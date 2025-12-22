"""
BOSS-ESSI-BO: Bayesian Optimization over Subspaces with ESSI.

This method implements a hierarchical BO approach that:
1. Uses an outer GP with Hamming kernel to model the quality of subspaces (binary masks)
2. Runs inner BO loops within selected subspaces
3. Uses local search to find the best subspace via acquisition maximization

Algorithm Overview:
- Initialization: Sample M0 random subspaces and run inner BO on each
- Main loop:
  1. Fit outer GP on subspace quality data with Hamming kernel
  2. Choose next subspace via acquisition maximization over {0,1}^D
  3. Run inner BO on the selected subspace
  4. Update subspace scores and global best

Reference:
    BOSS-ESSI-BO: Bayesian Optimization over Subspaces with Expected
    Subspace Selection Improvement
"""

import math
from typing import Optional, Tuple, List, Dict, Any, Set
import torch
from torch import Tensor
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

from gpytorch.kernels import Kernel
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.priors import GammaPrior
from gpytorch.constraints import GreaterThan
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from gpytorch.distributions import MultivariateNormal

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

from .base import BaseOptimizer
from .utils import DEVICE, DTYPE, ensure_tensor, clamp_to_bounds


# =============================================================================
# Hamming Kernel for Outer GP over Binary Masks
# =============================================================================

class HammingKernel(Kernel):
    """
    Hamming distance kernel for binary vectors.
    
    k(z, z') = σ_f^2 * exp(-λ * d_H(z, z'))
    
    where d_H(z, z') = sum_{j=1}^D 1[z_j ≠ z'_j] is the Hamming distance.
    
    Args:
        D: Dimension of the binary vectors
        lengthscale: Initial lengthscale (λ) parameter
        outputscale: Initial output scale (σ_f^2) parameter
    """
    
    has_lengthscale = True
    
    def __init__(
        self,
        D: int,
        lengthscale: float = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.D = D
        self.register_parameter(
            name="raw_lengthscale",
            parameter=torch.nn.Parameter(torch.tensor(lengthscale))
        )
        
    @property
    def lengthscale(self) -> Tensor:
        return torch.nn.functional.softplus(self.raw_lengthscale)
    
    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        **params
    ) -> Tensor:
        """
        Compute the Hamming kernel.
        
        Args:
            x1: First set of binary vectors (n1, D)
            x2: Second set of binary vectors (n2, D)
            diag: If True, return only diagonal elements
            
        Returns:
            Kernel matrix of shape (n1, n2) or (n1,) if diag=True
        """
        if diag:
            # Diagonal case: d_H(x1[i], x2[i])
            hamming_dist = (x1 != x2).float().sum(dim=-1)
            return torch.exp(-self.lengthscale * hamming_dist)
        else:
            # Full kernel matrix
            # Expand dimensions for broadcasting: (n1, 1, D) vs (1, n2, D)
            x1_expanded = x1.unsqueeze(1)  # (n1, 1, D)
            x2_expanded = x2.unsqueeze(0)  # (1, n2, D)
            
            # Compute Hamming distance
            hamming_dist = (x1_expanded != x2_expanded).float().sum(dim=-1)  # (n1, n2)
            
            return torch.exp(-self.lengthscale * hamming_dist)


class OuterGP(ExactGP):
    """
    Outer GP model with Hamming kernel for subspace quality modeling.
    
    Models the quality score y_z = SubspaceScore(z) for binary mask z.
    """
    
    def __init__(
        self,
        train_z: Tensor,  # Binary masks (N, D)
        train_y: Tensor,  # Subspace scores (N,) or (N, 1)
        D: int,
        likelihood: Optional[GaussianLikelihood] = None
    ):
        if train_y.dim() == 2:
            train_y = train_y.squeeze(-1)
            
        if likelihood is None:
            likelihood = GaussianLikelihood()
            
        super().__init__(train_z, train_y, likelihood)
        
        self.D = D
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            HammingKernel(D=D, lengthscale=1.0 / D)
        )
        
    def forward(self, z: Tensor) -> MultivariateNormal:
        """Forward pass through the GP."""
        mean_z = self.mean_module(z)
        covar_z = self.covar_module(z)
        return MultivariateNormal(mean_z, covar_z)


# =============================================================================
# Inner BO for Subspace Optimization
# =============================================================================

class InnerBO:
    """
    Inner BO optimizer for a fixed subspace.
    
    Optimizes the objective function in a d_sub-dimensional subspace
    defined by the active coordinate indices.
    """
    
    def __init__(
        self,
        full_dim: int,
        active_indices: List[int],
        bounds: Tensor,
        acq_func: str = "EI",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize inner BO.
        
        Args:
            full_dim: Full dimensionality D
            active_indices: List of active coordinate indices (S)
            bounds: Full bounds tensor (2, D)
            acq_func: Acquisition function type
            device: Device for computation
            dtype: Data type for tensors
        """
        self.full_dim = full_dim
        self.active_indices = sorted(active_indices)
        self.sub_dim = len(active_indices)
        self.device = device or DEVICE
        self.dtype = dtype or DTYPE
        
        # Full bounds
        self.bounds = bounds.to(device=self.device, dtype=self.dtype)
        
        # Subspace bounds
        self.sub_bounds = torch.stack([
            self.bounds[0, self.active_indices],
            self.bounds[1, self.active_indices]
        ])
        
        # Unit bounds for subspace
        self.unit_sub_bounds = torch.stack([
            torch.zeros(self.sub_dim, device=self.device, dtype=self.dtype),
            torch.ones(self.sub_dim, device=self.device, dtype=self.dtype)
        ])
        
        self.acq_func = acq_func
        
        # Training data in subspace
        self.train_X_sub = torch.empty((0, self.sub_dim), device=self.device, dtype=self.dtype)
        self.train_y = torch.empty((0, 1), device=self.device, dtype=self.dtype)
        
        # Best solution
        self.x_best_sub: Optional[Tensor] = None
        self.f_best: Optional[float] = None
        
        # GP model
        self.model: Optional[SingleTaskGP] = None
        
        # Statistics for standardization
        self._y_mean = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self._y_std = torch.tensor(1.0, device=self.device, dtype=self.dtype)
        
    def _normalize_X_sub(self, X_sub: Tensor) -> Tensor:
        """Normalize subspace X to [0, 1]^d_sub."""
        lb = self.sub_bounds[0]
        ub = self.sub_bounds[1]
        return (X_sub - lb) / (ub - lb + 1e-8)
    
    def _denormalize_X_sub(self, X_sub_norm: Tensor) -> Tensor:
        """Denormalize X from [0, 1]^d_sub to original subspace bounds."""
        lb = self.sub_bounds[0]
        ub = self.sub_bounds[1]
        return lb + X_sub_norm * (ub - lb)
    
    def _update_y_statistics(self):
        """Update Y mean and std for standardization."""
        if self.train_y.numel() > 0:
            self._y_mean = self.train_y.mean()
            self._y_std = self.train_y.std()
            if self._y_std < 1e-6:
                self._y_std = torch.tensor(1.0, device=self.device, dtype=self.dtype)
                
    def _standardize_y(self, y: Tensor) -> Tensor:
        """Standardize y to mean=0, std=1."""
        return (y - self._y_mean) / (self._y_std + 1e-8)
    
    def project_to_subspace(self, x_full: Tensor) -> Tensor:
        """
        Project full-dimensional point to subspace.
        
        Args:
            x_full: Full point (D,) or (n, D)
            
        Returns:
            Subspace point (d_sub,) or (n, d_sub)
        """
        if x_full.dim() == 1:
            return x_full[self.active_indices]
        else:
            return x_full[:, self.active_indices]
    
    def fill_full_vector(self, x_sub: Tensor, x_base: Tensor) -> Tensor:
        """
        Fill a full-dimensional vector from subspace values.
        
        Args:
            x_sub: Subspace values (d_sub,) or (n, d_sub)
            x_base: Base full vector (D,) or (n, D)
            
        Returns:
            Full vector with subspace coordinates replaced
        """
        x_full = x_base.clone()
        if x_sub.dim() == 1:
            for i, idx in enumerate(self.active_indices):
                x_full[idx] = x_sub[i]
        else:
            for i, idx in enumerate(self.active_indices):
                x_full[:, idx] = x_sub[:, i]
        return x_full
    
    def observe(self, x_sub: Tensor, y: Tensor):
        """
        Add observation to inner BO.
        
        Args:
            x_sub: Subspace point (d_sub,) or (1, d_sub)
            y: Objective value (scalar or (1,) or (1, 1))
        """
        x_sub = ensure_tensor(x_sub, device=self.device, dtype=self.dtype)
        y = ensure_tensor(y, device=self.device, dtype=self.dtype)
        
        if x_sub.dim() == 1:
            x_sub = x_sub.unsqueeze(0)
        if y.dim() == 0:
            y = y.unsqueeze(0).unsqueeze(-1)
        elif y.dim() == 1:
            y = y.unsqueeze(-1)
            
        self.train_X_sub = torch.cat([self.train_X_sub, x_sub], dim=0)
        self.train_y = torch.cat([self.train_y, y], dim=0)
        
        # Update best (minimization)
        y_val = y.item()
        if self.f_best is None or y_val < self.f_best:
            self.f_best = y_val
            self.x_best_sub = x_sub.squeeze(0).clone()
    
    def fit_model(self):
        """Fit GP model to current data."""
        if self.train_X_sub.numel() == 0:
            return
            
        self._update_y_statistics()
        
        X_norm = self._normalize_X_sub(self.train_X_sub)
        y_std = self._standardize_y(self.train_y)
        
        # Create GP with Matern kernel
        covar_module = ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=self.sub_dim,
                lengthscale_constraint=GreaterThan(1e-4),
            ),
            outputscale_constraint=GreaterThan(1e-4),
        )
        
        self.model = SingleTaskGP(
            train_X=X_norm,
            train_Y=y_std,
            covar_module=covar_module,
            input_transform=None,
            outcome_transform=None,
        ).to(device=self.device, dtype=self.dtype)
        
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)
    
    def suggest(self) -> Tensor:
        """
        Suggest next point in subspace.
        
        Returns:
            Suggested point in subspace coordinates (d_sub,)
        """
        if self.model is None or self.train_y.numel() < 2:
            # Random sampling if not enough data
            lb = self.sub_bounds[0]
            ub = self.sub_bounds[1]
            return lb + (ub - lb) * torch.rand(self.sub_dim, device=self.device, dtype=self.dtype)
        
        # Get standardized best value (for minimization, we want highest negated value)
        y_std = self._standardize_y(self.train_y)
        best_f = y_std.max()  # For minimization with negated standardization
        
        # Create acquisition function
        if self.acq_func == "UCB":
            acq = UpperConfidenceBound(self.model, beta=0.25)
        else:  # EI
            acq = ExpectedImprovement(self.model, best_f=best_f)
        
        # Optimize acquisition in unit cube
        try:
            candidates_norm, _ = optimize_acqf(
                acq,
                bounds=self.unit_sub_bounds,
                q=1,
                num_restarts=5,
                raw_samples=32,
            )
            
            # Denormalize
            candidates = self._denormalize_X_sub(candidates_norm)
            
            # Clamp to bounds
            candidates = torch.clamp(candidates, self.sub_bounds[0], self.sub_bounds[1])
            
            return candidates.squeeze(0)
        except Exception:
            # Fallback to random
            lb = self.sub_bounds[0]
            ub = self.sub_bounds[1]
            return lb + (ub - lb) * torch.rand(self.sub_dim, device=self.device, dtype=self.dtype)


# =============================================================================
# BOSS-ESSI-BO Main Class
# =============================================================================

class BOSSESSIBO(BaseOptimizer):
    """
    BOSS-ESSI-BO: Bayesian Optimization over Subspaces with ESSI.
    
    This method uses a hierarchical approach:
    1. Outer GP with Hamming kernel models subspace quality
    2. Inner BO optimizes within selected subspaces
    3. Acquisition maximization over binary masks selects subspaces
    
    Args:
        input_dim: Dimensionality of the input space (D)
        bounds: Bounds tensor of shape (2, input_dim)
        m_min: Minimum subspace size
        m_max: Maximum subspace size
        m0: Number of initial random subspaces
        b0: Inner BO budget per subspace
        outer_acq: Outer acquisition function type ('EI' or 'UCB')
        inner_acq: Inner acquisition function type ('EI' or 'UCB')
        max_ls_iters: Maximum local search iterations for subspace selection
        ucb_beta: UCB beta parameter for outer acquisition
        device: Device for computation
        dtype: Data type for tensors
    """
    
    def __init__(
        self,
        input_dim: int,
        bounds: Tensor,
        m_min: Optional[int] = None,
        m_max: Optional[int] = None,
        m0: int = 3,
        b0: int = 5,
        outer_acq: str = "EI",
        inner_acq: str = "EI",
        max_ls_iters: int = 20,
        ucb_beta: float = 2.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        # Initialize base class
        _device = device or DEVICE
        _dtype = dtype or DTYPE
        
        super().__init__(
            input_dim=input_dim,
            bounds=bounds,
            acq_func=inner_acq,
            device=_device,
            dtype=_dtype,
        )
        
        # Subspace size range
        self.m_min = m_min or max(1, input_dim // 4)
        self.m_max = m_max or max(self.m_min, input_dim // 2)
        
        # Ensure m_min <= m_max <= input_dim
        self.m_min = min(self.m_min, input_dim)
        self.m_max = min(self.m_max, input_dim)
        self.m_max = max(self.m_min, self.m_max)
        
        # Initialization parameters
        self.m0 = m0  # Number of initial subspaces
        self.b0 = b0  # Inner BO budget per subspace
        
        # Acquisition settings
        self.outer_acq = outer_acq
        self.inner_acq = inner_acq
        self.max_ls_iters = max_ls_iters
        self.ucb_beta = ucb_beta
        
        # Outer GP data: D_Z = {(z_i, y_i)}
        self.subspace_masks: List[Tensor] = []  # Binary masks
        self.subspace_scores: List[float] = []  # Quality scores
        
        # Set of subspaces with their inner models
        self.subspace_set: List[Dict[str, Any]] = []
        
        # Outer GP model
        self.outer_gp: Optional[OuterGP] = None
        
        # Global best
        self.x_best: Optional[Tensor] = None
        self.f_best: Optional[float] = None
        
        # Initialization state
        self._initialized = False
        self._init_subspace_idx = 0
        self._current_inner_bo: Optional[InnerBO] = None
        self._current_mask: Optional[Tensor] = None
        self._inner_bo_step = 0
        self._pending_mask: Optional[Tensor] = None
        
        # Full training data
        self.X: Optional[Tensor] = None
        self.y: Optional[Tensor] = None
        
    # -------------------------------------------------------------------------
    # Utility Functions
    # -------------------------------------------------------------------------
    
    def _sample_random_mask(self) -> Tensor:
        """
        Sample a random binary mask with size in [m_min, m_max].
        
        Returns:
            Binary mask tensor of shape (D,)
        """
        m = np.random.randint(self.m_min, self.m_max + 1)
        indices = np.random.choice(self.input_dim, size=m, replace=False)
        
        mask = torch.zeros(self.input_dim, device=self.device, dtype=self.dtype)
        mask[indices] = 1.0
        return mask
    
    def _mask_to_indices(self, mask: Tensor) -> List[int]:
        """Convert binary mask to list of active indices."""
        return (mask == 1.0).nonzero(as_tuple=True)[0].tolist()
    
    def _indices_to_mask(self, indices: List[int]) -> Tensor:
        """Convert list of indices to binary mask."""
        mask = torch.zeros(self.input_dim, device=self.device, dtype=self.dtype)
        for idx in indices:
            mask[idx] = 1.0
        return mask
    
    def _subspace_score(self, f_best_z: float) -> float:
        """
        Compute subspace score.
        
        Option A: Direct negative best value (for minimization).
        
        Args:
            f_best_z: Best value found in the subspace
            
        Returns:
            Score (higher is better for the outer GP)
        """
        # For minimization: lower f_best_z is better, so negate
        return -f_best_z
    
    def _generate_neighbors(self, mask: Tensor) -> List[Tensor]:
        """
        Generate neighboring masks by flipping single bits.
        
        Only generates neighbors with size in [m_min, m_max].
        
        Args:
            mask: Current binary mask
            
        Returns:
            List of neighbor masks
        """
        neighbors = []
        current_size = int(mask.sum().item())
        
        for j in range(self.input_dim):
            neighbor = mask.clone()
            neighbor[j] = 1.0 - neighbor[j]  # Flip bit j
            
            new_size = int(neighbor.sum().item())
            if self.m_min <= new_size <= self.m_max:
                neighbors.append(neighbor)
                
        return neighbors
    
    # -------------------------------------------------------------------------
    # Outer GP Methods
    # -------------------------------------------------------------------------
    
    def _fit_outer_gp(self):
        """Fit outer GP with Hamming kernel on subspace data."""
        if len(self.subspace_masks) < 2:
            return
            
        # Stack masks and scores
        train_z = torch.stack(self.subspace_masks)
        train_y = torch.tensor(
            self.subspace_scores, 
            device=self.device, 
            dtype=self.dtype
        )
        
        # Create and fit outer GP
        self.outer_gp = OuterGP(
            train_z=train_z,
            train_y=train_y,
            D=self.input_dim
        ).to(device=self.device, dtype=self.dtype)
        
        mll = ExactMarginalLogLikelihood(self.outer_gp.likelihood, self.outer_gp)
        try:
            fit_gpytorch_mll(mll)
        except Exception:
            # If fitting fails, continue with default parameters
            pass
    
    def _outer_acquisition(self, mask: Tensor) -> float:
        """
        Compute outer acquisition function value for a mask.
        
        Args:
            mask: Binary mask (D,)
            
        Returns:
            Acquisition value
        """
        if self.outer_gp is None:
            return 0.0
            
        self.outer_gp.eval()
        
        with torch.no_grad():
            mask_input = mask.unsqueeze(0)
            posterior = self.outer_gp(mask_input)
            mu = posterior.mean.item()
            sigma = posterior.variance.sqrt().item()
            
        if self.outer_acq == "UCB":
            return mu + self.ucb_beta * sigma
        else:  # EI
            if sigma <= 1e-10:
                return 0.0
            
            # Best score so far
            f_star = max(self.subspace_scores) if self.subspace_scores else 0.0
            
            z = (mu - f_star) / sigma
            ei = (mu - f_star) * norm.cdf(z) + sigma * norm.pdf(z)
            return max(ei, 0.0)
    
    def _local_search_for_mask(self) -> Tensor:
        """
        Local search to find best mask via acquisition maximization.
        
        Uses multi-start local search with 1-bit flips.
        
        Returns:
            Best mask found
        """
        # Initialize with existing masks and some random ones
        init_masks = list(self.subspace_masks)
        n_random = max(3, len(init_masks))
        for _ in range(n_random):
            init_masks.append(self._sample_random_mask())
            
        # Find best starting point
        best_mask = None
        best_acq = float('-inf')
        
        for mask in init_masks:
            acq = self._outer_acquisition(mask)
            if acq > best_acq:
                best_acq = acq
                best_mask = mask.clone()
                
        # Local search from each starting point
        for mask_start in init_masks[:min(5, len(init_masks))]:
            mask_curr = mask_start.clone()
            acq_curr = self._outer_acquisition(mask_curr)
            
            for _ in range(self.max_ls_iters):
                # Generate neighbors
                neighbors = self._generate_neighbors(mask_curr)
                if not neighbors:
                    break
                    
                # Find best neighbor
                best_neighbor = None
                best_neighbor_acq = acq_curr
                
                for neighbor in neighbors:
                    acq = self._outer_acquisition(neighbor)
                    if acq > best_neighbor_acq:
                        best_neighbor_acq = acq
                        best_neighbor = neighbor
                        
                # Move to best neighbor if improvement found
                if best_neighbor is not None and best_neighbor_acq > acq_curr:
                    mask_curr = best_neighbor
                    acq_curr = best_neighbor_acq
                else:
                    break  # Local optimum
                    
            # Update global best
            if acq_curr > best_acq:
                best_acq = acq_curr
                best_mask = mask_curr.clone()
                
        return best_mask if best_mask is not None else self._sample_random_mask()
    
    # -------------------------------------------------------------------------
    # Inner BO Methods
    # -------------------------------------------------------------------------
    
    def _run_inner_bo_step(
        self,
        inner_bo: InnerBO,
        x_base: Tensor,
        evaluate_func: Optional[callable] = None
    ) -> Tuple[Tensor, Optional[float]]:
        """
        Run a single step of inner BO.
        
        Args:
            inner_bo: Inner BO instance
            x_base: Base point for non-active coordinates
            evaluate_func: Optional function to evaluate (for simulation)
            
        Returns:
            Tuple of (full point, objective value if evaluated)
        """
        # Get suggestion from inner BO
        if inner_bo.train_X_sub.numel() == 0:
            # First point: use projection of x_base or random
            if x_base is not None:
                x_sub = inner_bo.project_to_subspace(x_base)
            else:
                lb = inner_bo.sub_bounds[0]
                ub = inner_bo.sub_bounds[1]
                x_sub = lb + (ub - lb) * torch.rand(
                    inner_bo.sub_dim, device=self.device, dtype=self.dtype
                )
        else:
            inner_bo.fit_model()
            x_sub = inner_bo.suggest()
            
        # Fill full vector
        base = x_base if x_base is not None else torch.zeros(
            self.input_dim, device=self.device, dtype=self.dtype
        )
        x_full = inner_bo.fill_full_vector(x_sub, base)
        
        # Clamp to bounds
        x_full = torch.clamp(x_full, self.bounds[0], self.bounds[1])
        
        return x_full, None
    
    # -------------------------------------------------------------------------
    # Main Interface
    # -------------------------------------------------------------------------
    
    def observe(self, X: Tensor, y: Tensor):
        """
        Update optimizer with new observations.
        
        Args:
            X: Input points (n, D) or (D,)
            y: Objective values (n,) or (n, 1) or scalar
        """
        X = ensure_tensor(X, device=self.device, dtype=self.dtype)
        y = ensure_tensor(y, device=self.device, dtype=self.dtype)
        
        if X.dim() == 1:
            X = X.unsqueeze(0)
        if y.dim() == 0:
            y = y.unsqueeze(0).unsqueeze(-1)
        elif y.dim() == 1:
            y = y.unsqueeze(-1)
            
        # Store in full dataset
        if self.X is None:
            self.X = X
            self.y = y
        else:
            self.X = torch.cat([self.X, X], dim=0)
            self.y = torch.cat([self.y, y], dim=0)
            
        # Also update base class for consistency
        self.train_X = self.X
        self.train_y = self.y
        
        # Update global best (minimization)
        for i in range(X.shape[0]):
            y_val = y[i].item()
            if self.f_best is None or y_val < self.f_best:
                self.f_best = y_val
                self.x_best = X[i].clone()
                
        # Update current inner BO if active
        if self._current_inner_bo is not None:
            for i in range(X.shape[0]):
                x_sub = self._current_inner_bo.project_to_subspace(X[i])
                self._current_inner_bo.observe(x_sub, y[i])
                
            self._inner_bo_step += 1
            
            # Check if inner BO is complete
            if self._inner_bo_step >= self.b0:
                self._complete_current_subspace()
    
    def _complete_current_subspace(self):
        """Complete current subspace optimization and update outer GP data."""
        if self._current_inner_bo is None or self._current_mask is None:
            return
            
        # Get best from inner BO
        f_best_z = self._current_inner_bo.f_best
        if f_best_z is None:
            f_best_z = float('inf')
            
        # Compute subspace score
        score = self._subspace_score(f_best_z)
        
        # Add to outer GP data
        self.subspace_masks.append(self._current_mask.clone())
        self.subspace_scores.append(score)
        
        # Add to subspace set
        self.subspace_set.append({
            'mask': self._current_mask.clone(),
            'inner_bo': self._current_inner_bo,
            'score': score,
            'f_best': f_best_z,
        })
        
        # Reset current inner BO
        self._current_inner_bo = None
        self._current_mask = None
        self._inner_bo_step = 0
        
        # Update initialization state
        if not self._initialized:
            self._init_subspace_idx += 1
            if self._init_subspace_idx >= self.m0:
                self._initialized = True
    
    def suggest(self, n_suggestions: int = 1) -> Tensor:
        """
        Suggest next point(s) to evaluate.
        
        Args:
            n_suggestions: Number of points to suggest
            
        Returns:
            Suggested points (n_suggestions, D)
        """
        suggestions = []
        
        for _ in range(n_suggestions):
            suggestion = self._suggest_one()
            suggestions.append(suggestion)
            
        return torch.stack(suggestions)
    
    def _suggest_one(self) -> Tensor:
        """Suggest a single point."""
        
        # Case 1: No data yet - return random point
        if self.X is None or self.X.numel() == 0:
            # Start with a random subspace
            self._current_mask = self._sample_random_mask()
            active_indices = self._mask_to_indices(self._current_mask)
            
            self._current_inner_bo = InnerBO(
                full_dim=self.input_dim,
                active_indices=active_indices,
                bounds=self.bounds,
                acq_func=self.inner_acq,
                device=self.device,
                dtype=self.dtype,
            )
            self._inner_bo_step = 0
            
            # Return random initial point
            lb = self.bounds[0]
            ub = self.bounds[1]
            return lb + (ub - lb) * torch.rand(self.input_dim, device=self.device, dtype=self.dtype)
        
        # Case 2: In middle of inner BO
        if self._current_inner_bo is not None and self._inner_bo_step < self.b0:
            x_full, _ = self._run_inner_bo_step(
                self._current_inner_bo,
                self.x_best
            )
            return x_full
        
        # Case 3: Need to select new subspace
        # First, complete previous if needed
        if self._current_inner_bo is not None:
            self._complete_current_subspace()
            
        # During initialization: use random subspaces
        if not self._initialized:
            new_mask = self._sample_random_mask()
        else:
            # Fit outer GP and find best subspace
            self._fit_outer_gp()
            
            if self.outer_gp is not None:
                new_mask = self._local_search_for_mask()
            else:
                new_mask = self._sample_random_mask()
        
        # Start new inner BO
        active_indices = self._mask_to_indices(new_mask)
        
        self._current_mask = new_mask
        self._current_inner_bo = InnerBO(
            full_dim=self.input_dim,
            active_indices=active_indices,
            bounds=self.bounds,
            acq_func=self.inner_acq,
            device=self.device,
            dtype=self.dtype,
        )
        self._inner_bo_step = 0
        
        # Get first suggestion from new inner BO
        x_full, _ = self._run_inner_bo_step(
            self._current_inner_bo,
            self.x_best
        )
        return x_full
    
    def get_best_point(self) -> Tuple[Tensor, Tensor]:
        """
        Get best observed point.
        
        Returns:
            Tuple of (best_x, best_y)
        """
        if self.y is None or self.y.numel() == 0:
            raise ValueError("No observations yet.")
            
        best_idx = self.y.argmin()  # Minimization
        return self.X[best_idx], self.y[best_idx]
    
    def reset(self):
        """Reset optimizer to initial state."""
        super().reset()
        
        self.subspace_masks = []
        self.subspace_scores = []
        self.subspace_set = []
        self.outer_gp = None
        
        self.x_best = None
        self.f_best = None
        
        self._initialized = False
        self._init_subspace_idx = 0
        self._current_inner_bo = None
        self._current_mask = None
        self._inner_bo_step = 0
        
        self.X = None
        self.y = None


class BOSSESSIBOMaximize(BOSSESSIBO):
    """
    BOSS-ESSI-BO variant for maximization problems.
    """
    
    def _subspace_score(self, f_best_z: float) -> float:
        """For maximization: higher f_best_z is better."""
        return f_best_z
    
    def observe(self, X: Tensor, y: Tensor):
        """Update with observations (maximization)."""
        X = ensure_tensor(X, device=self.device, dtype=self.dtype)
        y = ensure_tensor(y, device=self.device, dtype=self.dtype)
        
        if X.dim() == 1:
            X = X.unsqueeze(0)
        if y.dim() == 0:
            y = y.unsqueeze(0).unsqueeze(-1)
        elif y.dim() == 1:
            y = y.unsqueeze(-1)
            
        # Store in full dataset
        if self.X is None:
            self.X = X
            self.y = y
        else:
            self.X = torch.cat([self.X, X], dim=0)
            self.y = torch.cat([self.y, y], dim=0)
            
        # Also update base class
        self.train_X = self.X
        self.train_y = self.y
        
        # Update global best (maximization)
        for i in range(X.shape[0]):
            y_val = y[i].item()
            if self.f_best is None or y_val > self.f_best:
                self.f_best = y_val
                self.x_best = X[i].clone()
                
        # Update current inner BO if active
        if self._current_inner_bo is not None:
            for i in range(X.shape[0]):
                x_sub = self._current_inner_bo.project_to_subspace(X[i])
                # Negate for inner BO which minimizes
                self._current_inner_bo.observe(x_sub, -y[i])
                
            self._inner_bo_step += 1
            
            if self._inner_bo_step >= self.b0:
                self._complete_current_subspace_max()
    
    def _complete_current_subspace_max(self):
        """Complete current subspace for maximization."""
        if self._current_inner_bo is None or self._current_mask is None:
            return
            
        # Get best from inner BO (negated)
        f_best_z = self._current_inner_bo.f_best
        if f_best_z is not None:
            f_best_z = -f_best_z  # Un-negate
        else:
            f_best_z = float('-inf')
            
        # Compute subspace score
        score = self._subspace_score(f_best_z)
        
        # Add to outer GP data
        self.subspace_masks.append(self._current_mask.clone())
        self.subspace_scores.append(score)
        
        # Add to subspace set
        self.subspace_set.append({
            'mask': self._current_mask.clone(),
            'inner_bo': self._current_inner_bo,
            'score': score,
            'f_best': f_best_z,
        })
        
        # Reset current inner BO
        self._current_inner_bo = None
        self._current_mask = None
        self._inner_bo_step = 0
        
        # Update initialization state
        if not self._initialized:
            self._init_subspace_idx += 1
            if self._init_subspace_idx >= self.m0:
                self._initialized = True
    
    def get_best_point(self) -> Tuple[Tensor, Tensor]:
        """Get best observed point (maximization)."""
        if self.y is None or self.y.numel() == 0:
            raise ValueError("No observations yet.")
            
        best_idx = self.y.argmax()
        return self.X[best_idx], self.y[best_idx]
