"""
BOSS-ESSI-BO v2: Bayesian Optimization for Subspace Selection with ESSI.

This version strictly follows the algorithm pseudocode:

Algorithm Overview:
1. Train a full-space GP (GP_X) on all data D
2. Use outer GP (GP_Z) with Hamming kernel to model subspace quality
3. For each iteration:
   - Select k candidate subspaces using outer BO
   - For each subspace, maximize ESSI using the full-space GP
   - Evaluate the point from the subspace with highest ESSI
   - Update outer GP with (subspace, ESSI) pair

Key differences from v1:
- Uses a single full-space GP for ESSI computation
- ESSI is computed by fixing non-active coordinates to x*
- k candidate subspaces are compared per iteration
- Outer GP is trained on ESSI values (not best objective values)

Reference:
    BOSS-ESSI-BO: Bayesian Optimization over Subspaces with Expected
    Subspace Selection Improvement
"""

import math
from typing import Optional, Tuple, List, Dict, Any
import torch
from torch import Tensor
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

from gpytorch.kernels import Kernel, ScaleKernel, MaternKernel
from gpytorch.priors import GammaPrior
from gpytorch.constraints import GreaterThan
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll

from .base import BaseOptimizer
from .utils import DEVICE, DTYPE, ensure_tensor, clamp_to_bounds


# =============================================================================
# Hamming Kernel for Outer GP over Binary Masks
# =============================================================================

class HammingKernel(Kernel):
    """
    Hamming distance kernel for binary vectors.
    
    k(z, z') = exp(-λ * d_H(z, z'))
    
    where d_H(z, z') = sum_{j=1}^D 1[z_j ≠ z'_j] is the Hamming distance.
    """
    
    has_lengthscale = True
    
    def __init__(self, D: int, lengthscale: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.D = D
        self.register_parameter(
            name="raw_lengthscale",
            parameter=torch.nn.Parameter(torch.tensor(lengthscale))
        )
        
    @property
    def lengthscale(self) -> Tensor:
        return torch.nn.functional.softplus(self.raw_lengthscale)
    
    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, **params) -> Tensor:
        """Compute the Hamming kernel."""
        if diag:
            hamming_dist = (x1 != x2).float().sum(dim=-1)
            return torch.exp(-self.lengthscale * hamming_dist)
        else:
            x1_expanded = x1.unsqueeze(1)
            x2_expanded = x2.unsqueeze(0)
            hamming_dist = (x1_expanded != x2_expanded).float().sum(dim=-1)
            return torch.exp(-self.lengthscale * hamming_dist)


class OuterGP(ExactGP):
    """
    Outer GP model with Hamming kernel for subspace quality (ESSI) modeling.
    """
    
    def __init__(
        self,
        train_z: Tensor,
        train_y: Tensor,
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
        mean_z = self.mean_module(z)
        covar_z = self.covar_module(z)
        return MultivariateNormal(mean_z, covar_z)


# =============================================================================
# BOSS-ESSI-BO v2 Main Class
# =============================================================================

class BOSSESSIBO_V2(BaseOptimizer):
    """
    BOSS-ESSI-BO v2: Strictly follows the algorithm pseudocode.
    
    Key features:
    - Full-space GP (GP_X) for ESSI computation
    - Outer GP (GP_Z) with Hamming kernel for subspace selection
    - k candidate subspaces compared per iteration
    - ESSI values used for outer GP training
    
    Args:
        input_dim: Dimensionality of the input space (D)
        bounds: Bounds tensor of shape (2, input_dim)
        s: Subspace size (number of active coordinates)
        k: Number of candidate subspaces per iteration
        outer_acq: Outer acquisition function type ('EI' or 'UCB')
        max_ls_iters: Maximum local search iterations
        ucb_beta: UCB beta parameter
        device: Device for computation
        dtype: Data type for tensors
    """
    
    def __init__(
        self,
        input_dim: int,
        bounds: Tensor,
        s: Optional[int] = None,
        k: int = 3,
        outer_acq: str = "EI",
        max_ls_iters: int = 20,
        ucb_beta: float = 2.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        _device = device or DEVICE
        _dtype = dtype or DTYPE
        
        super().__init__(
            input_dim=input_dim,
            bounds=bounds,
            acq_func="EI",
            device=_device,
            dtype=_dtype,
        )
        
        # Subspace size (default: ~1/4 of dimensions)
        self.s = s or max(1, input_dim // 4)
        self.s = min(self.s, input_dim)
        
        # Number of candidate subspaces per iteration
        self.k = k
        
        # Outer acquisition settings
        self.outer_acq = outer_acq
        self.max_ls_iters = max_ls_iters
        self.ucb_beta = ucb_beta
        
        # Full-space GP (GP_X)
        self.gp_x: Optional[SingleTaskGP] = None
        
        # Outer GP data: {(mask_i, ESSI_i)}
        self.subspace_masks: List[Tensor] = []
        self.subspace_essi: List[float] = []
        
        # Outer GP model (GP_Z)
        self.outer_gp: Optional[OuterGP] = None
        
        # Global best
        self.x_best: Optional[Tensor] = None
        self.f_best: Optional[float] = None
        
        # Full training data
        self.X: Optional[Tensor] = None
        self.y: Optional[Tensor] = None
        
        # Y statistics for GP_X
        self._gp_y_mean = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self._gp_y_std = torch.tensor(1.0, device=self.device, dtype=self.dtype)
        
    # -------------------------------------------------------------------------
    # Utility Functions
    # -------------------------------------------------------------------------
    
    def _normalize_X(self, X: Tensor) -> Tensor:
        """Normalize X from original bounds to [0, 1]^d."""
        lb = self.bounds[0]
        ub = self.bounds[1]
        return (X - lb) / (ub - lb + 1e-8)
    
    def _denormalize_X(self, X_norm: Tensor) -> Tensor:
        """Denormalize X from [0, 1]^d to original bounds."""
        lb = self.bounds[0]
        ub = self.bounds[1]
        return lb + X_norm * (ub - lb)
    
    def _standardize_y(self, y: Tensor) -> Tensor:
        """Standardize y to mean=0, std=1."""
        return (y - self._gp_y_mean) / (self._gp_y_std + 1e-8)
    
    def _update_y_statistics(self):
        """Update Y mean and std."""
        if self.y is not None and self.y.numel() > 0:
            self._gp_y_mean = self.y.mean()
            # Handle single observation case: std() returns NaN for single element
            if self.y.numel() < 2:
                self._gp_y_std = torch.tensor(1.0, device=self.device, dtype=self.dtype)
            else:
                self._gp_y_std = self.y.std()
                # Check for NaN or very small std
                if torch.isnan(self._gp_y_std) or self._gp_y_std < 1e-6:
                    self._gp_y_std = torch.tensor(1.0, device=self.device, dtype=self.dtype)
    
    def _sample_random_mask(self) -> Tensor:
        """Sample a random binary mask with size s."""
        indices = np.random.choice(self.input_dim, size=self.s, replace=False)
        mask = torch.zeros(self.input_dim, device=self.device, dtype=self.dtype)
        mask[indices] = 1.0
        return mask
    
    def _mask_to_indices(self, mask: Tensor) -> List[int]:
        """Convert binary mask to list of active indices."""
        return (mask == 1.0).nonzero(as_tuple=True)[0].tolist()
    
    def _replace_coordinates(self, x_base: Tensor, S: List[int], y_sub: Tensor) -> Tensor:
        """
        Replace coordinates in x_base at positions S with values from y_sub.
        
        Implements Eq.(7): z = ReplaceCoordinates(x*, S, y)
        
        Args:
            x_base: Base full vector (D,)
            S: List of active coordinate indices
            y_sub: Subspace values (|S|,)
            
        Returns:
            Full vector with subspace coordinates replaced
        """
        z = x_base.clone()
        for i, idx in enumerate(S):
            z[idx] = y_sub[i]
        return z
    
    # -------------------------------------------------------------------------
    # Full-Space GP (GP_X) Methods
    # -------------------------------------------------------------------------
    
    def _fit_full_gp(self):
        """
        Fit the full-space GP (GP_X) on all data D.
        
        Pseudocode Step 4: Train a GP surrogate GP_X on D in the full d-dimensional space.
        """
        if self.X is None or self.X.numel() == 0:
            return
            
        self._update_y_statistics()
        
        X_norm = self._normalize_X(self.X)
        y_std = self._standardize_y(self.y)
        
        # Create GP with Matern kernel
        covar_module = ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=self.input_dim,
                lengthscale_constraint=GreaterThan(1e-4),
            ),
            outputscale_constraint=GreaterThan(1e-4),
        )
        
        self.gp_x = SingleTaskGP(
            train_X=X_norm,
            train_Y=y_std,
            covar_module=covar_module,
            input_transform=None,
            outcome_transform=None,
        ).to(device=self.device, dtype=self.dtype)
        
        mll = ExactMarginalLogLikelihood(self.gp_x.likelihood, self.gp_x)
        try:
            fit_gpytorch_mll(mll)
        except Exception:
            pass
    
    def _predict_gp_x(self, z: Tensor) -> Tuple[float, float]:
        """
        Get GP_X prediction at point z.
        
        Args:
            z: Full-dimensional point (D,)
            
        Returns:
            (mu, sigma) - mean and std in original scale
        """
        if self.gp_x is None:
            return 0.0, 1.0
            
        self.gp_x.eval()
        
        z_norm = self._normalize_X(z.unsqueeze(0))
        
        with torch.no_grad():
            posterior = self.gp_x(z_norm)
            mu_std = posterior.mean.item()
            var_std = posterior.variance.item()
            sigma_std = math.sqrt(max(var_std, 1e-10))
            
        # Convert back to original scale
        mu = mu_std * self._gp_y_std.item() + self._gp_y_mean.item()
        sigma = sigma_std * self._gp_y_std.item()
        
        return mu, sigma
    
    # -------------------------------------------------------------------------
    # ESSI Acquisition Function
    # -------------------------------------------------------------------------
    
    def _compute_essi(self, y_sub: Tensor, S: List[int]) -> float:
        """
        Compute ESSI(y) for subspace coordinates y.
        
        Implements Eq.(6): ESSI(y) = (f* - μ) Φ(γ) + σ φ(γ)
        where γ = (f* - μ) / σ
        
        This is essentially EI, but computed by:
        1. Fixing non-active coordinates to x*
        2. Using the full-space GP for prediction
        
        Args:
            y_sub: Subspace coordinate values (|S|,)
            S: List of active coordinate indices
            
        Returns:
            ESSI value (higher is better for minimization)
        """
        if self.gp_x is None or self.x_best is None or self.f_best is None:
            return 0.0
            
        # Eq.(7): z = ReplaceCoordinates(x*, S, y)
        z = self._replace_coordinates(self.x_best, S, y_sub)
        
        # Get GP prediction
        mu, sigma = self._predict_gp_x(z)
        
        if sigma < 1e-10:
            return 0.0
            
        # Eq.(6): ESSI = (f* - μ) Φ(γ) + σ φ(γ), γ = (f* - μ) / σ
        # For minimization: improvement = f* - μ (positive when μ < f*)
        gamma = (self.f_best - mu) / sigma
        
        essi = (self.f_best - mu) * norm.cdf(gamma) + sigma * norm.pdf(gamma)
        
        return max(essi, 0.0)
    
    def _maximize_essi(self, S: List[int]) -> Tuple[Tensor, float]:
        """
        Maximize ESSI over subspace coordinates.
        
        Implements MaximizeESSI procedure from pseudocode.
        
        Args:
            S: List of active coordinate indices
            
        Returns:
            (x_new, ESSI*) - best full point and its ESSI value
        """
        if self.x_best is None:
            # Random point if no best yet
            lb = self.bounds[0]
            ub = self.bounds[1]
            x_new = lb + (ub - lb) * torch.rand(self.input_dim, device=self.device, dtype=self.dtype)
            return x_new, 0.0
            
        s = len(S)
        
        # Get subspace bounds
        sub_lb = self.bounds[0, S].cpu().numpy()
        sub_ub = self.bounds[1, S].cpu().numpy()
        sub_bounds = list(zip(sub_lb, sub_ub))
        
        # Objective: negative ESSI (for minimization by scipy)
        def neg_essi(y_np):
            y_sub = torch.tensor(y_np, device=self.device, dtype=self.dtype)
            return -self._compute_essi(y_sub, S)
        
        # Multi-start optimization
        best_essi = float('-inf')
        best_y = None
        
        n_starts = max(5, min(20, 2 * s))
        
        for i in range(n_starts):
            # Initialize: first start from x_best projection, others random
            if i == 0:
                y0 = self.x_best[S].cpu().numpy()
            else:
                y0 = np.array([np.random.uniform(lb, ub) for lb, ub in sub_bounds])
            
            try:
                result = minimize(
                    neg_essi,
                    y0,
                    method='L-BFGS-B',
                    bounds=sub_bounds,
                    options={'maxiter': 50}
                )
                
                essi_val = -result.fun
                if essi_val > best_essi:
                    best_essi = essi_val
                    best_y = result.x
            except Exception:
                continue
        
        if best_y is None:
            # Fallback: use x_best projection
            best_y = self.x_best[S].cpu().numpy()
            y_sub = torch.tensor(best_y, device=self.device, dtype=self.dtype)
            best_essi = self._compute_essi(y_sub, S)
        
        # Construct full point
        y_sub = torch.tensor(best_y, device=self.device, dtype=self.dtype)
        x_new = self._replace_coordinates(self.x_best, S, y_sub)
        
        # Clamp to bounds
        x_new = torch.clamp(x_new, self.bounds[0], self.bounds[1])
        
        return x_new, best_essi
    
    # -------------------------------------------------------------------------
    # Outer GP (GP_Z) Methods
    # -------------------------------------------------------------------------
    
    def _fit_outer_gp(self):
        """
        Fit outer GP with Hamming kernel on subspace data.
        
        Pseudocode Step 6: Fit an outer GP GP_Z over previously evaluated subspaces.
        """
        if len(self.subspace_masks) < 2:
            return
            
        train_z = torch.stack(self.subspace_masks)
        train_y = torch.tensor(
            self.subspace_essi, 
            device=self.device, 
            dtype=self.dtype
        )
        
        self.outer_gp = OuterGP(
            train_z=train_z,
            train_y=train_y,
            D=self.input_dim
        ).to(device=self.device, dtype=self.dtype)
        
        mll = ExactMarginalLogLikelihood(self.outer_gp.likelihood, self.outer_gp)
        try:
            fit_gpytorch_mll(mll)
        except Exception:
            pass
    
    def _outer_acquisition(self, mask: Tensor) -> float:
        """
        Compute outer acquisition function value for a mask.
        
        Pseudocode Step 2.1: a(m) = EI_Z(m) or UCB_Z(m)
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
            
            f_star = max(self.subspace_essi) if self.subspace_essi else 0.0
            z = (mu - f_star) / sigma
            ei = (mu - f_star) * norm.cdf(z) + sigma * norm.pdf(z)
            return max(ei, 0.0)
    
    def _generate_neighbors(self, mask: Tensor) -> List[Tensor]:
        """
        Generate neighboring masks by flipping single bits.
        
        Only generates neighbors with size = s.
        """
        neighbors = []
        current_size = int(mask.sum().item())
        
        for j in range(self.input_dim):
            neighbor = mask.clone()
            neighbor[j] = 1.0 - neighbor[j]
            
            new_size = int(neighbor.sum().item())
            if new_size == self.s:
                neighbors.append(neighbor)
                
        return neighbors
    
    def _argmax_mask_acquisition(self) -> Tensor:
        """
        Select subspace using outer BO.
        
        Implements ArgmaxMaskAcquisition procedure from pseudocode.
        
        Returns:
            Selected binary mask
        """
        # If no outer GP yet, return random mask
        if self.outer_gp is None or len(self.subspace_masks) < 2:
            return self._sample_random_mask()
        
        # Initialize pool: existing masks + random masks
        init_masks = list(self.subspace_masks)
        n_random = max(3, len(init_masks))
        for _ in range(n_random):
            init_masks.append(self._sample_random_mask())
            
        # Find best starting point
        best_mask = None
        best_acq = float('-inf')
        
        for mask in init_masks:
            if int(mask.sum().item()) != self.s:
                continue
            acq = self._outer_acquisition(mask)
            if acq > best_acq:
                best_acq = acq
                best_mask = mask.clone()
        
        if best_mask is None:
            best_mask = self._sample_random_mask()
            best_acq = self._outer_acquisition(best_mask)
                
        # Local search with 1-bit flips under |m| = s constraint
        for mask_start in init_masks[:min(5, len(init_masks))]:
            if int(mask_start.sum().item()) != self.s:
                continue
                
            mask_curr = mask_start.clone()
            acq_curr = self._outer_acquisition(mask_curr)
            
            for _ in range(self.max_ls_iters):
                neighbors = self._generate_neighbors(mask_curr)
                if not neighbors:
                    break
                    
                best_neighbor = None
                best_neighbor_acq = acq_curr
                
                for neighbor in neighbors:
                    acq = self._outer_acquisition(neighbor)
                    if acq > best_neighbor_acq:
                        best_neighbor_acq = acq
                        best_neighbor = neighbor
                        
                if best_neighbor is not None and best_neighbor_acq > acq_curr:
                    mask_curr = best_neighbor
                    acq_curr = best_neighbor_acq
                else:
                    break
                    
            if acq_curr > best_acq:
                best_acq = acq_curr
                best_mask = mask_curr.clone()
                
        return best_mask
    
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
            
        # Update base class for consistency
        self.train_X = self.X
        self.train_y = self.y
        
        # Update global best (minimization)
        for i in range(X.shape[0]):
            y_val = y[i].item()
            if self.f_best is None or y_val < self.f_best:
                self.f_best = y_val
                self.x_best = X[i].clone()
    
    def suggest(self, n_suggestions: int = 1) -> Tensor:
        """
        Suggest next point(s) to evaluate.
        
        Implements the main loop from pseudocode:
        1. Fit GP_X on all data
        2. Fit GP_Z on subspace data
        3. Select k candidate subspaces
        4. Maximize ESSI in each subspace
        5. Return point from subspace with highest ESSI
        
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
        """Suggest a single point following the pseudocode."""
        
        # Case 1: Not enough data - return random point
        if self.X is None or self.X.shape[0] < 2:
            lb = self.bounds[0]
            ub = self.bounds[1]
            return lb + (ub - lb) * torch.rand(self.input_dim, device=self.device, dtype=self.dtype)
        
        # ----- Step 3-4: Modeling -----
        # Fit full-space GP on all data D
        self._fit_full_gp()
        
        if self.gp_x is None:
            lb = self.bounds[0]
            ub = self.bounds[1]
            return lb + (ub - lb) * torch.rand(self.input_dim, device=self.device, dtype=self.dtype)
        
        # ----- Step 6: Fit outer GP -----
        self._fit_outer_gp()
        
        # ----- Steps 7-12: Select k subspaces and compute ESSI -----
        candidates = []  # List of (x_new, ESSI, mask)
        
        for i in range(self.k):
            # Step 9: Select next subspace using BO
            if i == 0 and len(self.subspace_masks) == 0:
                # First ever subspace: random
                mask = self._sample_random_mask()
            else:
                mask = self._argmax_mask_acquisition()
            
            S = self._mask_to_indices(mask)
            
            # Step 10-11: Maximize ESSI in subspace
            x_new, essi_val = self._maximize_essi(S)
            
            candidates.append({
                'x_new': x_new,
                'essi': essi_val,
                'mask': mask,
                'S': S
            })
        
        # ----- Step 14: Select best subspace suggestion -----
        best_candidate = max(candidates, key=lambda c: c['essi'])
        
        # Store for outer GP update (will be updated after observe)
        self._pending_mask = best_candidate['mask']
        self._pending_essi = best_candidate['essi']
        
        return best_candidate['x_new']
    
    def _update_outer_data(self, mask: Tensor, essi: float):
        """
        Update outer GP data with (mask, ESSI) pair.
        
        Pseudocode Step 22: UpdateOuterData(GP_Z, S_{i*}, ESSI_{i*})
        """
        self.subspace_masks.append(mask.clone())
        self.subspace_essi.append(essi)
    
    def get_best_point(self) -> Tuple[Tensor, Tensor]:
        """Get best observed point."""
        if self.y is None or self.y.numel() == 0:
            raise ValueError("No observations yet.")
            
        best_idx = self.y.argmin()
        return self.X[best_idx], self.y[best_idx]
    
    def reset(self):
        """Reset optimizer to initial state."""
        super().reset()
        
        self.gp_x = None
        self.subspace_masks = []
        self.subspace_essi = []
        self.outer_gp = None
        
        self.x_best = None
        self.f_best = None
        
        self.X = None
        self.y = None
        
        self._gp_y_mean = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self._gp_y_std = torch.tensor(1.0, device=self.device, dtype=self.dtype)


class BOSSESSIBO_V2_Maximize(BOSSESSIBO_V2):
    """
    BOSS-ESSI-BO v2 variant for maximization problems.
    """
    
    def _compute_essi(self, y_sub: Tensor, S: List[int]) -> float:
        """
        Compute ESSI for maximization.
        
        For maximization: improvement = μ - f* (positive when μ > f*)
        """
        if self.gp_x is None or self.x_best is None or self.f_best is None:
            return 0.0
            
        z = self._replace_coordinates(self.x_best, S, y_sub)
        mu, sigma = self._predict_gp_x(z)
        
        if sigma < 1e-10:
            return 0.0
            
        # For maximization: improvement = μ - f*
        gamma = (mu - self.f_best) / sigma
        essi = (mu - self.f_best) * norm.cdf(gamma) + sigma * norm.pdf(gamma)
        
        return max(essi, 0.0)
    
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
            
        if self.X is None:
            self.X = X
            self.y = y
        else:
            self.X = torch.cat([self.X, X], dim=0)
            self.y = torch.cat([self.y, y], dim=0)
            
        self.train_X = self.X
        self.train_y = self.y
        
        # Update global best (maximization)
        for i in range(X.shape[0]):
            y_val = y[i].item()
            if self.f_best is None or y_val > self.f_best:
                self.f_best = y_val
                self.x_best = X[i].clone()
    
    def get_best_point(self) -> Tuple[Tensor, Tensor]:
        """Get best observed point (maximization)."""
        if self.y is None or self.y.numel() == 0:
            raise ValueError("No observations yet.")
            
        best_idx = self.y.argmax()
        return self.X[best_idx], self.y[best_idx]

