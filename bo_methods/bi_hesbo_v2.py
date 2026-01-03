"""
BiHeSBO-UCB v2: Bilevel HeSBO with Relaxed Embedding Search.

This version uses continuous relaxation for embedding optimization:
1. Outer level: Optimize continuous parameter η ∈ [0,1]^D using GP-UCB
2. Inner level: Run HeSBO-style BO with embedding S = RelaxToEmbedding(η)

Key innovations:
- Continuous parameterization of embedding via η
- GP-UCB for embedding parameter optimization
- Embedding quality metric combining GP fit and collision penalty

Algorithm Overview:
1. Initialize η_1, convert to embedding S_1, evaluate quality J_1
2. For each outer iteration:
   a. Select η_t by maximizing UCB over η-space
   b. Convert to embedding S_t = RelaxToEmbedding(η_t)
   c. Run T_inner BO steps with S_t
   d. Evaluate embedding quality J_t
   e. Update outer GP on (η, J) pairs

Reference:
    Based on HeSBO: Nayebi et al. "A Framework for Bayesian Optimization 
    in Embedded Subspaces" ICML 2019.
"""

import math
from typing import Optional, Tuple, List, Dict, Any
import torch
from torch import Tensor
import numpy as np

from gpytorch.kernels import ScaleKernel, MaternKernel, RBFKernel
from gpytorch.constraints import GreaterThan
from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf

from .base import BaseOptimizer
from .utils import DEVICE, DTYPE, ensure_tensor, clamp_to_bounds


# =============================================================================
# Relaxed Embedding Functions
# =============================================================================

def relax_to_embedding(
    eta: Tensor,
    d: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """
    Convert continuous parameter η to count-sketch embedding matrix S.
    
    Algorithm:
    1. Sort dimensions by η_i (importance)
    2. Assign top d dimensions to distinct buckets
    3. Assign remaining dimensions probabilistically
    
    Args:
        eta: Embedding parameter η ∈ [0,1]^D
        d: Low-dimensional embedding dimension
        device: Device for computation
        dtype: Data type
        
    Returns:
        S: Embedding matrix of shape (D, d)
    """
    D = eta.shape[0]
    S = torch.zeros(D, d, device=device, dtype=dtype)
    
    # Step 1: Sort dimensions by η_i (descending - higher = more important)
    _, perm = torch.sort(eta, descending=True)
    
    # Step 2: Assign top d dimensions to distinct buckets
    for j in range(min(d, D)):
        i = perm[j].item()
        sign = 1.0 if torch.rand(1).item() > 0.5 else -1.0
        S[i, j] = sign
    
    # Step 3: Assign remaining dimensions probabilistically
    if D > d:
        # Get η values of top d dimensions (bucket representatives)
        top_eta = eta[perm[:d]]
        
        for rank in range(d, D):
            i = perm[rank].item()
            eta_i = eta[i]
            
            # Compute similarity to each bucket's representative
            # Higher similarity = more likely to be assigned to that bucket
            similarities = 1.0 - torch.abs(top_eta - eta_i)
            similarities = torch.clamp(similarities, min=0.01)  # Avoid zero
            probs = similarities / similarities.sum()
            
            # Sample bucket
            j = torch.multinomial(probs, 1).item()
            sign = 1.0 if torch.rand(1).item() > 0.5 else -1.0
            S[i, j] = sign
    
    return S


def project_high_to_low(x_high: Tensor, S: Tensor) -> Tensor:
    """
    Project high-dimensional point to low-dimensional space.
    
    y = argmin_y ||Sy - x||^2 = (S^T S)^{-1} S^T x
    
    For count-sketch, this is approximately: y = S^T x (normalized)
    
    Args:
        x_high: High-dim point (D,) or (n, D)
        S: Embedding matrix (D, d)
        
    Returns:
        y_low: Low-dim point (d,) or (n, d)
    """
    squeeze = False
    if x_high.dim() == 1:
        x_high = x_high.unsqueeze(0)
        squeeze = True
    
    # Compute S^T x
    y_low = torch.mm(x_high, S)  # (n, d)
    
    # Normalize by bucket sizes (number of non-zero entries per column)
    bucket_sizes = (S != 0).float().sum(dim=0)  # (d,)
    bucket_sizes = torch.clamp(bucket_sizes, min=1.0)
    y_low = y_low / bucket_sizes.unsqueeze(0)
    
    if squeeze:
        y_low = y_low.squeeze(0)
    
    return y_low


def project_low_to_high(y_low: Tensor, S: Tensor, bounds: Tensor) -> Tensor:
    """
    Project low-dimensional point to high-dimensional space.
    
    x = S * y (element-wise for each row)
    
    Args:
        y_low: Low-dim point (d,) or (n, d)
        S: Embedding matrix (D, d)
        bounds: Original bounds (2, D)
        
    Returns:
        x_high: High-dim point (D,) or (n, D)
    """
    squeeze = False
    if y_low.dim() == 1:
        y_low = y_low.unsqueeze(0)
        squeeze = True
    
    # x = S * y
    x_high = torch.mm(y_low, S.T)  # (n, D)
    
    # Denormalize to original bounds
    lb = bounds[0]
    ub = bounds[1]
    x_high = lb + (x_high + 1) / 2 * (ub - lb)  # Map from [-1,1] to bounds
    
    # Clamp to bounds
    x_high = clamp_to_bounds(x_high, bounds)
    
    if squeeze:
        x_high = x_high.squeeze(0)
    
    return x_high


# =============================================================================
# Embedding Quality Evaluation
# =============================================================================

def evaluate_embedding_quality(
    S: Tensor,
    X_high: Tensor,
    f_values: Tensor,
    device: torch.device,
    dtype: torch.dtype,
    collision_weight: float = 0.1,
) -> float:
    """
    Evaluate embedding quality using GP fit + collision penalty.
    
    J = J_fit + λ * J_coll
    
    Args:
        S: Embedding matrix (D, d)
        X_high: High-dim data (n, D)
        f_values: Objective values (n, 1)
        device: Device
        dtype: Data type
        collision_weight: Weight for collision penalty (λ)
        
    Returns:
        J: Embedding quality score (higher is better)
    """
    n = X_high.shape[0]
    if n < 3:
        return 0.0
    
    try:
        # Project data to low-dim
        Y_low = project_high_to_low(X_high, S)
        
        # Normalize Y_low to [0, 1]
        Y_min = Y_low.min(dim=0).values
        Y_max = Y_low.max(dim=0).values
        Y_range = Y_max - Y_min + 1e-8
        Y_norm = (Y_low - Y_min) / Y_range
        Y_norm = torch.clamp(Y_norm, 0.0, 1.0)
        
        # Standardize f
        f_mean = f_values.mean()
        f_std = f_values.std()
        if f_std < 1e-6:
            f_std = torch.tensor(1.0, device=device, dtype=dtype)
        f_std_vals = (f_values - f_mean) / f_std
        
        # Fit GP
        model = SingleTaskGP(
            train_X=Y_norm,
            train_Y=f_std_vals,
        ).to(device=device, dtype=dtype)
        
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        
        # Compute GP marginal likelihood as J_fit
        model.eval()
        with torch.no_grad():
            output = model(Y_norm)
            J_fit = mll(output, f_std_vals.squeeze(-1)).item()
        
        # Compute collision penalty J_coll
        # Extract ARD lengthscales if available
        try:
            if hasattr(model.covar_module.base_kernel, 'lengthscale'):
                lengthscales = model.covar_module.base_kernel.lengthscale.squeeze()
                weights = 1.0 / (lengthscales ** 2 + 1e-8)
            else:
                weights = torch.ones(S.shape[1], device=device, dtype=dtype)
        except:
            weights = torch.ones(S.shape[1], device=device, dtype=dtype)
        
        # Collision penalty: dimensions in same bucket
        J_coll = 0.0
        D, d = S.shape
        for j in range(d):
            # Find dimensions assigned to bucket j
            bucket_dims = (S[:, j] != 0).nonzero(as_tuple=True)[0]
            if len(bucket_dims) > 1:
                # Penalty for collision
                w_j = weights[j].item() if j < len(weights) else 1.0
                J_coll -= len(bucket_dims) * w_j
        
        J = J_fit + collision_weight * J_coll
        return J
    
    except Exception:
        return -float('inf')


# =============================================================================
# BiHeSBO_V2 Main Class
# =============================================================================

class BiHeSBO_V2(BaseOptimizer):
    """
    BiHeSBO-UCB v2: Bilevel HeSBO with Relaxed Embedding Search.
    
    Uses continuous relaxation η ∈ [0,1]^D to parameterize embeddings,
    and GP-UCB to optimize over η-space.
    
    Args:
        input_dim: High-dimensional input space (D)
        low_dim: Low-dimensional embedding dimension (d)
        bounds: Bounds tensor (2, D)
        T_outer: Number of outer iterations (embedding updates)
        T_inner: Number of inner BO steps per embedding
        outer_beta: UCB beta for outer optimization
        collision_weight: Weight for collision penalty
        acq_func: Acquisition function for inner BO
        device: Device for computation
        dtype: Data type
    """
    
    def __init__(
        self,
        input_dim: int,
        low_dim: int,
        bounds: Tensor,
        T_outer: int = 10,
        T_inner: int = 10,
        outer_beta: float = 2.0,
        collision_weight: float = 0.1,
        acq_func: str = "EI",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        _device = device or DEVICE
        _dtype = dtype or DTYPE
        
        super().__init__(
            input_dim=input_dim,
            bounds=bounds,
            acq_func=acq_func,
            device=_device,
            dtype=_dtype,
        )
        
        self.low_dim = max(1, min(low_dim, input_dim))
        self.T_outer = T_outer
        self.T_inner = T_inner
        self.outer_beta = outer_beta
        self.collision_weight = collision_weight
        
        # Low-dim bounds (unit cube)
        self.low_bounds = torch.stack([
            -torch.ones(self.low_dim, device=self.device, dtype=self.dtype),
            torch.ones(self.low_dim, device=self.device, dtype=self.dtype)
        ])
        
        # Eta bounds (unit cube)
        self.eta_bounds = torch.stack([
            torch.zeros(input_dim, device=self.device, dtype=self.dtype),
            torch.ones(input_dim, device=self.device, dtype=self.dtype)
        ])
        
        # Initialize η_1 (random or 0.5)
        self.current_eta = torch.rand(input_dim, device=self.device, dtype=self.dtype)
        
        # Current embedding
        self.current_S = relax_to_embedding(
            self.current_eta, self.low_dim, self.device, self.dtype
        )
        
        # Outer GP data: {(η_i, J_i)}
        self.eta_data: List[Tensor] = []
        self.J_data: List[float] = []
        
        # Outer GP model
        self.outer_gp: Optional[SingleTaskGP] = None
        
        # Inner GP model
        self.inner_gp: Optional[SingleTaskGP] = None
        
        # Training data in high-dim
        self.X_high: Optional[Tensor] = None
        self.f_values: Optional[Tensor] = None
        
        # Training data in low-dim (current embedding)
        self.Y_low: Optional[Tensor] = None
        
        # Y statistics for inner GP
        self._f_mean = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self._f_std = torch.tensor(1.0, device=self.device, dtype=self.dtype)
        
        # Iteration counters
        self._outer_iter = 0
        self._inner_step = 0
        self._total_evals = 0
        
        # Best found
        self.f_best: Optional[float] = None
        self.x_best: Optional[Tensor] = None
        
        # Pending suggestion tracking
        self._pending_y_low: Optional[Tensor] = None
    
    def _update_f_statistics(self):
        """Update f mean and std for standardization."""
        if self.f_values is not None and self.f_values.numel() > 0:
            self._f_mean = self.f_values.mean()
            self._f_std = self.f_values.std()
            if self._f_std < 1e-6:
                self._f_std = torch.tensor(1.0, device=self.device, dtype=self.dtype)
    
    def _standardize_f(self, f: Tensor) -> Tensor:
        """Standardize f to mean=0, std=1."""
        return (f - self._f_mean) / (self._f_std + 1e-8)
    
    def _fit_outer_gp(self):
        """Fit outer GP on (η, J) data."""
        if len(self.eta_data) < 2:
            return
        
        # Stack eta data
        eta_tensor = torch.stack(self.eta_data)  # (n, D)
        J_tensor = torch.tensor(
            self.J_data, device=self.device, dtype=self.dtype
        ).unsqueeze(-1)  # (n, 1)
        
        # Normalize J
        J_mean = J_tensor.mean()
        J_std = J_tensor.std()
        if J_std < 1e-6:
            J_std = torch.tensor(1.0, device=self.device, dtype=self.dtype)
        J_norm = (J_tensor - J_mean) / J_std
        
        # Use RBF kernel for smooth optimization landscape
        covar_module = ScaleKernel(
            RBFKernel(ard_num_dims=self.input_dim)
        )
        
        self.outer_gp = SingleTaskGP(
            train_X=eta_tensor,
            train_Y=J_norm,
            covar_module=covar_module,
        ).to(device=self.device, dtype=self.dtype)
        
        mll = ExactMarginalLogLikelihood(self.outer_gp.likelihood, self.outer_gp)
        try:
            fit_gpytorch_mll(mll)
        except:
            pass
    
    def _select_next_eta(self) -> Tensor:
        """Select next η by maximizing UCB over η-space."""
        if self.outer_gp is None or len(self.eta_data) < 2:
            # Random η
            return torch.rand(self.input_dim, device=self.device, dtype=self.dtype)
        
        # UCB acquisition
        acq = UpperConfidenceBound(self.outer_gp, beta=self.outer_beta)
        
        try:
            candidates, _ = optimize_acqf(
                acq,
                bounds=self.eta_bounds,
                q=1,
                num_restarts=5,
                raw_samples=64,
            )
            return candidates.squeeze(0)
        except:
            return torch.rand(self.input_dim, device=self.device, dtype=self.dtype)
    
    def _fit_inner_gp(self):
        """Fit inner GP on low-dim data."""
        if self.Y_low is None or self.Y_low.shape[0] < 2:
            return
        
        self._update_f_statistics()
        
        # Normalize Y_low to [0, 1]
        Y_min = self.Y_low.min(dim=0).values
        Y_max = self.Y_low.max(dim=0).values
        Y_range = Y_max - Y_min + 1e-8
        Y_norm = (self.Y_low - Y_min) / Y_range
        Y_norm = torch.clamp(Y_norm, 0.0, 1.0)
        
        f_std = self._standardize_f(self.f_values)
        
        covar_module = ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=self.low_dim,
                lengthscale_constraint=GreaterThan(1e-4),
            ),
            outputscale_constraint=GreaterThan(1e-4),
        )
        
        self.inner_gp = SingleTaskGP(
            train_X=Y_norm,
            train_Y=f_std,
            covar_module=covar_module,
        ).to(device=self.device, dtype=self.dtype)
        
        mll = ExactMarginalLogLikelihood(self.inner_gp.likelihood, self.inner_gp)
        try:
            fit_gpytorch_mll(mll)
        except:
            pass
    
    def _select_next_y_low(self) -> Tensor:
        """Select next point in low-dim space."""
        if self.inner_gp is None or self.Y_low is None or self.Y_low.shape[0] < 2:
            # Random in low-dim
            return 2 * torch.rand(self.low_dim, device=self.device, dtype=self.dtype) - 1
        
        # Normalized bounds for optimization
        unit_bounds = torch.stack([
            torch.zeros(self.low_dim, device=self.device, dtype=self.dtype),
            torch.ones(self.low_dim, device=self.device, dtype=self.dtype)
        ])
        
        f_std = self._standardize_f(self.f_values)
        best_f = f_std.max()  # For minimization
        
        if self.acq_func == "UCB":
            acq = UpperConfidenceBound(self.inner_gp, beta=0.25)
        else:
            acq = ExpectedImprovement(self.inner_gp, best_f=best_f)
        
        try:
            candidates, _ = optimize_acqf(
                acq,
                bounds=unit_bounds,
                q=1,
                num_restarts=5,
                raw_samples=32,
            )
            
            # Denormalize
            Y_min = self.Y_low.min(dim=0).values
            Y_max = self.Y_low.max(dim=0).values
            Y_range = Y_max - Y_min + 1e-8
            y_low = Y_min + candidates.squeeze(0) * Y_range
            
            return y_low
        except:
            return 2 * torch.rand(self.low_dim, device=self.device, dtype=self.dtype) - 1
    
    def _update_embedding(self):
        """Update embedding at end of inner loop."""
        if self.X_high is None or self.X_high.shape[0] < 3:
            return
        
        # Evaluate current embedding quality
        J = evaluate_embedding_quality(
            self.current_S,
            self.X_high,
            self.f_values,
            self.device,
            self.dtype,
            self.collision_weight,
        )
        
        # Add to outer GP data
        self.eta_data.append(self.current_eta.clone())
        self.J_data.append(J)
        
        # Fit outer GP
        self._fit_outer_gp()
        
        # Select next η
        self.current_eta = self._select_next_eta()
        
        # Convert to embedding
        self.current_S = relax_to_embedding(
            self.current_eta, self.low_dim, self.device, self.dtype
        )
        
        # Re-project all data to new embedding
        self.Y_low = project_high_to_low(self.X_high, self.current_S)
        
        # Reset inner GP
        self.inner_gp = None
        
        # Increment outer iteration
        self._outer_iter += 1
        self._inner_step = 0
    
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
        
        # Store in high-dim dataset
        if self.X_high is None:
            self.X_high = X
            self.f_values = y
        else:
            self.X_high = torch.cat([self.X_high, X], dim=0)
            self.f_values = torch.cat([self.f_values, y], dim=0)
        
        # Update base class
        self.train_X = self.X_high
        self.train_y = self.f_values
        
        # Project to low-dim
        Y_new = project_high_to_low(X, self.current_S)
        if self.Y_low is None:
            self.Y_low = Y_new
        else:
            self.Y_low = torch.cat([self.Y_low, Y_new], dim=0)
        
        # Update best (minimization)
        for i in range(X.shape[0]):
            f_val = y[i].item()
            if self.f_best is None or f_val < self.f_best:
                self.f_best = f_val
                self.x_best = X[i].clone()
            
            self._total_evals += 1
            self._inner_step += 1
        
        # Check if we should update embedding
        if self._inner_step >= self.T_inner:
            self._update_embedding()
    
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
        
        # Not enough data: random point
        if self.X_high is None or self.X_high.shape[0] < 2:
            lb = self.bounds[0]
            ub = self.bounds[1]
            return lb + (ub - lb) * torch.rand(self.input_dim, device=self.device, dtype=self.dtype)
        
        # Fit inner GP
        self._fit_inner_gp()
        
        # Select next point in low-dim
        y_low = self._select_next_y_low()
        
        # Project to high-dim
        x_high = project_low_to_high(y_low, self.current_S, self.bounds)
        
        # Store for potential future use
        self._pending_y_low = y_low
        
        return x_high
    
    def get_best_point(self) -> Tuple[Tensor, Tensor]:
        """Get best observed point."""
        if self.f_values is None or self.f_values.numel() == 0:
            raise ValueError("No observations yet.")
        
        best_idx = self.f_values.argmin()
        return self.X_high[best_idx], self.f_values[best_idx]
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about embedding optimization."""
        return {
            'outer_iter': self._outer_iter,
            'inner_step': self._inner_step,
            'total_evals': self._total_evals,
            'n_embeddings_tried': len(self.eta_data),
            'J_values': self.J_data.copy() if self.J_data else [],
            'current_eta_mean': self.current_eta.mean().item(),
            'current_eta_std': self.current_eta.std().item(),
        }
    
    def reset(self):
        """Reset optimizer to initial state."""
        super().reset()
        
        self.current_eta = torch.rand(self.input_dim, device=self.device, dtype=self.dtype)
        self.current_S = relax_to_embedding(
            self.current_eta, self.low_dim, self.device, self.dtype
        )
        
        self.eta_data = []
        self.J_data = []
        self.outer_gp = None
        self.inner_gp = None
        
        self.X_high = None
        self.f_values = None
        self.Y_low = None
        
        self._f_mean = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self._f_std = torch.tensor(1.0, device=self.device, dtype=self.dtype)
        
        self._outer_iter = 0
        self._inner_step = 0
        self._total_evals = 0
        
        self.f_best = None
        self.x_best = None
        self._pending_y_low = None


class BiHeSBO_V2_Maximize(BiHeSBO_V2):
    """
    BiHeSBO-UCB v2 variant for maximization problems.
    """
    
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
        
        if self.X_high is None:
            self.X_high = X
            self.f_values = y
        else:
            self.X_high = torch.cat([self.X_high, X], dim=0)
            self.f_values = torch.cat([self.f_values, y], dim=0)
        
        self.train_X = self.X_high
        self.train_y = self.f_values
        
        Y_new = project_high_to_low(X, self.current_S)
        if self.Y_low is None:
            self.Y_low = Y_new
        else:
            self.Y_low = torch.cat([self.Y_low, Y_new], dim=0)
        
        # Update best (maximization)
        for i in range(X.shape[0]):
            f_val = y[i].item()
            if self.f_best is None or f_val > self.f_best:
                self.f_best = f_val
                self.x_best = X[i].clone()
            
            self._total_evals += 1
            self._inner_step += 1
        
        if self._inner_step >= self.T_inner:
            self._update_embedding()
    
    def get_best_point(self) -> Tuple[Tensor, Tensor]:
        """Get best observed point (maximization)."""
        if self.f_values is None or self.f_values.numel() == 0:
            raise ValueError("No observations yet.")
        
        best_idx = self.f_values.argmax()
        return self.X_high[best_idx], self.f_values[best_idx]

