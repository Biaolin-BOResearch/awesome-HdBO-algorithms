"""
AS-HeSBO: Adaptive-Sketch Hashing-Enhanced Subspace Bayesian Optimization.

This method extends HeSBO by adaptively updating the hash embedding during
optimization. Instead of fixing the embedding once at initialization, it
periodically evaluates candidate embeddings and replaces the current one
if a better quality score is achieved.

Key Idea:
- Embedding matrices are still count-sketch style (hash + sign)
- But instead of fixing once, we do periodic "search and replace"
- Candidates are sampled from the same HeSBO distribution (theoretical guarantee)
- A quality metric on observed data is used to filter/select embeddings
- This provably increases P(S ∈ G) compared to vanilla HeSBO

Algorithm Overview:
1. Initialize: Sample S_1 ~ μ (HeSBO distribution)
2. For each epoch k:
   a. Run L BO steps with current embedding S_curr
   b. Sample candidate S' ~ μ
   c. Compute quality scores J(S_curr) and J(S')
   d. If J(S') >= J(S_curr), replace S_curr with S'
3. Output best point found

Quality Metrics:
- GP marginal likelihood on re-embedded data
- Leave-one-out cross-validation error
- Prediction MSE on held-out points

Reference:
    Based on HeSBO: Nayebi et al. "A Framework for Bayesian Optimization 
    in Embedded Subspaces" ICML 2019.
"""

import math
from typing import Optional, Tuple, List, Dict, Any, Literal
import torch
from torch import Tensor
import numpy as np

from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.constraints import Interval, GreaterThan
from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf

from .base import BaseOptimizer
from .utils import DEVICE, DTYPE, ensure_tensor, clamp_to_bounds


# =============================================================================
# Count-Sketch Embedding
# =============================================================================

class CountSketchEmbedding:
    """
    Count-Sketch style embedding matrix S ∈ R^{D×d}.
    
    Implements the projection:
    - Forward: y[h[i]] += s[i] * x[i] for all i (high to low)
    - Inverse: x[i] = s[i] * y[h[i]] for all i (low to high)
    
    Where:
    - h: [D] -> [d] is a hash function
    - s: [D] -> {-1, +1} is a sign function
    """
    
    def __init__(
        self,
        input_dim: int,
        low_dim: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize a count-sketch embedding.
        
        Args:
            input_dim: High-dimensional input space (D)
            low_dim: Low-dimensional embedding space (d)
            device: Device for computation
            dtype: Data type
            seed: Random seed for reproducibility
        """
        self.input_dim = input_dim
        self.low_dim = low_dim
        self.device = device or DEVICE
        self.dtype = dtype or DTYPE
        
        if seed is not None:
            torch.manual_seed(seed)
        
        # Hash function: h[i] ∈ {0, 1, ..., d-1}
        self.h = torch.randint(0, low_dim, (input_dim,), device=self.device)
        
        # Sign function: s[i] ∈ {-1, +1}
        self.sigma = (
            (torch.randint(0, 2, (input_dim,), device=self.device) * 2 - 1)
            .to(dtype=self.dtype)
        )
    
    def high_to_low(self, X: Tensor) -> Tensor:
        """
        Project high-dimensional points to low-dimensional space.
        
        y[h[i]] += s[i] * x[i]
        
        Args:
            X: Points in high-dim space (n, D) or (D,)
            
        Returns:
            Points in low-dim space (n, d) or (d,)
        """
        squeeze = False
        if X.dim() == 1:
            X = X.unsqueeze(0)
            squeeze = True
        
        X = X.to(device=self.device, dtype=self.dtype)
        batch_size = X.shape[0]
        
        Y = torch.zeros((batch_size, self.low_dim), device=self.device, dtype=self.dtype)
        
        for i in range(self.input_dim):
            bucket_idx = self.h[i]
            Y[:, bucket_idx] += X[:, i] * self.sigma[i]
        
        if squeeze:
            Y = Y.squeeze(0)
        
        return Y
    
    def low_to_high(self, Y: Tensor, bounds: Optional[Tensor] = None) -> Tensor:
        """
        Project low-dimensional points back to high-dimensional space.
        
        x[i] = s[i] * y[h[i]]
        
        Args:
            Y: Points in low-dim space (n, d) or (d,)
            bounds: Optional bounds for clamping (2, D)
            
        Returns:
            Points in high-dim space (n, D) or (D,)
        """
        squeeze = False
        if Y.dim() == 1:
            Y = Y.unsqueeze(0)
            squeeze = True
        
        Y = Y.to(device=self.device, dtype=self.dtype)
        batch_size = Y.shape[0]
        
        X = torch.zeros((batch_size, self.input_dim), device=self.device, dtype=self.dtype)
        
        for i in range(self.input_dim):
            bucket_idx = self.h[i]
            X[:, i] = Y[:, bucket_idx] * self.sigma[i]
        
        if bounds is not None:
            X = clamp_to_bounds(X, bounds)
        
        if squeeze:
            X = X.squeeze(0)
        
        return X
    
    def re_embed(self, X_high: Tensor) -> Tensor:
        """
        Re-embed high-dimensional points using this embedding.
        
        For re-embedding observed data under a new embedding:
        y' = argmin_y ||S'y - x|| 
        
        For count-sketch, this is simply the forward projection.
        
        Args:
            X_high: Points in high-dim space (n, D)
            
        Returns:
            Re-embedded points in low-dim space (n, d)
        """
        return self.high_to_low(X_high)
    
    def clone(self) -> 'CountSketchEmbedding':
        """Create a copy of this embedding."""
        new_emb = CountSketchEmbedding(
            input_dim=self.input_dim,
            low_dim=self.low_dim,
            device=self.device,
            dtype=self.dtype,
        )
        new_emb.h = self.h.clone()
        new_emb.sigma = self.sigma.clone()
        return new_emb


# =============================================================================
# Quality Score Functions
# =============================================================================

def compute_gp_marginal_likelihood(
    Y_low: Tensor,
    f: Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    """
    Compute GP marginal log-likelihood on low-dim data.
    
    Higher is better (indicates better fit).
    
    Args:
        Y_low: Low-dim points (n, d)
        f: Objective values (n, 1)
        device: Device
        dtype: Data type
        
    Returns:
        Marginal log-likelihood (higher is better)
    """
    if Y_low.shape[0] < 3:
        return float('-inf')
    
    try:
        # Normalize data
        Y_norm = Y_low.clone()
        Y_min = Y_norm.min(dim=0).values
        Y_max = Y_norm.max(dim=0).values
        Y_range = Y_max - Y_min + 1e-8
        Y_norm = (Y_norm - Y_min) / Y_range
        
        f_mean = f.mean()
        f_std = f.std()
        if f_std < 1e-6:
            f_std = torch.tensor(1.0, device=device, dtype=dtype)
        f_norm = (f - f_mean) / f_std
        
        # Fit GP
        covar_module = ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=Y_low.shape[1],
            ),
        )
        
        model = SingleTaskGP(
            train_X=Y_norm,
            train_Y=f_norm,
            covar_module=covar_module,
        ).to(device=device, dtype=dtype)
        
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        
        # Compute marginal likelihood (without optimization)
        model.eval()
        with torch.no_grad():
            output = model(Y_norm)
            ll = mll(output, f_norm.squeeze(-1))
        
        return ll.item()
    
    except Exception:
        return float('-inf')


def compute_loocv_error(
    Y_low: Tensor,
    f: Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    """
    Compute leave-one-out cross-validation error.
    
    Lower is better (indicates better generalization).
    Returns negative error so higher is better (consistent with other metrics).
    
    Args:
        Y_low: Low-dim points (n, d)
        f: Objective values (n, 1)
        device: Device
        dtype: Data type
        
    Returns:
        Negative LOOCV MSE (higher is better)
    """
    n = Y_low.shape[0]
    if n < 4:
        return float('-inf')
    
    try:
        # Normalize
        Y_norm = Y_low.clone()
        Y_min = Y_norm.min(dim=0).values
        Y_max = Y_norm.max(dim=0).values
        Y_range = Y_max - Y_min + 1e-8
        Y_norm = (Y_norm - Y_min) / Y_range
        
        f_mean = f.mean()
        f_std = f.std()
        if f_std < 1e-6:
            f_std = torch.tensor(1.0, device=device, dtype=dtype)
        f_norm = (f - f_mean) / f_std
        
        # Leave-one-out CV
        errors = []
        
        for i in range(n):
            # Leave out point i
            mask = torch.ones(n, dtype=torch.bool, device=device)
            mask[i] = False
            
            Y_train = Y_norm[mask]
            f_train = f_norm[mask]
            Y_test = Y_norm[i:i+1]
            f_test = f_norm[i]
            
            # Fit GP on training data
            model = SingleTaskGP(
                train_X=Y_train,
                train_Y=f_train,
            ).to(device=device, dtype=dtype)
            
            # Predict on held-out point
            model.eval()
            with torch.no_grad():
                pred = model(Y_test)
                pred_mean = pred.mean.item()
            
            error = (pred_mean - f_test.item()) ** 2
            errors.append(error)
        
        mse = np.mean(errors)
        return -mse  # Negative so higher is better
    
    except Exception:
        return float('-inf')


def compute_prediction_correlation(
    Y_low: Tensor,
    f: Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    """
    Compute correlation between GP predictions and actual values.
    
    Higher is better.
    
    Args:
        Y_low: Low-dim points (n, d)
        f: Objective values (n, 1)
        device: Device
        dtype: Data type
        
    Returns:
        Correlation coefficient (higher is better)
    """
    n = Y_low.shape[0]
    if n < 4:
        return float('-inf')
    
    try:
        # Normalize
        Y_norm = Y_low.clone()
        Y_min = Y_norm.min(dim=0).values
        Y_max = Y_norm.max(dim=0).values
        Y_range = Y_max - Y_min + 1e-8
        Y_norm = (Y_norm - Y_min) / Y_range
        
        f_mean = f.mean()
        f_std = f.std()
        if f_std < 1e-6:
            return 0.0  # No variance in f
        f_norm = (f - f_mean) / f_std
        
        # Fit GP and get predictions
        model = SingleTaskGP(
            train_X=Y_norm,
            train_Y=f_norm,
        ).to(device=device, dtype=dtype)
        
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        
        model.eval()
        with torch.no_grad():
            pred = model(Y_norm)
            pred_mean = pred.mean
        
        # Compute correlation
        f_flat = f_norm.squeeze(-1)
        corr = torch.corrcoef(torch.stack([pred_mean, f_flat]))[0, 1].item()
        
        if np.isnan(corr):
            return 0.0
        
        return corr
    
    except Exception:
        return float('-inf')


# =============================================================================
# AS-HeSBO Main Class
# =============================================================================

class ASHeSBO(BaseOptimizer):
    """
    Adaptive-Sketch HeSBO: Periodically updates embedding based on quality scores.
    
    This method extends HeSBO by:
    1. Running BO in epochs of L steps each
    2. At each epoch boundary, sampling a candidate embedding
    3. Replacing current embedding if candidate has better quality score
    
    Quality metrics:
    - 'mll': GP marginal log-likelihood
    - 'loocv': Leave-one-out cross-validation (negative MSE)
    - 'corr': Prediction correlation
    
    Args:
        input_dim: High-dimensional input space (D)
        low_dim: Low-dimensional embedding space (d)
        bounds: Bounds tensor (2, D)
        L: Number of BO steps per epoch
        K: Maximum number of epochs (or None for unlimited)
        quality_metric: Which quality metric to use
        n_candidates: Number of candidate embeddings to evaluate per epoch
        acq_func: Acquisition function for inner BO
        device: Device for computation
        dtype: Data type
    """
    
    def __init__(
        self,
        input_dim: int,
        low_dim: int,
        bounds: Tensor,
        L: int = 10,
        K: Optional[int] = None,
        quality_metric: Literal['mll', 'loocv', 'corr'] = 'mll',
        n_candidates: int = 1,
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
        self.L = L  # Steps per epoch
        self.K = K  # Max epochs
        self.quality_metric = quality_metric
        self.n_candidates = n_candidates
        
        # Low-dim bounds (unit cube)
        self.low_bounds = torch.stack([
            torch.zeros(self.low_dim, device=self.device, dtype=self.dtype),
            torch.ones(self.low_dim, device=self.device, dtype=self.dtype)
        ])
        
        # Initialize first embedding S_1 ~ μ
        self.current_embedding = CountSketchEmbedding(
            input_dim=input_dim,
            low_dim=self.low_dim,
            device=self.device,
            dtype=self.dtype,
        )
        
        # Current quality score
        self.current_score: Optional[float] = None
        
        # Training data in high-dim
        self.X_high: Optional[Tensor] = None
        self.f_values: Optional[Tensor] = None
        
        # Training data in low-dim (current embedding)
        self.Y_low: Optional[Tensor] = None
        
        # GP model
        self.gp_model: Optional[SingleTaskGP] = None
        
        # Y statistics for standardization
        self._f_mean = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self._f_std = torch.tensor(1.0, device=self.device, dtype=self.dtype)
        
        # Epoch tracking
        self._step_in_epoch = 0
        self._current_epoch = 1
        self._embedding_updates = 0
        
        # Best found
        self.f_best: Optional[float] = None
        self.x_best: Optional[Tensor] = None
        
        # Statistics
        self.embedding_history: List[Dict[str, Any]] = []
    
    def _compute_quality_score(self, embedding: CountSketchEmbedding) -> float:
        """
        Compute quality score for an embedding using observed data.
        
        Args:
            embedding: The embedding to evaluate
            
        Returns:
            Quality score (higher is better)
        """
        if self.X_high is None or self.X_high.shape[0] < 3:
            return float('-inf')
        
        # Re-embed all observed data under this embedding
        Y_low = embedding.re_embed(self.X_high)
        
        # Clamp to [0, 1] for numerical stability
        Y_low = torch.clamp(Y_low, -10.0, 10.0)
        
        if self.quality_metric == 'mll':
            return compute_gp_marginal_likelihood(
                Y_low, self.f_values, self.device, self.dtype
            )
        elif self.quality_metric == 'loocv':
            return compute_loocv_error(
                Y_low, self.f_values, self.device, self.dtype
            )
        elif self.quality_metric == 'corr':
            return compute_prediction_correlation(
                Y_low, self.f_values, self.device, self.dtype
            )
        else:
            return compute_gp_marginal_likelihood(
                Y_low, self.f_values, self.device, self.dtype
            )
    
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
    
    def _try_update_embedding(self):
        """
        Try to update embedding at epoch boundary.
        
        Sample candidate embeddings and replace current if better.
        """
        if self.X_high is None or self.X_high.shape[0] < 3:
            return
        
        # Compute current embedding score
        self.current_score = self._compute_quality_score(self.current_embedding)
        
        best_candidate = None
        best_score = self.current_score
        
        # Sample n_candidates new embeddings
        for _ in range(self.n_candidates):
            candidate = CountSketchEmbedding(
                input_dim=self.input_dim,
                low_dim=self.low_dim,
                device=self.device,
                dtype=self.dtype,
            )
            
            score = self._compute_quality_score(candidate)
            
            if score > best_score:
                best_score = score
                best_candidate = candidate
        
        # Replace if better candidate found
        if best_candidate is not None:
            self.current_embedding = best_candidate
            self.current_score = best_score
            self._embedding_updates += 1
            
            # Re-embed all data under new embedding
            self.Y_low = self.current_embedding.re_embed(self.X_high)
            
            # Refit GP with new embedding
            self._fit_gp()
            
            self.embedding_history.append({
                'epoch': self._current_epoch,
                'action': 'replaced',
                'old_score': self.current_score,
                'new_score': best_score,
            })
        else:
            self.embedding_history.append({
                'epoch': self._current_epoch,
                'action': 'kept',
                'score': self.current_score,
            })
    
    def _fit_gp(self):
        """Fit GP model on low-dim data."""
        if self.Y_low is None or self.Y_low.shape[0] < 2:
            return
        
        self._update_f_statistics()
        
        # Normalize Y_low to [0, 1]
        Y_norm = self.Y_low.clone()
        Y_min = Y_norm.min(dim=0).values
        Y_max = Y_norm.max(dim=0).values
        Y_range = Y_max - Y_min + 1e-8
        Y_norm = (Y_norm - Y_min) / Y_range
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
        
        self.gp_model = SingleTaskGP(
            train_X=Y_norm,
            train_Y=f_std,
            covar_module=covar_module,
            input_transform=None,
            outcome_transform=None,
        ).to(device=self.device, dtype=self.dtype)
        
        mll = ExactMarginalLogLikelihood(self.gp_model.likelihood, self.gp_model)
        try:
            fit_gpytorch_mll(mll)
        except Exception:
            pass
    
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
        
        # Project to low-dim using current embedding
        Y_new = self.current_embedding.high_to_low(X)
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
        
        # Track epoch progress
        self._step_in_epoch += X.shape[0]
        
        # Check for epoch boundary
        if self._step_in_epoch >= self.L:
            self._step_in_epoch = 0
            self._current_epoch += 1
            
            # Try to update embedding
            self._try_update_embedding()
        
        # Fit GP
        self._fit_gp()
    
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
        
        # Not enough data: random in low-dim, project to high-dim
        if self.gp_model is None or self.Y_low is None or self.Y_low.shape[0] < 2:
            y_low = torch.rand(self.low_dim, device=self.device, dtype=self.dtype)
            
            # Project to high-dim
            lb = self.bounds[0]
            ub = self.bounds[1]
            x_high = self.current_embedding.low_to_high(y_low)
            x_high = lb + x_high * (ub - lb)
            x_high = clamp_to_bounds(x_high, self.bounds)
            
            return x_high
        
        # Get standardized best value
        f_std = self._standardize_f(self.f_values)
        best_f = f_std.max()  # For minimization
        
        # Create acquisition function
        if self.acq_func == "UCB":
            acq = UpperConfidenceBound(self.gp_model, beta=0.25)
        else:  # EI
            acq = ExpectedImprovement(self.gp_model, best_f=best_f)
        
        try:
            # Optimize in low-dim unit cube
            candidates, _ = optimize_acqf(
                acq,
                bounds=self.low_bounds,
                q=1,
                num_restarts=5,
                raw_samples=32,
            )
            y_low = candidates.squeeze(0)
        except Exception:
            y_low = torch.rand(self.low_dim, device=self.device, dtype=self.dtype)
        
        # Project to high-dim
        lb = self.bounds[0]
        ub = self.bounds[1]
        x_high = self.current_embedding.low_to_high(y_low)
        x_high = lb + x_high * (ub - lb)
        x_high = clamp_to_bounds(x_high, self.bounds)
        
        return x_high
    
    def get_best_point(self) -> Tuple[Tensor, Tensor]:
        """Get best observed point."""
        if self.f_values is None or self.f_values.numel() == 0:
            raise ValueError("No observations yet.")
        
        best_idx = self.f_values.argmin()
        return self.X_high[best_idx], self.f_values[best_idx]
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """
        Get statistics about embedding updates.
        
        Returns:
            Dict with embedding statistics
        """
        return {
            'current_epoch': self._current_epoch,
            'embedding_updates': self._embedding_updates,
            'current_score': self.current_score,
            'history': self.embedding_history,
        }
    
    def reset(self):
        """Reset optimizer to initial state."""
        super().reset()
        
        self.current_embedding = CountSketchEmbedding(
            input_dim=self.input_dim,
            low_dim=self.low_dim,
            device=self.device,
            dtype=self.dtype,
        )
        
        self.current_score = None
        self.X_high = None
        self.f_values = None
        self.Y_low = None
        self.gp_model = None
        
        self._f_mean = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self._f_std = torch.tensor(1.0, device=self.device, dtype=self.dtype)
        
        self._step_in_epoch = 0
        self._current_epoch = 1
        self._embedding_updates = 0
        
        self.f_best = None
        self.x_best = None
        
        self.embedding_history = []


class ASHeSBOMaximize(ASHeSBO):
    """
    AS-HeSBO variant for maximization problems.
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
        
        Y_new = self.current_embedding.high_to_low(X)
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
        
        self._step_in_epoch += X.shape[0]
        
        if self._step_in_epoch >= self.L:
            self._step_in_epoch = 0
            self._current_epoch += 1
            self._try_update_embedding()
        
        self._fit_gp()
    
    def _suggest_one(self) -> Tensor:
        """Suggest a single point (maximization uses negated values internally)."""
        
        if self.gp_model is None or self.Y_low is None or self.Y_low.shape[0] < 2:
            y_low = torch.rand(self.low_dim, device=self.device, dtype=self.dtype)
            lb = self.bounds[0]
            ub = self.bounds[1]
            x_high = self.current_embedding.low_to_high(y_low)
            x_high = lb + x_high * (ub - lb)
            x_high = clamp_to_bounds(x_high, self.bounds)
            return x_high
        
        # For maximization, best_f should be the standardized max
        f_std = self._standardize_f(self.f_values)
        best_f = f_std.max()
        
        if self.acq_func == "UCB":
            acq = UpperConfidenceBound(self.gp_model, beta=0.25)
        else:
            acq = ExpectedImprovement(self.gp_model, best_f=best_f)
        
        try:
            candidates, _ = optimize_acqf(
                acq,
                bounds=self.low_bounds,
                q=1,
                num_restarts=5,
                raw_samples=32,
            )
            y_low = candidates.squeeze(0)
        except Exception:
            y_low = torch.rand(self.low_dim, device=self.device, dtype=self.dtype)
        
        lb = self.bounds[0]
        ub = self.bounds[1]
        x_high = self.current_embedding.low_to_high(y_low)
        x_high = lb + x_high * (ub - lb)
        x_high = clamp_to_bounds(x_high, self.bounds)
        
        return x_high
    
    def get_best_point(self) -> Tuple[Tensor, Tensor]:
        """Get best observed point (maximization)."""
        if self.f_values is None or self.f_values.numel() == 0:
            raise ValueError("No observations yet.")
        
        best_idx = self.f_values.argmax()
        return self.X_high[best_idx], self.f_values[best_idx]

