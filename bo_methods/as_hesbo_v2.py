"""
AS-HeSBO v2: Adaptive-Sketch HeSBO with ESSI-based Embedding Search.

This version implements a stronger adaptive embedding strategy:
1. Inner loop: Run L BO steps with current embedding
2. GlobalRehashAndImprove: Sample new embedding + hill-climbing
3. EmbedESSI: Use ESSI as embedding quality metric
4. Replace if improved

Key innovations:
- ESSI-based embedding quality (not just GP likelihood)
- Global rehash from HeSBO distribution μ
- Local hill-climbing on (h, σ) space
- Neighbor generation via bucket changes / sign flips

Algorithm Overview:
1. Initialize S_curr from HeSBO distribution
2. For each epoch:
   a. Run L inner BO steps with S_curr
   b. GlobalRehashAndImprove: sample S_rand ~ μ, hill-climb
   c. Compare J_prop vs J_curr using EmbedESSI
   d. Replace if J_prop >= J_curr

Reference:
    Based on HeSBO: Nayebi et al. "A Framework for Bayesian Optimization 
    in Embedded Subspaces" ICML 2019.
"""

import math
from typing import Optional, Tuple, List, Dict, Any
import torch
from torch import Tensor
import numpy as np
from scipy.stats import norm

from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.constraints import GreaterThan
from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf

from .base import BaseOptimizer
from .utils import DEVICE, DTYPE, ensure_tensor, clamp_to_bounds


# =============================================================================
# Count-Sketch Embedding with (h, σ) representation
# =============================================================================

class HeSBOEmbedding:
    """
    Count-Sketch embedding with explicit (h, σ) representation.
    
    h[i] ∈ {0, 1, ..., d-1}: hash function mapping dimension i to bucket
    σ[i] ∈ {-1, +1}: sign function for dimension i
    """
    
    def __init__(
        self,
        input_dim: int,
        low_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        h: Optional[Tensor] = None,
        sigma: Optional[Tensor] = None,
    ):
        """
        Initialize embedding.
        
        Args:
            input_dim: High-dimensional space (D)
            low_dim: Low-dimensional space (d)
            device: Device
            dtype: Data type
            h: Optional pre-specified hash function
            sigma: Optional pre-specified sign function
        """
        self.input_dim = input_dim
        self.low_dim = low_dim
        self.device = device
        self.dtype = dtype
        
        # Hash function: h[i] ∈ {0, ..., d-1}
        if h is not None:
            self.h = h.clone().to(device=device)
        else:
            self.h = torch.randint(0, low_dim, (input_dim,), device=device)
        
        # Sign function: σ[i] ∈ {-1, +1}
        if sigma is not None:
            self.sigma = sigma.clone().to(device=device, dtype=dtype)
        else:
            self.sigma = (
                (torch.randint(0, 2, (input_dim,), device=device) * 2 - 1)
                .to(dtype=dtype)
            )
    
    def to_matrix(self) -> Tensor:
        """
        Convert (h, σ) to embedding matrix S.
        
        Returns:
            S: Embedding matrix (D, d)
        """
        S = torch.zeros(self.input_dim, self.low_dim, device=self.device, dtype=self.dtype)
        for i in range(self.input_dim):
            j = self.h[i].item()
            S[i, j] = self.sigma[i]
        return S
    
    def high_to_low(self, X: Tensor) -> Tensor:
        """
        Project high-dim to low-dim: y[h[i]] += σ[i] * x[i]
        
        Args:
            X: High-dim points (n, D) or (D,)
            
        Returns:
            Y: Low-dim points (n, d) or (d,)
        """
        squeeze = False
        if X.dim() == 1:
            X = X.unsqueeze(0)
            squeeze = True
        
        X = X.to(device=self.device, dtype=self.dtype)
        batch_size = X.shape[0]
        
        Y = torch.zeros((batch_size, self.low_dim), device=self.device, dtype=self.dtype)
        
        for i in range(self.input_dim):
            bucket_idx = self.h[i].item()
            Y[:, bucket_idx] += X[:, i] * self.sigma[i]
        
        if squeeze:
            Y = Y.squeeze(0)
        
        return Y
    
    def low_to_high(self, Y: Tensor, bounds: Tensor) -> Tensor:
        """
        Project low-dim to high-dim: x[i] = σ[i] * y[h[i]]
        
        Args:
            Y: Low-dim points (n, d) or (d,)
            bounds: Original bounds (2, D)
            
        Returns:
            X: High-dim points (n, D) or (D,)
        """
        squeeze = False
        if Y.dim() == 1:
            Y = Y.unsqueeze(0)
            squeeze = True
        
        Y = Y.to(device=self.device, dtype=self.dtype)
        batch_size = Y.shape[0]
        
        X = torch.zeros((batch_size, self.input_dim), device=self.device, dtype=self.dtype)
        
        for i in range(self.input_dim):
            bucket_idx = self.h[i].item()
            X[:, i] = Y[:, bucket_idx] * self.sigma[i]
        
        # Denormalize and clamp
        lb = bounds[0]
        ub = bounds[1]
        X = lb + (X + 1) / 2 * (ub - lb)
        X = clamp_to_bounds(X, bounds)
        
        if squeeze:
            X = X.squeeze(0)
        
        return X
    
    def clone(self) -> 'HeSBOEmbedding':
        """Create a copy of this embedding."""
        return HeSBOEmbedding(
            input_dim=self.input_dim,
            low_dim=self.low_dim,
            device=self.device,
            dtype=self.dtype,
            h=self.h,
            sigma=self.sigma,
        )
    
    @staticmethod
    def sample_from_hesbo(
        input_dim: int,
        low_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> 'HeSBOEmbedding':
        """Sample a fresh embedding from HeSBO distribution μ."""
        return HeSBOEmbedding(
            input_dim=input_dim,
            low_dim=low_dim,
            device=device,
            dtype=dtype,
        )


# =============================================================================
# Neighbor Generation
# =============================================================================

def generate_neighbors(
    embedding: HeSBOEmbedding,
    num_neighbors: int = 10,
) -> List[HeSBOEmbedding]:
    """
    Generate neighboring embeddings by modifying h or σ.
    
    Each neighbor differs by:
    - Changing one hash bucket, OR
    - Flipping one sign
    
    Args:
        embedding: Current embedding
        num_neighbors: Number of neighbors to generate
        
    Returns:
        List of neighbor embeddings
    """
    neighbors = []
    D = embedding.input_dim
    d = embedding.low_dim
    
    for _ in range(num_neighbors):
        # Copy current embedding
        new_h = embedding.h.clone()
        new_sigma = embedding.sigma.clone()
        
        # Random index to modify
        i = np.random.randint(0, D)
        
        if np.random.random() < 0.5:
            # Change hash bucket
            old_bucket = new_h[i].item()
            # Choose new bucket different from old
            available_buckets = [j for j in range(d) if j != old_bucket]
            if available_buckets:
                new_bucket = np.random.choice(available_buckets)
                new_h[i] = new_bucket
        else:
            # Flip sign
            new_sigma[i] = -new_sigma[i]
        
        neighbor = HeSBOEmbedding(
            input_dim=D,
            low_dim=d,
            device=embedding.device,
            dtype=embedding.dtype,
            h=new_h,
            sigma=new_sigma,
        )
        neighbors.append(neighbor)
    
    return neighbors


# =============================================================================
# EmbedESSI: Embedding-level ESSI Quality
# =============================================================================

def compute_embed_essi(
    embedding: HeSBOEmbedding,
    X_high: Tensor,
    f_values: Tensor,
    bounds: Tensor,
    n_anchors: int = 5,
    n_neighborhood_samples: int = 10,
) -> float:
    """
    Compute embedding-level ESSI quality score.
    
    J = (1/|A|) * Σ_{a ∈ A} max_{y ∈ N(y_a)} ESSI(y)
    
    Args:
        embedding: The embedding to evaluate
        X_high: High-dim data (n, D)
        f_values: Objective values (n, 1)
        bounds: Original bounds (2, D)
        n_anchors: Number of anchor points
        n_neighborhood_samples: Samples in neighborhood
        
    Returns:
        J: ESSI-based embedding quality score
    """
    n = X_high.shape[0]
    if n < 3:
        return 0.0
    
    device = embedding.device
    dtype = embedding.dtype
    
    try:
        # Project data to low-dim
        Y_low = embedding.high_to_low(X_high)
        
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
        try:
            fit_gpytorch_mll(mll)
        except:
            pass
        
        model.eval()
        
        # Get f_best (in standardized space)
        f_best_std = f_std_vals.min().item()
        f_best_orig = f_values.min().item()
        
        # Select anchors: top M best points + some random
        n_best = min(n_anchors // 2 + 1, n)
        n_random = n_anchors - n_best
        
        # Indices of best points
        _, best_indices = torch.sort(f_values.squeeze(), descending=False)
        anchor_indices = best_indices[:n_best].tolist()
        
        # Add random indices
        remaining = [i for i in range(n) if i not in anchor_indices]
        if remaining and n_random > 0:
            random_indices = np.random.choice(remaining, size=min(n_random, len(remaining)), replace=False)
            anchor_indices.extend(random_indices.tolist())
        
        # Compute ESSI sum
        essi_sum = 0.0
        
        for a_idx in anchor_indices:
            y_a = Y_norm[a_idx]
            
            # Sample neighborhood around y_a
            best_essi = 0.0
            
            for _ in range(n_neighborhood_samples):
                # Random perturbation in neighborhood
                delta = torch.randn_like(y_a) * 0.1
                y_candidate = torch.clamp(y_a + delta, 0.0, 1.0)
                
                # Compute ESSI at candidate
                with torch.no_grad():
                    pred = model(y_candidate.unsqueeze(0))
                    mu = pred.mean.item()
                    sigma = pred.variance.sqrt().item()
                
                if sigma < 1e-10:
                    essi = 0.0
                else:
                    # ESSI formula (minimization): (f* - μ) Φ(γ) + σ φ(γ)
                    gamma = (f_best_std - mu) / sigma
                    essi = (f_best_std - mu) * norm.cdf(gamma) + sigma * norm.pdf(gamma)
                    essi = max(essi, 0.0)
                
                best_essi = max(best_essi, essi)
            
            essi_sum += best_essi
        
        J = essi_sum / len(anchor_indices)
        return J
    
    except Exception:
        return 0.0


# =============================================================================
# GlobalRehashAndImprove
# =============================================================================

def global_rehash_and_improve(
    input_dim: int,
    low_dim: int,
    X_high: Tensor,
    f_values: Tensor,
    bounds: Tensor,
    device: torch.device,
    dtype: torch.dtype,
    L_hc: int = 5,
    num_neighbors: int = 5,
) -> HeSBOEmbedding:
    """
    GlobalRehashAndImprove procedure.
    
    1. Sample fresh embedding from HeSBO distribution
    2. Local hill-climbing on (h, σ) using EmbedESSI
    
    Args:
        input_dim: High-dim space D
        low_dim: Low-dim space d
        X_high: High-dim data (n, D)
        f_values: Objective values (n, 1)
        bounds: Original bounds (2, D)
        device: Device
        dtype: Data type
        L_hc: Number of hill-climbing iterations
        num_neighbors: Number of neighbors per iteration
        
    Returns:
        S_prop: Improved embedding proposal
    """
    # 1. Global rehash: sample fresh embedding
    S_curr = HeSBOEmbedding.sample_from_hesbo(input_dim, low_dim, device, dtype)
    J_curr = compute_embed_essi(S_curr, X_high, f_values, bounds)
    
    # 2. Local improvement via hill-climbing
    for _ in range(L_hc):
        neighbors = generate_neighbors(S_curr, num_neighbors)
        
        S_best = S_curr
        J_best = J_curr
        
        for S_candidate in neighbors:
            J_cand = compute_embed_essi(S_candidate, X_high, f_values, bounds)
            if J_cand > J_best:
                J_best = J_cand
                S_best = S_candidate
        
        if J_best > J_curr:
            S_curr = S_best
            J_curr = J_best
        else:
            break  # Local optimum reached
    
    return S_curr


# =============================================================================
# AS-HeSBO v2 Main Class
# =============================================================================

class ASHeSBO_V2(BaseOptimizer):
    """
    AS-HeSBO v2: Adaptive-Sketch HeSBO with ESSI-based Embedding Search.
    
    Uses ESSI as embedding quality metric and GlobalRehashAndImprove
    for adaptive embedding updates.
    
    Args:
        input_dim: High-dimensional input space (D)
        low_dim: Low-dimensional embedding dimension (d)
        bounds: Bounds tensor (2, D)
        K: Number of epochs (embedding update opportunities)
        L: Number of inner BO steps per epoch
        L_hc: Number of hill-climbing iterations in GlobalRehashAndImprove
        num_neighbors: Number of neighbors for hill-climbing
        n_anchors: Number of anchors for EmbedESSI
        acq_func: Acquisition function for inner BO
        device: Device for computation
        dtype: Data type
    """
    
    def __init__(
        self,
        input_dim: int,
        low_dim: int,
        bounds: Tensor,
        K: int = 10,
        L: int = 10,
        L_hc: int = 5,
        num_neighbors: int = 5,
        n_anchors: int = 5,
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
        self.K = K  # Epochs
        self.L = L  # Inner BO steps per epoch
        self.L_hc = L_hc  # Hill-climbing iterations
        self.num_neighbors = num_neighbors
        self.n_anchors = n_anchors
        
        # Low-dim bounds
        self.low_bounds = torch.stack([
            torch.zeros(self.low_dim, device=self.device, dtype=self.dtype),
            torch.ones(self.low_dim, device=self.device, dtype=self.dtype)
        ])
        
        # Initialize embedding from HeSBO distribution
        self.current_embedding = HeSBOEmbedding.sample_from_hesbo(
            input_dim, self.low_dim, self.device, self.dtype
        )
        
        # Current embedding quality
        self.current_J: Optional[float] = None
        
        # Inner GP model
        self.inner_gp: Optional[SingleTaskGP] = None
        
        # Training data
        self.X_high: Optional[Tensor] = None
        self.f_values: Optional[Tensor] = None
        self.Y_low: Optional[Tensor] = None
        
        # Y statistics
        self._f_mean = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self._f_std = torch.tensor(1.0, device=self.device, dtype=self.dtype)
        
        # Counters
        self._epoch = 0
        self._step_in_epoch = 0
        self._total_evals = 0
        self._embedding_updates = 0
        
        # Best found
        self.f_best: Optional[float] = None
        self.x_best: Optional[Tensor] = None
        
        # Statistics
        self.J_history: List[float] = []
    
    def _update_f_statistics(self):
        """Update f mean and std."""
        if self.f_values is not None and self.f_values.numel() > 0:
            self._f_mean = self.f_values.mean()
            self._f_std = self.f_values.std()
            if self._f_std < 1e-6:
                self._f_std = torch.tensor(1.0, device=self.device, dtype=self.dtype)
    
    def _standardize_f(self, f: Tensor) -> Tensor:
        """Standardize f to mean=0, std=1."""
        return (f - self._f_mean) / (self._f_std + 1e-8)
    
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
        """Select next point in low-dim space using acquisition."""
        if self.inner_gp is None or self.Y_low is None or self.Y_low.shape[0] < 2:
            return torch.rand(self.low_dim, device=self.device, dtype=self.dtype)
        
        f_std = self._standardize_f(self.f_values)
        best_f = f_std.max()  # For minimization
        
        if self.acq_func == "UCB":
            acq = UpperConfidenceBound(self.inner_gp, beta=0.25)
        else:
            acq = ExpectedImprovement(self.inner_gp, best_f=best_f)
        
        try:
            candidates, _ = optimize_acqf(
                acq,
                bounds=self.low_bounds,
                q=1,
                num_restarts=5,
                raw_samples=32,
            )
            return candidates.squeeze(0)
        except:
            return torch.rand(self.low_dim, device=self.device, dtype=self.dtype)
    
    def _try_update_embedding(self):
        """Try to update embedding at epoch boundary using GlobalRehashAndImprove."""
        if self.X_high is None or self.X_high.shape[0] < 3:
            return
        
        # Update current embedding quality
        self.current_J = compute_embed_essi(
            self.current_embedding,
            self.X_high,
            self.f_values,
            self.bounds,
            self.n_anchors,
        )
        
        # GlobalRehashAndImprove
        S_prop = global_rehash_and_improve(
            self.input_dim,
            self.low_dim,
            self.X_high,
            self.f_values,
            self.bounds,
            self.device,
            self.dtype,
            self.L_hc,
            self.num_neighbors,
        )
        
        # Evaluate proposal
        J_prop = compute_embed_essi(
            S_prop,
            self.X_high,
            self.f_values,
            self.bounds,
            self.n_anchors,
        )
        
        # Compare and update if better
        if J_prop >= self.current_J:
            self.current_embedding = S_prop
            self.current_J = J_prop
            self._embedding_updates += 1
            
            # Re-project all data to new embedding
            self.Y_low = self.current_embedding.high_to_low(self.X_high)
            
            # Reset inner GP
            self.inner_gp = None
        
        # Record history
        self.J_history.append(self.current_J)
        
        # Increment epoch
        self._epoch += 1
        self._step_in_epoch = 0
    
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
            
            self._total_evals += 1
            self._step_in_epoch += 1
        
        # Check if we should update embedding
        if self._step_in_epoch >= self.L:
            self._try_update_embedding()
    
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
        x_high = self.current_embedding.low_to_high(y_low, self.bounds)
        
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
            'epoch': self._epoch,
            'step_in_epoch': self._step_in_epoch,
            'total_evals': self._total_evals,
            'embedding_updates': self._embedding_updates,
            'current_J': self.current_J,
            'J_history': self.J_history.copy(),
        }
    
    def reset(self):
        """Reset optimizer to initial state."""
        super().reset()
        
        self.current_embedding = HeSBOEmbedding.sample_from_hesbo(
            self.input_dim, self.low_dim, self.device, self.dtype
        )
        
        self.current_J = None
        self.inner_gp = None
        
        self.X_high = None
        self.f_values = None
        self.Y_low = None
        
        self._f_mean = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self._f_std = torch.tensor(1.0, device=self.device, dtype=self.dtype)
        
        self._epoch = 0
        self._step_in_epoch = 0
        self._total_evals = 0
        self._embedding_updates = 0
        
        self.f_best = None
        self.x_best = None
        
        self.J_history = []


class ASHeSBO_V2_Maximize(ASHeSBO_V2):
    """
    AS-HeSBO v2 variant for maximization problems.
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
            
            self._total_evals += 1
            self._step_in_epoch += 1
        
        if self._step_in_epoch >= self.L:
            self._try_update_embedding()
    
    def get_best_point(self) -> Tuple[Tensor, Tensor]:
        """Get best observed point (maximization)."""
        if self.f_values is None or self.f_values.numel() == 0:
            raise ValueError("No observations yet.")
        
        best_idx = self.f_values.argmax()
        return self.X_high[best_idx], self.f_values[best_idx]

