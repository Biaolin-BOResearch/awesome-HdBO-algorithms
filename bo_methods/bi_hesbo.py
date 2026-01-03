"""
BiHeSBO-UCB: Bi-level Hashing-Enhanced Subspace Bayesian Optimization with UCB.

This method implements a bi-level optimization structure:
1. Outer level (Bandit): Maintains M independent hash embeddings and uses
   UCB/Thompson Sampling to select which embedding to allocate evaluation budget
2. Inner level (BO): Runs standard low-dimensional BO within the selected embedding

Algorithm Overview:
- Initialization: Create M hash embeddings, do L initial evaluations per embedding
- Main loop:
  1. Outer layer selects embedding j using UCB1/TS bandit algorithm
  2. Inner layer runs one BO step in embedding j's low-dim space
  3. Update embedding j's GP and best value
  4. Update global best and bandit statistics

Reference:
    Based on HeSBO: Nayebi et al. "A Framework for Bayesian Optimization 
    in Embedded Subspaces" ICML 2019.
"""

import math
from typing import Optional, Tuple, List, Dict, Any
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
# Single Embedding (HeSBO-style)
# =============================================================================

class HashEmbedding:
    """
    A single hash embedding using Count Sketch projection.
    
    Maintains its own:
    - Hash function (h) and sign function (sigma)
    - Low-dimensional GP model
    - Training data in low-dim space
    - Best value found in this embedding
    """
    
    def __init__(
        self,
        embedding_id: int,
        input_dim: int,
        low_dim: int,
        bounds: Tensor,
        acq_func: str = "EI",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize a hash embedding.
        
        Args:
            embedding_id: Unique identifier for this embedding
            input_dim: High-dimensional input space dimension (D)
            low_dim: Low-dimensional embedding dimension (d)
            bounds: Original bounds in high-dim space (2, D)
            acq_func: Acquisition function type for inner BO
            device: Device for computation
            dtype: Data type
        """
        self.embedding_id = embedding_id
        self.input_dim = input_dim
        self.low_dim = low_dim
        self.device = device or DEVICE
        self.dtype = dtype or DTYPE
        self.acq_func = acq_func
        
        # Original bounds
        self.bounds = bounds.to(device=self.device, dtype=self.dtype)
        
        # Low-dim bounds (unit cube)
        self.low_bounds = torch.stack([
            torch.zeros(low_dim, device=self.device, dtype=self.dtype),
            torch.ones(low_dim, device=self.device, dtype=self.dtype)
        ])
        
        # Initialize hash function (Count Sketch)
        # h: maps each high dimension i to a low dimension index h[i] in [0, low_dim)
        self.h = torch.randint(0, low_dim, (input_dim,), device=self.device)
        
        # sigma: random signs {-1, +1}
        self.sigma = (
            (torch.randint(0, 2, (input_dim,), device=self.device) * 2 - 1)
            .to(dtype=self.dtype)
        )
        
        # Training data in low-dim space
        self.train_Y_low = torch.empty((0, low_dim), device=self.device, dtype=self.dtype)
        self.train_f = torch.empty((0, 1), device=self.device, dtype=self.dtype)
        
        # Training data in high-dim space (for reference)
        self.train_X_high = torch.empty((0, input_dim), device=self.device, dtype=self.dtype)
        
        # GP model
        self.model: Optional[SingleTaskGP] = None
        
        # Best value found in this embedding (for minimization)
        self.f_best: Optional[float] = None
        self.x_best: Optional[Tensor] = None
        self.y_best: Optional[Tensor] = None  # Low-dim best
        
        # Y statistics for standardization
        self._y_mean = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self._y_std = torch.tensor(1.0, device=self.device, dtype=self.dtype)
        
        # Pull count (for UCB)
        self.n_pulls = 0
        self.total_reward = 0.0
        
    def _normalize_low(self, Y: Tensor) -> Tensor:
        """Normalize low-dim Y to [0, 1]^d."""
        # For Count Sketch, values can be outside [0,1]
        # We use min-max normalization based on seen data
        return torch.clamp(Y, 0.0, 1.0)
    
    def _update_y_statistics(self):
        """Update f mean and std for standardization."""
        if self.train_f.numel() > 0:
            self._y_mean = self.train_f.mean()
            # Handle single observation case: std() returns NaN for single element
            if self.train_f.numel() < 2:
                self._y_std = torch.tensor(1.0, device=self.device, dtype=self.dtype)
            else:
                self._y_std = self.train_f.std()
                # Check for NaN or very small std
                if torch.isnan(self._y_std) or self._y_std < 1e-6:
                    self._y_std = torch.tensor(1.0, device=self.device, dtype=self.dtype)
    
    def _standardize_f(self, f: Tensor) -> Tensor:
        """Standardize f to mean=0, std=1."""
        return (f - self._y_mean) / (self._y_std + 1e-8)
    
    def high_to_low(self, X: Tensor) -> Tensor:
        """
        Project high-dimensional points to low-dimensional space using Count Sketch.
        
        Implements: y[h[i]] += s[i] * x[i] for all i
        
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
    
    def low_to_high(self, Y: Tensor) -> Tensor:
        """
        Project low-dimensional points back to high-dimensional space.
        
        Implements: x[i] = s[i] * y[h[i]]
        
        Args:
            Y: Points in low-dim space (n, d) or (d,)
            
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
        
        # Denormalize and clamp
        lb = self.bounds[0]
        ub = self.bounds[1]
        X = lb + X * (ub - lb)
        X = clamp_to_bounds(X, self.bounds)
        
        if squeeze:
            X = X.squeeze(0)
            
        return X
    
    def observe(self, x_high: Tensor, y_low: Tensor, f: Tensor):
        """
        Add observation to this embedding.
        
        Args:
            x_high: High-dim point (D,) or (1, D)
            y_low: Low-dim point (d,) or (1, d)
            f: Objective value (scalar or (1,) or (1, 1))
        """
        x_high = ensure_tensor(x_high, device=self.device, dtype=self.dtype)
        y_low = ensure_tensor(y_low, device=self.device, dtype=self.dtype)
        f = ensure_tensor(f, device=self.device, dtype=self.dtype)
        
        if x_high.dim() == 1:
            x_high = x_high.unsqueeze(0)
        if y_low.dim() == 1:
            y_low = y_low.unsqueeze(0)
        if f.dim() == 0:
            f = f.unsqueeze(0).unsqueeze(-1)
        elif f.dim() == 1:
            f = f.unsqueeze(-1)
        
        # Clamp low-dim to [0, 1]
        y_low = torch.clamp(y_low, 0.0, 1.0)
        
        self.train_X_high = torch.cat([self.train_X_high, x_high], dim=0)
        self.train_Y_low = torch.cat([self.train_Y_low, y_low], dim=0)
        self.train_f = torch.cat([self.train_f, f], dim=0)
        
        # Update best (minimization)
        f_val = f.item()
        if self.f_best is None or f_val < self.f_best:
            self.f_best = f_val
            self.x_best = x_high.squeeze(0).clone()
            self.y_best = y_low.squeeze(0).clone()
        
        # Update pull count
        self.n_pulls += 1
    
    def fit_model(self):
        """Fit GP model on low-dim data."""
        if self.train_Y_low.numel() == 0 or self.train_Y_low.shape[0] < 2:
            return
        
        self._update_y_statistics()
        
        Y_low = self.train_Y_low
        f_std = self._standardize_f(self.train_f)
        
        # Create GP with Matern kernel
        covar_module = ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=self.low_dim,
                lengthscale_constraint=GreaterThan(1e-4),
            ),
            outputscale_constraint=GreaterThan(1e-4),
        )
        
        self.model = SingleTaskGP(
            train_X=Y_low,
            train_Y=f_std,
            covar_module=covar_module,
            input_transform=None,
            outcome_transform=None,
        ).to(device=self.device, dtype=self.dtype)
        
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        try:
            fit_gpytorch_mll(mll)
        except Exception:
            pass
    
    def suggest_low_dim(self) -> Tensor:
        """
        Suggest next point in low-dim space.
        
        Returns:
            Suggested point in low-dim space (d,)
        """
        if self.model is None or self.train_f.numel() < 2:
            # Random sampling
            return torch.rand(self.low_dim, device=self.device, dtype=self.dtype)
        
        # Get standardized best value
        f_std = self._standardize_f(self.train_f)
        best_f = f_std.max()  # For minimization with negated standardization
        
        # Create acquisition function
        if self.acq_func == "UCB":
            acq = UpperConfidenceBound(self.model, beta=0.25)
        else:  # EI
            acq = ExpectedImprovement(self.model, best_f=best_f)
        
        try:
            candidates, _ = optimize_acqf(
                acq,
                bounds=self.low_bounds,
                q=1,
                num_restarts=5,
                raw_samples=32,
            )
            return candidates.squeeze(0)
        except Exception:
            return torch.rand(self.low_dim, device=self.device, dtype=self.dtype)
    
    def get_reward(self, f_global_best: float) -> float:
        """
        Compute reward for this embedding (for bandit).
        
        For minimization: reward = -f_best (higher is better)
        Or: reward = improvement = f_global_best - f_best
        
        Args:
            f_global_best: Global best value across all embeddings
            
        Returns:
            Reward value
        """
        if self.f_best is None:
            return 0.0
        
        # Option 1: Negative best value
        # return -self.f_best
        
        # Option 2: Improvement over global best
        return f_global_best - self.f_best


# =============================================================================
# BiHeSBO-UCB Main Class
# =============================================================================

class BiHeSBO(BaseOptimizer):
    """
    Bi-level Hashing-Enhanced Subspace Bayesian Optimization.
    
    Maintains M independent hash embeddings and uses a bandit algorithm
    to select which embedding to allocate evaluation budget.
    
    Args:
        input_dim: High-dimensional input space dimension (D)
        low_dim: Low-dimensional embedding dimension (d)
        bounds: Bounds tensor of shape (2, D)
        M: Number of parallel embeddings
        L: Number of initial evaluations per embedding
        bandit_algo: Bandit algorithm ('UCB1', 'TS', 'EPS_GREEDY')
        ucb_c: Exploration parameter for UCB1
        eps: Epsilon for epsilon-greedy
        inner_acq: Acquisition function for inner BO
        device: Device for computation
        dtype: Data type
    """
    
    def __init__(
        self,
        input_dim: int,
        low_dim: int,
        bounds: Tensor,
        M: int = 5,
        L: int = 2,
        bandit_algo: str = "UCB1",
        ucb_c: float = 2.0,
        eps: float = 0.1,
        inner_acq: str = "EI",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        _device = device or DEVICE
        _dtype = dtype or DTYPE
        
        super().__init__(
            input_dim=input_dim,
            bounds=bounds,
            acq_func=inner_acq,
            device=_device,
            dtype=_dtype,
        )
        
        self.low_dim = max(1, min(low_dim, input_dim))
        self.M = M  # Number of embeddings
        self.L = L  # Initial evaluations per embedding
        self.bandit_algo = bandit_algo
        self.ucb_c = ucb_c
        self.eps = eps
        self.inner_acq = inner_acq
        
        # Create M embeddings
        self.embeddings: List[HashEmbedding] = []
        for j in range(M):
            emb = HashEmbedding(
                embedding_id=j,
                input_dim=input_dim,
                low_dim=self.low_dim,
                bounds=bounds,
                acq_func=inner_acq,
                device=_device,
                dtype=_dtype,
            )
            self.embeddings.append(emb)
        
        # Global best
        self.f_global_best: Optional[float] = None
        self.x_global_best: Optional[Tensor] = None
        
        # Full training data (for reference)
        self.X: Optional[Tensor] = None
        self.y: Optional[Tensor] = None
        
        # Initialization state
        self._initialized = False
        self._init_embedding_idx = 0
        self._init_step_in_embedding = 0
        
        # Current selected embedding for the pending suggestion
        self._current_embedding_idx: Optional[int] = None
        self._current_y_low: Optional[Tensor] = None
        
        # Total number of evaluations
        self._n_evals = 0
        
    # -------------------------------------------------------------------------
    # Bandit Selection Methods
    # -------------------------------------------------------------------------
    
    def _ucb1_score(self, embedding: HashEmbedding) -> float:
        """
        Compute UCB1 score for an embedding.
        
        UCB1 = mean_reward + c * sqrt(ln(n) / n_j)
        
        Args:
            embedding: The embedding to score
            
        Returns:
            UCB1 score
        """
        if embedding.n_pulls == 0:
            return float('inf')  # Explore unpulled arms first
        
        # Mean reward (negative best value for minimization)
        if embedding.f_best is None:
            mean_reward = 0.0
        else:
            # Use improvement over global best as reward
            if self.f_global_best is not None:
                mean_reward = (self.f_global_best - embedding.f_best) / max(1, embedding.n_pulls)
            else:
                mean_reward = -embedding.f_best
        
        # Exploration bonus
        total_pulls = sum(e.n_pulls for e in self.embeddings)
        if total_pulls == 0:
            return float('inf')
        
        exploration = self.ucb_c * math.sqrt(math.log(total_pulls + 1) / embedding.n_pulls)
        
        return mean_reward + exploration
    
    def _thompson_sampling_score(self, embedding: HashEmbedding) -> float:
        """
        Compute Thompson Sampling score for an embedding.
        
        Uses posterior sampling from a Beta distribution approximation.
        """
        if embedding.n_pulls == 0:
            return np.random.random()  # Random prior
        
        # Simple approximation: sample from normal posterior
        if embedding.f_best is None:
            mean = 0.0
        else:
            if self.f_global_best is not None:
                mean = self.f_global_best - embedding.f_best
            else:
                mean = -embedding.f_best
        
        # Posterior variance decreases with pulls
        std = 1.0 / math.sqrt(embedding.n_pulls + 1)
        
        return np.random.normal(mean, std)
    
    def _select_embedding(self) -> int:
        """
        Select which embedding to use based on bandit algorithm.
        
        Returns:
            Index of selected embedding
        """
        if self.bandit_algo == "UCB1":
            scores = [self._ucb1_score(e) for e in self.embeddings]
            return int(np.argmax(scores))
        
        elif self.bandit_algo == "TS":
            scores = [self._thompson_sampling_score(e) for e in self.embeddings]
            return int(np.argmax(scores))
        
        elif self.bandit_algo == "EPS_GREEDY":
            if np.random.random() < self.eps:
                # Explore: random embedding
                return np.random.randint(0, self.M)
            else:
                # Exploit: best embedding
                best_f = [e.f_best if e.f_best is not None else float('inf') 
                         for e in self.embeddings]
                return int(np.argmin(best_f))
        
        else:
            # Default: round-robin or random
            return np.random.randint(0, self.M)
    
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
        
        # Update base class
        self.train_X = self.X
        self.train_y = self.y
        
        # Process each observation
        for i in range(X.shape[0]):
            x_high = X[i]
            f_val = y[i]
            
            # Get the embedding that generated this point
            emb_idx = self._current_embedding_idx
            if emb_idx is None:
                # During initialization or if not tracked
                emb_idx = self._init_embedding_idx if not self._initialized else 0
            
            embedding = self.embeddings[emb_idx]
            
            # Project to low-dim
            if self._current_y_low is not None:
                y_low = self._current_y_low
            else:
                # Normalize x to [0,1] then project
                x_norm = (x_high - self.bounds[0]) / (self.bounds[1] - self.bounds[0] + 1e-8)
                y_low = embedding.high_to_low(x_norm)
            
            # Add observation to embedding
            embedding.observe(x_high, y_low, f_val)
            
            # Update global best (minimization)
            f_float = f_val.item()
            if self.f_global_best is None or f_float < self.f_global_best:
                self.f_global_best = f_float
                self.x_global_best = x_high.clone()
            
            self._n_evals += 1
            
            # Update initialization state
            if not self._initialized:
                self._init_step_in_embedding += 1
                if self._init_step_in_embedding >= self.L:
                    self._init_step_in_embedding = 0
                    self._init_embedding_idx += 1
                    if self._init_embedding_idx >= self.M:
                        self._initialized = True
        
        # Reset current tracking
        self._current_embedding_idx = None
        self._current_y_low = None
    
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
        
        # During initialization: cycle through embeddings
        if not self._initialized:
            emb_idx = self._init_embedding_idx
            embedding = self.embeddings[emb_idx]
            
            # Random point in low-dim space
            y_low = torch.rand(self.low_dim, device=self.device, dtype=self.dtype)
            
            # Project to high-dim
            x_high = embedding.low_to_high(y_low)
            
            # Track for observe
            self._current_embedding_idx = emb_idx
            self._current_y_low = y_low
            
            return x_high
        
        # Main loop: select embedding using bandit
        emb_idx = self._select_embedding()
        embedding = self.embeddings[emb_idx]
        
        # Fit GP for this embedding
        embedding.fit_model()
        
        # Get suggestion in low-dim space
        y_low = embedding.suggest_low_dim()
        
        # Project to high-dim
        x_high = embedding.low_to_high(y_low)
        
        # Track for observe
        self._current_embedding_idx = emb_idx
        self._current_y_low = y_low
        
        return x_high
    
    def get_best_point(self) -> Tuple[Tensor, Tensor]:
        """Get best observed point."""
        if self.y is None or self.y.numel() == 0:
            raise ValueError("No observations yet.")
        
        best_idx = self.y.argmin()
        return self.X[best_idx], self.y[best_idx]
    
    def get_embedding_stats(self) -> List[Dict[str, Any]]:
        """
        Get statistics for each embedding.
        
        Returns:
            List of dicts with stats for each embedding
        """
        stats = []
        for emb in self.embeddings:
            stats.append({
                'id': emb.embedding_id,
                'n_pulls': emb.n_pulls,
                'f_best': emb.f_best,
                'ucb_score': self._ucb1_score(emb) if self.bandit_algo == "UCB1" else None,
            })
        return stats
    
    def reset(self):
        """Reset optimizer to initial state."""
        super().reset()
        
        # Recreate embeddings
        self.embeddings = []
        for j in range(self.M):
            emb = HashEmbedding(
                embedding_id=j,
                input_dim=self.input_dim,
                low_dim=self.low_dim,
                bounds=self.bounds,
                acq_func=self.inner_acq,
                device=self.device,
                dtype=self.dtype,
            )
            self.embeddings.append(emb)
        
        self.f_global_best = None
        self.x_global_best = None
        self.X = None
        self.y = None
        
        self._initialized = False
        self._init_embedding_idx = 0
        self._init_step_in_embedding = 0
        self._current_embedding_idx = None
        self._current_y_low = None
        self._n_evals = 0


class BiHeSBOMaximize(BiHeSBO):
    """
    BiHeSBO variant for maximization problems.
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
        
        if self.X is None:
            self.X = X
            self.y = y
        else:
            self.X = torch.cat([self.X, X], dim=0)
            self.y = torch.cat([self.y, y], dim=0)
        
        self.train_X = self.X
        self.train_y = self.y
        
        for i in range(X.shape[0]):
            x_high = X[i]
            f_val = y[i]
            
            emb_idx = self._current_embedding_idx
            if emb_idx is None:
                emb_idx = self._init_embedding_idx if not self._initialized else 0
            
            embedding = self.embeddings[emb_idx]
            
            if self._current_y_low is not None:
                y_low = self._current_y_low
            else:
                x_norm = (x_high - self.bounds[0]) / (self.bounds[1] - self.bounds[0] + 1e-8)
                y_low = embedding.high_to_low(x_norm)
            
            # Negate for inner BO (which minimizes)
            embedding.observe(x_high, y_low, -f_val)
            
            # Update global best (maximization)
            f_float = f_val.item()
            if self.f_global_best is None or f_float > self.f_global_best:
                self.f_global_best = f_float
                self.x_global_best = x_high.clone()
            
            self._n_evals += 1
            
            if not self._initialized:
                self._init_step_in_embedding += 1
                if self._init_step_in_embedding >= self.L:
                    self._init_step_in_embedding = 0
                    self._init_embedding_idx += 1
                    if self._init_embedding_idx >= self.M:
                        self._initialized = True
        
        self._current_embedding_idx = None
        self._current_y_low = None
    
    def _ucb1_score(self, embedding: HashEmbedding) -> float:
        """UCB1 score for maximization."""
        if embedding.n_pulls == 0:
            return float('inf')
        
        # For maximization, higher f_best is better
        if embedding.f_best is None:
            mean_reward = 0.0
        else:
            # f_best is negated in embedding, so we use -f_best
            mean_reward = -embedding.f_best / max(1, embedding.n_pulls)
        
        total_pulls = sum(e.n_pulls for e in self.embeddings)
        if total_pulls == 0:
            return float('inf')
        
        exploration = self.ucb_c * math.sqrt(math.log(total_pulls + 1) / embedding.n_pulls)
        
        return mean_reward + exploration
    
    def get_best_point(self) -> Tuple[Tensor, Tensor]:
        """Get best observed point (maximization)."""
        if self.y is None or self.y.numel() == 0:
            raise ValueError("No observations yet.")
        
        best_idx = self.y.argmax()
        return self.X[best_idx], self.y[best_idx]

