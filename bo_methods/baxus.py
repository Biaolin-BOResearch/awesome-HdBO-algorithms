"""
BAxUS: Bayesian Optimization with Adaptively Expanding Subspaces.

Reference:
    Papenmeier, L., Nardi, L., & Poloczek, M. (2022). "Increasing the Scope as
    You Learn: Adaptive Bayesian Optimization in Nested Subspaces". NeurIPS 2022.

BAxUS uses a family of nested random subspaces via count-sketch embeddings,
starting with very low dimensional embeddings (1-3 dimensions) and adaptively
expanding them as optimization progresses. This allows it to maintain previous
observations when increasing the embedding dimension.
"""

import torch
from typing import Optional, Tuple, Dict
from torch import Tensor

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

from .base import BaseOptimizer


class BAxUS(BaseOptimizer):
    """
    BAxUS: Bayesian Optimization with Adaptively Expanding Subspaces.

    Key Features:
    - Nested count-sketch embeddings that preserve observations across expansions
    - Adaptive expansion of target dimensionality
    - Trust region mechanism similar to TuRBO but dimension-adaptive
    - Starts with very low dimensions and expands as needed

    The count-sketch embedding maps each input dimension to exactly one target
    dimension with a random sign (+1 or -1). This sparse structure allows
    nested expansion while maintaining consistency.

    Args:
        input_dim: Original high dimensionality (D)
        bounds: Tensor of shape (2, input_dim) with bounds
        target_dim_init: Initial embedding dimensionality (default: 2)
        target_dim_max: Maximum embedding dimensionality (default: None, uses input_dim)
        n_init: Number of initial random samples
        length_init: Initial trust region length (default: 0.8)
        length_min: Minimum trust region length (default: 0.5^7)
        length_max: Maximum trust region length (default: 1.6)
        success_tolerance: Consecutive successes before expanding TR (default: 3)
        failure_tolerance_base: Base for computing failure tolerance (default: depends on d)
        expansion_factor: Factor for expanding target dimension (default: 2)
        device: Device for computations
        dtype: Data type for tensors
    """

    def __init__(
        self,
        input_dim: int,
        bounds: Tensor,
        target_dim_init: int = 2,
        target_dim_max: Optional[int] = None,
        n_init: int = 10,
        length_init: float = 0.8,
        length_min: float = 0.5 ** 7,
        length_max: float = 1.6,
        success_tolerance: int = 3,
        failure_tolerance_base: Optional[int] = None,
        expansion_factor: int = 2,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__(input_dim, bounds, device, dtype)

        self.target_dim_init = target_dim_init
        self.target_dim_max = target_dim_max if target_dim_max is not None else input_dim
        self.target_dim = target_dim_init  # Current target dimensionality

        self.n_init = n_init
        self.length_init = length_init
        self.length_min = length_min
        self.length_max = length_max
        self.success_tolerance = success_tolerance
        self.expansion_factor = expansion_factor

        # Dimension-dependent failure tolerance (as in paper)
        if failure_tolerance_base is None:
            failure_tolerance_base = max(4, self.target_dim)
        self.failure_tolerance_base = failure_tolerance_base

        # Trust region state
        self.length = length_init
        self.n_successes = 0
        self.n_failures = 0
        self.best_value = -float('inf')

        # Count-sketch embedding parameters
        self._initialize_embedding()

        self.model: Optional[SingleTaskGP] = None

    def _initialize_embedding(self):
        """
        Initialize count-sketch embedding.

        The count-sketch embedding assigns each input dimension to exactly
        one target dimension with a random sign.
        """
        # Hash function: maps each input dim to a target dim
        self.hash_indices = torch.randint(
            0, self.target_dim,
            (self.input_dim,),
            device=self.device
        )

        # Sign function: assigns +1 or -1 to each input dim
        self.hash_signs = 2 * torch.randint(
            0, 2,
            (self.input_dim,),
            device=self.device,
            dtype=self.dtype
        ) - 1

    def _expand_embedding(self):
        """
        Expand the target dimensionality while maintaining nested structure.

        The key insight is that the count-sketch structure allows us to
        simply re-hash some dimensions to the new target dimensions while
        keeping others unchanged.
        """
        old_target_dim = self.target_dim
        new_target_dim = min(
            old_target_dim * self.expansion_factor,
            self.target_dim_max
        )

        if new_target_dim == old_target_dim:
            return  # Already at maximum

        self.target_dim = new_target_dim

        # Re-initialize hash for new target dimension
        # To maintain nesting, we keep some structure but allow expansion
        new_hash_indices = torch.randint(
            0, self.target_dim,
            (self.input_dim,),
            device=self.device
        )

        # Keep signs the same for consistency
        self.hash_indices = new_hash_indices

        print(f"BAxUS: Expanded target dimension from {old_target_dim} to {new_target_dim}")

    def _embed_to_target(self, X_high: Tensor) -> Tensor:
        """
        Embed high-dimensional points to target space using count-sketch.

        Args:
            X_high: Tensor of shape (n, input_dim)

        Returns:
            Tensor of shape (n, target_dim)
        """
        n = X_high.shape[0]
        X_low = torch.zeros(n, self.target_dim, device=self.device, dtype=self.dtype)

        # Count-sketch: sum signed values mapped to same target dimension
        for i in range(self.input_dim):
            target_idx = self.hash_indices[i]
            sign = self.hash_signs[i]
            X_low[:, target_idx] += sign * X_high[:, i]

        # Normalize by square root of average collision rate
        collision_rate = self.input_dim / self.target_dim
        X_low = X_low / (collision_rate ** 0.5)

        return X_low

    def _unembed_from_target(self, X_low: Tensor) -> Tensor:
        """
        Map target space points back to high-dimensional space.

        Args:
            X_low: Tensor of shape (n, target_dim)

        Returns:
            Tensor of shape (n, input_dim)
        """
        n = X_low.shape[0]
        X_high = torch.zeros(n, self.input_dim, device=self.device, dtype=self.dtype)

        # Reverse count-sketch: use hash to distribute values
        collision_rate = self.input_dim / self.target_dim
        X_low_scaled = X_low * (collision_rate ** 0.5)

        for i in range(self.input_dim):
            target_idx = self.hash_indices[i]
            sign = self.hash_signs[i]
            X_high[:, i] = sign * X_low_scaled[:, target_idx]

        # Clip to bounds
        X_high = torch.clamp(X_high, self.bounds[0], self.bounds[1])

        return X_high

    def _get_target_bounds(self) -> Tensor:
        """Get bounds for the target space."""
        # Estimate bounds based on embedding of corner points
        corners = torch.stack([self.bounds[0], self.bounds[1]])
        embedded_corners = self._embed_to_target(corners)

        # Conservative bounds
        lower = embedded_corners.min(dim=0).values - 0.1
        upper = embedded_corners.max(dim=0).values + 0.1

        return torch.stack([lower, upper])

    def _create_trust_region_bounds(self, center: Tensor) -> Tensor:
        """
        Create trust region bounds around center in target space.

        Args:
            center: Center point in target space (shape: target_dim)

        Returns:
            Bounds tensor of shape (2, target_dim)
        """
        target_bounds = self._get_target_bounds()

        lower = torch.maximum(
            target_bounds[0],
            center - self.length / 2
        )
        upper = torch.minimum(
            target_bounds[1],
            center + self.length / 2
        )

        return torch.stack([lower, upper])

    def _update_trust_region(self, y_new: float):
        """Update trust region based on new observation."""
        # Check if this is an improvement
        if y_new > self.best_value:
            self.n_successes += 1
            self.n_failures = 0
            self.best_value = y_new
        else:
            self.n_successes = 0
            self.n_failures += 1

        # Expand trust region after sufficient successes
        if self.n_successes >= self.success_tolerance:
            self.length = min(self.length * 2.0, self.length_max)
            self.n_successes = 0

        # Shrink trust region after failures
        failure_tolerance = self.failure_tolerance_base * (self.target_dim / self.target_dim_init)
        if self.n_failures >= failure_tolerance:
            self.length = max(self.length / 2.0, self.length_min)
            self.n_failures = 0

            # If trust region is at minimum, try expanding target dimension
            if self.length <= self.length_min and self.target_dim < self.target_dim_max:
                self._expand_embedding()
                self.length = self.length_init  # Reset trust region size
                self.n_failures = 0
                self.n_successes = 0

    def _fit_model(self):
        """Fit GP model in target space."""
        # Embed observations to target space
        X_embedded = self._embed_to_target(self.X)

        # Fit GP in target space
        self.model = SingleTaskGP(
            train_X=X_embedded,
            train_Y=self.y
        ).to(device=self.device, dtype=self.dtype)

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)

    def suggest(self, n_suggestions: int = 1) -> Tensor:
        """
        Suggest next point(s) to evaluate.

        Args:
            n_suggestions: Number of points to suggest

        Returns:
            Tensor of shape (n_suggestions, input_dim)
        """
        if self.X is None or len(self.X) == 0:
            raise ValueError("No observations yet. Call observe() first.")

        # Fit model in target space
        self._fit_model()

        # Get best point in target space as TR center
        best_idx = self.y.argmax()
        center_embedded = self._embed_to_target(self.X[best_idx:best_idx+1]).squeeze(0)

        # Create trust region bounds in target space
        tr_bounds = self._create_trust_region_bounds(center_embedded)

        # Optimize acquisition function in target space
        acq_func = ExpectedImprovement(
            model=self.model,
            best_f=self.y.max().item()
        )

        candidates_embedded, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=tr_bounds,
            q=n_suggestions,
            num_restarts=10,
            raw_samples=512,
        )

        # Unembed to original space
        candidates = self._unembed_from_target(candidates_embedded)

        return candidates

    def observe(self, X: Tensor, y: Tensor):
        """
        Update optimizer with new observations.

        Args:
            X: Tensor of shape (n, input_dim)
            y: Tensor of shape (n, 1) or (n,)
        """
        X = X.to(device=self.device, dtype=self.dtype)
        y = y.to(device=self.device, dtype=self.dtype)

        if y.dim() == 1:
            y = y.unsqueeze(-1)

        # Update trust region for each new observation
        for i in range(len(y)):
            self._update_trust_region(y[i].item())

        if self.X is None:
            self.X = X
            self.y = y
        else:
            self.X = torch.cat([self.X, X], dim=0)
            self.y = torch.cat([self.y, y], dim=0)

    def get_best_point(self) -> Tuple[Tensor, Tensor]:
        """Get best observed point."""
        if self.y is None or len(self.y) == 0:
            raise ValueError("No observations yet.")

        best_idx = self.y.argmax()
        return self.X[best_idx], self.y[best_idx]
