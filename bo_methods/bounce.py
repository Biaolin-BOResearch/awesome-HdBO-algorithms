"""
Bounce: Reliable High-Dimensional Bayesian Optimization for Combinatorial and Mixed Spaces.

Reference:
    Papenmeier, L., Nardi, L., & Poloczek, M. (2023). "Bounce: Reliable
    High-Dimensional Bayesian Optimization for Combinatorial and Mixed Spaces".
    NeurIPS 2023.

Bounce uses nested embeddings by partitioning variables into 'bins' (target
dimensions), enabling reliable optimization in mixed continuous/discrete spaces.
"""

import torch
from typing import Optional, Tuple, List
from torch import Tensor
import numpy as np

from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

from .base import BaseOptimizer


class Bounce(BaseOptimizer):
    """
    Bounce: Reliable High-Dimensional Bayesian Optimization.

    Uses nested embeddings via variable binning into target dimensions.
    Handles continuous, discrete, and mixed variable types.

    Key Features:
    - Nested binning structure for embeddings
    - Works with mixed variable types
    - Reliable performance across different optima structures
    - Lower-dimensional target space

    Args:
        input_dim: Dimensionality of input space
        bounds: Tensor of shape (2, input_dim) with bounds
        target_dim: Target embedding dimensionality (default: sqrt(d))
        n_bins_per_target: Number of bins per target dimension (default: 2)
        discrete_dims: List of discrete dimension indices (default: None)
        categorical_dims: Dict mapping dim index to num categories (default: None)
        device: Device for computations
        dtype: Data type for tensors
    """

    def __init__(
        self,
        input_dim: int,
        bounds: Tensor,
        target_dim: Optional[int] = None,
        n_bins_per_target: int = 2,
        discrete_dims: Optional[List[int]] = None,
        categorical_dims: Optional[dict] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__(input_dim, bounds, device, dtype)

        # Default target dim: sqrt(d)
        if target_dim is None:
            target_dim = max(2, int(np.sqrt(input_dim)))
        self.target_dim = min(target_dim, input_dim)

        self.n_bins_per_target = n_bins_per_target

        # Variable type specifications
        self.discrete_dims = discrete_dims if discrete_dims is not None else []
        self.categorical_dims = categorical_dims if categorical_dims is not None else {}

        # Create binning structure (partition input dims into target dims)
        self.bin_assignment = self._create_bin_assignment()

        self.model: Optional[SingleTaskGP] = None

    def _create_bin_assignment(self) -> Tensor:
        """
        Create binning assignment: map each input dim to a target bin.

        Returns:
            Tensor of shape (input_dim,) with bin assignments
        """
        # Simple round-robin assignment
        # Each input dimension is assigned to a target dimension
        bin_indices = torch.arange(self.input_dim, device=self.device) % self.target_dim

        # Shuffle for better mixing
        perm = torch.randperm(self.input_dim, device=self.device)
        return bin_indices[perm]

    def _embed_to_target(self, X: Tensor) -> Tensor:
        """
        Embed high-dimensional points to target space via binning.

        Args:
            X: Tensor of shape (n, input_dim)

        Returns:
            Tensor of shape (n, target_dim)
        """
        n = X.shape[0]
        X_target = torch.zeros(n, self.target_dim, device=self.device, dtype=self.dtype)

        # For each target dimension, aggregate assigned input dimensions
        for t in range(self.target_dim):
            # Find input dimensions assigned to this target dimension
            mask = self.bin_assignment == t
            assigned_dims = torch.where(mask)[0]

            if len(assigned_dims) > 0:
                # Aggregate: take mean of assigned dimensions
                X_target[:, t] = X[:, assigned_dims].mean(dim=1)

        return X_target

    def _unembed_from_target(
        self,
        X_target: Tensor,
        reference: Optional[Tensor] = None
    ) -> Tensor:
        """
        Map target space points back to input space.

        Args:
            X_target: Tensor of shape (n, target_dim)
            reference: Reference point to use for unassigned structure

        Returns:
            Tensor of shape (n, input_dim)
        """
        n = X_target.shape[0]

        if reference is None:
            # Use center of bounds as reference
            reference = (self.bounds[0] + self.bounds[1]) / 2

        X_full = reference.unsqueeze(0).repeat(n, 1)

        # For each target dimension, distribute to assigned input dimensions
        for t in range(self.target_dim):
            mask = self.bin_assignment == t
            assigned_dims = torch.where(mask)[0]

            if len(assigned_dims) > 0:
                # Distribute target value to all assigned input dimensions
                for dim in assigned_dims:
                    X_full[:, dim] = X_target[:, t]

        # Handle discrete dimensions by rounding
        for dim in self.discrete_dims:
            X_full[:, dim] = torch.round(X_full[:, dim])

        # Handle categorical dimensions
        for dim, n_categories in self.categorical_dims.items():
            # Discretize to valid category indices
            X_full[:, dim] = torch.clamp(
                torch.round(X_full[:, dim]),
                self.bounds[0, dim],
                self.bounds[1, dim]
            )

        # Clip to bounds
        X_full = torch.clamp(X_full, self.bounds[0], self.bounds[1])

        return X_full

    def _get_target_bounds(self) -> Tensor:
        """
        Get bounds for target space based on input space bounds.

        Returns:
            Tensor of shape (2, target_dim)
        """
        target_bounds = torch.zeros(2, self.target_dim, device=self.device, dtype=self.dtype)

        for t in range(self.target_dim):
            # Find input dimensions assigned to this target
            mask = self.bin_assignment == t
            assigned_dims = torch.where(mask)[0]

            if len(assigned_dims) > 0:
                # Target bounds are the average of assigned input bounds
                target_bounds[0, t] = self.bounds[0, assigned_dims].mean()
                target_bounds[1, t] = self.bounds[1, assigned_dims].mean()
            else:
                # Default to [0, 1]
                target_bounds[0, t] = 0.0
                target_bounds[1, t] = 1.0

        return target_bounds

    def _fit_model(self):
        """Fit GP model in target space."""
        # Embed observations to target space
        X_target = self._embed_to_target(self.X)
        target_bounds = self._get_target_bounds()

        # Standardize outputs manually to avoid BoTorch input warnings
        y_mean = self.y.mean()
        y_std = self.y.std()
        if y_std < 1e-6:
            y_std = torch.tensor(1.0, device=self.device, dtype=self.dtype)
        y_stdized = (self.y - y_mean) / y_std

        # Fit GP in target space with transforms
        self.model = SingleTaskGP(
            train_X=X_target,
            train_Y=y_stdized,
            input_transform=Normalize(d=self.target_dim, bounds=target_bounds),
            outcome_transform=None,
        ).to(device=self.device, dtype=self.dtype)

        self._y_mean = y_mean
        self._y_std = y_std

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

        # Get target space bounds
        target_bounds = self._get_target_bounds()

        # Optimize acquisition function in target space
        acq_func = ExpectedImprovement(
            model=self.model,
            best_f=self.y.max().item()
        )

        candidates_target, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=target_bounds,
            q=n_suggestions,
            num_restarts=10,
            raw_samples=512,
        )

        # Get best observed point as reference for unembedding
        best_idx = self.y.argmax()
        reference = self.X[best_idx]

        # Unembed to input space
        candidates = self._unembed_from_target(candidates_target, reference)

        # For discrete/categorical dims, ensure valid values
        for dim in self.discrete_dims:
            candidates[:, dim] = torch.round(candidates[:, dim])

        for dim, n_categories in self.categorical_dims.items():
            candidates[:, dim] = torch.clamp(
                torch.round(candidates[:, dim]),
                self.bounds[0, dim],
                self.bounds[1, dim]
            )

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

    def set_variable_types(
        self,
        discrete_dims: Optional[List[int]] = None,
        categorical_dims: Optional[dict] = None
    ):
        """
        Update variable type specifications.

        Args:
            discrete_dims: List of discrete dimension indices
            categorical_dims: Dict mapping dim index to num categories
        """
        if discrete_dims is not None:
            self.discrete_dims = discrete_dims

        if categorical_dims is not None:
            self.categorical_dims = categorical_dims
