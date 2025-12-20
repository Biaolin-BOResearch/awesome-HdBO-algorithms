"""
SOLID: Sequential Optimization in Locally Important Dimensions.

Reference:
    Winkel, M. A., Stallrich, J. W., Storlie, C. B., & Reich, B. J. (2021).
    "Sequential Optimization in Locally Important Dimensions". Technometrics, 63(2), 236-248.

SOLID performs both global and local variable selection to identify which
dimensions are important near the current optimum. It restricts the acquisition
function search space to locally active dimensions.
"""

import torch
from typing import Optional, Tuple, Set
from torch import Tensor

from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

from .base import BaseOptimizer


class SOLID(BaseOptimizer):
    """
    SOLID: Sequential Optimization in Locally Important Dimensions.

    Performs variable selection to identify globally and locally important
    dimensions, then focuses optimization on those dimensions.

    Key Features:
    - Global variable selection: identifies dimensions that never matter
    - Local variable selection: identifies dimensions that matter near optimum
    - Acquisition optimization restricted to active dimensions
    - Adaptive dimension selection based on surrogate model predictions

    Args:
        input_dim: Dimensionality of the input space
        bounds: Tensor of shape (2, input_dim) with bounds
        global_threshold: Threshold for global variable selection (default: 0.01)
        local_threshold: Threshold for local variable selection (default: 0.05)
        local_region_size: Size of local region around optimum (default: 0.1)
        min_active_dims: Minimum number of active dimensions (default: 2)
        acq_func: Acquisition function to use (default: 'EI')
        device: Device for computations
        dtype: Data type for tensors
    """

    def __init__(
        self,
        input_dim: int,
        bounds: Tensor,
        global_threshold: float = 0.01,
        local_threshold: float = 0.05,
        local_region_size: float = 0.1,
        min_active_dims: int = 2,
        acq_func: str = 'EI',
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__(input_dim, bounds, device, dtype)

        self.global_threshold = global_threshold
        self.local_threshold = local_threshold
        self.local_region_size = local_region_size
        self.min_active_dims = min_active_dims
        self.acq_func_name = acq_func

        # Track active dimensions
        self.globally_active_dims: Set[int] = set(range(input_dim))
        self.locally_active_dims: Set[int] = set(range(input_dim))

        self.model: Optional[SingleTaskGP] = None

    def _perform_global_variable_selection(self):
        """
        Identify globally active dimensions using lengthscale analysis.

        Dimensions with very large lengthscales relative to the domain
        are considered inactive globally.
        """
        if self.model is None:
            return

        # Get lengthscales from the kernel
        try:
            lengthscales = self.model.covar_module.base_kernel.lengthscale.detach()
        except AttributeError:
            # If no ARD kernel, assume all dimensions are active
            return

        if lengthscales.dim() == 0:
            # Isotropic kernel - all dimensions equally important
            return

        lengthscales = lengthscales.squeeze()

        # Compute domain ranges
        domain_ranges = self.bounds[1] - self.bounds[0]

        # Dimensions are globally inactive if lengthscale >> domain range
        # This means the function varies very slowly in that dimension
        relative_lengthscales = lengthscales / domain_ranges

        # Identify active dimensions (where lengthscale is not too large)
        active_mask = relative_lengthscales < (1.0 / self.global_threshold)

        self.globally_active_dims = set(
            torch.where(active_mask)[0].cpu().tolist()
        )

        # Ensure minimum number of active dimensions
        if len(self.globally_active_dims) < self.min_active_dims:
            # Keep dimensions with smallest relative lengthscales
            sorted_dims = torch.argsort(relative_lengthscales)
            self.globally_active_dims = set(
                sorted_dims[:self.min_active_dims].cpu().tolist()
            )

    def _perform_local_variable_selection(self, center: Tensor):
        """
        Identify locally active dimensions near the current best point.

        For each dimension, we compare predictions with and without
        varying that dimension in a local region around the center.

        Args:
            center: Current best point (shape: input_dim)
        """
        if self.model is None:
            return

        # Only consider globally active dimensions
        candidate_dims = sorted(self.globally_active_dims)

        if len(candidate_dims) == 0:
            candidate_dims = list(range(self.input_dim))

        # Generate local region samples
        n_samples = 50
        local_samples = self._generate_local_samples(center, n_samples)

        # Get baseline predictions
        with torch.no_grad():
            posterior = self.model.posterior(local_samples)
            baseline_mean = posterior.mean.squeeze(-1)
            baseline_std = baseline_mean.std()

        # For each dimension, measure impact of removing it
        importance_scores = torch.zeros(self.input_dim, device=self.device, dtype=self.dtype)

        for dim in candidate_dims:
            # Create samples with dimension fixed at center value
            fixed_samples = local_samples.clone()
            fixed_samples[:, dim] = center[dim]

            # Get predictions with dimension fixed
            with torch.no_grad():
                posterior_fixed = self.model.posterior(fixed_samples)
                fixed_mean = posterior_fixed.mean.squeeze(-1)

            # Measure change in predictions
            change = torch.abs(baseline_mean - fixed_mean).mean()

            # Normalize by baseline variability
            if baseline_std > 1e-6:
                importance_scores[dim] = change / baseline_std
            else:
                importance_scores[dim] = change

        # Select dimensions with importance above threshold
        active_mask = importance_scores > self.local_threshold

        self.locally_active_dims = set(
            torch.where(active_mask)[0].cpu().tolist()
        )

        # Ensure minimum number of active dimensions
        if len(self.locally_active_dims) < self.min_active_dims:
            # Keep dimensions with highest importance
            sorted_dims = torch.argsort(importance_scores, descending=True)
            self.locally_active_dims = set(
                sorted_dims[:self.min_active_dims].cpu().tolist()
            )

        # Intersect with globally active dimensions
        self.locally_active_dims = self.locally_active_dims.intersection(
            self.globally_active_dims
        )

        if len(self.locally_active_dims) < self.min_active_dims:
            # Fall back to globally active dimensions
            self.locally_active_dims = set(
                sorted(self.globally_active_dims)[:self.min_active_dims]
            )

    def _generate_local_samples(self, center: Tensor, n_samples: int) -> Tensor:
        """
        Generate samples in a local region around center.

        Args:
            center: Center point (shape: input_dim)
            n_samples: Number of samples

        Returns:
            Tensor of shape (n_samples, input_dim)
        """
        # Sample from Gaussian around center
        std = self.local_region_size * (self.bounds[1] - self.bounds[0])

        samples = torch.randn(
            n_samples, self.input_dim,
            device=self.device, dtype=self.dtype
        ) * std + center

        # Clip to bounds
        samples = torch.clamp(samples, self.bounds[0], self.bounds[1])

        return samples

    def _create_restricted_bounds(self, center: Tensor) -> Tensor:
        """
        Create bounds that fix inactive dimensions at center values.

        Args:
            center: Current best point

        Returns:
            Modified bounds tensor
        """
        bounds = self.bounds.clone()

        # Fix inactive dimensions
        for dim in range(self.input_dim):
            if dim not in self.locally_active_dims:
                bounds[0, dim] = center[dim]
                bounds[1, dim] = center[dim]

        return bounds

    def _fit_model(self):
        """Fit GP model with ARD kernel."""
        from gpytorch.kernels import ScaleKernel, MaternKernel

        covar_module = ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=self.input_dim,
            )
        )

        # Standardize outputs manually to avoid BoTorch input warnings
        y_mean = self.y.mean()
        y_std = self.y.std()
        if y_std < 1e-6:
            y_std = torch.tensor(1.0, device=self.device, dtype=self.dtype)
        y_stdized = (self.y - y_mean) / y_std

        self.model = SingleTaskGP(
            train_X=self.X,
            train_Y=y_stdized,
            covar_module=covar_module,
            input_transform=Normalize(d=self.input_dim, bounds=self.bounds),
            outcome_transform=None,
        ).to(device=self.device, dtype=self.dtype)

        # Save stats for potential post-processing
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

        # Fit model with ARD kernel
        self._fit_model()

        # Perform global variable selection
        self._perform_global_variable_selection()

        # Get current best point
        best_idx = self.y.argmax()
        best_point = self.X[best_idx]

        # Perform local variable selection around best point
        self._perform_local_variable_selection(best_point)

        print(f"SOLID: Globally active dims: {len(self.globally_active_dims)}/{self.input_dim}")
        print(f"SOLID: Locally active dims: {len(self.locally_active_dims)}/{self.input_dim}")

        # Create restricted bounds (fix inactive dimensions)
        restricted_bounds = self._create_restricted_bounds(best_point)

        # Optimize acquisition function in restricted space
        acq_func = ExpectedImprovement(
            model=self.model,
            best_f=self.y.max().item()
        )

        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=restricted_bounds,
            q=n_suggestions,
            num_restarts=10,
            raw_samples=512,
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
