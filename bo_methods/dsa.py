"""
DSA: Dimension Scheduling Algorithm for Bayesian Optimization.

Reference:
    Ulmasov, D., Baroukh, C., Chachuat, B., Deisenroth, M. P., & Misener, R. (2016).
    "Bayesian Optimization with Dimension Scheduling: Application to Biological Systems".
    26th European Symposium on Computer Aided Process Engineering.

DSA distributes training data across multiple GPs, each containing data from a subset
of dimensions. At each iteration, a new subset is sampled based on importance.
"""

import torch
from typing import Optional, Tuple, List
from torch import Tensor

from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

from .base import BaseOptimizer


class DSA(BaseOptimizer):
    """
    Dimension Scheduling Algorithm (DSA).

    Optimizes the objective function along a subset of dimensions at each iteration.
    The subset is sampled from a probability distribution that reflects parameter
    importance based on historical observations.

    Key Features:
    - Dimension importance estimation via variance analysis
    - Random dimension subsets for each iteration
    - Reduced computational cost per iteration
    - Can identify important dimensions adaptively

    Args:
        input_dim: Dimensionality of the input space
        bounds: Tensor of shape (2, input_dim) with bounds
        n_active_dims: Number of dimensions to optimize per iteration
        use_pca: Whether to use PCA for dimension importance (default: False)
        importance_update_freq: How often to update dimension importance (default: 5)
        device: Device for computations
        dtype: Data type for tensors
    """

    def __init__(
        self,
        input_dim: int,
        bounds: Tensor,
        n_active_dims: Optional[int] = None,
        use_pca: bool = False,
        importance_update_freq: int = 5,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__(input_dim, bounds, device, dtype)

        # Default to sqrt(d) active dimensions
        if n_active_dims is None:
            n_active_dims = max(2, int(input_dim ** 0.5))

        self.n_active_dims = min(n_active_dims, input_dim)
        self.use_pca = use_pca
        self.importance_update_freq = importance_update_freq

        # Dimension importance weights (uniform initially)
        self.dim_importance = torch.ones(
            input_dim, device=device, dtype=dtype
        ) / input_dim

        self.iteration_count = 0
        self.model: Optional[SingleTaskGP] = None

    def _update_dimension_importance(self):
        """
        Update dimension importance based on observed data.

        Uses variance in each dimension or PCA-based importance.
        """
        if self.X is None or len(self.X) < 3:
            # Not enough data, keep uniform
            return

        if self.use_pca and len(self.X) >= self.input_dim:
            # Use PCA for dimension importance
            self._update_importance_pca()
        else:
            # Use simple variance-based importance
            self._update_importance_variance()

    def _update_importance_variance(self):
        """
        Estimate importance based on variance and correlation with objective.
        """
        # Standardize X
        X_std = (self.X - self.X.mean(dim=0)) / (self.X.std(dim=0) + 1e-6)
        y_std = (self.y - self.y.mean()) / (self.y.std() + 1e-6)

        # Compute correlation with objective for each dimension
        correlations = torch.abs(
            (X_std * y_std).mean(dim=0)
        )

        # Compute variance in each dimension
        variances = self.X.var(dim=0)

        # Combine correlation and variance
        # High correlation or high variance suggests importance
        importance = correlations * 0.7 + (variances / variances.max()) * 0.3

        # Normalize to probability distribution
        self.dim_importance = importance / importance.sum()

        # Ensure minimum probability for exploration
        min_prob = 0.01 / self.input_dim
        self.dim_importance = torch.clamp(self.dim_importance, min=min_prob)
        self.dim_importance = self.dim_importance / self.dim_importance.sum()

    def _update_importance_pca(self):
        """
        Use PCA to determine dimension importance.
        """
        # Center the data
        X_centered = self.X - self.X.mean(dim=0)

        # Compute covariance matrix
        cov = (X_centered.T @ X_centered) / (len(self.X) - 1)

        # Eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)

        # Sort by eigenvalues (descending)
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Importance based on contribution to top principal components
        # Use top k components (k = n_active_dims)
        n_components = min(self.n_active_dims, len(eigenvalues))
        top_components = eigenvectors[:, :n_components]

        # Sum absolute loadings across top components
        importance = torch.abs(top_components).sum(dim=1)

        # Normalize
        self.dim_importance = importance / importance.sum()

        # Ensure minimum probability
        min_prob = 0.01 / self.input_dim
        self.dim_importance = torch.clamp(self.dim_importance, min=min_prob)
        self.dim_importance = self.dim_importance / self.dim_importance.sum()

    def _sample_active_dimensions(self) -> List[int]:
        """
        Sample a subset of dimensions based on importance weights.

        Returns:
            List of dimension indices to optimize
        """
        # Sample without replacement
        sampled_dims = torch.multinomial(
            self.dim_importance,
            num_samples=self.n_active_dims,
            replacement=False
        ).tolist()

        return sorted(sampled_dims)

    def _create_subspace_bounds(
        self,
        active_dims: List[int],
        center: Tensor
    ) -> Tensor:
        """
        Create bounds for optimization in the active subspace.

        Inactive dimensions are fixed at center values.

        Args:
            active_dims: List of active dimension indices
            center: Current best point

        Returns:
            Bounds tensor of shape (2, input_dim)
        """
        bounds = torch.zeros_like(self.bounds)

        for dim in range(self.input_dim):
            if dim in active_dims:
                # Active dimension: use full bounds
                bounds[0, dim] = self.bounds[0, dim]
                bounds[1, dim] = self.bounds[1, dim]
            else:
                # Inactive dimension: fix at center value
                bounds[0, dim] = center[dim]
                bounds[1, dim] = center[dim]

        return bounds

    def _fit_model(self):
        """Fit GP model to observed data."""
        # Standardize outputs manually to avoid BoTorch input warnings
        y_mean = self.y.mean()
        y_std = self.y.std()
        if y_std < 1e-6:
            y_std = torch.tensor(1.0, device=self.device, dtype=self.dtype)
        y_stdized = (self.y - y_mean) / y_std

        self.model = SingleTaskGP(
            train_X=self.X,
            train_Y=y_stdized,
            input_transform=Normalize(d=self.input_dim, bounds=self.bounds),
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

        # Update dimension importance periodically
        if self.iteration_count % self.importance_update_freq == 0:
            self._update_dimension_importance()

        self.iteration_count += 1

        # Fit model
        self._fit_model()

        # Get current best point as anchor
        best_idx = self.y.argmax()
        best_point = self.X[best_idx]

        # Sample active dimensions
        active_dims = self._sample_active_dimensions()

        print(f"DSA: Optimizing dimensions {active_dims} (importance: {self.dim_importance[active_dims].cpu().numpy()})")

        # Create subspace bounds
        subspace_bounds = self._create_subspace_bounds(active_dims, best_point)

        # Optimize acquisition function
        acq_func = ExpectedImprovement(
            model=self.model,
            best_f=self.y.max().item()
        )

        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=subspace_bounds,
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
