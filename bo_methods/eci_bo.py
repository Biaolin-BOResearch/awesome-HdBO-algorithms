"""
Expected Coordinate Improvement Bayesian Optimization (ECI-BO).

This method proposes the Expected Coordinate Improvement (ECI) criterion
to measure improvement along individual coordinates. It extends the
standard Expected Improvement to coordinate-wise optimization.

Key features:
1. ECI criterion: Measures expected improvement along each coordinate
2. Coordinate prioritization: Sorts coordinates by their maximal ECI values
3. Sequential coordinate optimization: Optimizes one coordinate at a time

Algorithm 2 (from paper):
1. Initialize with n_init samples
2. While budget not exhausted:
   - For each coordinate i, compute ECI_{i,max} = max ECI_i(x)
   - Sort coordinates by ECI_{i,max} in descending order
   - For each coordinate in sorted order:
     * Train GP model
     * Find x that maximizes ECI along that coordinate
     * Evaluate the new point
     * Update best solution
"""

import math
from typing import Optional, Tuple, List
import torch
from torch import Tensor
from scipy.stats import norm
from scipy.optimize import minimize_scalar, minimize

from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.priors import GammaPrior
from gpytorch.constraints import GreaterThan

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

from .base import BaseOptimizer
from .utils import DEVICE, DTYPE


def expected_improvement(mu: float, sigma: float, f_best: float) -> float:
    """
    Compute Expected Improvement.

    EI(x) = (f_best - mu) * Φ((f_best - mu) / sigma) + sigma * φ((f_best - mu) / sigma)

    For minimization, f_best is the current minimum.

    Args:
        mu: Posterior mean
        sigma: Posterior standard deviation
        f_best: Current best (minimum) value

    Returns:
        Expected improvement value
    """
    if sigma <= 1e-10:
        return 0.0

    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    return max(ei, 0.0)


class ECIBO(BaseOptimizer):
    """
    Expected Coordinate Improvement Bayesian Optimization.

    This method optimizes one coordinate at a time, using the Expected
    Coordinate Improvement (ECI) criterion. It prioritizes coordinates
    based on their potential for improvement.

    The ECI for coordinate i is defined as:
        ECI_i(x) = (f* - μ(z))Φ((f* - μ(z))/σ(z)) + σ(z)φ((f* - μ(z))/σ(z))

    where z = [x*_1, ..., x*_{i-1}, x, x*_{i+1}, ..., x*_d]

    This means we fix all coordinates except i to their current best values,
    and compute the EI as a function of the i-th coordinate only.

    Args:
        input_dim: Dimensionality of the input space
        bounds: Tensor of shape (2, input_dim) with lower and upper bounds
        n_coordinates_per_iter: Number of coordinates to optimize per iteration
            (default: all coordinates, i.e., input_dim)
        use_coordinate_sorting: Whether to sort coordinates by ECI_max
        device: Device to use for computations
        dtype: Data type for tensors
    """

    def __init__(
        self,
        input_dim: int,
        bounds: Tensor,
        n_coordinates_per_iter: Optional[int] = None,
        use_coordinate_sorting: bool = True,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
    ):
        # Initialize base class
        _device = device or DEVICE
        _dtype = dtype or DTYPE
        super().__init__(
            input_dim=input_dim,
            bounds=bounds,
            acq_func="EI",  # We use custom ECI
            device=_device,
            dtype=_dtype,
        )

        # Number of coordinates to optimize per iteration
        self.n_coordinates_per_iter = n_coordinates_per_iter or input_dim
        self.use_coordinate_sorting = use_coordinate_sorting

        # Model placeholder
        self.model: Optional[SingleTaskGP] = None
        self.mll: Optional[ExactMarginalLogLikelihood] = None

        # Training data
        self.X: Optional[Tensor] = None
        self.y: Optional[Tensor] = None

        # Current best solution
        self.x_best: Optional[Tensor] = None
        self.f_best: Optional[float] = None

    def _create_covar_module(self):
        """Create covariance module for GP."""
        covar_module = ScaleKernel(
            base_kernel=MaternKernel(
                nu=2.5,
                ard_num_dims=self.input_dim,
                lengthscale_constraint=GreaterThan(1e-4),
            ),
            outputscale_prior=GammaPrior(2.0, 0.15),
            outputscale_constraint=GreaterThan(1e-4),
        )
        return covar_module

    def _fit_model(self):
        """Fit GP model to current data."""
        covar_module = self._create_covar_module()

        self.model = SingleTaskGP(
            train_X=self.X,
            train_Y=self.y,
            covar_module=covar_module,
        ).to(device=self.device, dtype=self.dtype)

        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(self.mll)

    def _compute_eci(self, x_value: float, coord_idx: int) -> float:
        """
        Compute Expected Coordinate Improvement for a given coordinate value.

        ECI_i(x) = EI evaluated at z where:
        z = [x*_1, ..., x*_{i-1}, x, x*_{i+1}, ..., x*_d]

        Args:
            x_value: Value for coordinate coord_idx
            coord_idx: Index of the coordinate being optimized

        Returns:
            ECI value (negated for minimization-based optimizer)
        """
        # Construct the full point z
        z = self.x_best.clone()
        z[coord_idx] = x_value

        # Get posterior prediction
        self.model.eval()
        with torch.no_grad():
            z_tensor = z.unsqueeze(0)  # (1, d)
            posterior = self.model(z_tensor)
            mu = posterior.mean.item()
            sigma = posterior.variance.sqrt().item()

        # Compute EI (for minimization, f_best is minimum)
        ei = expected_improvement(mu, sigma, self.f_best)

        return ei

    def _maximize_eci_along_coordinate(self, coord_idx: int) -> Tuple[float, float]:
        """
        Maximize ECI along a single coordinate.

        Args:
            coord_idx: Index of the coordinate to optimize

        Returns:
            Tuple of (optimal_x_value, max_eci_value)
        """
        lb = self.bounds[0, coord_idx].item()
        ub = self.bounds[1, coord_idx].item()

        # Use scipy minimize_scalar for 1D optimization
        def neg_eci(x):
            return -self._compute_eci(x, coord_idx)

        # Try multiple starting points
        best_x = self.x_best[coord_idx].item()
        best_eci = self._compute_eci(best_x, coord_idx)

        # Grid search for initial points
        n_grid = 20
        grid_points = torch.linspace(lb, ub, n_grid)

        for x_init in grid_points:
            try:
                result = minimize_scalar(
                    neg_eci,
                    bounds=(lb, ub),
                    method='bounded',
                    options={'xatol': 1e-6}
                )
                if result.success:
                    eci_val = -result.fun
                    if eci_val > best_eci:
                        best_eci = eci_val
                        best_x = result.x
            except Exception:
                continue

        return best_x, best_eci

    def _compute_all_eci_max(self) -> List[Tuple[int, float]]:
        """
        Compute maximum ECI for all coordinates.

        Returns:
            List of (coordinate_index, max_eci) tuples sorted by ECI descending
        """
        eci_max_list = []

        for i in range(self.input_dim):
            _, max_eci = self._maximize_eci_along_coordinate(i)
            eci_max_list.append((i, max_eci))

        # Sort by ECI in descending order
        eci_max_list.sort(key=lambda x: x[1], reverse=True)

        return eci_max_list

    def observe(self, X: Tensor, y: Tensor):
        """
        Update the optimizer with new observations.

        Args:
            X: Tensor of shape (n, input_dim) with input points
            y: Tensor of shape (n,) or (n, 1) with observed values
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

        # Update best solution (for minimization)
        best_idx = self.y.argmin()
        self.x_best = self.X[best_idx].clone()
        self.f_best = self.y[best_idx].item()

    def suggest(self, n_suggestions: int = 1) -> Tensor:
        """
        Suggest next point(s) to evaluate using ECI-BO.

        For ECI-BO, we typically suggest one point at a time by optimizing
        along coordinates sequentially. For batch suggestions, we repeat
        the coordinate optimization process.

        Args:
            n_suggestions: Number of points to suggest

        Returns:
            Tensor of shape (n_suggestions, input_dim) with suggested points
        """
        if self.X is None or len(self.X) == 0:
            raise ValueError("No observations yet. Call observe() first.")

        suggestions = []

        for _ in range(n_suggestions):
            # Fit GP model
            self._fit_model()

            # Get coordinate priority order
            if self.use_coordinate_sorting:
                sorted_coords = self._compute_all_eci_max()
            else:
                # Use natural order
                sorted_coords = [(i, 0.0) for i in range(self.input_dim)]

            # Take top n_coordinates_per_iter coordinates
            coords_to_optimize = sorted_coords[:self.n_coordinates_per_iter]

            # Start from current best
            new_x = self.x_best.clone()

            # Optimize along each coordinate in priority order
            for coord_idx, _ in coords_to_optimize:
                optimal_x, _ = self._maximize_eci_along_coordinate(coord_idx)
                new_x[coord_idx] = optimal_x

            suggestions.append(new_x)

            # For batch suggestions, we need to update the model
            # Here we use a simple approach: just add the suggested point
            # with a predicted value (Thompson sampling style)
            if n_suggestions > 1 and len(suggestions) < n_suggestions:
                # Use posterior mean as pseudo-observation
                self.model.eval()
                with torch.no_grad():
                    pred = self.model(new_x.unsqueeze(0))
                    pseudo_y = pred.mean

                self.X = torch.cat([self.X, new_x.unsqueeze(0)], dim=0)
                self.y = torch.cat([self.y, pseudo_y.unsqueeze(-1)], dim=0)

        return torch.stack(suggestions)

    def suggest_one_coordinate(self) -> Tuple[Tensor, int]:
        """
        Suggest next point by optimizing a single coordinate (pure ECI-BO).

        This follows Algorithm 2 more closely: optimize one coordinate,
        evaluate, then move to the next coordinate.

        Returns:
            Tuple of (suggested_point, optimized_coordinate_index)
        """
        if self.X is None or len(self.X) == 0:
            raise ValueError("No observations yet. Call observe() first.")

        # Fit GP model
        self._fit_model()

        # Get coordinate with maximum ECI
        sorted_coords = self._compute_all_eci_max()
        best_coord_idx = sorted_coords[0][0]

        # Optimize along the best coordinate
        optimal_x, _ = self._maximize_eci_along_coordinate(best_coord_idx)

        # Construct the new point
        new_x = self.x_best.clone()
        new_x[best_coord_idx] = optimal_x

        return new_x, best_coord_idx

    def get_best_point(self) -> Tuple[Tensor, Tensor]:
        """
        Get the best observed point so far.

        Returns:
            Tuple of (best_X, best_y)
        """
        if self.y is None or len(self.y) == 0:
            raise ValueError("No observations yet.")

        best_idx = self.y.argmin()  # Minimization
        return self.X[best_idx], self.y[best_idx]

    def reset(self) -> None:
        """Reset the optimizer to its initial state."""
        self.X = None
        self.y = None
        self.model = None
        self.mll = None
        self.x_best = None
        self.f_best = None


class ECIBOMaximize(ECIBO):
    """
    ECI-BO variant for maximization problems.

    Same as ECIBO but for maximization instead of minimization.
    """

    def _compute_eci(self, x_value: float, coord_idx: int) -> float:
        """
        Compute ECI for maximization.

        For maximization: EI = (μ - f_best)Φ((μ - f_best)/σ) + σφ((μ - f_best)/σ)
        """
        z = self.x_best.clone()
        z[coord_idx] = x_value

        self.model.eval()
        with torch.no_grad():
            z_tensor = z.unsqueeze(0)
            posterior = self.model(z_tensor)
            mu = posterior.mean.item()
            sigma = posterior.variance.sqrt().item()

        if sigma <= 1e-10:
            return 0.0

        # For maximization
        z_score = (mu - self.f_best) / sigma
        ei = (mu - self.f_best) * norm.cdf(z_score) + sigma * norm.pdf(z_score)
        return max(ei, 0.0)

    def observe(self, X: Tensor, y: Tensor):
        """Update with new observations (maximization)."""
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

        # Update best solution (for maximization)
        best_idx = self.y.argmax()
        self.x_best = self.X[best_idx].clone()
        self.f_best = self.y[best_idx].item()

    def get_best_point(self) -> Tuple[Tensor, Tensor]:
        """Get the best observed point (maximization)."""
        if self.y is None or len(self.y) == 0:
            raise ValueError("No observations yet.")

        best_idx = self.y.argmax()
        return self.X[best_idx], self.y[best_idx]

