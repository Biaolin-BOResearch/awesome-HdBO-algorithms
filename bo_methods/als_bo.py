"""
Adaptive Line Search Bayesian Optimization (ALS-BO).

This method generates an adaptive search direction based on ECI (Expected
Coordinate Improvement) and then performs line search optimization along
that direction.

Algorithm 1 (from paper):
1. Initial design: Generate initial samples
2. While N < N_max:
   - GP building: Build GP model on data set D
   - Adaptive search direction: Generate direction d from ECI solutions
     - For each coordinate i: x'_i = argmax ECI_i(x), r_i = x'_i - x*_i
     - d = [r_1, r_2, ..., r_D] / ||[r_1, r_2, ..., r_D]||
   - Acquisition optimization: Find η* = argmax EI(d, η) along the line
     - x^(new) = x_min + η* * d
   - Expensive evaluation: Evaluate f(x^(new))
   - Best solution update: Update (x_min, f_min)

Key difference from ECI-BO:
- ECI-BO optimizes coordinates sequentially
- ALS-BO uses ECI to construct a single search direction, then does line search
"""

import math
from typing import Optional, Tuple, List
import torch
from torch import Tensor
from scipy.stats import norm
from scipy.optimize import minimize_scalar

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
    Compute Expected Improvement (for minimization).

    EI(x) = (f_best - mu) * Φ(z) + sigma * φ(z)
    where z = (f_best - mu) / sigma

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


class ALSBO(BaseOptimizer):
    """
    Adaptive Line Search Bayesian Optimization.

    This method uses Expected Coordinate Improvement (ECI) to construct
    an adaptive search direction, then optimizes the acquisition function
    along that line.

    The algorithm:
    1. For each coordinate i, find x'_i = argmax ECI_i(x)
    2. Compute direction: r_i = x'_i - x*_i for each i
    3. Normalize: d = [r_1, ..., r_D] / ||[r_1, ..., r_D]||
    4. Line search: η* = argmax EI(x_min + η*d)
    5. New point: x^(new) = x_min + η* * d

    Args:
        input_dim: Dimensionality of the input space
        bounds: Tensor of shape (2, input_dim) with lower and upper bounds
        eta_min: Minimum step size for line search (default: -1.0)
        eta_max: Maximum step size for line search (default: 1.0)
        n_eta_samples: Number of samples for line search initialization
        device: Device to use for computations
        dtype: Data type for tensors
    """

    def __init__(
        self,
        input_dim: int,
        bounds: Tensor,
        eta_min: float = -2.0,
        eta_max: float = 2.0,
        n_eta_samples: int = 100,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
    ):
        # Initialize base class
        _device = device or DEVICE
        _dtype = dtype or DTYPE
        super().__init__(
            input_dim=input_dim,
            bounds=bounds,
            acq_func="EI",
            device=_device,
            dtype=_dtype,
        )

        self.eta_min = eta_min
        self.eta_max = eta_max
        self.n_eta_samples = n_eta_samples

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

        # Normalize data for GP
        X_norm = self._normalize_X(self.X)
        self._update_y_statistics()
        y_std = self._standardize_y(self.y)
        
        self.model = SingleTaskGP(
            train_X=X_norm,
            train_Y=y_std,
            covar_module=covar_module,
            input_transform=None,
            outcome_transform=None,
        ).to(device=self.device, dtype=self.dtype)

        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(self.mll)

    def _compute_eci(self, x_value: float, coord_idx: int) -> float:
        """
        Compute Expected Coordinate Improvement for a given coordinate value.

        Args:
            x_value: Value for coordinate coord_idx
            coord_idx: Index of the coordinate being optimized

        Returns:
            ECI value
        """
        # Construct the full point z = [x*_1, ..., x*_{i-1}, x, x*_{i+1}, ..., x*_d]
        z = self.x_best.clone()
        z[coord_idx] = x_value

        # Get posterior prediction
        self.model.eval()
        with torch.no_grad():
            z_tensor = z.unsqueeze(0)
            posterior = self.model(z_tensor)
            mu = posterior.mean.item()
            sigma = posterior.variance.sqrt().item()

        # Compute EI
        ei = expected_improvement(mu, sigma, self.f_best)
        return ei

    def _maximize_eci_along_coordinate(self, coord_idx: int) -> float:
        """
        Find x'_i = argmax ECI_i(x) along coordinate i.

        Args:
            coord_idx: Index of the coordinate to optimize

        Returns:
            Optimal x value for this coordinate
        """
        lb = self.bounds[0, coord_idx].item()
        ub = self.bounds[1, coord_idx].item()

        def neg_eci(x):
            return -self._compute_eci(x, coord_idx)

        # Start from current best
        best_x = self.x_best[coord_idx].item()
        best_eci = self._compute_eci(best_x, coord_idx)

        # Try multiple starting points using bounded optimization
        try:
            result = minimize_scalar(
                neg_eci,
                bounds=(lb, ub),
                method='bounded',
                options={'xatol': 1e-6}
            )
            if result.success and -result.fun > best_eci:
                best_x = result.x
        except Exception:
            pass

        return best_x

    def _compute_adaptive_direction(self) -> Tensor:
        """
        Generate adaptive search direction d from ECI solutions.

        For each coordinate i:
            x'_i = argmax ECI_i(x)
            r_i = x'_i - x*_i

        d = [r_1, r_2, ..., r_D] / ||[r_1, r_2, ..., r_D]||

        Returns:
            Normalized direction vector of shape (input_dim,)
        """
        r = torch.zeros(self.input_dim, device=self.device, dtype=self.dtype)

        for i in range(self.input_dim):
            x_prime_i = self._maximize_eci_along_coordinate(i)
            r[i] = x_prime_i - self.x_best[i].item()

        # Normalize the direction
        norm_r = torch.linalg.norm(r)
        if norm_r < 1e-10:
            # If direction is zero, use a random direction
            r = torch.randn(self.input_dim, device=self.device, dtype=self.dtype)
            norm_r = torch.linalg.norm(r)

        d = r / norm_r
        return d

    def _compute_ei_along_line(self, eta: float, direction: Tensor) -> float:
        """
        Compute EI at point x_min + η * d.

        Args:
            eta: Step size along the direction
            direction: Normalized search direction

        Returns:
            EI value at the point
        """
        # Compute the point
        x = self.x_best + eta * direction

        # Clamp to bounds
        x = torch.clamp(x, self.bounds[0], self.bounds[1])

        # Get posterior prediction
        self.model.eval()
        with torch.no_grad():
            x_tensor = x.unsqueeze(0)
            posterior = self.model(x_tensor)
            mu = posterior.mean.item()
            sigma = posterior.variance.sqrt().item()

        # Compute EI
        ei = expected_improvement(mu, sigma, self.f_best)
        return ei

    def _optimize_along_line(self, direction: Tensor) -> Tuple[float, Tensor]:
        """
        Find η* = argmax EI(x_min + η*d) and return the new point.

        Args:
            direction: Normalized search direction

        Returns:
            Tuple of (optimal_eta, new_point)
        """
        # Compute bounds for eta based on box constraints
        # We need x_min + η*d to be within bounds
        eta_lb = self.eta_min
        eta_ub = self.eta_max

        # Refine eta bounds based on box constraints
        for i in range(self.input_dim):
            if direction[i] > 1e-10:
                # Upper bound constraint
                eta_ub_i = (self.bounds[1, i] - self.x_best[i]) / direction[i]
                eta_ub = min(eta_ub, eta_ub_i.item())
                # Lower bound constraint
                eta_lb_i = (self.bounds[0, i] - self.x_best[i]) / direction[i]
                eta_lb = max(eta_lb, eta_lb_i.item())
            elif direction[i] < -1e-10:
                # Upper bound constraint
                eta_lb_i = (self.bounds[1, i] - self.x_best[i]) / direction[i]
                eta_lb = max(eta_lb, eta_lb_i.item())
                # Lower bound constraint
                eta_ub_i = (self.bounds[0, i] - self.x_best[i]) / direction[i]
                eta_ub = min(eta_ub, eta_ub_i.item())

        # Ensure valid bounds
        if eta_lb >= eta_ub:
            eta_lb, eta_ub = -0.1, 0.1

        def neg_ei(eta):
            return -self._compute_ei_along_line(eta, direction)

        # Optimize using bounded scalar optimization
        best_eta = 0.0
        best_ei = self._compute_ei_along_line(0.0, direction)

        try:
            result = minimize_scalar(
                neg_ei,
                bounds=(eta_lb, eta_ub),
                method='bounded',
                options={'xatol': 1e-6}
            )
            if result.success and -result.fun > best_ei:
                best_eta = result.x
        except Exception:
            pass

        # Also try grid search for robustness
        eta_samples = torch.linspace(eta_lb, eta_ub, self.n_eta_samples)
        for eta in eta_samples:
            ei = self._compute_ei_along_line(eta.item(), direction)
            if ei > best_ei:
                best_ei = ei
                best_eta = eta.item()

        # Compute the new point
        new_x = self.x_best + best_eta * direction
        new_x = torch.clamp(new_x, self.bounds[0], self.bounds[1])

        return best_eta, new_x

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

        # Also update base class attributes for proper standardization
        self.train_X = self.X
        self.train_y = self.y

        # Update best solution (for minimization)
        best_idx = self.y.argmin()
        self.x_best = self.X[best_idx].clone()
        self.f_best = self.y[best_idx].item()

    def suggest(self, n_suggestions: int = 1) -> Tensor:
        """
        Suggest next point(s) to evaluate using ALS-BO.

        Args:
            n_suggestions: Number of points to suggest

        Returns:
            Tensor of shape (n_suggestions, input_dim) with suggested points
        """
        if self.X is None or len(self.X) == 0:
            raise ValueError("No observations yet. Call observe() first.")

        suggestions = []

        for _ in range(n_suggestions):
            # Step 3: GP building
            self._fit_model()

            # Step 4: Adaptive search direction
            direction = self._compute_adaptive_direction()

            # Step 5: Acquisition function optimization along the line
            _, new_x = self._optimize_along_line(direction)

            suggestions.append(new_x)

            # For batch suggestions, add pseudo-observation
            if n_suggestions > 1 and len(suggestions) < n_suggestions:
                self.model.eval()
                with torch.no_grad():
                    pred = self.model(new_x.unsqueeze(0))
                    pseudo_y = pred.mean

                self.X = torch.cat([self.X, new_x.unsqueeze(0)], dim=0)
                self.y = torch.cat([self.y, pseudo_y.unsqueeze(-1)], dim=0)

        return torch.stack(suggestions)

    def suggest_with_direction(self) -> Tuple[Tensor, Tensor, float]:
        """
        Suggest next point and return the search direction used.

        Returns:
            Tuple of (new_point, direction, optimal_eta)
        """
        if self.X is None or len(self.X) == 0:
            raise ValueError("No observations yet. Call observe() first.")

        # Fit GP model
        self._fit_model()

        # Compute adaptive direction
        direction = self._compute_adaptive_direction()

        # Optimize along the line
        optimal_eta, new_x = self._optimize_along_line(direction)

        return new_x, direction, optimal_eta

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


class ALSBOMaximize(ALSBO):
    """
    ALS-BO variant for maximization problems.
    """

    def _compute_eci(self, x_value: float, coord_idx: int) -> float:
        """Compute ECI for maximization."""
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

        # For maximization: EI = (μ - f_best)Φ(z) + σφ(z)
        z_score = (mu - self.f_best) / sigma
        ei = (mu - self.f_best) * norm.cdf(z_score) + sigma * norm.pdf(z_score)
        return max(ei, 0.0)

    def _compute_ei_along_line(self, eta: float, direction: Tensor) -> float:
        """Compute EI for maximization along the line."""
        x = self.x_best + eta * direction
        x = torch.clamp(x, self.bounds[0], self.bounds[1])

        self.model.eval()
        with torch.no_grad():
            x_tensor = x.unsqueeze(0)
            posterior = self.model(x_tensor)
            mu = posterior.mean.item()
            sigma = posterior.variance.sqrt().item()

        if sigma <= 1e-10:
            return 0.0

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

        # Also update base class attributes for proper standardization
        self.train_X = self.X
        self.train_y = self.y

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

