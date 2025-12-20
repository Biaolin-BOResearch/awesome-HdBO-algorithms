"""
Expected Subspace Selection Improvement Bayesian Optimization (ESSI-BO).

This method uses an adaptive subspace selection approach where multiple
random subspaces are sampled, ESSI is optimized in each subspace, and
the best subspace/solution is selected.

Algorithm 1: Adaptive Subspace Selection Approach
Require: Initial data D, subspace size s, candidates k
Ensure: Optimal solution (x*, f*)

1. Initialize (x*, f*) as best in D
2. while n < n_max:
   3. Modeling: Train GP surrogate using D
   4. Subspace Optimization:
      5. for i = 1 to k:
         6. Randomly select s-dim subspace S_i ⊆ {1, ..., d}
         7. x_i^(new), ESSI_i ← GA_Optimize(ESSI(·), S_i)
      8. end for
   9. Evaluation:
      10. Select i* = argmax_{1≤i≤k} ESSI_i
      11. Construct x^(new) by merging x* and x_{i*}^(new) on S_{i*}
      12. Evaluate f^(new) = f(x^(new))
      13. Update D and (x*, f*) if improved
14. end while

ESSI formula:
ESSI(y) = (f* - μ(z))Φ((f* - μ(z))/σ(z)) + σ(z)φ((f* - μ(z))/σ(z))

where z is formed by fixing non-subspace coordinates to x* values.
"""

import math
from typing import Optional, Tuple, List, Set
import torch
from torch import Tensor
import numpy as np
from scipy.stats import norm
from scipy.optimize import differential_evolution

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


class ESSIBO(BaseOptimizer):
    """
    Expected Subspace Selection Improvement Bayesian Optimization.

    This method optimizes in random subspaces and selects the best one
    based on ESSI (Expected Subspace Selection Improvement) values.

    Key features:
    1. Sample k random s-dimensional subspaces
    2. Optimize ESSI in each subspace using GA
    3. Select the subspace with highest ESSI value
    4. Merge the solution with current best to form new candidate

    Args:
        input_dim: Dimensionality of the input space (d)
        bounds: Tensor of shape (2, input_dim) with lower and upper bounds
        subspace_size: Size of each random subspace (s)
        n_candidates: Number of candidate subspaces to evaluate (k)
        ga_pop_size: Population size for genetic algorithm
        ga_max_iter: Maximum iterations for genetic algorithm
        device: Device to use for computations
        dtype: Data type for tensors
    """

    def __init__(
        self,
        input_dim: int,
        bounds: Tensor,
        subspace_size: Optional[int] = None,
        n_candidates: int = 5,
        ga_pop_size: int = 50,
        ga_max_iter: int = 100,
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

        # Subspace size: default to min(d, 5) as a reasonable choice
        self.subspace_size = subspace_size or min(input_dim, 5)
        self.n_candidates = n_candidates
        self.ga_pop_size = ga_pop_size
        self.ga_max_iter = ga_max_iter

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

    def _sample_random_subspace(self) -> List[int]:
        """
        Randomly select an s-dimensional subspace S ⊆ {0, ..., d-1}.

        Returns:
            List of coordinate indices in the subspace
        """
        indices = np.random.choice(
            self.input_dim,
            size=self.subspace_size,
            replace=False
        )
        return sorted(indices.tolist())

    def _compute_essi(self, subspace_values: np.ndarray, subspace_indices: List[int]) -> float:
        """
        Compute ESSI for a point in the subspace.

        ESSI(y) = (f* - μ(z))Φ((f* - μ(z))/σ(z)) + σ(z)φ((f* - μ(z))/σ(z))

        where z is formed by:
        - Taking x* as the base
        - Replacing coordinates in subspace_indices with subspace_values

        Args:
            subspace_values: Values for subspace coordinates
            subspace_indices: Indices of coordinates in the subspace

        Returns:
            ESSI value
        """
        # Construct full point z by merging x* and subspace values
        z = self.x_best.clone()
        for i, idx in enumerate(subspace_indices):
            z[idx] = subspace_values[i]

        # Get posterior prediction
        self.model.eval()
        with torch.no_grad():
            z_tensor = z.unsqueeze(0)
            posterior = self.model(z_tensor)
            mu = posterior.mean.item()
            sigma = posterior.variance.sqrt().item()

        # Compute ESSI (same as EI formula)
        essi = expected_improvement(mu, sigma, self.f_best)
        return essi

    def _optimize_essi_in_subspace(
        self,
        subspace_indices: List[int]
    ) -> Tuple[np.ndarray, float]:
        """
        Optimize ESSI in a given subspace using differential evolution (GA).

        x_i^(new), ESSI_i ← GA_Optimize(ESSI(·), S_i)

        Args:
            subspace_indices: Indices of coordinates in the subspace

        Returns:
            Tuple of (optimal_subspace_values, max_essi_value)
        """
        # Get bounds for subspace coordinates
        subspace_bounds = []
        for idx in subspace_indices:
            lb = self.bounds[0, idx].item()
            ub = self.bounds[1, idx].item()
            subspace_bounds.append((lb, ub))

        # Objective function (negative ESSI for minimization)
        def neg_essi(subspace_values):
            return -self._compute_essi(subspace_values, subspace_indices)

        # Use differential evolution as GA optimizer
        try:
            result = differential_evolution(
                neg_essi,
                bounds=subspace_bounds,
                maxiter=self.ga_max_iter,
                popsize=self.ga_pop_size // len(subspace_indices) + 1,
                tol=1e-6,
                seed=None,
                workers=1,
            )
            optimal_values = result.x
            max_essi = -result.fun
        except Exception:
            # Fallback: use current best values in subspace
            optimal_values = np.array([
                self.x_best[idx].item() for idx in subspace_indices
            ])
            max_essi = self._compute_essi(optimal_values, subspace_indices)

        return optimal_values, max_essi

    def _construct_new_point(
        self,
        subspace_values: np.ndarray,
        subspace_indices: List[int]
    ) -> Tensor:
        """
        Construct x^(new) by merging x* and x_{i*}^(new) on S_{i*}.

        Args:
            subspace_values: Optimal values for subspace coordinates
            subspace_indices: Indices of coordinates in the subspace

        Returns:
            New point tensor of shape (input_dim,)
        """
        new_x = self.x_best.clone()
        for i, idx in enumerate(subspace_indices):
            new_x[idx] = subspace_values[i]

        # Ensure within bounds
        new_x = torch.clamp(new_x, self.bounds[0], self.bounds[1])
        return new_x

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
        Suggest next point(s) to evaluate using ESSI-BO.

        Args:
            n_suggestions: Number of points to suggest

        Returns:
            Tensor of shape (n_suggestions, input_dim) with suggested points
        """
        if self.X is None or len(self.X) == 0:
            raise ValueError("No observations yet. Call observe() first.")

        suggestions = []

        for _ in range(n_suggestions):
            # Step 3: Train GP surrogate
            self._fit_model()

            # Step 4-8: Subspace Optimization
            candidate_results = []

            for _ in range(self.n_candidates):
                # Step 6: Randomly select s-dim subspace
                subspace_indices = self._sample_random_subspace()

                # Step 7: Optimize ESSI in subspace
                optimal_values, essi_value = self._optimize_essi_in_subspace(
                    subspace_indices
                )

                candidate_results.append({
                    'indices': subspace_indices,
                    'values': optimal_values,
                    'essi': essi_value,
                })

            # Step 10: Select i* = argmax ESSI_i
            best_candidate = max(candidate_results, key=lambda x: x['essi'])

            # Step 11: Construct x^(new)
            new_x = self._construct_new_point(
                best_candidate['values'],
                best_candidate['indices']
            )

            suggestions.append(new_x)

            # For batch suggestions, add pseudo-observation
            if n_suggestions > 1 and len(suggestions) < n_suggestions:
                self.model.eval()
                with torch.no_grad():
                    pred = self.model(new_x.unsqueeze(0))
                    pseudo_y = pred.mean

                self.X = torch.cat([self.X, new_x.unsqueeze(0)], dim=0)
                self.y = torch.cat([self.y, pseudo_y.unsqueeze(-1)], dim=0)

                # Update best if pseudo observation is better
                if pseudo_y.item() < self.f_best:
                    self.x_best = new_x.clone()
                    self.f_best = pseudo_y.item()

        return torch.stack(suggestions)

    def suggest_with_subspace_info(self) -> Tuple[Tensor, List[int], float]:
        """
        Suggest next point and return subspace information.

        Returns:
            Tuple of (new_point, selected_subspace_indices, essi_value)
        """
        if self.X is None or len(self.X) == 0:
            raise ValueError("No observations yet. Call observe() first.")

        # Train GP
        self._fit_model()

        # Evaluate all candidate subspaces
        candidate_results = []
        for _ in range(self.n_candidates):
            subspace_indices = self._sample_random_subspace()
            optimal_values, essi_value = self._optimize_essi_in_subspace(
                subspace_indices
            )
            candidate_results.append({
                'indices': subspace_indices,
                'values': optimal_values,
                'essi': essi_value,
            })

        # Select best
        best_candidate = max(candidate_results, key=lambda x: x['essi'])
        new_x = self._construct_new_point(
            best_candidate['values'],
            best_candidate['indices']
        )

        return new_x, best_candidate['indices'], best_candidate['essi']

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


class ESSIBOMaximize(ESSIBO):
    """
    ESSI-BO variant for maximization problems.
    """

    def _compute_essi(self, subspace_values: np.ndarray, subspace_indices: List[int]) -> float:
        """Compute ESSI for maximization."""
        z = self.x_best.clone()
        for i, idx in enumerate(subspace_indices):
            z[idx] = subspace_values[i]

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

