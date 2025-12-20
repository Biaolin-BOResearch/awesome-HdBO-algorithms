"""
Random Line Bayesian Optimization (RandomLineBO).

This method performs Bayesian optimization along random lines in the
high-dimensional space, effectively reducing the optimization to a
sequence of 1D problems along random directions.

Reference:
    Related to random subspace methods for high-dimensional optimization.
    The method samples random directions and optimizes along these lines.
"""

import torch
from typing import Optional, Tuple
from torch import Tensor

from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

from .base import BaseOptimizer


class RandomLineBO(BaseOptimizer):
    """
    Random Line Bayesian Optimization.

    This method optimizes the objective function along random lines in the
    high-dimensional space. At each iteration:
    1. Sample a random starting point in the space
    2. Sample a random direction (normalized)
    3. Perform 1D Bayesian optimization along this line
    4. The next suggestion is the best point found along the line

    This approach is effective when:
    - The function has structure along certain directions
    - Full high-dimensional optimization is too expensive
    - Simple exploration strategies are needed

    Args:
        input_dim: Dimensionality of the input space
        bounds: Tensor of shape (2, input_dim) with lower and upper bounds
        n_lines_per_iter: Number of random lines to sample per iteration
        line_samples: Number of samples to take along each line
        acq_func: Acquisition function ('EI' or 'UCB')
        use_anchor: If True, use current best point as anchor for lines
        device: Device to use for computations
        dtype: Data type for tensors
    """

    def __init__(
        self,
        input_dim: int,
        bounds: Tensor,
        n_lines_per_iter: int = 5,
        line_samples: int = 100,
        acq_func: str = 'EI',
        use_anchor: bool = True,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__(input_dim, bounds, device, dtype)

        self.n_lines_per_iter = n_lines_per_iter
        self.line_samples = line_samples
        self.acq_func_name = acq_func
        self.use_anchor = use_anchor

        self.model: Optional[SingleTaskGP] = None

    def _sample_random_line(
        self,
        anchor: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Sample a random line in the space.

        Args:
            anchor: Starting point for the line. If None, sampled randomly.

        Returns:
            Tuple of (anchor_point, direction_vector)
        """
        # Sample anchor point
        if anchor is None:
            anchor = torch.rand(
                self.input_dim,
                device=self.device,
                dtype=self.dtype
            ) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]

        # Sample random direction
        direction = torch.randn(
            self.input_dim,
            device=self.device,
            dtype=self.dtype
        )
        direction = direction / direction.norm()  # Normalize

        return anchor, direction

    def _sample_points_on_line(
        self,
        anchor: Tensor,
        direction: Tensor,
        n_samples: int
    ) -> Tensor:
        """
        Sample points along a line within the bounds.

        Args:
            anchor: Starting point (shape: input_dim)
            direction: Direction vector (shape: input_dim)
            n_samples: Number of samples

        Returns:
            Tensor of shape (n_samples, input_dim)
        """
        # Find valid range of t such that anchor + t * direction stays in bounds
        # For each dimension i: bounds[0,i] <= anchor[i] + t * direction[i] <= bounds[1,i]

        # Calculate maximum positive and negative t for each dimension
        t_max_pos = torch.full((self.input_dim,), float('inf'), device=self.device, dtype=self.dtype)
        t_max_neg = torch.full((self.input_dim,), float('-inf'), device=self.device, dtype=self.dtype)

        for i in range(self.input_dim):
            if direction[i] > 1e-8:
                t_max_pos[i] = (self.bounds[1, i] - anchor[i]) / direction[i]
                t_max_neg[i] = (self.bounds[0, i] - anchor[i]) / direction[i]
            elif direction[i] < -1e-8:
                t_max_pos[i] = (self.bounds[0, i] - anchor[i]) / direction[i]
                t_max_neg[i] = (self.bounds[1, i] - anchor[i]) / direction[i]

        t_min = t_max_neg.max().item()
        t_max = t_max_pos.min().item()

        # Sample t values uniformly in valid range
        t_values = torch.linspace(
            t_min, t_max, n_samples,
            device=self.device,
            dtype=self.dtype
        )

        # Generate points: anchor + t * direction
        points = anchor.unsqueeze(0) + t_values.unsqueeze(1) * direction.unsqueeze(0)

        # Clip to bounds for numerical safety
        points = torch.max(points, self.bounds[0].unsqueeze(0))
        points = torch.min(points, self.bounds[1].unsqueeze(0))

        return points

    def _fit_model(self):
        """Fit the GP model to observed data."""
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

        # Fit the model
        self._fit_model()

        # Determine anchor point
        if self.use_anchor and len(self.y) > 0:
            best_idx = self.y.argmax()
            anchor = self.X[best_idx]
        else:
            anchor = None

        # Collect candidates from multiple random lines
        all_candidates = []

        for _ in range(self.n_lines_per_iter):
            # Sample random line
            line_anchor, direction = self._sample_random_line(anchor)

            # Sample points along the line
            line_points = self._sample_points_on_line(
                line_anchor, direction, self.line_samples
            )

            # Evaluate acquisition function on line points
            if self.acq_func_name == 'EI':
                acq_func = ExpectedImprovement(
                    model=self.model,
                    best_f=self.y.max().item()
                )
            elif self.acq_func_name == 'UCB':
                beta = 2.0 * torch.log(torch.tensor(
                    self.input_dim * len(self.X) ** 2 * torch.pi ** 2 / 6.0
                ))
                acq_func = UpperConfidenceBound(
                    model=self.model,
                    beta=beta.item()
                )
            else:
                raise ValueError(f"Unknown acquisition function: {self.acq_func_name}")

            # Evaluate acquisition function
            with torch.no_grad():
                acq_values = acq_func(line_points.unsqueeze(1))  # Add q-batch dim

            # Find best point on this line
            best_idx = acq_values.argmax()
            all_candidates.append(line_points[best_idx])

        # Stack all candidates
        candidates = torch.stack(all_candidates)

        # If we need more suggestions than lines, sample additional random points
        if n_suggestions > len(candidates):
            additional = optimize_acqf(
                acq_function=acq_func,
                bounds=self.bounds,
                q=n_suggestions - len(candidates),
                num_restarts=5,
                raw_samples=256,
            )[0]
            candidates = torch.cat([candidates, additional], dim=0)

        # Return top n_suggestions by acquisition value
        with torch.no_grad():
            acq_values = acq_func(candidates.unsqueeze(1))
        top_indices = acq_values.topk(min(n_suggestions, len(candidates))).indices
        return candidates[top_indices.squeeze()]

    def observe(self, X: Tensor, y: Tensor):
        """
        Update the optimizer with new observations.

        Args:
            X: Tensor of shape (n, input_dim) with input points
            y: Tensor of shape (n, 1) or (n,) with observed values
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
        """
        Get the best observed point so far.

        Returns:
            Tuple of (best_X, best_y)
        """
        if self.y is None or len(self.y) == 0:
            raise ValueError("No observations yet.")

        best_idx = self.y.argmax()
        return self.X[best_idx], self.y[best_idx]
