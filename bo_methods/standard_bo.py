"""
Standard Bayesian Optimization with Gaussian Process regression.

This module implements the classic BO algorithm using a GP surrogate model
and common acquisition functions (EI, PI, UCB).
"""

from typing import Optional
import torch
from gpytorch.module import Module
from gpytorch.means.mean import Mean

from .base import BaseOptimizer
from .utils import DEVICE, DTYPE, ensure_tensor


class StandardBO(BaseOptimizer):
    """
    Standard Bayesian Optimization using Gaussian Processes.

    This is the vanilla BO algorithm that maintains a GP model over the
    entire input space and optimizes an acquisition function to select
    the next query point.
    """

    def __init__(
        self,
        input_dim: int,
        bounds: torch.Tensor,
        acq_func: str = "EI",
        covar_module: Optional[Module] = None,
        mean_module: Optional[Mean] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize Standard Bayesian Optimization.

        Args:
            input_dim: Dimension of the input space
            bounds: Bounds tensor of shape (2, input_dim) with [lower_bounds, upper_bounds]
            acq_func: Acquisition function type ('EI', 'PI', 'UCB')
            covar_module: Optional custom covariance module for the GP
            mean_module: Optional custom mean module for the GP
            device: Device for computation (CPU or CUDA)
            dtype: Data type for tensors

        Example:
            >>> bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
            >>> optimizer = StandardBO(input_dim=2, bounds=bounds, acq_func='EI')
        """
        super().__init__(
            input_dim=input_dim,
            bounds=bounds,
            acq_func=acq_func,
            covar_module=covar_module,
            mean_module=mean_module,
            device=device,
            dtype=dtype,
        )

    def suggest(self, n_suggestions: int = 1, num_restarts: int = 5) -> torch.Tensor:
        """
        Suggest next points to evaluate using acquisition function optimization.

        Args:
            n_suggestions: Number of points to suggest (batch size)
            num_restarts: Number of restarts for acquisition function optimization

        Returns:
            Tensor of shape (n_suggestions, input_dim) with suggested points

        Raises:
            ValueError: If model has not been initialized (no observations yet)

        Note:
            Output shape is always (n_suggestions, input_dim), even if n_suggestions=1

        Example:
            >>> X_next = optimizer.suggest(n_suggestions=3)
            >>> assert X_next.shape == (3, input_dim)
        """
        if self.model is None:
            raise ValueError(
                "Model not initialized. Please call observe() with initial data first."
            )

        candidates = self._optimize_acquisition(q=n_suggestions, num_restarts=num_restarts)
        assert candidates.shape == (n_suggestions, self.input_dim), \
            f"Expected shape ({n_suggestions}, {self.input_dim}), got {candidates.shape}"
        return candidates

    def observe(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """
        Update the optimizer with new observations.

        Args:
            X: Observed input points of shape (n, input_dim)
            y: Observed output values of shape (n,) or (n, 1)

        Example:
            >>> X = torch.tensor([[0.5, 0.3], [0.2, 0.8]])
            >>> y = torch.tensor([0.6, 0.4])
            >>> optimizer.observe(X, y)
        """
        X = ensure_tensor(X, device=self.device, dtype=self.dtype)
        y = ensure_tensor(y, device=self.device, dtype=self.dtype)

        self._update_model(X, y)

    def predict(self, X: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Generate posterior samples at given points.

        Args:
            X: Points to predict at, shape (n, input_dim)
            num_samples: Number of samples to draw from the posterior

        Returns:
            Samples of shape (num_samples, n)

        Raises:
            ValueError: If model has not been initialized

        Example:
            >>> X_test = torch.tensor([[0.3, 0.7]])
            >>> samples = optimizer.predict(X_test, num_samples=100)
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call observe() first.")

        X = ensure_tensor(X, device=self.device, dtype=self.dtype)
        return self.model(X).sample(torch.Size((num_samples,)))
