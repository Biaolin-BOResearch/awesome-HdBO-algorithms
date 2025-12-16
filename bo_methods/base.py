"""
Base classes and interfaces for Bayesian Optimization methods.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Callable
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import (
    ExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
)
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.module import Module
from gpytorch.means.mean import Mean
from gpytorch.distributions.multivariate_normal import MultivariateNormal

from .utils import DEVICE, DTYPE, ensure_tensor, clamp_to_bounds


class BaseOptimizer(ABC):
    """
    Abstract base class for Bayesian Optimization methods.

    This class defines the common interface that all BO methods must implement.
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
        Initialize the base optimizer.

        Args:
            input_dim: Dimension of the input space
            bounds: Bounds tensor of shape (2, input_dim)
            acq_func: Acquisition function type ('EI', 'PI', 'UCB')
            covar_module: Optional custom covariance module
            mean_module: Optional custom mean module
            device: Device for computation
            dtype: Data type for tensors
        """
        self.input_dim = input_dim
        self.device = device or DEVICE
        self.dtype = dtype or DTYPE

        self.bounds = bounds.to(device=self.device, dtype=self.dtype)
        self.acq_func = acq_func
        self.covar_module = covar_module
        self.mean_module = mean_module

        # Training data
        self.train_X = torch.empty((0, input_dim), device=self.device, dtype=self.dtype)
        self.train_y = torch.empty((0, 1), device=self.device, dtype=self.dtype)

        # Model
        self.model: Optional[SingleTaskGP] = None

    @abstractmethod
    def suggest(self, n_suggestions: int = 1) -> torch.Tensor:
        """
        Suggest next points to evaluate.

        Args:
            n_suggestions: Number of points to suggest

        Returns:
            Tensor of shape (n_suggestions, input_dim) with suggested points
        """
        pass

    @abstractmethod
    def observe(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """
        Update the optimizer with new observations.

        Args:
            X: Observed input points of shape (n, input_dim)
            y: Observed output values of shape (n,) or (n, 1)
        """
        pass

    def _update_model(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """
        Update the GP model with new data.

        Args:
            X: Training inputs of shape (n, input_dim)
            y: Training outputs of shape (n, 1)
        """
        # Ensure correct shapes
        X = X.reshape(-1, self.input_dim)
        y = y.reshape(-1, 1)

        # Update training data
        self.train_X = torch.cat([self.train_X, X], dim=0)
        self.train_y = torch.cat([self.train_y, y], dim=0)

        # Create and fit GP model
        self.model = SingleTaskGP(
            self.train_X,
            self.train_y,
            covar_module=self.covar_module,
            mean_module=self.mean_module,
        ).to(self.device)

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_model(mll)

    def _get_acquisition_function(self):
        """
        Get the acquisition function based on the specified type.

        Returns:
            Acquisition function instance
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call observe() first.")

        best_f = self.train_y.max()

        if self.acq_func == "EI":
            return ExpectedImprovement(self.model, best_f=best_f)
        elif self.acq_func == "PI":
            return ProbabilityOfImprovement(self.model, best_f=best_f)
        elif self.acq_func == "UCB":
            return UpperConfidenceBound(self.model, beta=0.25)
        else:
            raise ValueError(f"Acquisition function '{self.acq_func}' not supported")

    def _optimize_acquisition(
        self, q: int = 1, num_restarts: int = 5, raw_samples: int = 32
    ) -> torch.Tensor:
        """
        Optimize the acquisition function.

        Args:
            q: Batch size
            num_restarts: Number of restarts for optimization
            raw_samples: Number of raw samples for initialization

        Returns:
            Optimal points of shape (q, input_dim)
        """
        acq_func = self._get_acquisition_function()

        candidates, _ = optimize_acqf(
            acq_func,
            bounds=self.bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )

        return candidates

    def get_best_point(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the best observed point and its value.

        Returns:
            Tuple of (best_x, best_y)
        """
        best_idx = self.train_y.argmax()
        return self.train_X[best_idx], self.train_y[best_idx]

    def reset(self) -> None:
        """Reset the optimizer to its initial state."""
        self.train_X = torch.empty((0, self.input_dim), device=self.device, dtype=self.dtype)
        self.train_y = torch.empty((0, 1), device=self.device, dtype=self.dtype)
        self.model = None


class EmbeddingOptimizer(BaseOptimizer):
    """
    Base class for dimensionality reduction-based Bayesian Optimization methods.

    These methods operate in a lower-dimensional embedding space.
    """

    def __init__(
        self,
        input_dim: int,
        low_dim: int,
        bounds: torch.Tensor,
        acq_func: str = "EI",
        covar_module: Optional[Module] = None,
        mean_module: Optional[Mean] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize the embedding optimizer.

        Args:
            input_dim: Dimension of the original high-dimensional space
            low_dim: Dimension of the low-dimensional embedding space
            bounds: Bounds tensor of shape (2, input_dim) for the high-dimensional space
            acq_func: Acquisition function type
            covar_module: Optional custom covariance module
            mean_module: Optional custom mean module
            device: Device for computation
            dtype: Data type for tensors
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

        self.low_dim = min(low_dim, input_dim)

        # Training data in low-dimensional space
        self.train_X_low = torch.empty((0, self.low_dim), device=self.device, dtype=self.dtype)

        # Bounds for low-dimensional space (typically [0, 1]^low_dim)
        self.low_dim_bounds = torch.stack([
            torch.zeros(self.low_dim, device=self.device, dtype=self.dtype),
            torch.ones(self.low_dim, device=self.device, dtype=self.dtype),
        ])

    @abstractmethod
    def high_to_low(self, X: torch.Tensor) -> torch.Tensor:
        """
        Project from high-dimensional space to low-dimensional space.

        Args:
            X: Points in high-dimensional space of shape (n, input_dim)

        Returns:
            Points in low-dimensional space of shape (n, low_dim)
        """
        pass

    @abstractmethod
    def low_to_high(self, X: torch.Tensor) -> torch.Tensor:
        """
        Project from low-dimensional space to high-dimensional space.

        Args:
            X: Points in low-dimensional space of shape (n, low_dim)

        Returns:
            Points in high-dimensional space of shape (n, input_dim)
        """
        pass

    def observe(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """
        Update the optimizer with new observations.

        Args:
            X: Observed input points in high-dimensional space
            y: Observed output values
        """
        X = ensure_tensor(X, device=self.device, dtype=self.dtype)
        y = ensure_tensor(y, device=self.device, dtype=self.dtype)

        # Project to low-dimensional space
        X_low = self.high_to_low(X)

        # Update model in low-dimensional space
        X_low = X_low.reshape(-1, self.low_dim)
        y = y.reshape(-1, 1)

        self.train_X_low = torch.cat([self.train_X_low, X_low], dim=0)
        self.train_y = torch.cat([self.train_y, y], dim=0)

        # Update GP model
        self.model = SingleTaskGP(
            self.train_X_low,
            self.train_y,
            covar_module=self.covar_module,
            mean_module=self.mean_module,
        ).to(self.device)

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_model(mll)

    def suggest(self, n_suggestions: int = 1) -> torch.Tensor:
        """
        Suggest next points to evaluate in high-dimensional space.

        Args:
            n_suggestions: Number of points to suggest

        Returns:
            Suggested points in high-dimensional space of shape (n_suggestions, input_dim)

        Note:
            Output shape is always (n_suggestions, input_dim), even if n_suggestions=1
        """
        # Optimize in low-dimensional space
        acq_func = self._get_acquisition_function()

        candidates_low, _ = optimize_acqf(
            acq_func,
            bounds=self.low_dim_bounds,
            q=n_suggestions,
            num_restarts=5,
            raw_samples=32,
        )

        # Project back to high-dimensional space
        candidates_high = self.low_to_high(candidates_low)

        # Ensure within bounds
        candidates_high = clamp_to_bounds(candidates_high, self.bounds)

        # Validate shape
        assert candidates_high.shape == (n_suggestions, self.input_dim), \
            f"Expected shape ({n_suggestions}, {self.input_dim}), got {candidates_high.shape}"

        return candidates_high

    def reset(self) -> None:
        """Reset the optimizer to its initial state."""
        super().reset()
        self.train_X_low = torch.empty((0, self.low_dim), device=self.device, dtype=self.dtype)
