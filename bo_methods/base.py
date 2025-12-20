"""
Base classes and interfaces for Bayesian Optimization methods.

All data is internally normalized to unit cube [0,1]^d and standardized (mean=0, std=1)
for better GP fitting. Transforms are applied transparently.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Callable, Tuple
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
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
    
    Data Normalization:
        - Input X is normalized to [0,1]^d using the provided bounds
        - Output Y is standardized to mean=0, std=1
        - All internal operations use normalized data
        - Suggested points are denormalized before returning
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
            bounds: Bounds tensor of shape (2, input_dim) - original bounds
            acq_func: Acquisition function type ('EI', 'PI', 'UCB')
            covar_module: Optional custom covariance module
            mean_module: Optional custom mean module
            device: Device for computation
            dtype: Data type for tensors
        """
        self.input_dim = input_dim
        self.device = device or DEVICE
        self.dtype = dtype or DTYPE

        # Original bounds (for denormalization)
        self.bounds = bounds.to(device=self.device, dtype=self.dtype)
        
        # Unit cube bounds (for normalized space)
        self.unit_bounds = torch.stack([
            torch.zeros(input_dim, device=self.device, dtype=self.dtype),
            torch.ones(input_dim, device=self.device, dtype=self.dtype),
        ])
        
        self.acq_func = acq_func
        self.covar_module = covar_module
        self.mean_module = mean_module

        # Training data in ORIGINAL space (for user access)
        self.train_X = torch.empty((0, input_dim), device=self.device, dtype=self.dtype)
        self.train_y = torch.empty((0, 1), device=self.device, dtype=self.dtype)
        
        # Training data in NORMALIZED space (for GP)
        self._train_X_normalized = torch.empty((0, input_dim), device=self.device, dtype=self.dtype)
        self._train_y_standardized = torch.empty((0, 1), device=self.device, dtype=self.dtype)
        
        # Y statistics for standardization
        self._y_mean = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self._y_std = torch.tensor(1.0, device=self.device, dtype=self.dtype)

        # Model
        self.model: Optional[SingleTaskGP] = None

    def _normalize_X(self, X: torch.Tensor) -> torch.Tensor:
        """Normalize X from original bounds to [0, 1]^d."""
        lb = self.bounds[0]
        ub = self.bounds[1]
        return (X - lb) / (ub - lb + 1e-8)
    
    def _denormalize_X(self, X_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize X from [0, 1]^d to original bounds."""
        lb = self.bounds[0]
        ub = self.bounds[1]
        return lb + X_norm * (ub - lb)
    
    def _standardize_y(self, y: torch.Tensor) -> torch.Tensor:
        """Standardize y to mean=0, std=1."""
        return (y - self._y_mean) / (self._y_std + 1e-8)
    
    def _unstandardize_y(self, y_std: torch.Tensor) -> torch.Tensor:
        """Unstandardize y back to original scale."""
        return y_std * self._y_std + self._y_mean
    
    def _update_y_statistics(self):
        """Update Y mean and std for standardization."""
        if self.train_y.numel() > 0:
            self._y_mean = self.train_y.mean()
            self._y_std = self.train_y.std()
            if self._y_std < 1e-6:
                self._y_std = torch.tensor(1.0, device=self.device, dtype=self.dtype)

    @abstractmethod
    def suggest(self, n_suggestions: int = 1) -> torch.Tensor:
        """
        Suggest next points to evaluate.

        Args:
            n_suggestions: Number of points to suggest

        Returns:
            Tensor of shape (n_suggestions, input_dim) with suggested points
            in ORIGINAL space (not normalized)
        """
        pass

    @abstractmethod
    def observe(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """
        Update the optimizer with new observations.

        Args:
            X: Observed input points of shape (n, input_dim) in ORIGINAL space
            y: Observed output values of shape (n,) or (n, 1)
        """
        pass

    def _update_model(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """
        Update the GP model with new data.

        Args:
            X: Training inputs in ORIGINAL space of shape (n, input_dim)
            y: Training outputs of shape (n, 1)
        """
        # Ensure correct shapes
        X = X.reshape(-1, self.input_dim)
        y = y.reshape(-1, 1)

        # Update training data (original space)
        self.train_X = torch.cat([self.train_X, X], dim=0)
        self.train_y = torch.cat([self.train_y, y], dim=0)
        
        # Update Y statistics and normalize data
        self._update_y_statistics()
        self._train_X_normalized = self._normalize_X(self.train_X)
        self._train_y_standardized = self._standardize_y(self.train_y)

        # Create and fit GP model in normalized space
        # Disable default transforms since we handle normalization manually
        self.model = SingleTaskGP(
            self._train_X_normalized,
            self._train_y_standardized,
            covar_module=self.covar_module,
            mean_module=self.mean_module,
            input_transform=None,
            outcome_transform=None,
        ).to(self.device)

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)

    def _get_acquisition_function(self):
        """
        Get the acquisition function based on the specified type.

        Returns:
            Acquisition function instance
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call observe() first.")

        # Best value in standardized space
        best_f = self._train_y_standardized.max()

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
        Optimize the acquisition function in normalized space and return 
        denormalized candidates.

        Args:
            q: Batch size
            num_restarts: Number of restarts for optimization
            raw_samples: Number of raw samples for initialization

        Returns:
            Optimal points of shape (q, input_dim) in ORIGINAL space
        """
        acq_func = self._get_acquisition_function()

        # Optimize in unit cube [0,1]^d
        candidates_norm, _ = optimize_acqf(
            acq_func,
            bounds=self.unit_bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )
        
        # Denormalize to original space
        candidates = self._denormalize_X(candidates_norm)

        return candidates

    def get_best_point(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the best observed point and its value in ORIGINAL space.

        Returns:
            Tuple of (best_x, best_y) in original space
        """
        if self.train_y.numel() == 0:
            return None, None
        best_idx = self.train_y.argmax()
        return self.train_X[best_idx], self.train_y[best_idx]

    def reset(self) -> None:
        """Reset the optimizer to its initial state."""
        self.train_X = torch.empty((0, self.input_dim), device=self.device, dtype=self.dtype)
        self.train_y = torch.empty((0, 1), device=self.device, dtype=self.dtype)
        self._train_X_normalized = torch.empty((0, self.input_dim), device=self.device, dtype=self.dtype)
        self._train_y_standardized = torch.empty((0, 1), device=self.device, dtype=self.dtype)
        self._y_mean = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self._y_std = torch.tensor(1.0, device=self.device, dtype=self.dtype)
        self.model = None


class EmbeddingOptimizer(BaseOptimizer):
    """
    Base class for dimensionality reduction-based Bayesian Optimization methods.

    These methods operate in a lower-dimensional embedding space.
    All data is normalized before projection.
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

        # Training data in low-dimensional space (already normalized)
        self.train_X_low = torch.empty((0, self.low_dim), device=self.device, dtype=self.dtype)

        # Bounds for low-dimensional space (unit cube)
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
               Should be normalized to [0,1]^d

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
            Should be in [0,1]^d (normalized)
        """
        pass

    def observe(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """
        Update the optimizer with new observations.

        Args:
            X: Observed input points in ORIGINAL high-dimensional space
            y: Observed output values
        """
        X = ensure_tensor(X, device=self.device, dtype=self.dtype)
        y = ensure_tensor(y, device=self.device, dtype=self.dtype)
        
        # Store original data
        X = X.reshape(-1, self.input_dim)
        y = y.reshape(-1, 1)
        self.train_X = torch.cat([self.train_X, X], dim=0)
        self.train_y = torch.cat([self.train_y, y], dim=0)
        
        # Update Y statistics (using ALL y data)
        self._update_y_statistics()

        # Normalize X to [0,1]^d, then project to low-dim
        X_norm = self._normalize_X(X)
        X_low = self.high_to_low(X_norm)
        # Clamp to [0,1] to ensure low-dim data is in unit cube (avoid BoTorch warnings)
        X_low = torch.clamp(X_low, 0.0, 1.0)

        # Update low-dim data
        self.train_X_low = torch.cat([self.train_X_low, X_low], dim=0)
        
        # Re-standardize ALL Y data with updated mean/std (critical for correct standardization)
        self._train_y_standardized = self._standardize_y(self.train_y)

        # Update GP model in low-dimensional space
        # Disable default transforms since we handle normalization manually
        self.model = SingleTaskGP(
            self.train_X_low,
            self._train_y_standardized,
            covar_module=self.covar_module,
            mean_module=self.mean_module,
            input_transform=None,
            outcome_transform=None,
        ).to(self.device)

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)

    def suggest(self, n_suggestions: int = 1) -> torch.Tensor:
        """
        Suggest next points to evaluate in ORIGINAL high-dimensional space.

        Args:
            n_suggestions: Number of points to suggest

        Returns:
            Suggested points in ORIGINAL high-dimensional space of shape (n_suggestions, input_dim)
        """
        # Best value in standardized space
        best_f = self._train_y_standardized.max()
        
        # Create acquisition function
        if self.acq_func == "EI":
            acq_func = ExpectedImprovement(self.model, best_f=best_f)
        elif self.acq_func == "UCB":
            acq_func = UpperConfidenceBound(self.model, beta=0.25)
        else:
            acq_func = ExpectedImprovement(self.model, best_f=best_f)

        # Optimize in low-dimensional unit cube
        candidates_low, _ = optimize_acqf(
            acq_func,
            bounds=self.low_dim_bounds,
            q=n_suggestions,
            num_restarts=5,
            raw_samples=32,
        )

        # Project back to high-dimensional normalized space
        candidates_high_norm = self.low_to_high(candidates_low)
        
        # Clamp to [0,1] and denormalize to original bounds
        candidates_high_norm = torch.clamp(candidates_high_norm, 0.0, 1.0)
        candidates_high = self._denormalize_X(candidates_high_norm)

        # Ensure within original bounds
        candidates_high = clamp_to_bounds(candidates_high, self.bounds)

        # Validate shape
        assert candidates_high.shape == (n_suggestions, self.input_dim), \
            f"Expected shape ({n_suggestions}, {self.input_dim}), got {candidates_high.shape}"

        return candidates_high

    def reset(self) -> None:
        """Reset the optimizer to its initial state."""
        super().reset()
        self.train_X_low = torch.empty((0, self.low_dim), device=self.device, dtype=self.dtype)
