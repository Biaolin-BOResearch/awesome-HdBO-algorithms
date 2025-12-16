"""
Sparse Axis-Aligned Subspace Bayesian Optimization (SAASBO).

This module implements SAASBO, which uses fully Bayesian inference with
hierarchical sparsity priors to handle high-dimensional optimization problems.
SAASBO is particularly effective for problems with limited evaluation budgets
and high-dimensional search spaces.

Reference:
    Eriksson and Jankowiak. "High-Dimensional Bayesian Optimization with
    Sparse Axis-Aligned Subspaces." UAI 2021.
    GitHub: https://github.com/pytorch/botorch
"""

from typing import Optional
import torch
from botorch import fit_fully_bayesian_model_nuts
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.transforms import Standardize
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.module import Module
from gpytorch.means.mean import Mean

from .base import BaseOptimizer
from .utils import DEVICE, DTYPE, ensure_tensor


class SAASBO(BaseOptimizer):
    """
    Sparse Axis-Aligned Subspace Bayesian Optimization.

    SAASBO uses fully Bayesian inference with hierarchical sparsity priors
    on inverse lengthscales to enable efficient high-dimensional optimization.
    It uses Hamiltonian Monte Carlo (NUTS) for inference rather than MAP estimation.

    Key features:
    - Hierarchical sparsity priors prevent overfitting in high dimensions
    - Fully Bayesian treatment captures model uncertainty
    - Uses qLogEI acquisition function for numerical stability
    - Designed for limited evaluation budgets (typically < 1000 evaluations)
    """

    def __init__(
        self,
        input_dim: int,
        bounds: torch.Tensor,
        warmup_steps: int = 256,
        num_samples: int = 128,
        thinning: int = 16,
        disable_progbar: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize SAASBO optimizer.

        Args:
            input_dim: Dimension of the input space
            bounds: Bounds tensor of shape (2, input_dim) with [lower_bounds, upper_bounds]
            warmup_steps: Number of warmup steps for NUTS sampler (default: 256)
            num_samples: Number of MCMC samples to draw (default: 128)
            thinning: Thinning interval for MCMC samples (default: 16)
            disable_progbar: Whether to disable progress bar during NUTS sampling
            device: Device for computation (CPU or CUDA)
            dtype: Data type for tensors

        Note:
            The original paper recommends at least 512 warmup steps and 256 samples
            for production use. Default values are reduced for faster experimentation.

        Example:
            >>> bounds = torch.tensor([[0.0] * 50, [1.0] * 50])  # 50D problem
            >>> optimizer = SAASBO(input_dim=50, bounds=bounds)
        """
        super().__init__(
            input_dim=input_dim,
            bounds=bounds,
            acq_func="LogEI",  # SAASBO uses LogEI for numerical stability
            covar_module=None,  # Not used - SAASBO uses its own model
            mean_module=None,   # Not used - SAASBO uses its own model
            device=device,
            dtype=dtype,
        )

        # NUTS sampling parameters
        self.warmup_steps = warmup_steps
        self.num_samples = num_samples
        self.thinning = thinning
        self.disable_progbar = disable_progbar

        # SAASBO uses a fully Bayesian model instead of standard GP
        self.saas_model: Optional[SaasFullyBayesianSingleTaskGP] = None

    def suggest(self, n_suggestions: int = 1, num_restarts: int = 10, raw_samples: int = 256) -> torch.Tensor:
        """
        Suggest next points to evaluate using qLogEI acquisition function.

        Args:
            n_suggestions: Number of points to suggest (batch size)
            num_restarts: Number of restarts for acquisition function optimization
            raw_samples: Number of raw samples for initialization

        Returns:
            Tensor of shape (n_suggestions, input_dim) with suggested points

        Raises:
            ValueError: If model has not been initialized (no observations yet)

        Note:
            Output shape is always (n_suggestions, input_dim), even if n_suggestions=1
            SAASBO uses more restarts by default than standard BO for better optimization.

        Example:
            >>> X_next = optimizer.suggest(n_suggestions=1)
            >>> assert X_next.shape == (1, input_dim)
        """
        if self.saas_model is None:
            raise ValueError(
                "Model not initialized. Please call observe() with initial data first."
            )

        # Get qLogEI acquisition function
        acq_func = self._get_acquisition_function()

        # Optimize acquisition function
        candidates, _ = optimize_acqf(
            acq_func,
            bounds=self.bounds,
            q=n_suggestions,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )

        assert candidates.shape == (n_suggestions, self.input_dim), \
            f"Expected shape ({n_suggestions}, {self.input_dim}), got {candidates.shape}"
        return candidates

    def observe(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """
        Update the optimizer with new observations.

        This method fits the fully Bayesian SAAS model using NUTS sampling,
        which can be computationally expensive but provides better uncertainty
        quantification than MAP estimation.

        Args:
            X: Observed input points of shape (n, input_dim)
            y: Observed output values of shape (n,) or (n, 1)

        Note:
            The NUTS fitting procedure scales cubically with the number of datapoints,
            so it's recommended to use SAASBO with limited evaluation budgets.

        Example:
            >>> X = torch.tensor([[0.5] * 50])  # 50D point
            >>> y = torch.tensor([0.6])
            >>> optimizer.observe(X, y)
        """
        X = ensure_tensor(X, device=self.device, dtype=self.dtype)
        y = ensure_tensor(y, device=self.device, dtype=self.dtype)

        # Ensure correct shapes
        X = X.reshape(-1, self.input_dim)
        y = y.reshape(-1, 1)

        # Update training data
        self.train_X = torch.cat([self.train_X, X], dim=0)
        self.train_y = torch.cat([self.train_y, y], dim=0)

        # Fit SAAS model with NUTS
        self._fit_saas_model()

    def _fit_saas_model(self) -> None:
        """
        Fit the fully Bayesian SAAS GP model using NUTS sampling.

        This method creates a SaasFullyBayesianSingleTaskGP model and fits it
        using Hamiltonian Monte Carlo (NUTS) to draw samples from the posterior
        over hyperparameters.
        """
        # Create SAAS model with outcome standardization
        self.saas_model = SaasFullyBayesianSingleTaskGP(
            train_X=self.train_X,
            train_Y=self.train_y,
            train_Yvar=None,  # Assume unknown/homoscedastic noise
            outcome_transform=Standardize(m=1),  # Standardize outputs
        )

        # Fit using NUTS sampler
        fit_fully_bayesian_model_nuts(
            self.saas_model,
            warmup_steps=self.warmup_steps,
            num_samples=self.num_samples,
            thinning=self.thinning,
            disable_progbar=self.disable_progbar,
        )

    def _get_acquisition_function(self):
        """
        Get the qLogEI acquisition function.

        SAASBO uses qLogExpectedImprovement instead of standard EI
        for better numerical stability, especially in high dimensions.

        Returns:
            qLogExpectedImprovement acquisition function instance
        """
        if self.saas_model is None:
            raise ValueError("Model not initialized. Call observe() first.")

        best_f = self.train_y.max()

        return qLogExpectedImprovement(
            model=self.saas_model,
            best_f=best_f,
        )

    def predict(self, X: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Generate posterior samples at given points.

        For SAASBO, this integrates over the posterior samples from NUTS,
        providing better uncertainty quantification than MAP-based approaches.

        Args:
            X: Points to predict at, shape (n, input_dim)
            num_samples: Number of samples to draw from the posterior

        Returns:
            Samples of shape (num_samples, n)

        Raises:
            ValueError: If model has not been initialized

        Example:
            >>> X_test = torch.tensor([[0.3] * 50])  # 50D test point
            >>> samples = optimizer.predict(X_test, num_samples=100)
        """
        if self.saas_model is None:
            raise ValueError("Model not initialized. Call observe() first.")

        X = ensure_tensor(X, device=self.device, dtype=self.dtype)

        # Get posterior distribution
        posterior = self.saas_model.posterior(X)

        # Draw samples
        samples = posterior.rsample(torch.Size((num_samples,)))

        return samples.squeeze(-1)  # Shape: (num_samples, n)

    def reset(self) -> None:
        """Reset the optimizer to its initial state."""
        super().reset()
        self.saas_model = None
