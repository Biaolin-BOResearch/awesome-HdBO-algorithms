"""
Vanilla Bayesian Optimization with dimensionality-dependent priors.

Reference:
    Hvarfner et al. (2024). "Vanilla Bayesian Optimization Performs Great in
    High Dimensions." International Conference on Machine Learning (ICML).

All data is normalized to unit cube [0,1]^d and standardized (mean=0, std=1)
for better GP fitting.
"""

import math
from typing import Optional, Tuple, Dict
import torch
from torch import Tensor

from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.priors import LogNormalPrior, GammaPrior
from gpytorch.constraints import GreaterThan
from gpytorch.likelihoods import GaussianLikelihood

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import (
    ExpectedImprovement,
    UpperConfidenceBound,
    qLogNoisyExpectedImprovement,
)
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

from .base import BaseOptimizer
from .utils import DEVICE, DTYPE


class VanillaBO(BaseOptimizer):
    """
    Vanilla Bayesian Optimization with dimensionality-dependent priors.
    
    Inherits normalization from BaseOptimizer:
    - X is normalized to [0,1]^d
    - Y is standardized to mean=0, std=1
    """

    def __init__(
        self,
        input_dim: int,
        bounds: Tensor,
        acq_func: str = "EI",
        ls_loc: float = 1.4,
        ls_scale: float = 1.73205,  # sqrt(3)
        noise_loc: float = -4.0,
        noise_scale: float = 1.0,
        use_outputscale: bool = False,
        outputscale_alpha: float = 2.0,
        outputscale_beta: float = 0.15,
        ls_constraint: float = 1e-4,
        noise_constraint: float = 1e-4,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
    ):
        _device = device or DEVICE
        _dtype = dtype or DTYPE
        super().__init__(
            input_dim=input_dim,
            bounds=bounds,
            acq_func=acq_func,
            device=_device,
            dtype=_dtype,
        )

        self.acq_func_name = acq_func
        self.use_outputscale = use_outputscale

        # Dimension-dependent lengthscale prior
        self.ls_loc = ls_loc + 0.5 * math.log(input_dim)
        self.ls_scale = ls_scale
        self.noise_loc = noise_loc
        self.noise_scale = noise_scale
        self.outputscale_alpha = outputscale_alpha
        self.outputscale_beta = outputscale_beta
        self.ls_constraint = ls_constraint
        self.noise_constraint = noise_constraint

        self.mll: Optional[ExactMarginalLogLikelihood] = None

    def _create_covar_module(self):
        """Create covariance module with dimension-dependent priors."""
        lengthscale_prior = LogNormalPrior(loc=self.ls_loc, scale=self.ls_scale)
        
        base_kernel = RBFKernel(
            ard_num_dims=self.input_dim,
            lengthscale_prior=lengthscale_prior,
            lengthscale_constraint=GreaterThan(self.ls_constraint),
        )

        if self.use_outputscale:
            covar_module = ScaleKernel(
                base_kernel=base_kernel,
                outputscale_prior=GammaPrior(
                    concentration=self.outputscale_alpha,
                    rate=self.outputscale_beta,
                ),
                outputscale_constraint=GreaterThan(self.ls_constraint),
            )
        else:
            covar_module = base_kernel

        return covar_module.to(device=self.device, dtype=self.dtype)

    def _create_likelihood(self):
        """Create likelihood with noise prior."""
        noise_prior = LogNormalPrior(loc=self.noise_loc, scale=self.noise_scale)
        likelihood = GaussianLikelihood(
            noise_prior=noise_prior,
            noise_constraint=GreaterThan(self.noise_constraint),
        )
        return likelihood.to(device=self.device, dtype=self.dtype)

    def _fit_model(self):
        """Fit the GP model to the observed data in normalized space."""
        covar_module = self._create_covar_module()
        likelihood = self._create_likelihood()

        # Use normalized data - disable BoTorch's default transforms
        # since we handle normalization/standardization manually
        self.model = SingleTaskGP(
            train_X=self._train_X_normalized,
            train_Y=self._train_y_standardized,
            covar_module=covar_module,
            likelihood=likelihood,
            input_transform=None,    # X already normalized to [0,1]^d
            outcome_transform=None,  # Y already standardized to mean=0, std=1
        ).to(device=self.device, dtype=self.dtype)

        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(self.mll)

    def observe(self, X: Tensor, y: Tensor):
        """Update with new observations in ORIGINAL space."""
        X = X.to(device=self.device, dtype=self.dtype)
        y = y.to(device=self.device, dtype=self.dtype)

        if y.dim() == 1:
            y = y.unsqueeze(-1)

        # Store original data
        if self.train_X.numel() == 0:
            self.train_X = X.reshape(-1, self.input_dim)
            self.train_y = y.reshape(-1, 1)
        else:
            self.train_X = torch.cat([self.train_X, X.reshape(-1, self.input_dim)], dim=0)
            self.train_y = torch.cat([self.train_y, y.reshape(-1, 1)], dim=0)

        # Update normalization statistics and normalized data
        self._update_y_statistics()
        self._train_X_normalized = self._normalize_X(self.train_X)
        self._train_y_standardized = self._standardize_y(self.train_y)

    def suggest(self, n_suggestions: int = 1) -> Tensor:
        """Suggest next point(s) in ORIGINAL space."""
        if self.train_X is None or self.train_X.numel() == 0:
            raise ValueError("No observations yet. Call observe() first.")

        # Fit the model
        self._fit_model()

        # Best value in standardized space
        best_f = self._train_y_standardized.max().item()

        # Create acquisition function
        if self.acq_func_name == "EI":
            acq_func = ExpectedImprovement(model=self.model, best_f=best_f)
        elif self.acq_func_name == "UCB":
            beta = 2.0 * math.log(self.input_dim * len(self.train_X) ** 2 * math.pi ** 2 / 6.0)
            acq_func = UpperConfidenceBound(model=self.model, beta=beta)
        elif self.acq_func_name == "qLogNEI":
            acq_func = qLogNoisyExpectedImprovement(
                model=self.model,
                X_baseline=self._train_X_normalized,
            )
        else:
            raise ValueError(f"Unknown acquisition function: {self.acq_func_name}")

        # Optimize in unit cube [0,1]^d
        candidates_norm, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=self.unit_bounds,
            q=n_suggestions,
            num_restarts=10,
            raw_samples=512,
        )

        # Denormalize to original space
        candidates = self._denormalize_X(candidates_norm)

        return candidates

    def get_best_point(self) -> Tuple[Tensor, Tensor]:
        """Get the best observed point in ORIGINAL space."""
        if self.train_y is None or self.train_y.numel() == 0:
            return None, None
        best_idx = self.train_y.argmax()
        return self.train_X[best_idx], self.train_y[best_idx]

    def reset(self) -> None:
        """Reset the optimizer."""
        super().reset()
        self.model = None
        self.mll = None
