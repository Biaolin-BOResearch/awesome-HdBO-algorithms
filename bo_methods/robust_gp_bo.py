"""
Robust GP for High-Dimensional Bayesian Optimization.

Reference:
    Xu et al. (2025). "Standard Gaussian Process is All You Need for
    High-Dimensional Bayesian Optimization." ICLR 2025.

This implementation addresses the gradient vanishing problem in GP training
for high-dimensional settings through:

1. Matérn kernel (more robust than SE/RBF kernel in high dimensions)
2. Robust lengthscale initialization: ℓ₀ = c * √d (Section 5)

Key insights from the paper:

Section 4 (Theoretical Analysis):
- SE kernel gradients vanish as dimension d grows
- The squared distance ρ² grows with d, causing exp(-ρ²) → 0
- Matérn kernel decays slower (ρ²/e^√(5ρ) vs ρ²/e^ρ²)

Section 5 (Robust Initialization):
- Initialize lengthscale as ℓ₀ = c√d where c > 0
- This prevents gradient vanishing by keeping ρ/ℓ from growing too fast
- Formula (8): ℓ₀ = c√d, c > 0
- Lemma 5.1: With ℓ₀ = c√d, probability of gradient vanishing decreases
  exponentially with dimension d
"""

import math
from typing import Optional, Tuple
import torch
from torch import Tensor

from gpytorch.kernels import ScaleKernel, MaternKernel, RBFKernel
from gpytorch.priors import GammaPrior, UniformPrior
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.likelihoods import GaussianLikelihood

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import (
    ExpectedImprovement,
    UpperConfidenceBound,
    LogExpectedImprovement,
)
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

from .base import BaseOptimizer
from .utils import DEVICE, DTYPE


class RobustGPBO(BaseOptimizer):
    """
    Robust GP-based Bayesian Optimization for High Dimensions.

    This method uses standard GP with two key modifications for high-dimensional
    robustness:
    1. Matérn kernel instead of SE/RBF (more robust gradient behavior)
    2. Dimension-dependent lengthscale initialization: ℓ₀ = c√d

    The paper shows that with these modifications, standard GP can match or
    exceed the performance of sophisticated high-dimensional BO methods.

    Args:
        input_dim: Dimensionality of the input space (D)
        bounds: Tensor of shape (2, input_dim) with lower and upper bounds
        acq_func: Acquisition function ('EI', 'UCB', 'LogEI')
        kernel_type: 'matern' (recommended) or 'rbf'
        nu: Smoothness parameter for Matérn kernel (0.5, 1.5, or 2.5)
        use_ard: Whether to use ARD (automatic relevance determination)
        lengthscale_init_factor: Factor c for ℓ₀ = c√d initialization (default: 1.0)
        use_sqrt_d_init: Whether to use √d initialization (True recommended)
        beta: UCB exploration parameter (default: 1.5)
        ls_prior_type: Lengthscale prior type ('uniform' or 'gamma')
        outputscale_prior: Tuple (alpha, beta) for Gamma prior on outputscale
        device: Device to use for computations
        dtype: Data type for tensors
    """

    def __init__(
        self,
        input_dim: int,
        bounds: Tensor,
        acq_func: str = "UCB",
        kernel_type: str = "matern",
        nu: float = 2.5,
        use_ard: bool = True,
        lengthscale_init_factor: float = 1.0,
        use_sqrt_d_init: bool = True,
        beta: float = 1.5,
        ls_prior_type: str = "uniform",
        outputscale_prior: Tuple[float, float] = (2.0, 0.15),
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
    ):
        # Initialize base class
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
        self.kernel_type = kernel_type.lower()
        self.nu = nu
        self.use_ard = use_ard
        self.lengthscale_init_factor = lengthscale_init_factor
        self.use_sqrt_d_init = use_sqrt_d_init
        self.beta = beta
        self.ls_prior_type = ls_prior_type.lower()
        self.outputscale_prior = outputscale_prior

        # Compute robust initialization value
        # Key formula from Section 5: ℓ₀ = c * √d
        self.initial_lengthscale = (
            lengthscale_init_factor * math.sqrt(input_dim)
            if use_sqrt_d_init
            else 0.6931  # Default botorch/gpytorch value
        )

        # Model placeholders
        self.model: Optional[SingleTaskGP] = None
        self.mll: Optional[ExactMarginalLogLikelihood] = None

        # Training data
        self.X: Optional[Tensor] = None
        self.y: Optional[Tensor] = None

    def _create_lengthscale_prior_and_constraint(self):
        """
        Create lengthscale prior and constraint based on configuration.

        Returns:
            Tuple of (prior, constraint)
        """
        if self.ls_prior_type == "uniform":
            # As in reference implementation: wider bounds for high-dim
            if self.input_dim >= 100:
                upper = 30.0
            else:
                upper = 10.0
            ls_prior = UniformPrior(1e-10, upper)
            ls_constraint = Interval(lower_bound=1e-10, upper_bound=upper)
        elif self.ls_prior_type == "gamma":
            ls_prior = GammaPrior(3.0, 6.0)
            ls_constraint = GreaterThan(1e-4)
        else:
            # No prior, just constraint
            ls_prior = None
            ls_constraint = GreaterThan(1e-4)

        return ls_prior, ls_constraint

    def _create_covar_module(self):
        """
        Create covariance module with robust initialization.

        Returns:
            Covariance module (ScaleKernel wrapping Matérn or RBF)
        """
        ls_prior, ls_constraint = self._create_lengthscale_prior_and_constraint()

        # Create base kernel
        if self.kernel_type == "matern":
            base_kernel = MaternKernel(
                nu=self.nu,
                ard_num_dims=self.input_dim if self.use_ard else None,
                lengthscale_prior=ls_prior,
                lengthscale_constraint=ls_constraint,
            )
        else:  # rbf
            base_kernel = RBFKernel(
                ard_num_dims=self.input_dim if self.use_ard else None,
                lengthscale_prior=ls_prior,
                lengthscale_constraint=ls_constraint,
            )

        # Wrap with ScaleKernel
        covar_module = ScaleKernel(
            base_kernel=base_kernel,
            outputscale_prior=GammaPrior(
                concentration=self.outputscale_prior[0],
                rate=self.outputscale_prior[1],
            ),
            outputscale_constraint=GreaterThan(1e-4),
        )

        # Apply robust initialization: ℓ₀ = c * √d (Section 5)
        if self.use_sqrt_d_init:
            ls = torch.ones_like(covar_module.base_kernel.lengthscale) * self.initial_lengthscale
            covar_module.base_kernel._set_lengthscale(ls)

        return covar_module

    def _create_model(self) -> SingleTaskGP:
        """
        Create a GP model with robust initialization.

        Returns:
            A SingleTaskGP model with Matérn kernel and √d initialization
        """
        covar_module = self._create_covar_module()

        model = SingleTaskGP(
            train_X=self.X,
            train_Y=self.y,
            covar_module=covar_module,
        )

        return model

    def _fit_model(self):
        """Fit the GP model to the observed data."""
        self.model = self._create_model().to(device=self.device, dtype=self.dtype)
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(self.mll)

    def suggest(self, n_suggestions: int = 1) -> Tensor:
        """
        Suggest next point(s) to evaluate.

        Args:
            n_suggestions: Number of points to suggest (default: 1)

        Returns:
            Tensor of shape (n_suggestions, input_dim) with suggested points
        """
        if self.X is None or len(self.X) == 0:
            raise ValueError("No observations yet. Call observe() first.")

        # Fit the model
        self._fit_model()

        # Create acquisition function
        if self.acq_func_name == "EI":
            acq_func = ExpectedImprovement(
                model=self.model,
                best_f=self.y.max().item(),
            )
        elif self.acq_func_name == "LogEI":
            acq_func = LogExpectedImprovement(
                model=self.model,
                best_f=self.y.max().item(),
            )
        elif self.acq_func_name == "UCB":
            acq_func = UpperConfidenceBound(
                model=self.model,
                beta=self.beta,
            )
        else:
            raise ValueError(f"Unknown acquisition function: {self.acq_func_name}")

        # Optimize acquisition function
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=self.bounds,
            q=n_suggestions,
            num_restarts=10,
            raw_samples=1000,
        )

        return candidates

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

    def get_lengthscales(self) -> Optional[Tensor]:
        """
        Get the current learned lengthscales.

        Returns:
            Lengthscale tensor or None if model not fitted
        """
        if self.model is None:
            return None
        return self.model.covar_module.base_kernel.lengthscale.detach()

    def reset(self) -> None:
        """Reset the optimizer to its initial state."""
        self.X = None
        self.y = None
        self.model = None
        self.mll = None


def compute_gradient_vanishing_threshold(
    input_dim: int,
    machine_epsilon: float = 2e-53,
) -> float:
    """
    Compute the threshold τ for gradient vanishing (from Lemma 4.2).

    For SE kernel: τ_SE = (1/2) + sqrt(1/4 - log(ξ))
    For Matérn kernel: τ_Matérn = (1 + sqrt(1 + log(1/5) - log(ξ))) / √5

    Args:
        input_dim: Dimensionality of the problem
        machine_epsilon: Machine epsilon (default: double precision)

    Returns:
        Dictionary with thresholds for SE and Matérn kernels
    """
    log_eps = math.log(machine_epsilon)

    # SE kernel threshold (Proposition 4.1)
    tau_se = 0.5 + math.sqrt(0.25 - log_eps)

    # Matérn kernel threshold (Proposition 4.3)
    sqrt_5 = math.sqrt(5)
    tau_matern = (1 + math.sqrt(1 + math.log(1/5) - log_eps)) / sqrt_5

    return {
        "tau_se": tau_se,
        "tau_matern": tau_matern,
        "ratio": tau_matern / tau_se,  # Matérn allows ~3x larger ρ
    }


def compute_recommended_init(input_dim: int, c: float = 1.0) -> float:
    """
    Compute the recommended lengthscale initialization (Lemma 5.1).

    ℓ₀ = c * √d

    Args:
        input_dim: Dimensionality of the problem
        c: Scaling factor (default: 1.0)

    Returns:
        Recommended initial lengthscale value
    """
    return c * math.sqrt(input_dim)

