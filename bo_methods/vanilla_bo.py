"""
Vanilla Bayesian Optimization with dimensionality-dependent priors.

Reference:
    Hvarfner et al. (2024). "Vanilla Bayesian Optimization Performs Great in
    High Dimensions." International Conference on Machine Learning (ICML).

This implementation shows that simple Bayesian optimization with appropriate
dimensionality-dependent priors on lengthscales can perform competitively
against more elaborate high-dimensional methods.

Key insights from the paper:

1. Dimension-dependent lengthscale prior (Section 4):
   Use LogNormal priors on lengthscales with mean that scales as sqrt(d),
   i.e., loc = base_loc + 0.5 * log(d).

2. Calibrating Epistemic Uncertainty (Section 5.2):
   Fix σ_f^2 = 1 (outputscale) to match the scale of standardized observations.
   When data is repeatedly re-normalized, the signal variance would decrease
   over time if learned. The paper recommends fixing outputscale = 1.
   Formula: σ_f^2 = (1/n) * y^T * K^{-1} * y

Note: BoTorch's SingleTaskGP uses Standardize outcome transform by default,
which normalizes Y to mean=0, std=1. This is compatible with fixing σ_f^2 = 1.
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

    This method uses standard GP-based Bayesian optimization but with carefully
    chosen priors on the GP hyperparameters that scale appropriately with the
    dimensionality of the problem. The key insight is that proper priors can
    make vanilla BO competitive with complex high-dimensional methods.

    Key Features from the paper:
    - LogNormal priors on lengthscales with dimension-dependent mean:
      loc = base_loc + 0.5 * log(d)
    - RBF kernel with ARD (not Matern)
    - LogNormal prior on noise
    - Fixed outputscale = 1 (Section 5.2: Calibrating Epistemic Uncertainty)
      This prevents σ_f^2 from diminishing over time when data is re-normalized.

    Two modes available (matching reference implementation):
    - "default" mode (use_outputscale=False): RBFKernel only, outputscale fixed to 1
    - "with_ops" mode (use_outputscale=True): ScaleKernel with learned outputscale

    Args:
        input_dim: Dimensionality of the input space
        bounds: Tensor of shape (2, input_dim) with lower and upper bounds
        acq_func: Acquisition function to use ('EI', 'UCB', or 'qLogNEI')
        ls_loc: Base location parameter for lengthscale LogNormal prior (default: 1.4)
        ls_scale: Scale parameter for lengthscale LogNormal prior (default: sqrt(3))
        noise_loc: Location parameter for noise LogNormal prior (default: -4.0)
        noise_scale: Scale parameter for noise LogNormal prior (default: 1.0)
        use_outputscale: Whether to use ScaleKernel with learned outputscale.
            False (default): Matches paper's "default" config, fixes σ_f^2 = 1.
            True: Matches paper's "with_ops" config, learns outputscale.
        outputscale_alpha: Shape parameter for outputscale Gamma prior (default: 2.0)
        outputscale_beta: Rate parameter for outputscale Gamma prior (default: 0.15)
        ls_constraint: Lower bound constraint for lengthscale (default: 1e-4)
        noise_constraint: Lower bound constraint for noise (default: 1e-4)
        device: Device to use for computations
        dtype: Data type for tensors
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
        use_outputscale: bool = False,  # Paper recommends fixing σ_f^2 = 1 (Section 5.2)
        outputscale_alpha: float = 2.0,
        outputscale_beta: float = 0.15,
        ls_constraint: float = 1e-4,
        noise_constraint: float = 1e-4,
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
        self.use_outputscale = use_outputscale

        # Dimension-dependent lengthscale prior
        # Key insight from the paper: loc = base_loc + 0.5 * log(d)
        self.ls_loc = ls_loc + 0.5 * math.log(input_dim)
        self.ls_scale = ls_scale

        # Noise prior parameters
        self.noise_loc = noise_loc
        self.noise_scale = noise_scale

        # Outputscale prior parameters
        self.outputscale_alpha = outputscale_alpha
        self.outputscale_beta = outputscale_beta

        # Constraints
        self.ls_constraint = ls_constraint
        self.noise_constraint = noise_constraint

        # Model placeholders
        self.model: Optional[SingleTaskGP] = None
        self.mll: Optional[ExactMarginalLogLikelihood] = None

        # Training data (using different names to avoid conflict with base class)
        self.X: Optional[Tensor] = None
        self.y: Optional[Tensor] = None

    def _create_covar_module(self):
        """
        Create covariance module with dimension-dependent priors.

        Returns:
            Covariance module (RBFKernel or ScaleKernel(RBFKernel))
        """
        # Create lengthscale prior with dimension scaling
        lengthscale_prior = LogNormalPrior(
            loc=self.ls_loc,
            scale=self.ls_scale,
        )

        # Create RBF kernel with ARD
        base_kernel = RBFKernel(
            ard_num_dims=self.input_dim,
            lengthscale_prior=lengthscale_prior,
            lengthscale_constraint=GreaterThan(self.ls_constraint),
        )

        if self.use_outputscale:
            # "with_ops" mode: Wrap with ScaleKernel and outputscale prior
            covar_module = ScaleKernel(
                base_kernel=base_kernel,
                outputscale_prior=GammaPrior(
                    concentration=self.outputscale_alpha,
                    rate=self.outputscale_beta,
                ),
                outputscale_constraint=GreaterThan(self.ls_constraint),
            )
        else:
            # "default" mode: Fix σ_f^2 = 1 as recommended in Section 5.2
            # This calibrates epistemic uncertainty and prevents the signal
            # variance from diminishing over time during re-normalization.
            covar_module = base_kernel

        return covar_module

    def _create_likelihood(self):
        """
        Create likelihood with noise prior.

        Returns:
            GaussianLikelihood with LogNormal noise prior
        """
        noise_prior = LogNormalPrior(
            loc=self.noise_loc,
            scale=self.noise_scale,
        )

        likelihood = GaussianLikelihood(
            noise_prior=noise_prior,
            noise_constraint=GreaterThan(self.noise_constraint),
        )

        return likelihood

    def _create_model(self) -> SingleTaskGP:
        """
        Create a GP model with dimensionality-dependent priors.

        Returns:
            A SingleTaskGP model with appropriate priors
        """
        covar_module = self._create_covar_module()
        likelihood = self._create_likelihood()

        model = SingleTaskGP(
            train_X=self.X,
            train_Y=self.y,
            covar_module=covar_module,
            likelihood=likelihood,
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
        elif self.acq_func_name == "UCB":
            # Use dimension-dependent beta parameter for UCB
            beta = 2.0 * math.log(
                self.input_dim * len(self.X) ** 2 * math.pi ** 2 / 6.0
            )
            acq_func = UpperConfidenceBound(
                model=self.model,
                beta=beta,
            )
        elif self.acq_func_name == "qLogNEI":
            # qLogNoisyExpectedImprovement as used in the reference implementation
            acq_func = qLogNoisyExpectedImprovement(
                model=self.model,
                X_baseline=self.X,
            )
        else:
            raise ValueError(f"Unknown acquisition function: {self.acq_func_name}")

        # Optimize acquisition function
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=self.bounds,
            q=n_suggestions,
            num_restarts=10,
            raw_samples=512,
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
            Tuple of (best_X, best_y) where best_X has shape (input_dim,)
            and best_y has shape (1,)
        """
        if self.y is None or len(self.y) == 0:
            raise ValueError("No observations yet.")

        best_idx = self.y.argmax()
        return self.X[best_idx], self.y[best_idx]

    def reset(self) -> None:
        """Reset the optimizer to its initial state."""
        self.X = None
        self.y = None
        self.model = None
        self.mll = None


def get_vanilla_bo_prior_params(
    input_dim: int,
    base_ls_loc: float = 1.4,
    base_ls_scale: float = 1.73205,
) -> Dict[str, float]:
    """
    Compute the dimension-dependent prior parameters for VanillaBO.

    This is a utility function to show how the priors scale with dimension.

    Args:
        input_dim: Dimensionality of the problem
        base_ls_loc: Base location for lengthscale prior
        base_ls_scale: Base scale for lengthscale prior

    Returns:
        Dictionary with scaled prior parameters
    """
    return {
        "ls_loc": base_ls_loc + 0.5 * math.log(input_dim),
        "ls_scale": base_ls_scale,
        "ls_prior_mean": math.exp(base_ls_loc + 0.5 * math.log(input_dim) + base_ls_scale ** 2 / 2),
        "ls_prior_median": math.exp(base_ls_loc + 0.5 * math.log(input_dim)),
    }
