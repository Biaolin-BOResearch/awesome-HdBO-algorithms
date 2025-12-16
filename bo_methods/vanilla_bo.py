"""
Vanilla Bayesian Optimization with dimensionality-dependent priors.

Reference:
    Hvarfner et al. (2024). "Vanilla Bayesian Optimization Performs Great in
    High Dimensions." International Conference on Machine Learning (ICML).

This implementation shows that simple Bayesian optimization with appropriate
dimensionality-dependent priors on lengthscales can perform competitively
against more elaborate high-dimensional methods.
"""

import torch
from typing import Optional, Tuple
from torch import Tensor

import gpytorch
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.priors import GammaPrior
from gpytorch.constraints import Interval

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

from .base import BaseOptimizer


class VanillaBO(BaseOptimizer):
    """
    Vanilla Bayesian Optimization with dimensionality-dependent priors.

    This method uses standard GP-based Bayesian optimization but with carefully
    chosen priors on the GP hyperparameters that scale appropriately with the
    dimensionality of the problem. The key insight is that proper priors can
    make vanilla BO competitive with complex high-dimensional methods.

    Key Features:
    - Dimensionality-dependent lengthscale priors
    - Standard Matérn 5/2 kernel with ARD
    - Expected Improvement or UCB acquisition function
    - No complex trust regions or embeddings

    Args:
        input_dim: Dimensionality of the input space
        bounds: Tensor of shape (2, input_dim) with lower and upper bounds
        acq_func: Acquisition function to use ('EI' or 'UCB')
        lengthscale_prior_alpha: Shape parameter for Gamma prior (default: 2.0)
        lengthscale_prior_beta: Rate parameter for Gamma prior (scaled by dimension)
        outputscale_prior_alpha: Shape parameter for output scale prior
        outputscale_prior_beta: Rate parameter for output scale prior
        noise_prior_alpha: Shape parameter for noise prior
        noise_prior_beta: Rate parameter for noise prior
        device: Device to use for computations
        dtype: Data type for tensors
    """

    def __init__(
        self,
        input_dim: int,
        bounds: Tensor,
        acq_func: str = 'EI',
        lengthscale_prior_alpha: float = 2.0,
        lengthscale_prior_beta: Optional[float] = None,
        outputscale_prior_alpha: float = 2.0,
        outputscale_prior_beta: float = 0.5,
        noise_prior_alpha: float = 1.1,
        noise_prior_beta: float = 0.05,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__(input_dim, bounds, device, dtype)

        self.acq_func_name = acq_func

        # Dimensionality-dependent lengthscale prior
        # The key insight: lengthscale prior should scale with sqrt(d)
        if lengthscale_prior_beta is None:
            lengthscale_prior_beta = input_dim ** 0.5

        self.lengthscale_prior = GammaPrior(
            concentration=lengthscale_prior_alpha,
            rate=lengthscale_prior_beta
        )

        self.outputscale_prior = GammaPrior(
            concentration=outputscale_prior_alpha,
            rate=outputscale_prior_beta
        )

        self.noise_prior = GammaPrior(
            concentration=noise_prior_alpha,
            rate=noise_prior_beta
        )

        self.model: Optional[SingleTaskGP] = None
        self.mll: Optional[ExactMarginalLogLikelihood] = None

    def _create_model(self) -> SingleTaskGP:
        """
        Create a GP model with dimensionality-dependent priors.

        Returns:
            A SingleTaskGP model with appropriate priors
        """
        # Create Matérn 5/2 kernel with ARD and dimensionality-dependent priors
        covar_module = ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=self.input_dim,
                lengthscale_prior=self.lengthscale_prior,
                lengthscale_constraint=Interval(1e-4, 1e4)
            ),
            outputscale_prior=self.outputscale_prior,
            outputscale_constraint=Interval(1e-4, 1e4)
        )

        # Create the GP model
        model = SingleTaskGP(
            train_X=self.X,
            train_Y=self.y,
            covar_module=covar_module
        )

        # Set noise prior
        model.likelihood.noise_covar.noise_prior = self.noise_prior
        model.likelihood.noise_covar.register_constraint(
            "raw_noise",
            Interval(1e-6, 0.1)
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
        if self.acq_func_name == 'EI':
            acq_func = ExpectedImprovement(
                model=self.model,
                best_f=self.y.max().item()
            )
        elif self.acq_func_name == 'UCB':
            # Use dimension-dependent beta parameter for UCB
            beta = 2.0 * torch.log(torch.tensor(
                self.input_dim * len(self.X) ** 2 * torch.pi ** 2 / 6.0
            ))
            acq_func = UpperConfidenceBound(
                model=self.model,
                beta=beta.item()
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
            y: Tensor of shape (n, 1) with observed values
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
