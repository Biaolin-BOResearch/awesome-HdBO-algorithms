"""
Trust Region Bayesian Optimization (TuRBO) - Aligned with Reference Implementation.

This implementation closely follows the original TuRBO code from Eriksson et al.
while adapting to the suggest/observe API pattern.

Reference:
    Eriksson et al. "Scalable Global Optimization via Local Bayesian Optimization"
    NeurIPS 2019.
    GitHub: https://github.com/uber-research/TuRBO
"""

import math
from copy import deepcopy
from typing import Optional

import torch
import gpytorch
import numpy as np
from torch.quasirandom import SobolEngine

from .base import BaseOptimizer
from .utils import DEVICE, DTYPE, ensure_tensor


class TuRBO(BaseOptimizer):
    """
    Trust Region Bayesian Optimization (TuRBO-1).

    Uses a single trust region with Thompson sampling for efficient
    high-dimensional Bayesian optimization.
    """

    def __init__(
        self,
        input_dim: int,
        bounds: torch.Tensor,
        n_init: int = None,
        max_cholesky_size: int = 2000,
        n_training_steps: int = 50,
        use_ard: bool = True,
        n_candidates: int = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize TuRBO optimizer.

        Args:
            input_dim: Dimension of the input space
            bounds: Bounds tensor of shape (2, input_dim)
            n_init: Number of initial random points (default: 2*dim)
            max_cholesky_size: Maximum size for Cholesky decomposition
            n_training_steps: Number of training steps for GP hyperparameters
            use_ard: Whether to use ARD in the kernel
            n_candidates: Number of candidate points (default: min(100*dim, 5000))
            device: Device for computation
            dtype: Data type for tensors
        """
        super().__init__(
            input_dim=input_dim,
            bounds=bounds,
            acq_func="thompson",
            device=device,
            dtype=dtype,
        )

        # Settings
        self.n_init = n_init if n_init is not None else 2 * input_dim
        self.max_cholesky_size = max_cholesky_size
        self.n_training_steps = n_training_steps
        self.use_ard = use_ard
        self.n_candidates = n_candidates if n_candidates is not None else min(100 * input_dim, 5000)

        # Trust region parameters (from original TuRBO)
        self.length_min = 0.5 ** 7
        self.length_max = 1.6
        self.length_init = 0.8

        # Tolerances (from original TuRBO)
        self.fail_tol = max(4.0, float(input_dim))  # For batch_size=1
        self.succ_tol = 3

        # State variables
        self.length = self.length_init
        self.fail_count = 0
        self.succ_count = 0
        self.n_evals = 0
        self.gp_hypers = {}

        # Track whether we need initialization
        self.needs_initialization = True

    def suggest(self, n_suggestions: int = 1) -> torch.Tensor:
        """
        Suggest next points to evaluate.

        Args:
            n_suggestions: Number of points to suggest (will be returned one by one)

        Returns:
            Suggested points of shape (n_suggestions, input_dim)

        Note:
            Even though we return multiple suggestions, they are generated
            one at a time to maintain consistency.
        """
        suggestions = []

        for _ in range(n_suggestions):
            if self.needs_initialization or self.train_X.shape[0] < self.n_init:
                # Generate random initialization point
                X_next = self._generate_random_point()
            else:
                # Generate point using Thompson sampling in trust region
                X_next = self._generate_thompson_sample()

            suggestions.append(X_next)

        # Stack to ensure (n_suggestions, input_dim) shape
        result = torch.stack(suggestions)
        assert result.shape == (n_suggestions, self.input_dim), \
            f"Expected shape ({n_suggestions}, {self.input_dim}), got {result.shape}"
        return result

    def observe(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """
        Update the optimizer with new observations.

        Args:
            X: Observed input points
            y: Observed output values (to be minimized)
        """
        X = ensure_tensor(X, device=self.device, dtype=self.dtype)
        y = ensure_tensor(y, device=self.device, dtype=self.dtype)

        X = X.reshape(-1, self.input_dim)
        y = y.reshape(-1, 1)

        # Add to training data
        self.train_X = torch.cat([self.train_X, X], dim=0)
        self.train_y = torch.cat([self.train_y, y], dim=0)
        self.n_evals += X.shape[0]

        # Check if we finished initialization
        if self.needs_initialization and self.train_X.shape[0] >= self.n_init:
            self.needs_initialization = False
            if self.train_X.shape[0] > self.n_init:
                # Already have more than n_init, start adjusting TR
                self._adjust_trust_region(y)
        elif not self.needs_initialization:
            # Adjust trust region based on new observations
            self._adjust_trust_region(y)

        # Check for restart
        if self.length < self.length_min:
            self._restart_trust_region()

    def _generate_random_point(self) -> torch.Tensor:
        """
        Generate a random point within bounds (for initialization).

        Returns:
            Random point of shape (input_dim,)
        """
        X = torch.rand(self.input_dim, device=self.device, dtype=self.dtype)
        X = self.bounds[0] + (self.bounds[1] - self.bounds[0]) * X
        return X

    def _adjust_trust_region(self, y_new: torch.Tensor) -> None:
        """
        Adjust trust region length based on improvement.

        Following original TuRBO logic:
        - Success if y_new < y_best - 1e-3 * |y_best|
        - Expand after succ_tol successes
        - Shrink after fail_tol failures
        """
        # Get current best
        y_best = self.train_y[:-y_new.shape[0]].min() if self.train_y.shape[0] > y_new.shape[0] else float('inf')

        # Check for improvement
        improvement_threshold = 1e-3 * abs(y_best)
        if y_new.min() < y_best - improvement_threshold:
            # Success: improvement found
            self.succ_count += 1
            self.fail_count = 0
        else:
            # Failure: no improvement
            self.succ_count = 0
            self.fail_count += 1

        # Adjust trust region length
        if self.succ_count == self.succ_tol:
            # Expand trust region
            self.length = min(2.0 * self.length, self.length_max)
            self.succ_count = 0
        elif self.fail_count == self.fail_tol:
            # Shrink trust region
            self.length = self.length / 2.0
            self.fail_count = 0

    def _restart_trust_region(self) -> None:
        """Restart trust region when it becomes too small."""
        print(f"TuRBO: Restarting (n_evals={self.n_evals}, best={self.train_y.min().item():.4f})")

        # Reset TR state
        self.length = self.length_init
        self.fail_count = 0
        self.succ_count = 0
        self.gp_hypers = {}

        # Clear old data and request reinitialization
        self.train_X = torch.empty((0, self.input_dim), device=self.device, dtype=self.dtype)
        self.train_y = torch.empty((0, 1), device=self.device, dtype=self.dtype)
        self.needs_initialization = True

    def _generate_thompson_sample(self) -> torch.Tensor:
        """
        Generate a single Thompson sample using the GP posterior.

        Returns:
            Suggested point of shape (input_dim,)
        """
        # Normalize inputs to [0, 1]
        X_norm = self._normalize_to_unit_cube(self.train_X)

        # Standardize outputs
        y_mean = self.train_y.mean()
        y_std = self.train_y.std()
        if y_std < 1e-6:
            y_std = torch.tensor(1.0, device=self.device, dtype=self.dtype)
        y_norm = (self.train_y - y_mean) / y_std

        # Train GP
        gp = self._train_gp(X_norm, y_norm)

        # Create trust region bounds using ARD weights
        x_center = X_norm[y_norm.argmin().item(), :]

        # Get ARD lengthscales
        if self.use_ard:
            lengthscales = gp.covar_module.base_kernel.lengthscale.detach().cpu().numpy().ravel()
            weights = lengthscales / lengthscales.mean()
            weights = weights / np.prod(np.power(weights, 1.0 / len(weights)))
        else:
            weights = np.ones(self.input_dim)

        # Compute TR bounds
        lb = np.clip(x_center.cpu().numpy() - weights * self.length / 2.0, 0.0, 1.0)
        ub = np.clip(x_center.cpu().numpy() + weights * self.length / 2.0, 0.0, 1.0)

        # Generate candidates in trust region
        sobol = SobolEngine(self.input_dim, scramble=True)
        candidates_unit = sobol.draw(self.n_candidates).to(device=self.device, dtype=self.dtype)
        candidates = torch.from_numpy(lb).to(device=self.device, dtype=self.dtype) + \
                     (torch.from_numpy(ub - lb).to(device=self.device, dtype=self.dtype)) * candidates_unit

        # Apply perturbation mask (only perturb some dimensions)
        prob_perturb = min(20.0 / self.input_dim, 1.0)
        mask = torch.rand(self.n_candidates, self.input_dim, device=self.device) <= prob_perturb
        # Ensure at least one dimension is perturbed
        mask[mask.sum(dim=1) == 0, torch.randint(0, self.input_dim, (1,))] = True

        # Create candidate points
        x_center_expanded = x_center.unsqueeze(0).expand(self.n_candidates, -1)
        X_cand = x_center_expanded.clone()
        X_cand[mask] = candidates[mask]

        # Thompson sampling: sample from GP posterior
        with torch.no_grad(), gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            posterior = gp.likelihood(gp(X_cand))
            # Sample once for each candidate
            y_samples = posterior.sample()  # Shape: (n_candidates,)

        # Destandardize samples
        y_samples = y_mean + y_std * y_samples

        # Select best sample
        best_idx = y_samples.argmin()
        X_next_norm = X_cand[best_idx]

        # Denormalize back to original space
        X_next = self._denormalize_from_unit_cube(X_next_norm.unsqueeze(0)).squeeze(0)

        return X_next

    def _train_gp(self, X: torch.Tensor, y: torch.Tensor):
        """
        Train a GP model on the given data.

        Args:
            X: Normalized training inputs in [0,1]^d
            y: Standardized training outputs

        Returns:
            Trained GP model
        """
        # Define kernel
        if self.use_ard:
            kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(
                    nu=2.5,
                    ard_num_dims=self.input_dim,
                    lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 6.0),
                ),
                outputscale_prior=gpytorch.priors.GammaPrior(2.0, 0.15),
            )
        else:
            kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(
                    nu=2.5,
                    lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 6.0),
                ),
                outputscale_prior=gpytorch.priors.GammaPrior(2.0, 0.15),
            )

        # Create likelihood and model
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_prior=gpytorch.priors.GammaPrior(1.1, 0.05)
        )

        # Create GP model
        class ExactGPModel(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood, kernel):
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = kernel

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        gp = ExactGPModel(X, y.squeeze(-1), likelihood, kernel).to(device=self.device, dtype=self.dtype)

        # Train the GP
        gp.train()
        likelihood.train()

        optimizer = torch.optim.Adam(gp.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp)

        for _ in range(self.n_training_steps):
            optimizer.zero_grad()
            output = gp(X)
            loss = -mll(output, y.squeeze(-1))
            loss.backward()
            optimizer.step()

        # Switch to eval mode
        gp.eval()
        likelihood.eval()

        return gp

    def _normalize_to_unit_cube(self, X: torch.Tensor) -> torch.Tensor:
        """Normalize from original bounds to [0, 1]^d."""
        return (X - self.bounds[0]) / (self.bounds[1] - self.bounds[0])

    def _denormalize_from_unit_cube(self, X: torch.Tensor) -> torch.Tensor:
        """Denormalize from [0, 1]^d to original bounds."""
        return self.bounds[0] + (self.bounds[1] - self.bounds[0]) * X
