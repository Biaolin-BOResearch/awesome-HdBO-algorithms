"""
Trust Region Bayesian Optimization with Multiple Trust Regions (TuRBO-M).

This implementation follows the original TuRBO-M code from Eriksson et al.
while adapting to the suggest/observe API pattern with one-by-one point generation.

Reference:
    Eriksson et al. "Scalable Global Optimization via Local Bayesian Optimization"
    NeurIPS 2019.
    GitHub: https://github.com/uber-research/TuRBO
"""

import math
from copy import deepcopy
from typing import Optional, List

import torch
import gpytorch
import numpy as np
from torch.quasirandom import SobolEngine

from .base import BaseOptimizer
from .utils import DEVICE, DTYPE, ensure_tensor


class TurboM(BaseOptimizer):
    """
    Trust Region Bayesian Optimization with Multiple Trust Regions (TuRBO-M).

    Maintains multiple independent trust regions that can explore different
    parts of the space simultaneously, providing better exploration.
    """

    def __init__(
        self,
        input_dim: int,
        bounds: torch.Tensor,
        n_trust_regions: int = 5,
        n_init: int = None,
        max_cholesky_size: int = 2000,
        n_training_steps: int = 50,
        use_ard: bool = True,
        n_candidates: int = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize TuRBO-M optimizer.

        Args:
            input_dim: Dimension of the input space
            bounds: Bounds tensor of shape (2, input_dim)
            n_trust_regions: Number of independent trust regions
            n_init: Number of initial random points per TR (default: 2*dim)
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
        self.n_trust_regions = n_trust_regions
        self.n_init = n_init if n_init is not None else 2 * input_dim
        self.max_cholesky_size = max_cholesky_size
        self.n_training_steps = n_training_steps
        self.use_ard = use_ard
        self.n_candidates = n_candidates if n_candidates is not None else min(100 * input_dim, 5000)

        # Trust region parameters (from original TuRBO)
        self.length_min = 0.5 ** 7
        self.length_max = 1.6
        self.length_init = 0.8

        # Tolerances (from original TuRBO-M)
        self.fail_tol = max(5, input_dim)  # Different from TuRBO-1
        self.succ_tol = 3

        # State variables for each trust region
        self.lengths = torch.full((n_trust_regions,), self.length_init, device=self.device, dtype=self.dtype)
        self.fail_counts = torch.zeros(n_trust_regions, dtype=torch.long, device=self.device)
        self.succ_counts = torch.zeros(n_trust_regions, dtype=torch.long, device=self.device)
        self.gp_hypers = [{} for _ in range(n_trust_regions)]

        # Track which trust region each point belongs to
        # -1 means point is inactive (from restarted TR)
        self.tr_indices = torch.empty((0,), dtype=torch.long, device=self.device)

        # Track initialization state for each TR
        self.tr_init_counts = torch.full((n_trust_regions,), self.n_init, dtype=torch.long, device=self.device)

        self.n_evals = 0

    def suggest(self, n_suggestions: int = 1) -> torch.Tensor:
        """
        Suggest next points to evaluate.

        Points are generated one-by-one, selecting the best candidate across
        all trust regions using Thompson sampling.

        Args:
            n_suggestions: Number of points to suggest

        Returns:
            Suggested points of shape (n_suggestions, input_dim)
        """
        suggestions = []

        for _ in range(n_suggestions):
            # Check if any TR needs initialization
            needs_init = self.tr_init_counts.sum().item() > 0

            if needs_init:
                # Generate random initialization point for the first TR that needs it
                X_next, tr_idx = self._generate_init_point()
            else:
                # Generate Thompson sample from all trust regions
                X_next, tr_idx = self._generate_thompson_sample_multi_tr()

            suggestions.append(X_next)

            # Store which TR this suggestion came from (for later observe)
            # This is temporary storage, will be properly added in observe()

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

        # For each new observation, determine which TR it belongs to
        for i in range(X.shape[0]):
            x_new = X[i:i+1]
            y_new = y[i:i+1]

            # Determine TR index
            if self.tr_init_counts.sum() > 0:
                # During initialization, assign to first TR that needs points
                tr_idx = torch.where(self.tr_init_counts > 0)[0][0].item()
                self.tr_init_counts[tr_idx] -= 1
            else:
                # After initialization, find which TR this point came from
                # For now, we assign to the TR with best match (could track this better)
                tr_idx = self._assign_to_best_tr(x_new, y_new)

            # Add to training data
            self.train_X = torch.cat([self.train_X, x_new], dim=0)
            self.train_y = torch.cat([self.train_y, y_new], dim=0)
            self.tr_indices = torch.cat([self.tr_indices, torch.tensor([tr_idx], device=self.device)])
            self.n_evals += 1

            # Adjust trust region if initialization is done
            if self.tr_init_counts.sum() == 0:
                self._adjust_trust_region(tr_idx, y_new)

            # Check for restart
            if self.lengths[tr_idx] < self.length_min:
                self._restart_trust_region(tr_idx)

    def _generate_init_point(self) -> tuple[torch.Tensor, int]:
        """
        Generate a random initialization point for a TR that needs it.

        Returns:
            Tuple of (random_point, tr_index)
        """
        # Find first TR that needs initialization
        tr_idx = torch.where(self.tr_init_counts > 0)[0][0].item()

        # Generate random point
        X = torch.rand(self.input_dim, device=self.device, dtype=self.dtype)
        X = self.bounds[0] + (self.bounds[1] - self.bounds[0]) * X

        return X, tr_idx

    def _generate_thompson_sample_multi_tr(self) -> tuple[torch.Tensor, int]:
        """
        Generate Thompson samples from all trust regions and select the best.

        Returns:
            Tuple of (best_point, tr_index)
        """
        best_sample = None
        best_value = float('inf')
        best_tr_idx = 0

        # Generate candidates from each trust region
        for tr_idx in range(self.n_trust_regions):
            # Get data for this trust region
            mask = self.tr_indices == tr_idx
            if mask.sum() < 3:  # Need at least 3 points to fit GP
                continue

            X_tr = self.train_X[mask]
            y_tr = self.train_y[mask]

            try:
                # Generate Thompson sample from this TR
                X_sample, y_sample = self._thompson_sample_from_tr(X_tr, y_tr, self.lengths[tr_idx], tr_idx)

                # Keep track of best sample across all TRs
                if y_sample < best_value:
                    best_value = y_sample
                    best_sample = X_sample
                    best_tr_idx = tr_idx

            except Exception as e:
                # If GP fitting fails for this TR, skip it
                continue

        # If no valid sample found, generate random point
        if best_sample is None:
            best_sample = torch.rand(self.input_dim, device=self.device, dtype=self.dtype)
            best_sample = self.bounds[0] + (self.bounds[1] - self.bounds[0]) * best_sample
            best_tr_idx = 0

        return best_sample, best_tr_idx

    def _thompson_sample_from_tr(
        self,
        X_tr: torch.Tensor,
        y_tr: torch.Tensor,
        length: float,
        tr_idx: int
    ) -> tuple[torch.Tensor, float]:
        """
        Generate a Thompson sample from a specific trust region.

        Args:
            X_tr: Training data for this TR
            y_tr: Training targets for this TR
            length: Trust region length
            tr_idx: Trust region index

        Returns:
            Tuple of (sampled_point, sampled_value)
        """
        # Normalize inputs to [0, 1]
        X_norm = self._normalize_to_unit_cube(X_tr)

        # Standardize outputs
        y_mean = y_tr.mean()
        y_std = y_tr.std()
        if y_std < 1e-6:
            y_std = torch.tensor(1.0, device=self.device, dtype=self.dtype)
        y_norm = (y_tr - y_mean) / y_std

        # Train or reuse GP
        if not self.gp_hypers[tr_idx]:
            # Need to train GP
            gp = self._train_gp(X_norm, y_norm)
            self.gp_hypers[tr_idx] = gp.state_dict()
        else:
            # Reuse existing hypers
            gp = self._train_gp(X_norm, y_norm)
            gp.load_state_dict(self.gp_hypers[tr_idx])

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
        lb = np.clip(x_center.cpu().numpy() - weights * length / 2.0, 0.0, 1.0)
        ub = np.clip(x_center.cpu().numpy() + weights * length / 2.0, 0.0, 1.0)

        # Generate candidates in trust region
        sobol = SobolEngine(self.input_dim, scramble=True)
        candidates_unit = sobol.draw(self.n_candidates).to(device=self.device, dtype=self.dtype)
        candidates = torch.from_numpy(lb).to(device=self.device, dtype=self.dtype) + \
                     (torch.from_numpy(ub - lb).to(device=self.device, dtype=self.dtype)) * candidates_unit

        # Apply perturbation mask
        prob_perturb = min(20.0 / self.input_dim, 1.0)
        mask = torch.rand(self.n_candidates, self.input_dim, device=self.device) <= prob_perturb
        mask[mask.sum(dim=1) == 0, torch.randint(0, self.input_dim, (1,))] = True

        x_center_expanded = x_center.unsqueeze(0).expand(self.n_candidates, -1)
        X_cand = x_center_expanded.clone()
        X_cand[mask] = candidates[mask]

        # Thompson sampling: sample from GP posterior
        with torch.no_grad(), gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            posterior = gp.likelihood(gp(X_cand))
            y_samples = posterior.sample()

        # Destandardize samples
        y_samples = y_mean + y_std * y_samples

        # Select best sample
        best_idx = y_samples.argmin()
        X_next_norm = X_cand[best_idx]
        y_next = y_samples[best_idx].item()

        # Denormalize back to original space
        X_next = self._denormalize_from_unit_cube(X_next_norm.unsqueeze(0)).squeeze(0)

        return X_next, y_next

    def _assign_to_best_tr(self, X: torch.Tensor, y: torch.Tensor) -> int:
        """
        Assign a new point to the trust region it most likely came from.

        This is a heuristic: assign to the TR whose center is closest.

        Args:
            X: New point
            y: New value

        Returns:
            Trust region index
        """
        min_dist = float('inf')
        best_tr = 0

        for tr_idx in range(self.n_trust_regions):
            mask = self.tr_indices == tr_idx
            if mask.sum() == 0:
                continue

            X_tr = self.train_X[mask]
            y_tr = self.train_y[mask]

            # Find center of this TR (best point)
            center = X_tr[y_tr.argmin()]

            # Compute distance
            dist = torch.norm(X.squeeze() - center)

            if dist < min_dist:
                min_dist = dist
                best_tr = tr_idx

        return best_tr

    def _adjust_trust_region(self, tr_idx: int, y_new: torch.Tensor) -> None:
        """
        Adjust trust region length based on improvement.

        Args:
            tr_idx: Trust region index
            y_new: New observed value
        """
        # Get best value in this TR before new observation
        mask = self.tr_indices == tr_idx
        y_tr = self.train_y[mask]

        if y_tr.shape[0] <= 1:
            return

        y_best = y_tr[:-1].min()  # Exclude the new observation

        # Check for improvement (from original TuRBO-M)
        improvement_threshold = 1e-3 * abs(y_best.item())
        if y_new.min() < y_best - improvement_threshold:
            # Success: improvement found
            self.succ_counts[tr_idx] += 1
            self.fail_counts[tr_idx] = 0
        else:
            # Failure: no improvement
            self.succ_counts[tr_idx] = 0
            self.fail_counts[tr_idx] += 1

        # Adjust trust region length
        if self.succ_counts[tr_idx] == self.succ_tol:
            # Expand trust region
            self.lengths[tr_idx] = min(2.0 * self.lengths[tr_idx], self.length_max)
            self.succ_counts[tr_idx] = 0
        elif self.fail_counts[tr_idx] >= self.fail_tol:
            # Shrink trust region
            self.lengths[tr_idx] = self.lengths[tr_idx] / 2.0
            self.fail_counts[tr_idx] = 0

    def _restart_trust_region(self, tr_idx: int) -> None:
        """
        Restart a trust region when it becomes too small.

        Args:
            tr_idx: Trust region index
        """
        print(f"TuRBO-M: Restarting TR-{tr_idx} (n_evals={self.n_evals}, "
              f"best={self.train_y[self.tr_indices == tr_idx].min().item():.4f})")

        # Reset TR state
        self.lengths[tr_idx] = self.length_init
        self.fail_counts[tr_idx] = 0
        self.succ_counts[tr_idx] = 0
        self.gp_hypers[tr_idx] = {}

        # Mark old points as inactive
        self.tr_indices[self.tr_indices == tr_idx] = -1

        # Request reinitialization
        self.tr_init_counts[tr_idx] = self.n_init

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

    def get_best_per_tr(self) -> List[tuple[torch.Tensor, torch.Tensor]]:
        """
        Get the best point from each trust region.

        Returns:
            List of (best_x, best_y) tuples for each TR
        """
        results = []
        for tr_idx in range(self.n_trust_regions):
            mask = self.tr_indices == tr_idx
            if mask.sum() == 0:
                results.append((None, None))
            else:
                y_tr = self.train_y[mask]
                X_tr = self.train_X[mask]
                best_idx = y_tr.argmin()
                results.append((X_tr[best_idx], y_tr[best_idx]))
        return results
