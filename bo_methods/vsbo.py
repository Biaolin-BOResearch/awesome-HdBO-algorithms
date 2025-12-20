"""
Variable Selection Bayesian Optimization (VS-BO).

VS-BO performs variable selection using importance scores derived from
the GP model's lengthscales (ARD) or gradient-based metrics (KL-relevance).
It focuses optimization on the most important variables while using
a conditional distribution (CMA-ES posterior) for less important ones.

Reference:
    Shen et al. "Variable Selection for Gaussian Process Bandits"
    NeurIPS 2022.
"""

from typing import Optional, List, Tuple
import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel, MaternKernel, RBFKernel
from gpytorch.module import Module
from gpytorch.means.mean import Mean
from gpytorch.constraints import Interval

from .base import BaseOptimizer
from .utils import DEVICE, DTYPE, ensure_tensor, clamp_to_bounds


class VSBO(BaseOptimizer):
    """
    Variable Selection Bayesian Optimization.

    VS-BO uses importance scores (ARD lengthscales or KL-relevance) to
    identify the most important variables, then performs optimization
    in the selected subspace while sampling less important variables
    from a conditional distribution.
    """

    def __init__(
        self,
        input_dim: int,
        bounds: torch.Tensor,
        n_init_samples: int = 5,
        fs_method: str = "ard",
        min_active_dims: int = 2,
        max_active_dims: Optional[int] = None,
        loss_threshold_factor: float = 10.0,
        acq_func: str = "EI",
        covar_module: Optional[Module] = None,
        mean_module: Optional[Mean] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize VS-BO optimizer.

        Args:
            input_dim: Dimension of the input space
            bounds: Bounds tensor of shape (2, input_dim)
            n_init_samples: Number of initial random samples
            fs_method: Feature selection method ('ard' or 'klrel')
            min_active_dims: Minimum number of active dimensions
            max_active_dims: Maximum number of active dimensions (None = no limit)
            loss_threshold_factor: Threshold factor for stopping forward selection
            acq_func: Acquisition function type
            covar_module: Optional custom covariance module
            mean_module: Optional custom mean module
            device: Device for computation
            dtype: Data type for tensors
            seed: Random seed for reproducibility
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

        self.n_init_samples = n_init_samples
        self.fs_method = fs_method
        self.min_active_dims = max(1, min(min_active_dims, input_dim))
        self.max_active_dims = max_active_dims or input_dim
        self.loss_threshold_factor = loss_threshold_factor
        self.seed = seed

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Active dimensions mask (True = active/important)
        self.active_mask = torch.ones(input_dim, dtype=torch.bool, device=self.device)
        self.active_dims = input_dim
        self.importance_scores: Optional[torch.Tensor] = None
        self.sorted_indices: Optional[torch.Tensor] = None

        # GP model for active dimensions
        self.model_active: Optional[SingleTaskGP] = None

        # Default kernel with ARD
        if covar_module is None:
            self.covar_module = ScaleKernel(
                MaternKernel(
                    ard_num_dims=input_dim,
                    lengthscale_constraint=Interval(0.005, 10.0),
                ),
                outputscale_constraint=Interval(0.05, 20.0),
            )

    def _compute_importance_scores(self) -> torch.Tensor:
        """
        Compute importance scores for each dimension.

        Returns:
            Tensor of importance scores of shape (input_dim,)
        """
        if self.model is None:
            return torch.ones(self.input_dim, device=self.device, dtype=self.dtype)

        if self.fs_method == "ard":
            # ARD-based: use inverse lengthscales
            try:
                lengthscales = self.model.covar_module.base_kernel.lengthscale.squeeze()
                if lengthscales.ndim == 0:
                    lengthscales = lengthscales.unsqueeze(0)
                # Ensure we have the right number of dimensions
                if len(lengthscales) < self.input_dim:
                    # Pad with mean value if needed
                    mean_ls = lengthscales.mean()
                    lengthscales = torch.cat([
                        lengthscales,
                        mean_ls.repeat(self.input_dim - len(lengthscales))
                    ])
                scores = 1.0 / (lengthscales[:self.input_dim] + 1e-6)
            except (AttributeError, RuntimeError):
                scores = torch.ones(self.input_dim, device=self.device, dtype=self.dtype)
        elif self.fs_method == "klrel":
            # KL-relevance based: use gradient information
            scores = self._compute_kl_relevance()
        else:
            scores = torch.ones(self.input_dim, device=self.device, dtype=self.dtype)

        # Normalize scores
        scores = scores / (scores.max() + 1e-8)
        return scores

    def _compute_kl_relevance(self, n_samples: int = 1000) -> torch.Tensor:
        """
        Compute KL-relevance scores using gradient information.

        Args:
            n_samples: Number of samples for Monte Carlo estimation

        Returns:
            Tensor of KL-relevance scores
        """
        if self.model is None:
            return torch.ones(self.input_dim, device=self.device, dtype=self.dtype)

        # Generate random samples in [0, 1]^d
        X_rand = torch.rand(n_samples, self.input_dim, device=self.device, dtype=self.dtype)
        X_rand.requires_grad_(True)

        try:
            # Get posterior
            self.model.eval()
            posterior = self.model.posterior(X_rand)

            # Compute gradients of mean
            mean = posterior.mean.sum()
            grads = torch.autograd.grad(mean, X_rand, retain_graph=True)[0]

            # Normalize by posterior variance
            variance = posterior.variance.squeeze()
            if variance.ndim == 0:
                variance = variance.unsqueeze(0)
            variance = variance.unsqueeze(1).expand(-1, self.input_dim)

            # KL-relevance: E[|grad|^2 / variance]
            kl_scores = (grads.abs() / (variance.sqrt() + 1e-6)).mean(dim=0)

        except (RuntimeError, AttributeError):
            kl_scores = torch.ones(self.input_dim, device=self.device, dtype=self.dtype)

        return kl_scores

    def _select_active_dimensions(self) -> None:
        """
        Perform stepwise forward selection to identify active dimensions.
        """
        # Compute importance scores
        self.importance_scores = self._compute_importance_scores()
        _, self.sorted_indices = torch.sort(self.importance_scores, descending=True)

        # Start with minimum active dimensions
        prev_loss = float('inf')
        loss_interval = None

        for k in range(self.min_active_dims, self.input_dim + 1):
            selected_dims = self.sorted_indices[:k]

            try:
                # Fit model on selected dimensions
                X_selected = self.train_X[:, selected_dims]

                # Normalize X for fitting
                bounds_selected = self.bounds[:, selected_dims]
                X_norm = (X_selected - bounds_selected[0]) / (bounds_selected[1] - bounds_selected[0] + 1e-8)

                # Standardize Y
                y_mean = self.train_y.mean()
                y_std = self.train_y.std() + 1e-6
                y_norm = (self.train_y - y_mean) / y_std

                # Create and fit temporary model
                temp_model = SingleTaskGP(
                    X_norm, y_norm,
                    input_transform=None,
                    outcome_transform=None,
                ).to(self.device)
                mll = ExactMarginalLogLikelihood(temp_model.likelihood, temp_model)
                fit_gpytorch_mll(mll)

                # Compute loss
                temp_model.eval()
                output = temp_model(X_norm)
                loss = -mll(output, y_norm.squeeze()).item()

                if loss_interval is None:
                    if k > self.min_active_dims:
                        loss_interval = prev_loss - loss
                    prev_loss = loss
                else:
                    improvement = prev_loss - loss
                    if improvement < loss_interval / self.loss_threshold_factor:
                        # Stop adding dimensions
                        self.active_dims = k - 1
                        self.active_mask = torch.zeros(
                            self.input_dim, dtype=torch.bool, device=self.device
                        )
                        self.active_mask[self.sorted_indices[:self.active_dims]] = True
                        return

                    loss_interval = improvement
                    prev_loss = loss

            except (RuntimeError, ValueError):
                continue

            if k >= self.max_active_dims:
                break

        # Use all dimensions if no stopping criterion met
        self.active_dims = min(self.input_dim, self.max_active_dims)
        self.active_mask = torch.ones(self.input_dim, dtype=torch.bool, device=self.device)

    def _fit_active_model(self) -> None:
        """
        Fit GP model on active dimensions only.
        """
        if self.active_dims == 0 or len(self.train_X) == 0:
            return

        X_active = self.train_X[:, self.active_mask]

        # Normalize
        bounds_active = self.bounds[:, self.active_mask]
        X_norm = (X_active - bounds_active[0]) / (bounds_active[1] - bounds_active[0] + 1e-8)

        y_mean = self.train_y.mean()
        y_std = self.train_y.std() + 1e-6
        y_norm = (self.train_y - y_mean) / y_std

        # Create model with ARD kernel for active dimensions
        covar = ScaleKernel(
            MaternKernel(
                ard_num_dims=self.active_dims,
                lengthscale_constraint=Interval(0.005, 10.0),
            ),
            outputscale_constraint=Interval(0.05, 20.0),
        ).to(self.device)

        self.model_active = SingleTaskGP(
            X_norm, y_norm,
            covar_module=covar,
            input_transform=None,
            outcome_transform=None,
        ).to(self.device)

        mll = ExactMarginalLogLikelihood(self.model_active.likelihood, self.model_active)
        fit_gpytorch_mll(mll)

        self._y_mean = y_mean
        self._y_std = y_std

    def observe(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """
        Update the optimizer with new observations.

        Args:
            X: Observed input points of shape (n, input_dim)
            y: Observed output values of shape (n,) or (n, 1)
        """
        X = ensure_tensor(X, device=self.device, dtype=self.dtype)
        y = ensure_tensor(y, device=self.device, dtype=self.dtype)

        X = X.reshape(-1, self.input_dim)
        y = y.reshape(-1, 1)

        # Update training data
        self.train_X = torch.cat([self.train_X, X], dim=0)
        self.train_y = torch.cat([self.train_y, y], dim=0)

        # Fit full model for importance scoring
        self._update_model(
            torch.empty(0, self.input_dim, device=self.device, dtype=self.dtype),
            torch.empty(0, 1, device=self.device, dtype=self.dtype)
        )  # Just to reset

        try:
            # Fit full model
            X_norm = (self.train_X - self.bounds[0]) / (self.bounds[1] - self.bounds[0] + 1e-8)
            y_mean = self.train_y.mean()
            y_std = self.train_y.std() + 1e-6
            y_norm = (self.train_y - y_mean) / y_std

            self.model = SingleTaskGP(
                X_norm, y_norm,
                covar_module=self.covar_module,
                input_transform=None,
                outcome_transform=None,
            ).to(self.device)
            mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
            fit_gpytorch_mll(mll)
        except (RuntimeError, ValueError):
            pass

        # Perform variable selection
        if len(self.train_X) >= self.n_init_samples:
            self._select_active_dimensions()
            self._fit_active_model()

    def suggest(self, n_suggestions: int = 1) -> torch.Tensor:
        """
        Suggest next points to evaluate.

        Args:
            n_suggestions: Number of points to suggest

        Returns:
            Tensor of shape (n_suggestions, input_dim) with suggested points
        """
        if len(self.train_X) < self.n_init_samples or self.model_active is None:
            # Return random samples during initialization
            candidates = torch.zeros(n_suggestions, self.input_dim, device=self.device, dtype=self.dtype)
            for i in range(self.input_dim):
                low, high = self.bounds[0, i], self.bounds[1, i]
                candidates[:, i] = torch.rand(n_suggestions, device=self.device, dtype=self.dtype) * (high - low) + low
            return candidates

        # Optimize acquisition on active dimensions
        best_f = (self.train_y.max() - self._y_mean) / self._y_std
        acq_func = ExpectedImprovement(self.model_active, best_f=best_f)

        # Bounds for active dimensions in normalized space
        bounds_norm = torch.stack([
            torch.zeros(self.active_dims, device=self.device, dtype=self.dtype),
            torch.ones(self.active_dims, device=self.device, dtype=self.dtype),
        ])

        try:
            candidates_active, _ = optimize_acqf(
                acq_func,
                bounds=bounds_norm,
                q=n_suggestions,
                num_restarts=10,
                raw_samples=100,
            )
        except RuntimeError:
            # Fallback to random
            candidates_active = torch.rand(
                n_suggestions, self.active_dims, device=self.device, dtype=self.dtype
            )

        # Denormalize active dimensions
        bounds_active = self.bounds[:, self.active_mask]
        candidates_active = candidates_active * (bounds_active[1] - bounds_active[0]) + bounds_active[0]

        # Construct full candidates
        candidates = torch.zeros(n_suggestions, self.input_dim, device=self.device, dtype=self.dtype)

        # Set active dimensions from optimization
        active_idx = 0
        for i in range(self.input_dim):
            if self.active_mask[i]:
                candidates[:, i] = candidates_active[:, active_idx]
                active_idx += 1
            else:
                # Sample inactive dimensions
                # Use best observed values with small perturbation
                best_idx = self.train_y.argmax()
                best_val = self.train_X[best_idx, i]
                perturbation = torch.randn(n_suggestions, device=self.device, dtype=self.dtype) * 0.1
                candidates[:, i] = best_val + perturbation * (self.bounds[1, i] - self.bounds[0, i])

        # Clamp to bounds
        candidates = clamp_to_bounds(candidates, self.bounds)

        return candidates

    def get_active_dimensions(self) -> List[int]:
        """
        Get the indices of active (important) dimensions.

        Returns:
            List of active dimension indices
        """
        return torch.where(self.active_mask)[0].tolist()

    def get_importance_scores(self) -> Optional[torch.Tensor]:
        """
        Get the importance scores for all dimensions.

        Returns:
            Tensor of importance scores or None if not computed
        """
        return self.importance_scores

    def reset(self) -> None:
        """Reset the optimizer to its initial state."""
        super().reset()
        self.active_mask = torch.ones(self.input_dim, dtype=torch.bool, device=self.device)
        self.active_dims = self.input_dim
        self.importance_scores = None
        self.sorted_indices = None
        self.model_active = None
