"""
Generalized Product of Experts Trust Region Bayesian Optimization (gPOE-TuRBO).

Combines the gPOE ensemble approach with TuRBO's trust region framework
for efficient high-dimensional optimization. Implementation aligned with
https://github.com/saultaut/Scalable-BO-with-gPoE

Reference:
    Eriksson et al. "Scalable Global Optimization via Local Bayesian Optimization"
    NeurIPS 2019 (for TuRBO).
"""

from typing import Optional
import math
from dataclasses import dataclass
from copy import deepcopy
import torch
import numpy as np
import gpytorch
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.fit import fit_gpytorch_model
from torch.quasirandom import SobolEngine
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints import GreaterThan

from .base import BaseOptimizer
from .utils import DEVICE, DTYPE, ensure_tensor


@dataclass
class TurboState:
    """State variables for TuRBO trust region management."""
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = None  # Will be set in __post_init__
    success_counter: int = 0
    success_tolerance: int = 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        """Initialize failure tolerance based on dimensionality and batch size."""
        if self.failure_tolerance is None:
            self.failure_tolerance = math.ceil(
                max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
            )


class GPOETRBO(BaseOptimizer):
    """
    gPOE-based Trust Region Bayesian Optimization.

    Combines ensemble of GP experts with trust region optimization
    for improved scalability in high dimensions.

    This implementation follows the reference code from:
    https://github.com/saultaut/Scalable-BO-with-gPoE
    """

    def __init__(
        self,
        input_dim: int,
        bounds: torch.Tensor,
        points_per_expert: int = 20,
        n_init: int = 50,
        prob_perturb: float = 0.2,
        perturb_rate: int = 200,
        n_candidates: int = 5000,
        batch_size: int = 5000,
        beta: float = 1.0,
        weighting: str = "diff_entr",
        length_min: float = 0.5**7,
        length_max: float = 1.6,
        length_init: float = 0.8,
        success_tolerance: int = 3,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize gPOE-TuRBO optimizer.

        Args:
            input_dim: Dimension of the input space
            bounds: Bounds tensor of shape (2, input_dim)
            points_per_expert: Number of data points per expert (default: 20)
            n_init: Number of random initialization points (default: 50)
            prob_perturb: Probability of perturbing each dimension (default: 0.2)
            perturb_rate: Rate at which to update perturbation mask (default: 200)
            n_candidates: Number of candidate points for optimization (default: 5000)
            batch_size: Batch size for GP predictions (default: 5000)
            beta: UCB exploration parameter (default: 1.0)
            weighting: Method for combining expert predictions (default: 'diff_entr')
            length_min: Minimum trust region length
            length_max: Maximum trust region length
            length_init: Initial trust region length
            success_tolerance: Number of successes before expanding TR
            device: Device for computation
            dtype: Data type for tensors
        """
        super().__init__(
            input_dim=input_dim,
            bounds=bounds,
            acq_func="ucb",
            device=device,
            dtype=dtype,
        )

        # gPOE parameters
        self.points_per_expert = points_per_expert
        self.n_init = n_init
        self.n_candidates = n_candidates
        self.batch_size = batch_size
        self.beta = torch.tensor(beta, device=self.device, dtype=self.dtype)
        self.weighting = weighting

        # Trust region parameters
        self.prob_perturb = prob_perturb
        self.perturb_rate = perturb_rate

        # TuRBO state
        self.state = TurboState(
            dim=input_dim,
            batch_size=1,  # Single point suggestions
            length=length_init,
            length_min=length_min,
            length_max=length_max,
            success_tolerance=success_tolerance,
        )

        # Evaluation counter
        self.n_evals = 0

        # Perturbation mask (for partial perturbations within TR)
        self.mask = None

    def suggest(self, n_suggestions: int = 1) -> torch.Tensor:
        """
        Suggest next points using gPOE within trust region.

        Args:
            n_suggestions: Number of points to suggest

        Returns:
            Suggested points of shape (n_suggestions, input_dim)

        Note:
            Output shape is always (n_suggestions, input_dim), even if n_suggestions=1
        """
        suggestions = []

        for _ in range(n_suggestions):
            # Handle initialization phase
            if self.train_X.shape[0] < self.n_init:
                X_next = self._random_suggestion()
            else:
                # Handle restart
                if self.state.restart_triggered:
                    print(f"{self.n_evals}) Restarting with fbest = {self.train_y.max():.4f}")
                    self.state = TurboState(dim=self.input_dim, batch_size=1)
                    X_next = self._random_suggestion()
                else:
                    # Normal TR-based suggestion
                    X_next = self._suggest_in_trust_region()

            suggestions.append(X_next)
            self.n_evals += 1

        # Stack to ensure (n_suggestions, input_dim) shape
        result = torch.stack(suggestions) if len(suggestions) > 1 else suggestions[0].unsqueeze(0)
        assert result.shape == (n_suggestions, self.input_dim), \
            f"Expected shape ({n_suggestions}, {self.input_dim}), got {result.shape}"
        return result

    def observe(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """
        Update with new observations and adjust trust region.

        Args:
            X: Observed input points of shape (n, input_dim)
            y: Observed output values of shape (n,) or (n, 1)
        """
        X = ensure_tensor(X, device=self.device, dtype=self.dtype)
        y = ensure_tensor(y, device=self.device, dtype=self.dtype)

        X = X.reshape(-1, self.input_dim)
        y = y.reshape(-1, 1)

        # Update data
        self.train_X = torch.cat([self.train_X, X], dim=0)
        self.train_y = torch.cat([self.train_y, y], dim=0)

        # Update trust region after initialization
        if self.train_X.shape[0] >= self.n_init:
            self._update_trust_region(y)

    def _random_suggestion(self) -> torch.Tensor:
        """
        Generate a single random suggestion using Sobol sampling.

        Returns:
            Random point of shape (input_dim,)
        """
        sobol = SobolEngine(dimension=self.input_dim, scramble=True)
        X_sobol = sobol.draw(n=1).to(device=self.device, dtype=self.dtype).squeeze(0)

        # Scale to bounds
        X_scaled = self.bounds[0] + (self.bounds[1] - self.bounds[0]) * X_sobol

        return X_scaled

    def _suggest_in_trust_region(self) -> torch.Tensor:
        """
        Generate suggestion within trust region using gPOE ensemble.

        Returns:
            Suggested point of shape (input_dim,)
        """
        # Determine number of experts
        n_experts = max(1, int(self.train_X.shape[0] / self.points_per_expert))
        n_per_expert = int(self.train_X.shape[0] / n_experts)

        if self.n_evals % 20 == 0:
            print(f'Number of experts: {n_experts}')

        # Random partition of data across experts
        partition = np.random.choice(
            self.train_X.shape[0],
            size=(n_experts, n_per_expert),
            replace=False,
        )

        # Standardize targets
        train_y_mean = self.train_y.mean()
        # Handle single observation case: std() returns NaN for single element
        if self.train_y.numel() < 2:
            train_y_std = torch.tensor(1.0, device=self.device, dtype=self.dtype)
        else:
            train_y_std = self.train_y.std()
            # Check for NaN or very small std
            if torch.isnan(train_y_std) or train_y_std < 1e-8:
                train_y_std = torch.tensor(1.0, device=self.device, dtype=self.dtype)
        train_y_normalized = (self.train_y - train_y_mean) / train_y_std

        # Get best point (center of trust region)
        best_idx = self.train_y.argmax()
        x_best = self.train_X[best_idx]

        # Create batched training data
        batched_X = torch.stack([self.train_X[partition[k]] for k in range(n_experts)])
        batched_y = torch.stack([train_y_normalized[partition[k]] for k in range(n_experts)])

        # Train ensemble
        try:
            model = self._get_ensemble_model(batched_X, batched_y)
        except Exception as ex:
            print(f"Error fitting ensemble: {ex}")
            return self._random_suggestion()

        # Compute ARD weights for trust region
        weights = self._compute_ard_weights(model)

        # Normalize to unit cube for TR calculations
        x_best_norm = self._normalize_to_unit_cube(x_best)

        # Compute trust region bounds
        tr_lb = torch.clamp(x_best_norm - weights * self.state.length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_best_norm + weights * self.state.length / 2.0, 0.0, 1.0)

        # Generate perturbations within trust region
        sobol = SobolEngine(dimension=self.input_dim, scramble=True)
        pert = sobol.draw(n=self.n_candidates).to(device=self.device, dtype=self.dtype)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        # Update perturbation mask periodically
        if self.n_evals % self.perturb_rate == 0 or self.n_evals == self.n_init or self.mask is None:
            mask = np.random.rand(self.n_candidates, self.input_dim) <= self.prob_perturb
            # Ensure at least one dimension is perturbed
            ind = np.where(np.sum(mask, axis=1) == 0)[0]
            if len(ind) > 0:
                mask[ind, np.random.randint(0, self.input_dim, size=len(ind))] = True
            self.mask = mask

        # Create candidates by perturbing only selected dimensions
        ones_matrix = np.ones((self.n_candidates, self.input_dim))
        X_test = x_best_norm.cpu().numpy().copy() * ones_matrix
        X_test[self.mask] = pert.cpu().numpy()[self.mask]
        X_test = torch.tensor(X_test).to(device=self.device, dtype=self.dtype)

        # Get predictions from ensemble
        mu_s, var_s, prior = self._predict_ensemble(model, X_test, n_experts)

        # Combine using gPOE
        mu, var = self._combine_experts(mu_s, var_s, prior)

        # Compute UCB acquisition
        ucb = mu + torch.sqrt(self.beta) * var.sqrt()

        # Select best candidate
        best_idx = ucb.argmax()
        X_next_norm = X_test[best_idx]

        # Denormalize to original bounds
        X_next = self._denormalize_from_unit_cube(X_next_norm)

        return X_next

    def _update_trust_region(self, y_new: torch.Tensor) -> None:
        """
        Update trust region state based on new observations.

        Args:
            y_new: New observed values
        """
        # Check for improvement
        improvement_threshold = 1e-3 * abs(self.state.best_value)
        if y_new.max().item() > self.state.best_value + improvement_threshold:
            # Success
            self.state.success_counter += 1
            self.state.failure_counter = 0
            self.state.best_value = y_new.max().item()
        else:
            # Failure
            self.state.success_counter = 0
            self.state.failure_counter += 1

        # Adjust trust region length
        if self.state.success_counter == self.state.success_tolerance:
            # Expand
            self.state.length = min(2.0 * self.state.length, self.state.length_max)
            self.state.success_counter = 0

        elif self.state.failure_counter == self.state.failure_tolerance:
            # Shrink
            self.state.length /= 2.0
            self.state.failure_counter = 0

        # Update best value tracked across all observations
        self.state.best_value = max(self.state.best_value, self.train_y.max().item())

        # Check for restart
        if self.state.length < self.state.length_min:
            self.state.restart_triggered = True

    def _compute_ard_weights(self, model: SingleTaskGP) -> torch.Tensor:
        """
        Compute ARD weights from GP lengthscales.

        Args:
            model: Fitted GP model

        Returns:
            ARD weights of shape (input_dim,)
        """
        try:
            # Extract lengthscales from the kernel
            lengthscales = model.covar_module.base_kernel.lengthscale.detach()[0, 0, :]
            weights = lengthscales / lengthscales.mean()
            # Geometric mean normalization
            weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
            return weights
        except:
            # Fallback to uniform weights
            return torch.ones(self.input_dim, device=self.device, dtype=self.dtype)

    def _normalize_to_unit_cube(self, X: torch.Tensor) -> torch.Tensor:
        """Normalize from original bounds to [0, 1]^d."""
        return (X - self.bounds[0]) / (self.bounds[1] - self.bounds[0])

    def _denormalize_from_unit_cube(self, X: torch.Tensor) -> torch.Tensor:
        """Denormalize from [0, 1]^d to original bounds."""
        return self.bounds[0] + (self.bounds[1] - self.bounds[0]) * X

    def _get_ensemble_model(
        self, batched_X: torch.Tensor, batched_y: torch.Tensor
    ) -> SingleTaskGP:
        """
        Create and fit batched GP model for ensemble.

        Args:
            batched_X: Batched training inputs of shape (n_experts, n_per_expert, input_dim)
            batched_y: Batched training targets of shape (n_experts, n_per_expert, 1)

        Returns:
            Fitted SingleTaskGP model
        """
        model = SingleTaskGP(
            batched_X,
            batched_y,  # already standardized upstream
            input_transform=Normalize(d=self.input_dim, bounds=self.bounds),
            outcome_transform=None,
        )
        model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-5))

        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll, options={'maxiter': 500})

        return model

    def _predict_ensemble(
        self, model: SingleTaskGP, X_test: torch.Tensor, n_experts: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get predictions from ensemble of experts.

        Args:
            model: Batched GP model
            X_test: Test points of shape (n_candidates, input_dim)
            n_experts: Number of experts

        Returns:
            Tuple of (means, variances, prior_variances) each of shape (n_experts, n_candidates)
        """
        mu_s = torch.zeros(n_experts, self.n_candidates, device=self.device, dtype=self.dtype)
        var_s = torch.zeros(n_experts, self.n_candidates, device=self.device, dtype=self.dtype)
        prior = torch.zeros(n_experts, self.n_candidates, device=self.device, dtype=self.dtype)

        # Process in batches
        num_batches = math.ceil(self.n_candidates / self.batch_size)

        for i in range(num_batches):
            start_i = i * self.batch_size
            end_i = min((i + 1) * self.batch_size, self.n_candidates)
            X_batch = X_test[start_i:end_i]

            try:
                # Get prior variance
                with gpytorch.settings.prior_mode(True):
                    y_prior = model.likelihood(model(X_batch))
                    prior[:, start_i:end_i] = y_prior.variance.detach()

                # Get posterior predictions
                posterior = model.posterior(X_batch)
                y_pred = model.likelihood(posterior.mvn)

                mu_s[:, start_i:end_i] = y_pred.mean.detach()
                var_s[:, start_i:end_i] = y_pred.variance.detach()

            except Exception as e:
                print(f"Prediction error in batch {i}: {e}")
                continue

        return mu_s, var_s, prior

    def _combine_experts(
        self,
        mu_s: torch.Tensor,
        var_s: torch.Tensor,
        prior: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Combine expert predictions using generalized Product of Experts.

        Args:
            mu_s: Expert means of shape (n_experts, n_candidates)
            var_s: Expert variances of shape (n_experts, n_candidates)
            prior: Prior variances of shape (n_experts, n_candidates)

        Returns:
            Tuple of (combined_mean, combined_variance) each of shape (n_candidates,)
        """
        # Compute weights
        weight_matrix = self._compute_weights(mu_s, var_s, prior)

        # Convert to precision
        prec_s = 1.0 / (var_s + 1e-8)

        # Normalize weights
        weight_matrix = self._normalize_weights(weight_matrix)

        # Combined precision
        prec = torch.sum(weight_matrix * prec_s, dim=0)

        # Combined variance
        var = 1.0 / (prec + 1e-8)

        # Combined mean
        mu = var * torch.sum(weight_matrix * prec_s * mu_s, dim=0)

        return mu, var

    def _compute_weights(
        self,
        mu_s: torch.Tensor,
        var_s: torch.Tensor,
        prior: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weights for combining expert predictions."""
        if self.weighting == "uniform":
            return torch.ones_like(mu_s) / mu_s.shape[0]

        elif self.weighting == "diff_entr":
            # Differential entropy weighting
            return 0.5 * (torch.log(prior + 1e-8) - torch.log(var_s + 1e-8))

        elif self.weighting == "variance":
            power = 10
            return torch.exp(-power * var_s)

        elif self.weighting == "no_weights":
            return torch.ones_like(mu_s)

        else:
            raise ValueError(f"Unknown weighting method: {self.weighting}")

    def _normalize_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """Normalize weights to sum to 1 along expert dimension."""
        sum_weights = torch.sum(weights, dim=0, keepdim=True)
        return weights / (sum_weights + 1e-8)

    def reset(self) -> None:
        """Reset the optimizer to its initial state."""
        super().reset()
        self.state = TurboState(dim=self.input_dim, batch_size=1)
        self.n_evals = 0
        self.mask = None
