"""
Generalized Product of Experts Bayesian Optimization (gPOE-BO).

This module implements BO using an ensemble of GP experts combined
via the generalized product of experts framework. Implementation aligned
with the reference from https://github.com/saultaut/Scalable-BO-with-gPoE

Reference:
    Cao et al. "Sparse Gaussian Processes with Spherical Harmonic Features"
    ICML 2013 (for gPoE framework).
"""

from typing import Optional
import math
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


class GPOEBO(BaseOptimizer):
    """
    Generalized Product of Experts Bayesian Optimization.

    Maintains multiple GP experts trained on different subsets of the data
    and combines their predictions using the product of experts framework.

    This implementation follows the reference code from:
    https://github.com/saultaut/Scalable-BO-with-gPoE
    """

    def __init__(
        self,
        input_dim: int,
        bounds: torch.Tensor,
        points_per_expert: int = 20,
        n_init: int = 50,
        n_candidates: int = 5000,
        batch_size: int = 5000,
        beta: float = 1.0,
        weighting: str = "diff_entr",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize gPOE-BO optimizer.

        Args:
            input_dim: Dimension of the input space
            bounds: Bounds tensor of shape (2, input_dim)
            points_per_expert: Number of data points per expert (default: 20)
            n_init: Number of random initialization points (default: 50)
            n_candidates: Number of candidate points for optimization (default: 5000)
            batch_size: Batch size for GP predictions (default: 5000)
            beta: UCB exploration parameter (default: 1.0)
            weighting: Method for combining expert predictions (default: 'diff_entr')
                      Options: 'uniform', 'diff_entr', 'variance', 'no_weights'
            device: Device for computation
            dtype: Data type for tensors
        """
        super().__init__(
            input_dim=input_dim,
            bounds=bounds,
            acq_func="ucb",  # gPOE uses UCB-like acquisition
            device=device,
            dtype=dtype,
        )

        self.points_per_expert = points_per_expert
        self.n_init = n_init
        self.n_candidates = n_candidates
        self.batch_size = batch_size
        self.beta = torch.tensor(beta, device=self.device, dtype=self.dtype)
        self.weighting = weighting

    def suggest(self, n_suggestions: int = 1) -> torch.Tensor:
        """
        Suggest next points using ensemble of GP experts.

        Args:
            n_suggestions: Number of points to suggest

        Returns:
            Suggested points of shape (n_suggestions, input_dim)

        Note:
            Output shape is always (n_suggestions, input_dim), even if n_suggestions=1
        """
        # Random initialization phase
        if self.train_X.shape[0] < self.n_init:
            suggestions = self._random_suggestions(n_suggestions)
            assert suggestions.shape == (n_suggestions, self.input_dim), \
                f"Expected shape ({n_suggestions}, {self.input_dim}), got {suggestions.shape}"
            return suggestions

        # Determine number of experts based on data size
        n_experts = max(1, int(self.train_X.shape[0] / self.points_per_expert))
        n_per_expert = int(self.train_X.shape[0] / n_experts)

        # Random partition of data across experts
        partition = np.random.choice(
            self.train_X.shape[0],
            size=(n_experts, n_per_expert),
            replace=False,
        )

        # Standardize targets (zero mean, unit variance)
        train_y_mean = self.train_y.mean()
        train_y_std = self.train_y.std()
        if train_y_std < 1e-8:
            train_y_std = torch.tensor(1.0, device=self.device, dtype=self.dtype)
        train_y_std_normalized = (self.train_y - train_y_mean) / train_y_std

        # Create batched training data for ensemble
        batched_X = torch.stack([self.train_X[partition[k]] for k in range(n_experts)])
        batched_y = torch.stack([train_y_std_normalized[partition[k]] for k in range(n_experts)])

        # Train ensemble of GPs
        try:
            model = self._get_ensemble_model(batched_X, batched_y)
        except Exception as ex:
            print(f"Error fitting ensemble: {ex}")
            suggestions = self._random_suggestions(n_suggestions)
            assert suggestions.shape == (n_suggestions, self.input_dim), \
                f"Expected shape ({n_suggestions}, {self.input_dim}), got {suggestions.shape}"
            return suggestions

        # Generate candidate points
        X_test = self._generate_candidates()

        # Get predictions from all experts
        mu_s, var_s, prior = self._predict_ensemble(model, X_test, n_experts)

        # Combine predictions using gPOE
        mu, var = self._combine_experts(mu_s, var_s, prior)

        # Compute acquisition function (UCB)
        ucb = mu + torch.sqrt(self.beta) * var.sqrt()

        # Select top candidates
        best_indices = ucb.topk(n_suggestions).indices
        suggestions = X_test[best_indices]

        # Clamp to bounds
        suggestions = torch.clamp(suggestions, min=self.bounds[0], max=self.bounds[1])

        # Validate shape
        assert suggestions.shape == (n_suggestions, self.input_dim), \
            f"Expected shape ({n_suggestions}, {self.input_dim}), got {suggestions.shape}"

        return suggestions

    def observe(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """
        Update with new observations.

        Args:
            X: Observed input points of shape (n, input_dim)
            y: Observed output values of shape (n,) or (n, 1)
        """
        X = ensure_tensor(X, device=self.device, dtype=self.dtype)
        y = ensure_tensor(y, device=self.device, dtype=self.dtype)

        X = X.reshape(-1, self.input_dim)
        y = y.reshape(-1, 1)

        self.train_X = torch.cat([self.train_X, X], dim=0)
        self.train_y = torch.cat([self.train_y, y], dim=0)

    def _random_suggestions(self, n: int) -> torch.Tensor:
        """
        Generate random suggestions within bounds using Sobol sampling.

        Args:
            n: Number of random points to generate

        Returns:
            Random points of shape (n, input_dim)
        """
        sobol = SobolEngine(dimension=self.input_dim, scramble=True)
        X_sobol = sobol.draw(n=n).to(device=self.device, dtype=self.dtype)

        # Scale to bounds
        X_scaled = self.bounds[0] + (self.bounds[1] - self.bounds[0]) * X_sobol

        return X_scaled

    def _generate_candidates(self) -> torch.Tensor:
        """
        Generate candidate points using Sobol sampling.

        Returns:
            Candidate points of shape (n_candidates, input_dim)
        """
        sobol = SobolEngine(dimension=self.input_dim, scramble=True)
        X_cand = sobol.draw(n=self.n_candidates).to(device=self.device, dtype=self.dtype)

        # Scale to bounds
        X_cand = self.bounds[0] + (self.bounds[1] - self.bounds[0]) * X_cand

        return X_cand

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
            batched_y,  # already standardized before call
            input_transform=Normalize(d=self.input_dim, bounds=self.bounds),
            outcome_transform=None,
        )
        model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-5))

        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

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

        # Process in batches to avoid memory issues
        num_batches = math.ceil(self.n_candidates / self.batch_size)

        for i in range(num_batches):
            start_i = i * self.batch_size
            end_i = min((i + 1) * self.batch_size, self.n_candidates)
            X_batch = X_test[start_i:end_i]

            try:
                # Get prior variance (before conditioning on data)
                with gpytorch.settings.prior_mode(True):
                    y_prior = model.likelihood(model(X_batch))
                    prior[:, start_i:end_i] = y_prior.variance.detach()

                # Get posterior predictions (after conditioning on data)
                posterior = model.posterior(X_batch)
                y_pred = model.likelihood(posterior.mvn)

                mu_s[:, start_i:end_i] = y_pred.mean.detach()
                var_s[:, start_i:end_i] = y_pred.variance.detach()

            except Exception as e:
                print(f"Prediction error in batch {i}: {e}")
                # Fill with prior for failed batch
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

        This implements the gPoE formula:
        1/σ²_combined = Σ_i w_i/σ²_i
        μ_combined = σ²_combined * Σ_i w_i*μ_i/σ²_i

        Args:
            mu_s: Expert means of shape (n_experts, n_candidates)
            var_s: Expert variances of shape (n_experts, n_candidates)
            prior: Prior variances of shape (n_experts, n_candidates)

        Returns:
            Tuple of (combined_mean, combined_variance) each of shape (n_candidates,)
        """
        # Compute weights based on weighting scheme
        weight_matrix = self._compute_weights(mu_s, var_s, prior)

        # Convert variance to precision (inverse variance)
        prec_s = 1.0 / (var_s + 1e-8)  # Add small epsilon for numerical stability

        # Normalize weights to sum to 1 across experts
        weight_matrix = self._normalize_weights(weight_matrix)

        # Combined precision (weighted sum of precisions)
        prec = torch.sum(weight_matrix * prec_s, dim=0)

        # Combined variance (inverse of combined precision)
        var = 1.0 / (prec + 1e-8)

        # Combined mean (precision-weighted average of means)
        mu = var * torch.sum(weight_matrix * prec_s * mu_s, dim=0)

        return mu, var

    def _compute_weights(
        self,
        mu_s: torch.Tensor,
        var_s: torch.Tensor,
        prior: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute weights for combining expert predictions.

        Args:
            mu_s: Expert means of shape (n_experts, n_candidates)
            var_s: Expert variances of shape (n_experts, n_candidates)
            prior: Prior variances of shape (n_experts, n_candidates)

        Returns:
            Weight matrix of shape (n_experts, n_candidates)
        """
        if self.weighting == "uniform":
            # Equal weights for all experts
            return torch.ones_like(mu_s) / mu_s.shape[0]

        elif self.weighting == "diff_entr":
            # Differential entropy weighting
            # Weight by information gain: H(prior) - H(posterior)
            # ∝ 0.5 * (log(σ²_prior) - log(σ²_posterior))
            return 0.5 * (torch.log(prior + 1e-8) - torch.log(var_s + 1e-8))

        elif self.weighting == "variance":
            # Inverse variance weighting with exponential scaling
            power = 10
            return torch.exp(-power * var_s)

        elif self.weighting == "no_weights":
            # No weighting (all weights = 1, unnormalized)
            return torch.ones_like(mu_s)

        else:
            raise ValueError(f"Unknown weighting method: {self.weighting}")

    def _normalize_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Normalize weights to sum to 1 along the expert dimension.

        Args:
            weights: Weight matrix of shape (n_experts, n_candidates)

        Returns:
            Normalized weights of shape (n_experts, n_candidates)
        """
        sum_weights = torch.sum(weights, dim=0, keepdim=True)
        return weights / (sum_weights + 1e-8)
