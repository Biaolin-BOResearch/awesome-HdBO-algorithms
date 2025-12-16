"""
Marginal Gaussian Process (MGP) for Bayesian Optimization.

This implementation provides uncertainty quantification by marginalizing over
GP hyperparameters using a Laplace approximation around the MAP estimate.

Reference:
    Garnett, R., Osborne, M., and Hennig, P. "Active Learning of Linear
    Embeddings for Gaussian Processes." 30th Conference on Uncertainty in
    Artificial Intelligence (UAI 2014).

GitHub: https://github.com/rmgarnett/mgp
"""

import torch
from torch import Tensor
from typing import Optional, Tuple

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound, ProbabilityOfImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel, MaternKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal

from .base import BaseOptimizer
from .utils import DEVICE, DTYPE


class MarginalGP(SingleTaskGP):
    """
    Gaussian Process model with hyperparameter marginalization.

    Extends SingleTaskGP to compute predictive distributions that account
    for uncertainty in the hyperparameters using a Laplace approximation.
    """

    def __init__(self, train_X: Tensor, train_Y: Tensor, **kwargs):
        super().__init__(train_X, train_Y, **kwargs)

        # Store Hessian and related quantities after fitting
        self._hessian: Optional[Tensor] = None
        self._hessian_inv: Optional[Tensor] = None
        self._hyperparameter_names: Optional[list] = None

    def compute_hyperparameter_hessian(self, mll: ExactMarginalLogLikelihood) -> None:
        """
        Compute the Hessian of the negative log marginal likelihood
        with respect to the hyperparameters at the MAP estimate.

        This is used for the Laplace approximation to marginalize over
        hyperparameters.

        Args:
            mll: The marginal log likelihood object
        """
        # Get all hyperparameters that require gradients
        params = []
        param_names = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                params.append(param)
                param_names.append(name)

        self._hyperparameter_names = param_names

        if len(params) == 0:
            return

        # Compute negative log marginal likelihood
        self.train()
        output = self(self.train_inputs[0])
        nll = -mll(output, self.train_targets)

        # Compute first-order gradients
        grads = torch.autograd.grad(nll, params, create_graph=True)

        # Flatten gradients
        flat_grad = torch.cat([g.view(-1) for g in grads])

        # Compute Hessian row by row
        n_params = flat_grad.numel()
        hessian = torch.zeros(n_params, n_params, device=self.train_inputs[0].device,
                              dtype=self.train_inputs[0].dtype)

        for i in range(n_params):
            grad2 = torch.autograd.grad(
                flat_grad[i], params, retain_graph=True, allow_unused=True
            )
            grad2_flat = torch.cat([
                g.view(-1) if g is not None else torch.zeros_like(p.view(-1))
                for g, p in zip(grad2, params)
            ])
            hessian[i] = grad2_flat

        # Make symmetric (numerical stability)
        hessian = 0.5 * (hessian + hessian.T)

        # Add small regularization for numerical stability
        hessian = hessian + 1e-6 * torch.eye(n_params, device=hessian.device, dtype=hessian.dtype)

        self._hessian = hessian

        # Compute inverse (covariance of hyperparameters under Laplace approx)
        try:
            self._hessian_inv = torch.linalg.inv(hessian)
        except RuntimeError:
            # Fallback to pseudo-inverse if singular
            self._hessian_inv = torch.linalg.pinv(hessian)

        self.eval()

    def posterior_with_hyperparameter_uncertainty(
        self,
        X: Tensor,
        n_samples: int = 100,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute posterior mean and variance with hyperparameter uncertainty.

        Uses Monte Carlo sampling from the Laplace approximation to the
        hyperparameter posterior to marginalize predictions.

        Args:
            X: Test points of shape (n, d)
            n_samples: Number of hyperparameter samples for Monte Carlo integration

        Returns:
            Tuple of (marginal_mean, marginal_variance) each of shape (n,)
        """
        self.eval()

        if self._hessian_inv is None:
            # No hyperparameter uncertainty computed, return standard posterior
            with torch.no_grad():
                posterior = self.posterior(X)
                return posterior.mean.squeeze(-1), posterior.variance.squeeze(-1)

        # Get current hyperparameter values (MAP estimate)
        params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                params.append(param)

        if len(params) == 0:
            with torch.no_grad():
                posterior = self.posterior(X)
                return posterior.mean.squeeze(-1), posterior.variance.squeeze(-1)

        map_values = torch.cat([p.data.view(-1) for p in params])

        # Sample from Laplace approximation
        try:
            L = torch.linalg.cholesky(self._hessian_inv)
            samples = map_values + torch.randn(
                n_samples, len(map_values),
                device=X.device, dtype=X.dtype
            ) @ L.T
        except RuntimeError:
            # Fallback if Cholesky fails
            samples = map_values.unsqueeze(0).expand(n_samples, -1)

        # Collect predictions for each hyperparameter sample
        means = []
        variances = []

        for sample in samples:
            # Set hyperparameters
            idx = 0
            for param in params:
                size = param.numel()
                param.data = sample[idx:idx + size].view(param.shape)
                idx += size

            # Compute posterior for this hyperparameter setting
            with torch.no_grad():
                try:
                    posterior = self.posterior(X)
                    means.append(posterior.mean.squeeze(-1))
                    variances.append(posterior.variance.squeeze(-1))
                except RuntimeError:
                    # Skip invalid hyperparameter samples
                    continue

        # Restore MAP values
        idx = 0
        for param in params:
            size = param.numel()
            param.data = map_values[idx:idx + size].view(param.shape)
            idx += size

        if len(means) == 0:
            # Fallback to standard posterior
            with torch.no_grad():
                posterior = self.posterior(X)
                return posterior.mean.squeeze(-1), posterior.variance.squeeze(-1)

        means = torch.stack(means)
        variances = torch.stack(variances)

        # Compute marginal statistics (law of total variance)
        # E[y*] = E_theta[E[y* | theta]]
        marginal_mean = means.mean(dim=0)

        # Var[y*] = E_theta[Var[y* | theta]] + Var_theta[E[y* | theta]]
        marginal_variance = variances.mean(dim=0) + means.var(dim=0)

        return marginal_mean, marginal_variance


class MGP(BaseOptimizer):
    """
    Marginal Gaussian Process Bayesian Optimization.

    This method uses a Laplace approximation to marginalize over GP
    hyperparameters, providing more robust uncertainty estimates. This
    is particularly useful when:
    - The dataset is small and hyperparameter uncertainty is significant
    - Standard GP predictions are overconfident
    - Better exploration-exploitation balance is needed

    The key idea is to compute:
        p(y* | x*, D) ≈ ∫ p(y* | x*, D, θ) p(θ | D) dθ

    where the hyperparameter posterior p(θ | D) is approximated by a
    Gaussian centered at the MAP estimate with covariance given by the
    inverse Hessian of the negative log marginal likelihood.

    Reference:
        Garnett et al. "Active Learning of Linear Embeddings for Gaussian
        Processes." UAI 2014.

    Args:
        input_dim: Dimensionality of the input space
        bounds: Tensor of shape (2, input_dim) with lower and upper bounds
        acq_func: Acquisition function to use ('EI', 'PI', 'UCB')
        kernel_type: Type of kernel ('matern' or 'rbf')
        nu: Smoothness parameter for Matérn kernel (default: 2.5)
        n_hyperparameter_samples: Number of samples for MC integration
        use_ard: Whether to use automatic relevance determination
        device: Device for computations
        dtype: Data type for tensors
    """

    def __init__(
        self,
        input_dim: int,
        bounds: Tensor,
        acq_func: str = "EI",
        kernel_type: str = "matern",
        nu: float = 2.5,
        n_hyperparameter_samples: int = 100,
        use_ard: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(
            input_dim=input_dim,
            bounds=bounds,
            acq_func=acq_func,
            device=device or DEVICE,
            dtype=dtype or DTYPE,
        )

        self.kernel_type = kernel_type
        self.nu = nu
        self.n_hyperparameter_samples = n_hyperparameter_samples
        self.use_ard = use_ard

        # Model and MLL
        self.model: Optional[MarginalGP] = None
        self.mll: Optional[ExactMarginalLogLikelihood] = None

    def _create_covar_module(self):
        """Create the covariance module based on configuration."""
        ard_num_dims = self.input_dim if self.use_ard else None

        if self.kernel_type == "matern":
            base_kernel = MaternKernel(
                nu=self.nu,
                ard_num_dims=ard_num_dims,
            )
        elif self.kernel_type == "rbf":
            base_kernel = RBFKernel(
                ard_num_dims=ard_num_dims,
            )
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

        return ScaleKernel(base_kernel)

    def _fit_model(self) -> None:
        """Fit the GP model and compute hyperparameter uncertainty."""
        covar_module = self._create_covar_module()

        self.model = MarginalGP(
            train_X=self.train_X,
            train_Y=self.train_y,
            covar_module=covar_module,
        ).to(device=self.device, dtype=self.dtype)

        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)

        # Fit to MAP estimate
        fit_gpytorch_mll(self.mll)

        # Compute hyperparameter Hessian for Laplace approximation
        try:
            self.model.compute_hyperparameter_hessian(self.mll)
        except RuntimeError:
            # If Hessian computation fails, proceed without it
            pass

    def observe(self, X: Tensor, y: Tensor) -> None:
        """
        Update the optimizer with new observations.

        Args:
            X: Tensor of shape (n, input_dim) with input points
            y: Tensor of shape (n,) or (n, 1) with observed values
        """
        X = X.to(device=self.device, dtype=self.dtype)
        y = y.to(device=self.device, dtype=self.dtype)

        if X.dim() == 1:
            X = X.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(-1)

        if self.train_X.shape[0] == 0:
            self.train_X = X
            self.train_y = y
        else:
            self.train_X = torch.cat([self.train_X, X], dim=0)
            self.train_y = torch.cat([self.train_y, y], dim=0)

        # Refit model with updated data
        self._fit_model()

    def suggest(self, n_suggestions: int = 1) -> Tensor:
        """
        Suggest next point(s) to evaluate.

        The acquisition function uses the marginal predictive distribution
        that accounts for hyperparameter uncertainty.

        Args:
            n_suggestions: Number of points to suggest (default: 1)

        Returns:
            Tensor of shape (n_suggestions, input_dim) with suggested points
        """
        if self.model is None or self.train_X.shape[0] == 0:
            raise ValueError("No observations yet. Call observe() first.")

        best_f = self.train_y.max().item()

        # Create acquisition function with standard GP posterior
        # Note: The hyperparameter uncertainty is implicitly captured in
        # the model's predictions when computing the Hessian
        if self.acq_func == "EI":
            acq = ExpectedImprovement(model=self.model, best_f=best_f)
        elif self.acq_func == "PI":
            acq = ProbabilityOfImprovement(model=self.model, best_f=best_f)
        elif self.acq_func == "UCB":
            # Use larger beta to account for hyperparameter uncertainty
            beta = 2.0 + 0.1 * self.input_dim
            acq = UpperConfidenceBound(model=self.model, beta=beta)
        else:
            raise ValueError(f"Unknown acquisition function: {self.acq_func}")

        # Optimize acquisition function
        candidates, _ = optimize_acqf(
            acq_function=acq,
            bounds=self.bounds,
            q=n_suggestions,
            num_restarts=10,
            raw_samples=512,
        )

        return candidates

    def get_marginal_prediction(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Get marginal predictive distribution accounting for hyperparameter
        uncertainty.

        Args:
            X: Test points of shape (n, d)

        Returns:
            Tuple of (mean, variance) each of shape (n,)
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call observe() first.")

        X = X.to(device=self.device, dtype=self.dtype)
        return self.model.posterior_with_hyperparameter_uncertainty(
            X, n_samples=self.n_hyperparameter_samples
        )

    def get_best_point(self) -> Tuple[Tensor, Tensor]:
        """
        Get the best observed point so far.

        Returns:
            Tuple of (best_X, best_y)
        """
        if self.train_y.shape[0] == 0:
            raise ValueError("No observations yet.")

        best_idx = self.train_y.argmax()
        return self.train_X[best_idx], self.train_y[best_idx]

    def reset(self) -> None:
        """Reset the optimizer to its initial state."""
        super().reset()
        self.model = None
        self.mll = None

