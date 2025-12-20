"""
LassoBO - LASSO-based Bayesian Optimization for High-Dimensional Problems.

LassoBO uses L1 regularization on the GP lengthscales to identify important
variables, then performs BO on the selected subspace with a fixed background
for unimportant variables.

Reference:
    Song et al. "LassoBO: LASSO-based Bayesian Optimization for 
    High-Dimensional Black-Box Functions" 2023.
"""

from typing import Optional, List, Tuple
import torch
import numpy as np
from gpytorch.module import Module
from gpytorch.means.mean import Mean
from gpytorch.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from botorch.models import SingleTaskGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize

from .base import BaseOptimizer
from .utils import DEVICE, DTYPE, ensure_tensor, clamp_to_bounds


class LassoGP(ExactGP, GPyTorchModel):
    """GP model with L1 regularization for variable selection."""
    _num_outputs = 1

    def __init__(self, train_x, train_y, likelihood, ard_dims):
        super().__init__(train_x, train_y, likelihood)
        self.ard_dims = ard_dims
        self.mean_module = ConstantMean()
        base_kernel = RBFKernel(
            lengthscale_constraint=Interval(1e-3, 1e6),
            ard_num_dims=ard_dims
        )
        self.covar_module = ScaleKernel(
            base_kernel,
            outputscale_constraint=Interval(0.1, 10)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class BestKStrategy:
    """Strategy for filling unimportant variables using best-k values."""

    def __init__(self, dims, k=3):
        self.dims = dims
        self.k = k
        self.best_xs = []
        self.best_ys = []

    def init_strategy(self, xs, ys):
        self.best_xs = []
        self.best_ys = []
        for x, y in zip(xs, ys):
            self.update(x, y)

    def update(self, x, y):
        if len(self.best_xs) < self.k:
            self.best_xs.append(x)
            self.best_ys.append(y)
        else:
            min_y = min(self.best_ys)
            if y > min_y:
                idx = self.best_ys.index(min_y)
                self.best_xs[idx] = x
                self.best_ys[idx] = y

    def get_background(self, lb, ub, n=2):
        if len(self.best_xs) == 0:
            return np.random.rand(n, self.dims) * (ub - lb) + lb
        best_xs = np.array(self.best_xs)
        background = np.zeros((n, self.dims))
        for i in range(n):
            for dim in range(self.dims):
                background[i, dim] = np.clip(
                    np.random.choice(best_xs[:, dim]) + 0.05 * np.random.randn() * (ub[dim] - lb[dim]),
                    lb[dim], ub[dim]
                )
        return background


class LassoBO(BaseOptimizer):
    """
    LassoBO - LASSO-based Bayesian Optimization.
    
    Uses L1 regularization on GP lengthscales to identify important variables
    and performs BO on the selected subspace.
    """

    def __init__(
        self,
        input_dim: int,
        bounds: torch.Tensor,
        n_init_samples: int = 10,
        alpha: float = 0.9,
        l1_weight: float = 1e-4,
        n_background: int = 5,
        max_tol: int = 11,
        acq_func: str = "UCB",
        covar_module: Optional[Module] = None,
        mean_module: Optional[Mean] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize LassoBO.

        Args:
            input_dim: Dimension of the input space
            bounds: Bounds tensor of shape (2, input_dim)
            n_init_samples: Number of initial random samples
            alpha: Fraction of total importance to capture with selected dims
            l1_weight: Weight for L1 regularization
            n_background: Number of background samples
            max_tol: Maximum tolerance before retraining
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
        self.alpha = alpha
        self.l1_weight = l1_weight
        self.n_background = n_background
        self.max_tol = max_tol
        self.seed = seed

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.lb = bounds[0].cpu().numpy()
        self.ub = bounds[1].cpu().numpy()

        # State
        self.samples = []
        self.best_value = float('-inf')
        self.best_sample = None
        self.hypers = None
        self.is_train = True
        self.tol = 0
        self.rec_scores = []
        self.prior = np.array([])
        self.rd_background = None
        self.active_dims = list(range(input_dim))
        self.axis_scores = np.zeros(input_dim)
        self.lasso_model = None

        # Strategy for unimportant variables
        self.uipt_solver = BestKStrategy(dims=input_dim, k=3)

    def _latin_hypercube(self, n, dims):
        points = np.zeros((n, dims))
        centers = (1.0 + 2.0 * np.arange(0.0, n)) / float(2 * n)
        for i in range(dims):
            points[:, i] = centers[np.random.permutation(n)]
        perturbation = np.random.uniform(-1.0, 1.0, (n, dims)) / float(2 * n)
        points += perturbation
        return np.clip(points, 0, 1)

    def _from_unit_cube(self, points):
        return points * (self.ub - self.lb) + self.lb

    def _shuffle_data(self, X):
        X_new = np.zeros_like(X)
        for i in range(X.shape[1]):
            X_new[:, i] = np.random.permutation(X[:, i])
        return X_new

    def _compute_axis_scores(self, hypers=None, prior=None):
        """Compute importance scores using L1-regularized GP training."""
        if len(self.samples) < 3:
            return np.ones(self.input_dim), None, None
        
        if prior is None:
            prior = []

        X = np.array([s[0] for s in self.samples])
        y = np.array([s[1] for s in self.samples])

        train_x = torch.from_numpy(X).to(self.device, self.dtype)
        train_y = torch.from_numpy(y).to(self.device, self.dtype)

        # Normalize - ensure bounds on correct device
        bounds_tensor = torch.stack([
            torch.tensor(self.lb, device=self.device, dtype=self.dtype),
            torch.tensor(self.ub, device=self.device, dtype=self.dtype)
        ])
        train_x = normalize(train_x, bounds_tensor)
        # Standardize Y to mean=0, std=1 (use mean, not median)
        mu = train_y.mean()
        sigma = train_y.std()
        if sigma < 1e-6:
            sigma = torch.tensor(1.0, device=train_y.device, dtype=train_y.dtype)
        train_y = (train_y - mu) / sigma

        # Create GP with LASSO regularization
        noise_constraint = Interval(0, 5e-2)
        likelihood = GaussianLikelihood(noise_constraint=noise_constraint)
        likelihood = likelihood.to(train_x.device, train_x.dtype)

        model = LassoGP(train_x, train_y, likelihood, self.input_dim)
        model = model.to(train_x.device, train_x.dtype)
        model.train()
        likelihood.train()

        mll = ExactMarginalLogLikelihood(likelihood, model)

        # Initialize hyperparameters
        if hypers is None:
            model.covar_module.outputscale = 1.1
            model.covar_module.base_kernel.lengthscale = torch.ones(
                1, self.input_dim, device=self.device, dtype=self.dtype
            ) * 0.5
            model.likelihood.noise = 1e-2
        else:
            try:
                model.load_state_dict(hypers)
            except Exception:
                pass

        # Train with L1 regularization
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        best_loss = float('inf')
        best_hypers = None
        tol_count = 0

        for _ in range(250):
            try:
                optimizer.zero_grad()
                output = model(train_x)
                
                # L1 regularization on inverse lengthscales (promotes sparsity)
                ls = model.covar_module.base_kernel.lengthscale[0]
                l1_norm = sum(1.0 / (p ** 2 + 1e-6) for k, p in enumerate(ls) if k not in prior)
                
                loss = -mll(output, train_y) + self.l1_weight * l1_norm
                loss.backward()
                optimizer.step()

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_hypers = model.state_dict()
                    tol_count = 0
                else:
                    scheduler.step()
                    tol_count += 1
                    if tol_count > 33:
                        break
            except Exception:
                break

        model.eval()
        likelihood.eval()

        # Extract importance scores (inverse lengthscales)
        try:
            inv_ls = 1.0 / model.covar_module.base_kernel.lengthscale[0].cpu().detach().numpy()
        except Exception:
            inv_ls = np.ones(self.input_dim)

        self.lasso_model = model
        return inv_ls, best_hypers, model

    def _find_num_active_dims(self, scores, alpha=0.9):
        """Find number of active dimensions to capture alpha fraction of importance."""
        d = 0
        score = np.array(scores[-1]) if len(scores) > 0 else np.ones(self.input_dim)
        full_sum = np.sum(score)
        sorted_score = np.sort(score)[::-1]
        selected_sum = 0
        for s in sorted_score:
            selected_sum += s
            d += 1
            if selected_sum >= alpha * full_sum:
                break
        return np.clip(d, 1, int(self.input_dim * 0.6))

    def observe(self, X: torch.Tensor, y: torch.Tensor) -> None:
        X = ensure_tensor(X, device=self.device, dtype=self.dtype)
        y = ensure_tensor(y, device=self.device, dtype=self.dtype)
        X = X.reshape(-1, self.input_dim)
        y = y.reshape(-1)

        X_np = X.cpu().numpy()
        y_np = y.cpu().numpy()

        for i in range(len(X_np)):
            x, val = X_np[i], y_np[i]
            self.samples.append((x, val))
            self.uipt_solver.update(x, val)

            if val > self.best_value:
                self.best_value = val
                self.best_sample = x.copy()
                self.tol = 0
            else:
                self.tol += 1
                if self.tol > self.max_tol:
                    self.tol = 0
                    self.is_train = True

        self.train_X = torch.cat([self.train_X, X], dim=0)
        self.train_y = torch.cat([self.train_y, y.reshape(-1, 1)], dim=0)

    def suggest(self, n_suggestions: int = 1) -> torch.Tensor:
        candidates = []

        for _ in range(n_suggestions):
            if len(self.samples) < self.n_init_samples:
                # Random initialization
                x = self._latin_hypercube(1, self.input_dim)
                x = self._from_unit_cube(x)[0]
            else:
                # Compute importance scores
                if self.is_train:
                    self.axis_scores, self.hypers, _ = self._compute_axis_scores(None, [])
                    self.rec_scores = []
                    self.rd_background = self._latin_hypercube(self.n_background, self.input_dim)
                    self.is_train = False
                else:
                    self.axis_scores, self.hypers, _ = self._compute_axis_scores(self.hypers, [])
                    self.rd_background = self._shuffle_data(self.rd_background)

                self.rec_scores.append(self.axis_scores)

                # Find active dimensions
                d = self._find_num_active_dims(self.rec_scores, self.alpha)
                self.active_dims = np.argsort(-self.axis_scores)[:d].tolist()

                # Optimize on active dimensions
                x = self._optimize_on_active_dims()

            candidates.append(x)

        candidates = np.array(candidates)
        return torch.tensor(candidates, device=self.device, dtype=self.dtype)

    def _optimize_on_active_dims(self):
        """Optimize using BO on active dimensions with fixed background."""
        if len(self.samples) < 3 or self.lasso_model is None:
            x = self._latin_hypercube(1, self.input_dim)
            return self._from_unit_cube(x)[0]

        X = np.array([s[0] for s in self.samples])
        y = np.array([s[1] for s in self.samples])

        train_x = torch.from_numpy(X).to(self.device, self.dtype)
        train_y = torch.from_numpy(y).to(self.device, self.dtype)

        # Ensure bounds on correct device
        bounds_tensor = torch.stack([
            torch.tensor(self.lb, device=self.device, dtype=self.dtype),
            torch.tensor(self.ub, device=self.device, dtype=self.dtype)
        ])
        # Normalize X to unit cube
        train_x_norm = normalize(train_x, bounds_tensor)

        # Standardize y to zero-mean, unit-std for GP stability
        mu = train_y.mean()
        sigma = train_y.std()
        if sigma < 1e-6:
            sigma = torch.tensor(1.0, device=train_y.device, dtype=train_y.dtype)
        train_y_norm = ((train_y - mu) / sigma).reshape(-1, 1)

        # Get background samples
        best_background = self.uipt_solver.get_background(
            np.zeros(self.input_dim), np.ones(self.input_dim), n=2
        )
        background = np.vstack([self.rd_background, best_background])

        # Create GP model
        try:
            gp = SingleTaskGP(
                train_x_norm, train_y_norm,
                input_transform=None,
                outcome_transform=None,
            )
            beta = 0.75 + 0.75 * np.random.rand()
            UCB = UpperConfidenceBound(gp, beta=beta)

            std_bounds = torch.stack([
                torch.zeros(self.input_dim, device=self.device, dtype=self.dtype),
                torch.ones(self.input_dim, device=self.device, dtype=self.dtype)
            ])

            max_acq = float('-inf')
            new_x = None

            for i in range(min(len(background), 4)):
                inactive_dims = [j for j in range(self.input_dim) if j not in self.active_dims]
                fixed_features = {j: float(background[i, j]) for j in inactive_dims}

                try:
                    candidate, acq_value = optimize_acqf(
                        UCB, bounds=std_bounds, q=1,
                        fixed_features=fixed_features,
                        num_restarts=8, raw_samples=1000,
                    )
                    if acq_value.item() > max_acq:
                        new_x = unnormalize(candidate, bounds_tensor.to(candidate))[0]
                        max_acq = acq_value.item()
                except Exception:
                    continue

            if new_x is not None:
                return new_x.cpu().numpy()
        except Exception:
            pass

        # Fallback to random
        x = self._latin_hypercube(1, self.input_dim)
        return self._from_unit_cube(x)[0]

    def get_active_dimensions(self):
        return self.active_dims

    def get_importance_scores(self):
        return self.axis_scores

    def reset(self):
        super().reset()
        self.samples = []
        self.best_value = float('-inf')
        self.best_sample = None
        self.hypers = None
        self.is_train = True
        self.tol = 0
        self.rec_scores = []
        self.prior = np.array([])
        self.rd_background = None
        self.active_dims = list(range(self.input_dim))
        self.axis_scores = np.zeros(self.input_dim)
        self.lasso_model = None
        self.uipt_solver = BestKStrategy(dims=self.input_dim, k=3)
