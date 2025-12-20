"""
Monte Carlo Tree Search with Variable Selection (MCTS-VS).

MCTS-VS uses Monte Carlo Tree Search to identify important variables
in high-dimensional optimization problems.

Reference:
    Song et al. "Monte Carlo Tree Search Based Variable Selection for
    High Dimensional Bayesian Optimization" NeurIPS 2022.
"""

from typing import Optional, List, Dict, Tuple
import torch
import numpy as np
from gpytorch.module import Module
from gpytorch.means.mean import Mean
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF as SklearnRBF, ConstantKernel
from scipy.stats import norm

from .base import BaseOptimizer
from .utils import DEVICE, DTYPE, ensure_tensor, clamp_to_bounds


class MCTSNode:
    """Node in the MCTS tree representing a subset of variables."""
    _counter = 0

    def __init__(self, parent, dims, active_dims_idx, min_num_variables=3, reset_id=False):
        if reset_id:
            MCTSNode._counter = 0
        self.id = MCTSNode._counter
        MCTSNode._counter += 1
        self.parent = parent
        self.dims = dims
        self.active_dims_idx = active_dims_idx
        self.min_num_variables = min_num_variables
        self.kids = []
        self.n = 0
        self.q = 0.0

    def is_leaf(self):
        return len(self.kids) == 0

    def get_uct(self, Cp):
        if self.n == 0:
            return float('inf')
        parent_n = self.parent.n if self.parent else self.n
        exploitation = self.q / self.n
        exploration = Cp * np.sqrt(2 * np.log(parent_n + 1) / (self.n + 1e-8))
        return exploitation + exploration

    def update(self, reward):
        self.n += 1
        self.q += reward


class BestKStrategy:
    """Strategy for filling unimportant variables using best-k observed values."""

    def __init__(self, dims, k=5):
        self.dims = dims
        self.k = k
        self.best_xs = []
        self.best_ys = []

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

    def get_full_variable(self, fixed_variables, lb, ub):
        new_x = np.zeros(self.dims)
        if len(self.best_xs) > 0:
            best_xs_arr = np.array(self.best_xs)
            for dim in range(self.dims):
                if dim in fixed_variables:
                    new_x[dim] = fixed_variables[dim]
                else:
                    base_val = np.random.choice(best_xs_arr[:, dim])
                    perturbation = 0.05 * np.random.randn() * (ub[dim] - lb[dim])
                    new_x[dim] = np.clip(base_val + perturbation, lb[dim], ub[dim])
        else:
            for dim in range(self.dims):
                if dim in fixed_variables:
                    new_x[dim] = fixed_variables[dim]
                else:
                    new_x[dim] = np.random.uniform(lb[dim], ub[dim])
        return new_x


class MCTSVS(BaseOptimizer):
    """
    Monte Carlo Tree Search with Variable Selection.
    Uses MCTS to identify important variables and performs BO in selected subspace.
    """

    def __init__(
        self,
        input_dim: int,
        bounds: torch.Tensor,
        n_init_samples: int = 10,
        Cp: float = 0.1,
        min_num_variables: int = 3,
        select_right_threshold: int = 5,
        k: int = 20,
        acq_func: str = "EI",
        covar_module: Optional[Module] = None,
        mean_module: Optional[Mean] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        seed: Optional[int] = None,
    ):
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
        self.Cp = Cp
        self.min_num_variables = min_num_variables
        self.select_right_threshold = select_right_threshold
        self.k = k
        self.seed = seed

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.lb = bounds[0].cpu().numpy()
        self.ub = bounds[1].cpu().numpy()

        self._init_tree()
        self.samples = []
        self.best_value = float('-inf')
        self.best_sample = None
        self.uipt_solver = BestKStrategy(dims=input_dim, k=k)
        self.num_select_right = 0
        self.active_dims_idx = list(range(input_dim))
        self.axis_scores = np.zeros(input_dim)

    def _init_tree(self):
        self.nodes = []
        self.root = MCTSNode(
            parent=None, dims=self.input_dim,
            active_dims_idx=list(range(self.input_dim)),
            min_num_variables=self.min_num_variables, reset_id=True,
        )
        self.nodes.append(self.root)

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

    def _compute_axis_scores(self):
        if len(self.samples) < 3:
            return np.ones(self.input_dim)
        X = np.array([s[0] for s in self.samples])
        y = np.array([s[1] for s in self.samples])
        X_norm = (X - self.lb) / (self.ub - self.lb + 1e-8)
        y_norm = (y - y.mean()) / (y.std() + 1e-6)
        try:
            kernel = ConstantKernel(1.0) * SklearnRBF(
                length_scale=np.ones(self.input_dim), length_scale_bounds=(1e-3, 1e3))
            gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-2, n_restarts_optimizer=3)
            gpr.fit(X_norm, y_norm)
            ls = gpr.kernel_.k2.length_scale
            if np.isscalar(ls):
                ls = np.ones(self.input_dim) * ls
            scores = 1.0 / (ls + 1e-6)
            scores = scores / (scores.max() + 1e-8)
        except Exception:
            scores = np.ones(self.input_dim)
        return scores

    def _select_node(self):
        current = self.root
        path = []
        while not current.is_leaf():
            uct_values = [kid.get_uct(self.Cp) for kid in current.kids]
            choice = np.random.choice(np.argwhere(uct_values == np.max(uct_values)).flatten())
            path.append((current, choice))
            current = current.kids[choice]
            self.num_select_right += choice
        return current, path

    def _split_node(self, node):
        if len(node.active_dims_idx) <= self.min_num_variables:
            return None, None
        active_scores = self.axis_scores[node.active_dims_idx]
        if active_scores.sum() < 1e-8:
            return None, None
        sorted_idx = np.argsort(active_scores)[::-1]
        split_point = max(len(sorted_idx) // 2, self.min_num_variables)
        left_dims = [node.active_dims_idx[i] for i in sorted_idx[:split_point]]
        left_child = MCTSNode(parent=node, dims=self.input_dim,
            active_dims_idx=left_dims, min_num_variables=self.min_num_variables)
        right_dims = [node.active_dims_idx[i] for i in sorted_idx[split_point:]]
        if len(right_dims) < self.min_num_variables:
            right_dims = [node.active_dims_idx[i] for i in sorted_idx[-self.min_num_variables:]]
        right_child = MCTSNode(parent=node, dims=self.input_dim,
            active_dims_idx=right_dims, min_num_variables=self.min_num_variables)
        node.kids = [left_child, right_child]
        return left_child, right_child

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
            if val > self.best_value:
                self.best_value = val
                self.best_sample = x.copy()
            self.uipt_solver.update(x, val)
        self.train_X = torch.cat([self.train_X, X], dim=0)
        self.train_y = torch.cat([self.train_y, y.reshape(-1, 1)], dim=0)
        self.axis_scores = self._compute_axis_scores()
        if self.num_select_right >= self.select_right_threshold:
            self._init_tree()
            self.num_select_right = 0

    def suggest(self, n_suggestions: int = 1) -> torch.Tensor:
        candidates = []
        for _ in range(n_suggestions):
            if len(self.samples) < self.n_init_samples:
                x = self._latin_hypercube(1, self.input_dim)
                x = self._from_unit_cube(x)[0]
            else:
                leaf, path = self._select_node()
                self.active_dims_idx = leaf.active_dims_idx
                x = self._optimize_on_active_dims(leaf.active_dims_idx)
                left, right = self._split_node(leaf)
                if left is not None and right is not None:
                    self.nodes.append(left)
                    self.nodes.append(right)
            candidates.append(x)
        candidates = np.array(candidates)
        return torch.tensor(candidates, device=self.device, dtype=self.dtype)

    def _optimize_on_active_dims(self, active_dims):
        if len(self.samples) < 3:
            x = self._latin_hypercube(1, self.input_dim)
            return self._from_unit_cube(x)[0]
        X = np.array([s[0] for s in self.samples])
        y = np.array([s[1] for s in self.samples])
        X_active = X[:, active_dims]
        lb_active = self.lb[active_dims]
        ub_active = self.ub[active_dims]
        X_active_norm = (X_active - lb_active) / (ub_active - lb_active + 1e-8)
        y_norm = (y - y.mean()) / (y.std() + 1e-6)
        try:
            kernel = ConstantKernel(1.0) * SklearnRBF(length_scale=1.0)
            gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-2, n_restarts_optimizer=2)
            gpr.fit(X_active_norm, y_norm)
            n_cand = 1000
            cands = np.random.rand(n_cand, len(active_dims))
            mu, sigma = gpr.predict(cands, return_std=True)
            best_y = y_norm.max()
            with np.errstate(divide='ignore', invalid='ignore'):
                imp = mu - best_y
                Z = imp / (sigma + 1e-8)
                ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
                ei[sigma < 1e-8] = 0.0
            best_idx = np.argmax(ei)
            best_active = cands[best_idx]
            best_active = best_active * (ub_active - lb_active) + lb_active
        except Exception:
            best_active = np.random.rand(len(active_dims))
            best_active = best_active * (ub_active - lb_active) + lb_active
        fixed_vars = {dim: val for dim, val in zip(active_dims, best_active)}
        full_x = self.uipt_solver.get_full_variable(fixed_vars, self.lb, self.ub)
        return full_x

    def get_active_dimensions(self):
        return self.active_dims_idx

    def get_importance_scores(self):
        return self.axis_scores

    def reset(self):
        super().reset()
        self._init_tree()
        self.samples = []
        self.best_value = float('-inf')
        self.best_sample = None
        self.uipt_solver = BestKStrategy(dims=self.input_dim, k=self.k)
        self.num_select_right = 0
        self.active_dims_idx = list(range(self.input_dim))
        self.axis_scores = np.zeros(self.input_dim)
