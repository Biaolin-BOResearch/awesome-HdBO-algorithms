"""
HDS: Hierarchical Diagonal Sampling for Bayesian Optimization.

Reference:
    Chen, B., Castro, R. M., & Krause, A. (2012). "Joint Optimization and
    Variable Selection of High-Dimensional Gaussian Processes". ICML 2012.

HDS identifies active variables through a binary tree of dimension subsets
and fits GPs in lower-dimensional projections corresponding to active subspaces.
"""

import torch
from typing import Optional, Tuple, List, Set
from torch import Tensor
import numpy as np

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

from .base import BaseOptimizer


class HDS(BaseOptimizer):
    """
    Hierarchical Diagonal Sampling (HDS).

    Uses a binary tree structure to hierarchically partition dimensions and
    identify active subspaces. Fits separate GPs for different subsets and
    performs joint optimization and variable selection.

    Key Features:
    - Binary tree of dimension subsets
    - Hierarchical variable selection
    - Lower-dimensional GP projections
    - Adaptive active set identification

    Args:
        input_dim: Dimensionality of input space
        bounds: Tensor of shape (2, input_dim) with bounds
        max_active_dims: Maximum number of active dimensions (default: sqrt(d))
        tree_depth: Depth of binary tree for partitioning (default: auto)
        selection_threshold: Threshold for variable activity (default: 0.1)
        n_candidates_per_node: Candidates to sample per tree node (default: 5)
        device: Device for computations
        dtype: Data type for tensors
    """

    def __init__(
        self,
        input_dim: int,
        bounds: Tensor,
        max_active_dims: Optional[int] = None,
        tree_depth: Optional[int] = None,
        selection_threshold: float = 0.1,
        n_candidates_per_node: int = 5,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__(input_dim, bounds, device, dtype)

        # Default max active dims: sqrt(d)
        if max_active_dims is None:
            max_active_dims = max(2, int(np.sqrt(input_dim)))
        self.max_active_dims = min(max_active_dims, input_dim)

        # Default tree depth: log2(d)
        if tree_depth is None:
            tree_depth = max(1, int(np.log2(input_dim)))
        self.tree_depth = tree_depth

        self.selection_threshold = selection_threshold
        self.n_candidates_per_node = n_candidates_per_node

        # Build binary tree of dimension subsets
        self.tree_nodes = self._build_dimension_tree()

        # Track active dimensions
        self.active_dims: Set[int] = set(range(input_dim))

        # GP models for each tree node (lazy initialization)
        self.node_models = {}

    def _build_dimension_tree(self) -> List[Set[int]]:
        """
        Build binary tree of dimension subsets.

        Returns:
            List of dimension sets, one per tree node
        """
        nodes = []
        all_dims = set(range(self.input_dim))

        # Root node contains all dimensions
        nodes.append(all_dims)

        # Build tree level by level
        for depth in range(self.tree_depth):
            # Get nodes at current level
            level_start = 2**depth - 1
            level_end = 2**(depth + 1) - 1

            if level_start >= len(nodes):
                break

            # Split each node at this level
            for node_idx in range(level_start, min(level_end, len(nodes))):
                parent_dims = list(nodes[node_idx])

                if len(parent_dims) <= 1:
                    # Can't split further
                    continue

                # Split roughly in half
                mid = len(parent_dims) // 2
                left_dims = set(parent_dims[:mid])
                right_dims = set(parent_dims[mid:])

                nodes.append(left_dims)
                nodes.append(right_dims)

        return nodes

    def _get_node_importance(self, node_dims: Set[int]) -> float:
        """
        Estimate importance of a dimension subset using variance.

        Args:
            node_dims: Set of dimension indices

        Returns:
            Importance score
        """
        if self.X is None or len(self.X) < 3:
            return 1.0

        # Get data projected onto this subset
        dim_list = sorted(list(node_dims))
        X_proj = self.X[:, dim_list]

        # Compute variance in outputs when varying these dimensions
        # versus when fixing them

        # Simple heuristic: variance of X in these dimensions
        X_var = X_proj.var(dim=0).mean()

        # Correlation with objective
        if len(self.y) > 1:
            y_centered = self.y - self.y.mean()
            correlations = []
            for dim in dim_list:
                x_centered = self.X[:, dim] - self.X[:, dim].mean()
                corr = torch.abs((x_centered * y_centered.squeeze()).mean())
                correlations.append(corr)

            if correlations:
                avg_corr = torch.stack(correlations).mean()
                importance = 0.5 * X_var + 0.5 * avg_corr
                return importance.item()

        return X_var.item()

    def _select_active_dimensions(self):
        """
        Select active dimensions using hierarchical tree structure.

        Walks down the tree and selects important subsets.
        """
        if self.X is None or len(self.X) < 3:
            # Not enough data, keep all dimensions
            return

        # Evaluate importance of each tree node
        node_importance = {}
        for i, node_dims in enumerate(self.tree_nodes):
            if len(node_dims) > 0:
                node_importance[i] = self._get_node_importance(node_dims)
            else:
                node_importance[i] = 0.0

        # Select top nodes by importance
        sorted_nodes = sorted(
            node_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Collect dimensions from important nodes
        selected_dims = set()
        for node_idx, importance in sorted_nodes:
            if importance < self.selection_threshold:
                break

            node_dims = self.tree_nodes[node_idx]

            # Add these dimensions if we haven't exceeded max
            if len(selected_dims) + len(node_dims) <= self.max_active_dims:
                selected_dims.update(node_dims)
            elif len(selected_dims) < self.max_active_dims:
                # Add dimensions one by one until we hit max
                remaining = self.max_active_dims - len(selected_dims)
                selected_dims.update(list(node_dims)[:remaining])
                break

            if len(selected_dims) >= self.max_active_dims:
                break

        # Ensure we have at least some dimensions
        if len(selected_dims) == 0:
            # Take top dimensions by individual importance
            dim_importance = []
            for d in range(self.input_dim):
                imp = self._get_node_importance({d})
                dim_importance.append((d, imp))

            dim_importance.sort(key=lambda x: x[1], reverse=True)
            selected_dims = set([d for d, _ in dim_importance[:self.max_active_dims]])

        self.active_dims = selected_dims

    def _fit_node_model(self, node_dims: Set[int]) -> SingleTaskGP:
        """
        Fit GP model for a specific dimension subset.

        Args:
            node_dims: Set of dimensions for this node

        Returns:
            Fitted GP model
        """
        dim_list = sorted(list(node_dims))
        X_proj = self.X[:, dim_list]

        model = SingleTaskGP(
            train_X=X_proj,
            train_Y=self.y
        ).to(device=self.device, dtype=self.dtype)

        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        return model

    def _optimize_on_subspace(
        self,
        active_dims: Set[int],
        center: Tensor,
        n_candidates: int
    ) -> Tensor:
        """
        Optimize acquisition function on active subspace.

        Args:
            active_dims: Active dimension indices
            center: Center point for inactive dimensions
            n_candidates: Number of candidates to generate

        Returns:
            Candidates in full space
        """
        dim_list = sorted(list(active_dims))

        # Fit model on active subspace
        model = self._fit_node_model(active_dims)

        # Create bounds for active subspace
        active_bounds = torch.stack([
            self.bounds[0, dim_list],
            self.bounds[1, dim_list]
        ])

        # Optimize acquisition function
        acq_func = ExpectedImprovement(
            model=model,
            best_f=self.y.max().item()
        )

        candidates_proj, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=active_bounds,
            q=n_candidates,
            num_restarts=5,
            raw_samples=256,
        )

        # Embed back to full space
        candidates_full = center.unsqueeze(0).repeat(n_candidates, 1)
        candidates_full[:, dim_list] = candidates_proj

        return candidates_full

    def suggest(self, n_suggestions: int = 1) -> Tensor:
        """
        Suggest next point(s) to evaluate.

        Args:
            n_suggestions: Number of points to suggest

        Returns:
            Tensor of shape (n_suggestions, input_dim)
        """
        if self.X is None or len(self.X) == 0:
            raise ValueError("No observations yet. Call observe() first.")

        # Select active dimensions using tree structure
        self._select_active_dimensions()

        print(f"HDS: Selected {len(self.active_dims)}/{self.input_dim} active dimensions")
        print(f"     Active dims: {sorted(list(self.active_dims))[:10]}...")

        # Get current best point as center
        best_idx = self.y.argmax()
        center = self.X[best_idx]

        # Generate candidates from important tree nodes
        all_candidates = []

        # Optimize on the full active set
        candidates = self._optimize_on_subspace(
            self.active_dims,
            center,
            n_suggestions
        )
        all_candidates.append(candidates)

        # Also try smaller subspaces from tree nodes
        important_nodes = []
        for i, node_dims in enumerate(self.tree_nodes):
            if len(node_dims) > 0 and node_dims.issubset(self.active_dims):
                importance = self._get_node_importance(node_dims)
                important_nodes.append((i, node_dims, importance))

        important_nodes.sort(key=lambda x: x[2], reverse=True)

        # Sample from top few nodes
        for i, node_dims, _ in important_nodes[:3]:
            if len(node_dims) >= 2:
                try:
                    node_candidates = self._optimize_on_subspace(
                        node_dims,
                        center,
                        self.n_candidates_per_node
                    )
                    all_candidates.append(node_candidates)
                except:
                    pass  # Skip if optimization fails

        # Combine all candidates
        all_candidates = torch.cat(all_candidates, dim=0)

        # Fit full model to rank candidates
        full_model = SingleTaskGP(
            train_X=self.X,
            train_Y=self.y
        ).to(device=self.device, dtype=self.dtype)

        mll = ExactMarginalLogLikelihood(full_model.likelihood, full_model)
        fit_gpytorch_mll(mll)

        acq_func = ExpectedImprovement(
            model=full_model,
            best_f=self.y.max().item()
        )

        # Rank by acquisition value
        with torch.no_grad():
            acq_values = acq_func(all_candidates.unsqueeze(1))

        # Select top candidates
        top_indices = acq_values.topk(min(n_suggestions, len(all_candidates))).indices
        return all_candidates[top_indices.squeeze()]

    def observe(self, X: Tensor, y: Tensor):
        """
        Update optimizer with new observations.

        Args:
            X: Tensor of shape (n, input_dim)
            y: Tensor of shape (n, 1) or (n,)
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
        """Get best observed point."""
        if self.y is None or len(self.y) == 0:
            raise ValueError("No observations yet.")

        best_idx = self.y.argmax()
        return self.X[best_idx], self.y[best_idx]
