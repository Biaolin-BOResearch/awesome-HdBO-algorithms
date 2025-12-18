"""
Adaptive Linear Embedding for Bayesian Optimization (ALEBO).

ALEBO uses a random linear embedding with a Mahalanobis kernel and
constrained acquisition optimization to avoid nonlinear distortions.

Reference:
    Letham et al. "Re-Examining Linear Embeddings for High-Dimensional
    Bayesian Optimization" NeurIPS 2020.
"""

from typing import Optional
import torch
from scipy.optimize import minimize
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import Kernel
from gpytorch.constraints import Positive
from gpytorch.module import Module
from gpytorch.means.mean import Mean

from .base import EmbeddingOptimizer
from .utils import DEVICE, DTYPE, ensure_tensor


class MahalanobisKernel(Kernel):
    """
    Mahalanobis distance kernel for ALEBO.

    k(y, y') = σ² exp(-(y-y')^T Γ (y-y'))

    where Γ = (T B^†)^T D (T B^†) and D = diag(1/(2*l_i²)).

    Since we work in the embedding space directly, we parameterize
    Γ directly as a symmetric positive definite matrix.
    """

    has_lengthscale = False

    def __init__(
        self,
        low_dim: int,
        B_pinv: torch.Tensor,
        **kwargs,
    ):
        """
        Initialize the Mahalanobis kernel.

        Args:
            low_dim: Dimension of embedding space (d_e)
            B_pinv: Pseudo-inverse of embedding matrix B, shape (d_e, D)
        """
        super().__init__(**kwargs)
        self.low_dim = low_dim
        self.register_buffer("B_pinv", B_pinv)

        # Initialize lengthscales for the Mahalanobis metric
        # The number of parameters is d_e (diagonal elements)
        self.register_parameter(
            "raw_lengthscales",
            torch.nn.Parameter(torch.ones(low_dim))
        )
        self.register_constraint("raw_lengthscales", Positive())

    @property
    def lengthscales(self) -> torch.Tensor:
        return self.raw_lengthscales_constraint.transform(self.raw_lengthscales)

    def _compute_gamma(self) -> torch.Tensor:
        """
        Compute Γ matrix for Mahalanobis distance.

        Γ = (T B^†)^T D (T B^†)

        For simplicity, we assume T = I (identity) and compute:
        Γ = B^{†T} D B^†

        where D = diag(1/(2*l_i²))
        """
        # D = diag(1/(2*l_i²))
        D_diag = 1.0 / (2.0 * self.lengthscales ** 2)

        # For the full Mahalanobis kernel as in the paper:
        # Γ = B^{†T} D B^†
        # But in embedding space, we use a simpler diagonal approximation
        # Γ = D (diagonal in embedding space)
        return torch.diag(D_diag)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        **params,
    ) -> torch.Tensor:
        """
        Compute kernel values.

        Args:
            x1: First input tensor of shape (..., n1, d)
            x2: Second input tensor of shape (..., n2, d)
            diag: If True, return only diagonal elements

        Returns:
            Kernel matrix of shape (..., n1, n2) or (..., n1) if diag=True
        """
        Gamma = self._compute_gamma()

        if diag:
            # Compute diagonal: k(x1_i, x2_i)
            diff = x1 - x2  # (..., n, d)
            # Mahalanobis: diff @ Gamma @ diff^T for each point
            quad_form = (diff @ Gamma * diff).sum(dim=-1)
            return torch.exp(-quad_form)
        else:
            # Compute full kernel matrix
            # x1: (..., n1, d), x2: (..., n2, d)
            diff = x1.unsqueeze(-2) - x2.unsqueeze(-3)  # (..., n1, n2, d)
            # Mahalanobis distance: diff @ Gamma @ diff^T
            quad_form = (diff @ Gamma * diff).sum(dim=-1)  # (..., n1, n2)
            return torch.exp(-quad_form)


class ALEBO(EmbeddingOptimizer):
    """
    Adaptive Linear Embedding for Bayesian Optimization.

    Key features:
    1. Random projection matrix B with rows sampled from hypersphere S^{d-1}
    2. Mahalanobis kernel that adapts to the linear embedding
    3. Constrained acquisition optimization: -1 ≤ B^†y ≤ 1
    4. Rejection sampling for initial points
    """

    def __init__(
        self,
        input_dim: int,
        low_dim: int,
        bounds: torch.Tensor,
        acq_func: str = "EI",
        covar_module: Optional[Module] = None,
        mean_module: Optional[Mean] = None,
        n_init: int = 10,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize ALEBO optimizer.

        Args:
            input_dim: Dimension of the high-dimensional input space (D)
            low_dim: Dimension of the low-dimensional embedding space (d_e)
            bounds: Bounds tensor of shape (2, input_dim)
            acq_func: Acquisition function type
            covar_module: Optional custom covariance module (ignored, uses Mahalanobis)
            mean_module: Optional custom mean module
            n_init: Number of initial points for rejection sampling
            device: Device for computation
            dtype: Data type for tensors
        """
        # Don't pass covar_module to parent - we'll create our own
        super().__init__(
            input_dim=input_dim,
            low_dim=low_dim,
            bounds=bounds,
            acq_func=acq_func,
            covar_module=None,  # We use Mahalanobis kernel
            mean_module=mean_module,
            device=device,
            dtype=dtype,
        )

        self.n_init = n_init

        # Initialize embedding matrix B with rows from hypersphere S^{d-1}
        # B: D x d_e matrix where each row is on the unit hypersphere
        self.embedding_matrix = self._create_hypersphere_embedding()

        # Compute pseudo-inverse B^† for projection constraints
        self.embedding_matrix_pinv = torch.linalg.pinv(self.embedding_matrix)

        # Set embedding bounds to [-1, 1]^d_e as per paper
        self.low_dim_bounds = torch.stack([
            -torch.ones(self.low_dim, device=self.device, dtype=self.dtype),
            torch.ones(self.low_dim, device=self.device, dtype=self.dtype),
        ])

    def _create_hypersphere_embedding(self) -> torch.Tensor:
        """
        Create embedding matrix B by sampling D points from S^{d_e - 1}.

        Each row of B is uniformly sampled from the (d_e-1)-dimensional
        unit hypersphere in R^{d_e}.

        Returns:
            Embedding matrix of shape (D, d_e)
        """
        # Sample from standard normal
        B = torch.randn(
            (self.input_dim, self.low_dim),
            device=self.device,
            dtype=self.dtype
        )

        # Normalize each row to unit length (projects onto hypersphere)
        row_norms = torch.linalg.norm(B, dim=1, keepdim=True)
        B = B / row_norms

        return B

    def _generate_initial_points_rejection(self, n_points: int) -> torch.Tensor:
        """
        Generate initial points using rejection sampling.

        Sample points y in embedding space such that B^†y lies within
        the high-dimensional bounds (scaled to [-1, 1]).

        Args:
            n_points: Number of points to generate

        Returns:
            Points in embedding space of shape (n_points, d_e)
        """
        points = []
        max_attempts = n_points * 1000

        attempts = 0
        while len(points) < n_points and attempts < max_attempts:
            # Sample uniformly in [-1, 1]^d_e
            y = 2 * torch.rand(self.low_dim, device=self.device, dtype=self.dtype) - 1

            # Check if B^†y is within [-1, 1]^D (normalized bounds)
            x_projected = self.embedding_matrix_pinv.T @ y

            if torch.all(x_projected >= -1) and torch.all(x_projected <= 1):
                points.append(y)

            attempts += 1

        if len(points) < n_points:
            # If rejection sampling fails, fill with uniform samples
            # (this can happen if d_e is much smaller than D)
            remaining = n_points - len(points)
            for _ in range(remaining):
                y = 2 * torch.rand(self.low_dim, device=self.device, dtype=self.dtype) - 1
                points.append(y)

        return torch.stack(points)

    def high_to_low(self, X: torch.Tensor) -> torch.Tensor:
        """
        Project high-dimensional points to low-dimensional space.

        Uses the embedding matrix directly: y = B^T x
        (This projects onto the column space of B)

        Args:
            X: Points in high-dimensional space of shape (n, D)

        Returns:
            Points in low-dimensional space of shape (n, d_e)
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        X = X.to(device=self.device, dtype=self.dtype)

        # Normalize X to [-1, 1] based on bounds
        X_normalized = self._normalize_to_unit(X)

        # Project: y = B^T x
        # B: (D, d_e), X_normalized^T: (D, n)
        # Result: (d_e, n) -> transpose to (n, d_e)
        result = (self.embedding_matrix.T @ X_normalized.T).T

        return result

    def low_to_high(self, X: torch.Tensor) -> torch.Tensor:
        """
        Project low-dimensional points to high-dimensional space.

        Uses the embedding matrix: x = B y

        Args:
            X: Points in low-dimensional space of shape (n, d_e)

        Returns:
            Points in high-dimensional space of shape (n, D)
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        X = X.to(device=self.device, dtype=self.dtype)

        # Project: x = B y
        # B: (D, d_e), X^T: (d_e, n)
        # Result: (D, n) -> transpose to (n, D)
        result_normalized = (self.embedding_matrix @ X.T).T

        # Clamp to [-1, 1] first (the projection constraint)
        result_normalized = torch.clamp(result_normalized, -1, 1)

        # Denormalize to original bounds
        result = self._denormalize_from_unit(result_normalized)

        return result

    def _normalize_to_unit(self, X: torch.Tensor) -> torch.Tensor:
        """Normalize X from bounds to [-1, 1]."""
        lb, ub = self.bounds[0], self.bounds[1]
        return 2 * (X - lb) / (ub - lb) - 1

    def _denormalize_from_unit(self, X: torch.Tensor) -> torch.Tensor:
        """Denormalize X from [-1, 1] to bounds."""
        lb, ub = self.bounds[0], self.bounds[1]
        return lb + (X + 1) / 2 * (ub - lb)

    def observe(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """
        Update the optimizer with new observations.

        Args:
            X: Observed input points in high-dimensional space
            y: Observed output values
        """
        X = ensure_tensor(X, device=self.device, dtype=self.dtype)
        y = ensure_tensor(y, device=self.device, dtype=self.dtype)

        # Project to low-dimensional space
        X_low = self.high_to_low(X)

        # Update training data
        X_low = X_low.reshape(-1, self.low_dim)
        y = y.reshape(-1, 1)

        self.train_X_low = torch.cat([self.train_X_low, X_low], dim=0)
        self.train_y = torch.cat([self.train_y, y], dim=0)

        # Create Mahalanobis kernel
        mahalanobis_kernel = MahalanobisKernel(
            low_dim=self.low_dim,
            B_pinv=self.embedding_matrix_pinv,
        ).to(self.device)

        # Create and fit GP model with Mahalanobis kernel
        self.model = SingleTaskGP(
            self.train_X_low,
            self.train_y,
            covar_module=mahalanobis_kernel,
            mean_module=self.mean_module,
        ).to(self.device)

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)

    def _optimize_acquisition_constrained(
        self,
        n_suggestions: int = 1,
        num_restarts: int = 10,
        raw_samples: int = 256,
    ) -> torch.Tensor:
        """
        Optimize acquisition function with linear constraints.

        max α(y) subject to -1 ≤ B^†y ≤ 1

        Uses scipy.optimize.minimize with SLSQP.

        Args:
            n_suggestions: Number of points to suggest
            num_restarts: Number of random restarts
            raw_samples: Number of samples for initialization

        Returns:
            Optimal points in embedding space of shape (n_suggestions, d_e)
        """
        acq_func = self._get_acquisition_function()

        best_candidates = []

        for _ in range(n_suggestions):
            best_val = float('-inf')
            best_candidate = None

            # Generate initial points via rejection sampling
            init_points = self._generate_initial_points_rejection(num_restarts)

            for i in range(num_restarts):
                y0 = init_points[i].cpu().numpy()

                # Define constraint: -1 ≤ B^†y ≤ 1
                B_pinv_np = self.embedding_matrix_pinv.T.cpu().numpy()

                def constraint_lower(y):
                    return y @ B_pinv_np.T + 1  # Should be >= 0

                def constraint_upper(y):
                    return 1 - y @ B_pinv_np.T  # Should be >= 0

                constraints = [
                    {'type': 'ineq', 'fun': constraint_lower},
                    {'type': 'ineq', 'fun': constraint_upper},
                ]

                # Also constrain y to be in reasonable range
                bounds_scipy = [(-5, 5)] * self.low_dim

                # Objective: negative acquisition (we minimize)
                def objective(y):
                    y_tensor = torch.tensor(y, device=self.device, dtype=self.dtype).reshape(1, -1)
                    with torch.no_grad():
                        val = acq_func(y_tensor.unsqueeze(0))  # Add q dimension
                    return -val.item()

                try:
                    result = minimize(
                        objective,
                        y0,
                        method='SLSQP',
                        bounds=bounds_scipy,
                        constraints=constraints,
                        options={'maxiter': 100, 'ftol': 1e-6},
                    )

                    if result.success and -result.fun > best_val:
                        best_val = -result.fun
                        best_candidate = result.x
                except Exception:
                    continue

            if best_candidate is not None:
                best_candidates.append(
                    torch.tensor(best_candidate, device=self.device, dtype=self.dtype)
                )
            else:
                # Fallback: use random point
                best_candidates.append(init_points[0])

        return torch.stack(best_candidates)

    def suggest(self, n_suggestions: int = 1) -> torch.Tensor:
        """
        Suggest next points to evaluate in high-dimensional space.

        Uses constrained acquisition optimization as per ALEBO paper.

        Args:
            n_suggestions: Number of points to suggest

        Returns:
            Suggested points in high-dimensional space of shape (n_suggestions, D)
        """
        if self.model is None:
            # No observations yet - use rejection sampling for initial points
            init_low = self._generate_initial_points_rejection(n_suggestions)
            candidates_high = self.low_to_high(init_low)
        else:
            # Optimize acquisition with constraints
            candidates_low = self._optimize_acquisition_constrained(
                n_suggestions=n_suggestions
            )
            candidates_high = self.low_to_high(candidates_low)

        # Validate shape
        assert candidates_high.shape == (n_suggestions, self.input_dim), \
            f"Expected shape ({n_suggestions}, {self.input_dim}), got {candidates_high.shape}"

        return candidates_high

    def reset(self) -> None:
        """
        Reset the optimizer and reinitialize embedding matrix.
        """
        super().reset()

        # Reinitialize embedding with new hypersphere samples
        self.embedding_matrix = self._create_hypersphere_embedding()
        self.embedding_matrix_pinv = torch.linalg.pinv(self.embedding_matrix)
