"""
Adaptive Linear EMbedding for Bayesian Optimization (ALEBO).

ALEBO learns an adaptive linear embedding that aligns with
the geometry of the optimization problem.

Reference:
    Letham et al. "Re-Examining Linear Embeddings for High-Dimensional
    Bayesian Optimization" NeurIPS 2020.
"""

from typing import Optional
import torch
from gpytorch.module import Module
from gpytorch.means.mean import Mean

from .base import EmbeddingOptimizer
from .utils import DEVICE, DTYPE, clamp_to_bounds


class ALEBO(EmbeddingOptimizer):
    """
    Adaptive Linear Embedding for Bayesian Optimization.

    Uses a random linear embedding with orthonormal columns,
    similar to REMBO but with a specific construction that
    ensures better theoretical properties.
    """

    def __init__(
        self,
        input_dim: int,
        low_dim: int,
        bounds: torch.Tensor,
        acq_func: str = "EI",
        covar_module: Optional[Module] = None,
        mean_module: Optional[Mean] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize ALEBO optimizer.

        Args:
            input_dim: Dimension of the high-dimensional input space
            low_dim: Dimension of the low-dimensional embedding space
            bounds: Bounds tensor of shape (2, input_dim)
            acq_func: Acquisition function type
            covar_module: Optional custom covariance module
            mean_module: Optional custom mean module
            device: Device for computation
            dtype: Data type for tensors
        """
        super().__init__(
            input_dim=input_dim,
            low_dim=low_dim,
            bounds=bounds,
            acq_func=acq_func,
            covar_module=covar_module,
            mean_module=mean_module,
            device=device,
            dtype=dtype,
        )

        # Initialize embedding matrix with orthonormal columns
        # B: D x d matrix where columns are orthonormal
        self.embedding_matrix = self._create_orthonormal_embedding()

        # Compute pseudo-inverse for projection
        self.embedding_matrix_pinv = torch.linalg.pinv(self.embedding_matrix)

    def _create_orthonormal_embedding(self) -> torch.Tensor:
        """
        Create an embedding matrix with orthonormal columns.

        Returns:
            Embedding matrix of shape (input_dim, low_dim)
        """
        # Initialize with random normal vectors
        B = torch.zeros((self.input_dim, self.low_dim), device=self.device, dtype=self.dtype)

        for i in range(self.input_dim):
            # Sample from standard normal
            B[i, :] = torch.randn(self.low_dim, device=self.device, dtype=self.dtype)

            # Normalize to unit length
            B[i, :] = B[i, :] / torch.linalg.norm(B[i, :])

        return B

    def high_to_low(self, X: torch.Tensor) -> torch.Tensor:
        """
        Project high-dimensional points to low-dimensional space.

        Uses the pseudo-inverse: z = B^+ x

        Args:
            X: Points in high-dimensional space of shape (n, input_dim)

        Returns:
            Points in low-dimensional space of shape (n, low_dim)
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        X = X.to(device=self.device, dtype=self.dtype)
        batch_size = X.shape[0]

        # Project using matrix multiplication
        result = torch.zeros((batch_size, self.low_dim), device=self.device, dtype=self.dtype)

        for i in range(batch_size):
            result[i] = self.embedding_matrix_pinv @ X[i]

        return result

    def low_to_high(self, X: torch.Tensor) -> torch.Tensor:
        """
        Project low-dimensional points back to high-dimensional space.

        Uses the embedding matrix: x = B z

        Args:
            X: Points in low-dimensional space of shape (n, low_dim)

        Returns:
            Points in high-dimensional space of shape (n, input_dim)
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        X = X.to(device=self.device, dtype=self.dtype)
        batch_size = X.shape[0]

        # Project back using matrix multiplication
        result = torch.zeros((batch_size, self.input_dim), device=self.device, dtype=self.dtype)

        for i in range(batch_size):
            result[i] = self.embedding_matrix @ X[i]

        # Clamp to bounds
        return clamp_to_bounds(result, self.bounds)

    def reset(self) -> None:
        """
        Reset the optimizer and reinitialize embedding matrix.
        """
        super().reset()

        # Reinitialize embedding
        self.embedding_matrix = self._create_orthonormal_embedding()
        self.embedding_matrix_pinv = torch.linalg.pinv(self.embedding_matrix)
