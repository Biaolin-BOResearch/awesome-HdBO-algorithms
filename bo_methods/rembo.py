"""
Random EMbedding Bayesian Optimization (REMBO).

REMBO uses a random linear projection to embed high-dimensional
problems into a low-dimensional space.

Reference:
    Wang et al. "Bayesian Optimization in High Dimensions via Random Embeddings"
    IJCAI 2013.
"""

from typing import Optional
import torch
from gpytorch.module import Module
from gpytorch.means.mean import Mean

from .base import EmbeddingOptimizer
from .utils import DEVICE, DTYPE, clamp_to_bounds


class REMBO(EmbeddingOptimizer):
    """
    Random Embedding Bayesian Optimization.

    Uses a random linear projection matrix to embed high-dimensional
    optimization problems into a lower-dimensional space.
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
        Initialize REMBO optimizer.

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

        # Initialize random embedding matrix
        # A: D x d matrix (where D = input_dim, d = low_dim)
        self.embedding_matrix = torch.randn(
            (self.input_dim, self.low_dim),
            device=self.device,
            dtype=self.dtype,
        )

        # Compute pseudo-inverse for projection
        self.embedding_matrix_pinv = torch.linalg.pinv(self.embedding_matrix)

    def high_to_low(self, X: torch.Tensor) -> torch.Tensor:
        """
        Project high-dimensional points to low-dimensional space.

        Uses the pseudo-inverse of the embedding matrix: z = A^+ x

        Args:
            X: Points in high-dimensional space of shape (n, input_dim)

        Returns:
            Points in low-dimensional space of shape (n, low_dim)
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        X = X.to(device=self.device, dtype=self.dtype)
        batch_size = X.shape[0]

        # Project: z = A^+ x (for each point in batch)
        result = torch.zeros((batch_size, self.low_dim), device=self.device, dtype=self.dtype)

        for i in range(batch_size):
            result[i] = self.embedding_matrix_pinv @ X[i]

        return result

    def low_to_high(self, X: torch.Tensor) -> torch.Tensor:
        """
        Project low-dimensional points back to high-dimensional space.

        Uses the embedding matrix: x = A z

        Args:
            X: Points in low-dimensional space of shape (n, low_dim)

        Returns:
            Points in high-dimensional space of shape (n, input_dim)
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        X = X.to(device=self.device, dtype=self.dtype)
        batch_size = X.shape[0]

        # Project back: x = A z
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

        # Reinitialize random embedding
        self.embedding_matrix = torch.randn(
            (self.input_dim, self.low_dim),
            device=self.device,
            dtype=self.dtype,
        )
        self.embedding_matrix_pinv = torch.linalg.pinv(self.embedding_matrix)
