"""
Hashing-Enhanced Subspace Bayesian Optimization (HesBO).

HesBO uses a randomized hashing-based dimensionality reduction
to enable efficient BO in high dimensions.

Reference:
    Nayebi et al. "A Framework for Bayesian Optimization in Embedded Subspaces"
    ICML 2019.
"""

from typing import Optional
import torch
from gpytorch.module import Module
from gpytorch.means.mean import Mean
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.constraints import Interval

from .base import EmbeddingOptimizer
from .utils import DEVICE, DTYPE, clamp_to_bounds


class HesBO(EmbeddingOptimizer):
    """
    Hashing-Enhanced Subspace Bayesian Optimization.

    Uses a randomized hash function to project high-dimensional inputs
    into a low-dimensional subspace while preserving local structure.
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
        Initialize HesBO optimizer.

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
        # Ensure low_dim is valid
        low_dim = max(1, min(low_dim, input_dim))

        # Default kernel for low-dimensional space
        if covar_module is None:
            covar_module = ScaleKernel(
                MaternKernel(
                    ard_num_dims=low_dim,
                    lengthscale_constraint=Interval(0.005, 4.0),
                ),
                outputscale_constraint=Interval(0.05, 20.0),
            )

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

        # Initialize hash function parameters (Count Sketch)
        # h: maps each high dimension i to a low dimension index h[i] in [0, low_dim)
        # This implements the hash function h: [D] -> [d] from the paper
        self.h = torch.randint(0, self.low_dim, (input_dim,), device=self.device)

        # sigma: random signs {-1, +1}
        # This implements the sign function s: [D] -> {-1, +1} from the paper
        self.sigma = (
            (torch.randint(0, 2, (input_dim,), device=self.device) * 2 - 1)
            .to(dtype=self.dtype)
        )

    def high_to_low(self, X: torch.Tensor) -> torch.Tensor:
        """
        Project high-dimensional points to low-dimensional space using Count Sketch.

        Implements the projection: y[h[i]] += s[i] * x[i] for all i
        where h is the hash function and s is the sign function.

        Args:
            X: Points in high-dimensional space of shape (n, input_dim)

        Returns:
            Points in low-dimensional space of shape (n, low_dim)
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        X = X.to(device=self.device, dtype=self.dtype)
        batch_size = X.shape[0]

        # Initialize low-dimensional representation
        Y = torch.zeros((batch_size, self.low_dim), device=self.device, dtype=self.dtype)

        # Count Sketch projection: y[h[i]] += s[i] * x[i]
        # For each high dimension i, add its signed value to bucket h[i]
        for i in range(self.input_dim):
            bucket_idx = self.h[i]
            Y[:, bucket_idx] += X[:, i] * self.sigma[i]

        return Y

    def low_to_high(self, X: torch.Tensor) -> torch.Tensor:
        """
        Project low-dimensional points back to high-dimensional space.

        Implements the inverse projection: x[i] = s[i] * y[h[i]]
        where h is the hash function and s is the sign function.

        Args:
            X: Points in low-dimensional space of shape (n, low_dim)

        Returns:
            Points in high-dimensional space of shape (n, input_dim)
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        X = X.to(device=self.device, dtype=self.dtype)
        batch_size = X.shape[0]

        # Initialize high-dimensional representation
        Y = torch.zeros((batch_size, self.input_dim), device=self.device, dtype=self.dtype)

        # Inverse projection: x[i] = s[i] * y[h[i]]
        # For each high dimension i, gather from bucket h[i] and apply sign
        for i in range(self.input_dim):
            bucket_idx = self.h[i]
            Y[:, i] = X[:, bucket_idx] * self.sigma[i]

        # Clamp to bounds
        return clamp_to_bounds(Y, self.bounds)

    def reset(self) -> None:
        """
        Reset the optimizer and reinitialize hash function.
        """
        super().reset()

        # Reinitialize hash function and signs
        self.h = torch.randint(0, self.low_dim, (self.input_dim,), device=self.device)
        self.sigma = (
            (torch.randint(0, 2, (self.input_dim,), device=self.device) * 2 - 1)
            .to(dtype=self.dtype)
        )
