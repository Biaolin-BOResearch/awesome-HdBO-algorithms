"""
Utility functions and common configurations for Bayesian Optimization methods.
"""

import torch
from torch.quasirandom import SobolEngine
from typing import Optional, Tuple


# Device and dtype configuration
def get_device() -> torch.device:
    """Get the default device for computations."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dtype() -> torch.dtype:
    """Get the default dtype for computations."""
    return torch.double


# Global device and dtype
DEVICE = get_device()
DTYPE = get_dtype()


def get_initial_points(
    dim: int,
    n_pts: int,
    bounds: torch.Tensor,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Generate initial points uniformly within the given bounds.

    Args:
        dim: Dimensionality of the problem
        n_pts: Number of initial points to generate
        bounds: Tensor of shape (2, dim) with lower and upper bounds
        seed: Random seed for reproducibility
        device: Device to use for computation
        dtype: Data type for the tensor

    Returns:
        Tensor of shape (n_pts, dim) with initial points
    """
    if device is None:
        device = DEVICE
    if dtype is None:
        dtype = DTYPE

    if seed is not None:
        torch.manual_seed(seed)

    # Generate random points within bounds
    X_init = torch.zeros(n_pts, dim, device=device, dtype=dtype)
    for i in range(dim):
        low, high = bounds[0][i], bounds[1][i]
        X_init[:, i] = torch.rand(n_pts, device=device, dtype=dtype) * (high - low) + low

    return X_init


def generate_sobol_samples(
    dim: int,
    n_samples: int,
    bounds: Optional[torch.Tensor] = None,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Generate Sobol quasi-random samples.

    Args:
        dim: Dimensionality of the problem
        n_samples: Number of samples to generate
        bounds: Optional bounds tensor of shape (2, dim). If None, returns samples in [0,1]^dim
        seed: Random seed for reproducibility
        device: Device to use for computation
        dtype: Data type for the tensor

    Returns:
        Tensor of shape (n_samples, dim) with Sobol samples
    """
    if device is None:
        device = DEVICE
    if dtype is None:
        dtype = DTYPE

    sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
    X = sobol.draw(n=n_samples).to(dtype=dtype, device=device)

    if bounds is not None:
        # Scale to bounds
        X = bounds[0] + (bounds[1] - bounds[0]) * X

    return X


def normalize_bounds(
    X: torch.Tensor, bounds: torch.Tensor, reverse: bool = False
) -> torch.Tensor:
    """
    Normalize points to [0, 1]^d or denormalize from [0, 1]^d.

    Args:
        X: Points to normalize/denormalize
        bounds: Bounds tensor of shape (2, dim)
        reverse: If True, denormalize from [0, 1] to original bounds

    Returns:
        Normalized or denormalized points
    """
    if reverse:
        # Denormalize: [0, 1] -> [lb, ub]
        return bounds[0] + (bounds[1] - bounds[0]) * X
    else:
        # Normalize: [lb, ub] -> [0, 1]
        return (X - bounds[0]) / (bounds[1] - bounds[0])


def ensure_tensor(
    X,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Ensure input is a PyTorch tensor with specified device and dtype.

    Args:
        X: Input data (tensor, numpy array, or list)
        device: Target device
        dtype: Target dtype

    Returns:
        PyTorch tensor
    """
    if device is None:
        device = DEVICE
    if dtype is None:
        dtype = DTYPE

    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X)

    return X.to(device=device, dtype=dtype)


def clamp_to_bounds(X: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
    """
    Clamp points to stay within bounds.

    Args:
        X: Points to clamp
        bounds: Bounds tensor of shape (2, dim)

    Returns:
        Clamped points
    """
    return torch.clamp(X, min=bounds[0], max=bounds[1])
