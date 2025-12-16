# Bayesian Optimization Methods Collection

A comprehensive, well-structured collection of state-of-the-art Bayesian optimization algorithms implemented in PyTorch with BoTorch and GPyTorch.

## Features

- **Clean Architecture**: Well-organized codebase following best practices with abstract base classes and consistent interfaces
- **Multiple Algorithms**: Includes StandardBO, TuRBO, HesBO, ALEBO, REMBO, gPOE-BO, and gPOE-TuRBO
- **GPU Support**: All methods support CUDA acceleration
- **Type Hints**: Full type annotations for better code quality
- **Shape Guarantees**: All methods return tensors of shape `(n_suggestions, input_dim)` - no surprises!
- **Comprehensive Documentation**: Detailed docstrings and usage examples

## Installation

### From Source

```bash
git clone https://github.com/yourusername/bayesian-optimization-collection.git
cd bayesian-optimization-collection
pip install -e .
```

### Dependencies

- Python >= 3.8
- PyTorch >= 2.0.0
- GPyTorch >= 1.11.0
- BoTorch >= 0.9.0
- NumPy >= 1.24.0

## Available Methods

### 1. StandardBO
Classic Bayesian Optimization using Gaussian Process regression with common acquisition functions (EI, PI, UCB).

**Best for**: Low to medium-dimensional problems (d < 20), well-understood problems

### 2. TuRBO (Trust Region Bayesian Optimization)
Uses multiple trust regions that dynamically expand/shrink based on optimization progress.

**Best for**: High-dimensional problems (d = 50-1000), local optimization

**Reference**: Eriksson et al. "Scalable Global Optimization via Local Bayesian Optimization" (NeurIPS 2019)

### 3. HesBO (Hashing-Enhanced Subspace BO)
Uses randomized hashing for dimensionality reduction while preserving local structure.

**Best for**: High-dimensional problems with low effective dimensionality

**Reference**: Nayebi et al. "A Framework for Bayesian Optimization in Embedded Subspaces" (ICML 2019)

### 4. ALEBO (Adaptive Linear Embedding BO)
Learns adaptive linear embeddings aligned with the problem geometry.

**Best for**: High-dimensional problems with linear structure

**Reference**: Letham et al. "Re-Examining Linear Embeddings for High-Dimensional Bayesian Optimization" (NeurIPS 2020)

### 5. REMBO (Random Embedding BO)
Uses random linear projections to embed high-dimensional problems into low-dimensional spaces.

**Best for**: High-dimensional problems with intrinsic low dimensionality

**Reference**: Wang et al. "Bayesian Optimization in High Dimensions via Random Embeddings" (IJCAI 2013)

### 6. gPOE-BO (Generalized Product of Experts BO)
Maintains ensemble of GP experts combined via generalized product of experts framework.

**Best for**: Large-scale problems requiring distributed computation, complex landscapes

### 7. gPOE-TuRBO
Combines gPOE ensemble approach with TuRBO's trust region framework.

**Best for**: Very high-dimensional problems (d > 100), combining benefits of ensemble and trust regions

---

## New Methods from HDBO Benchmark

The following methods are from the HDBO (High-Dimensional Bayesian Optimization) benchmark survey's Figure 1 timeline:

### 8. VanillaBO
Simple Bayesian optimization with dimensionality-dependent priors on lengthscales. Recent research shows this can perform competitively against elaborate high-dimensional methods.

**Best for**: High-dimensional problems where simpler methods may suffice, baseline comparisons

**Reference**: Hvarfner et al. "Vanilla Bayesian Optimization Performs Great in High Dimensions" (ICML 2024)

### 9. RandomLineBO
Optimizes along random lines in the high-dimensional space, reducing to 1D optimization problems.

**Best for**: Exploratory optimization, problems with directional structure

### 10. BAxUS (Bayesian optimization with Adaptively Expanding Subspaces)
Uses nested count-sketch embeddings that start with very low dimensions (1-3) and adaptively expand as optimization progresses.

**Best for**: High-dimensional problems with unknown effective dimensionality, adaptive dimensionality reduction

**Reference**: Papenmeier et al. "Increasing the Scope as You Learn: Adaptive Bayesian Optimization in Nested Subspaces" (NeurIPS 2022)

### 11. SOLID (Sequential Optimization in Locally Important Dimensions)
Performs both global and local variable selection to identify important dimensions near the current optimum.

**Best for**: High-dimensional problems where only a subset of dimensions are locally important

**Reference**: Winkel et al. "Sequential Optimization in Locally Important Dimensions" (Technometrics 2021)

### 12. DSA (Dimension Scheduling Algorithm)
Optimizes along random subsets of dimensions at each iteration, with dimension importance estimated from historical data.

**Best for**: High-dimensional problems with computational constraints, parallel evaluation scenarios

**Reference**: Ulmasov et al. "Bayesian Optimization with Dimension Scheduling" (2016)

### 13. HDS (Hierarchical Diagonal Sampling)
Uses a binary tree structure to hierarchically partition dimensions and identify active subspaces through joint optimization.

**Best for**: Structured dimensionality reduction, problems with hierarchical dimension relationships

**Reference**: Chen et al. "Joint Optimization and Variable Selection of High-Dimensional Gaussian Processes" (ICML 2012)

### 14. Bounce
Reliable BO for mixed continuous/discrete/categorical spaces using nested binning embeddings.

**Best for**: Combinatorial optimization, mixed variable types, hardware design, neural architecture search

**Reference**: Papenmeier et al. "Bounce: Reliable High-Dimensional Bayesian Optimization for Combinatorial and Mixed Spaces" (NeurIPS 2023)

### 15. LSBO (Latent Space Bayesian Optimization)
Learns non-linear latent representations via VAE and performs BO in the learned latent space.

**Best for**: Structured data (molecules, sequences), problems with complex non-linear structure

**Reference**: Gómez-Bombarelli et al. "Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules" (ACS Central Science 2018)

## Quick Start

### Basic Usage

```python
import torch
from bo_methods import StandardBO

# Define optimization bounds
bounds = torch.tensor([[0.0, 0.0, 0.0],
                       [1.0, 1.0, 1.0]])

# Initialize optimizer
optimizer = StandardBO(
    input_dim=3,
    bounds=bounds,
    acq_func='EI'
)

# Initial random observations
X_init = torch.rand(5, 3)
y_init = objective_function(X_init)

# Update optimizer with initial data
optimizer.observe(X_init, y_init)

# Optimization loop
for iteration in range(50):
    # Suggest next point to evaluate
    X_next = optimizer.suggest(n_suggestions=1)

    # Evaluate objective function
    y_next = objective_function(X_next)

    # Update optimizer
    optimizer.observe(X_next, y_next)

    # Get current best
    best_x, best_y = optimizer.get_best_point()
    print(f"Iteration {iteration}: Best value = {best_y.item():.4f}")
```

### High-Dimensional Optimization with TuRBO

```python
from bo_methods import TuRBO

# High-dimensional problem (100 dimensions)
bounds = torch.stack([
    torch.zeros(100),
    torch.ones(100)
])

optimizer = TuRBO(
    input_dim=100,
    bounds=bounds,
    n_trust_regions=5,
    n_init=20,
    length_init=0.8
)

# Initial random sampling
from bo_methods.utils import get_initial_points

X_init = get_initial_points(
    dim=100,
    n_pts=20,
    bounds=bounds
)
y_init = objective_function(X_init)
optimizer.observe(X_init, y_init)

# Optimization loop
for iteration in range(200):
    X_next = optimizer.suggest(n_suggestions=1)
    y_next = objective_function(X_next)
    optimizer.observe(X_next, y_next)
```

### Embedding-Based Methods (HesBO, ALEBO, REMBO)

```python
from bo_methods import HesBO

# High-dimensional problem embedded in low-dimensional subspace
optimizer = HesBO(
    input_dim=100,      # Original dimension
    low_dim=10,         # Embedding dimension
    bounds=bounds,
    acq_func='EI'
)

# Similar optimization loop as above
X_init = get_initial_points(dim=100, n_pts=20, bounds=bounds)
y_init = objective_function(X_init)
optimizer.observe(X_init, y_init)

for iteration in range(100):
    X_next = optimizer.suggest()
    y_next = objective_function(X_next)
    optimizer.observe(X_next, y_next)
```

### Ensemble Methods (gPOE-BO, gPOE-TuRBO)

```python
from bo_methods import GPOEBO, GPOETRBO

# gPOE-BO with ensemble of experts
optimizer = GPOEBO(
    input_dim=50,
    bounds=bounds,
    points_per_expert=20,
    n_init=50,
    weighting='diff_entr'
)

# gPOE-TuRBO combining ensemble and trust regions
optimizer = GPOETRBO(
    input_dim=100,
    bounds=bounds,
    points_per_expert=20,
    length_init=0.8
)
```

## Advanced Usage

### Custom Acquisition Functions

All methods support different acquisition functions:

```python
# Expected Improvement (default)
optimizer = StandardBO(input_dim=5, bounds=bounds, acq_func='EI')

# Probability of Improvement
optimizer = StandardBO(input_dim=5, bounds=bounds, acq_func='PI')

# Upper Confidence Bound
optimizer = StandardBO(input_dim=5, bounds=bounds, acq_func='UCB')
```

### Custom GP Kernels

```python
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.constraints import Interval

# Custom kernel with specific lengthscale constraints
covar_module = ScaleKernel(
    MaternKernel(
        nu=2.5,
        ard_num_dims=10,
        lengthscale_constraint=Interval(0.01, 10.0)
    )
)

optimizer = StandardBO(
    input_dim=10,
    bounds=bounds,
    covar_module=covar_module
)
```

### Batch Optimization

```python
# Suggest multiple points in parallel
X_next = optimizer.suggest(n_suggestions=5)

# Evaluate in parallel (if possible)
y_next = objective_function(X_next)

# Update with batch observations
optimizer.observe(X_next, y_next)
```

### GPU Acceleration

```python
import torch

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# All tensors should be on the same device
bounds = bounds.to(device)
optimizer = StandardBO(
    input_dim=10,
    bounds=bounds,
    device=device
)
```

## Benchmarking Example

```python
import torch
from bo_methods import StandardBO, TuRBO, HesBO
from bo_methods.utils import get_initial_points

def ackley(X):
    """Ackley test function"""
    a, b, c = 20, 0.2, 2 * torch.pi
    d = X.shape[-1]
    sum1 = (X ** 2).sum(dim=-1)
    sum2 = torch.cos(c * X).sum(dim=-1)
    return -a * torch.exp(-b * torch.sqrt(sum1 / d)) - torch.exp(sum2 / d) + a + torch.e

# Compare methods
methods = {
    'StandardBO': StandardBO(input_dim=10, bounds=bounds),
    'TuRBO': TuRBO(input_dim=10, bounds=bounds, n_trust_regions=3),
    'HesBO': HesBO(input_dim=10, low_dim=5, bounds=bounds)
}

results = {}
for name, optimizer in methods.items():
    # Initialize
    X_init = get_initial_points(dim=10, n_pts=20, bounds=bounds)
    y_init = -ackley(X_init).reshape(-1, 1)  # Negate for maximization
    optimizer.observe(X_init, y_init)

    # Optimize
    best_values = []
    for i in range(100):
        X_next = optimizer.suggest()
        y_next = -ackley(X_next).reshape(-1, 1)
        optimizer.observe(X_next, y_next)
        best_values.append(optimizer.get_best_point()[1].item())

    results[name] = best_values
    print(f"{name}: Best = {max(best_values):.6f}")
```

## Project Structure

```
bayesian-optimization-collection/
├── bo_methods/
│   ├── __init__.py          # Package initialization
│   ├── base.py              # Abstract base classes
│   ├── utils.py             # Utility functions
│   ├── standard_bo.py       # Standard BO implementation
│   ├── turbo.py             # TuRBO implementation
│   ├── hesbo.py             # HesBO implementation
│   ├── alebo.py             # ALEBO implementation
│   ├── rembo.py             # REMBO implementation
│   ├── gpoe_bo.py           # gPOE-BO implementation
│   └── gpoe_turbo.py        # gPOE-TuRBO implementation
├── tests/                   # Unit tests
├── examples/                # Usage examples
├── docs/                    # Documentation
├── requirements.txt         # Dependencies
├── setup.py                 # Installation script
└── README.md               # This file
```

## Best Practices

1. **Start Simple**: Begin with StandardBO for problems with d < 20
2. **Scale Up**: Use TuRBO for high-dimensional problems (d = 50-1000)
3. **Leverage Structure**: Use embedding methods (HesBO, ALEBO, REMBO) when you suspect low effective dimensionality
4. **Ensemble for Robustness**: Use gPOE methods for complex landscapes or when you need robustness
5. **Monitor Trust Regions**: Watch TR expansion/shrinking in TuRBO-based methods
6. **Normalize Your Objective**: Better to work with standardized/normalized objectives
7. **Use Enough Initial Points**: Generally 2d to 5d initial random samples

## Testing

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run with coverage
pytest --cov=bo_methods tests/
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## Citation

If you use this code in your research, please cite:

```bibtex
@software{bayesian_optimization_collection,
  title = {Bayesian Optimization Methods Collection},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/bayesian-optimization-collection}
}
```

## License

MIT License - see LICENSE file for details

## References

1. Eriksson et al. "Scalable Global Optimization via Local Bayesian Optimization" NeurIPS 2019
2. Nayebi et al. "A Framework for Bayesian Optimization in Embedded Subspaces" ICML 2019
3. Letham et al. "Re-Examining Linear Embeddings for High-Dimensional Bayesian Optimization" NeurIPS 2020
4. Wang et al. "Bayesian Optimization in High Dimensions via Random Embeddings" IJCAI 2013

## Acknowledgments

This collection builds upon the excellent BoTorch and GPyTorch libraries.
