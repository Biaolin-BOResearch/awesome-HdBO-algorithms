# Usage Guide

## Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Choosing the Right Method](#choosing-the-right-method)
4. [Advanced Configuration](#advanced-configuration)
5. [Troubleshooting](#troubleshooting)

## Installation

### Basic Installation

```bash
cd bayesian-optimization-collection
pip install -e .
```

### With Development Tools

```bash
pip install -e ".[dev]"
```

### Verify Installation

```python
import torch
from bo_methods import StandardBO, TuRBO, HesBO

print("Installation successful!")
```

## Quick Start

### 1. Define Your Optimization Problem

```python
import torch

def objective_function(X):
    """
    Your black-box function to optimize.

    Args:
        X: Input tensor of shape (n, d)

    Returns:
        Output values of shape (n,) or (n, 1)
    """
    # Example: sum of squares
    return -(X ** 2).sum(dim=-1)
```

### 2. Set Up Bounds

```python
# For a 5-dimensional problem with bounds [0, 1]
input_dim = 5
bounds = torch.stack([
    torch.zeros(input_dim),  # Lower bounds
    torch.ones(input_dim)     # Upper bounds
])
```

### 3. Choose an Optimizer

```python
from bo_methods import StandardBO

optimizer = StandardBO(
    input_dim=input_dim,
    bounds=bounds,
    acq_func='EI'
)
```

### 4. Initialize with Random Points

```python
from bo_methods.utils import get_initial_points

# Generate 10 initial random points
X_init = get_initial_points(
    dim=input_dim,
    n_pts=10,
    bounds=bounds
)

# Evaluate objective
y_init = objective_function(X_init).reshape(-1, 1)

# Update optimizer
optimizer.observe(X_init, y_init)
```

### 5. Run Optimization Loop

```python
n_iterations = 50

for i in range(n_iterations):
    # Suggest next point(s)
    X_next = optimizer.suggest(n_suggestions=1)

    # Evaluate objective
    y_next = objective_function(X_next).reshape(-1, 1)

    # Update optimizer
    optimizer.observe(X_next, y_next)

    # Get current best
    best_x, best_y = optimizer.get_best_point()

    if i % 10 == 0:
        print(f"Iteration {i}: Best value = {best_y.item():.4f}")
```

## Choosing the Right Method

### Decision Tree

```
Is your problem dimension < 20?
├─ Yes → Use StandardBO
│         ├─ Simple, well-understood
│         └─ Full Gaussian Process
│
└─ No → Is dimension 20-100?
         ├─ Yes → Use TuRBO
         │        ├─ Trust region optimization
         │        └─ Good exploration-exploitation
         │
         └─ No → Dimension > 100?
                  └─ Yes → Do you suspect low effective dimensionality?
                           ├─ Yes → Use HesBO, ALEBO, or REMBO
                           │        └─ Embedding-based methods
                           │
                           └─ No → Use gPOE-TuRBO
                                    └─ Ensemble + Trust regions
```

### Method Comparison Table

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| StandardBO | d < 20 | Simple, interpretable | Poor scaling to high-D |
| TuRBO | 20 ≤ d ≤ 1000 | Good high-D performance | Requires tuning TR params |
| HesBO | d > 50 with structure | Fast, hash-based | Assumes additive structure |
| ALEBO | d > 50 with linearity | Adaptive embedding | Linear assumption |
| REMBO | d > 50 | Simple random embedding | Fixed embedding |
| gPOE-BO | Complex landscapes | Robust ensemble | More computation |
| gPOE-TuRBO | d > 100 | Best of both worlds | Most complex |

## Advanced Configuration

### Custom Acquisition Functions

```python
# Expected Improvement (best for balanced exploration-exploitation)
optimizer = StandardBO(input_dim=d, bounds=bounds, acq_func='EI')

# Probability of Improvement (more exploitative)
optimizer = StandardBO(input_dim=d, bounds=bounds, acq_func='PI')

# Upper Confidence Bound (more exploratory)
optimizer = StandardBO(input_dim=d, bounds=bounds, acq_func='UCB')
```

### Custom GP Kernels

```python
from gpytorch.kernels import ScaleKernel, MaternKernel, RBFKernel
from gpytorch.constraints import Interval

# Matérn kernel with ARD
covar_module = ScaleKernel(
    MaternKernel(
        nu=2.5,
        ard_num_dims=input_dim,
        lengthscale_constraint=Interval(0.01, 10.0)
    )
)

optimizer = StandardBO(
    input_dim=input_dim,
    bounds=bounds,
    covar_module=covar_module
)
```

### TuRBO Configuration

```python
from bo_methods import TuRBO

optimizer = TuRBO(
    input_dim=100,
    bounds=bounds,
    n_trust_regions=5,          # Number of parallel trust regions
    n_init=20,                   # Initial random points per TR
    length_init=0.8,             # Initial TR length
    length_min=0.5**7,           # Minimum TR length (restart trigger)
    length_max=1.6,              # Maximum TR length
    success_tolerance=3,         # Expand after 3 successes
    failure_tolerance=5,         # Shrink after 5 failures
)
```

### Embedding Methods Configuration

```python
from bo_methods import HesBO, ALEBO, REMBO

# HesBO with custom embedding dimension
optimizer = HesBO(
    input_dim=100,
    low_dim=10,      # Embedding dimension
    bounds=bounds,
    acq_func='EI'
)

# ALEBO with orthonormal embedding
optimizer = ALEBO(
    input_dim=100,
    low_dim=15,
    bounds=bounds
)

# REMBO with random linear projection
optimizer = REMBO(
    input_dim=100,
    low_dim=8,
    bounds=bounds
)
```

### gPOE Configuration

```python
from bo_methods import GPOEBO, GPOETRBO

# gPOE-BO
optimizer = GPOEBO(
    input_dim=50,
    bounds=bounds,
    points_per_expert=20,    # Data points per GP expert
    n_init=50,               # Initial random points
    n_candidates=5000,       # Candidates for optimization
    beta=1.0,                # UCB exploration parameter
    weighting='diff_entr'    # Expert weighting method
)

# gPOE-TuRBO
optimizer = GPOETRBO(
    input_dim=100,
    bounds=bounds,
    points_per_expert=20,
    length_init=0.8,
    weighting='diff_entr'
)
```

### Batch Optimization

```python
# Suggest multiple points in parallel
batch_size = 5
X_next = optimizer.suggest(n_suggestions=batch_size)

# Evaluate in parallel (if your objective supports it)
y_next = objective_function(X_next)

# Update with batch
optimizer.observe(X_next, y_next)
```

### GPU Acceleration

```python
import torch

# Enable CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.double

# Ensure bounds are on GPU
bounds = bounds.to(device=device, dtype=dtype)

# Create optimizer with GPU support
optimizer = StandardBO(
    input_dim=input_dim,
    bounds=bounds,
    device=device,
    dtype=dtype
)

# All tensors will automatically be on GPU
```

## Troubleshooting

### Problem: "Model not initialized" error

**Solution**: Make sure to call `observe()` with initial data before calling `suggest()`.

```python
# Wrong
optimizer = StandardBO(input_dim=5, bounds=bounds)
X_next = optimizer.suggest()  # ERROR!

# Correct
optimizer = StandardBO(input_dim=5, bounds=bounds)
optimizer.observe(X_init, y_init)  # First observe
X_next = optimizer.suggest()  # Now works
```

### Problem: Poor optimization performance

**Solutions**:
1. **Increase initial samples**: Use 2d to 5d random initial points
2. **Check objective scale**: Normalize outputs to have mean ~0, std ~1
3. **Try different acquisition functions**: EI, PI, or UCB
4. **Adjust trust region parameters** (for TuRBO)
5. **Increase embedding dimension** (for embedding methods)

```python
# Normalize objective values
y_mean = y_init.mean()
y_std = y_init.std()
y_normalized = (y_init - y_mean) / y_std
```

### Problem: Out of memory on GPU

**Solutions**:
1. **Reduce batch size** for predictions
2. **Use smaller embedding dimensions**
3. **Process candidates in smaller chunks**

```python
# For gPOE methods
optimizer = GPOEBO(
    input_dim=50,
    bounds=bounds,
    batch_size=1000,  # Reduce from default 5000
    n_candidates=2000  # Reduce from default 5000
)
```

### Problem: Trust region converging too quickly (TuRBO)

**Solutions**:
1. Increase `failure_tolerance`
2. Increase `success_tolerance`
3. Adjust `length_min` and `length_max`

```python
optimizer = TuRBO(
    input_dim=100,
    bounds=bounds,
    failure_tolerance=10,     # More tolerant of failures
    success_tolerance=5,      # Need more successes to expand
    length_min=0.5**5,        # Allow smaller regions
)
```

### Problem: Suggestions outside bounds

**Solution**: This shouldn't happen, but if it does, manually clamp:

```python
X_next = optimizer.suggest()
X_next = torch.clamp(X_next, min=bounds[0], max=bounds[1])
```

### Problem: Slow GP fitting

**Solutions**:
1. Use approximate inference (not implemented in base version)
2. Reduce number of training points (for gPOE)
3. Use simpler kernels
4. Enable GPU acceleration

```python
# Simpler kernel (faster)
from gpytorch.kernels import RBFKernel

covar_module = RBFKernel(
    ard_num_dims=input_dim
)
```

## Best Practices Checklist

- [ ] Start with 2d to 5d initial random samples
- [ ] Normalize/standardize your objective function
- [ ] Use appropriate method for your dimensionality
- [ ] Monitor trust region behavior (for TuRBO)
- [ ] Use GPU when available for d > 50
- [ ] Validate bounds are correct
- [ ] Check for NaN/Inf values in observations
- [ ] Use consistent device/dtype for all tensors
- [ ] Consider batch optimization for parallel evaluations
- [ ] Save results regularly during long runs

## Common Patterns

### Pattern 1: Hyperparameter Tuning

```python
def train_model(hyperparams):
    """Train ML model with given hyperparameters."""
    # Your training code here
    return validation_accuracy

# Define search space
bounds = torch.tensor([
    [0.001, 0.1, 10, 32],      # [lr_min, dropout_min, layers_min, batch_min]
    [0.1, 0.5, 100, 512]       # [lr_max, dropout_max, layers_max, batch_max]
])

optimizer = TuRBO(input_dim=4, bounds=bounds)

# Initialize
X_init = get_initial_points(dim=4, n_pts=10, bounds=bounds)
y_init = torch.tensor([train_model(x) for x in X_init]).reshape(-1, 1)
optimizer.observe(X_init, y_init)

# Optimize
for i in range(50):
    X_next = optimizer.suggest()
    y_next = torch.tensor([train_model(X_next[0])]).reshape(-1, 1)
    optimizer.observe(X_next, y_next)
```

### Pattern 2: Multi-Fidelity Optimization

```python
def evaluate_at_fidelity(X, fidelity):
    """Evaluate at different fidelities (e.g., dataset sizes)."""
    # fidelity = 0.1: 10% of data
    # fidelity = 1.0: 100% of data
    return objective(X, fidelity)

# Use cheap low-fidelity evaluations for exploration
# Use expensive high-fidelity for final validation
```

### Pattern 3: Constraint Handling

```python
def penalized_objective(X):
    """Apply penalty for constraint violations."""
    y = objective(X)

    # Example constraint: sum of parameters < 1
    violation = torch.clamp(X.sum(dim=-1) - 1.0, min=0)
    penalty = 1000 * violation

    return y - penalty
```

## Further Reading

- [BoTorch Documentation](https://botorch.org/)
- [GPyTorch Documentation](https://gpytorch.ai/)
- Original papers listed in main README
