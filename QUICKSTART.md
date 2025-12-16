# Quick Start Guide

Get started with Bayesian Optimization in 5 minutes!

## Installation

```bash
cd bayesian-optimization-collection
pip install -e .
```

## 30-Second Example

```python
import torch
from bo_methods import StandardBO
from bo_methods.utils import get_initial_points

# 1. Define your optimization problem
def objective(X):
    """Function to maximize"""
    return -(X ** 2).sum(dim=-1).reshape(-1, 1)

# 2. Set bounds
bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])

# 3. Create optimizer
optimizer = StandardBO(input_dim=2, bounds=bounds)

# 4. Initialize with random points
X_init = get_initial_points(dim=2, n_pts=5, bounds=bounds)
y_init = objective(X_init)
optimizer.observe(X_init, y_init)

# 5. Run optimization
for i in range(20):
    X_next = optimizer.suggest()
    y_next = objective(X_next)
    optimizer.observe(X_next, y_next)

# 6. Get best result
best_x, best_y = optimizer.get_best_point()
print(f"Best point: {best_x}")
print(f"Best value: {best_y.item():.4f}")
```

## Choose Your Method

### For Low Dimensions (d < 20)
```python
from bo_methods import StandardBO
optimizer = StandardBO(input_dim=5, bounds=bounds, acq_func='EI')
```

### For High Dimensions (d = 20-1000)
```python
from bo_methods import TuRBO
optimizer = TuRBO(input_dim=100, bounds=bounds, n_trust_regions=5)
```

### For Very High Dimensions with Structure (d > 100)
```python
from bo_methods import HesBO
optimizer = HesBO(input_dim=200, low_dim=10, bounds=bounds)
```

## Common Patterns

### Pattern 1: Hyperparameter Tuning
```python
# Define search space for learning rate, dropout, etc.
bounds = torch.tensor([
    [0.001, 0.1, 10],     # [lr_min, dropout_min, layers_min]
    [0.1, 0.5, 100]       # [lr_max, dropout_max, layers_max]
])

def train_and_evaluate(hyperparams):
    lr, dropout, n_layers = hyperparams
    # Train your model...
    return validation_accuracy

optimizer = TuRBO(input_dim=3, bounds=bounds)
# ... run optimization ...
```

### Pattern 2: Batch Optimization
```python
# Suggest multiple points at once
X_next = optimizer.suggest(n_suggestions=5)

# Evaluate in parallel
y_next = torch.stack([objective(x) for x in X_next])

# Update
optimizer.observe(X_next, y_next)
```

### Pattern 3: GPU Acceleration
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bounds = bounds.to(device)

optimizer = StandardBO(input_dim=10, bounds=bounds, device=device)
```

## Complete Working Example

```python
"""
Optimize the Ackley function - a common benchmark
"""
import torch
from bo_methods import TuRBO
from bo_methods.utils import get_initial_points

def ackley(X):
    """Ackley test function (minimize)"""
    a, b, c = 20, 0.2, 2 * torch.pi
    d = X.shape[-1]
    sum1 = (X ** 2).sum(dim=-1)
    sum2 = torch.cos(c * X).sum(dim=-1)
    return -(-a * torch.exp(-b * torch.sqrt(sum1 / d))
            - torch.exp(sum2 / d) + a + torch.e)

# Setup
dim = 10
bounds = torch.stack([
    torch.full((dim,), -5.0),
    torch.full((dim,), 5.0)
])

# Create optimizer
optimizer = TuRBO(
    input_dim=dim,
    bounds=bounds,
    n_trust_regions=3,
    n_init=20
)

# Initialize
X_init = get_initial_points(dim=dim, n_pts=20, bounds=bounds)
y_init = ackley(X_init).reshape(-1, 1)  # Negate for maximization
optimizer.observe(X_init, y_init)

# Optimize
print("Starting optimization...")
for i in range(100):
    X_next = optimizer.suggest()
    y_next = ackley(X_next).reshape(-1, 1)
    optimizer.observe(X_next, y_next)

    if (i + 1) % 20 == 0:
        best_x, best_y = optimizer.get_best_point()
        print(f"Iteration {i+1}: Best = {-best_y.item():.6f}")

# Final result
best_x, best_y = optimizer.get_best_point()
print(f"\nOptimization complete!")
print(f"Best value found: {-best_y.item():.6f}")
print(f"Global optimum: 0.0")
```

## Next Steps

1. **Read the full documentation**: [README.md](README.md)
2. **Explore examples**: [examples/basic_example.py](examples/basic_example.py)
3. **Compare methods**: [docs/METHOD_COMPARISON.md](docs/METHOD_COMPARISON.md)
4. **Detailed usage**: [docs/USAGE.md](docs/USAGE.md)

## Common Issues

### Issue: "Model not initialized"
**Solution**: Call `observe()` before `suggest()`

```python
# Wrong
optimizer.suggest()  # ERROR

# Correct
optimizer.observe(X_init, y_init)  # First observe
optimizer.suggest()  # Now works
```

### Issue: Poor performance
**Solution**: Add more initial points

```python
# Use 2d to 5d initial points
n_init = 2 * input_dim  # Good rule of thumb
X_init = get_initial_points(dim=input_dim, n_pts=n_init, bounds=bounds)
```

### Issue: Slow computation
**Solution**: Enable GPU or use embedding methods

```python
# Option 1: GPU
device = torch.device('cuda')
optimizer = StandardBO(input_dim=d, bounds=bounds, device=device)

# Option 2: Embedding for high-D
optimizer = HesBO(input_dim=200, low_dim=10, bounds=bounds)
```

## Help & Support

- **Documentation**: See [README.md](README.md)
- **Issues**: Report bugs on GitHub
- **Examples**: Check [examples/](examples/) directory

## Quick Reference

| Method | Best For | Typical Use |
|--------|----------|-------------|
| StandardBO | d < 20 | `StandardBO(input_dim=10, bounds=bounds)` |
| TuRBO | 20 ≤ d ≤ 1000 | `TuRBO(input_dim=100, bounds=bounds)` |
| HesBO | d > 50, additive | `HesBO(input_dim=200, low_dim=10, bounds=bounds)` |
| ALEBO | d > 50, linear | `ALEBO(input_dim=100, low_dim=15, bounds=bounds)` |
| REMBO | d > 50 | `REMBO(input_dim=100, low_dim=8, bounds=bounds)` |
| gPOE-BO | Complex landscape | `GPOEBO(input_dim=50, bounds=bounds)` |
| gPOE-TuRBO | d > 100, complex | `GPOETRBO(input_dim=150, bounds=bounds)` |

Happy optimizing! 🚀
