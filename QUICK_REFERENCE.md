# HDBO Methods - Quick Reference Card

## 🚀 One-Liner Usage

```python
from bo_methods import VanillaBO, BAxUS, SOLID, DSA, HDS, Bounce, LSBO, RandomLineBO
```

---

## 📋 Method Selection Guide

### Just Getting Started?
```python
optimizer = VanillaBO(input_dim=d, bounds=bounds)  # Start here!
```

### Mixed Variables (discrete/categorical)?
```python
optimizer = Bounce(input_dim=d, bounds=bounds, discrete_dims=[0,1,2])
```

### Unknown Dimensionality?
```python
optimizer = BAxUS(input_dim=d, bounds=bounds, target_dim_init=2)
```

### Sparse Importance?
```python
optimizer = SOLID(input_dim=d, bounds=bounds)
```

### Need Speed?
```python
optimizer = DSA(input_dim=d, bounds=bounds, n_active_dims=10)
```

### Hierarchical Structure?
```python
optimizer = HDS(input_dim=d, bounds=bounds)
```

### Complex Non-linear?
```python
optimizer = LSBO(input_dim=d, bounds=bounds, latent_dim=10)
```

### Exploration?
```python
optimizer = RandomLineBO(input_dim=d, bounds=bounds)
```

---

## 💡 Common Patterns

### Basic Workflow
```python
# 1. Setup
import torch
from bo_methods import BAxUS

bounds = torch.stack([torch.zeros(d), torch.ones(d)])
optimizer = BAxUS(input_dim=d, bounds=bounds)

# 2. Initialize
X_init = torch.rand(20, d)
y_init = objective(X_init)
optimizer.observe(X_init, y_init)

# 3. Optimize
for i in range(100):
    X_next = optimizer.suggest()
    y_next = objective(X_next)
    optimizer.observe(X_next, y_next)

# 4. Get result
best_x, best_y = optimizer.get_best_point()
```

---

## 📊 Quick Comparison

| Need | Use This | Why |
|------|----------|-----|
| Simple baseline | VanillaBO | Surprisingly good! |
| Fast iterations | DSA, RandomLineBO | Low overhead |
| Adaptive dims | BAxUS | Expands as needed |
| Sparse features | SOLID, HDS | Variable selection |
| Mixed types | Bounce | Handles discrete/categorical |
| Deep structure | LSBO | VAE embeddings |
| Unknown structure | VanillaBO → BAxUS | Safe progression |

---

## 🎯 Dimension Guidelines

```
d < 20      → VanillaBO, StandardBO
20 ≤ d < 100 → SOLID, DSA, HDS
d ≥ 100     → BAxUS, TuRBO, LSBO
```

---

## ⚙️ Common Parameters

### All Methods Support:
- `input_dim`: Input dimensionality
- `bounds`: Tensor of shape (2, input_dim)
- `device`: 'cpu' or 'cuda'
- `dtype`: torch.float32 or torch.float64

### Method-Specific:

**VanillaBO**:
- `acq_func`: 'EI' or 'UCB'
- `lengthscale_prior_beta`: Auto-set to √d

**BAxUS**:
- `target_dim_init`: Starting dimension (default: 2)
- `target_dim_max`: Maximum dimension
- `expansion_factor`: How fast to expand (default: 2)

**SOLID**:
- `local_threshold`: Sensitivity (default: 0.05)
- `min_active_dims`: Minimum active dimensions

**DSA**:
- `n_active_dims`: Dims per iteration (default: √d)
- `use_pca`: Use PCA for importance

**HDS**:
- `max_active_dims`: Max active dimensions
- `tree_depth`: Depth of binary tree

**Bounce**:
- `target_dim`: Target embedding dimension
- `discrete_dims`: List of discrete dimension indices
- `categorical_dims`: Dict {dim: n_categories}

**LSBO**:
- `latent_dim`: VAE latent dimension
- `vae_hidden_dim`: VAE architecture
- `vae_train_steps`: Training iterations

---

## 🔧 Troubleshooting

### Method fails?
```python
# Start simple
optimizer = VanillaBO(input_dim=d, bounds=bounds)
```

### Too slow?
```python
# Use faster method
optimizer = DSA(input_dim=d, bounds=bounds, n_active_dims=10)
```

### Bad results?
```python
# Try adaptive method
optimizer = BAxUS(input_dim=d, bounds=bounds, target_dim_init=2)
```

### Mixed variables?
```python
# Use Bounce
optimizer = Bounce(input_dim=d, bounds=bounds, discrete_dims=[...])
```

---

## 📖 Learn More

- **Full docs**: See README.md
- **Examples**: `examples/all_hdbo_methods_demo.py`
- **Technical**: HDBO_METHODS_FINAL.md
- **Summary**: FINAL_SUMMARY.md

---

## 🎓 Citation Template

```bibtex
@inproceedings{method_year,
  title={Method Title},
  author={Authors},
  booktitle={Venue},
  year={Year}
}
```

See individual method files for specific citations.

---

## ✅ Quick Checklist

Before optimizing:
- [ ] Define `bounds` correctly
- [ ] Normalize objective if needed
- [ ] Set random seed for reproducibility
- [ ] Use enough initial samples (2d to 5d)
- [ ] Choose appropriate method
- [ ] Set reasonable iteration budget

---

**Made with ❤️ for the BO community**
