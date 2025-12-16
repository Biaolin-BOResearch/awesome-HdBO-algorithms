# HDBO Benchmark Methods Implementation Summary

This document summarizes the implementation of high-dimensional Bayesian optimization methods from the HDBO benchmark survey's Figure 1 timeline.

## Implemented Methods (5/17+ from Figure 1)

### ✅ Completed Implementations

#### 1. **VanillaBO** (Hvarfner et al., 2024)
- **File**: `bo_methods/vanilla_bo.py`
- **Category**: Baseline method with dimensionality-dependent priors
- **Key Features**:
  - Dimensionality-dependent Gamma priors on lengthscales (scales with √d)
  - Standard Matérn 5/2 kernel with ARD
  - Expected Improvement or UCB acquisition
  - No complex trust regions or embeddings

**Usage**:
```python
from bo_methods import VanillaBO

optimizer = VanillaBO(
    input_dim=100,
    bounds=bounds,
    acq_func='EI',
    lengthscale_prior_beta=None  # Auto-set to sqrt(d)
)
```

#### 2. **RandomLineBO**
- **File**: `bo_methods/random_line_bo.py`
- **Category**: Random subspace method
- **Key Features**:
  - Samples random lines in high-dimensional space
  - Performs 1D optimization along each line
  - Can use current best point as anchor
  - Effective for directional structure

**Usage**:
```python
from bo_methods import RandomLineBO

optimizer = RandomLineBO(
    input_dim=100,
    bounds=bounds,
    n_lines_per_iter=5,
    line_samples=100,
    use_anchor=True
)
```

#### 3. **BAxUS** (Papenmeier et al., 2022)
- **File**: `bo_methods/baxus.py`
- **Category**: Adaptive linear embedding
- **Key Features**:
  - Nested count-sketch embeddings
  - Starts with very low dimensions (default: 2)
  - Adaptively expands target dimensionality
  - Trust region mechanism
  - Preserves observations across dimension expansions

**Usage**:
```python
from bo_methods import BAxUS

optimizer = BAxUS(
    input_dim=100,
    bounds=bounds,
    target_dim_init=2,
    target_dim_max=20,
    expansion_factor=2
)
```

#### 4. **SOLID** (Winkel et al., 2021)
- **File**: `bo_methods/solid.py`
- **Category**: Variable selection
- **Key Features**:
  - Global variable selection via lengthscale analysis
  - Local variable selection near current optimum
  - Fixes inactive dimensions during acquisition optimization
  - Adaptive dimension importance

**Usage**:
```python
from bo_methods import SOLID

optimizer = SOLID(
    input_dim=100,
    bounds=bounds,
    global_threshold=0.01,
    local_threshold=0.05,
    min_active_dims=2
)
```

#### 5. **DSA** (Ulmasov et al., 2016)
- **File**: `bo_methods/dsa.py`
- **Category**: Variable selection / Dimension scheduling
- **Key Features**:
  - Optimizes random subsets of dimensions per iteration
  - Dimension importance via variance and correlation analysis
  - Optional PCA-based importance (when enough data)
  - Reduced computational cost per iteration

**Usage**:
```python
from bo_methods import DSA

optimizer = DSA(
    input_dim=100,
    bounds=bounds,
    n_active_dims=10,  # Default: sqrt(d)
    use_pca=False,
    importance_update_freq=5
)
```

---

## Previously Implemented Methods (Already in Codebase)

The following methods from Figure 1 were already implemented:

1. **TuRBO** - Trust Region BO (Eriksson et al., 2019) ✅
2. **ALEBO** - Adaptive Linear Embedding BO (Letham et al., 2020) ✅
3. **HesBO** - Hashing-Enhanced Subspace BO (Nayebi et al., 2019) ✅
4. **REMBO** - Random Embedding BO (Wang et al., 2013) ✅
5. **SAASBO** - Sparse Axis-Aligned Subspace BO ✅
6. **gPOE-BO** - Generalized Product of Experts BO ✅
7. **gPOE-TuRBO** - gPOE + TuRBO combination ✅

---

## Pending Implementations

The following methods from Figure 1 are not yet implemented:

### Variable Selection Methods
- **HDS** (Hierarchical Diagonal Sampling) - Chen et al., 2012
- **SAAS** (Sparse Axis-Aligned Subspace) - Raponi et al., 2020

### Trust Region Methods
- **MORBO** (Multi-Objective TuRBO) - Daulton et al., 2022
- **CASMOPOLITAN** - Papenmeier et al., 2024

### Non-linear Embedding Methods
- **LSBO** (Latent Space BO) - Gómez-Bombarelli et al., 2018
- **LaMBO** - Stanton et al., 2022
- **VEBO** (VAE-based BO) - Tripp et al., 2020

### Structured Spaces Methods
- **BODi** - Deshwal et al., 2021
- **ProbRep** (Probabilistic Reparametrization) - Daulton et al., 2022
- **Bounce** - Papenmeier et al., 2023

### Additive Models
- **ADMM-based approaches**

---

## Method Categories and When to Use

### For Variable Selection:
- **SOLID**: When you know only a subset of dimensions matter locally
- **DSA**: When you want computational efficiency and can tolerate approximate optimization
- **SAASBO**: When you suspect sparse axis-aligned structure

### For Embedding Methods:
- **BAxUS**: When effective dimensionality is unknown and needs adaptation
- **ALEBO**: When you suspect linear structure
- **HesBO**: When you want randomized hashing for structure preservation
- **REMBO**: Simple random embeddings for quick prototyping

### For Trust Regions:
- **TuRBO**: Single-objective, high-dimensional problems
- **gPOE-TuRBO**: Very high-dimensional with ensemble benefits
- **MORBO**: Multi-objective high-dimensional problems (to be implemented)

### For Baselines:
- **VanillaBO**: Always try this first! Recent research shows it's competitive
- **RandomLineBO**: Exploratory baseline for directional problems

---

## Testing and Validation

An example script is provided to compare all methods:

```bash
python examples/hdbo_methods_example.py
```

This runs all implemented HDBO methods on the 50D Ackley function and compares their performance.

---

## Implementation Quality

All implemented methods:
- ✅ Follow the same `BayesianOptimizer` interface
- ✅ Support GPU acceleration
- ✅ Include type hints
- ✅ Have detailed docstrings
- ✅ Return tensors of shape `(n_suggestions, input_dim)`
- ✅ Are based on published research papers

---

## Future Work

1. **Complete remaining HDBO methods**: MORBO, LSBO, LaMBO, etc.
2. **Add comprehensive benchmarks**: Compare on standard test suites
3. **Implement hybrid methods**: Combine best features from multiple approaches
4. **Add multi-objective support**: Extend more methods to MO setting
5. **Discrete/categorical optimization**: Extend to structured spaces

---

## References

### Newly Implemented Methods:

1. **VanillaBO**: Hvarfner, C., et al. (2024). "Vanilla Bayesian Optimization Performs Great in High Dimensions." ICML.

2. **BAxUS**: Papenmeier, L., Nardi, L., & Poloczek, M. (2022). "Increasing the Scope as You Learn: Adaptive Bayesian Optimization in Nested Subspaces." NeurIPS.

3. **SOLID**: Winkel, M. A., et al. (2021). "Sequential Optimization in Locally Important Dimensions." Technometrics, 63(2), 236-248.

4. **DSA**: Ulmasov, D., et al. (2016). "Bayesian Optimization with Dimension Scheduling: Application to Biological Systems." 26th European Symposium on Computer Aided Process Engineering.

### Survey Paper:

- **HDBO Benchmark**: Grosnit, A., et al. (2024). "A Survey and Benchmark of High-Dimensional Bayesian Optimization of Discrete Sequences." arXiv:2406.04739.

---

## Contact

For questions or contributions, please open an issue on GitHub.
