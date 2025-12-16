# HesBO Implementation Fix

## Issue

The original HesBO implementation in this collection did not correctly implement the Count Sketch projection from the paper.

## What Was Wrong

### Original (Incorrect) Implementation

```python
# WRONG: Used scatter_add with normalization
values = (X * self.sigma.unsqueeze(0)) * self.scaling  # 1/sqrt(D) scaling
result.scatter_add_(1, indices, values)
return result.clamp_(0.0, 1.0)  # Clamped to [0,1]
```

Problems:
1. ❌ Added unnecessary `scaling = 1/sqrt(D)` normalization
2. ❌ Clamped output to `[0, 1]` which distorts the projection
3. ❌ Used `scatter_add_` which is correct but with wrong values

### Fixed Implementation

```python
# CORRECT: Pure Count Sketch
for i in range(self.input_dim):
    bucket_idx = self.h[i]
    Y[:, bucket_idx] += X[:, i] * self.sigma[i]
return Y  # No clamping, no scaling
```

## Count Sketch Algorithm

### From the Paper (Nayebi et al., ICML 2019)

Count Sketch uses two random functions:
- **Hash function** `h: [D] → [d]` - maps each high dimension to a low dimension bucket
- **Sign function** `s: [D] → {-1, +1}` - random signs

### Forward Projection (high_to_low)

**Formula**: `y[h[i]] += s[i] * x[i]` for all `i ∈ [D]`

**Implementation**:
```python
def high_to_low(self, X: torch.Tensor) -> torch.Tensor:
    Y = torch.zeros((batch_size, self.low_dim))
    for i in range(self.input_dim):
        bucket_idx = self.h[i]
        Y[:, bucket_idx] += X[:, i] * self.sigma[i]
    return Y
```

### Backward Projection (low_to_high)

**Formula**: `x[i] = s[i] * y[h[i]]` for all `i ∈ [D]`

**Implementation**:
```python
def low_to_high(self, X: torch.Tensor) -> torch.Tensor:
    Y = torch.zeros((batch_size, self.input_dim))
    for i in range(self.input_dim):
        bucket_idx = self.h[i]
        Y[:, i] = X[:, bucket_idx] * self.sigma[i]
    return clamp_to_bounds(Y, self.bounds)  # Only clamp to problem bounds
```

## Verification Against Reference

### Reference Implementation (`HesBO/count_sketch.py`)

```python
# Forward projection
for i in range(high_dim):
    high_to_low[i] = np.random.choice(range(low_dim))
    low_obs[:,high_to_low[i]] = X[:,i]*sign[i] + low_obs[:,high_to_low[i]]

# Backward projection
for i in range(high_dim):
    high_obs[:,i] = sign[i]*low_obs[:,high_to_low[i]]
```

### Our Fixed Implementation

```python
# Forward projection
for i in range(self.input_dim):
    bucket_idx = self.h[i]
    Y[:, bucket_idx] += X[:, i] * self.sigma[i]

# Backward projection
for i in range(self.input_dim):
    bucket_idx = self.h[i]
    Y[:, i] = X[:, bucket_idx] * self.sigma[i]
```

✅ **Matches perfectly!**

## Key Properties of Count Sketch

### 1. Hash Collisions

Multiple high dimensions can map to the same low dimension:
```python
# Example: both dim 0 and dim 5 map to bucket 2
h[0] = 2
h[5] = 2

# Then: Y[2] = s[0]*X[0] + s[5]*X[5] + ...
```

This is **intentional** and enables dimensionality reduction.

### 2. Lossy Compression

Count Sketch is a **lossy projection**:
```python
X_original = torch.rand(100)
Y = high_to_low(X_original)      # 100 → 10
X_reconstructed = low_to_high(Y) # 10 → 100

# X_original != X_reconstructed (due to collisions)
```

### 3. No Normalization

Unlike some other sketching methods:
- ❌ No `1/sqrt(D)` scaling
- ❌ No normalization to `[0, 1]`
- ✅ Just sum with random signs

## Changes Made

### File: `bo_methods/hesbo.py`

#### 1. Removed Scaling Factor
```python
# REMOVED:
self.scaling = 1.0 / torch.sqrt(torch.tensor(float(input_dim)))
```

#### 2. Fixed Forward Projection
```python
# OLD (WRONG):
values = (X * self.sigma.unsqueeze(0)) * self.scaling
result.scatter_add_(1, indices, values)
return result.clamp_(0.0, 1.0)

# NEW (CORRECT):
for i in range(self.input_dim):
    bucket_idx = self.h[i]
    Y[:, bucket_idx] += X[:, i] * self.sigma[i]
return Y
```

#### 3. Fixed Backward Projection
```python
# OLD (WRONG):
gathered_values = X[:, self.h]
result = (gathered_values * self.sigma.unsqueeze(0)).div_(self.scaling)

# NEW (CORRECT):
for i in range(self.input_dim):
    bucket_idx = self.h[i]
    Y[:, i] = X[:, bucket_idx] * self.sigma[i]
```

#### 4. Fixed Reset Method
```python
# OLD (WRONG):
self.h = torch.nn.Parameter(
    torch.randint(...),
    requires_grad=False
)

# NEW (CORRECT):
self.h = torch.randint(...)  # Just a tensor, not a Parameter
```

## Testing

Run the test script to verify:

```bash
python test_hesbo_projection.py
```

Expected output:
```
Testing HesBO Count Sketch Projection
==============================
✓ Forward projection matches manual calculation
✓ Backward projection matches manual calculation
✓ Integration with BO works correctly
ALL TESTS PASSED ✓
```

## Impact

### What Changed
- ✅ HesBO now implements pure Count Sketch from the paper
- ✅ Matches reference implementation in `HesBO/count_sketch.py`
- ✅ Maintains shape consistency guarantees

### What Didn't Change
- ✅ Algorithm logic preserved
- ✅ API remains the same
- ✅ Integration with other components unchanged

## References

1. **Paper**: Nayebi et al. "A Framework for Bayesian Optimization in Embedded Subspaces" ICML 2019
2. **Reference Code**: `/mnt/h/BOResearch-25fall/HesBO/count_sketch.py`
3. **Count Sketch**: Charikar, M., Chen, K., & Farach-Colton, M. (2004). Finding frequent items in data streams.

## Summary

| Aspect | Before | After |
|--------|--------|-------|
| Forward Projection | ❌ Scaled & clamped | ✅ Pure Count Sketch |
| Backward Projection | ❌ Scaled | ✅ Pure Count Sketch |
| Normalization | ❌ `1/sqrt(D)` | ✅ None |
| Clamping | ❌ To `[0,1]` | ✅ Only to bounds |
| Reference Match | ❌ No | ✅ Yes |
| Paper Match | ❌ No | ✅ Yes |

**Status**: ✅ FIXED - HesBO now correctly implements Count Sketch!
