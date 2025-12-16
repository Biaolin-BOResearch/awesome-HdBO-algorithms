# Shape Guarantees

## Overview

All Bayesian optimization methods in this collection provide **strict shape guarantees** for their output tensors. This ensures consistency across different methods and makes it easy to swap between algorithms.

## Core Guarantee

**All `suggest()` methods return tensors of shape `(n_suggestions, input_dim)`**

This is true:
- For all 7 implemented methods
- Regardless of `n_suggestions` value (including `n_suggestions=1`)
- During initialization phase and normal operation
- Even when fallback to random sampling occurs

## Method-by-Method Guarantees

### StandardBO

```python
optimizer = StandardBO(input_dim=10, bounds=bounds)
optimizer.observe(X_init, y_init)

X_next = optimizer.suggest(n_suggestions=1)
assert X_next.shape == (1, 10)  # Always 2D

X_batch = optimizer.suggest(n_suggestions=5)
assert X_batch.shape == (5, 10)  # Always (n_suggestions, input_dim)
```

### TuRBO

```python
optimizer = TuRBO(input_dim=100, bounds=bounds)

# During initialization
X_next = optimizer.suggest(n_suggestions=1)
assert X_next.shape == (1, 100)

# After initialization
optimizer.observe(X_init, y_init)
X_next = optimizer.suggest(n_suggestions=3)
assert X_next.shape == (3, 100)
```

### Embedding Methods (HesBO, ALEBO, REMBO)

All embedding methods map from low-dimensional to high-dimensional space, always returning high-dimensional candidates:

```python
optimizer = HesBO(input_dim=200, low_dim=10, bounds=bounds)
optimizer.observe(X_init, y_init)

# Returns in HIGH-dimensional space
X_next = optimizer.suggest(n_suggestions=1)
assert X_next.shape == (1, 200)  # NOT (1, 10)!

X_batch = optimizer.suggest(n_suggestions=4)
assert X_batch.shape == (4, 200)
```

**Important**: Even though these methods work in a low-dimensional embedding space internally, `suggest()` always returns candidates in the **original high-dimensional space**.

### Ensemble Methods (gPOE-BO, gPOE-TuRBO)

```python
optimizer = GPOEBO(input_dim=50, bounds=bounds)

# During initialization
X_next = optimizer.suggest(n_suggestions=1)
assert X_next.shape == (1, 50)

# After initialization
optimizer.observe(X_init, y_init)
X_batch = optimizer.suggest(n_suggestions=3)
assert X_batch.shape == (3, 50)
```

## Implementation Details

### Shape Validation

All methods include **runtime shape assertions** to catch bugs early:

```python
def suggest(self, n_suggestions: int = 1) -> torch.Tensor:
    # ... optimization logic ...

    # Validate shape before returning
    assert candidates.shape == (n_suggestions, self.input_dim), \
        f"Expected shape ({n_suggestions}, {self.input_dim}), got {candidates.shape}"

    return candidates
```

These assertions:
- Run in both debug and production code
- Provide clear error messages if shape is incorrect
- Help catch bugs during development

### Never Squeeze Dimensions

**We never squeeze the batch dimension**, even for single suggestions:

```python
# ✓ CORRECT
X_next = optimizer.suggest(n_suggestions=1)
print(X_next.shape)  # (1, 10) - keeps batch dimension

# ✗ WRONG (not done in this library)
X_next = optimizer.suggest(n_suggestions=1)
X_next = X_next.squeeze(0)  # DON'T DO THIS
print(X_next.shape)  # (10,) - inconsistent!
```

Why? Consistency and compatibility:
- Easy to concatenate batches
- Compatible with vectorized objective functions
- Works seamlessly with PyTorch/NumPy operations
- No special cases for `n_suggestions=1`

## Usage Patterns

### Pattern 1: Sequential Optimization

```python
for iteration in range(100):
    # Always returns (1, input_dim)
    X_next = optimizer.suggest(n_suggestions=1)

    # Evaluate objective
    y_next = objective(X_next)  # Can handle (1, input_dim) input

    # Update
    optimizer.observe(X_next, y_next)
```

### Pattern 2: Batch Optimization

```python
# Get multiple suggestions at once
X_batch = optimizer.suggest(n_suggestions=10)
assert X_batch.shape == (10, input_dim)

# Evaluate in parallel
y_batch = torch.stack([objective(x) for x in X_batch])

# Or vectorized
y_batch = objective(X_batch)  # If objective supports batching

# Update with batch
optimizer.observe(X_batch, y_batch)
```

### Pattern 3: Concatenating Results

```python
# Collect suggestions from multiple optimizers
suggestions = []

for optimizer in [opt1, opt2, opt3]:
    X = optimizer.suggest(n_suggestions=2)
    suggestions.append(X)

# Easy to concatenate - all have same shape structure
all_suggestions = torch.cat(suggestions, dim=0)
print(all_suggestions.shape)  # (6, input_dim)
```

### Pattern 4: Indexing

```python
# Get batch
X_batch = optimizer.suggest(n_suggestions=5)

# Index individual points
x1 = X_batch[0]    # Shape: (input_dim,)
x2 = X_batch[1:3]  # Shape: (2, input_dim)

# Iterate
for x in X_batch:  # Each x has shape (input_dim,)
    print(x.shape)
```

## Testing Shape Consistency

We provide comprehensive tests to ensure shape consistency:

### Running Tests

```bash
# Run shape consistency tests
python test_shapes_manual.py

# Or with pytest
pytest tests/test_shape_consistency.py -v
```

### Example Test

```python
def test_consistency():
    """Verify all methods return same shape."""
    methods = [StandardBO, TuRBO, HesBO, ALEBO, REMBO]

    for Method in methods:
        optimizer = Method(input_dim=10, bounds=bounds)
        optimizer.observe(X_init, y_init)

        X = optimizer.suggest(n_suggestions=3)
        assert X.shape == (3, 10), f"{Method.__name__} failed"
```

## Common Pitfalls to Avoid

### ✗ Don't Assume Single-Element Shape

```python
# BAD - assumes squeezed output
X_next = optimizer.suggest(n_suggestions=1)
y_next = objective(X_next[0])  # Requires indexing

# GOOD - handles 2D shape correctly
X_next = optimizer.suggest(n_suggestions=1)
y_next = objective(X_next)  # Works with (1, input_dim)
```

### ✗ Don't Mix Shapes

```python
# BAD - inconsistent shapes
if n_suggestions == 1:
    X = optimizer.suggest(1).squeeze(0)  # (input_dim,)
else:
    X = optimizer.suggest(n_suggestions)  # (n_suggestions, input_dim)

# GOOD - always 2D
X = optimizer.suggest(n_suggestions)  # Always (n_suggestions, input_dim)
```

### ✗ Don't Manually Reshape

```python
# BAD - unnecessary reshaping
X = optimizer.suggest(n_suggestions=1)
X = X.reshape(-1)  # Changes to (input_dim,)

# GOOD - use as-is
X = optimizer.suggest(n_suggestions=1)  # Keep as (1, input_dim)
```

## Type Annotations

All methods have proper type hints:

```python
def suggest(self, n_suggestions: int = 1) -> torch.Tensor:
    """
    Returns:
        torch.Tensor: Shape (n_suggestions, input_dim)
    """
    pass
```

Use mypy for type checking:

```bash
mypy bo_methods/
```

## Design Rationale

### Why Always 2D?

1. **Consistency**: No special cases, easier to reason about
2. **Batching**: Natural support for batch operations
3. **Type Safety**: Single return type regardless of batch size
4. **Composability**: Easy to combine outputs from different methods
5. **PyTorch Convention**: Matches PyTorch's batch-first convention

### Why Not Optional Squeeze?

We considered but rejected:

```python
# Rejected design
def suggest(self, n_suggestions=1, squeeze=True):
    if n_suggestions == 1 and squeeze:
        return candidates.squeeze(0)  # ✗ Inconsistent
    return candidates  # ✗ Type depends on parameters
```

Problems:
- Type changes based on runtime values
- Easy to forget `squeeze` parameter
- Harder to compose and pipeline
- More error-prone

## Summary

**Key Points:**
- ✓ Always returns `(n_suggestions, input_dim)`
- ✓ Never squeezed, even for single suggestions
- ✓ Consistent across all 7 methods
- ✓ Runtime assertions catch shape errors
- ✓ Well-tested with comprehensive test suite

**Quick Reference:**
```python
# All of these are guaranteed to work
X = optimizer.suggest(n_suggestions=1)    # (1, input_dim)
X = optimizer.suggest(n_suggestions=5)    # (5, input_dim)
X = optimizer.suggest(n_suggestions=100)  # (100, input_dim)
```

No surprises, no edge cases, just consistent behavior! 🎯
