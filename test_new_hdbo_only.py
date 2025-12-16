"""
Quick test to verify new HDBO methods work correctly.
This bypasses the old base.py import issues.
"""

import torch
import sys


def simple_objective(X):
    """Simple quadratic function for testing."""
    return -((X - 0.5) ** 2).sum(dim=-1, keepdim=True)


def get_initial_points_simple(dim, n_pts, bounds):
    """Simple random initial points generator."""
    lower = bounds[0]
    upper = bounds[1]
    return torch.rand(n_pts, dim) * (upper - lower) + lower


def test_method(name, optimizer_class, optimizer_kwargs, input_dim=10):
    """Test a single optimizer."""
    print(f"\nTesting {name}...")

    bounds = torch.stack([
        torch.zeros(input_dim),
        torch.ones(input_dim)
    ])

    try:
        # Create optimizer
        optimizer = optimizer_class(input_dim, bounds, **optimizer_kwargs)
        print(f"  ✓ Instantiated")

        # Initial observations
        X_init = get_initial_points_simple(input_dim, 5, bounds)
        y_init = simple_objective(X_init)

        # Observe
        optimizer.observe(X_init, y_init)
        print(f"  ✓ Initial observe()")

        # Suggest
        X_next = optimizer.suggest(n_suggestions=1)
        assert X_next.shape == (1, input_dim), f"Wrong shape: {X_next.shape}"
        print(f"  ✓ suggest() returned shape {X_next.shape}")

        # Observe again
        y_next = simple_objective(X_next)
        optimizer.observe(X_next, y_next)
        print(f"  ✓ Second observe()")

        # Get best
        best_x, best_y = optimizer.get_best_point()
        assert best_x.shape == (input_dim,), f"Wrong best_x shape"
        assert best_y.shape == (1,), f"Wrong best_y shape"
        print(f"  ✓ get_best_point() works, best_y = {best_y.item():.4f}")

        print(f"✅ {name} PASSED")
        return True

    except Exception as e:
        print(f"❌ {name} FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Test all new HDBO methods."""
    print("="*60)
    print("Testing New HDBO Methods")
    print("="*60)

    # Import directly from files
    sys.path.insert(0, 'bo_methods')

    from vanilla_bo import VanillaBO
    from random_line_bo import RandomLineBO
    from baxus import BAxUS
    from solid import SOLID
    from dsa import DSA

    input_dim = 20

    tests = [
        ("VanillaBO", VanillaBO, {}),
        ("RandomLineBO", RandomLineBO, {'n_lines_per_iter': 3, 'line_samples': 50}),
        ("BAxUS", BAxUS, {'target_dim_init': 2, 'target_dim_max': 10}),
        ("SOLID", SOLID, {}),
        ("DSA", DSA, {'n_active_dims': 5}),
    ]

    results = []
    for name, cls, kwargs in tests:
        passed = test_method(name, cls, kwargs, input_dim)
        results.append((name, passed))

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    all_passed = True
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name:20s}: {status}")
        all_passed = all_passed and passed

    print("="*60)

    if all_passed:
        print("\n🎉 All new HDBO methods work correctly!")
        return 0
    else:
        print("\n⚠️  Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
