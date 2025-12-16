"""
Manual test script to verify shape consistency across all BO methods.
"""

import torch
import sys
sys.path.insert(0, '.')

from bo_methods import StandardBO, TuRBO, HesBO, ALEBO, REMBO, GPOEBO, GPOETRBO
from bo_methods.utils import get_initial_points


def test_method_shape(name, optimizer, input_dim, bounds, n_init=15):
    """Test a single method for shape consistency."""
    print(f"\nTesting {name}...")

    # Initialize with data
    X_init = get_initial_points(dim=input_dim, n_pts=n_init, bounds=bounds)
    y_init = torch.randn(n_init, 1)
    optimizer.observe(X_init, y_init)

    # Test single suggestion
    X_next = optimizer.suggest(n_suggestions=1)
    expected_shape = (1, input_dim)
    assert X_next.shape == expected_shape, \
        f"  ✗ Single suggestion: Expected {expected_shape}, got {X_next.shape}"
    print(f"  ✓ Single suggestion shape: {X_next.shape}")

    # Test batch suggestions
    X_next = optimizer.suggest(n_suggestions=5)
    expected_shape = (5, input_dim)
    assert X_next.shape == expected_shape, \
        f"  ✗ Batch suggestion: Expected {expected_shape}, got {X_next.shape}"
    print(f"  ✓ Batch suggestion shape: {X_next.shape}")

    print(f"  ✓ {name} passed all shape tests!")
    return True


def main():
    """Run all shape consistency tests."""
    print("="*60)
    print("SHAPE CONSISTENCY TESTS")
    print("="*60)

    # Setup
    input_dim = 10
    low_dim = 5
    bounds = torch.stack([
        torch.zeros(input_dim),
        torch.ones(input_dim)
    ])

    results = {}

    # Test StandardBO
    try:
        optimizer = StandardBO(input_dim=input_dim, bounds=bounds)
        results['StandardBO'] = test_method_shape('StandardBO', optimizer, input_dim, bounds)
    except Exception as e:
        print(f"  ✗ StandardBO failed: {e}")
        results['StandardBO'] = False

    # Test TuRBO
    try:
        optimizer = TuRBO(input_dim=input_dim, bounds=bounds, n_trust_regions=2, n_init=10)
        results['TuRBO'] = test_method_shape('TuRBO', optimizer, input_dim, bounds, n_init=10)
    except Exception as e:
        print(f"  ✗ TuRBO failed: {e}")
        results['TuRBO'] = False

    # Test HesBO
    try:
        optimizer = HesBO(input_dim=input_dim, low_dim=low_dim, bounds=bounds)
        results['HesBO'] = test_method_shape('HesBO', optimizer, input_dim, bounds)
    except Exception as e:
        print(f"  ✗ HesBO failed: {e}")
        results['HesBO'] = False

    # Test ALEBO
    try:
        optimizer = ALEBO(input_dim=input_dim, low_dim=low_dim, bounds=bounds)
        results['ALEBO'] = test_method_shape('ALEBO', optimizer, input_dim, bounds)
    except Exception as e:
        print(f"  ✗ ALEBO failed: {e}")
        results['ALEBO'] = False

    # Test REMBO
    try:
        optimizer = REMBO(input_dim=input_dim, low_dim=low_dim, bounds=bounds)
        results['REMBO'] = test_method_shape('REMBO', optimizer, input_dim, bounds)
    except Exception as e:
        print(f"  ✗ REMBO failed: {e}")
        results['REMBO'] = False

    # Test gPOE-BO
    try:
        optimizer = GPOEBO(
            input_dim=input_dim,
            bounds=bounds,
            points_per_expert=10,
            n_init=20
        )
        results['gPOE-BO'] = test_method_shape('gPOE-BO', optimizer, input_dim, bounds, n_init=20)
    except Exception as e:
        print(f"  ✗ gPOE-BO failed: {e}")
        results['gPOE-BO'] = False

    # Test gPOE-TuRBO
    try:
        optimizer = GPOETRBO(
            input_dim=input_dim,
            bounds=bounds,
            points_per_expert=10,
            n_init=20
        )
        results['gPOE-TuRBO'] = test_method_shape('gPOE-TuRBO', optimizer, input_dim, bounds, n_init=20)
    except Exception as e:
        print(f"  ✗ gPOE-TuRBO failed: {e}")
        results['gPOE-TuRBO'] = False

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    passed = sum(results.values())
    total = len(results)

    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8s} {name}")

    print("="*60)
    print(f"Passed: {passed}/{total}")
    print("="*60)

    if passed == total:
        print("\n✓ ALL TESTS PASSED - All methods return consistent shapes!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
