"""
Quick test to verify all HDBO methods can be imported and instantiated.
"""

import torch
from bo_methods.utils import get_initial_points


def test_imports():
    """Test that all methods can be imported."""
    print("Testing imports...")

    try:
        from bo_methods import (
            VanillaBO,
            RandomLineBO,
            BAxUS,
            SOLID,
            DSA
        )
        print("✓ All HDBO methods imported successfully!")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_instantiation():
    """Test that all methods can be instantiated."""
    print("\nTesting instantiation...")

    from bo_methods import VanillaBO, RandomLineBO, BAxUS, SOLID, DSA

    input_dim = 10
    bounds = torch.stack([
        torch.zeros(input_dim),
        torch.ones(input_dim)
    ])

    methods = {
        'VanillaBO': lambda: VanillaBO(input_dim, bounds),
        'RandomLineBO': lambda: RandomLineBO(input_dim, bounds),
        'BAxUS': lambda: BAxUS(input_dim, bounds, target_dim_init=2),
        'SOLID': lambda: SOLID(input_dim, bounds),
        'DSA': lambda: DSA(input_dim, bounds, n_active_dims=5),
    }

    all_success = True
    for name, constructor in methods.items():
        try:
            optimizer = constructor()
            print(f"✓ {name} instantiated successfully")
        except Exception as e:
            print(f"✗ {name} failed: {e}")
            all_success = False

    return all_success


def test_basic_workflow():
    """Test basic observe-suggest workflow."""
    print("\nTesting basic workflow...")

    from bo_methods import VanillaBO, RandomLineBO, BAxUS, SOLID, DSA

    input_dim = 10
    bounds = torch.stack([
        torch.zeros(input_dim),
        torch.ones(input_dim)
    ])

    # Simple quadratic function
    def objective(X):
        return -((X - 0.5) ** 2).sum(dim=-1, keepdim=True)

    methods = {
        'VanillaBO': VanillaBO(input_dim, bounds),
        'RandomLineBO': RandomLineBO(input_dim, bounds, n_lines_per_iter=3),
        'BAxUS': BAxUS(input_dim, bounds, target_dim_init=2),
        'SOLID': SOLID(input_dim, bounds),
        'DSA': DSA(input_dim, bounds, n_active_dims=5),
    }

    all_success = True

    for name, optimizer in methods.items():
        try:
            # Initial observations
            X_init = get_initial_points(dim=input_dim, n_pts=5, bounds=bounds)
            y_init = objective(X_init)

            # Observe
            optimizer.observe(X_init, y_init)

            # Suggest
            X_next = optimizer.suggest(n_suggestions=1)

            # Verify shape
            assert X_next.shape == (1, input_dim), f"Wrong shape: {X_next.shape}"

            # Observe again
            y_next = objective(X_next)
            optimizer.observe(X_next, y_next)

            # Get best
            best_x, best_y = optimizer.get_best_point()

            assert best_x.shape == (input_dim,), f"Wrong best_x shape: {best_x.shape}"
            assert best_y.shape == (1,), f"Wrong best_y shape: {best_y.shape}"

            print(f"✓ {name} workflow test passed")

        except Exception as e:
            print(f"✗ {name} workflow test failed: {e}")
            import traceback
            traceback.print_exc()
            all_success = False

    return all_success


def main():
    """Run all tests."""
    print("="*60)
    print("HDBO Methods Test Suite")
    print("="*60)

    results = []

    # Test imports
    results.append(("Imports", test_imports()))

    # Test instantiation
    results.append(("Instantiation", test_instantiation()))

    # Test basic workflow
    results.append(("Workflow", test_basic_workflow()))

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    all_passed = True
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:20s}: {status}")
        all_passed = all_passed and passed

    print("="*60)

    if all_passed:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠ Some tests failed!")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
