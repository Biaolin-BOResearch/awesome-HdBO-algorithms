"""
Test shape consistency across all BO methods.

Ensures that all methods return candidates with consistent shape (n_suggestions, input_dim).
"""

import pytest
import torch
from bo_methods import StandardBO, TuRBO, HesBO, ALEBO, REMBO, GPOEBO, GPOETRBO
from bo_methods.utils import get_initial_points


class TestShapeConsistency:
    """Test cases for shape consistency across all methods."""

    @pytest.fixture
    def setup_low_dim(self):
        """Setup for low-dimensional test."""
        input_dim = 5
        bounds = torch.stack([
            torch.zeros(input_dim),
            torch.ones(input_dim)
        ])
        return input_dim, bounds

    @pytest.fixture
    def setup_high_dim(self):
        """Setup for high-dimensional test."""
        input_dim = 50
        bounds = torch.stack([
            torch.zeros(input_dim),
            torch.ones(input_dim)
        ])
        return input_dim, bounds

    def test_standard_bo_shape(self, setup_low_dim):
        """Test StandardBO output shape."""
        input_dim, bounds = setup_low_dim
        optimizer = StandardBO(input_dim=input_dim, bounds=bounds)

        # Initialize with data
        X_init = get_initial_points(dim=input_dim, n_pts=10, bounds=bounds)
        y_init = torch.randn(10, 1)
        optimizer.observe(X_init, y_init)

        # Test single suggestion
        X_next = optimizer.suggest(n_suggestions=1)
        assert X_next.shape == (1, input_dim), \
            f"Expected (1, {input_dim}), got {X_next.shape}"

        # Test batch suggestions
        X_next = optimizer.suggest(n_suggestions=5)
        assert X_next.shape == (5, input_dim), \
            f"Expected (5, {input_dim}), got {X_next.shape}"

    def test_turbo_shape(self, setup_high_dim):
        """Test TuRBO output shape."""
        input_dim, bounds = setup_high_dim
        optimizer = TuRBO(
            input_dim=input_dim,
            bounds=bounds,
            n_trust_regions=3,
            n_init=20
        )

        # Test during initialization phase
        X_next = optimizer.suggest(n_suggestions=1)
        assert X_next.shape == (1, input_dim), \
            f"Expected (1, {input_dim}), got {X_next.shape}"

        # Initialize with data
        X_init = get_initial_points(dim=input_dim, n_pts=20, bounds=bounds)
        y_init = torch.randn(20, 1)
        optimizer.observe(X_init, y_init)

        # Test single suggestion
        X_next = optimizer.suggest(n_suggestions=1)
        assert X_next.shape == (1, input_dim), \
            f"Expected (1, {input_dim}), got {X_next.shape}"

        # Test batch suggestions
        X_next = optimizer.suggest(n_suggestions=3)
        assert X_next.shape == (3, input_dim), \
            f"Expected (3, {input_dim}), got {X_next.shape}"

    def test_hesbo_shape(self, setup_high_dim):
        """Test HesBO output shape."""
        input_dim, bounds = setup_high_dim
        low_dim = 10

        optimizer = HesBO(
            input_dim=input_dim,
            low_dim=low_dim,
            bounds=bounds
        )

        # Initialize with data
        X_init = get_initial_points(dim=input_dim, n_pts=15, bounds=bounds)
        y_init = torch.randn(15, 1)
        optimizer.observe(X_init, y_init)

        # Test single suggestion
        X_next = optimizer.suggest(n_suggestions=1)
        assert X_next.shape == (1, input_dim), \
            f"Expected (1, {input_dim}), got {X_next.shape}"

        # Test batch suggestions
        X_next = optimizer.suggest(n_suggestions=4)
        assert X_next.shape == (4, input_dim), \
            f"Expected (4, {input_dim}), got {X_next.shape}"

    def test_alebo_shape(self, setup_high_dim):
        """Test ALEBO output shape."""
        input_dim, bounds = setup_high_dim
        low_dim = 10

        optimizer = ALEBO(
            input_dim=input_dim,
            low_dim=low_dim,
            bounds=bounds
        )

        # Initialize with data
        X_init = get_initial_points(dim=input_dim, n_pts=15, bounds=bounds)
        y_init = torch.randn(15, 1)
        optimizer.observe(X_init, y_init)

        # Test single suggestion
        X_next = optimizer.suggest(n_suggestions=1)
        assert X_next.shape == (1, input_dim), \
            f"Expected (1, {input_dim}), got {X_next.shape}"

        # Test batch suggestions
        X_next = optimizer.suggest(n_suggestions=4)
        assert X_next.shape == (4, input_dim), \
            f"Expected (4, {input_dim}), got {X_next.shape}"

    def test_rembo_shape(self, setup_high_dim):
        """Test REMBO output shape."""
        input_dim, bounds = setup_high_dim
        low_dim = 8

        optimizer = REMBO(
            input_dim=input_dim,
            low_dim=low_dim,
            bounds=bounds
        )

        # Initialize with data
        X_init = get_initial_points(dim=input_dim, n_pts=15, bounds=bounds)
        y_init = torch.randn(15, 1)
        optimizer.observe(X_init, y_init)

        # Test single suggestion
        X_next = optimizer.suggest(n_suggestions=1)
        assert X_next.shape == (1, input_dim), \
            f"Expected (1, {input_dim}), got {X_next.shape}"

        # Test batch suggestions
        X_next = optimizer.suggest(n_suggestions=4)
        assert X_next.shape == (4, input_dim), \
            f"Expected (4, {input_dim}), got {X_next.shape}"

    def test_gpoe_bo_shape(self, setup_high_dim):
        """Test gPOE-BO output shape."""
        input_dim, bounds = setup_high_dim

        optimizer = GPOEBO(
            input_dim=input_dim,
            bounds=bounds,
            points_per_expert=10,
            n_init=20
        )

        # Test during initialization phase
        X_next = optimizer.suggest(n_suggestions=1)
        assert X_next.shape == (1, input_dim), \
            f"Expected (1, {input_dim}), got {X_next.shape}"

        # Initialize with data
        X_init = get_initial_points(dim=input_dim, n_pts=30, bounds=bounds)
        y_init = torch.randn(30, 1)
        optimizer.observe(X_init, y_init)

        # Test single suggestion
        X_next = optimizer.suggest(n_suggestions=1)
        assert X_next.shape == (1, input_dim), \
            f"Expected (1, {input_dim}), got {X_next.shape}"

        # Test batch suggestions
        X_next = optimizer.suggest(n_suggestions=3)
        assert X_next.shape == (3, input_dim), \
            f"Expected (3, {input_dim}), got {X_next.shape}"

    def test_gpoe_turbo_shape(self, setup_high_dim):
        """Test gPOE-TuRBO output shape."""
        input_dim, bounds = setup_high_dim

        optimizer = GPOETRBO(
            input_dim=input_dim,
            bounds=bounds,
            points_per_expert=10,
            n_init=20
        )

        # Test during initialization phase
        X_next = optimizer.suggest(n_suggestions=1)
        assert X_next.shape == (1, input_dim), \
            f"Expected (1, {input_dim}), got {X_next.shape}"

        # Initialize with data
        X_init = get_initial_points(dim=input_dim, n_pts=30, bounds=bounds)
        y_init = torch.randn(30, 1)
        optimizer.observe(X_init, y_init)

        # Test single suggestion
        X_next = optimizer.suggest(n_suggestions=1)
        assert X_next.shape == (1, input_dim), \
            f"Expected (1, {input_dim}), got {X_next.shape}"

        # Test batch suggestions
        X_next = optimizer.suggest(n_suggestions=3)
        assert X_next.shape == (3, input_dim), \
            f"Expected (3, {input_dim}), got {X_next.shape}"

    def test_all_methods_consistency(self, setup_low_dim):
        """Test that all methods return consistent shapes."""
        input_dim, bounds = setup_low_dim
        n_suggestions = 3

        # Create all optimizers
        optimizers = {
            'StandardBO': StandardBO(input_dim=input_dim, bounds=bounds),
            'TuRBO': TuRBO(input_dim=input_dim, bounds=bounds, n_init=10),
            'HesBO': HesBO(input_dim=input_dim, low_dim=3, bounds=bounds),
            'ALEBO': ALEBO(input_dim=input_dim, low_dim=3, bounds=bounds),
            'REMBO': REMBO(input_dim=input_dim, low_dim=3, bounds=bounds),
        }

        # Initialize all with same data
        X_init = get_initial_points(dim=input_dim, n_pts=15, bounds=bounds)
        y_init = torch.randn(15, 1)

        shapes = {}
        for name, optimizer in optimizers.items():
            optimizer.observe(X_init, y_init)
            X_next = optimizer.suggest(n_suggestions=n_suggestions)
            shapes[name] = X_next.shape

        # Check all have same shape
        expected_shape = (n_suggestions, input_dim)
        for name, shape in shapes.items():
            assert shape == expected_shape, \
                f"{name} returned shape {shape}, expected {expected_shape}"

        print(f"✓ All methods return consistent shape: {expected_shape}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
