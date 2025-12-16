"""
Unit tests for Standard Bayesian Optimization.
"""

import pytest
import torch
from bo_methods import StandardBO
from bo_methods.utils import get_initial_points


class TestStandardBO:
    """Test cases for StandardBO class."""

    @pytest.fixture
    def setup(self):
        """Setup test fixtures."""
        input_dim = 5
        bounds = torch.stack([
            torch.zeros(input_dim),
            torch.ones(input_dim)
        ])
        return input_dim, bounds

    def test_initialization(self, setup):
        """Test optimizer initialization."""
        input_dim, bounds = setup

        optimizer = StandardBO(
            input_dim=input_dim,
            bounds=bounds,
            acq_func='EI'
        )

        assert optimizer.input_dim == input_dim
        assert optimizer.acq_func == 'EI'
        assert optimizer.train_X.shape == (0, input_dim)
        assert optimizer.train_y.shape == (0, 1)
        assert optimizer.model is None

    def test_observe(self, setup):
        """Test observation update."""
        input_dim, bounds = setup
        optimizer = StandardBO(input_dim=input_dim, bounds=bounds)

        # Generate observations
        X = torch.rand(5, input_dim)
        y = torch.randn(5, 1)

        # Update optimizer
        optimizer.observe(X, y)

        assert optimizer.train_X.shape == (5, input_dim)
        assert optimizer.train_y.shape == (5, 1)
        assert optimizer.model is not None

    def test_suggest(self, setup):
        """Test suggestion generation."""
        input_dim, bounds = setup
        optimizer = StandardBO(input_dim=input_dim, bounds=bounds)

        # Need observations before suggesting
        X = torch.rand(10, input_dim)
        y = torch.randn(10, 1)
        optimizer.observe(X, y)

        # Suggest next points
        X_next = optimizer.suggest(n_suggestions=3)

        assert X_next.shape == (3, input_dim)
        assert (X_next >= bounds[0]).all()
        assert (X_next <= bounds[1]).all()

    def test_get_best_point(self, setup):
        """Test retrieval of best observed point."""
        input_dim, bounds = setup
        optimizer = StandardBO(input_dim=input_dim, bounds=bounds)

        # Add observations
        X = torch.rand(10, input_dim)
        y = torch.randn(10, 1)
        optimizer.observe(X, y)

        # Get best point
        best_x, best_y = optimizer.get_best_point()

        assert best_x.shape == (input_dim,)
        assert best_y.item() == y.max().item()

    def test_reset(self, setup):
        """Test optimizer reset."""
        input_dim, bounds = setup
        optimizer = StandardBO(input_dim=input_dim, bounds=bounds)

        # Add observations
        X = torch.rand(5, input_dim)
        y = torch.randn(5, 1)
        optimizer.observe(X, y)

        # Reset
        optimizer.reset()

        assert optimizer.train_X.shape == (0, input_dim)
        assert optimizer.train_y.shape == (0, 1)
        assert optimizer.model is None

    def test_different_acq_functions(self, setup):
        """Test different acquisition functions."""
        input_dim, bounds = setup

        for acq_func in ['EI', 'PI', 'UCB']:
            optimizer = StandardBO(
                input_dim=input_dim,
                bounds=bounds,
                acq_func=acq_func
            )

            X = torch.rand(10, input_dim)
            y = torch.randn(10, 1)
            optimizer.observe(X, y)

            X_next = optimizer.suggest()
            assert X_next.shape == (1, input_dim)

    def test_batch_observations(self, setup):
        """Test batch observation updates."""
        input_dim, bounds = setup
        optimizer = StandardBO(input_dim=input_dim, bounds=bounds)

        # First batch
        X1 = torch.rand(5, input_dim)
        y1 = torch.randn(5, 1)
        optimizer.observe(X1, y1)

        # Second batch
        X2 = torch.rand(3, input_dim)
        y2 = torch.randn(3, 1)
        optimizer.observe(X2, y2)

        assert optimizer.train_X.shape == (8, input_dim)
        assert optimizer.train_y.shape == (8, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
