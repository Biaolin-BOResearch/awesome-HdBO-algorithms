"""
Example usage of HDBO benchmark methods.

This script demonstrates the new high-dimensional Bayesian optimization
methods from the HDBO benchmark Figure 1 timeline.
"""

import torch
import numpy as np
from bo_methods import VanillaBO, RandomLineBO, BAxUS, SOLID, DSA
from bo_methods.utils import get_initial_points


def ackley(X: torch.Tensor) -> torch.Tensor:
    """
    Ackley test function (to be minimized, we negate for maximization).

    Args:
        X: Tensor of shape (n, d)

    Returns:
        Tensor of shape (n, 1) with function values (negated for maximization)
    """
    a, b, c = 20, 0.2, 2 * torch.pi
    d = X.shape[-1]

    sum1 = (X ** 2).sum(dim=-1)
    sum2 = torch.cos(c * X).sum(dim=-1)

    result = -a * torch.exp(-b * torch.sqrt(sum1 / d)) - torch.exp(sum2 / d) + a + torch.e

    # Negate for maximization (BO maximizes)
    return -result.unsqueeze(-1)


def run_optimization(optimizer_name: str, optimizer, n_iterations: int = 100):
    """
    Run optimization loop for a given optimizer.

    Args:
        optimizer_name: Name of the optimizer for display
        optimizer: Optimizer instance
        n_iterations: Number of optimization iterations

    Returns:
        List of best values found at each iteration
    """
    print(f"\n{'='*60}")
    print(f"Running {optimizer_name}")
    print(f"{'='*60}")

    best_values = []

    for i in range(n_iterations):
        # Suggest next point
        X_next = optimizer.suggest(n_suggestions=1)

        # Evaluate objective
        y_next = ackley(X_next)

        # Update optimizer
        optimizer.observe(X_next, y_next)

        # Track best value
        best_x, best_y = optimizer.get_best_point()
        best_values.append(best_y.item())

        if (i + 1) % 20 == 0:
            print(f"Iteration {i+1:3d}: Best value = {best_y.item():.6f}")

    final_best = max(best_values)
    print(f"\nFinal best value: {final_best:.6f}")
    print(f"(Ackley optimum is 0, we found {-final_best:.6f})")

    return best_values


def main():
    """Main function to demonstrate all HDBO methods."""

    # Problem setup
    input_dim = 50
    n_init = 20
    n_iterations = 100

    # Define bounds
    bounds = torch.stack([
        torch.ones(input_dim) * -5.0,
        torch.ones(input_dim) * 5.0
    ])

    print(f"Optimizing {input_dim}-dimensional Ackley function")
    print(f"Initial samples: {n_init}")
    print(f"Optimization iterations: {n_iterations}")

    # Generate initial data
    X_init = get_initial_points(dim=input_dim, n_pts=n_init, bounds=bounds)
    y_init = ackley(X_init)

    results = {}

    # 1. VanillaBO
    print("\n" + "="*60)
    print("1. VanillaBO - Dimensionality-dependent priors")
    print("="*60)

    optimizer = VanillaBO(
        input_dim=input_dim,
        bounds=bounds,
        acq_func='EI'
    )
    optimizer.observe(X_init, y_init)
    results['VanillaBO'] = run_optimization('VanillaBO', optimizer, n_iterations)

    # 2. RandomLineBO
    print("\n" + "="*60)
    print("2. RandomLineBO - Optimization along random lines")
    print("="*60)

    optimizer = RandomLineBO(
        input_dim=input_dim,
        bounds=bounds,
        n_lines_per_iter=5,
        line_samples=100
    )
    optimizer.observe(X_init.clone(), y_init.clone())
    results['RandomLineBO'] = run_optimization('RandomLineBO', optimizer, n_iterations)

    # 3. BAxUS
    print("\n" + "="*60)
    print("3. BAxUS - Adaptively expanding subspaces")
    print("="*60)

    optimizer = BAxUS(
        input_dim=input_dim,
        bounds=bounds,
        target_dim_init=2,
        target_dim_max=min(20, input_dim),
        n_init=n_init
    )
    optimizer.observe(X_init.clone(), y_init.clone())
    results['BAxUS'] = run_optimization('BAxUS', optimizer, n_iterations)

    # 4. SOLID
    print("\n" + "="*60)
    print("4. SOLID - Local variable selection")
    print("="*60)

    optimizer = SOLID(
        input_dim=input_dim,
        bounds=bounds,
        local_threshold=0.05,
        global_threshold=0.01
    )
    optimizer.observe(X_init.clone(), y_init.clone())
    results['SOLID'] = run_optimization('SOLID', optimizer, n_iterations)

    # 5. DSA
    print("\n" + "="*60)
    print("5. DSA - Dimension scheduling")
    print("="*60)

    optimizer = DSA(
        input_dim=input_dim,
        bounds=bounds,
        n_active_dims=int(input_dim ** 0.5),
        use_pca=False
    )
    optimizer.observe(X_init.clone(), y_init.clone())
    results['DSA'] = run_optimization('DSA', optimizer, n_iterations)

    # Print comparison
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)

    for name, values in results.items():
        final_value = max(values)
        print(f"{name:15s}: {final_value:.6f} (Ackley error: {-final_value:.6f})")

    # Find best method
    best_method = max(results.items(), key=lambda x: max(x[1]))
    print(f"\nBest method: {best_method[0]} with value {max(best_method[1]):.6f}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    main()
