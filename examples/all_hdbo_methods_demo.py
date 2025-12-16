"""
Comprehensive demo of all 8 newly implemented HDBO methods.

This script demonstrates:
1. VanillaBO - Dimensionality-dependent priors
2. RandomLineBO - Random line optimization
3. BAxUS - Adaptive expanding subspaces
4. SOLID - Local variable selection
5. DSA - Dimension scheduling
6. HDS - Hierarchical diagonal sampling
7. Bounce - Mixed/combinatorial spaces
8. LSBO - Latent space BO with VAE
"""

import torch
import numpy as np
from typing import Callable
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from bo_methods import (
    VanillaBO, RandomLineBO, BAxUS, SOLID, DSA,
    HDS, Bounce, LSBO
)
from bo_methods.utils import get_initial_points


# ============================================================================
# Test Functions
# ============================================================================

def ackley(X: torch.Tensor, a: float = 20, b: float = 0.2, c: float = 2*np.pi) -> torch.Tensor:
    """Ackley function (to minimize, we negate for maximization)."""
    d = X.shape[-1]
    sum1 = (X ** 2).sum(dim=-1)
    sum2 = torch.cos(c * X).sum(dim=-1)
    result = -a * torch.exp(-b * torch.sqrt(sum1 / d)) - torch.exp(sum2 / d) + a + np.e
    return -result.unsqueeze(-1)  # Negate for maximization


def rosenbrock(X: torch.Tensor) -> torch.Tensor:
    """Rosenbrock function (negated for maximization)."""
    result = torch.zeros(X.shape[0])
    for i in range(X.shape[1] - 1):
        result += 100 * (X[:, i+1] - X[:, i]**2)**2 + (1 - X[:, i])**2
    return -result.unsqueeze(-1)


def sphere(X: torch.Tensor) -> torch.Tensor:
    """Sphere function (negated for maximization)."""
    return -((X - 0.5) ** 2).sum(dim=-1, keepdim=True)


# ============================================================================
# Helper Functions
# ============================================================================

def run_optimizer_demo(
    name: str,
    optimizer,
    objective: Callable,
    n_iterations: int = 50,
    verbose: bool = True
):
    """
    Run optimization and track performance.

    Args:
        name: Optimizer name
        optimizer: Optimizer instance
        objective: Objective function
        n_iterations: Number of iterations
        verbose: Print progress

    Returns:
        List of best values
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Testing {name}")
        print(f"{'='*60}")

    best_values = []

    for i in range(n_iterations):
        try:
            # Suggest next point
            X_next = optimizer.suggest(n_suggestions=1)

            # Evaluate
            y_next = objective(X_next)

            # Update
            optimizer.observe(X_next, y_next)

            # Track best
            best_x, best_y = optimizer.get_best_point()
            best_values.append(best_y.item())

            if verbose and (i + 1) % 10 == 0:
                print(f"  Iter {i+1:3d}: Best = {best_y.item():.6f}")

        except Exception as e:
            if verbose:
                print(f"  Error at iteration {i}: {e}")
            break

    if verbose:
        final_best = max(best_values) if best_values else float('-inf')
        print(f"  Final Best: {final_best:.6f}")

    return best_values


# ============================================================================
# Main Demo
# ============================================================================

def main():
    """Run comprehensive demo of all methods."""

    print("="*70)
    print("HDBO Methods Comprehensive Demo")
    print("="*70)

    # Problem setup
    input_dim = 30
    n_init = 15
    n_iterations = 50

    print(f"\nProblem Configuration:")
    print(f"  Input dimensionality: {input_dim}")
    print(f"  Initial samples: {n_init}")
    print(f"  Optimization iterations: {n_iterations}")
    print(f"  Test function: Ackley")

    # Define bounds
    bounds = torch.stack([
        torch.ones(input_dim) * -5.0,
        torch.ones(input_dim) * 5.0
    ])

    # Objective function
    objective = ackley

    # Generate initial data
    print("\nGenerating initial samples...")
    X_init = get_initial_points(dim=input_dim, n_pts=n_init, bounds=bounds)
    y_init = objective(X_init)

    results = {}

    # ========================================================================
    # 1. VanillaBO
    # ========================================================================
    print("\n\n" + "="*70)
    print("1/8: VanillaBO - Dimensionality-Dependent Priors")
    print("="*70)
    print("Description: Simple BO with smart priors that scale with dimension")

    optimizer = VanillaBO(
        input_dim=input_dim,
        bounds=bounds,
        acq_func='EI'
    )
    optimizer.observe(X_init.clone(), y_init.clone())
    results['VanillaBO'] = run_optimizer_demo('VanillaBO', optimizer, objective, n_iterations)

    # ========================================================================
    # 2. RandomLineBO
    # ========================================================================
    print("\n\n" + "="*70)
    print("2/8: RandomLineBO - Random Line Search")
    print("="*70)
    print("Description: Optimizes along random 1D lines in high-D space")

    optimizer = RandomLineBO(
        input_dim=input_dim,
        bounds=bounds,
        n_lines_per_iter=3,
        line_samples=50,
        use_anchor=True
    )
    optimizer.observe(X_init.clone(), y_init.clone())
    results['RandomLineBO'] = run_optimizer_demo('RandomLineBO', optimizer, objective, n_iterations)

    # ========================================================================
    # 3. BAxUS
    # ========================================================================
    print("\n\n" + "="*70)
    print("3/8: BAxUS - Adaptive Expanding Subspaces")
    print("="*70)
    print("Description: Starts with low dimensions and adaptively expands")

    optimizer = BAxUS(
        input_dim=input_dim,
        bounds=bounds,
        target_dim_init=2,
        target_dim_max=min(15, input_dim),
        expansion_factor=2
    )
    optimizer.observe(X_init.clone(), y_init.clone())
    results['BAxUS'] = run_optimizer_demo('BAxUS', optimizer, objective, n_iterations)

    # ========================================================================
    # 4. SOLID
    # ========================================================================
    print("\n\n" + "="*70)
    print("4/8: SOLID - Local Variable Selection")
    print("="*70)
    print("Description: Identifies globally and locally important dimensions")

    optimizer = SOLID(
        input_dim=input_dim,
        bounds=bounds,
        global_threshold=0.01,
        local_threshold=0.05,
        min_active_dims=3
    )
    optimizer.observe(X_init.clone(), y_init.clone())
    results['SOLID'] = run_optimizer_demo('SOLID', optimizer, objective, n_iterations)

    # ========================================================================
    # 5. DSA
    # ========================================================================
    print("\n\n" + "="*70)
    print("5/8: DSA - Dimension Scheduling")
    print("="*70)
    print("Description: Optimizes random dimension subsets with adaptive importance")

    optimizer = DSA(
        input_dim=input_dim,
        bounds=bounds,
        n_active_dims=max(5, int(input_dim ** 0.5)),
        use_pca=False,
        importance_update_freq=5
    )
    optimizer.observe(X_init.clone(), y_init.clone())
    results['DSA'] = run_optimizer_demo('DSA', optimizer, objective, n_iterations)

    # ========================================================================
    # 6. HDS
    # ========================================================================
    print("\n\n" + "="*70)
    print("6/8: HDS - Hierarchical Diagonal Sampling")
    print("="*70)
    print("Description: Uses binary tree of dimension subsets")

    optimizer = HDS(
        input_dim=input_dim,
        bounds=bounds,
        max_active_dims=max(10, int(np.sqrt(input_dim))),
        tree_depth=max(2, int(np.log2(input_dim)))
    )
    optimizer.observe(X_init.clone(), y_init.clone())
    results['HDS'] = run_optimizer_demo('HDS', optimizer, objective, n_iterations)

    # ========================================================================
    # 7. Bounce
    # ========================================================================
    print("\n\n" + "="*70)
    print("7/8: Bounce - Mixed/Combinatorial Spaces")
    print("="*70)
    print("Description: Nested binning for continuous/discrete variables")

    optimizer = Bounce(
        input_dim=input_dim,
        bounds=bounds,
        target_dim=max(5, int(np.sqrt(input_dim))),
        n_bins_per_target=2
    )
    optimizer.observe(X_init.clone(), y_init.clone())
    results['Bounce'] = run_optimizer_demo('Bounce', optimizer, objective, n_iterations)

    # ========================================================================
    # 8. LSBO
    # ========================================================================
    print("\n\n" + "="*70)
    print("8/8: LSBO - Latent Space BO with VAE")
    print("="*70)
    print("Description: Learns non-linear latent space via VAE")

    optimizer = LSBO(
        input_dim=input_dim,
        bounds=bounds,
        latent_dim=max(5, int(input_dim ** 0.5)),
        vae_hidden_dim=64,
        vae_train_steps=50,  # Reduced for demo
        vae_lr=0.001
    )
    optimizer.observe(X_init.clone(), y_init.clone())
    results['LSBO'] = run_optimizer_demo('LSBO', optimizer, objective, n_iterations//2)  # Fewer iters (VAE is slow)

    # ========================================================================
    # Final Comparison
    # ========================================================================
    print("\n\n" + "="*70)
    print("FINAL PERFORMANCE COMPARISON")
    print("="*70)

    print(f"\n{'Method':<20} {'Final Best':>12} {'Improvement':>12} {'Status':>10}")
    print("-" * 70)

    initial_best = y_init.max().item()

    sorted_results = sorted(
        results.items(),
        key=lambda x: max(x[1]) if x[1] else float('-inf'),
        reverse=True
    )

    for name, values in sorted_results:
        if values:
            final_best = max(values)
            improvement = final_best - initial_best
            status = "✓" if improvement > 0 else "✗"

            print(f"{name:<20} {final_best:>12.6f} {improvement:>+12.6f} {status:>10}")
        else:
            print(f"{name:<20} {'FAILED':>12} {'-':>12} {'✗':>10}")

    print("-" * 70)
    print(f"{'Initial Best':<20} {initial_best:>12.6f}")
    print(f"{'Ackley Optimum':<20} {0.0:>12.6f} (we negate to maximize)")

    # Best method
    if sorted_results:
        best_method, best_values = sorted_results[0]
        print(f"\n🏆 Best Method: {best_method} with final value {max(best_values):.6f}")

    print("\n" + "="*70)
    print("Demo Complete!")
    print("="*70)


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    main()
