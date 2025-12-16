"""
Basic example demonstrating the usage of different BO methods
on a synthetic test function.
"""

import torch
import matplotlib.pyplot as plt
from typing import Dict, List

# Import BO methods
import sys
sys.path.append('..')
from bo_methods import StandardBO, TuRBO, HesBO, ALEBO, REMBO
from bo_methods.utils import get_initial_points


def ackley(X: torch.Tensor, a: float = 20, b: float = 0.2, c: float = 2 * torch.pi) -> torch.Tensor:
    """
    Ackley test function (minimization).

    Global minimum: f(0, ..., 0) = 0

    Args:
        X: Input tensor of shape (n, d)
        a, b, c: Function parameters

    Returns:
        Function values of shape (n,)
    """
    if len(X.shape) == 1:
        X = X.unsqueeze(0)

    d = X.shape[-1]
    sum1 = (X ** 2).sum(dim=-1)
    sum2 = torch.cos(c * X).sum(dim=-1)

    return -a * torch.exp(-b * torch.sqrt(sum1 / d)) - torch.exp(sum2 / d) + a + torch.e


def run_optimization(
    optimizer,
    objective_fn,
    bounds: torch.Tensor,
    n_init: int = 20,
    n_iterations: int = 100,
    verbose: bool = True
) -> List[float]:
    """
    Run Bayesian optimization with given optimizer.

    Args:
        optimizer: BO optimizer instance
        objective_fn: Objective function to minimize
        bounds: Bounds tensor
        n_init: Number of initial random points
        n_iterations: Number of optimization iterations
        verbose: Whether to print progress

    Returns:
        List of best values at each iteration
    """
    dim = bounds.shape[1]
    device = bounds.device
    dtype = bounds.dtype

    # Initial random sampling
    X_init = get_initial_points(
        dim=dim,
        n_pts=n_init,
        bounds=bounds,
        device=device,
        dtype=dtype
    )

    # Evaluate objective (negate for maximization)
    y_init = -objective_fn(X_init).reshape(-1, 1)

    # Update optimizer
    optimizer.observe(X_init, y_init)

    # Track best values
    best_values = []
    best_x, best_y = optimizer.get_best_point()
    best_values.append(best_y.item())

    if verbose:
        print(f"Initial best: {-best_y.item():.6f}")

    # Optimization loop
    for iteration in range(n_iterations):
        # Suggest next point
        X_next = optimizer.suggest(n_suggestions=1)

        # Evaluate objective (negate for maximization)
        y_next = -objective_fn(X_next).reshape(-1, 1)

        # Update optimizer
        optimizer.observe(X_next, y_next)

        # Track progress
        best_x, best_y = optimizer.get_best_point()
        best_values.append(best_y.item())

        if verbose and (iteration + 1) % 20 == 0:
            print(f"Iteration {iteration + 1}: Best = {-best_y.item():.6f}")

    return best_values


def plot_results(results: Dict[str, List[float]], save_path: str = "optimization_results.png"):
    """
    Plot optimization results for different methods.

    Args:
        results: Dictionary mapping method names to best values over iterations
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))

    for method_name, best_values in results.items():
        # Convert to actual objective values (we negated for maximization)
        obj_values = [-v for v in best_values]
        plt.plot(obj_values, label=method_name, linewidth=2)

    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Best Objective Value (Ackley)', fontsize=12)
    plt.title('Comparison of Bayesian Optimization Methods', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved to {save_path}")


def main():
    """Main function to run comparisons."""
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Problem setup
    input_dim = 10
    bounds = torch.stack([
        torch.full((input_dim,), -5.0),
        torch.full((input_dim,), 5.0)
    ])

    print(f"Optimizing {input_dim}D Ackley function")
    print(f"Bounds: [{bounds[0][0].item()}, {bounds[1][0].item()}]")
    print(f"Global minimum: f(0, ..., 0) = 0\n")

    # Initialize optimizers
    optimizers = {
        'StandardBO': StandardBO(
            input_dim=input_dim,
            bounds=bounds,
            acq_func='EI'
        ),
        'TuRBO': TuRBO(
            input_dim=input_dim,
            bounds=bounds,
            n_trust_regions=3,
            n_init=20
        ),
        'HesBO': HesBO(
            input_dim=input_dim,
            low_dim=5,
            bounds=bounds,
            acq_func='EI'
        ),
        'ALEBO': ALEBO(
            input_dim=input_dim,
            low_dim=5,
            bounds=bounds,
            acq_func='EI'
        ),
        'REMBO': REMBO(
            input_dim=input_dim,
            low_dim=5,
            bounds=bounds,
            acq_func='EI'
        )
    }

    # Run optimization for each method
    results = {}
    for method_name, optimizer in optimizers.items():
        print(f"\n{'='*60}")
        print(f"Running {method_name}")
        print(f"{'='*60}")

        best_values = run_optimization(
            optimizer=optimizer,
            objective_fn=ackley,
            bounds=bounds,
            n_init=20,
            n_iterations=100,
            verbose=True
        )

        results[method_name] = best_values

        # Print final result
        final_best = -best_values[-1]  # Convert back to minimization
        print(f"\n{method_name} Final Best: {final_best:.6f}")

    # Plot comparison
    print(f"\n{'='*60}")
    print("Generating comparison plot...")
    print(f"{'='*60}")
    plot_results(results)

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for method_name, best_values in results.items():
        final_best = -best_values[-1]
        print(f"{method_name:15s}: {final_best:.6f}")


if __name__ == "__main__":
    main()
