"""
Verbalized Bayesian Optimization (VBO) Main Optimizer

Integrates all VBO components into a unified optimizer that is compatible
with the BaseLLMOptimizer interface.

Configuration options:
- Surrogate Mode: in_context / belief_based
- Surrogate Output: text_only / text_with_numeric
- Acquisition Output: text_only / text_with_numeric
- Optimization Method: random_search / grid_search / single_start_tga / multi_start_tga
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from .surrogate import (
    VBOSurrogateModel,
    InContextSurrogate,
    BeliefBasedSurrogate,
    SurrogateMode,
    OutputMode,
    SurrogatePrediction,
    create_surrogate,
)

from .acquisition import (
    VBOAcquisitionFunction,
    AcquisitionOutputMode,
    AcquisitionEvaluation,
    TextGradOptimizableAcquisition,
)

from .optimization import (
    AcquisitionOptimizer,
    OptimizationMethod,
    OptimizationStats,
    create_optimizer,
)


@dataclass
class VBOConfig:
    """
    Configuration for Verbalized Bayesian Optimization.
    
    Attributes:
        surrogate_mode: in_context or belief_based
        surrogate_output: text_only or text_with_numeric
        acquisition_output: text_only or text_with_numeric  
        optimization_method: random_search, grid_search, single_start_tga, or multi_start_tga
        use_textgrad_for_acquisition: Whether to optimize acquisition strategy with TextGrad
        textgrad_update_interval: How often to update acquisition strategy (in iterations)
    """
    surrogate_mode: SurrogateMode = SurrogateMode.IN_CONTEXT
    surrogate_output: OutputMode = OutputMode.TEXT_WITH_NUMERIC
    acquisition_output: AcquisitionOutputMode = AcquisitionOutputMode.TEXT_WITH_NUMERIC
    optimization_method: OptimizationMethod = OptimizationMethod.RANDOM_SEARCH
    
    # TextGrad for acquisition strategy
    use_textgrad_for_acquisition: bool = False
    textgrad_update_interval: int = 5
    
    # Optimization method parameters
    n_candidates: int = 20  # For random search
    n_per_dim: int = 5  # For grid search
    n_steps: int = 10  # For TGA
    n_starts: int = 5  # For multi-start TGA
    step_size: float = 0.1  # For TGA
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "surrogate_mode": self.surrogate_mode.value,
            "surrogate_output": self.surrogate_output.value,
            "acquisition_output": self.acquisition_output.value,
            "optimization_method": self.optimization_method.value,
            "use_textgrad_for_acquisition": self.use_textgrad_for_acquisition,
            "textgrad_update_interval": self.textgrad_update_interval,
            "n_candidates": self.n_candidates,
            "n_per_dim": self.n_per_dim,
            "n_steps": self.n_steps,
            "n_starts": self.n_starts,
            "step_size": self.step_size,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VBOConfig":
        return cls(
            surrogate_mode=SurrogateMode(data.get("surrogate_mode", "in_context")),
            surrogate_output=OutputMode(data.get("surrogate_output", "text_with_numeric")),
            acquisition_output=AcquisitionOutputMode(data.get("acquisition_output", "text_with_numeric")),
            optimization_method=OptimizationMethod(data.get("optimization_method", "random_search")),
            use_textgrad_for_acquisition=data.get("use_textgrad_for_acquisition", False),
            textgrad_update_interval=data.get("textgrad_update_interval", 5),
            n_candidates=data.get("n_candidates", 20),
            n_per_dim=data.get("n_per_dim", 5),
            n_steps=data.get("n_steps", 10),
            n_starts=data.get("n_starts", 5),
            step_size=data.get("step_size", 0.1),
        )


class VBOptimizer:
    """
    Verbalized Bayesian Optimization Optimizer.
    
    A fully LLM-driven Bayesian optimization algorithm that uses:
    - LLM as surrogate model (with configurable belief strategy)
    - LLM as acquisition function (with configurable output)
    - LLM for acquisition optimization (including Textual Gradient Ascent)
    
    All components are modular and can be freely combined.
    """
    
    def __init__(
        self,
        input_dim: int,
        bounds: Tensor,
        llm_client: Any,
        model_name: str = "",
        task_description: str = "",
        maximize: bool = False,
        config: Optional[VBOConfig] = None,
        param_names: Optional[List[str]] = None,
        param_space_description: str = "",
        **kwargs,
    ):
        """
        Initialize VBO optimizer.
        
        Args:
            input_dim: Dimensionality of the search space
            bounds: Tensor of shape (2, input_dim) with [lower, upper] bounds
            llm_client: LLM client for all queries
            model_name: Name of the LLM model
            task_description: Description of the optimization task
            maximize: Whether to maximize (True) or minimize (False)
            config: VBO configuration
            param_names: Optional list of parameter names
            param_space_description: Description of parameter space (for belief-based)
            **kwargs: Additional configuration overrides
        """
        self.input_dim = input_dim
        self.bounds = bounds
        self.llm_client = llm_client
        self.model_name = model_name or getattr(llm_client, 'model_name', 'unknown')
        self.task_description = task_description
        self.maximize = maximize
        self.param_names = param_names or [f"x{i}" for i in range(input_dim)]
        self.param_space_description = param_space_description
        
        # Configuration
        self.config = config or VBOConfig()
        
        # Apply kwargs overrides with proper type conversion
        enum_mappings = {
            'surrogate_mode': SurrogateMode,
            'surrogate_output': OutputMode,
            'acquisition_output': AcquisitionOutputMode,
            'optimization_method': OptimizationMethod,
        }
        
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                # Convert strings to enums if needed
                if key in enum_mappings and isinstance(value, str):
                    value = enum_mappings[key](value)
                setattr(self.config, key, value)
        
        # State
        self.X: Optional[Tensor] = None
        self.y: Optional[Tensor] = None
        self.iteration = 0
        self.best_value: Optional[float] = None
        self.best_x: Optional[Tensor] = None
        
        # Initialize components
        self._init_surrogate()
        self._init_acquisition()
        self._init_optimizer()
        
        # Tracking
        self.optimization_stats_history: List[Dict] = []
        self.performance_history: List[Dict] = []
    
    def _init_surrogate(self) -> None:
        """Initialize surrogate model."""
        self.surrogate = create_surrogate(
            mode=self.config.surrogate_mode,
            llm_client=self.llm_client,
            output_mode=self.config.surrogate_output,
            task_description=self.task_description,
            maximize=self.maximize,
            param_space_description=self.param_space_description,
        )
    
    def _init_acquisition(self) -> None:
        """Initialize acquisition function."""
        if self.config.use_textgrad_for_acquisition:
            self.acquisition = TextGradOptimizableAcquisition(
                llm_client=self.llm_client,
                output_mode=self.config.acquisition_output,
                task_description=self.task_description,
                maximize=self.maximize,
            )
        else:
            self.acquisition = VBOAcquisitionFunction(
                llm_client=self.llm_client,
                output_mode=self.config.acquisition_output,
                task_description=self.task_description,
                maximize=self.maximize,
            )
    
    def _init_optimizer(self) -> None:
        """Initialize acquisition optimizer."""
        self.acq_optimizer = create_optimizer(
            method=self.config.optimization_method,
            bounds=self.bounds,
            llm_client=self.llm_client,
            n_candidates=self.config.n_candidates,
            n_per_dim=self.config.n_per_dim,
            n_steps=self.config.n_steps,
            n_starts=self.config.n_starts,
            step_size=self.config.step_size,
        )
    
    def suggest(self, n_candidates: int = 1) -> Tensor:
        """
        Suggest next point(s) to evaluate.
        
        Args:
            n_candidates: Number of candidates to suggest
            
        Returns:
            Tensor of shape (n_candidates, input_dim)
        """
        if self.X is None or len(self.X) == 0:
            # No observations yet, return random point
            return self.random_points(n_candidates)
        
        # Get current best
        best_value = self._get_best_value()
        n_observations = len(self.X)
        
        # Optimize acquisition
        best_x, stats = self.acq_optimizer.optimize(
            surrogate=self.surrogate,
            acquisition=self.acquisition,
            best_value=best_value,
            n_observations=n_observations,
            iteration=self.iteration,
            total_iterations=100,  # Assumed default
        )
        
        # Track stats
        self.optimization_stats_history.append(stats.to_dict())
        
        return best_x.unsqueeze(0) if n_candidates == 1 else best_x.unsqueeze(0).repeat(n_candidates, 1)
    
    def update(self, X: Tensor, y: Tensor) -> None:
        """
        Update with new observations.
        
        Args:
            X: New points, shape (n, input_dim)
            y: New values, shape (n, 1) or (n,)
        """
        # Ensure correct shapes
        if X.dim() == 1:
            X = X.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(1)
        
        # Update internal state
        if self.X is None:
            self.X = X.clone()
            self.y = y.clone()
        else:
            self.X = torch.cat([self.X, X], dim=0)
            self.y = torch.cat([self.y, y], dim=0)
        
        # Update surrogate with each new observation
        for i in range(len(X)):
            x_i = X[i]
            y_i = y[i].item()
            self.surrogate.update(x_i, y_i)
            
            # Track performance for TextGrad
            improved = self._check_improvement(y_i)
            self.performance_history.append({
                'x': x_i.tolist(),
                'y': y_i,
                'improved': improved,
                'iteration': self.iteration,
            })
        
        # Update best
        self._update_best()
        
        # Increment iteration
        self.iteration += 1
        
        # Update acquisition strategy if using TextGrad
        if (self.config.use_textgrad_for_acquisition and 
            self.iteration > 0 and 
            self.iteration % self.config.textgrad_update_interval == 0):
            self._update_acquisition_strategy()
    
    def initialize(self, X: Tensor, y: Tensor) -> None:
        """
        Initialize with existing observations.
        
        Args:
            X: Initial points
            y: Initial values
        """
        self.update(X, y)
    
    def random_points(self, n: int) -> Tensor:
        """Generate n random points within bounds."""
        return (
            self.bounds[0] + 
            torch.rand(n, self.input_dim, dtype=torch.float64) * 
            (self.bounds[1] - self.bounds[0])
        )
    
    def warmstart(self, n_points: int, objective_func: Any = None) -> Tensor:
        """
        Generate warmstart points using LLM.
        
        For VBO, we use the surrogate model's initial belief to suggest
        promising starting points.
        
        Args:
            n_points: Number of warmstart points
            objective_func: Optional objective function (not used here)
            
        Returns:
            Tensor of initial points
        """
        # For belief-based surrogate, the belief is already initialized
        # Generate diverse points using acquisition optimization
        points = []
        
        for _ in range(n_points):
            point = self.random_points(1).squeeze(0)
            points.append(point)
        
        return torch.stack(points)
    
    def _get_best_value(self) -> float:
        """Get current best observed value."""
        if self.best_value is not None:
            return self.best_value
        if self.y is not None and len(self.y) > 0:
            if self.maximize:
                return self.y.max().item()
            else:
                return self.y.min().item()
        return 0.0
    
    def _update_best(self) -> None:
        """Update best observed point and value."""
        if self.y is None or len(self.y) == 0:
            return
        
        if self.maximize:
            idx = self.y.argmax()
        else:
            idx = self.y.argmin()
        
        self.best_value = self.y[idx].item()
        self.best_x = self.X[idx].clone()
    
    def _check_improvement(self, y: float) -> bool:
        """Check if new value is an improvement."""
        if self.best_value is None:
            return True
        if self.maximize:
            return y > self.best_value
        else:
            return y < self.best_value
    
    def _update_acquisition_strategy(self) -> None:
        """Update acquisition strategy using TextGrad."""
        if not isinstance(self.acquisition, TextGradOptimizableAcquisition):
            return
        
        # Get recent performance data
        recent = self.performance_history[-self.config.textgrad_update_interval:]
        
        if len(recent) < 2:
            return
        
        # Add acquisition scores from history
        for p in recent:
            p['acquisition_score'] = 5.0  # Default score if not tracked
        
        # Compute gradient and update
        gradient = self.acquisition.compute_gradient(recent)
        self.acquisition.update_strategy(gradient)
    
    def get_best(self) -> Tuple[Tensor, float]:
        """Get best observed point and value."""
        return self.best_x, self.best_value
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        stats = {
            "config": self.config.to_dict(),
            "n_observations": len(self.X) if self.X is not None else 0,
            "n_iterations": self.iteration,
            "best_value": self.best_value,
            "best_x": self.best_x.tolist() if self.best_x is not None else None,
            "optimization_stats": self.optimization_stats_history,
        }
        
        # Add surrogate stats
        if hasattr(self.surrogate, 'get_belief'):
            stats["final_belief"] = self.surrogate.get_belief()
        
        # Add acquisition stats
        stats["acquisition_stats"] = self.acquisition.get_stats()
        
        # Add strategy evolution if using TextGrad
        if isinstance(self.acquisition, TextGradOptimizableAcquisition):
            stats["strategy_history"] = self.acquisition.get_strategy_history()
        
        return stats
    
    def clip_to_bounds(self, X: Tensor) -> Tensor:
        """Clip points to be within bounds."""
        return torch.max(torch.min(X, self.bounds[1]), self.bounds[0])


# =============================================================================
# Convenience Factory Functions
# =============================================================================

def create_vbo_optimizer(
    input_dim: int,
    bounds: Tensor,
    llm_client: Any,
    surrogate_mode: str = "in_context",
    surrogate_output: str = "text_with_numeric",
    acquisition_output: str = "text_with_numeric",
    optimization_method: str = "random_search",
    **kwargs,
) -> VBOptimizer:
    """
    Convenience factory for creating VBO optimizer.
    
    Args:
        input_dim: Dimensionality
        bounds: Parameter bounds
        llm_client: LLM client
        surrogate_mode: "in_context" or "belief_based"
        surrogate_output: "text_only" or "text_with_numeric"
        acquisition_output: "text_only" or "text_with_numeric"
        optimization_method: "random_search", "grid_search", "single_start_tga", "multi_start_tga"
        **kwargs: Additional configuration
        
    Returns:
        Configured VBOptimizer
    """
    config = VBOConfig(
        surrogate_mode=SurrogateMode(surrogate_mode),
        surrogate_output=OutputMode(surrogate_output),
        acquisition_output=AcquisitionOutputMode(acquisition_output),
        optimization_method=OptimizationMethod(optimization_method),
        **{k: v for k, v in kwargs.items() if hasattr(VBOConfig, k)},
    )
    
    return VBOptimizer(
        input_dim=input_dim,
        bounds=bounds,
        llm_client=llm_client,
        config=config,
        **{k: v for k, v in kwargs.items() if not hasattr(VBOConfig, k)},
    )


# =============================================================================
# List All Configurations for Experiments
# =============================================================================

def get_all_vbo_configurations() -> List[Dict[str, str]]:
    """
    Get all possible VBO configuration combinations.
    
    Returns:
        List of configuration dictionaries
    """
    configs = []
    
    for surrogate_mode in SurrogateMode:
        for surrogate_output in OutputMode:
            for acquisition_output in AcquisitionOutputMode:
                for optimization_method in OptimizationMethod:
                    configs.append({
                        "surrogate_mode": surrogate_mode.value,
                        "surrogate_output": surrogate_output.value,
                        "acquisition_output": acquisition_output.value,
                        "optimization_method": optimization_method.value,
                    })
    
    return configs


def get_vbo_config_name(config: Dict[str, str]) -> str:
    """Generate a short name for a VBO configuration."""
    sm = "IC" if config["surrogate_mode"] == "in_context" else "BB"
    so = "T" if config["surrogate_output"] == "text_only" else "TN"
    ao = "T" if config["acquisition_output"] == "text_only" else "TN"
    om = {
        "random_search": "RS",
        "grid_search": "GS",
        "single_start_tga": "sTGA",
        "multi_start_tga": "mTGA",
    }[config["optimization_method"]]
    
    return f"VBO_{sm}_{so}_{ao}_{om}"
