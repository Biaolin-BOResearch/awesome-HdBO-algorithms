"""
Acquisition Function Optimization Methods

Four methods for optimizing the acquisition function:
1. Random Search: Sample random candidates and evaluate
2. Grid Search: Systematically search a grid
3. Single-Start Textual Gradient Ascent (TGA): Gradient ascent from one starting point
4. Multi-Start Textual Gradient Ascent: Gradient ascent from multiple starting points

Each method tracks optimization efficiency for comparison.
"""

import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from .surrogate import VBOSurrogateModel, SurrogatePrediction
from .acquisition import VBOAcquisitionFunction, AcquisitionEvaluation


class OptimizationMethod(Enum):
    """Acquisition optimization method."""
    RANDOM_SEARCH = "random_search"
    GRID_SEARCH = "grid_search"
    SINGLE_START_TGA = "single_start_tga"
    MULTI_START_TGA = "multi_start_tga"


@dataclass
class OptimizationStats:
    """Statistics for acquisition optimization."""
    method: str
    n_candidates_evaluated: int
    n_llm_calls: int
    wall_time_seconds: float
    best_acquisition_value: Optional[float]
    best_x: Any
    convergence_history: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "n_candidates_evaluated": self.n_candidates_evaluated,
            "n_llm_calls": self.n_llm_calls,
            "wall_time_seconds": round(self.wall_time_seconds, 3),
            "best_acquisition_value": self.best_acquisition_value,
            "convergence_history": self.convergence_history,
        }


# =============================================================================
# Textual Gradient Prompts
# =============================================================================

COMPUTE_GRADIENT_PROMPT = """You are computing the textual gradient for Bayesian optimization acquisition maximization.

Current point: x = {current_x}
Current acquisition value: {current_value}

Surrogate prediction at this point:
{prediction_text}

Acquisition evaluation:
{acquisition_text}

Parameter bounds:
{bounds}

Based on this information, compute the gradient direction for MAXIMIZING the acquisition value:

1. **Direction Analysis**: In which direction should x move to increase acquisition value?
   - Consider both exploration (move towards uncertain regions) and exploitation (move towards high-value regions)

2. **Magnitude Suggestion**: How large should the step be?
   - Consider distance to bounds
   - Consider the uncertainty level

3. **Specific Adjustment**: Provide concrete numerical adjustments for each dimension.

Output in JSON format:
{{
    "gradient_direction": "Textual description of optimal direction...",
    "magnitude": "small/medium/large",
    "dimension_adjustments": [<adjustment for dim 0>, <adjustment for dim 1>, ...],
    "reasoning": "Why this direction improves acquisition..."
}}"""


REFINE_POINT_PROMPT = """You are refining a candidate point for Bayesian optimization.

Starting point: x = {start_x}
Target: Maximize acquisition value

Gradient information:
{gradient_text}

Current acquisition value: {current_value}
Parameter bounds: {bounds}

Apply the gradient to compute a new, improved point.
Consider:
1. Step size appropriate for the gradient magnitude
2. Stay within parameter bounds
3. Balance between following gradient and exploring

Output the new point:
{{
    "new_x": [<value for dim 0>, <value for dim 1>, ...],
    "expected_improvement": "Description of expected improvement...",
    "confidence": "high/medium/low"
}}"""


# =============================================================================
# Base Optimizer
# =============================================================================

class AcquisitionOptimizer(ABC):
    """Abstract base class for acquisition optimizers."""
    
    def __init__(
        self,
        bounds: Tensor,
        llm_client: Any,
    ):
        """
        Initialize optimizer.
        
        Args:
            bounds: Tensor of shape (2, dim) with [lower, upper] bounds
            llm_client: LLM client for TGA methods
        """
        self.bounds = bounds
        self.llm_client = llm_client
        self.dim = bounds.shape[1]
        self.stats: Optional[OptimizationStats] = None
    
    @abstractmethod
    def optimize(
        self,
        surrogate: VBOSurrogateModel,
        acquisition: VBOAcquisitionFunction,
        best_value: float,
        n_observations: int,
        iteration: int = 0,
        total_iterations: int = 100,
    ) -> Tuple[Tensor, OptimizationStats]:
        """
        Optimize acquisition function to find best next point.
        
        Args:
            surrogate: Surrogate model for predictions
            acquisition: Acquisition function for evaluation
            best_value: Current best observed value
            n_observations: Number of observations so far
            iteration: Current iteration
            total_iterations: Total iterations planned
            
        Returns:
            Tuple of (best_x, optimization_stats)
        """
        pass
    
    def _random_point(self) -> Tensor:
        """Generate a random point within bounds."""
        return (
            self.bounds[0] + 
            torch.rand(self.dim, dtype=torch.float64) * 
            (self.bounds[1] - self.bounds[0])
        )
    
    def _clip_to_bounds(self, x: Tensor) -> Tensor:
        """Clip point to be within bounds."""
        return torch.max(torch.min(x, self.bounds[1]), self.bounds[0])
    
    def _format_x(self, x: Any) -> str:
        """Format point as string."""
        if isinstance(x, Tensor):
            return "[" + ", ".join(f"{v:.4f}" for v in x.tolist()) + "]"
        elif isinstance(x, (list, tuple)):
            return "[" + ", ".join(f"{v:.4f}" for v in x) + "]"
        return str(x)
    
    def _format_bounds(self) -> str:
        """Format bounds as string."""
        lower = self.bounds[0].tolist()
        upper = self.bounds[1].tolist()
        return f"Lower: {lower}, Upper: {upper}"
    
    def get_stats(self) -> Optional[Dict]:
        """Get optimization statistics."""
        return self.stats.to_dict() if self.stats else None


# =============================================================================
# Random Search
# =============================================================================

class RandomSearchOptimizer(AcquisitionOptimizer):
    """
    Random Search for acquisition optimization.
    
    Samples random candidates and evaluates them with the acquisition function.
    """
    
    def __init__(
        self,
        bounds: Tensor,
        llm_client: Any,
        n_candidates: int = 50,
    ):
        super().__init__(bounds, llm_client)
        self.n_candidates = n_candidates
    
    def optimize(
        self,
        surrogate: VBOSurrogateModel,
        acquisition: VBOAcquisitionFunction,
        best_value: float,
        n_observations: int,
        iteration: int = 0,
        total_iterations: int = 100,
    ) -> Tuple[Tensor, OptimizationStats]:
        """Optimize using random search."""
        start_time = time.time()
        n_llm_calls = 0
        
        best_x = None
        best_acq_value = float('-inf')
        convergence = []
        
        for i in range(self.n_candidates):
            # Generate random candidate
            x = self._random_point()
            
            # Get surrogate prediction
            prediction = surrogate.predict(x)
            n_llm_calls += 1
            
            # Evaluate acquisition
            evaluation = acquisition.evaluate(
                x=x,
                prediction=prediction,
                best_value=best_value,
                n_observations=n_observations,
                iteration=iteration,
                total_iterations=total_iterations,
            )
            n_llm_calls += 1
            
            # Get score (use 0.5 as default for text-only mode)
            score = evaluation.score if evaluation.score is not None else 0.5
            
            if score > best_acq_value:
                best_acq_value = score
                best_x = x.clone()
            
            convergence.append(best_acq_value)
        
        elapsed = time.time() - start_time
        
        self.stats = OptimizationStats(
            method="random_search",
            n_candidates_evaluated=self.n_candidates,
            n_llm_calls=n_llm_calls,
            wall_time_seconds=elapsed,
            best_acquisition_value=best_acq_value,
            best_x=best_x.tolist() if best_x is not None else None,
            convergence_history=convergence,
        )
        
        return best_x if best_x is not None else self._random_point(), self.stats


# =============================================================================
# Grid Search
# =============================================================================

class GridSearchOptimizer(AcquisitionOptimizer):
    """
    Grid Search for acquisition optimization.
    
    Creates a grid over the search space and evaluates candidates systematically.
    """
    
    def __init__(
        self,
        bounds: Tensor,
        llm_client: Any,
        n_per_dim: int = 5,
        max_candidates: int = 100,
    ):
        super().__init__(bounds, llm_client)
        self.n_per_dim = n_per_dim
        self.max_candidates = max_candidates
    
    def _create_grid(self) -> Tensor:
        """Create grid of candidate points."""
        # Create 1D grids for each dimension
        grids = []
        for d in range(self.dim):
            grids.append(torch.linspace(
                self.bounds[0, d].item(),
                self.bounds[1, d].item(),
                self.n_per_dim,
            ))
        
        # Create meshgrid
        mesh = torch.meshgrid(*grids, indexing='ij')
        
        # Flatten to get all candidates
        candidates = torch.stack([m.flatten() for m in mesh], dim=1)
        
        # Subsample if too many
        if len(candidates) > self.max_candidates:
            indices = torch.randperm(len(candidates))[:self.max_candidates]
            candidates = candidates[indices]
        
        return candidates.to(torch.float64)
    
    def optimize(
        self,
        surrogate: VBOSurrogateModel,
        acquisition: VBOAcquisitionFunction,
        best_value: float,
        n_observations: int,
        iteration: int = 0,
        total_iterations: int = 100,
    ) -> Tuple[Tensor, OptimizationStats]:
        """Optimize using grid search."""
        start_time = time.time()
        n_llm_calls = 0
        
        # Create grid
        candidates = self._create_grid()
        
        best_x = None
        best_acq_value = float('-inf')
        convergence = []
        
        for i in range(len(candidates)):
            x = candidates[i]
            
            # Get surrogate prediction
            prediction = surrogate.predict(x)
            n_llm_calls += 1
            
            # Evaluate acquisition
            evaluation = acquisition.evaluate(
                x=x,
                prediction=prediction,
                best_value=best_value,
                n_observations=n_observations,
                iteration=iteration,
                total_iterations=total_iterations,
            )
            n_llm_calls += 1
            
            # Get score
            score = evaluation.score if evaluation.score is not None else 0.5
            
            if score > best_acq_value:
                best_acq_value = score
                best_x = x.clone()
            
            convergence.append(best_acq_value)
        
        elapsed = time.time() - start_time
        
        self.stats = OptimizationStats(
            method="grid_search",
            n_candidates_evaluated=len(candidates),
            n_llm_calls=n_llm_calls,
            wall_time_seconds=elapsed,
            best_acquisition_value=best_acq_value,
            best_x=best_x.tolist() if best_x is not None else None,
            convergence_history=convergence,
        )
        
        return best_x if best_x is not None else self._random_point(), self.stats


# =============================================================================
# Single-Start Textual Gradient Ascent
# =============================================================================

class SingleStartTGA(AcquisitionOptimizer):
    """
    Single-Start Textual Gradient Ascent for acquisition optimization.
    
    Starts from one point and uses LLM to compute textual gradients for
    iterative improvement.
    """
    
    def __init__(
        self,
        bounds: Tensor,
        llm_client: Any,
        n_steps: int = 10,
        step_size: float = 0.1,
    ):
        super().__init__(bounds, llm_client)
        self.n_steps = n_steps
        self.step_size = step_size
    
    def _compute_textual_gradient(
        self,
        x: Tensor,
        prediction: SurrogatePrediction,
        evaluation: AcquisitionEvaluation,
    ) -> Dict[str, Any]:
        """Compute textual gradient using LLM."""
        prompt = COMPUTE_GRADIENT_PROMPT.format(
            current_x=self._format_x(x),
            current_value=f"{evaluation.score:.4f}" if evaluation.score else "N/A",
            prediction_text=f"Distribution: {prediction.distribution_text}\nConfidence: {prediction.confidence_text}",
            acquisition_text=f"Evaluation: {evaluation.evaluation_text}\nExploration: {evaluation.exploration_text}\nExploitation: {evaluation.exploitation_text}",
            bounds=self._format_bounds(),
        )
        
        response = self._query_llm(prompt)
        
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
        
        # Fallback: random direction
        return {
            "gradient_direction": "random perturbation",
            "magnitude": "small",
            "dimension_adjustments": [0.0] * self.dim,
            "reasoning": "Fallback to random exploration",
        }
    
    def _apply_gradient(
        self,
        x: Tensor,
        gradient: Dict[str, Any],
    ) -> Tensor:
        """Apply textual gradient to get new point."""
        # Parse magnitude
        magnitude_map = {"small": 0.05, "medium": 0.1, "large": 0.2}
        magnitude = magnitude_map.get(gradient.get("magnitude", "medium"), 0.1)
        
        # Get dimension adjustments
        adjustments = gradient.get("dimension_adjustments", [])
        
        # If we have specific adjustments, use them
        if adjustments and len(adjustments) == self.dim:
            delta = torch.tensor(adjustments, dtype=torch.float64)
            # Normalize and scale
            if delta.norm() > 0:
                delta = delta / delta.norm() * magnitude * (self.bounds[1] - self.bounds[0]).norm()
        else:
            # Use LLM to refine
            prompt = REFINE_POINT_PROMPT.format(
                start_x=self._format_x(x),
                gradient_text=gradient.get("gradient_direction", "explore nearby"),
                current_value=str(gradient.get("current_value", "unknown")),
                bounds=self._format_bounds(),
            )
            
            response = self._query_llm(prompt)
            
            try:
                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    data = json.loads(json_match.group())
                    new_x = data.get("new_x", [])
                    if new_x and len(new_x) == self.dim:
                        return self._clip_to_bounds(torch.tensor(new_x, dtype=torch.float64))
            except json.JSONDecodeError:
                pass
            
            # Fallback: random perturbation
            delta = torch.randn(self.dim, dtype=torch.float64)
            delta = delta / delta.norm() * magnitude * (self.bounds[1] - self.bounds[0]).norm()
        
        new_x = x + delta
        return self._clip_to_bounds(new_x)
    
    def _query_llm(self, prompt: str) -> str:
        """Query LLM."""
        response = self.llm_client.chat.completions.create(
            model=self.llm_client.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    
    def optimize(
        self,
        surrogate: VBOSurrogateModel,
        acquisition: VBOAcquisitionFunction,
        best_value: float,
        n_observations: int,
        iteration: int = 0,
        total_iterations: int = 100,
    ) -> Tuple[Tensor, OptimizationStats]:
        """Optimize using textual gradient ascent."""
        start_time = time.time()
        n_llm_calls = 0
        n_candidates = 0
        
        # Start from random point
        x = self._random_point()
        
        best_x = x.clone()
        best_acq_value = float('-inf')
        convergence = []
        
        for step in range(self.n_steps):
            # Get prediction
            prediction = surrogate.predict(x)
            n_llm_calls += 1
            
            # Evaluate acquisition
            evaluation = acquisition.evaluate(
                x=x,
                prediction=prediction,
                best_value=best_value,
                n_observations=n_observations,
                iteration=iteration,
                total_iterations=total_iterations,
            )
            n_llm_calls += 1
            n_candidates += 1
            
            current_score = evaluation.score if evaluation.score is not None else 0.5
            
            if current_score > best_acq_value:
                best_acq_value = current_score
                best_x = x.clone()
            
            convergence.append(best_acq_value)
            
            # Compute textual gradient
            gradient = self._compute_textual_gradient(x, prediction, evaluation)
            gradient["current_value"] = current_score
            n_llm_calls += 1
            
            # Apply gradient
            x = self._apply_gradient(x, gradient)
            # Might need another LLM call for refinement
            n_llm_calls += 1
        
        elapsed = time.time() - start_time
        
        self.stats = OptimizationStats(
            method="single_start_tga",
            n_candidates_evaluated=n_candidates,
            n_llm_calls=n_llm_calls,
            wall_time_seconds=elapsed,
            best_acquisition_value=best_acq_value,
            best_x=best_x.tolist(),
            convergence_history=convergence,
        )
        
        return best_x, self.stats


# =============================================================================
# Multi-Start Textual Gradient Ascent
# =============================================================================

class MultiStartTGA(AcquisitionOptimizer):
    """
    Multi-Start Textual Gradient Ascent for acquisition optimization.
    
    Runs TGA from multiple starting points and returns the best result.
    """
    
    def __init__(
        self,
        bounds: Tensor,
        llm_client: Any,
        n_starts: int = 5,
        n_steps_per_start: int = 5,
        step_size: float = 0.1,
    ):
        super().__init__(bounds, llm_client)
        self.n_starts = n_starts
        self.n_steps_per_start = n_steps_per_start
        self.step_size = step_size
        
        # Create single-start optimizer for each run
        self.single_tga = SingleStartTGA(
            bounds=bounds,
            llm_client=llm_client,
            n_steps=n_steps_per_start,
            step_size=step_size,
        )
    
    def optimize(
        self,
        surrogate: VBOSurrogateModel,
        acquisition: VBOAcquisitionFunction,
        best_value: float,
        n_observations: int,
        iteration: int = 0,
        total_iterations: int = 100,
    ) -> Tuple[Tensor, OptimizationStats]:
        """Optimize using multi-start TGA."""
        start_time = time.time()
        
        all_results = []
        total_candidates = 0
        total_llm_calls = 0
        all_convergence = []
        
        for start_idx in range(self.n_starts):
            # Run single-start TGA
            x, stats = self.single_tga.optimize(
                surrogate=surrogate,
                acquisition=acquisition,
                best_value=best_value,
                n_observations=n_observations,
                iteration=iteration,
                total_iterations=total_iterations,
            )
            
            all_results.append((x, stats.best_acquisition_value))
            total_candidates += stats.n_candidates_evaluated
            total_llm_calls += stats.n_llm_calls
            all_convergence.extend(stats.convergence_history)
        
        # Find best among all starts
        best_idx = max(range(len(all_results)), key=lambda i: all_results[i][1] or 0)
        best_x, best_acq_value = all_results[best_idx]
        
        elapsed = time.time() - start_time
        
        self.stats = OptimizationStats(
            method="multi_start_tga",
            n_candidates_evaluated=total_candidates,
            n_llm_calls=total_llm_calls,
            wall_time_seconds=elapsed,
            best_acquisition_value=best_acq_value,
            best_x=best_x.tolist() if isinstance(best_x, Tensor) else best_x,
            convergence_history=all_convergence,
        )
        
        return best_x, self.stats


# =============================================================================
# Factory Function
# =============================================================================

def create_optimizer(
    method: OptimizationMethod,
    bounds: Tensor,
    llm_client: Any,
    **kwargs,
) -> AcquisitionOptimizer:
    """
    Factory function to create an acquisition optimizer.
    
    Args:
        method: Optimization method to use
        bounds: Parameter bounds
        llm_client: LLM client
        **kwargs: Method-specific parameters
        
    Returns:
        Configured optimizer
    """
    if method == OptimizationMethod.RANDOM_SEARCH:
        return RandomSearchOptimizer(
            bounds=bounds,
            llm_client=llm_client,
            n_candidates=kwargs.get("n_candidates", 50),
        )
    elif method == OptimizationMethod.GRID_SEARCH:
        return GridSearchOptimizer(
            bounds=bounds,
            llm_client=llm_client,
            n_per_dim=kwargs.get("n_per_dim", 5),
            max_candidates=kwargs.get("max_candidates", 100),
        )
    elif method == OptimizationMethod.SINGLE_START_TGA:
        return SingleStartTGA(
            bounds=bounds,
            llm_client=llm_client,
            n_steps=kwargs.get("n_steps", 10),
            step_size=kwargs.get("step_size", 0.1),
        )
    elif method == OptimizationMethod.MULTI_START_TGA:
        return MultiStartTGA(
            bounds=bounds,
            llm_client=llm_client,
            n_starts=kwargs.get("n_starts", 5),
            n_steps_per_start=kwargs.get("n_steps_per_start", 5),
            step_size=kwargs.get("step_size", 0.1),
        )
    else:
        raise ValueError(f"Unknown optimization method: {method}")
