"""
LLAMBO: Large Language Model for Bayesian Optimization

This module implements LLAMBO, which uses LLMs as both surrogate models
and acquisition functions for Bayesian optimization.

Reference:
    Liu et al., "Large Language Models to Enhance Bayesian Optimization"
    ICLR 2024
"""

import torch
from torch import Tensor
from typing import Optional, List, Tuple, Dict, Any
import json
import random
import numpy as np
from scipy.stats import norm

from .base import BaseLLMOptimizer, DEVICE, DTYPE


class LLAMBO(BaseLLMOptimizer):
    """
    LLAMBO: LLM-based Bayesian Optimization.
    
    Uses LLM for:
    1. Candidate point generation (via prompting with historical data)
    2. Surrogate modeling (LLM predicts function values)
    3. Expected Improvement computation
    
    Attributes:
        n_candidates: Number of candidate points to generate per iteration.
        n_surrogate_samples: Number of LLM queries for uncertainty estimation.
        alpha: Target score percentile parameter.
        use_warmstart: Whether to use LLM for initial point generation.
    """
    
    def __init__(
        self,
        input_dim: int,
        bounds: Tensor,
        llm_client: Any = None,
        model_name: str = "gpt-4",
        task_description: str = "",
        maximize: bool = True,
        n_candidates: int = 10,
        n_surrogate_samples: int = 10,
        alpha: float = 0.1,
        use_warmstart: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize LLAMBO optimizer.
        
        Args:
            input_dim: Dimensionality of the input space.
            bounds: Tensor of shape (2, input_dim) with [lower, upper] bounds.
            llm_client: Client for LLM API calls.
            model_name: Name of the LLM model.
            task_description: Description of the optimization task.
            maximize: Whether to maximize or minimize.
            n_candidates: Number of candidates to sample each iteration.
            n_surrogate_samples: Number of samples for surrogate uncertainty.
            alpha: Parameter for target score calculation.
            use_warmstart: Whether to use LLM warmstart.
            device: Torch device.
            dtype: Torch data type.
        """
        super().__init__(
            input_dim=input_dim,
            bounds=bounds,
            llm_client=llm_client,
            model_name=model_name,
            task_description=task_description,
            maximize=maximize,
            device=device,
            dtype=dtype,
        )
        
        self.n_candidates = n_candidates
        self.n_surrogate_samples = n_surrogate_samples
        self.alpha = alpha
        self.use_warmstart = use_warmstart
        
        # Surrogate model cache
        self._surrogate_cache: Dict[tuple, Tuple[float, float]] = {}
        
    def warmstart(self, n_points: int, objective_func: Any = None) -> Tensor:
        """
        Generate initial points using LLM.
        
        Args:
            n_points: Number of initial points to generate.
            objective_func: Optional objective function for evaluation.
            
        Returns:
            Tensor of initial points.
        """
        system_msg = (
            "You are an AI assistant helping with black-box optimization. "
            f"The goal is to {'maximize' if self.maximize else 'minimize'} "
            f"a function. {self.task_description}"
        )
        
        prompt = f"""
You are helping to initialize optimization of a black-box function.
The search space has {self.input_dim} dimensions.
Bounds: [{self.bounds[0].tolist()}, {self.bounds[1].tolist()}]

Suggest {n_points} diverse starting points that would be good for exploring this space.
Return strictly as a JSON array of {self.input_dim}-dimensional arrays.
Example format: [[0.1, 0.2], [0.5, 0.6], ...]
Do not include any explanation, only the JSON array.
"""
        
        response = self.query_llm(prompt, system_message=system_msg)
        
        try:
            points = json.loads(response)
            if isinstance(points, list) and len(points) >= n_points:
                points = points[:n_points]
                points_tensor = torch.tensor(points, device=self.device, dtype=self.dtype)
                return self.clip_to_bounds(points_tensor)
        except json.JSONDecodeError:
            pass
            
        # Fallback to random
        return self.random_points(n_points)
    
    def _sample_candidate_point(self, history_str: str, target_score: float) -> Optional[Tensor]:
        """
        Sample a candidate point from LLM.
        
        Args:
            history_str: Formatted history string.
            target_score: Target score to achieve.
            
        Returns:
            Candidate point tensor or None.
        """
        prompt = f"""
The following are past evaluations of a black-box function:
{history_str}

The search space bounds are [{self.bounds[0].tolist()}, {self.bounds[1].tolist()}].
Recommend a new point x that can achieve a function value of {target_score:.6f}.
Return only a single {self.input_dim}-dimensional numerical vector.
Format: [x1, x2, ..., x{self.input_dim}]
Do not include any explanations.
"""
        
        system_msg = (
            f"You are an AI assistant helping to {'maximize' if self.maximize else 'minimize'} "
            f"a black-box function. {self.task_description}"
        )
        
        response = self.query_llm(prompt, system_message=system_msg)
        return self.parse_point_from_response(response)
    
    def _predict_with_llm(self, x: Tensor, history_str: str) -> float:
        """
        Predict function value at x using LLM as surrogate.
        
        Args:
            x: Point to predict.
            history_str: Formatted history string.
            
        Returns:
            Predicted function value.
        """
        prompt = f"""
The following are past evaluations of a black-box function:
{history_str}

The search space bounds are [{self.bounds[0].tolist()}, {self.bounds[1].tolist()}].
Predict the function value at x = {x.tolist()}.
Return only a single numerical value.
"""
        
        system_msg = (
            f"You are an AI assistant predicting values of a black-box function. "
            f"{self.task_description}"
        )
        
        response = self.query_llm(prompt, system_message=system_msg)
        
        try:
            return float(response.strip())
        except ValueError:
            # Try to extract a number
            import re
            match = re.search(r'-?\d+\.?\d*', response)
            if match:
                return float(match.group())
            return 0.0
    
    def _compute_surrogate(
        self,
        candidates: List[Tensor],
    ) -> Dict[tuple, Tuple[float, float]]:
        """
        Compute surrogate model predictions for candidates.
        
        Uses multiple LLM queries with shuffled history to estimate
        mean and standard deviation.
        
        Args:
            candidates: List of candidate points.
            
        Returns:
            Dictionary mapping point tuples to (mean, std) predictions.
        """
        results = {}
        
        for candidate in candidates:
            key = tuple(candidate.tolist())
            
            if key in self._surrogate_cache:
                results[key] = self._surrogate_cache[key]
                continue
                
            predictions = []
            for _ in range(self.n_surrogate_samples):
                # Shuffle history for each sample
                history = list(zip(self.X.tolist(), self.y.squeeze().tolist()))
                random.shuffle(history)
                history_str = "\n".join([
                    f"x: {x}, f(x): {y}" for x, y in history
                ])
                
                pred = self._predict_with_llm(candidate, history_str)
                predictions.append(pred)
                
            mean = np.mean(predictions)
            std = np.std(predictions)
            
            results[key] = (mean, std)
            self._surrogate_cache[key] = (mean, std)
            
        return results
    
    def _expected_improvement(
        self,
        mean: float,
        std: float,
        best_so_far: float,
        xi: float = 0.01,
    ) -> float:
        """
        Compute Expected Improvement.
        
        Args:
            mean: Predicted mean.
            std: Predicted standard deviation.
            best_so_far: Best observed value.
            xi: Exploration-exploitation trade-off parameter.
            
        Returns:
            Expected improvement value.
        """
        if std < 1e-9:
            return 0.0
            
        if self.maximize:
            improvement = mean - best_so_far - xi
        else:
            improvement = best_so_far - mean - xi
            
        z = improvement / std
        ei = improvement * norm.cdf(z) + std * norm.pdf(z)
        return max(ei, 0.0)
    
    def suggest(self, n_candidates: int = 1) -> Tensor:
        """
        Suggest new points to evaluate.
        
        Uses LLAMBO algorithm:
        1. Sample candidate points from LLM
        2. Compute surrogate predictions
        3. Select points with highest Expected Improvement
        
        Args:
            n_candidates: Number of candidates to suggest.
            
        Returns:
            Tensor of suggested points.
        """
        if self.X is None or self.y is None:
            return self.random_points(n_candidates)
            
        # Compute target score for candidate generation
        if self.maximize:
            best_y = self.y.max().item()
            worst_y = self.y.min().item()
            target_score = best_y + self.alpha * (best_y - worst_y)
        else:
            best_y = self.y.min().item()
            worst_y = self.y.max().item()
            target_score = best_y - self.alpha * (worst_y - best_y)
            
        # Sample candidates from LLM
        candidates = []
        for _ in range(self.n_candidates):
            # Shuffle history
            history = list(zip(self.X.tolist(), self.y.squeeze().tolist()))
            random.shuffle(history)
            history_str = "\n".join([
                f"x: {x}, f(x): {y}" for x, y in history
            ])
            
            candidate = self._sample_candidate_point(history_str, target_score)
            if candidate is not None:
                candidates.append(candidate)
                
        # Fallback if no candidates generated
        if len(candidates) == 0:
            return self.random_points(n_candidates)
            
        # Compute surrogate predictions
        surrogate_results = self._compute_surrogate(candidates)
        
        # Compute EI for each candidate
        best_so_far = best_y
        ei_values = []
        for candidate in candidates:
            key = tuple(candidate.tolist())
            mean, std = surrogate_results.get(key, (0.0, 0.0))
            ei = self._expected_improvement(mean, std, best_so_far)
            ei_values.append(ei)
            
        # Select top candidates by EI
        sorted_indices = np.argsort(ei_values)[::-1]
        selected = [candidates[i] for i in sorted_indices[:n_candidates]]
        
        return torch.stack(selected)


class LLAMBOLight(BaseLLMOptimizer):
    """
    LLAMBO-Light: Simplified version without surrogate modeling.
    
    Uses LLM directly for candidate generation with exploration-exploitation
    balance encoded in the prompt.
    """
    
    def __init__(
        self,
        input_dim: int,
        bounds: Tensor,
        llm_client: Any = None,
        model_name: str = "gpt-3.5-turbo",
        task_description: str = "",
        maximize: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(
            input_dim=input_dim,
            bounds=bounds,
            llm_client=llm_client,
            model_name=model_name,
            task_description=task_description,
            maximize=maximize,
            device=device,
            dtype=dtype,
        )
        
    def suggest(self, n_candidates: int = 1) -> Tensor:
        """
        Suggest points using direct LLM prompting.
        """
        if self.X is None or self.y is None:
            return self.random_points(n_candidates)
            
        # Format history
        history = list(zip(self.X.tolist(), self.y.squeeze().tolist()))
        random.shuffle(history)
        history_str = "\n".join([f"x: {x}, f(x): {y}" for x, y in history])
        
        prompt = f"""
The following are past evaluations of a black-box function:
{history_str}

The search space bounds are [{self.bounds[0].tolist()}, {self.bounds[1].tolist()}].

Based on the past data, recommend the next point to evaluate that balances:
- Exploration: selecting points in unexplored regions far from evaluated points
- Exploitation: selecting points close to high-performing evaluations

The goal is to find the global {'maximum' if self.maximize else 'minimum'}.
Return only a {self.input_dim}-dimensional numerical vector.
Format: [x1, x2, ..., x{self.input_dim}]
"""
        
        system_msg = (
            f"You are an AI assistant for black-box optimization. "
            f"{self.task_description}"
        )
        
        candidates = []
        for _ in range(n_candidates):
            response = self.query_llm(prompt, system_message=system_msg)
            point = self.parse_point_from_response(response)
            if point is not None:
                candidates.append(point)
                
        if len(candidates) == 0:
            return self.random_points(n_candidates)
            
        return torch.stack(candidates)

