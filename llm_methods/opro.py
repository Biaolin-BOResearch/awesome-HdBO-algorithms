"""
OPRO: Optimization by PROmpting

This module implements OPRO, which uses LLMs as optimizers by describing
the optimization problem in natural language and iteratively prompting
the LLM to generate better solutions.

The key idea is to include historical solution-score pairs in the prompt
(sorted by score) so the LLM can learn the pattern and generate improved solutions.

Reference:
    Yang et al., "Large Language Models as Optimizers"
    NeurIPS 2023
    https://github.com/google-deepmind/opro
"""

import torch
from torch import Tensor
from typing import Optional, Any, List, Tuple, Dict
import numpy as np
import json
import re

from .base import BaseLLMOptimizer, DEVICE, DTYPE


class OPRO(BaseLLMOptimizer):
    """
    OPRO: Optimization by PROmpting.
    
    Uses LLM as optimizer by:
    1. Describing the optimization problem in natural language
    2. Including historical solution-score pairs in the prompt
    3. Asking LLM to generate new solutions that improve upon existing ones
    
    The meta-prompt contains:
    - Problem description
    - Previous solutions and their scores (sorted in ascending order)
    - Instruction to generate a new, better solution
    
    Attributes:
        max_num_instructions: Maximum number of past solutions in prompt.
        score_threshold: Minimum score threshold for inclusion in prompt.
        num_score_buckets: Number of buckets for score discretization.
    """
    
    def __init__(
        self,
        input_dim: int,
        bounds: Tensor,
        llm_client: Any = None,
        model_name: str = "gpt-4",
        task_description: str = "",
        maximize: bool = True,
        max_num_instructions: int = 20,
        score_threshold: float = 0.0,
        num_score_buckets: int = 20,
        temperature: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize OPRO optimizer.
        
        Args:
            input_dim: Dimensionality of the input space.
            bounds: Tensor of shape (2, input_dim) with [lower, upper] bounds.
            llm_client: Client for LLM API calls.
            model_name: Name of the LLM model.
            task_description: Description of the optimization problem.
            maximize: Whether to maximize or minimize.
            max_num_instructions: Maximum number of past solutions in prompt.
            score_threshold: Minimum score for inclusion in prompt.
            num_score_buckets: Number of buckets for score discretization.
            temperature: LLM sampling temperature.
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
        
        self.max_num_instructions = max_num_instructions
        self.score_threshold = score_threshold
        self.num_score_buckets = num_score_buckets
        self.temperature = temperature
        
        # History as list of (solution, score, step_index)
        self.solution_history: List[Tuple[List[float], float, int]] = []
        self.current_step = 0
    
    def initialize(self, X: Tensor, y: Tensor) -> None:
        """
        Initialize the optimizer with observed data.
        
        Also populates solution_history for OPRO-specific logic.
        
        Args:
            X: Initial observed points of shape (n, input_dim).
            y: Initial observed values of shape (n, 1) or (n,).
        """
        super().initialize(X, y)
        
        # Populate solution_history from initialized data
        for i in range(X.shape[0]):
            solution = X[i].tolist()
            score = y[i].item() if y[i].dim() == 0 else y[i, 0].item()
            self.solution_history.append((solution, score, self.current_step))
            self.current_step += 1
    
    def update(self, X_new: Tensor, y_new: Tensor) -> None:
        """
        Update the optimizer with new observations.
        
        Also updates solution_history for OPRO-specific logic.
        
        Args:
            X_new: New observed points.
            y_new: New observed values.
        """
        super().update(X_new, y_new)
        
        # Update solution_history
        X_new = X_new.to(device=self.device, dtype=self.dtype)
        y_new = y_new.to(device=self.device, dtype=self.dtype)
        if y_new.dim() == 1:
            y_new = y_new.unsqueeze(-1)
            
        for i in range(X_new.shape[0]):
            solution = X_new[i].tolist()
            score = y_new[i, 0].item()
            self.solution_history.append((solution, score, self.current_step))
            self.current_step += 1
        
    def _bucketize_score(self, score: float, min_score: float = 0.0, max_score: float = 1.0) -> int:
        """
        Convert continuous score to discrete bucket.
        
        Args:
            score: The score to bucketize.
            min_score: Minimum score value.
            max_score: Maximum score value.
            
        Returns:
            Bucket index (0 to num_score_buckets).
        """
        normalized = (score - min_score) / (max_score - min_score + 1e-9)
        normalized = max(0, min(1, normalized))
        return round(normalized * self.num_score_buckets)
    
    def _gen_solution_score_pairs_str(self) -> Tuple[str, List[Tuple[List[float], float, int]]]:
        """
        Generate string of solution-score pairs for meta-prompt.
        
        Solutions are sorted by score (ascending for maximization) and
        filtered by score threshold.
        
        Returns:
            Tuple of (formatted string, filtered solutions list).
        """
        # Sort by score (ascending so best is last)
        if self.maximize:
            sorted_solutions = sorted(self.solution_history, key=lambda x: x[1])
        else:
            sorted_solutions = sorted(self.solution_history, key=lambda x: -x[1])
        
        # Take last max_num_instructions
        sorted_solutions = sorted_solutions[-self.max_num_instructions:]
        
        # Filter by threshold
        filtered_solutions = []
        for solution, score, step in sorted_solutions:
            if self.maximize and score >= self.score_threshold:
                filtered_solutions.append((solution, score, step))
            elif not self.maximize and score <= self.score_threshold:
                filtered_solutions.append((solution, score, step))
        
        # Generate string
        lines = []
        for solution, score, _ in filtered_solutions:
            # Optionally bucketize score
            if self.num_score_buckets < float('inf'):
                score_str = str(self._bucketize_score(score))
            else:
                score_str = f"{score:.4f}"
            lines.append(f"solution:\n{solution}\nscore:\n{score_str}\n")
        
        return "\n".join(lines), filtered_solutions
    
    def _gen_meta_prompt(self) -> str:
        """
        Generate the meta-prompt for OPRO.
        
        The meta-prompt includes:
        1. Task description
        2. Historical solution-score pairs
        3. Instruction to generate better solution
        
        Returns:
            The meta-prompt string.
        """
        solution_str, _ = self._gen_solution_score_pairs_str()
        
        direction = "maximize" if self.maximize else "minimize"
        bounds_lower = self.bounds[0].tolist()
        bounds_upper = self.bounds[1].tolist()
        
        meta_prompt = f"""You are optimizing a {self.input_dim}-dimensional black-box function.

Task: {self.task_description}
Goal: {direction} the function value.
Search space: Each dimension is bounded by [{bounds_lower}, {bounds_upper}].

Below are some previous solutions with their scores. 
The solutions are arranged in ascending order based on their scores, where {"higher" if self.maximize else "lower"} scores indicate better quality.

{solution_str}

Generate a new solution that is different from all solutions above and has a {"higher" if self.maximize else "lower"} score than all solutions above.

Return the solution as a JSON array of {self.input_dim} numbers.
Format: [x1, x2, ..., x{self.input_dim}]
Do not include any explanation, only the JSON array.
"""
        return meta_prompt
    
    def _parse_solution(self, response: str) -> Optional[List[float]]:
        """
        Parse solution from LLM response.
        
        Args:
            response: LLM response string.
            
        Returns:
            Parsed solution as list, or None if parsing fails.
        """
        try:
            # Try to extract JSON array
            match = re.search(r'\[[\d\s,.\-e]+\]', response)
            if match:
                solution = json.loads(match.group())
                if isinstance(solution, list) and len(solution) == self.input_dim:
                    # Clip to bounds
                    solution = [
                        max(self.bounds[0, i].item(), min(self.bounds[1, i].item(), float(v)))
                        for i, v in enumerate(solution)
                    ]
                    return solution
        except (json.JSONDecodeError, ValueError):
            pass
        return None
    
    def suggest(self, n_candidates: int = 1) -> Tensor:
        """
        Suggest new solutions using OPRO.
        
        Args:
            n_candidates: Number of solutions to generate.
            
        Returns:
            Tensor of suggested solutions.
        """
        candidates = []
        
        for _ in range(n_candidates):
            if len(self.solution_history) == 0:
                # No history, generate random solution
                solution = self.random_points(1).squeeze(0).tolist()
            else:
                # Generate meta-prompt and query LLM
                meta_prompt = self._gen_meta_prompt()
                
                response = self.query_llm(
                    meta_prompt,
                    temperature=self.temperature,
                )
                
                solution = self._parse_solution(response)
                
                # Fallback to random if parsing fails
                if solution is None:
                    solution = self.random_points(1).squeeze(0).tolist()
            
            candidates.append(torch.tensor(solution, dtype=self.dtype, device=self.device))
        
        return torch.stack(candidates)
    
    def update_with_score(self, solution: List[float], score: float) -> None:
        """
        Update history with new solution and score.
        
        Args:
            solution: The evaluated solution.
            score: The function value.
        """
        self.solution_history.append((solution, score, self.current_step))
        self.current_step += 1
        
        # Also update base class data
        x_tensor = torch.tensor(solution, dtype=self.dtype, device=self.device).unsqueeze(0)
        y_tensor = torch.tensor([[score]], dtype=self.dtype, device=self.device)
        self.update(x_tensor, y_tensor)
    
    def get_best_solution(self) -> Tuple[List[float], float]:
        """
        Get the best solution found so far.
        
        Returns:
            Tuple of (best_solution, best_score).
        """
        if not self.solution_history:
            return None, None
        
        if self.maximize:
            best = max(self.solution_history, key=lambda x: x[1])
        else:
            best = min(self.solution_history, key=lambda x: x[1])
        
        return best[0], best[1]


class OPROWithExemplars(OPRO):
    """
    OPRO with few-shot exemplars.
    
    Extends OPRO to include a few example evaluations in the prompt
    to help the LLM understand the function behavior better.
    """
    
    def __init__(
        self,
        input_dim: int,
        bounds: Tensor,
        llm_client: Any = None,
        model_name: str = "gpt-4",
        task_description: str = "",
        maximize: bool = True,
        num_exemplars: int = 3,
        **kwargs,
    ):
        """
        Initialize OPRO with exemplars.
        
        Args:
            input_dim: Dimensionality.
            bounds: Search bounds.
            llm_client: LLM client.
            model_name: LLM model name.
            task_description: Task description.
            maximize: Whether to maximize.
            num_exemplars: Number of exemplars to show.
            **kwargs: Additional arguments for OPRO.
        """
        super().__init__(
            input_dim=input_dim,
            bounds=bounds,
            llm_client=llm_client,
            model_name=model_name,
            task_description=task_description,
            maximize=maximize,
            **kwargs,
        )
        self.num_exemplars = num_exemplars
    
    def _gen_meta_prompt(self) -> str:
        """Generate meta-prompt with exemplars."""
        solution_str, filtered_solutions = self._gen_solution_score_pairs_str()
        
        direction = "maximize" if self.maximize else "minimize"
        bounds_lower = self.bounds[0].tolist()
        bounds_upper = self.bounds[1].tolist()
        
        # Select exemplars (best and worst)
        exemplars_str = ""
        if len(filtered_solutions) >= 2:
            exemplars_str = "\nExample evaluations showing the pattern:\n"
            # Show worst
            worst = filtered_solutions[0]
            exemplars_str += f"  Low score example: {worst[0]} -> {worst[1]:.4f}\n"
            # Show best
            best = filtered_solutions[-1]
            exemplars_str += f"  High score example: {best[0]} -> {best[1]:.4f}\n"
        
        meta_prompt = f"""You are optimizing a {self.input_dim}-dimensional black-box function.

Task: {self.task_description}
Goal: {direction} the function value.
Search space: Each dimension is bounded by [{bounds_lower}, {bounds_upper}].
{exemplars_str}
Below are previous solutions with their scores (sorted by quality):

{solution_str}

Generate a new solution that:
1. Is different from all solutions above
2. Has a {"higher" if self.maximize else "lower"} score than all solutions above

Return only a JSON array of {self.input_dim} numbers: [x1, x2, ..., x{self.input_dim}]
"""
        return meta_prompt

