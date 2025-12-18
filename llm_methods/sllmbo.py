"""
SLLMBO: Structured LLM-based Bayesian Optimization

This module implements SLLMBO, which uses LLMs with structured prompting
and intelligent summarization for hyperparameter optimization.

Reference:
    "SLLMBO: A Structured Approach to LLM-based Black-box Optimization"
"""

import torch
from torch import Tensor
from typing import Optional, Any, Dict, List
import json

from .base import BaseLLMOptimizer, DEVICE, DTYPE


class SLLMBO(BaseLLMOptimizer):
    """
    Structured LLM-based Bayesian Optimization.
    
    Uses LLMs with:
    1. Structured prompting with parameter ranges
    2. Intelligent summarization of optimization history
    3. Adaptive parameter range refinement
    
    Attributes:
        param_ranges: Dictionary of parameter ranges.
        n_summarize_iter: Iterations between summarization.
        max_iters_no_improvement: Early stopping threshold.
    """
    
    def __init__(
        self,
        input_dim: int,
        bounds: Tensor,
        llm_client: Any = None,
        model_name: str = "gpt-4",
        task_description: str = "",
        maximize: bool = True,
        param_names: Optional[List[str]] = None,
        n_summarize_iter: int = 10,
        max_iters_no_improvement: int = 10,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize SLLMBO optimizer.
        
        Args:
            input_dim: Dimensionality of the input space.
            bounds: Tensor of shape (2, input_dim) with [lower, upper] bounds.
            llm_client: Client for LLM API calls.
            model_name: Name of the LLM model.
            task_description: Description of the optimization task.
            maximize: Whether to maximize or minimize.
            param_names: Optional names for each dimension.
            n_summarize_iter: Number of iterations between history summarization.
            max_iters_no_improvement: Early stopping if no improvement.
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
        
        self.param_names = param_names or [f"x{i}" for i in range(input_dim)]
        self.n_summarize_iter = n_summarize_iter
        self.max_iters_no_improvement = max_iters_no_improvement
        
        # Initialize parameter ranges from bounds
        self.param_ranges = self._init_param_ranges()
        
        # Tracking
        self.current_iteration = 0
        self.iters_no_improvement = 0
        self.best_score = float('-inf') if maximize else float('inf')
        self.best_params = None
        
        # System message
        self.system_message = self._create_system_message()
        
    def _init_param_ranges(self) -> Dict[str, Dict[str, float]]:
        """
        Initialize parameter ranges from bounds.
        """
        ranges = {}
        for i, name in enumerate(self.param_names):
            ranges[name] = {
                "min": self.bounds[0, i].item(),
                "max": self.bounds[1, i].item(),
                "type": "continuous"
            }
        return ranges
    
    def _create_system_message(self) -> str:
        """
        Create the system message for LLM.
        """
        direction = "maximize" if self.maximize else "minimize"
        return f"""You are an AI assistant helping with hyperparameter optimization.
Your goal is to {direction} the objective function.

Task Description: {self.task_description}

You should:
1. Analyze past evaluations to understand the objective landscape
2. Suggest promising parameter configurations
3. Balance exploration of new regions with exploitation of good areas
4. Adaptively refine parameter ranges when appropriate

Always respond in valid JSON format as specified in each prompt."""

    def _format_param_dict(self, x: Tensor) -> Dict[str, float]:
        """
        Convert tensor to parameter dictionary.
        """
        return {
            name: x[i].item()
            for i, name in enumerate(self.param_names)
        }
    
    def _parse_param_dict(self, params: Dict[str, float]) -> Tensor:
        """
        Convert parameter dictionary to tensor.
        """
        values = [params.get(name, 0.0) for name in self.param_names]
        return torch.tensor(values, device=self.device, dtype=self.dtype)
    
    def _intelligent_summarize(self) -> None:
        """
        Summarize conversation history using LLM.
        """
        if len(self.conversation_history) < 4:
            return
            
        summary_prompt = f"""
Please summarize the following optimization conversation history into a condensed form.
Retain the key insights about:
1. Which parameter regions work well
2. Which regions to avoid
3. Any patterns in the objective function

Conversation history:
{json.dumps(self.conversation_history, indent=2)}

Return as a JSON list of messages in format:
[{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}]
Keep it concise but informative.
"""
        
        try:
            # Use direct API call to avoid adding to history
            messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": summary_prompt}
            ]
            
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=2000,
                temperature=0.3,
            )
            
            summary = response.choices[0].message.content
            summarized_history = json.loads(summary)
            
            if isinstance(summarized_history, list):
                self.conversation_history = summarized_history
                
        except (json.JSONDecodeError, Exception):
            # Keep original history on failure
            pass
    
    def initialize_optimization(self) -> Dict[str, Any]:
        """
        Initialize optimization by querying LLM for starting point.
        """
        init_prompt = f"""
Initialize optimization for the following problem:

Parameter ranges:
{json.dumps(self.param_ranges, indent=2)}

Suggest initial parameter values and confirm the ranges.
Return JSON with format:
{{
    "initial_params": {{"param_name": value, ...}},
    "param_ranges": {{"param_name": {{"min": float, "max": float, "type": "continuous"}}, ...}}
}}
"""
        
        response = self.query_llm(
            init_prompt, 
            system_message=self.system_message,
            temperature=0.7
        )
        
        try:
            result = json.loads(response)
            if "param_ranges" in result:
                self.param_ranges = result["param_ranges"]
            return result
        except json.JSONDecodeError:
            # Return default
            default_params = {
                name: (self.bounds[0, i].item() + self.bounds[1, i].item()) / 2
                for i, name in enumerate(self.param_names)
            }
            return {"initial_params": default_params, "param_ranges": self.param_ranges}
    
    def suggest(self, n_candidates: int = 1) -> Tensor:
        """
        Suggest next parameter configuration.
        """
        if self.X is None or self.y is None:
            # Initialize
            init_result = self.initialize_optimization()
            params = init_result.get("initial_params", {})
            point = self._parse_param_dict(params)
            return self.clip_to_bounds(point.unsqueeze(0))
            
        # Summarize if needed
        if self.current_iteration > 0 and self.current_iteration % self.n_summarize_iter == 0:
            self._intelligent_summarize()
            
        # Get current best
        if self.maximize:
            best_idx = self.y.argmax()
        else:
            best_idx = self.y.argmin()
            
        best_x = self.X[best_idx]
        best_y = self.y[best_idx].item()
        
        # Update best tracking
        improved = (self.maximize and best_y > self.best_score) or \
                   (not self.maximize and best_y < self.best_score)
                   
        if improved:
            self.best_score = best_y
            self.best_params = self._format_param_dict(best_x)
            self.iters_no_improvement = 0
        else:
            self.iters_no_improvement += 1
            
        # Create optimization prompt
        opt_prompt = f"""
Current optimization status:
- Best score: {self.best_score:.6f}
- Best parameters: {json.dumps(self.best_params)}
- Current parameter ranges: {json.dumps(self.param_ranges)}
- Iterations without improvement: {self.iters_no_improvement}

Recent evaluations:
"""
        # Add recent history
        n_recent = min(10, len(self.X))
        for i in range(-n_recent, 0):
            params = self._format_param_dict(self.X[i])
            score = self.y[i].item()
            opt_prompt += f"  {params} -> {score:.6f}\n"
            
        opt_prompt += f"""
Based on the optimization history:
1. Suggest the next parameter configuration to try
2. Indicate if parameter ranges should be updated

Return JSON with format:
{{
    "next_params": {{"param_name": value, ...}},
    "update_param_ranges": true/false,
    "new_param_ranges": {{"param_name": {{"min": float, "max": float}}, ...}} (if update_param_ranges is true)
}}
"""
        
        response = self.query_llm(
            opt_prompt,
            system_message=self.system_message,
            temperature=0.7
        )
        
        try:
            result = json.loads(response)
            
            # Update ranges if suggested
            if result.get("update_param_ranges", False):
                new_ranges = result.get("new_param_ranges", {})
                for name, range_info in new_ranges.items():
                    if name in self.param_ranges:
                        self.param_ranges[name].update(range_info)
                        
            # Parse suggested parameters
            next_params = result.get("next_params", {})
            point = self._parse_param_dict(next_params)
            
        except json.JSONDecodeError:
            # Fallback to random point near best
            if self.best_params:
                best_tensor = self._parse_param_dict(self.best_params)
                noise = torch.randn(self.input_dim, device=self.device, dtype=self.dtype)
                noise = noise * (self.bounds[1] - self.bounds[0]) * 0.1
                point = best_tensor + noise
            else:
                point = self.random_points(1).squeeze(0)
                
        self.current_iteration += 1
        return self.clip_to_bounds(point.unsqueeze(0))
    
    def should_stop(self) -> bool:
        """
        Check if optimization should stop early.
        """
        return self.iters_no_improvement >= self.max_iters_no_improvement

