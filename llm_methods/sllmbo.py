"""
SLLMBO: Sequential Large Language Model-Based Hyper-Parameter Optimization

This module implements SLLMBO, which uses LLMs with structured prompting,
dynamic search space adaptability, and intelligent summarization for
hyperparameter optimization.

Reference:
    SLLMBO: Sequential Large Language Model-Based Hyper-Parameter Optimization
    https://github.com/KananMahammadli/SLLMBO
"""

import torch
from torch import Tensor
from typing import Optional, Any, Dict, List, Tuple
import json
import re

from .base import BaseLLMOptimizer, DEVICE, DTYPE


# Default prompts matching the reference SLLMBO implementation
DEFAULT_SYSTEM_PROMPT = """You are an AI assistant specialized in hyperparameter tuning for machine learning models.
Your goal is to optimize hyperparameters for a {model_name} model for the following problem:
    {problem_description}
The {direction} average and minimize standard deviation of the {metric} metric used to evaluate the performance of the model.
        
When asked for initialization, provide:
    1. A list of hyperparameters to tune and their value ranges, if parameter is numeric, provide range as [min, max], if parameter is categorical, provide list of options.
    2. Starting values for these parameters.
        
Format your response as a JSON object with keys 'param_ranges' and 'initial_params'.
        
During the optimization process, you will be asked to:
    1. Decide whether to update parameter ranges or continue with the last range.
    2. Suggest new parameter values based on previous parameter ranges, suggested parameter values and results, and last used parameter range.
        
Balance exploration and exploitation as the optimization progresses.
Keep in mind that goal is to find the best possible hyperparameter values for {model_name} model to {direction} the {metric} metric."""

DEFAULT_INIT_PROMPT = "Provide the initial hyperparameters to tune, their ranges, and starting values."

DEFAULT_OPT_PROMPT = """Goal: Find the best possible hyperparameter values for {model_name} model to {direction} the {metric} metric.
Current best score: {best_score}
Current best parameters: {best_params}
Last used parameter ranges: {current_ranges}

1. Decide whether to update the parameter ranges or continue with the current ranges.
2. Suggest the next set of parameters to try.

Provide your response as a JSON object with keys:
    - 'update_param_ranges': boolean
    - 'new_param_ranges': dictionary of new ranges (if update_param_ranges is true)
    - 'next_params': dictionary of next parameter values to try

When there is no improvement to best {metric} for some iterations, consider exploration too, and balance between exploration and exploitation."""

DEFAULT_SUMMARIZATION_PROMPT = """Summarize the conversation history, focusing on the most important information for hyperparameter tuning:
    1. Keep track of the best parameters and scores found so far.
    2. Summarize key trends and decisions made during the optimization.
    3. Maintain a balance between exploration and exploitation.
    4. Ensure the summary is concise while retaining crucial information.

Current conversation history: {conversation_history}
Provide your summary as a list of messages, each with 'role' and 'content' keys."""


class SLLMBO(BaseLLMOptimizer):
    """
    Sequential Large Language Model-Based Hyper-Parameter Optimization.
    
    SLLMBO leverages LLMs with:
    1. Dynamic search space adaptability
    2. Enhanced parameter landscape exploitation
    3. Intelligent summarization of optimization history
    4. Early stopping based on no improvement
    
    This implementation aligns with the reference SLLMBO paper and code.
    
    Attributes:
        param_ranges: Dictionary of parameter ranges with min, max, type.
        n_summarize_iter: Number of iterations between history summarization.
        max_iters_no_improvement: Maximum iterations without improvement before stopping.
        metric: Evaluation metric name.
        model_name_display: Display name for the model being optimized.
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
        metric: str = "objective",
        model_name_display: str = "model",
        system_prompt: Optional[str] = None,
        init_prompt: Optional[str] = None,
        opt_prompt: Optional[str] = None,
        summarization_prompt: Optional[str] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize SLLMBO optimizer.
        
        Args:
            input_dim: Dimensionality of the input space.
            bounds: Tensor of shape (2, input_dim) with [lower, upper] bounds.
            llm_client: Client for LLM API calls (e.g., OpenAI client).
            model_name: Name of the LLM model to use for optimization.
            task_description: Description of the optimization task/problem.
            maximize: Whether to maximize (True) or minimize (False) the objective.
            param_names: Optional names for each dimension/parameter.
            n_summarize_iter: Number of iterations between intelligent summarization.
            max_iters_no_improvement: Max iterations without improvement for early stopping.
            metric: Name of the evaluation metric.
            model_name_display: Display name of the model being optimized (for prompts).
            system_prompt: Custom system prompt (uses default if None).
            init_prompt: Custom initialization prompt (uses default if None).
            opt_prompt: Custom optimization prompt (uses default if None).
            summarization_prompt: Custom summarization prompt (uses default if None).
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
        self.metric = metric
        self.model_name_display = model_name_display
        
        # Prompt templates
        self.system_prompt_template = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.init_prompt_template = init_prompt or DEFAULT_INIT_PROMPT
        self.opt_prompt_template = opt_prompt or DEFAULT_OPT_PROMPT
        self.summarization_prompt_template = summarization_prompt or DEFAULT_SUMMARIZATION_PROMPT
        
        # Initialize parameter ranges from bounds
        self.param_ranges = self._init_param_ranges()
        
        # Tracking state
        self.current_iteration = 0
        self.initialization_iter = 0
        self.iters_no_improvement = 0
        self.best_score = float('-inf') if maximize else float('inf')
        self.best_params: Optional[Dict[str, float]] = None
        
        # System message (initialized lazily)
        self._system_message: Optional[str] = None
        
    def _init_param_ranges(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize parameter ranges from bounds.
        
        Returns:
            Dictionary mapping parameter names to their range specifications.
        """
        ranges = {}
        for i, name in enumerate(self.param_names):
            ranges[name] = {
                "min": self.bounds[0, i].item(),
                "max": self.bounds[1, i].item(),
                "type": "continuous"
            }
        return ranges
    
    @property
    def system_message(self) -> str:
        """Get the formatted system message."""
        if self._system_message is None:
            direction = "maximize" if self.maximize else "minimize"
            self._system_message = self.system_prompt_template.format(
                model_name=self.model_name_display,
                problem_description=self.task_description,
                metric=self.metric,
                direction=direction,
            )
        return self._system_message

    def _format_param_dict(self, x: Tensor) -> Dict[str, float]:
        """
        Convert tensor to parameter dictionary.
        
        Args:
            x: Parameter tensor of shape (input_dim,).
            
        Returns:
            Dictionary mapping parameter names to values.
        """
        return {
            name: x[i].item()
            for i, name in enumerate(self.param_names)
        }
    
    def _parse_param_dict(self, params: Dict[str, Any]) -> Tensor:
        """
        Convert parameter dictionary to tensor.
        
        Args:
            params: Dictionary mapping parameter names to values.
            
        Returns:
            Parameter tensor of shape (input_dim,).
        """
        values = []
        for name in self.param_names:
            val = params.get(name, 0.0)
            # Handle case where value might be a dict (e.g., from range)
            if isinstance(val, dict):
                val = (val.get("min", 0.0) + val.get("max", 0.0)) / 2
            values.append(float(val))
        return torch.tensor(values, device=self.device, dtype=self.dtype)
    
    def _extract_json(self, response: str) -> Dict[str, Any]:
        """
        Extract JSON from LLM response, handling various formats.
        
        Args:
            response: Raw LLM response string.
            
        Returns:
            Parsed dictionary from the response.
        """
        try:
            # First, try to parse the entire response as JSON
            parsed = json.loads(response)
            if isinstance(parsed, dict):
                return parsed
            # If it's a list, try to find a dict in it or wrap it
            if isinstance(parsed, list) and len(parsed) > 0:
                if isinstance(parsed[0], dict):
                    return parsed[0]  # Return first dict
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON object from the response
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
        
        # Fallback: extract key-value pairs
        return self._extract_key_value_pairs(response)
    
    def _extract_key_value_pairs(self, response: str) -> Dict[str, Any]:
        """
        Extract key-value pairs from non-JSON response.
        
        Args:
            response: Response string to parse.
            
        Returns:
            Extracted dictionary.
        """
        pairs = {}
        lines = response.split("\n")
        for line in lines:
            match = re.match(r'^\s*(["\']?)(\w+)\1\s*:\s*(.+)$', line)
            if match:
                key, value = match.group(2), match.group(3)
                try:
                    pairs[key] = json.loads(value)
                except json.JSONDecodeError:
                    pairs[key] = value.strip()
        return pairs
    
    def _call_llm(
        self,
        user_message: str,
        curr_iter: int,
    ) -> str:
        """
        Call the LLM with the given prompt (single-turn conversation).
        
        Each call is independent - only system message and user message are sent.
        No conversation history is maintained between calls.
        
        For reasoning models (e.g., Thinking models), the thinking process
        wrapped in <think> tags is automatically stripped from the response.
        
        Args:
            user_message: The user prompt to send.
            curr_iter: Current iteration number (kept for API compatibility).
            
        Returns:
            LLM response string (with thinking content stripped).
        """
        # Single-turn conversation: only system + user message
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": user_message},
        ]
        
        response = self.llm_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
        )
        assistant_response = response.choices[0].message.content
        
        # Strip thinking content for reasoning models
        assistant_response = self._strip_thinking_content(assistant_response)
        
        # Track query count
        self.llm_query_count += 1
        
        return assistant_response
    
    def _strip_thinking_content(self, response: str) -> str:
        """Strip <think>...</think> and similar tags from reasoning model output."""
        import re
        if response is None:
            return ""
        patterns = [
            r'<think>.*?</think>',
            r'<thinking>.*?</thinking>',
            r'<thought>.*?</thought>',
            r'<reasoning>.*?</reasoning>',
        ]
        result = response
        for pattern in patterns:
            result = re.sub(pattern, '', result, flags=re.DOTALL | re.IGNORECASE)
        return result.strip()
    
    def _intelligent_summarize(self) -> None:
        """
        Summarize conversation history using LLM to reduce context length.
        
        This method replaces the conversation history with a summarized version
        that retains key optimization insights.
        """
        if len(self.conversation_history) < 4:
            return
        
        prompt = self.summarization_prompt_template.format(
            conversation_history=json.dumps(self.conversation_history)
        )
        
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": prompt},
        ]
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
            )
            summary_response = response.choices[0].message.content
            
            # Try to parse as list of messages
            summarized_history = json.loads(summary_response)
            if isinstance(summarized_history, list):
                self.conversation_history = summarized_history
                
        except (json.JSONDecodeError, Exception) as e:
            # Keep original history on failure
            pass
    
    def initialize_optimization(self) -> Dict[str, Any]:
        """
        Initialize optimization by querying LLM for starting parameters and ranges.
        
        Returns:
            Dictionary with 'param_ranges' and 'initial_params'.
        """
        response = self._call_llm(
            user_message=self.init_prompt_template,
            curr_iter=self.initialization_iter,
        )
        
        return self._parse_init_response(response)
    
    def _parse_init_response(self, response: str) -> Dict[str, Any]:
        """
        Parse initialization response from LLM.
        
        Args:
            response: LLM response string.
            
        Returns:
            Dictionary with param_ranges and initial_params.
        """
        parsed = self._extract_json(response)
        
        if not isinstance(parsed, dict):
            raise ValueError("Response is not a dictionary")
        
        # Check for required keys
        if "param_ranges" in parsed and "initial_params" in parsed:
            return parsed
        
        # Try to infer param_ranges and initial_params
        param_ranges, initial_params = self._infer_params(parsed)
        return {
            "param_ranges": param_ranges,
            "initial_params": initial_params,
        }
    
    def _infer_params(
        self, 
        parsed_response: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Infer param_ranges and initial_params from a parsed response.
        
        Args:
            parsed_response: Parsed dictionary from LLM.
            
        Returns:
            Tuple of (param_ranges, initial_params).
        """
        param_ranges = {}
        initial_params = {}
        
        for key, value in parsed_response.items():
            if isinstance(value, list):
                param_ranges[key] = value
                initial_params[key] = value[0] if len(value) > 0 else None
            elif isinstance(value, dict) and "min" in value and "max" in value:
                param_ranges[key] = [value["min"], value["max"]]
                initial_params[key] = (value["min"] + value["max"]) / 2
            else:
                param_ranges[key] = value
                initial_params[key] = value
                
        return param_ranges, initial_params
    
    def get_next_parameters_and_ranges(
        self,
        current_ranges: Dict[str, Any],
        best_score: float,
        best_params: Dict[str, Any],
        curr_iter: int,
    ) -> Dict[str, Any]:
        """
        Get the next set of parameters to try and decide whether to update ranges.
        
        Args:
            current_ranges: Current parameter ranges.
            best_score: Best score achieved so far.
            best_params: Parameters that achieved best score.
            curr_iter: Current iteration number.
            
        Returns:
            Dictionary with update_param_ranges, new_param_ranges (optional), next_params.
        """
        direction = "maximize" if self.maximize else "minimize"
        
        prompt = self.opt_prompt_template.format(
            model_name=self.model_name_display,
            direction=direction,
            metric=self.metric,
            best_score=best_score,
            best_params=json.dumps(best_params),
            current_ranges=json.dumps(current_ranges),
        )
        
        response = self._call_llm(
            user_message=prompt,
            curr_iter=curr_iter,
        )
        
        return self._parse_opt_response(response, best_params)
    
    def _parse_opt_response(
        self, 
        response: str, 
        default_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Parse optimization step response from LLM.
        
        Args:
            response: LLM response string.
            default_params: Default parameters to use on parse failure.
            
        Returns:
            Dictionary with update_param_ranges, new_param_ranges, next_params.
        """
        try:
            parsed = self._extract_json(response)
            
            if not isinstance(parsed, dict):
                raise ValueError("Response is not a dictionary")
            
            # Check for required keys
            if "update_param_ranges" in parsed and "next_params" in parsed:
                return parsed
            
            # Try to infer from response
            update_param_ranges, next_params = self._infer_next_params(
                parsed, default_params
            )
            return {
                "update_param_ranges": update_param_ranges,
                "next_params": next_params,
            }
            
        except (json.JSONDecodeError, ValueError):
            return {
                "update_param_ranges": False,
                "next_params": default_params
            }
    
    def _infer_next_params(
        self,
        parsed_response: Dict[str, Any],
        default_params: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Infer update_param_ranges and next_params from a parsed response.
        
        Args:
            parsed_response: Parsed dictionary from LLM.
            default_params: Default parameters to use as base.
            
        Returns:
            Tuple of (update_param_ranges, next_params).
        """
        update_param_ranges = False
        next_params = default_params.copy()
        
        for key, value in parsed_response.items():
            if key.lower() == "update_param_ranges":
                update_param_ranges = bool(value)
            elif isinstance(value, (int, float, str, bool)):
                next_params[key] = value
                
        return update_param_ranges, next_params
    
    def _check_improvement(self, score: float) -> bool:
        """
        Check if the new score is an improvement.
        
        Args:
            score: New score to check.
            
        Returns:
            True if score is an improvement, False otherwise.
        """
        if self.maximize:
            return score >= self.best_score
        else:
            return score <= self.best_score
    
    def suggest(self, n_candidates: int = 1) -> Tensor:
        """
        Suggest next parameter configuration.
        
        This method implements the main SLLMBO optimization loop, querying
        the LLM for parameter suggestions based on optimization history.
        
        Args:
            n_candidates: Number of candidates to suggest (currently supports 1).
            
        Returns:
            Tensor of shape (n_candidates, input_dim) with suggested points.
        """
        # First iteration: initialize optimization
        if self.X is None or self.y is None:
            try:
                init_result = self.initialize_optimization()
                
                # Update param_ranges if provided
                if "param_ranges" in init_result:
                    self.param_ranges = init_result["param_ranges"]
                
                # Get initial parameters
                initial_params = init_result.get("initial_params", {})
                
                # Convert to tensor
                point = self._parse_param_dict(initial_params)
                return self.clip_to_bounds(point.unsqueeze(0))
            
            except Exception:
                # Fallback to center of bounds
                center = (self.bounds[0] + self.bounds[1]) / 2
                return center.unsqueeze(0)
            
        # Get current best
        if self.maximize:
            best_idx = self.y.argmax()
        else:
            best_idx = self.y.argmin()
            
        best_x = self.X[best_idx]
        best_y = self.y[best_idx].item()
        
        # Update best tracking
        if self._check_improvement(best_y):
            self.best_score = best_y
            self.best_params = self._format_param_dict(best_x)
            self.iters_no_improvement = 0
        else:
            self.iters_no_improvement += 1
            
        # Ensure we have best_params initialized
        if self.best_params is None:
            self.best_params = self._format_param_dict(best_x)
        
        # Increment iteration counter
        self.current_iteration += 1
        
        try:
            # Get next parameters from LLM
            next_step = self.get_next_parameters_and_ranges(
                current_ranges=self.param_ranges,
                best_score=self.best_score,
                best_params=self.best_params,
                curr_iter=self.current_iteration,
            )
            
            # Update ranges if suggested
            if next_step.get("update_param_ranges", False):
                new_ranges = next_step.get("new_param_ranges", {})
                if new_ranges:
                    for name, range_info in new_ranges.items():
                        if name in self.param_ranges:
                            if isinstance(range_info, dict):
                                self.param_ranges[name].update(range_info)
                            elif isinstance(range_info, list) and len(range_info) == 2:
                                self.param_ranges[name]["min"] = range_info[0]
                                self.param_ranges[name]["max"] = range_info[1]
            
            # Parse suggested parameters
            next_params = next_step.get("next_params", {})
            point = self._parse_param_dict(next_params)
            
        except Exception:
            # Fallback: sample near best with noise
            if self.best_params:
                best_tensor = self._parse_param_dict(self.best_params)
                noise = torch.randn(self.input_dim, device=self.device, dtype=self.dtype)
                noise = noise * (self.bounds[1] - self.bounds[0]) * 0.1
                point = best_tensor + noise
            else:
                point = self.random_points(1).squeeze(0)
                
        return self.clip_to_bounds(point.unsqueeze(0))
    
    def should_stop(self) -> bool:
        """
        Check if optimization should stop early.
        
        Returns:
            True if no improvement for max_iters_no_improvement iterations.
        """
        return self.iters_no_improvement >= self.max_iters_no_improvement

    def get_stats(self) -> Dict[str, Any]:
        """
        Get optimization statistics.
        
        Returns:
            Dictionary with optimization statistics including SLLMBO-specific metrics.
        """
        stats = super().get_stats()
        stats.update({
            "current_iteration": self.current_iteration,
            "iters_no_improvement": self.iters_no_improvement,
            "best_score": self.best_score,
            "best_params": self.best_params,
            "param_ranges": self.param_ranges,
        })
        return stats
