"""
Base class for LLM-based Bayesian Optimization methods.

This module provides the abstract base class that all LLM-enhanced BO
methods should inherit from. It defines the common interface and utilities
shared across different LLM-BO implementations.
"""

import torch
from torch import Tensor
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Dict, Any, Callable
import json
import os


# Default device and dtype
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


class BaseLLMOptimizer(ABC):
    """
    Abstract base class for LLM-based Bayesian Optimization methods.
    
    This class provides the common interface for all LLM-enhanced BO algorithms.
    Subclasses should implement the specific optimization logic while leveraging
    the LLM for various tasks such as candidate generation, surrogate modeling,
    or acquisition function optimization.
    
    Attributes:
        input_dim: Dimensionality of the input space.
        bounds: Tensor of shape (2, input_dim) specifying [lower, upper] bounds.
        device: Torch device for computations.
        dtype: Torch data type for tensors.
        X: Observed input points.
        y: Observed function values.
        llm_client: Client for LLM API calls.
        model_name: Name of the LLM model to use.
        task_description: Description of the optimization task for LLM context.
    """
    
    def __init__(
        self,
        input_dim: int,
        bounds: Tensor,
        llm_client: Any = None,
        model_name: str = "gpt-4",
        task_description: str = "",
        maximize: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize the LLM-based optimizer.
        
        Args:
            input_dim: Dimensionality of the input space.
            bounds: Tensor of shape (2, input_dim) with [lower, upper] bounds.
            llm_client: Client for LLM API calls (e.g., OpenAI client).
            model_name: Name of the LLM model to use.
            task_description: Description of the optimization task.
            maximize: Whether to maximize (True) or minimize (False) the objective.
            device: Torch device for computations.
            dtype: Torch data type for tensors.
        """
        self.input_dim = input_dim
        self.device = device or DEVICE
        self.dtype = dtype or DTYPE
        self.bounds = bounds.to(device=self.device, dtype=self.dtype)
        
        self.llm_client = llm_client
        self.model_name = model_name
        self.task_description = task_description
        self.maximize = maximize
        
        # Observation storage
        self.X: Optional[Tensor] = None
        self.y: Optional[Tensor] = None
        
        # LLM interaction tracking
        self.llm_query_count = 0
        self.llm_query_cost = 0.0
        self.conversation_history: List[Dict[str, str]] = []
        
    def initialize(self, X: Tensor, y: Tensor) -> None:
        """
        Initialize the optimizer with observed data.
        
        Args:
            X: Initial observed points of shape (n, input_dim).
            y: Initial observed values of shape (n, 1) or (n,).
        """
        self.X = X.to(device=self.device, dtype=self.dtype)
        self.y = y.to(device=self.device, dtype=self.dtype)
        if self.y.dim() == 1:
            self.y = self.y.unsqueeze(-1)
            
    def update(self, X_new: Tensor, y_new: Tensor) -> None:
        """
        Update the optimizer with new observations.
        
        Args:
            X_new: New observed points.
            y_new: New observed values.
        """
        X_new = X_new.to(device=self.device, dtype=self.dtype)
        y_new = y_new.to(device=self.device, dtype=self.dtype)
        if y_new.dim() == 1:
            y_new = y_new.unsqueeze(-1)
            
        if self.X is None:
            self.X = X_new
            self.y = y_new
        else:
            self.X = torch.cat([self.X, X_new], dim=0)
            self.y = torch.cat([self.y, y_new], dim=0)
            
    @abstractmethod
    def suggest(self, n_candidates: int = 1) -> Tensor:
        """
        Suggest new candidate points to evaluate.
        
        Args:
            n_candidates: Number of candidates to suggest.
            
        Returns:
            Tensor of shape (n_candidates, input_dim) with suggested points.
        """
        pass
    
    def get_best(self) -> Tuple[Tensor, Tensor]:
        """
        Get the best observed point and value.
        
        Returns:
            Tuple of (best_x, best_y) tensors.
        """
        if self.X is None or self.y is None:
            raise ValueError("No observations available.")
            
        if self.maximize:
            best_idx = self.y.argmax()
        else:
            best_idx = self.y.argmin()
            
        return self.X[best_idx], self.y[best_idx]
    
    def query_llm(
        self,
        prompt: str,
        system_message: str = "",
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """
        Query the LLM with a prompt (single-turn conversation).
        
        Each call is independent - no conversation history is used.
        The system_message and prompt are sent as a fresh conversation.
        
        For reasoning models (e.g., Thinking models), the thinking process
        wrapped in <think> tags is automatically stripped from the response.
        
        Args:
            prompt: The user prompt to send to the LLM.
            system_message: Optional system message for context.
            temperature: Sampling temperature.
            **kwargs: Additional parameters (ignored for compatibility).
            
        Returns:
            The LLM's response as a string (with thinking content stripped).
        """
        if self.llm_client is None:
            raise ValueError("LLM client not configured.")
        
        # Single-turn conversation: only system + user message, no history
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        # No max_tokens limit - let model generate complete response
        response = self.llm_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
        )
        
        assistant_response = response.choices[0].message.content
        
        # Strip thinking content for reasoning models
        assistant_response = self._strip_thinking_content(assistant_response)
        
        # Track query count (conversation_history is kept for logging but not used in calls)
        self.llm_query_count += 1
        
        return assistant_response
    
    def _strip_thinking_content(self, response: str) -> str:
        """
        Strip thinking/reasoning content from model response.
        
        Reasoning models (like Qwen-Thinking) wrap their reasoning process
        in <think>...</think> tags. This method extracts only the final answer.
        
        Args:
            response: Raw LLM response string.
            
        Returns:
            Response with thinking content removed.
        """
        import re
        
        if response is None:
            return ""
        
        # Pattern to match <think>...</think> blocks (including multiline)
        # Also handles variations like <thinking>...</thinking>
        patterns = [
            r'<think>.*?</think>',
            r'<thinking>.*?</thinking>',
            r'<thought>.*?</thought>',
            r'<reasoning>.*?</reasoning>',
        ]
        
        result = response
        for pattern in patterns:
            result = re.sub(pattern, '', result, flags=re.DOTALL | re.IGNORECASE)
        
        # Clean up extra whitespace
        result = result.strip()
        
        return result
    
    def format_observations_for_prompt(
        self,
        max_points: int = 20,
        include_bounds: bool = True,
    ) -> str:
        """
        Format observed data for inclusion in LLM prompts.
        
        Args:
            max_points: Maximum number of points to include.
            include_bounds: Whether to include bound information.
            
        Returns:
            Formatted string representation of observations.
        """
        if self.X is None or self.y is None:
            return "No observations available."
            
        n_points = min(len(self.X), max_points)
        
        # Sort by value for better context
        if self.maximize:
            sorted_indices = torch.argsort(self.y.squeeze(), descending=True)
        else:
            sorted_indices = torch.argsort(self.y.squeeze(), descending=False)
            
        lines = []
        
        if include_bounds:
            lower = self.bounds[0].tolist()
            upper = self.bounds[1].tolist()
            lines.append(f"Search space bounds: [{lower}, {upper}]")
            lines.append(f"Dimension: {self.input_dim}")
            lines.append("")
            
        lines.append("Past evaluations (sorted by value):")
        for i in range(n_points):
            idx = sorted_indices[i]
            x = self.X[idx].tolist()
            y_val = self.y[idx].item()
            lines.append(f"  x = {x}, f(x) = {y_val:.6f}")
            
        return "\n".join(lines)
    
    def parse_point_from_response(self, response: str) -> Optional[Tensor]:
        """
        Parse a point from LLM response.
        
        Args:
            response: LLM response string.
            
        Returns:
            Parsed point as tensor, or None if parsing fails.
        """
        try:
            # Try to extract JSON array
            import re
            
            # Look for array pattern
            match = re.search(r'\[[\d\s,.\-e]+\]', response)
            if match:
                point_list = json.loads(match.group())
                if len(point_list) == self.input_dim:
                    point = torch.tensor(point_list, device=self.device, dtype=self.dtype)
                    # Clip to bounds
                    point = torch.clamp(point, self.bounds[0], self.bounds[1])
                    return point
                    
            return None
        except (json.JSONDecodeError, ValueError):
            return None
            
    def clip_to_bounds(self, X: Tensor) -> Tensor:
        """
        Clip points to the search bounds.
        
        Args:
            X: Points to clip.
            
        Returns:
            Clipped points.
        """
        return torch.clamp(X, self.bounds[0], self.bounds[1])
    
    def random_points(self, n: int) -> Tensor:
        """
        Generate random points within the search bounds.
        
        Args:
            n: Number of points to generate.
            
        Returns:
            Random points tensor of shape (n, input_dim).
        """
        return (
            torch.rand(n, self.input_dim, device=self.device, dtype=self.dtype)
            * (self.bounds[1] - self.bounds[0])
            + self.bounds[0]
        )
        
    def clear_history(self) -> None:
        """Clear the LLM conversation history."""
        self.conversation_history = []
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get optimization statistics.
        
        Returns:
            Dictionary with optimization statistics.
        """
        stats = {
            "n_observations": len(self.X) if self.X is not None else 0,
            "llm_query_count": self.llm_query_count,
            "llm_query_cost": self.llm_query_cost,
        }
        
        if self.X is not None and self.y is not None:
            best_x, best_y = self.get_best()
            stats["best_x"] = best_x.tolist()
            stats["best_y"] = best_y.item()
            
        return stats

