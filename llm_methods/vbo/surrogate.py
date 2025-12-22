"""
Verbalized Surrogate Model

Two strategies:
1. In-Context: Direct inference from historical observations
2. Belief-Based: Maintain a textual belief about the black-box function

Two output modes:
1. Text-only: Pure natural language description of distribution
2. Text+Numeric: Natural language + mean and std values
"""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor


class SurrogateMode(Enum):
    """Surrogate model strategy mode."""
    IN_CONTEXT = "in_context"
    BELIEF_BASED = "belief_based"


class OutputMode(Enum):
    """Output mode for predictions."""
    TEXT_ONLY = "text_only"
    TEXT_WITH_NUMERIC = "text_with_numeric"


@dataclass
class SurrogatePrediction:
    """
    Prediction result from surrogate model.
    
    Attributes:
        x: Input point
        distribution_text: Natural language description of value distribution
        confidence_text: Natural language description of confidence
        reasoning: Chain-of-thought reasoning
        mean: Predicted mean value (optional, for TEXT_WITH_NUMERIC mode)
        std: Predicted standard deviation (optional, for TEXT_WITH_NUMERIC mode)
    """
    x: Any
    distribution_text: str
    confidence_text: str
    reasoning: str
    mean: Optional[float] = None
    std: Optional[float] = None
    
    def has_numeric(self) -> bool:
        """Check if numeric values are available."""
        return self.mean is not None and self.std is not None


# =============================================================================
# Prompt Templates
# =============================================================================

IN_CONTEXT_PROMPT_TEXT_ONLY = """You are a Bayesian optimization surrogate model that predicts function values using natural language.

Task: {task_description}

Historical observations (x -> y):
{history}

Predict the function value at x = {x_query}

Provide your prediction as a natural language description:
1. **Distribution Description**: Describe the likely distribution of function values at this point.
   - What range is most probable?
   - Is the distribution unimodal, multimodal, or skewed?
   - Example: "Most likely between 0.6 and 0.8, with peak probability around 0.72"

2. **Confidence Narrative**: Explain your confidence level and its basis.
   - How close is this point to observed data?
   - What evidence supports or contradicts this prediction?

3. **Reasoning**: Step-by-step reasoning based on the observations.

Output in JSON format:
{{
    "distribution_text": "...",
    "confidence_text": "...",
    "reasoning": "..."
}}"""


IN_CONTEXT_PROMPT_WITH_NUMERIC = """You are a Bayesian optimization surrogate model that predicts function values.

Task: {task_description}

Historical observations (x -> y):
{history}

Predict the function value at x = {x_query}

Provide both textual description AND numerical estimates:

1. **Distribution Description**: Natural language description of the value distribution.

2. **Confidence Narrative**: Explain your confidence level.

3. **Reasoning**: Step-by-step reasoning.

4. **Numerical Estimates**:
   - mean: Your best estimate of the function value
   - std: Your uncertainty (standard deviation)

Output in JSON format:
{{
    "distribution_text": "...",
    "confidence_text": "...",
    "reasoning": "...",
    "mean": <number>,
    "std": <number>
}}"""


BELIEF_UPDATE_PROMPT = """You are maintaining a textual belief about an unknown black-box function.

Current belief about the function:
{current_belief}

New observation: f({x_new}) = {y_new}

Task: {task_description}

Update your belief about the function based on this new observation.
Consider:
- Does this confirm or contradict your previous understanding?
- What regions might be promising?
- What is the overall shape/behavior of the function?

Output your updated belief as a comprehensive text description:
{{
    "updated_belief": "..."
}}"""


BELIEF_PREDICT_PROMPT_TEXT_ONLY = """You are a Bayesian optimization surrogate model using a textual belief about the function.

Task: {task_description}

Your current belief about the function:
{current_belief}

Historical observations for reference:
{history}

Predict the function value at x = {x_query}

Based on your belief, provide:
1. **Distribution Description**: Natural language description of likely values
2. **Confidence Narrative**: How confident are you based on your belief?
3. **Reasoning**: How does your belief inform this prediction?

Output in JSON format:
{{
    "distribution_text": "...",
    "confidence_text": "...",
    "reasoning": "..."
}}"""


BELIEF_PREDICT_PROMPT_WITH_NUMERIC = """You are a Bayesian optimization surrogate model using a textual belief about the function.

Task: {task_description}

Your current belief about the function:
{current_belief}

Historical observations for reference:
{history}

Predict the function value at x = {x_query}

Based on your belief, provide both textual and numerical predictions:
1. **Distribution Description**: Natural language description
2. **Confidence Narrative**: Confidence explanation
3. **Reasoning**: Reasoning from belief
4. **Numerical**: mean and std estimates

Output in JSON format:
{{
    "distribution_text": "...",
    "confidence_text": "...",
    "reasoning": "...",
    "mean": <number>,
    "std": <number>
}}"""


INITIAL_BELIEF_PROMPT = """You are initializing a belief about an unknown black-box function.

Task: {task_description}

Parameter space:
{param_space}

Provide an initial belief about this function. Consider:
- What behavior might be expected based on the task?
- What are reasonable assumptions about smoothness, modality, etc.?
- What regions might be worth exploring first?

Output your initial belief:
{{
    "initial_belief": "..."
}}"""


# =============================================================================
# Surrogate Model Implementations
# =============================================================================

class VBOSurrogateModel(ABC):
    """Abstract base class for Verbalized Surrogate Models."""
    
    def __init__(
        self,
        llm_client: Any,
        output_mode: OutputMode = OutputMode.TEXT_WITH_NUMERIC,
        task_description: str = "",
        maximize: bool = False,
    ):
        self.llm_client = llm_client
        self.output_mode = output_mode
        self.task_description = task_description
        self.maximize = maximize
        self.history: List[Tuple[Any, float]] = []
    
    @abstractmethod
    def predict(self, x: Any) -> SurrogatePrediction:
        """Predict function value at x."""
        pass
    
    @abstractmethod
    def update(self, x: Any, y: float) -> None:
        """Update model with new observation."""
        pass
    
    def _format_history(self) -> str:
        """Format observation history as string."""
        if not self.history:
            return "No observations yet."
        
        lines = []
        for x, y in self.history:
            if isinstance(x, (list, tuple)):
                x_str = ", ".join(f"{v:.4f}" for v in x)
            elif isinstance(x, Tensor):
                x_str = ", ".join(f"{v:.4f}" for v in x.tolist())
            else:
                x_str = f"{x:.4f}" if isinstance(x, float) else str(x)
            lines.append(f"  x = [{x_str}] -> y = {y:.4f}")
        
        return "\n".join(lines)
    
    def _format_x(self, x: Any) -> str:
        """Format input x as string."""
        if isinstance(x, (list, tuple)):
            return "[" + ", ".join(f"{v:.4f}" for v in x) + "]"
        elif isinstance(x, Tensor):
            return "[" + ", ".join(f"{v:.4f}" for v in x.tolist()) + "]"
        else:
            return f"{x:.4f}" if isinstance(x, float) else str(x)
    
    def _query_llm(self, prompt: str) -> str:
        """Query the LLM and return response."""
        response = self.llm_client.chat.completions.create(
            model=self.llm_client.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    
    def _parse_prediction(self, response: str, x: Any) -> SurrogatePrediction:
        """Parse LLM response into SurrogatePrediction."""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                # Fallback: try to parse entire response as JSON
                data = json.loads(response)
            
            return SurrogatePrediction(
                x=x,
                distribution_text=data.get("distribution_text", response[:200]),
                confidence_text=data.get("confidence_text", "Moderate confidence"),
                reasoning=data.get("reasoning", "Based on available data"),
                mean=data.get("mean"),
                std=data.get("std"),
            )
        except json.JSONDecodeError:
            # Fallback for non-JSON responses
            return SurrogatePrediction(
                x=x,
                distribution_text=response[:300],
                confidence_text="Confidence level uncertain",
                reasoning=response,
                mean=None,
                std=None,
            )


class InContextSurrogate(VBOSurrogateModel):
    """
    In-Context Surrogate Model.
    
    Directly infers predictions from historical observations without
    maintaining an explicit belief state.
    """
    
    def __init__(
        self,
        llm_client: Any,
        output_mode: OutputMode = OutputMode.TEXT_WITH_NUMERIC,
        task_description: str = "",
        maximize: bool = False,
    ):
        super().__init__(llm_client, output_mode, task_description, maximize)
    
    def predict(self, x: Any) -> SurrogatePrediction:
        """Predict function value using in-context learning."""
        if self.output_mode == OutputMode.TEXT_ONLY:
            prompt_template = IN_CONTEXT_PROMPT_TEXT_ONLY
        else:
            prompt_template = IN_CONTEXT_PROMPT_WITH_NUMERIC
        
        prompt = prompt_template.format(
            task_description=self.task_description or "Optimize an unknown function",
            history=self._format_history(),
            x_query=self._format_x(x),
        )
        
        response = self._query_llm(prompt)
        return self._parse_prediction(response, x)
    
    def update(self, x: Any, y: float) -> None:
        """Add observation to history."""
        self.history.append((x, y))


class BeliefBasedSurrogate(VBOSurrogateModel):
    """
    Belief-Based Surrogate Model.
    
    Maintains a textual belief about the black-box function that is
    updated with each observation. Predictions are made based on this belief.
    """
    
    def __init__(
        self,
        llm_client: Any,
        output_mode: OutputMode = OutputMode.TEXT_WITH_NUMERIC,
        task_description: str = "",
        maximize: bool = False,
        param_space_description: str = "",
    ):
        super().__init__(llm_client, output_mode, task_description, maximize)
        self.param_space_description = param_space_description
        self.current_belief: str = ""
        self.belief_history: List[str] = []
        
        # Initialize belief if we have task info
        if task_description:
            self._initialize_belief()
    
    def _initialize_belief(self) -> None:
        """Initialize the belief about the function."""
        prompt = INITIAL_BELIEF_PROMPT.format(
            task_description=self.task_description,
            param_space=self.param_space_description or "Continuous parameters",
        )
        
        response = self._query_llm(prompt)
        
        try:
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                self.current_belief = data.get("initial_belief", response)
            else:
                self.current_belief = response
        except json.JSONDecodeError:
            self.current_belief = response
        
        self.belief_history.append(self.current_belief)
    
    def predict(self, x: Any) -> SurrogatePrediction:
        """Predict function value based on current belief."""
        if not self.current_belief:
            self._initialize_belief()
        
        if self.output_mode == OutputMode.TEXT_ONLY:
            prompt_template = BELIEF_PREDICT_PROMPT_TEXT_ONLY
        else:
            prompt_template = BELIEF_PREDICT_PROMPT_WITH_NUMERIC
        
        prompt = prompt_template.format(
            task_description=self.task_description or "Optimize an unknown function",
            current_belief=self.current_belief,
            history=self._format_history(),
            x_query=self._format_x(x),
        )
        
        response = self._query_llm(prompt)
        return self._parse_prediction(response, x)
    
    def update(self, x: Any, y: float) -> None:
        """Update belief with new observation."""
        # Add to history
        self.history.append((x, y))
        
        # Update belief
        prompt = BELIEF_UPDATE_PROMPT.format(
            current_belief=self.current_belief,
            x_new=self._format_x(x),
            y_new=f"{y:.4f}",
            task_description=self.task_description,
        )
        
        response = self._query_llm(prompt)
        
        try:
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                self.current_belief = data.get("updated_belief", response)
            else:
                self.current_belief = response
        except json.JSONDecodeError:
            self.current_belief = response
        
        self.belief_history.append(self.current_belief)
    
    def get_belief(self) -> str:
        """Get current belief text."""
        return self.current_belief
    
    def get_belief_history(self) -> List[str]:
        """Get history of belief updates."""
        return self.belief_history.copy()


def create_surrogate(
    mode: SurrogateMode,
    llm_client: Any,
    output_mode: OutputMode = OutputMode.TEXT_WITH_NUMERIC,
    task_description: str = "",
    maximize: bool = False,
    **kwargs,
) -> VBOSurrogateModel:
    """
    Factory function to create a surrogate model.
    
    Args:
        mode: IN_CONTEXT or BELIEF_BASED
        llm_client: LLM client for queries
        output_mode: TEXT_ONLY or TEXT_WITH_NUMERIC
        task_description: Description of the optimization task
        maximize: Whether to maximize (True) or minimize (False)
        **kwargs: Additional arguments for specific surrogate types
    
    Returns:
        Configured surrogate model
    """
    if mode == SurrogateMode.IN_CONTEXT:
        return InContextSurrogate(
            llm_client=llm_client,
            output_mode=output_mode,
            task_description=task_description,
            maximize=maximize,
        )
    elif mode == SurrogateMode.BELIEF_BASED:
        return BeliefBasedSurrogate(
            llm_client=llm_client,
            output_mode=output_mode,
            task_description=task_description,
            maximize=maximize,
            param_space_description=kwargs.get("param_space_description", ""),
        )
    else:
        raise ValueError(f"Unknown surrogate mode: {mode}")
