"""
Verbalized Acquisition Function

Evaluates acquisition value using LLM-based textual reasoning.

Two output modes:
1. Text-only: Pure natural language evaluation
2. Text+Numeric: Natural language + numerical acquisition score
"""

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from .surrogate import SurrogatePrediction


class AcquisitionOutputMode(Enum):
    """Output mode for acquisition evaluation."""
    TEXT_ONLY = "text_only"
    TEXT_WITH_NUMERIC = "text_with_numeric"


@dataclass
class AcquisitionEvaluation:
    """
    Result of acquisition function evaluation.
    
    Attributes:
        x: Candidate point
        evaluation_text: Natural language evaluation of acquisition value
        exploration_text: Analysis of exploration value
        exploitation_text: Analysis of exploitation value
        reasoning: Chain-of-thought reasoning
        score: Numerical acquisition score (optional, for TEXT_WITH_NUMERIC mode)
    """
    x: Any
    evaluation_text: str
    exploration_text: str
    exploitation_text: str
    reasoning: str
    score: Optional[float] = None
    
    def has_numeric(self) -> bool:
        """Check if numeric score is available."""
        return self.score is not None


# =============================================================================
# Prompt Templates
# =============================================================================

ACQUISITION_PROMPT_TEXT_ONLY = """You are an acquisition function for Bayesian optimization.
Your role is to evaluate whether a candidate point is worth evaluating.

Task: {task_description}
Goal: {goal}

Current best observed value: {best_value}
Number of observations so far: {n_observations}
Optimization phase: {phase}

Candidate point: x = {x_candidate}

Surrogate model prediction for this point:
{prediction_text}

Evaluate this candidate point by considering:

1. **Exploration Value**: 
   - How uncertain is the prediction at this point?
   - Is this an under-explored region?
   - Could we learn something new by evaluating here?

2. **Exploitation Value**:
   - How likely is this point to improve on the current best?
   - What is the expected improvement potential?

3. **Risk-Reward Balance**:
   - Given the current optimization phase, should we explore or exploit more?
   - What are the potential downsides of evaluating this point?

4. **Overall Recommendation**:
   - Is this point worth evaluating? Why or why not?
   - How does it compare to other regions we might explore?

Output in JSON format:
{{
    "evaluation_text": "Overall evaluation of this candidate...",
    "exploration_text": "Analysis of exploration value...",
    "exploitation_text": "Analysis of exploitation value...",
    "reasoning": "Step-by-step reasoning..."
}}"""


ACQUISITION_PROMPT_WITH_NUMERIC = """You are an acquisition function for Bayesian optimization.
Your role is to evaluate whether a candidate point is worth evaluating.

Task: {task_description}
Goal: {goal}

Current best observed value: {best_value}
Number of observations so far: {n_observations}
Optimization phase: {phase}

Candidate point: x = {x_candidate}

Surrogate model prediction for this point:
{prediction_text}

Evaluate this candidate point by considering:

1. **Exploration Value**: 
   - How uncertain is the prediction at this point?
   - Is this an under-explored region?

2. **Exploitation Value**:
   - How likely is this point to improve on the current best?
   - What is the expected improvement potential?

3. **Risk-Reward Balance**:
   - Given the current phase, should we explore or exploit more?

4. **Score Assignment** (0-10 scale):
   - 0-3: Poor candidate, unlikely to be useful
   - 4-6: Moderate candidate, might be worth trying
   - 7-10: Excellent candidate, high potential

Output in JSON format:
{{
    "evaluation_text": "Overall evaluation...",
    "exploration_text": "Exploration analysis...",
    "exploitation_text": "Exploitation analysis...",
    "reasoning": "Step-by-step reasoning...",
    "score": <number between 0 and 10>
}}"""


COMPARATIVE_ACQUISITION_PROMPT = """You are an acquisition function for Bayesian optimization.
Compare multiple candidate points and rank them.

Task: {task_description}
Goal: {goal}

Current best observed value: {best_value}
Number of observations: {n_observations}

Candidates to compare:
{candidates_text}

For each candidate, consider:
1. Exploration potential (uncertainty, novelty)
2. Exploitation potential (expected improvement)
3. Risk-reward trade-off

Rank all candidates and provide reasoning.

Output in JSON format:
{{
    "rankings": [
        {{"rank": 1, "candidate_id": <id>, "reasoning": "..."}},
        {{"rank": 2, "candidate_id": <id>, "reasoning": "..."}},
        ...
    ],
    "best_candidate_analysis": "Why the top candidate is best...",
    "overall_strategy": "Current acquisition strategy explanation..."
}}"""


# =============================================================================
# Acquisition Function Implementation
# =============================================================================

class VBOAcquisitionFunction:
    """
    Verbalized Acquisition Function.
    
    Uses LLM to evaluate acquisition value of candidate points based on
    surrogate model predictions.
    """
    
    def __init__(
        self,
        llm_client: Any,
        output_mode: AcquisitionOutputMode = AcquisitionOutputMode.TEXT_WITH_NUMERIC,
        task_description: str = "",
        maximize: bool = False,
        strategy_text: Optional[str] = None,
    ):
        """
        Initialize the acquisition function.
        
        Args:
            llm_client: LLM client for queries
            output_mode: TEXT_ONLY or TEXT_WITH_NUMERIC
            task_description: Description of the optimization task
            maximize: Whether to maximize (True) or minimize (False)
            strategy_text: Optional custom strategy text for evaluation
        """
        self.llm_client = llm_client
        self.output_mode = output_mode
        self.task_description = task_description
        self.maximize = maximize
        self.strategy_text = strategy_text
        
        # Tracking
        self.n_evaluations = 0
        self.evaluation_history: List[AcquisitionEvaluation] = []
    
    def evaluate(
        self,
        x: Any,
        prediction: SurrogatePrediction,
        best_value: float,
        n_observations: int,
        iteration: int = 0,
        total_iterations: int = 100,
    ) -> AcquisitionEvaluation:
        """
        Evaluate acquisition value for a candidate point.
        
        Args:
            x: Candidate point
            prediction: Surrogate model prediction for x
            best_value: Current best observed value
            n_observations: Number of observations so far
            iteration: Current iteration number
            total_iterations: Total planned iterations
            
        Returns:
            AcquisitionEvaluation result
        """
        # Determine optimization phase
        progress = iteration / max(total_iterations, 1)
        if progress < 0.3:
            phase = "Early exploration phase - prioritize discovering the landscape"
        elif progress < 0.7:
            phase = "Middle phase - balance exploration and exploitation"
        else:
            phase = "Late exploitation phase - focus on refinement"
        
        # Format prediction text
        prediction_text = self._format_prediction(prediction)
        
        # Select prompt template
        if self.output_mode == AcquisitionOutputMode.TEXT_ONLY:
            prompt_template = ACQUISITION_PROMPT_TEXT_ONLY
        else:
            prompt_template = ACQUISITION_PROMPT_WITH_NUMERIC
        
        # Build prompt
        prompt = prompt_template.format(
            task_description=self.task_description or "Optimize an unknown function",
            goal="Maximize" if self.maximize else "Minimize",
            best_value=f"{best_value:.4f}",
            n_observations=n_observations,
            phase=phase,
            x_candidate=self._format_x(x),
            prediction_text=prediction_text,
        )
        
        # Query LLM
        response = self._query_llm(prompt)
        
        # Parse response
        evaluation = self._parse_evaluation(response, x)
        
        # Track
        self.n_evaluations += 1
        self.evaluation_history.append(evaluation)
        
        return evaluation
    
    def compare_candidates(
        self,
        candidates: List[tuple],  # List of (x, prediction) pairs
        best_value: float,
        n_observations: int,
    ) -> List[int]:
        """
        Compare multiple candidates and return ranking.
        
        Args:
            candidates: List of (x, prediction) pairs
            best_value: Current best value
            n_observations: Number of observations
            
        Returns:
            List of candidate indices, ordered by rank (best first)
        """
        # Format candidates text
        candidates_text = []
        for i, (x, pred) in enumerate(candidates):
            pred_text = self._format_prediction(pred)
            candidates_text.append(f"Candidate {i}:\n  x = {self._format_x(x)}\n  Prediction:\n{pred_text}")
        
        prompt = COMPARATIVE_ACQUISITION_PROMPT.format(
            task_description=self.task_description,
            goal="Maximize" if self.maximize else "Minimize",
            best_value=f"{best_value:.4f}",
            n_observations=n_observations,
            candidates_text="\n\n".join(candidates_text),
        )
        
        response = self._query_llm(prompt)
        
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                rankings = data.get("rankings", [])
                # Extract ordered indices
                ranked_indices = [r["candidate_id"] for r in sorted(rankings, key=lambda x: x["rank"])]
                return ranked_indices
        except (json.JSONDecodeError, KeyError):
            pass
        
        # Fallback: return original order
        return list(range(len(candidates)))
    
    def _format_prediction(self, prediction: SurrogatePrediction) -> str:
        """Format surrogate prediction as text."""
        lines = [
            f"Distribution: {prediction.distribution_text}",
            f"Confidence: {prediction.confidence_text}",
            f"Reasoning: {prediction.reasoning}",
        ]
        
        if prediction.has_numeric():
            lines.append(f"Numerical estimates: mean = {prediction.mean:.4f}, std = {prediction.std:.4f}")
        
        return "\n".join(lines)
    
    def _format_x(self, x: Any) -> str:
        """Format input x as string."""
        if isinstance(x, (list, tuple)):
            return "[" + ", ".join(f"{v:.4f}" for v in x) + "]"
        elif hasattr(x, 'tolist'):  # Tensor
            return "[" + ", ".join(f"{v:.4f}" for v in x.tolist()) + "]"
        else:
            return f"{x:.4f}" if isinstance(x, float) else str(x)
    
    def _query_llm(self, user_prompt: str, system_prompt: str = None) -> str:
        """
        Query the LLM with a single-turn conversation.
        
        Args:
            user_prompt: The user prompt to send.
            system_prompt: Optional system prompt. If None, uses default.
            
        Returns:
            LLM response string.
        """
        if system_prompt is None:
            system_prompt = (
                "You are an acquisition function for Bayesian optimization. "
                "Your role is to evaluate candidate points and determine which are most "
                "promising to evaluate next, balancing exploration and exploitation."
            )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response = self.llm_client.chat.completions.create(
            model=self.llm_client.model_name,
            messages=messages,
        )
        return response.choices[0].message.content
    
    def _parse_evaluation(self, response: str, x: Any) -> AcquisitionEvaluation:
        """Parse LLM response into AcquisitionEvaluation."""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)
            
            # Parse score if present
            score = data.get("score")
            if score is not None:
                score = float(score)
                # Normalize to [0, 1] for internal use
                score = max(0.0, min(10.0, score)) / 10.0
            
            return AcquisitionEvaluation(
                x=x,
                evaluation_text=data.get("evaluation_text", response[:200]),
                exploration_text=data.get("exploration_text", ""),
                exploitation_text=data.get("exploitation_text", ""),
                reasoning=data.get("reasoning", ""),
                score=score,
            )
        except json.JSONDecodeError:
            # Fallback for non-JSON responses
            return AcquisitionEvaluation(
                x=x,
                evaluation_text=response[:300],
                exploration_text="",
                exploitation_text="",
                reasoning=response,
                score=None,
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get acquisition function statistics."""
        scores = [e.score for e in self.evaluation_history if e.score is not None]
        
        return {
            "n_evaluations": self.n_evaluations,
            "avg_score": sum(scores) / len(scores) if scores else None,
            "output_mode": self.output_mode.value,
        }


class TextGradOptimizableAcquisition(VBOAcquisitionFunction):
    """
    Acquisition function with strategy that can be optimized via Textual Gradient.
    
    The strategy text is treated as a learnable parameter that can be updated
    based on optimization performance feedback.
    """
    
    DEFAULT_STRATEGY = """Acquisition Strategy v1.0:

1. Exploration-Exploitation Balance:
   - In early phases (< 30% budget): Weight exploration at 70%, exploitation at 30%
   - In middle phases (30-70% budget): Equal 50-50 balance
   - In late phases (> 70% budget): Weight exploitation at 80%, exploration at 20%

2. Exploration Scoring:
   - High uncertainty regions: +3 points
   - Far from all observations: +2 points
   - Near promising but unexplored areas: +2 points

3. Exploitation Scoring:
   - High predicted value (above current best): +4 points
   - Near current best with lower uncertainty: +2 points
   - Consistent with observed high-value pattern: +2 points

4. Risk Adjustment:
   - Very high uncertainty with no nearby data: -1 point
   - Already well-explored region: -2 points
   - Contradicts observed trends: -2 points
"""
    
    def __init__(
        self,
        llm_client: Any,
        output_mode: AcquisitionOutputMode = AcquisitionOutputMode.TEXT_WITH_NUMERIC,
        task_description: str = "",
        maximize: bool = False,
        initial_strategy: Optional[str] = None,
    ):
        super().__init__(
            llm_client=llm_client,
            output_mode=output_mode,
            task_description=task_description,
            maximize=maximize,
            strategy_text=initial_strategy or self.DEFAULT_STRATEGY,
        )
        
        self.strategy_history: List[str] = [self.strategy_text]
        self.performance_history: List[Dict] = []
    
    def update_strategy(self, gradient_text: str) -> None:
        """
        Update strategy using textual gradient.
        
        Args:
            gradient_text: Textual gradient describing how to improve strategy
        """
        prompt = f"""Update the acquisition strategy based on the following gradient:

Current Strategy:
{self.strategy_text}

Improvement Gradient:
{gradient_text}

Generate an improved strategy that:
1. Addresses the issues identified in the gradient
2. Keeps successful elements from the current strategy
3. Is clear and actionable
4. Increments the version number

Output the complete new strategy:
{{
    "updated_strategy": "..."
}}"""
        
        response = self._query_llm(prompt)
        
        try:
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                self.strategy_text = data.get("updated_strategy", self.strategy_text)
            else:
                self.strategy_text = response
        except json.JSONDecodeError:
            self.strategy_text = response
        
        self.strategy_history.append(self.strategy_text)
    
    def compute_gradient(self, performance_data: List[Dict]) -> str:
        """
        Compute textual gradient based on performance feedback.
        
        Args:
            performance_data: List of dicts with 'x', 'predicted', 'actual', 'improved'
            
        Returns:
            Textual gradient describing how to improve the strategy
        """
        self.performance_history.extend(performance_data)
        
        # Format performance data
        good_cases = [p for p in performance_data if p.get('improved', False)]
        bad_cases = [p for p in performance_data if not p.get('improved', False)]
        
        cases_text = []
        for i, p in enumerate(performance_data[:10]):  # Limit to 10 cases
            cases_text.append(
                f"Case {i+1}: x={self._format_x(p['x'])}, "
                f"acq_score={p.get('acquisition_score', 'N/A')}, "
                f"actual_improvement={'Yes' if p.get('improved') else 'No'}"
            )
        
        prompt = f"""Analyze the acquisition function performance and provide improvement suggestions.

Current Strategy:
{self.strategy_text}

Recent Performance:
- Total decisions: {len(performance_data)}
- Successful (found improvement): {len(good_cases)}
- Unsuccessful: {len(bad_cases)}
- Success rate: {len(good_cases) / max(len(performance_data), 1) * 100:.1f}%

Sample Cases:
{chr(10).join(cases_text)}

Analyze:
1. What patterns lead to successful acquisition decisions?
2. What patterns lead to unsuccessful decisions?
3. Is the exploration-exploitation balance appropriate?
4. Are there scoring rules that should be adjusted?

Provide a gradient (improvement direction) for the strategy:
{{
    "gradient": "Detailed description of how to improve the strategy..."
}}"""
        
        response = self._query_llm(prompt)
        
        try:
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return data.get("gradient", response)
        except json.JSONDecodeError:
            pass
        
        return response
    
    def get_strategy(self) -> str:
        """Get current strategy text."""
        return self.strategy_text
    
    def get_strategy_history(self) -> List[str]:
        """Get history of strategy versions."""
        return self.strategy_history.copy()
