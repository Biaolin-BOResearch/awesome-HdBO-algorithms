"""
LLINBO: Trustworthy LLM-in-the-Loop Bayesian Optimization

A hybrid framework that combines Large Language Models with statistical 
surrogates (e.g., Gaussian Processes) for Bayesian Optimization.

The core philosophy is to leverage contextual reasoning strengths of LLMs 
for early exploration, while relying on principled statistical models to 
guide efficient exploitation.

Three mechanisms are provided:
1. LLINBO-Transient: Gradually shifts from LLM to GP as optimization progresses
2. LLINBO-Justify: Uses GP to justify LLM proposals with a decreasing threshold
3. LLINBO-Constrained: Uses posterior samples to validate LLM candidates

Reference:
    "LLINBO: Trustworthy LLM-in-the-Loop Bayesian Optimization"
    https://github.com/your-repo/LLM-in-the-Loop-BO
"""

import torch
from torch import Tensor
from typing import Optional, Any, List, Tuple, Dict
import numpy as np
import json
import random
from collections import defaultdict
from scipy.stats import norm

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound
from botorch.acquisition.analytic import PosteriorMean
from botorch.optim import optimize_acqf

from .base import BaseLLMOptimizer, DEVICE, DTYPE


# ============================================================================
# Helper functions (aligned with helper_func.py)
# ============================================================================

def train_gp(history: List[Tuple[tuple, float]]) -> SingleTaskGP:
    """
    Train a GP model on history data.
    
    Args:
        history: List of (x_tuple, y_value) pairs.
        
    Returns:
        Fitted SingleTaskGP model.
    """
    X = torch.tensor([list(x) for x, _ in history], dtype=torch.float64)
    Y = torch.tensor([[y] for _, y in history], dtype=torch.float64)
    gp = SingleTaskGP(X, Y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    return gp


def optimize_acqf_ucb(model: SingleTaskGP, bounds: Tensor, beta: float) -> Tensor:
    """
    Optimize UCB acquisition function.
    
    Args:
        model: GP model.
        bounds: Search bounds.
        beta: UCB exploration parameter.
        
    Returns:
        Candidate point tensor.
    """
    ucb = UpperConfidenceBound(model, beta=beta)
    candidate, _ = optimize_acqf(
        ucb,
        bounds=bounds,
        q=1,
        num_restarts=20,
        raw_samples=200,
    )
    return candidate


def find_max_variance_bound(model: SingleTaskGP, bounds: Tensor, dim: int) -> float:
    """
    Find maximum posterior variance in the search space.
    
    Args:
        model: GP model.
        bounds: Search bounds.
        dim: Dimensionality.
        
    Returns:
        Maximum variance value.
    """
    axes = [torch.linspace(bounds[0, i], bounds[1, i], 10) for i in range(dim)]
    mesh = torch.meshgrid(*axes, indexing='ij')
    grid = torch.stack([m.reshape(-1) for m in mesh], dim=-1)
    
    model.eval()
    with torch.no_grad():
        posterior = model.posterior(grid)
        variances = posterior.variance
        max_var, _ = torch.max(variances, dim=0)
    return max_var.item()


def find_gp_maximum(model: SingleTaskGP, bounds: Tensor, 
                    num_restarts: int = 10, raw_samples: int = 100) -> float:
    """
    Find maximum of GP posterior mean.
    
    Args:
        model: GP model.
        bounds: Search bounds.
        num_restarts: Number of optimization restarts.
        raw_samples: Number of raw samples for initialization.
        
    Returns:
        Maximum posterior mean value.
    """
    posterior_mean = PosteriorMean(model)
    _, best_obj = optimize_acqf(
        acq_function=posterior_mean,
        bounds=bounds,
        q=1,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )
    return best_obj.item()


def select_next_design_point_bound(model_dict: Dict[int, SingleTaskGP], 
                                    bounds: Tensor, 
                                    beta_t: float, 
                                    dim: int) -> List[float]:
    """
    Select next design point using CGP-UCB (for LLINBO-Constrained).
    
    Args:
        model_dict: Dictionary of GP models.
        bounds: Search bounds.
        beta_t: UCB parameter.
        dim: Dimensionality.
        
    Returns:
        Selected point as list.
    """
    # Construct a grid
    axes = [torch.linspace(bounds[0, i], bounds[1, i], 40) for i in range(dim)]
    mesh = torch.meshgrid(*axes, indexing='ij')
    grid = torch.stack([m.reshape(-1) for m in mesh], dim=-1)
    sampled_points = grid[torch.randperm(grid.shape[0])[:20000]]
    
    n_models = len(model_dict)
    n_grid = sampled_points.shape[0]
    
    mu_matrix = torch.zeros(n_models, n_grid, dtype=torch.float64)
    shared_variance = torch.zeros(n_grid, dtype=torch.float64)
    
    for i, model in model_dict.items():
        model.eval()
        with torch.no_grad():
            posterior = model.posterior(sampled_points)
            mu_matrix[i] = posterior.mean.view(-1).to(torch.float64)
            if i == 0:
                shared_variance = posterior.variance.view(-1).to(torch.float64)
    
    mu_mean = mu_matrix.mean(dim=0)
    mu_sample_var = mu_matrix.var(dim=0, unbiased=True)
    acq = mu_mean + beta_t * torch.sqrt(shared_variance + mu_sample_var)
    best_idx = torch.argmax(acq)
    best_x = sampled_points[best_idx]
    return best_x.tolist()


# ============================================================================
# LLAMBO Agent (aligned with LLAMAGENT in LLM_agent_BBFO.py)
# ============================================================================

class LLAMBOAgent:
    """
    LLAMBO Agent for candidate sampling and surrogate modeling via LLM.
    
    Uses LLM for:
    1. Candidate point sampling (with target score)
    2. Surrogate modeling (predicting function values)
    3. Expected Improvement computation
    """
    
    def __init__(
        self,
        history: List[Tuple[tuple, float]],
        dim: int = 2,
        alpha: float = 0.1,
        num_cand: int = 10,
        max_surrogate_eval: int = 10,
        func_desc: str = "a black-box function",
        llm_client: Any = None,
        model_name: str = "gpt-3.5-turbo",
    ):
        """
        Initialize LLAMBO Agent.
        
        Args:
            history: List of (x_tuple, y_value) pairs.
            dim: Dimensionality of the search space.
            alpha: Parameter for target score calculation.
            num_cand: Number of candidate points to sample.
            max_surrogate_eval: Number of surrogate evaluations per candidate.
            func_desc: Description of the objective function.
            llm_client: OpenAI client.
            model_name: LLM model name.
        """
        self.dim = dim
        self.alpha = alpha
        self.history = [(tuple(x) if not isinstance(x, tuple) else x, y) for x, y in history]
        self.grid_results = {}  # Store grid evaluations
        self.num_cand = num_cand
        self.func_desc = func_desc
        self.max_surrogate_eval = max_surrogate_eval
        self.llm_client = llm_client
        self.model_name = model_name
        
    def query_llm(self, prompt: str) -> str:
        """Query the LLM with a prompt."""
        messages = [
            {"role": "system", "content": "You are an AI assistant that helps people find a maximum of a black box function."},
            {"role": "user", "content": prompt}
        ]
        response = self.llm_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
        )
        return response.choices[0].message.content
    
    def llm_warmstarting(self, num_warmstart: int, objective_function: Any) -> List[Tuple[tuple, float]]:
        """
        Generate initial points using LLM warmstarting.
        
        Args:
            num_warmstart: Number of warmstart points.
            objective_function: The objective function to evaluate.
            
        Returns:
            List of (x_tuple, y_value) pairs.
        """
        if objective_function is None:
            raise ValueError("Objective function must be provided for warm-starting.")
        
        prompt = f"""
        You are assisting me with maximize a black-box function, which is {self.func_desc}.
        Suggest {num_warmstart} promising starting points for this task in the range [0, 1]^{self.dim}.
        Return the points strictly in JSON format as a list of {self.dim}-dimensional vectors. Do not include any explanations, labels, formatting, or extra text. The response must be strictly valid JSON.
        """
        
        while True:
            llm_output = self.query_llm(prompt)
            try:
                warmstart_points = json.loads(llm_output)
                if isinstance(warmstart_points, list) and all(
                    isinstance(x, list) and len(x) == self.dim for x in warmstart_points
                ):
                    history = [(tuple(x), objective_function(x)) for x in warmstart_points]
                    return history
            except json.JSONDecodeError:
                print("LLM warmstarting response could not be parsed! Retrying...")
                continue
    
    def sample_candidate_points(self) -> List[tuple]:
        """
        Sample candidate points from LLM.
        
        Uses target score based on best and worst observed values.
        
        Returns:
            List of candidate point tuples.
        """
        best_y = max(self.history, key=lambda x: x[1])[1]
        worst_y = min(self.history, key=lambda x: x[1])[1]
        target_score = best_y - self.alpha * (best_y - worst_y)
        
        # Prepare permuted histories
        permuted_histories = []
        for _ in range(self.max_surrogate_eval):
            shuffled = self.history.copy()
            random.shuffle(shuffled)
            permuted_histories.append(shuffled)
        
        candidates = []
        for history_variant in permuted_histories[:self.num_cand]:
            history_str = "\n".join([f"x: {h[0]}, f(x): {h[1]}" for h in history_variant])
            candidate = self._sample_one_candidate(history_str, target_score)
            if candidate is not None:
                candidates.append(candidate)
                
        return candidates
    
    def _sample_one_candidate(self, history_str: str, target_score: float) -> Optional[tuple]:
        """Sample a single candidate point from LLM."""
        prompt = f"""
        The following are past evaluations of a black-box function. The function is {self.func_desc}.
        {history_str}
        The allowable ranges for x is [0, 1]^{self.dim}.
        Recommend a new x that can achieve the function value of {target_score}.
        Return only a single {self.dim}-dimensional numerical vector with the highest possible precision. 
        Do not include any explanations, labels, formatting, or extra text. The response must be strictly valid JSON.
        """
        
        try:
            response = self.query_llm(prompt)
            extracted_value = json.loads(response.strip())
            if isinstance(extracted_value, list) and len(extracted_value) == self.dim:
                return tuple(float(v) for v in extracted_value)
        except (ValueError, json.JSONDecodeError):
            pass
        return None
    
    def surrogate_model(self, candidates: List[tuple]) -> None:
        """
        Build LLM-based surrogate model for candidates.
        
        Uses multiple LLM queries with permuted history to estimate
        mean and standard deviation for each candidate.
        
        Args:
            candidates: List of candidate point tuples.
        """
        permuted_histories = []
        for _ in range(self.max_surrogate_eval):
            shuffled = self.history.copy()
            random.shuffle(shuffled)
            permuted_histories.append(shuffled)
        
        # Predict scores for each candidate
        for x in candidates:
            scores = []
            for history_variant in permuted_histories[:self.max_surrogate_eval]:
                history_str = "\n".join([f"x: {h[0]}, f(x): {h[1]}" for h in history_variant])
                score = self._predict_llm_score(x, history_str)
                if score is not None:
                    scores.append(score)
            
            if scores:
                mean, std = np.mean(scores), np.std(scores)
                self.grid_results[tuple(x)] = (mean, std)
    
    def _predict_llm_score(self, x: tuple, history_str: str) -> Optional[float]:
        """Predict function value at x using LLM."""
        prompt = f"""
        The following are past evaluations of a black-box function, which is {self.func_desc}.    
        {history_str}     
        The allowable ranges for x is [0, 1]^{self.dim}.
        Predict the function value at x = {x}.
        Return only a single numerical value. Do not include any explanations, labels, formatting, or extra text. The response must be strictly a valid floating-point number.
        """
        
        try:
            response = self.query_llm(prompt)
            return float(response.strip())
        except ValueError:
            return None
    
    def expected_improvement(self, mean: float, std: float, best_so_far: float, xi: float = 0.01) -> float:
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
        if mean is None or std is None:
            return -np.inf
        improvement = mean - best_so_far - xi
        z = improvement / (std + 1e-9)
        ei = improvement * norm.cdf(z) + std * norm.pdf(z)
        return ei
    
    def find_best_candidate(self) -> Optional[tuple]:
        """
        Find best candidate using EI on LLM surrogate.
        
        Returns:
            Best candidate tuple, or None if no history.
        """
        if not self.history:
            return None
        
        best_so_far = max(self.history, key=lambda x: x[1])[1]
        candidates = self.sample_candidate_points()
        
        if not candidates:
            return None
        
        self.surrogate_model(candidates)
        best_candidate = None
        best_ei = -np.inf
        
        for candidate in candidates:
            mean, std = self.grid_results.get(tuple(candidate), (None, None))
            ei = self.expected_improvement(mean, std, best_so_far)
            
            if ei > best_ei:
                best_ei = ei
                best_candidate = candidate
        
        return best_candidate


# ============================================================================
# LLAMBO-Light Agent (aligned with LLAMAGENT_L in LLM_agent_BBFO.py)
# ============================================================================

class LLAMBOLightAgent:
    """
    LLAMBO-Light Agent for direct candidate generation via LLM.
    
    A simplified version that directly generates candidates without
    explicit surrogate modeling, relying on the LLM's internal
    understanding of exploration-exploitation trade-off.
    """
    
    def __init__(
        self,
        history: List[Tuple[tuple, float]],
        dim: int,
        func_desc: str = "a black-box function",
        llm_client: Any = None,
        model_name: str = "gpt-3.5-turbo",
    ):
        """
        Initialize LLAMBO-Light Agent.
        
        Args:
            history: List of (x_tuple, y_value) pairs.
            dim: Dimensionality of the search space.
            func_desc: Description of the objective function.
            llm_client: OpenAI client.
            model_name: LLM model name.
        """
        self.dim = dim
        self.func_desc = func_desc
        self.history = [(tuple(x) if not isinstance(x, tuple) else x, y) for x, y in history]
        self.llm_client = llm_client
        self.model_name = model_name
        
    def query_llm(self, prompt: str) -> str:
        """Query the LLM with a prompt."""
        messages = [
            {"role": "system", "content": "You are an AI assistant that helps people find a maximum of a black box function."},
            {"role": "user", "content": prompt}
        ]
        response = self.llm_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
        )
        return response.choices[0].message.content
    
    def llm_warmstarting(self, num_warmstart: int, objective_function: Any) -> List[Tuple[tuple, float]]:
        """
        Generate initial points using LLM warmstarting.
        """
        prompt = f"""
        You are assisting me with maximize a black-box function, which is {self.func_desc}.
        Suggest {num_warmstart} promising starting points for this task in the range [0, 1]^{self.dim}.
        Return the points strictly in JSON format as a list of {self.dim}-dimensional vectors. Do not include any explanations, labels, formatting, or extra text. The response must be strictly valid JSON.
        """
        
        while True:
            llm_output = self.query_llm(prompt)
            try:
                warmstart_points = json.loads(llm_output)
                if isinstance(warmstart_points, list) and all(
                    isinstance(x, list) and len(x) == self.dim for x in warmstart_points
                ):
                    history = [(tuple(x), objective_function(x)) for x in warmstart_points]
                    return history
            except json.JSONDecodeError:
                print("LLM warmstarting response could not be parsed! Retrying...")
                continue
    
    def sample_candidate_points(self) -> List[float]:
        """
        Sample a candidate point that balances exploration and exploitation.
        
        Returns:
            Candidate point as list.
        """
        shuffled_history = self.history.copy()
        random.shuffle(shuffled_history)
        
        history_str = "\n".join([f"x: {x}, f(x): {y}" for x, y in shuffled_history])
        prompt = f"""
        The following are past evaluations of a black-box function, which is {self.func_desc}.
        {history_str}
        The allowable ranges for x is [0, 1]^{self.dim}.
        Based on the past data, recommend the next point to evaluate that balances exploration and exploitation:
        - Exploration means selecting a point in an unexplored or less-sampled region that is far from the previously evaluated points.
        - Exploitation means selecting a point close to the previously high-performing evaluations.
        The goal is to eventually find the global maximum. Return only a single {self.dim}-dimensional numerical vector with high precision. The response must be valid JSON with no explanations, labels, or extra formatting.
        Return only a single {self.dim}-dimensional numerical vector with the highest possible precision. Do not include any explanations, labels, formatting, or extra text.
        """
        
        while True:
            llm_output = self.query_llm(prompt)
            try:
                cand_points = json.loads(llm_output)
                return cand_points
            except json.JSONDecodeError:
                print("LLM candidate response could not be parsed! Retrying...")
                continue


# ============================================================================
# LLINBO Main Class (aligned with LLMIBO_BFO in LLM_agent_BBFO.py)
# ============================================================================

class LLINBO(BaseLLMOptimizer):
    """
    LLINBO: LLM-in-the-Loop Bayesian Optimization.
    
    A hybrid framework combining LLMs with statistical surrogates (GP)
    for Bayesian optimization.
    
    Supported methods:
    - 'llambo': Full LLAMBO with LLM surrogate and EI
    - 'llmbo': LLAMBO-Light with direct LLM candidate generation
    - 'bo': Standard GP-based BO with UCB
    - 'transient': Gradually shifts from LLM to GP
    - 'justify': Uses GP to justify LLM proposals
    - 'constrained': Uses posterior samples to validate LLM candidates
    - 'rs': Random search baseline
    
    Reference:
        "LLINBO: Trustworthy LLM-in-the-Loop Bayesian Optimization"
    """
    
    def __init__(
        self,
        input_dim: int,
        bounds: Tensor,
        llm_client: Any = None,
        model_name: str = "gpt-3.5-turbo",
        task_description: str = "a black-box function",
        maximize: bool = True,
        method: str = "transient",
        n_iterations: int = 20,
        n_init: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize LLINBO optimizer.
        
        Args:
            input_dim: Dimensionality of the search space.
            bounds: Tensor of shape (2, input_dim) with [lower, upper] bounds.
            llm_client: OpenAI client for LLM API calls.
            model_name: Name of the LLM model (e.g., "gpt-3.5-turbo").
            task_description: Description of the objective function.
            maximize: Whether to maximize (True) or minimize (False).
            method: LLINBO method ('llambo', 'llmbo', 'bo', 'transient', 'justify', 'constrained', 'rs').
            n_iterations: Total number of optimization iterations.
            n_init: Number of initial points (default: input_dim).
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
        
        self.method = method.lower()
        self.n_iterations = n_iterations
        self.n_init = n_init if n_init is not None else input_dim
        self.current_iteration = 0
        
        # History as list of (tuple, float) for compatibility
        self.history: List[Tuple[tuple, float]] = []
        
        # For justify method
        self._max_var: Optional[float] = None
        
        # Method dispatcher
        self._methods = {
            'rs': self._suggest_rs,
            'llambo': self._suggest_llambo,
            'llmbo': self._suggest_llmbo,
            'bo': self._suggest_bo,
            'transient': self._suggest_transient,
            'justify': self._suggest_justify,
            'constrained': self._suggest_constrained,
        }
        
        if self.method not in self._methods:
            raise ValueError(f"Method '{self.method}' is not implemented. "
                           f"Choose from: {list(self._methods.keys())}")
    
    def _compute_beta(self, t: int) -> float:
        """Compute UCB beta parameter based on iteration."""
        return np.log((t + 1) * self.input_dim * np.pi**2 / 0.1 * 6) * 2
    
    def _history_to_tensor(self) -> Tuple[Tensor, Tensor]:
        """Convert history to tensors."""
        if not self.history:
            return None, None
        X = torch.tensor([list(x) for x, _ in self.history], dtype=self.dtype, device=self.device)
        y = torch.tensor([[y] for _, y in self.history], dtype=self.dtype, device=self.device)
        return X, y
    
    def warmstart(self, objective_function: Any) -> List[Tuple[tuple, float]]:
        """
        Warmstart optimization using LLM.
        
        Args:
            objective_function: The objective function to optimize.
            
        Returns:
            Initial history as list of (tuple, float) pairs.
        """
        agent = LLAMBOLightAgent(
            history=[],
            dim=self.input_dim,
            func_desc=self.task_description,
            llm_client=self.llm_client,
            model_name=self.model_name,
        )
        self.history = agent.llm_warmstarting(
            num_warmstart=self.n_init,
            objective_function=objective_function
        )
        return self.history
    
    def initialize_random(self, objective_function: Any) -> List[Tuple[tuple, float]]:
        """
        Initialize with random samples.
        
        Args:
            objective_function: The objective function.
            
        Returns:
            Initial history.
        """
        X_init = torch.rand(self.n_init, self.input_dim, dtype=self.dtype, device=self.device)
        X_init = self.bounds[0] + (self.bounds[1] - self.bounds[0]) * X_init
        
        self.history = []
        for i in range(self.n_init):
            x = X_init[i].tolist()
            y = objective_function(x)
            self.history.append((tuple(x), y))
        
        return self.history
    
    def suggest(self, n_candidates: int = 1) -> Tensor:
        """
        Suggest next point(s) to evaluate.
        
        Args:
            n_candidates: Number of candidates to suggest.
            
        Returns:
            Tensor of suggested points.
        """
        candidates = []
        for _ in range(n_candidates):
            candidate = self._methods[self.method]()
            candidates.append(candidate)
            self.current_iteration += 1
        
        return torch.stack(candidates)
    
    def update_history(self, x: List[float], y: float) -> None:
        """
        Update history with new observation.
        
        Args:
            x: New point as list.
            y: Function value.
        """
        self.history.append((tuple(x), y))
        
        # Also update base class data
        x_tensor = torch.tensor(x, dtype=self.dtype, device=self.device).unsqueeze(0)
        y_tensor = torch.tensor([[y]], dtype=self.dtype, device=self.device)
        self.update(x_tensor, y_tensor)
    
    # ========================================================================
    # Method implementations
    # ========================================================================
    
    def _suggest_rs(self) -> Tensor:
        """Random search."""
        x = torch.rand(self.input_dim, dtype=self.dtype, device=self.device)
        x = self.bounds[0] + (self.bounds[1] - self.bounds[0]) * x
        return x
    
    def _suggest_llambo(self) -> Tensor:
        """Full LLAMBO with LLM surrogate."""
        agent = LLAMBOAgent(
            history=self.history,
            dim=self.input_dim,
            func_desc=self.task_description,
            llm_client=self.llm_client,
            model_name=self.model_name,
        )
        next_x = agent.find_best_candidate()
        
        if next_x is None:
            return self._suggest_rs()
        
        return torch.tensor(next_x, dtype=self.dtype, device=self.device)
    
    def _suggest_llmbo(self) -> Tensor:
        """LLAMBO-Light with direct LLM candidate generation."""
        agent = LLAMBOLightAgent(
            history=self.history,
            dim=self.input_dim,
            func_desc=self.task_description,
            llm_client=self.llm_client,
            model_name=self.model_name,
        )
        next_x = agent.sample_candidate_points()
        return torch.tensor(next_x, dtype=self.dtype, device=self.device)
    
    def _suggest_bo(self) -> Tensor:
        """Standard GP-based BO with UCB."""
        model = train_gp(self.history)
        beta_t = self._compute_beta(self.current_iteration)
        
        bounds = torch.stack([
            torch.zeros(self.input_dim, dtype=torch.float64),
            torch.ones(self.input_dim, dtype=torch.float64)
        ])
        
        next_x = optimize_acqf_ucb(model, bounds=bounds, beta=beta_t)
        return next_x.squeeze(0).to(dtype=self.dtype, device=self.device)
    
    def _suggest_transient(self) -> Tensor:
        """
        Transient method: Gradually shifts from LLM to GP.
        
        Probability of using GP: p_t = min((t^2 / T), 1)
        """
        t = self.current_iteration
        T = self.n_iterations
        p_t = min((t**2 / T), 1)
        
        if np.random.rand() < p_t:
            # Use GP-UCB
            return self._suggest_bo()
        else:
            # Use LLM
            return self._suggest_llmbo()
    
    def _suggest_justify(self) -> Tensor:
        """
        Justify method: Uses GP to justify LLM proposals.
        
        Accept LLM candidate only if:
        UCB(x_gp) > UCB(x_llm) + psi_t
        
        where psi_t = max_var / (t + 1) decreases over time.
        """
        model = train_gp(self.history)
        t = self.current_iteration
        beta_t = self._compute_beta(t)
        
        bounds = torch.stack([
            torch.zeros(self.input_dim, dtype=torch.float64),
            torch.ones(self.input_dim, dtype=torch.float64)
        ])
        
        # Compute max_var once at the beginning
        if self._max_var is None:
            self._max_var = find_max_variance_bound(model, bounds, self.input_dim)
        
        psi_t = self._max_var / (t + 1)
        
        # Get GP candidate
        next_x_gp = optimize_acqf_ucb(model, bounds=bounds, beta=beta_t)
        
        # Get LLM candidate
        agent = LLAMBOLightAgent(
            history=self.history,
            dim=self.input_dim,
            func_desc=self.task_description,
            llm_client=self.llm_client,
            model_name=self.model_name,
        )
        next_x_llm = agent.sample_candidate_points()
        next_x_llm_tensor = torch.tensor([next_x_llm], dtype=torch.float64)
        
        # Compare UCB values
        ucb = UpperConfidenceBound(model, beta=beta_t)
        ucb_gp = ucb(next_x_gp).item()
        ucb_llm = ucb(next_x_llm_tensor).item()
        
        if ucb_gp > ucb_llm + psi_t:
            return next_x_gp.squeeze(0).to(dtype=self.dtype, device=self.device)
        else:
            return torch.tensor(next_x_llm, dtype=self.dtype, device=self.device)
    
    def _suggest_constrained(self) -> Tensor:
        """
        Constrained method: Uses posterior samples to validate LLM candidates.
        
        Accept LLM candidate only if some posterior samples beat the 
        current GP maximum. Number of samples decreases with iteration.
        """
        model = train_gp(self.history)
        t = self.current_iteration
        beta_t = self._compute_beta(t)
        
        bounds = torch.stack([
            torch.zeros(self.input_dim, dtype=torch.float64),
            torch.ones(self.input_dim, dtype=torch.float64)
        ])
        
        # Get LLM candidate
        agent = LLAMBOLightAgent(
            history=self.history,
            dim=self.input_dim,
            func_desc=self.task_description,
            llm_client=self.llm_client,
            model_name=self.model_name,
        )
        next_x_llm = agent.sample_candidate_points()
        
        # Find GP maximum
        snew = 10000
        post_max = find_gp_maximum(model, bounds, num_restarts=10, raw_samples=100)
        sraw = int(np.floor(snew / (t + 1)**2))
        
        better_samples = []
        if sraw > 1:
            with torch.no_grad():
                posterior = model.posterior(
                    torch.tensor(next_x_llm, dtype=torch.float64).unsqueeze(0)
                )
                samples = posterior.rsample(sample_shape=torch.Size([sraw]))
            for s in samples.view(-1):
                if s.item() > post_max:
                    better_samples.append(s.item())
        
        if len(better_samples) == 0:
            # Fall back to GP-UCB
            next_x = optimize_acqf_ucb(model, bounds=bounds, beta=beta_t)
            return next_x.squeeze(0).to(dtype=self.dtype, device=self.device)
        else:
            # Build model ensemble and use CGP-UCB
            model_dict = {}
            for i, sample_val in enumerate(better_samples):
                extended_history = self.history + [(tuple(next_x_llm), sample_val)]
                X = torch.tensor([list(x) for x, _ in extended_history], dtype=torch.double)
                Y = torch.tensor([[y] for _, y in extended_history], dtype=torch.double)
                new_model = SingleTaskGP(X, Y)
                mll = ExactMarginalLogLikelihood(new_model.likelihood, new_model)
                fit_gpytorch_mll(mll)
                model_dict[i] = new_model
            
            next_x = select_next_design_point_bound(
                model_dict=model_dict, 
                bounds=bounds, 
                beta_t=beta_t, 
                dim=self.input_dim
            )
            return torch.tensor(next_x, dtype=self.dtype, device=self.device)
