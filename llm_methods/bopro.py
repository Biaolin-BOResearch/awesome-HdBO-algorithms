"""
BOPRO: Bayesian Optimization with LLM Proposals

This module implements BOPRO, which uses LLMs to propose candidate solutions
that are then evaluated and ranked using traditional BO surrogate models
(e.g., Gaussian Processes) and acquisition functions.

The key insight is that LLMs can leverage domain knowledge and contextual
understanding to propose high-quality initial candidates, while GP-based
surrogates provide principled uncertainty quantification.

Reference:
    "BOPRO: Bayesian Optimization with LLM Proposals"
    ICLR 2025
    https://github.com/amazon-science/bopro
"""

import torch
from torch import Tensor
from typing import Optional, Any, List, Tuple, Dict
import numpy as np
import json
import re

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.kernels import ScaleKernel, MaternKernel, RBFKernel
from gpytorch.priors import GammaPrior

from .base import BaseLLMOptimizer, DEVICE, DTYPE


class BOPRO(BaseLLMOptimizer):
    """
    BOPRO: Bayesian Optimization with LLM Proposals.
    
    Combines LLM candidate generation with GP-based acquisition function
    evaluation. The workflow is:
    
    1. Use LLM to propose N candidate solutions based on context and history
    2. Fit a GP surrogate model on observed data
    3. Evaluate acquisition function (EI, UCB, etc.) on LLM proposals
    4. Select the proposal with highest acquisition value
    
    This hybrid approach leverages:
    - LLM's ability to understand context and propose reasonable candidates
    - GP's principled uncertainty quantification for exploration-exploitation
    
    Attributes:
        n_proposals: Number of LLM proposals per iteration.
        acquisition: Acquisition function type ('EI', 'UCB', 'LogEI').
        kernel_type: GP kernel type ('matern', 'rbf').
        ucb_beta: UCB exploration parameter.
    """
    
    def __init__(
        self,
        input_dim: int,
        bounds: Tensor,
        llm_client: Any = None,
        model_name: str = "gpt-4",
        task_description: str = "",
        maximize: bool = True,
        n_proposals: int = 10,
        acquisition: str = "EI",
        kernel_type: str = "matern",
        ucb_beta: float = 2.0,
        use_warmstart: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize BOPRO optimizer.
        
        Args:
            input_dim: Dimensionality of the input space.
            bounds: Tensor of shape (2, input_dim) with [lower, upper] bounds.
            llm_client: Client for LLM API calls.
            model_name: Name of the LLM model.
            task_description: Description of the optimization problem.
            maximize: Whether to maximize or minimize.
            n_proposals: Number of LLM proposals per iteration.
            acquisition: Acquisition function ('EI', 'UCB', 'LogEI').
            kernel_type: GP kernel type ('matern', 'rbf').
            ucb_beta: UCB exploration parameter.
            use_warmstart: Whether to use LLM for initial point generation.
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
        
        self.n_proposals = n_proposals
        self.acquisition = acquisition.upper()
        self.kernel_type = kernel_type.lower()
        self.ucb_beta = ucb_beta
        self.use_warmstart = use_warmstart
        
        # GP model
        self.gp_model: Optional[SingleTaskGP] = None
        
    def _build_gp(self) -> SingleTaskGP:
        """
        Build and fit GP surrogate model.
        
        Returns:
            Fitted SingleTaskGP model.
        """
        X = self.X.to(torch.float64)
        y = self.y.to(torch.float64)
        
        # Create kernel
        if self.kernel_type == "matern":
            base_kernel = MaternKernel(
                nu=2.5,
                ard_num_dims=self.input_dim,
                lengthscale_prior=GammaPrior(3.0, 6.0),
            )
        else:
            base_kernel = RBFKernel(
                ard_num_dims=self.input_dim,
                lengthscale_prior=GammaPrior(3.0, 6.0),
            )
        
        covar_module = ScaleKernel(
            base_kernel=base_kernel,
            outputscale_prior=GammaPrior(2.0, 0.15),
        )
        
        model = SingleTaskGP(X, y, covar_module=covar_module)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        
        return model
    
    def _get_acquisition_function(self, model: SingleTaskGP):
        """
        Create acquisition function.
        
        Args:
            model: GP model.
            
        Returns:
            Acquisition function object.
        """
        if self.maximize:
            best_f = self.y.max().item()
        else:
            best_f = self.y.min().item()
        
        if self.acquisition == "EI":
            return ExpectedImprovement(model=model, best_f=best_f, maximize=self.maximize)
        elif self.acquisition == "LOGEI":
            return LogExpectedImprovement(model=model, best_f=best_f)
        elif self.acquisition == "UCB":
            return UpperConfidenceBound(model=model, beta=self.ucb_beta)
        else:
            return ExpectedImprovement(model=model, best_f=best_f, maximize=self.maximize)
    
    def _generate_llm_proposals(self) -> List[List[float]]:
        """
        Generate candidate proposals using LLM.
        
        Returns:
            List of proposed solutions.
        """
        # Format history for prompt
        if self.X is not None and len(self.X) > 0:
            history = list(zip(self.X.tolist(), self.y.squeeze().tolist()))
            
            # Sort by value
            if self.maximize:
                history.sort(key=lambda x: x[1], reverse=True)
            else:
                history.sort(key=lambda x: x[1])
            
            # Take recent/best points
            history = history[:min(20, len(history))]
            history_str = "\n".join([
                f"  x = {x}, f(x) = {y:.6f}" for x, y in history
            ])
        else:
            history_str = "No evaluations yet."
        
        bounds_lower = self.bounds[0].tolist()
        bounds_upper = self.bounds[1].tolist()
        direction = "maximize" if self.maximize else "minimize"
        
        prompt = f"""You are helping optimize a {self.input_dim}-dimensional black-box function.

Task Description: {self.task_description}
Goal: {direction} the function value.
Search space bounds: [{bounds_lower}, {bounds_upper}]

Past evaluations (sorted by quality):
{history_str}

Based on this data, propose {self.n_proposals} diverse candidate solutions that:
1. Explore promising regions near the best solutions
2. Also explore some unexplored regions for diversity
3. Stay within the search bounds

Return ONLY a JSON array of {self.n_proposals} solutions. Each solution is a {self.input_dim}-dimensional array.
Format: [[x1, x2, ...], [x1, x2, ...], ...]
Do not include any explanation.
"""
        
        system_msg = "You are an expert optimization assistant."
        
        try:
            response = self.query_llm(prompt, system_message=system_msg, max_tokens=1000)
            
            # Parse response
            match = re.search(r'\[\s*\[[\d\s,.\-e\[\]]+\]\s*\]', response, re.DOTALL)
            if match:
                proposals = json.loads(match.group())
                
                # Validate and clip
                valid_proposals = []
                for p in proposals:
                    if isinstance(p, list) and len(p) == self.input_dim:
                        clipped = [
                            max(bounds_lower[i], min(bounds_upper[i], float(v)))
                            for i, v in enumerate(p)
                        ]
                        valid_proposals.append(clipped)
                
                if valid_proposals:
                    return valid_proposals
                    
        except (json.JSONDecodeError, Exception) as e:
            print(f"LLM proposal parsing failed: {e}")
        
        # Fallback: random proposals
        random_proposals = self.random_points(self.n_proposals)
        return random_proposals.tolist()
    
    def warmstart(self, n_points: int, objective_func: Any = None) -> Tensor:
        """
        Generate initial points using LLM.
        
        Args:
            n_points: Number of initial points.
            objective_func: Optional objective function.
            
        Returns:
            Tensor of initial points.
        """
        bounds_lower = self.bounds[0].tolist()
        bounds_upper = self.bounds[1].tolist()
        direction = "maximize" if self.maximize else "minimize"
        
        prompt = f"""Suggest {n_points} diverse starting points for optimizing a {self.input_dim}-dimensional function.

Task: {self.task_description}
Goal: {direction} the function.
Bounds: [{bounds_lower}, {bounds_upper}]

Return ONLY a JSON array of {n_points} points, each with {self.input_dim} dimensions.
Format: [[x1, x2, ...], [x1, x2, ...], ...]
"""
        
        try:
            response = self.query_llm(prompt, max_tokens=500)
            match = re.search(r'\[\s*\[[\d\s,.\-e\[\]]+\]\s*\]', response, re.DOTALL)
            if match:
                points = json.loads(match.group())
                if len(points) >= n_points:
                    points = points[:n_points]
                    points_tensor = torch.tensor(points, device=self.device, dtype=self.dtype)
                    return self.clip_to_bounds(points_tensor)
        except:
            pass
        
        return self.random_points(n_points)
    
    def suggest(self, n_candidates: int = 1) -> Tensor:
        """
        Suggest new points using BOPRO.
        
        1. Generate LLM proposals
        2. Fit GP and compute acquisition values
        3. Select best proposals by acquisition value
        
        Args:
            n_candidates: Number of candidates to return.
            
        Returns:
            Tensor of suggested points.
        """
        if self.X is None or len(self.X) < 2:
            # Not enough data for GP
            return self.random_points(n_candidates)
        
        # Build GP model
        self.gp_model = self._build_gp()
        
        # Get acquisition function
        acq_func = self._get_acquisition_function(self.gp_model)
        
        # Generate LLM proposals
        proposals = self._generate_llm_proposals()
        
        # Also add some GP-optimized candidates for comparison
        try:
            gp_candidate, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=self.bounds.to(torch.float64),
                q=1,
                num_restarts=10,
                raw_samples=100,
            )
            proposals.append(gp_candidate.squeeze(0).tolist())
        except:
            pass
        
        # Evaluate acquisition function on proposals
        proposals_tensor = torch.tensor(proposals, dtype=torch.float64, device=self.device)
        
        with torch.no_grad():
            acq_values = acq_func(proposals_tensor.unsqueeze(1))
        
        # Sort by acquisition value
        sorted_indices = torch.argsort(acq_values, descending=True)
        
        # Select top candidates
        selected = []
        for i in range(min(n_candidates, len(proposals))):
            idx = sorted_indices[i]
            selected.append(proposals_tensor[idx])
        
        return torch.stack(selected).to(dtype=self.dtype, device=self.device)


class BOPROWithRanking(BOPRO):
    """
    BOPRO with Thompson Sampling ranking.
    
    Instead of using deterministic acquisition function, uses
    Thompson Sampling to rank LLM proposals.
    """
    
    def __init__(self, *args, n_thompson_samples: int = 100, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_thompson_samples = n_thompson_samples
    
    def suggest(self, n_candidates: int = 1) -> Tensor:
        """
        Suggest points using Thompson Sampling ranking.
        """
        if self.X is None or len(self.X) < 2:
            return self.random_points(n_candidates)
        
        # Build GP
        self.gp_model = self._build_gp()
        
        # Generate proposals
        proposals = self._generate_llm_proposals()
        proposals_tensor = torch.tensor(proposals, dtype=torch.float64, device=self.device)
        
        # Thompson Sampling: sample from posterior and rank
        with torch.no_grad():
            posterior = self.gp_model.posterior(proposals_tensor)
            samples = posterior.rsample(sample_shape=torch.Size([self.n_thompson_samples]))
            
            # Average rank across samples
            if self.maximize:
                ranks = torch.argsort(torch.argsort(samples, dim=1, descending=True), dim=1)
            else:
                ranks = torch.argsort(torch.argsort(samples, dim=1, descending=False), dim=1)
            
            mean_ranks = ranks.float().mean(dim=0)
        
        # Select candidates with best (lowest) average rank
        sorted_indices = torch.argsort(mean_ranks)
        
        selected = []
        for i in range(min(n_candidates, len(proposals))):
            idx = sorted_indices[i]
            selected.append(proposals_tensor[idx])
        
        return torch.stack(selected).to(dtype=self.dtype, device=self.device)

