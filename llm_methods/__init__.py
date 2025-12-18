"""
LLM-based Bayesian Optimization Methods Collection

A collection of LLM-augmented Bayesian optimization algorithms
that leverage Large Language Models for various aspects of optimization.

Methods implemented:
- LLAMBO: LLM as surrogate model and acquisition function
- LLAMBOLight: Simplified LLAMBO with direct candidate generation
- LLINBO: LLM-in-the-Loop BO with multiple strategies (transient, justify, constrained)
- SLLMBO: Structured LLM-based optimization with intelligent summarization
- OPRO: Optimization by PROmpting (LLM as optimizer)
- BOPRO: BO with LLM proposals
"""

from .base import BaseLLMOptimizer

# LLAMBO - LLM as surrogate and acquisition function
from .llambo import LLAMBO, LLAMBOLight

# LLINBO - LLM-in-the-Loop BO
from .llinbo import (
    LLINBO,
    LLAMBOAgent,
    LLAMBOLightAgent,
    train_gp,
    optimize_acqf_ucb,
    find_max_variance_bound,
    find_gp_maximum,
    select_next_design_point_bound,
)

# SLLMBO - Structured LLM-based optimization
from .sllmbo import SLLMBO

# OPRO - Optimization by PROmpting
from .opro import OPRO, OPROWithExemplars

# BOPRO - BO with LLM proposals
from .bopro import BOPRO, BOPROWithRanking


__version__ = "1.0.0"

__all__ = [
    # Base class
    "BaseLLMOptimizer",
    
    # LLAMBO
    "LLAMBO",
    "LLAMBOLight",
    
    # LLINBO (LLM-in-the-Loop BO)
    "LLINBO",
    "LLAMBOAgent",
    "LLAMBOLightAgent",
    
    # SLLMBO
    "SLLMBO",
    
    # OPRO
    "OPRO",
    "OPROWithExemplars",
    
    # BOPRO
    "BOPRO",
    "BOPROWithRanking",
    
    # Utility functions
    "train_gp",
    "optimize_acqf_ucb",
    "find_max_variance_bound",
    "find_gp_maximum",
    "select_next_design_point_bound",
]
