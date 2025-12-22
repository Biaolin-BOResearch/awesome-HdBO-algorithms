"""
Verbalized Bayesian Optimization (VBO) Framework

A fully LLM-driven Bayesian Optimization framework with:
- Configurable Surrogate Model strategies (In-Context / Belief-based)
- Configurable output modes (Text-only / Text+Numeric)
- Configurable Acquisition Function evaluation
- Multiple Acquisition optimization methods (Random/Grid/TGA)

All components can be freely combined for experiments.
"""

from .surrogate import (
    VBOSurrogateModel,
    InContextSurrogate,
    BeliefBasedSurrogate,
    SurrogateMode,
    OutputMode,
    SurrogatePrediction,
    create_surrogate,
)

from .acquisition import (
    VBOAcquisitionFunction,
    AcquisitionOutputMode,
    AcquisitionEvaluation,
    TextGradOptimizableAcquisition,
)

from .optimization import (
    AcquisitionOptimizer,
    RandomSearchOptimizer,
    GridSearchOptimizer,
    SingleStartTGA,
    MultiStartTGA,
    OptimizationMethod,
    OptimizationStats,
    create_optimizer,
)

from .vbo_optimizer import (
    VBOptimizer,
    VBOConfig,
    create_vbo_optimizer,
    get_all_vbo_configurations,
    get_vbo_config_name,
)

__all__ = [
    # Surrogate
    "VBOSurrogateModel",
    "InContextSurrogate",
    "BeliefBasedSurrogate",
    "SurrogateMode",
    "OutputMode",
    "SurrogatePrediction",
    "create_surrogate",
    # Acquisition
    "VBOAcquisitionFunction",
    "AcquisitionOutputMode",
    "AcquisitionEvaluation",
    "TextGradOptimizableAcquisition",
    # Optimization
    "AcquisitionOptimizer",
    "RandomSearchOptimizer",
    "GridSearchOptimizer",
    "SingleStartTGA",
    "MultiStartTGA",
    "OptimizationMethod",
    "OptimizationStats",
    "create_optimizer",
    # Main optimizer
    "VBOptimizer",
    "VBOConfig",
    "create_vbo_optimizer",
    "get_all_vbo_configurations",
    "get_vbo_config_name",
]
