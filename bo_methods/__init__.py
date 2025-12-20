"""
Bayesian Optimization Methods Collection

A collection of state-of-the-art Bayesian optimization algorithms
for hyperparameter tuning and black-box optimization.
"""

from .standard_bo import StandardBO
from .turbo import TuRBO
from .turbo_m import TurboM
from .gpoe_bo import GPOEBO
from .gpoe_turbo import GPOETRBO
from .hesbo import HesBO
from .alebo import ALEBO
from .rembo import REMBO
from .saasbo import SAASBO

# New methods from HDBO benchmark Figure 1
from .vanilla_bo import VanillaBO
from .random_line_bo import RandomLineBO
from .baxus import BAxUS
from .solid import SOLID
from .dsa import DSA
from .hds import HDS
from .bounce import Bounce
from .lsbo import LSBO
from .mgp import MGP

# Robust GP methods for high dimensions
from .robust_gp_bo import RobustGPBO

# Coordinate-wise optimization methods
from .eci_bo import ECIBO, ECIBOMaximize

# Adaptive line search methods
from .als_bo import ALSBO, ALSBOMaximize

# Subspace selection methods
from .essi_bo import ESSIBO, ESSIBOMaximize

# Variable selection methods
from .vsbo import VSBO
from .mcts_vs import MCTSVS
from .lasso_bo import LassoBO

__version__ = "1.8.0"
__all__ = [
    "StandardBO",
    "TuRBO",
    "TurboM",
    "GPOEBO",
    "GPOETRBO",
    "HesBO",
    "ALEBO",
    "REMBO",
    "SAASBO",
    # HDBO benchmark methods
    "VanillaBO",
    "RandomLineBO",
    "BAxUS",
    "SOLID",
    "DSA",
    "HDS",
    "Bounce",
    "LSBO",
    # Hyperparameter marginalization methods
    "MGP",
    # Robust GP methods for high dimensions
    "RobustGPBO",
    # Coordinate-wise optimization methods
    "ECIBO",
    "ECIBOMaximize",
    # Adaptive line search methods
    "ALSBO",
    "ALSBOMaximize",
    # Subspace selection methods
    "ESSIBO",
    "ESSIBOMaximize",
    # Variable selection methods
    "VSBO",
    "MCTSVS",
    "LassoBO",
]
