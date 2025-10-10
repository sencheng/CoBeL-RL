"""
This module provides optimizer classes for
the fitting of model hyper-parameters to behavioral data.

Classes
-------
Optimizer
    Abstract optimizer class from which all optimizers classes are derived
GridSearchOptimizer
    A simple grid search optimizer
EAOptimizer
    A basic evolutionary algorithm optimizer

"""

from .optimizer import Optimizer
from .grid_search import GridSearchOptimizer
from .evolution import EAOptimizer
