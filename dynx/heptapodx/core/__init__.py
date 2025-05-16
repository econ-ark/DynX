"""
Heptapod-B Core Module

This module provides the core components of the Heptapod-B library:
- FunctionalProblem class for mathematical representation
- Validation utilities for state space and configuration
"""

from .functional_problem import FunctionalProblem, AttrDict

__all__ = ["FunctionalProblem", "AttrDict"]

__version__ = "1.6.0"
