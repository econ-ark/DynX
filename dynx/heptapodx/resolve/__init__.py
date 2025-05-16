"""
Heptapod-B Resolve Module

This module provides utilities for resolving references and aliases in configuration.
"""

from .methods import (
    _resolve_method_references,
    resolve_grid_type,
    resolve_shock_method
)

__all__ = [
    "_resolve_method_references",
    "resolve_grid_type",
    "resolve_shock_method"
]
