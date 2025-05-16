"""
Heptapod-B Initialization Module

This module provides functions for initializing:
- Stage models (build_stage)
- Mover models (build_mover)
- Perch models (build_perch)
"""

from .stage import build_stage

__all__ = ["build_stage"]
