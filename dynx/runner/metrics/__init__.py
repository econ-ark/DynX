"""
Metrics package for CircuitRunner.

This package provides metric functions that can be used with CircuitRunner
to evaluate model performance and compare against reference models.
"""

from .deviations import (
    dev_c_L2,
    dev_c_Linf,
    dev_a_L2,
    dev_a_Linf,
    dev_v_L2,
    dev_v_Linf,
    dev_pol_L2,
    dev_pol_Linf,
    make_policy_dev_metric,
)

__all__ = [
    "dev_c_L2",
    "dev_c_Linf", 
    "dev_a_L2",
    "dev_a_Linf",
    "dev_v_L2",
    "dev_v_Linf",
    "dev_pol_L2",
    "dev_pol_Linf",
    "make_policy_dev_metric",
] 