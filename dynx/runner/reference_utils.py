"""
Reference model utilities for loading previously solved models.

This module provides functions to locate and load reference models
that have been previously solved and saved as bundles.
"""

from pathlib import Path
from typing import Any, Optional

import numpy as np

from dynx.runner.circuit_runner import CircuitRunner

# Default method to use as reference
DEFAULT_REF_METHOD = "VFI_HDGRID"


def ref_bundle_path(runner: CircuitRunner, x: np.ndarray) -> Optional[Path]:
    """
    Get path to reference bundle directory.

    In method-aware mode, uses DEFAULT_REF_METHOD instead of the method
    specified in the parameter vector.

    Args:
        runner: CircuitRunner instance
        x: Parameter vector

    Returns:
        Path to reference bundle directory, or None if no output_root

    Note:
        If method_param_path is not in runner.param_paths, returns the
        bundle path for the original parameter vector (no substitution).
        This means the "reference" would be the same as the method being
        tested, resulting in zero deviation.
    """
    if not runner.output_root or not runner.method_param_path:
        return None

    # More efficient: directly modify the method parameter index
    if runner.method_param_path in runner.param_paths:
        x_ref = x.copy()
        method_idx = runner.param_paths.index(runner.method_param_path)
        x_ref[method_idx] = DEFAULT_REF_METHOD
    else:
        # If method parameter not found, use original vector
        x_ref = x

    # Get bundle path for reference method
    return runner._bundle_path(x_ref)


def load_reference_model(runner: CircuitRunner, x: np.ndarray) -> Optional[Any]:
    """
    Load reference model if it exists.

    Looks for a bundle saved with DEFAULT_REF_METHOD for the given parameters.

    Args:
        runner: CircuitRunner instance
        x: Parameter vector

    Returns:
        ModelCircuit if reference exists and loads successfully, None otherwise
    """
    ref_path = ref_bundle_path(runner, x)
    if not ref_path:
        return None

    return runner._maybe_load_bundle(ref_path)
