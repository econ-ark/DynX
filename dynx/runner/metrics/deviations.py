"""
Deviation metrics for comparing model policies against reference solutions.

This module provides metric functions that compute L2 and L∞ norms of the
difference between a model's policy and a reference model's policy.
"""

from typing import Any, Callable, Literal, Optional

import numpy as np

from dynx.runner.circuit_runner import CircuitRunner
from dynx.runner.reference_utils import load_reference_model


def _extract_policy(
    model: Any, stage: str = "OWNC", sol_attr: str = "policy", key: str = "c"
) -> Optional[np.ndarray]:
    """
    Extract policy array from a model's stage structure.

    Args:
        model: ModelCircuit instance
        stage: Stage name to extract from (default: "OWNC")
        sol_attr: Solution attribute (default: "policy")
        key: Policy key within the solution (default: "c")

    Returns:
        Policy array or None if not found
    """
    try:
        # Try stage-based extraction first
        stage_obj = model.get_stage(stage)
        if hasattr(stage_obj, "dcsn") and hasattr(stage_obj.dcsn, "sol"):
            sol = stage_obj.dcsn.sol
            # Handle both attribute and dict-style access
            if hasattr(sol, sol_attr):
                policy_obj = getattr(sol, sol_attr)
                if hasattr(policy_obj, key):
                    return np.asarray(getattr(policy_obj, key))
                elif isinstance(policy_obj, dict) and key in policy_obj:
                    return np.asarray(policy_obj[key])
                elif isinstance(policy_obj, np.ndarray) and key == "":
                    # Policy is already an array (no nested key)
                    return policy_obj
                elif isinstance(policy_obj, np.ndarray) and not key:
                    # Also handle None key
                    return policy_obj

        # Fallback to cntn perch
        if hasattr(stage_obj, "cntn") and hasattr(stage_obj.cntn, "sol"):
            sol = stage_obj.cntn.sol
            if hasattr(sol, sol_attr):
                policy_obj = getattr(sol, sol_attr)
                if hasattr(policy_obj, key):
                    return np.asarray(getattr(policy_obj, key))
                elif isinstance(policy_obj, dict) and key in policy_obj:
                    return np.asarray(policy_obj[key])
                elif isinstance(policy_obj, np.ndarray) and key == "":
                    # Policy is already an array (no nested key)
                    return policy_obj
                elif isinstance(policy_obj, np.ndarray) and not key:
                    # Also handle None key
                    return policy_obj

    except (AttributeError, KeyError):
        pass

    # Last resort: try direct attribute on model
    try:
        return np.asarray(getattr(model, key))
    except AttributeError:
        return None


def make_policy_dev_metric(
    policy_attr: str,
    norm: Literal["L2", "Linf"],
    stage: str = "OWNC",
    sol_attr: str = "policy",
) -> Callable[[Any, CircuitRunner, np.ndarray], float]:
    """
    Factory to create policy deviation metrics.

    Args:
        policy_attr: Policy key/attribute name (e.g., "c", "a", "v")
        norm: Norm type - "L2" for Euclidean norm, "Linf" for infinity norm
        stage: Stage name to extract from (default: "OWNC")
        sol_attr: Solution attribute (default: "policy")

    Returns:
        Metric function that accepts (model, *, _runner, _x) and returns float

    Note:
        The L2 norm returns the raw Euclidean norm (not normalized).
        For RMS error, divide by sqrt(n) where n is the number of elements.
    """

    def metric(
        model: Any, *, _runner: Optional[CircuitRunner] = None, _x: Optional[np.ndarray] = None
    ) -> float:
        """Compute deviation of model policy from reference."""
        # Handle backward compatibility
        if _runner is None or _x is None:
            return np.nan

        # Load reference model
        ref_model = load_reference_model(_runner, _x)
        if ref_model is None:
            return np.nan

        # Get policies using the extractor
        policy = _extract_policy(model, stage, sol_attr, policy_attr)
        ref_policy = _extract_policy(ref_model, stage, sol_attr, policy_attr)

        if policy is None or ref_policy is None:
            return np.nan

        # Compute difference
        diff = policy - ref_policy

        # Return appropriate norm
        if norm == "L2":
            return float(np.linalg.norm(diff.ravel(), ord=2))
        else:  # Linf
            return float(np.linalg.norm(diff.ravel(), ord=np.inf))

    # Set a meaningful name for the metric
    metric.__name__ = f"dev_{policy_attr}_{norm}"
    metric.__doc__ = f"{norm} deviation of {policy_attr} from reference model"

    return metric


# Create concrete metric functions for consumption only
# (most reliable since consumption is commonly available)
dev_c_L2 = make_policy_dev_metric("c", "L2")
dev_c_Linf = make_policy_dev_metric("c", "Linf")

# These require specific model structures and may not work for all models
# Users should create their own with appropriate stage/sol_attr parameters
dev_a_L2 = make_policy_dev_metric("a", "L2")
dev_a_Linf = make_policy_dev_metric("a", "Linf")
dev_v_L2 = make_policy_dev_metric("v", "L2", sol_attr="value")
dev_v_Linf = make_policy_dev_metric("v", "Linf", sol_attr="value")
dev_pol_L2 = make_policy_dev_metric("pol", "L2")
dev_pol_Linf = make_policy_dev_metric("pol", "Linf")
