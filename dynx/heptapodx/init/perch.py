"""
Perch initialization module for Heptapod-B.

This module provides functions for initializing perch models from the stage model.
"""

from typing import Dict, Any
from ..core.functional_problem import FunctionalProblem


def build_perch(
    stage_problem: FunctionalProblem,
) -> Dict[str, FunctionalProblem]:
    """
    Build perch models from the stage model's state definitions.

    Creates a simplified FunctionalProblem for each perch (state) in the stage model.
    Unlike mover models which have a complete set of components, perch models
    by default only contain state entries.

    Parameters
    ----------
    stage_problem : FunctionalProblem
        The stage problem containing the state definitions

    Returns
    -------
    dict
        Dictionary mapping perch names to their FunctionalProblem instances
    """
    perch_problems = {}

    # Process each state in the stage problem
    for state_name, state_info in stage_problem.math["state_space"].items():
        # Create a new FunctionalProblem for this perch
        perch_problem = FunctionalProblem()

        # Copy only the state definition for this perch
        perch_problem._math["state_space"][state_name] = state_info

        # Copy all parameters from the stage model
        if hasattr(stage_problem, "parameters_dict"):
            perch_problem.parameters_dict = dict(stage_problem.parameters_dict)

        # Copy all settings from the stage model
        if hasattr(stage_problem, "settings_dict"):
            perch_problem.settings_dict = dict(stage_problem.settings_dict)

        # Store the perch problem
        perch_problems[state_name] = perch_problem

    return perch_problems
