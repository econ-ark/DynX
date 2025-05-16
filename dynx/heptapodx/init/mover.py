"""
Mover initialization module for Heptapod-B.

This module provides functions for initializing mover models from configuration.
"""

from typing import Dict, Any
from ..core.functional_problem import FunctionalProblem
import warnings

def build_mover(
    config: Dict[str, Any], stage_problem: FunctionalProblem
) -> Dict[str, FunctionalProblem]:
    """
    Build mover models from the configuration.

    Creates a FunctionalProblem for each mover defined in the configuration, copying
    relevant components from the stage model based on the mover's source and target perches.

    Each mover problem will:
    - Only include parameters and settings explicitly specified in its config (or all if 'all' is specified)
    - Include functions referenced by the mover
    - Include constraint definitions from the stage problem
    - Include state definitions for source and target perches only
    - Include shocks referenced in the mover config
    - Have a separate operator attribute containing operator information

    Parameters
    ----------
    config : dict
        The full configuration data
    stage_problem : FunctionalProblem
        The stage problem containing the state definitions

    Returns
    -------
    dict
        Dictionary mapping mover names to their FunctionalProblem instances
    """
    mover_problems = {}

    # Check if the config has a movers section
    if "movers" not in config:
        return mover_problems

    # Process each mover
    for mover_name, mover_config in config["movers"].items():
        # Create a new FunctionalProblem for this mover
        mover_problem = FunctionalProblem()

        # Get source and target perches
        source_perch = mover_config.get("source")
        target_perch = mover_config.get("target")

        # Copy the entire stage methods dictionary
        mover_problem.methods = dict(stage_problem.methods)

        # Handle parameters - check for inherit_parameters flag first
        if (
            "inherit_parameters" in mover_config
            and mover_config["inherit_parameters"]
        ):
            # Copy all parameters from stage
            if hasattr(stage_problem, "parameters_dict"):
                mover_problem.parameters_dict = dict(stage_problem.parameters_dict)
        # Fallback to old behavior - specific parameters list
        elif "parameters" in mover_config:
            parameter_spec = mover_config["parameters"]
            # If 'all' is in the list, copy all parameters
            if "all" in parameter_spec:
                mover_problem.parameters_dict = dict(stage_problem.parameters_dict)
            else:
                # Copy only the specified parameters
                for param in parameter_spec:
                    if param in stage_problem.parameters_dict:
                        mover_problem.parameters_dict[param] = (
                            stage_problem.parameters_dict[param]
                        )

        # Handle settings - check for inherit_settings flag first
        if (
            "inherit_settings" in mover_config
            and mover_config["inherit_settings"]
        ):
            # Copy all settings from stage
            if hasattr(stage_problem, "settings_dict"):
                mover_problem.settings_dict = dict(stage_problem.settings_dict)
        # Fallback to old behavior - specific settings list
        elif "settings" in mover_config:
            settings_spec = mover_config["settings"]
            # If 'all' is in the list, copy all settings
            if "all" in settings_spec:
                mover_problem.settings_dict = dict(stage_problem.settings_dict)
            else:
                # Copy only the specified settings
                for setting in settings_spec:
                    if setting in stage_problem.settings:
                        mover_problem.settings_dict[setting] = (
                            stage_problem.settings_dict[setting]
                        )

        # --- Copy relevant math components ---

        # Copy required functions
        if "functions" in mover_config:
            for func_name in mover_config["functions"]:
                if func_name in stage_problem.math["functions"]:
                    mover_problem._math["functions"][func_name] = (
                        stage_problem.math["functions"][func_name]
                    )

        # Copy all constraints (simpler than checking dependencies)
        mover_problem._math["constraints"] = dict(
            stage_problem.math["constraints"]
        )

        # Copy state definitions for source and target perches
        # Make sure source_perch is a string (not a list)
        if source_perch and isinstance(source_perch, str) and source_perch in stage_problem.math["state_space"]:
            mover_problem._math["state_space"][source_perch] = stage_problem.math[
                "state_space"
            ][source_perch]
        # Make sure target_perch is a string (not a list)
        if target_perch and isinstance(target_perch, str) and target_perch in stage_problem.math["state_space"]:
            mover_problem._math["state_space"][target_perch] = stage_problem.math[
                "state_space"
            ][target_perch]

        # Add required grids specified in the configuration
        if "required_grids" in mover_config:
            for grid_name in mover_config["required_grids"]:
                if grid_name in stage_problem.math["state_space"]:
                    mover_problem._math["state_space"][grid_name] = stage_problem.math[
                        "state_space"
                    ][grid_name]
                else:
                    warnings.warn(f"Required grid '{grid_name}' not found in stage problem. Skipping.")

        # Copy referenced shocks
        if "shocks" in mover_config and isinstance(
            mover_config["shocks"], list
        ):
            for shock_name in mover_config["shocks"]:
                if shock_name in stage_problem.math["shocks"]:
                    mover_problem._math["shocks"][shock_name] = (
                        stage_problem.math["shocks"][shock_name]
                    )

        # Store operator information in a separate attribute
        if "operator" in mover_config:
            mover_problem.operator = mover_config["operator"]

        # Store the mover problem
        mover_problems[mover_name] = mover_problem

    return mover_problems
