"""
Stage initialization module for Heptapod-B.

This module provides functions for initializing a Stage model from configuration.
"""

from typing import Dict, List, Any, Optional, Union, Callable, Tuple
import warnings
import os

from ..core.functional_problem import FunctionalProblem
from ..core.validation import (
    validate_state_space_config,
    validate_interpolation_method,
    validate_shock_method,
    validate_compilation_method,
    validate_multi_output_function,
    emit_deprecation_warning
)
from ..resolve.methods import (
    _resolve_method_references,
    resolve_parameter_references
)
from ..io.yaml_loader import load_config


def build_stage(config: Union[str, Dict[str, Any]], master_config: Optional[Dict[str, Any]] = None) -> FunctionalProblem:
    """
    Builds a FunctionalProblem from a configuration file or dictionary.
    
    Parameters
    ----------
    config : str or dict
        Path to the configuration file or a configuration dictionary
    master_config : dict, optional
        Master configuration dictionary
        
    Returns
    -------
    FunctionalProblem
        A configured FunctionalProblem instance
    """
    # If string is provided, assume it's a path to a YAML file
    config_path = None
    if isinstance(config, str):
        config_path = config
        config = load_config(config)
    
    # Create a new instance of FunctionalProblem
    problem = FunctionalProblem()
    
    # Process master file if specified
    master_functions = None
    master_math_functions = None
    
    if "master_file" in config:
        # Get path to master file, resolving relative paths
        master_path = config["master_file"]
        
        # Determine the directory containing the stage config file
        if config_path:
            config_dir = os.path.dirname(os.path.abspath(config_path))
            master_path = os.path.join(config_dir, master_path)
        elif "_source_file" in config:
            config_dir = os.path.dirname(config["_source_file"])
            master_path = os.path.join(config_dir, master_path)
        
        # Load master config
        try:
            master_config = load_config(master_path)
        except FileNotFoundError:
            # Try once more relative to the current working directory
            try:
                master_config = load_config(config["master_file"])
            except FileNotFoundError:
                raise FileNotFoundError(f"Master configuration file not found: {master_path}")
        
        # Store master config in the original config for reference
        config["_master"] = master_config
        
        # Extract functions and math.functions from master if they exist
        if "functions" in master_config:
            master_functions = master_config["functions"]
        
        if "math" in master_config and "functions" in master_config["math"]:
            master_math_functions = master_config["math"]["functions"]
            # Store for reference in the original config
            if "_master" not in config:
                config["_master"] = {}
            config["_master"]["math_functions"] = master_math_functions
    # If master_config is directly provided as a parameter, use it
    elif master_config is not None:
        # Extract functions and math.functions from master_config if they exist
        if "functions" in master_config:
            master_functions = master_config["functions"]
        
        if "math" in master_config and "functions" in master_config["math"]:
            master_math_functions = master_config["math"]["functions"]
            # Store for reference in the original config
            if "_master" not in config:
                config["_master"] = {}
            config["_master"]["math_functions"] = master_math_functions
    
    # Resolve parameter references using both stage and master parameters
    config = resolve_parameter_references(config, master_config)
    
    # Special handling for stage parameters - ensure they're properly resolved
    if "stage" in config and "parameters" in config["stage"]:
        stage_parameters = config["stage"]["parameters"]
        resolved_parameters = {}
        
        for param_name, param_value in stage_parameters.items():
            # Handle special case of list reference format ['param_name']
            if isinstance(param_value, list) and len(param_value) == 1 and isinstance(param_value[0], str):
                ref_key = param_value[0]
                # Look up in master config first
                if master_config and "parameters" in master_config and ref_key in master_config["parameters"]:
                    resolved_parameters[param_name] = master_config["parameters"][ref_key]
                else:
                    # Keep the original value if reference can't be resolved
                    resolved_parameters[param_name] = param_value
            else:
                resolved_parameters[param_name] = param_value
        
        # Update the stage parameters with resolved values
        config["stage"]["parameters"] = resolved_parameters
    
    # Extract stage config from the loaded configuration
    stage_config = config.get("stage", {})
    
    # Set up basic properties of the problem
    problem.name = stage_config.get("name", "UnnamedStage")
    problem.description = stage_config.get("description", "")
    problem.version = stage_config.get("version", "1.0.0")
    problem.is_portable = stage_config.get("is_portable", False)
    
    # Set stage optimization method
    problem.method = stage_config.get("method", "default")
    problem.kind = stage_config.get("kind", "sequential")
    
    # Set parameters dict and settings dict
    problem.parameters_dict = stage_config.get("parameters", {})
    problem.settings_dict = stage_config.get("settings", {})
    
    # Copy methods if available (not same as method)
    if "methods" in stage_config:
        # Create resolved methods dictionary
        resolved_methods = {}
        
        # Iterate through methods to resolve references
        for method_name, method_value in stage_config["methods"].items():
            # Check if the value is a reference in the format ["method_name"]
            if isinstance(method_value, list) and len(method_value) == 1 and isinstance(method_value[0], str):
                ref_key = method_value[0]
                # Look up in master config first
                if master_config and "methods" in master_config and ref_key in master_config["methods"]:
                    resolved_methods[method_name] = master_config["methods"][ref_key]
                else:
                    # Keep the original value if reference can't be resolved
                    resolved_methods[method_name] = method_value
            else:
                # Keep non-reference values as they are
                resolved_methods[method_name] = method_value
        
        # Assign the resolved methods to the problem
        problem.methods = resolved_methods
    
    # Set up math component (math namespaces contain functions, constraints, state spaces)
    math_config = stage_config.get("math", {})
    
    # Initialize math dictionary for the problem
    problem.math = {
        "functions": {},
        "constraints": {},
        "state_space": {},
    }
    
    # Process math component if it exists
    if math_config:
        math_dict = problem.math
        
        # Process functions section - handle inheritance from master file
        if "functions" in math_config:
            # Extract stage functions
            stage_functions = math_config["functions"]
            merged_functions = {}
            
            # Process each function in the stage file
            for func_name, func_info in stage_functions.items():
                # Check for inheritance (several syntaxes supported)
                
                # Case 1: Bracket notation inheritance - ["function_name"]
                if isinstance(func_info, list) and len(func_info) == 1 and isinstance(func_info[0], str):
                    base_func_name = func_info[0]
                    if master_math_functions is not None and base_func_name in master_math_functions:
                        merged_functions[func_name] = master_math_functions[base_func_name].copy()
                    else:
                        warnings.warn(
                            f"Function '{base_func_name}' referenced for inheritance not found in master file",
                            UserWarning
                        )
                        # Keep the reference anyway
                        merged_functions[func_name] = func_info
                
                # Case 2: Direct reference inheritance - function_name: function_name
                elif isinstance(func_info, str) and master_math_functions is not None and func_info in master_math_functions:
                    merged_functions[func_name] = master_math_functions[func_info].copy()
                
                # Case 3: Function dict with inherit property
                elif isinstance(func_info, dict) and "inherit" in func_info:
                    inherit_value = func_info["inherit"]
                    
                    # Case 3a: inherit: true - inherit same named function from master
                    if inherit_value is True:
                        if master_math_functions is not None and func_name in master_math_functions:
                            # Get the base function and create a deep copy
                            merged_func = master_math_functions[func_name].copy()
                            
                            # Apply any local overrides except the inherit key
                            for key, value in func_info.items():
                                if key != "inherit":  # Skip the inherit tag
                                    merged_func[key] = value
                                    
                            merged_functions[func_name] = merged_func
                        else:
                            warnings.warn(
                                f"Function '{func_name}' specified for inheritance but not found in master file",
                                UserWarning
                            )
                            # Keep the local version anyway (might just have inherit flag)
                            merged_functions[func_name] = func_info.copy()
                            # Remove inherit tag
                            merged_functions[func_name].pop("inherit")
                    
                    # Case 3b: inherit: "function_name" - inherit from named function with overrides
                    elif isinstance(inherit_value, str):
                        base_func_name = inherit_value
                        if master_math_functions is not None and base_func_name in master_math_functions:
                            # Start with base function from master
                            merged_func = master_math_functions[base_func_name].copy()
                            # Override with local properties
                            for key, value in func_info.items():
                                if key != "inherit":  # Skip the inherit tag
                                    merged_func[key] = value
                            merged_functions[func_name] = merged_func
                        else:
                            warnings.warn(
                                f"Base function '{base_func_name}' for inheritance not found in master file",
                                UserWarning
                            )
                            # Keep local version but remove inherit tag
                            local_func = func_info.copy()
                            local_func.pop("inherit")
                            merged_functions[func_name] = local_func
                    
                    # Case 3c: Invalid inherit value
                    else:
                        warnings.warn(
                            f"Invalid 'inherit' value for function '{func_name}': {inherit_value}",
                            UserWarning
                        )
                        # Keep local version but remove inherit tag
                        local_func = func_info.copy()
                        local_func.pop("inherit")
                        merged_functions[func_name] = local_func
                
                # Case 4: No inheritance - use function as defined in stage
                else:
                    merged_functions[func_name] = func_info
            
            # Add remaining functions from master that weren't overridden or explicitly inherited
            if master_math_functions is not None:
                for func_name, func_info in master_math_functions.items():
                    if func_name not in merged_functions:
                        merged_functions[func_name] = func_info
            
            # Validate function definitions
            for func_name, func_info in merged_functions.items():
                validate_compilation_method(func_name, func_info)
                validate_multi_output_function(func_name, func_info)
            
            math_dict["functions"] = merged_functions
        elif master_math_functions is not None:
            # If no stage functions but master functions exist, use master functions
            for func_name, func_info in master_math_functions.items():
                validate_compilation_method(func_name, func_info)
                validate_multi_output_function(func_name, func_info)
            
            math_dict["functions"] = master_math_functions

        # Add constraints
        if "constraints" in math_config:
            # Validate constraint definitions
            for constraint_name, constraint_info in math_config["constraints"].items():
                validate_compilation_method(constraint_name, constraint_info)
            
            math_dict["constraints"] = math_config["constraints"]

        # Add shocks if available
        if "shocks" in math_config:
            # Resolve references within math.shocks methods
            resolved_shocks = {}
            for shock_name, shock_info in math_config["shocks"].items():
                # Validate shock method
                validate_shock_method(shock_name, shock_info)
                
                resolved_info = shock_info.copy()  # Avoid modifying original config dict
                if "methods" in resolved_info:
                    resolved_info["methods"] = _resolve_method_references(
                        resolved_info["methods"], problem.methods
                    )
                resolved_shocks[shock_name] = resolved_info
            math_dict["shocks"] = resolved_shocks

        # Add state_space definitions from math.state_space
        if "state_space" in math_config:
            for state_name, state_info in math_config["state_space"].items():
                # Validate state space configuration
                validate_state_space_config(state_name, state_info)
                validate_interpolation_method(state_name, state_info)
                
                # Create state entry
                math_dict["state_space"][state_name] = {
                    "description": state_info.get("description", ""),
                    "dimensions": state_info.get("dimensions", []),
                }

                # Copy settings if available
                if "settings" in state_info:
                    math_dict["state_space"][state_name]["settings"] = state_info["settings"]

                # Handle explicit grid specification if present
                if "grid" in state_info:
                    math_dict["state_space"][state_name]["grid"] = state_info["grid"]

                # Resolve references in methods and store
                if "methods" in state_info:
                    resolved_methods = _resolve_method_references(
                        state_info["methods"], problem.methods
                    )
                    math_dict["state_space"][state_name]["methods"] = resolved_methods
        
        # Copy other math sections if present
        for key, value in math_config.items():
            if key not in ["functions", "constraints", "shocks", "state_space"]:
                math_dict[key] = value

    # If top-level functions exist in stage or master, add them to the problem
    if "functions" in config:
        problem.functions = config["functions"]
    elif master_functions:
        problem.functions = master_functions

    return problem
