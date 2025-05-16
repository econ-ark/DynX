"""
Validation utilities for Heptapod-B.

This module provides functions to validate:
- State space configurations
- Grid configurations
- YAML structure
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import warnings


# Grid generation alias table mapping shorthand names to canonical types
GRID_TYPE_ALIASES = {
    "uniform": "linspace",
    "log": "geomspace",
    "chebyshev": "chebyshev",
    "int_range": "int_range",
    "enum": "list",
    "list": "list"
}

# Required keys for each grid type
GRID_TYPE_REQUIRED_KEYS = {
    "linspace": ["min", "max", "points"],
    "geomspace": ["min", "max", "points"],
    "chebyshev": ["min", "max", "points"],
    "int_range": ["start", "stop"],  # step is optional with default 1
    "list": ["values"]
}


def emit_deprecation_warning(old_key: str, new_key: str, location: str = "") -> None:
    """
    Emit a consistent deprecation warning.

    Parameters
    ----------
    old_key : str
        The deprecated key
    new_key : str
        The replacement key to use
    location : str, optional
        Where the deprecated key was found (for context)
    """
    loc_msg = f" in {location}" if location else ""
    warnings.warn(
        f"'{old_key}'{loc_msg} is deprecated. Use '{new_key}' instead. "
        f"This will be removed in v1.7.",
        DeprecationWarning,
        stacklevel=2
    )


def validate_state_space_config(state_name: str, state_info: Dict[str, Any]) -> None:
    """
    Validates state space configuration according to the specification.
    
    Parameters
    ----------
    state_name : str
        Name of the state being validated
    state_info : dict
        State space configuration to validate
        
    Raises
    ------
    ValueError
        If validation fails
    """
    # Check for both grid and grid_generation
    has_grid = "grid" in state_info
    has_grid_gen = "methods" in state_info and "grid_generation" in state_info.get("methods", {})
    has_grid_type = "methods" in state_info and "grid.type" in state_info.get("methods", {})
    
    # Check for both old and new grid generation methods
    if has_grid_gen:
        emit_deprecation_warning("grid_generation", "grid.type", f"state '{state_name}'")
    
    if has_grid and (has_grid_gen or has_grid_type):
        raise ValueError(
            f"State '{state_name}' cannot have both 'grid' and 'grid_generation'/'grid.type' defined. "
            "Use 'grid' for manual specification or 'grid.type' for algorithmic generation."
        )
    
    # Validate grid if present
    if has_grid:
        grid_spec = state_info["grid"]
        
        # If grid is a flat list
        if isinstance(grid_spec, list):
            if len(state_info.get("dimensions", [])) != 1:
                raise ValueError(
                    f"State '{state_name}' defines grid as a flat list but has {len(state_info.get('dimensions', []))} dimensions. "
                    "Flat list grids can only be used with a single dimension."
                )
                
        # If grid is a mapping
        elif isinstance(grid_spec, dict):
            dimensions = state_info.get("dimensions", [])
            
            # Check each dimension has a corresponding grid definition
            for dim in dimensions:
                if dim not in grid_spec:
                    raise ValueError(
                        f"State '{state_name}' is missing grid specification for dimension '{dim}'. "
                        "Each dimension must have a corresponding entry in the grid mapping."
                    )
            
            # Check each grid entry is valid (list, int_range style, or grid.type spec)
            for dim, dim_grid in grid_spec.items():
                if dim not in dimensions:
                    raise ValueError(
                        f"State '{state_name}' has grid entry for unknown dimension '{dim}'. "
                        f"Valid dimensions are: {dimensions}"
                    )
                
                if not (isinstance(dim_grid, list) or 
                        (isinstance(dim_grid, dict) and "start" in dim_grid and "stop" in dim_grid) or
                        (isinstance(dim_grid, dict) and "type" in dim_grid)):
                    raise ValueError(
                        f"State '{state_name}' has invalid grid specification for dimension '{dim}'. "
                        "Grid must be either a list of values, an int_range-style mapping with 'start' and 'stop' keys, "
                        "or an algorithmic grid specification with 'type' key."
                    )
        else:
            raise ValueError(
                f"State '{state_name}' has invalid grid specification. "
                "Grid must be either a list (for single dimension) or a mapping (for multiple dimensions)."
            )
    
    # Check for create_mesh flag with full tensor
    if (has_grid and isinstance(state_info["grid"], dict) and 
        len(state_info.get("dimensions", [])) > 1 and 
        "methods" in state_info and state_info["methods"].get("create_mesh", True) is False):
        # This is valid - user is providing a full tensor and disabling mesh creation
        pass


def validate_interpolation_method(state_name: str, state_info: Dict[str, Any]) -> None:
    """
    Validates interpolation method in state space configuration.
    
    Parameters
    ----------
    state_name : str
        Name of the state being validated
    state_info : dict
        State space configuration to validate
        
    Raises
    ------
    ValueError
        If validation fails
    """
    methods = state_info.get("methods", {})
    
    # Check for both old and new interpolation methods
    if "interpolation" in methods:
        emit_deprecation_warning("interpolation", "interp", f"state '{state_name}'")
    
    if "interpolation" in methods and "interp" in methods:
        raise ValueError(
            f"State '{state_name}' cannot have both 'interpolation' and 'interp' defined. "
            "Use only 'interp' for the interpolation method."
        )


def validate_shock_method(shock_name: str, shock_info: Dict[str, Any]) -> None:
    """
    Validates shock method in shock configuration.
    
    Parameters
    ----------
    shock_name : str
        Name of the shock being validated
    shock_info : dict
        Shock configuration to validate
        
    Raises
    ------
    ValueError
        If validation fails
    """
    methods = shock_info.get("methods", {})
    
    # Check for both old and new shock methods
    if "shock_distribution" in methods:
        emit_deprecation_warning("shock_distribution", "shock_method", f"shock '{shock_name}'")
    
    if "shock_distribution" in methods and "shock_method" in methods:
        raise ValueError(
            f"Shock '{shock_name}' cannot have both 'shock_distribution' and 'shock_method' defined. "
            "Use only 'shock_method' for the shock method."
        )


def validate_compilation_method(func_name: str, func_info: Dict[str, Any]) -> None:
    """
    Validates compilation method in function configuration.
    
    Parameters
    ----------
    func_name : str
        Name of the function being validated
    func_info : dict
        Function configuration to validate
        
    Raises
    ------
    ValueError
        If validation fails
    """
    # Skip if not a dict with metadata
    if not isinstance(func_info, dict) or "expr" not in func_info:
        return
    
    # Check for both old and new compilation methods
    if "function_compilation" in func_info:
        emit_deprecation_warning("function_compilation", "compilation", f"function '{func_name}'")
    
    if "constraint_compilation" in func_info:
        emit_deprecation_warning("constraint_compilation", "compilation", f"function '{func_name}'")
    
    if "function_compilation" in func_info and "compilation" in func_info:
        raise ValueError(
            f"Function '{func_name}' cannot have both 'function_compilation' and 'compilation' defined. "
            "Use only 'compilation' for the compilation method."
        )


def validate_multi_output_function(func_name: str, func_info: Dict[str, Any]) -> None:
    """
    Validates a multi-output function definition.
    
    Parameters
    ----------
    func_name : str
        Name of the function being validated
    func_info : dict
        Function configuration to validate
        
    Raises
    ------
    ValueError
        If validation fails
    """
    # Skip if not a multi-output function
    if not isinstance(func_info, dict) or "expr" in func_info:
        return
    
    # Check each output definition
    for output_name, output_info in func_info.items():
        # Skip if not a dictionary (direct expression)
        if not isinstance(output_info, dict):
            continue
        
        # Skip description or other metadata
        if output_name in ["description", "documentation"]:
            continue
        
        # Check for both old and new compilation methods
        if "function_compilation" in output_info:
            emit_deprecation_warning(
                "function_compilation", "compilation", 
                f"function '{func_name}.{output_name}'"
            )
        
        if "function_compilation" in output_info and "compilation" in output_info:
            raise ValueError(
                f"Function output '{func_name}.{output_name}' cannot have both 'function_compilation' "
                f"and 'compilation' defined. Use only 'compilation' for the compilation method."
            )


def is_reference_format(value: Any) -> bool:
    """
    Check if a value is in the reference format: either a single-element list
    or a string enclosed in square brackets.
    
    Parameters
    ----------
    value : Any
        The value to check
    
    Returns
    -------
    bool
        True if the value is in reference format, False otherwise
    """
    # Case 1: Value is a list like ['key']
    if (isinstance(value, list) and len(value) == 1 and isinstance(value[0], str)):
        return True
    
    # Case 2: Value is a string like '[key]'
    if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
        return True
    
    return False


def extract_reference_key(value: Any) -> str:
    """
    Extract the reference key from a value in reference format.
    
    Parameters
    ----------
    value : Any
        The value in reference format
    
    Returns
    -------
    str
        The extracted reference key
    
    Raises
    ------
    ValueError
        If the value is not in reference format
    """
    if not is_reference_format(value):
        raise ValueError(f"Value '{value}' is not in reference format")
    
    # Case 1: Value is a list like ['key']
    if isinstance(value, list):
        return value[0]
    
    # Case 2: Value is a string like '[key]'
    return value[1:-1] 