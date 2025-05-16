"""
Method reference resolution utilities for Heptapod-B.

This module provides functions to:
- Resolve [key] references in config dictionaries
- Resolve method aliases
- Validate reference resolution
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import re
import warnings
from ..core.validation import (
    is_reference_format, 
    extract_reference_key,
    GRID_TYPE_ALIASES,
    GRID_TYPE_REQUIRED_KEYS,
    emit_deprecation_warning
)


def _resolve_method_references(
    method_dict: Dict[str, Any], stage_methods: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Resolves [key] references in a methods dictionary using the stage_methods mapping.
    Handles values that are strings '[key]' or lists containing one string 'key'.

    Parameters
    ----------
    method_dict : dict
        The dictionary potentially containing [key] references.
    stage_methods : dict
        The top-level dictionary mapping keys to actual method names.

    Returns
    -------
    dict
        A new dictionary with references resolved.
    """
    resolved_methods = {}
    if not isinstance(method_dict, dict):
        return method_dict

    for key, value in method_dict.items():
        if is_reference_format(value):
            # Extract the reference key
            ref_key = extract_reference_key(value)
            
            # Resolve if it's in stage_methods
            if ref_key in stage_methods:
                resolved_methods[key] = stage_methods[ref_key]
            else:
                # Keep original value
                resolved_methods[key] = value
                warnings.warn(
                    f"Method reference key '{ref_key}' (from {value}) not found in stage methods. "
                    f"Keeping original.",
                    UserWarning
                )
        # Handle deprecated keys and aliases
        elif key == "grid_generation" and isinstance(value, dict) and "type" in value:
            # Handle grid generation deprecation
            emit_deprecation_warning("grid_generation", "grid.type")
            
            # Clone the dictionary to avoid modifying the original
            grid_gen_dict = value.copy()
            
            # Apply alias mapping if the type is a known alias
            grid_type = grid_gen_dict["type"]
            if isinstance(grid_type, str) and grid_type in GRID_TYPE_ALIASES:
                grid_gen_dict["type"] = GRID_TYPE_ALIASES[grid_type]
                
            # Validate required keys for the grid type
            canonical_type = grid_gen_dict["type"]
            if canonical_type in GRID_TYPE_REQUIRED_KEYS:
                required_keys = GRID_TYPE_REQUIRED_KEYS[canonical_type]
                missing_keys = [key for key in required_keys if key not in grid_gen_dict]
                if missing_keys:
                    warnings.warn(
                        f"Missing required keys {missing_keys} for grid type '{canonical_type}'.",
                        UserWarning
                    )
            
            resolved_methods[key] = grid_gen_dict
        elif key == "shock_distribution" and not isinstance(value, dict):
            # Handle shock_distribution deprecation
            emit_deprecation_warning("shock_distribution", "shock_method")
            resolved_methods["shock_method"] = value
        elif key == "interpolation" and not isinstance(value, dict):
            # Handle interpolation deprecation
            emit_deprecation_warning("interpolation", "interp")
            resolved_methods["interp"] = value
        elif key == "function_compilation" and not isinstance(value, dict):
            # Handle function_compilation deprecation
            emit_deprecation_warning("function_compilation", "compilation")
            resolved_methods["compilation"] = value
        elif key == "constraint_compilation" and not isinstance(value, dict):
            # Handle constraint_compilation deprecation
            emit_deprecation_warning("constraint_compilation", "compilation")
            resolved_methods["compilation"] = value
        else:
            # Not a reference, keep original value
            resolved_methods[key] = value

    return resolved_methods


def resolve_parameter_references(
    config_dict: Dict[str, Any], master_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Recursively resolves parameter references in a configuration dictionary 
    using stage-level parameters first, then master file parameters if needed.
    
    Parameters
    ----------
    config_dict : dict
        The configuration dictionary potentially containing parameter references
    master_config : dict, optional
        Master configuration with parameters and settings sections
        
    Returns
    -------
    dict
        A new dictionary with parameter references resolved
    """
    if not isinstance(config_dict, dict):
        return config_dict
    
    # If no master config is provided but there is one in the config_dict, use it
    if master_config is None and "_master" in config_dict:
        master_config = config_dict["_master"]
    
    # Extract stage-level parameters and settings (if available)
    stage_params = {}
    stage_settings = {}
    if "stage" in config_dict:
        stage_params = config_dict["stage"].get("parameters", {})
        stage_settings = config_dict["stage"].get("settings", {})
    
    # Get master parameters and settings (if available)
    master_params = {}
    master_settings = {}
    if master_config:
        master_params = master_config.get("parameters", {})
        master_settings = master_config.get("settings", {})
    
    # Create new dict to avoid modifying original
    resolved_dict = {}
    
    # Process each item in the dictionary
    for key, value in config_dict.items():
        # Skip _master to avoid circular references
        if key == "_master":
            continue
            
        # If the value is a dictionary, recursively resolve it
        if isinstance(value, dict):
            resolved_dict[key] = resolve_parameter_references(value, master_config)
        # If the value is a list, check each item
        elif isinstance(value, list):
            resolved_list = []
            
            # Special case: ['param_name'] reference format
            if len(value) == 1 and isinstance(value[0], str):
                ref_key = value[0]
                # Check in order of precedence
                if ref_key in stage_params:
                    resolved_dict[key] = stage_params[ref_key]
                    continue
                elif ref_key in stage_settings:
                    resolved_dict[key] = stage_settings[ref_key]
                    continue
                elif ref_key in master_params:
                    resolved_dict[key] = master_params[ref_key]
                    continue
                elif ref_key in master_settings:
                    resolved_dict[key] = master_settings[ref_key]
                    continue
            
            # Process each item in the list
            for item in value:
                if isinstance(item, dict):
                    resolved_list.append(resolve_parameter_references(item, master_config))
                elif is_reference_format(item):
                    ref_key = extract_reference_key(item)
                    # First try to resolve from stage parameters
                    if ref_key in stage_params:
                        resolved_list.append(stage_params[ref_key])
                    # Then try stage settings
                    elif ref_key in stage_settings:
                        resolved_list.append(stage_settings[ref_key])
                    # Fall back to master parameters
                    elif ref_key in master_params:
                        resolved_list.append(master_params[ref_key])
                    # Finally try master settings
                    elif ref_key in master_settings:
                        resolved_list.append(master_settings[ref_key])
                    else:
                        resolved_list.append(item)
                else:
                    resolved_list.append(item)
            resolved_dict[key] = resolved_list
        # If the value is a reference, resolve it
        elif is_reference_format(value):
            ref_key = extract_reference_key(value)
            # First try to resolve from stage parameters
            if ref_key in stage_params:
                resolved_dict[key] = stage_params[ref_key]
            # Then try stage settings
            elif ref_key in stage_settings:
                resolved_dict[key] = stage_settings[ref_key]
            # Fall back to master parameters
            elif ref_key in master_params:
                resolved_dict[key] = master_params[ref_key]
            # Finally try master settings
            elif ref_key in master_settings:
                resolved_dict[key] = master_settings[ref_key]
            else:
                resolved_dict[key] = value
        # Otherwise keep the original value
        else:
            resolved_dict[key] = value
    
    return resolved_dict


def resolve_grid_type(grid_spec: Dict[str, Any], stage_methods: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolves grid type references and aliases in a grid specification.

    Parameters
    ----------
    grid_spec : dict
        The grid specification dictionary (containing 'type' and other parameters)
    stage_methods : dict
        The stage-level methods dictionary with defaults

    Returns
    -------
    dict
        A new dictionary with resolved grid type
    """
    if not isinstance(grid_spec, dict) or "type" not in grid_spec:
        return grid_spec
    
    result = grid_spec.copy()
    grid_type = result["type"]
    
    # Handle type references
    if is_reference_format(grid_type):
        ref_key = extract_reference_key(grid_type)
        if ref_key in stage_methods:
            # Set the actual grid type from the reference
            result["type"] = stage_methods[ref_key]
        else:
            warnings.warn(
                f"Grid type reference key '{ref_key}' not found in stage methods.",
                UserWarning
            )
    
    # Apply alias mapping if the type is a known alias
    if isinstance(result["type"], str) and result["type"] in GRID_TYPE_ALIASES:
        result["type"] = GRID_TYPE_ALIASES[result["type"]]
    
    return result


def resolve_shock_method(shock_info: Dict[str, Any], stage_methods: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolves shock method references in a shock configuration.

    Parameters
    ----------
    shock_info : dict
        The shock configuration dictionary
    stage_methods : dict
        The stage-level methods dictionary with defaults

    Returns
    -------
    dict
        A new dictionary with resolved shock method
    """
    if not isinstance(shock_info, dict) or "methods" not in shock_info:
        return shock_info
    
    result = shock_info.copy()
    methods = result.get("methods", {})
    
    # Handle shock_distribution deprecated key
    if "shock_distribution" in methods and "shock_method" not in methods:
        emit_deprecation_warning("shock_distribution", "shock_method")
        methods["shock_method"] = methods["shock_distribution"]
    
    # Resolve method references
    result["methods"] = _resolve_method_references(methods, stage_methods)
    
    return result 