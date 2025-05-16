"""
YAML loading utilities for Heptapod-B.

This module provides functions to load YAML configuration files,
including custom tag handling and safe loading.
"""

import os
import yaml
from typing import Dict, Any, Union, Optional
import warnings


def load_config(config_file: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Parameters
    ----------
    config_file : str
        Path to the YAML configuration file

    Returns
    -------
    dict
        Dictionary with configuration data
    
    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist
    ValueError
        If there is an error loading the configuration file
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(
            f"Configuration file not found: {config_file}"
        )

    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise ValueError(f"Error loading configuration file: {str(e)}")
    
    # Check if this config references a master file for parameters
    if "master_file" in config:
        master_file_path = config["master_file"]
        # Handle relative paths based on the current config file's location
        if not os.path.isabs(master_file_path):
            base_dir = os.path.dirname(os.path.abspath(config_file))
            master_file_path = os.path.join(base_dir, master_file_path)
        
        # Load and merge the master file parameters
        config = merge_master_parameters(config, master_file_path)
    
    return config


def merge_master_parameters(config: Dict[str, Any], master_file_path: str) -> Dict[str, Any]:
    """
    Load a master configuration file and merge its parameters, settings, and functions with the stage config.
    
    Parameters
    ----------
    config : dict
        The stage configuration dictionary
    master_file_path : str
        Path to the master configuration file
        
    Returns
    -------
    dict
        The merged configuration with master parameters available
        
    Raises
    ------
    FileNotFoundError
        If the master file doesn't exist
    ValueError
        If there is an error loading the master file
    """
    if not os.path.exists(master_file_path):
        raise FileNotFoundError(
            f"Master configuration file not found: {master_file_path}"
        )
    
    try:
        with open(master_file_path, "r") as f:
            master_config = yaml.safe_load(f)
    except Exception as e:
        raise ValueError(f"Error loading master configuration file: {str(e)}")
    
    # Store the master information in the config
    config["_master"] = {
        "file_path": master_file_path,
        "parameters": master_config.get("parameters", {}),
        "settings": master_config.get("settings", {})
    }
    
    # If master contains functions, store them too
    if "functions" in master_config:
        config["_master"]["functions"] = master_config["functions"]
    
    # If master contains math.functions, store them separately
    if "math" in master_config and "functions" in master_config["math"]:
        config["_master"]["math_functions"] = master_config["math"]["functions"]
    
    return config


def dump_config(config: Dict[str, Any], file_path: str) -> None:
    """
    Dump configuration to a YAML file.

    Parameters
    ----------
    config : dict
        Configuration dictionary to dump
    file_path : str
        Path to the output YAML file
    
    Raises
    ------
    ValueError
        If there is an error dumping the configuration
    """
    try:
        with open(file_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    except Exception as e:
        raise ValueError(f"Error dumping configuration: {str(e)}")


def load_functions_from_yaml(yaml_file: str) -> Dict[str, Dict[str, Any]]:
    """
    Load function definitions from a YAML file.
    
    This is a convenience function for loading only function definitions
    from a YAML file, typically used for testing or standalone function libraries.

    Parameters
    ----------
    yaml_file : str
        Path to the YAML file containing function definitions

    Returns
    -------
    dict
        Dictionary of function definitions
    
    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist
    ValueError
        If there is an error loading the YAML file or it doesn't contain functions
    """
    config = load_config(yaml_file)
    
    # Try to find functions in various locations
    if "functions" in config:
        return config["functions"]
    elif "stage" in config and "math" in config["stage"] and "functions" in config["stage"]["math"]:
        return config["stage"]["math"]["functions"]
    else:
        warnings.warn(
            "No functions found in YAML file. Returning empty dictionary.",
            UserWarning
        )
        return {}
