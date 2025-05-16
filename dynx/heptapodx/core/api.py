"""
Heptapod-B Public API

This module serves as the main public API for Heptapod-B, exposing
all the functions and classes that users should interact with.
"""

# Re-export core components
from .functional_problem import FunctionalProblem, AttrDict

# Re-export initialization functions
from ..init.stage import build_stage as initialize_stage
from ..init.mover import build_mover as initialize_mover
from ..init.perch import build_perch as initialize_perch

# Re-export IO functions
from ..io.yaml_loader import load_config, load_functions_from_yaml

# Re-export numerical generator functions
from ..num.compile import (
    compile_function, 
    compile_eval_function,
    compile_sympy_function,
    compile_numba_function
)
from ..num.generate import compile_num as generate_numerical_model

# Main initialization function
def initialize_model(config, master_config=None):
    """
    Initialize a stage model from a configuration file or dictionary.
    
    This is the main entry point for model initialization from StageCraft.
    
    Parameters
    ----------
    config : str or dict
        Path to the configuration file or configuration dictionary
    master_config : dict, optional
        The master configuration dictionary for parameter inheritance
        
    Returns
    -------
    stage_model : dict
        The stage model
    mover_models : dict
        Dictionary of mover models
    perch_models : dict
        Dictionary of perch models
    """
    # Load the configuration
    if isinstance(config, dict):
        # Already loaded as a dictionary
        config_dict = config
    else:
        # Load from file
        config_dict = load_config(config)
    
    # If master_config is provided, use it directly in the config
    if master_config is not None:
        # Ensure master_config has the expected structure
        if not "_master" in config_dict:
            config_dict["_master"] = {}
        
        # Store the math.functions from master for function inheritance
        if "math" in master_config and "functions" in master_config["math"]:
            config_dict["_master"]["math_functions"] = master_config["math"]["functions"]
        else:
            # Add an empty dictionary if no math.functions found
            config_dict["_master"]["math_functions"] = {}
        
        # Store parameters for parameter inheritance
        if "parameters" in master_config:
            config_dict["_master"]["parameters"] = master_config["parameters"]
        
        # Store settings for settings inheritance
        if "settings" in master_config:
            config_dict["_master"]["settings"] = master_config["settings"]
        
        # Remove the master_file reference to avoid confusion
        if "master_file" in config_dict:
            del config_dict["master_file"]
    
    # Initialize the stage
    stage_problem = initialize_stage(config_dict, master_config)
    
    # Extract perch models from the stage problem
    perch_models = initialize_perch(stage_problem)
    
    # Extract mover models from the stage problem
    mover_models = initialize_mover(config_dict, stage_problem)
    
    return stage_problem, mover_models, perch_models


# Define the public API
__all__ = [
    # Core classes
    "FunctionalProblem",
    "AttrDict",
    
    # Initialization functions
    "initialize_model",
    "initialize_stage",
    "initialize_mover",
    "initialize_perch",
    
    # IO functions
    "load_config",
    "load_functions_from_yaml",
    
    # Numerical functions
    "generate_numerical_model",
    "compile_function",
    "compile_eval_function",
    "compile_sympy_function",
    "compile_numba_function"
] 