"""
1D Endogenous Grid Method (EGM) whisperer and operator factories for lifecycle models.

This module provides the interface layer between the core computational operators
defined in horses.py and the ModCraft framework. It includes:

- operator_factory_in_situ: Creates operators for in-situ solving within the Stage
- whisperer_external: Implements the external solver approach 
- whisper_ls: Legacy function maintained for backward compatibility
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Import the core EGM operators from the horses module
from .horses import operator_factory_cntn_to_dcsn, operator_factory_dcsn_to_arvl


def whisper_ls(stage):
    """Whisper operator for the lifecycle model (legacy function).

    Parameters
    ----------
    stage : Stage
        The stage to solve

    Returns
    -------
    dict
        Dictionary containing the operators for the lifecycle model
    """
    
    cntn_to_dcsn_op = operator_factory_cntn_to_dcsn(stage.cntn_to_dcsn)
    dcsn_to_arvl_op = operator_factory_dcsn_to_arvl(stage.dcsn_to_arvl)

    return {
        "backward": {
            "cntn_to_dcsn": cntn_to_dcsn_op,
            "dcsn_to_arvl": dcsn_to_arvl_op,
        },
        "forward": {},
    }


def operator_factory_in_situ(stage):
    """In-situ operator factory for the lifecycle model.
    
    Creates operators for the stage that implement the EGM solution
    approach directly within the model's computational structure.

    Parameters
    ----------
    stage : Stage
        The stage to solve

    Returns
    -------
    dict
        Dictionary containing the operators for the lifecycle model
    """
    
    # Create the operators using the core EGM implementation
    cntn_to_dcsn_op = operator_factory_cntn_to_dcsn(stage.cntn_to_dcsn)
    dcsn_to_arvl_op = operator_factory_dcsn_to_arvl(stage.dcsn_to_arvl)

    return {
        "backward": {
            "cntn_to_dcsn": cntn_to_dcsn_op,
            "dcsn_to_arvl": dcsn_to_arvl_op,
        },
        "forward": {},
    }


def whisperer_external(stage):
    """External whisperer for the lifecycle model.
    
    Demonstrates the external solver approach where calculations are
    performed outside the stage and then results are pushed back into the perches.

    Parameters
    ----------
    stage : Stage
        The stage to solve externally

    Returns
    -------
    bool
        True if the external solving was successful, False otherwise
    """
    print("Running external whisperer solver...")

    try:
        # 1. Extract necessary data from stage perches and movers
        # In a real external solver, this data would be passed to an external library
        cntn_sol = stage.perches["cntn"].sol
        
        # Get models from movers
        cntn_to_dcsn_model = stage.cntn_to_dcsn.model
        dcsn_to_arvl_model = stage.dcsn_to_arvl.model
        
        # 2. Perform the external solution
        # For demonstration, we'll use the same operators but pretend they're external
        cntn_to_dcsn_op = operator_factory_cntn_to_dcsn(stage.cntn_to_dcsn)
        dcsn_result = cntn_to_dcsn_op(cntn_sol)
        
        dcsn_to_arvl_op = operator_factory_dcsn_to_arvl(stage.dcsn_to_arvl)  
        arvl_result = dcsn_to_arvl_op(dcsn_result)
        
        # 3. Update the stage's perches with the results
        stage.perches["dcsn"].sol = dcsn_result
        stage.perches["arvl"].sol = arvl_result
        
        # 4. Mark the stage as solved
        stage.status_flags["solved"] = True
        
        print("External whisperer solver completed successfully.")
        return True
        
    except Exception as e:
        print(f"Error in external whisperer: {e}")
        return False


def plot_results(stage, results_dcsn, results_arvl, image_dir, filename_suffix=""):
    """Plot the results of the lifecycle model.

    Parameters
    ----------
    stage : Stage
        The stage object containing the model
    results_dcsn : dict
        The results at the decision perch
    results_arvl : dict
        The results at the arrival perch
    image_dir : str
        Directory to save the plots to
    filename_suffix : str, optional
        Suffix to add to the output filenames, by default ""
    """
    plt.figure(figsize=(12, 10))

    # 1. Consumption Policy Function (from results_dcsn)
    plt.subplot(2, 2, 1)
    m_grid = stage.dcsn.grid.m
    policy_dcsn_array = results_dcsn["policy"]
    plt.plot(m_grid, policy_dcsn_array, label="Consumption Policy c(m)")
    plt.plot(m_grid, m_grid, "k--", label="m=c (45-degree line)", linewidth=0.5)
    plt.title("Consumption Policy Function")
    plt.xlabel("Cash-on-Hand (m)")
    plt.ylabel("Consumption (c)")
    plt.legend()
    plt.grid(True)

    # 2. Value Function (Decision Perch) (from results_dcsn)
    plt.subplot(2, 2, 2)
    vlu_dcsn_array = results_dcsn["vlu_dcsn"]
    plt.plot(m_grid, vlu_dcsn_array, label="Value Function V(m)")
    plt.title("Value Function (Decision Perch)")
    plt.xlabel("Cash-on-Hand (m)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    # 3. Expected Value Function (Arrival Perch) (from results_arvl)
    plt.subplot(2, 2, 3)
    a_grid = stage.arvl.grid.a
    vlu_arvl_array = results_arvl["vlu_arvl"]
    plt.plot(a_grid, vlu_arvl_array, label="Expected Value Function E[V(a)]")
    plt.title("Expected Value Function (Arrival Perch)")
    plt.xlabel("Assets (a)")
    plt.ylabel("Expected Value")
    plt.legend()
    plt.grid(True)

    # 4. Shock Distribution (from stage model)
    plt.subplot(2, 2, 4)
    shock_info = stage.dcsn_to_arvl.model.num.get("shocks", {}).get("income_shock", {})
    shock_grid = shock_info.get("grid")
    shock_probs = shock_info.get("probabilities")

    if shock_grid is not None and shock_probs is not None:
        bar_width = (
            (shock_grid[1] - shock_grid[0]) * 0.8 if len(shock_grid) > 1 else 0.8
        )
        plt.bar(
            shock_grid, shock_probs, width=bar_width, label="Income Shock Distribution"
        )
    else:
        plt.text(0.5, 0.5, "Shock info not available", horizontalalignment="center")

    plt.title("Income Shock Distribution")
    plt.xlabel("Shock Value (y)")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, f"heptapod_lifecycle_t1_results{filename_suffix}.png"))


def build_lifecycle_t1_stage(config_file, stage=None):
    """Initialize a Stage for the penultimate period (T-1) of a lifecycle model.

    Sets up a lifecycle model stage with proper terminal value functions 
    and initial arrays for continuation perch.

    Parameters
    ----------
    config_file : str
        Path to the configuration YAML file
    stage : Stage, optional
        A pre-configured Stage object to be initialized. If None, a new Stage will be created.
        
    Returns
    -------
    Stage
        The initialized lifecycle stage with proper perch data
    """
    # Handle both function signatures for backward compatibility
    if stage is None:
        # Import here to avoid circular imports
        import sys
        import os
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
        from src.stagecraft import Stage
        
        try:
            from src.heptapod import model_init as heptapod_model_init
            from src.heptapod import gen_num as heptapod_gen_num
        except ImportError:
            # Fall back to old path if needed
            from plugins.heptapod.src.heptapod import model_init as heptapod_model_init
            from plugins.heptapod.src.heptapod import gen_num as heptapod_gen_num
            
        # Create a new Stage
        stage = Stage(
            name="LifecycleT1Stage", 
            init_rep=heptapod_model_init, 
            num_rep=heptapod_gen_num
        )
        stage.load_config(config_file)
        stage.build_computational_model()
        
        # Return the new Stage
        standalone_mode = True
    else:
        # We're just initializing an existing Stage
        standalone_mode = False
    
    # Extract parameters
    parameters = stage.model.parameters_dict
    gamma = parameters["gamma"]

    # Set up initial conditions
    # Fix: Access the a_nxt grid through the main stage model, not the mover model
    a_nxt_grid = stage.cntn.grid.a_nxt

    # Calculate initial vlu_cntn and lambda_cntn arrays on a_nxt_grid
    if gamma == 1:
        vlu_cntn_init = np.log(np.maximum(a_nxt_grid, 1e-9))
    else:
        vlu_cntn_init = (np.maximum(a_nxt_grid, 1e-9) ** (1 - gamma)) / (1 - gamma)

    lambda_cntn_init = np.maximum(a_nxt_grid, 1e-9) ** (-gamma)

    # Attach initial arrays to the continuation perch
    initial_perch_data = {
        "vlu_cntn": vlu_cntn_init,
        "lambda_cntn": lambda_cntn_init,
    }
    stage.perches["cntn"].sol = initial_perch_data
    
    if standalone_mode:
        return stage