#!/usr/bin/env python3
"""
Model Overview Example

This script demonstrates how to load a model from a configuration file
and provides a comprehensive overview of all models generated in a stage:
- Stage model
- Perch models (arvl, dcsn, cntn)
- Mover models (arvl_to_dcsn, dcsn_to_cntn, cntn_to_dcsn, dcsn_to_arvl)

The script shows:
1. The initialization of the models from configuration using `init_rep`.
2. The numerical compilation of the models using `num_rep`.
3. How to access different components of each model.
4. How to check the `status_flags` dictionary.
"""

import os
import sys
import numpy as np
import pprint

# Add the modcraft root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import required modules
from src.stagecraft import Stage
# Import the specific representation modules
from plugins.heptapod.src.heptapod import model_init, gen_num

def print_section(title):
    """Print a section header to make output more readable."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)

def print_dict_summary(d, max_items=5, name="", indent=0):
    """Print a summarized view of a dictionary, skipping empty collections."""
    indent_str = " " * indent
    if not isinstance(d, dict):
        print(f"{indent_str}{name}: {type(d).__name__}")
        return
        
    # Filter out empty collections first
    non_empty_items = {k: v for k, v in d.items() if not (isinstance(v, (dict, list, set)) and not v)}
    
    if not non_empty_items:
        print(f"{indent_str}{name}: {{}} (contains only empty collections or is empty)")
        return
        
    print(f"{indent_str}{name}:")
    items_shown_count = 0
    for key, value in non_empty_items.items():
        if items_shown_count >= max_items:
            print(f"{indent_str}  ... ({len(non_empty_items) - max_items} more non-empty items)")
            break
            
        if isinstance(value, dict):
             # Recurse slightly for nested dicts if needed, or just show count
             print(f"{indent_str}  {key}: (dict with {len(value)} items)") # Could add recursive call here if desired
        elif isinstance(value, list):
            print(f"{indent_str}  {key}: (list with {len(value)} items)")
        elif isinstance(value, np.ndarray):
            print(f"{indent_str}  {key}: (array with shape {value.shape} and dtype {value.dtype})")
        elif callable(value):
            print(f"{indent_str}  {key}: (callable)")
        else:
            print(f"{indent_str}  {key}: {type(value).__name__}") # Show type for clarity
            
        items_shown_count += 1


def print_model_summary(model, name="Model"):
    """Print a summary of a model's structure and contents."""
    print(f"\n--- {name} Summary ---")
    
    if model is None:
        print("(Model is None)")
        return

    # Helper to safely get length or return 0
    def safe_len(obj):
        if hasattr(obj, '__len__'):
            return len(obj)
        return 0

    # Print basic information, checking attribute existence
    print(f"Parameters: {safe_len(getattr(model, 'parameters', None))} items")
    print(f"Settings: {safe_len(getattr(model, 'settings', None))} items")
    print(f"Methods: {safe_len(getattr(model, 'methods', None))} items")
    
    # Print mathematical components
    math = getattr(model, 'math', None)
    if isinstance(math, dict) and math:
        print("\nMathematical Components:")
        print_dict_summary(math, name="  Math Dict", indent=2)
    
    # Print numerical components if available
    num = getattr(model, 'num', None)
    if isinstance(num, dict) and num:
        print("\nNumerical Components:")
        print_dict_summary(num, name="  Num Dict", indent=2)
        
        # Example: Access specific numerical parts like state space grids
        state_space = num.get('state_space', {})
        if state_space:
             print("    Example Grid Info:")
             for space_name, space_vars in state_space.items():
                  if isinstance(space_vars, dict):
                       for var_name, grid_array in space_vars.items():
                            if isinstance(grid_array, np.ndarray):
                                 print(f"      - {space_name}.{var_name}: {len(grid_array)} points [{grid_array[0]:.2f} to {grid_array[-1]:.2f}]")
    
    # Print operator information for movers
    operator = getattr(model, 'operator', None)
    if isinstance(operator, dict) and operator:
        print("\nOperator:")
        print_dict_summary(operator, name="  Details", indent=2)

def main():
    """Main function to demonstrate model initialization and overview."""
    # Get the path to the config file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, 'example_config.yml')
    
    print_section("MODEL INITIALIZATION")
    print(f"Loading configuration from: {config_file}")
    
    # Step 1: Create a Stage with init_rep and num_rep
    stage = Stage(name="ModOverview", init_rep=model_init, num_rep=gen_num)
    
    # Step 2: Load the configuration file
    stage.load_config(config_file)
    print("\nStatus flags after initialization:")
    pprint.pprint(stage.status_flags)
    
    # Display basic information about the stage
    print(f"\nStage name: {stage.name}")
    print(f"Is portable: {stage.status_flags.get('portable', False)}") # Access via status_flags
    
    # Print model structure overview
    print_section("STAGE MODEL OVERVIEW (Initial)")
    stage.print_model_structure() # Use the built-in detailed print method
    
    # Print individual model summaries using custom helper
    print_section("INDIVIDUAL MODEL SUMMARIES (Initial)")
    print_model_summary(stage.model, "Stage Model")
    for perch_name in ["arvl", "dcsn", "cntn"]:
        perch = getattr(stage, perch_name)
        if perch: print_model_summary(perch.model, f"{perch_name.capitalize()} Perch Model")
    
    # Use direct property access for movers
    movers_to_check = {"arvl_to_dcsn": stage.arvl_to_dcsn, 
                       "dcsn_to_cntn": stage.dcsn_to_cntn,
                       "cntn_to_arvl": stage.cntn_to_arvl, # Note: This mover exists but might not have a model from heptapod config
                       "dcsn_to_arvl": stage.dcsn_to_arvl, 
                       "cntn_to_dcsn": stage.cntn_to_dcsn}
                       
    for mover_name, mover in movers_to_check.items():
        if mover: print_model_summary(mover.model, f"{mover_name} Mover Model")
    
    # Step 3: Numerical compilation
    print_section("NUMERICAL COMPILATION")
    
    # Remove the deprecated model_rep assignment
    # stage.model_rep = gen_num 
    
    # Build the computational model using num_rep provided at init
    success = stage.build_computational_model()
    print(f"Compilation successful: {success}")
    print("\nStatus flags after compilation:")
    pprint.pprint(stage.status_flags)
    
    # Show numerical components using custom helper
    print_section("INDIVIDUAL MODEL SUMMARIES (Compiled)")
    print_model_summary(stage.model, "Compiled Stage Model")
    for perch_name in ["arvl", "dcsn", "cntn"]:
        perch = getattr(stage, perch_name)
        if perch: print_model_summary(perch.model, f"Compiled {perch_name.capitalize()} Perch Model")
        
    for mover_name, mover in movers_to_check.items():
         if mover: print_model_summary(mover.model, f"Compiled {mover_name} Mover Model")
         
    # Show compiled model structure using built-in method
    print_section("STAGE MODEL OVERVIEW (Compiled)")
    stage.print_model_structure() 
    
    # Show access examples
    print_section("ACCESS EXAMPLES")
    
    # Example 1: Access stage parameters
    print("Stage parameters access:")
    if hasattr(stage.model, 'parameters_dict') and stage.model.parameters_dict:
        print("\nParameters:")
        for param_name in sorted(stage.model.parameters_dict.keys()):
            if param_name in stage.model.parameters_dict:
                print(f"  {param_name} = {stage.model.parameters_dict[param_name]}")
    else:
         print("  No parameters found in stage model.")

    # Example 2: Access state space grids
    print("\nState space grids access:")
    if hasattr(stage.model, 'num') and stage.model.num:
        state_space = stage.model.num.get('state_space', {})
        if state_space:
            for perch_name, grids in state_space.items():
                print(f"  {perch_name} grids:")
                for grid_name, grid in grids.get('grids', {}).items():
                    if isinstance(grid, np.ndarray):
                        print(f"    {grid_name}: {len(grid)} points - Range: [{grid[0]:.2f}, {grid[-1]:.2f}]")
    
    # Example 3: Check if a stage is solvable based on having sol (formerly up) and dist (formerly down) data
    print("\nStage solvability check:")
    
    # Set some example data for demonstration
    cntn_sol_data = {"vlu_cntn": np.array([1.0, 2.0, 3.0]), "lambda_cntn": np.array([0.1, 0.2, 0.3])}
    arvl_dist_data = np.array([0.2, 0.5, 0.3])  # Some distribution over states
    
    # Set data to perches
    stage.perches["cntn"].sol = cntn_sol_data  # Setting cntn.sol (formerly cntn.up)
    stage.perches["arvl"].dist = arvl_dist_data  # Setting arvl.dist (formerly arvl.down)
    
    # Check if the stage is now solvable
    stage._check_solvability()
    print(f"  Stage is solvable: {stage.status_flags['solvable']}")
    print(f"  cntn.sol is set: {stage.perches['cntn'].sol is not None}")  # Check cntn.sol
    print(f"  arvl.dist is set: {stage.perches['arvl'].dist is not None}")  # Check arvl.dist
    
    # Demonstrate access to perch data
    print("\nPerch data access:")
    print(f"  cntn.sol type: {type(stage.perches['cntn'].sol)}")
    if isinstance(stage.perches['cntn'].sol, dict):
        for key, value in stage.perches['cntn'].sol.items():
            if isinstance(value, np.ndarray):
                print(f"    {key}: array shape {value.shape}, first value {value[0]}")
    
    print(f"  arvl.dist type: {type(stage.perches['arvl'].dist)}")
    if isinstance(stage.perches['arvl'].dist, np.ndarray):
        print(f"    distribution shape: {stage.perches['arvl'].dist.shape}")
    
    print_section("EXAMPLE COMPLETE")
    print("You have now seen how to initialize, compile, and access a stage model.")

if __name__ == "__main__":
    main() 