"""
Housing model with master parameter configuration.

This example demonstrates how to load and initialize a model stage using
a configuration file that references a master parameter file.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the src directory to the path if running the script directly
script_dir = Path(__file__).parent.resolve()
repo_root = script_dir.parent.parent
sys.path.append(str(repo_root))

from src.heptapod_b.init.stage import build_stage
from src.heptapod_b.io.yaml_loader import load_config

def load_housing_model():
    """
    Load the housing model stage using master parameter configuration.
    """
    # Define paths
    configs_dir = repo_root / "examples" / "configs"
    stage_file = configs_dir / "OWNC_stage_with_master.yml"
    master_file = configs_dir / "housing_renting_master.yml"
    
    print(f"Loading housing model with master parameter configuration...")
    print(f"Stage file: {stage_file}")
    print(f"Master file: {master_file}")
    
    # Verify files exist
    if not stage_file.exists():
        raise FileNotFoundError(f"Stage file not found: {stage_file}")
    if not master_file.exists():
        raise FileNotFoundError(f"Master file not found: {master_file}")
    
    # Load the master config for reference
    master_config = load_config(str(master_file))
    print("\nMaster configuration contains:")
    print(f"  Parameters: {len(master_config['parameters'])} parameters")
    print(f"  Settings: {len(master_config['settings'])} settings")
    
    # Load the stage config for comparison
    stage_config = load_config(str(stage_file))
    print("\nStage configuration contains:")
    print(f"  Parameters: {len(stage_config['stage']['parameters'])} parameters")
    print(f"  Settings: {len(stage_config['stage']['settings'])} settings")
    
    # Highlight the parameter override
    master_r = master_config['parameters']['r']
    stage_r = stage_config['stage']['parameters']['r']
    print(f"\nParameter override example:")
    print(f"  'r' in master file: {master_r}")
    print(f"  'r' in stage file: {stage_r}")
    
    # Build the stage model (which automatically resolves parameter references)
    print("\nBuilding stage model...")
    stage = build_stage(str(stage_file))
    
    # Print summary of stage model
    print("\nStage model summary:")
    print(f"  Name: {stage.name if hasattr(stage, 'name') else 'Unnamed'}")
    print(f"  Parameters: {len(stage.parameters_dict)} parameters")
    print(f"  Settings: {len(stage.settings)} settings")
    
    # Verify that critical parameters are correct
    print("\nVerifying parameters:")
    
    # Parameters that should come from master
    for param_name in ['beta', 'alpha', 'gamma']:
        master_value = master_config['parameters'][param_name]
        stage_value = stage.parameters_dict[param_name] 
        print(f"  {param_name}: {stage_value} (master: {master_value})")
        assert stage_value == master_value, f"Parameter '{param_name}' does not match master value"
    
    # Parameter that's overridden in the stage
    master_value = master_config['parameters']['r']
    stage_value = stage.parameters_dict['r']
    expected_value = stage_config['stage']['parameters']['r']
    print(f"  r: {stage_value} (master: {master_value}, expected: {expected_value})")
    assert stage_value == expected_value, "Parameter 'r' does not match stage override value"
    assert stage_value != master_value, "Parameter 'r' incorrectly used master value instead of stage override"
    
    # Check math.parameters section which uses parameter references
    if 'parameters' in stage.math:
        print("\nChecking math.parameters section:")
        for param_name, param_value in stage.math['parameters'].items():
            print(f"  {param_name}: {param_value}")
        
        # Verify that interest_rate resolves to the stage value of r
        assert stage.math['parameters']['interest_rate'] == stage_config['stage']['parameters']['r'], \
            "math.parameters.interest_rate doesn't match stage r value"
    
    # Check state space dimensions and grid sizes
    print("\nState space dimensions:")
    for perch_name, perch_info in stage.math.state_space.items():
        print(f"  {perch_name}: {perch_info['dimensions']}")
        
        # Check grid settings against master file
        if 'grid' in perch_info:
            print(f"  Grid settings for {perch_name}:")
            for dim_name, grid_spec in perch_info['grid'].items():
                if 'points' in grid_spec:
                    print(f"    {dim_name}: {grid_spec['points']} points")
    
    # Create a simple visualization of parameter relationships
    plt.figure(figsize=(10, 6))
    
    # Parameters to visualize
    params = ['beta', 'r', 'alpha', 'gamma', 'iota', 'kappa']
    values = [stage.parameters_dict[p] for p in params]
    
    # Create bar chart
    bars = plt.bar(params, values, color=['skyblue' if p != 'r' else 'orange' for p in params])
    plt.title('Housing Model Parameters (With Stage Override for r)')
    plt.xlabel('Parameter')
    plt.ylabel('Value')
    plt.ylim(0, max(values) * 1.2)
    
    # Add value labels on top of bars
    for i, v in enumerate(values):
        plt.text(i, v + 0.05, f"{v:.3f}", ha='center')
    
    # Add a legend for overridden parameters
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='skyblue', label='From Master'),
        Patch(facecolor='orange', label='Stage Override')
    ]
    plt.legend(handles=legend_elements)
    
    # Save the figure
    output_file = script_dir / "housing_master_params.png"
    plt.savefig(output_file)
    print(f"\nParameter visualization saved to: {output_file}")
    
    return stage

if __name__ == "__main__":
    stage = load_housing_model()
    print("\nHousing model stage loaded successfully!")
    print("The model is initialized and ready for use in simulations or optimization.")
    print("All parameters were correctly resolved, with stage overrides taking precedence.") 