#!/usr/bin/env python
"""
Simple test script demonstrating model, period, and stage access in the housing model.
This script loads the housing model and shows how to access various components.
"""

import os
import sys
import numpy as np

# Add modcraft root to path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, repo_root)

# Import from ModCraft
from src.stagecraft import Stage
from dynx.stagecraft.makemod import initialize_model_Circuit
from src.heptapod_b.io.yaml_loader import load_config
from src.heptapod_b.num.generate import compile_num as generate_numerical_model

def load_configs():
    """Load all configuration files for the housing model."""
    config_dir = os.path.join(repo_root, "examples/economic_models/housing/config")
    master_path = os.path.join(config_dir, "housing_master.yml")
    ownh_path = os.path.join(config_dir, "OWNH_stage.yml") 
    ownc_path = os.path.join(config_dir, "OWNC_stage.yml")
    connections_path = os.path.join(config_dir, "connections.yml")
    
    # Load configurations
    print("Loading configurations...")
    master_config = load_config(master_path)
    ownh_config = load_config(ownh_path)
    ownc_config = load_config(ownc_path)
    connections_config = load_config(connections_path)
    
    return {
        "master": master_config,
        "ownh": ownh_config,
        "ownc": ownc_config,
        "connections": connections_config
    }

def setup_housing_model():
    """Initialize the housing model and return the model circuit."""
    # Load configurations
    configs = load_configs()
    
    # Prepare stages config dictionary
    stage_configs = {
        "OWNH": configs["ownh"],
        "OWNC": configs["ownc"]
    }
    
    # Build the model circuit
    print("Building model circuit using configuration...")
    model_circuit = initialize_model_Circuit(
        master_config=configs["master"],
        stage_configs=stage_configs,
        connections_config=configs["connections"]
    )
    
    # Set numerical representation for stages
    # This is needed for building the computational model
    period_index = 0  # Use the first period
    for stage_name in model_circuit.periods_list[period_index].stages:
        model_circuit.periods_list[period_index].stages[stage_name].num_rep = generate_numerical_model
    
    return model_circuit

def demonstrate_access_patterns(model_circuit):
    """Demonstrate various access patterns for model components."""
    print("\n--- MODEL CIRCUIT ACCESS ---")
    print(f"Number of periods: {len(model_circuit.periods_list)}")
    
    # Access a specific period
    period = model_circuit.get_period(0)
    print(f"\n--- PERIOD ACCESS ---")
    print(f"Period index: 0")
    print(f"Stages in period: {list(period.stages.keys())}")
    
    # Access a specific stage
    ownh_stage = period.get_stage("OWNH")
    ownc_stage = period.get_stage("OWNC")
    print(f"\n--- STAGE ACCESS ---")
    print(f"OWNH stage name: {ownh_stage.name}")
    print(f"OWNH status flags: {ownh_stage.status_flags}")
    
    # Build computational model for OWNH stage
    print(f"\n--- BUILDING COMPUTATIONAL MODEL FOR OWNH ---")
    ownh_stage.build_computational_model()
    print(f"OWNH compiled: {ownh_stage.status_flags.get('compiled', False)}")
    
    # Access perches and movers
    print(f"\n--- PERCH ACCESS ---")
    print(f"Perches in OWNH: {list(ownh_stage.perches.keys())}")
    print(f"OWNH arvl model exists: {ownh_stage.arvl.model is not None}")
    
    print(f"\n--- MOVER ACCESS ---")
    print(f"Movers in OWNH: {len(ownh_stage.forward_movers) + len(ownh_stage.backward_movers)}")
    print(f"OWNH arvl_to_dcsn exists: {ownh_stage.arvl_to_dcsn is not None}")
    
    # Access model parameters and settings
    print(f"\n--- MODEL ATTRIBUTE ACCESS ---")
    model = ownh_stage.model
    print(f"OWNH parameters count: {len(model.parameters_dict)}")
    # Access a parameter both ways
    if 'r' in model.parameters_dict:
        print(f"Parameter r via dict: {model.parameters_dict['r']}")
        print(f"Parameter r via property: {model.param.r}")
    
    # Access numerical components
    print(f"\n--- NUMERICAL COMPONENT ACCESS ---")
    if hasattr(model.num, 'state_space'):
        print(f"State space keys: {model.num.state_space.keys()}")
        
        # Access grid data using direct path
        if 'arvl' in model.num.state_space:
            arvl_state_space = model.num.state_space['arvl']
            if 'grids' in arvl_state_space:
                print(f"OWNH arvl grid variables: {arvl_state_space['grids'].keys()}")
                if 'a' in arvl_state_space['grids']:
                    a_grid = arvl_state_space['grids']['a']
                    print(f"Asset grid shape: {a_grid.shape}, min: {a_grid.min()}, max: {a_grid.max()}")
    
    # Access grid data using proxy (if implemented)
    print(f"\n--- GRID PROXY ACCESS ---")
    if hasattr(ownh_stage.arvl, 'grid'):
        print(f"Grid proxy exists on arvl perch")
        if hasattr(ownh_stage.arvl.grid, 'a'):
            a_grid = ownh_stage.arvl.grid.a
            print(f"Asset grid via proxy - shape: {a_grid.shape}, min: {a_grid.min()}, max: {a_grid.max()}")
    
    # Access shock data
    print(f"\n--- SHOCK ACCESS ---")
    if hasattr(model.num, 'shocks'):
        print(f"Shocks: {model.num.shocks.keys()}")
        if 'income_shock' in model.num.shocks:
            shock = model.num.shocks['income_shock']
            print(f"Income shock values: {shock['values']}")
            if 'transition_matrix' in shock:
                print(f"Transition matrix shape: {shock['transition_matrix'].shape}")
    
    return {
        "ownh_stage": ownh_stage,
        "ownc_stage": ownc_stage
    }

def main():
    """Main function to run the access test."""
    print("=== HOUSING MODEL ACCESS TEST ===")
    
    # Setup the housing model
    model_circuit = setup_housing_model()
    
    # Demonstrate access patterns
    stages = demonstrate_access_patterns(model_circuit)
    
    print("\n=== TEST COMPLETED SUCCESSFULLY ===")
    return model_circuit, stages

if __name__ == "__main__":
    model_circuit, stages = main() 