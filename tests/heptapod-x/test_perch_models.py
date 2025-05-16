#!/usr/bin/env python3
"""
Test Perch Models using Heptapod-B.

This script demonstrates how perch models are created from a stage model.
"""

import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from heptapod_b.io.yaml_loader import load_config
from heptapod_b.init.stage import build_stage
from heptapod_b.init.perch import build_perch
from heptapod_b.num.generate import compile_num

if __name__ == "__main__":
    # Load the model config
    config_file = os.path.join(os.path.dirname(__file__), '../configs', 'ConsInd_multi.yml')
    print(f"Loading config from {os.path.abspath(config_file)}")
    config = load_config(config_file)
    
    # Build the stage model
    print("\nBuilding stage model...")
    stage_problem = build_stage(config)
    
    # Print information about the stage model first
    print("\n===== Stage Model =====")
    print(f"Parameters: {len(stage_problem.parameters_dict)} parameters")
    print(f"Functions: {len(stage_problem.math['functions'])} functions")
    print(f"Constraints: {len(stage_problem.math['constraints'])} constraints")
    
    # Get all perch names from the math["state_space"] keys
    perches = list(stage_problem.math["state_space"].keys())
    print(f"Perches: {', '.join(perches)}")
    
    # Build perch models
    print("\nBuilding perch models...")
    perch_problems = build_perch(stage_problem)
    print(f"Created {len(perch_problems)} perch models")
    
    # Process each perch model
    for perch_name, perch_problem in perch_problems.items():
        print(f"\n----- Perch: {perch_name} -----")
        
        # Print global parameters inherited
        if hasattr(perch_problem, "parameters_dict"):
            print("Global parameters inherited:")
            params = perch_problem.parameters_dict
            for key in ["beta", "gamma", "r"]:
                if key in params:
                    print(f"  {key}: {params[key]}")
        
        # Print state info copied from stage
        states = list(perch_problem.math["state_space"].keys())
        print(f"States: {', '.join(states)}")
        
        # Print dimensions for this perch
        dimensions = perch_problem.math["state_space"][perch_name].get("dimensions", [])
        print(f"Dimensions: {', '.join(dimensions)}")
        
        # Print settings if any
        if "settings" in perch_problem.math["state_space"][perch_name]:
            settings = perch_problem.math["state_space"][perch_name]["settings"]
            print(f"Settings: {settings}")
        
        # Print grid info if any
        if "grid" in perch_problem.math["state_space"][perch_name]:
            grid_spec = perch_problem.math["state_space"][perch_name]["grid"]
            print(f"Grid specification: {grid_spec}")
    
    # Compile each perch model
    print("\nCompiling perch models...")
    compiled_perches = {}
    for perch_name, perch_problem in perch_problems.items():
        print(f"Compiling {perch_name}...")
        try:
            compiled_perch = compile_num(perch_problem)
            compiled_perches[perch_name] = compiled_perch
            print(f"  Success!")
        except Exception as e:
            print(f"  Error: {type(e).__name__}: {str(e)}")
    
    # Test parameter resolution in compiled perches
    print("\nTesting parameter resolution in compiled perches:")
    for perch_name, perch_problem in compiled_perches.items():
        print(f"\n=== {perch_name} ===")
        
        # Check numerical state space
        if hasattr(perch_problem, "num") and "state_space" in perch_problem.num:
            perch_state_space = perch_problem.num["state_space"]
            
            for state_name, state_data in perch_state_space.items():
                print(f"State: {state_name}")
                
                # Print dimensions
                if "dimensions" in state_data:
                    print(f"  Dimensions: {state_data['dimensions']}")
                
                # Print grids
                if "grids" in state_data:
                    for dim_name, grid in state_data["grids"].items():
                        if hasattr(grid, "__len__"):
                            try:
                                print(f"  Grid '{dim_name}': {len(grid)} points, range: [{min(grid):.6f} - {max(grid):.6f}]")
                            except Exception as e:
                                print(f"  Grid '{dim_name}': Error displaying ({type(e).__name__})")
                
                # Special case for dcsn perch - check if m_min reference was resolved
                if perch_name == "dcsn" and state_name == "dcsn" and "grids" in state_data:
                    if "m" in state_data["grids"]:
                        m_grid = state_data["grids"]["m"]
                        m_min_actual = min(m_grid)
                        m_min_expected = 1E-500  # From the YAML config
                        
                        print(f"  m_min reference resolution:")
                        print(f"    Expected: {m_min_expected}")
                        print(f"    Actual: {m_min_actual}")
                        
                        # For very small numbers, just check if both are very close to zero
                        if m_min_actual < 1E-100 and m_min_expected < 1E-100:
                            print("    ✓ m_min reference correctly resolved (both very small)")
                        else:
                            print("    ✗ m_min reference resolution failed")
        else:
            print("  No numerical state space compiled")
    
    print("\nPerch testing complete!") 