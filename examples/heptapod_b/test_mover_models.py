#!/usr/bin/env python3
"""
Test Mover Models using Heptapod-B.

This script demonstrates how mover models are created from a stage model.
"""

import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from heptapod_b.io.yaml_loader import load_config
from heptapod_b.init.stage import build_stage
from heptapod_b.init.mover import build_mover
from heptapod_b.num.generate import compile_num

if __name__ == "__main__":
    # Load the model config
    config_file = os.path.join(os.path.dirname(__file__), '../configs', 'ConsInd_multi.yml')
    print(f"Loading config from {os.path.abspath(config_file)}")
    config = load_config(config_file)
    
    # Build the stage model
    print("Building stage model...")
    stage_problem = build_stage(config)
    
    # Build mover models
    print("\nBuilding mover models...")
    mover_problems = build_mover(config, stage_problem)
    
    # Print information about each mover
    for mover_name, mover_problem in mover_problems.items():
        print(f"\n=== Mover: {mover_name} ===")
        
        # Print mover type and operator
        mover_type = mover_problem.operator.get("method") if hasattr(mover_problem, "operator") else "unknown"
        print(f"Type: {mover_type}")
        
        # Print states in this mover
        states = list(mover_problem.math["state_space"].keys())
        print(f"States: {', '.join(states)}")
        
        # Print functions
        functions = list(mover_problem.math["functions"].keys())
        print(f"Functions: {', '.join(functions[:3])}{'...' if len(functions) > 3 else ''}")
        
        # Check for parameters
        has_params = hasattr(mover_problem, "parameters_dict") and len(mover_problem.parameters_dict) > 0
        print(f"Parameters inherited: {has_params}")
        
        # Check if settings were inherited
        has_settings = hasattr(mover_problem, "settings") and len(mover_problem.settings) > 0
        print(f"Settings inherited: {has_settings}")
    
    # Compile each mover model separately
    print("\nCompiling mover models...")
    compiled_movers = {}
    for mover_name, mover_problem in mover_problems.items():
        print(f"Compiling {mover_name}...")
        try:
            # Create a copy to avoid modifying the original
            compiled_problem = compile_num(mover_problem)
            compiled_movers[mover_name] = compiled_problem
            print(f"  Success!")
        except Exception as e:
            print(f"  Error compiling {mover_name}: {type(e).__name__}: {e}")
    
    # Test compiled functions in each mover
    print("\nTesting compiled functions in movers...")
    for mover_name, mover_problem in compiled_movers.items():
        print(f"\n=== Testing {mover_name} ===")
        
        # Get the list of compiled functions
        if hasattr(mover_problem, "num") and "functions" in mover_problem.num:
            functions = list(mover_problem.num["functions"].keys())
            print(f"Compiled functions: {len(functions)}")
            
            # Try to call a function (if any exist)
            if functions:
                # Find a simple function to test (preferably u_func if it exists)
                test_func = None
                if "u_func" in mover_problem.num["functions"]:
                    test_func = "u_func"
                elif "uc_func" in mover_problem.num["functions"]:
                    test_func = "uc_func"
                else:
                    test_func = functions[0]
                
                try:
                    # Call the function with c=1.0 (common parameter for utility functions)
                    result = mover_problem.num["functions"][test_func](c=1.0)
                    print(f"  Called {test_func}(c=1.0) = {result}")
                except Exception as e:
                    print(f"  Error calling {test_func}: {type(e).__name__}: {e}")
            else:
                print("  No functions to test")
        else:
            print("  No compiled functions found")
    
    print("\nMover model testing complete!") 