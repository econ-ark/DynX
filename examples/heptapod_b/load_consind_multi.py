#!/usr/bin/env python3
"""
Load ConsInd_multi.yml Example

This script loads the ConsInd_multi.yml file and creates a stage problem with
multi-dimensional functions, then generates the numerical model using
generate_numerical_model().

It also demonstrates the new direct attribute access features.
"""

import os
import sys
import numpy as np

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == "__main__":
    from src.heptapod.model_init import load_config, initialize_stage
    from src.heptapod.gen_num import generate_numerical_model
    
    # Load the model
    config_file = os.path.join(os.path.dirname(__file__), 'configs', 'ConsInd_multi.yml')
    config = load_config(config_file)
    problem = initialize_stage(config)
    
    # Generate numerical model
    generate_numerical_model(problem)
    
    # Print multi-dimensional functions
    print(f"Model loaded from {config_file}")
    print(f"Numerical model generated")
    
    # Check numerical state space after generation using new attribute access
    print(f"\n==== NUMERICAL STATE SPACE (attribute access) ====")
    for perch, state_space in problem.num.state_space.items():
        print(f"\nPerch: {perch}")
        for dim, grid in state_space.items():
            try:
                if hasattr(grid, '__len__'):
                    # Handle array/list type grids
                    if len(grid) > 0:
                        if np.issubdtype(type(grid[0]), np.number):
                            print(f"  {dim}: {len(grid)} points, range: [{min(grid):.4f}, {max(grid):.4f}]")
                        else:
                            print(f"  {dim}: {len(grid)} points, values: {grid[:3]}...")
                    else:
                        print(f"  {dim}: empty grid")
                else:
                    # Handle other types
                    print(f"  {dim}: {type(grid).__name__} type")
            except Exception as e:
                print(f"  {dim}: Error displaying grid - {type(e).__name__}")
    
    # Find multi-output functions in v1.6 format using attribute access
    # Multi-output functions have multiple key-value pairs that aren't system keys
    system_keys = ['expr', 'description', 'compilation', 'vector_axis']
    multi_funcs = []
    
    for func_name, func_def in problem.math.functions.items():
        if isinstance(func_def, dict):
            output_keys = [k for k in func_def.keys() if k not in system_keys]
            if len(output_keys) > 1:  # More than one output
                multi_funcs.append((func_name, output_keys))
    
    print(f"\nFound {len(multi_funcs)} multi-dimensional functions:")
    for func_name, outputs in multi_funcs:
        print(f"- {func_name}: {outputs}")
    
    # Test parameter access via param property
    print(f"\n==== DIRECT PARAMETER ACCESS ====")
    print("Available parameters:")
    for param_name in dir(problem.param):
        try:
            value = getattr(problem.param, param_name)
            print(f"  problem.param.{param_name} = {value}")
        except Exception as e:
            print(f"  Error accessing {param_name}: {e}")
    
    # Show available functions using attribute access
    print(f"\nAvailable compiled functions (attribute access):")
    for i, func_name in enumerate(problem.num.functions.keys()):
        if i < 20 or i > len(problem.num.functions) - 5:  # Show first 20 and last 5
            print(f"  problem.num.functions.{func_name}")
        elif i == 20:
            print(f"  ... ({len(problem.num.functions) - 25} more functions) ...")
    
    # Test a multi-dimensional function - util_and_mutil
    print("\nTesting util_and_mutil function with c=1.0 (attribute access):")
    try:
        # Try direct call to multi-output function using attribute access
        if hasattr(problem.num.functions, 'util_and_mutil'):
            result = problem.num.functions.util_and_mutil(c=1.0)
            if isinstance(result, dict):
                for key, value in result.items():
                    print(f"  {key}: {value}")
            else:
                print(f"  Result (not a dict): {result}")
        
        # Try individual output accessors using attribute access
        for output in next((outputs for name, outputs in multi_funcs if name == 'util_and_mutil'), []):
            try:
                output_func_name = f"util_and_mutil_{output}"
                if hasattr(problem.num.functions, output_func_name):
                    value = getattr(problem.num.functions, output_func_name)(c=1.0)
                    print(f"  Direct accessor {output_func_name}: {value}")
            except Exception as sub_e:
                print(f"  Error accessing {output_func_name}: {type(sub_e).__name__}: {sub_e}")
                
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")
    
    # Try using individual keys directly with attribute access
    print("\nTrying direct access to util and mutil functions (attribute access):")
    try:
        if hasattr(problem.num.functions, 'util_and_mutil_util'):
            result = problem.num.functions.util_and_mutil_util(c=2.0)
            print(f"  problem.num.functions.util_and_mutil_util(c=2.0): {result}")
        
        if hasattr(problem.num.functions, 'util_and_mutil_mutil'):
            result = problem.num.functions.util_and_mutil_mutil(c=2.0)
            print(f"  problem.num.functions.util_and_mutil_mutil(c=2.0): {result}")
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")
    
    # Test the standard scalar functions to make sure they work
    print("\nTesting scalar functions (attribute access):")
    try:
        if hasattr(problem.num.functions, 'u_func'):
            result = problem.num.functions.u_func(c=1.0)
            print(f"  problem.num.functions.u_func(c=1.0): {result}")
    except Exception as e:
        print(f"  Error in u_func: {type(e).__name__}: {e}")
        
    # Compare old and new access methods
    print("\n==== COMPARISON OF ACCESS METHODS ====")
    print("Old dictionary access: problem.num['functions']['u_func'](c=1.0)")
    print("New attribute access: problem.num.functions.u_func(c=1.0)")
    
    try:
        # Access same function both ways
        if 'u_func' in problem.num['functions'] and hasattr(problem.num.functions, 'u_func'):
            old_result = problem.num['functions']['u_func'](c=1.0)
            new_result = problem.num.functions.u_func(c=1.0)
            print(f"  Same result: {old_result == new_result} (old={old_result}, new={new_result})")
    except Exception as e:
        print(f"  Error in comparison: {type(e).__name__}: {e}") 