#!/usr/bin/env python3
"""
Load ConsInd_multi.yml Example using Heptapod-B

This script loads the ConsInd_multi.yml file and creates a stage problem with
multi-dimensional functions, then generates the numerical model using
the new Heptapod-B package.

It also demonstrates the direct attribute access features.
"""

import os
import sys
import numpy as np

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

if __name__ == "__main__":
    # Import from the new heptapod_b package
    from src.heptapod_b.io.yaml_loader import load_config
    from src.heptapod_b.init.stage import build_stage
    from src.heptapod_b.num.generate import compile_num
    
    # Load the model
    config_file = os.path.join(os.path.dirname(__file__), '../configs', 'ConsInd_multi.yml')
    config = load_config(config_file)
    problem = build_stage(config)
    
    # Generate numerical model
    compile_num(problem)

    # Testing parameter access methods
    print("\n==== PARAMETER ACCESS METHODS ====")
    
    # 1. Using the param accessor property (recommended)
    print("Using param accessor property (recommended):")
    print(f"  problem.param.beta = {problem.param.beta}")
    print(f"  problem.param.gamma = {problem.param.gamma}")
    
    # 2. Using direct parameters_dict access
    print("\nUsing direct parameters_dict access:")
    print(f"  problem.parameters_dict['beta'] = {problem.parameters_dict['beta']}")
    print(f"  problem.parameters_dict['gamma'] = {problem.parameters_dict['gamma']}")
    
    # 3. Verify both methods return the same values
    print("\nVerifying consistency between access methods:")
    print(f"  Same beta: {problem.param.beta == problem.parameters_dict['beta']}")
    print(f"  Same gamma: {problem.param.gamma == problem.parameters_dict['gamma']}")
    
    # Print multi-dimensional functions
    print(f"\nModel loaded from {config_file}")
    print(f"Numerical model generated")

    # Check numerical state space after generation using new attribute access
    print(f"\n==== NUMERICAL STATE SPACE (attribute access) ====")
    
    # Now display each perch's state space in detail
    for perch, state_space in problem.num.state_space.items():
        print(f"\nPerch: {perch}")
        
        # Show dimensions
        if "dimensions" in state_space:
            print(f"  Dimensions: {state_space['dimensions']}")
        
        # Show grids (if available)
        if "grids" in state_space:
            print(f"  Grids:")
            for grid_name, grid_values in state_space["grids"].items():
                if hasattr(grid_values, '__len__') and len(grid_values) > 0:
                    print(f"    {grid_name}: {len(grid_values)} points, range: [{min(grid_values):.4f}, {max(grid_values):.4f}]")
        
        # Show mesh (if available)
        if "mesh" in state_space:
            print(f"  Mesh grid available with dimensions: {', '.join(state_space['mesh'].keys())}")
            
        # Show tensor (if available)
        if "tensor" in state_space:
            tensor = state_space["tensor"]
            if hasattr(tensor, 'shape'):
                print(f"  Tensor grid with shape: {tensor.shape}")
            elif hasattr(tensor, '__len__'):
                print(f"  Tensor grid with {len(tensor)} points")
                
        # Show any other attributes
        for key in state_space.keys():
            if key not in ["dimensions", "grids", "mesh", "tensor"]:
                print(f"  Other attribute: {key} ({type(state_space[key]).__name__})")
        
        print("------------------")
    
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
    
    # Check for function_variables
    print("\n==== CHECKING FOR FUNCTION_VARIABLES ====")
    if hasattr(problem.num, 'function_variables'):
        print("  problem.num.function_variables exists!")
        print(f"  Type: {type(problem.num.function_variables)}")
        
        if isinstance(problem.num.function_variables, dict):
            print(f"  Keys: {list(problem.num.function_variables.keys())}")
            # Print a sample of the content if it's a dictionary
            if problem.num.function_variables:
                sample_key = next(iter(problem.num.function_variables))
                print(f"  Sample entry ({sample_key}): {problem.num.function_variables[sample_key]}")
    else:
        print("  problem.num.function_variables does not exist")
        print("  Available attributes in problem.num:")
        for attr in dir(problem.num):
            if not attr.startswith('_'):
                print(f"    - {attr}")

    # Test container-based attribute access
    print("\n==== TESTING CONTAINER-BASED ATTRIBUTE ACCESS ====")
    print("1. Accessing functions via problem.num.functions.function_name:")
    try:
        # Try accessing a simple utility function
        if hasattr(problem.num.functions, 'u_func'):
            result = problem.num.functions.u_func(c=2.0)
            print(f"  problem.num.functions.u_func(c=2.0) = {result}")
            
            # Also try the dictionary style access for comparison
            dict_result = problem.num['functions']['u_func'](c=2.0)
            print(f"  problem.num['functions']['u_func'](c=2.0) = {dict_result}")
            print(f"  Results match: {result == dict_result}")
    except Exception as e:
        print(f"  Error accessing function: {type(e).__name__}: {e}") 