#!/usr/bin/env python3
"""
Simple Movers Example using Heptapod-B.

This script demonstrates how movers are constructed from a stage model.
"""

import os
import sys
import yaml
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from heptapod_b.init.stage import build_stage
from heptapod_b.init.mover import build_mover
from heptapod_b.num.generate import compile_num

# Define a simple model configuration inline
CONFIG = """
stage:
  name: "SimpleSavings"
  parameters:
    beta: 0.9
    gamma: 2.0
    r: 1.05
  
  methods:
    default_grid: 'linspace'
    compilation: 'eval'
  
  math:
    functions:
      # Simple utility function
      u_func:
        expr: "c**(1-gamma)/(1-gamma)"
        description: "CRRA utility"
      
      # Marginal utility
      mu_func:
        expr: "c**(-gamma)"
        description: "Marginal utility"
      
      # Budget constraint function (safe to call without extra params)
      budget:
        expr: "a + y - c - a_next"
        description: "Budget constraint"
      
      # Simple transition
      transition:
        expr: "a_next = (1+r)*a"
        description: "Asset transition"
    
    state_space:
      state1:
        description: "Initial state"
        dimensions: ["a"]
        grid:
          a: [0.1, 0.5, 1.0, 5.0, 10.0]
      
      state2:
        description: "Terminal state"
        dimensions: ["a_next"]
        grid:
          a_next: [0.1, 0.5, 1.0, 5.0, 10.0]

movers:
  forward_mover:
    type: "forward"
    source: "state1"
    target: "state2"
    functions:
      - u_func
      - transition
    operator:
      method: simulation
    inherit_parameters: true
    inherit_settings: true
  
  backward_mover:
    type: "backward"
    source: "state2"
    target: "state1"
    functions:
      - mu_func
      - budget
    operator:
      method: integration
    inherit_parameters: true
    inherit_settings: true
"""

if __name__ == "__main__":
    # Load the model config from the inline YAML
    config = yaml.safe_load(CONFIG)
    
    print("Building stage model...")
    stage_problem = build_stage(config)
    
    # Build mover models
    print("\nBuilding mover models...")
    mover_problems = build_mover(config, stage_problem)
    
    # Print information about each mover
    for mover_name, mover_problem in mover_problems.items():
        print(f"\n=== Mover: {mover_name} ===")
        
        # Print problem info
        print(f"Mover type: {mover_name}")
        print(f"Operator type: {mover_problem.operator.get('type', 'unknown')}")
        print(f"Parameters: {', '.join(f'{k}={v}' for k, v in mover_problem.parameters_dict.items())}")
        
        # Print states (source and target)
        states = list(mover_problem.math["state_space"].keys())
        print(f"States: {', '.join(states)}")
        
        # Print functions
        functions = list(mover_problem.math["functions"].keys())
        print(f"Functions: {', '.join(functions)}")
    
    # Compile each mover model separately
    print("\nCompiling mover models...")
    compiled_movers = {}
    for mover_name, mover_problem in mover_problems.items():
        print(f"Compiling {mover_name}...")
        try:
            # Compile the mover
            compiled_problem = compile_num(mover_problem)
            compiled_movers[mover_name] = compiled_problem
            print(f"  Success!")
        except Exception as e:
            print(f"  Error compiling {mover_name}: {type(e).__name__}: {e}")
    
    # Test utility functions in each mover
    print("\nTesting functions in movers...")
    for mover_name, mover_problem in compiled_movers.items():
        print(f"\n=== Testing {mover_name} ===")
        
        if hasattr(mover_problem, "num") and "functions" in mover_problem.num:
            functions = list(mover_problem.num["functions"].keys())
            print(f"Compiled functions: {', '.join(functions)}")
            
            # Test u_func in forward_mover
            if mover_name == "forward_mover" and "u_func" in functions:
                result = mover_problem.num["functions"]["u_func"](c=2.0)
                print(f"  u_func(c=2.0) = {result}")
            
            # Test mu_func in backward_mover
            if mover_name == "backward_mover" and "mu_func" in functions:
                result = mover_problem.num["functions"]["mu_func"](c=2.0)
                print(f"  mu_func(c=2.0) = {result}")
                
            # Test budget function
            if "budget" in functions:
                result = mover_problem.num["functions"]["budget"](a=1.0, y=0.5, c=0.3, a_next=1.2)
                print(f"  budget(a=1.0, y=0.5, c=0.3, a_next=1.2) = {result}")
        else:
            print("  No compiled functions found")
    
    print("\nMover model testing complete!") 