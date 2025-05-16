#!/usr/bin/env python3
"""
Test mover parameter resolution in Heptapod-B.

This example demonstrates how parameters are passed from stage to movers
and how they can be overridden in movers.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to sys.path
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(parent_dir))

from heptapod_b.core import FunctionalProblem
from heptapod_b.init import build_stage, build_mover, build_perch


# Define a test stage model configuration
config = {
    "stage": {
        "parameters": {
            "mean": 1.0,
            "std": 0.1,
            "beta": 0.96,
            "gamma": 2.0,
        },
        "methods": {
            "shock_method": "tauchen",
        },
        "math": {
            "functions": {
                "f": {
                    "expr": "mean + x * std"
                }
            },
            "state": {
                "perch1": {
                    "grid": {
                        "type": "linspace",
                        "min": 0.0,
                        "max": 10.0,
                        "points": 10
                    }
                },
                "perch2": {
                    "grid": {
                        "type": "linspace",
                        "min": 0.0,
                        "max": 10.0,
                        "points": 10
                    }
                }
            }
        },
        "movers": {
            "mover1": {
                "source": "perch1",
                "target": "perch2",
                "inherit_parameters": True,
                "functions": ["f"],
                "operator": {
                    "type": "forward",
                    "method": "direct"
                }
            },
            "mover2": {
                "source": "perch1",
                "target": "perch2",
                "parameters": ["mean", "std"],  # Explicitly include these parameters
                "functions": ["f"],
                "operator": {
                    "type": "forward",
                    "method": "direct"
                }
            },
            "mover3": {
                "source": "perch1",
                "target": "perch2",
                "parameters": ["beta"],  # Only include beta parameter
                "functions": ["f"],
                "operator": {
                    "type": "forward",
                    "method": "direct"
                }
            }
        }
    }
}


def main():
    """Run the mover parameter resolution test."""
    # Build the stage model
    stage_problem = build_stage(config)
    
    # Build the mover models
    mover_problems = build_mover(config, stage_problem)
    
    # Check initial parameters
    print("Initial Parameters:")
    for mover_name, mover_problem in mover_problems.items():
        print(f"  {mover_name}: mean={mover_problem.parameters_dict.get('mean')}, std={mover_problem.parameters_dict.get('std')}")
    
    # Modify parameters in one mover
    print("\nModifying parameters in mover2:")
    mover_problems["mover2"].parameters_dict["mean"] = 2.0  # Change mean from 1.0 to 2.0
    mover_problems["mover2"].parameters_dict["std"] = 0.5   # Change std from 0.1 to 0.5
    
    # Check parameters after modification
    print("\nParameters after modification:")
    for mover_name, mover_problem in mover_problems.items():
        print(f"  {mover_name}: mean={mover_problem.parameters_dict.get('mean')}, std={mover_problem.parameters_dict.get('std')}")
    
    # Now try to evaluate the function in each mover
    print("\nEvaluating functions with modified parameters:")
    
    # This is just a mock evaluation - in a real scenario you would compile the
    # numerical model and use the compiled functions
    for mover_name, mover_problem in mover_problems.items():
        print(f"\n{mover_name}:")
        
        # Get function expression and relevant parameters for evaluation
        func_expr = mover_problem.math["functions"]["f"]["expr"]
        
        # Extract parameter values - preferring mover-specific ones
        expected_mean = mover_problem.parameters_dict.get("mean", "unknown")
        print(f"  Mean parameter: {expected_mean}")
        
        expected_std = mover_problem.parameters_dict.get("std", "unknown")
        print(f"  Std parameter: {expected_std}")
        
        # Show the expression with expected parameter substitution
        print(f"  Expression: {func_expr}")
        if expected_mean != "unknown" and expected_std != "unknown":
            for x_value in [0.0, 1.0, 2.0]:
                expected_result = expected_mean + x_value * expected_std
                print(f"  f({x_value}) expected to be: {expected_result}")


if __name__ == "__main__":
    main() 