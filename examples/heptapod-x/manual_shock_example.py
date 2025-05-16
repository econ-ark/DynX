"""
Example demonstrating the use of manual shock processes in Heptapod-B.
"""

import yaml
import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt

# Use the appropriate API import
try:
    from heptapod_b.core.api import initialize_model
    # Import compile_num from the appropriate location
    from heptapod_b.num.generate import compile_num
except ImportError:
    # Fallback to older import path if needed
    from heptapod.model_init import initialize_model
    from heptapod.gen_num import generate_numerical_model as compile_num

# Configuration YAML with manual shock processes
CONFIG = """
stage:
  name: SimpleManualShockExample
  parameters:
    beta: 0.96
    gamma: 2.0
  settings:
    example_setting: 123
  state_space:
    a: {grid_type: linspace, n_points: 10, min: 0, max: 100}
    z: {shock: income_shock}  # Link state variable to shock name
  
  # Important: Shocks should be in the math section
  math:
    shocks:
      income_shock:
        description: "Manually specified income shock (Markov)"
        dimension: "z"
        methods:
          method: "manual"            # Specify manual method
          shock_method: "DiscreteMarkov"
        # Provide values directly
        transition_matrix: [[0.9, 0.1], [0.2, 0.8]]
        values: [0.2, 1.3]
        labels: ["low", "high"]
      
      iid_shock:
        description: "Manually specified IID shock"
        dimension: "eps"
        methods:
          method: "manual"
          shock_method: "IID"
        values: [-0.1, 0.0, 0.1]  # Uniform distribution by default
  
  functions:
    utility:
      expr: "c**(1-gamma)/(1-gamma)"
"""

def main():
    """Run the example."""
    print("Loading model configuration...")
    config = yaml.safe_load(CONFIG)
    
    print("Initializing model...")
    stage_model, mover_models, perch_models = initialize_model(config)
    
    print("Compiling numerical model...")
    # This is the key step to generate numerical components including shocks
    compile_num(stage_model)
    
    print("\nExamining the model:")
    print(f"Model name: {stage_model.name}")
    print(f"Parameters: beta={stage_model.param.beta}, gamma={stage_model.param.gamma}")
    
    # Check if math section has shocks
    print("\nShocks in math section:")
    if hasattr(stage_model, "math") and "shocks" in stage_model.math:
        print(f"Found {len(stage_model.math['shocks'])} shocks: {list(stage_model.math['shocks'].keys())}")
    else:
        print("No shocks found in math section")
    
    # Verify shocks were created
    print("\nShocks in the numerical model:")
    
    # Debug printing
    print(f"Available keys in stage_model.num: {list(stage_model.num.keys())}")
    
    if "shocks" in stage_model.num:
        for shock_name, shock_data in stage_model.num["shocks"].items():
            print(f"\n  {shock_name}:")
            print(f"    Available keys: {list(shock_data.keys())}")
            print(f"    Dimension: {shock_data.get('dimensions')}")
            
            if "values" in shock_data:
                print(f"    Values: {shock_data['values']}")
            if "probs" in shock_data:
                print(f"    Probabilities: {shock_data['probs']}")
            if "process" in shock_data:
                print(f"    Process type: {type(shock_data['process']).__name__}")
            
            # If it's a Markov process, show the transition matrix
            if "transition_matrix" in shock_data:
                print(f"    Transition Matrix:")
                for row in shock_data["transition_matrix"]:
                    print(f"      {row}")
                
                # Visualize transition matrix if matplotlib is available
                try:
                    plt.figure(figsize=(5, 4))
                    plt.imshow(shock_data["transition_matrix"], cmap='Blues')
                    plt.colorbar(label='Probability')
                    plt.title(f'{shock_name} Transition Matrix')
                    plt.xlabel('To State')
                    plt.ylabel('From State')
                    
                    # Add state labels if available
                    if "labels" in shock_data:
                        plt.xticks(range(len(shock_data["labels"])), shock_data["labels"])
                        plt.yticks(range(len(shock_data["labels"])), shock_data["labels"])
                    else:
                        plt.xticks(range(len(shock_data["values"])))
                        plt.yticks(range(len(shock_data["values"])))
                    
                    # Save the figure
                    os.makedirs("output", exist_ok=True)
                    plt.savefig(f"output/{shock_name}_transition_matrix.png")
                    plt.close()
                    print(f"    Transition matrix visualization saved to output/{shock_name}_transition_matrix.png")
                except Exception as e:
                    print(f"    Could not visualize transition matrix: {e}")
    else:
        print("No shocks found in the model's num dictionary.")
    
    print("\nManual shock example completed successfully!")

if __name__ == "__main__":
    main() 