#!/usr/bin/env python3
"""
Markov Shock Example Script

This script demonstrates the use of DiscreteMarkov shock grids in Heptapod,
showing how shock indices map to actual shock values.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the src directory to the path (adjust based on examples folder location)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import Heptapod modules
from src.heptapod.model_init import initialize_model
from src.heptapod.gen_num import generate_numerical_model

def plot_transition_matrix(transition_matrix, grid, title="Transition Matrix"):
    """Plot a heatmap of the transition matrix"""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(transition_matrix, cmap='viridis')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Probability", rotation=-90, va="bottom")
    
    # Add labels and ticks
    ax.set_xlabel("Next State Index")
    ax.set_ylabel("Current State Index")
    ax.set_title(title)
    
    # Add grid values as tick labels (rounded to 2 decimal places)
    grid_labels = [f"{val:.2f}" for val in grid]
    ax.set_xticks(np.arange(len(grid)))
    ax.set_yticks(np.arange(len(grid)))
    ax.set_xticklabels(grid_labels)
    ax.set_yticklabels(grid_labels)
    
    # Rotate tick labels and set alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations in each cell
    for i in range(len(grid)):
        for j in range(len(grid)):
            text = ax.text(j, i, f"{transition_matrix[i, j]:.2f}",
                          ha="center", va="center", 
                          color="white" if transition_matrix[i, j] > 0.5 else "black")
    
    fig.tight_layout()
    return fig, ax

def main():
    """Main function to demonstrate multi-dimensional grids"""
    # Load model from YAML file
    config_file = os.path.join(os.path.dirname(__file__), 'configs', 'markov_shock_example.yml')
    print(f"Loading model from {config_file}...")
    stage, movers, perches = initialize_model(config_file)
    
    # Generate numerical model
    print("Generating numerical model...")
    generate_numerical_model(stage)
    
    # Get shock and state space information
    shock_name = 'productivity'
    shock = stage.num['shocks'][shock_name]
    grid = shock['grid']
    transition_matrix = shock['transition_matrix']
    
    # Get state space grid (indices)
    z_grid = stage.num['state_space']['arvl']['z']
    
    print(f"\n==== SHOCK INDICES VS VALUES ====")
    print(f"The state space uses shock indices (z_idx), which map to shock values:")
    
    print(f"\n  GRID OF INDICES (state variable):")
    if len(z_grid) <= 20:
        print(f"  {z_grid}")
    else:
        print(f"  First few: {z_grid[:5]}...")
        print(f"  Last few: {z_grid[-5:]}...")
    
    print(f"\n  CORRESPONDING SHOCK VALUES:")
    print(f"  Index │ Shock Value")
    print(f"  ------┼------------")
    for idx in range(len(grid)):
        print(f"  {idx:5d} │ {grid[idx]:10.4f}")
    
    print(f"\n==== TRANSITION PROBABILITIES ====")
    print(f"The transition matrix gives P(z_idx' | z_idx):")
    
    # Show an example for index 2
    example_idx = 2
    print(f"\nIf current state has z_idx = {example_idx} (shock value = {grid[example_idx]:.4f}):")
    print(f"Probability of transitioning to:")
    
    # Show transition probabilities for non-negligible transitions
    for next_idx, prob in enumerate(transition_matrix[example_idx]):
        if prob > 0.001:  # Only show non-negligible probabilities
            print(f"  z_idx = {next_idx} (value = {grid[next_idx]:8.4f}): {prob:.4f}")
    
    # Plot the transition matrix
    try:
        fig, ax = plot_transition_matrix(transition_matrix, grid, 
                          f"Markov Transition Matrix (rho={shock['parameters']['rho']}, sigma={shock['parameters']['sigma']})")
        output_file = os.path.join(os.path.dirname(__file__), 'markov_transition_matrix.png')
        plt.savefig(output_file)
        print(f"\nTransition matrix plot saved to '{output_file}'")
        plt.close()
    except ImportError:
        print("Matplotlib not available, skipping plot creation")
    
    print(f"\n==== MODEL USAGE ====")
    print(f"When using this shock in your model:")
    print(f"1. The state variable stores the INDEX (0-8), not the actual shock value")
    print(f"2. To get the actual shock value: shock_value = shock_grid[shock_idx]")
    print(f"3. For transitions, use the row of transition_matrix corresponding to current index")
    print(f"4. For a model with z_idx and assets (a), the state would be (z_idx, a)")
    print(f"   - z_idx = 3 corresponds to productivity value of {grid[3]:.4f}")

if __name__ == "__main__":
    main() 