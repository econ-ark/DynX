#!/usr/bin/env python
"""
Housing model driver for ModCraft.

This script loads, initializes, and solves the owner-only housing model
using the StageCraft and Heptapod-B architecture.

Usage:
    python Housing.py [--periods N] [--plot] [--no-solve]

Options:
    --periods N     Number of periods to simulate (default: 10)
    --plot          Generate and save plots
    --no-solve      Skip solving (for testing loading only)
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import copy
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

# Add modcraft root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

# Import from ModCraft
from src.stagecraft import Stage
from src.stagecraft.config_loader import initialize_model_Circuit, compile_all_stages
from src.heptapod_b.io.yaml_loader import load_config
from src.heptapod_b.core.api import initialize_model
from src.heptapod_b.num.generate import compile_num as generate_numerical_model

current_dir = os.path.dirname(os.path.abspath(__file__))

repo_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.insert(0, repo_root)


# Import housing model
from models.housing.whisperer import (
    build_operators,
    solve_stage,
    run_time_iteration,
)

# Add new import
try:
    from FUES.math_funcs import mask_jumps
except ImportError:
    # Fallback function if FUES is not installed
    def mask_jumps(y, threshold=0.02):
        return y

def load_configs():
    """Load all configuration files."""
    config_dir = os.path.join(os.path.dirname(__file__), "config_v2")
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

def initialize_housing_model():
    """Initialize the housing model stages using YAML configuration.
    
    Returns
    -------
    dict
        Dictionary containing the OWNH and OWNC stages
    """
    print("Initializing housing model...")
    
    # Load configurations
    configs = load_configs()
    
    # Prepare stages config dictionary
    stage_configs = {
        "OWNH": configs["ownh"],
        "OWNC": configs["ownc"]
    }
    
    # Let the framework handle all the complexity of creating stages,
    # perches, movers, and connections
    print("Building model circuit using configuration...")
    model_circuit = initialize_model_Circuit(
        master_config=configs["master"],
        stage_configs=stage_configs,
        connections_config=configs["connections"]
    )

    compile_all_stages(model_circuit)
    
    # Get the individual stages from the model circuit
    # For the housing model, we have just one period with two stages
    period = model_circuit.get_period(0)
    ownh_stage = period.get_stage("OWNH")
    ownc_stage = period.get_stage("OWNC")
    
    # Debug: Print information about the Markov shock process
    print("\nShock information for OWNH stage:")
    shock_info = ownh_stage.model.num.shocks.income_shock
    #print(f"Shock values: {shock_info.grid}")
    print(f"Transition matrix shape: {shock_info.transition_matrix.shape}")
    
    # Mark final period stages
    ownh_stage.status_flags["is_terminal"] = False
    ownc_stage.status_flags["is_terminal"] = True
    
    # Set external mode for stages
    ownh_stage.model_mode = "external"
    ownc_stage.model_mode = "external"
    
    # Return both stages as a dictionary
    return {
        "OWNH": ownh_stage,
        "OWNC": ownc_stage
    }

def create_multi_period_model():
    """Create a multi-period model circuit.
    
    Parameters
    ----------
    n_periods : int
        Number of periods to create
        
    Returns
    -------
    ModelCircuit
        The multi-period model circuit
    """
    print(f"Creating multi-period model with  periods...")
    
    # Load configurations
    configs = load_configs()
    
    # Prepare stages config dictionary
    stage_configs = {
        "OWNH": configs["ownh"],
        "OWNC": configs["ownc"]
    }
    
    # Set the number of periods in the master config
    # This is critical to properly create a multi-period model
    master_config = copy.deepcopy(configs["master"])
    
    # Build the multi-period model circuit
    print("Building multi-period model circuit...")
    model_circuit = initialize_model_Circuit(
        master_config=master_config,
        stage_configs=stage_configs,
        connections_config=configs["connections"]
    )

    compile_all_stages(model_circuit)
    
    return model_circuit

def plot_policies(stages, filename=None):
    """Plot policy functions similar to fella.py's output.
    
    Parameters
    ----------
    stages : list
        List of stage dictionaries for each period
    filename : str, optional
        Filename to save the plot, by default None
    """
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Extract the latest stage
    latest_stage = stages[-1]
    ownh_stage = latest_stage["OWNH"]
    ownc_stage = latest_stage["OWNC"]
    
    # Plot 1: Consumption policy function
    ax = axs[0, 0]
    w_grid = ownc_stage.model.num.dcsn.w
    H_idx = 1  # Middle housing value
    y_idx = 0  # First income state
    
    consumption = ownc_stage.perches["dcsn"].sol["policy"][:, H_idx, y_idx]
    ax.plot(w_grid, consumption, label="Consumption Policy")
    ax.plot(w_grid, w_grid, 'k--', label="45-degree Line")
    ax.set_title("Consumption Policy Function")
    ax.set_xlabel("Cash-on-Hand (w)")
    ax.set_ylabel("Consumption (c)")
    ax.legend()
    ax.grid(True)
    
    # Plot 2: Housing policy function
    ax = axs[0, 1]
    a_grid = ownh_stage.model.num.dcsn.a
    H_idx = 1  # Middle housing value
    y_idx = 0  # First income state
    
    H_policy_idx = ownh_stage.perches["dcsn"].sol["H_policy"][:, H_idx, y_idx]
    H_nxt_grid = ownh_stage.model.num.cntn.H_nxt
    H_policy = H_nxt_grid[H_policy_idx]
    
    ax.step(a_grid, H_policy, where='post', label="Housing Policy")
    ax.set_title("Housing Policy Function")
    ax.set_xlabel("Assets (a)")
    ax.set_ylabel("Housing Choice (H)")
    ax.legend()
    ax.grid(True)
    
    # Plot 3: Value function at arrival (showing for different y_pre values)
    ax = axs[1, 0]
    a_grid = ownh_stage.model.num.arvl.a
    H_idx = 1  # Middle housing value
    
    # Plot value function for each previous shock state
    for i_y_pre in range(ownh_stage.perches["arvl"].sol["vlu_arvl"].shape[2]):
        value = ownh_stage.perches["arvl"].sol["vlu_arvl"][:, H_idx, i_y_pre]
        y_pre_val = ownh_stage.model.num.shocks.income_shock.grid[i_y_pre]
        ax.plot(a_grid, value, label=f"y_pre={y_pre_val:.2f}")
    
    ax.set_title("Value Function at Arrival")
    ax.set_xlabel("Assets (a)")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)
    
    # Plot 4: Policy functions across time
    ax = axs[1, 1]
    w_grid = ownc_stage.model.num.dcsn.w
    
    # Plot consumption policies for multiple periods
    for i, stage_dict in enumerate(stages):
        period = i + 1
        ownc_stage_t = stage_dict["OWNC"]
        consumption = ownc_stage_t.perches["dcsn"].sol["policy"][:, H_idx, y_idx]
        ax.plot(w_grid, consumption, label=f"T-{period}")
    
    ax.set_title("Consumption Policies Across Time")
    ax.set_xlabel("Cash-on-Hand (w)")
    ax.set_ylabel("Consumption (c)")
    ax.legend()
    ax.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename)
        print(f"Plot saved to {filename}")
    
    return fig

def plot_multiperiod_results(model_circuit, image_dir):
    """Plot policy functions similar to fella.py using the model_circuit.
    
    Parameters
    ----------
    model_circuit : ModelCircuit
        The model circuit containing the periods and stages
    image_dir : str
        Directory to save the output images
    """
    # Close any existing plots
    plt.close()
    
    # Set seaborn style to match run_fella.py
    sns.set(style="white", rc={"font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11})
    
    # Get the first period (index 0)
    first_period = model_circuit.periods_list[0]
    
    # Get the stages from the first period
    ownc_stage = first_period.get_stage("OWNC")
    ownh_stage = first_period.get_stage("OWNH")
    
    # Create figure for consumption policies - match run_fella.py exactly
    fig, ax = plt.subplots(1, 2, figsize=(8, 6))
    
    # Apply styling to both subplots
    for a in ax:
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        a.spines['left'].set_visible(True)
        a.spines['bottom'].set_visible(True)
        a.grid(True)
        a.set_yticklabels(a.get_yticks(), size=9)
        a.set_xticklabels(a.get_xticks(), size=9)
        a.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        a.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    
    # Define colors for different housing values - exact same as run_fella.py
    colors = ['blue', 'red', 'green']
    
    # Get cash-on-hand grid
    w_grid = ownc_stage.dcsn.grid.w
    
    # Get housing grid
    H_grid = ownc_stage.dcsn.grid.H_nxt
    
    # Select housing indices for plotting - EXACTLY as in run_fella.py
    H_inds = [1, 2, 4]
    
    # Fixed income state for all plots (middle state = 1)
    y_idx = 1
    
    # Initialize min/max values for dynamic axis scaling
    c_min, c_max = float('inf'), float('-inf')
    w_min, w_max = float('inf'), float('-inf')
    
    # Store consumption data for both subplots
    consumption_data = []
    
    # Plot 1: Consumption policy for different housing values (left subplot)
    for i, (H_idx, color, label) in enumerate(zip(H_inds, colors, ['$H_{t}$ = low', '$H_{t}$ = med.', '$H_{t}$ = high'])):
        # Get consumption policy - consistently use income state 1 (middle)
        consumption = ownc_stage.dcsn.sol["policy"][:, H_idx, y_idx]
        
        # Handle NaN and inf values
        consumption = np.nan_to_num(consumption, nan=0.0, posinf=None, neginf=0.0)
        
        # Apply mask_jumps to smooth out any discontinuities - same as run_fella.py
        consumption_masked = mask_jumps(consumption, threshold=0.2)
        
        # Update min/max values for dynamic scaling
        c_min = min(c_min, np.nanmin(consumption_masked))
        c_max = max(c_max, np.nanmax(consumption_masked))
        w_min = min(w_min, np.nanmin(w_grid))
        w_max = max(w_max, np.nanmax(w_grid))
        
        # Store data for later plotting
        consumption_data.append((w_grid, consumption_masked, color, label))
    
    # Add padding to axis limits (20% padding)
    c_range = c_max - c_min
    w_range = w_max - w_min
    c_min = max(0, c_min - 0.1 * c_range)  # Don't go below 0 for consumption
    c_max = c_max + 0.2 * c_range
    w_min = max(0, w_min - 0.1 * w_range)  # Don't go below 0 for wealth
    w_max = w_max + 0.2 * w_range
    
    # Now actually plot the data with appropriate limits
    for w_grid, consumption_masked, color, label in consumption_data:
        ax[0].plot(w_grid, consumption_masked, color=color, linestyle='-', linewidth=1, label=label)
        ax[1].plot(w_grid, consumption_masked, color=color, linestyle='-', linewidth=1, label=label)
    
    # Set dynamic limits for consumption plots
    ax[0].set_xlim([w_min, w_max])
    ax[0].set_ylim([c_min, c_max])
    ax[0].set_title("FUES", fontsize=11)
    ax[0].set_ylabel('Consumption at time $t$', fontsize=11)
    
    ax[1].set_xlim([w_min, w_max])
    ax[1].set_ylim([c_min, c_max])
    ax[1].set_title("DC-EGM", fontsize=11)
    ax[1].set_ylabel('Consumption at time $t$', fontsize=11)
    
    # Add legends with same formatting as run_fella.py
    ax[0].legend(frameon=False, prop={'size': 10})
    ax[1].legend(frameon=False, prop={'size': 10})
    
    # Add a common x-label - same format as run_fella.py
    fig.supxlabel(r'Financial assets at time $t$', fontsize=11)
    
    # Set tight layout
    fig.tight_layout()
    
    # Save the consumption policy figure
    output_path = os.path.join(image_dir, "housing_policy_functions.png")
    fig.savefig(output_path)
    print(f"Policy function plot saved to {output_path}")
    
    # Create a second plot for housing policy - same structure as run_fella.py
    fig2, ax2 = plt.subplots(1, 2, figsize=(8, 6))
    
    # Apply styling
    for a in ax2:
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        a.spines['left'].set_visible(True)
        a.spines['bottom'].set_visible(True)
        a.grid(True)
        a.set_yticklabels(a.get_yticks(), size=9)
        a.set_xticklabels(a.get_xticks(), size=9)
        a.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        a.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    
    # Get asset grid
    a_grid = ownh_stage.dcsn.grid.a
    
    # Initialize min/max values for dynamic axis scaling
    h_min, h_max = float('inf'), float('-inf')
    a_min, a_max = float('inf'), float('-inf')
    
    # Store housing data for both subplots
    housing_data_left = []
    housing_data_right = []
    
    # Plot housing policy for different housing values - left subplot
    for i, (H_idx, color, label) in enumerate(zip(H_inds, colors, ['$H_{t}$ = low', '$H_{t}$ = med.', '$H_{t}$ = high'])):
        if "H_policy" in ownh_stage.dcsn.sol:
            H_policy = ownh_stage.dcsn.sol["H_policy"][:, H_idx, y_idx]  # Consistent income state 1
            H_values = ownh_stage.cntn.grid.H_nxt[H_policy]
            
            # Handle NaN and inf values
            H_values = np.nan_to_num(H_values, nan=0.0, posinf=None, neginf=0.0)
            
            # Apply mask_jumps to smoothen any discontinuities
            H_values_masked = mask_jumps(H_values, threshold=0.02)
            
            # Update min/max values for dynamic scaling
            h_min = min(h_min, np.nanmin(H_values_masked))
            h_max = max(h_max, np.nanmax(H_values_masked))
            a_min = min(a_min, np.nanmin(a_grid))
            a_max = max(a_max, np.nanmax(a_grid))
            
            # Store for later plotting
            housing_data_left.append((a_grid, H_values_masked, color, label))
    
    # Plot housing policy for different income values - right subplot
    # Use middle housing value
    H_idx = H_inds[1]  # Use medium housing index
    
    for i, (income_idx, color, label) in enumerate(zip([0, 1, 2], colors, ['$y_{t}$ = low', '$y_{t}$ = med.', '$y_{t}$ = high'])):
        if "H_policy" in ownh_stage.dcsn.sol and income_idx < ownh_stage.dcsn.sol['H_policy'].shape[2]:
            H_policy = ownh_stage.dcsn.sol["H_policy"][:, H_idx, income_idx]
            H_values = ownh_stage.cntn.grid.H_nxt[H_policy]
            
            # Handle NaN and inf values
            H_values = np.nan_to_num(H_values, nan=0.0, posinf=None, neginf=0.0)
            
            # Apply mask_jumps to smoothen any discontinuities
            H_values_masked = mask_jumps(H_values, threshold=0.02)
            
            # Update min/max values for dynamic scaling
            h_min = min(h_min, np.nanmin(H_values_masked))
            h_max = max(h_max, np.nanmax(H_values_masked))
            
            # Store for later plotting
            housing_data_right.append((a_grid, H_values_masked, color, label))
    
    # Add padding to axis limits (20% padding)
    h_range = h_max - h_min
    a_range = a_max - a_min
    h_min = max(0, h_min - 0.1 * h_range)  # Don't go below 0 for housing
    h_max = h_max + 0.2 * h_range
    a_min = max(0, a_min - 0.1 * a_range)  # Don't go below 0 for assets
    a_max = a_max + 0.2 * a_range
    
    # Now plot housing data with appropriate limits
    for a_grid, H_values_masked, color, label in housing_data_left:
        ax2[0].plot(a_grid, H_values_masked, color=color, linestyle='-', linewidth=1, label=label)
    
    for a_grid, H_values_masked, color, label in housing_data_right:
        ax2[1].plot(a_grid, H_values_masked, color=color, linestyle='-', linewidth=1, label=label)
    
    # Set dynamic limits and titles for housing plots
    ax2[0].set_xlim([a_min, a_max])
    ax2[0].set_ylim([h_min, h_max])
    ax2[0].set_title("Housing Policy By Housing Value", fontsize=11)
    ax2[0].set_ylabel('Housing Choice', fontsize=11)
    
    ax2[1].set_xlim([a_min, a_max])
    ax2[1].set_ylim([h_min, h_max])
    ax2[1].set_title("Housing Policy By Income", fontsize=11)
    ax2[1].set_ylabel('Housing Choice', fontsize=11)
    
    # Add legends with same formatting as run_fella.py
    ax2[0].legend(frameon=False, prop={'size': 10})
    ax2[1].legend(frameon=False, prop={'size': 10})
    
    # Add a common x-label - same format as run_fella.py
    fig2.supxlabel(r'Financial assets at time $t$', fontsize=11)
    
    # Set tight layout
    fig2.tight_layout()
    
    # Save the housing policy figure
    output_path2 = os.path.join(image_dir, "housing_policy_housing.png")
    fig2.savefig(output_path2)
    print(f"Housing policy plot saved to {output_path2}")
    
    plt.close('all')

def create_image_dir():
    """Create directory for output images."""
    image_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "images"))
    if not os.path.exists(image_dir):
        print(f"Creating image directory: {image_dir}")
        os.makedirs(image_dir)
    return image_dir

def main():
    """Main driver function."""
    # Create image directory
    image_dir = create_image_dir()
    
    # Initialize model
    model_circuit = create_multi_period_model()

    # Solve the multi-period model - set verbose to False to disable debug output,
    # but we'll still get timing information
    all_stages_solved = run_time_iteration(model_circuit, verbose=False)

    # Generate and save plots
    plot_multiperiod_results(model_circuit, image_dir)

if __name__ == "__main__":
    main()

    print("Done.")
    