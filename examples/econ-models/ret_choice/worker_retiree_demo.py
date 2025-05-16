#!/usr/bin/env python3
# examples/worker_retiree_demo.py

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Try different import approaches to make the script runnable from various locations
try:
    # When running from the project root or if package is installed
    from stagecraft import Stage
    from stagecraft import ModelCircuit 
    from stagecraft import Period
except ImportError:
    try:
        # When running from examples directory
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from src.stagecraft import Stage
        from src.stagecraft import ModelCircuit
        from src.stagecraft import Period
    except ImportError:
        raise ImportError(
            "Unable to import stagecraft. Make sure you're either:\n"
            "1. Running from the project root directory\n"
            "2. Have installed the package with 'pip install -e .'\n"
            "3. Have added the project root to your PYTHONPATH"
        )

# Define model parameters
model_parameters = {
    "worker": {
        "wage": 1.0,
        "prob_shock": 0.1
    },
    "retiree": {
        "pension": 0.7,
        "health_cost": 0.1
    },
    "discount_factor": 0.95
}

def main():
    """
    Create a demo model with worker, retiree, and discrete choice stages
    to demonstrate the custom labeling in stage graph visualization.
    """
    # Create a new model sequence
    model = ModelCircuit("WorkerRetireeDemo")
    
    # Create 3 periods
    for t in range(3):
        period = Period(time_index=t)
        
        # Create stages with different types
        worker_stage = Stage(name=f"worker_t{t}")
        worker_stage.stage_type = "worker"  # Set stage type for proper labeling
        
        retiree_stage = Stage(name=f"retiree_t{t}")
        retiree_stage.stage_type = "retiree"  # Set stage type for proper labeling
        
        disc_choice_stage = Stage(name=f"disc_choice_t{t}", decision_type="discrete")
        disc_choice_stage.stage_type = "disc.choice"  # Set stage type for proper labeling
        
        # Add stages to the period
        period.add_stage("worker", worker_stage)
        period.add_stage("retiree", retiree_stage)
        period.add_stage("disc_choice", disc_choice_stage)
        
        # Add intra-period connections with proper branch keys
        period.add_connection(
            source_stage=disc_choice_stage,
            target_stage=worker_stage,
            source_perch_attr="cntn",
            target_perch_attr="arvl",
            direction="forward",
            branch_key="worker",  # Add branch_key for the connection
            create_transpose=False  # Disable automatic transpose creation
        )
        
        period.add_connection(
            source_stage=disc_choice_stage,
            target_stage=retiree_stage,
            source_perch_attr="cntn",
            target_perch_attr="arvl",
            direction="forward",
            branch_key="retiree",  # Add branch_key for the connection
            create_transpose=False  # Disable automatic transpose creation
        )
        
        # Add manual transpose connections with proper branch keys
        period.add_connection(
            source_stage=worker_stage,
            target_stage=disc_choice_stage,
            source_perch_attr="arvl",
            target_perch_attr="cntn",
            direction="backward",
            branch_key="worker"  # Specify branch key for backward connection
        )
        
        period.add_connection(
            source_stage=retiree_stage,
            target_stage=disc_choice_stage,
            source_perch_attr="arvl",
            target_perch_attr="cntn",
            direction="backward",
            branch_key="retiree"  # Specify branch key for backward connection
        )
        
        # Add period to model sequence
        model.add_period(period)
    
    # Add inter-period connections
    # Worker to worker
    for t in range(2):
        # We need to store the branch_key for later, but it's not a parameter for Mover.__init__
        mover = model.add_inter_period_connection(
            source_period=model.get_period(t),
            target_period=model.get_period(t+1),
            source_stage=model.get_period(t).get_stage("worker"),
            target_stage=model.get_period(t+1).get_stage("disc_choice"),
            source_perch_attr="cntn",
            target_perch_attr="arvl"
        )
        # Set branch_key as an attribute after creation
        mover.branch_key = "worker"
    
    # Retiree to retiree
    for t in range(2):
        # We need to store the branch_key for later, but it's not a parameter for Mover.__init__
        mover = model.add_inter_period_connection(
            source_period=model.get_period(t),
            target_period=model.get_period(t+1),
            source_stage=model.get_period(t).get_stage("retiree"),
            target_stage=model.get_period(t+1).get_stage("retiree"),
            source_perch_attr="cntn",
            target_perch_attr="arvl"
        )
        # Set branch_key as an attribute after creation
        mover.branch_key = "retiree"
    
    # Create all transpose connections
    # model.create_all_transpose_connections()  # Comment out this line to prevent automatic creation
    
    # Customize the labels for visualization
    custom_labels = {
        'worker': 'Worker',
        'retiree': 'Retiree',
        'disc.choice': 'Discrete Choice'
    }
    
    # Define abbreviated names for shorter labels
    abbreviated_names = {
        'worker': 'W',
        'retiree': 'R',
        'disc.choice': 'DC'
    }
    
    # Define full stage names for legend
    stage_full_names = {
        'worker': 'Worker',
        'retiree': 'Retiree',
        'disc.choice': 'Discrete Choice'
    }
    
    # Define node color mapping
    type_colors = {
        'worker': '#377eb8',     # Blue for worker
        'retiree': '#4daf4a',    # Green for retiree
        'disc.choice': '#e41a1c' # Red for discrete choice
    }
    
    # Define edge style mapping
    edge_styles = {
        'forward_intra': {'color': 'blue', 'style': 'solid', 'width': 1.5, 'alpha': 0.7, 'arrowsize': 15},
        'backward_intra': {'color': 'purple', 'style': 'dashed', 'width': 1.5, 'alpha': 0.7, 'arrowsize': 15},
        'forward_inter': {'color': 'darkblue', 'style': 'solid', 'width': 2.0, 'alpha': 0.9, 'arrowsize': 20},
        'backward_inter': {'color': 'darkviolet', 'style': 'dashed', 'width': 2.0, 'alpha': 0.9, 'arrowsize': 20}
    }
    
    # Create directory for images if it doesn't exist
    image_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'images', 'worker_retiree'))
    if not os.path.exists(image_dir):
        print(f"Creating directory: {image_dir}")
        os.makedirs(image_dir)
    
    print("\nVisualizing stage graph (forward edges only)...")
    fig1 = model.visualize_stage_graph(
        edge_type='forward',
        custom_labels=custom_labels,
        include_period_in_label=True,
        node_color_mapping=type_colors,
        edge_style_mapping=edge_styles,
        abbreviated_names=abbreviated_names,
        stage_full_names=stage_full_names,
        filename=os.path.join(image_dir, "worker_retiree_forward.png")
    )
    
    print("\nVisualizing stage graph (all edges)...")
    fig2 = model.visualize_stage_graph(
        edge_type='both',
        custom_labels=custom_labels,
        include_period_in_label=True,
        node_color_mapping=type_colors,
        edge_style_mapping=edge_styles,
        abbreviated_names=abbreviated_names,
        stage_full_names=stage_full_names,
        filename=os.path.join(image_dir, "worker_retiree_all.png")
    )
    
    print("\nVisualizing stage graph (alternative layout)...")
    fig3 = model.visualize_stage_graph(
        edge_type='both',
        layout='period_spring',
        custom_labels=custom_labels,
        include_period_in_label=True,
        node_color_mapping=type_colors,
        edge_style_mapping=edge_styles,
        abbreviated_names=abbreviated_names,
        stage_full_names=stage_full_names,
        short_labels=True,
        legend_loc='upper left',
        mark_special_nodes=True,
        node_size=1000,
        figsize=(14, 10),
        filename=os.path.join(image_dir, "worker_retiree_spring.png")
    )
    
    plt.close('all')
    print("\nStage graph visualization complete.")
    print(f"Files saved in: {image_dir}")

if __name__ == "__main__":
    main() 