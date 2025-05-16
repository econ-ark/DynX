#!/usr/bin/env python3
# examples/monetary_policy_mortgage_model.py

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Add the repository root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.insert(0, parent_dir)

from src.stagecraft.model_circuit import ModelCircuit
from src.stagecraft.period import Period
from src.stagecraft.stage import Stage

def main():
    """
    Create a model sequence for a lifecycle model with mortgage and housing decisions.
    
    Model structure based on the monetary policy transmission with mortgages model,
    with discrete lifecycle periods and multiple decision stages within each period.
    
    Key features:
    - Housing tenure decisions (rent vs. own)
    - Housing adjustment decisions for owners
    - Consumption and savings decisions
    - Mortgage payment and refinancing decisions
    """
    # Create a new model sequence for the lifecycle model
    model = ModelCircuit("MortgageLifecycleModel")
    
    # Number of lifecycle periods (e.g., ages 25-85 in annual periods)
    num_periods = 5  # Using 5 periods for demonstration (can be extended)
    
    # Create periods and stages for the lifecycle model
    for t in range(num_periods):
        period = Period(time_index=t)
        
        # Create stages for different decision nodes within each period
        
        # 1. Housing tenure decision stage (discrete choice between renting and owning)
        tenure_choice_stage = Stage(name=f"tenure_choice_t{t}", decision_type="discrete")
        tenure_choice_stage.stage_type = "tenure_choice"
        
        # 2. Renter stage (for those who choose to rent)
        renter_stage = Stage(name=f"renter_t{t}")
        renter_stage.stage_type = "renter"
        
        # 3. Owner stages
        
        # 3a. Housing adjustment decision (keep current house or adjust)
        housing_adj_stage = Stage(name=f"housing_adj_t{t}", decision_type="discrete")
        housing_adj_stage.stage_type = "housing_adj"
        
        # 3b. Stage for homeowners who keep their current house
        keep_house_stage = Stage(name=f"keep_house_t{t}")
        keep_house_stage.stage_type = "keep_house"
        
        # 3c. Stage for homeowners who adjust their housing stock
        adjust_house_stage = Stage(name=f"adjust_house_t{t}")
        adjust_house_stage.stage_type = "adjust_house"
        
        # 4. Consumption and mortgage payment decision stage
        cons_mort_stage = Stage(name=f"cons_mort_t{t}")
        cons_mort_stage.stage_type = "cons_mort"
        
        # Add all stages to the period
        period.add_stage("tenure_choice", tenure_choice_stage)
        period.add_stage("renter", renter_stage)
        period.add_stage("housing_adj", housing_adj_stage)
        period.add_stage("keep_house", keep_house_stage)
        period.add_stage("adjust_house", adjust_house_stage)
        period.add_stage("cons_mort", cons_mort_stage)
        
        # Set up intra-period connections using the new v1.5.14 methods
        
        # From tenure choice to renter or housing adjustment decision
        period.connect_fwd(
            src="tenure_choice",
            tgt="renter",
            branch_key="renter_choice"
        )
        
        period.connect_fwd(
            src="tenure_choice",
            tgt="housing_adj",
            branch_key="housing_choice"
        )
        
        # From housing adjustment decision to either keep or adjust house
        period.connect_fwd(
            src="housing_adj",
            tgt="keep_house",
            branch_key="keep_house_choice"
        )
        
        period.connect_fwd(
            src="housing_adj",
            tgt="adjust_house",
            branch_key="adjust_house_choice"
        )
        
        # From all housing outcome stages to consumption/mortgage decision
        period.connect_fwd(
            src="renter",
            tgt="cons_mort",
            branch_key="renter_path"
        )
        
        period.connect_fwd(
            src="keep_house",
            tgt="cons_mort",
            branch_key="keep_house_path"
        )
        
        period.connect_fwd(
            src="adjust_house",
            tgt="cons_mort",
            branch_key="adjust_house_path"
        )
        
        # Create transpose connections within the period for backward solving
        period.create_transpose_connections()
        
        # Add period to model sequence
        model.add_period(period)
    
    # Add inter-period connections
    # The consumption/mortgage decision in one period connects to the tenure choice in the next
    for t in range(num_periods - 1):
        model.add_inter_period_connection(
            source_period=model.get_period(t),
            target_period=model.get_period(t+1),
            source_stage=model.get_period(t).get_stage("cons_mort"),
            target_stage=model.get_period(t+1).get_stage("tenure_choice"),
            source_perch_attr="cntn",
            target_perch_attr="arvl"
        )
    
    # Create transpose connections for inter-period connections
    model.create_transpose_connections()
    
    # Customize the labels for visualization
    custom_labels = {
        'tenure_choice': 'Housing Tenure Choice',
        'renter': 'Renter Decision',
        'housing_adj': 'Housing Adjustment Decision',
        'keep_house': 'Keep Current House',
        'adjust_house': 'Adjust Housing Stock',
        'cons_mort': 'Consumption & Mortgage'
    }
    
    # Define abbreviated names for shorter labels
    abbreviated_names = {
        'tenure_choice': 'TC',
        'renter': 'R',
        'housing_adj': 'HA',
        'keep_house': 'KH',
        'adjust_house': 'AH',
        'cons_mort': 'CM'
    }
    
    # Define full stage names for legend
    stage_full_names = {
        'tenure_choice': 'Housing Tenure Choice',
        'renter': 'Renter Decision',
        'housing_adj': 'Housing Adjustment Decision',
        'keep_house': 'Keep Current House',
        'adjust_house': 'Adjust Housing Stock',
        'cons_mort': 'Consumption & Mortgage'
    }
    
    # Configure font properties for better readability
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'figure.figsize': (16, 12),
        'figure.dpi': 300
    })
    
    # Create color map for node types with more distinct colors
    type_colors = {
        'tenure_choice': '#4daf4a',       # Green for tenure choice
        'renter': '#377eb8',              # Blue for renter
        'housing_adj': '#ff7f00',         # Orange for housing adjustment
        'keep_house': '#984ea3',          # Purple for keep house
        'adjust_house': '#e41a1c',        # Red for adjust house
        'cons_mort': '#f781bf'            # Pink for consumption & mortgage
    }
    
    # Create edge style mapping with more distinctive styles for backward edges
    edge_styles = {
        'forward_intra': {'color': 'blue', 'style': 'solid', 'width': 1.5, 'alpha': 0.7, 'arrowsize': 15},
        'backward_intra': {'color': 'red', 'style': 'dashed', 'width': 2.5, 'alpha': 1.0, 'arrowsize': 20},
        'forward_inter': {'color': 'darkblue', 'style': 'solid', 'width': 2.0, 'alpha': 0.9, 'arrowsize': 20},
        'backward_inter': {'color': 'darkred', 'style': 'dashed', 'width': 2.5, 'alpha': 1.0, 'arrowsize': 25}
    }
    
    # Create a colormap for periods to add as a visual key
    period_colors = ['#a6cee3', '#b2df8a', '#fdbf6f', '#cab2d6', '#fb9a99']
    
    # Create directory for images if it doesn't exist
    image_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'images', 'mortgages'))
    if not os.path.exists(image_dir):
        print(f"Creating directory: {image_dir}")
        os.makedirs(image_dir)
    
    # 1. Spring layout (force-directed graph)
    print("\nVisualizing mortgage lifecycle model with spring layout...")
    model.visualize_stage_graph(
        edge_type='both',                 # Show both forward and backward edges
        layout='period_spring',           # Period-grouped spring layout
        custom_labels=custom_labels,      # Use custom labels
        include_period_in_label=True,     # Include period in labels
        short_labels=True,                # Use abbreviated labels
        node_color_mapping=type_colors,   # Use custom node colors
        node_size=1200,                   # Slightly smaller nodes to reduce overlap
        legend_loc='upper left',          # Position legend in upper left
        figsize=(16, 12),                 # Larger figure size for better spacing
        mark_special_nodes=True,          # Mark initial and terminal nodes
        filename=os.path.join(image_dir, "mortgage_lifecycle_spring.png"),
        abbreviated_names=abbreviated_names,  # Pass the abbreviated names
        stage_full_names=stage_full_names,    # Pass the full stage names
        edge_style_mapping=edge_styles,       # Pass the edge style mapping
        connectionstyle='arc3,rad=0.3',   # Use more curved edges to separate forward/backward
        dpi=300                           # Higher resolution output
    )
    plt.close('all')
    
    # 2. Dot layout (hierarchical layout, top to bottom)
    print("\nVisualizing mortgage lifecycle model with hierarchical (dot) layout...")
    model.visualize_stage_graph(
        edge_type='forward',              # Only show forward edges for clarity
        layout='dot',                     # Hierarchical layout
        custom_labels=custom_labels,
        include_period_in_label=True,
        abbreviated_names=abbreviated_names,  # Pass the abbreviated names
        stage_full_names=stage_full_names,    # Pass the full stage names
        edge_style_mapping=edge_styles,       # Pass the edge style mapping
        filename=os.path.join(image_dir, "mortgage_lifecycle_hierarchical.png")
    )
    plt.close('all')
    
    # 3. Circo layout (circular layout, good for cyclical structures)
    print("\nVisualizing mortgage lifecycle model with circular layout...")
    model.visualize_stage_graph(
        edge_type='forward',              # Only show forward edges for clarity
        layout='circo',                   # Circular layout
        custom_labels=custom_labels,
        include_period_in_label=True,
        abbreviated_names=abbreviated_names,  # Pass the abbreviated names
        stage_full_names=stage_full_names,    # Pass the full stage names
        edge_style_mapping=edge_styles,       # Pass the edge style mapping
        filename=os.path.join(image_dir, "mortgage_lifecycle_circular.png")
    )
    plt.close('all')
    
    # Print model structure summary
    print("\nModel Structure Summary:")
    print(f"Model name: {model.name}")
    
    # Access periods using index-based approach to avoid attribute errors
    period_count = 0
    while True:
        try:
            period = model.get_period(period_count)
            period_count += 1
        except:
            break
    
    print(f"Number of periods: {period_count}")
    print(f"Total number of stages created: {period_count * 6}")  # 6 stages per period
    print(f"Images saved in: {image_dir}")
    
    return model

if __name__ == "__main__":
    model = main() 