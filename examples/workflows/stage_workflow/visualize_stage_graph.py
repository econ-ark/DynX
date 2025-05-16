"""
Visualize the Stage's built-in graph

This script visualizes the Stage's built-in graph structure using NetworkX,
with styling consistent with ModelSequence visualization.
"""

# Add this to your notebook as a new cell after accessing stage.graph

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch
import numpy as np

def visualize_stage_graph(stage, 
                          edge_type='both',  # 'forward', 'backward', or 'both'
                          layout='spring',   # 'spring' or 'fixed'
                          custom_labels=None,
                          node_color_mapping=None,
                          edge_style_mapping=None,
                          abbreviated_names=None,
                          short_labels=False,
                          figsize=(10, 8),
                          node_size=3000,
                          legend_loc='upper center',
                          filename=None,
                          exclude_cntn_to_arvl=True,
                          show_edge_labels=True):
    """
    Visualize the Stage's built-in graph with customizable styling.
    Compatible with ModelSequence visualization style.
    
    Parameters:
    -----------
    stage : Stage
        The stage object containing the graph to visualize
    edge_type : str, optional
        Type of edges to display: 'forward', 'backward', or 'both'
    layout : str, optional
        Layout to use: 'spring' for force-directed or 'fixed' for predefined positions
    custom_labels : dict, optional
        Mapping of node names to custom display labels
    node_color_mapping : dict, optional
        Mapping of node types to colors
    edge_style_mapping : dict, optional
        Mapping of edge types to styling parameters
    abbreviated_names : dict, optional
        Mapping of node types to abbreviated names for shorter labels
    short_labels : bool, optional
        Whether to use short labels (abbreviated names)
    figsize : tuple, optional
        Figure size (width, height) in inches
    node_size : int, optional
        Size of the nodes in the visualization
    legend_loc : str, optional
        Location of the legend
    filename : str, optional
        If provided, save the figure to this filename
    exclude_cntn_to_arvl : bool, optional
        Whether to exclude the cntn_to_arvl edge from the visualization
    show_edge_labels : bool, optional
        Whether to show edge labels. Compatible with ModelSequence parameter
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the visualization
    """
    G = stage.graph  # Use the Stage's built-in graph
    
    # Filter edges if needed
    filtered_G = nx.DiGraph()
    
    for edge in G.edges(data=True):
        # Handle both regular edges (u, v) and multigraph edges (u, v, key)
        if len(edge) == 2:
            source, target = edge
            data = {}
        elif len(edge) == 3:
            source, target, data = edge
            
        # Skip cntn_to_arvl edge if requested
        if exclude_cntn_to_arvl and source == 'cntn' and target == 'arvl':
            continue
            
        filtered_G.add_edge(source, target, **data)
    
    # Categorize edges
    forward_edges = []
    backward_edges = []
    
    for source, target in filtered_G.edges():
        if ((source == 'arvl' and target == 'dcsn') or 
            (source == 'dcsn' and target == 'cntn')):
            forward_edges.append((source, target))
        else:
            backward_edges.append((source, target))
    
    # Filter edges based on edge_type parameter
    display_edges = []
    if edge_type == 'forward':
        display_edges = forward_edges
    elif edge_type == 'backward':
        display_edges = backward_edges
    else:  # 'both'
        display_edges = forward_edges + backward_edges
    
    # Create a subgraph with only the edges we want to display
    display_G = nx.DiGraph()
    for source, target in display_edges:
        display_G.add_edge(source, target)
    
    # Now create the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set up node positions based on layout type
    if layout == 'fixed':
        pos = {
            "arvl": (0, 0),
            "dcsn": (1, 0),
            "cntn": (2, 0)
        }
    else:  # 'spring'
        pos = nx.spring_layout(display_G, seed=42, k=1.5)
    
    # Default node colors if not specified
    if node_color_mapping is None:
        node_color_mapping = {
            'arvl': 'lightblue',
            'dcsn': 'lightgreen',
            'cntn': 'lightyellow'
        }
    
    # Default edge styles if not specified - compatible with ModelSequence style
    if edge_style_mapping is None:
        edge_style_mapping = {
            'forward': {'color': 'blue', 'style': 'solid', 'width': 2, 'alpha': 0.8, 'mutation_scale': 15},
            'backward': {'color': 'red', 'style': 'dashed', 'width': 2, 'alpha': 0.8, 'mutation_scale': 15}
        }
    
    # Create a mapping of nodes to colors
    node_colors = []
    for node in display_G.nodes():
        # Use node type if available, otherwise use the node name
        node_type = getattr(stage, 'perch_type', {}).get(node, node)
        node_colors.append(node_color_mapping.get(node_type, 'lightgreen'))
    
    # Calculate node radius for proper arrow shrinkage
    # This converts node size to approximately the radius in points
    node_radius = np.sqrt(node_size / np.pi)  # More accurate calculation for shrinkage
    
    # First draw the nodes so they appear behind the edges
    nodes = nx.draw_networkx_nodes(display_G, pos, 
                         node_size=node_size, 
                         node_color=node_colors, 
                         alpha=0.8,
                         ax=ax)
    
    # Store edge curves for label positioning
    edge_curves = {}
    
    # Draw forward edges
    for source, target in [e for e in display_edges if e in forward_edges]:
        style = edge_style_mapping.get('forward', 
                                      {'color': 'blue', 'style': 'solid', 'width': 2})
        
        # Use consistent curve direction based on node positions
        # This ensures forward and backward edges mirror each other
        dx = pos[target][0] - pos[source][0]
        dy = pos[target][1] - pos[source][1]
        
        # Determine if the edge is predominantly horizontal or vertical
        is_horizontal = abs(dx) > abs(dy)
        
        # Choose curve direction based on orientation and direction of flow
        if is_horizontal:
            # For horizontal edges, forward flows curve down
            rad = -0.3 if dx > 0 else 0.3
        else:
            # For vertical edges, forward flows curve right
            rad = 0.3 if dy > 0 else -0.3
            
        # Store the curve radius for label positioning
        edge_curves[(source, target)] = rad
        
        ax.annotate("", 
                  xy=pos[target], xycoords='data',
                  xytext=pos[source], textcoords='data',
                  arrowprops=dict(
                      arrowstyle="->", 
                      color=style.get('color', 'blue'),
                      linestyle='-' if style.get('style', 'solid') == 'solid' else '--',
                      linewidth=style.get('width', 2),
                      alpha=style.get('alpha', 0.8),
                      mutation_scale=style.get('mutation_scale', 15),
                      connectionstyle=f"arc3,rad={rad}",
                      shrinkA=node_radius,  # Use calculated node radius for proper shrinkage
                      shrinkB=node_radius   # Use calculated node radius for proper shrinkage
                  ))
    
    # Draw backward edges
    for source, target in [e for e in display_edges if e in backward_edges]:
        style = edge_style_mapping.get('backward', 
                                       {'color': 'red', 'style': 'dashed', 'width': 2})
        
        # Use consistent curve direction based on node positions
        # This ensures forward and backward edges mirror each other
        dx = pos[target][0] - pos[source][0]
        dy = pos[target][1] - pos[source][1]
        
        # Determine if the edge is predominantly horizontal or vertical
        is_horizontal = abs(dx) > abs(dy)
        
        # Choose curve direction based on orientation and direction of flow
        # Opposite of what we did for forward edges to create mirroring effect
        if is_horizontal:
            # For horizontal edges, backward flows curve up (opposite of forward)
            rad = 0.3 if dx > 0 else -0.3
        else:
            # For vertical edges, backward flows curve left (opposite of forward)
            rad = -0.3 if dy > 0 else 0.3
        
        # Store the curve radius for label positioning
        edge_curves[(source, target)] = rad
            
        ax.annotate("", 
                  xy=pos[target], xycoords='data',
                  xytext=pos[source], textcoords='data',
                  arrowprops=dict(
                      arrowstyle="->", 
                      color=style.get('color', 'red'),
                      linestyle='-' if style.get('style', 'dashed') == 'solid' else '--',
                      linewidth=style.get('width', 2),
                      alpha=style.get('alpha', 0.8),
                      mutation_scale=style.get('mutation_scale', 15),
                      connectionstyle=f"arc3,rad={rad}",
                      shrinkA=node_radius,  # Use calculated node radius for proper shrinkage  
                      shrinkB=node_radius   # Use calculated node radius for proper shrinkage
                  ))
    
    # Draw node labels (on top of nodes, after edges)
    node_labels = {}
    for node in display_G.nodes():
        if custom_labels and node in custom_labels:
            node_labels[node] = custom_labels[node]
        elif abbreviated_names and short_labels and node in abbreviated_names:
            node_labels[node] = abbreviated_names[node]
        else:
            node_labels[node] = node
    
    nx.draw_networkx_labels(display_G, pos, labels=node_labels,
                          font_size=14, font_weight='bold', ax=ax)
    
    # Add edge labels
    edge_labels = {
        ('arvl', 'dcsn'): 'arvl_to_dcsn',
        ('dcsn', 'cntn'): 'dcsn_to_cntn',
        ('cntn', 'dcsn'): 'cntn_to_dcsn',
        ('dcsn', 'arvl'): 'dcsn_to_arvl',
    }
    
    # Position edge labels if requested
    if show_edge_labels:
        for (source, target), label in edge_labels.items():
            if display_G.has_edge(source, target):
                # Get the curvature value for this edge
                rad = edge_curves.get((source, target), 0)
                
                # For curved edges, we need to calculate point along the curve
                # Start with the midpoint between nodes
                midpoint_x = (pos[source][0] + pos[target][0]) / 2
                midpoint_y = (pos[source][1] + pos[target][1]) / 2
                
                # Direction from source to target
                dx = pos[target][0] - pos[source][0]
                dy = pos[target][1] - pos[source][1]
                
                # Normalize direction
                length = np.sqrt(dx**2 + dy**2)
                if length > 0:
                    dx = dx / length
                    dy = dy / length
                
                # Perpendicular direction (rotated 90 degrees)
                perp_dx = -dy
                perp_dy = dx
                
                # Get the type of edge (forward or backward) for better label positioning
                is_backward = (source, target) in backward_edges
                
                # Calculate offset distances based on the edge type and length
                # Use larger offsets for better separation
                along_offset = 0.0  # Position along the edge (0 = middle)
                
                if is_backward:
                    # For backward edges, move labels away from the center
                    along_offset = -0.1  # Slightly towards the source
                    perp_offset = 0.25 * abs(rad) * length  # Increased perpendicular offset
                else:
                    # For forward edges, move labels away from the center in the opposite direction
                    along_offset = 0.1   # Slightly towards the target
                    perp_offset = 0.25 * abs(rad) * length  # Increased perpendicular offset
                
                # Adjust midpoint along the edge based on along_offset
                midpoint_x += dx * along_offset * length
                midpoint_y += dy * along_offset * length
                
                # Apply perpendicular offset based on curve direction
                if rad > 0:
                    label_x = midpoint_x + perp_dx * perp_offset
                    label_y = midpoint_y + perp_dy * perp_offset
                else:
                    label_x = midpoint_x - perp_dx * perp_offset
                    label_y = midpoint_y - perp_dy * perp_offset
                
                # Create a text box with better visibility
                ax.text(label_x, label_y, label, 
                        bbox=dict(facecolor='white', edgecolor='black', alpha=0.9, pad=3, boxstyle="round,pad=0.3"),
                        horizontalalignment='center', 
                        verticalalignment='center',
                        fontsize=10,
                        fontweight='bold',
                        zorder=5)  # Ensure labels appear over edges
    
    # Add legend
    legend_handles = []
    
    if any(e in forward_edges for e in display_edges):
        fwd_style = edge_style_mapping.get('forward', 
                                          {'color': 'blue', 'style': 'solid', 'width': 2})
        forward_patch = plt.Line2D([0], [0], 
                                  color=fwd_style.get('color', 'blue'),
                                  linestyle='-' if fwd_style.get('style', 'solid') == 'solid' else '--',
                                  linewidth=fwd_style.get('width', 2),
                                  label='Forward Flow')
        legend_handles.append(forward_patch)
    
    if any(e in backward_edges for e in display_edges):
        bck_style = edge_style_mapping.get('backward', 
                                           {'color': 'red', 'style': 'dashed', 'width': 2})
        backward_patch = plt.Line2D([0], [0], 
                                   color=bck_style.get('color', 'red'),
                                   linestyle='-' if bck_style.get('style', 'dashed') == 'solid' else '--',
                                   linewidth=bck_style.get('width', 2),
                                   label='Backward Flow')
        legend_handles.append(backward_patch)
    
    if legend_handles:
        ax.legend(handles=legend_handles, loc=legend_loc)
    
    # Set title
    layout_name = 'Spring' if layout == 'spring' else 'Fixed'
    ax.set_title(f'ModCraft Stage Graph ({layout_name} Layout)')
    ax.axis('off')
    plt.tight_layout()
    
    # Save figure if filename is provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    # Print graph information
    print("\nStage Graph Analysis:")
    print(f"Number of nodes (perches): {display_G.number_of_nodes()}")
    print(f"Number of edges (movers): {display_G.number_of_edges()}")
    
    if exclude_cntn_to_arvl:
        print("Note: cntn_to_arvl edge excluded from visualization")
    
    # Check for cycles
    try:
        cycles = list(nx.simple_cycles(display_G))
        if cycles:
            print(f"Graph contains {len(cycles)} cycles:")
            for i, cycle in enumerate(cycles):
                print(f"  Cycle {i+1}: {' -> '.join(cycle)} -> {cycle[0]}")
        else:
            print("Graph is acyclic (no cycles)")
    except Exception as e:
        print(f"Could not check for cycles: {e}")
    
    return fig  # Return the figure for further customization if needed

# Example usage:
"""
# Define custom node colors
node_colors = {
    'arvl': '#377eb8',     # Blue
    'dcsn': '#4daf4a',     # Green
    'cntn': '#e41a1c'      # Red
}

# Define custom edge styles - compatible with ModelSequence
edge_styles = {
    'forward': {'color': 'blue', 'style': 'solid', 'width': 1.5, 'alpha': 0.7, 'mutation_scale': 15},
    'backward': {'color': 'purple', 'style': 'dashed', 'width': 1.5, 'alpha': 0.7, 'mutation_scale': 15}
}

# Call the visualization function
fig = visualize_stage_graph(
    stage,
    edge_type='both',
    layout='spring',
    node_color_mapping=node_colors,
    edge_style_mapping=edge_styles,
    show_edge_labels=True
)
plt.show()
"""