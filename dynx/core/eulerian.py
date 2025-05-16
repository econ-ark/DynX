"""
Eulerian circuit checking functionality for CircuitCraft.

This module provides methods to check if a CircuitBoard forms an Eulerian circuit,
particularly one that starts at a terminal perch, traverses through backward movers,
and returns to the terminal perch via forward movers.
"""

import networkx as nx
from .circuit_board import CircuitBoard

def is_eulerian_circuit(circuit: CircuitBoard) -> bool:
    """
    Check if a circuit forms an Eulerian cycle from terminal perches through
    backward movers and back via forward movers.
    
    In CircuitCraft v1.2.4+, the combined forward and backward sub-graphs must form
    an Eulerian cycle for a full back-then-forward solution to work correctly.
    
    Parameters
    ----------
    circuit : CircuitBoard
        The circuit to check
        
    Returns
    -------
    bool
        True if the circuit forms an Eulerian cycle, False otherwise
    """
    # Check if both graphs have edges
    if not circuit.backward_graph.edges() or not circuit.forward_graph.edges():
        return False  # Need both backward and forward edges for a complete circuit
    
    # Get the terminal perches in the backward graph
    # Terminal perch for the entire circuit is the backward graph's initial node
    terminal_perches = circuit._get_terminal_perches("backward")
    if not terminal_perches:
        return False  # No terminal perches, cannot be Eulerian
    
    # Create a combined graph with both backward and forward edges
    # but with appropriate attributes to distinguish them
    combined_graph = nx.DiGraph()
    
    # Add nodes from either graph (they should have the same nodes)
    combined_graph.add_nodes_from(circuit.backward_graph.nodes())
    
    # Add edges with attributes indicating direction
    for u, v, data in circuit.backward_graph.edges(data=True):
        combined_graph.add_edge(u, v, edge_type="backward")
    
    for u, v, data in circuit.forward_graph.edges(data=True):
        combined_graph.add_edge(u, v, edge_type="forward")
    
    # For an Eulerian circuit in a directed graph:
    # 1. All nodes must have equal in-degree and out-degree
    # 2. All nodes must be in a single strongly connected component
    
    # Check in-degree equals out-degree for all nodes
    for node in combined_graph.nodes():
        if combined_graph.in_degree(node) != combined_graph.out_degree(node):
            return False
    
    # Check if graph is strongly connected (one SCC containing all nodes)
    if not nx.is_strongly_connected(combined_graph):
        return False
    
    # If we have terminal perches, check that there is a valid Eulerian cycle
    # that starts at a terminal perch, follows backward edges, and returns via forward edges
    for terminal_perch in terminal_perches:
        # See if we can find a valid path
        path = find_backward_forward_path(combined_graph, terminal_perch)
        if path:
            return True
    
    return False

def find_backward_forward_path(graph, start_node):
    """
    Find a path that starts at the given node, follows backward edges until it can't anymore,
    then follows forward edges back to the start node.
    
    In CircuitCraft v1.2.4+ terminology:
    - start_node is the "terminal perch" for the entire circuit (backward graph's initial node)
    - The ending point of the backward path is the "initial perch" for the entire circuit
      (forward graph's initial node)
    
    Parameters
    ----------
    graph : nx.DiGraph
        The combined graph with edge_type attributes
    start_node : str
        The starting node (terminal perch)
    
    Returns
    -------
    list or None
        A valid path if found, None otherwise
    """
    # Create subgraphs for backward and forward edges
    backward_edges = [(u, v) for u, v, data in graph.edges(data=True) 
                      if data.get('edge_type') == 'backward']
    forward_edges = [(u, v) for u, v, data in graph.edges(data=True) 
                    if data.get('edge_type') == 'forward']
    
    backward_graph = nx.DiGraph()
    backward_graph.add_nodes_from(graph.nodes())
    backward_graph.add_edges_from(backward_edges)
    
    forward_graph = nx.DiGraph()
    forward_graph.add_nodes_from(graph.nodes())
    forward_graph.add_edges_from(forward_edges)
    
    # Find all nodes reachable from start_node via backward edges
    initial_perches = set()
    for node in backward_graph.nodes():
        # If we can reach this node from the start_node (terminal perch) via backward edges,
        # it's an initial perch from the perspective of our Eulerian path
        if nx.has_path(backward_graph, start_node, node):
            initial_perches.add(node)
    
    # For each initial perch, check if we can return to the terminal perch via forward edges
    for initial_perch in initial_perches:
        if nx.has_path(forward_graph, initial_perch, start_node):
            # We found a valid path: terminal_perch to initial_perch via backward edges,
            # then initial_perch back to terminal_perch via forward edges
            try:
                backward_path = nx.shortest_path(backward_graph, start_node, initial_perch)
                forward_path = nx.shortest_path(forward_graph, initial_perch, start_node)
                return backward_path + forward_path[1:]  # Avoid duplicating the initial_perch
            except nx.NetworkXNoPath:
                # This should not happen given our checks, but handle it anyway
                continue
    
    return None

def find_eulerian_path(circuit: CircuitBoard):
    """
    Find an Eulerian path in the circuit, starting from a terminal perch,
    going through all backward movers, and returning via forward movers.
    
    Parameters
    ----------
    circuit : CircuitBoard
        The circuit to check
        
    Returns
    -------
    list or None
        A list of perch names forming the path, or None if no path exists
    """
    # Check if both graphs have edges
    if not circuit.backward_graph.edges() or not circuit.forward_graph.edges():
        return None  # Need both backward and forward edges for a complete path
    
    # Create a combined graph with both backward and forward edges
    combined_graph = nx.DiGraph()
    combined_graph.add_nodes_from(circuit.backward_graph.nodes())
    
    # Add edges with attributes indicating direction
    for u, v, data in circuit.backward_graph.edges(data=True):
        combined_graph.add_edge(u, v, edge_type="backward")
    
    for u, v, data in circuit.forward_graph.edges(data=True):
        combined_graph.add_edge(u, v, edge_type="forward")
    
    # Get the terminal perches in the backward graph
    terminal_perches = circuit._get_terminal_perches("backward")
    if not terminal_perches:
        return None  # No terminal perches, cannot form Eulerian path
    
    # Try to find a valid path for each terminal perch
    for terminal_perch in terminal_perches:
        path = find_backward_forward_path(combined_graph, terminal_perch)
        if path:
            return path
    
    return None

def visualize_eulerian_path(circuit: CircuitBoard, path=None):
    """
    Generate a visualization of the Eulerian path in the circuit.
    
    Parameters
    ----------
    circuit : CircuitBoard
        The circuit to visualize
    path : list, optional
        The Eulerian path to highlight. If None, attempt to find one.
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the visualization
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import networkx as nx
        from matplotlib.patches import FancyArrowPatch
    except ImportError:
        raise ImportError("Matplotlib is required for visualization. Install with 'pip install matplotlib'")
    
    if path is None:
        path = find_eulerian_path(circuit)
    
    # Create a figure with extra vertical space
    fig, ax = plt.subplots(figsize=(14, 12))  # Further increased figure size
    plt.rcParams['path.simplify'] = False  # Avoid simplifying complex paths
    plt.rcParams['path.simplify_threshold'] = 0.0  # No simplification
    
    # Create separate lists for backward and forward edges
    backward_edges = []
    forward_edges = []
    
    # Extract movers from the circuit
    for u, v, data in circuit.backward_graph.edges(data=True):
        mover = data.get("mover")
        backward_edges.append((u, v, mover))
    
    for u, v, data in circuit.forward_graph.edges(data=True):
        mover = data.get("mover")
        forward_edges.append((u, v, mover))
    
    # Collect all nodes
    all_nodes = set()
    for u, v, _ in backward_edges:
        all_nodes.add(u)
        all_nodes.add(v)
    for u, v, _ in forward_edges:
        all_nodes.add(u)
        all_nodes.add(v)
    
    # Create a dictionary to map node names to positions
    # Place nodes in a horizontal line with EXTREME SPACING
    pos = {}
    sorted_nodes = sorted(all_nodes)
    spacing = 7.0  # Much larger node spacing
    y_position = 0  # Center y-level for nodes
    
    for i, node in enumerate(sorted_nodes):
        pos[node] = (i * spacing, y_position)
    
    # Draw nodes
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Draw nodes as circles with increased size
    for node in sorted_nodes:
        circle = plt.Circle(pos[node], radius=0.6, facecolor='white', edgecolor='black', linewidth=2.5, zorder=5)
        ax.add_patch(circle)
        ax.text(pos[node][0], pos[node][1], node, ha='center', va='center', 
                fontsize=18, fontweight='bold', zorder=6)
    
    # If we have a path, highlight those edges
    path_edges = []
    if path and len(path) > 1:
        # Create a combined graph with edge types
        combined_graph = nx.DiGraph()
        
        # Add backward edges with type
        for u, v, mover in backward_edges:
            combined_graph.add_edge(u, v, edge_type="backward")
            
        # Add forward edges with type
        for u, v, mover in forward_edges:
            combined_graph.add_edge(u, v, edge_type="forward")
            
        # Extract edges from the path with their types
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            if combined_graph.has_edge(u, v):
                edge_type = combined_graph[u][v]["edge_type"]
                path_edges.append((u, v, edge_type))
    
    # First identify node pairs with both forward and backward edges between the same nodes
    bidirectional_pairs = set()
    backward_pairs = {(u, v) for u, v, _ in backward_edges}
    forward_pairs = {(u, v) for u, v, _ in forward_edges}
    
    for u, v in backward_pairs:
        if (v, u) in forward_pairs:  # If there's a corresponding forward edge in opposite direction
            bidirectional_pairs.add((u, v))
            bidirectional_pairs.add((v, u))
    
    # Offset for nodes in backward/forward arcs (vertical separation)
    offset_y = 0.5
    
    # Common settings for both edge types
    path_line_width_thick = 5.0  # Line width for path edges
    path_line_width_normal = 3.5  # Line width for normal edges
    arrow_size = 0.6  # Size of arrowhead
    arrow_pull_back = 0.15  # How far to pull back arrowhead from endpoint
    control_point_magnitude = 3.0  # Base magnitude for control point offset
    
    # Draw backward edges as manually constructed curves with arrows (similar to forward edges)
    for u, v, mover in backward_edges:
        # Determine if this is part of a bidirectional pair
        is_bidirectional = (u, v) in bidirectional_pairs
        
        # Check if this edge is part of the Eulerian path
        is_path_edge = (u, v, "backward") in path_edges
        
        # Calculate positions with offset
        pos_u = (pos[u][0], pos[u][1] + offset_y)
        pos_v = (pos[v][0], pos[v][1] + offset_y)
        
        # Calculate control point for quadratic curve - FORCE IT UPWARD
        edge_len = abs(sorted_nodes.index(u) - sorted_nodes.index(v))
        # For longer edges, scale curvature based on distance
        control_offset = min(control_point_magnitude, 1.0 + edge_len * 0.5)  # Positive for upward
        if is_bidirectional:
            control_offset *= 1.2  # More extreme for bidirectional
            
        # Calculate a control point above the midpoint
        mid_x = (pos_u[0] + pos_v[0]) / 2
        mid_y = (pos_u[1] + pos_v[1]) / 2 + control_offset  # Force upward
        
        # Create a Path object for the curved line
        from matplotlib.path import Path
        import matplotlib.patches as patches
        
        # Define the vertices of the quadratic Bezier curve
        verts = [
            pos_u,          # Start point
            (mid_x, mid_y), # Control point
            pos_v           # End point
        ]
        
        # Define the path codes
        codes = [
            Path.MOVETO,    # Start
            Path.CURVE3,    # Control point
            Path.CURVE3     # End point
        ]
        
        # Create the path
        path = Path(verts, codes)
        
        # Create the patch
        line_width = path_line_width_thick if is_path_edge else path_line_width_normal
        patch = patches.PathPatch(
            path, 
            facecolor='none', 
            edgecolor='blue', 
            lw=line_width,
            alpha=1.0 if is_path_edge else 0.8,
            zorder=2
        )
        ax.add_patch(patch)
        
        # Add an arrowhead - manual approach
        # Calculate the direction at the endpoint
        dx = pos_v[0] - mid_x
        dy = pos_v[1] - mid_y
        
        # Normalize the direction
        import numpy as np
        length = np.sqrt(dx**2 + dy**2)
        dx, dy = dx/length, dy/length
        
        # Calculate arrowhead points - forcing correct orientation
        arrow_tip = (pos_v[0] - dx * arrow_pull_back, pos_v[1] - dy * arrow_pull_back)  # Pull back from endpoint
        arrow_left = (arrow_tip[0] - arrow_size * (dx*0.7 - dy*0.7), 
                      arrow_tip[1] - arrow_size * (dy*0.7 + dx*0.7))
        arrow_right = (arrow_tip[0] - arrow_size * (dx*0.7 + dy*0.7), 
                       arrow_tip[1] - arrow_size * (dy*0.7 - dx*0.7))
        
        # Create a polygon for the arrowhead
        arrow_head = plt.Polygon([arrow_tip, arrow_left, arrow_right], 
                                 closed=True, fc='blue', ec='blue', 
                                 zorder=3, alpha=1.0 if is_path_edge else 0.8)
        ax.add_patch(arrow_head)
        
        # Add label
        if mover and hasattr(mover, 'comp') and mover.comp:
            if hasattr(mover.comp, '__name__'):
                label = mover.comp.__name__
            else:
                label = mover.comp.__class__.__name__
        else:
            label = f"{u}→{v}"
        
        # Position label above the arc
        label_y = mid_y + 1.0  # Ensure label is above the curve
        
        ax.text(mid_x, label_y, label, color='blue', fontsize=14, fontweight='bold',
                ha='center', va='center',
                bbox=dict(facecolor='white', alpha=1.0, edgecolor='blue', boxstyle='round,pad=0.7'),
                zorder=4)
    
    # Draw forward edges as manually constructed curves with arrows
    for u, v, mover in forward_edges:
        # Determine if this is part of a bidirectional pair
        is_bidirectional = (u, v) in bidirectional_pairs
        
        # Check if this edge is part of the Eulerian path
        is_path_edge = (u, v, "forward") in path_edges
        
        # Calculate positions with offset
        pos_u = (pos[u][0], pos[u][1] - offset_y)
        pos_v = (pos[v][0], pos[v][1] - offset_y)
        
        # Calculate control point for quadratic curve - FORCE IT DOWNWARD
        edge_len = abs(sorted_nodes.index(u) - sorted_nodes.index(v))
        # For longer edges, scale curvature based on distance, matching backward edge magnitude
        control_offset = -min(control_point_magnitude, 1.0 + edge_len * 0.5)  # Negative for downward
        if is_bidirectional:
            control_offset *= 1.2  # More extreme for bidirectional
            
        # Calculate a control point below the midpoint
        mid_x = (pos_u[0] + pos_v[0]) / 2
        mid_y = (pos_u[1] + pos_v[1]) / 2 + control_offset  # Force downward
        
        # Create a Path object for the curved line
        from matplotlib.path import Path
        import matplotlib.patches as patches
        
        # Define the vertices of the quadratic Bezier curve
        verts = [
            pos_u,          # Start point
            (mid_x, mid_y), # Control point
            pos_v           # End point
        ]
        
        # Define the path codes
        codes = [
            Path.MOVETO,    # Start
            Path.CURVE3,    # Control point
            Path.CURVE3     # End point
        ]
        
        # Create the path
        path = Path(verts, codes)
        
        # Create the patch
        line_width = path_line_width_thick if is_path_edge else path_line_width_normal
        patch = patches.PathPatch(
            path, 
            facecolor='none', 
            edgecolor='red', 
            lw=line_width,
            alpha=1.0 if is_path_edge else 0.8,
            zorder=2
        )
        ax.add_patch(patch)
        
        # Add an arrowhead - manual approach
        # Calculate the direction at the endpoint
        dx = pos_v[0] - mid_x
        dy = pos_v[1] - mid_y
        
        # Normalize the direction
        import numpy as np
        length = np.sqrt(dx**2 + dy**2)
        dx, dy = dx/length, dy/length
        
        # Calculate arrowhead points - forcing correct orientation
        arrow_tip = (pos_v[0] - dx * arrow_pull_back, pos_v[1] - dy * arrow_pull_back)  # Pull back from endpoint
        arrow_left = (arrow_tip[0] - arrow_size * (dx*0.7 - dy*0.7), 
                      arrow_tip[1] - arrow_size * (dy*0.7 + dx*0.7))
        arrow_right = (arrow_tip[0] - arrow_size * (dx*0.7 + dy*0.7), 
                       arrow_tip[1] - arrow_size * (dy*0.7 - dx*0.7))
        
        # Create a polygon for the arrowhead
        arrow_head = plt.Polygon([arrow_tip, arrow_left, arrow_right], 
                                 closed=True, fc='red', ec='red', 
                                 zorder=3, alpha=1.0 if is_path_edge else 0.8)
        ax.add_patch(arrow_head)
        
        # Add label
        if mover and hasattr(mover, 'comp') and mover.comp:
            if hasattr(mover.comp, '__name__'):
                label = mover.comp.__name__
            else:
                label = mover.comp.__class__.__name__
        else:
            label = f"{u}→{v}"
        
        # Position label below the arc
        label_y = mid_y - 1.0  # Ensure label is below the curve
        
        ax.text(mid_x, label_y, label, color='red', fontsize=14, fontweight='bold',
                ha='center', va='center', 
                bbox=dict(facecolor='white', alpha=1.0, edgecolor='red', boxstyle='round,pad=0.7'),
                zorder=4)
    
    # Add enhanced legend
    backward_patch = mpatches.Patch(color='blue', label='Backward Movers (comp values)')
    forward_patch = mpatches.Patch(color='red', label='Forward Movers (sim values)')
    
    legend_handles = [backward_patch, forward_patch]
    legend_labels = ['Backward Movers (comp values)', 'Forward Movers (sim values)']
    
    if path:
        path_patch = mpatches.Patch(edgecolor='black', facecolor='lightgray', 
                                    label='Eulerian Path', linewidth=2)
        legend_handles.append(path_patch)
        legend_labels.append('Eulerian Path')
    
    ax.legend(handles=legend_handles, labels=legend_labels, 
              loc='upper right', frameon=True, framealpha=1, 
              facecolor='white', edgecolor='black',
              fontsize=12)
    
    # Add title
    ax.set_title("Eulerian Circuit Visualization", fontsize=20, pad=25)
    
    # Set axis limits with EXTREME padding
    all_x = [p[0] for p in pos.values()]
    min_x, max_x = min(all_x), max(all_x)
    width = max_x - min_x
    ax.set_xlim(min_x - width*0.4, max_x + width*0.4)
    ax.set_ylim(-8, 8)  # Much more vertical space
    
    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    
    plt.tight_layout()
    
    return fig

def add_to_circuit_board():
    """
    Add Eulerian circuit checking functionality to CircuitBoard class.
    """
    from .circuit_board import CircuitBoard
    
    # Add Eulerian methods to CircuitBoard
    CircuitBoard.is_eulerian = is_eulerian_circuit
    CircuitBoard.find_eulerian_path = find_eulerian_path
    CircuitBoard.visualize_eulerian_path = visualize_eulerian_path 
    
    # Original finalize_model method
    original_finalize = CircuitBoard.finalize_model
    
    def finalize_model_with_eulerian_check(self):
        """
        Extended finalize_model method that includes Eulerian circuit check.
        """
        # Call the original method
        result = original_finalize(self)
        
        # Perform Eulerian check if we have both forward and backward edges
        if (self.backward_graph.edges() and self.forward_graph.edges() and 
            not is_eulerian_circuit(self)):
            import warnings
            warnings.warn(
                "This circuit is not Eulerian. A full back-then-forward solution "
                "may fail or produce inconsistent results. Consider restructuring "
                "the circuit to satisfy Eulerian properties."
            )
        
        return result
    
    # Replace the original method with the extended one
    CircuitBoard.finalize_model = finalize_model_with_eulerian_check

# Automatically add Eulerian functionality when the module is imported
add_to_circuit_board() 