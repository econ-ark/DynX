import pickle
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable

import networkx as nx
import numpy as np

from .perch import Perch
from .mover import Mover


class CircuitBoard:
    """
    A circuit-board in CircuitCraft represented as a directed graph.
    
    In CircuitCraft, circuits are represented as graphs where nodes can be any entity
    (such as Stages, Perches, or other objects) and edges are Movers that facilitate
    data transfer between nodes.
    
    The graph maintains both backward and forward edges:
    - Backward edges: For operations like Coleman solving (successor to predecessor)
    - Forward edges: For operations like push-forward (predecessor to successor)
    
    Status flags track the state of the circuit board.
    """
    
    def __init__(self, name: str = "circuit_board"):
        """
        Initialize a circuit board.
        
        Parameters
        ----------
        name : str, optional
            Name of the circuit board, default is "circuit_board".
        """
        self.name = name
        self.perches: Dict[str, Perch] = {}
        
        # Use two separate directed graphs for backward and forward operations
        self.backward_graph = nx.DiGraph()
        self.forward_graph = nx.DiGraph()
        
        # Main graph for general use (can store any type of node)
        self.graph = nx.MultiDiGraph()
        
        # Status flags
        self.status_flags = {
            "eulerian": False # Default, can be set by specific implementations
        }
    
    def add_perch(self, perch: Perch) -> None:
        """
        Add a perch to the circuit board.
        
        This method is primarily used by the Stage class. In a general CircuitBoard,
        nodes can be any entity, not just perches.
        
        Parameters
        ----------
        perch : Perch
            The perch to add.
            
        Raises
        ------
        ValueError
            If a perch with the same name already exists.
        """
        if perch.name in self.perches:
            raise ValueError(f"Perch with name '{perch.name}' already exists")
        
        self.perches[perch.name] = perch
        
        # Add the perch as a node to the graphs
        self.backward_graph.add_node(perch.name)
        self.forward_graph.add_node(perch.name)
        self.graph.add_node(perch.name, perch=perch)
    
    def add_node(self, node_id: str, **attrs) -> None:
        """
        Add a general node to the circuit board.
        
        This method allows adding any type of node to the CircuitBoard graphs.
        
        Parameters
        ----------
        node_id : str
            Identifier for the node.
        **attrs
            Additional attributes to store with the node.
            
        Raises
        ------
        ValueError
            If a node with the same ID already exists.
        """
        if node_id in self.graph:
            raise ValueError(f"Node with ID '{node_id}' already exists")
        
        # Add the node to all graphs
        self.backward_graph.add_node(node_id, **attrs)
        self.forward_graph.add_node(node_id, **attrs)
        self.graph.add_node(node_id, **attrs)
    
    def add_mover(self, source_name: str, target_name: str,
                 map_data: Optional[Dict[str, Any]] = None,
                 parameters: Optional[Dict[str, Any]] = None,
                 numerical_hyperparameters: Optional[Dict[str, Any]] = None,
                 source_key: Optional[str] = None, 
                 source_keys: Optional[List[str]] = None,
                 target_key: Optional[str] = None, 
                 stage_name: Optional[str] = None,
                 edge_type: str = "forward",
                 model: Optional[Any] = None) -> Mover:
        """
        Add a mover to the circuit board.
        
        Creates a connection between two nodes in the graph.
        Nodes can be any entities (Stages, Perches, etc.) that have been added to the graph.
        
        Parameters
        ----------
        source_name : str
            Name/ID of the source node.
        target_name : str
            Name/ID of the target node.
        map_data : Dict[str, Any], optional
            Legacy parameter. Use model instead.
        parameters : Dict[str, Any], optional
            Legacy parameter. Include in model instead.
        numerical_hyperparameters : Dict[str, Any], optional
            Legacy parameter. Include in model instead.
        source_key : str, optional
            DEPRECATED: Use source_keys instead.
            Key from source to use in the operation.
        source_keys : List[str], optional
            Keys from source to use in the operation.
            Default is ["up"] for backward movers and ["down"] for forward movers.
        target_key : str, optional
            Key in target where the result will be stored.
            Default is "up" for backward movers and "down" for forward movers.
        edge_type : str, optional
            Type of mover: "forward" or "backward".
            Default is "forward".
        model : Any, optional
            Model representation for this mover. Takes precedence over map_data.
            
        Returns
        -------
        Mover
            The created mover object
            
        Raises
        ------
        ValueError
            If either node doesn't exist in the graph.
        """
        # Check if source and target exist in the graph, rather than in perches
        if not self.graph.has_node(source_name):
            raise ValueError(f"Source node '{source_name}' doesn't exist in the graph")
        if not self.graph.has_node(target_name):
            raise ValueError(f"Target node '{target_name}' doesn't exist in the graph")
        
        # Get the appropriate graph
        graph = self._get_graph(edge_type)
        
        # Handle deprecated source_key parameter
        if source_key is not None:
            if source_keys is not None:
                raise ValueError("Cannot provide both source_key and source_keys. Use source_keys only.")
            source_keys = [source_key]
            
        # Set default source_keys and target_key if not provided
        if source_keys is None:
            if edge_type == "backward":
                # For backward movers: Source = up data of the preceding node
                source_keys = ["up"]
            else:  # forward
                # For forward movers: Source = down data from the preceding node
                source_keys = ["down"]
            
        if target_key is None:
            if edge_type == "backward":
                # For backward movers: Target = up data of the succeeding node
                target_key = "up"
            else:  # forward
                # For forward movers: Target = down data of the succeeding node
                target_key = "down"
                
        # Handle legacy parameters by merging them into the model if needed
        if map_data is not None and model is None:
            model = map_data
            
        if (parameters or numerical_hyperparameters) and isinstance(model, dict):
            # Make a copy of the model to avoid modifying the original
            model_copy = model.copy() if model else {}
            
            if parameters:
                model_copy['parameters'] = parameters
            if numerical_hyperparameters:
                model_copy['numerical_hyperparameters'] = numerical_hyperparameters
                
            model = model_copy
                
        # Create the mover
        mover = Mover(
            source_name=source_name,
            target_name=target_name,
            edge_type=edge_type,
            model=model,
            source_keys=source_keys,
            target_key=target_key, 
            stage_name=stage_name
        )
        
        # Add the edge with the mover object as an attribute
        # Check if edge exists first; if so, update it rather than create a new one
        if graph.has_edge(source_name, target_name):
            # Use direct attribute assignment for existing edge
            graph[source_name][target_name]['mover'] = mover
        else:
            # Create new edge with mover attribute
            graph.add_edge(source_name, target_name, mover=mover)
            
        # Also add to the multi-graph with a key for multi-edge support
        mover_name = getattr(mover, 'name', f"{source_name}_to_{target_name}_{edge_type}")
        self.graph.add_edge(source_name, target_name, key=mover_name, mover=mover)
        
        return mover
    
    def set_mover_comp(self, source_name: str, target_name: str, edge_type: str, comp_func: Callable) -> None:
        """
        Set the computational method for a mover edge.
        
        Parameters
        ----------
        source_name : str
            Name/ID of the source node.
        target_name : str
            Name/ID of the target node.
        edge_type : str
            Type of mover: "forward" or "backward".
        comp_func : Callable
            The computational function that will transform data.
            
        Raises
        ------
        ValueError
            If the mover doesn't exist.
        """
        graph = self._get_graph(edge_type)
        
        if not graph.has_edge(source_name, target_name):
            raise ValueError(f"{edge_type} mover from '{source_name}' to '{target_name}' doesn't exist")
            
        # Get the mover from edge data
        mover = graph[source_name][target_name]["mover"]
        mover.set_comp(comp_func)
        print(f"Mover {source_name} set with comp {comp_func}")
        
    
    def _get_graph(self, edge_type: str) -> nx.DiGraph:
        """Get the appropriate graph based on edge type."""
        if edge_type == "forward":
            return self.forward_graph
        elif edge_type == "backward":
            return self.backward_graph
        else:
            raise ValueError(f"Unknown edge type: {edge_type}")
    
    def finalize_model(self) -> None:
        """
        Indicate that all perches and movers have been created.
        This method is now primarily a placeholder or for future extension.
        """
        pass # Keep the method signature for compatibility if needed
    
    def _get_terminal_perches(self, edge_type: str) -> List[str]:
        """Get terminal perches (no outgoing edges) for the specified edge type."""
        graph = self._get_graph(edge_type)
        return [n for n in graph.nodes() if graph.out_degree(n) == 0]
    
    def _get_initial_perches(self, edge_type: str) -> List[str]:
        """Get initial perches (no incoming edges) for the specified edge type."""
        graph = self._get_graph(edge_type)
        return [n for n in graph.nodes() if graph.in_degree(n) == 0]
    
    def create_comps_from_maps(self, comp_factory: Callable[[Any], Callable]) -> None:
        """
        Create computational methods (comps) from models for all movers.
        
        Parameters
        ----------
        comp_factory : Callable
            Function that takes a model and returns a comp callable.
        """
        # Process backward movers
        for source, target, data in self.backward_graph.edges(data=True):
            mover = data["mover"]
            if mover.has_model and not mover.has_comp:
                mover.create_comp_from_map(comp_factory)
        
        # Process forward movers
        for source, target, data in self.forward_graph.edges(data=True):
            mover = data["mover"]
            if mover.has_model and not mover.has_comp:
                mover.create_comp_from_map(comp_factory)
    
    def execute_mover(self, source_name: str, target_name: str, edge_type: str = "forward") -> Any:
        """
        Execute a single mover in the circuit.
        
        This method is primarily intended for execution between perch nodes.
        For non-perch nodes, make sure to handle data extraction and application 
        according to your node types.
        
        Parameters
        ----------
        source_name : str
            Name/ID of the source node.
        target_name : str
            Name/ID of the target node.
        edge_type : str
            Type of mover: "forward" or "backward".
            
        Returns
        -------
        Any
            Result from executing the mover.
            
        Raises
        ------
        ValueError
            If the mover doesn't exist or has no comp method.
        """
        graph = self._get_graph(edge_type)
        
        if not graph.has_edge(source_name, target_name):
            raise ValueError(f"{edge_type} mover from '{source_name}' to '{target_name}' doesn't exist")
        
        # Get the mover from edge data
        mover = graph[source_name][target_name]["mover"]
        
        if not mover.has_comp:
            raise ValueError(f"{edge_type} mover from '{source_name}' to '{target_name}' has no comp method")
        
        # Check if source and target are perches
        # If they are, use perch methods to get and set data
        source_is_perch = source_name in self.perches
        target_is_perch = target_name in self.perches
        
        # Extract data from source based on source_keys
        input_data = {}
        
        if source_is_perch:
            # Source is a perch, use perch methods
            source_perch = self.perches[source_name]
            for key in mover.source_keys:
                input_data[key] = source_perch.get_data(key)
        else:
            # Source is not a perch, use node data from the graph
            source_node_data = graph.nodes[source_name]
            for key in mover.source_keys:
                # Try to get data from node attributes
                if key in source_node_data:
                    input_data[key] = source_node_data[key]
                else:
                    # Check if there's a custom data accessor method
                    source_obj = source_node_data.get('object')
                    if source_obj and hasattr(source_obj, 'get_data'):
                        input_data[key] = source_obj.get_data(key)
                    else:
                        # Default to None if data not found
                        input_data[key] = None
            
        # If there's only one source key, pass the value directly instead of a dictionary
        if len(mover.source_keys) == 1:
            input_data = input_data[mover.source_keys[0]]
            
        # Execute the mover's comp function
        result = mover.execute(input_data)
        
        # Apply the result to the target, if needed
        if result is not None:
            if target_is_perch:
                # Target is a perch, use perch methods
                target_perch = self.perches[target_name]
                
                if isinstance(result, dict):
                    # If mover has a target_key, store the entire dictionary there
                    if mover.target_key:
                        target_perch.set_data(mover.target_key, result)
                    else:
                        # Only if no target_key is specified, try to update individual keys
                        for key, value in result.items():
                            if key in target_perch.get_data_keys():
                                target_perch.set_data(key, value)
                else:
                    # If result is not a dictionary, update the target_key directly
                    if mover.target_key:
                        target_perch.set_data(mover.target_key, result)
            else:
                # Target is not a perch, update node data in the graph
                if isinstance(result, dict):
                    # If result is a dictionary, update multiple attributes
                    for key, value in result.items():
                        graph.nodes[target_name][key] = value
                else:
                    # If result is not a dictionary, update the target_key directly
                    if mover.target_key:
                        graph.nodes[target_name][mover.target_key] = result
                
                # If target node has a custom data setter method, use it
                target_node_data = graph.nodes[target_name]
                target_obj = target_node_data.get('object')
                if target_obj and hasattr(target_obj, 'set_data'):
                    if isinstance(result, dict):
                        for key, value in result.items():
                            target_obj.set_data(key, value)
                    elif mover.target_key:
                        target_obj.set_data(mover.target_key, result)
                
        return result
    
    def get_movers_dict(self, mover_type=None):
        """
        Get a dictionary mapping from source perch to list of target perches
        with associated movers.
        
        Parameters
        ----------
        mover_type : str, optional
            If provided, only include movers of this type (e.g., 'backward', 'forward')
            
        Returns
        -------
        dict
            Dictionary with source perch names as keys and lists of target perch names as values
        """
        movers_dict = {}
        
        # Find movers in the appropriate graph
        if mover_type == "backward":
            graph = self.backward_graph
        elif mover_type == "forward":
            graph = self.forward_graph
        else:
            # If no type specified, include both graphs
            backward_edges = list(self.backward_graph.edges(data=True))
            forward_edges = list(self.forward_graph.edges(data=True))
            all_edges = backward_edges + forward_edges
            
            for source, target, data in all_edges:
                if source not in movers_dict:
                    movers_dict[source] = []
                movers_dict[source].append(target)
            
            return movers_dict
        
        # Process edges from the specific graph
        for source, target, data in graph.edges(data=True):
            if source not in movers_dict:
                movers_dict[source] = []
            movers_dict[source].append(target)
            
        return movers_dict
    
    def _detect_value_change(self, previous_value: Any, current_value: Any) -> bool:
        """
        Detect if a value has changed, handling various data types appropriately.
        
        Parameters
        ----------
        previous_value : Any
            The previous value to compare
        current_value : Any
            The current value to compare
            
        Returns
        -------
        bool
            True if the value has changed, False otherwise
        """
        # Check for None transitions
        if previous_value is None and current_value is not None:
            return True
        elif current_value is None and previous_value is not None:
            return True
        
        # Handle NumPy arrays
        if (isinstance(previous_value, np.ndarray) and 
                isinstance(current_value, np.ndarray)):
            return not np.array_equal(previous_value, current_value)
        
        # Use __eq__ for objects that define it
        if hasattr(previous_value, "__eq__") and previous_value is not None:
            return previous_value != current_value
        
        # Fallback to id comparison for objects without proper equality
        return id(previous_value) != id(current_value)
    
    def solve_backward(self):
        """
        Solve all backward movers in the circuit.
        
        This performs backward operations in a single pass using a topological sort
        of the backward graph. For acyclic dependency networks, a single pass is
        sufficient to solve the circuit.
        
        Streamlined version that uses NetworkX capabilities for more efficient 
        processing.
        """
        # Check if any perch has a comp value - we need initial values
        initial_perches = [name for name, perch in self.perches.items() 
                          if perch.up is not None]
        if not initial_perches:
            raise RuntimeError("Cannot solve backwards: No perch has a comp value")
        
        print("Perches with initial comp values:", initial_perches)
        
        # Check for cycles explicitly before attempting topological sort
        if list(nx.simple_cycles(self.backward_graph)):
            raise RuntimeError("Backward graph contains cycles; cannot solve")
        
        # Create a topological sort of the backward graph
        try:
            # In backward graph: A->B means B's value depends on A's
            topo_order = list(nx.lexicographical_topological_sort(self.backward_graph))
            
            if not topo_order:
                raise RuntimeError("Cannot solve backwards: Backward graph is empty")
            
            # Track which perches have been updated
            updated_perches = set()
            
            # Single pass through the graph in topological order
            # This ensures that for each perch, all its dependencies are processed first
            print("Performing backward solve in single pass...")
            
            for perch_name in topo_order:
                # Get all the incoming edges (dependencies)
                for source in self.backward_graph.predecessors(perch_name):
                    source_perch = self.perches[source]
                    target_perch = self.perches[perch_name]
                    
                    # Get the mover for this edge
                    mover = self.backward_graph[source][perch_name].get("mover")
                    
                    # Skip if mover doesn't exist or has no comp method
                    if not mover or not mover.has_comp:
                        continue
                    
                    # Check if source has the required source key values
                    source_has_data = True
                    if mover.source_keys:
                        for key in mover.source_keys:
                            if source_perch.get_data(key) is None:
                                source_has_data = False
                                break
                    
                    if not source_has_data:
                        print(f"  Skipping {source} -> {perch_name}: "
                              f"Source lacks required data")
                        continue
                    
                    # Store previous value to detect changes (for reporting only)
                    previous_target_value = None
                    if mover.target_key:
                        previous_target_value = target_perch.get_data(mover.target_key)
                    
                    try:
                        # Execute the mover's comp function - maintains the principle of 
                        # passing raw values and getting raw values back
                        result = self.execute_mover(source, perch_name, "backward")
                        
                        # Check if the target value has changed (for reporting only)
                        current_target_value = None
                        if mover.target_key:
                            current_target_value = target_perch.get_data(mover.target_key)
                        
                        # Use the helper method to detect value changes
                        value_changed = self._detect_value_change(
                            previous_target_value, current_target_value)
                        
                        if value_changed:
                            #print(f"  Value updated: {source} -> {perch_name}: "
                            #      f"{previous_target_value} -> {current_target_value}")
                            updated_perches.add(perch_name)
                    
                    except Exception as e:
                        print(f"Error executing backward mover from {source} "
                              f"to {perch_name}: {e}")
            
            # Report on results
            if updated_perches:
                print(f"Backward solve updated {len(updated_perches)} perches: "
                      f"{sorted(updated_perches)}")
            else:
                print("No perches were updated during backward solve")
            
            # Check for perches that still have no comp value
            unresolved_perches = [
                name for name, perch in self.perches.items() 
                if perch.up is None and self.backward_graph.in_degree(name) > 0
            ]
            if unresolved_perches:
                print(f"Warning: {len(unresolved_perches)} perches still have no comp value: "
                      f"{sorted(unresolved_perches)}")
                # Analyze why these perches weren't updated
                for perch_name in unresolved_perches:
                    print(f"Perch {perch_name} was not updated. Analyzing dependencies:")
                    # Find all paths to this perch from perches with initial values
                    for initial in initial_perches:
                        try:
                            paths = list(nx.all_simple_paths(
                                self.backward_graph, initial, perch_name))
                            for path in paths:
                                print(f"  Dependency chain: {' -> '.join(path)}")
                                # Identify where the chain breaks
                                for i in range(len(path) - 1):
                                    src, tgt = path[i], path[i + 1]
                                    src_perch = self.perches[src]
                                    mover = self.backward_graph[src][tgt].get("mover")
                                    if src_perch.up is None:
                                        print(f"    Break at {src} -> {tgt}: "
                                              f"Source perch has no comp value")
                                        break
                                    elif not mover or not mover.has_comp:
                                        print(f"    Break at {src} -> {tgt}: No valid mover")
                                        break
                        except (nx.NetworkXNoPath, nx.NodeNotFound):
                            print(f"  No path from {initial} to {perch_name}")
            
            print("Backward solve completed successfully")
            
        except nx.NetworkXUnfeasible:
            # Graph has cycles - should be caught by the explicit check above
            raise RuntimeError("Backward graph contains cycles; cannot perform topological sort")
    
    def solve_forward(self) -> None:
        """
        Solve all forward movers in the circuit.
        
        This performs forward operations in a single pass using a topological sort
        of the forward graph. For acyclic dependency networks, a single pass is
        sufficient to simulate the circuit.
        
        Streamlined version that uses NetworkX capabilities for more efficient 
        processing.
        
        Raises
        ------
        RuntimeError
            If the circuit is not solvable or an error occurs during solving.
        """
        # Check if any perch has a comp value - otherwise we can't simulate
        if not any(perch.up is not None for perch in self.perches.values()):
            raise RuntimeError("Cannot simulate forward pass: No perch has a comp value.")
        
        # Find initial perches (those with both comp and sim values)
        initial_perches = []
        for name, perch in self.perches.items():
            if perch.up is not None and perch.down is not None:
                initial_perches.append(name)
        
        if not initial_perches:
            raise RuntimeError(
                "Cannot simulate forward pass: No perch has both comp and sim values.")
        
        print(f"Initial perches for forward solving: {initial_perches}")
        
        # Check for cycles explicitly
        if list(nx.simple_cycles(self.forward_graph)):
            raise RuntimeError("Forward graph contains cycles; cannot solve")
        
        # Get topological order for the forward graph
        try:
            # Use lexicographical topological sort for deterministic ordering
            topo_order = list(nx.lexicographical_topological_sort(self.forward_graph))
            
            # Mark nodes for tracking
            nx.set_node_attributes(self.forward_graph, False, "processed")
            
            # Track which perches have been updated
            updated_perches = set()
            
            # Single pass through the graph in topological order
            print("Performing forward solve in single pass...")
            
            for perch_name in topo_order:
                # Mark initial perches as processed
                if perch_name in initial_perches:
                    self.forward_graph.nodes[perch_name]["processed"] = True
                    continue
                    
                # Find all predecessors of this perch in the forward graph
                predecessors = list(self.forward_graph.predecessors(perch_name))
                
                # Skip if no predecessors
                if not predecessors:
                    continue
                    
                # Try each predecessor to see if we can compute this perch's sim value
                for pred in predecessors:
                    # Skip predecessors that don't have sim values
                    if self.perches[pred].down is None:
                        continue
                        
                    # Get the mover from predecessor to this perch
                    mover = self.forward_graph[pred][perch_name].get("mover")
                    if not mover or not mover.has_comp:
                        continue
                        
                    # Store previous value to detect changes (for reporting only)
                    previous_value = None
                    if mover.target_key:
                        previous_value = self.perches[perch_name].get_data(mover.target_key)
                    
                    # Execute the mover while maintaining the principle that mover.comp
                    # receives and returns raw values
                    try:
                        result = self.execute_mover(pred, perch_name, "forward")
                        
                        # Check if the target value has changed (for reporting only)
                        current_value = None
                        if mover.target_key:
                            current_value = self.perches[perch_name].get_data(
                                mover.target_key)
                        
  

                    except Exception as e:
                        print(f"Error executing forward mover from {pred} "
                              f"to {perch_name}: {e}")
            
            # Report on results
            if updated_perches:
                print(f"Forward solve updated {len(updated_perches)} perches: "
                      f"{sorted(updated_perches)}")
            else:
                print("No perches were updated during forward solve")
            
            # For debugging purposes, show any unprocessed nodes and their dependencies
            unprocessed = [
                n for n in self.forward_graph.nodes() 
                if self.forward_graph.nodes[n].get("processed") is False
            ]
            if unprocessed:
                print(f"Warning: {len(unprocessed)} perches were not processed:")
                for node in unprocessed:
                    print(f"- {node}")
                    # Show incoming edges
                    for pred in self.forward_graph.predecessors(node):
                        mover = self.forward_graph[pred][node].get("mover")
                        has_comp = mover.has_comp if mover else False
                        print(f"  <- {pred} (has sim: {self.perches[pred].down is not None}, "
                              f"mover has comp: {has_comp})")
                
                # Identify paths from initial perches
                for initial in initial_perches:
                    try:
                        paths = list(nx.all_simple_paths(
                            self.forward_graph, initial, node))
                        for path in paths:
                            print(f"  Path from initial perch: {' -> '.join(path)}")
                            # Identify where the chain breaks
                            for i in range(len(path) - 1):
                                src, tgt = path[i], path[i + 1]
                                src_perch = self.perches[src]
                                mover = self.forward_graph[src][tgt].get("mover")
                                if src_perch.down is None:
                                    print(f"    Break at {src} -> {tgt}: "
                                          f"Source perch has no sim value")
                                    break
                                elif not mover or not mover.has_comp:
                                    print(f"    Break at {src} -> {tgt}: No valid mover")
                                    break
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        continue
            
            print("Forward solve complete.")
            
        except nx.NetworkXError:
            raise RuntimeError(
                "Forward graph contains cycles; cannot perform topological sort")
    
    def solve(self):
        """
        Solve the circuit by running backward and forward solving in sequence.
        
        This method will:
        1. Solve backward to populate comp values for all perches
        2. Solve forward to populate sim values for all perches
        
        Returns
        -------
        bool
            True if the circuit was solved successfully, False otherwise.
            
        Raises
        ------
        RuntimeError
            If the circuit is not solvable or an error occurs during solving.
        """
        # Solve backward to compute comp values
        try:
            self.solve_backward()
        except Exception as e:
            print(f"Error during backward solving: {e}")
            return False
            
        # Solve forward to compute sim values
        try:
            self.solve_forward()
        except Exception as e:
            print(f"Error during forward solving: {e}")
            return False
            
        return True
    
    def get_perch_data(self, perch_name: str, key: str) -> Any:
        """
        Get data from a perch by key.
        
        Parameters
        ----------
        perch_name : str
            Name of the perch.
        key : str
            Key of the data to retrieve.
            
        Returns
        -------
        Any
            The requested data.
            
        Raises
        ------
        ValueError
            If the perch doesn't exist.
        KeyError
            If the key doesn't exist in the perch.
        """
        if perch_name not in self.perches:
            raise ValueError(f"Perch '{perch_name}' doesn't exist")
            
        return self.perches[perch_name].get_data(key)
    
    def create_transpose_connections(self, edge_type: str = "forward") -> List[Mover]:
        """
        Create transpose connections for all movers of the specified type.
        
        For each forward mover, create a corresponding backward mover, or vice versa.
        Maintains proper data flow conventions:
        - Forward movers: source.cntn.down -> target.arvl.down
        - Backward movers: source.arvl.up -> target.cntn.up
        
        Parameters
        ----------
        edge_type : str
            Type of movers to create transposes for: "forward" or "backward".
            Default is "forward" (creates backward transposes).
            
        Returns
        -------
        List[Mover]
            List of newly created transpose movers.
            
        Notes
        -----
        When creating a transpose of a forward edge:
        - Source becomes target, target becomes source
        - Data flows from arvl.up to cntn.up instead of cntn.down to arvl.down
        
        When creating a transpose of a backward edge:
        - Source becomes target, target becomes source
        - Data flows from cntn.down to arvl.down instead of arvl.up to cntn.up
        """
        source_graph = self._get_graph(edge_type)
        transpose_type = "backward" if edge_type == "forward" else "forward"
        target_graph = self._get_graph(transpose_type)
        
        created_movers = []
        
        # Process all edges in the source graph
        for source, target, data in source_graph.edges(data=True):
            mover = data.get("mover")
            if not mover:
                print(f"Warning: Edge from {source} to {target} has no mover")
                continue
                
            # Check if a transpose edge already exists
            if target_graph.has_edge(target, source):
                print(f"Transpose edge from {target} to {source} already exists, skipping")
                continue
            
            # Determine source and target keys for the transpose mover
            if edge_type == "forward":
                # Creating backward transpose of a forward mover
                # Original: cntn.down -> arvl.down
                # Transpose: arvl.up -> cntn.up
                source_keys = ["up"]
                target_key = "up"
            else:
                # Creating forward transpose of a backward mover
                # Original: arvl.up -> cntn.up
                # Transpose: cntn.down -> arvl.down
                source_keys = ["down"]
                target_key = "down"
            
            # Create model for the transpose mover if needed
            # For now, we'll just copy the original model, but this could be customized
            transpose_model = None
            if mover.has_model:
                transpose_model = mover.model  # Consider deep copying if needed
            
            # Create the transpose mover
            transpose_mover_name = f"{getattr(mover, 'name', 'mover')}_transpose"
            
            # Add the mover to the transpose graph
            self.add_mover(
                source_name=target,  # Original target becomes source
                target_name=source,  # Original source becomes target
                edge_type=transpose_type,
                model=transpose_model,
                source_keys=source_keys,
                target_key=target_key,
            )
            
            # Get the newly created mover
            if target_graph.has_edge(target, source):
                new_mover = target_graph[target][source].get("mover")
                if new_mover:
                    created_movers.append(new_mover)
                    print(f"Created transpose mover: {target} -> {source} ({transpose_type})")
            
        return created_movers
    
    def set_perch_data(self, perch_name: str, data: Dict[str, Any]) -> None:
        """
        Set data on a perch.
        
        Parameters
        ----------
        perch_name : str
            Name of the perch.
        data : Dict[str, Any]
            Dictionary of data to set on the perch.
            
        Raises
        ------
        ValueError
            If the perch doesn't exist.
        KeyError
            If any key doesn't exist in the perch.
        """
        if perch_name not in self.perches:
            raise ValueError(f"Perch '{perch_name}' doesn't exist")
            
        perch = self.perches[perch_name]
        for key, value in data.items():
            perch.set_data(key, value)
            
        # Adding data might make the circuit solvable
        self._check_solvability()
    
    def save(self, filepath: str) -> None:
        """
        Save the circuit to a file.
        
        This serializes the circuit state using pickle, ensuring that
        all contained objects are serializable.
        
        Parameters
        ----------
        filepath : str
            Path to save the circuit to.
            
        Raises
        ------
        RuntimeError
            If the circuit is not portable.
        """
        if not self.is_portable:
            raise RuntimeError("Circuit is not portable. Call make_portable() before saving.")
            
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
        except Exception as e:
            raise RuntimeError(f"Failed to save circuit: {str(e)}")
    
    @classmethod
    def load(cls, filepath: str) -> 'CircuitBoard':
        """
        Load a circuit from a file.
        
        Parameters
        ----------
        filepath : str
            Path to load the circuit from.
            
        Returns
        -------
        CircuitBoard
            The loaded circuit.
            
        Raises
        ------
        RuntimeError
            If loading fails.
        """
        try:
            with open(filepath, 'rb') as f:
                circuit = pickle.load(f)
                
            if not isinstance(circuit, cls):
                raise TypeError(f"Loaded object is not a {cls.__name__}")
                
            return circuit
        except Exception as e:
            raise RuntimeError(f"Failed to load circuit from {filepath}: {str(e)}")
    
    def make_portable(self, comp_factory: Optional[Callable] = None) -> None:
        """
        Make the circuit portable by ensuring all maps have comps.
        
        Parameters
        ----------
        comp_factory : Callable, optional
            Function that creates comps from maps. If None, a default is used.
        """
        # If no factory provided, create a default that just passes through the data
        if comp_factory is None:
            def default_factory(map_data):
                def comp(input_data: Dict[str, Any]) -> Dict[str, Any]:
                    # Simple identity function - just return the input
                    return input_data
                return comp
            comp_factory = default_factory
        
        # Create comps for all maps
        self.create_comps_from_maps(comp_factory)
    
    def __str__(self) -> str:
        """String representation of the circuit board."""
        perch_count = len(self.perches)
        backward_edge_count = self.backward_graph.number_of_edges()
        forward_edge_count = self.forward_graph.number_of_edges()
        
        # Removed status checks based on old flags
        # status = []
        # if self.has_model:
        #     status.append("modeled")
        # if self.is_portable:
        #     status.append("portable")
        # if self.is_solved:
        #     status.append("solved")
        # if self.is_simulated:
        #     status.append("simulated")
        # 
        # status_str = ", ".join(status) if status else "empty"
        
        # Simplified string representation without old status flags
        return (f"CircuitBoard({self.name}, {perch_count} perches, "
                f"{backward_edge_count} backward movers, "
                f"{forward_edge_count} forward movers)") 