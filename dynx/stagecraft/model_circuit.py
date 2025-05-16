from typing import List, Dict, Optional, Callable, Any, Union, Tuple, TYPE_CHECKING

# Import only what's needed directly
from dynx.core.mover import Mover
from dynx.core.perch import Perch
from dynx.core.circuit_board import CircuitBoard

# Import Period directly, not just for type checking
from dynx.stagecraft.period import Period
    
# We use Stage directly, so we need it at runtime
from dynx.stagecraft.stage import Stage

class ModelCircuit(CircuitBoard):
    """Manages an ordered sequence of Period sub-graphs and inter-period connections.

    This class holds the complete dynamic model structure over time.
    It orchestrates the overall solution process (backward/forward passes)
    by iterating through its Periods and handling inter-period data flow.

    Attributes
    ----------
    name : str
        Name of the model circuit.
    periods_list : List[Period]
        Ordered list of Period objects.
    periods_map : Dict[int, Period]
        Map from time index to Period object for quick lookup.
    inter_period_movers : List[Mover]
        List of Mover objects representing connections between stages in different periods.
    """
    def __init__(self, name: str, verbose: bool = False):
        """Initializes the ModelCircuit.

        Parameters
        ----------
        name : str
            Name of the model circuit.
        verbose : bool, optional
            Whether to print status messages, by default False
        """
        # Initialize parent CircuitBoard
        super().__init__(name=name)
        
        self.name = name
        self.periods_list: List[Period] = []
        self.periods_map: Dict[int, Period] = {}
        self.inter_period_movers: List[Mover] = []
        self.verbose = verbose
        if self.verbose:
            print(f"\n=== Creating ModelCircuit: {name} ===")

    def add_period(self, period: Period, verbose: Optional[bool] = None):
        """Adds a Period object to the sequence.

        Ensures periods are added in order and time indices are unique.
        Also incorporates the period's nodes and edges into the ModelCircuit's graphs.

        Parameters
        ----------
        period : Period
            The Period object to add.
        verbose : bool, optional
            Whether to print status messages. If None, uses the ModelCircuit's verbose flag.
        """
        # Use the provided verbose flag, or fall back to the class verbose flag
        verbose = verbose if verbose is not None else self.verbose
        
        if not isinstance(period, Period):
             raise TypeError(f"Expected a Period object, got {type(period)}")
        time_index = period.time_index
        if time_index in self.periods_map:
            raise ValueError(f"Period with time_index {time_index} already exists in ModelCircuit '{self.name}'.")

        self.periods_map[time_index] = period
        self.periods_list.append(period)
        # Keep the list sorted by time index
        self.periods_list.sort(key=lambda p: p.time_index)
        
        # Incorporate the period's graph structure into ModelCircuit's graphs
        self._incorporate_period_graph(period)
        if verbose:
            print(f"Added Period {time_index} to ModelCircuit '{self.name}'. Current order: {[p.time_index for p in self.periods_list]}")
        
    def _incorporate_period_graph(self, period: Period):
        """
        Incorporates a period's graph structure into this ModelCircuit's graphs.
        
        For each stage node in the period's graphs, creates a corresponding node in the
        ModelCircuit graphs with a prefix to ensure uniqueness. Also adds all edges.
        
        Parameters
        ----------
        period : Period
            The Period object whose graph structure should be incorporated.
        """
        time_index = period.time_index
        
        # Add nodes from the period's forward and backward graphs
        for stage_id in period.stages:
            # Create the global node ID for this stage
            global_node_id = f"p{time_index}:{stage_id}"
            
            # Get stage attributes to copy
            stage = period.stages[stage_id]
            
            # Add to both forward and backward graphs with stage reference
            self.forward_graph.add_node(global_node_id, 
                                       stage=stage,
                                       period=period,
                                       period_idx=time_index,
                                       stage_id=stage_id)
                                       
            self.backward_graph.add_node(global_node_id, 
                                        stage=stage,
                                        period=period,
                                        period_idx=time_index,
                                        stage_id=stage_id)
        
        # Add edges from the period's forward graph
        for source_id, target_id, edge_data in period.forward_graph.edges(data=True):
            source_global_id = f"p{time_index}:{source_id}"
            target_global_id = f"p{time_index}:{target_id}"
            
            # Copy all edge attributes
            edge_attr_dict = edge_data.copy()
            
            # Set the edge type explicitly based on mover's edge_type
            mover = edge_attr_dict.get('mover', None)
            if mover and hasattr(mover, 'edge_type'):
                edge_attr_dict['type'] = mover.edge_type
            elif mover and hasattr(mover, 'name') and '_transpose' in mover.name.lower():
                edge_attr_dict['type'] = 'backward'
            else:
                edge_attr_dict['type'] = 'forward'  # Default type
            
            # Copy all edge attributes
            self.forward_graph.add_edge(source_global_id, target_global_id, **edge_attr_dict)
            
        # Add edges from the period's backward graph
        for source_id, target_id, edge_data in period.backward_graph.edges(data=True):
            source_global_id = f"p{time_index}:{source_id}"
            target_global_id = f"p{time_index}:{target_id}"
            
            # Copy all edge attributes
            edge_attr_dict = edge_data.copy()
            
            # Set the edge type explicitly based on mover's edge_type
            mover = edge_attr_dict.get('mover', None)
            if mover and hasattr(mover, 'edge_type'):
                edge_attr_dict['type'] = mover.edge_type
            elif mover and hasattr(mover, 'name') and ('_transpose' in mover.name.lower() or '_t' in mover.name.lower()):
                edge_attr_dict['type'] = 'backward'
            else:
                edge_attr_dict['type'] = 'backward'  # Default type for backward graph
            
            # Copy all edge attributes
            self.backward_graph.add_edge(source_global_id, target_global_id, **edge_attr_dict)

    def get_period(self, time_index: int) -> Period:
        """Retrieves a Period by its time index.

        Parameters
        ----------
        time_index : int
            The time index of the Period to retrieve.

        Returns
        -------
        Period
            The requested Period object.

        Raises
        -------
        KeyError
            If no Period with the given time_index exists.
        """
        if time_index not in self.periods_map:
            raise KeyError(f"Period with time_index {time_index} not found in ModelCircuit '{self.name}'.")
        return self.periods_map[time_index]

    def add_inter_period_connection(
        self,
        source_period: Period,
        target_period: Period,
        source_stage: Stage,
        target_stage: Stage,
        source_perch_attr: str = "cntn",
        target_perch_attr: str = "arvl",
        forward_comp: Optional[Callable] = None,
        mover_name: Optional[str] = None,
        create_transpose: bool = True,
        verbose: Optional[bool] = None,
        **mover_kwargs
    ) -> Mover:
        """
        Connect a stage in one period to a stage in another.
        
        Creates an inter-period connection (mover) between specified perches
        of stages in different periods.
        
        Parameters
        ----------
        source_period : Period
            The source period containing the source stage.
        target_period : Period
            The target period containing the target stage.
        source_stage : Stage
            The source stage containing the source perch.
        target_stage : Stage
            The target stage containing the target perch.
        source_perch_attr : str, optional
            Attribute name of the source perch in the source stage. Default is "cntn".
        target_perch_attr : str, optional
            Attribute name of the target perch in the target stage. Default is "arvl".
        forward_comp : Callable, optional
            Computational function for the mover. Defaults to None.
        mover_name : str, optional
            Name for the mover. If None, a default name is generated.
        create_transpose : bool, optional
            Whether to automatically create a transpose connection. Default is True.
        verbose : bool, optional
            Whether to print status messages. If None, uses the ModelCircuit's verbose flag.
        **mover_kwargs
            Additional arguments for the mover.

        Returns
        -------
        Mover
            The newly created mover.

        Raises
        ------
        AttributeError
            If the source_perch_attr or target_perch_attr doesn't exist.
        """
        # Use the provided verbose flag, or fall back to the class verbose flag
        verbose = verbose if verbose is not None else self.verbose
        
        # Get source and target periods by time index
        source_period_idx = source_period.time_index
        target_period_idx = target_period.time_index
        
        # Get stage IDs for reference
        source_stage_id = next((k for k, v in source_period.stages.items() if v == source_stage), None)
        target_stage_id = next((k for k, v in target_period.stages.items() if v == target_stage), None)
        
        if source_stage_id is None:
            raise ValueError(f"Source stage not found in period {source_period_idx}")
        if target_stage_id is None:
            raise ValueError(f"Target stage not found in period {target_period_idx}")
        
        # Verify the perches exist but don't keep direct references
        source_perch_obj = getattr(source_stage, source_perch_attr, None)
        target_perch_obj = getattr(target_stage, target_perch_attr, None)
        
        if source_perch_obj is None:
            raise AttributeError(f"Source stage does not have a '{source_perch_attr}' perch")
        if target_perch_obj is None:
            raise AttributeError(f"Target stage does not have a '{target_perch_attr}' perch")
        
        # Generate a default mover name if not provided
        if mover_name is None:
            mover_name = f"{source_stage_id}_{source_period_idx}_to_{target_stage_id}_{target_period_idx}"
        
        # Create global node IDs
        source_node_id = f"p{source_period_idx}:{source_stage_id}"
        target_node_id = f"p{target_period_idx}:{target_stage_id}"
        
        # Define source keys and target key for forward mover
        # Forward mover: source_keys = ["cntn"]["sol", "dist"], target_key = ["arvl"]["dist"]
        source_keys = [f"{source_perch_attr}.sol", f"{source_perch_attr}.dist"]
        target_key = f"{target_perch_attr}.dist"
        
        # Create forward mover with global stage IDs as names
        forward_mover = Mover(
            source_name=source_node_id, # Use global ID
            target_name=target_node_id, # Use global ID
            edge_type="forward",
            comp=forward_comp,
            source_keys=source_keys,
            target_key=target_key,
            **mover_kwargs
        )
        
        # Set additional attributes (only the essential ones)
        forward_mover.name = mover_name
        forward_mover.source_period_idx = source_period_idx
        forward_mover.target_period_idx = target_period_idx
        
        # Add the edge to the forward graph
        self.forward_graph.add_edge(source_node_id, target_node_id, 
                                   mover=forward_mover,
                                   source_perch_attr=source_perch_attr, 
                                   target_perch_attr=target_perch_attr,
                                   inter_period=True,
                                   source_period_idx=source_period_idx,
                                   target_period_idx=target_period_idx)
        
        # Add to inter_period_movers list
        self.inter_period_movers.append(forward_mover)
        if verbose:
            print(f"Added Inter-Period Mover: {mover_name} ({source_period_idx}:{source_stage_id}.{source_perch_attr} -> {target_period_idx}:{target_stage_id}.{target_perch_attr})")
        
        # Create transpose connection if requested
        if create_transpose:
            transpose_name = f"{mover_name}_transpose"
            
            # Define source keys and target key for backward mover
            # Backward mover: source_keys = ["arvl"]["sol"], target_key = ["cntn"]["sol"]
            transpose_source_keys = [f"{target_perch_attr}.sol"]
            transpose_target_key = f"{source_perch_attr}.sol"
            
            # Create backward mover
            backward_mover = Mover(
                source_name=target_node_id,
                target_name=source_node_id,
                edge_type="backward",
                comp=None,  # No comp function for now
                source_keys=transpose_source_keys,
                target_key=transpose_target_key,
                **mover_kwargs
            )
            
            # Set additional attributes (only the essential ones)
            backward_mover.name = transpose_name
            backward_mover.source_perch_attr = target_perch_attr
            backward_mover.target_perch_attr = source_perch_attr
            backward_mover.source_period_idx = target_period_idx
            backward_mover.target_period_idx = source_period_idx
            
            # Add the edge to the backward graph
            self.backward_graph.add_edge(target_node_id, source_node_id, 
                                        mover=backward_mover,
                                        source_perch_attr=target_perch_attr, 
                                        target_perch_attr=source_perch_attr,
                                        inter_period=True,
                                        source_period_idx=target_period_idx,
                                        target_period_idx=source_period_idx)
            
            # Add to inter_period_movers list
            self.inter_period_movers.append(backward_mover)
            
            if verbose:
                print(f"Added transpose inter-period mover: {transpose_name} (backward)")
        
        return forward_mover

    def solve_backward(self, verbose: Optional[bool] = None):
        """Orchestrates the backward solution pass across all periods.
        
        Parameters
        ----------
        verbose : bool, optional
            Whether to print status messages. If None, uses the ModelCircuit's verbose flag.
        """
        # Use the provided verbose flag, or fall back to the class verbose flag
        verbose = verbose if verbose is not None else self.verbose
        
        if verbose:
            print(f"\n--- Starting ModelCircuit '{self.name}' Backward Solve ---")
        if not self.periods_list:
            if verbose:
                print("No periods in sequence. Nothing to solve.")
            return

        # Iterate backward through periods (T, T-1, ..., 0 or 1)
        for period in reversed(self.periods_list):
            if verbose:
                print(f"-- Solving Period {period.time_index} Backward --")
            # 1. Execute inter-period backward movers targeting this period
            current_period_idx = period.time_index
            
            # Find all incoming backward edges to stages in this period
            current_period_nodes = [f"p{current_period_idx}:{stage_id}" for stage_id in period.stages]
            incoming_backward_edges = []
            
            for target_node in current_period_nodes:
                for source_node, _, edge_data in self.backward_graph.in_edges(target_node, data=True):
                    # Check if this is an inter-period edge (source from a different period)
                    if edge_data.get('inter_period', False) and edge_data.get('mover', None):
                        incoming_backward_edges.append((source_node, target_node, edge_data))
            
            if incoming_backward_edges:
                if verbose:
                    print(f"  Executing {len(incoming_backward_edges)} incoming inter-period backward mover(s) to t={current_period_idx}...")
                for source_node, target_node, edge_data in incoming_backward_edges:
                    mover = edge_data.get('mover')
                    
                    # Extract information
                    source_node_parts = source_node.split(':')
                    target_node_parts = target_node.split(':')
                    source_period_idx = int(source_node_parts[0][1:])
                    target_period_idx = int(target_node_parts[0][1:])
                    source_stage_id = source_node_parts[1]
                    target_stage_id = target_node_parts[1]
                    
                    # Get perch attributes
                    source_perch_attr = edge_data.get('source_perch_attr', getattr(mover, 'source_perch_attr', 'arvl'))
                    target_perch_attr = edge_data.get('target_perch_attr', getattr(mover, 'target_perch_attr', 'cntn'))
                    
                    if verbose:
                        print(f"    Executing inter-period backward mover: {mover.name if hasattr(mover, 'name') else 'unnamed'}")
                    
                    # Look up the actual stage and perch objects
                    try:
                        source_period = self.get_period(source_period_idx)
                        target_period = self.get_period(target_period_idx)
                        source_stage = source_period.get_stage(source_stage_id)
                        target_stage = target_period.get_stage(target_stage_id)
                        source_perch = getattr(source_stage, source_perch_attr, None)
                        target_perch = getattr(target_stage, target_perch_attr, None)
                        
                        if source_perch is None or target_perch is None:
                            if verbose:
                                print(f"      ERROR: Could not find perches {source_perch_attr} or {target_perch_attr}")
                            continue
                        
                        if hasattr(mover, 'execute') and callable(mover.execute):
                            try:
                                # --- Actual execution should happen here --- #
                                # mover.execute()
                                if verbose:
                                    print(f"      (Conceptual) Executed {mover.name if hasattr(mover, 'name') else 'mover'}")
                                
                                # --- Placeholder Simulation for Debugging --- #
                                source_val = getattr(source_perch, 'sol', f"VAL_FROM_{source_node}")
                                if hasattr(target_perch, 'sol'):
                                    if not isinstance(target_perch.sol, dict):
                                        target_perch.sol = {}
                                    branch_key = getattr(mover, 'branch_key', 'default')
                                    target_perch.sol[branch_key] = source_val
                                    if verbose:
                                        print(f"      -> Simulated: {target_node}.sol['{branch_key}'] set.")
                                # --- End Placeholder --- #
                            except Exception as exec_err:
                                if verbose:
                                    print(f"      ERROR executing inter-period mover: {exec_err}")
                        else:
                            if verbose:
                                print(f"      Warning: Mover has no execute method.")
                    except Exception as lookup_err:
                        if verbose:
                            print(f"      ERROR: Could not find stages or periods: {lookup_err}")
                        continue

            # 2. Solve the period internally (runs topological sort and stage.solve_backward)
            period.solve_backward()

        if verbose:
            print(f"--- ModelCircuit '{self.name}' Backward Solve Finished ---")

    def solve_forward(self, verbose: Optional[bool] = None):
        """Orchestrates the forward simulation pass across all periods.
        
        Parameters
        ----------
        verbose : bool, optional
            Whether to print status messages. If None, uses the ModelCircuit's verbose flag.
        """
        # Use the provided verbose flag, or fall back to the class verbose flag
        verbose = verbose if verbose is not None else self.verbose
        
        if verbose:
            print(f"\n--- Starting ModelCircuit '{self.name}' Forward Solve ---")
        if not self.periods_list:
            if verbose:
                print("No periods in sequence. Nothing to solve.")
            return

        # Iterate forward through periods (0 or 1, ..., T)
        for i, current_period in enumerate(self.periods_list):
            if verbose:
                print(f"-- Solving Period {current_period.time_index} Forward --")
            current_period_idx = current_period.time_index
            
            # 1. Execute inter-period forward movers targeting this period
            if i > 0:  # Skip for the first period
                current_period_nodes = [f"p{current_period_idx}:{stage_id}" for stage_id in current_period.stages]
                incoming_forward_edges = []
                
                for target_node in current_period_nodes:
                    for source_node, _, edge_data in self.forward_graph.in_edges(target_node, data=True):
                        # Check if this is an inter-period edge (source from a different period)
                        if edge_data.get('inter_period', False) and edge_data.get('mover', None):
                            incoming_forward_edges.append((source_node, target_node, edge_data))
                
                if incoming_forward_edges:
                    if verbose:
                        print(f"  Executing {len(incoming_forward_edges)} incoming inter-period forward mover(s) to t={current_period_idx}...")
                    for source_node, target_node, edge_data in incoming_forward_edges:
                        mover = edge_data.get('mover')
                        
                        # Extract information
                        source_node_parts = source_node.split(':')
                        target_node_parts = target_node.split(':')
                        source_period_idx = int(source_node_parts[0][1:])
                        target_period_idx = int(target_node_parts[0][1:])
                        source_stage_id = source_node_parts[1]
                        target_stage_id = target_node_parts[1]
                        
                        # Get perch attributes
                        source_perch_attr = edge_data.get('source_perch_attr', getattr(mover, 'source_perch_attr', 'cntn'))
                        target_perch_attr = edge_data.get('target_perch_attr', getattr(mover, 'target_perch_attr', 'arvl'))
                        
                        if verbose:
                            print(f"    Executing inter-period forward mover: {mover.name if hasattr(mover, 'name') else 'unnamed'}")
                        
                        # Look up the actual stage and perch objects
                        try:
                            source_period = self.get_period(source_period_idx)
                            target_period = self.get_period(target_period_idx)
                            source_stage = source_period.get_stage(source_stage_id)
                            target_stage = target_period.get_stage(target_stage_id)
                            source_perch = getattr(source_stage, source_perch_attr, None)
                            target_perch = getattr(target_stage, target_perch_attr, None)
                            
                            if source_perch is None or target_perch is None:
                                if verbose:
                                    print(f"      ERROR: Could not find perches {source_perch_attr} or {target_perch_attr}")
                                continue
                            
                            if hasattr(mover, 'execute') and callable(mover.execute):
                                try:
                                    # --- Actual execution should happen here --- #
                                    # mover.execute() 
                                    if verbose:
                                        print(f"      (Conceptual) Executed {mover.name if hasattr(mover, 'name') else 'mover'}")
                                    
                                    # --- Placeholder Simulation for Debugging --- #
                                    source_val = getattr(source_perch, 'dist', f"STATE_FROM_{source_node}")
                                    if hasattr(target_perch, 'dist'):
                                        target_perch.dist = source_val
                                        if verbose:
                                            print(f"      -> Simulated: {target_node}.dist set to value from {source_node}.")
                                    # --- End Placeholder --- #
                                except Exception as exec_err:
                                    if verbose:
                                        print(f"      ERROR executing inter-period mover: {exec_err}")
                            else:
                                if verbose:
                                    print(f"      Warning: Mover has no execute method.")
                        except Exception as lookup_err:
                            if verbose:
                                print(f"      ERROR: Could not find stages or periods: {lookup_err}")
                            continue

            # 2. Solve the current period internally (runs topological sort and stage.solve_forward)
            current_period.solve_forward()

        if verbose:
            print(f"--- ModelCircuit '{self.name}' Forward Solve Finished ---")

    def create_transpose_connections(self, edge_type: str = "both", verbose: Optional[bool] = None) -> List[Mover]:
        """
        Create transpose connections for all inter-period movers.
        
        For each forward inter-period mover, create a corresponding backward mover, and vice versa.
        This ensures that for any user-defined connection in one direction, 
        there's an automatic corresponding connection in the opposite direction.
        
        Parameters
        ----------
        edge_type : str
            Type of movers to create transposes for: "forward", "backward", or "both".
            Default is "both" (creates both forward and backward transposes).
        verbose : bool, optional
            Whether to print status messages. If None, uses the ModelCircuit's verbose flag.
            
        Returns
        -------
        List[Mover]
            List of newly created transpose movers.
            
        Notes
        -----
        This method maintains proper data flow conventions:
        - Forward movers: source.cntn.dist -> target.arvl.dist
        - Backward movers: source.arvl.sol -> target.cntn.sol
        
        It also creates transposes for all intra-period movers within each Period
        by calling their create_transpose_connections method.
        """
        # Use the provided verbose flag, or fall back to the class verbose flag
        verbose = verbose if verbose is not None else self.verbose
        
        created_movers = []
        
        # Create transposes for inter-period movers
        forward_movers = [m for m in self.inter_period_movers if m.edge_type == "forward"]
        backward_movers = [m for m in self.inter_period_movers if m.edge_type == "backward"]
        
        # Process forward movers if requested
        if edge_type in ["forward", "both"]:
            for mover in forward_movers:
                # Create backward transpose if it doesn't exist
                if not self._has_transpose_mover(mover):
                    transpose = self._create_transpose_mover(mover, verbose=verbose)
                    if transpose:
                        created_movers.append(transpose)
                        if verbose:
                            print(f"Created backward transpose for inter-period mover: {mover.name}")
        
        # Process backward movers if requested
        if edge_type in ["backward", "both"]:
            for mover in backward_movers:
                # Create forward transpose if it doesn't exist
                if not self._has_transpose_mover(mover):
                    transpose = self._create_transpose_mover(mover, verbose=verbose)
                    if transpose:
                        created_movers.append(transpose)
                        if verbose:
                            print(f"Created forward transpose for inter-period mover: {mover.name}")
        
        # Also create transposes for all intra-period movers within each Period
        for period in self.periods_list:
            if hasattr(period, "create_transpose_connections"):
                period_transposes = period.create_transpose_connections(edge_type=edge_type, verbose=verbose)
                if period_transposes and verbose:
                    print(f"Created {len(period_transposes)} transpose movers in Period {period.time_index}")
        
        return created_movers
    
    def _has_transpose_mover(self, mover: Mover) -> bool:
        """
        Check if a transpose mover already exists for the given mover.
        
        Parameters
        ----------
        mover : Mover
            The mover to check for a transpose.
            
        Returns
        -------
        bool
            True if a transpose mover exists, False otherwise.
        """
        # Determine source and target for the potential transpose
        transpose_type = "backward" if mover.edge_type == "forward" else "forward"
        
        # Get information from the mover
        source_name = getattr(mover, 'target_name', None)
        target_name = getattr(mover, 'source_name', None)
        source_perch_attr = getattr(mover, 'target_perch_attr', None)
        target_perch_attr = getattr(mover, 'source_perch_attr', None)
        
        # Check if any existing mover matches the transpose pattern
        for existing in self.inter_period_movers:
            # Skip if edge type doesn't match transpose type
            if existing.edge_type != transpose_type:
                continue
                
            # Check if the name patterns match
            if (hasattr(existing, 'source_name') and hasattr(existing, 'target_name') and
                hasattr(mover, 'source_name') and hasattr(mover, 'target_name')):
                # For a transpose, the source_name should match target_name and vice versa
                if (existing.source_name == source_name and 
                    existing.target_name == target_name):
                    return True
                    
            # Check if perch attributes match
            if hasattr(existing, 'source_perch_attr') and hasattr(existing, 'target_perch_attr'):
                if existing.source_perch_attr == source_perch_attr and existing.target_perch_attr == target_perch_attr:
                    return True
        
        return False
    
    def _create_transpose_mover(self, mover: Mover, verbose: Optional[bool] = None) -> Optional[Mover]:
        """
        Create a transpose mover for the given mover.
        
        Parameters
        ----------
        mover : Mover
            The mover to create a transpose for.
        verbose : bool, optional
            Whether to print status messages. If None, uses the ModelCircuit's verbose flag.
            
        Returns
        -------
        Optional[Mover]
            The newly created transpose mover, or None if creation failed.
        """
        # Use the provided verbose flag, or fall back to the class verbose flag
        verbose = verbose if verbose is not None else self.verbose
        
        transpose_type = "backward" if mover.edge_type == "forward" else "forward"
        
        # --- Refactored Logic for Perch Attributes --- 
        # Parse perch attributes from the original mover's keys
        original_source_perch = None
        original_target_perch = None
        
        # Try to parse from target_key (e.g., "arvl.dist")
        if mover.target_key and '.' in mover.target_key:
            original_target_perch = mover.target_key.split('.')[0]
            
        # Try to parse from the first source_key (e.g., "cntn.sol")
        if mover.source_keys and mover.source_keys[0] and '.' in mover.source_keys[0]:
            original_source_perch = mover.source_keys[0].split('.')[0]
            
        # If parsing failed, use defaults (e.g., 'arvl' and 'cntn')
        # The new source perch is the original target perch, and vice versa
        source_perch_attr = original_target_perch if original_target_perch else ('arvl' if mover.edge_type == 'forward' else 'cntn')
        target_perch_attr = original_source_perch if original_source_perch else ('cntn' if mover.edge_type == 'forward' else 'arvl')
        # --- End Refactored Logic ---
        
        # Get source and target names safely (these should be global IDs)
        source_name = getattr(mover, 'target_name', None)
        target_name = getattr(mover, 'source_name', None)
        
        # Handle case where names might be missing (should not happen ideally)
        if source_name is None or target_name is None:
            if verbose:
                print(f"Warning: Could not get source/target name from mover {getattr(mover, 'name', 'unnamed')}. Skipping transpose.")
            return None
            
        # Get period indices
        source_period_idx = getattr(mover, 'target_period_idx', None)
        target_period_idx = getattr(mover, 'source_period_idx', None)
        
        # Determine source keys and target key based on mover type and parsed perch attrs
        if mover.edge_type == "forward":
            # Original: Forward mover (e.g., cntn -> arvl)
            # Transpose: Backward mover (e.g., arvl -> cntn)
            source_keys = [f"{source_perch_attr}.sol"] # e.g., arvl.sol
            target_key = f"{target_perch_attr}.sol" # e.g., cntn.sol
        else: # Original: Backward mover (e.g., arvl -> cntn)
            # Transpose: Forward mover (e.g., cntn -> arvl)
            source_keys = [f"{source_perch_attr}.sol", f"{source_perch_attr}.dist"] # e.g., cntn.sol, cntn.dist
            target_key = f"{target_perch_attr}.dist" # e.g., arvl.dist
        
        # Create a name for the transpose mover
        transpose_name = f"{getattr(mover, 'name', 'mover')}_transpose"
        
        # Create the transpose mover using the correct constructor parameters
        transpose_mover = Mover(
            source_name=source_name,
            target_name=target_name,
            edge_type=transpose_type,
            model=getattr(mover, 'model', None),
            comp=getattr(mover, 'comp', None),
            source_keys=source_keys,
            target_key=target_key
        )
        
        # Set additional essential attributes (name and period indices)
        setattr(transpose_mover, 'name', transpose_name)
        # transpose_mover.source_perch_attr = source_perch_attr # Removed
        # transpose_mover.target_perch_attr = target_perch_attr # Removed
        
        # Set period indices if available
        if source_period_idx is not None:
            transpose_mover.source_period_idx = source_period_idx
        if target_period_idx is not None:
            transpose_mover.target_period_idx = target_period_idx
        
        # Add the new mover to inter_period_movers and the appropriate graph
        self.inter_period_movers.append(transpose_mover)
        graph = self.backward_graph if transpose_type == "backward" else self.forward_graph
        graph.add_edge(source_name, target_name, 
                       mover=transpose_mover,
                       source_perch_attr=source_perch_attr, # Keep for graph data, if needed?
                       target_perch_attr=target_perch_attr, # Keep for graph data, if needed?
                       inter_period=True,
                       source_period_idx=source_period_idx,
                       target_period_idx=target_period_idx)
        
        if verbose:
            print(f"Added transpose inter-period mover: {transpose_name} ({transpose_type})")
        
        return transpose_mover

    def create_all_transpose_connections(self, verbose: Optional[bool] = None) -> List[Mover]:
        """Creates transpose connections for all movers in the model circuit.
        
        Parameters
        ----------
        verbose : bool, optional
            Whether to print status messages. If None, uses the ModelCircuit's verbose flag.
            
        Returns
        -------
        List[Mover]
            List of newly created transpose movers.
        """
        # Use the provided verbose flag, or fall back to the class verbose flag
        verbose = verbose if verbose is not None else self.verbose
        
        if verbose:
            print(f"\n--- Creating All Transpose Connections for ModelCircuit: {self.name} ---")
        return self.create_transpose_connections('both', verbose=verbose)

    def build_stage_graph(self, edge_type='both'):
        """
        Build a comprehensive stage-to-stage graph for the entire model.
        
        This method creates a graph where nodes are individual stages (identified by period:stage_id)
        and edges represent connections between stages, either within periods or across periods.
        
        Parameters
        ----------
        edge_type : str, optional
            Type of edges to include: 'forward', 'backward', or 'both'. Default is 'both'.
            
        Returns
        -------
        nx.DiGraph
            A directed graph representing stage-to-stage connections across the entire model.
        """
        import networkx as nx
        
        # Create a new directed graph
        stage_graph = nx.DiGraph()
        
        # Use the existing graph structure that already incorporates all period nodes and edges
        if edge_type in ['forward', 'both']:
            # Add nodes and edges from the forward graph
            for node, node_attrs in self.forward_graph.nodes(data=True):
                stage_graph.add_node(node, **node_attrs)
                
            for source, target, edge_attrs in self.forward_graph.edges(data=True):
                # Create a copy of edge attributes to avoid modifying the original
                attrs_copy = edge_attrs.copy()
                # Set the edge type to 'forward' if it doesn't already exist
                attrs_copy['type'] = 'forward'
                stage_graph.add_edge(source, target, **attrs_copy)
                
        if edge_type in ['backward', 'both'] and edge_type != 'forward':
            # Add nodes from the backward graph if not already added
            for node, node_attrs in self.backward_graph.nodes(data=True):
                if node not in stage_graph:
                    stage_graph.add_node(node, **node_attrs)
                    
            # Add edges from the backward graph
            for source, target, edge_attrs in self.backward_graph.edges(data=True):
                # Create a copy of edge attributes to avoid modifying the original
                attrs_copy = edge_attrs.copy()
                # Set the edge type to 'backward' if it doesn't already exist
                attrs_copy['type'] = 'backward'
                stage_graph.add_edge(source, target, **attrs_copy)
        
        return stage_graph

    def visualize_stage_graph(self, edge_type='both', figsize=(12, 8), 
                            node_size=1500, layout='period_layout', 
                            show_node_labels=True, show_edge_labels=False,
                            filename=None, ax=None, title=None,
                            node_color_mapping=None, edge_style_mapping=None,
                            custom_labels=None, include_period_in_label=True,
                            short_labels=False, legend_loc='upper right',
                            mark_special_nodes=False, **kwargs):
        """
        Visualize the stage-to-stage graph for the entire model.
        
        Parameters
        ----------
        edge_type : str, optional
            Type of edges to include: 'forward', 'backward', or 'both'. Default is 'both'.
        figsize : tuple, optional
            Figure size as (width, height) in inches. Default is (12, 8).
        node_size : int, optional
            Size of the nodes in the visualization. Default is 1500.
        layout : str, optional
            Layout method: 'period_layout' (arrange by period), 'spring' (force-directed), 
            'kamada_kawai', 'circular', 'random'. Default is 'period_layout'.
        show_node_labels : bool, optional
            Whether to show node labels. Default is True.
        show_edge_labels : bool, optional
            Whether to show edge labels. Default is False.
        filename : str, optional
            If provided, save the figure to this file path. Default is None.
        ax : matplotlib.axes.Axes, optional
            If provided, draw on this axis. Default is None.
        title : str, optional
            Title for the figure. Default is None.
        node_color_mapping : dict, optional
            Custom mapping for node colors. Default is None.
        edge_style_mapping : dict, optional
            Custom mapping for edge styles. Default is None.
        custom_labels : dict, optional
            Custom labels for specific node types (e.g., {'worker': 'Worker', 'retiree': 'Retiree'}).
            Default is {'worker': 'Worker', 'retiree': 'Retiree', 'disc.choice': 'Discrete Choice'}.
        include_period_in_label : bool, optional
            Whether to include the period number in node labels. Default is True.
        short_labels : bool, optional
            Whether to use abbreviated labels (1-2 characters) for nodes. Default is False.
        legend_loc : str, optional
            Position for the legend. Default is 'upper right'.
        mark_special_nodes : bool, optional
            Whether to mark initial nodes with a star (*) and terminal nodes with a plus sign (+)
            for each period in the forward graph. Default is False.
        **kwargs
            Additional keyword arguments for custom labels and stage names.
            
        Returns
        -------
        matplotlib.figure.Figure
            The figure object containing the visualization.
        """
        import matplotlib.pyplot as plt
        import networkx as nx
        import numpy as np
        
        # Default custom labels if not provided
        if custom_labels is None:
            custom_labels = {
                'worker': 'Worker',
                'retiree': 'Retiree',
                'disc.choice': 'Discrete Choice',
                'discrete': 'Discrete Choice'
            }
        
        # Build the stage graph
        G = self.build_stage_graph(edge_type=edge_type)
        
        if len(G) == 0:
            print("No nodes in graph to visualize.")
            return None
        
        # Create a new figure if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        # Default node color mapping based on stage types or decision types (now provided by caller)
        node_color_mapping = node_color_mapping or kwargs.get('node_color_mapping', {})
        
        # Default edge style mapping
        if edge_style_mapping is None:
            edge_style_mapping = kwargs.get('edge_style_mapping', {})
        
        # Determine node positions
        if layout == 'period_layout':
            # Custom layout that arranges nodes by period
            period_spacing = 3.0
            stage_spacing = 2.0
            pos = {}
            
            # Get unique period indices and sort them
            # Extract period_idx attribute, which is an integer (not Period object)
            period_indices = sorted(list(set(
                data.get('period_idx', int(node.split(':')[0][1:]))  # Get period_idx or extract from node ID
                for node, data in G.nodes(data=True)
            )))
            
            for period_idx in period_indices:
                # Get nodes for this period
                period_nodes = [node for node, data in G.nodes(data=True) 
                               if data.get('period_idx', int(node.split(':')[0][1:])) == period_idx]
                
                # Position them vertically
                for i, node in enumerate(sorted(period_nodes)):
                    pos[node] = (period_idx * period_spacing, i * stage_spacing)
        
        elif layout == 'spring':
            # Use stronger repulsion (k) to prevent node overlap, and increase iterations for better spacing
            pos = nx.spring_layout(G, seed=42, k=1.5, iterations=100)
        elif layout == 'period_spring':
            # Initialize positions based on periods to group same-period nodes together
            pos_init = {}
            
            # Get unique period indices and sort them
            period_indices = sorted(list(set(
                data.get('period_idx', int(node.split(':')[0][1:]))
                for node, data in G.nodes(data=True)
            )))
            
            # Create a circle layout for each period
            for period_idx in period_indices:
                # Get nodes for this period
                period_nodes = [node for node, data in G.nodes(data=True) 
                               if data.get('period_idx', int(node.split(':')[0][1:])) == period_idx]
                
                # Position this period's nodes in a small circle at a location based on period index
                period_center_x = 5 * period_idx
                period_center_y = 0
                
                # Create a circle layout for nodes in this period
                for i, node in enumerate(period_nodes):
                    angle = 2 * np.pi * i / len(period_nodes)
                    radius = 3.0
                    pos_init[node] = (
                        period_center_x + radius * np.cos(angle),
                        period_center_y + radius * np.sin(angle)
                    )
            
            # Use these initial positions for spring layout
            pos = nx.spring_layout(G, pos=pos_init, fixed=None, k=1.0, 
                                  iterations=50, weight='weight', seed=42)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'random':
            pos = nx.random_layout(G, seed=42)
        else:
            pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes
        node_colors = []
        node_types = {}  # Store the node type for each node for labeling
        
        # Mappings for stage names and abbreviations (now provided by the caller)
        abbreviated_names = kwargs.get('abbreviated_names', {})
        
        # Store full stage names for legend
        stage_full_names = kwargs.get('stage_full_names', {})
        
        # Get node types for coloring
        for node, data in G.nodes(data=True):
            stage = data.get('stage')
            
            # Extract stage ID and check for special node types
            stage_id = data.get('stage_id', '').lower()
            
            # Direct mapping from stage_id
            if stage_id in node_color_mapping:
                stage_type = stage_id
            # Check for stage_type attribute
            elif hasattr(stage, 'stage_type'):
                stage_type = stage.stage_type.lower()
            # Check for decision_type attribute
            elif data.get('decision_type', '').lower() in node_color_mapping:
                stage_type = data.get('decision_type', '').lower()
            # Default to unknown
            else:
                stage_type = 'unknown'
            
            # Store the node type
            node_types[node] = stage_type
            
            # Get the color based on stage type
            color = node_color_mapping.get(stage_type, 'lightgray')
            node_colors.append(color)
        
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_colors, 
                              alpha=0.7, ax=ax)
        
        # Identify initial and terminal nodes for each period if requested
        initial_nodes = {}
        terminal_nodes = {}
        
        if mark_special_nodes:
            # Get unique period indices
            period_indices = sorted(list(set(
                data.get('period_idx', int(node.split(':')[0][1:]))
                for node, data in G.nodes(data=True)
            )))
            
            for period_idx in period_indices:
                # Get nodes for this period
                period_nodes = [node for node, data in G.nodes(data=True) 
                                if data.get('period_idx', int(node.split(':')[0][1:])) == period_idx]
                
                # Create a subgraph for this period with only forward edges
                period_subgraph = G.subgraph(period_nodes).copy()
                
                # Keep only forward edges within this period
                forward_edges = [(u, v) for u, v, data in period_subgraph.edges(data=True)
                                 if data.get('type') == 'forward' and not data.get('inter_period', False)]
                
                # Create a new directed graph with just forward edges
                forward_graph = nx.DiGraph()
                forward_graph.add_nodes_from(period_nodes)
                forward_graph.add_edges_from(forward_edges)
                
                # Find initial nodes (no incoming forward edges within this period)
                period_initial_nodes = [node for node in period_nodes if forward_graph.in_degree(node) == 0]
                
                # Find terminal nodes (no outgoing forward edges within this period)
                period_terminal_nodes = [node for node in period_nodes if forward_graph.out_degree(node) == 0]
                
                # Store for this period
                initial_nodes[period_idx] = period_initial_nodes
                terminal_nodes[period_idx] = period_terminal_nodes
        
        # Draw node labels if requested
        if show_node_labels:
            # Create labels, applying custom labels for special node types and including period numbers
            labels = {}
            for node, data in G.nodes(data=True):
                # Get period number
                period_idx = data.get('period_idx', int(node.split(':')[0][1:]))
                
                # Get the stage ID
                stage_id = data.get('stage_id', node.split(':')[-1])
                
                # Determine if this is a special node to be marked
                node_mark = ""
                if mark_special_nodes:
                    if node in initial_nodes.get(period_idx, []):
                        node_mark = "*"  # Star for initial nodes
                    elif node in terminal_nodes.get(period_idx, []):
                        node_mark = "+"  # Plus for terminal nodes
                
                if short_labels:
                    # Use abbreviated label for this stage type
                    abbr = abbreviated_names.get(stage_id.lower(), stage_id[:2].upper())
                    if include_period_in_label:
                        labels[node] = f"P{period_idx}:{abbr}{node_mark}"
                    else:
                        labels[node] = f"{abbr}{node_mark}"
                else:
                    # Check if this node has a special type with a custom label
                    node_type = node_types.get(node, 'unknown')
                    if node_type in custom_labels:
                        # Apply the custom label for this node type
                        label = custom_labels[node_type]
                    else:
                        # Use the default stage ID
                        label = stage_id
                    
                    # Include period in label if requested
                    if include_period_in_label:
                        labels[node] = f"P{period_idx}: {label}{node_mark}"
                    else:
                        labels[node] = f"{label}{node_mark}"
            
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, 
                                   font_weight='bold', ax=ax)
        
        # Draw edges with different styles for different types
        for edge_type in ['forward_intra', 'backward_intra', 'forward_inter', 'backward_inter']:
            edge_type_parts = edge_type.split('_')
            direction, connection = edge_type_parts[0], edge_type_parts[1]
            
            # Filter edges by type
            edges = [(u, v) for u, v, data in G.edges(data=True) 
                    if data.get('type') == direction and 
                    (data.get('inter_period') if connection == 'inter' else not data.get('inter_period', False))]
            
            if edges:
                style = edge_style_mapping.get(edge_type, {})
                nx.draw_networkx_edges(
                    G, pos, edgelist=edges, 
                    width=style.get('width', 1.5),
                    alpha=style.get('alpha', 0.7),
                    edge_color=style.get('color', 'black'),
                    style=style.get('style', 'solid'),
                    arrowsize=style.get('arrowsize', 15),
                    connectionstyle='arc3,rad=0.1',  # Use curved edges to avoid overlap
                    ax=ax
                )
        
        # Draw edge labels if requested
        if show_edge_labels:
            edge_labels = {}
            for u, v, data in G.edges(data=True):
                mover = data.get('mover')
                if mover and hasattr(mover, 'name'):
                    edge_labels[(u, v)] = mover.name
            
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, 
                                        font_size=8, ax=ax)
        
        # Add a title if provided
        if title:
            plt.title(title)
        else:
            # Get min and max period indices from the graph nodes
            min_period = min(data.get('period_idx', int(node.split(':')[0][1:])) 
                            for node, data in G.nodes(data=True))
            max_period = max(data.get('period_idx', int(node.split(':')[0][1:])) 
                            for node, data in G.nodes(data=True))
            periods_range = f"{min_period}-{max_period}"
            plt.title(f"Stage Graph for {self.name} (Periods {periods_range})")
        
        # Remove axis ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # Create a better legend that doesn't overlap with the graph
        from matplotlib.patches import Patch
        
        # Only include node types that are actually present in the graph
        present_types = set(node_types.values())
        
        # Use full names in the legend
        legend_elements = []
        for node_type in present_types:
            # Get the full name for the legend
            if node_type in stage_full_names:
                legend_label = stage_full_names[node_type]
            else:
                legend_label = custom_labels.get(node_type, node_type.capitalize())
            
            # Create the patch with the color
            legend_elements.append(
                Patch(facecolor=node_color_mapping.get(node_type, 'lightgray'), 
                      edgecolor='black', alpha=0.7, 
                      label=legend_label)
            )
            
        # Add legend elements for special nodes if requested
        if mark_special_nodes:
            # Add explanation for the star and plus markers
            from matplotlib.lines import Line2D
            legend_elements.append(
                Line2D([0], [0], marker='', linestyle='', label="* Initial node", 
                       markerfacecolor='white', markersize=0)
            )
            legend_elements.append(
                Line2D([0], [0], marker='', linestyle='', label="+ Terminal node", 
                       markerfacecolor='white', markersize=0)
            )
        
        # Add legend elements for edge types
        import matplotlib.lines as mlines
        for edge_type, style in edge_style_mapping.items():
            if '_' in edge_type:
                direction, period_type = edge_type.split('_')
                # Check if this edge type exists in the graph
                exists = any(
                    data.get('type') == direction and
                    (data.get('inter_period') if period_type == 'inter' else not data.get('inter_period', False))
                    for _, _, data in G.edges(data=True)
                )
                if exists:
                    line = mlines.Line2D([], [], color=style.get('color', 'black'),
                                        linestyle='-' if style.get('style', 'solid') == 'solid' else '--',
                                        linewidth=style.get('width', 1.5),
                                        alpha=style.get('alpha', 0.7),
                                        label=f"{direction.capitalize()} {period_type.capitalize()}")
                    legend_elements.append(line)
        
        if legend_elements:
            # Create legend outside of the main plot area to prevent overlap
            if layout in ['spring', 'period_spring']:
                # For spring layout, place legend in a fixed position that won't overlap
                legend = ax.legend(handles=legend_elements, loc=legend_loc, fontsize=8, 
                           bbox_to_anchor=(1.02, 1), borderaxespad=0.)
            else:
                legend = ax.legend(handles=legend_elements, loc=legend_loc, fontsize=8)
        
        # Add annotation with graph statistics
        stats_text = (
            f"Nodes: {len(G.nodes())}\n"
            f"Intra-period edges: {sum(1 for _, _, d in G.edges(data=True) if not d.get('inter_period', False))}\n"
            f"Inter-period edges: {sum(1 for _, _, d in G.edges(data=True) if d.get('inter_period', False))}"
        )
        plt.figtext(0.02, 0.02, stats_text, fontsize=8, ha='left')
        
        # Save the figure if a filename is provided
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {filename}")
        
        plt.tight_layout()
        return fig 