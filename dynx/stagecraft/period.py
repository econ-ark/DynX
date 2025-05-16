# src/stagecraft/period.py

import networkx as nx
from typing import Dict, Optional, Callable, List, Any, Union, TYPE_CHECKING
import warnings
from copy import deepcopy

from dynx.core.circuit_board import CircuitBoard
from dynx.core.perch import Perch
from dynx.core.mover import Mover

# Import Stage only for type checking
if TYPE_CHECKING:
    from dynx.stagecraft.stage import Stage

def build_transpose(m: Mover, *, branch_key: str | None = None) -> Mover:
    """Return a new Mover that is the CDC-conform transpose of `m`.
    
    Parameters
    ----------
    m : Mover
        The original mover to create a transpose for
    branch_key : str | None, optional
        Branch key for the transpose mover, by default None
        
    Returns
    -------
    Mover
        A new Mover that is the CDC-conform transpose of the input mover
    """
    t = deepcopy(m)
    t.edge_type = "backward" if m.edge_type == "forward" else "forward"
    t.source_name, t.target_name = m.target_name, m.source_name

    if m.edge_type == "forward":      # F → B
        bk = branch_key or m.branch_key or m.source_name
        t.source_keys = ["arvl.sol"]
        t.target_key = "cntn.sol"
        t.branch_key = bk
    else:                             # B → F
        t.source_keys = ["cntn.dist"]
        t.target_key = "arvl.dist"
        t.branch_key = branch_key or m.branch_key
    
    t.name = f"{m.name}_T" if hasattr(m, 'name') else f"{t.source_name}_to_{t.target_name}_{t.edge_type}_T"
    return t

class Period(CircuitBoard):
    # Minimal docstring
    """Container for Stages and intra-period Movers at a time index 't'."""

    def __init__(self, time_index: int):
        """
        Initialize a Period object.
        
        Parameters
        ----------
        time_index : int
            The time index of this period.
        """
        # Call the parent CircuitBoard constructor with the period name
        super().__init__(name=f"Period_{time_index}")
        # Store the time index
        self.time_index = time_index
        # Initialize the stages dictionary
        self.stages: Dict[str, 'Stage'] = {}
        # Initialize a list to store all movers explicitly
        self.movers: List[Mover] = []
        # Initialize the main graphs (stage-to-stage connections only)
        self.forward_graph = nx.DiGraph()  # Stage-level forward graph
        self.backward_graph = nx.DiGraph() # Stage-level backward graph
        # Print initialization message
        print(f"Initialized Period {time_index}")

    def _auto_branch_key(self, src_id: str, existing: dict, direction: str) -> str:
        """
        Auto-generate a unique branch key when none is supplied.
        
        Parameters
        ----------
        src_id : str
            The source stage ID to use as default branch key
        existing : dict
            Dictionary of existing keys to check for collisions
        direction : str
            Direction ("forward" or "backward") for the warning message
            
        Returns
        -------
        str
            A unique branch key
        """
        default = src_id
        if default in existing:
            warnings.warn(
                f"[{direction}] branch_key not supplied; auto-renaming "
                f"duplicate '{default}' → '{default}_1'.", RuntimeWarning)
            i = 1
            while f"{default}_{i}" in existing:
                i += 1
            return f"{default}_{i}"
        return default

    def connect_bwd(self, src: Union[str, 'Stage'], tgt: Union[str, 'Stage'],
                   *, branch_key: Optional[str] = None, create_transpose: bool = False, **mover_kw) -> Mover:
        """
        Create a backward connection from a source stage to a target stage.
        
        Parameters
        ----------
        src : str | Stage
            Source stage or its ID
        tgt : str | Stage
            Target stage or its ID
        branch_key : str | None, optional
            Key used to index multiple incoming objects inside target's cntn.sol dict,
            by default None (auto-generated from source ID)
        create_transpose : bool, optional
            Whether to automatically create a transpose connection, by default False
        **mover_kw
            Additional keyword arguments for the Mover constructor
            
        Returns
        -------
        Mover
            The created backward mover
            
        Raises
        ------
        ValueError
            If either source or target stage is not found
        """
        # Resolve stage objects and IDs
        source_stage = src if not isinstance(src, str) else self.get_stage(src)
        target_stage = tgt if not isinstance(tgt, str) else self.get_stage(tgt)
        
        source_id = src if isinstance(src, str) else next(
            k for k, v in self.stages.items() if v == src)
        target_id = tgt if isinstance(tgt, str) else next(
            k for k, v in self.stages.items() if v == tgt)
        
        # Check if we need to auto-generate a branch key
        if branch_key is None:
            # Check existing target cntn.sol keys if it's a dict
            existing = {}
            if hasattr(target_stage, 'cntn') and hasattr(target_stage.cntn, 'sol'):
                if isinstance(target_stage.cntn.sol, dict):
                    existing = target_stage.cntn.sol
            
            # Check existing movers targeting the same stage with backward edges
            for _, t, data in self.backward_graph.in_edges(target_id, data=True):
                if 'mover' in data and hasattr(data['mover'], 'branch_key') and data['mover'].branch_key:
                    existing[data['mover'].branch_key] = True
            
            branch_key = self._auto_branch_key(source_id, existing, "backward")
        
        # Create the backward mover
        return self.add_connection(
            source_stage=source_stage,
            target_stage=target_stage,
            source_perch_attr="arvl",
            target_perch_attr="cntn",
            direction="backward",
            branch_key=branch_key,
            create_transpose=create_transpose,
            **mover_kw
        )

    def connect_fwd(self, src: Union[str, 'Stage'], tgt: Union[str, 'Stage'],
                   *, branch_key: Optional[str] = None, create_transpose: bool = False, **mover_kw) -> Mover:
        """
        Create a forward connection from a source stage to a target stage.
        
        Parameters
        ----------
        src : str | Stage
            Source stage or its ID
        tgt : str | Stage
            Target stage or its ID
        branch_key : str | None, optional
            Key used to index multiple incoming objects inside target's arvl.dist dict,
            by default None (auto-generated from source ID)
        create_transpose : bool, optional
            Whether to automatically create a transpose connection, by default False
        **mover_kw
            Additional keyword arguments for the Mover constructor
            
        Returns
        -------
        Mover
            The created forward mover
            
        Raises
        ------
        ValueError
            If either source or target stage is not found
        """
        # Resolve stage objects and IDs
        source_stage = src if not isinstance(src, str) else self.get_stage(src)
        target_stage = tgt if not isinstance(tgt, str) else self.get_stage(tgt)
        
        source_id = src if isinstance(src, str) else next(
            k for k, v in self.stages.items() if v == src)
        target_id = tgt if isinstance(tgt, str) else next(
            k for k, v in self.stages.items() if v == tgt)
        
        # Check if we need to auto-generate a branch key
        if branch_key is None:
            # Check existing target arvl.dist keys if it's a dict
            existing = {}
            if hasattr(target_stage, 'arvl') and hasattr(target_stage.arvl, 'dist'):
                if isinstance(target_stage.arvl.dist, dict):
                    existing = target_stage.arvl.dist
            
            # Check existing movers targeting the same stage with forward edges
            for _, t, data in self.forward_graph.in_edges(target_id, data=True):
                if 'mover' in data and hasattr(data['mover'], 'branch_key') and data['mover'].branch_key:
                    existing[data['mover'].branch_key] = True
            
            branch_key = self._auto_branch_key(source_id, existing, "forward")
        
        # Create the forward mover
        return self.add_connection(
            source_stage=source_stage,
            target_stage=target_stage,
            source_perch_attr="cntn",
            target_perch_attr="arvl",
            direction="forward",
            branch_key=branch_key,
            create_transpose=create_transpose,
            **mover_kw
        )
        
    def add_stage(self, stage_id: str, stage_obj: 'Stage'):
        """
        Add a Stage to this Period.
        
        Parameters
        ----------
        stage_id : str
            ID for the stage within this period.
        stage_obj : Stage
            The Stage object to add.
        """
        # Add the stage to the stages dictionary
        if stage_id in self.stages:
            print(f"Warning: Stage {stage_id} already exists in Period {self.time_index}. Overwriting.")
        self.stages[stage_id] = stage_obj
        
        # Add the stage as a node to both graphs
        self.forward_graph.add_node(stage_id, stage=stage_obj)
        self.backward_graph.add_node(stage_id, stage=stage_obj)
        
        # NOTE: We intentionally DO NOT add the stage's internal edges to the period's graph
        # Intra-stage connections remain within the stage's own graph
        # Inter-stage connections are added separately through add_connection or similar methods
        print(f"Added Stage '{stage_id}' to Period {self.time_index} (added {len(stage_obj.perches)} perches)")

    def get_stage(self, stage_id: str) -> 'Stage':
        """
        Get a Stage by its ID.
        
        Parameters
        ----------
        stage_id : str
            ID of the stage to get.
            
        Returns
        -------
        Stage
            The requested Stage object.
            
        Raises
        ------
        ValueError
            If no Stage with the given ID exists.
        """
        if stage_id not in self.stages:
            raise ValueError(f"Stage '{stage_id}' not found in Period {self.time_index}.")
        return self.stages[stage_id]

    def _has_transpose(self, m: Mover) -> bool:
        """
        Check if a transpose mover already exists for the given mover.
        
        Parameters
        ----------
        m : Mover
            The mover to check for an existing transpose
            
        Returns
        -------
        bool
            True if a transpose already exists, False otherwise
        """
        # Determine the expected transpose attributes
        transpose_edge_type = "backward" if m.edge_type == "forward" else "forward"
        transpose_source = m.target_name
        transpose_target = m.source_name
        
        # Get the appropriate graph
        graph = self.backward_graph if transpose_edge_type == "backward" else self.forward_graph
        
        # Check if an edge already exists with matching properties
        if not graph.has_edge(transpose_source, transpose_target):
            return False
            
        # Get the edge data
        edge_data = graph.get_edge_data(transpose_source, transpose_target)
        
        # Handle different edge data structures
        if isinstance(edge_data, dict):
            # If the edge_data contains a 'mover' key directly, it's a single edge
            if 'mover' in edge_data:
                edge_mover = edge_data['mover']
                if (edge_mover.edge_type == transpose_edge_type and
                    edge_mover.branch_key == m.branch_key):
                    return True
            else:
                # Otherwise, it could be a MultiDiGraph with multiple edges
                for key, data in edge_data.items():
                    if isinstance(data, dict) and 'mover' in data:
                        edge_mover = data['mover']
                        if (edge_mover.edge_type == transpose_edge_type and
                            edge_mover.branch_key == m.branch_key):
                            return True
        
        return False

    def _insert_edge(self, m: Mover) -> None:
        """
        Insert a mover into the appropriate graph.
        
        Parameters
        ----------
        m : Mover
            The mover to insert
        """
        # Determine which graph to use
        graph = self.backward_graph if m.edge_type == "backward" else self.forward_graph
        
        # Add the edge with the mover
        graph.add_edge(m.source_name, m.target_name, mover=m)
        
        # Add to the movers list
        self.movers.append(m)

    def create_transpose_connections(self, edge_type="both") -> list[Mover]:
        """
        Create transpose connections for all movers in the period.
        
        Parameters
        ----------
        edge_type : str, optional
            Type of edges to create transposes for: "forward", "backward", or "both",
            by default "both"
            
        Returns
        -------
        list[Mover]
            List of newly created transpose movers
        """
        created = []
        graph_pairs = ((self.forward_graph, "forward"),
                      (self.backward_graph, "backward"))

        for G, gtype in graph_pairs:
            if edge_type not in (gtype, "both"):
                continue
            for u, v, d in list(G.edges(data=True)):
                if 'mover' not in d:
                    continue
                m = d["mover"]
                if self._has_transpose(m):
                    continue
                t = build_transpose(m)
                self._insert_edge(t)
                created.append(t)
                print(f"Created transpose mover: {t.source_name} → {t.target_name} ({t.edge_type})")
        
        # Also create transposes for each Stage's internal movers
        for stage_id, stage in self.stages.items():
            if hasattr(stage, "create_transpose_connections"):
                stage_transposes = stage.create_transpose_connections(edge_type=edge_type)
                if stage_transposes:
                    print(f"Created {len(stage_transposes)} transpose movers in Stage {stage_id}")
                    created.extend(stage_transposes)
                    
        return created

    def add_connection(
        self,
        source_stage: 'Stage',
        target_stage: 'Stage',
        source_perch_attr: str = "cntn",
        target_perch_attr: str = "arvl",
        direction: str = "forward",
        branch_key: Optional[str] = None,
        forward_comp: Optional[Callable] = None,
        mover_name: Optional[str] = None,
        create_transpose: bool = True,
        **mover_kwargs
    ) -> Mover:
        """
        Add a connection between two stages in this period.
        
        Parameters
        ----------
        source_stage : Stage
            The source stage.
        target_stage : Stage
            The target stage.
        source_perch_attr : str, optional
            Attribute name of the source perch in the source stage. Default is "cntn".
        target_perch_attr : str, optional
            Attribute name of the target perch in the target stage. Default is "arvl".
        direction : str, optional
            Direction of the connection: 'backward' or 'forward'. Default is 'forward'.
        branch_key : str, optional
            For backward connections to discrete stages, the branch key to identify the value.
        forward_comp : Callable, optional
            Computational function for the mover in forward direction.
        mover_name : str, optional
            Explicit name for the mover. If None, a default name is generated.
        create_transpose : bool, optional
            Whether to automatically create a transpose connection. Default is True.
        **mover_kwargs
            Additional arguments for the Mover constructor.
            
        Returns
        -------
        Mover
            The newly created Mover object.
            
        Raises
        ------
        ValueError
            If direction is invalid or branch_key is missing for discrete stages.
        AttributeError
            If the source_perch_attr or target_perch_attr doesn't exist.
        """
        # Get stage IDs for reference
        source_stage_id = next((k for k, v in self.stages.items() if v == source_stage), None)
        target_stage_id = next((k for k, v in self.stages.items() if v == target_stage), None)
        
        if source_stage_id is None:
            raise ValueError(f"Source stage not found in period {self.time_index}")
        if target_stage_id is None:
            raise ValueError(f"Target stage not found in period {self.time_index}")
        
        # Verify the perches exist but don't keep direct references
        source_perch_obj = getattr(source_stage, source_perch_attr, None)
        target_perch_obj = getattr(target_stage, target_perch_attr, None)
        
        if source_perch_obj is None:
            raise AttributeError(f"Source stage does not have a '{source_perch_attr}' perch")
        if target_perch_obj is None:
            raise AttributeError(f"Target stage does not have a '{target_perch_attr}' perch")
        
        # Set up edge type, saved branch key, and comp function
        edge_type = direction
        saved_branch_key = None
        mover_comp = forward_comp if direction == "forward" else None
        
        # Additional validation for backward direction
        if direction == 'backward':
            if target_stage.decision_type == 'discrete' and branch_key is None:
                raise ValueError(f"Backward connection to discrete stage '{target_stage_id}' requires 'branch_key'.")
            if branch_key is None: 
                branch_key = source_stage_id
            saved_branch_key = branch_key
            # Remove branch_key from mover_kwargs if present
            if 'branch_key' in mover_kwargs:
                mover_kwargs.pop('branch_key')
        elif direction != 'forward':
            raise ValueError("Direction must be 'backward' or 'forward'.")
        
        # Generate a default mover name if not provided
        if mover_name is None:
            branch_suffix = f"_{saved_branch_key}" if direction == 'backward' and saved_branch_key else ""
            mover_name = f"{source_stage_id}_to_{target_stage_id}_{direction}{branch_suffix}"
        
        # Determine appropriate source keys and target key based on direction
        if direction == 'forward':
            # Forward mover: source_keys = ["cntn"]["sol", "dist"], target_key = ["arvl"]["dist"]
            source_keys = [f"{source_perch_attr}.sol", f"{source_perch_attr}.dist"]
            target_key = f"{target_perch_attr}.dist"
        else:  # backward
            # Backward mover: source_keys = ["arvl"]["sol"], target_key = ["cntn"]["sol"]
            source_keys = [f"{source_perch_attr}.sol"]
            target_key = f"{target_perch_attr}.sol"
        
        # Create the mover with stage IDs as source_name and target_name
        mover = Mover(
            source_name=source_stage_id,
            target_name=target_stage_id,
            edge_type=edge_type,
            comp=mover_comp,
            source_keys=source_keys,
            target_key=target_key,
            **mover_kwargs
        )
        
        # Set additional attributes on the mover (only the essential ones)
        mover.name = mover_name
        
        # For backward movers, preserve the branch key
        if saved_branch_key is not None:
            mover.branch_key = saved_branch_key
        
        # Add period reference for context
        mover.period_idx = self.time_index
        
        # Add the mover to the appropriate graph (between stage nodes, not perches)
        graph = self.backward_graph if edge_type == "backward" else self.forward_graph
        graph.add_edge(source_stage_id, target_stage_id, mover=mover, 
                      source_perch_attr=source_perch_attr, target_perch_attr=target_perch_attr)
        
        # Add the mover to the movers list
        self.movers.append(mover)
        
        print(f"Added connection Mover: {mover_name} ({source_stage_id}.{source_perch_attr} -> {target_stage_id}.{target_perch_attr})")

        # Add connections to the CircuitBoard graph too
        source_id = f"{source_stage_id}.{source_perch_attr}"
        target_id = f"{target_stage_id}.{target_perch_attr}"
        mover_graph = self.forward_graph if edge_type == "forward" else self.backward_graph

        # Create the transpose mover if requested
        if create_transpose:
            transpose_direction = "backward" if direction == "forward" else "forward"
            transpose_branch_key = saved_branch_key if transpose_direction == "backward" else None
            
            # Handle special case: when creating backward transpose to a discrete stage
            if transpose_direction == "backward" and source_stage.decision_type == "discrete":
                # If we're creating a backward connection TO a discrete stage,
                # we need to ensure there's a branch_key
                if transpose_branch_key is None:
                    # Use target_stage_id as default branch_key if none is provided
                    transpose_branch_key = target_stage_id
                    print(f"Warning: Auto-generating branch_key '{transpose_branch_key}' for backward connection to discrete stage '{source_stage_id}'")
            
            # Recursive call to create the transpose but disable create_transpose to prevent infinite loop
            self.add_connection(
                source_stage=target_stage if transpose_direction == "backward" else source_stage,
                target_stage=source_stage if transpose_direction == "backward" else target_stage,
                source_perch_attr=target_perch_attr if transpose_direction == "backward" else source_perch_attr,
                target_perch_attr=source_perch_attr if transpose_direction == "backward" else target_perch_attr,
                direction=transpose_direction,
                branch_key=transpose_branch_key,
                mover_name=f"{mover_name}_transpose",
                create_transpose=False,  # Prevent infinite recursion
                **mover_kwargs
            )
        
        return mover

    def _get_subgraph(self, edge_type_filter: str) -> nx.DiGraph:
        """
        Get a filtered subgraph containing only edges of a specific type.
        
        Parameters
        ----------
        edge_type_filter : str
            The edge type to filter by: "forward" or "backward".
            
        Returns
        -------
        nx.DiGraph
            A directed graph containing only edges of the specified type.
        """
        # Graph that we'll be filtering
        if edge_type_filter == "forward":
            main_graph = self.forward_graph
        elif edge_type_filter == "backward":
            main_graph = self.backward_graph
        else:
            raise ValueError(f"Invalid edge_type_filter: {edge_type_filter}")
        
        # Create a new directed graph
        filtered_graph = nx.DiGraph()
        
        # Add all nodes from the main graph
        for node in main_graph.nodes:
            filtered_graph.add_node(node, **main_graph.nodes[node])
        
        # Filtered_graph now has all nodes from the main graph but no edges
        return filtered_graph

    def solve_backward(self):
        """
        Solve all stages in the period in backward order using the topological sort.
        
        Implements fan-out in the backward direction - a source stage's arvl.sol
        can be sent to multiple target stages' cntn.sol using branch keys.
        
        Returns
        -------
        None
        """
        print(f"[Period {self.time_index}] Starting backward solve")
        
        # Get the order of nodes for backward solution (from terminal to initial)
        # Use the backward graph for the order, since we're solving backward
        stage_topo_order = self._get_topological_order("backward")
        
        # If there are no stages, return early
        if not stage_topo_order:
            print(f"[Period {self.time_index}] No stages to solve backward.")
            return
        
        # Print the order we'll use for solving
        print(f"[Period {self.time_index}] Backward solve order: {', '.join(stage_topo_order)}")
        
        # Process each stage in the backward topological order
        for target_stage_id in stage_topo_order:
            target_stage = self.stages[target_stage_id]
            
            # Get incoming backward edges to this target stage
            # Only backward graph edges are relevant for backward solution direction
            incoming_edges = list(self.backward_graph.in_edges(target_stage_id, data=True))
            
            # Process branch connections for incoming mover data
            branch_connections = {}
            
            # Initialize target_stage.cntn.sol as dict if not already
            if not hasattr(target_stage, 'cntn') or not hasattr(target_stage.cntn, 'sol'):
                print(f"Warning: Target stage {target_stage_id} missing cntn.sol attribute.")
            else:
                if target_stage.cntn.sol is None:
                    target_stage.cntn.sol = {}
                elif not isinstance(target_stage.cntn.sol, dict) and incoming_edges:
                    # Convert to dict if it's not already and we have incoming edges
                    print(f"Converting {target_stage_id}.cntn.sol to dict for fan-in.")
                    target_stage.cntn.sol = {"default": target_stage.cntn.sol}
            
            # Fan-in: Copy data from all source stages to target stage using branch keys
            for source_id, _, edge_data in incoming_edges:
                if 'mover' in edge_data:
                    mover = edge_data['mover']
                    source_stage = self.stages[source_id]
                    
                    # Only consider backward movers for backward solve
                    if mover.edge_type == 'backward':
                        # Get the branch key (use source_id as default if not specified)
                        branch_key = mover.branch_key or source_id
                        
                        # Get the actual perches
                        source_perch = getattr(source_stage, "arvl")
                        target_perch = getattr(target_stage, "cntn")
                        
                        # Get the source perch sol data
                        source_data = source_perch.sol
                        
                        # Check that source has data before connecting
                        if source_data is not None:
                            # Update the branch_key entry in the target perch's sol dict
                            target_perch.sol[branch_key] = source_data
                            branch_connections[branch_key] = source_id
                        else:
                            print(f"Warning: Source perch {source_id}.arvl does not have sol data to connect to target {target_stage_id}.cntn")
            
            # Print branch connections for debugging
            if branch_connections:
                branch_info = ", ".join([f"{k}: {v}" for k, v in branch_connections.items()])
                print(f"[Period {self.time_index}] Connected branches to {target_stage_id}: {branch_info}")
            
            # Now solve the target stage (with branch connections set up)
            print(f"[Period {self.time_index}] Solving backward: {target_stage_id}")
            target_stage.solve_backward()
        
        print(f"[Period {self.time_index}] Backward solve complete")

    def solve_forward(self):
        """
        Solve all stages in the period in forward order using the topological sort.
        
        Implements fan-in in the forward direction - multiple source stages' cntn.dist
        values can be collected into a target stage's arvl.dist using branch keys.
        
        Returns
        -------
        None
        """
        print(f"[Period {self.time_index}] Starting forward solve")
        
        # Get the order of nodes for forward solution (from initial to terminal)
        # Use the forward graph for the order, since we're solving forward
        stage_topo_order = self._get_topological_order("forward")
        
        # If there are no stages, return early
        if not stage_topo_order:
            print(f"[Period {self.time_index}] No stages to solve forward.")
            return
        
        # Print the order we'll use for solving
        print(f"[Period {self.time_index}] Forward solve order: {', '.join(stage_topo_order)}")
        
        # Process each stage in the forward topological order
        for target_stage_id in stage_topo_order:
            target_stage = self.stages[target_stage_id]
            
            # Get incoming forward edges to this target stage
            # Only forward graph edges are relevant for forward solution direction
            incoming_edges = list(self.forward_graph.in_edges(target_stage_id, data=True))
            
            # Process connections for incoming mover data
            if incoming_edges:
                # Initialize a dictionary to collect incoming dist data
                incoming = {}
                
                # Fan-in: Collect dist data from all source stages
                for source_id, _, edge_data in incoming_edges:
                    if 'mover' in edge_data:
                        mover = edge_data['mover']
                        source_stage = self.stages[source_id]
                        
                        # Only consider forward movers for forward solve
                        if mover.edge_type == 'forward':
                            # Get the branch key (use source_id as default if not specified)
                            branch_key = mover.branch_key or source_id
                            
                            # Get the source perch dist data
                            source_perch = getattr(source_stage, "cntn")
                            source_data = source_perch.dist
                            
                            # Check that source has data before connecting
                            if source_data is not None:
                                # Add to incoming dict with branch_key
                                incoming[branch_key] = source_data
                            else:
                                print(f"Warning: Source perch {source_id}.cntn does not have dist data to connect to target {target_stage_id}.arvl")
                
                # If we collected any incoming data, assign it to target
                if incoming:
                    incoming_info = ", ".join(list(incoming.keys()))
                    print(f"[Period {self.time_index}] Connected data to {target_stage_id}: {incoming_info}")
                    
                    # Assign the collected dict to target stage's arvl.dist
                    target_stage.arvl.dist = incoming
            
            # Now solve the target stage (with connections set up)
            print(f"[Period {self.time_index}] Solving forward: {target_stage_id}")
            target_stage.solve_forward()
        
        print(f"[Period {self.time_index}] Forward solve complete")

    def get_forward_graph(self) -> nx.DiGraph:
        """Returns the forward stage-to-stage graph within this period.

        Nodes are stage IDs. Edges represent forward connections
        made via `add_connection`. Useful for determining forward simulation order.

        Returns
        -------
        nx.DiGraph
            The forward dependency graph.
        """
        return self.forward_graph

    def get_backward_graph(self) -> nx.DiGraph:
        """Returns the backward dependency graph within this period.

        Nodes are stage IDs. Edges represent backward connections
        made via `add_connection`. Useful for determining backward solution order.

        Returns
        -------
        nx.DiGraph
            The backward dependency graph.
        """
        return self.backward_graph

    def get_initial_stages(self, graph_type: str = 'forward') -> List[str]:
        """Finds stages with in-degree 0 in the specified dependency graph.

        For 'forward', these are entry points for simulation within the period.
        For 'backward', these are the starting points for the backward solve
        within the period (stages whose values don't depend on other stages
        *within this period* via backward connections).

        Parameters
        ----------
        graph_type : str, optional
            The type of graph to analyze ('forward' or 'backward' dependency).
            Defaults to 'forward'.

        Returns
        -------
        List[str]
            A list of stage IDs that are initial nodes in the specified graph.
        """
        if graph_type == 'forward':
            graph = self.get_forward_graph()
        elif graph_type == 'backward':
            graph = self.get_backward_graph()
        else:
            raise ValueError("graph_type must be 'forward' or 'backward'")

        if not graph: # Handle empty graph case
            return list(self.stages.keys()) # All stages are initial if no connections

        return [node for node, degree in graph.in_degree() if degree == 0]

    def get_terminal_stages(self, graph_type: str = 'forward') -> List[str]:
        """Finds stages with out-degree 0 in the specified dependency graph.

        For 'forward', these are exit points for simulation within the period.
        For 'backward', these are the final calculation points in the backward
        flow within the period (stages that are not depended upon by other stages
        *within this period* via backward connections).

        Parameters
        ----------
        graph_type : str, optional
            The type of graph to analyze ('forward' or 'backward' dependency).
            Defaults to 'forward'.

        Returns
        -------
        List[str]
            A list of stage IDs that are terminal nodes in the specified graph.
        """
        if graph_type == 'forward':
            graph = self.get_forward_graph()
        elif graph_type == 'backward':
            graph = self.get_backward_graph()
        else:
            raise ValueError("graph_type must be 'forward' or 'backward'")

        if not graph: # Handle empty graph case
            return list(self.stages.keys()) # All stages are terminal if no connections

        return [node for node, degree in graph.out_degree() if degree == 0]

    def _get_topological_order(self, graph_type: str) -> List[str]:
        """Computes the topological sort order for the specified graph type.

        Ensures the graph is a Directed Acyclic Graph (DAG).

        Parameters
        ----------
        graph_type : str
            The type of graph ('forward' or 'backward' dependency).

        Returns
        -------
        List[str]
            The list of stage IDs in topological order.

        Raises
        -------
        nx.NetworkXUnfeasible
            If the graph contains cycles.
        ValueError
            If graph_type is invalid.
        """
        if graph_type == 'forward':
            graph = self.get_forward_graph()
        elif graph_type == 'backward':
            graph = self.get_backward_graph()
        else:
            raise ValueError("graph_type must be 'forward' or 'backward'")

        if not graph: # Handle empty graph
             return list(self.stages.keys()) # Return all stages if no connections

        if not nx.is_directed_acyclic_graph(graph):
            cycles = list(nx.simple_cycles(graph))
            raise nx.NetworkXUnfeasible(f"The {graph_type} graph for Period {self.time_index} contains cycles, cannot topologically sort. Cycles: {cycles}")

        return list(nx.topological_sort(graph)) 