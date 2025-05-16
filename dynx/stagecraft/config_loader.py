#!/usr/bin/env python3
"""
Stage Model Configuration Loader.

This module provides functions to build a multi-period model from
configuration dictionaries. All file I/O operations MUST be performed
at the front end, not within these core functions.

Core functions:
    initialize_model_Circuit: Build a model circuit from config dictionaries.
    create_stage: Create a Stage object from configuration dictionaries.
    visualize_model: Create visualizations of the model's structure.

Phases of model building:
    _phase0_validate_inputs: Validate input configurations.
    _phase1_create_periods: Create periods with stages.
    _phase2_create_intra_period_edges: Add connections within periods.
    _phase3_register_periods: Add periods to the model.
    _phase4_create_inter_period_edges: Add connections between periods.
"""

# Standard library imports
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Set, Iterator, Iterable, Tuple

# Third-party imports
import matplotlib.pyplot as plt
import networkx as nx

# Local imports
from dynx.stagecraft.stage import Stage
from dynx.core.mover import Mover
from dynx.stagecraft.period import Period
from dynx.stagecraft.model_circuit import ModelCircuit as Model
from dynx.heptapodx.core.api import initialize_model, generate_numerical_model 

# ==================
# Constants
# ==================

_EDGE_STYLES = {
    'forward_intra': {'color': '#1f77b4', 'width': 2.5, 'style': 'solid', 'alpha': 0.9, 'arrowsize': 20},
    'backward_intra': {'color': '#d62728', 'width': 2.5, 'style': 'dashed', 'alpha': 0.9, 'arrowsize': 20},
    'forward_inter': {'color': '#2ca02c', 'width': 2.5, 'style': 'solid', 'alpha': 0.9, 'arrowsize': 20},
    'backward_inter': {'color': '#9467bd', 'width': 2.5, 'style': 'dashed', 'alpha': 0.9, 'arrowsize': 20}
}

_VISUALIZATION_LAYOUTS = ["hierarchical", "circular", "forward", "period_spring"]

_DEFAULT_NODE_SIZE = 1500
_DEFAULT_CONNECTION_STYLE = 'arc3,rad=0.1'

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================
# Custom Exceptions
# ==================

class LoaderError(RuntimeError):
    """Top-level error thrown by config_loader."""
    pass

class ConfigKeyError(LoaderError):
    """Raised when required key is missing in config dictionaries."""
    pass

# ==================
# Utility Functions
# ==================

def _as_int(x: Any, *, context: str = "") -> int:
    """
    Safely cast a value to int, with helpful error context.
    
    Args:
        x: Value to cast to int
        context: Description of what this value represents (for error messages)
        
    Returns:
        Integer representation of the value
        
    Raises:
        ConfigKeyError: If the value cannot be cast to int
    """
    try:
        return int(x)
    except (ValueError, TypeError):
        msg = f"Expected integer value but got {type(x).__name__}: {x}"
        if context:
            msg = f"{context}: {msg}"
        raise ConfigKeyError(msg)

def _ensure_list(obj: Any) -> List:
    """
    Ensure an object is a list.
    
    Args:
        obj: The object to convert to a list if it's not already
        
    Returns:
        A list representation of the object
    """
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    if obj == "all":
        return []  # Special case for "all" keyword
    return [obj]  # Wrap scalar in a list

def _resolve_period_indices(conn_obj: Dict) -> Set[int]:
    """
    Extract all period indices from a connection descriptor.
    
    Args:
        conn_obj: A dictionary describing connections
        
    Returns:
        Set of period indices referenced in the connection
    """
    result = set()
    
    # Handle period/periods field
    if "period" in conn_obj:
        try:
            period_idx = _as_int(conn_obj["period"], context="period field")
            result.add(period_idx)
        except ConfigKeyError as e:
            logger.warning(str(e))
    
    if "periods" in conn_obj:
        periods = conn_obj["periods"]
        if periods == "all":
            pass  # "all" is handled by the caller
        elif isinstance(periods, list):
            for p in periods:
                try:
                    period_idx = _as_int(p, context="periods list item")
                    result.add(period_idx)
                except ConfigKeyError as e:
                    logger.warning(str(e))
    
    # Handle source/target period fields
    for field in ["source_period", "target_period"]:
        if field in conn_obj:
            try:
                period_idx = _as_int(conn_obj[field], context=field)
                result.add(period_idx)
            except ConfigKeyError as e:
                logger.warning(str(e))
    
    # Handle source/target periods arrays
    for field in ["source_periods", "target_periods"]:
        if field in conn_obj:
            periods = _ensure_list(conn_obj[field])
            for p in periods:
                try:
                    period_idx = _as_int(p, context=f"{field} item")
                    result.add(period_idx)
                except ConfigKeyError as e:
                    logger.warning(str(e))
    
    return result

# ==================
# Public Helper Functions
# ==================

def create_stage(stage_name: str, stage_config: Dict, master_config: Dict) -> Stage:
    """
    Create a Stage object from configuration dictionaries.
    
    Args:
        stage_name: Name of the stage.
        stage_config: Stage configuration dictionary.
        master_config: Master configuration dictionary for parameter inheritance.
        
    Returns:
        Initialized Stage object.
        
    Raises:
        LoaderError: If stage creation fails
    """
    logger.info(f"Creating stage '{stage_name}'")
    
    try:
        # Create the stage and pass initialize_model and configs to it
        # The stage will handle initialization internally using the init_rep (initialize_model)
        stage = Stage(
            name=stage_name, 
            init_rep=initialize_model,  # Pass initialize_model from heptapod_b.api
            master_config=master_config,
            config=stage_config  # Pass the stage_config directly to the Stage
        )
        
        return stage
    except Exception as e:
        raise LoaderError(f"Failed to create stage '{stage_name}': {e}") from e

def compile_all_stages(
        model: 'ModelCircuit',  # Use quotes for potential forward reference
        *,
        force: bool = False,
        **gen_kwargs
) -> None:
    """
    Compile (numerically generate) every Stage in an already-constructed
    ModelCircuit.

    Parameters
    ----------
    model : ModelCircuit
        The ModelCircuit returned by ``initialize_model_Circuit``.
    force : bool, default False
        • False  → skip a stage if it advertises itself as ``compiled``
          (i.e. ``stage.status_flags.get("compiled", False)``).
        • True   → run compilation even if the flag is already set.
    **gen_kwargs
        Extra keyword arguments forwarded verbatim to
        ``generate_numerical_model``.  Leave empty for default behaviour.

    Notes
    -----
    1. This call is **side-effect only** – it mutates each Stage in place and
       does *not* return anything.
    2. It is *not* invoked anywhere in ``initialize_model_Circuit``;
       the caller must decide when to compile.
    """
    logger.info("Numerically compiling all stages …")

    # Attempt to import ModelCircuit for type checking if needed, handle if not found
    # This is mainly for clarity; the function relies on duck typing.
    try:
        from .model_circuit import ModelCircuit
    except ImportError:
        logger.debug("Could not import ModelCircuit for type hint check, continuing...")
        ModelCircuit = None # Type check below will effectively be skipped

    # Basic check if the passed object seems like a ModelCircuit
    if ModelCircuit and not isinstance(model, ModelCircuit):
        logger.warning(f"Input 'model' might not be a ModelCircuit (type: {type(model)}). Proceeding, but ensure it has a '.periods_list' attribute.")
    elif not hasattr(model, 'periods_list'):
         logger.error("Input 'model' does not have a '.periods_list' attribute. Cannot compile stages.")
         return

    for period in model.periods_list: # Use periods_list instead of periods
        if not hasattr(period, 'stages') or not isinstance(period.stages, dict):
            logger.warning(f"Period {getattr(period, 'time_index', '?')} does not have a valid 'stages' dictionary. Skipping.")
            continue
        if not hasattr(period, 'time_index'):
             logger.warning(f"Period object does not have a 'time_index'. Using '?' for logging.")
             period_time_index = '?'
        else:
             period_time_index = period.time_index

        for stage_name, stage in period.stages.items():
            # Defensive check for stage object
            if stage is None:
                logger.warning(f"Stage '{stage_name}' in Period {period_time_index} is None. Skipping.")
                continue

            # Check if stage has status_flags, default to empty dict if not
            # Use dict now instead of set
            stage_status_flags = getattr(stage, "status_flags", {})
            if not isinstance(stage_status_flags, dict):
                logger.warning(f"Stage '{stage_name}' has status_flags, but it's not a dict (type: {type(stage_status_flags)}). Reinitializing as empty dict for checks.")
                setattr(stage, "status_flags", {}) # Ensure it's a dict
                stage_status_flags = stage.status_flags
            
            # Ensure status_flags is assigned back if created
            if not hasattr(stage, "status_flags"):
                 setattr(stage, "status_flags", stage_status_flags)

            already_compiled = stage_status_flags.get("compiled", False)
            if already_compiled and not force:
                logger.debug(f"Stage '{stage_name}' already compiled – skipping")
                continue

            try:
                # 1. Ensure the Stage has an initialised state if possible
                # Check if the stage has an 'initialize' method and is not already marked as initialized
                is_initialized = stage_status_flags.get("initialized", False) or getattr(stage, "initialized", False)
                if hasattr(stage, "initialize") and not is_initialized:
                    logger.debug(f"Initializing Stage '{stage_name}' before compilation...")
                    stage.initialize() # Call initialize if it exists and stage isn't initialized
                    # Assume initialize sets the flag correctly
                    stage_status_flags = getattr(stage, "status_flags", {}) # Re-fetch flags

                # 2. Call the Numerical generator (Assumed to be imported as generate_numerical_model)
                logger.debug(f"Calling generate_numerical_model for Stage '{stage_name}'...")
                stage.num_rep = generate_numerical_model
                stage.build_computational_model()

                # 3. Attach grid/mesh proxies, if the Stage object supports it
                if hasattr(stage, "_attach_grid_proxies") and callable(stage._attach_grid_proxies):
                    logger.debug(f"Attaching grid/mesh proxies for Stage '{stage_name}'...")
                    stage._attach_grid_proxies()
                else:
                     logger.debug(f"Stage '{stage_name}' does not have an _attach_grid_proxies method. Skipping attachment.")

                # 4. Mark as compiled (use dict assignment)
                stage.status_flags["compiled"] = True
                logger.debug(f"Marked Stage '{stage_name}' as compiled.")

                logger.info(f"✓ Compiled Stage '{stage_name}' in Period {period_time_index}")

            except Exception as exc:
                logger.warning(
                    f"⚠️ Failed to compile Stage '{stage_name}' "
                    f"(Period {period_time_index}): {exc}",
                    exc_info=True # Log traceback for debugging
                )

def visualize_model(model: Model, output_dir: Union[str, Path], prefix: str = "", save_svg: bool = False) -> None:
    """
    Create visualizations of the model's structure.
    
    Args:
        model: Model to visualize.
        output_dir: Directory to save visualization files.
        prefix: Prefix for output filenames.
        save_svg: Whether to also save SVG copies of the visualizations.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations for each layout
    for layout in _VISUALIZATION_LAYOUTS:
        title = f"{model.name} - {layout.title()} View"
        filename_png = output_dir / f"{prefix}model_{layout}.png"
        
        # Adjust parameters based on layout
        node_size = _DEFAULT_NODE_SIZE
        connection_style = _DEFAULT_CONNECTION_STYLE
        show_edge_labels = True
        
        if layout == "circular":
            connection_style = 'arc3,rad=0.2'
            show_edge_labels = False
        elif layout == "period_spring":
            node_size = 1800
            connection_style = 'arc3,rad=0.2'
            show_edge_labels = False
        
        # Only show forward edges for the "forward" layout
        edge_type = 'forward' if layout == "forward" else 'both'
        
        # Determine which edge styles to use based on edge_type
        edge_style_mapping = _EDGE_STYLES
        if edge_type == 'forward':
            edge_style_mapping = {
                k: v for k, v in _EDGE_STYLES.items() 
                if k.startswith('forward')
            }
        
        # Additional parameters for period_spring layout
        kwargs = {}
        if layout == "period_spring":
            kwargs.update({
                'label_offset': 0.3,
                'figsize': (14, 10),
                'dpi': 150,
                'with_edge_legend': True
            })
        
        # Call the model's visualization method
        model.visualize_stage_graph(
            edge_type=edge_type,
            layout=layout,
            node_size=node_size,
            title=title,
            edge_style_mapping=edge_style_mapping,
            show_node_labels=True,
            show_edge_labels=show_edge_labels,
            connectionstyle=connection_style,
            filename=filename_png,
            **kwargs
        )
        
        logger.info(f"Created {layout} visualization: {filename_png}")
        
        # Optionally save SVG version
        if save_svg:
            filename_svg = output_dir / f"{prefix}model_{layout}.svg"
            model.visualize_stage_graph(
                edge_type=edge_type,
                layout=layout,
                node_size=node_size,
                title=title,
                edge_style_mapping=edge_style_mapping,
                show_node_labels=True,
                show_edge_labels=show_edge_labels,
                connectionstyle=connection_style,
                filename=filename_svg,
                **kwargs
            )
            logger.info(f"Created {layout} SVG visualization: {filename_svg}")

# ==================
# Model Builder
# ==================

def initialize_model_Circuit(master_config: Dict, stage_configs: Dict[str, Dict], connections_config: Dict) -> Model:
    """
    Build a model circuit from configuration dictionaries. This function mimics the approach
    used in housing_renting_demo.py but in an automated manner.
    
    Args:
        master_config: Master configuration dictionary.
        stage_configs: Dictionary of stage configuration dictionaries (key=stage_name).
                      Should contain configs for ALL stages mentioned in master_config.
        connections_config: Connections configuration dictionary.
        
    Returns:
        Fully configured Model object.
    """
    logger.info(f"Building model from provided configurations")
    
    # Phase 0: Validate inputs
    _phase0_validate_inputs(master_config, stage_configs, connections_config)
    
    # Create the model
    model_name = master_config.get("name", "UnnamedModel")
    model = Model(name=model_name)
    logger.info(f"Creating ModelCircuit: {model_name}")
    
    # Determine the maximum number of periods from the master config
    max_periods = master_config.get("num_periods", master_config.get("horizon", 1))
    
    # Determine which periods actually need to be created based on connections
    required_periods = determine_required_periods(connections_config)
    
    # Ensure required_periods doesn't exceed max_periods
    required_periods = {p for p in required_periods if p < max_periods}
    
    # If no required_periods were found, default to period range 0 to max_periods
    if not required_periods:
        required_periods = set(range(max_periods))
    
    logger.info(f"Creating model with {len(required_periods)} periods: {sorted(required_periods)}")
    
    # Phase 1: Create all periods with their stages
    periods = _phase1_create_periods(master_config, stage_configs, required_periods)
    
    # Phase 2: Establish intra-period connections
    _phase2_create_intra_period_edges(periods, connections_config)
    
    # Phase 3: Register periods with the model
    _phase3_register_periods(model, periods)
    
    # Phase 4: Create inter-period connections
    _phase4_create_inter_period_edges(model, connections_config)
    
    return model

def _phase0_validate_inputs(master_cfg: Dict, stage_cfgs: Dict[str, Dict], conn_cfg: Dict) -> None:
    """
    Validate presence of mandatory keys and data types.
    
    Args:
        master_cfg: Master configuration dictionary.
        stage_cfgs: Dictionary of stage configurations.
        conn_cfg: Connections configuration dictionary.
        
    Raises:
        LoaderError: If validation fails
    """
    # Validate master config
    if not isinstance(master_cfg, dict):
        raise LoaderError(f"Master config must be a dictionary, got {type(master_cfg).__name__}")
    
    # Ensure all imported stages have configs
    if "imports" in master_cfg:
        imports = master_cfg.get("imports", [])
        for import_item in imports:
            stage_alias = import_item.get("alias", import_item.get("stage_name"))
            
            if not stage_alias:
                raise LoaderError(f"Import item missing 'alias' or 'stage_name': {import_item}")
                
            if stage_alias not in stage_cfgs:
                raise LoaderError(f"Stage '{stage_alias}' imported in master config but missing from stage_cfgs")
    
    # Validate connection config structure
    if not isinstance(conn_cfg, dict):
        raise LoaderError(f"Connections config must be a dictionary, got {type(conn_cfg).__name__}")
    
    # Validate intra-period connections
    intra_period = conn_cfg.get("intra_period", {})
    if not (isinstance(intra_period, dict) or isinstance(intra_period, list)):
        raise LoaderError(f"intra_period must be a dict or list, got {type(intra_period).__name__}")
    
    # Validate inter-period connections
    inter_period = conn_cfg.get("inter_period", [])
    if not isinstance(inter_period, list):
        raise LoaderError(f"inter_period must be a list, got {type(inter_period).__name__}")

def _phase1_create_periods(
        master_cfg: Dict,
        stage_cfgs: Dict[str, Dict],
        required_idx: Set[int]
) -> Dict[int, Period]:
    """
    Create unregistered Period instances with fully populated stages but no movers yet.
    
    Args:
        master_cfg: Master configuration dictionary.
        stage_cfgs: Dictionary of stage configurations.
        required_idx: Set of period indices to create.
        
    Returns:
        Dictionary mapping period indices to Period instances.
    """
    periods_created = {}  # Store created periods by index
    
    for period_idx in sorted(required_idx):
        logger.info(f"Setting up Period {period_idx}")
        
        # Create the period
        period = Period(time_index=period_idx)
        periods_created[period_idx] = period
        
        # Get stage configs from either imports or stages section
        if "imports" in master_cfg:
            imports = master_cfg.get("imports", [])
            for import_item in imports:
                stage_alias = import_item.get("alias", import_item.get("stage_name"))
                
                if stage_alias and stage_alias in stage_cfgs:
                    stage_config = stage_cfgs[stage_alias]
                    
                    # Create and initialize the stage with both configs
                    stage = create_stage(stage_alias, stage_config, master_cfg)
                    
                    # Add the stage to the period
                    period.add_stage(stage_alias, stage)
                else:
                    logger.warning(f"Stage {stage_alias} configuration not found in stage_cfgs dictionary")
        else:
            # Process stage configurations for this period
            stages_config = master_cfg.get("stages", {})
            for stage_name, stage_info in stages_config.items():
                # Get the stage configuration from the provided stage_configs dictionary
                if stage_name in stage_cfgs:
                    stage_config = stage_cfgs[stage_name]
                    
                    # Create and initialize the stage with both configs
                    stage = create_stage(stage_name, stage_config, master_cfg)
                    
                    # Add the stage to the period
                    period.add_stage(stage_name, stage)
                else:
                    logger.warning(f"Stage {stage_name} configuration not found in stage_cfgs dictionary")
    
    return periods_created

def _phase2_create_intra_period_edges(
        periods: Dict[int, Period],
        connections_config: Dict
) -> None:
    """
    Create intra-period connections for all periods.
    
    Args:
        periods: Dictionary mapping period indices to Period instances.
        connections_config: Connections configuration dictionary.
    """
    for period_idx, period in periods.items():
        # Process intra-period connections for this period
        for conn_dict in _iter_intra_conn(period_idx, connections_config):
            _apply_intra_conn(period, conn_dict)

def _phase3_register_periods(model: Model, periods: Dict[int, Period]) -> None:
    """
    Add periods to the model in ascending order.
    
    Args:
        model: The model to add periods to.
        periods: Dictionary mapping period indices to Period instances.
    """
    # Add periods to the model in order
    for period_idx in sorted(periods.keys()):
        period = periods[period_idx]
        model.add_period(period)
        logger.info(f"Added Period {period_idx} to model with {len(period.stages)} stages")

def _phase4_create_inter_period_edges(model: Model, connections_config: Dict) -> None:
    """
    Create inter-period connections between periods in the model.
    
    Args:
        model: The model to add connections to.
        connections_config: Connections configuration dictionary.
    """
    # Process inter-period connections
    create_inter_period_connections(model, connections_config)

# ==================
# Internal Helper Functions
# ==================

def determine_required_periods(connections_config: Dict) -> Set[int]:
    """
    Determine the periods that need to be created based on connections.
    
    Args:
        connections_config: Dictionary containing connection specifications.
        
    Returns:
        Set of period indices that need to be created.
    """
    required_periods = set()
    
    # Check intra-period connections
    intra_period_connections = connections_config.get("intra_period", {})
    
    # Handle if intra_period_connections is a dictionary with period indices as keys
    if isinstance(intra_period_connections, dict):
        for period_idx_str in intra_period_connections.keys():
            try:
                period_idx = _as_int(period_idx_str, context="intra_period key")
                required_periods.add(period_idx)
            except ConfigKeyError as e:
                logger.warning(str(e))
    # Handle if intra_period_connections is a list of dictionaries with 'period' keys
    elif isinstance(intra_period_connections, list):
        for conn in intra_period_connections:
            if isinstance(conn, dict):
                # Extract period indices from the connection object
                periods = _resolve_period_indices(conn)
                required_periods.update(periods)
    
    # Check inter-period connections
    inter_period_connections = connections_config.get("inter_period", [])
    for conn in inter_period_connections:
        if not isinstance(conn, dict):
            continue
        
        # Extract period indices from the connection object
        periods = _resolve_period_indices(conn)
        required_periods.update(periods)
    
    return required_periods

def _iter_intra_conn(period_idx: int, conn_cfg: Dict) -> Iterator[Dict]:
    """
    Yield normalized intra-period connection dictionaries.
    
    Args:
        period_idx: The period index to get connections for.
        conn_cfg: The overall connections configuration dictionary.
        
    Yields:
        Normalized connection dictionaries with 'source', 'target', 'direction', and 'branch_key' keys.
    """
    intra_period_connections = conn_cfg.get("intra_period", {})
    
    # Handle dictionary format with period indices as keys
    if isinstance(intra_period_connections, dict):
        period_connections = intra_period_connections.get(str(period_idx), {})
        
        # Process forward connections
        for conn in period_connections.get("forward", []):
            if "source" in conn and "target" in conn:
                yield {
                    "source": conn["source"],
                    "target": conn["target"],
                    "direction": "forward",
                    "branch_key": conn.get("branch_key")
                }
        
        # Process backward connections
        for conn in period_connections.get("backward", []):
            if "source" in conn and "target" in conn:
                yield {
                    "source": conn["source"],
                    "target": conn["target"],
                    "direction": "backward",
                    "branch_key": conn.get("branch_key")
                }
    
    # Handle list format
    elif isinstance(intra_period_connections, list):
        for conn in intra_period_connections:
            if not isinstance(conn, dict):
                continue
                
            # Check if this connection applies to our period
            periods_spec = conn.get('periods', conn.get('period', 'all'))
            
            apply_to_this_period = False
            if periods_spec == 'all':
                apply_to_this_period = True
            elif isinstance(periods_spec, list) and period_idx in periods_spec:
                apply_to_this_period = True
            elif isinstance(periods_spec, (int, str)) and _as_int(periods_spec, context="periods field") == period_idx:
                apply_to_this_period = True
            
            if not apply_to_this_period:
                continue
                
            # Extract required fields
            source_name = conn.get("source")
            target_name = conn.get("target")
            
            if not source_name or not target_name:
                logger.warning(f"Missing source or target in connection: {conn}")
                continue
                
            # Determine direction
            direction = conn.get("direction", "forward")
            
            yield {
                "source": source_name,
                "target": target_name,
                "direction": direction,
                "branch_key": conn.get("branch_key")
            }

def _apply_intra_conn(period: Period, conn_dict: Dict) -> None:
    """
    Apply an intra-period connection to a period.
    
    Args:
        period: The period to add the connection to.
        conn_dict: The normalized connection dictionary.
    """
    source_name = conn_dict["source"]
    target_name = conn_dict["target"]
    direction = conn_dict["direction"]
    branch_key = conn_dict.get("branch_key")
    
    try:
        if direction == "forward":
            period.connect_fwd(
                src=source_name,
                tgt=target_name,
                branch_key=branch_key
            )
            logger.info(f"Created forward connection: {source_name} -> {target_name} in period {period.time_index}")
        else:
            period.connect_bwd(
                src=source_name,
                tgt=target_name,
                branch_key=branch_key
            )
            logger.info(f"Created backward connection: {source_name} -> {target_name} in period {period.time_index}")
    except Exception as e:
        logger.warning(f"Error creating {direction} connection {source_name} -> {target_name} in period {period.time_index}: {e}")

def create_inter_period_connections(model: Model, connections_config: Dict) -> None:
    """
    Create inter-period connections between periods in the model.
    
    Args:
        model: The model to add connections to.
        connections_config: Dictionary containing connection specifications.
    """
    # Process inter-period connections
    inter_period_connections = connections_config.get("inter_period", [])
    for conn in inter_period_connections:
        # Handle source_periods and target_periods arrays
        if "source_periods" in conn and "target_periods" in conn:
            _create_multi_inter_period_connections(model, conn)
        else:
            # Handle single source_period and target_period
            _create_single_inter_period_connection(model, conn)

def _create_multi_inter_period_connections(model: Model, conn: Dict) -> None:
    """
    Create multiple inter-period connections for paired arrays of periods.
    
    Args:
        model: The model to add connections to.
        conn: Connection configuration with source_periods and target_periods arrays.
    """
    source_periods = _ensure_list(conn.get("source_periods", []))
    target_periods = _ensure_list(conn.get("target_periods", []))
    source_stage_name = conn.get("source")
    target_stage_name = conn.get("target")
    
    if not source_stage_name or not target_stage_name:
        logger.warning(f"Missing source or target stage in inter-period connection: {conn}")
        return
    
    # Ensure the lists have the same length
    if len(source_periods) != len(target_periods):
        logger.warning(f"source_periods and target_periods have different lengths")
        return
    
    # Create connections for each pair
    for source_period_idx, target_period_idx in zip(source_periods, target_periods):
        try:
            source_period_idx = _as_int(source_period_idx, context="source_periods item")
            target_period_idx = _as_int(target_period_idx, context="target_periods item")
            
            _create_single_inter_period_connection_impl(
                model,
                source_period_idx,
                target_period_idx,
                source_stage_name,
                target_stage_name,
                branch_key=conn.get("branch_key"),
                create_transpose=conn.get("create_transpose", True),
                mover_name=conn.get("name"),
                source_perch_attr=conn.get("source_perch", "cntn"),
                target_perch_attr=conn.get("target_perch", "arvl")
            )
        except (ConfigKeyError, Exception) as e:
            logger.warning(f"Error creating inter-period connection: {e}")

def _create_single_inter_period_connection(model: Model, conn: Dict) -> None:
    """
    Create a single inter-period connection.
    
    Args:
        model: The model to add connections to.
        conn: Connection configuration with single source_period and target_period.
    """
    source_period_idx = conn.get("source_period")
    target_period_idx = conn.get("target_period")
    source_stage_name = conn.get("source")
    target_stage_name = conn.get("target")
    
    if source_period_idx is None or target_period_idx is None:
        logger.warning(f"Missing source_period or target_period in inter-period connection: {conn}")
        return
        
    if not source_stage_name or not target_stage_name:
        logger.warning(f"Missing source or target stage in inter-period connection: {conn}")
        return
    
    try:
        source_period_idx = _as_int(source_period_idx, context="source_period")
        target_period_idx = _as_int(target_period_idx, context="target_period")
        
        _create_single_inter_period_connection_impl(
            model,
            source_period_idx,
            target_period_idx,
            source_stage_name,
            target_stage_name,
            branch_key=conn.get("branch_key"),
            create_transpose=conn.get("create_transpose", True),
            mover_name=conn.get("name"),
            source_perch_attr=conn.get("source_perch", "cntn"),
            target_perch_attr=conn.get("target_perch", "arvl")
        )
    except (ConfigKeyError, Exception) as e:
        logger.warning(f"Error creating inter-period connection: {e}")

def _create_single_inter_period_connection_impl(
    model: Model,
    source_period_idx: int,
    target_period_idx: int,
    source_stage_name: str,
    target_stage_name: str,
    branch_key: Optional[str] = None,
    create_transpose: bool = True,
    mover_name: Optional[str] = None,
    source_perch_attr: str = "cntn",
    target_perch_attr: str = "arvl"
) -> None:
    """
    Implementation of creating a single connection between stages in different periods.
    
    Args:
        model: Model containing the periods.
        source_period_idx: Index of the source period.
        target_period_idx: Index of the target period.
        source_stage_name: Name of the source stage.
        target_stage_name: Name of the target stage.
        branch_key: Optional branch key for conditional flow.
        create_transpose: Whether to create a transpose connection.
        mover_name: Optional name for the mover.
        source_perch_attr: Perch attribute for the source.
        target_perch_attr: Perch attribute for the target.
    """
    try:
        # Get periods
        source_period = model.get_period(source_period_idx)
        target_period = model.get_period(target_period_idx)
        
        if not source_period or not target_period:
            logger.warning(f"Period {source_period_idx} or {target_period_idx} not found, skipping connection")
            return
        
        # Get stages
        if source_stage_name not in source_period.stages:
            logger.warning(f"Source stage '{source_stage_name}' not found in period {source_period_idx}")
            return
        if target_stage_name not in target_period.stages:
            logger.warning(f"Target stage '{target_stage_name}' not found in period {target_period_idx}")
            return
        
        source_stage = source_period.stages[source_stage_name]
        target_stage = target_period.stages[target_stage_name]
        
        # Use parameter names and order exactly matching housing_renting_demo.py
        model.add_inter_period_connection(
            source_period=source_period,
            target_period=target_period,
            source_stage=source_stage,
            target_stage=target_stage,
            source_perch_attr=source_perch_attr,
            target_perch_attr=target_perch_attr,
            branch_key=branch_key,
            mover_name=mover_name,
            create_transpose=create_transpose
        )
        
        logger.info(
            f"Created inter-period connection: Period {source_period_idx}.{source_stage_name} -> "
            f"Period {target_period_idx}.{target_stage_name}"
        )
    except Exception as e:
        logger.warning(f"Error creating inter-period connection: {e}")



