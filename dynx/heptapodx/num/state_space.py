"""
State space generation module for Heptapod-B.

This module provides functions to generate numerical grids based on
various specifications:
- Uniform (linspace)
- Logarithmic (geomspace)
- Chebyshev
- Integer range
- Explicit list
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import warnings
import copy

from ..core.validation import GRID_TYPE_ALIASES, GRID_TYPE_REQUIRED_KEYS


def resolve_reference(value, params=None, _recursion_depth=0, _seen_refs=None):
    """
    Recursively resolves a reference to its actual value.
    
    Parameters
    ----------
    value : Any
        The value to resolve which might be a reference
    params : dict, optional
        Dictionary of parameters to resolve references against
    _recursion_depth : int, optional
        Current recursion depth (internal use for detecting infinite recursion)
    _seen_refs : set, optional
        Set of references already seen (internal use for detecting circular references)
        
    Returns
    -------
    Any
        The resolved value
        
    Raises
    ------
    ValueError
        If a reference could not be resolved or a circular reference is detected
    RecursionError
        If maximum recursion depth is exceeded
    """
    # Maximum recursion depth to prevent infinite recursion
    MAX_RECURSION_DEPTH = 10
    
    if _recursion_depth > MAX_RECURSION_DEPTH:
        raise RecursionError(f"Maximum recursion depth ({MAX_RECURSION_DEPTH}) exceeded while resolving references")
    
    # Initialize tracking set if this is the first call
    if _seen_refs is None:
        _seen_refs = set()
    
    # Non-reference values pass through unchanged
    if not params or value is None or isinstance(value, (int, float, bool)):
        return value
        
    # Handle list references - format: ['reference_name']
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], str):
        ref_key = value[0]
        
        # Detect cycles - if we've seen this reference before, raise error
        if ref_key in _seen_refs:
            raise ValueError(f"Circular reference detected: {ref_key}")
            
        # Add this reference to seen set
        _seen_refs.add(ref_key)
            
        if ref_key in params:
            # Get the parameter value
            param_value = params[ref_key]
            
            # Special case - if the parameter value is already a primitive type,
            # return it directly without further resolution
            if isinstance(param_value, (int, float, bool, str)):
                return param_value
                
            # Otherwise, recursively resolve the reference in case it points to another reference
            return resolve_reference(param_value, params, _recursion_depth + 1, _seen_refs)
        else:
            # If we can't find the reference but it seems like a numeric string, try converting it
            if ref_key.isdigit():
                try:
                    return int(ref_key)
                except (ValueError, TypeError):
                    pass
                
            # If all else fails, raise error    
            raise ValueError(f"Unresolved reference: {ref_key}")
    
    # Handle string references - direct parameter names
    elif isinstance(value, str):
        # Handle direct grid types without further resolution
        if value in ['linspace', 'geomspace', 'chebyshev', 'int_range', 'list']:
            return value
            
        # Detect cycles - if we've seen this reference before, raise error
        if value in _seen_refs:
            raise ValueError(f"Circular reference detected: {value}")
            
        # Check if it's a direct parameter reference
        if value in params:
            # Add this reference to seen set
            _seen_refs.add(value)
            
            # Special case - if a string reference is the same as its name, it's a circular reference
            # For example, if we see "m_min" and params["m_min"] is also "m_min", it's circular
            if params[value] == value:
                raise ValueError(f"Self-referential parameter: {value}")
                
            # Recursively resolve the reference in case it points to another reference
            return resolve_reference(params[value], params, _recursion_depth + 1, _seen_refs)
            
        # If we can't find the reference but it seems like a numeric string, try converting it
        if value.isdigit():
            try:
                return int(value)
            except (ValueError, TypeError):
                pass
    
    # Return unchanged if not a reference or reference not found
    return value


def generate_grid(dim_spec, all_params):
    """
    Generate a grid of points for a single dimension based on a specification.
    
    Args:
        dim_spec: Dictionary specifying the dimension grid.
            Must contain the key 'type' which can be one of:
            - 'linspace': Requires 'min', 'max', 'n' parameters
            - 'geomspace': Requires 'min', 'max', 'n' parameters
            - 'chebyshev': Requires 'min', 'max', 'n' parameters
            - 'int_range': Requires 'start', 'stop', 'step' parameters
            - 'list': Requires 'values' parameter
            Or a reference to a grid type in the format ['grid_type_name']
        all_params: Dictionary containing all parameter values to resolve references
    
    Returns:
        numpy.ndarray: A 1D array representing the grid for this dimension
    
    Raises:
        ValueError: If required keys are missing or if an unknown grid type is specified
    """
    # Deep copy to avoid modifying the original
    dim_spec = copy.deepcopy(dim_spec)
    
    # Validate required keys
    if 'type' not in dim_spec:
        raise ValueError("Missing required key 'type' in dimension specification")
    
    # Resolve the grid type reference if needed
    try:
        grid_type = resolve_reference(dim_spec['type'], all_params)
    except ValueError as e:
        warnings.warn(f"Error resolving grid type reference: {e}. Using 'linspace' as fallback.")
        grid_type = 'linspace'
    
    # Handle default grid or grid type aliases
    if grid_type == 'default_grid' or grid_type == 'unspecified':
        if 'default_grid' in all_params:
            grid_type = all_params['default_grid']
        else:
            # Use linspace as fallback only if specifically configured that way
            grid_type = 'linspace'
    elif grid_type in GRID_TYPE_ALIASES:
        grid_type = GRID_TYPE_ALIASES[grid_type]
    
    # Create a parameters dictionary for this grid
    params = {}
    
    # Copy relevant parameters and resolve references
    for key, value in dim_spec.items():
        if key != 'type':  # Skip the type key
            try:
                params[key] = resolve_reference(value, all_params)
            except Exception as e:
                warnings.warn(f"Error resolving parameter '{key}': {e}. Using value directly.")
                params[key] = value
    
    # Handle 'points' key if present (convert to 'n' for compatibility)
    if 'points' in params and 'n' not in params:
        params['n'] = params['points']
    
    # Generate grid based on the specified type
    if grid_type == 'linspace':
        # Check required parameters
        for req_param in ['min', 'max', 'n']:
            if req_param not in params:
                raise ValueError(f"Missing required parameter '{req_param}' for linspace grid")
        
        # Ensure parameters are numeric
        try:
            min_val = float(params['min'])
            max_val = float(params['max'])
            n_points = int(params['n'])
        except (ValueError, TypeError) as e:
            raise ValueError(f"Error converting linspace parameters: {e}")
        
        return np.linspace(min_val, max_val, n_points)
    
    elif grid_type == 'geomspace':
        # Check required parameters
        for req_param in ['min', 'max', 'n']:
            if req_param not in params:
                raise ValueError(f"Missing required parameter '{req_param}' for geomspace grid")
        
        # Ensure parameters are numeric
        try:
            min_val = float(params['min'])
            max_val = float(params['max'])
            n_points = int(params['n'])
        except (ValueError, TypeError) as e:
            raise ValueError(f"Error converting geomspace parameters: {e}")
        
        # Ensure min is positive for geomspace
        min_val = max(min_val, 1e-10)  # Avoid zero or negative values
        return np.geomspace(min_val, max_val, n_points)
    
    elif grid_type == 'chebyshev':
        # Check required parameters
        for req_param in ['min', 'max', 'n']:
            if req_param not in params:
                raise ValueError(f"Missing required parameter '{req_param}' for chebyshev grid")
        
        # Ensure parameters are numeric
        try:
            min_val = float(params['min'])
            max_val = float(params['max'])
            n_points = int(params['n'])
        except (ValueError, TypeError) as e:
            raise ValueError(f"Error converting chebyshev parameters: {e}")
        
        # Generate Chebyshev nodes in [-1, 1]
        k = np.arange(n_points)
        cheb_nodes = np.cos((2 * k + 1) * np.pi / (2 * n_points))
        
        # Transform to desired range
        return 0.5 * (min_val + max_val) + 0.5 * (max_val - min_val) * cheb_nodes
    
    elif grid_type == 'int_range':
        # Check required parameters
        for req_param in ['start', 'stop', 'step']:
            if req_param not in params:
                raise ValueError(f"Missing required parameter '{req_param}' for int_range grid")
        
        # Convert parameters to integers
        try:
            start = int(params['start'])
            stop = int(params['stop'])
            step = int(params['step'])
        except (ValueError, TypeError) as e:
            raise ValueError(f"Error converting int_range values: {str(e)}")
        
        return np.arange(start, stop + 1, step)  # +1 to include stop value
    
    elif grid_type == 'list':
        # Check required parameters
        if 'values' not in params:
            raise ValueError("Missing required parameter 'values' for list grid")
        
        values = params['values']
        if not isinstance(values, list):
            raise ValueError(f"Parameter 'values' must be a list, got {type(values)}")
        
        return np.array(values)
    
    else:
        warnings.warn(f"Unknown grid type: {grid_type}. Using linspace as fallback.")
        
        # Check if we have the necessary parameters for linspace
        if all(k in params for k in ['min', 'max', 'n']):
            try:
                min_val = float(params['min'])
                max_val = float(params['max'])
                n_points = int(params['n'])
                return np.linspace(min_val, max_val, n_points)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Error using linspace fallback: {e}")
        else:
            raise ValueError(f"Cannot use linspace fallback: missing required parameters")


def int_range(start, stop, step):
    """
    Create an integer range from start to stop (inclusive) with the given step.
    
    Args:
        start: First value in the range
        stop: Last value in the range (inclusive)
        step: Step size between values
        
    Returns:
        numpy.ndarray: Array of integers from start to stop (inclusive) with given step
    """
    return np.arange(start, stop + 1, step)


def generate_chebyshev_grid(
    min_val: float,
    max_val: float,
    n_points: int
) -> np.ndarray:
    """
    Generate a Chebyshev grid on the interval [min_val, max_val].
    
    Parameters
    ----------
    min_val : float
        Minimum value of the grid
    max_val : float
        Maximum value of the grid
    n_points : int
        Number of grid points
        
    Returns
    -------
    numpy.ndarray
        Chebyshev grid
    """
    # Generate Chebyshev nodes on [-1, 1]
    k = np.arange(n_points)
    x = -np.cos((2*k + 1) * np.pi / (2*n_points))
    
    # Scale to [min_val, max_val]
    return 0.5 * (max_val - min_val) * (x + 1) + min_val


def create_mesh_grid(dimension_grids: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Create a mesh grid from 1D grids for each dimension.
    
    Parameters
    ----------
    dimension_grids : dict
        Dictionary mapping dimension names to 1D grids
        
    Returns
    -------
    tuple
        Tuple containing:
        - Dictionary mapping dimension names to mesh grid arrays
        - Full tensor as an array of shape (n_points, n_dimensions)
    """
    # Extract dimension names and grids
    dimensions = list(dimension_grids.keys())
    grids = list(dimension_grids.values())
    
    # Create mesh grid
    mesh_arrays = np.meshgrid(*grids, indexing='ij')
    
    # Create dictionary mapping dimension names to mesh arrays
    mesh_dict = {dim: mesh_arrays[i].flatten() for i, dim in enumerate(dimensions)}
    
    # Create full tensor
    full_tensor = np.stack([mesh_dict[dim] for dim in dimensions], axis=1)
    
    return mesh_dict, full_tensor


def generate_numerical_state_space(problem, methods=None):
    """
    Generate numerical state space grids from analytical state definitions.

    This function iterates through the state definitions in problem.math['state_space']
    and generates numerical grids for each dimension specified within each state.
    
    Parameters
    ----------
    problem : FunctionalProblem
        The model with analytical state definitions
    methods : dict, optional
        Methods dictionary with grid generation settings
        
    Returns
    -------
    dict
        Dictionary representing the numerical state space structure
    """
    if not hasattr(problem, "math") or "state_space" not in problem.math:
        warnings.warn(
            "problem.math['state_space'] not found. Skipping state space generation.",
            UserWarning
        )
        return {}

    # Get the analytical state definitions (from the model)
    analytical_states = problem.math["state_space"]

    # Ensure the numerical state space container exists
    if not hasattr(problem, "num"):
        problem.num = {}
    if "state_space" not in problem.num:
        problem.num["state_space"] = {}

    # Get global parameters and settings
    # These are pure values that can be used directly
    global_params = {}
    if hasattr(problem, "parameters_dict"):
        global_params.update(problem.parameters_dict)
    if hasattr(problem, "settings_dict"):
        global_params.update(problem.settings_dict)
    if hasattr(problem, "methods"):
        global_params.update(problem.methods)
    
    # Keep a clean copy of the global parameters for reference resolution
    global_params_clean = global_params.copy()
    
    # Iterate through each defined state (perch)
    for state_name, state_info in analytical_states.items():
        # Skip if already processed
        if state_name in problem.num["state_space"] and problem.num["state_space"][state_name]:
            continue
            
        # Ensure perch-level storage exists
        problem.num["state_space"][state_name] = {}

        # Get dimensions for this state
        dimensions = state_info.get("dimensions")
        if not dimensions or not isinstance(dimensions, list):
            warnings.warn(
                f"Invalid or missing 'dimensions' list for state '{state_name}'. Skipping grid generation.",
                UserWarning
            )
            continue

        # Store the dimensions in the numerical state space
        problem.num["state_space"][state_name]["dimensions"] = dimensions

        # Process state-specific settings:
        # 1. Extract raw state settings
        # 2. Resolve any references to global parameters
        state_settings = {}
        if "settings_dict" in state_info:
            # Copy state-specific settings
            raw_settings = state_info["settings_dict"].copy()
            
            # Process each setting, resolving any global references
            for key, value in raw_settings.items():
                if isinstance(value, list) and len(value) == 1 and isinstance(value[0], str):
                    # This is a reference, check if it exists in globals
                    ref_key = value[0]
                    if ref_key in global_params_clean:
                        # Use the global value directly
                        state_settings[key] = global_params_clean[ref_key]
                    else:
                        # Keep as a reference to be resolved later
                        state_settings[key] = value
                else:
                    # Use the value directly
                    state_settings[key] = value
        
        # Get state methods
        state_methods = {}
        if "methods" in state_info:
            state_methods.update(state_info["methods"])
                
        # Build the final combined params:
        # 1. Start with global parameters
        # 2. Override with resolved state settings
        # 3. Override with state methods
        combined_params = {**global_params}
        combined_params.update(state_settings)
        combined_params.update(state_methods)
        
        # Add default_grid explicitly
        if "default_grid" not in combined_params and hasattr(problem, "methods"):
            if "default_grid" in problem.methods:
                combined_params["default_grid"] = problem.methods["default_grid"]
            else:
                combined_params["default_grid"] = "linspace"  # Default fallback

        # Initialize dictionary to store grids for each dimension
        dimension_grids = {}

        # Handle different grid specification formats
        grid_spec = state_info.get("grid")
        if grid_spec is None:
            # For perch models, check if there's a state definition in problem.math.state 
            if hasattr(problem.math, "state"):
                # Try to extract grid from state definitions
                state_def_found = True
                dimension_grids = {}
                
                for dim in dimensions:
                    if hasattr(problem.math.state, dim):
                        state_def = getattr(problem.math.state, dim)
                        if hasattr(state_def, "grid"):
                            try:
                                # Generate grid for this dimension
                                grid = generate_grid(state_def.grid, combined_params)
                                dimension_grids[dim] = grid
                            except Exception as e:
                                warnings.warn(
                                    f"Error generating grid for dimension '{dim}' in state '{state_name}': {e}",
                                    UserWarning
                                )
                    else:
                        state_def_found = False
                        warnings.warn(
                            f"Dimension '{dim}' not found in state definitions for state '{state_name}'",
                            UserWarning
                        )
                
                if not state_def_found or not dimension_grids:
                    warnings.warn(
                        f"No grid specification found for state '{state_name}'. Skipping.",
                        UserWarning
                    )
                    continue
            else:
                warnings.warn(
                    f"No grid specification found for state '{state_name}'. Skipping.",
                    UserWarning
                )
                continue

        # Case 1: Grid is a list (for 1D states only)
        elif isinstance(grid_spec, list):
            if len(dimensions) != 1:
                warnings.warn(
                    f"State '{state_name}' has multiple dimensions but grid is a flat list.",
                    UserWarning
                )
                continue
                
            # Use the list directly as the grid for the single dimension
            dimension_grids[dimensions[0]] = np.array(grid_spec)

        # Case 2: Grid is a mapping of dimension names to grid specifications
        elif isinstance(grid_spec, dict):
            for dim_name in dimensions:
                if dim_name not in grid_spec:
                    warnings.warn(
                        f"Missing grid specification for dimension '{dim_name}' in state '{state_name}'.",
                        UserWarning
                    )
                    continue

                dim_grid_spec = grid_spec[dim_name]

                # Case 2.1: Direct list of values
                if isinstance(dim_grid_spec, list):
                    dimension_grids[dim_name] = np.array(dim_grid_spec)

                # Case 2.2: int_range style mapping
                elif isinstance(dim_grid_spec, dict) and "start" in dim_grid_spec and "stop" in dim_grid_spec:
                    step = dim_grid_spec.get("step", 1)
                    
                    # Make sure we convert all values to proper numeric types
                    # First try resolving references
                    start_val = resolve_reference(dim_grid_spec["start"], combined_params)
                    stop_val = resolve_reference(dim_grid_spec["stop"], combined_params) 
                    step_val = resolve_reference(step, combined_params)
                    
                    # Then convert to integers
                    start = int(start_val)
                    stop = int(stop_val)
                    step = int(step_val)
                    
                    dimension_grids[dim_name] = np.arange(
                        start,
                        stop + 1,  # Make it inclusive
                        step
                    )
                # Case 2.3: Algorithmic grid specification with 'type'
                elif isinstance(dim_grid_spec, dict) and "type" in dim_grid_spec:
                    dimension_grids[dim_name] = generate_grid(dim_grid_spec, combined_params)
                else:
                    warnings.warn(
                        f"Invalid grid specification for dimension '{dim_name}' in state '{state_name}'.",
                        UserWarning
                    )
                    continue
        else:
            warnings.warn(
                f"Invalid grid specification format for state '{state_name}'.",
                UserWarning
            )
            continue

        # All dimensions have valid grids
        if len(dimension_grids) != len(dimensions):
            warnings.warn(
                f"Not all dimensions have valid grids for state '{state_name}'. "
                f"Expected {len(dimensions)}, got {len(dimension_grids)}.",
                UserWarning
            )
            continue

        # Store the 1D grids in the numerical state space
        problem.num["state_space"][state_name]["grids"] = dimension_grids

        # Create mesh grid if needed
        create_mesh = state_info.get("methods", {}).get("create_mesh", True)
        if create_mesh and len(dimensions) > 1:
            try:
                mesh_dict, full_tensor = create_mesh_grid(dimension_grids)
                problem.num["state_space"][state_name]["mesh"] = mesh_dict
                problem.num["state_space"][state_name]["tensor"] = full_tensor
            except Exception as e:
                warnings.warn(
                    f"Error creating mesh grid for state '{state_name}': {str(e)}",
                    UserWarning
                )

    return problem.num["state_space"]


def build_state_space(problem):
    """
    Build the state space for a problem by generating grids for each dimension of each state.
    
    Args:
        problem: A FunctionalProblem instance with problem configuration
        
    Returns:
        Dictionary mapping state names to dictionaries containing:
            - 'dimensions': List of dimension names for the state
            - 'grid_dict': Dictionary mapping dimension names to grid arrays
            - 'grid_tuple': Tuple of grid arrays in the order of dimensions
            - 'grid_shape': Tuple of grid sizes in the order of dimensions
            - 'grid_size': Total size of the grid (product of all dimension sizes)
    """
    state_space = {}
    all_params = problem.params.copy()  # Get all parameters from the problem
    
    for state_name, state_config in problem.states.items():
        # Initialize state entry
        state_space[state_name] = {
            'dimensions': [],
            'grid_dict': {},
            'grid_tuple': (),
            'grid_shape': (),
            'grid_size': 0
        }
        
        # Get dimensions for this state
        dimensions = state_config.get('dimensions', [])
        if not dimensions:
            warnings.warn(f"No dimensions specified for state '{state_name}'")
            continue
            
        state_space[state_name]['dimensions'] = dimensions
        
        # Generate grids for each dimension
        grid_dict = {}
        grid_arrays = []
        grid_shape = []
        
        for dim_name in dimensions:
            # Check if this dimension is defined in the problem's grid configuration
            if dim_name not in problem.grid:
                warnings.warn(f"Dimension '{dim_name}' referenced in state '{state_name}' "
                              f"but not defined in grid configuration")
                continue
                
            dim_spec = problem.grid[dim_name]
            
            try:
                # Generate the grid for this dimension
                grid = generate_grid(dim_spec, all_params)
                
                # Store the grid
                grid_dict[dim_name] = grid
                grid_arrays.append(grid)
                grid_shape.append(len(grid))
                
            except Exception as e:
                # Log the error and continue with other dimensions
                warnings.warn(f"Error generating grid for dimension '{dim_name}' in state '{state_name}': {str(e)}")
                continue
        
        # Check if we have grids for all dimensions
        if len(grid_arrays) != len(dimensions):
            warnings.warn(f"Not all dimensions have valid grids for state '{state_name}'. "
                          f"Expected {len(dimensions)}, got {len(grid_arrays)}.")
            
        # Update state space entry
        if grid_arrays:
            state_space[state_name]['grid_dict'] = grid_dict
            state_space[state_name]['grid_tuple'] = tuple(grid_arrays)
            state_space[state_name]['grid_shape'] = tuple(grid_shape)
            state_space[state_name]['grid_size'] = np.prod(grid_shape)
        
    return state_space


def build_grid(dim_name, dim_config, params):
    """
    Build a grid for a dimension based on its configuration.
    
    Args:
        dim_name: Name of the dimension
        dim_config: Configuration for the dimension
        params: Additional parameters
        
    Returns:
        numpy array of grid points
    """
    grid_type = resolve_reference(dim_config.get('type', ['default_grid']), params)
    
    # Handle the case where grid_type is returned as 'default_grid'
    if grid_type == 'default_grid':
        grid_type = 'linspace'
    
    # If grid_type is a string that matches a valid grid type, use it directly
    if isinstance(grid_type, str) and grid_type in GRID_TYPE_ALIASES:
        grid_function_name = GRID_TYPE_ALIASES[grid_type]
        
        # Get the grid function
        try:
            grid_function = globals()[grid_function_name]
        except KeyError:
            raise ValueError(f"Grid function '{grid_function_name}' not found")
        
        # Extract and resolve parameters for the grid function
        grid_params = {}
        for param_name, param_value in dim_config.items():
            if param_name != 'type':
                try:
                    grid_params[param_name] = resolve_reference(param_value, params)
                except (TypeError, ValueError) as e:
                    message = f"Error converting {param_name} values for dimension '{dim_name}': {str(e)}"
                    print(message)
                    return None
                    
        # Call the grid function with the resolved parameters
        try:
            return grid_function(**grid_params)
        except Exception as e:
            message = f"Error generating grid for dimension '{dim_name}': {str(e)}"
            print(message)
            return None
    else:
        print(f"Unknown grid type: '{grid_type}'")
        return None 