"""
Circuit Runner for parameter sweeps of economic models.

This module provides the CircuitRunner class, which prepares configurations
for parameter sweeps. It supports patching configurations with parameter
values, building and running models, and collecting metrics. It also includes
utilities for parallel execution with MPI.

Note — samplers should use `np.random.default_rng` for side-effect-free randomness.
"""

import pickle
import copy
import hashlib
import json
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    HAS_MPI = False

from dynx.runner.telemetry import RunRecorder


def set_deep(d: Dict[str, Any], path: str, val: Any) -> None:
    """
    Set a value in a nested dictionary, creating intermediate dictionaries if needed.

    Args:
        d: Dictionary to modify
        path: Path in the form "key1.key2.key3" specifying the location in the dictionary
        val: Value to set at the path

    Examples:
        >>> d = {}
        >>> set_deep(d, "a.b.c", 42)
        >>> d
        {'a': {'b': {'c': 42}}}
    """
    keys = path.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = val


class CircuitRunner:
    """
    Run economic models with parameter sweeps.

    CircuitRunner is designed to efficiently run economic models with different
    parameter values. It handles configuration patching and metric collection.

    Attributes:
        base_cfg: Base configuration dictionary for the model
        param_paths: List of parameter paths to patch (in the form "path.to.param")
        model_factory: Function to create model instances from a configuration
        solver: Function to solve a model
        simulator: Optional function to simulate a solved model
        metric_fns: Optional dictionary of functions that compute metrics from a model
        cache: Whether to cache results based on parameter values
        _cache: Internal cache of (parameter vector, metrics) pairs
    """

    def __init__(
        self,
        base_cfg: Dict[str, Any],
        param_paths: List[str],
        model_factory: Callable[[Dict[str, Any]], Any],
        solver: Callable[[Any], None],
        metric_fns: Optional[Dict[str, Callable[[Any], float]]] = None,
        simulator: Optional[Callable[[Any], None]] = None,
        sampler: Optional[Any] = None,
        cache: bool = False,
        validate_paths: bool = False,
    ):
        """
        Initialize a CircuitRunner.

        Args:
            base_cfg: Base configuration dictionary for the model
            param_paths: List of parameter paths to patch (in the form "path.to.param")
            model_factory: Function that creates model instances from a configuration
            solver: Function that solves a model
            metric_fns: Optional dictionary of functions that compute metrics from a model
            simulator: Optional function that simulates a solved model
            sampler: Optional sampler to use for generating parameter values (deprecated)
            cache: Whether to cache results based on parameter values
            validate_paths: Whether to validate that parameter paths exist in the configuration
        """
        if sampler is not None:
            warnings.warn(
                "`sampler` argument is deprecated and will be removed in v1.7; "
                "draw xs with dynx.runner.sampler.build_design() instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            
        self.base_cfg = copy.deepcopy(base_cfg)
        self.param_paths = param_paths
        self.model_factory = model_factory
        self.solver = solver
        self.metric_fns = metric_fns or {}
        self.simulator = simulator
        self.cache = cache
        self._cache = {} if cache else None
        
        # Validate parameter paths if requested
        if validate_paths:
            self._validate_param_paths()
    
    def _validate_param_paths(self) -> None:
        """
        Validate that all parameter paths exist in the configuration.
        
        Raises:
            ValueError: If a parameter path does not exist in the base configuration.
        """
        for path in self.param_paths:
            keys = path.split(".")
            d = self.base_cfg
            
            try:
                for key in keys[:-1]:
                    d = d[key]
                
                # Check if final key exists (but don't access it yet)
                if keys[-1] not in d:
                    raise KeyError
                
            except (KeyError, TypeError):
                raise ValueError(f"Parameter path '{path}' not found in configuration")
    
    def pack(self, d: Dict[str, Any]) -> np.ndarray:
        """
        Pack a dictionary of parameter values into a numpy array.
        
        Args:
            d: Dictionary mapping parameter paths to values
            
        Returns:
            Parameter vector with dtype=object for supporting mixed types
            
        Raises:
            ValueError: If a parameter path in d is not in param_paths
        """
        # Create array with object dtype to support mixed types
        x = np.empty(len(self.param_paths), dtype=object)
        
        # Fill array based on parameter paths order
        for i, path in enumerate(self.param_paths):
            if path in d:
                x[i] = d[path]
            else:
                raise ValueError(f"Parameter path '{path}' not found in input dictionary")
        
        return x
    
    def unpack(self, x: np.ndarray) -> Dict[str, Any]:
        """
        Unpack a parameter vector into a dictionary.
        
        Args:
            x: Parameter vector
            
        Returns:
            Dictionary mapping parameter paths to values
            
        Raises:
            ValueError: If the parameter vector length doesn't match param_paths
        """
        if len(x) != len(self.param_paths):
            raise ValueError(
                f"Parameter vector length ({len(x)}) doesn't match "
                f"number of parameters ({len(self.param_paths)})"
            )
        
        result = {}
        for i, path in enumerate(self.param_paths):
            result[path] = x[i]
            
        return result
    
    def patch_config(self, param_vec: np.ndarray) -> Dict[str, Any]:
        """
        Create a patched configuration by applying parameter values.

        Args:
            param_vec: Parameter vector of values to patch into the configuration

        Returns:
            A deep copy of the configuration with parameters patched

        Raises:
            ValueError: If the parameter vector length doesn't match param_paths
        """
        if len(param_vec) != len(self.param_paths):
            raise ValueError(
                f"Parameter vector length ({len(param_vec)}) doesn't match "
                f"number of parameters ({len(self.param_paths)})"
            )
        
        # Deep copy to avoid modifying the original
        cfg = copy.deepcopy(self.base_cfg)
        
        # Apply each parameter value
        for i, path in enumerate(self.param_paths):
            set_deep(cfg, path, param_vec[i])
        
        return cfg
    
    def run(
        self, 
        x: np.ndarray, 
        return_model: bool = False
    ) -> Union[Dict[str, Any], Tuple[Dict[str, Any], Any]]:
        """
        Run a model with the given parameter vector and collect metrics.

        Args:
            x: Parameter vector
            return_model: Whether to return the model instance

        Returns:
            dict[str, Any]: Dictionary of metrics
            model: (Optional) Model instance

        Raises:
            ValueError: If x doesn't match the number of parameters
        """
        if self.cache:
            # Hash the parameter vector for caching
            if x.dtype == object:
                key_bytes = pickle.dumps(x, protocol=5)
            else:
                key_bytes = x.tobytes()
            key = hashlib.md5(key_bytes).hexdigest()
            
            if key in self._cache:
                if return_model:
                    return self._cache[key], None
                return self._cache[key]
        
        # Create a new configuration with parameters patched
        cfg = self.patch_config(x)
        
        # Create model
        model = self.model_factory(cfg)
        
        # Create recorder for metrics
        recorder = RunRecorder()
        
        # Solve model
        self.solver(model, recorder=recorder)
        
        # Run simulator if provided
        if self.simulator is not None:
            self.simulator(model, recorder=recorder)
        
        # Collect metrics from functions
        metrics = {}
        for name, fn in self.metric_fns.items():
            metrics[name] = fn(model)
        
        # Add metrics from recorder
        metrics.update(recorder.metrics)
        
        # Cache result
        if self.cache:
            self._cache[key] = metrics
        
        if return_model:
            return metrics, model
        return metrics


def mpi_map(
    runner: CircuitRunner,
    xs: np.ndarray,
    return_models: bool = False,
    mpi: bool = True,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, List[Any]]]:
    """
    Apply CircuitRunner.run to multiple parameter vectors, potentially in parallel.

    This function maps the runner's run method across multiple parameter vectors.
    If MPI is available and enabled, it distributes the work across MPI ranks.

    Args:
        runner: CircuitRunner instance
        xs: Array of parameter vectors (shape: n_samples x n_params)
        return_models: Whether to return model instances along with metrics
        mpi: Whether to use MPI for parallelization (if available)

    Returns:
        Union[pd.DataFrame, Tuple[pd.DataFrame, List[Any]]]: DataFrame with parameter values and metrics, 
        optionally with a list of model instances

    Raises:
        ValueError: If xs doesn't have the expected shape
    """
    if xs.ndim != 2:
        raise ValueError(f"Expected 2D array, got {xs.ndim}D")
    
    use_mpi = mpi and HAS_MPI
    
    if use_mpi:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        if rank == 0:
            # Master process distributes work
            chunks = np.array_split(xs, size)
        else:
            chunks = None
        
        # Scatter chunks to all processes
        chunk = comm.scatter(chunks, root=0)
    else:
        # No MPI, just use the full array
        chunk = xs
    
    # Process local chunk
    results = []
    models_list = [] if return_models else None
    
    for x in chunk:
        if return_models:
            metrics, model = runner.run(x, return_model=True)
            results.append(metrics)
            models_list.append(model)
        else:
            metrics = runner.run(x)
            results.append(metrics)
    
    # Convert complex values to JSON strings for DataFrame compatibility
    for result in results:
        for k, v in result.items():
            if not np.isscalar(v) and not isinstance(v, str):
                result[k] = json.dumps(v)
    
    # Create DataFrame with parameters and metrics
    param_values = [list(x) for x in chunk]
    
    # Create a DataFrame from the parameters
    if len(param_values) > 0:
        param_df = pd.DataFrame(param_values, columns=runner.param_paths)
        
        # Create a DataFrame from the metrics
        metrics_df = pd.DataFrame(results)
        
        # Combine parameters and metrics
        local_df = pd.concat([param_df, metrics_df], axis=1)
    else:
        # Empty DataFrame with correct columns
        columns = runner.param_paths + list(runner.metric_fns.keys())
        local_df = pd.DataFrame(columns=columns)
    
    if use_mpi:
        # Gather results from all processes
        all_dfs = comm.gather(local_df, root=0)
        
        if return_models:
            all_models = comm.gather(models_list, root=0)
        
        if rank == 0:
            # Combine all DataFrames
            df = pd.concat(all_dfs, ignore_index=True)
            
            if return_models:
                # Flatten list of lists
                models = [m for sublist in all_models for m in sublist]
                return df, models
            return df
        return None  # Non-root processes return None
    
    if return_models:
        return local_df, models_list
    return local_df


def plot_metrics(
    df: pd.DataFrame, 
    metric_col: str, 
    param_col: str, 
    second_param_col: Optional[str] = None,
) -> None:
    """Plot metrics from a parameter sweep."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm
    except ImportError:
        warnings.warn("matplotlib is required for plotting")
        return
    
    if second_param_col is None:
        # 1D plot
        plt.figure(figsize=(10, 6))
        plt.plot(df[param_col], df[metric_col], 'o-')
        plt.xlabel(param_col)
        plt.ylabel(metric_col)
        plt.grid(True)
        plt.title(f"{metric_col} vs {param_col}")
    else:
        # 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        x = df[param_col].values
        y = df[second_param_col].values
        z = df[metric_col].values
        
        surf = ax.plot_trisurf(
            x, y, z, cmap=cm.viridis, linewidth=0.2, alpha=0.8
        )
        
        ax.set_xlabel(param_col)
        ax.set_ylabel(second_param_col)
        ax.set_zlabel(metric_col)
        
        plt.colorbar(surf)
        plt.title(f"{metric_col} vs {param_col} and {second_param_col}")
    
    plt.tight_layout()
    plt.show()


def plot_errors(df: pd.DataFrame, error_col: str, param_col: str) -> None:
    """Plot errors from a parameter sweep."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib is required for plotting")
        return
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(df[param_col], df[error_col], 'o-')
    plt.xlabel(param_col)
    plt.ylabel(error_col)
    plt.grid(True)
    plt.title(f"{error_col} vs {param_col}")
    plt.tight_layout()
    plt.show() 