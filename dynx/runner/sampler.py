"""
Sampler module providing sampling utilities for parameter sweeps in economic models.

This module provides various samplers for generating parameter combinations
for use with CircuitRunner.
"""

import copy
import itertools
import logging
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from pyDOE import lhs

# Type aliases
DesignMatrix = NDArray[np.object_]
SamplerCallable = Callable[[int, List[str], Dict[str, Dict[str, Any]], Optional[int]], DesignMatrix]
ParameterRanges = List[Tuple[float, float]]

logger = logging.getLogger(__name__)


def _is_categorical(meta_entry: Dict[str, Any]) -> bool:
    """Check if a parameter is categorical.
    
    Parameters are categorical if they have 'enum' or 'values' keys in their metadata.
    
    Args:
        meta_entry: Metadata entry for a parameter.
        
    Returns:
        True if parameter is categorical, False otherwise.
    """
    return "enum" in meta_entry or "values" in meta_entry


def _has_nan(array: NDArray[np.object_]) -> bool:
    """Safely check if an object array contains any NaN values.
    
    This function handles object arrays by checking each element individually
    and only using np.isnan for numeric types.
    
    Args:
        array: Array to check for NaN values.
        
    Returns:
        True if the array contains any NaN values, False otherwise.
    """
    # First make a fast-path check for arrays that definitely don't have NaNs
    if array.dtype.kind not in ['O', 'f']:
        return False
    
    # For object arrays, check each element individually
    for item in array.flat:
        # Check if item is a float or np.float type
        if isinstance(item, (float, np.floating)):
            if np.isnan(item):
                return True
        # np.nan is also a valid value for object arrays
        elif item is np.nan:
            return True
    
    return False


class BaseSampler:
    """Base class for samplers."""

    def __call__(
        self,
        n: Optional[int],
        param_paths: List[str],
        meta: Dict[str, Dict[str, Any]],
        seed: Optional[int] = None,
    ) -> NDArray[np.object_]:
        """Generate parameter draws.
        
        This is the preferred API for samplers. Legacy API is provided for backward
        compatibility with existing code.
        
        Args:
            n: Number of samples to generate. For some samplers like FixedSampler 
               or FullGridSampler, this argument is ignored.
            param_paths: List of parameter paths to sample.
            meta: Metadata for parameters, including min/max bounds for numeric parameters
                 and enum/values for categorical parameters.
            seed: Random seed for reproducibility.
            
        Returns:
            2D array of parameter draws, shape (n_draws, len(param_paths))
        """
        raise NotImplementedError


class MVNormSampler(BaseSampler):
    """Generate samples from a multivariate normal distribution.

    Clips samples to respect bounds if provided in the metadata.

    Attributes:
        mean: Mean of the distribution.
        cov: Covariance matrix.
        clip_bounds: Whether to clip samples to respect bounds.
        sample_size: Default number of samples to generate.
    """

    def __init__(
        self,
        mean: NDArray[np.float64],
        cov: NDArray[np.float64],
        clip_bounds: bool = True,
        sample_size: Optional[int] = None,  # For backward compatibility
    ):
        """Initialize.

        Args:
            mean: Mean of the distribution.
            cov: Covariance matrix.
            clip_bounds: Whether to clip samples to respect bounds.
            sample_size: Default number of samples to generate.
        """
        self.mean = np.asarray(mean, dtype=np.float64)
        self.cov = np.asarray(cov, dtype=np.float64)
        self.clip_bounds = clip_bounds
        self.sample_size = sample_size

    def __call__(
        self,
        n: Optional[int],
        param_paths: List[str],
        meta: Dict[str, Dict[str, Any]],
        seed: Optional[int] = None,
    ) -> NDArray[np.object_]:
        """Generate parameter draws from a multivariate normal distribution.

        Args:
            n: Number of samples to generate (overrides sample_size if provided).
            param_paths: List of parameter paths.
            meta: Metadata for parameters.
            seed: Random seed for reproducibility.

        Returns:
            2D array of parameter draws, shape (n, len(param_paths))

        Raises:
            ValueError: If parameters are non-numeric or out of bounds.
        """
        # Check if we're dealing with categorical parameters
        for path in param_paths:
            if path in meta and _is_categorical(meta[path]):
                raise ValueError(
                    f"MVNormSampler can only handle numeric parameters, but {path} "
                    f"is categorical: {meta[path]}"
                )

        # If no meta provided or no bounds in meta, just generate samples
        if meta is None:
            meta = {}

        # Check dimensions
        dim = self.mean.shape[0]
        if len(param_paths) != dim:
            raise ValueError(
                f"Dimension mismatch: len(param_paths)={len(param_paths)}, "
                f"but mean.shape={self.mean.shape}"
            )

        # Determine number of samples
        samples = n if n is not None else self.sample_size
        if samples is None:
            raise ValueError("Number of samples not specified")

        # Set seed for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Generate samples
        X = np.random.multivariate_normal(self.mean, self.cov, size=samples)

        # Clip samples to respect bounds if provided
        if self.clip_bounds:
            for i, path in enumerate(param_paths):
                if path in meta:
                    if "min" in meta[path]:
                        X[:, i] = np.maximum(X[:, i], meta[path]["min"])
                    if "max" in meta[path]:
                        X[:, i] = np.minimum(X[:, i], meta[path]["max"])

        # Check if any sample is out of bounds
        for i, path in enumerate(param_paths):
            if path in meta:
                if "min" in meta[path]:
                    if np.any(X[:, i] < meta[path]["min"]):
                        raise ValueError(
                            f"Sample out of bounds for {path}: min = {meta[path]['min']}, "
                            f"got {np.min(X[:, i])}"
                        )
                if "max" in meta[path]:
                    if np.any(X[:, i] > meta[path]["max"]):
                        raise ValueError(
                            f"Sample out of bounds for {path}: max = {meta[path]['max']}, "
                            f"got {np.max(X[:, i])}"
                        )

        # Convert to object array to match BaseSampler interface
        X_obj = X.astype(object)
        return X_obj


class FixedSampler(BaseSampler):
    """Sampler that returns fixed, predetermined parameter draws."""

    def __init__(self, rows: NDArray[np.object_]):
        """Initialize the fixed sampler.
        
        Args:
            rows: Matrix of parameter draws, shape (n_draws, n_params).
                 May contain NaN values which will be treated as wildcards
                 to be filled in by other samplers or by grid expansion.
        """
        self.rows = rows

    def __call__(
        self,
        n: Optional[int],
        param_paths: List[str],
        meta: Dict[str, Dict[str, Any]],
        seed: Optional[int] = None,
    ) -> NDArray[np.object_]:
        """Return the fixed parameter draws.
        
        Args:
            n: Ignored for FixedSampler.
            param_paths: List of parameter paths.
            meta: Metadata for parameters (ignored).
            seed: Ignored for FixedSampler.
            
        Returns:
            The fixed parameter draws provided at initialization.
        """
        # Check dimensions
        if self.rows.shape[1] != len(param_paths):
            raise ValueError(
                f"Number of columns in rows ({self.rows.shape[1]}) does not match "
                f"number of parameters ({len(param_paths)})"
            )
        return self.rows


class FullGridSampler(BaseSampler):
    """Sampler that generates a full grid of parameter combinations."""

    def __init__(self, grid_values: Dict[str, List[Any]]):
        """Initialize the full grid sampler.
        
        Args:
            grid_values: Dictionary mapping parameter paths to lists of possible values.
        """
        self.grid_values = grid_values

    def __call__(
        self,
        n: Optional[int],
        param_paths: List[str],
        meta: Dict[str, Dict[str, Any]],
        seed: Optional[int] = None,
    ) -> NDArray[np.object_]:
        """Generate a full grid of parameter combinations.
        
        Args:
            n: Ignored for FullGridSampler.
            param_paths: List of parameter paths.
            meta: Metadata for parameters (used to extract enum/values for parameters
                 not specified in grid_values).
            seed: Ignored for FullGridSampler.
            
        Returns:
            A grid of all possible parameter combinations.
        """
        # Calculate grid size
        total_size = 1
        param_sets = []
        valid_params = []
        
        for path in param_paths:
            if path in self.grid_values:
                # Use values specified in grid_values
                values = self.grid_values[path]
                valid_params.append(path)
            elif path in meta and _is_categorical(meta[path]):
                # Use categorical values from meta
                if "enum" in meta[path]:
                    values = meta[path]["enum"]
                else:
                    values = meta[path]["values"]
                valid_params.append(path)
            else:
                # Skip parameters not in grid_values and not categorical
                continue
            
            param_sets.append(values)
            total_size *= len(values)
        
        # Generate grid using itertools.product
        grid = list(itertools.product(*param_sets))
        
        # Convert to numpy array with all param_paths (fill missing with NaN)
        result = np.full((len(grid), len(param_paths)), np.nan, dtype=object)
        
        # Fill in the valid parameters
        for i, combo in enumerate(grid):
            for j, path in enumerate(valid_params):
                idx = param_paths.index(path)
                result[i, idx] = combo[j]
        
        return result


class LatinHypercubeSampler(BaseSampler):
    """Generate samples using Latin Hypercube Sampling.

    This ensures good coverage of the parameter space by dividing each dimension
    into equal intervals and placing exactly one sample in each interval.

    Attributes:
        ranges: List of (min, max) tuples for each parameter.
        sample_size: Number of samples to generate.
    """

    def __init__(
        self,
        ranges: ParameterRanges,
        sample_size: int = 1,
    ):
        """Initialize.

        Args:
            ranges: List of (min, max) tuples for each parameter.
            sample_size: Number of samples to generate.
        """
        self.ranges = ranges
        self.sample_size = sample_size

    def __call__(
        self,
        n: Optional[int],
        param_paths: List[str],
        meta: Dict[str, Dict[str, Any]],
        seed: Optional[int] = None,
    ) -> NDArray[np.object_]:
        """Generate parameter draws using Latin Hypercube Sampling.

        Args:
            n: Number of samples to generate (overrides sample_size if provided).
            param_paths: List of parameter paths.
            meta: Metadata for parameters (used to check if parameters are numeric).
            seed: Random seed for reproducibility.

        Returns:
            2D array of parameter draws, shape (n, len(param_paths))
        """
        # Determine number of samples
        samples = n if n is not None else self.sample_size

        # Check that we're only sampling numeric parameters
        for path in param_paths:
            if path in meta and _is_categorical(meta[path]):
                raise ValueError(
                    f"LatinHypercubeSampler can only handle numeric parameters, but {path} "
                    f"is categorical: {meta[path]}"
                )

        # Check dimensions
        dim = len(self.ranges)
        if len(param_paths) != dim:
            raise ValueError(
                f"Dimension mismatch: len(param_paths)={len(param_paths)}, "
                f"but len(ranges)={dim}"
            )

        # Set seed for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Generate Latin Hypercube samples
        try:
            from scipy.stats.qmc import LatinHypercube
            lhs = LatinHypercube(d=dim)
            X = lhs.random(n=samples)
        except ImportError:
            try:
                # Fallback to pyDOE
                import pyDOE
                X = pyDOE.lhs(dim, samples=samples)
            except ImportError:
                # Final fallback to simple stratified sampling
                X = np.zeros((samples, dim))
                for j in range(dim):
                    perms = np.random.permutation(samples)
                    for i in range(samples):
                        X[i, j] = (perms[i] + np.random.random()) / samples

        # Scale samples to parameter ranges
        for j, (min_val, max_val) in enumerate(self.ranges):
            X[:, j] = X[:, j] * (max_val - min_val) + min_val

        # Convert to object array to match BaseSampler interface
        X_obj = X.astype(object)
        return X_obj


def build_design(
    param_paths: List[str],
    samplers: List[BaseSampler],
    Ns: List[Optional[int]],
    meta: Dict[str, Dict[str, Any]],
    seed: Optional[int] = None,
) -> Tuple[DesignMatrix, Dict[str, List[str]]]:
    """Build a design matrix from samplers.

    Args:
        param_paths: List of parameter paths to include in the design.
        samplers: List of samplers to use.
        Ns: List of sample sizes for each sampler.
        meta: Dictionary mapping parameter paths to metadata containing bounds.
        seed: Random seed for reproducibility.

    Returns:
        X: A (N, D) design matrix of parameter draws.
        info: Dictionary with sampler information for each row in the design.
    """
    if len(samplers) != len(Ns):
        raise ValueError(f"Number of samplers ({len(samplers)}) must match "
                         f"number of sample sizes ({len(Ns)})")

    # Identify categorical parameters
    categorical_params = [p for p in param_paths if p in meta and _is_categorical(meta[p])]
    numeric_params = [p for p in param_paths if p not in categorical_params]

    # Process parameters with samplers
    sampler_param_indices = []  # Indices of parameters handled by samplers
    sampler_outputs = []  # Sampler outputs
    sampler_tags = []  # Sampler tags for info dictionary

    # For each sampler, get parameters for that sampler
    for i, (sampler, N) in enumerate(zip(samplers, Ns)):
        # For samplers that don't support categorical parameters, filter out categorical parameters
        sampler_param_paths = []
        if isinstance(sampler, (MVNormSampler, LatinHypercubeSampler)):
            # These samplers only work with numeric parameters
            sampler_param_paths = numeric_params
        else:
            # Other samplers can handle all parameters
            sampler_param_paths = param_paths

        # If no parameters for this sampler, skip
        if not sampler_param_paths:
            continue

        # Get indices of parameters in param_paths for this sampler
        indices = [param_paths.index(p) for p in sampler_param_paths]
        sampler_param_indices.append(indices)

        # Generate parameter draws using the sampler
        output = sampler(N, sampler_param_paths, meta, seed=seed)
        sampler_outputs.append(output)

        # Create sampler tags
        sampler_name = sampler.__class__.__name__
        sampler_tags.append([f"{sampler_name}"] * len(output))

    # Create cartesian product over categorical parameters if any
    if categorical_params:
        # Get all possible values for each categorical parameter
        cat_values = []
        for param in categorical_params:
            if param in meta:
                if "enum" in meta[param]:
                    cat_values.append(meta[param]["enum"])
                elif "values" in meta[param]:
                    cat_values.append(meta[param]["values"])
                else:
                    raise ValueError(f"Categorical parameter {param} has no 'enum' or 'values' key")
            else:
                raise ValueError(f"Parameter {param} not found in meta")

        # Create cartesian product
        cat_grid = list(itertools.product(*cat_values))
        cat_indices = [param_paths.index(p) for p in categorical_params]
    else:
        cat_grid = [()]  # Empty tuple as placeholder when no categoricals
        cat_indices = []

    # Combine sampler outputs with categorical grid
    rows = []
    row_info = {"sampler": []}

    # Create cartesian product between sampler outputs and categorical grid
    for sampler_idx, (output, tags) in enumerate(zip(sampler_outputs, sampler_tags)):
        indices = sampler_param_indices[sampler_idx]

        for i, row in enumerate(output):
            for cat_row in cat_grid:
                # Create a full row with NaNs
                full_row = np.full(len(param_paths), np.nan, dtype=object)

                # Fill in values from sampler output for appropriate indices
                for j, idx in enumerate(indices):
                    full_row[idx] = row[j]

                # Fill in categorical values
                for j, idx in enumerate(cat_indices):
                    if j < len(cat_row):  # Safety check
                        full_row[idx] = cat_row[j]

                rows.append(full_row)
                # Add grid suffix to sampler tag if categorical parameters exist
                tag_suffix = "×grid" if categorical_params else ""
                row_info["sampler"].append(f"{tags[i]}{tag_suffix}")

    # Check if any rows were created
    if not rows:
        # If no rows were created from samplers, create grid-only design
        for cat_row in cat_grid:
            full_row = np.full(len(param_paths), np.nan, dtype=object)
            for j, idx in enumerate(cat_indices):
                if j < len(cat_row):
                    full_row[idx] = cat_row[j]
            rows.append(full_row)
            row_info["sampler"].append("GridOnly")

    # Convert rows to design matrix
    X = np.array(rows, dtype=object)

    # Check for NaN values in the design
    if _has_nan(X):
        raise ValueError("Design matrix contains NaN values. "
                         "Check if param_specs or samplers cover all parameters.")

    return X, row_info


def build_design_legacy(
    param_paths: List[str],
    param_specs: Dict[str, Any],
    N: int,
    meta: Optional[Dict[str, Dict[str, Any]]] = None,
    seed: Optional[int] = None,
) -> Tuple[DesignMatrix, Dict[str, List[str]]]:
    """Legacy version of build_design for backward compatibility.
    
    Args:
        param_paths: List of parameter paths to include in the design.
        param_specs: Dictionary mapping parameter paths to samplers or values.
        N: Number of samples for samplers.
        meta: Metadata for parameters.
        seed: Random seed for reproducibility.
        
    Returns:
        Tuple of (design_matrix, info_dict).
    """
    # Convert param_specs to samplers and categorical values in meta
    samplers = []
    Ns = []
    
    # Ensure meta exists
    if meta is None:
        meta = {}
    meta = copy.deepcopy(meta)  # Make a copy to avoid modifying the original
    
    # Record which parameters will be handled by samplers
    sampler_params = set()
    
    # For each parameter, check if it has a sampler or should be treated as categorical
    for path in param_paths:
        if path in param_specs:
            spec = param_specs[path]
            
            if isinstance(spec, BaseSampler):
                # Parameter has a sampler
                samplers.append(spec)
                Ns.append(N)
                sampler_params.add(path)
            elif isinstance(spec, list):
                # Parameter has a list of values, make it categorical
                if path not in meta:
                    meta[path] = {}
                
                if all(isinstance(x, str) for x in spec):
                    meta[path]["enum"] = spec
                else:
                    meta[path]["values"] = spec
    
    # If no samplers were provided, create a default one
    if not samplers:
        # Create a default MVNormSampler for any numeric parameters without samplers
        numeric_params = []
        for path in param_paths:
            if path not in meta or not _is_categorical(meta[path]):
                numeric_params.append(path)
        
        if numeric_params:
            mean = np.zeros(len(numeric_params))
            cov = np.eye(len(numeric_params))
            default_sampler = MVNormSampler(mean, cov)
            samplers.append(default_sampler)
            Ns.append(N)
    
    # Call new build_design with converted parameters
    return build_design(param_paths, samplers, Ns, meta, seed=seed) 