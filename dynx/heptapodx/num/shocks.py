"""
Shock grid generation module for Heptapod-B.

This module provides functions to generate numerical shock grids:
- Normal distribution
- Lognormal distribution
- Adaptive grid
- Markov discretization (Tauchen and Rouwenhorst)
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import warnings

# Import resolve_reference from state_space to resolve parameter references
from .state_space import resolve_reference

# Import manual shock process functionality
from .manual_shocks import create_manual_shock_process


def build_normal_shock_grid(mean=0.0, std=1.0, size=7, method='gauss-hermite', bounds=None, **kwargs):
    """
    Build a grid for a normal shock distribution.
    
    Args:
        mean: Mean of the normal distribution
        std: Standard deviation of the normal distribution
        size: Number of points in the grid
        method: Method to use for discretization ('gauss-hermite', 'equiprobable', or 'tauchen')
        bounds: Tuple of (min, max) values for the grid (only used for 'equiprobable' and 'tauchen')
        **kwargs: Additional parameters (ignored)
        
    Returns:
        Tuple of (points, weights) where points is an array of grid points and weights is 
        an array of probabilities
    """
    if method == 'gauss-hermite':
        # Generate Gauss-Hermite quadrature points and weights
        points, weights = np.polynomial.hermite.hermgauss(size)
        
        # Scale points by sqrt(2) to match normal distribution
        points = points * np.sqrt(2)
        
        # Apply mean and standard deviation
        points = mean + std * points
        
        # Normalize weights to sum to 1
        weights = weights / np.sum(weights)
        
        return points, weights
        
    elif method == 'equiprobable':
        # Check if bounds are provided
        if bounds is None:
            # Default to +/- 3 standard deviations
            bounds = (mean - 3*std, mean + 3*std)
            
        # Generate equiprobable grid
        points = np.zeros(size)
        weights = np.ones(size) / size
        
        # Calculate CDF values for equidistant probabilities
        cdf_values = np.linspace(0, 1, size+2)[1:-1]
        
        # Convert CDF values to quantiles
        for i in range(size):
            points[i] = stats.norm.ppf(cdf_values[i], loc=mean, scale=std)
            
        return points, weights
        
    elif method == 'tauchen':
        # Check if bounds are provided
        if bounds is None:
            # Default to +/- 3 standard deviations
            bounds = (mean - 3*std, mean + 3*std)
            
        # Generate Tauchen method grid
        min_val, max_val = bounds
        points = np.linspace(min_val, max_val, size)
        
        # Calculate transition probabilities
        step = (max_val - min_val) / (size - 1)
        weights = np.zeros(size)
        
        for i in range(size):
            if i == 0:
                weights[i] = stats.norm.cdf(points[i] + step/2, loc=mean, scale=std)
            elif i == size - 1:
                weights[i] = 1 - stats.norm.cdf(points[i] - step/2, loc=mean, scale=std)
            else:
                weights[i] = stats.norm.cdf(points[i] + step/2, loc=mean, scale=std) - stats.norm.cdf(points[i] - step/2, loc=mean, scale=std)
                
        return points, weights
        
    else:
        raise ValueError(f"Unknown method '{method}' for normal shock grid. "
                         f"Must be one of: 'gauss-hermite', 'equiprobable', 'tauchen'.")


def build_lognormal_shock_grid(
    mu: float = 0.0,
    sigma: float = 0.1,
    n_points: int = 7,
    width: float = 3,
    prob_zero_income: float = 0.0,
    zero_income_value: float = 1e-9
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a lognormal shock grid with the specified parameters.
    
    Parameters
    ----------
    mu : float
        Mean of the log of the distribution
    sigma : float
        Standard deviation of the log of the distribution
    n_points : int
        Number of grid points
    width : float
        Width of the grid in standard deviations
    prob_zero_income : float
        Probability of zero income
    zero_income_value : float
        Value to use for zero income
        
    Returns
    -------
    tuple
        (shock_values, shock_probs) arrays
    """
    # Handle zero income case similar to normal grid
    if prob_zero_income > 0:
        # Reduce n_points by 1 to make room for zero income
        if n_points > 1:
            n_grid = n_points - 1
        else:
            n_grid = 1
            warnings.warn(
                "Cannot accommodate zero income with n_points=1. Using n_grid=1 anyway.",
                UserWarning
            )
        
        # Compute the lognormal quantiles for equiprobable grid excluding zero income
        remaining_prob = 1.0 - prob_zero_income
        
        # Special case for n_grid = 1
        if n_grid == 1:
            # Just use the exp(mu) for the single grid point
            spacing = np.array([np.exp(mu)])
            probs = np.array([remaining_prob])
        else:
            # Compute boundary points of equiprobable bins for the standard normal
            inner_probs = np.linspace(0, remaining_prob, n_grid + 1)
            # Use the lognormal PPF directly
            bin_edges = stats.lognorm.ppf(inner_probs, s=sigma, scale=np.exp(mu))
            
            # Use midpoints of bins as grid points
            spacing = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            
            # All grid points get equal probability
            probs = np.full(n_grid, remaining_prob / n_grid)
        
        # Add zero income point
        spacing = np.append(spacing, zero_income_value)
        probs = np.append(probs, prob_zero_income)
        
        # Sort by value in ascending order
        idx = np.argsort(spacing)
        spacing = spacing[idx]
        probs = probs[idx]
    else:
        # Standard case: equiprobable grid with no zero income
        if width <= 0:
            raise ValueError("width must be positive")
        
        # For lognormal, work with the log-transformed grid
        # Which is normally distributed with mean mu and std dev sigma
        norm_grid, norm_probs = build_normal_shock_grid(mu, sigma, n_points, width)
        
        # Convert back to lognormal
        spacing = np.exp(norm_grid)
        probs = norm_probs  # Probabilities remain the same
    
    return spacing, probs


def build_adaptive_shock_grid(
    mean: float = 0.0,
    std: float = 0.1,
    n_points: int = 7,
    width: float = 3,
    prob_zero_income: float = 0.0,
    zero_income_value: float = 1e-9
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build an adaptive shock grid with points concentrated where probability mass is higher.
    
    Parameters
    ----------
    mean : float
        Mean of the normal distribution
    std : float
        Standard deviation of the normal distribution
    n_points : int
        Number of grid points
    width : float
        Width of the grid in standard deviations
    prob_zero_income : float
        Probability of zero income
    zero_income_value : float
        Value to use for zero income
        
    Returns
    -------
    tuple
        (shock_values, shock_probs) arrays
    """
    # For adaptive grid, create equiprobable grid using normal CDF
    if prob_zero_income > 0:
        # Handle zero income case
        if n_points <= 1:
            warnings.warn(
                "n_points too small for adaptive grid with zero income. Using basic normal grid.",
                UserWarning
            )
            return build_normal_shock_grid(
                mean, std, n_points, width, prob_zero_income, zero_income_value
            )
        
        # Compute remaining grid points after accounting for zero income
        n_grid = n_points - 1
        remaining_prob = 1.0 - prob_zero_income
        
        # Create quantiles at equiprobable points
        quantiles = np.linspace(0, remaining_prob, n_grid + 1)
        grid_values = stats.norm.ppf(quantiles, loc=mean, scale=std)
        
        # Use quantile values directly instead of bin midpoints for better adaptation
        values = grid_values[1:]  # Skip the leftmost edge
        probs = np.full(n_grid, remaining_prob / n_grid)
        
        # Add zero income point
        values = np.append(values, zero_income_value)
        probs = np.append(probs, prob_zero_income)
        
        # Sort by value in ascending order
        idx = np.argsort(values)
        values = values[idx]
        probs = probs[idx]
    else:
        # Standard case: equiprobable quantile-based grid
        quantiles = np.linspace(0, 1, n_points + 1)
        grid_edges = stats.norm.ppf(quantiles, loc=mean, scale=std)
        
        # Use quantile values directly instead of bin midpoints
        values = grid_edges[1:-1]  # Skip the leftmost and rightmost edges
        
        # Add the mean to ensure it's represented
        values = np.append(values, mean)
        
        # Add the +/- width*std boundary points if not too close to existing points
        left_edge = mean - width * std
        right_edge = mean + width * std
        
        # Only add if not too close to existing points
        min_dist = 0.1 * std
        if np.min(np.abs(values - left_edge)) > min_dist:
            values = np.append(values, left_edge)
        if np.min(np.abs(values - right_edge)) > min_dist:
            values = np.append(values, right_edge)
        
        # Sort and remove duplicates
        values = np.unique(values)
        
        # Compute probabilities using normal PDF, normalized to sum to 1
        probs = stats.norm.pdf(values, loc=mean, scale=std)
        probs = probs / np.sum(probs)
    
    return values, probs


def build_discrete_markov_shock_grid(
    rho: float = 0.9,
    sigma: float = 0.1,
    n_points: int = 7,
    width: float = 3,
    mean: float = 0.0,
    method: str = "tauchen",
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a discrete Markov process approximation for AR(1) shock.
    
    Parameters
    ----------
    rho : float
        Persistence parameter of AR(1) process
    sigma : float
        Standard deviation of shock innovations
    n_points : int
        Number of grid points
    width : float
        Width of the grid in standard deviations
    mean : float
        Mean of the AR(1) process
    method : str
        Method to use: 'tauchen' or 'rouwenhorst'
    **kwargs
        Additional parameters for specific methods
        
    Returns
    -------
    tuple
        (shock_values, shock_probs, transition_matrix) arrays
    """
    if method.lower() == "tauchen":
        return tauchen_method(rho, sigma, n_points, width, mean, **kwargs)
    elif method.lower() == "rouwenhorst":
        return rouwenhorst_method(rho, sigma, n_points, mean, **kwargs)
    else:
        warnings.warn(
            f"Unknown method: {method}. Using Tauchen method as fallback.",
            UserWarning
        )
        return tauchen_method(rho, sigma, n_points, width, mean, **kwargs)


def tauchen_method(
    rho: float,
    sigma: float,
    n_points: int,
    width: float = 3,
    mean: float = 0.0,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Implement Tauchen method for discretizing AR(1) process.
    
    Parameters
    ----------
    rho : float
        Persistence parameter of AR(1) process
    sigma : float
        Standard deviation of shock innovations
    n_points : int
        Number of grid points
    width : float
        Width of the grid in standard deviations
    mean : float
        Mean of the AR(1) process
    **kwargs
        Additional parameters
        
    Returns
    -------
    tuple
        (shock_values, shock_probs, transition_matrix) arrays
    """
    # Verify parameters
    if not 0 <= abs(rho) < 1:
        warnings.warn(
            f"rho={rho} should be in [0,1). Results may be unreliable.",
            UserWarning
        )
    
    # Step 1: Create the state grid for z
    sigma_z = sigma / np.sqrt(1 - rho**2)  # Unconditional std dev
    
    # Linear state grid centered at mean
    z_min = mean - width * sigma_z / 2
    z_max = mean + width * sigma_z / 2
    z_grid = np.linspace(z_min, z_max, n_points)
    
    # Step 2: Compute the transition matrix
    dz = (z_max - z_min) / (n_points - 1)
    transition = np.zeros((n_points, n_points))
    
    # Compute the conditional transition probabilities
    for i in range(n_points):
        for j in range(n_points):
            # For the first state (j=0)
            if j == 0:
                transition[i, j] = stats.norm.cdf(
                    (z_grid[j] - mean - rho * (z_grid[i] - mean) + dz/2) / sigma
                )
            # For the last state (j=n-1)
            elif j == n_points - 1:
                transition[i, j] = 1.0 - stats.norm.cdf(
                    (z_grid[j] - mean - rho * (z_grid[i] - mean) - dz/2) / sigma
                )
            # For all interior states
            else:
                transition[i, j] = stats.norm.cdf(
                    (z_grid[j] - mean - rho * (z_grid[i] - mean) + dz/2) / sigma
                ) - stats.norm.cdf(
                    (z_grid[j] - mean - rho * (z_grid[i] - mean) - dz/2) / sigma
                )
    
    # Step 3: Compute the stationary distribution (ergodic probs)
    # We'll compute this by finding the eigenvector corresponding to eigenvalue 1
    transition_transpose = transition.T
    eigenvalues, eigenvectors = np.linalg.eig(transition_transpose)
    
    # Find which eigenvalue is closest to 1
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    
    # Extract the corresponding eigenvector and normalize
    stationary_dist = np.real(eigenvectors[:, idx])
    stationary_dist = stationary_dist / np.sum(stationary_dist)
    
    return z_grid, stationary_dist, transition


def rouwenhorst_method(
    rho: float,
    sigma: float,
    n_points: int,
    mean: float = 0.0,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Implement Rouwenhorst method for discretizing AR(1) process.
    
    Parameters
    ----------
    rho : float
        Persistence parameter of AR(1) process
    sigma : float
        Standard deviation of shock innovations
    n_points : int
        Number of grid points
    mean : float
        Mean of the AR(1) process
    **kwargs
        Additional parameters
        
    Returns
    -------
    tuple
        (shock_values, shock_probs, transition_matrix) arrays
    """
    # The Rouwenhorst method parameters
    sigma_z = sigma / np.sqrt(1 - rho**2)  # Unconditional std dev
    p = (1 + rho) / 2  # Probability of staying in the same state
    
    # Create the grid using Rouwenhorst's equidistant grid centering on mean
    z_min = mean - np.sqrt(n_points - 1) * sigma_z
    z_max = mean + np.sqrt(n_points - 1) * sigma_z
    z_grid = np.linspace(z_min, z_max, n_points)
    
    # Special case: N = 2
    if n_points == 2:
        transition = np.array([[p, 1-p], [1-p, p]])
        stationary_dist = np.array([0.5, 0.5])  # Equal probability for 2-state process
        return z_grid, stationary_dist, transition
    
    # Initialize for N = 2
    p_mat = np.array([[p, 1-p], [1-p, p]])
    
    # Build up recursively for N > 2
    for n in range(3, n_points + 1):
        # Initialize a matrix of zeros of size n x n
        p_n = np.zeros((n, n))
        
        # Expand using the recursive procedure
        # P_(n) = [ p*P_(n-1)    (1-p)*P_(n-1)    0 ] +
        #         [ 0    p*P_(n-1)    (1-p)*P_(n-1) ]
        
        # First term
        p_n[0:n-1, 0:n-1] += p * p_mat
        p_n[0:n-1, 1:n] += (1-p) * p_mat
        
        # Second term
        p_n[1:n, 0:n-1] += (1-p) * p_mat
        p_n[1:n, 1:n] += p * p_mat
        
        # Normalize to ensure row sums are 1 (probability transition matrix)
        for i in range(n):
            p_n[i, :] = p_n[i, :] / np.sum(p_n[i, :])
        
        # Update for next iteration
        p_mat = p_n.copy()
    
    # Compute the stationary distribution (same method as Tauchen)
    transition_transpose = p_mat.T
    eigenvalues, eigenvectors = np.linalg.eig(transition_transpose)
    
    # Find which eigenvalue is closest to 1
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    
    # Extract the corresponding eigenvector and normalize
    stationary_dist = np.real(eigenvectors[:, idx])
    stationary_dist = stationary_dist / np.sum(stationary_dist)
    
    return z_grid, stationary_dist, p_mat


def build_shock_grid(shock_type, params=None, **kwargs):
    """
    Build a shock grid based on the specified shock type and parameters.
    
    Args:
        shock_type: Type of shock ('normal', 'lognormal', 'uniform', etc.)
        params: Dictionary of parameters for the shock
        **kwargs: Additional parameters that override those in params
        
    Returns:
        Tuple of (points, weights) where points is an array of grid points and 
        weights is an array of probabilities
    """
    # Initialize parameters dictionary if None
    if params is None:
        params = {}
    
    # Override with kwargs
    all_params = {**params, **kwargs}
    
    # Handle different shock types
    if shock_type.lower() == 'normal':
        # Extract parameters needed for normal distribution
        normal_params = {k: v for k, v in all_params.items() 
                        if k in ['mean', 'std', 'size', 'method', 'bounds']}
        return build_normal_shock_grid(**normal_params)
        
    elif shock_type.lower() == 'lognormal':
        # Convert normal mean and std to lognormal parameters
        mu = all_params.get('mean', 0.0)
        sigma = all_params.get('std', 1.0)
        size = all_params.get('size', 7)
        method = all_params.get('method', 'gauss-hermite')
        bounds = all_params.get('bounds', None)
        
        # Build normal grid for log values
        log_points, weights = build_normal_shock_grid(
            mean=mu, std=sigma, size=size, method=method, bounds=bounds
        )
        
        # Transform points with exp
        points = np.exp(log_points)
        
        return points, weights
        
    elif shock_type.lower() == 'uniform':
        # Extract uniform parameters
        a = all_params.get('min', 0.0)
        b = all_params.get('max', 1.0)
        size = all_params.get('size', 7)
        
        # Generate uniform grid
        points = np.linspace(a, b, size)
        weights = np.ones(size) / size
        
        return points, weights
        
    elif shock_type.lower() == 'discrete':
        # Extract discrete parameters
        values = all_params.get('values', [0, 1])
        probs = all_params.get('probs', None)
        
        # Ensure values is a numpy array
        values = np.array(values)
        
        # If probs is not specified, use uniform probabilities
        if probs is None:
            probs = np.ones(len(values)) / len(values)
        else:
            probs = np.array(probs)
            # Normalize probs to sum to 1
            probs = probs / np.sum(probs)
            
        return values, probs
        
    elif shock_type.lower() == 'discretemarkov':
        # Extract parameters for Markov process
        rho = all_params.get('rho', 0.9)
        sigma = all_params.get('std', 0.1)
        n_points = all_params.get('n_points', 7)
        width = all_params.get('width', 3)
        mean = all_params.get('mean', 0.0)
        method = all_params.get('method', 'tauchen')
        
        # Build discrete Markov shock grid
        values, probs, transition_matrix = build_discrete_markov_shock_grid(
            rho=rho, sigma=sigma, n_points=n_points, 
            width=width, mean=mean, method=method
        )
        
        # Store transition matrix in all_params for later access
        all_params['transition_matrix'] = transition_matrix
        
        return values, probs
        
    else:
        raise ValueError(f"Unknown shock type '{shock_type}'. "
                         f"Must be one of: 'normal', 'lognormal', 'uniform', 'discrete', 'discretemarkov'.")


def generate_numerical_shocks(problem, methods=None):
    """
    Generate numerical shock distributions from analytical shock definitions.
    
    Parameters
    ----------
    problem : FunctionalProblem
        The model with analytical shock definitions
    methods : dict, optional
        Methods dictionary with shock generation settings
        
    Returns
    -------
    dict
        Dictionary representing the numerical shock distributions
    """
    if not hasattr(problem, "math") or "shocks" not in problem.math:
        warnings.warn(
            "problem.math['shocks'] not found. Skipping shock generation.",
            UserWarning
        )
        return {}

    analytical_shocks = problem.math["shocks"]

    # Ensure the numerical shocks container exists
    if "shocks" not in problem.num:
        problem.num["shocks"] = {}

    # Get global parameters and methods
    all_params = {
        **getattr(problem, "parameters_dict", {}),
        **getattr(problem, "methods", {}),
        **getattr(problem, "settings_dict", {}),
    }

    # Iterate through each defined shock
    for shock_name, shock_info in analytical_shocks.items():
        try:
            # Create storage for this shock if it doesn't exist
            if shock_name not in problem.num["shocks"]:
                problem.num["shocks"][shock_name] = {}
            
            # Copy dimensions info
            dimensions = shock_info.get("dimensions", [])
            problem.num["shocks"][shock_name]["dimensions"] = dimensions
            
            # Get shock type and methods
            methods_dict = shock_info.get("methods", {})
            settings_dict = shock_info.get("settings", {})
            params_dict = shock_info.get("parameters", {})
            #print(shock_info)
            
            # Resolve method references if they are in reference format
            resolved_methods = {}
            for method_name, method_value in methods_dict.items():
                try:
                    resolved_methods[method_name] = resolve_reference(method_value, all_params)
                except (ValueError, TypeError):
                    # If we can't resolve, use a default
                    if method_name == 'shock_method':
                        resolved_methods[method_name] = 'normal'
                    elif method_name == 'integration_method':
                        resolved_methods[method_name] = 'discretize'
                    else:
                        resolved_methods[method_name] = method_value
            
            # Check for manual shock method
            generation_method = resolved_methods.get("method", None)
            if generation_method and generation_method.lower() in ['manual', 'explicit']:
                # Use manual shock process creation
                try:
                    shock_process = create_manual_shock_process(shock_info['parameters'], problem)
                    
                    # Store the process object and its data for access patterns
                    problem.num["shocks"][shock_name]["process"] = shock_process
                    problem.num["shocks"][shock_name]["values"] = shock_process.values
                    problem.num["shocks"][shock_name]["probs"] = shock_process.probs
                    
                    # Store transition matrix if it's a Markov process
                    if hasattr(shock_process, 'transition_matrix'):
                        problem.num["shocks"][shock_name]["transition_matrix"] = shock_process.transition_matrix
                        
                    # Skip the rest of the loop for manual shocks
                    continue
                except Exception as e:
                    warnings.warn(f"Error creating manual shock process for '{shock_name}': {e}", UserWarning)
                    # Fall through to algorithmic method for fallback
            
            # Standard algorithmic shock process generation (for non-manual methods)
            # Resolve parameter references
            resolved_params = {}
            for param_name, param_value in params_dict.items():
                try:
                    resolved_params[param_name] = resolve_reference(param_value, all_params)
                except (ValueError, TypeError) as e:
                    warnings.warn(f"Could not resolve parameter '{param_name}' for shock '{shock_name}': {e}")
                    # Use default values for common parameters
                    if param_name == 'mean':
                        resolved_params[param_name] = 0.0
                    elif param_name == 'std':
                        resolved_params[param_name] = 0.1
                    else:
                        resolved_params[param_name] = param_value
            
            # Resolve settings references
            resolved_settings = {}
            for setting_name, setting_value in settings_dict.items():
                try:
                    resolved_settings[setting_name] = resolve_reference(setting_value, all_params)
                except (ValueError, TypeError) as e:
                    warnings.warn(f"Could not resolve setting '{setting_name}' for shock '{shock_name}': {e}")
                    # Use default values for common settings
                    if setting_name == 'n_points':
                        resolved_settings[setting_name] = 7
                    elif setting_name == 'width':
                        resolved_settings[setting_name] = 3
                    else:
                        resolved_settings[setting_name] = setting_value
            
            # Get the shock type
            shock_type = resolved_methods.get("shock_method", "normal")
            
            # Combine params and settings
            build_params = {**resolved_params, **resolved_settings}
            
            # Build the shock grid
            values_and_probs = build_shock_grid(shock_type, build_params)
            
            # Store in num
            if values_and_probs:
                values, probs = values_and_probs
                problem.num["shocks"][shock_name]["values"] = values
                problem.num["shocks"][shock_name]["probs"] = probs
        
        except Exception as e:
            warnings.warn(
                f"Error generating shock grid for shock '{shock_name}': {str(e)}",
                UserWarning
            )
            # Create a fallback default shock grid
            try:
                # Simple normal shock with default parameters
                default_values, default_probs = build_normal_shock_grid(mean=1.0, std=0.1, size=7)
                problem.num["shocks"][shock_name]["values"] = default_values
                problem.num["shocks"][shock_name]["probs"] = default_probs
                warnings.warn(f"Using fallback normal shock grid for '{shock_name}'", UserWarning)
            except Exception:
                continue

    return problem.num["shocks"] 