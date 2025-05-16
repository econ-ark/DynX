"""
Manual shock process creation and validation for Heptapod-B.

This module provides functionality to create shock processes directly from
manually specified transition matrices and shock values.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, Union
import warnings

# Import resolve_reference from state_space
try:
    from .state_space import resolve_reference
except ImportError:
    # Default implementation if not found
    def resolve_reference(param, model):
        """Basic parameter resolution."""
        # If param is a single-element list, try to resolve it as a reference
        if isinstance(param, list) and len(param) == 1 and isinstance(param[0], str):
            key = param[0]
            # Try to resolve from various model attributes
            if hasattr(model, 'parameters_dict') and isinstance(model.parameters_dict, dict) and key in model.parameters_dict:
                return model.parameters_dict[key]
            elif hasattr(model, 'param') and hasattr(model.param, key):
                return getattr(model.param, key)
            elif hasattr(model, 'methods') and isinstance(model.methods, dict) and key in model.methods:
                return model.methods[key]
            elif hasattr(model, 'settings') and isinstance(model.settings, dict) and key in model.settings:
                return model.settings[key]
            elif hasattr(model, '__contains__') and key in model:
                return model[key]
            else:
                raise ValueError(f"Could not resolve reference: {key}")
        return param

# Define shock process classes if they're not available from core
class ShockProcess:
    """Base class for shock processes."""
    def __init__(self, values, probs=None, labels=None):
        self.values = values
        self.probs = probs if probs is not None else np.ones_like(values) / len(values)
        self.labels = labels

class DiscreteMarkovProcess(ShockProcess):
    """Discrete Markov shock process with specified transition matrix."""
    def __init__(self, transition_matrix, values, labels=None, stationary_distribution=None):
        """
        Initialize a discrete Markov process.
        
        Parameters
        ----------
        transition_matrix : ndarray
            Square matrix of transition probabilities.
        values : ndarray
            Values for each state.
        labels : list, optional
            Labels for each state.
        stationary_distribution : ndarray, optional
            Pre-computed stationary distribution (if available).
        """
        self.transition_matrix = transition_matrix
        self.values = values
        self.labels = labels
        
        # Compute stationary distribution if not provided
        if stationary_distribution is None:
            try:
                # Simple power method to compute stationary distribution
                # For large matrices or more precision, use numpy.linalg.eig
                n_states = len(values)
                p = np.ones(n_states) / n_states  # Initial uniform distribution
                for _ in range(100):  # Usually converges quickly
                    p_new = p @ transition_matrix
                    if np.allclose(p, p_new, rtol=1e-10, atol=1e-12):
                        break
                    p = p_new
                self.stationary_distribution = p
            except Exception as e:
                warnings.warn(f"Failed to compute stationary distribution: {e}")
                self.stationary_distribution = np.ones(len(values)) / len(values)
        else:
            self.stationary_distribution = stationary_distribution
        
        # Use stationary distribution as probabilities
        self.probs = self.stationary_distribution

class IIDProcess(ShockProcess):
    """IID shock process with specified values and probabilities."""
    def __init__(self, values, probabilities=None, labels=None):
        """
        Initialize an IID shock process.
        
        Parameters
        ----------
        values : ndarray
            Values for each state.
        probabilities : ndarray, optional
            Probabilities for each state. If not provided, uniform.
        labels : list, optional
            Labels for each state.
        """
        super().__init__(values, probabilities, labels)
        # For IID, no transition matrix needed


def validate_manual_shock_parameters(transition_matrix: np.ndarray, values: np.ndarray):
    """
    Validate manually specified shock parameters.
    
    Parameters
    ----------
    transition_matrix : numpy.ndarray
        Transition probability matrix
    values : numpy.ndarray
        Shock values
        
    Raises
    ------
    ValueError
        If parameters are invalid
    """
    # Check values array
    if len(values.shape) != 1:
        raise ValueError("Values must be 1-dimensional")
    
    # Check dimensions (skip if transition_matrix is None for IID case)
    if transition_matrix is None:
        return
        
    if len(transition_matrix.shape) != 2:
        raise ValueError("Transition matrix must be 2-dimensional")
        
    if transition_matrix.shape[0] != transition_matrix.shape[1]:
        raise ValueError("Transition matrix must be square")
        
    # Check matrix dimensions match values
    if transition_matrix.shape[0] != len(values):
        raise ValueError(f"Transition matrix dimensions ({transition_matrix.shape[0]}x{transition_matrix.shape[1]}) "
                        f"do not match number of values ({len(values)})")
    
    # Check row sums are approximately 1
    row_sums = np.sum(transition_matrix, axis=1)
    if not np.allclose(row_sums, 1.0, rtol=1e-5, atol=1e-8):
        warnings.warn(f"Transition matrix rows do not sum to 1: {row_sums}", UserWarning)
    
    # Check for non-negative probabilities
    if np.any(transition_matrix < 0):
        raise ValueError("Transition matrix contains negative probabilities")
        
    if np.any(transition_matrix > 1):
        raise ValueError("Transition matrix contains probabilities greater than 1")


def create_manual_shock_process(spec: Dict[str, Any], model: Optional[Any]) -> ShockProcess:
    """
    Create a shock process from manually specified values.
    
    Parameters
    ----------
    spec : dict
        Shock specification including transition_matrix and values
    model : FunctionalProblem
        Model object for parameter resolution
        
    Returns
    -------
    ShockProcess
        Instance of appropriate shock process class
    """
    # Extract required parameters
    methods_dict = spec.get('methods', {})
    #print(spec)
    shock_method = methods_dict.get('shock_method', 'DiscreteMarkov')
    
    # Get values and transition matrix
    if 'values' not in spec:
        raise ValueError("Manual shock method requires 'values' in specification")
    
    values = spec.get('values')
    # Resolve parameter references
    values = resolve_reference(values, model)
    values = np.asarray(values)
    
    # Get transition matrix (required for Markov, optional for IID)
    transition_matrix = None
    if shock_method == 'DiscreteMarkov':
        if 'transition_matrix' not in spec:
            raise ValueError("Manual shock method with 'DiscreteMarkov' requires transition_matrix")
        
        transition_matrix = spec.get('transition_matrix')
        # Resolve parameter references
        transition_matrix = resolve_reference(transition_matrix, model)
        transition_matrix = np.asarray(transition_matrix)
    
    # Validate parameters
    validate_manual_shock_parameters(transition_matrix, values)
    
    # Get optional parameters
    labels = spec.get('labels')
    stationary_distribution = spec.get('stationary_distribution')
    
    # Resolve stationary_distribution if provided
    if stationary_distribution is not None:
        stationary_distribution = resolve_reference(stationary_distribution, model)
        if stationary_distribution is not None:
            stationary_distribution = np.asarray(stationary_distribution)
    
    # Create appropriate shock process based on shock_method
    if shock_method == 'DiscreteMarkov':
        return DiscreteMarkovProcess(
            transition_matrix=transition_matrix,
            values=values,
            labels=labels,
            stationary_distribution=stationary_distribution
        )
    elif shock_method == 'IID':
        # For IID, get probabilities if specified, otherwise uniform
        probabilities = spec.get('probabilities')
        if probabilities is not None:
            probabilities = resolve_reference(probabilities, model)
            probabilities = np.asarray(probabilities)
        
        return IIDProcess(
            values=values,
            probabilities=probabilities,
            labels=labels
        )
    else:
        raise ValueError(f"Unsupported shock_method '{shock_method}' for manual shock process") 