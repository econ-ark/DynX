"""
FunctionalProblem class definition for Heptapod-B.

This module provides the core FunctionalProblem class for mathematical representation
of a problem, with separate .math and .num dictionaries for symbolic and numerical
components.
"""

from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from collections import OrderedDict


class AttrDict:
    """
    Helper class to provide attribute access to dictionary items.
    
    This class wraps a dictionary to allow attribute-style access to its items,
    while maintaining dictionary-style access through subscripting.
    """
    
    def __init__(self, data_dict=None):
        self._data = data_dict or {}
    
    def __getattr__(self, name):
        # Try to get from the data dictionary
        if name in self._data:
            # For attribute access, wrap nested dicts but don't recursively wrap
            if isinstance(self._data[name], dict):
                return AttrDict(self._data[name])
            return self._data[name]
        
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
    
    def __setattr__(self, name, value):
        # Special case for _data attribute
        if name == '_data':
            super().__setattr__(name, value)
            return
        
        # Set in the data dictionary
        self._data[name] = value
    
    def __getitem__(self, key):
        # For dict-style access, return the original dict value (no wrapping)
        if key in self._data:
            return self._data[key]
        raise KeyError(f"'{key}' not found")
    
    def __setitem__(self, key, value):
        # Support dictionary-style assignment
        self._data[key] = value
    
    def get(self, key, default=None):
        # Support get() method like a dictionary
        return self._data.get(key, default)
    
    def __dir__(self):
        """Return a list of available attributes"""
        return list(self._data.keys())
        
    def __iter__(self):
        """Support iteration over keys"""
        return iter(self._data)
        
    def items(self):
        """Support items() method like a dictionary"""
        return self._data.items()
        
    def keys(self):
        """Support keys() method like a dictionary"""
        return self._data.keys()
        
    def values(self):
        """Support values() method like a dictionary"""
        return self._data.values()
        
    def __contains__(self, key):
        """Support 'in' operator"""
        return key in self._data
        
    def __len__(self):
        """Support len() function"""
        return len(self._data)


class FunctionalProblem:
    """
    Class for mathematical representation of a problem.
    
    This class provides a structured approach to model representation with a consistent
    hierarchical access pattern.
    
    Attributes
    ----------
    math : AttrDict
        Dictionary containing mathematical components:
        - math.functions: Function definitions
        - math.constraints: Constraint definitions
        - math.state_space: State space definitions
        - math.shocks: Shock definitions and probabilities
    
    num : AttrDict
        Dictionary containing numerical components:
        - num.functions: Compiled function callables
        - num.constraints: Compiled constraint callables
        - num.state_space: Numerical grids and dimensions
        - num.shocks: Numerical shock grids and distributions
    
    parameters_dict : dict
        Dictionary of parameter values
    
    param : property
        Direct attribute-style access to parameters (e.g., problem.param.beta)
    
    methods : dict
        Dictionary of method references populated from configuration
    
    settings : dict
        Dictionary of numerical settings like tolerances
    
    operator : dict
        Operator configuration for movers
    
    Access Patterns
    --------------
    The recommended access pattern is container-based:
    
    - Mathematical components: problem.math.functions.function_name
    - Numerical components: problem.num.functions.function_name
    - Parameters: problem.param.parameter_name
    
    This enforces a consistent hierarchical structure.
    """
    def __init__(self):
        """Initialize an empty FunctionalProblem with default structure"""
        # Mathematical representation - original definitions
        self._math = {
            "functions": {},  # Function definitions
            "constraints": {},  # Constraint definitions
            "state_space": {},  # State space definitions (renamed from "state")
            "shocks": {},  # Shock definitions and probabilities
        }

        # Numerical representation - compiled components
        self._num = {
            "functions": {},  # Compiled function callables
            "constraints": {},  # Compiled constraint callables
            "state_space": {},  # Numerical grids and dimensions
            "shocks": {},  # Numerical shock grids and distributions
        }

        # Method references - initialized as empty and populated from config
        self.methods = {}  # Methods directly from configuration

        # Top-level elements - core model attributes for direct access
        self.parameters_dict = {}  # Dictionary mapping parameter names to their values
        self.settings = {}  # Dictionary of numerical settings like tolerances
        
        # Operator information (for movers only)
        self.operator = {}  # Operator config for movers
        
        # Define all properties that can be directly accessed
        self._direct_access_properties = {
            'parameters_dict': True, 
            'math': True,
            'num': True,
            'methods': True,
            'settings': True,
            'operator': True,
            'param': True,
        }

    def __getitem__(self, key):
        """Allow subscript access to properties"""
        if key in self._direct_access_properties:
            return getattr(self, key)
        else:
            raise KeyError(f"{key} is not a valid property")
            
    def __setitem__(self, key, value):
        """Allow subscript setting of properties"""
        if key in self._direct_access_properties:
            setattr(self, key, value)
        else:
            raise KeyError(f"{key} is not a valid property")

    @property
    def param(self):
        """Direct attribute-style access to parameters"""
        class _ParameterAccessor:
            def __init__(self, params_dict):
                self._params = params_dict
                
            def __getattr__(self, name):
                if name in self._params:
                    return self._params[name]
                raise AttributeError(f"Parameter '{name}' not found")
                
            def __dir__(self):
                return list(self._params.keys())
                
        return _ParameterAccessor(self.parameters_dict)



    @property
    def math(self):
        """Property that returns AttrDict wrapper for math dict"""
        return AttrDict(self._math)
    
    @math.setter
    def math(self, value):
        """Setter for math property"""
        self._math = value
        
    @property
    def num(self):
        """Property that returns AttrDict wrapper for num dict"""
        return AttrDict(self._num)
    
    @num.setter
    def num(self, value):
        """Setter for num property"""
        self._num = value 