"""
Function compilation module for Heptapod-B.

This module provides functions to compile mathematical expressions
into callable Python functions using various methods:
- eval (default)
- sympy
- numba
"""

from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from collections import OrderedDict
import re
import warnings


def compile_function(
    expr: str,
    parameters: Dict[str, Any],
    method: str = 'eval',
    all_compiled_funcs: Optional[Dict[str, Callable]] = None
) -> Callable:
    """
    Compile a function expression using the specified method.
    
    Parameters
    ----------
    expr : str
        Function expression to compile
    parameters : dict
        Dictionary of parameters to use in function
    method : str
        Compilation method: 'eval', 'sympy', or 'numba'
    all_compiled_funcs : dict, optional
        Dictionary of already compiled functions to use in compilation
        
    Returns
    -------
    callable
        Compiled function
    """
    # Extract formula part for eval
    if isinstance(expr, str) and '=' in expr:
        formula = expr.split('=', 1)[1].strip()
    else:
        formula = expr
    
    # Handle deprecated methods
    if method == 'function_compilation':
        warnings.warn(
            "'function_compilation' is deprecated. Use 'compilation' instead. "
            "This will be removed in v1.7.",
            DeprecationWarning,
            stacklevel=2
        )
        method = 'eval'  # Default fallback
    
    if method == 'sympy':
        return compile_sympy_function(formula, parameters, all_compiled_funcs)
    elif method == 'numba':
        return compile_numba_function(formula, parameters, all_compiled_funcs)
    else:  # Default to 'eval'
        return compile_eval_function(formula, parameters, all_compiled_funcs)


def compile_eval_function(
    expr: str,
    parameters: Dict[str, Any],
    all_compiled_funcs: Optional[Dict[str, Callable]] = None
) -> Callable:
    """
    Compile a function expression using Python's eval function.
    
    Parameters
    ----------
    expr : str
        Function expression to compile
    parameters : dict
        Dictionary of parameters to use in function
    all_compiled_funcs : dict, optional
        Dictionary of already compiled functions to use in compilation
        
    Returns
    -------
    callable
        Compiled function that accepts **kwargs
    """
    import math
    import numpy as np
    
    # Create a dictionary with available functions and parameters
    eval_globals = {
        'math': math,
        'np': np,
        'sin': math.sin,
        'cos': math.cos,
        'exp': math.exp,
        'log': math.log,
        'sqrt': math.sqrt,
        'abs': abs,
        'max': max,
        'min': min,
        **parameters
    }
    
    # Add already compiled functions if provided
    if all_compiled_funcs:
        eval_globals.update(all_compiled_funcs)
    
    # Create the callable function
    def scalar_func(*args, **kwargs):
        # Build a fresh namespace on *every* call so that newly compiled
        # functions added to ``all_compiled_funcs`` **after** this
        # function was created are still visible (resolves forward
        # references like Q_func using u_func defined later).

        dynamic_globals = eval_globals.copy()
        if all_compiled_funcs:
            # Merge in any functions compiled after this one
            dynamic_globals.update(all_compiled_funcs)

        # Combine with user-supplied keyword arguments
        local_vars = {**dynamic_globals, **kwargs}
        
        # If positional arguments are provided, try to map them to expected arguments
        # This handles the case where the function is called with positional args instead of named args
        if args:
            # Try to infer parameter names from the expression
            # Look for patterns that might indicate variable names (e.g., "c**alpha", "np.log(c)")
            import re
            # Pattern matches variable names in various contexts (standalone, function args, operations)
            var_pattern = r'\b([a-zA-Z_]\w*)\b(?!\s*\()'
            potential_vars = re.findall(var_pattern, expr)
            
            # Filter out known functions, keywords, and parameters
            potential_vars = [v for v in potential_vars if v not in ['np', 'math', 'sin', 'cos', 'exp', 
                                                                     'log', 'sqrt', 'abs', 'max', 'min', 
                                                                     'if', 'else', 'for', 'while', 'and', 
                                                                     'or', 'not', 'in', 'is'] 
                             and v not in eval_globals]
            
            # Remove duplicates while preserving order
            seen = set()
            unique_vars = [x for x in potential_vars if not (x in seen or seen.add(x))]
            
            # Map positional args to variable names
            for i, arg_val in enumerate(args):
                if i < len(unique_vars):
                    local_vars[unique_vars[i]] = arg_val
        
        # Use Python's built-in eval function with a restricted globals dict
        # but allow all key functions to be accessed
        builtins_dict = {
            'abs': abs, 'all': all, 'any': any, 'bool': bool, 'dict': dict,
            'float': float, 'int': int, 'len': len, 'list': list, 'max': max, 
            'min': min, 'range': range, 'round': round, 'sum': sum, 'tuple': tuple,
            'zip': zip, 'map': map, 'filter': filter, 'enumerate': enumerate,
            'isinstance': isinstance, 'issubclass': issubclass
        }
        globals_dict = {"__builtins__": builtins_dict}
        try:
            return eval(expr, globals_dict, local_vars)
        except Exception as e:
            raise ValueError(f"Error evaluating expression '{expr}': {str(e)}")
    
    return scalar_func


def compile_sympy_function(
    expr: str,
    parameters: Dict[str, Any],
    all_compiled_funcs: Optional[Dict[str, Callable]] = None
) -> Callable:
    """
    Compile a function expression using SymPy's lambdify.
    
    Parameters
    ----------
    expr : str
        Function expression to compile
    parameters : dict
        Dictionary of parameters to use in function
    all_compiled_funcs : dict, optional
        Dictionary of already compiled functions to use in compilation
        
    Returns
    -------
    callable
        Compiled function that accepts **kwargs
    """
    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import parse_expr
        from sympy.utilities.lambdify import lambdify
    except ImportError:
        warnings.warn(
            "SymPy not installed. Falling back to 'eval' compilation method.",
            ImportWarning
        )
        return compile_eval_function(expr, parameters, all_compiled_funcs)
    
    # Parse the expression
    try:
        # First, substitute any pre-compiled functions with their symbolic equivalents
        if all_compiled_funcs:
            # This is a simplification - for full support, you'd need to parse
            # the expression and create symbolic functions for each compiled function
            for func_name in all_compiled_funcs:
                # Just a simple placeholder to avoid errors during parsing
                if func_name in expr:
                    warnings.warn(
                        f"Referenced function '{func_name}' in sympy-compiled expression "
                        f"may not work as expected",
                        UserWarning
                    )
        
        # Create symbols for all parameters
        param_symbols = {k: sp.Symbol(k) for k in parameters}
        
        # Parse the expression
        sympy_expr = parse_expr(expr, local_dict=param_symbols)
        
        # Identify all symbols used in the expression
        used_symbols = [sym for sym in sympy_expr.free_symbols if sym.name not in parameters]
        used_symbol_names = [sym.name for sym in used_symbols]
        
        # Create a lambda function using sympy's lambdify
        lambda_func = lambdify(used_symbols, sympy_expr, modules=['numpy'])
        
        # Create the wrapper function that accepts arbitrary kwargs
        def wrapper_func(**kwargs):
            # Extract the values for each symbol
            args = [kwargs[name] for name in used_symbol_names]
            return lambda_func(*args)
        
        return wrapper_func
    except Exception as e:
        warnings.warn(
            f"Error in sympy compilation: {str(e)}. Falling back to 'eval'.",
            UserWarning
        )
        return compile_eval_function(expr, parameters, all_compiled_funcs)


def compile_numba_function(
    expr: str,
    parameters: Dict[str, Any],
    all_compiled_funcs: Optional[Dict[str, Callable]] = None
) -> Callable:
    """
    Compile a function expression using Numba's JIT compiler.
    
    Parameters
    ----------
    expr : str
        Function expression to compile
    parameters : dict
        Dictionary of parameters to use in function
    all_compiled_funcs : dict, optional
        Dictionary of already compiled functions to use in compilation
        
    Returns
    -------
    callable
        Compiled function that accepts **kwargs
    """
    try:
        import numba
    except ImportError:
        warnings.warn(
            "Numba not installed. Falling back to 'eval' compilation method.",
            ImportWarning
        )
        return compile_eval_function(expr, parameters, all_compiled_funcs)
    
    try:
        # For simplicity, we'll first create an eval function then JIT-compile it
        # This approach is not ideal for complex expressions but provides a path forward
        eval_func = compile_eval_function(expr, parameters, all_compiled_funcs)
        
        # Calculate function signature based on the expression
        # Parse out variable names using a simple regex
        var_names = set(re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', expr))
        # Remove Python keywords and built-in functions
        var_names = var_names - set([
            'if', 'else', 'for', 'while', 'return', 'and', 'or', 'not',
            'True', 'False', 'None', 'import', 'from', 'as', 'def', 'class',
            'print', 'min', 'max', 'abs', 'sum', 'sin', 'cos', 'exp', 'log', 'sqrt'
        ])
        # Remove parameter names
        var_names = var_names - set(parameters.keys())
        # Remove already compiled function names
        if all_compiled_funcs:
            var_names = var_names - set(all_compiled_funcs.keys())
        
        # Now var_names should contain only the actual variables needed
        
        # Create a simple JIT function for these variables
        @numba.jit(nopython=True)
        def numba_func(*args):
            # This is a major simplification and won't work for complex expressions
            # Real Numba compilation would parse the expression and generate
            # optimized code with proper type annotations
            return eval_func(**{name: arg for name, arg in zip(var_names, args)})
        
        # Create a wrapper that converts kwargs to positional args
        def wrapper_func(**kwargs):
            args = [kwargs[name] for name in var_names]
            return numba_func(*args)
        
        return wrapper_func
    except Exception as e:
        warnings.warn(
            f"Error in numba compilation: {str(e)}. Falling back to 'eval'.",
            UserWarning
        )
        return compile_eval_function(expr, parameters, all_compiled_funcs) 