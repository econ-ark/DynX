"""
Numerical model generation module for Heptapod-B.

This module provides functions to generate a complete numerical model
from a FunctionalProblem instance with analytical definitions.
"""

from typing import Dict, List, Any, Optional, Union, Callable, Tuple
import warnings
from collections import OrderedDict

from ..core.functional_problem import FunctionalProblem
from ..core.validation import emit_deprecation_warning
from .compile import compile_function
from .state_space import generate_numerical_state_space
from .shocks import generate_numerical_shocks


def compile_num(problem: FunctionalProblem, methods: Optional[Dict[str, Any]] = None) -> FunctionalProblem:
    """
    Generate a numerical model from a FunctionalProblem instance.

    This function takes a FunctionalProblem with analytical definitions (.math)
    and creates the corresponding numerical implementations (.num) based on
    the methods specified in the model itself.

    Parameters
    ----------
    problem : FunctionalProblem
        The model with analytical definitions
    methods : dict, optional
        Additional methods to override the model's methods.
        If None, uses only the methods from problem.methods

    Returns
    -------
    FunctionalProblem
        The same model instance with populated .num attributes
    """
    # Ensure model has methods
    if not hasattr(problem, "methods"):
        problem.methods = {}

    # Initialize methods with defaults if not provided
    model_methods = initialize_methods(problem)

    # If external methods are provided, they override the model's methods
    if methods is not None:
        # Create a copy of model_methods to avoid modifying the original
        merged_methods = model_methods.copy()
        # Update with provided methods
        merged_methods.update(methods)
    else:
        # Only use model's methods
        merged_methods = model_methods

    # Generate numerical components, using the model's methods
    generate_numerical_functions(problem, merged_methods)
    generate_numerical_constraints(problem, merged_methods)
    
    # Explicitly generate numerical state spaces and ensure they're in problem.num
    if hasattr(problem, "math") and "state_space" in problem.math:
        state_spaces = generate_numerical_state_space(problem, merged_methods)
        # Explicitly ensure the generated state spaces are assigned to problem.num
        if "state_space" not in problem.num or not problem.num["state_space"]:
            problem.num["state_space"] = state_spaces
    
    generate_numerical_shocks(problem, merged_methods)

    return problem


def initialize_methods(problem: FunctionalProblem) -> Dict[str, Any]:
    """
    Initialize compilation methods in the model based on configuration.
    
    This function sets up the compilation methods using the standardized keys,
    with appropriate deprecation warnings for old keys.
    
    Parameters
    ----------
    problem : FunctionalProblem
        The model instance
        
    Returns
    -------
    dict
        Dictionary of initialized methods
    """
    if not hasattr(problem, "methods"):
        problem.methods = {}
        
    methods = problem.methods
    
    # Handle global compilation method
    if "compilation" not in methods:
        # Check for deprecated keys with warnings
        if "function_compilation" in methods:
            emit_deprecation_warning("function_compilation", "compilation")
            methods["compilation"] = methods["function_compilation"]
        elif "constraint_compilation" in methods:
            emit_deprecation_warning("constraint_compilation", "compilation")
            methods["compilation"] = methods["constraint_compilation"]
        else:
            # Default to 'eval' if not specified
            methods["compilation"] = "eval"
    
    # Handle grid type
    if "grid.type" not in methods and "default_grid" not in methods:
        if "grid_generation" in methods:
            emit_deprecation_warning("grid_generation", "grid.type")
            methods["grid.type"] = methods["grid_generation"]
        else:
            # Default to 'linspace' if not specified
            methods["grid.type"] = "linspace"
    
    # Handle shock method
    if "shock_method" not in methods:
        if "shock_distribution" in methods:
            emit_deprecation_warning("shock_distribution", "shock_method")
            methods["shock_method"] = methods["shock_distribution"]
        else:
            # Default to 'normal' if not specified
            methods["shock_method"] = "normal"
            
    return methods


def generate_numerical_functions(problem: FunctionalProblem, methods: Optional[Dict[str, Any]] = None) -> Dict[str, Callable]:
    """
    Generate numerical functions from analytical expressions in problem.math.
    
    This function iterates through the functions defined in problem.math['functions']
    and compiles them into callable Python functions stored in problem.num['functions'].
    It uses the compilation method specified in methods or problem.methods.compilation.
    
    Parameters
    ----------
    problem : FunctionalProblem
        The model with analytical function definitions
    methods : dict, optional
        Methods dictionary with compilation settings
        
    Returns
    -------
    dict
        Dictionary of compiled functions
    """
    if not hasattr(problem, "math") or "functions" not in problem.math:
        warnings.warn(
            "problem.math['functions'] not found. Skipping function compilation.",
            UserWarning
        )
        return {}
    
    # Ensure numerical functions container exists
    if "functions" not in problem.num:
        problem.num["functions"] = {}
    
    # Initialize parameters dict
    parameters = getattr(problem, "parameters_dict", {})
    
    # Initialize methods with defaults if not provided
    if methods is None:
        # Use model's own methods
        if hasattr(problem, "methods"):
            methods = problem.methods
        else:
            methods = {}
    
    # Get compilation method with deprecation handling
    compilation_method = methods.get("compilation", "eval")
    
    # Store the dictionary of all compiled functions for reference within multi-output functions
    all_compiled_funcs = {}
    
    # Process each function
    for func_name, func_def in problem.math["functions"].items():
        # Skip if already compiled
        if func_name in problem.num["functions"]:
            continue
        
        # Handle multi-output functions (dictionaries with named sub-expressions)
        if isinstance(func_def, dict) and "expr" not in func_def:
            # Skip metadata keys (description, documentation)
            output_names = [k for k in func_def.keys() if k not in ["description", "documentation", "compilation"]]
            
            # Create individual output expressions
            output_functions = {}
            for output_name in output_names:
                output_expr = func_def[output_name]
                
                # Each output is a string expression or a dict with 'expr' key
                if isinstance(output_expr, dict) and "expr" in output_expr:
                    # Extract expression from dict
                    expr = output_expr["expr"]
                    # Extract specific method for this output if available
                    output_method = output_expr.get("compilation", 
                                                   output_expr.get("function_compilation", compilation_method))
                    
                    # Check for deprecated key in output
                    if "function_compilation" in output_expr:
                        emit_deprecation_warning(
                            "function_compilation", "compilation", 
                            f"function '{func_name}.{output_name}'"
                        )
                else:
                    # Direct expression
                    expr = output_expr
                    output_method = func_def.get("compilation", compilation_method)
                
                # Compile the individual output function
                try:
                    output_functions[output_name] = compile_function(
                        expr, parameters, method=output_method, all_compiled_funcs=all_compiled_funcs
                    )
                except Exception as e:
                    warnings.warn(
                        f"Error compiling function '{func_name}.{output_name}': {e}",
                        UserWarning
                    )
                    # Create a simple error function
                    output_functions[output_name] = lambda *args, error=str(e): exec(f"raise ValueError('{error}')")
            
            # Define a multi-output function that returns all outputs
            def create_multi_output_func(name, outputs, expressions):
                def multi_output_func(**kwargs):
                    result = OrderedDict()
                    for out_name in outputs:
                        result[out_name] = expressions[out_name](**kwargs)
                    return result
                return multi_output_func
            
            # Define wrapper functions to get individual outputs with the same signature
            def create_scalar_wrapper(main_func, output):
                def wrapper(**kwargs):
                    return main_func(**kwargs)[output]
                return wrapper
            
            # Create the main multi-output function
            main_func = create_multi_output_func(func_name, output_names, output_functions)
            main_func.__name__ = func_name
            
            # Store the main function and scalar accessors
            problem.num["functions"][func_name] = main_func
            all_compiled_funcs[func_name] = main_func
            
            # Create and store direct accessor functions for each output
            for output_name in output_names:
                accessor_name = f"{func_name}_{output_name}"
                accessor_func = create_scalar_wrapper(main_func, output_name)
                accessor_func.__name__ = accessor_name
                problem.num["functions"][accessor_name] = accessor_func
                all_compiled_funcs[accessor_name] = accessor_func
                
        else:
            # Handle scalar functions (either a string expression or a dict with 'expr' key)
            if isinstance(func_def, dict) and "expr" in func_def:
                # Extract expression from dict
                expr = func_def["expr"]
                # Extract specific method for this function if available
                func_method = func_def.get("compilation", 
                                          func_def.get("function_compilation", compilation_method))
                
                # Check for deprecated key
                if "function_compilation" in func_def:
                    emit_deprecation_warning("function_compilation", "compilation", f"function '{func_name}'")
            else:
                # Direct expression
                expr = func_def
                func_method = compilation_method
            
            # Compile the scalar function
            try:
                func = compile_function(
                    expr, parameters, method=func_method, all_compiled_funcs=all_compiled_funcs
                )
                func.__name__ = func_name
                problem.num["functions"][func_name] = func
                all_compiled_funcs[func_name] = func
            except Exception as e:
                warnings.warn(
                    f"Error compiling function '{func_name}': {e}",
                    UserWarning
                )
                # Create a simple error function
                problem.num["functions"][func_name] = lambda *args, error=str(e): exec(f"raise ValueError('{error}')")
    
    return problem.num["functions"]


def generate_numerical_constraints(problem: FunctionalProblem, methods: Optional[Dict[str, Any]] = None) -> Dict[str, Callable]:
    """
    Generate numerical constraints from analytical constraint definitions.

    This function uses the model's own method specifications to determine how
    to compile each constraint, ensuring internal consistency.

    Parameters
    ----------
    problem : FunctionalProblem
        The model with analytical constraint definitions
    methods : dict, optional
        Methods dictionary with compilation settings

    Returns
    -------
    dict
        Dictionary of compiled constraints
    """
    if not hasattr(problem, "math") or "constraints" not in problem.math:
        return {}

    # Ensure numerical constraints container exists
    if "constraints" not in problem.num:
        problem.num["constraints"] = {}

    # Get parameters dictionary
    parameters = getattr(problem, "parameters_dict", {})
    
    # Initialize methods with defaults if not provided
    if methods is None:
        # Use model's own methods
        if hasattr(problem, "methods"):
            methods = problem.methods
        else:
            methods = {}
    
    # Get compilation method with deprecation handling
    compilation_method = methods.get("compilation", "eval")
    
    # Store all compiled functions and constraints for reference
    all_compiled_funcs = {}
    if hasattr(problem, "num") and "functions" in problem.num:
        all_compiled_funcs.update(problem.num["functions"])
    
    # Process each constraint
    for constraint_name, constraint_def in problem.math["constraints"].items():
        # Skip if already compiled
        if constraint_name in problem.num["constraints"]:
            continue
            
        if isinstance(constraint_def, dict) and "expr" in constraint_def:
            # Complex constraint definition with metadata
            expr = constraint_def["expr"]
            
            # Get constraint-specific compilation method
            constraint_method = constraint_def.get("compilation", 
                                                  constraint_def.get("constraint_compilation", compilation_method))
            
            # Check for deprecated key
            if "constraint_compilation" in constraint_def:
                emit_deprecation_warning("constraint_compilation", "compilation", f"constraint '{constraint_name}'")
        else:
            # Simple expression
            expr = constraint_def
            constraint_method = compilation_method
            
        # Compile the constraint
        try:
            constraint_func = compile_function(
                expr, parameters, method=constraint_method, all_compiled_funcs=all_compiled_funcs
            )
            constraint_func.__name__ = constraint_name
            problem.num["constraints"][constraint_name] = constraint_func
            all_compiled_funcs[constraint_name] = constraint_func
        except Exception as e:
            warnings.warn(
                f"Error compiling constraint '{constraint_name}': {e}",
                UserWarning
            )
            # Create a simple error function
            problem.num["constraints"][constraint_name] = lambda *args, error=str(e): exec(f"raise ValueError('{error}')")
    
    return problem.num["constraints"] 