"""
Heptapod-B Numerical Module

This module provides functions for generating numerical components:
- Function compilation
- Grid generation
- Shock grid generation
- Numerical model generation
"""

from .compile import (
    compile_function,
    compile_eval_function,
    compile_sympy_function,
    compile_numba_function
)

from .state_space import (
    generate_grid,
    generate_chebyshev_grid,
    create_mesh_grid,
    generate_numerical_state_space
)

from .shocks import (
    build_normal_shock_grid,
    build_lognormal_shock_grid,
    build_adaptive_shock_grid,
    build_discrete_markov_shock_grid,
    tauchen_method,
    rouwenhorst_method,
    generate_numerical_shocks
)

from .generate import (
    compile_num,
    initialize_methods,
    generate_numerical_functions,
    generate_numerical_constraints
)

__all__ = [
    # Function compilation
    "compile_function",
    "compile_eval_function",
    "compile_sympy_function",
    "compile_numba_function",
    
    # Grid generation
    "generate_grid",
    "generate_chebyshev_grid",
    "create_mesh_grid",
    "generate_numerical_state_space",
    
    # Shock generation
    "build_normal_shock_grid",
    "build_lognormal_shock_grid",
    "build_adaptive_shock_grid",
    "build_discrete_markov_shock_grid",
    "tauchen_method",
    "rouwenhorst_method",
    "generate_numerical_shocks",
    
    # Numerical model generation
    "compile_num",
    "initialize_methods",
    "generate_numerical_functions",
    "generate_numerical_constraints"
]
