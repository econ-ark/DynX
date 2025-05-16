"""
DynX HeptapodX Module

HeptapodX is a YAML-based functional problem definition layer for economic models.
It allows users to define economic models in a declarative way, with a focus on
mathematical expressions and state spaces.
"""

# Re-export public API
from dynx.heptapodx.core.api import (
    FunctionalProblem,
    AttrDict,
    initialize_model,
    initialize_stage,
    initialize_mover,
    initialize_perch,
    load_config,
    load_functions_from_yaml,
    generate_numerical_model,
    compile_function,
    compile_eval_function,
    compile_sympy_function,
    compile_numba_function
)

__all__ = [
    "FunctionalProblem",
    "AttrDict",
    "initialize_model",
    "initialize_stage",
    "initialize_mover",
    "initialize_perch",
    "load_config",
    "load_functions_from_yaml",
    "generate_numerical_model",
    "compile_function",
    "compile_eval_function",
    "compile_sympy_function",
    "compile_numba_function"
]
