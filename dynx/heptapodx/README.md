# Heptapod-B

**Heptapod-B** is an *embarrassingly* vectorised DSL for functional-graph modelling.  


It doesn't out-smart NumPy—it organises:
* interpolation & integration
* bilinear operators
* symbolic → numeric compilation  

so you can write  
```yaml
util_and_mutil:
  util:  "c**(1-g)/(1-g)"
  mutil: "c**(-g)"
```  
once and evaluate it on any mesh, Markov shock, or solver scheme.

It doesn't out-smart YOU. You define the functions you need for each stage and mover and implement them in YOUR solver as you see fit.

## Usage

```python
from dynx.heptapodx.core import FunctionalProblem
from dynx.heptapodx.init.stage import build_stage
from dynx.heptapodx.num.generate import compile_num

# Load configuration from YAML
stage_problem = build_stage("config.yml")

# Compile numerical model
compile_num(stage_problem)

# Access computed values
print(stage_problem.num.functions)
```

## Package Structure

Heptapod-B is a PEP 420 namespace package with the following structure:

```
dynx/heptapodx/
├── __init__.py           # Re-exports top-level API
├── core/                 # Core components
│   ├── functional_problem.py  # FunctionalProblem class + AttrDict wrapper
│   └── validation.py          # state-space & config validators
├── init/                 # Initialization components
│   ├── stage.py          # build_stage (was initialize_stage)
│   ├── mover.py          # build_mover (was initialize_mover)
│   └── perch.py          # build_perch (was initialize_perch)
├── resolve/              # Reference resolution
│   └── methods.py        # _resolve_method_references, alias tables
├── num/                  # Numerical components
│   ├── compile.py        # compile_function dispatch (eval/sympy/numba)
│   ├── state_space.py    # grid builders incl. Tauchen/Rouwenhorst
│   ├── shocks.py         # normal/lognormal/adaptive grids
│   └── generate.py       # generate_numerical_model orchestrator
└── io/                   # Input/output utilities
    └── yaml_loader.py    # safe-load + custom tags
```

## Compatibility

For backward compatibility, you can still use the old import paths:

```python
import heptapod_b as hb
```

These will emit a `DeprecationWarning` and will be removed in a future version. Please use the new import paths:

```python
import dynx.heptapodx as hx
```

# Heptapod-B Configuration Parameter Resolution

After reviewing the code, I can provide documentation for how parameter resolution works in Heptapod-B:

## Configuration Hierarchy and Parameter Resolution

Heptapod-B supports a hierarchical configuration system with parameters defined at multiple levels:

1. **Global level**: Parameters and settings defined at the stage level
2. **State/Perch level**: Each state (perch) can have its own settings and parameters
3. **Mover level**: Each mover can have its own settings and parameters

### Parameter Resolution Rules

Parameter resolution follows these precedence rules:

1. **Direct numerical values**: If a numerical value is directly provided, it's used as-is
2. **Local references**: References first look in the local scope (state or mover settings)
3. **Inherited references**: If not found locally and `inherit_parameters: true`, they're resolved from global settings

### State Space Grid Resolution

For grid specifications within a state:

```yaml
grid:
  m:
    type: "linspace"
    min: m_min       # Reference to state-level setting
    max: m_max       # Reference to state-level setting
    points: m_points # Reference to state-level setting
```

The grid parameters (like `m_min`) are resolved as follows:
1. First, check the state's own settings section
2. If not found and inheritance is enabled, check global settings
3. If still not found, raise an error (no fallbacks)

This is implemented in `generate_numerical_state_space()` which:
1. Collects global parameters, settings, and methods
2. For each state, collects its specific settings and methods
3. Combines them with proper precedence (state settings override global)
4. Passes the combined parameters to grid generation functions

```python
# Get global parameters and settings
global_params = {}
if hasattr(problem, "parameters"):
    global_params.update(problem.parameters)
if hasattr(problem, "settings"):
    global_params.update(problem.settings)
if hasattr(problem, "methods"):
    global_params.update(problem.methods)

# Get state-specific settings
state_settings = {}
if "settings" in state_info:
    state_settings.update(state_info["settings"])
if "methods" in state_info:
    state_settings.update(state_info["methods"])

# Combine with state settings taking precedence
combined_params = {**global_params}
combined_params.update(state_settings)
```

### Mover Parameter Resolution

For movers with `inherit_parameters: true`, the same resolution process applies:
1. Mover-specific parameters take precedence
2. Global parameters are used as fallbacks

This ensures that when functions are built at the mover level, they have access to all required parameters.

### Reference Resolution Implementation

The `resolve_reference()` function handles resolving parameter references:
1. If a value is a direct reference (string or list format), it looks up the actual value
2. It handles both formats: `['param_name']` and direct string `'param_name'`
3. It has recursion protection to prevent infinite loops
4. It converts string values to numeric types when possible

```python
# Handle list references - format: ['reference_name']
if isinstance(value, list) and len(value) == 1 and isinstance(value[0], str):
    ref_key = value[0]
    if ref_key in params:
        # Recursively resolve the reference
        return resolve_reference(params[ref_key], params, _recursion_depth + 1)
    else:
        raise ValueError(f"Unresolved reference: {ref_key}")

# Handle string references - direct parameter names
elif isinstance(value, str):
    # Check if it's a direct parameter reference
    if value in params:
        # Recursively resolve the reference
        return resolve_reference(params[value], params, _recursion_depth + 1)
```

## Current Behavior and Compliance

The current implementation correctly follows these parameter resolution rules, ensuring that:

1. State-level grid specifications correctly use parameters from the state's settings section
2. Fallback to global settings happens when a parameter isn't found locally
3. References are fully resolved before being used in numerical calculations
4. Type conversion happens appropriately to ensure parameters are in the correct format

This parameter resolution system allows for a flexible but predictable configuration system where parameters can be reused across different levels of the configuration.
