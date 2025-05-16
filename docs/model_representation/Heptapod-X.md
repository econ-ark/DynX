# Heptapod-X
> *Mathematical representation and parameter resolution for recursive economic models*
>

This document outlines the architecture and functionality of Heptapod-X.

Heptapod-B is an ***embarrassingly vectorized*** model representation for functional operations. 
It directly uses NumPy's existing vectorized operations and adds minimal extra syntax to represent:
- interpolation (as function evaluation)
- integration
- and bilinear operations.  

All algebraic operations in Heptapod-B are defined on vectors and return vectors, providing two key advantages:
- Operations automatically handle arrays of any dimension without requiring explicit loops. 
- Since dynamic optimization ***is*** a functional operation, expressing model equations directly on vector spaces decisively eliminates ambiguities between mathematical model specification and numerical implementation.
- This leads to direct translation from mathematical expressions, to numerical functions to solution algorithms.

>Tv-v =**0**

## Contents
1. [Mathematical Representation](#mathematical-representation)
   - Symbolic vs. Numerical Representations
   - FunctionalProblem Architecture
   - Structure of Model Attributes
   - Parameters and Settings Structure
2. [Parameter Resolution](#parameter-resolution)
   - Reference Syntax
   - Resolution Process
   - Scope Levels
   - Resolution Definition
   - Standard Methods and Keys
3. [Model Structure](#model-structure)
   - Two-Level Hierarchy: Stage and Substages
   - Substage Types
   - YAML Representation
4. [Integration with StageCraft](#integration-with-stagecraft)
   - Model Connection to Perches and Movers
   - Whisperer Integration
5. [Key Components](#key-components)
   - Function Compilation
   - Grid Generation
   - State Space Construction
6. [Vectorization Principles](#vectorization-principles)
   - Automatic Vectorization
   - Vector Semantics
   - Performance Considerations

## Mathematical Representation

Heptapod-B separates symbolic and numerical representations of functional equations specified in the YAML configuration file.

The primary object created by heptapod-b is the `FunctionalProblem` which is instantiated into a model:

| Component | Description | Subobjects | Access |
|-----------|-------------|------------|--------|
| **`.math`** | Symbolic, analytical representation | `functions`, `state_space`,`constraints`, `shocks` | `model.math` |
| **`.num`** | Numerical, computational representation | `functions`, `state_space`, `constraints`, `shocks` | `model.num` |
| **`.params`** | Economic parameters | `beta`, `gamma`, `r`, etc. (user-defined direct numeric values) | `model.params` |
| **`.settings`** | Computational settings | `grid`, `tolerance`, `max_iter`, etc. (user-defined direct numeric values)  | `model.settings` |
| **`.methods`** | Solution methods | `interp_linear`, `optimization`, `shock_method`, etc. (user-defined key - method pairs)| `model.methods` |

This separation enables:
- Distinction between model specification and numerical implementation
- Late binding of numerical methods to mathematical definitions
- Alternative solution approaches for the same model specification

### FunctionalProblem Architecture

The `FunctionalProblem` class encapsulates the model representation, following the structure in `model_init.py`:

```python
from heptapod_b.core import FunctionalProblem
from heptapod_b.init.stage import build_stage
from heptapod_b.num.generate import compile_num

# Create symbolic representation
problem = build_stage("config.yaml")  # loads YAML → FunctionalProblem

# Compile numerical representation
compile_num(problem)  # FunctionalProblem.math → FunctionalProblem.num
```


### Structure of Model Attributes

The following table shows the correspondence between `.math` and `.num` attributes:

| Component | `.math` (Symbolic) | `.num` (Numerical) |
|-----------|-------------------|-------------------|
| **.functions** | String expressions<br>`"c**(1-gamma)/(1-gamma)"` | Callable functions<br>`<function utility at 0x...>` |
| **.state_space** | Grid specifications<br>`{"type": "geomspace", "min": 0.1, ...}` | NumPy arrays<br>`array([0.1, 0.28, 0.79, ...])` |
| **.shocks** | Distribution parameters<br>`{"rho": 0.9, "sigma": 0.2}` | Discretized values & matrices<br>`{"values": array([...]), "probs": array([...])}` |
| **.constraints** | - | - |

## Parameters and Settings Structure

Parameters and settings are collections of numeric values that can be accessed as attributes:

```python
# Parameter values
problem.parameters = {
    "beta": 0.96,        # Discount factor
    "gamma": 2.0,        # Risk aversion
    "r": 0.03,           # Interest rate
    "w": 1.0,            # Wage
    "rho": 0.8,          # AR(1) persistence
    "sigma": 0.2,        # Shock standard deviation
    "grid_min": 0.0,     # Grid minimum
    "grid_max": 100.0,   # Grid maximum
    "grid_size": 100     # Grid size
}

# Access as attributes
beta = problem.parameters.beta       # 0.96
gamma = problem.parameters.gamma     # 2.0
```
```python
# Settings values
problem.settings = {
    "grid": {
        "type": "geomspace",
        "min": 0.1,
        "max": 100.0,
        "points": 50
    },
    "tolerance": 1e-6,
    "max_iter": 1000,
    "damping": 0.5,
    "debug": False
}

# Access as attributes
grid_type = problem.settings.grid.type       # "geomspace"
tolerance = problem.settings.tolerance       # 1e-6
```

### Parameter Resolution

Parameter resolution implements reference-based value assignment and lookup.

#### Reference Syntax



```yaml
# Reference to a method or setting (will be resolved)
shock_method: [tauchen]

# Literal value (will not be resolved)
shock_method: tauchen

# Alternative reference syntax with string
shock_method: "[tauchen]"
```

### Resolution Process

Parameter resolution follows these steps:

1. Identify reference patterns (`[name]` or `"[name]"`)
2. Search for the reference in methods (`methods` attribute)
3. If not found in methods, check settings (`settings` attribute)
4. If still not found, leave as is and emit a warning
5. Apply recursively until all references are resolved or max depth reached

### Scope Levels

Parameters have a resolution hierarchy:

| Level | Scope | Override Precedence |
|-------|-------|---------------------|
| Component-specific | Limited to specific function or state | Highest |
| Method-specific | Available to all components using the method | Medium |
| Global | Available to all components | Lowest |

Example:

```yaml
# Global parameter
parameters:
  beta: 0.96

# Method-specific parameter
methods:
  interp_linear:
    extrapolate: false

# Component-specific parameter override
state:
  assets:
    grid:
      type: geomspace
      min: 0.1
      max: 100.0
      points: 50
```

### Resolution Definition

In Heptapod-B, **resolution** refers to the process of replacing reference placeholders with their actual values. A reference is **resolved** when:

1. A reference token (`[name]` or `"[name]"`) is identified in the configuration
2. The referenced value is found in either `methods` or `settings`
3. The reference token is replaced with the actual value from the referenced location
4. All nested references within that value are also recursively resolved
5. The final concrete value (with no remaining references) is used in place of the original reference

This resolution happens in two phases:
- **Initialization phase**: When the YAML configuration is loaded
- **Compilation phase**: When numerical representations are created from symbolic ones

References that cannot be resolved during either phase will keep their original form and a warning will be emitted.

### Standard Methods and Keys

Heptapod-B defines several standard methods with specific allowed keys. These methods control how different components of the model are processed:

#### Compilation Methods
| Method | Description | Allowed Keys |
|--------|-------------|--------------|
| **`eval`** | Direct Python evaluation | None |
| **`sympy`** | Symbolic mathematics via SymPy | `use_lambdify` (bool) |
| **`numba`** | Just-in-time compilation | `nopython` (bool), `parallel` (bool), `fastmath` (bool) |

#### Grid Generation Methods
| Method | Description | Required Keys | Optional Keys |
|--------|-------------|--------------|---------------|
| **`linspace`** | Uniform grid | `min`, `max`, `points` | None |
| **`geomspace`** | Geometric grid | `min`, `max`, `points` | None |
| **`chebyshev`** | Chebyshev nodes | `min`, `max`, `points` | None |
| **`int_range`** | Integer range | `start`, `stop` | `step` |
| **`list`** / **`enum`** | Custom points | `values` | `labels` |

#### Interpolation Methods
| Method | Description | Allowed Keys |
|--------|-------------|--------------|
| **`linear`** | Linear interpolation | `extrapolate` (bool) |
| **`cubic`** | Cubic spline interpolation | `extrapolate` (bool), `bc_type` (str) |
| **`nearest`** | Nearest neighbor interpolation | `extrapolate` (bool) |
| **`piecewise`** | Piecewise constant interpolation | `extrapolate` (bool) |

#### Shock Methods
| Method | Description | Required Keys | Optional Keys |
|--------|-------------|--------------|---------------|
| **`normal`** | Normal distribution | `mean`, `std` | `n_points`, `width` |
| **`lognormal`** | Log-normal distribution | `mean`, `std` | `n_points`, `width` |
| **`discretemarkov`** | Markov process | `rho`, `sigma` | `method` (tauchen/rouwenhorst), `n_points` |
| **`uniform`** | Uniform distribution | `min`, `max` | `n_points` |
| **`discrete`** | Discrete distribution | `values`, `probs` | None |

#### Integration Methods
| Method | Description | Allowed Keys |
|--------|-------------|--------------|
| **`quadrature`** | Gaussian quadrature | `order` (int) |
| **`montecarlo`** | Monte Carlo integration | `n_samples` (int) |
| **`discretize`** | Discrete approximation | `n_points` (int) |

#### Optimization Methods
| Method | Description | Allowed Keys |
|--------|-------------|--------------|
| **`EGM`** | Endogenous Grid Method | `monotone` (bool), `interp` (str) |
| **`VFI`** | Value Function Iteration | `brute_force` (bool), `grid_search` (bool) |
| **`PFI`** | Policy Function Iteration | `newton` (bool), `max_iter` (int) |

## Model Structure

Heptapod-B implements a two-level hierarchy: the stage model and its substages (movers).

### Two-Level Hierarchy: Stage and Substages

The model hierarchy consists of two levels:

```
FunctionalProblem (Stage Model)
└── Substages (Movers)
    ├── arvl_to_dcsn (Mover Model)
    ├── dcsn_to_cntn (Mover Model)
    ├── cntn_to_dcsn (Mover Model)
    └── dcsn_to_arvl (Mover Model)
```

The stage and each mover are `FunctionalProblem` instances with:
- `.math` - Symbolic representations
- `.num` - Numerical representations
- `.parameters` - Economic parameters
- `.settings` - Computational settings
- `.methods` - Solution methods

### Substage Types

The substages (movers) are specialized by function:

1. **Forward Movers**: Transform distributions
   - `arvl_to_dcsn`: Arrival to Decision (shock realization)
   - `dcsn_to_cntn`: Decision to Continuation (asset transition)

2. **Backward Movers**: Implement factored Bellman operations
   - `cntn_to_dcsn`: Continuation to Decision (optimization)
   - `dcsn_to_arvl`: Decision to Arrival (expectation)

### Perches as Special Components

Perches in StageCraft connect to specialized substages containing only state-space information:

- **Arrival Perch**: Connected to arrival state-space substage
- **Decision Perch**: Connected to decision state-space substage
- **Continuation Perch**: Connected to continuation state-space substage

The `perch.py` module creates these state-space-only substages from the YAML configuration.

### YAML Representation

The YAML configuration reflects the two-level hierarchy:

```yaml
# Stage-level parameters and functions
parameters:
  beta: 0.96
  gamma: 2.0

functions:
  utility:
    inputs: [c]
    expr: "c**(1-gamma)/(1-gamma)"

# State spaces (used by perches)
state_space:
  arrival:
    dimensions: [assets]
  decision:
    dimensions: [assets, income]
  continuation:
    dimensions: [assets]

# State definitions
state:
  assets:
    grid:
      type: linspace
      min: 0.0
      max: 100.0
      points: 100
  income:
    grid:
      type: custom
      values: [0.8, 1.0, 1.2]

# Substages (movers)
substages:
  arvl_to_dcsn:
    functions:
      shock_realization:
        inputs: [assets, shock]
        expr: "assets * (1 + shock)"
        
  dcsn_to_cntn:
    functions:
      asset_transition:
        inputs: [assets, consumption]
        expr: "assets - consumption"
        
  cntn_to_dcsn:
    methods:
      optimization: [egm]
      
  dcsn_to_arvl:
    methods:
      expectation: [quadrature]
```

## Integration with StageCraft

Heptapod-B integrates with StageCraft by providing model representations for stages, perches, and movers.

### Model Connection to Perches and Movers

The connection follows the two-level hierarchy:

1. The stage FunctionalProblem attaches to `stage.model`
2. The substage FunctionalProblems attach to movers:
   - `stage.arvl_to_dcsn.model`: Arrival to Decision mover model
   - `stage.dcsn_to_cntn.model`: Decision to Continuation mover model
   - `stage.cntn_to_dcsn.model`: Continuation to Decision mover model
   - `stage.dcsn_to_arvl.model`: Decision to Arrival mover model

3. Perch models are specialized state-space substages:
   - `stage.arvl.model`: Arrival perch model (state space)
   - `stage.dcsn.model`: Decision perch model (state space)
   - `stage.cntn.model`: Continuation perch model (state space)

```python
# Initialize stage with Heptapod-B model
stage = Stage()
stage.model = build_stage("lifecycle_model.yaml")

# Compile numerical representation
compile_num(stage.model)

# Models attached to movers (substages)
assert stage.arvl_to_dcsn.model is not None
assert stage.dcsn_to_cntn.model is not None
assert stage.cntn_to_dcsn.model is not None
assert stage.dcsn_to_arvl.model is not None

# State space models attached to perches
assert stage.arvl.model is not None
assert stage.dcsn.model is not None
assert stage.cntn.model is not None
```

### Whisperer Integration

The Heptapod-B whisperer connects stages to specialized solvers:

```python
# Create a whisperer with the appropriate solver
from heptapod_b.solvers import EGMWhisperer
whisperer = EGMWhisperer()

# Attach the whisperer to the stage
stage.attach_whisperer(whisperer)

# Solve the stage
stage.solve_backward()
```

The whisperer:
- Extracts model information from `model.num`
- Creates operators for each mover
- Attaches operators to `mover.comp`
- Orchestrates the solution process

## Key Components

### Function Compilation

Functions can be compiled through multiple methods:

| Method | Description | Use Case |
|--------|-------------|----------|
| **eval** | Direct Python evaluation | Simple functions |
| **sympy** | Symbolic mathematics | Derivatives, analytical operations |
| **numba** | Just-in-time compilation | Computation-intensive algorithms |

```yaml
functions:
  utility:
    inputs: [c]
    expr: "c**(1-gamma)/(1-gamma)"
    compilation: "eval"  # Alternative: "sympy" or "numba"
```

Multi-output functions:

```yaml
util_and_mutil:           # Rᶰ→R²
  util:   "c**(1-gamma)/(1-gamma)"
  mutil:  "c**(-gamma)"
  compilation: "eval"
```

### Grid Generation

Grids define numerical approximation of continuous state spaces:

```yaml
state:
  assets:
    grid:
      type: linspace  # Alternatives: geomspace, logspace, etc.
      min: 0.0
      max: 100.0
      points: 100
```

Grid types:

| Grid Type | Description | Common Use |
|-----------|-------------|------------|
| **linspace** | Uniform grid | General purpose, consumption |
| **geomspace** | Geometric spacing | Assets, wealth |
| **custom** | User-defined points | Special cases, adaptive grids |

### State Space Construction

State spaces combine grids into computational domains:

```yaml
state_space:
  decision:
    dimensions:
      - assets
      - income_shock
    grid:
      create_mesh: true  # Create full tensor product
```

For multi-dimensional spaces, Heptapod-B:
- Creates tensor product meshes when needed
- Maps between index spaces and value spaces
- Supports mixed continuous and discrete dimensions
- Handles vectorized operations

> **Performance Note**: For high-dimensional state spaces, Heptapod-B provides automatic dimensional reduction methods that avoid the curse of dimensionality when possible. 

## Vectorization Principles

Heptapod-B is an embarrassingly vectorized framework built on NumPy's vectorization capabilities.

### Automatic Vectorization

All functions in Heptapod-B are automatically vectorized:

1. **Point-wise Operations**: Functions written as scalar operations are automatically applied element-wise when given array inputs
2. **No Explicit Loops**: Operations on arrays don't require explicit loops
3. **Broadcasting**: NumPy's broadcasting rules are preserved, allowing operations between arrays of different but compatible shapes

```python
# A simple function defined in YAML
# utility:
#   expr: "c**(1-gamma)/(1-gamma)"

# Automatically vectorized when called with array input
c_values = np.linspace(0.1, 10.0, 100)  # Array of consumption values
u_values = model.num.functions.utility(c_values)  # Returns array of utility values
```

### Vector Semantics

Heptapod-B maintains consistent vector semantics throughout the model:

1. **All Operations Preserve Vectorization**: Every stage of computation maintains vector semantics
2. **Grid Operations**: Grid values are stored as NumPy arrays and operations on grids maintain array structure
3. **Expectation Operations**: Integration over shocks is performed in vectorized form
4. **Interpolation**: Evaluations at off-grid points use vectorized interpolation through SciPy's N-dimensional interpolators

```yaml
# Function that automatically preserves vector semantics
transition_budget:
  m:     "y + a*r"        # Vectorized income calculation
  a_nxt: "m - c"          # Vectorized asset transition
  value: "u + beta*cntn"  # Vectorized value calculation
```

### Performance Considerations

Vectorization provides significant performance benefits:

1. **Memory Locality**: Vectorized operations benefit from CPU cache efficiency
2. **SIMD Instructions**: Modern CPUs can process multiple data points in a single instruction
3. **Lazy Evaluation**: Many NumPy operations use lazy evaluation, avoiding unnecessary computations
4. **C-Level Performance**: Most NumPy operations execute at optimized C speed

Heptapod-B doesn't "out-smart" NumPy—it organizes interpolation, integration, and bilinear operations while preserving NumPy's vectorization efficiency. This approach enables writing clean mathematical expressions that automatically handle arrays of any dimension.

```python
# Every algebraic expression works on potentially arbitrary vectors
# No need for manual looping over state spaces

# Example: Evaluating a multi-output function on a grid
grid_points = model.num.state_space.decision.grid  # Mesh grid
results = model.num.functions.asset_transition_value(grid_points)

# Results dictionary contains vectorized outputs for each defined function
# results = {
#   'm': array([...]),     # Income values at each grid point
#   'a_nxt': array([...]), # Next period assets at each grid point
#   'util': array([...]),  # Utility values at each grid point
#   'mutil': array([...])  # Marginal utility values at each grid point
# }
``` 