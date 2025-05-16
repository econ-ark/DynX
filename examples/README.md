# ModCraft Examples

This directory contains example configuration files, applications, and usage patterns for ModCraft.

## Directory Structure

- `heptapod_b/` - Examples demonstrating Heptapod-B package features
  - Parameter resolution, movers, perches, shocks, and basic usage

- `economic_models/` - Examples of economic models implemented using ModCraft
  - Housing models, consumption-income shock models, retirement choice models

- `configs/` - Example YAML configuration files
  - Manual grid specification, parameter resolution, model configuration

- `workflows/` - Examples demonstrating recommended workflows
  - Stage workflow examples and patterns

- `circuitCraft/` - Examples specific to CircuitCraft functionality

## Getting Started

For newcomers to ModCraft, we recommend starting with:

1. First, explore the `heptapod_b/load_consind_multi_b.py` example to see how to load a model
2. Next, review the YAML configuration examples in the `configs/` directory
3. Then, explore specific features in the `heptapod_b/` directory
4. Finally, check out complete economic models in the `economic_models/` directory

## Parameter Resolution in Heptapod-B (v1.6+)

Heptapod-B features a powerful parameter reference system that enables reuse of values across different parts of your model. Understanding how parameter references work is crucial for building maintainable models.

### Reference Syntax

References can be specified in two ways:

1. **List-style reference**: `["param_name"]` - Recommended for clarity
2. **Direct string reference**: `param_name` - Works but may cause issues with recursive resolution

### Parameter Inheritance

Parameters follow a hierarchical structure:

```
Stage (Global) Parameters
       │
       ├── Stage Settings
       │
       ├── State Space Settings
       │
       └── Mover Parameters (inherited or explicit)
```

#### Stage to Mover Inheritance

When `inherit_parameters: true` is set on a mover, all parameters from the stage model are copied to the mover:

```yaml
movers:
  my_mover:
    # ...
    inherit_parameters: true
    inherit_settings: true
```

Each mover gets its own independent copy of parameters, which can be modified without affecting other movers.

### Resolution Timing

Parameter references are resolved during **compilation** (not during initialization):

1. Stage model is built with `build_stage()` 
2. Mover models are built with `build_mover()`
3. References are fully resolved during `compile_num()`

This allows you to modify parameters between building and compiling if needed.

### Resolution Environment

Each component resolves references in its own environment:

1. **Stage model**: Uses its own parameters and settings
2. **Mover models**: Use their inherited copies of parameters
3. **Each mover resolves independently**: Changes to one mover's parameters only affect that mover

### Examples

#### Global Parameter Reference in Grid Specification

```yaml
stage:
  settings:
    a_min: 0.01
    a_max: 10.0
  
  math:
    state_space:
      assets:
        dimensions: ["a"]
        grid:
          a:
            type: "linspace"
            min: a_min       # Direct reference to state-level setting
            max: a_max       # Direct reference to state-level setting
            points: 20
```

#### Proper Global References in State Settings

```yaml
stage:
  settings:
    a_min: 0.01
  
  math:
    state_space:
      assets:
        settings:
          my_min: ["a_min"]  # List-style reference to global setting
```

#### Shock Parameters with Global References

```yaml
stage:
  parameters:
    mean: 1.0
    std: 0.1
  
  math:
    shocks:
      income_shock:
        settings:
          mean: ["mean"]     # References global parameter
          std: ["std"]       # References global parameter
```

### Best Practices

1. **Use list-style references** `["param_name"]` for clarity and to avoid circular references
2. **Keep parameter names unique** across different scopes to avoid confusion
3. **Modify parameters before compilation** if you need different values than those specified in YAML
4. **Use stage parameters** for values shared across multiple components
5. **Use state-specific settings** for values unique to a particular state

### Debugging Tips

If you encounter parameter resolution issues:

1. Check for circular references (e.g., `a_min: a_min`)
2. Verify parameter names match exactly (references are case-sensitive)
3. Use list-style references `["param_name"]` for global parameters
4. Examine compilation warnings that indicate unresolved references

For practical examples demonstrating parameter resolution:
- `heptapod_b/test_mover_parameter_resolution.py` - Python script demonstrating how mover-specific parameters affect shock compilation
- `configs/ConsInd_multi.yml` - Shows parameter references in a full model

## Manual Grid Specification (v1.5.19+)

ModCraft v1.5.19 introduces manual grid specification in YAML configurations, giving users precise control over state space grids. This feature allows explicit definition of grid points rather than relying solely on algorithmic generation.

### Features

#### 1. Flat List Grid

Directly specify all grid points for a single dimension as a flat list:

```yaml
state_space:
  assets:
    dimensions: ["assets"]
    grid: [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
```

#### 2. Dimension Mapping

For multi-dimensional grids, map dimension names to their respective grid points:

```yaml
state_space:
  assets_income:
    dimensions: ["assets", "income"]
    grid:
      assets: [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
      income: [-0.5, -0.2, 0.0, 0.2, 0.5]
```

#### 3. Integer Range

Specify integer grids using start, stop, and step parameters:

```yaml
state_space:
  periods:
    dimensions: ["period"]
    grid:
      start: 0
      stop: 10
      step: 1
```

#### 4. Mixed Specifications

Combine different grid specification types in a single state space:

```yaml
state_space:
  mixed_grid:
    dimensions: ["assets", "age"]
    grid:
      assets: [0.0, 0.1, 0.5, 1.0, 5.0, 10.0]  # Explicit list
      age:                                      # Integer range
        start: 20
        stop: 65
        step: 5
```

#### 5. Mesh Creation Control

Control whether to create a mesh grid for tensor product spaces:

```yaml
state_space:
  no_mesh:
    dimensions: ["x", "y"]
    grid:
      x: [0.0, 1.0, 2.0]
      y: [0.0, 0.5, 1.0]
    methods:
      create_mesh: false
```

### Grid Type Aliases

ModCraft supports several aliases for algorithmic grid generation:

| Alias | Description | Required Parameters |
|-------|-------------|---------------------|
| `uniform` | Uniform grid (alias for `linspace`) | `min`, `max`, `n` |
| `log` | Logarithmic grid | `min`, `max`, `n` |
| `chebyshev` | Chebyshev nodes for better interpolation | `min`, `max`, `n` |
| `integer` | Integer range | `start`, `stop`, `step` |
| `categorical` | Enumeration for categorical variables | `values` |

Example of algorithmic grid specification:

```yaml
state_space:
  algorithmic:
    dimensions: ["wealth"]
    methods:
      grid:
        type: chebyshev
        min: 0.1
        max: 50.0
        n: 15
```

For a comprehensive example demonstrating all grid specification features, see the `configs/manual_grid_example.yml` 