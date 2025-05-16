# StageCraft Configuration Loader

The `config_loader` module provides functionality to build StageCraft model circuits directly from YAML configuration files, enabling a declarative approach to defining complex economic models.

## Overview

This module allows you to:

1. Define stages, their properties, and state spaces in YAML files
2. Specify a master configuration with shared parameters and settings
3. Connect stages within periods and across periods
4. Generate multi-period models with automated connection handling
5. Visualize the resulting model circuit

## Configuration Structure

### Master Configuration File

The master configuration file defines the overall model structure and imports individual stage files:

```yaml
name: "ModelName"
description: "Description of the model"
version: "1.0.0"

# Global parameters shared across all stages
parameters:
  beta: 0.95            # Discount factor
  r: 1.04               # Interest rate
  # ... more parameters

# Global settings
settings:
  tol: 1.0e-6           # Convergence tolerance
  max_iter: 1000        # Maximum iterations
  # ... more settings
  
# Import individual stage configurations
imports:
  - file: "stage1.yml"
    stage_name: "Stage1"
    alias: "ST1"
  
  - file: "stage2.yml"
    stage_name: "Stage2"
    alias: "ST2"

# Stage connections
connections:
  # Intra-period connections
  - source: "ST1"       # Source stage
    target: "ST2"       # Target stage
    mapping:            # State variable mapping
      x: "x_input"
      y: "y_input"
  
  # Inter-period connections
  - source: "ST2"       # Source stage
    target: "ST1"       # Target stage
    mapping:
      x_next: "x"
      y_next: "y"
    period_offset: 1    # Indicates inter-period connection
```

### Stage Configuration File

Each stage is defined in its own YAML file:

```yaml
# Reference to master configuration file (optional)
master_file: "master.yml"

# Stage definition
stage:
  name: "StageName"
  is_portable: true
  method: "EGM"         # or "discrete_choice", etc.
  kind: "sequential"    # or "branching"

  # Stage-specific parameters (can reference master parameters)
  parameters:
    beta: ["beta"]      # Reference to master parameter
    gamma: 2.0          # Stage-specific parameter
  
  # Stage-specific settings
  settings:
    tol: ["tol"]        # Reference to master setting
    max_iter: 100       # Stage-specific setting
  
  # Math section defining functions and state spaces
  math:
    functions:
      # Function definitions
      u_func:
        expr: "log(c)"
        description: "Utility function"
    
    state_space:
      # Define perches (arrival, decision, continuation, etc.)
      arvl:
        description: "Arrival state space"
        dimensions: ["x", "y", "z"]
      
      dcsn:
        description: "Decision state space"
        dimensions: ["x", "y", "z"]
      
      cntn:
        description: "Continuation state space"
        dimensions: ["x_next", "y_next", "z_next"]

# Movers section defining transitions between perches
movers:
  # Arrival to Decision
  arvl_to_dcsn:
    type: "forward"
    source: "arvl"
    target: "dcsn"
    functions:
      - identity_mapping
    operator:
      method: simulation
    description: "Arrival->Decision transition"
  
  # Decision to Continuation
  dcsn_to_cntn:
    type: "forward"
    source: "dcsn"
    target: "cntn"
    functions:
      - budget_constraint
    operator:
      method: optimization
      objective: u_func
    description: "Decision->Continuation with optimization"
    required_variables:
      - c
  
  # Continuation to Decision (backward step)
  cntn_to_dcsn:
    type: "backward"
    source: "cntn"
    target: "dcsn"
    functions:
      - egm_operations
    operator:
      method: EGM
    description: "Continuation->Decision backward step"
```

## Parameter References

You can reference parameters from a master file using the following syntax:

```yaml
parameters:
  beta: ["beta"]        # Reference to master parameter
```

## Using the Config Loader

### Building a Model

```python
from src.stagecraft.config_loader import build_model_from_configs, visualize_model

# Build a model with 3 periods
master_config_path = "path/to/master.yml"
model = build_model_from_configs(master_config_path, periods=3)

# Create visualizations
visualize_model(model, output_dir="path/to/images", prefix="model_")
```

### Individual Functions

The module provides several functions for more granular control:

- `load_yaml_config(config_path)`: Load a YAML configuration file
- `load_stage_config(stage_path, master_config)`: Load a stage configuration with optional master config
- `create_stage_from_config(stage_config, stage_name)`: Create a Stage object from configuration
- `build_model_from_configs(master_config_path, periods)`: Build a complete model circuit
- `visualize_model(model, output_dir, prefix)`: Generate visualizations of the model

## Example

See `examples/economic_models/housing/load_housing_model.py` for a complete example of loading and visualizing a housing rental choice model from configuration files.

## Best Practices

1. Define shared parameters in the master file for consistency
2. Use meaningful aliases for stages in the master file
3. Keep stage configurations focused on stage-specific details
4. Use the reference syntax `["parameter_name"]` to inherit from master
5. Include detailed descriptions for functions, state spaces, and movers 