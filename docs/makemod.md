# makemod Module

The `makemod` module provides functionality to build StageCraft model circuits directly from YAML configuration files, enabling a declarative approach to defining complex economic models.

## Overview

The makemod system automates the construction of multi-period economic models from configuration files, handling:
- Stage creation and initialization
- Period assembly
- Intra-period connections (within stages)
- Inter-period connections (between periods)
- Model visualization

## Core Functions

### `initialize_model_Circuit(master_config, stage_configs, connections_config)`

Builds a complete `ModelCircuit` from configuration dictionaries.

**Parameters:**
- `master_config`: Master configuration dictionary containing global parameters
- `stage_configs`: Dictionary of stage configurations (key=stage_name)
- `connections_config`: Dictionary defining all connections (edges) in the model

**Returns:**
- Fully configured `ModelCircuit` object

### `create_stage(stage_name, stage_config, master_config)`

Creates a single `Stage` object from configuration.

**Parameters:**
- `stage_name`: Name identifier for the stage
- `stage_config`: Stage-specific configuration dictionary
- `master_config`: Master configuration for parameter inheritance

**Returns:**
- Initialized `Stage` object

### `compile_all_stages(model, force=False, **gen_kwargs)`

Compiles (numerically generates) every Stage in an already-constructed ModelCircuit.

**Parameters:**
- `model`: The ModelCircuit to compile
- `force`: If True, recompile even if already compiled
- `**gen_kwargs`: Extra arguments for numerical generation

### `visualize_model(model, output_dir, prefix="", save_svg=False)`

Generates multiple visualizations of the model structure.

**Parameters:**
- `model`: The `ModelCircuit` to visualize
- `output_dir`: Directory to save visualization files
- `prefix`: Optional prefix for output filenames
- `save_svg`: Whether to also save SVG versions

## Configuration Structure

### Master Configuration

```yaml
name: MyModel
horizon: 10
parameters:
  beta: 0.95
  sigma: 2.0
imports:
  - alias: worker_stage
    stage_name: worker
  - alias: retiree_stage
    stage_name: retiree
```

### Stage Configuration

```yaml
name: worker
parameters:
  wage: 25000
state_space:
  assets:
    type: linspace
    min: 0
    max: 100000
    n: 50
```

### Connections Configuration

```yaml
intra_period:
  0:  # Period index
    forward:
      - source: arvl
        target: dcsn
    backward:
      - source: dcsn
        target: cntn

inter_period:
  - source_period: 0
    target_period: 1
    source: worker_stage
    target: worker_stage
    source_perch: cntn
    target_perch: arvl
```

## Model Building Phases

The makemod follows a structured 4-phase approach:

1. **Phase 0 - Validation**: Validates all input configurations
2. **Phase 1 - Period Creation**: Creates all periods with populated stages
3. **Phase 2 - Intra-period Edges**: Establishes connections within periods
4. **Phase 3 - Period Registration**: Registers periods with the model
5. **Phase 4 - Inter-period Edges**: Creates connections between periods

## Usage Example

```python
from dynx.stagecraft.makemod import initialize_model_Circuit, visualize_model, compile_all_stages
import yaml

# Load configurations
with open('master.yml') as f:
    master_cfg = yaml.safe_load(f)
    
stage_cfgs = {}
for stage_file in ['worker_stage.yml', 'retiree_stage.yml']:
    with open(stage_file) as f:
        name = stage_file.replace('_stage.yml', '')
        stage_cfgs[name] = yaml.safe_load(f)
        
with open('connections.yml') as f:
    conn_cfg = yaml.safe_load(f)

# Build the model
model = initialize_model_Circuit(master_cfg, stage_cfgs, conn_cfg)

# Compile all stages
compile_all_stages(model)

# Visualize the model
visualize_model(model, output_dir='./visualizations')
```

## Integration with StageCraft

The makemod integrates seamlessly with the broader StageCraft framework:
- Uses `Stage` objects from `dynx.stagecraft.stage`
- Creates `Period` containers from `dynx.stagecraft.period`
- Produces `ModelCircuit` instances from `dynx.stagecraft.model_circuit`
- Leverages Heptapod-X's `initialize_model` for model initialization

## Error Handling

The module provides comprehensive error handling:
- `LoaderError`: Top-level errors during model loading
- `ConfigKeyError`: Missing required configuration keys
- Detailed logging throughout the build process

## Advanced Features

- **Dynamic Period Creation**: Only creates periods that are referenced in connections
- **Flexible Connection Formats**: Supports both dictionary and list formats for connections
- **Branch Key Support**: Handles conditional flows with branch keys
- **Transpose Connections**: Automatically creates backward connections when specified

## Migration from config_loader

If you're migrating from the old `config_loader` module:
1. Update imports: `from dynx.stagecraft.config_loader import ...` → `from dynx.stagecraft.makemod import ...`
2. All function names remain the same
3. All functionality is preserved 