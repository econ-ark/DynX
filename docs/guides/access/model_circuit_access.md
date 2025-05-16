# Model Circuit Access Cheatsheet

This document provides a quick reference for accessing components in the Dyn-X framework.

## Model Circuit Structure

```python
# Access a model circuit
model_circuit = initialize_model_Circuit(master_config, stage_configs, connections_config)

# Get number of periods
num_periods = len(model_circuit.periods_list)

# Access a specific period (two equivalent methods)
period = model_circuit.periods_list[period_idx]
period = model_circuit.get_period(period_idx)

# Access a specific stage in a period (two equivalent methods)
stage = period.stages["STAGE_NAME"]
stage = period.get_stage("STAGE_NAME")

# Check if all stages are compiled
all_compiled = model_circuit.all_stages_compiled()
```

## Model Circuit Access

| Path | Access Example | Object Type | Description |
|------|----------------|-------------|-------------|
| `model_circuit.name` | `name = model_circuit.name` | str | Name of the model circuit |
| `model_circuit.periods_list` | `periods = model_circuit.periods_list` | list | List of all periods in the circuit |
| `model_circuit.periods_list[i]` | `period = model_circuit.periods_list[0]` | Period | Access specific period by index |
| `model_circuit.get_period(i)` | `period = model_circuit.get_period(0)` | Period | Alternative way to access a period |
| `model_circuit.all_stages_compiled()` | `compiled = model_circuit.all_stages_compiled()` | bool | Check if all stages are compiled |
| `model_circuit.solve_all_stages()` | `model_circuit.solve_all_stages()` | method | Solve all stages in the circuit |
| `model_circuit.simulate_all_stages()` | `model_circuit.simulate_all_stages()` | method | Simulate all stages in the circuit |
| `model_circuit.inter_period_movers` | `movers = model_circuit.inter_period_movers` | dict | Dictionary of movers between periods |

## Period Access

| Path | Access Example | Object Type | Description |
|------|----------------|-------------|-------------|
| `period.stages` | `stages = period.stages` | dict | Dictionary of all stages in the period |
| `period.stages[stage_name]` | `stage = period.stages["OWNH"]` | Stage | Access specific stage by name |
| `period.get_stage(stage_name)` | `stage = period.get_stage("OWNH")` | Stage | Alternative way to access a stage |
| `period.connections` | `connections = period.connections` | list | List of all connections in the period |
| `period.update_connections()` | `period.update_connections()` | method | Update all connections in the period |
| `period.solve_stages(order)` | `period.solve_stages(["OWNH", "OWNC"])` | method | Solve stages in specified order |
| `period.simulate_stages(order)` | `period.simulate_stages(["OWNH", "OWNC"])` | method | Simulate stages in specified order |

## Stage Access

```python
# Direct access to standard perches via properties
arvl_perch = stage.arvl
dcsn_perch = stage.dcsn
cntn_perch = stage.cntn

# Access any perch by name
perch = stage.perches["perch_name"]

# Direct access to standard movers via properties
arvl_to_dcsn = stage.arvl_to_dcsn
dcsn_to_cntn = stage.dcsn_to_cntn
cntn_to_dcsn = stage.cntn_to_dcsn
dcsn_to_arvl = stage.dcsn_to_arvl

# Access any mover by source/target
mover = stage._get_mover("source_perch", "target_perch", "forward")

# Build computational model
stage.build_computational_model()

# Check status flags
is_compiled = stage.status_flags.get("compiled", False)
is_initialized = stage.status_flags.get("initialized", False)
```

## Model Attribute Access

```python
# Get the model associated with a stage
model = stage.model

# Access parameters (two equivalent ways)
param_value = model.parameters_dict["param_name"]  # Dictionary access
param_value = model.param.param_name              # Attribute-style access

# Access settings (two equivalent ways)
setting_value = model.settings_dict["setting_name"]  # Dictionary access
setting_value = model.settings.setting_name         # Attribute-style access

# Access methods
method_value = model.methods["method_name"]
```

## Numerical Component Access

### State Space and Grids

```python
# Direct path to state space
state_space = model.num.state_space

# Direct access to perch state space 
perch_state_space = model.num.state_space["perch_name"]

# Direct access to grid variables
grid_var = model.num.state_space["perch_name"]["grids"]["var_name"]

# Access via grid proxy (preferred, more concise)
grid_var = stage.perch_name.grid.var_name

# Example: Access asset grid for the arrival perch
asset_grid = stage.arvl.grid.a
```

### Mesh Grids

```python
# Direct access to mesh data
mesh_var = model.num.state_space["perch_name"]["mesh"]["var_name"]

# Access via mesh proxy (if available)
mesh_var = stage.perch_name.mesh.var_name

# Example: Access asset mesh for the arrival perch
asset_mesh = stage.arvl.mesh.a
```

### Shocks

```python
# Access shock definitions
shocks = model.num.shocks

# Access a specific shock
shock = model.num.shocks["shock_name"]

# Access shock components
shock_values = model.num.shocks["shock_name"]["values"]
shock_probs = model.num.shocks["shock_name"]["probs"]
transition_matrix = model.num.shocks["shock_name"]["transition_matrix"]  # If applicable
```

## Solution/Distribution Access

```python
# Access solution data on a perch
solution = perch.sol

# Access a specific solution component
component = perch.sol["component_name"]

# Access distribution data on a perch
distribution = perch.dist

# Example: Access value function for arrival perch
vlu_arvl = stage.arvl.sol["vlu_arvl"]

# Example: Access policy function for decision perch
policy = stage.dcsn.sol["policy"]
```

## Common Tasks

```python
# Initialize a housing model
configs = load_configs()
stage_configs = {
    "OWNH": configs["ownh"],
    "OWNC": configs["ownc"]
}
model_circuit = initialize_model_Circuit(
    master_config=configs["master"],
    stage_configs=stage_configs,
    connections_config=configs["connections"]
)

# Set numerical representation for stages
for stage_name in model_circuit.periods_list[0].stages:
    model_circuit.periods_list[0].stages[stage_name].num_rep = generate_numerical_model

# Build computational models for all stages
for period in model_circuit.periods_list:
    for stage_name, stage in period.stages.items():
        stage.build_computational_model()
        
# Alternative: Use compile_all_stages utility
from src.stagecraft.config_loader import compile_all_stages
compile_all_stages(model=model_circuit)
```

## Multi-Period Connections

| Path | Access Example | Object Type | Description |
|------|----------------|-------------|-------------|
| `model_circuit.inter_period_movers` | `movers = model_circuit.inter_period_movers` | dict | Dictionary of all inter-period movers |
| `model_circuit.inter_period_movers["mover_name"]` | `mover = model_circuit.inter_period_movers["OWNC_0_to_OWNH_1"]` | Mover | Access specific inter-period mover |
| `model_circuit.add_inter_period_connection()` | `model_circuit.add_inter_period_connection(period1, "stage1", period2, "stage2", type="forward")` | method | Add a new connection between periods |

## Model Circuit Navigation Examples

```python
# Iterating through all periods and stages
for period_idx, period in enumerate(model_circuit.periods_list):
    print(f"Period {period_idx}:")
    for stage_name, stage in period.stages.items():
        print(f"  Stage: {stage_name}, Compiled: {stage.status_flags.get('compiled', False)}")
        
# Accessing a model in a specific stage
model = model_circuit.periods_list[0].stages["OWNH"].model

# Accessing a specific grid in a specific stage
asset_grid = model_circuit.periods_list[0].stages["OWNH"].arvl.grid.a

# Accessing a solution component in a specific stage
if model_circuit.periods_list[0].stages["OWNC"].dcsn.sol:
    policy = model_circuit.periods_list[0].stages["OWNC"].dcsn.sol.get("policy")
```
