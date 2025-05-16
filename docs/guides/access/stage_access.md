# Stage Access Path Cheat-Sheet

This table lists the core dotted attribute paths used to access components within a compiled `Stage` object. All paths should be accessed after calling `stage.build_computational_model()`.

## Stage Components

| Path | Access Example | Object Type | Description |
|------|----------------|-------------|-------------|
| `stage.model` | `model = stage.model` | FunctionalProblem | The main model representation |
| `stage.model.math` | `math = stage.model.math` | AttrDict | Mathematical representations |
| `stage.model.math.functions` | `functions = stage.model.math.functions` | dict | Mathematical functions |
| `stage.model.math.state_space` | `state_space = stage.model.math.state_space` | dict | State space definitions |
| `stage.model.parameters_dict` | `params = stage.model.parameters_dict` | dict | Model parameters dictionary |
| `stage.model.param` | `param_value = stage.model.param.some_param` | _ParameterAccessor | Property-based access to parameters |
| `stage.model.settings_dict` | `settings = stage.model.settings_dict` | dict | Model settings dictionary |
| `stage.model.settings` | `setting_value = stage.model.settings.some_setting` | _SettingsAccessor | Property-based access to settings |
| `stage.model.methods` | `methods = stage.model.methods` | dict | Model methods dictionary |
| `stage.model.num` | `num = stage.model.num` | AttrDict | Numerical objects |
| `stage.model.num.state_space` | `ss = stage.model.num.state_space` | AttrDict | Numerical state spaces |

## Perch Access

| Path | Access Example | Object Type | Description |
|------|----------------|-------------|-------------|
| `stage.arvl` | `arvl = stage.arvl` | Perch | Arrival perch |
| `stage.arvl.name` | `name = stage.arvl.name` | str | Perch name ("arvl") |
| `stage.arvl.model` | `model = stage.arvl.model` | FunctionalProblem | Model attached to perch |
| `stage.arvl.model.math` | `math = stage.arvl.model.math` | AttrDict | Math representation |
| `stage.arvl.model.parameters_dict` | `params = stage.arvl.model.parameters_dict` | dict | Perch parameters dictionary |
| `stage.arvl.model.param` | `param_value = stage.arvl.model.param.some_param` | _ParameterAccessor | Property-based access to parameters |
| `stage.arvl.model.settings_dict` | `settings = stage.arvl.model.settings_dict` | dict | Perch settings dictionary |
| `stage.arvl.model.settings` | `setting_value = stage.arvl.model.settings.some_setting` | _SettingsAccessor | Property-based access to settings |
| `stage.arvl.model.methods_dict` | `methods_dict = stage.arvl.model.methods_dict` | dict | Perch methods dictionary |
| `stage.arvl.model.methods` | `methods = stage.arvl.model.methods` | dict | Perch methods |
| `stage.arvl.model.num` | `num = stage.arvl.model.num` | AttrDict | Numerical objects |
| `stage.arvl.model.num.state_space` | `ss = stage.arvl.model.num.state_space` | AttrDict | State space |
| `stage.arvl.model.num.state_space.arvl` | `arvl_ss = stage.arvl.model.num.state_space.arvl` | AttrDict | Arrival state space |
| `stage.arvl.model.num.state_space.arvl.grids` | `grids = stage.arvl.model.num.state_space.arvl.grids` | AttrDict | Grid dictionary |
| `stage.arvl.model.num.state_space.arvl.grids.a` | `a_grid = stage.arvl.model.num.state_space.arvl.grids.a` | ndarray | Asset grid |
| `stage.arvl.model.num.state_space.arvl.grids.age` | `age_grid = stage.arvl.model.num.state_space.arvl.grids.age` | ndarray | Age grid |
| `stage.arvl.model.num.state_space.arvl.mesh` | `mesh = stage.arvl.model.num.state_space.arvl.mesh` | dict | Mesh grid dictionary |
| `stage.arvl.model.num.state_space.arvl.mesh.a` | `a_mesh = stage.arvl.model.num.state_space.arvl.mesh.a` | ndarray | Asset mesh grid (flattened) |
| `stage.arvl.grid.a` | `a_grid = stage.arvl.grid.a` | ndarray | Asset grid via proxy |
| `stage.arvl.grid.age` | `age_grid = stage.arvl.grid.age` | ndarray | Age grid via proxy |
| `stage.arvl.mesh.a` | `a_mesh = stage.arvl.mesh.a` | ndarray | Asset mesh grid via proxy |
| `stage.dcsn` | `dcsn = stage.dcsn` | Perch | Decision perch |
| `stage.dcsn.model` | `model = stage.dcsn.model` | FunctionalProblem | Model attached to perch |
| `stage.dcsn.model.num.state_space.dcsn.grids.m` | `m_grid = stage.dcsn.model.num.state_space.dcsn.grids.m` | ndarray | Cash-on-hand grid |
| `stage.dcsn.grid.m` | `m_grid = stage.dcsn.grid.m` | ndarray | Cash-on-hand grid via proxy |
| `stage.cntn` | `cntn = stage.cntn` | Perch | Continuation perch |
| `stage.cntn.model` | `model = stage.cntn.model` | FunctionalProblem | Model attached to perch |
| `stage.cntn.model.num.state_space.cntn.grids.a_nxt` | `a_nxt = stage.cntn.model.num.state_space.cntn.grids.a_nxt` | ndarray | Next-period asset grid |
| `stage.cntn.grid.a_nxt` | `a_nxt = stage.cntn.grid.a_nxt` | ndarray | Next-period asset grid via proxy |

## Mover Access

| Path | Access Example | Object Type | Description |
|------|----------------|-------------|-------------|
| `stage.arvl_to_dcsn` | `mover = stage.arvl_to_dcsn` | Mover | Arrival to decision mover |
| `stage.arvl_to_dcsn.source_name` | `source = stage.arvl_to_dcsn.source_name` | str | Source perch name ("arvl") |
| `stage.arvl_to_dcsn.target_name` | `target = stage.arvl_to_dcsn.target_name` | str | Target perch name ("dcsn") |
| `stage.arvl_to_dcsn.edge_type` | `edge_type = stage.arvl_to_dcsn.edge_type` | str | Edge type ("forward") |
| `stage.arvl_to_dcsn.model` | `model = stage.arvl_to_dcsn.model` | FunctionalProblem | Model attached to mover |
| `stage.arvl_to_dcsn.model.math` | `math = stage.arvl_to_dcsn.model.math` | AttrDict | Math representation |
| `stage.arvl_to_dcsn.model.parameters_dict` | `params = stage.arvl_to_dcsn.model.parameters_dict` | dict | Mover parameters dictionary |
| `stage.arvl_to_dcsn.model.param` | `param_value = stage.arvl_to_dcsn.model.param.some_param` | _ParameterAccessor | Property-based access to parameters |
| `stage.arvl_to_dcsn.model.settings_dict` | `settings = stage.arvl_to_dcsn.model.settings_dict` | dict | Mover settings dictionary |
| `stage.arvl_to_dcsn.model.settings` | `setting_value = stage.arvl_to_dcsn.model.settings.some_setting` | _SettingsAccessor | Property-based access to settings |
| `stage.arvl_to_dcsn.model.methods_dict` | `methods_dict = stage.arvl_to_dcsn.model.methods_dict` | dict | Mover methods dictionary |
| `stage.arvl_to_dcsn.model.methods` | `methods = stage.arvl_to_dcsn.model.methods` | dict | Mover methods |
| `stage.dcsn_to_cntn` | `mover = stage.dcsn_to_cntn` | Mover | Decision to continuation mover |
| `stage.dcsn_to_cntn.model` | `model = stage.dcsn_to_cntn.model` | FunctionalProblem | Model attached to mover |
| `stage.cntn_to_dcsn` | `mover = stage.cntn_to_dcsn` | Mover | Continuation to decision mover |
| `stage.cntn_to_dcsn.model` | `model = stage.cntn_to_dcsn.model` | FunctionalProblem | Model attached to mover |
| `stage.dcsn_to_arvl` | `mover = stage.dcsn_to_arvl` | Mover | Decision to arrival mover |
| `stage.dcsn_to_arvl.model` | `model = stage.dcsn_to_arvl.model` | FunctionalProblem | Model attached to mover |

## Stage Status Flags

| Path | Access Example | Object Type | Description |
|------|----------------|-------------|-------------|
| `stage.status_flags` | `flags = stage.status_flags` | dict | Dictionary of all status flags |
| `stage.status_flags["initialized"]` | `init = stage.status_flags["initialized"]` | bool | Stage is initialized |
| `stage.status_flags["compiled"]` | `compiled = stage.status_flags["compiled"]` | bool | Stage is compiled |
| `stage.status_flags["solvable"]` | `solvable = stage.status_flags["solvable"]` | bool | Stage has required data for solving |
| `stage.status_flags["solved"]` | `solved = stage.status_flags["solved"]` | bool | Stage has been solved backward |
| `stage.status_flags["simulated"]` | `simulated = stage.status_flags["simulated"]` | bool | Stage has been simulated forward |
| `stage.status_flags["portable"]` | `portable = stage.status_flags["portable"]` | bool | Stage can be serialized |
| `stage.status_flags["all_perches_initialized"]` | `perches_init = stage.status_flags["all_perches_initialized"]` | bool | All perches have been initialized |
| `stage.status_flags["all_perches_compiled"]` | `perches_comp = stage.status_flags["all_perches_compiled"]` | bool | All perches have been compiled |
| `stage.status_flags["all_movers_initialized"]` | `movers_init = stage.status_flags["all_movers_initialized"]` | bool | All movers have been initialized |
| `stage.status_flags["all_movers_compiled"]` | `movers_comp = stage.status_flags["all_movers_compiled"]` | bool | All movers have been compiled |

## Alternative Access Methods

### Direct Dictionary Access

In addition to attribute-style access, many components can be accessed using dictionary-style notation:

| Path | Access Example | Object Type | Description |
|------|----------------|-------------|-------------|
| `stage.arvl.model.num.state_space.arvl['grids']` | `grids = stage.arvl.model.num.state_space.arvl['grids']` | dict | Grid dictionary via dict access |
| `stage.arvl.model.num.state_space.arvl['grids']['a']` | `a_grid = stage.arvl.model.num.state_space.arvl['grids']['a']` | ndarray | Asset grid via nested dict access |
| `stage.arvl.model.num.state_space.arvl['mesh']` | `mesh = stage.arvl.model.num.state_space.arvl['mesh']` | dict | Mesh grid dictionary via dict access |
| `stage.arvl.model.num.state_space.arvl['mesh']['a']` | `a_mesh = stage.arvl.model.num.state_space.arvl['mesh']['a']` | ndarray | Asset mesh grid via nested dict access |

### Mesh Grid Access and Creation

There are three ways to work with mesh grids:

1. **Direct access to pre-computed mesh grids:**
```python
# Access pre-computed mesh directly from the model
a_mesh = stage.arvl.model.num.state_space.arvl.mesh.a

# Access pre-computed mesh via mesh proxy
a_mesh = stage.arvl.mesh.a
```

2. **Creating mesh grids from individual grids:**
```python
# Get 1D grids
a_grid = stage.arvl.grid.a
age_grid = stage.arvl.grid.age

# Create mesh grids
import numpy as np
a_mesh, age_mesh = np.meshgrid(a_grid, age_grid, indexing='ij')

# Access mesh grid values
a_mesh[0, 0]  # First asset value at first age
age_mesh[0, 0]  # First age value at first asset
```

### Using Mesh Grids for State Space Calculations

Working with functions defined over the state space:

```python
# Create a value function defined over the state space
value_function = np.zeros((len(a_grid), len(age_grid)))
for i in range(len(a_grid)):
    for j in range(len(age_grid)):
        value_function[i, j] = a_grid[i] * 100 - age_grid[j]  # Example calculation

# Access values at specific state space points
young_poor_value = value_function[0, 0]  # Value for youngest age, lowest assets
old_rich_value = value_function[-1, -1]  # Value for oldest age, highest assets

# Example of getting value at a specific state point
idx_a = len(a_grid) // 2  # Middle of asset grid
idx_age = len(age_grid) // 2  # Middle of age grid
mid_value = value_function[idx_a, idx_age]
```

Note: The `.grid` attribute is automatically added by the Stage's `_attach_grid_proxies()` method during `build_computational_model()` and provides direct access to grids without needing to navigate the full path through `state_space`. Similarly, the `.mesh` attribute is added if mesh grids exist, providing direct access to mesh grids. 