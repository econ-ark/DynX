# I/O Examples

This directory contains examples demonstrating the I/O capabilities of the DynX framework, particularly focused on configuration loading and model saving/loading.

## Examples

### folder_config_example.py

Demonstrates the new folder-based configuration loading functionality introduced in v0.1.8.dev3:

- **load_config**: Load configurations from a structured directory
- **Config modification**: Programmatically modify loaded configurations
- **Model building**: Initialize a ModelCircuit from loaded configs
- **Save/Load**: Save and load model circuits with attached solutions

### solution_rehydration_example.py

Demonstrates the automatic Solution object re-hydration feature introduced in v0.1.8.dev4:

- **Solution behavior**: Shows how Solution objects convert to/from dictionaries
- **Re-hydration process**: Explains the automatic detection and conversion
- **Benefits**: Ensures compatibility with downstream code expecting Solution attributes
- **Error prevention**: Prevents `AttributeError` when accessing `.EGM.unrefined` etc.

### branch_dict_example.py

Demonstrates branch dictionary handling for stages with multiple solution branches (e.g., TENU):

- **Branch solutions**: Shows how to create dictionaries mapping branch keys to Solution objects
- **Serialization problem**: Explains why numba typed.Dict objects cause pickling failures
- **Smart handling**: How the io module detects and properly serializes branch dictionaries
- **Usage patterns**: How to access branch-specific solutions after loading

## Expected Config Directory Structure

The `load_config` function expects the following directory structure:

```
config_dir/
├── master.yml         # Master configuration
├── stages/            # Stage configurations
│   ├── stage1.yml     # Both .yml and .yaml extensions are supported
│   ├── stage2.yaml
│   └── ...
└── connections.yml    # Connection definitions
```

**Note**: Stage configuration files in the `stages/` directory can use either `.yml` or `.yaml` extensions.

## Usage

```python
from dynx import load_config, initialize_model_Circuit

# Load configurations
cfg = load_config("path/to/config_dir")

# Modify if needed
cfg["master"]["periods"] = 5

# Build model
model = initialize_model_Circuit(
    master_config=cfg["master"],
    stage_configs=cfg["stages"],
    connections_config=cfg["connections"]
)
```

## See Also

- [StageCraft Documentation](../../dynx/stagecraft/README.md)
- [Saver Module Tests](../../tests/saver/test_saver.py)