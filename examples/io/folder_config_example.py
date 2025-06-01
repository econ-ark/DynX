#!/usr/bin/env python
"""
Example demonstrating the new folder-based configuration loading functionality.

This example shows how to:
1. Use load_config to load configurations from a structured directory
2. Modify loaded configurations programmatically
3. Initialize a model circuit with the loaded configs
4. Save and load model circuits with the new folder-based approach
"""

from pathlib import Path
from dynx import load_config, initialize_model_Circuit, save_circuit, load_circuit
from dynx.stagecraft.makemod import compile_all_stages

def demonstrate_load_config():
    """Demonstrate loading configurations from a folder structure."""
    print("=== Folder-Based Config Loading Example ===\n")
    
    # Assume we have a config directory structured as:
    # examples/config_HR/
    # ├── master.yml
    # ├── stages/
    # │   ├── stage1.yml
    # │   └── stage2.yml
    # └── connections.yml
    
    config_dir = Path("examples/config_HR")
    
    if not config_dir.exists():
        print(f"Config directory '{config_dir}' not found.")
        print("Please create a config directory with the expected structure.")
        return None
    
    # Load all configurations at once
    print(f"Loading configurations from: {config_dir}")
    cfg = load_config(config_dir)
    
    print(f"\nLoaded configuration keys: {list(cfg.keys())}")
    print(f"Master config name: {cfg['master'].get('name', 'Unknown')}")
    print(f"Number of stages: {len(cfg['stages'])}")
    print(f"Stage names: {list(cfg['stages'].keys())}")
    
    return cfg

def demonstrate_config_modification(cfg):
    """Show how to modify loaded configurations before building the model."""
    if cfg is None:
        return
    
    print("\n=== Modifying Loaded Configurations ===\n")
    
    # Example: Change the number of periods
    original_periods = cfg["master"].get("periods", 1)
    cfg["master"]["periods"] = 3
    print(f"Changed periods from {original_periods} to {cfg['master']['periods']}")
    
    # Example: Add a parameter to all stages
    for stage_name, stage_config in cfg["stages"].items():
        if "parameters" not in stage_config:
            stage_config["parameters"] = {}
        stage_config["parameters"]["custom_param"] = 0.99
        print(f"Added custom_param to stage '{stage_name}'")
    
    return cfg

def demonstrate_model_building(cfg):
    """Build a model circuit from loaded configurations."""
    if cfg is None:
        return None
    
    print("\n=== Building Model Circuit ===\n")
    
    # Initialize the model circuit
    mc = initialize_model_Circuit(
        master_config=cfg["master"],
        stage_configs=cfg["stages"],
        connections_config=cfg["connections"]
    )
    
    print(f"Model circuit created: {mc.name}")
    print(f"Number of periods: {len(mc.periods_list)}")
    
    # Example: Compile all stages (if numerical representation is available)
    try:
        compile_all_stages(mc)
        print("All stages compiled successfully")
    except:
        print("Stage compilation skipped (numerical representation not available)")
    
    return mc

def demonstrate_save_load(mc, tmp_dir):
    """Demonstrate saving and loading a model circuit."""
    if mc is None:
        return
    
    print("\n=== Saving and Loading Model Circuit ===\n")
    
    # Create a temporary directory for saving
    save_dir = Path(tmp_dir) / "saved_models"
    save_dir.mkdir(exist_ok=True)
    
    # Save the circuit
    print(f"Saving model circuit to: {save_dir}")
    saved_path = save_circuit(
        circuit=mc,
        dest=save_dir,
        config_src=Path("examples/config_HR"),  # Original config source
        model_id="example_model_v1"
    )
    print(f"Model saved to: {saved_path}")
    
    # Load the circuit back
    print("\nLoading model circuit from saved directory...")
    loaded_mc = load_circuit(saved_path)
    print(f"Loaded model: {loaded_mc.name}")
    print(f"Number of periods: {len(loaded_mc.periods_list)}")
    
    # Load without restoring data
    print("\nLoading without data restoration...")
    loaded_mc_no_data = load_circuit(saved_path, restore_data=False)
    print(f"Loaded model (no data): {loaded_mc_no_data.name}")

def main():
    """Run all demonstrations."""
    # Load configurations
    cfg = demonstrate_load_config()
    
    # Modify configurations
    cfg = demonstrate_config_modification(cfg)
    
    # Build model circuit
    mc = demonstrate_model_building(cfg)
    
    # Save and load
    demonstrate_save_load(mc, "/tmp")
    
    print("\n=== Example Complete ===")

if __name__ == "__main__":
    main() 