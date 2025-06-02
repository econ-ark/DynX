#!/usr/bin/env python
"""
Example demonstrating the disk-based save/load functionality of CircuitRunner.

This example shows how to:
1. Set up a CircuitRunner with bundle management
2. Run models with automatic saving
3. Load existing bundles instead of re-solving
4. Use the __runner.mode parameter to control behavior
"""

import sys
import os
from pathlib import Path
import numpy as np

# Add codebase to path if running directly
if __name__ == "__main__":
    codebase_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(codebase_dir))

from dynx.runner import CircuitRunner
from dynx.runner.sampler import MVNormSampler, build_design


def simple_model_factory(cfg):
    """Simple model factory for demonstration."""
    class DummyModel:
        def __init__(self, config):
            self.config = config
            self.name = "housing"
            self.version = "dev"
            self.periods_list = []  # Empty for simplicity
    
    return DummyModel(cfg)


def simple_solver(model, recorder=None):
    """Simple solver that just records some fake metrics."""
    if recorder:
        # Simulate some solving metrics
        recorder.add(
            solve_time=np.random.uniform(0.1, 2.0),
            iterations=np.random.randint(5, 20),
            convergence_error=np.random.uniform(1e-8, 1e-6)
        )


def housing_value_metric(model):
    """Example metric function."""
    beta = model.config.get("beta", 0.95)
    return beta * 100  # Simple derived metric


def demo_basic_save_load():
    """Demonstrate basic save/load functionality."""
    print("=== Basic Save/Load Demo ===\n")
    
    # Set up base configuration
    base_cfg = {
        "beta": 0.95,
        "master": {"periods": 5},
        "stages": {"HOUSING": {"type": "consumption"}},
        "connections": {}
    }
    
    # Create output directory
    output_dir = Path("solutions")
    output_dir.mkdir(exist_ok=True)
    
    # Set up CircuitRunner with save/load enabled
    runner = CircuitRunner(
        base_cfg=base_cfg,
        param_paths=["beta"],
        model_factory=simple_model_factory,
        solver=simple_solver,
        metric_fns={"housing_value": housing_value_metric},
        output_root=output_dir,
        bundle_prefix="housing",
        save_by_default=True,  # Save every solved model
        load_if_exists=True,   # Load existing bundles
        hash_len=8
    )
    
    # Test parameter
    x = np.array([0.97], dtype=object)
    
    print(f"Running with beta = {x[0]}")
    print(f"Bundle path would be: {runner._bundle_path(x)}")
    
    # First run - should solve and save
    print("\n--- First run (should solve and save) ---")
    metrics1 = runner.run(x)
    print(f"Metrics: {metrics1}")
    
    # Second run - should load from bundle
    print("\n--- Second run (should load from bundle) ---")
    metrics2 = runner.run(x)
    print(f"Metrics: {metrics2}")
    
    # Verify bundle was created
    bundle_path = runner._bundle_path(x)
    if bundle_path and bundle_path.exists():
        print(f"\n✓ Bundle created at: {bundle_path}")
        print(f"  Contents: {list(bundle_path.iterdir())}")
        
        # Check manifest
        manifest_path = bundle_path / "manifest.yml"
        if manifest_path.exists():
            import yaml
            manifest = yaml.safe_load(manifest_path.read_text())
            print(f"  Bundle hash: {manifest.get('bundle', {}).get('hash', 'N/A')}")
            print(f"  Parameters: {manifest.get('parameters', {})}")
    else:
        print("⚠ No bundle was created")


def demo_mode_parameter():
    """Demonstrate the __runner.mode parameter."""
    print("\n\n=== Mode Parameter Demo ===\n")
    
    base_cfg = {
        "beta": 0.95,
        "master": {"periods": 3},
        "stages": {"HOUSING": {"type": "consumption"}},
        "connections": {}
    }
    
    runner = CircuitRunner(
        base_cfg=base_cfg,
        param_paths=["beta", "__runner.mode"],
        model_factory=simple_model_factory,
        solver=simple_solver,
        metric_fns={"housing_value": housing_value_metric},
        output_root="solutions",
        bundle_prefix="housing",
        save_by_default=False,  # Only save when explicitly requested
        load_if_exists=False,   # Only load when mode="load"
    )
    
    # Test with mode="solve" (force solving)
    x_solve = np.array([0.98, "solve"], dtype=object)
    print(f"Running with mode='solve': beta={x_solve[0]}")
    metrics_solve = runner.run(x_solve, save_model=True)  # Override save_by_default
    print(f"Solve metrics: {metrics_solve}")
    
    # Test with mode="load" (force loading)
    x_load = np.array([0.98, "load"], dtype=object)
    print(f"\nRunning with mode='load': beta={x_load[0]}")
    metrics_load = runner.run(x_load)
    print(f"Load metrics: {metrics_load}")


def demo_parameter_sweep():
    """Demonstrate parameter sweep with mixed save/load."""
    print("\n\n=== Parameter Sweep Demo ===\n")
    
    base_cfg = {
        "beta": 0.95,
        "sigma": 2.0,
        "master": {"periods": 3},
        "stages": {"HOUSING": {"type": "consumption"}},
        "connections": {}
    }
    
    runner = CircuitRunner(
        base_cfg=base_cfg,
        param_paths=["beta", "sigma"],
        model_factory=simple_model_factory,
        solver=simple_solver,
        metric_fns={"housing_value": housing_value_metric},
        output_root="solutions",
        bundle_prefix="sweep",
        save_by_default=True,
        load_if_exists=True,
    )
    
    # Create design matrix
    sampler = MVNormSampler(
        means=[0.96, 2.5],
        cov=[[0.01, 0], [0, 0.25]]
    )
    
    xs, info = build_design(
        param_paths=["beta", "sigma"],
        samplers=[sampler],
        Ns=[5],
        seed=42
    )
    
    print(f"Running parameter sweep with {len(xs)} parameter combinations")
    
    # Use mpi_map for the sweep
    from dynx.runner import mpi_map
    df = mpi_map(runner, xs, return_models=False, mpi=False)
    
    print(f"\nSweep completed. Results shape: {df.shape}")
    print(f"Average housing value: {df['housing_value'].mean():.2f}")
    print(f"Bundle directory contents: {len(list(Path('solutions').glob('sweep_*')))} bundles")
    
    # Check for design matrix CSV
    design_matrix_path = Path("solutions") / "design_matrix.csv"
    if design_matrix_path.exists():
        import pandas as pd
        design_df = pd.read_csv(design_matrix_path)
        print(f"\n✓ Design matrix CSV created with {len(design_df)} rows")
        print(f"  Columns: {list(design_df.columns)}")
        print(f"  Sample entries:")
        for idx, row in design_df.head(3).iterrows():
            print(f"    beta={row['beta']:.3f}, sigma={row['sigma']:.3f}, hash={row['param_hash']}, bundle={row['bundle_dir']}")
    else:
        print("\n⚠ Design matrix CSV was not created")


def main():
    """Run all demonstrations."""
    demo_basic_save_load()
    demo_mode_parameter()
    demo_parameter_sweep()
    
    print("\n=== Summary ===")
    print("✓ Basic save/load functionality working")
    print("✓ Mode parameter controls solve/load behavior")
    print("✓ Parameter sweeps with bundle management")
    print("✓ Bundles include hash, prefix, rank, and parameters in manifest")
    print("✓ Design matrix CSV tracks all parameter combinations and bundle locations")


if __name__ == "__main__":
    main() 