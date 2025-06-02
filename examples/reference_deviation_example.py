"""
Example demonstrating reference model deviation metrics in DynX.

This example shows how to:
1. Configure a runner with method awareness
2. Solve a reference model
3. Compare fast methods against the reference
4. Use custom deviation metrics
"""

import numpy as np
import pandas as pd
from dynx.runner import CircuitRunner
from dynx.runner.metrics import dev_c_L2, dev_c_Linf, make_policy_dev_metric


# Mock model for demonstration
class MockModel:
    """Simple model with method-dependent policies."""
    
    def __init__(self, config):
        self.beta = config.get("beta", 0.96)
        self.sigma = config.get("sigma", 2.0)
        self.method = config["master"]["methods"]["upper_envelope"]
        
        # Mock policy that varies by method
        if self.method == "VFI_HDGRID":
            self.c = np.ones(100) * 0.7  # High accuracy reference
        elif self.method == "FUES":
            self.c = np.ones(100) * 0.71  # Slightly different
        elif self.method == "CONSAV":
            self.c = np.ones(100) * 0.715  # More different
        else:
            self.c = np.ones(100) * 0.72  # Most different


def mock_factory(config):
    """Factory to create mock models."""
    return MockModel(config)


def mock_solver(model, recorder=None):
    """Mock solver (model is already 'solved' in __init__)."""
    pass


def main():
    """Run the example."""
    # Base configuration
    base_cfg = {
        "beta": 0.96,
        "sigma": 2.0,
        "master": {
            "methods": {
                "upper_envelope": "VFI_HDGRID"
            }
        }
    }
    
    # Configure runner with method awareness
    runner = CircuitRunner(
        base_cfg=base_cfg,
        param_paths=["beta", "sigma", "master.methods.upper_envelope"],
        model_factory=mock_factory,
        solver=mock_solver,
        metric_fns={
            "dev_c_L2": dev_c_L2,
            "dev_c_Linf": dev_c_Linf,
        },
        output_root="./results_demo/",
        save_by_default=True,
        method_param_path="master.methods.upper_envelope"
    )
    
    # First, solve the reference model
    print("Solving reference model...")
    x_ref = runner.pack({
        "beta": 0.96,
        "sigma": 2.0,
        "master.methods.upper_envelope": "VFI_HDGRID"
    })
    
    metrics_ref = runner.run(x_ref)
    print(f"Reference metrics: {metrics_ref}")
    print()
    
    # Now compare fast methods
    print("Comparing fast methods...")
    methods = ["VFI_HDGRID", "FUES", "CONSAV", "DCEGM"]
    results = []
    
    for method in methods:
        x = runner.pack({
            "beta": 0.96,
            "sigma": 2.0,
            "master.methods.upper_envelope": method
        })
        metrics = runner.run(x)
        results.append({
            "method": method,
            "dev_c_L2": metrics["dev_c_L2"],
            "dev_c_Linf": metrics["dev_c_Linf"]
        })
    
    # Display results
    df_results = pd.DataFrame(results)
    print("\nDeviation Results:")
    print(df_results)
    print()
    
    # Show bundle organization
    print("Bundle paths:")
    for method in methods:
        x = runner.pack({
            "beta": 0.96,
            "sigma": 2.0,
            "master.methods.upper_envelope": method
        })
        bundle_path = runner._bundle_path(x)
        if bundle_path:
            rel_path = bundle_path.relative_to(runner.output_root)
            print(f"  {method}: {rel_path}")


if __name__ == "__main__":
    main() 