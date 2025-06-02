# Reference Model Deviation Metrics

This guide explains how to use DynX's reference model deviation metrics to compare fast solver methods against high-accuracy reference solutions.

## Overview

The reference model deviation metrics feature allows you to:
- Solve a high-accuracy reference model once
- Compare multiple fast solver methods against the reference
- Automatically compute L2 and L∞ norms of policy differences
- Organize bundles by method for easy comparison

## Quick Start

```python
from dynx.runner import CircuitRunner
from dynx.runner.metrics import dev_c_L2, dev_c_Linf

# Configure runner with method awareness
runner = CircuitRunner(
    base_cfg=cfg,
    param_paths=["beta", "sigma", "master.methods.upper_envelope"],
    model_factory=make_model,
    solver=solve,
    metric_fns={
        "dev_c_L2": dev_c_L2,      # L2 norm of consumption policy deviation
        "dev_c_Linf": dev_c_Linf,  # L∞ norm of consumption policy deviation
    },
    output_root="results/",
    save_by_default=True,
    method_param_path="master.methods.upper_envelope"
)
```

## Workflow

### 1. Solve Reference Model

First, solve your model with a high-accuracy method (typically `VFI_HDGRID`):

```bash
python solve.py \
    --ue-method VFI_HDGRID \
    --output-root results_HR \
    --save-bundles
```

This creates: `results_HR/bundles/<hash>/VFI_HDGRID/`

### 2. Compare Fast Methods

Then solve with fast methods and compute deviations:

```bash
python solve.py \
    --ue-method "FUES,CONSAV,DCEGM" \
    --output-root results_HR \
    --save-bundles \
    --metric dev_c_L2,dev_c_Linf
```

## Bundle Organization

With `method_param_path` set, bundles are organized by method:

```
results_HR/
└── bundles/
    └── <param_hash>/          # Same hash for same parameters
        ├── VFI_HDGRID/        # Reference method
        ├── FUES/              # Fast method 1
        ├── CONSAV/            # Fast method 2
        └── DCEGM/             # Fast method 3
```

## Available Metrics

The following deviation metrics are provided:

| Metric | Description |
|--------|-------------|
| `dev_c_L2` | L2 norm of consumption policy deviation |
| `dev_c_Linf` | L∞ norm of consumption policy deviation |
| `dev_a_L2` | L2 norm of asset policy deviation |
| `dev_a_Linf` | L∞ norm of asset policy deviation |
| `dev_v_L2` | L2 norm of value function deviation |
| `dev_v_Linf` | L∞ norm of value function deviation |
| `dev_pol_L2` | L2 norm of general policy deviation |
| `dev_pol_Linf` | L∞ norm of general policy deviation |

## Custom Deviation Metrics

You can create custom deviation metrics using the factory function:

```python
from dynx.runner.metrics.deviations import make_policy_dev_metric

# Create a custom metric for labor supply deviation
dev_labor_L2 = make_policy_dev_metric("labor", "L2")
dev_labor_Linf = make_policy_dev_metric("labor", "Linf")

# For policies in specific stages or solution attributes
dev_q_L2 = make_policy_dev_metric("Q", "L2", stage="RNTC", sol_attr="value")

# For policies that are plain arrays (not nested in dictionaries)
dev_wealth_L2 = make_policy_dev_metric("", "L2", stage="SAVE", sol_attr="wealth")
```

## Method-less Projects

If your project doesn't use a method parameter, set `method_param_path=None`:

```python
runner = CircuitRunner(
    ...,
    method_param_path=None  # Disables method-aware organization
)
```

This will use the original flat bundle structure and deviation metrics will return `NaN`.

## Configuration

The default reference method is `VFI_HDGRID`. You can change it by modifying:

```python
from dynx.runner import reference_utils
reference_utils.DEFAULT_REF_METHOD = "YOUR_REFERENCE_METHOD"
```

## Example Results

After running the workflow, you might see results like:

| master.methods.upper_envelope | dev_c_L2 | dev_c_Linf | euler_error |
|------------------------------|----------|------------|-------------|
| VFI_HDGRID                   | 0.000000 | 0.000000   | 1.2e-10     |
| FUES                         | 0.015634 | 0.118750   | 9.4e-04     |
| CONSAV                       | 0.021112 | 0.146228   | 1.5e-03     |
| DCEGM                        | 0.013507 | 0.097881   | 8.9e-04     |

## Notes

- Reference models must be solved and saved before computing deviations
- Deviation metrics return `NaN` if no reference is found
- The parameter hash excludes the method parameter to ensure consistency
- All existing metric functions remain backward compatible 