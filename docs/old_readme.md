

**StageCraft**

Specialized structure using Dyn-X, StageCraft module is a modular directed acyclic graph (DAG) based framework for computing recursive optimization problems in economics. 

At its core is a three-node Eulerian circuit called a **stage**, which encapsulates the smallest independent, non-trivial processing of information in a stochastic dynamic recursive model.

This design is particularly useful for economic models since functional objects, as part of a graph relationship, are directly related to simulation and other analyses (e.g., comparative statics).

**Heptapod-B**

Model representation for vector operations using NumpPy syntax. 

> This package is preliminary and under active develepment.

## Package Structure

### Core Stage Components

- **Perch (`perch.py`)**: The basic node type that stores functional objects (solution and distributions).
- **Mover (`mover.py`)**: Edge objects that define computational transitions between perches.
- **Stage (`stage.py`)**: The fundamental modeling unit based on a CDA (Continuation-Decision-Arrival) structure of perches.

### Higher-Level Graph Structures

- **Period (`period.py`)**: A collection of interconnected stages representing dynamics within a single time period.
- **ModelCircuit (`model_circuit.py`)**: A sequence of periods that forms a complete economic model with inter-period connections.
- **CircuitBoard (`circuit_board.py`)**: Base class for graph-based computational elements.

###  Model Representation

Model representation—the mathematical form of each mover and stage, its numerical components, parameters, and methods—is handled by a plug-in called `heptapod`.

The `heptapod` package provides the mathematical representation class, which specifies:
- Mathematical representation
    - State spaces
    - Transition functions
    - Constraints
    - Shocks
- Parameters
- Numerical settings 
- Methods 
- Numerical implementations
    - Transition functions
    - Constraints
    - State space (grids)
    - Shocks 

The separation between computational structure (`stagecraft`) and model representation (`heptapod`) allows for maximum flexibility in model development and solution.

While the Stage orchestrates the relationships between functional operators and functional objects, users have the flexibility to represent their models in any way they want.

## Repository Structure

The development repository is organized as follows:

```
StageCraft/
├── src/                     # Source code for the core packages
│   ├── stagecraft/          # Main computational framework
│   └── heptapod/            # Model representation plugin
├── examples/                # Example implementations and demonstrations
│   ├── mortgage_mon/        # Monetary policy and mortgage model
│   ├── ret_choice/          # Worker-retiree choice model
│   ├── cons_indshock/       # Consumption model with income shocks
│   ├── stage_workflow/      # Examples of stage workflow patterns
│   └── circuitCraft/        # Basic framework demonstrations
├── docs/                    # Documentation
│   ├── concepts.md          # Core concepts and definitions
│   ├── theory.md            # Mathematical theory
│   ├── stage.md             # Stage design documentation
│   └── *.png, *.tex         # Diagrams and visualizations
├── project-notes/           # Development planning and documentation
│   ├── Todo.md              # Current development tasks and roadmap
│   └── prompts/             # Development notes and AI prompts
├── tests/                   # Unit and integration tests
└── models/                  # Model implementations
```

### Key Directories

- **src/**: Contains the core framework code, including the main `stagecraft` package and the `heptapod` model representation plugin.
- **examples/**: Provides practical implementations and demonstrations of the framework, including economic models and workflow patterns.
- **docs/**: Includes comprehensive documentation of the framework's theoretical foundations and implementation details.
- **project-notes/**: Contains development planning materials and tasks for ongoing development.
- **tests/**: Houses unit and integration tests for the framework.
- **models/**: Reserved for full model implementations built with the framework.


## Notes 

### Model representation config.yml files

The current architecture for model representation is as follows is rudimentary and needs to be improved. 
## Parameter and State Space Inheritance Architecture

The current system for managing state spaces, parameters, and variable inheritance works as follows:

### State Space and Perch Initialization
- State space specifications create the info to create perch models 
- Perches automatically inherit parameters and settings from their parent stage during initialization
- Perch state space creation accesses model parameters and settings via direct references

### Parameter Inheritance Flow
- Mover models inherit model parameters and settings when `inherit = true` is specified

### Shock Variable Declaration
- Shock variables must be explicitly declared using bracket notation: `["var_name"]`
- This syntax indicates a reference rather than a literal value

### Equation Access to Parameters
- Equations should access settings and parameters from their relevant context:
  - Stage-level equations use stage parameters
  - Mover-level equations use mover parameters
- Equations are not created at the perch level
- Parameter access should match the scope of the equation being defined

# dynx_runner

A lightweight package for parameter sweeping and optimization of economic models.

## Overview

`dynx_runner` provides a framework for running economic models with different parameter sets, collecting metrics, and visualizing results. It is designed to work with any model that follows a simple interface pattern, making it easy to integrate with existing codebases.

Key features:
- Parameter sweeping with custom sampling strategies
- Caching of results to avoid redundant computations
- Parallel execution using MPI
- Visualization tools for analyzing results

## Installation

```bash
# Basic installation
pip install .

# With MPI support
pip install ".[mpi]"

# With test dependencies
pip install ".[test]"
```

## Usage Example

```python
import numpy as np
import matplotlib.pyplot as plt
from dynx_runner import CircuitRunner, plot_bars

# Define configuration dictionaries for your model
epochs_cfgs = {"main": {"parameters": {"beta": 0.95}}}
stage_cfgs = {"OWNH": {"parameters": {"r": 0.03}}}
conn_cfg = {}

# Define model factory and solver functions
def model_factory(epochs_cfgs, stage_cfgs, conn_cfg):
    # Create and return your model
    ...

def solver(model):
    # Solve the model
    ...

# Create a CircuitRunner
runner = CircuitRunner(
    epochs_cfgs=epochs_cfgs,
    stage_cfgs=stage_cfgs,
    conn_cfg=conn_cfg,
    param_specs={
        "main.parameters.beta": (0.8, 0.99, lambda n: np.random.uniform(0.8, 0.99, n)),
        "OWNH.parameters.r": (0.01, 0.05, lambda n: np.random.uniform(0.01, 0.05, n)),
    },
    model_factory=model_factory,
    solver=solver,
    metric_fns={
        "LL": lambda m: -m.log_likelihood(),
        "EulerError": lambda m: m.euler_error_max(),
    },
)

# Sample from prior and run models
xs = runner.sample_prior(10)
results = []

for x in xs:
    metrics = runner.run(x)
    params = runner.unpack(x)
    results.append({**params, **metrics})

# Create DataFrame for analysis
import pandas as pd
df = pd.DataFrame(results)

# Visualize results
fig, ax = plt.subplots(figsize=(10, 6))
plot_bars(df, "LL", ax=ax)
plt.show()

# Using MPI for parallel execution
from dynx_runner import mpi_map
df = mpi_map(runner, xs)
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/runner/test_pack_unpack.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.