# dynx_runner

A package for parameter sweeping and optimization of economic models.

## Features

- **Parameter Sweeping**: Define parameter spaces and efficiently compute model solutions across them.
- **Multi-Metric Support**: Collect arbitrary metrics during model execution, with the only constraint being picklability.
- **Telemetry**: Track metrics during model execution using the `RunRecorder` class.
- **Parallel Execution**: Use MPI for parallel parameter sweeping across multiple nodes.
- **Caching**: Avoid redundant computations by caching model results.
- **Visualization**: Built-in plotting functions for common visualization tasks.

## Usage

### Basic Usage

```python
from dynx_runner import CircuitRunner, mpi_map

# Create a CircuitRunner with your configuration
runner = CircuitRunner(
    epochs_cfgs=your_epochs_configs,
    stage_cfgs=your_stage_configs,
    conn_cfg=your_connection_config,
    param_specs={
        "main.parameters.beta": (0.8, 0.99, lambda n: np.random.uniform(0.8, 0.99, n)),
        "main.parameters.delta": (0.02, 0.1, lambda n: np.random.uniform(0.02, 0.1, n)),
    },
    model_factory=your_model_factory,
    solver=your_solver_function,
    metric_fns={"log_likelihood": lambda m: m.log_likelihood()},
)

# Sample parameters and run the model
xs = runner.sample_prior(n=100)
results = mpi_map(runner, xs, mpi=True)

# Plot results
from dynx_runner import plot_bars
import matplotlib.pyplot as plt
fig = plot_bars(results, metric="log_likelihood")
plt.show()
```

### Using the RunRecorder for Telemetry

The new `RunRecorder` class allows solvers and simulators to add metrics during model execution:

```python
from dynx_runner import RunRecorder

def my_solver(model, recorder=None):
    # Do solver work...
    iterations = 15
    time_taken = 0.5
    
    # Record metrics if recorder is provided
    if recorder is not None:
        recorder.add(
            solver_iterations=iterations,
            solver_time=time_taken,
            solver_converged=True,
            detailed_errors={"state1": 0.001, "state2": 0.005}  # Can be complex objects
        )
```

The `CircuitRunner` will automatically pass a `RunRecorder` to your solver and simulator functions if they accept a `recorder` parameter.

### Multi-Metric Support

The `CircuitRunner.run()` method now returns a dictionary of arbitrary metrics, which can include complex objects:

```python
# Run model with parameter vector
x = np.array([0.95, 0.05])
metrics = runner.run(x)

# Access scalar metrics
print(f"Log likelihood: {metrics['log_likelihood']}")
print(f"Solver iterations: {metrics['solver_iterations']}")

# Access complex metrics
print(f"Detailed errors: {metrics['detailed_errors']}")
```

When using `mpi_map`, only scalar metrics (int, float, bool, str) are allowed in the resulting DataFrame. The function will raise a `TypeError` if non-scalar metrics are encountered.

## API Reference

### CircuitRunner

Main class for running economic models with parameter sweeping capabilities.

### RunRecorder

Records metrics during model execution. Provides a simple interface for solvers and simulators to add metrics.

### mpi_map

Apply the CircuitRunner to each parameter vector using MPI parallelization. 