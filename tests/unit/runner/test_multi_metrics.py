"""
Tests for metric collection functionality in CircuitRunner.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from dynx.runner.circuit_runner import CircuitRunner, mpi_map
from dynx.runner.telemetry import RunRecorder


def mock_model_factory(cfg):
    """Mock model factory for testing."""
    
    class MockModel:
        def __init__(self):
            self.cfg = cfg
            self.iterations = 100
            self.energy = -5.0
            self.error = 0.01
            self.convergence = 0.001
            
        def log_likelihood(self):
            """For testing metrics"""
            return -1000.0
    
    return MockModel()


def mock_solver_with_recorder(model, recorder=None):
    """Mock solver that records metrics if a recorder is provided."""
    if recorder is not None:
        recorder.add(
            iterations=model.iterations,
            energy=model.energy,
            period_timings={"p0": 0.1, "p1": 0.2},
            stage_timings=[0.1, 0.2, 0.3]
        )


def mock_simulator_with_recorder(model, recorder=None):
    """Mock simulator that records metrics if a recorder is provided."""
    if recorder is not None:
        recorder.add(
            error=model.error,
            convergence=model.convergence
        )


def test_run_recorder_basics():
    """Test basic functionality of the RunRecorder class."""
    # Create recorder
    recorder = RunRecorder()
    
    # Add metrics
    recorder.add(simple=1.0, complex={"a": 1, "b": 2})
    
    # Check metrics
    assert recorder.metrics["simple"] == 1.0
    assert recorder.metrics["complex"] == {"a": 1, "b": 2}
    
    # Metrics should be a copy - modifying the original shouldn't affect the copy
    recorder.metrics["simple"] = 2.0
    new_recorder = RunRecorder()
    new_recorder.add(simple=3.0)
    
    assert recorder.metrics["simple"] == 2.0
    assert new_recorder.metrics["simple"] == 3.0


def test_circuit_runner_with_recorder():
    """Test CircuitRunner integration with RunRecorder."""
    runner = CircuitRunner(
        base_cfg={"main": {"parameters": {}}},
        param_paths=["main.parameters.beta"],
        model_factory=mock_model_factory,
        solver=mock_solver_with_recorder,
        simulator=mock_simulator_with_recorder,
        metric_fns={"LL": lambda m: -m.log_likelihood()},
    )
    
    # Run with parameter vector
    x = np.array([0.95])
    metrics = runner.run(x)
    
    # Check that metrics from recorder are included
    assert "iterations" in metrics
    assert "energy" in metrics
    assert "error" in metrics
    assert "convergence" in metrics
    assert "LL" in metrics
    
    # Check exact values
    assert metrics["iterations"] == 100
    assert metrics["energy"] == -5.0
    assert metrics["error"] == 0.01
    assert metrics["convergence"] == 0.001
    assert metrics["LL"] == 1000.0


def test_complex_metrics():
    """Test CircuitRunner with complex (non-scalar) metrics."""
    runner = CircuitRunner(
        base_cfg={"main": {"parameters": {}}},
        param_paths=["main.parameters.beta"],
        model_factory=mock_model_factory,
        solver=mock_solver_with_recorder,
        simulator=mock_simulator_with_recorder,
    )
    
    # Run with parameter vector
    x = np.array([0.95])
    metrics = runner.run(x)
    
    # Complex metrics from recorder should be present
    assert "period_timings" in metrics
    assert "stage_timings" in metrics
    
    # Check they have the right structure
    assert isinstance(metrics["period_timings"], dict)
    assert isinstance(metrics["stage_timings"], list)
    assert metrics["period_timings"]["p0"] == 0.1
    assert metrics["stage_timings"] == [0.1, 0.2, 0.3]


def test_mpi_map_with_complex_metrics():
    """Test mpi_map with complex metrics."""
    runner = CircuitRunner(
        base_cfg={"main": {"parameters": {}}},
        param_paths=["main.parameters.beta"],
        model_factory=mock_model_factory,
        solver=mock_solver_with_recorder,
        simulator=mock_simulator_with_recorder,
    )
    
    # Generate parameter vectors
    xs = np.array([[0.95], [0.96], [0.97]])
    
    # Run mpi_map
    df = mpi_map(runner, xs, mpi=False)
    
    # Check that scalar metrics are included
    assert "iterations" in df.columns
    assert "energy" in df.columns
    assert "error" in df.columns
    assert "convergence" in df.columns
    
    # Verify the values
    assert list(df["iterations"]) == [100, 100, 100]
    
    # For complex metrics, the JSON strings should be in the DataFrame
    assert "period_timings" in df.columns
    assert isinstance(df["period_timings"].iloc[0], str)  # Should be JSON string


def test_return_model_with_recorder():
    """Test CircuitRunner.run with return_model=True and recorder."""
    runner = CircuitRunner(
        base_cfg={"main": {"parameters": {}}},
        param_paths=["main.parameters.beta"],
        model_factory=mock_model_factory,
        solver=mock_solver_with_recorder,
        simulator=mock_simulator_with_recorder,
    )
    
    # Run with parameter vector and return_model=True
    x = np.array([0.95])
    metrics, model = runner.run(x, return_model=True)
    
    # Should have metrics from recorder
    assert "iterations" in metrics
    assert "energy" in metrics
    assert "error" in metrics
    assert "convergence" in metrics
    
    # And the model should be returned
    assert hasattr(model, "log_likelihood") 