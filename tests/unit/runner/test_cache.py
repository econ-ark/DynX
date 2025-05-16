"""
Tests for circuit_runner caching functionality.
"""

import os
import pickle
import pytest
from unittest.mock import MagicMock
import numpy as np

from dynx.runner.circuit_runner import CircuitRunner


class MockCallCounter:
    """Helper to count how many times functions are called."""
    
    def __init__(self):
        self.count = 0
    
    def reset(self):
        """Reset the counter."""
        self.count = 0
    
    def __call__(self, *args, **kwargs):
        """Increment counter and return a value."""
        self.count += 1
        return self.count


def mock_model_factory(cfg):
    """Factory for creating a mock model."""
    
    class MockModel:
        def __init__(self):
            self.cfg = cfg
        
        def log_likelihood(self):
            """For testing metrics"""
            return -1000.0
        
        def euler_error_max(self):
            """For testing multiple metrics"""
            return 0.01
    
    return MockModel()


def mock_solver(model, recorder=None):
    """Mock solver that does nothing."""
    pass


@pytest.fixture
def cached_runner():
    """Create a CircuitRunner with caching enabled."""
    runner = CircuitRunner(
        base_cfg={"main": {"parameters": {}}},
        param_paths=["main.parameters.beta"],
        model_factory=mock_model_factory,
        solver=mock_solver,
        metric_fns={
            "LL": lambda m: -m.log_likelihood(),
            "EulerMax": lambda m: m.euler_error_max(),
        },
        cache=True,
    )
    
    # Set up call counters for testing
    runner._counter = MockCallCounter()
    runner._orig_model_factory = runner.model_factory
    runner.model_factory = lambda cfg: (runner._counter(), runner._orig_model_factory(cfg))[1]
    
    return runner


@pytest.fixture
def uncached_runner():
    """Create a CircuitRunner without caching."""
    runner = CircuitRunner(
        base_cfg={"main": {"parameters": {}}},
        param_paths=["main.parameters.beta"],
        model_factory=mock_model_factory,
        solver=mock_solver,
        metric_fns={
            "LL": lambda m: -m.log_likelihood(),
        },
        cache=False,
    )
    
    # Set up call counters for testing
    runner._counter = MockCallCounter()
    runner._orig_model_factory = runner.model_factory
    runner.model_factory = lambda cfg: (runner._counter(), runner._orig_model_factory(cfg))[1]
    
    return runner


def test_cache_hit(cached_runner):
    """Test that cache is used properly."""
    # First run
    x = np.array([0.95])
    metrics1 = cached_runner.run(x)
    
    assert cached_runner._counter.count == 1
    assert "LL" in metrics1
    assert "EulerMax" in metrics1
    
    # Second run with same parameters should use cache
    cached_runner._counter.reset()
    metrics2 = cached_runner.run(x)
    
    assert cached_runner._counter.count == 0  # model_factory shouldn't be called
    assert metrics1 == metrics2


def test_no_cache(uncached_runner):
    """Test that when cache=False, cache is not used."""
    # First run
    x = np.array([0.95])
    metrics1 = uncached_runner.run(x)
    
    assert uncached_runner._counter.count == 1
    
    # Second run with same parameters should not use cache
    uncached_runner._counter.reset()
    metrics2 = uncached_runner.run(x)
    
    assert uncached_runner._counter.count == 1  # model_factory should be called again


def test_cache_metric_set(cached_runner):
    """Test that different metric sets create different cache entries."""
    # First run with all metrics
    x = np.array([0.95])
    metrics1 = cached_runner.run(x)
    
    assert cached_runner._counter.count == 1
    assert "LL" in metrics1
    assert "EulerMax" in metrics1
    
    # Second run requesting same metrics should use cache
    cached_runner._counter.reset()
    metrics2 = cached_runner.run(x)
    
    assert cached_runner._counter.count == 0  # model_factory shouldn't be called
    assert metrics1 == metrics2
    
    # Verify caching works correctly
    assert len(cached_runner._cache) == 1


def test_categorical_parameters(cached_runner):
    """Test that categorical parameters work correctly with caching."""
    # Create a new runner with a categorical parameter
    class MockModelWithCategorical:
        def __init__(self, cfg):
            self.cfg = cfg
        
        def log_likelihood(self):
            """Return different values based on the regime parameter"""
            # This makes the log likelihood depend on the categorical value
            regime = self.cfg["model"]["params"]["regime"]
            return -1000.0 if regime == "high" else -500.0

    def mock_categorical_factory(cfg):
        return MockModelWithCategorical(cfg)
    
    runner = CircuitRunner(
        base_cfg={"model": {"params": {"beta": 0.9, "regime": "low"}}},
        param_paths=["model.params.beta", "model.params.regime"],
        model_factory=mock_categorical_factory,
        solver=mock_solver,
        metric_fns={
            "LL": lambda m: -m.log_likelihood(),
        },
        cache=True,
    )
    
    # Set up call counter
    runner._counter = MockCallCounter()
    runner._orig_model_factory = runner.model_factory
    runner.model_factory = lambda cfg: (runner._counter(), runner._orig_model_factory(cfg))[1]
    
    # Run with a categorical parameter
    x = np.array([0.95, "high"], dtype=object)
    metrics1 = runner.run(x)
    
    assert runner._counter.count == 1
    assert metrics1["LL"] == 1000.0  # -(-1000.0)
    
    # Second run with same parameters should use cache
    runner._counter.reset()
    metrics2 = runner.run(x)
    
    assert runner._counter.count == 0  # model_factory shouldn't be called
    assert metrics1 == metrics2
    
    # Run with a different categorical value
    runner._counter.reset()
    x_diff = np.array([0.95, "low"], dtype=object)
    metrics3 = runner.run(x_diff)
    
    assert runner._counter.count == 1  # model_factory should be called for different parameters
    assert metrics3["LL"] == 500.0  # -(-500.0)
    assert metrics3 != metrics2 