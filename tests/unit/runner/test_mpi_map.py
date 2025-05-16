"""
Tests for MPI mapping functionality in CircuitRunner.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from dynx.runner.circuit_runner import CircuitRunner, mpi_map


class MockMPI:
    """Mock MPI implementation for testing."""
    def __init__(self, size=4, rank=0):
        self.size = size
        self.rank = rank
    
    class Comm:
        def __init__(self, size, rank):
            self.size = size
            self.rank = rank
        
        def Get_size(self):
            return self.size
        
        def Get_rank(self):
            return self.rank
        
        def bcast(self, obj, root=0):
            return obj
        
        def scatter(self, sendobj, root=0):
            # Simplified scatter that just returns a portion of the list
            if isinstance(sendobj, list):
                chunk_size = max(1, len(sendobj) // self.size)
                start = self.rank * chunk_size
                end = min(start + chunk_size, len(sendobj))
                return sendobj[start:end]
            return sendobj
        
        def gather(self, sendobj, root=0):
            # Just return a list with the sendobj repeated for each rank
            return [sendobj] * self.size


def mock_model_factory(cfg):
    """Factory for creating a mock model."""
    
    class MockModel:
        def __init__(self):
            self.cfg = cfg
        
        def log_likelihood(self):
            """For testing metrics"""
            return -1000.0
    
    return MockModel()


def mock_solver(model, recorder=None):
    """Mock solver that does nothing."""
    pass


@pytest.fixture
def test_runner():
    """Create a CircuitRunner for testing."""
    return CircuitRunner(
        base_cfg={"main": {"parameters": {}}},
        param_paths=[
            "main.parameters.beta",
            "main.parameters.delta"
        ],
        model_factory=mock_model_factory,
        solver=mock_solver,
        metric_fns={"LL": lambda m: -m.log_likelihood()},
    )


def test_mpi_map_local(test_runner):
    """Test mpi_map with local processing (no MPI)."""
    xs = np.array([
        [0.9, 0.05],
        [0.92, 0.06],
        [0.94, 0.07],
        [0.96, 0.08]
    ])
    
    # Run mpi_map with local processing
    df = mpi_map(test_runner, xs, mpi=False)
    
    # Check results
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    assert "main.parameters.beta" in df.columns
    assert "main.parameters.delta" in df.columns
    assert "LL" in df.columns  # Metric should be included


def test_mpi_map_with_mock_mpi(monkeypatch, test_runner):
    """Test mpi_map with mocked MPI implementation."""
    # Skip this test if mpi4py is not available
    pytest.importorskip("mpi4py", reason="mpi4py is not installed")
    
    # Mock MPI
    mock_mpi = MockMPI(size=2, rank=0)
    
    # Patch the mpi4py.MPI import inside circuit_runner.py
    with patch('mpi4py.MPI', mock_mpi):
        with patch('dynx.runner.circuit_runner.HAS_MPI', True):
            xs = np.array([
                [0.9, 0.05],
                [0.92, 0.06],
                [0.94, 0.07],
                [0.96, 0.08]
            ])
            
            # Run mpi_map with mocked MPI
            df = mpi_map(test_runner, xs, mpi=True)
            
            # On rank 0, we should get a DataFrame
            if mock_mpi.rank == 0:
                assert isinstance(df, pd.DataFrame)
                assert len(df) > 0
                assert "main.parameters.beta" in df.columns
                assert "main.parameters.delta" in df.columns
                assert "LL" in df.columns


def test_mpi_map_with_categorical_values(test_runner):
    """Test mpi_map with categorical parameters."""
    # Create a runner with categorical parameters
    runner = CircuitRunner(
        base_cfg={"model": {"params": {}}},
        param_paths=[
            "model.params.beta", 
            "model.params.regime"
        ],
        model_factory=mock_model_factory,
        solver=mock_solver,
        metric_fns={"LL": lambda m: -m.log_likelihood()},
    )
    
    # Create parameter vectors with mixed types
    xs = np.array([
        [0.9, "low"],
        [0.92, "high"],
        [0.94, "medium"],
        [0.96, "low"]
    ], dtype=object)
    
    # Run mpi_map
    df = mpi_map(runner, xs, mpi=False)
    
    # Check results
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    assert "model.params.beta" in df.columns
    assert "model.params.regime" in df.columns
    assert "LL" in df.columns
    
    # Check that categorical values are preserved
    assert set(df["model.params.regime"]) == {"low", "high", "medium"}


def test_mpi_map_with_return_models(test_runner):
    """Test mpi_map with return_models=True."""
    xs = np.array([
        [0.9, 0.05],
        [0.92, 0.06],
        [0.94, 0.07],
        [0.96, 0.08]
    ])
    
    # Run mpi_map with return_models=True
    df, models = mpi_map(test_runner, xs, return_models=True, mpi=False)
    
    # Check results
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    assert len(models) == 4
    assert all(hasattr(m, "log_likelihood") for m in models) 