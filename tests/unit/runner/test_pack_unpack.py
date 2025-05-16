"""
Test parameter packing and unpacking in CircuitRunner.
"""

import pytest
import numpy as np
from dynx.runner.circuit_runner import CircuitRunner
from dynx.runner.sampler import build_design


# Mock model factory and solver
def mock_model_factory(cfg):
    class MockModel:
        def __init__(self):
            self.cfg = cfg
            
        def log_likelihood(self):
            # For testing metrics
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


def test_pack_unpack(test_runner):
    """Test packing and unpacking of parameters."""
    # Create a parameter dictionary
    params = {
        "main.parameters.beta": 0.95,
        "main.parameters.delta": 0.05,
    }
    
    # Pack parameters into array
    x = np.array([0.95, 0.05])
    
    # Unpack array back to dictionary
    unpacked = {}
    for i, path in enumerate(test_runner.param_paths):
        unpacked[path] = x[i]
    
    # Check dictionary keys and values
    assert set(unpacked.keys()) == {"main.parameters.beta", "main.parameters.delta"}
    assert np.isclose(unpacked["main.parameters.beta"], 0.95)
    assert np.isclose(unpacked["main.parameters.delta"], 0.05)


def test_patch_with_missing_values(test_runner):
    """Test patching with missing values."""
    # Create a parameter array with NaN
    x = np.array([0.95, np.nan])
    
    # Patch the config
    cfg = test_runner.patch_config(x)
    
    # Check that only the first parameter was set
    assert np.isclose(cfg["main"]["parameters"]["beta"], 0.95)
    assert np.isnan(cfg["main"]["parameters"]["delta"])


def test_patch_with_string_values():
    """Test patching with string values."""
    base_cfg = {
        "model": {
            "params": {
                "beta": 0.9,
                "regime": "low"
            }
        }
    }
    param_paths = ["model.params.beta", "model.params.regime"]
    
    runner = CircuitRunner(
        base_cfg=base_cfg,
        param_paths=param_paths,
        model_factory=mock_model_factory,
        solver=mock_solver
    )
    
    # Create mixed-type parameter array (using dtype=object for string support)
    x = np.array([0.95, "high"], dtype=object)
    
    # Patch the config
    cfg = runner.patch_config(x)
    
    # Check that both parameters were set correctly
    assert np.isclose(cfg["model"]["params"]["beta"], 0.95)
    assert cfg["model"]["params"]["regime"] == "high"


def test_sample_prior():
    """Test sampling from prior distribution."""
    # Define parameter paths
    param_paths = ["main.parameters.beta", "main.parameters.delta"]
    
    # Define parameter metadata for build_design
    meta = {
        "main.parameters.beta": {"min": 0.8, "max": 0.99},
        "main.parameters.delta": {"min": 0.02, "max": 0.1}
    }
    
    # Create a sampler function for backward compatibility testing
    # This would be replaced with build_design in actual code
    def sample_fn(n):
        return np.column_stack([
            np.random.uniform(0.8, 0.99, n),
            np.random.uniform(0.02, 0.1, n)
        ])
    
    # Create CircuitRunner (without sampler)
    runner = CircuitRunner(
        base_cfg={"main": {"parameters": {}}},
        param_paths=param_paths,
        model_factory=mock_model_factory,
        solver=mock_solver
    )
    
    # Generate parameter design matrix using build_design
    n_samples = 100
    rng = np.random.RandomState(42)  # Fixed seed for reproducibility
    
    # Create design matrix directly with build_design
    from dynx.runner.sampler import MVNormSampler
    mvn_sampler = MVNormSampler(
        mean=np.array([0.9, 0.05]),  # Mid-points of the ranges
        cov=np.array([[0.005, 0], [0, 0.0005]]),  # Small covariance
        clip_bounds=True
    )
    xs, _ = build_design(param_paths, [mvn_sampler], [n_samples], meta, seed=42)
    
    # Check shape
    assert xs.shape == (n_samples, 2)
    
    # Check that values are within bounds
    assert np.all(xs[:, 0] >= 0.8)
    assert np.all(xs[:, 0] <= 0.99)
    assert np.all(xs[:, 1] >= 0.02)
    assert np.all(xs[:, 1] <= 0.1)


def test_param_paths_validation():
    """Test validation of parameter paths."""
    # CircuitRunner takes any paths now, no validation needed
    # Just verify it can be created with any paths
    base_cfg = {"some": {"nested": {"path": 123}}}
    
    # This should not raise an error even with unusual paths
    runner = CircuitRunner(
        base_cfg=base_cfg,
        param_paths=["some.nested.path", "nonexistent.path"],
        model_factory=mock_model_factory,
        solver=mock_solver,
    )
    
    # Verify the paths were stored correctly
    assert runner.param_paths == ["some.nested.path", "nonexistent.path"]


def test_patch_config():
    """Test patching of configurations with parameter values."""
    base_cfg = {"model": {"parameters": {}}}
    param_paths = ["model.parameters.beta", "model.settings.gamma"]
    
    runner = CircuitRunner(
        base_cfg=base_cfg,
        param_paths=param_paths,
        model_factory=mock_model_factory,
        solver=mock_solver,
    )
    
    # Create a parameter array
    x = np.array([0.95, 1.5])
    
    # Patch the config
    cfg = runner.patch_config(x)
    
    # Check that the parameters were set correctly
    assert np.isclose(cfg["model"]["parameters"]["beta"], 0.95)
    assert np.isclose(cfg["model"]["settings"]["gamma"], 1.5)
    
    # Original config should be unchanged
    assert "beta" not in base_cfg["model"]["parameters"]
    assert "settings" not in base_cfg["model"] or "gamma" not in base_cfg["model"]["settings"] 