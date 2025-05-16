import numpy as np
import pytest

from dynx.runner.sampler import (
    MVNormSampler,
    build_design,
)


def _is_close(value, target, atol=1e-6):
    """Check if a value is close to a target with a tolerance."""
    if isinstance(value, (float, np.floating)):
        return abs(value - target) <= atol
    return False


def test_mvnorm_bounds():
    """Test that MVNormSampler correctly enforces min/max bounds"""
    # Setup test data - small variance to ensure we stay within bounds most of the time
    # but large enough to sometimes exceed bounds
    mean = np.array([0.5])
    cov = np.array([[0.1]])
    
    # Create a sampler with bounds
    sampler = MVNormSampler(mean, cov, clip_bounds=True)
    
    # Generate a large number of samples to ensure we test the bounds
    n_samples = 1000
    param_paths = ["param"]
    meta = {"param": {"min": 0.0, "max": 1.0}}
    
    # Build design with just this sampler
    samplers = [sampler]
    Ns = [n_samples]
    xs, info = build_design(param_paths, samplers, Ns, meta)
    
    # All samples should be within bounds
    assert all(x >= 0.0 for x in xs[:, 0])
    assert all(x <= 1.0 for x in xs[:, 0])
    
    # With this many samples, we should see some values that would have been out of bounds
    # and got clipped (values very close to the bounds)
    close_to_min = [_is_close(x, 0.0) for x in xs[:, 0]]
    close_to_max = [_is_close(x, 1.0) for x in xs[:, 0]]
    
    # We should have at least a few samples clipped to the bounds
    assert sum(close_to_min) > 0 or sum(close_to_max) > 0


def test_mvnorm_bounds_without_clipping():
    """Test that MVNormSampler can generate samples without bounds clipping"""
    # Setup test data with narrow variance to ensure we stay within bounds
    mean = np.array([0.5])
    cov = np.array([[0.01]])  # Smaller variance
    
    # Create a sampler without bounds clipping
    sampler = MVNormSampler(mean, cov, clip_bounds=False)
    
    # Generate samples
    n_samples = 100
    param_paths = ["param"]
    meta = {"param": {"min": 0.0, "max": 1.0}}
    
    # Build design with just this sampler
    samplers = [sampler]
    Ns = [n_samples]
    xs, info = build_design(param_paths, samplers, Ns, meta)
    
    # Since we're not clipping, some samples could be outside bounds
    # But with our narrow distribution, most should still be within bounds
    in_bounds = [0.0 <= x <= 1.0 for x in xs[:, 0]]
    assert sum(in_bounds) / len(in_bounds) > 0.95


def test_mvnorm_out_of_bounds_detection():
    """Test that MVNormSampler correctly detects out-of-bounds samples"""
    # Setup test with mean at the boundary
    mean = np.array([0.0])  # At the lower bound
    cov = np.array([[0.1]])
    
    # Create a sampler that will detect out-of-bounds samples
    sampler = MVNormSampler(mean, cov, clip_bounds=True)
    
    # Generate samples
    n_samples = 100
    param_paths = ["param"]
    meta = {"param": {"min": 0.0, "max": 1.0}}
    
    # Build design with just this sampler
    samplers = [sampler]
    Ns = [n_samples]
    xs, info = build_design(param_paths, samplers, Ns, meta)
    
    # All samples should be within bounds
    assert all(x >= 0.0 for x in xs[:, 0])
    assert all(x <= 1.0 for x in xs[:, 0])
    
    # With mean at the boundary, we should see many samples clipped at the min bound
    close_to_min = [_is_close(x, 0.0) for x in xs[:, 0]]
    assert sum(close_to_min) > 10  # Should have plenty of samples at the boundary


def test_mvnorm_resampling():
    """Test that MVNormSampler can resample instead of clipping"""
    # This is a test that would be more relevant if we implement the resampling logic
    # Currently we're using clipping, but this test is included for completeness
    mean = np.array([0.5])
    cov = np.array([[0.1]])
    
    # Hypothetical sampler that would resample instead of clip
    # For now, we'll just test with clipping
    sampler = MVNormSampler(mean, cov, clip_bounds=True)
    
    # Generate samples
    n_samples = 100
    param_paths = ["param"]
    meta = {"param": {"min": 0.2, "max": 0.8}}  # Narrower bounds
    
    # Build design with just this sampler
    samplers = [sampler]
    Ns = [n_samples]
    xs, info = build_design(param_paths, samplers, Ns, meta)
    
    # All samples should be within the narrower bounds
    assert all(x >= 0.2 for x in xs[:, 0])
    assert all(x <= 0.8 for x in xs[:, 0])


def test_mvnorm_with_categorical_error():
    """Test that MVNormSampler raises an error when given categorical parameters"""
    # Setup
    mean = np.array([0.5])
    cov = np.array([[0.1]])
    sampler = MVNormSampler(mean, cov)
    
    # Try to sample a categorical parameter
    param_paths = ["cat_param"]
    meta = {"cat_param": {"enum": ["A", "B", "C"]}}
    
    # This should raise an error since MVNormSampler can't handle categoricals
    with pytest.raises(ValueError, match="can only handle numeric parameters"):
        sampler(n=5, param_paths=param_paths, meta=meta)


def test_mvnorm_with_discrete_values_error():
    """Test that MVNormSampler raises an error when given discrete list parameters"""
    # Setup
    mean = np.array([0.5])
    cov = np.array([[0.1]])
    sampler = MVNormSampler(mean, cov)
    
    # Try to sample a discrete parameter
    param_paths = ["discrete_param"]
    meta = {"discrete_param": {"values": [0.1, 0.2, 0.3]}}
    
    # This should raise an error since MVNormSampler can't handle discrete lists
    with pytest.raises(ValueError, match="can only handle numeric parameters"):
        sampler(n=5, param_paths=param_paths, meta=meta) 