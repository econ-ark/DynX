"""
Tests for the sampler module.
"""

import numpy as np
import pytest
import itertools
from unittest.mock import patch

from dynx.runner.sampler import (
    MVNormSampler,
    FullGridSampler,
    LatinHypercubeSampler,
    FixedSampler,
    build_design,
)


class TestMVNormSampler:
    """Test MVNormSampler class."""

    def test_init(self):
        """Test initialization with valid parameters."""
        mean = np.array([0.0, 1.0])
        cov = np.array([[1.0, 0.5], [0.5, 2.0]])
        sampler = MVNormSampler(mean, cov, sample_size=10)
        
        # Check attributes
        assert np.array_equal(sampler.mean, mean)
        assert np.array_equal(sampler.cov, cov)
        assert sampler.sample_size == 10

    def test_invalid_init(self):
        """Test initialization with invalid parameters."""
        # Mean with wrong shape
        with pytest.raises(ValueError):
            # Create cov with wrong shape for the mean
            mean = np.array([0.0])
            cov = np.array([[1.0, 0.5], [0.5, 2.0]])
            # This should raise ValueError since dimensions don't match
            # Note: We need to actually call the sampler to trigger the validation
            sampler = MVNormSampler(mean, cov, sample_size=10)
            sampler(10, ["param1", "param2"], {})

    def test_call(self):
        """Test generating samples."""
        mean = np.array([0.0, 1.0])
        cov = np.array([[1.0, 0.5], [0.5, 2.0]])
        sampler = MVNormSampler(mean, cov, sample_size=10)
        
        # Set seed for reproducibility
        np.random.seed(42)
        
        # Generate samples
        param_paths = ["param1", "param2"]
        meta = {"param1": {"min": -1.0, "max": 1.0}, "param2": {"min": 0.0, "max": 2.0}}
        
        samples = sampler(n=10, param_paths=param_paths, meta=meta, seed=42)
        
        # Check shape
        assert samples.shape == (10, 2)
        
        # Check bounds are respected
        assert np.all(samples[:, 0] >= -1.0)
        assert np.all(samples[:, 0] <= 1.0)
        assert np.all(samples[:, 1] >= 0.0)
        assert np.all(samples[:, 1] <= 2.0)


class TestFullGridSampler:
    """Test FullGridSampler class."""

    def test_init(self):
        """Test initialization with valid parameters."""
        grid_values = {"param1": [0.0, 1.0], "param2": [2.0, 3.0, 4.0]}
        sampler = FullGridSampler(grid_values)
        
        # Check attributes
        assert sampler.grid_values == grid_values
        
    def test_call(self):
        """Test generating grid samples."""
        grid_values = {"param1": [0.0, 1.0], "param2": [2.0, 3.0, 4.0]}
        sampler = FullGridSampler(grid_values)
        
        # Generate grid
        param_paths = ["param1", "param2"]
        meta = {}
        
        grid = sampler(n=None, param_paths=param_paths, meta=meta)
        
        # Check shape: 2 param1 values × 3 param2 values = 6 rows
        assert grid.shape == (6, 2)
        
        # Check all combinations exist
        expected_combos = list(itertools.product([0.0, 1.0], [2.0, 3.0, 4.0]))
        for i, row in enumerate(grid):
            assert tuple(row) in expected_combos


class TestLatinHypercubeSampler:
    """Test LatinHypercubeSampler class."""

    def test_init(self):
        """Test initialization with valid parameters."""
        param_ranges = [(0.0, 1.0), (2.0, 4.0)]
        sampler = LatinHypercubeSampler(param_ranges, sample_size=10)
        
        # Check attributes
        assert sampler.ranges == param_ranges
        assert sampler.sample_size == 10

    def test_call(self):
        """Test generating LHS samples."""
        param_ranges = [(0.0, 1.0), (2.0, 4.0)]
        sampler = LatinHypercubeSampler(param_ranges, sample_size=10)
        
        # Generate samples
        param_paths = ["param1", "param2"]
        meta = {}
        
        samples = sampler(n=10, param_paths=param_paths, meta=meta, seed=42)
        
        # Check shape
        assert samples.shape == (10, 2)
        
        # Check bounds are respected
        assert np.all(samples[:, 0] >= 0.0)
        assert np.all(samples[:, 0] <= 1.0)
        assert np.all(samples[:, 1] >= 2.0)
        assert np.all(samples[:, 1] <= 4.0)


class TestFixedSampler:
    """Test FixedSampler class."""

    def test_init(self):
        """Test initialization with valid parameters."""
        param_sets = np.array([
            [0.0, 2.0],
            [0.5, 3.0],
            [1.0, 4.0],
        ])
        sampler = FixedSampler(param_sets)
        
        # Check attributes
        assert sampler.rows.shape == (3, 2)
        assert np.array_equal(sampler.rows, param_sets)

    def test_call(self):
        """Test returning fixed samples."""
        param_sets = np.array([
            [0.0, 2.0],
            [0.5, 3.0],
            [1.0, 4.0],
        ])
        sampler = FixedSampler(param_sets)
        
        # Return fixed samples
        param_paths = ["param1", "param2"]
        meta = {}
        
        samples = sampler(n=None, param_paths=param_paths, meta=meta)
        
        # Check shape
        assert samples.shape == (3, 2)
        
        # Check values match fixed set
        assert np.array_equal(samples, param_sets)


class TestBuildDesign:
    """Test build_design function."""

    def test_with_samplers(self):
        """Test building a design with samplers."""
        # Define samplers with matching dimensions
        mvnorm_sampler = MVNormSampler(
            mean=np.array([0.0, 0.5]),  # 2D mean to match 2 parameters
            cov=np.array([[1.0, 0.0], [0.0, 1.0]]),
            sample_size=5
        )
        
        lhs_sampler = LatinHypercubeSampler(
            ranges=[(0.0, 1.0), (0.0, 1.0)],  # 2 ranges to match 2 parameters
            sample_size=3
        )
        
        # Define param_paths
        param_paths = ['model.param1', 'model.param2']
        
        # Set up meta
        meta = {
            'model.param1': {'min': -1.0, 'max': 1.0},
            'model.param2': {'min': 0.0, 'max': 1.0}
        }
        
        # Build design with two samplers
        samplers = [mvnorm_sampler, lhs_sampler]
        Ns = [5, 3]
        
        # Set seed for reproducibility
        xs, info = build_design(param_paths, samplers, Ns, meta, seed=42)
        
        # Check shape: 5 rows from mvnorm + 3 rows from lhs = 8 rows
        assert xs.shape[0] == 8
        assert xs.shape[1] == 2
        
        # Check sampler tags
        assert len(info["sampler"]) == 8
        assert all("MVNormSampler" in tag for tag in info["sampler"][:5])
        assert all("LatinHypercubeSampler" in tag for tag in info["sampler"][5:])

    def test_with_value_lists(self):
        """Test building a design with samplers that can handle categorical values."""
        # Define param_paths
        param_paths = ['model.param1', 'model.param2', 'model.param3']
        
        # Set up meta with a categorical parameter
        meta = {
            'model.param1': {'min': 0.0, 'max': 1.0},
            'model.param2': {'values': [2.0]},
            'model.param3': {'enum': ['A', 'B', 'C', 'D', 'E']},
        }
        
        # Create samplers
        mvnorm_sampler = MVNormSampler(
            mean=np.array([0.5]),
            cov=np.array([[0.1]]),
            sample_size=5
        )
        
        # Build design
        samplers = [mvnorm_sampler]
        Ns = [5]
        
        xs, info = build_design(param_paths, samplers, Ns, meta, seed=42)
        
        # Check shape: 5 samples × 1 value for param2 × 5 values for param3 = 25 rows
        assert xs.shape[0] == 25
        assert xs.shape[1] == 3
        
        # Check categorical values
        param3_values = set(xs[:, 2])
        assert param3_values == {'A', 'B', 'C', 'D', 'E'}
        
        # Check all param2 values are 2.0
        assert np.all(xs[:, 1] == 2.0)

    def test_fixed_sampler_size_mismatch(self):
        """Test with a FixedSampler for a subset of parameters."""
        # Create a fixed sampler with both parameters
        param_sets = np.array([
            [0.0, 'A'],
            [0.5, 'B'],
            [1.0, 'A'],
        ], dtype=object)  # Use object dtype to allow mixed types
        fixed_sampler = FixedSampler(param_sets)
        
        # Define param_paths with categorical parameters
        param_paths = ['model.param1', 'model.param2']
        
        # Set up meta
        meta = {
            'model.param1': {'min': 0.0, 'max': 1.0},
            'model.param2': {'enum': ['A', 'B']}
        }
        
        # Build design
        samplers = [fixed_sampler]
        Ns = [None]  # None for FixedSampler
        
        xs, info = build_design(param_paths, samplers, Ns, meta)
        
        # Check shape: 3 fixed samples × 2 categories = 6 rows
        # The build_design function creates a cartesian product with the categorical values
        assert xs.shape[0] == 6
        assert xs.shape[1] == 2
        
        # Check categorical values are present
        param2_values = set(xs[:, 1])
        assert param2_values == {'A', 'B'} 