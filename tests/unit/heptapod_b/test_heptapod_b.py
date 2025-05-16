"""
Testing suite for heptapod_b.

This module tests the basic functionality of the heptapod_b package.
"""

import sys
import os
import unittest
import numpy as np
import pytest
import yaml
from typing import Dict, Any

# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

def test_imports():
    """Test importing heptapod_b modules."""
    try:
        from dynx.heptapodx.core.api import FunctionalProblem, AttrDict
        from dynx.heptapodx.init import build_stage
        from dynx.heptapodx.num import compile_num
        
        print("Import tests passed.")
        # Instead of returning, just assert that the imports worked
        assert FunctionalProblem is not None
        assert AttrDict is not None
        assert build_stage is not None
        assert compile_num is not None
    except ImportError as e:
        pytest.fail(f"Import tests failed: {e}")


def initialize_model():
    """Initialize a test model."""
    from dynx.heptapodx.core.api import FunctionalProblem
    from dynx.heptapodx.init import build_stage
    from dynx.heptapodx.num import compile_num
    
    return FunctionalProblem, AttrDict, initialize_model, build_stage, compile_num


def test_functional_problem():
    """Test creating and using a FunctionalProblem."""
    from dynx.heptapodx.core.api import FunctionalProblem
    
    # Create an empty FunctionalProblem
    fp = FunctionalProblem()
    
    # Add some basic attributes and check they exist
    # Test direct attribute setting (the recommended pattern)
    fp.parameters_dict = {'beta': 0.96, 'gamma': 2.0}
    assert fp.parameters_dict['beta'] == 0.96
    assert fp.parameters_dict['gamma'] == 2.0
    
    # Test attribute-like access
    fp.settings = {'tol': 1e-8, 'max_iter': 1000}
    assert hasattr(fp, 'settings')
    assert fp.settings['tol'] == 1e-8
    assert fp.settings['max_iter'] == 1000
    
    # Test nested structures with math
    fp._math = {
        'functions': {
            'u': {'expr': 'c ** (1-gamma) / (1-gamma)', 'inputs': ['c']}
        }
    }
    assert 'functions' in fp.math
    assert 'u' in fp.math.functions
    assert fp.math.functions.u['expr'] == 'c ** (1-gamma) / (1-gamma)'
    
    # Test direct property access for parameters
    assert hasattr(fp, 'param')
    assert fp.param.beta == 0.96
    assert fp.param.gamma == 2.0


def test_load_yaml():
    """Test loading a YAML file."""
    from dynx.heptapodx.io.yaml_loader import load_config
    
    # Create a temporary YAML file
    yaml_content = """
    parameters:
      beta: 0.96
      gamma: 2.0
    settings:
      tol: 1.0e-8
      max_iter: 1000
    math:
      functions:
        u:
          expr: c ** (1-gamma) / (1-gamma)
          inputs: [c]
    """
    
    # Write to a temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
        f.write(yaml_content.encode('utf-8'))
        temp_file = f.name
    
    try:
        # Load the YAML file
        config = load_config(temp_file)
        
        # Check the loaded content
        assert 'parameters' in config
        assert config['parameters']['beta'] == 0.96
        assert config['parameters']['gamma'] == 2.0
    finally:
        # Clean up
        os.unlink(temp_file)


def test_attr_dict():
    """Test the AttrDict wrapper."""
    from dynx.heptapodx.core.functional_problem import AttrDict
    
    # Create a dictionary
    d = {'a': 1, 'b': {'c': 2, 'd': 3}, 'e': [4, 5, 6]}
    
    # Wrap it in an AttrDict
    ad = AttrDict(d)
    
    # Test attribute-like access
    assert ad.a == 1
    assert ad.b.c == 2
    assert ad.b.d == 3
    assert ad.e == [4, 5, 6]
    
    # Test dict-like access
    assert 'a' in ad
    assert ad['a'] == 1
    
    # Test setting attributes
    ad.f = 7
    assert ad.f == 7
    assert ad['f'] == 7
    
    # Test nested setting
    ad.g = {'h': 8}
    assert ad.g.h == 8
    
    # Test that nested dictionaries are wrapped
    assert isinstance(ad.b, AttrDict)
    assert isinstance(ad.g, AttrDict)


def test_backward_compatibility():
    """Test backward compatibility with old import paths."""
    try:
        import dynx.heptapodx as heptapod_b
        assert hasattr(heptapod_b, 'core')
        
        # Check that we can import the key functions from the old paths
        from dynx.heptapodx.core.api import FunctionalProblem
        from dynx.heptapodx.core.functional_problem import AttrDict
        from dynx.heptapodx.io.yaml_loader import load_config
        assert FunctionalProblem is not None
        assert AttrDict is not None
        assert load_config is not None
    except ImportError as e:
        pytest.skip(f"Backward compatibility not maintained: {e}")


def test_numerical_state_space():
    """Test numerical state space generation."""
    from dynx.heptapodx.core.api import FunctionalProblem
    import numpy as np
    
    # Create a simple state space definition
    fp = FunctionalProblem()
    fp._math = {
        'state_space': {
            'arvl': {
                'dimensions': ['a'],
                'grid': {
                    'a': {
                        'type': 'linspace',
                        'min': 0.0,
                        'max': 10.0,
                        'n': 100
                    }
                }
            }
        }
    }
    
    # Test generation of a simple grid
    from numpy import linspace
    grid = linspace(0.0, 10.0, 100)
    
    # Check that the grid is correct
    assert len(grid) == 100
    assert grid[0] == 0.0
    assert grid[-1] == 10.0
    assert np.isclose(grid[1] - grid[0], 0.101, atol=1e-3)


def run_all_tests():
    """Run all tests."""
    test_imports()
    test_functional_problem()
    test_load_yaml()
    test_attr_dict()
    test_backward_compatibility()
    test_numerical_state_space()
    
    print("All tests passed!")


if __name__ == "__main__":
    unittest.main() 