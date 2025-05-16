"""
Tests for manual shock process functionality in Heptapod-B.
"""

import pytest
import numpy as np
from unittest.mock import Mock

# Add path for importing from src if needed
import sys
import os
import importlib.util

# To handle potential import issues, check if heptapod_b is importable
try:
    import heptapod_b
except ImportError:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    src_path = os.path.join(project_root, 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
        print(f"Added {src_path} to sys.path")

# Try importing the manual shock functionality
try:
    from heptapod_b.num.manual_shocks import (
        create_manual_shock_process,
        validate_manual_shock_parameters,
        DiscreteMarkovProcess,
        IIDProcess,
        ShockProcess
    )
    
    # Also check if state_space has resolve_reference for tests
    if importlib.util.find_spec("heptapod_b.num.state_space"):
        from heptapod_b.num.state_space import resolve_reference
    else:
        # Use the resolve_reference from manual_shocks as fallback
        from heptapod_b.num.manual_shocks import resolve_reference
    
    # Test fixture for parameter resolution
    @pytest.fixture
    def mock_model():
        """Create a mock model with parameters for testing."""
        model = Mock()
        # Make parameters_dict a real dictionary
        model.parameters_dict = {
            "Pi": np.array([[0.9, 0.1], [0.2, 0.8]]),
            "z_vals": np.array([0.5, 1.5])
        }
        model.methods = {}
        model.settings = {}
        
        # Make the mock itself act dict-like for in/getitem operations
        model.__contains__ = lambda self, key: key in self.parameters_dict
        model.__getitem__ = lambda self, key: self.parameters_dict[key]
        
        return model
    
    class TestValidation:
        """Tests for parameter validation."""
        
        def test_valid_parameters(self):
            """Test that valid parameters pass validation."""
            # Create valid test data
            tm = np.array([[0.9, 0.1], [0.2, 0.8]])
            values = np.array([0.5, 1.5])
            
            # Should not raise
            validate_manual_shock_parameters(tm, values)
        
        def test_non_square_matrix(self):
            """Test validation fails for non-square matrix."""
            tm = np.array([[0.9, 0.1, 0], [0.2, 0.7, 0.1]])  # 2x3
            values = np.array([0.5, 1.5, 2.5])
            
            with pytest.raises(ValueError, match="Transition matrix must be square"):
                validate_manual_shock_parameters(tm, values)
        
        def test_dimension_mismatch(self):
            """Test validation fails if matrix dims don't match values."""
            tm = np.array([[0.9, 0.1], [0.2, 0.8]])  # 2x2
            values = np.array([0.5, 1.5, 2.5])  # 3 values
            
            with pytest.raises(ValueError, match="do not match number of values"):
                validate_manual_shock_parameters(tm, values)
        
        def test_non_one_row_sums(self):
            """Test warning for row sums not equal to 1."""
            tm = np.array([[0.8, 0.1], [0.2, 0.7]])  # Rows sum to 0.9
            values = np.array([0.5, 1.5])
            
            with pytest.warns(UserWarning, match="do not sum to 1"):
                validate_manual_shock_parameters(tm, values)
        
        def test_negative_probabilities(self):
            """Test validation fails for negative probabilities."""
            tm = np.array([[1.1, -0.1], [0.2, 0.8]])
            values = np.array([0.5, 1.5])
            
            with pytest.raises(ValueError, match="negative probabilities"):
                validate_manual_shock_parameters(tm, values)
    
    class TestCreation:
        """Tests for shock process creation."""
        
        def test_direct_markov_process(self):
            """Test creating MarkovProcess with direct values."""
            spec = {
                "methods": {
                    "method": "manual",
                    "shock_method": "DiscreteMarkov"
                },
                "transition_matrix": [[0.9, 0.1], [0.2, 0.8]],
                "values": [0.5, 1.5],
                "labels": ["low", "high"]
            }
            
            process = create_manual_shock_process(spec, None)
            
            assert isinstance(process, DiscreteMarkovProcess)
            assert np.allclose(process.values, [0.5, 1.5])
            assert np.allclose(process.transition_matrix, [[0.9, 0.1], [0.2, 0.8]])
            assert process.labels == ["low", "high"]
            # Check stationary distribution was computed
            assert hasattr(process, "stationary_distribution")
            assert process.stationary_distribution.shape == (2,)
        
        def test_direct_iid_process(self):
            """Test creating IIDProcess with direct values."""
            spec = {
                "methods": {
                    "method": "manual",
                    "shock_method": "IID"
                },
                "values": [-1, 0, 1],
                "probabilities": [0.3, 0.4, 0.3]
            }
            
            process = create_manual_shock_process(spec, None)
            
            assert isinstance(process, IIDProcess)
            assert np.allclose(process.values, [-1, 0, 1])
            assert np.allclose(process.probs, [0.3, 0.4, 0.3])
        
        def test_parameter_references(self, mock_model):
            """Test creating process with parameter references."""
            spec = {
                "methods": {
                    "method": "manual",
                    "shock_method": "DiscreteMarkov"
                },
                "transition_matrix": ["Pi"],
                "values": ["z_vals"]
            }
            
            process = create_manual_shock_process(spec, mock_model)
            
            assert isinstance(process, DiscreteMarkovProcess)
            assert np.allclose(process.values, [0.5, 1.5])
            assert np.allclose(process.transition_matrix, [[0.9, 0.1], [0.2, 0.8]])
        
        def test_missing_required_params(self):
            """Test error when required parameters are missing."""
            # Missing values
            spec1 = {
                "methods": {
                    "method": "manual",
                    "shock_method": "DiscreteMarkov"
                },
                "transition_matrix": [[0.9, 0.1], [0.2, 0.8]]
            }
            
            with pytest.raises(ValueError, match="requires 'values' in specification"):
                create_manual_shock_process(spec1, None)
            
            # Missing transition_matrix for MarkovProcess
            spec2 = {
                "methods": {
                    "method": "manual",
                    "shock_method": "DiscreteMarkov"
                },
                "values": [0.5, 1.5]
            }
            
            with pytest.raises(ValueError, match="requires transition_matrix"):
                create_manual_shock_process(spec2, None)
        
        def test_unsupported_shock_method(self):
            """Test error for unsupported shock_method."""
            spec = {
                "methods": {
                    "method": "manual",
                    "shock_method": "UnsupportedType"
                },
                "transition_matrix": [[0.9, 0.1], [0.2, 0.8]],
                "values": [0.5, 1.5]
            }
            
            with pytest.raises(ValueError, match="Unsupported shock_method"):
                create_manual_shock_process(spec, None)

except ImportError as e:
    # Skip all tests if imports fail
    pytestmark = pytest.mark.skip(reason=f"Failed to import manual_shocks module: {e}") 