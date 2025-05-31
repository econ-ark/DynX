"""
Unit tests for the Solution container class.
"""

import pytest
import numpy as np
import tempfile
import os
from numba import njit
from numba import typed, types

from dynx.stagecraft.solmaker import Solution


class TestSolution:
    """Test suite for Solution container."""
    
    def test_empty_construction(self):
        """Test that empty Solution has all core slots."""
        sol = Solution()
        
        # Check core arrays exist and are empty
        assert sol.vlu.shape == (0,)
        assert sol.Q.shape == (0,)
        assert sol.lambda_.shape == (0,)
        assert sol.phi.shape == (0,)
        
        # Check dictionaries exist
        assert len(sol.policy) == 0
        assert len(sol.timing) == 0
        assert len(sol.EGM.unrefined) == 0
        assert len(sol.EGM.refined) == 0
        assert len(sol.EGM.interpolated) == 0
    
    def test_policy_access(self):
        """Test adding and accessing policy entries."""
        sol = Solution()
        
        # Add via attribute
        c_array = np.ones((10, 5))
        sol.policy.c = c_array
        assert np.array_equal(sol.policy.c, c_array)
        
        # Add via mapping
        a_array = np.zeros((10, 5))
        sol.policy["a"] = a_array
        assert np.array_equal(sol.policy["a"], a_array)
        
        # Check both access methods work
        assert np.array_equal(sol.policy.c, sol.policy["c"])
        assert np.array_equal(sol.policy.a, sol.policy["a"])
    
    def test_egm_access(self):
        """Test adding and accessing EGM entries."""
        sol = Solution()
        
        # Add to different layers
        m_unref = np.linspace(0, 1, 20)
        sol.EGM.unrefined.m = m_unref
        
        m_ref = np.linspace(0, 1, 30)
        sol.EGM.refined["m"] = m_ref
        
        kappa_interp = np.ones(25)
        sol.EGM.interpolated.kappa = kappa_interp
        
        # Verify
        assert np.array_equal(sol.EGM.unrefined.m, m_unref)
        assert np.array_equal(sol.EGM.refined.m, m_ref)
        assert np.array_equal(sol.EGM.interpolated.kappa, kappa_interp)
    
    def test_core_array_assignment(self):
        """Test setting core arrays."""
        sol = Solution()
        
        # Set via attribute
        vlu = np.random.randn(10, 5, 3)
        sol.vlu = vlu
        assert np.array_equal(sol.vlu, vlu)
        assert sol._filled["vlu"]
        
        # Set via mapping
        Q = np.random.randn(10, 5, 3)
        sol["Q"] = Q
        assert np.array_equal(sol["Q"], Q)
        assert sol._filled["Q"]
    
    def test_timing_dict(self):
        """Test timing dictionary."""
        sol = Solution()
        
        # Add timing entries
        sol.timing["ue_time_avg"] = 0.5
        sol.timing["total_time"] = 1.2
        
        assert sol.timing["ue_time_avg"] == 0.5
        assert sol.timing["total_time"] == 1.2
        
        # Set entire timing dict
        new_timing = {"step1": 0.1, "step2": 0.2}
        sol.timing = new_timing
        assert dict(sol.timing) == new_timing
    
    def test_attr_mapping_equality(self):
        """Test that attribute and mapping access work correctly."""
        sol = Solution()
        
        # Set some data
        sol.vlu = np.ones((5, 3))
        sol.policy.c = np.zeros((5, 3))
        sol.EGM.refined.m = np.linspace(0, 1, 10)
        
        # Check core arrays (no reshaping)
        assert np.array_equal(sol.vlu, sol["vlu"])
        
        # Check policy - attribute access returns reshaped, dict access returns flattened
        assert sol.policy.c.shape == (5, 3)  # Attribute access preserves shape
        assert sol["policy"]["c"].shape == (5, 3)  # Dict access also reshapes in __getitem__
        assert np.array_equal(sol.policy.c, sol["policy"]["c"])
        
        # Check dictionary form of EGM
        egm_dict = sol["EGM"]
        assert np.array_equal(egm_dict["refined"]["m"], sol.EGM.refined.m)
    
    def test_save_load_roundtrip(self):
        """Test save/load preserves all data."""
        sol = Solution()
        
        # Populate with data
        sol.vlu = np.random.randn(10, 5)
        sol.Q = np.random.randn(10, 5)
        sol.policy.c = np.random.randn(10, 5)
        sol.policy.a = np.random.randn(10, 5)
        sol.timing["ue_time"] = 0.123
        sol.EGM.unrefined.m = np.linspace(0, 1, 20)
        sol.EGM.refined.m = np.linspace(0, 1, 30)
        
        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            basename = os.path.join(tmpdir, "test_sol")
            sol.save(basename)
            
            # Check files exist
            assert os.path.exists(f"{basename}.npz")
            assert os.path.exists(f"{basename}_meta.json")
            
            # Load
            sol2 = Solution.load(basename)
        
        # Verify all data preserved
        assert np.array_equal(sol.vlu, sol2.vlu)
        assert np.array_equal(sol.Q, sol2.Q)
        assert np.array_equal(sol.policy.c, sol2.policy.c)
        assert np.array_equal(sol.policy.a, sol2.policy.a)
        assert sol2.timing["ue_time"] == 0.123
        assert np.array_equal(sol.EGM.unrefined.m, sol2.EGM.unrefined.m)
        assert np.array_equal(sol.EGM.refined.m, sol2.EGM.refined.m)
    
    def test_numba_compatibility(self):
        """Test that solution can be used in numba kernels."""
        from numba import typed, types
        
        # Create typed dicts directly
        str_type = types.unicode_type
        array_type = types.float64[:]
        
        policy_dict = typed.Dict.empty(str_type, array_type)
        policy_dict["c"] = np.zeros(15)  # Flattened 5x3
        
        # Define a numba kernel
        @njit
        def access_solution(policy):
            # Access policy dict
            c_sum = policy["c"].sum()
            
            # Add to policy
            policy["test"] = np.ones(10)
            
            return c_sum
        
        # Call kernel
        c_sum = access_solution(policy_dict)
        
        assert c_sum == 0.0
        assert "test" in policy_dict
        assert np.array_equal(policy_dict["test"], np.ones(10))
    
    def test_as_dict_from_dict(self):
        """Test conversion to/from plain dict."""
        sol = Solution()
        
        # Populate
        sol.vlu = np.ones((5, 3))
        sol.policy.c = np.zeros((5, 3))
        sol.timing["test"] = 1.0
        sol.EGM.refined.m = np.linspace(0, 1, 10)
        
        # Convert to dict
        d = sol.as_dict()
        
        # Create new solution from dict
        sol2 = Solution.from_dict(d)
        
        # Verify
        assert np.array_equal(sol.vlu, sol2.vlu)
        assert np.array_equal(sol.policy.c, sol2.policy.c)
        assert sol2.timing["test"] == 1.0
        assert np.array_equal(sol.EGM.refined.m, sol2.EGM.refined.m)
    
    def test_pickle_roundtrip(self):
        """Test pickle save/load."""
        sol = Solution()
        sol.vlu = np.random.randn(5, 3)
        sol.policy.c = np.random.randn(5, 3)
        
        with tempfile.NamedTemporaryFile(suffix=".pkl") as f:
            sol.pkl(f.name)
            sol2 = Solution.from_pickle(f.name)
        
        assert np.array_equal(sol.vlu, sol2.vlu)
        assert np.array_equal(sol.policy.c, sol2.policy.c)
    
    def test_error_handling(self):
        """Test appropriate errors are raised."""
        sol = Solution()
        
        # Invalid attribute access
        with pytest.raises(AttributeError):
            _ = sol.invalid_attr
        
        # Invalid key access
        with pytest.raises(KeyError):
            _ = sol["invalid_key"]
        
        # Invalid attribute setting
        with pytest.raises(AttributeError):
            sol.invalid_attr = 1
        
        # Invalid key setting
        with pytest.raises(KeyError):
            sol["invalid_key"] = 1
        
        # Wrong type for timing
        with pytest.raises(ValueError):
            sol.timing = "not a dict"
        
        # Wrong type for policy
        with pytest.raises(ValueError):
            sol["policy"] = "not a dict" 