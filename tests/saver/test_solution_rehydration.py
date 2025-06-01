"""Test that Solution objects are properly re-hydrated when loading."""

from pathlib import Path
import pickle
import pytest
from dynx.stagecraft.io import load_circuit, save_circuit, _dump_object
from dynx.stagecraft.solmaker import Solution
import numpy as np


def test_solution_rehydration(tmp_path: Path, monkeypatch):
    """Test that Solution dicts are re-hydrated to proper Solution objects."""
    
    # Create a mock Solution-like dict (simulating what Solution.pkl() would save)
    solution_dict = {
        "policy": {"c": np.array([1.0, 2.0, 3.0])},
        "EGM": {
            "unrefined": {"e": np.array([0.1, 0.2, 0.3])},
            "refined": {"e": np.array([0.11, 0.21, 0.31])}
        },
        "timing": {"solve_time": 1.23, "iterations": 10},
        "meta": {"solver": "test"}
    }
    
    # Create a minimal circuit structure for testing
    from tests.saver.test_saver import build_dummy_circuit, make_config_dir, patch_builder
    
    # Build a dummy circuit
    circ = build_dummy_circuit()
    cfg_dir = make_config_dir(tmp_path)
    
    # Manually place the solution dict on a perch (simulating what happens when pkl() saves as dict)
    circ.periods_list[0].stages["Stage0"].perches["arvl"].sol = Solution.from_dict(solution_dict)
    
    # Save the circuit
    save_root = save_circuit(circ, tmp_path, cfg_dir, model_id="test_rehydration")
    
    # Now manually corrupt the saved solution to be just a dict (simulate old behavior)
    sol_path = save_root / "data" / "period_0" / "Stage0" / "arvl" / "sol.pkl"
    with sol_path.open("wb") as f:
        pickle.dump(solution_dict, f, protocol=5)  # Save as plain dict
    
    # Load the circuit
    patch_builder(monkeypatch, build_dummy_circuit)
    loaded_circ = load_circuit(save_root)
    
    # Verify the solution was re-hydrated to a proper Solution object
    arvl = loaded_circ.periods_list[0].stages["Stage0"].perches["arvl"]
    assert arvl.sol is not None
    assert isinstance(arvl.sol, Solution), "Solution should be re-hydrated from dict"
    
    # Verify we can access Solution properties
    assert hasattr(arvl.sol, "policy")
    assert hasattr(arvl.sol, "EGM")
    assert hasattr(arvl.sol.EGM, "unrefined")
    
    # Verify data integrity
    assert np.array_equal(arvl.sol.policy["c"], np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(arvl.sol.EGM.unrefined.e, np.array([0.1, 0.2, 0.3]))
    assert arvl.sol.timing["solve_time"] == 1.23


def test_non_solution_dict_preserved(tmp_path: Path, monkeypatch):
    """Test that non-Solution dicts are preserved as-is."""
    
    # Create a regular dict that doesn't look like a Solution
    regular_dict = {
        "some_key": "some_value",
        "numbers": [1, 2, 3],
        "nested": {"data": True}
    }
    
    from tests.saver.test_saver import build_dummy_circuit, make_config_dir, patch_builder
    
    # Build a dummy circuit
    circ = build_dummy_circuit()
    cfg_dir = make_config_dir(tmp_path)
    
    # Place the regular dict on a perch
    circ.periods_list[0].stages["Stage0"].perches["arvl"].sol = regular_dict
    
    # Save the circuit
    save_root = save_circuit(circ, tmp_path, cfg_dir, model_id="test_regular_dict")
    
    # Load the circuit
    patch_builder(monkeypatch, build_dummy_circuit)
    loaded_circ = load_circuit(save_root)
    
    # Verify the dict was preserved as-is (not converted)
    arvl = loaded_circ.periods_list[0].stages["Stage0"].perches["arvl"]
    assert isinstance(arvl.sol, dict), "Regular dict should remain a dict"
    assert not isinstance(arvl.sol, Solution), "Regular dict should not be converted to Solution"
    assert arvl.sol == regular_dict


def test_branch_dict_solutions_rehydration(tmp_path: Path, monkeypatch):
    """Test that branch dictionaries with Solution values are properly handled."""
    
    # Create Solutions for each branch
    owner_solution_dict = {
        "policy": {"c": np.array([1.0, 2.0, 3.0])},
        "EGM": {"unrefined": {"e": np.array([0.1, 0.2])}, "refined": {"e": np.array([0.11, 0.21])}},
        "timing": {"solve_time": 1.0, "iterations": 5},
    }
    
    renter_solution_dict = {
        "policy": {"c": np.array([4.0, 5.0, 6.0])},
        "EGM": {"unrefined": {"e": np.array([0.3, 0.4])}, "refined": {"e": np.array([0.31, 0.41])}},
        "timing": {"solve_time": 2.0, "iterations": 7},
    }
    
    # Create branch dictionary with Solution objects
    branch_dict = {
        "from_owner": Solution.from_dict(owner_solution_dict),
        "from_renter": Solution.from_dict(renter_solution_dict)
    }
    
    from tests.saver.test_saver import build_dummy_circuit, make_config_dir, patch_builder
    
    # Build a dummy circuit
    circ = build_dummy_circuit()
    cfg_dir = make_config_dir(tmp_path)
    
    # Place the branch dict on a perch (simulating TENU.cntn.sol)
    circ.periods_list[0].stages["Stage0"].perches["cntn"].sol = branch_dict
    
    # Save the circuit
    save_root = save_circuit(circ, tmp_path, cfg_dir, model_id="test_branch_dict")
    
    # Load the circuit
    patch_builder(monkeypatch, build_dummy_circuit)
    loaded_circ = load_circuit(save_root)
    
    # Verify the branch dict was loaded correctly
    cntn = loaded_circ.periods_list[0].stages["Stage0"].perches["cntn"]
    assert cntn.sol is not None
    assert isinstance(cntn.sol, dict), "Branch dict should remain a dict"
    assert "from_owner" in cntn.sol
    assert "from_renter" in cntn.sol
    
    # Verify each branch was re-hydrated to a Solution object
    assert isinstance(cntn.sol["from_owner"], Solution), "Branch should be re-hydrated to Solution"
    assert isinstance(cntn.sol["from_renter"], Solution), "Branch should be re-hydrated to Solution"
    
    # Verify data integrity
    assert np.array_equal(cntn.sol["from_owner"].policy["c"], np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(cntn.sol["from_renter"].policy["c"], np.array([4.0, 5.0, 6.0]))
    assert cntn.sol["from_owner"].timing["solve_time"] == 1.0
    assert cntn.sol["from_renter"].timing["iterations"] == 7
    
    # Verify we can access nested attributes
    assert hasattr(cntn.sol["from_owner"].EGM, "unrefined")
    assert np.array_equal(cntn.sol["from_owner"].EGM.unrefined.e, np.array([0.1, 0.2])) 