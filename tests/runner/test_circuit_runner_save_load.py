"""
Tests for CircuitRunner disk-based save/load functionality.

Tests cover the specification from prompt_v0.1.18dev5.md including:
- Bundle creation and naming
- Save/load behavior
- Mode parameter control  
- MPI compatibility
- Manifest generation
"""

import pytest
import tempfile
from pathlib import Path
import numpy as np
import yaml
from unittest.mock import Mock, patch

import sys
from pathlib import Path
codebase_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(codebase_dir))

from dynx.runner import CircuitRunner


class DummyModel:
    """Mock model for testing."""
    def __init__(self, config):
        self.config = config
        self.name = "test_model"
        self.version = "dev" 
        self.periods_list = []


def dummy_model_factory(cfg):
    return DummyModel(cfg)


def dummy_solver(model, recorder=None):
    if recorder:
        recorder.add(solve_time=1.0, iterations=10)


def dummy_metric(model):
    return model.config.get("beta", 0.95) * 100


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for bundle testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def base_runner_config():
    """Base configuration for CircuitRunner."""
    return {
        "base_cfg": {
            "beta": 0.95,
            "master": {"periods": 3},
            "stages": {"TEST": {"type": "test"}},
            "connections": {}
        },
        "param_paths": ["beta"],
        "model_factory": dummy_model_factory,
        "solver": dummy_solver,
        "metric_fns": {"test_metric": dummy_metric}
    }


class TestBundlePathGeneration:
    """Test bundle path generation and hashing."""
    
    def test_bundle_path_without_output_root(self, base_runner_config):
        """Test that bundle_path returns None when output_root is not set."""
        runner = CircuitRunner(**base_runner_config)
        x = np.array([0.97], dtype=object)
        
        assert runner._bundle_path(x) is None
    
    def test_bundle_path_with_output_root(self, base_runner_config, temp_output_dir):
        """Test bundle path generation with output_root."""
        runner = CircuitRunner(
            **base_runner_config,
            output_root=temp_output_dir,
            bundle_prefix="test",
            hash_len=8
        )
        x = np.array([0.97], dtype=object)
        
        bundle_path = runner._bundle_path(x)
        assert bundle_path is not None
        assert bundle_path.parent == temp_output_dir
        assert bundle_path.name.startswith("test_")
        assert len(bundle_path.name.split("_")[1]) == 8  # hash length
    
    def test_hash_consistency(self, base_runner_config, temp_output_dir):
        """Test that same parameter vector produces same hash."""
        runner = CircuitRunner(
            **base_runner_config,
            output_root=temp_output_dir,
            hash_len=8
        )
        x = np.array([0.97], dtype=object)
        
        hash1 = runner._hash_param_vec(x)
        hash2 = runner._hash_param_vec(x)
        assert hash1 == hash2
        assert len(hash1) == 8
    
    def test_different_params_different_hash(self, base_runner_config, temp_output_dir):
        """Test that different parameters produce different hashes."""
        runner = CircuitRunner(
            **base_runner_config,
            output_root=temp_output_dir
        )
        x1 = np.array([0.97], dtype=object)
        x2 = np.array([0.98], dtype=object)
        
        hash1 = runner._hash_param_vec(x1)
        hash2 = runner._hash_param_vec(x2)
        assert hash1 != hash2


class TestSaveLoadBehavior:
    """Test save and load behavior."""
    
    @patch('dynx.runner.circuit_runner.HAS_IO', True)
    @patch('dynx.runner.circuit_runner.save_circuit')
    def test_save_by_default_true(self, mock_save_circuit, base_runner_config, temp_output_dir):
        """Test that save_by_default=True saves models."""
        mock_save_circuit.return_value = temp_output_dir / "test_bundle"
        
        runner = CircuitRunner(
            **base_runner_config,
            output_root=temp_output_dir,
            save_by_default=True
        )
        x = np.array([0.97], dtype=object)
        
        runner.run(x)
        
        # Should have called save_circuit
        mock_save_circuit.assert_called_once()
    
    @patch('dynx.runner.circuit_runner.HAS_IO', True)
    @patch('dynx.runner.circuit_runner.save_circuit')
    def test_save_model_override(self, mock_save_circuit, base_runner_config, temp_output_dir):
        """Test that save_model parameter overrides save_by_default."""
        mock_save_circuit.return_value = temp_output_dir / "test_bundle"
        
        runner = CircuitRunner(
            **base_runner_config,
            output_root=temp_output_dir,
            save_by_default=False  # Default is False
        )
        x = np.array([0.97], dtype=object)
        
        # Should not save
        runner.run(x)
        mock_save_circuit.assert_not_called()
        
        # Should save when overridden
        runner.run(x, save_model=True)
        mock_save_circuit.assert_called_once()
    
    @patch('dynx.runner.circuit_runner.HAS_IO', True)
    @patch('dynx.runner.circuit_runner.load_circuit')
    def test_load_if_exists(self, mock_load_circuit, base_runner_config, temp_output_dir):
        """Test load_if_exists functionality."""
        # Create a fake bundle directory
        bundle_dir = temp_output_dir / "run_12345678"
        bundle_dir.mkdir()
        
        mock_model = DummyModel({"beta": 0.97})
        mock_load_circuit.return_value = mock_model
        
        runner = CircuitRunner(
            **base_runner_config,
            output_root=temp_output_dir,
            load_if_exists=True,
            bundle_prefix="run",
            hash_len=8
        )
        
        # Mock _hash_param_vec to return known hash
        runner._hash_param_vec = Mock(return_value="12345678")
        
        x = np.array([0.97], dtype=object)
        result = runner.run(x)
        
        # Should have attempted to load
        mock_load_circuit.assert_called_once()
        assert "test_metric" in result


class TestModeParameter:
    """Test __runner.mode parameter functionality."""
    
    def test_mode_parameter_extraction(self, base_runner_config, temp_output_dir):
        """Test that mode parameter is properly extracted and removed."""
        runner = CircuitRunner(
            **base_runner_config,
            param_paths=["beta", "__runner.mode"],
            output_root=temp_output_dir
        )
        
        x = np.array([0.97, "solve"], dtype=object)
        
        # Check that unpack properly handles the mode parameter
        param_dict = runner.unpack(x)
        assert param_dict["beta"] == 0.97
        assert param_dict["__runner.mode"] == "solve"
    
    @patch('dynx.runner.circuit_runner.HAS_IO', True)
    @patch('dynx.runner.circuit_runner.load_circuit')
    def test_mode_load_forces_loading(self, mock_load_circuit, base_runner_config, temp_output_dir):
        """Test that mode='load' forces loading even when load_if_exists=False."""
        bundle_dir = temp_output_dir / "run_12345678"
        bundle_dir.mkdir()
        
        mock_model = DummyModel({"beta": 0.97})
        mock_load_circuit.return_value = mock_model
        
        runner = CircuitRunner(
            **base_runner_config,
            param_paths=["beta", "__runner.mode"],
            output_root=temp_output_dir,
            load_if_exists=False,  # Normally wouldn't load
            bundle_prefix="run"
        )
        runner._hash_param_vec = Mock(return_value="12345678")
        
        x = np.array([0.97, "load"], dtype=object)
        runner.run(x)
        
        # Should have loaded despite load_if_exists=False
        mock_load_circuit.assert_called_once()
    
    @patch('dynx.runner.circuit_runner.HAS_IO', True)
    @patch('dynx.runner.circuit_runner.load_circuit')
    def test_mode_solve_prevents_loading(self, mock_load_circuit, base_runner_config, temp_output_dir):
        """Test that mode='solve' prevents loading even when load_if_exists=True."""
        bundle_dir = temp_output_dir / "run_12345678"
        bundle_dir.mkdir()
        
        runner = CircuitRunner(
            **base_runner_config,
            param_paths=["beta", "__runner.mode"],
            output_root=temp_output_dir,
            load_if_exists=True,  # Normally would load
            bundle_prefix="run"
        )
        runner._hash_param_vec = Mock(return_value="12345678")
        
        x = np.array([0.97, "solve"], dtype=object)
        runner.run(x)
        
        # Should not have loaded despite load_if_exists=True
        mock_load_circuit.assert_not_called()


class TestManifestGeneration:
    """Test manifest file generation."""
    
    @patch('dynx.runner.circuit_runner.HAS_IO', True)
    @patch('dynx.runner.circuit_runner.save_circuit')
    def test_manifest_content(self, mock_save_circuit, base_runner_config, temp_output_dir):
        """Test that manifest contains required fields."""
        bundle_dir = temp_output_dir / "housing_12345678"
        bundle_dir.mkdir()
        manifest_path = bundle_dir / "manifest.yml"
        
        # Create initial manifest
        manifest_path.write_text(yaml.safe_dump({"model_id": "housing_12345678"}))
        mock_save_circuit.return_value = bundle_dir
        
        runner = CircuitRunner(
            **base_runner_config,
            param_paths=["beta", "__runner.mode"],
            output_root=temp_output_dir,
            bundle_prefix="housing",
            save_by_default=True
        )
        runner._hash_param_vec = Mock(return_value="12345678")
        
        x = np.array([0.97, "solve"], dtype=object)
        runner.run(x)
        
        # Check manifest was updated
        manifest = yaml.safe_load(manifest_path.read_text())
        assert "bundle" in manifest
        assert manifest["bundle"]["hash"] == "12345678"
        assert manifest["bundle"]["prefix"] == "housing"
        assert "saved_by_rank" in manifest["bundle"]
        assert "parameters" in manifest
        assert manifest["parameters"]["beta"] == 0.97
        assert manifest["parameters"]["__runner.mode"] == "solve"


class TestIOAvailability:
    """Test behavior when IO functionality is not available."""
    
    @patch('dynx.runner.circuit_runner.HAS_IO', False)
    def test_warning_when_io_unavailable(self, base_runner_config, temp_output_dir):
        """Test that warning is issued when IO is unavailable but bundle features requested."""
        with pytest.warns(UserWarning, match="Bundle save/load functionality requires"):
            CircuitRunner(
                **base_runner_config,
                output_root=temp_output_dir,
                save_by_default=True
            )
    
    @patch('dynx.runner.circuit_runner.HAS_IO', False)
    def test_no_warning_when_no_bundle_features(self, base_runner_config):
        """Test no warning when IO unavailable but no bundle features requested."""
        # Should not raise any warnings
        CircuitRunner(**base_runner_config)


class TestReturnModelParameter:
    """Test return_model parameter functionality."""
    
    def test_return_model_false(self, base_runner_config):
        """Test return_model=False returns only metrics."""
        runner = CircuitRunner(**base_runner_config)
        x = np.array([0.97], dtype=object)
        
        result = runner.run(x, return_model=False)
        assert isinstance(result, dict)
        assert "test_metric" in result
    
    def test_return_model_true(self, base_runner_config):
        """Test return_model=True returns metrics and model."""
        runner = CircuitRunner(**base_runner_config)
        x = np.array([0.97], dtype=object)
        
        result = runner.run(x, return_model=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        metrics, model = result
        assert isinstance(metrics, dict)
        assert "test_metric" in metrics
        assert isinstance(model, DummyModel)
        assert model.config["beta"] == 0.97


class TestDesignMatrixCSV:
    """Test design matrix CSV functionality."""
    
    def test_design_matrix_creation_single_proc(self, base_runner_config, temp_output_dir):
        """Test that design_matrix.csv is created after a parameter sweep."""
        import pandas as pd
        from dynx.runner import mpi_map
        
        runner = CircuitRunner(
            **base_runner_config,
            output_root=temp_output_dir,
            bundle_prefix="test",
            save_by_default=True
        )
        
        # Create a small design matrix
        xs = np.array([
            [0.95],
            [0.96], 
            [0.97]
        ], dtype=object)
        
        # Run the sweep
        df = mpi_map(runner, xs, return_models=False, mpi=False)
        
        # Check that design matrix CSV was created
        design_csv_path = temp_output_dir / "design_matrix.csv"
        assert design_csv_path.exists(), "design_matrix.csv should be created"
        
        # Load and verify contents
        design_df = pd.read_csv(design_csv_path)
        assert len(design_df) == 3, "Should have 3 rows"
        assert "beta" in design_df.columns, "Should have beta column"
        assert "param_hash" in design_df.columns, "Should have param_hash column"
        assert "bundle_dir" in design_df.columns, "Should have bundle_dir column"
        
        # Verify all bundle directories exist
        for bundle_dir in design_df["bundle_dir"]:
            bundle_path = temp_output_dir / bundle_dir
            assert bundle_path.exists(), f"Bundle directory {bundle_dir} should exist"
    
    def test_design_matrix_deduplication(self, base_runner_config, temp_output_dir):
        """Test that design matrix avoids duplicates when appending."""
        import pandas as pd
        from dynx.runner import mpi_map
        
        runner = CircuitRunner(
            **base_runner_config,
            output_root=temp_output_dir,
            bundle_prefix="test",
            save_by_default=True
        )
        
        # Create design matrix
        xs = np.array([[0.95], [0.96]], dtype=object)
        
        # Run sweep twice
        mpi_map(runner, xs, return_models=False, mpi=False)
        mpi_map(runner, xs, return_models=False, mpi=False)
        
        # Check that design matrix doesn't have duplicates
        design_csv_path = temp_output_dir / "design_matrix.csv"
        design_df = pd.read_csv(design_csv_path)
        
        # Should still only have 2 unique rows
        assert len(design_df) == 2, "Should not have duplicate rows"
        assert len(design_df["param_hash"].unique()) == 2, "Should have 2 unique hashes"
    
    def test_design_matrix_no_output_root(self, base_runner_config):
        """Test that no design matrix is created when output_root is None."""
        from dynx.runner import mpi_map
        
        runner = CircuitRunner(**base_runner_config)  # No output_root
        xs = np.array([[0.95]], dtype=object)
        
        # Should not raise any errors
        df = mpi_map(runner, xs, return_models=False, mpi=False)
        assert len(df) == 1
    
    @patch('dynx.runner.circuit_runner.HAS_IO', True)
    @patch('dynx.runner.circuit_runner.save_circuit')
    @patch('dynx.runner.circuit_runner.load_circuit')
    def test_design_matrix_with_load_only_rows(self, mock_load_circuit, mock_save_circuit, base_runner_config, temp_output_dir):
        """Test that design matrix CSV is created correctly even when some rows are load-only."""
        import pandas as pd
        from dynx.runner import mpi_map
        
        # Mock load_circuit to return a model for existing bundles
        mock_model = DummyModel({"beta": 0.95})
        mock_load_circuit.return_value = mock_model
        mock_save_circuit.return_value = temp_output_dir / "test_bundle"
        
        # Create fake bundle directories to simulate existing bundles
        bundle_dir1 = temp_output_dir / "test_12345678"
        bundle_dir2 = temp_output_dir / "test_abcdefgh"
        bundle_dir1.mkdir()
        bundle_dir2.mkdir()
        
        runner = CircuitRunner(
            **base_runner_config,
            output_root=temp_output_dir,
            bundle_prefix="test",
            load_if_exists=True,  # Will load existing bundles
            save_by_default=True  # Will save new bundles
        )
        
        # Mock hash function to return predictable values
        def mock_hash(x):
            if x[0] == 0.95:
                return "12345678"  # Exists, will load
            elif x[0] == 0.96:
                return "abcdefgh"  # Exists, will load
            else:
                return "newbundl"  # Doesn't exist, will save
        
        runner._hash_param_vec = mock_hash
        
        # Create design matrix with mix of load and save scenarios
        xs = np.array([
            [0.95],  # Will load existing bundle
            [0.96],  # Will load existing bundle
            [0.97]   # Will create new bundle
        ], dtype=object)
        
        # Run the sweep
        df = mpi_map(runner, xs, return_models=False, mpi=False)
        
        # Check that design matrix CSV was created
        design_csv_path = temp_output_dir / "design_matrix.csv"
        assert design_csv_path.exists(), "design_matrix.csv should be created"
        
        # Load and verify contents
        design_df = pd.read_csv(design_csv_path)
        assert len(design_df) == 3, "Should have 3 rows (including load-only rows)"
        
        # Verify all expected bundle directories are listed
        expected_bundles = {"test_12345678", "test_abcdefgh", "test_newbundl"}
        actual_bundles = set(design_df["bundle_dir"])
        assert actual_bundles == expected_bundles, f"Expected {expected_bundles}, got {actual_bundles}"
        
        # Verify load_circuit was called for existing bundles
        assert mock_load_circuit.call_count == 2, "Should have attempted to load 2 existing bundles"
        
        # Verify save_circuit was called for new bundle
        assert mock_save_circuit.call_count == 1, "Should have saved 1 new bundle" 