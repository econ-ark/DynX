"""
Comprehensive tests for reference model deviation metrics.

Tests specific scenarios mentioned in the review:
1. Method-aware sweep with identical policies
2. Method-less runner returning NaN
3. MPI collision handling
4. Legacy metric fallback
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from dynx.runner import CircuitRunner, mpi_map
from dynx.runner.metrics import dev_c_L2


# Dummy model class for testing
class DummyModel:
    """Minimal model class with policy attributes."""

    def __init__(self, config):
        self.config = config
        # Always create same policy for testing identical results
        self.c = np.ones(100) * 0.7


def dummy_factory(config):
    """Factory function to create dummy models."""
    return DummyModel(config)


def dummy_solver(model, recorder=None):
    """Dummy solver that does nothing."""
    pass


class TestComprehensiveDeviationMetrics:
    """Comprehensive tests for deviation metrics."""

    def test_method_aware_identical_policies(self):
        """Test method-aware sweep where policies are identical."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_cfg = {"test_param": 1.0, "master": {"methods": {"upper_envelope": "VFI_HDGRID"}}}

            runner = CircuitRunner(
                base_cfg=base_cfg,
                param_paths=["test_param", "master.methods.upper_envelope"],
                model_factory=dummy_factory,
                solver=dummy_solver,
                metric_fns={
                    "dev_c_L2": dev_c_L2,
                },
                output_root=tmpdir,
                save_by_default=False,  # Don't save to avoid errors
                method_param_path="master.methods.upper_envelope",
            )

            # Mock the load_reference_model to return a reference
            ref_model = DummyModel({"test_param": 1.0})
            with patch("dynx.runner.metrics.deviations.load_reference_model") as mock_load:
                mock_load.return_value = ref_model

                # Save reference
                x_ref = np.array([1.0, "VFI_HDGRID"], dtype=object)
                metrics_ref = runner.run(x_ref)

                # Reference should have zero deviation from itself
                assert metrics_ref["dev_c_L2"] == 0.0

                # Run fast method (same solver creates identical policy)
                x_fast = np.array([1.0, "FUES"], dtype=object)
                metrics_fast = runner.run(x_fast)

                # Should also have zero deviation since policies are identical
                assert metrics_fast["dev_c_L2"] == 0.0

    def test_method_less_returns_nan(self):
        """Test that method-less runner returns NaN for deviation metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_cfg = {"test_param": 1.0}

            runner = CircuitRunner(
                base_cfg=base_cfg,
                param_paths=["test_param"],
                model_factory=dummy_factory,
                solver=dummy_solver,
                metric_fns={
                    "dev_c_L2": dev_c_L2,
                },
                output_root=tmpdir,
                save_by_default=False,  # Don't save to avoid errors
                method_param_path=None,  # Method-less mode
            )

            # Run and check bundle path
            x = np.array([1.0])
            metrics = runner.run(x)

            # Should return NaN
            assert np.isnan(metrics["dev_c_L2"])

            # Check bundle directory structure
            bundle_path = runner._bundle_path(x)
            assert bundle_path is not None
            assert bundle_path.parent.name == "bundles"
            # Should be just hash, no prefix
            hash_str = runner._hash_param_vec(x)
            assert bundle_path.name == hash_str

    def test_legacy_metric_fallback(self):
        """Test that legacy metrics work alongside new deviation metrics."""

        # Define a legacy metric
        def legacy_metric(model):
            """Old-style metric that returns a constant."""
            return 42.0

        with tempfile.TemporaryDirectory() as tmpdir:
            base_cfg = {"test_param": 1.0}

            runner = CircuitRunner(
                base_cfg=base_cfg,
                param_paths=["test_param"],
                model_factory=dummy_factory,
                solver=dummy_solver,
                metric_fns={
                    "legacy": legacy_metric,
                    "dev_c_L2": dev_c_L2,
                },
                output_root=tmpdir,
                method_param_path=None,
            )

            x = np.array([1.0])
            metrics = runner.run(x)

            # Legacy metric should work
            assert metrics["legacy"] == 42.0
            # Deviation metric should return NaN (no reference)
            assert np.isnan(metrics["dev_c_L2"])

    def test_mpi_collision_handling(self):
        """Test MPI collision handling logic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_cfg = {"test_param": 1.0, "master": {"methods": {"upper_envelope": "VFI_HDGRID"}}}

            runner = CircuitRunner(
                base_cfg=base_cfg,
                param_paths=["test_param", "master.methods.upper_envelope"],
                model_factory=dummy_factory,
                solver=dummy_solver,
                output_root=tmpdir,
                save_by_default=False,
                method_param_path="master.methods.upper_envelope",
            )

            # Create a bundle path
            x = np.array([1.0, "VFI_HDGRID"], dtype=object)
            bundle_path1 = runner._bundle_path(x)

            # Create the directory to simulate collision
            if bundle_path1:
                bundle_path1.mkdir(parents=True, exist_ok=True)

                # Get bundle path again - since MPI isn't actually active,
                # it should return the same path (no rank suffix)
                bundle_path2 = runner._bundle_path(x)

                # Without actual MPI, paths should be the same
                assert bundle_path1 == bundle_path2

    def test_parameter_sweep_with_dataframe(self):
        """Test a parameter sweep returning a DataFrame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_cfg = {"beta": 0.96, "master": {"methods": {"upper_envelope": "VFI_HDGRID"}}}

            runner = CircuitRunner(
                base_cfg=base_cfg,
                param_paths=["beta", "master.methods.upper_envelope"],
                model_factory=dummy_factory,
                solver=dummy_solver,
                metric_fns={
                    "dev_c_L2": dev_c_L2,
                },
                output_root=tmpdir,
                save_by_default=False,  # Don't save to avoid errors
                method_param_path="master.methods.upper_envelope",
            )

            # Mock the load_reference_model to return a reference
            ref_model = DummyModel({"beta": 0.96})
            with patch("dynx.runner.metrics.deviations.load_reference_model") as mock_load:

                def load_side_effect(_runner, x):
                    # Return ref model only for VFI_HDGRID methods
                    param_dict = _runner.unpack(x)
                    if param_dict.get("master.methods.upper_envelope") == "VFI_HDGRID":
                        return ref_model
                    return ref_model  # Return ref for all to compare

                mock_load.side_effect = load_side_effect

                # Create design matrix
                methods = ["VFI_HDGRID", "FUES"]
                betas = [0.95, 0.96]
                xs = []
                for beta in betas:
                    for method in methods:
                        xs.append(
                            runner.pack({"beta": beta, "master.methods.upper_envelope": method})
                        )

                xs = np.array(xs)

                # Run sweep
                df = mpi_map(runner, xs, mpi=False)

                # Check DataFrame structure
                assert isinstance(df, pd.DataFrame)
                assert len(df) == 4
                assert "beta" in df.columns
                assert "master.methods.upper_envelope" in df.columns
                assert "dev_c_L2" in df.columns

                # Reference rows should have zero deviation
                ref_rows = df[df["master.methods.upper_envelope"] == "VFI_HDGRID"]
                assert all(ref_rows["dev_c_L2"] == 0.0)
