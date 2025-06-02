"""
Unit tests for reference model deviation metrics.

Tests the method-aware bundle storage and deviation metric calculation.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from dynx.runner import CircuitRunner
from dynx.runner.metrics import dev_c_L2, dev_c_Linf
from dynx.runner.reference_utils import DEFAULT_REF_METHOD, load_reference_model


# Dummy model class for testing
class DummyModel:
    """Minimal model class with policy attributes."""

    def __init__(self, config):
        self.config = config
        # Extract test parameter to create different policies
        self.test_param = config.get("test_param", 1.0)

        # Create dummy policies based on test_param
        size = 100
        self.c = np.ones(size) * self.test_param
        self.a = np.ones(size) * self.test_param * 2
        self.v = np.ones(size) * self.test_param * 3
        self.pol = np.ones(size) * self.test_param * 4


def dummy_factory(config):
    """Factory function to create dummy models."""
    return DummyModel(config)


def dummy_solver(model, recorder=None):
    """Dummy solver that does nothing."""
    pass


class TestMethodAwareDeviation:
    """Test method-aware bundle storage and deviation metrics."""

    def test_method_aware_bundle_paths(self):
        """Test that bundle paths are computed correctly with method subdirectories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_cfg = {"test_param": 1.0, "master": {"methods": {"upper_envelope": "VFI_HDGRID"}}}

            runner = CircuitRunner(
                base_cfg=base_cfg,
                param_paths=["test_param", "master.methods.upper_envelope"],
                model_factory=dummy_factory,
                solver=dummy_solver,
                output_root=tmpdir,
                save_by_default=False,  # Don't actually save
                method_param_path="master.methods.upper_envelope",
            )

            # Create parameter vectors
            x_ref = np.array([1.5, "VFI_HDGRID"], dtype=object)
            x_alt = np.array([1.5, "FUES"], dtype=object)

            # Check bundle paths are computed correctly
            bundle_path_ref = runner._bundle_path(x_ref)
            bundle_path_alt = runner._bundle_path(x_alt)

            assert bundle_path_ref is not None
            assert bundle_path_alt is not None

            # Method should be in path
            assert "VFI_HDGRID" in str(bundle_path_ref)
            assert "FUES" in str(bundle_path_alt)

            # Same hash directory (method excluded from hash)
            assert bundle_path_ref.parent == bundle_path_alt.parent
            assert bundle_path_ref.parent.name == runner._hash_param_vec(x_ref)
            assert bundle_path_ref.parent.name == runner._hash_param_vec(x_alt)

    def test_deviation_metrics_with_reference(self):
        """Test deviation metrics when reference model exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_cfg = {"test_param": 1.0, "master": {"methods": {"upper_envelope": "VFI_HDGRID"}}}

            runner = CircuitRunner(
                base_cfg=base_cfg,
                param_paths=["test_param", "master.methods.upper_envelope"],
                model_factory=dummy_factory,
                solver=dummy_solver,
                metric_fns={
                    "dev_c_L2": dev_c_L2,
                    "dev_c_Linf": dev_c_Linf,
                },
                output_root=tmpdir,
                save_by_default=False,
                method_param_path="master.methods.upper_envelope",
            )

            # Mock the load_reference_model to return a reference model
            x_ref = np.array([1.0, DEFAULT_REF_METHOD], dtype=object)
            x_alt = np.array([1.0, "FUES"], dtype=object)

            # Create reference model
            ref_model = DummyModel({"test_param": 1.0})

            # Patch load_reference_model to return our reference
            with patch("dynx.runner.metrics.deviations.load_reference_model") as mock_load:
                # When called with x_ref, return ref_model (comparing to itself)
                # When called with x_alt, also return ref_model (for comparison)
                mock_load.return_value = ref_model

                # Reference should have zero deviation from itself
                metrics_ref, _ = runner.run(x_ref, return_model=True)
                assert metrics_ref["dev_c_L2"] == 0.0
                assert metrics_ref["dev_c_Linf"] == 0.0

                # Now test with different model (different policy values)
                runner.model_factory = lambda cfg: DummyModel({"test_param": 1.5})
                metrics_alt = runner.run(x_alt)

                # Should have non-zero deviation
                assert metrics_alt["dev_c_L2"] > 0.0
                assert metrics_alt["dev_c_Linf"] > 0.0

                # L2 norm should be sqrt(100 * 0.5^2) = 5.0
                expected_diff = 0.5  # 1.5 - 1.0
                expected_l2 = np.sqrt(100 * expected_diff**2)
                assert np.isclose(metrics_alt["dev_c_L2"], expected_l2)

                # Linf norm should be 0.5
                assert np.isclose(metrics_alt["dev_c_Linf"], expected_diff)

    def test_deviation_metrics_without_reference(self):
        """Test deviation metrics return NaN when reference doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_cfg = {"test_param": 1.0, "master": {"methods": {"upper_envelope": "FUES"}}}

            runner = CircuitRunner(
                base_cfg=base_cfg,
                param_paths=["test_param", "master.methods.upper_envelope"],
                model_factory=dummy_factory,
                solver=dummy_solver,
                metric_fns={
                    "dev_c_L2": dev_c_L2,
                },
                output_root=tmpdir,
                save_by_default=False,
                method_param_path="master.methods.upper_envelope",
            )

            # Run without reference existing (load_reference_model returns None by default)
            x = np.array([1.0, "FUES"], dtype=object)
            metrics = runner.run(x)

            # Should return NaN
            assert np.isnan(metrics["dev_c_L2"])

    def test_method_less_mode(self):
        """Test that method_param_path=None creates flat bundle structure."""
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
                save_by_default=False,
                method_param_path=None,  # Method-less mode
            )

            # Run twice with different parameters
            x1 = np.array([1.0])
            x2 = np.array([2.0])

            metrics1 = runner.run(x1)
            metrics2 = runner.run(x2)

            # Both should return NaN (no reference in method-less mode)
            assert np.isnan(metrics1["dev_c_L2"])
            assert np.isnan(metrics2["dev_c_L2"])

            # Check bundle paths are computed correctly
            bundle_path1 = runner._bundle_path(x1)
            bundle_path2 = runner._bundle_path(x2)

            assert bundle_path1 is not None
            assert bundle_path2 is not None

            # Should be in bundles/ directory with just hash (no prefix)
            assert bundle_path1.parent.name == "bundles"
            assert bundle_path2.parent.name == "bundles"
            # Bundle names should be just the hash
            assert not bundle_path1.name.startswith(runner.bundle_prefix)
            assert not bundle_path2.name.startswith(runner.bundle_prefix)

            # Different hashes for different parameters
            assert bundle_path1.name != bundle_path2.name

    def test_backward_compatibility(self):
        """Test that old-style metrics still work."""

        def old_metric(model):
            """Old-style metric that only accepts model."""
            return model.test_param * 10

        with tempfile.TemporaryDirectory() as tmpdir:
            base_cfg = {"test_param": 2.5}

            runner = CircuitRunner(
                base_cfg=base_cfg,
                param_paths=["test_param"],
                model_factory=dummy_factory,
                solver=dummy_solver,
                metric_fns={
                    "old_metric": old_metric,
                    "dev_c_L2": dev_c_L2,
                },
                output_root=tmpdir,
                method_param_path=None,
            )

            x = np.array([2.5])
            metrics = runner.run(x)

            # Old metric should work
            assert metrics["old_metric"] == 25.0

            # New metric should return NaN (no reference)
            assert np.isnan(metrics["dev_c_L2"])
