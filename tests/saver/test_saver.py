"""Unit tests for ``dynx.stagecraft.io``
=================================================
A *minimal* stub DynX hierarchy is constructed so that we can test the
save/load logic in isolation from the full framework.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pytest
import yaml

# Import from ModCraft
from dynx.stagecraft import Stage
from dynx.stagecraft.makemod import initialize_model_Circuit, compile_all_stages
from dynx.stagecraft.io import save_circuit, load_circuit, load_config

# ---------------------------------------------------------------------------
# Tiny stub graph classes
# ---------------------------------------------------------------------------

class DummyPerch:
    """Mimic just enough of StageCraft's *Perch* object."""

    def __init__(self, name: str):
        self.name = name
        self.sol = None
        self.sim = None


class DummyStage:
    """Very small stand‑in for a *Stage*."""

    def __init__(self, name: str, perch_names: List[str]):
        self.name = name
        self.perches = {pn: DummyPerch(pn) for pn in perch_names}


class DummyPeriod:
    def __init__(self, period_idx: int, n_stages: int = 1, perch_names: List[str] | None = None):
        perch_names = perch_names or ["arvl", "dcsn", "cntn"]
        self.stages = {f"Stage{s}": DummyStage(f"Stage{s}", perch_names) for s in range(n_stages)}
        self.time_index = period_idx


class DummyCircuit:
    def __init__(self, n_periods: int = 1, n_stages_per_period: int = 1):
        self.periods_list = [DummyPeriod(p, n_stages_per_period) for p in range(n_periods)]
        self.name = "TestModel"
        self.version = "1.0"


# ---------------------------------------------------------------------------
# Helper builders and fixtures
# ---------------------------------------------------------------------------

def build_dummy_circuit() -> DummyCircuit:
    """Create a circuit and put some sample data into one perch."""
    circ = DummyCircuit(n_periods=2, n_stages_per_period=1)
    arvl = circ.periods_list[0].stages["Stage0"].perches["arvl"]
    arvl.sol = np.array([1, 2, 3])
    arvl.sim = {"foo": "bar"}
    return circ


def build_dummy_circuit_no_data() -> DummyCircuit:
    """Create a circuit without any data attached."""
    return DummyCircuit(n_periods=2, n_stages_per_period=1)


def make_config_dir(tmp_path: Path) -> Path:
    """Create a folder structured config directory."""
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    
    # Create master.yml
    (cfg_dir / "master.yml").write_text("name: TestModel\nhorizon: 2\n")
    
    # Create connections.yml
    (cfg_dir / "connections.yml").write_text("intra_period:\n  forward: []\n")
    
    # Create stages directory
    stages_dir = cfg_dir / "stages"
    stages_dir.mkdir()
    
    # Create stage configs
    (stages_dir / "Stage0.yml").write_text("name: Stage0\ndummy: true\n")
    (stages_dir / "Stage1.yml").write_text("name: Stage1\ndummy: true\n")
    
    return cfg_dir


def patch_builder(monkeypatch, factory):
    """Monkey‑patch makemod.initialize_model_Circuit to return *factory()*."""
    import dynx.stagecraft.makemod as _cl

    monkeypatch.setattr(_cl, "initialize_model_Circuit", lambda *a, **kw: factory())


class CustomPickle:
    def __init__(self, val: int):
        self.val = val

    def __eq__(self, other):
        return isinstance(other, CustomPickle) and other.val == self.val


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_load_config(tmp_path: Path):
    """Test the load_config function."""
    cfg_dir = make_config_dir(tmp_path)
    
    # Test successful loading
    config = load_config(cfg_dir)
    
    assert "master" in config
    assert "stages" in config
    assert "connections" in config
    
    assert config["master"]["name"] == "TestModel"
    assert "STAGE0" in config["stages"]
    assert "STAGE1" in config["stages"]
    
    # Test missing master file
    bad_dir = tmp_path / "bad_config"
    bad_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="master.yml not found"):
        load_config(bad_dir)


def test_roundtrip_identity(tmp_path: Path, monkeypatch):
    """Test save and load circuit functionality."""
    circ = build_dummy_circuit()
    cfg_dir = make_config_dir(tmp_path)
    
    # Import io directly to use for save_circuit
    from dynx.stagecraft import io
    
    save_root = save_circuit(circ, tmp_path, cfg_dir, model_id="mani")
    manifest_path = save_root / "manifest.yml"
    assert manifest_path.exists()

    manifest = yaml.safe_load(manifest_path.read_text())

    # Collect all path strings inside manifest recursively
    def _paths(node):
        if isinstance(node, str):
            yield node
        elif isinstance(node, list):
            for v in node:
                yield from _paths(v)
        elif isinstance(node, dict):
            for v in node.values():
                yield from _paths(v)

    for rel in _paths(manifest):
        path = (save_root / rel) if not Path(rel).is_absolute() else Path(rel)
        # Only consider files (with suffix)
        if path.suffix:
            assert path.exists(), f"{rel} listed in manifest but missing"
    
    # Test loading with the new folder-based approach
    patch_builder(monkeypatch, build_dummy_circuit_no_data)
    loaded_circ = load_circuit(save_root)
    
    # Verify the loaded circuit has the expected structure
    assert len(loaded_circ.periods_list) == 2
    assert "Stage0" in loaded_circ.periods_list[0].stages
    
    # Verify the saved data was restored
    arvl = loaded_circ.periods_list[0].stages["Stage0"].perches["arvl"]
    assert arvl.sol is not None
    assert np.array_equal(arvl.sol, np.array([1, 2, 3]))
    assert arvl.sim == {"foo": "bar"}
    
    # Test loading without restoring data
    loaded_no_data = load_circuit(save_root, restore_data=False)
    arvl_no_data = loaded_no_data.periods_list[0].stages["Stage0"].perches["arvl"]
    assert arvl_no_data.sol is None
    assert arvl_no_data.sim is None


def test_folder_config_structure(tmp_path: Path):
    """Test that save_circuit creates the expected folder structure."""
    circ = build_dummy_circuit()
    cfg_dir = make_config_dir(tmp_path)
    
    save_root = save_circuit(circ, tmp_path, cfg_dir, model_id="test_model")
    
    # Check that configs were copied with the expected structure
    configs_dir = save_root / "configs"
    assert (configs_dir / "master.yml").exists()
    assert (configs_dir / "connections.yml").exists()
    assert (configs_dir / "stages").is_dir()
    assert (configs_dir / "stages" / "Stage0.yml").exists()
    assert (configs_dir / "stages" / "Stage1.yml").exists()
    
    # Test that we can load the config folder
    config = load_config(configs_dir)
    assert config["master"]["name"] == "TestModel"
    assert "STAGE0" in config["stages"]
    assert "STAGE1" in config["stages"]
