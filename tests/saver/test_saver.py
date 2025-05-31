"""Unit tests for ``dynx.stagecraft.saver``
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
from dynx.stagecraft.config_loader import initialize_model_Circuit, compile_all_stages
from dynx.stagecraft.saver import save_circuit    

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
        self.stages = [DummyStage(f"Stage{s}", perch_names) for s in range(n_stages)]


class DummyCircuit:
    def __init__(self, n_periods: int = 1, n_stages_per_period: int = 1):
        self.periods = [DummyPeriod(p, n_stages_per_period) for p in range(n_periods)]


# ---------------------------------------------------------------------------
# Helper builders and fixtures
# ---------------------------------------------------------------------------

def build_dummy_circuit() -> DummyCircuit:
    """Create a circuit and put some sample data into one perch."""
    circ = DummyCircuit(n_periods=2, n_stages_per_period=1)
    arvl = circ.periods[0].stages[0].perches["arvl"]
    arvl.sol = np.array([1, 2, 3])
    arvl.sim = {"foo": "bar"}
    return circ


def make_config_dir(tmp_path: Path) -> Path:
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    (cfg_dir / "master.yml").write_text("dummy: true\n")
    return cfg_dir


def patch_builder(monkeypatch, factory):
    """Monkey‑patch ConfigLoader.initialize_model_Circuit to return *factory()*."""
    import dynx.stagecraft.config_loader as _cl

    monkeypatch.setattr(_cl, "initialize_model_Circuit", lambda *a, **kw: factory())


class CustomPickle:
    def __init__(self, val: int):
        self.val = val

    def __eq__(self, other):
        return isinstance(other, CustomPickle) and other.val == self.val


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_roundtrip_identity(tmp_path: Path, monkeypatch):
    circ = build_dummy_circuit()
    cfg_dir = make_config_dir(tmp_path)
    save_root = saver.save_circuit(circ, tmp_path, cfg_dir, model_id="mani")
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
