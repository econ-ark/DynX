"""
dynx.stagecraft.saver
---------------------
Persist *artifacts* (configs + each perch's .sol / .sim) of a solved
StageCraft ModelCircuit.  The ModelCircuit itself is *not* pickled; it is
re-built from the copied configs when `load_circuit` is called.
"""

from __future__ import annotations

import datetime as _dt
import pickle
import logging
import shutil
from pathlib import Path
from typing import Any, List

import yaml

logger = logging.getLogger("dynx.stagecraft.saver")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _safe_pickle(obj: Any, path: Path) -> None:
    """Pickle *obj* with protocol 5."""
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=5)


def _dump_object(obj: Any, path: Path) -> None:
    """
    Best-effort pickle helper.

    Order of preference
    -------------------
    1. obj.pkl(path)             # e.g. dynx.stagecraft.solmaker.Solution
    2. pickle obj.as_dict()      # if obj exposes .as_dict()
    3. plain pickle              # protocol 5
    """
    try:
        if hasattr(obj, "pkl") and callable(obj.pkl):
            obj.pkl(path)                      # type: ignore[attr-defined]
            return

        if hasattr(obj, "as_dict") and callable(obj.as_dict):
            _safe_pickle(obj.as_dict(), path)
            return

        _safe_pickle(obj, path)

    except (pickle.PicklingError, TypeError) as exc:
        logger.warning("Skipping %s – not pickle-able (%s)", path.name, exc)



def _copy_configs(src: str | Path | dict | List[Any], dst: Path) -> None:
    """Copy directories / files *or* dump Python‐dict configs to YAML."""

    dst.mkdir(parents=True, exist_ok=True)

    # normalise to an iterable
    if not isinstance(src, (list, tuple)):
        src = [src]

    for item in src:
        # ------------------------------------------------------------------
        # 1)  Python dict → write a fresh YAML
        # ------------------------------------------------------------------
        if isinstance(item, dict):
            # Decide a filename
            if item.get("model_type", "").lower() == "master" or "horizon" in item:
                out = dst / "master.yml"
            elif item.get("connections") is True or "edges" in item:
                out = dst / "connections.yml"
            else:  # assume stage config
                stage_name = item.get("name", "stage").upper()
                out = dst / f"{stage_name}_stage.yml"

            with out.open("w") as f:
                yaml.safe_dump(item, f, sort_keys=False)
            continue  # go to next item

        # ------------------------------------------------------------------
        # 2)  Pathlike sources (dir or single file)
        # ------------------------------------------------------------------
        p = Path(item).expanduser().resolve()
        if p.is_dir():
            for fn in p.rglob("*"):
                if fn.suffix.lower() in {".yml", ".yaml", ".json", ".toml"}:
                    rel = fn.relative_to(p)
                    (dst / rel.parent).mkdir(parents=True, exist_ok=True)
                    shutil.copy2(fn, dst / rel)
        elif p.is_file():
            shutil.copy2(p, dst / p.name)
        else:
            logger.warning("Config source %s not recognised; skipped", item)



def stamp_model_id(circuit: Any, version: str | None = None) -> str:
    root = getattr(circuit, "name", "model")
    today = _dt.date.today().isoformat()
    if version is None:
        version = getattr(circuit, "version", "dev")
    return f"{root}_{version}_{today}"


# ---------------------------------------------------------------------------
# main API
# ---------------------------------------------------------------------------

def save_circuit(
    circuit: Any,
    dest: str | Path,
    config_src: str | Path | List[Path],
    model_id: str | None = None,
) -> Path:
    """
    Save configs and every *.sol / .sim* attached to a ModelCircuit.

    Returns the path to ``<dest>/<model_id>``.
    """
    dest = Path(dest).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    if model_id is None:
        model_id = stamp_model_id(circuit)

    target_dir = dest / model_id
    if target_dir.exists():
        logger.warning("Overwriting existing directory: %s", target_dir)
        shutil.rmtree(target_dir)
    target_dir.mkdir()

    # 1) copy configs verbatim
    _copy_configs(config_src, target_dir / "configs")

    # 2) discover periods robustly
    if hasattr(circuit, "periods_list"):
        periods_iter = list(circuit.periods_list)
    elif hasattr(circuit, "periods"):
        pa = circuit.periods
        periods_iter = list(pa.values() if isinstance(pa, dict) else pa)
    else:
        raise AttributeError("Circuit has no 'periods_list' or 'periods' attribute")

    if not periods_iter:
        raise ValueError("Circuit contains zero periods; nothing to save")

    data_dir = target_dir / "data"
    files_written: list[str] = []

    # 3) walk periods → stages → perches
    for p_idx, period in enumerate(periods_iter):
        period_dir = data_dir / f"period_{p_idx}"

        stages_attr = getattr(period, "stages", {})
        stages_iter = stages_attr.values() if isinstance(stages_attr, dict) else stages_attr

        for stage in stages_iter:
            stage_dir = period_dir / stage.name

            for perch_name, perch in getattr(stage, "perches", {}).items():
                perch_dir = stage_dir / perch_name
                perch_dir.mkdir(parents=True, exist_ok=True)

                # ---- sol ---------------------------------------------------
                if getattr(perch, "sol", None) is not None:
                    sol_path = perch_dir / "sol.pkl"
                    _dump_object(perch.sol, sol_path)
                    files_written.append(sol_path.relative_to(target_dir).as_posix())

                # ---- sim / dist -------------------------------------------
                sim_obj = (
                    getattr(perch, "sim", None)
                    if hasattr(perch, "sim")
                    else getattr(perch, "dist", None)
                )
                if sim_obj is not None:
                    sim_path = perch_dir / "sim.pkl"
                    _dump_object(sim_obj, sim_path)
                    files_written.append(sim_path.relative_to(target_dir).as_posix())

    # 4) write manifest
    manifest = {
        "model_id": model_id,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "files": sorted(files_written),
    }
    (target_dir / "manifest.yml").write_text(yaml.safe_dump(manifest, sort_keys=False))

    return target_dir


# ---------------------------------------------------------------------------  
# Re-create a ModelCircuit from a folder produced by `save_circuit`  
# ---------------------------------------------------------------------------
def load_circuit(saved_dir: str | Path) -> Any:
    """
    Reload a model folder written by `save_circuit`.

    * Re-parses YAML configs that were dumped/copied into ``configs/``.
    * Builds a fresh ModelCircuit via `initialize_model_Circuit`.
    * Attaches every saved ``sol.pkl`` / ``sim.pkl`` back to the correct perch.
    """
    from dynx.stagecraft import config_loader
    from dynx.heptapodx.io.yaml_loader import load_config

    saved_dir = Path(saved_dir).expanduser().resolve()
    cfg_dir   = saved_dir / "configs"
    data_dir  = saved_dir / "data"

    # --------------------------- 1. read YAMLs ----------------------------
    master_path = next(cfg_dir.glob("*master*.yml"), None)
    if master_path is None:
        raise FileNotFoundError("No master YAML found in saved configs")
    master_cfg = load_config(master_path)

    stage_cfgs: dict[str, dict] = {}
    for yml in cfg_dir.glob("*_stage.yml"):
        name = yml.stem.replace("_stage", "").upper()
        stage_cfgs[name] = load_config(yml)

    conn_path = cfg_dir / "connections.yml"
    connections_cfg = load_config(conn_path) if conn_path.exists() else {}

    # Align horizon/period count with folders present
    n_periods = len([p for p in data_dir.glob("period_*") if p.is_dir()])
    if n_periods:
        master_cfg["horizon"] = n_periods
        master_cfg["periods"] = n_periods  # backward-compat

    # --------------------------- 2. rebuild circuit -----------------------
    circuit = config_loader.initialize_model_Circuit(
        master_config     = master_cfg,
        stage_configs     = stage_cfgs,
        connections_config= connections_cfg,
    )

    config_loader.compile_all_stages(circuit)
    # helper: convert dict → Solution where possible
    try:
        from dynx.stagecraft.solmaker import Solution  # optional
    except ImportError:
        Solution = None  # type: ignore

    # --------------------------- 3. re-attach data ------------------------
    periods_iter = getattr(circuit, "periods_list",
                    getattr(circuit, "periods", None))
    if periods_iter is None:
        raise AttributeError("Circuit exposes neither 'periods_list' nor 'periods'")

    for p_idx, period in enumerate(periods_iter):
        stages_iter = period.stages.values() if isinstance(period.stages, dict) else period.stages
        for stage in stages_iter:
            for perch_name, perch in stage.perches.items():
                perch_dir = data_dir / f"period_{p_idx}" / stage.name / perch_name

                # ---------- SOL ----------
                sol_path = perch_dir / "sol.pkl"
                if sol_path.exists() and sol_path.stat().st_size:
                    try:
                        with sol_path.open("rb") as f:
                            obj = pickle.load(f)
                        if isinstance(obj, dict) and Solution:
                            obj = Solution.from_dict(obj)  # type: ignore[attr-defined]
                        perch.sol = obj
                    except Exception as exc:
                        logger.warning("Could not load %s: %s (skipped)", sol_path, exc)

                # ---------- SIM ----------
                sim_path = perch_dir / "sim.pkl"
                if sim_path.exists() and sim_path.stat().st_size:
                    try:
                        with sim_path.open("rb") as f:
                            obj = pickle.load(f)
                        perch.sim = obj
                    except Exception as exc:
                        logger.warning("Could not load %s: %s (skipped)", sim_path, exc)

    return circuit


