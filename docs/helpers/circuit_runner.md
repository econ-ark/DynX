# CircuitRunner User Guide

This short guide walks you through **running parameter sweeps** with the new `CircuitRunner` (v1.6.12+) and shows how to build the design matrix with the **sampler** toolbox.

---

## 1  Quick start

```python
from dynx.runner import CircuitRunner, mpi_map
from dynx.runner.sampler import MVNormSampler, build_design

# ❶ baseline config – ordinary nested dict
base_cfg = {
    "policy": {"beta": 0.96},
    "pricing": {"regime": "low"},
}

# ❷ parameter paths (order matters)
param_paths = ["pricing.regime", "policy.beta"]

# ❸ draw a small design
sampler  = MVNormSampler(mean=[0.0], cov=[[1e-4]])  # only beta numeric
meta     = {"policy.beta": {"min": 0.9, "max": 0.99},
            "pricing.regime": {"enum": ["low", "high"]}}
xs, _    = build_design(param_paths, [sampler], [5], meta, seed=0)

# ❹ create the runner
runner = CircuitRunner(
    base_cfg, param_paths,
    model_factory=lambda cfg: cfg,   # stub factory
    solver=lambda m, **_: None,
)

# ❺ run all rows (serial)
for x in xs:
    metrics = runner.run(x)
```

### 1.5  Minor observations & caveats

* **Global RNG side‑effects** – built‑in samplers now rely on `np.random.default_rng(seed)`. If you roll your own, avoid `np.random.seed` so parallel sweeps don’t collide.
* **Deep‑copy cost** – each call to `runner.run` performs a `copy.deepcopy` of `base_cfg`. For very large configs consider pre‑splitting immutable parts.
* **`sampler` argument deprecated** – The constructor still accepts `sampler` but raises a `DeprecationWarning`. Draw your design externally via `build_design`.
* **Return type** – `runner.run` yields `dict[str, Any]` (may include non‑floats from your `metric_fns` or `RunRecorder`).

---

## 2  API reference

### 2.1  `CircuitRunner` constructor

```python
CircuitRunner(
    base_cfg: dict,
    param_paths: list[str],
    model_factory: Callable[[dict], Any],
    solver: Callable[[Any], None],
    metric_fns: dict[str, Callable[[Any], float]] | None = None,
    simulator: Callable[[Any], None] | None = None,
    cache: bool = True,
)
```

* **`base_cfg`** – any nested dictionary that represents *one* model configuration.
* **`param_paths`** – dot‑paths whose leaves will be overwritten.
* **`model_factory(cfg)`** – must return a *fresh* model instance.
* **`solver(model)`** – mutates `model` in‑place; telemetry goes via `RunRecorder`.
* The optional **`simulator`** runs *after* solving.

### 2.2  `.run(x, return_model=False)`

* `x` is one row from `xs` (`dtype=object`).
* Returns a metric dict (`dict[str, Any]`) or `(metrics, model)`.

### 2.3  `mpi_map(runner, xs, mpi=True)`

* Evaluates many rows; returns a `pandas.DataFrame` (+ models if asked).
* Works single‑process when `mpi=False`.

### 2.4  Helper functions

* **`pack(d)` / `unpack(x)`** – convert between dict and array.
* **`set_deep(d, path, val)`** – utility for patching nested dicts.

---

# Sampler User Guide

The sampler toolbox creates the **design matrix** `xs` you feed to `CircuitRunner`.

## 1  Meta specification

| key in `meta[path]` | meaning               | example                    |
| ------------------- | --------------------- | -------------------------- |
| `"min", "max"`      | numeric bounds        | `{"min":0.9,"max":0.99}`   |
| `"enum"`            | categorical strings   | `{"enum":["low","high"]}`  |
| `"values"`          | discrete numeric list | `{"values":[0.1,0.2,0.3]}` |

## 2  Built‑in samplers

| class                                        | draws                     | notes                                                 |
| -------------------------------------------- | ------------------------- | ----------------------------------------------------- |
| `MVNormSampler(mean, cov, clip_bounds=True)` | joint numeric block       | fills categorical cols with `np.nan`; bounds clipped. |
| `LatinHypercubeSampler(ranges, sample_size)` | independent numeric draws | numeric only                                          |
| `FullGridSampler(grid_values)`               | Cartesian product         | can include strings and/or numbers                    |
| `FixedSampler(rows)`                         | manual rows               | use first for baselines                               |

## 3  `build_design()`

```python
xs, info = build_design(
    param_paths,
    samplers,   # list of sampler instances
    Ns,         # list of sample counts (None for FixedSampler)
    meta,
    seed=123,
)
```

* Numeric block(s) from samplers are **replicated** across the full grid of every categorical column.
* Result `xs` is `dtype=object`; no `np.nan` remains.
* `info["sampler"]` lists which sampler generated each row (with "×grid" suffix when categorical expansion applied).

## 4  Tips & gotchas

* **MVN dimension ≙ consecutive paths** – a k‑dimensional MVN must map to *k* consecutive entries in `param_paths`.
* **Random seeds** – samplers accept `seed`; they currently use `np.random.seed`, so set it once per build if you worry about global RNG effects.
* **All‑numeric sweeps** – when no categorical column exists, `xs` remains a plain `float` array and Cartesian logic is a no‑op.

---

## 5  Troubleshooting checklist

| symptom                                           | likely cause                                               | fix                                                                                                             |
| ------------------------------------------------- | ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `ValueError: Design matrix contains NaN`          | at least one path never filled                             | ensure every sampler outputs all columns **or** provide explicit values via `FullGridSampler` / `FixedSampler`. |
| `Dimension mismatch` from `MVNormSampler`         | `mean`/`cov` length ≠ number of numeric paths it handles   | align dimensions or split into smaller MVN samplers.                                                            |
| Categorical strings appear as `nan` in final `xs` | forgot to include categorical path in `meta` with `"enum"` | add the entry or switch to `FullGridSampler`.                                                                   |

Happy sweeping! 🎛️
