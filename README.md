# Dyn-X: Modular Dynamic Programming Using Graphs

⚠️ **Experimental Development Build** – APIs and documentation are incomplete.⚠️ 

Dyn-X is a directed-acyclic-graph (DAG) framework for solving recursive
dynamic models. It combines functional recursion with a practical
graph representation of what would otherwise be an infinite-dimensional
problem.

Unlike general graph libraries (e.g. *TensorFlow*), Dyn-X represents **both**
computational operators **and** the functional objects that define a
recursive problem.

---

## Installation

Dyn-X is not yet published on PyPI. You can install the latest
development build directly from GitHub:

**Main branch (bleeding-edge)**

```bash
pip install "git+https://github.com/akshayshanker/dynx.git#egg=dynx"
```

**Specific dev release (v0.18.dev0)**

```bash
pip install "git+https://github.com/akshayshanker/dynx.git@v0.18.dev0#egg=dynx"
```

To upgrade an existing installation:

```bash
pip install --upgrade --force-reinstall \
    "git+https://github.com/akshayshanker/dynx.git#egg=dynx"
```

---

## Documentation

Comprehensive documentation is in progress. For now, please see the
examples in `examples/` and the in-code docstrings.


