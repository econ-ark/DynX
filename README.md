# Dyn-X: Functional Recursion using Graphs

⚠️ **Experimental Development Build**  
This is an experimental development build of Dyn-X. APIs are not stable and may change without notice.

Dyn-X is a directed acyclic graph (DAG) based framework for computing recursive dynamic models.

The key innovation of Dyn-X  is to combine recursive functional operations with a practicable graph-based representation of an otherwise infinite-dimensional process.

Unlike other available graph-based libraries (e.g., TensorFlow), Dyn-X stores both **computational operators**  and **functional objects**  that define a recursive problem.

- Functional objects (value functions, marginal value functions, distributions, policies, etc.) are stored in nodes, called `perches`.
- Computational operations (optimization, expectation, simulation, and push-forward operators) between perches are stored in edges, called `movers`.
