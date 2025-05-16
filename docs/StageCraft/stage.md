# StageCraft: Stage

The `Stage` class is the central organizing structure in ModCraft, representing an **evaluation of the factored Bellman operator** and the **push-forwards** under its solution.

## Contents
- [Core Components](#core-components)
- [Perches](#perches)
- [Movers](#movers)
- [External Components](#external-components-provided-by-modules)
- [Solutions and Operators](#solutions-and-operators)
- [Graphs](#graphs)
- [Usage Pattern and Status Flags](#usage-pattern-and-status-flags)
- [Grid Access](#grid-access)

## Core Components

| Component | Description | Access Pattern |
|-----------|-------------|----------------|
| `stage.model` | Model representation | Direct attribute |
| `stage.perches` | Dictionary of perch instances | `stage.perches['arvl']` or `stage.arvl` |
| `stage.movers` | Dictionary of mover instances | `stage.movers['arvl_to_dcsn']` or `stage.arvl_to_dcsn` |


### Perches
`Perch` class instances that are attributes of a Stage. Each perch holds **functional objects** at specific points in the computational flow of the factored Bellman operator.

| Perch | Description | Access |
|-------|-------------|--------|
| **`stage.arvl`** (arrival) | functional objects at the beginning of a period | `stage.arvl` |
| **`stage.dcsn`** (decision) | functional objects at the decision point | `stage.dcsn` |
| **`stage.cntn`** (continuation) | functional objects at the end of a period | `stage.cntn` |

#### Perch Attributes
- **`perch.model`**: The model representation of the perch (arbitrary object)
- **`perch.sol`**: The solution to the perch (arbitrary callable)
- **`perch.dist`**: The distribution over the perch (arbitrary callable)
- **`perch.grid`**: Access to state space grids (available after building computational model)

### Movers
`Mover` class instances that are attributes of a Stage. Each mover contains the **model representation** that defines a mapping between perches:

| Mover | Direction | Function | Access |
|-------|-----------|----------|--------|
| **`stage.arvl_to_dcsn`** | Forward | Transforms arrival states to decision states | `stage.arvl_to_dcsn` |
| **`stage.dcsn_to_cntn`** | Forward | Maps decisions to continuation states | `stage.dcsn_to_cntn` |
| **`stage.cntn_to_dcsn`** | Backward | Backward induction (solving) continuation value function (or other functional state object) to decision value function | `stage.cntn_to_dcsn` |
| **`stage.dcsn_to_arvl`** | Backward | Backward induction (solving) decision value function (or other functional state object) to arrival value function | `stage.dcsn_to_arvl` |

#### Mover Attributes
Movers have built-in source and target attributes that point to the perch instances they connect:
- **`arvl_to_dcsn.source_name`**: `stage.arvl`
- **`arvl_to_dcsn.target_name`**: `stage.dcsn`
- **`dcsn_to_cntn.source_keys`**: `sol`
- **`dcsn_to_cntn.target_keys`**: `sol`

Movers also have:

- **`mover.model`**: The model representation of the mover 
- **`mover.comp`**: The numerical instantiation of the **computational operator**

> **Implementation Note**: The classes `Stage`, `Perch` and `Mover` are defined within the `StageCraft` package. Instances of `model` are an arbitrary object from the point of view of the core StageCraft package; their representation is handled by the `heptapod` package.

## External Components (Provided by Modules)

ModCraft's architecture emphasizes separation between model-representation infrastructure, application-specific operators and core **graph infrastructure**:

### Model Representation

The economic model itself is provided externally and defines the problem structure. In the `heptapod` package, the model is an instance of the `FunctionalProblem` class:

| Component | Description |
|-----------|-------------|
| **`stage.model.math`** | Contains mathematical representation of the model |
| **`stage.model.parameters`** | Contains model structural economic parameters |
| **`stage.model.settings`** | Contains model settings |
| **`stage.model.methods`** | Contains model methods |
| **`stage.model.num`** | Contains numerical representation of the model (**compiled**) |

> **Model Scope**: An instance of a model class is a self-contained set of instructions; models should be defined at the stage level, perch level, and mover level:
> - At the **stage level**, the model contains all information relevant to the factored Bellman operator
> - At the **mover level**, the model contains all information relevant to transformation operators for the particular mover
> - At the **perch level**, the model contains all information relevant to evaluating the functional objects at the perch

#### Model Representation Workflow
Model representations proceed in three steps (provided by `heptapod`):
1. **`init_rep`**: Takes a config file and defines the mathematical model, parameters, settings, and methods
2. **`num_rep`**: Compiles mathematical model, parameters, settings, and math models into numerical representations of functions, grids, and state-spaces required for numerical solution

## Solutions and Operators

> **Important**: A numerical representation of a mover (including transition functions $g$) in `model.num` is **not** the same as the instantiation of the computational operation $\mathbb{T}^{a}$ associated with each Mover.

Each Stage, once compiled, can be solved in two ways:

1. **In-situ solution**: Using a computational operator attached directly to the Stage by the `whisperer`
2. **External solution**: Implemented through external solvers by the `whisperer`

In both cases, the application or solver must provide either:
- The appropriate operator for each Mover, or
- Implementation of the solution method

### Operator Factory

In in-situ mode, the `whisperer` provides an `operator_factory` to the stage. The stage then uses the model information, feeds it to the `operator_factory`, and then populates the `mover.comp` numerical operators:

| Aspect | Description |
|--------|-------------|
| **Creation** | `operator_factory` creates numerical instantiations of functional operators for Stage Movers |
| **Origin** | `operator_factory` created **outside** the Stage and specific to particular solution methods |
| **Independence** | `operator_factory` is independent of the model representation package (heptapod), but uses `model.num` to create numerical representations of operators |
| **Result** | `operator_factory` produces an operator that is attached to `mover.comp` attribute |

This approach is strongly advantageous when the whisperer needs to leverage the graph's inbuilt features to solve complex model topologies, perform autodifferentiation, etc.

### External Solution using Whisperer

The Mover contains all information needed by the 'horse' to solve the model (via a whisperer), but attaching the configured operator to the `mover.comp` attribute remains optional and depends on the solution approach.

A whisperer may also work on the stage externally, extracting the necessary information, instructing the horse to perform the solution and then attaching the solutions to the stage. 

## Graphs

Beyond the implicit relationships between perches and movers defined by their source and target attributes, a Stage also contains built-in NetworkX graphs to represent the computational `links` between the perches. 

From a graphical point of view, a each `perch` within the stage is a `node` and each `mover` is an `edge`.

The purpose of these graphs is to provide a formal, standardized representation of the computational flow. For simulation, generating moments, auto-differentiation, checking dimensional reduction, etc., the intra-stage graph can be connected to the complete `modelCircuit` graph to form a complete computational graph.

### Graph Types
| Graph | Description |
|-------|-------------|
| **`stage.forward_graph`** | Represents simulation flow |
| **`stage.backward_graph`** | Represents solution flow |
| **`stage.combined_graph`** | Represents complete computational structure |

> **Note**: The forward graph is simply the transpose of the backward graph.

> **Structure**: The central organizing stage is the smallest possible Eulerian circuit with non-trivial sub-circuits. Each perch in a stage may have only one incoming and one outgoing edge to another perch within the stage.

Whether the graph is called upon to aid the whisperer is determined entirely by the problem at hand.

> **Design Principle**: StageCraft is designed so that the stage object is thin, it is not an instance of a graph or a particular model representation. However, where graph relationships are made explicit, they are done so using a benchmark package (NetworkX) rather than ad hoc code.

## Usage Pattern and Status Flags

### Status Flags

| Flag | Description | True When |
|------|-------------|-----------|
| **`stage.is_initialized`** | Initialization status | Stage has `model.math`, `model.parameters`, `model.settings`, and `model.methods` |
| **`stage.is_compiled`** | Compilation status | Stage has `model.num` |
| **`stage.is_solvable`** | Solvability status | Backward graph has non-empty `sol` in initial perch, forward graph has non-empty `dist` in initial perch |
| **`stage.is_solved`** | Solution status | All `sol` attributes have been populated |
| **`stage.is_simulated`** | Simulation status | All `dist` attributes have been populated |

### Connection Patterns

#### Internal Connections
- `arvl` → `dcsn` → `cntn` (Forward flow)
- `cntn` → `dcsn` → `arvl` (Backward flow)

#### External Connections
- A stage's continuation perch may have multiple incoming backward edges from external stages
- Inter-stage connections are managed at the Period and ModelCircuit level

## Grid Access

The Stage class provides direct access to state space grids through perch objects. After building the computational model with `build_computational_model()`, each perch automatically has a `.grid` attribute that gives access to the state space grids.

```python
# Access asset grid in arrival perch
a_grid = stage.arvl.grid.a

# Access cash-on-hand grid in decision perch
m_grid = stage.dcsn.grid.m

# Access next-period asset grid in continuation perch
a_nxt_grid = stage.cntn.grid.a_nxt


### Example Usage

```python
# Initialize a stage from configuration
stage = Stage(config_file='example_config.yaml')

# Compile the stage
stage.compile()

# Check status
print(f"Stage initialized: {stage.is_initialized}")
print(f"Stage compiled: {stage.is_compiled}")
print(f"Stage solvable: {stage.is_solvable}")

# Access state space grids (after compilation)
assets = stage.arvl.grid.a
cash = stage.dcsn.grid.m

# Solve the stage (assuming a whisperer is available)
whisperer.solve_stage(stage)

# Check solution status
print(f"Stage solved: {stage.is_solved}")
```
