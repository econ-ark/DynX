# Key Concepts

This document outlines the fundamental concepts in the StageCraft package.

## 1. Mathematical foundations 

### Functional Objects
- **Definition**: Functional objects that the Bellman operator and push-forward operation takes as inputs and returns as outputs.
- **Representation**: Mathematical points in a Banach space. Computational implementation: [`perch.sol`](../src/circuitcraft/stagecraft/perch.py#L120) (solution function) and [`perch.dist`](../src/circuitcraft/stagecraft/perch.py#L140) (distribution function) callables.
- **Instantiation**: Numerically specified by the user as needed.

### Euclidean State-Space
- **Definition**: The domain of the functional objects (functions) returned by the Bellman operator. These are commonly understood as an economic model's "state spaces" (e.g., assets).
- **Representation**: Mathematical: Symbolically stored in [`model.math`](../src/heptapod/functional_problem.py) (analytical representation). Computational implementation: appropriate grid in [`model.num`](../src/heptapod/numerical_problem.py) (numerical representation), depending on [`model.method`](../src/heptapod/functional_problem.py) (solution method).

### Factored Bellman Operator
- **Definition**: The Bellman operator as applied to continuation values. In this package, we use "Bellman operator" and "factored Bellman operator" interchangeably. We also refer to any functional operation that solves the factored Bellman equation as a "Bellman operator". For instance, the Coleman operator may return optimal policy functions without iterating on the value function. 
- **Representation**: Mathematical: Symbolically stored in [`stage.model.math`](../src/circuitcraft/stagecraft/stage.py) (Stage's mathematical model). Computational implementation: See [Functional Operators](#functional-operators).

### Functional Operators
- **Definition**: Operators (e.g., expectation, optimization) that compose the Bellman operator and simulations (e.g., Coleman or Stachurski operators).
- **Representation**: Symbolically stored in [`model.math`](../src/heptapod/functional_problem.py) within a [`Mover`](../src/circuitcraft/stagecraft/mover.py). Computational implementation: optionally stored as a callable in [`mover.model.comp`](../src/circuitcraft/stagecraft/mover.py) if stage is in `in-situ` mode. Otherwise, the computational implementation is external to the stage. In both cases, the functional operators are produced externally by the solution package ('horse'). 

### Transition Functions
- **Definition**: Measurable functions between Euclidean spaces that define state transitions (often noted as "g" functions).
- **Representation**: Symbolically stored in [`model.math`](../src/heptapod/functional_problem.py) within a [`Mover`](../src/circuitcraft/stagecraft/mover.py). Computational implementation: stored in [`model.num`](../src/heptapod/numerical_problem.py) as callables. 
- **Note**: These are not the same as functional operators but are part of the operator's structure.

## 2. Model Representation

Model representation is instantiated in the `model` object. In the current package, the representation class and its methods are defined in the [`heptapod`](../src/heptapod/) package. However, the [`Stage`](../src/circuitcraft/stagecraft/stage.py) class requires that the representation class has the following attributes:
- **[`model.math`](../src/heptapod/functional_problem.py)**: Analytical representation of model
- **[`model.parameters`](../src/heptapod/functional_problem.py)**: Economic parameters
- **[`model.settings`](../src/heptapod/functional_problem.py)**: Numerical settings
- **[`model.methods`](../src/heptapod/functional_problem.py)**: Methods to create numerical objects and operations
- **[`model.num`](../src/heptapod/numerical_problem.py)**: Numerical representation of grids, transition functions, reward functions, and other elements needed to form computational operators.

Note: [`model.num`](../src/heptapod/numerical_problem.py) needs to be fully defined by all the elements preceding it.

### 2.2 Representation Process
- **Initialization**: Converting configuration to a mathematical model (`init_rep`)
- **Compilation**: Converting the mathematical model to a numerical representation (`num_rep`)

The step after compilation to create the functional operators is not explicitly defined as a status in the stage class as this is largely handled by the solution package ('horse').
## 3. Core StageCraft Components

### Perches
- **Definition**: Computational objects that hold representation of functional objects and include the mathematical/numerical instructions needed for evaluating those states.
- **Implementation**: [`Perch`](../src/circuitcraft/stagecraft/perch.py) class instances that are attributes of a [`Stage`](../src/circuitcraft/stagecraft/stage.py).

Perches "types" are defined by the particular instance of a perch. In a stage, there are three perches:

### Arrival Perch
- **Definition**: The perch at the beginning of a stage before shocks are realized.
- **Implementation**: [`Perch`](../src/circuitcraft/stagecraft/perch.py) class instance.

### Decision Perch
- **Definition**: The perch after shocks are realized before decision is made.
- **Implementation**: [`Perch`](../src/circuitcraft/stagecraft/perch.py) class instance.

### Continuation Perch
- **Definition**: End of period perch after decision is made.
- **Implementation**: [`Perch`](../src/circuitcraft/stagecraft/perch.py) class instance.

## Movers
- **Definition**: Computational objects containing the mathematical/numerical instructions, parameters, and methods required to build the computational implementation of functional operations linking perches.
- **Implementation**: [`Mover`](../src/circuitcraft/stagecraft/mover.py) class instances that are attributes of a [`Stage`](../src/circuitcraft/stagecraft/stage.py).
- **Note**: The actual computational operator can be optionally attached as [`mover.comp`](../src/circuitcraft/stagecraft/mover.py).

### Mover Connection Attributes
Movers have built-in attributes that define connections between perches:
- **[`mover.source_perch`](../src/circuitcraft/stagecraft/mover.py)**: Reference to the source perch
- **[`mover.target_perch`](../src/circuitcraft/stagecraft/mover.py)**: Reference to the target perch
- **[`mover.source_key`](../src/circuitcraft/stagecraft/mover.py)**: Name of the attribute in the source perch to use (e.g., "sol" or "dist")
- **[`mover.target_key`](../src/circuitcraft/stagecraft/mover.py)**: Name of the attribute in the target perch to populate

Without a graph structure, these attributes are the only things that define connections between perches. The mover uses these attributes to locate and access the relevant perches and their functional objects.

With a graph structure, these same connections are represented as edges in a directed graph, with the movers themselves serving as the edges and perches as nodes. The graph structure provides additional capabilities like traversal algorithms, cycle detection, and visualization while maintaining the same underlying connectivity defined by the mover attributes.

### Mover Connectivity
- **Intra-Stage**: Movers connect perches within a stage (e.g., arrival→decision→continuation).
- **Inter-Stage**: Movers connect perches between different stages (e.g., the arrival perch of one stage connects to the continuation perch of another stage).
- **Inter-Period**: In a multi-period model, movers connect stages across time periods (e.g., the arrival perch of a stage in period t connects to the continuation perch of a stage in period t-1).

### Backward Movers
- **Definition**: Movers that link perches in the order of backward induction and contain representations of operators that comprise the Bellman operator. 
- **Implementation**: [`Mover`](../src/circuitcraft/stagecraft/mover.py) class instance.

### Forward Movers
- **Definition**: Movers that link perches in the order of simulation for push-forward.
- **Implementation**: [`Mover`](../src/circuitcraft/stagecraft/mover.py) class instance.

## Stage 
- **Definition**: The central organizing structure representing an evaluation of the factored Bellman operator in two directions: forward and backward.
- **Implementation**: [`Stage`](../src/circuitcraft/stagecraft/stage.py) class instance.

### Branching stage
- **Definition**: A stage that has multiple incoming edges.
- **Implementation**: Option in [`Stage`](../src/circuitcraft/stagecraft/stage.py) class that allows for multiple incoming edges in the backward direction

A branching stage has multiple income edges (or incoming inter-stage movers) in the backward direction into the continuation perch. However, everything else within the stage remains the same. The number of movers required to represent the operation at the decision perch and arrival perch remains one. 

## Period 
- **Definition**: A collection of stages representing a single time period. The stages are connected through inter-stage movers.
- **Implementation**: [`Period`](../src/circuitcraft/stagecraft/period.py) class instance.

Note: A period is only connected together through its common time index.

## ModelCircuit
- **Definition**: A collection of periods and inter-period movers (these movers connect stages within a period to stages in adjacent periods) representing a sequence of economic dynamics.
- **Implementation**: [`ModelCircuit`](../src/circuitcraft/model_circuit.py) class instance.


> While periods may have an arbitrary number of incoming edges and outgoing edges, a well-defined 'modelcircuit' has only one incoming edge in the forward direction. Unlike a period, a modelcircuit must also be an Eulerian circuit.

## 4. Graph Structure
Movers and perches implicitly define a graph, but this relationship can be made explicit using NetworkX:
- **Movers**: Represented as directed edges (backward movers for solution, forward movers for simulation).
- **Perches**: Represented as nodes.
- **Implementation**: The [`Stage`](../src/circuitcraft/stagecraft/stage.py) class maintains [`forward_graph`](../src/circuitcraft/stagecraft/stage.py), [`backward_graph`](../src/circuitcraft/stagecraft/stage.py), and [`combined_graph`](../src/circuitcraft/stagecraft/stage.py) attributes.

Because the NetworkX graphs are intentionally "thin," one can treat movers and perches as purely non-graph objects until graph-based analysis is required.

## 5. Horse and Whisperer

### 5.1 Horse

A Horse is any external computational engine that implements numerical algorithms for solving the factored Bellman operator and populating the perches in a [`ModelCircuit`](../src/circuitcraft/model_circuit.py). It handles backward induction, computes value or shadow value and policy functions. The Horse consumes [`model.num`](../src/heptapod/numerical_problem.py) representations and produces solutions that populate [`perch.sol`](../src/circuitcraft/stagecraft/perch.py#L120) attributes within the [`Stage`](../src/circuitcraft/stagecraft/stage.py) structure.

### 5.2 Whisperer

The Whisperer orchestrates the relationship between the Horse and the [`ModelCircuit`](../src/circuitcraft/model_circuit.py) by extracting relevant model information, determining graph topology, and using that information to "instruct" the Horse on how to solve the recursive problem. The Whisperer then populates the perches with the solution.

A Whisperer may simply attach an [`operator_factory`](../src/circuitcraft/stagecraft/stage.py) to the stage, in which case it implicitly uses the stage's internal backward and forward topologies to execute the Horse's solution algorithm.


## Related Classes
- [`Stage`](../src/circuitcraft/stagecraft/stage.py): The central organizing structure representing an evaluation of the factored Bellman operator.
- [`Period`](../src/circuitcraft/stagecraft/period.py): Encapsulates one or more `Stage` instances representing dynamics within a single time period.
- [`ModelCircuit`](../src/circuitcraft/model_circuit.py): Manages an ordered sequence of `Period` instances for multi-period economic dynamics.

## Related Packages
- [`heptapod`](../src/heptapod/): Provides model representation functionality for economic models.
- [`circuitcraft`](../src/circuitcraft/): Implements computational graph infrastructure for economic modeling.
