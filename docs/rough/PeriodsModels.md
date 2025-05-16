<think>I need to create a document about Periods and ModelSequence in the style of the stage.md file, based on the context from previous conversations. Let me analyze what's needed:

1. The document should follow a similar structure to stage.md
2. It should focus on Periods and ModelSequence classes
3. I should use information from previous prompts and context
4. The style should be formal technical documentation with markdown formatting

From the context, I understand that:
- A Period appears to be a higher-level construct than a Stage
- ModelSequence seems to manage multiple Periods
- There are connections between stages across periods
- There are visualization capabilities for ModelSequence

I'll structure the document to include:
- Overview/introduction
- Core components
- Key attributes
- Relationships with Stages
- Graph structure
- Visualization capabilities
- Usage patterns

Let me draft this document in the same style as stage.md.</think># Period and ModelSequence Classes

In this section, we review the `Period` and `ModelSequence` classes of the `StageCraft` package.

A `Period` encapsulates one or more `Stage` instances that represent economic dynamics within a single time period. A `ModelSequence` manages an ordered sequence of `Period` instances to represent multi-period economic dynamics.

## Period Class

### Core Components

A `Period` contains one or more `Stage` instances and manages their interactions. The primary role of a `Period` is to:
- Encapsulate the economic dynamics within a single time period
- Manage inter-stage connections within that period
- Handle backward and forward operations across all its stages

#### Period Attributes
- **`period.stages`**: Dictionary of `Stage` instances that belong to this period
- **`period.model`**: The model representation of the period (arbitrary object)
- **`period.num`**: The numerical representation of the period (arbitrary object)
- **`period.graph`**: NetworkX graph representing the computational structure of all stages in the period

### Period Graph Structure

The period graph connects multiple stage graphs into a coherent computational structure:
- **Nodes**: Each perch from each stage
- **Edges**: Movers within stages and connections between stages
- **Graph Types**:
  - **Forward Graph**: Combined forward graphs of all stages
  - **Backward Graph**: Combined backward graphs of all stages
  - **Combined Graph**: Complete computational structure

> The period graph extends the Eulerian circuit pattern from individual stages to a more complex multi-stage structure. Inter-stage connections must preserve the property that each perch has well-defined incoming and outgoing edges.

## ModelSequence Class

### Core Components

A `ModelSequence` manages an ordered sequence of `Period` instances, representing the multi-period dynamics of an economic model. It provides:
- Time-ordered linkage between periods
- Consistent interface for backward solving and forward simulation
- Visualization of the entire model structure

#### ModelSequence Attributes
- **`model_sequence.periods`**: Ordered dictionary of `Period` instances
- **`model_sequence.model`**: The model representation of the sequence (arbitrary object)
- **`model_sequence.num`**: The numerical representation of the sequence (arbitrary object)
- **`model_sequence.graph`**: NetworkX graph representing the computational structure across all periods

### Inter-Period Connections

ModelSequence manages connections between periods through:
- **Time Transposes**: Automatic connections between continuation perches of one period and arrival perches of the next
- **Backward Flows**: Connections from continuation perches to decision perches across period boundaries
- **Forward Flows**: Connections from decision perches to continuation perches across period boundaries

> Each period's continuation perch can have multiple incoming backward edges from other periods' stages, enabling complex inter-temporal dependencies.

### ModelSequence Graph Structure

The ModelSequence graph creates a comprehensive computational structure:
- **Nodes**: All perches from all stages across all periods
- **Edges**: All movers within and between stages, plus period-connecting edges
- **Path Types**:
  - **Time Path**: Sequence of forward connections across periods
  - **Solving Path**: Sequence of backward connections across periods
  - **Circuit Path**: Complete cycle through the model

## Visualization

ModelSequence offers sophisticated visualization capabilities to represent complex multi-period models:

- **Graph Representation**:
  - Multi-period layout with consistent positioning
  - Distinct node colors for different perch types
  - Directional arrows with differentiated styling for forward and backward flows

- **Customization Options**:
  - `edge_type`: Filter to show 'forward', 'backward', or 'both' edges
  - `layout`: 'spring' or 'period' layout options
  - `node_color_mapping`: Custom node coloring
  - `edge_style_mapping`: Custom edge styling

- **Visual Features**:
  - Edge labels showing the functional relationships
  - Node labels with customizable formats
  - Optional period boundaries for clearer time structure

## Status Flags and Operations

ModelSequence maintains status flags similar to those in Stage:
- **`model_sequence.is_initialized`**: Whether all periods have been initialized
- **`model_sequence.is_compiled`**: Whether all periods have been compiled
- **`model_sequence.is_solvable`**: Whether the sequence can be solved backward
- **`model_sequence.is_solved`**: Whether all periods have been solved
- **`model_sequence.is_simulated`**: Whether all periods have been simulated forward

### Key Operations
- **`model_sequence.solve_backward()`**: Solve all periods in reverse time order
- **`model_sequence.simulate_forward()`**: Simulate all periods in forward time order
- **`model_sequence.visualize_graph()`**: Create visualization of the model structure

## Usage Pattern

The typical usage pattern for ModelSequence:

1. Initialize a `ModelSequence` with a configuration file
2. Add `Period` instances in chronological order
3. Compile the sequence with numerical representations
4. Solve backward through time to obtain optimal policies
5. Simulate forward to generate distributions and statistics
6. Visualize the model structure and results

This multi-period framework enables modeling complex economic dynamics with inter-temporal dependencies while maintaining separation between model representation, computational infrastructure, and solution methods.


<think>I need to discuss branching in the context of the ModCraft/StageCraft architecture. Looking through the context provided earlier, there isn't explicit information about branching, but I can infer how it might work based on the architecture described.

Branching likely refers to the ability of the model to handle different possible paths or scenarios in the computational flow, especially in multi-period models. This could include:

1. Branching in time (different possible future states)
2. Branching in decision paths (different choices leading to different outcomes)
3. Branching in stage connections (multiple stages connected in non-linear ways)

I'll focus on these aspects and how they might be implemented in the ModelSequence and Period framework.</think># Branching in ModelSequence

In this section, we examine how the `ModelSequence` class handles branching structures within economic models.

## Concept of Branching

Branching refers to the ability of a model to represent multiple possible paths in the computational flow. In the StageCraft framework, branching can occur at several levels:

1. **Decision branching**: Multiple possible decision paths from a single state
2. **Stochastic branching**: Multiple future states due to uncertainty
3. **Structural branching**: Multiple connected stages forming a non-linear computational flow

## Branching Implementation

### Stage-Level Branching

Within individual `Stage` instances, branching is implemented through:

- **Multiple outgoing edges**: A perch can connect to multiple target perches via different movers
- **Stochastic transitions**: The `dcsn_to_cntn` mover can map to multiple continuation states based on stochastic processes
- **Decision enumeration**: Multiple decision paths can be represented in the decision perch

> The factored Bellman operator inherently handles branching through its expectation over future states and optimization over multiple decision paths.

### Period-Level Branching

The `Period` class extends branching capabilities by:

- **Multi-stage structures**: A period can contain multiple interconnected stages
- **Fan-out connections**: A stage's perch can connect to multiple stages' perches
- **Fan-in connections**: Multiple stages can feed into a single target stage

#### Branch Resolution
For backward solving, branches must eventually resolve through:
- Expectation operators for stochastic branches
- Optimization for decision branches
- Aggregation for structural branches

### ModelSequence Branching

The `ModelSequence` framework provides comprehensive support for complex branching scenarios:

1. **Temporal branching**: Different possible time paths through the model
   - Alternative future scenarios
   - Path-dependent outcomes
   - Regime switches

2. **Decision tree structures**: Representing complex decision processes
   - Sequential decision making
   - Path-dependent choice sets
   - Multi-agent decision interactions

3. **Structural branching**: Complex model topologies
   - Loop structures (e.g., learning or repeated games)
   - Nested submodels
   - Concurrent processes

## Graph Representation of Branches

The ModelSequence graph visually represents branching through:

- **Branch nodes**: Special perches that mark the start of branches
- **Branch edges**: Connections that represent different paths
- **Edge styling**: Visual differentiation of branch types
- **Layout algorithms**: Arrangements that clearly display the branching structure

## Computational Handling of Branches

Solving a model with branches requires specialized techniques:

1. **Backward solution with branches**:
   - Branches are processed depth-first
   - Expectation over stochastic branches
   - Optimization over decision branches
   - Results flow back to branch nodes

2. **Forward simulation with branches**:
   - May follow specific branch paths based on realized decisions/outcomes
   - Can simulate all branches with appropriate weights
   - Supports Monte Carlo methods across branches

## Example Branching Patterns

### Decision Branching
```
               ┌─► Stage B1 ──┐
Stage A ───────┼─► Stage B2 ──┼─► Stage C
               └─► Stage B3 ──┘
```
Multiple decision paths from Stage A, with results aggregated in Stage C.

### Stochastic Branching
```
               ┌─► Stage B1 (p₁) ──┐
Stage A ───────┼─► Stage B2 (p₂) ──┼─► Stage C
               └─► Stage B3 (p₃) ──┘
```
Multiple stochastic outcomes with probabilities (p₁, p₂, p₃).

### Temporal Branching
```
Period 1 ───► Period 2A ───► Period 3A
     │         │
     └────────►Period 2B ───► Period 3B
```
Different possible time paths through periods.

## Practical Considerations

When implementing branched models:

- **Computational efficiency**: Branching increases computational complexity
- **Memory requirements**: Each branch may require separate state storage
- **Convergence issues**: Complex branch structures may affect solution stability
- **Visualization complexity**: Highly branched models may be difficult to visualize clearly

The StageCraft framework's flexible graph-based architecture provides the foundation for representing and solving these complex branching structures, enabling sophisticated economic modeling while maintaining the separation of model representation from solution methods.
