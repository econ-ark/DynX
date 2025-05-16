# 3. Graphs

The ModCraft framework employs a graph-theoretic view to model economic problems:

## 3.1 Graph Structure

The computational structure is represented as a directed graph where:
- **Nodes (Perches)**: Represent states in the MDP
- **Edges (Movers)**: Represent transitions between states

This graph structure enables:
1. **Clear Visualization**: The problem structure can be easily visualized and communicated.
2. **Efficient Algorithms**: Graph algorithms can be used for solving and analyzing the models.
3. **Modular Composition**: Complex models can be built by connecting simpler sub-graphs.

## 3.2 DAG (Directed Acyclic Graph) for Solution

The backward induction solution process follows a DAG structure, ensuring:
- Value functions are computed in the correct order
- Information flows properly between components
- Circular dependencies are avoided

## 3.3 Circular Graph for Simulation

For forward simulation, the graph creates a circular structure that captures:
- The flow of states from one period to the next
- The propagation of shocks through the system
- The iteration of decision rules over time

## 3. Computational Graphs

### 3.1 Introduction to Computational Graphs

The ModCraft framework represents dynamic programming problems as computational graphs, providing a visual and mathematical abstraction that clarifies structure and facilitates computation:

1. **Definition**: A computational graph in ModCraft is a directed graph where:
   - Nodes represent computational points (continuation, arrival, decision points)
   - Edges represent transitions between these points
   - The graph topology encodes the structure of the dynamic programming problem

2. **Graph Properties**:
   - **Directed**: Transitions flow in one direction, reflecting the temporal progression of the decision process
   - **Potentially Cyclic**: Cycles represent recurring states in infinite-horizon problems
   - **Labeled**: Nodes and edges carry semantic information about the economic process

### 3.2 Node Types in ModCraft Graphs

The three fundamental node types correspond to the CDC structure outlined in the theoretical framework:

1. **Continuation (C) Nodes**: 
   - Represent points where uncertainty may be resolved
   - Compute expected values over possible outcomes
   - Connect to Arrival nodes via probability-weighted edges

2. **Arrival (A) Nodes**: 
   - Represent points where a specific uncertainty realization has occurred
   - Update state variables based on the arrival of new information
   - Connect to Decision nodes to enable action selection in the new state

3. **Decision (D) Nodes**: 
   - Represent points where economic agents make choices
   - Optimize over possible actions given the current state
   - Connect to Continuation nodes to move forward in the process

### 3.3 Graph Construction and Manipulation

The computational graph approach offers several advantages for model development:

1. **Intuitive Visualization**: The graph structure provides a clear visual representation of the economic model's decision process.

2. **Modular Construction**: Subgraphs representing specific economic components can be developed independently and later integrated.

3. **Topological Analysis**: The graph structure can be analyzed to identify:
   - Decision bottlenecks
   - Independent subproblems that can be solved separately
   - Optimizable graph traversal sequences

4. **Automatic Differentiation**: The graph structure facilitates automatic differentiation for optimization, especially when implemented in frameworks like JAX or PyTorch.

This graph-based representation provides the structural foundation upon which ModCraft's computational elements operate, enabling efficient solution of complex dynamic programming problems. 