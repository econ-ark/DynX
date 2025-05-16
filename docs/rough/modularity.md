# 2. Modularity

### 2.1 The Module Concept

A fundamental aspect of the ModCraft framework is its approach to modularity, which extends beyond mere code organization to encompass economic modeling principles:

1. **Module Definition**: A module in ModCraft represents a self-contained economic component with:
   - Well-defined inputs and outputs
   - Internal computational logic
   - Clear economic interpretation

2. **Module Types**:
   - **Basic Modules**: Implement fundamental economic behaviors (e.g., consumption decisions, labor supply)
   - **Composite Modules**: Combine multiple modules to represent more complex economic structures

3. **Module Properties**:
   - **Composability**: Modules can be combined in various ways to create new economic structures
   - **Reusability**: The same module can be used across different models
   - **Interpretability**: Each module maintains a clear economic meaning

### 2.2 Module Design Principles

The design of modules in ModCraft adheres to several key principles:

1. **Single Responsibility**: Each module should address one specific economic concept or calculation.

2. **Interface Standardization**: Modules communicate through standardized interfaces, enabling seamless integration.

3. **Economic Coherence**: The internal logic of a module should reflect established economic principles, maintaining theoretical consistency.

4. **Testability**: Modules can be tested individually, allowing verification of economic properties before integration into larger models.

### 2.3 Modular Model Construction

ModCraft enables economists to construct models by assembling modules in a manner similar to building with Lego blocks:

1. **Bottom-Up Construction**: Complex models are built by progressively combining simpler modules.

2. **Substitution and Variation**: Modules implementing alternative economic theories can be swapped to test different hypotheses within the same model structure.

3. **Incremental Refinement**: Models can be improved by enhancing individual modules without disrupting the overall structure.

This modular approach significantly reduces the complexity of model development, enables more effective collaboration, and facilitates the exploration of different economic specifications.

## 2.1 Key Modularity Principles

- **Horse and Whisperer Pattern**: Computational backends (horses) and adapters (whisperers) are separated, allowing for different frameworks to be used.
- **No Python-Native Mathematical Representation**: Mathematical concepts are represented in configuration files rather than directly in Python.
- **Proper Constraint Placement**: Constraints are placed on movers (transitions) rather than perches (states).
- **Clean Extension Points**: The system is designed to be easily extensible for different perch types and computational backends.

## 2.2 Separation of Responsibilities

The ModCraft framework clearly separates responsibilities among three main components:

### Circuit/Stage Package

The circuit/stage package serves as the **orchestration layer** and is responsible for:

- **Structural Definition**: Defines the CDC pattern with perches and movers in a graph structure
- **Workflow Management**: Coordinates the loading, compilation, solving, and simulation processes
- **Data Management**: Stores and retrieves computational results (value functions, policies, distributions)
- **Circuit Connectivity**: Handles the connections between stages and the flow of information
- **Execution Control**: Manages the sequence of operations and maintains state information

This layer is agnostic to the specific mathematical formulations or computational implementations.

### Heptapod Plugin

The Heptapod plugin acts as the **model definition layer** and is responsible for:

- **Mathematical Representation**: Defines the symbolic mathematical objects (functions, constraints, domains)
- **Parameter Specification**: Declares model parameters and their relationships
- **Variable Definitions**: Specifies state variables, control variables, and their domains
- **Configuration Parsing**: Reads and interprets model specifications from configuration files
- **Mathematical Structure Creation**: Builds the `.math` component of the FunctionalProblem

The Heptapod plugin focuses solely on the mathematical representation without concerning itself with how these objects will be numerically implemented or computationally solved.

### Whisperer/Horse Pattern

The whisperer/horse pattern forms the **computational implementation layer** and is divided into two components:

1. **Horse (Computational Backend)**:
   - **Numerical Operations**: Performs the actual computational operations (e.g., optimization, integration)
   - **Algorithm Implementation**: Implements specific numerical methods (e.g., value function iteration)
   - **Numerical Representation**: Represents mathematical objects in numerical form (e.g., grids, arrays)
   - **Computational Efficiency**: Optimizes performance using specific frameworks (JAX, PyTorch, NumPy)

2. **Whisperer (Adapter)**:
   - **Translation**: Translates between the framework-agnostic model representation and the horse-specific format
   - **Configuration Adaptation**: Adapts configuration data to the horse's expected format
   - **Result Interpretation**: Converts computational results back to the framework-agnostic format
   - **Method Selection**: Chooses appropriate computational methods based on model characteristics

This separation allows different computational backends to be used interchangeably without changing the model definition or circuit structure.

## 2.3 Interaction Flow

The interaction between these components follows a clear pattern:

1. **Configuration Phase**:
   - User provides configuration files
   - Heptapod plugin parses the configuration and creates mathematical structures
   - Circuit/stage package stores these structures in appropriate perches and movers

2. **Compilation Phase**:
   - Whisperer receives mathematical structures from the circuit/stage package
   - Whisperer translates these structures for the horse
   - Horse creates numerical implementations (grids, functions, etc.)
   - Whisperer returns these implementations to the circuit/stage package

3. **Solution Phase**:
   - Circuit/stage package determines the solution order
   - Whisperer communicates with the horse to perform backward operations
   - Horse executes the actual computations
   - Results are stored in the perches' `.up` attributes by the circuit/stage package

4. **Simulation Phase**:
   - Circuit/stage package determines the simulation order
   - Whisperer communicates with the horse to perform forward operations
   - Horse executes the actual simulations
   - Results are stored in the perches' `.down` attributes by the circuit/stage package

This modular design allows changes in one component to have minimal impact on others, facilitating maintenance, extension, and adaptation to different problem domains. 