# 5. Circuit Structure

The circuit structure of ModCraft creates a comprehensive framework for solving and simulating economic models:

## 5.1 Core Circuit Design

The circuit is designed with:
- **Perches as Nodes**: Storing computational objects at specific states
- **Movers as Edges**: Transforming functions between perches
- **Up and Down Functions**: Each perch stores both "up" (forward) and "down" (backward) function dictionaries

## 5.2 CDC Pattern Implementation

The CDC pattern is implemented as a circuit where:
1. **Continuation Perch**: Stores continuation values and terminal conditions
2. **Arrival Perch**: Manages the arrival of shocks and their integration
3. **Decision Perch**: Handles optimization and decision rules

## 5.3 Functional Problem Structure

The evolving FunctionalProblem structure separates:
- **.math**: Pure algebraic definitions (functions, constraints, domains)
- **.num**: Compiled numeric objects (but not the final solver operators)

## 5.4 Configuration System

The circuit is configured using YAML files that define:
- Mathematical specifications
- Parameter values
- Method configurations
- Simulation settings

This configuration system separates mathematical representation from symbolic referencing and numeric implementation.

## 5.5 Stage Model Structure

A **Stage** object orchestrates the model's computation and holds its core components. Here are its primary attributes:

- **name** (`str`): A user-defined name for the stage instance.  
- **status_flags** (`Dict[str, bool]`): A dictionary tracking the computational status (e.g., `initialized`, `compiled`, `solved`).  
- **perches** (`Dict[str, Perch]`): A dictionary containing the **Perch** objects (nodes) of the circuit, keyed by their names (e.g., `'arvl'`, `'dcsn'`, `'cntn'`). Perches hold state variables and computed data (`.up`, `.down`).  
- **forward_graph** (`networkx.DiGraph`): A directed graph representing the forward flow of computation. Edges connect perches and hold **Mover** objects.
- **backward_graph** (`networkx.DiGraph`): A directed graph representing the backward flow of computation. Edges connect perches and hold **Mover** objects.
- **model** (`FunctionalProblem`): An aggregated representation of the entire stage's model, combining parameters, settings, mathematical definitions (`.math`), and numerical components (`.num`) from perches and movers.
- **init_rep** (`Module | Object`): A reference to the plugin component responsible for parsing the configuration and defining the initial mathematical structure (`.math`).  
- **num_rep** (`Module | Object`): A reference to the plugin component responsible for generating the numerical implementation (`.num`) from the mathematical structure (`.math`).  

### Status Flags

The stage tracks its state using `status_flags`. Here's what each core flag indicates:

- **initialized**:  
  True after `stage.load_config()` successfully parses the configuration. The `.math`, `.parameters`, and `.settings` are loaded.  
- **compiled**:  
  True after `stage.build_computational_model()` generates numerical components (`.num`).  
- **solvable**:  
  True once the prerequisites for solving (e.g., boundary conditions, data in `cntn.up` or `arvl.down`) are met.  
- **solved**:  
  True after the backward pass is completed (the `.up` attributes of perches are populated).  
- **simulated**:  
  True after the forward pass is completed (the `.down` attributes of perches are populated).  
- **portable**:  
  True once the Stage's movers have attached computational methods (`.comp`), making it self-contained for serialization or reuse.

### Model Initialization Flow

The typical flow:

1. **Create Stage**  
2. **Load Config** (`stage.load_config()` → sets `initialized`)  
3. **Build Model** (`stage.build_computational_model()` → sets `compiled`)  
4. **Set Initial Values** (User provides data → sets `solvable`)  
5. **Solve** (`stage.solve()` → sets `solved`)  
6. **Simulate** (`stage.simulate()` → sets `simulated`)

### Common Access Patterns

```python
# Access Status Flags
print(f"Is compiled: {stage.status_flags['compiled']}") 

# Direct property access to components
stage_model = stage.model          # Access stage model instance (FunctionalProblem)
arvl_perch = stage.perches["arvl"] # e.g., to get 'arvl' perch object
fwd_mover = stage.forward_graph["arvl"]["dcsn"]["mover"] # Access a specific forward mover

# Stage model properties
beta_param = stage_model.parameters['beta'] 
tol_setting = stage_model.settings['tol']    

# Mathematical components
math_funcs = stage_model.math.get('functions', {}) 
util_func_def = math_funcs.get('u_func', {}) 

# Numerical components
num_funcs = stage_model.num.get('functions', {})
util_callable = num_funcs.get('u_func')             

arvl_grid = stage_model.num.get('state_space', {}).get('arvl', {}).get('grids', {}).get('a') 
income_shock_grid = stage_model.num.get('shocks', {}).get('income_shock', {}).get('grid') 

# Perch data access (populated after solve/simulate)
arvl_up_data = arvl_perch.up 
dcsn_down_data = stage.perches["dcsn"].down 
```

## 5.6 Stage and Plugin Responsibilities

The StageCraft framework separates the **core computational structure** (Stage) from the **model-specific details** (Plugin). This promotes modularity and allows different economic models or numerical methods to be plugged into the same orchestration engine.

1. **Stage Class Responsibilities**:

   - **Orchestration & Workflow**: Manages the overall lifecycle (`load_config`, `build_computational_model`, `solve`, `simulate`).  
   - **Structural Container**: Holds the core components: `perches` dictionary, `forward_graph`, `backward_graph`, and `Mover` instances attached to graph edges.  
   - **Status Tracking**: Owns and updates the primary `status_flags` dictionary (e.g. `initialized`, `compiled`, `solvable`, `solved`, `simulated`, `portable`).  
   - **Runtime Data Storage**: Stores the results of computation in the `.up` and `.down` attributes of its Perch objects.  
   - **Plugin References**: Holds references to the specific plugin modules/objects provided during initialization (`init_rep`, `num_rep`).

2. **Plugin Responsibilities (`init_rep` & `num_rep`)**:

   - **Model Definition** (`init_rep` — Initialization Representation):
     - **Parses Configuration**: Reads the user-provided config file (e.g., YAML).  
     - **Defines Mathematical Structure**: Creates the `FunctionalProblem` instances for the stage, perches, and movers, populating their `.math`, `.parameters`, and `.settings` attributes from the config.  
     - **Returns Structure**: Provides these initialized structures back to the Stage upon `load_config()`.  

   - **Numerical Implementation** (`num_rep` — Numerical Representation):
     - **Generates Numerical Components**: Takes the initialized model structures (containing `.math`).  
     - **Builds `.num`**: Populates the `.num` attribute within the `FunctionalProblem` instances, creating compiled functions (callables), numerical state space grids, shock representations, etc.  
     - **Returns Implementation**: Provides these numerically compiled structures back to the Stage upon `build_computational_model()`.

## 5.7 Stage to Period Connections

When multiple stages are combined into a single Period, each stage's `arvl.up` is unique (one element), but the `cntn.up` can hold multiple entries—one for each incoming edge from an upstream Stage. This allows for complex model structures where:

- Stage A → Stage B (in the backward sense) means "B's `cntn.up[...]` references A's `arvl.up`"
- If Stage B depends on multiple stages (e.g., A and C), then `B.cntn.up["A"]` and `B.cntn.up["C"]` hold references to the respective arrival perches
- During the backward pass, these references unify with the actual `A.arvl.up` and `C.arvl.up`
- The user's aggregator or solver in Stage B merges these multiple `.up[...]` entries in the continuation perch

**Example**:

```python
# Suppose we define a Period with multiple stages
period.add_stage("A", stageA)
period.add_stage("B", stageB)

# Connect in a backward sense: A -> B
# meaning B depends on A. B.cntn.up["A"] references A.arvl.up
period.connect_stages("A", "B", branch_key="A_branch")

# Later at solve time:
stageB.perches["cntn"].up["A_branch"] = stageA.perches["arvl"].up
# B merges these multiple references if needed
stageB.solve_backward()
```

This structure lets multiple Stage arrivals feed into a single Stage's continuation node. Each stage remains an independent CDC sub-graph, but **the Period** merges them by specifying how the `cntn.up[...]` references are assigned to different upstream `arvl.up`s.

## 5.8 Branching Stages and Discrete Choice Models

The ModCraft framework supports branching structures that are essential for modeling discrete choice problems and multi-path dynamic models:

### Branching Stage Architecture

Branching stages allow the model to represent decision points with multiple possible outcomes:

1. **Structure**: In a branching stage, a single arrival perch (`arvl`) and decision perch (`dcsn`) can connect to multiple continuation outcomes.

2. **Continuation References**: The continuation perch (`cntn`) can hold references to multiple branches via its dictionary-structured `.up` attribute. Each branch is identified by a unique `branch_key`.

3. **Backward Connections**: When connecting stages in the backward direction (for solving), branches are specified with a `branch_key` parameter:
   ```python
   # Example: Connect worker_stage to choice_stage with branch key 'work'
   period.connect_stages("worker_stage", "choice_stage", 
                         direction="backward", 
                         branch_key="work")
   ```

### Discrete Choice Implementation

Discrete choice models are implemented as a special case of branching stages:

1. **Discrete Stage Type**: Stages can be designated with `decision_type="discrete"` to indicate they contain discrete choices.

2. **Value Function Aggregation**: In discrete stages, the backward mover from `cntn` to `dcsn` performs a maximum operation (or expected maximum) over the different branch values in `cntn.up`.

3. **Branch Value References**: Each branch in `cntn.up` references a different alternative's value function. For example:
   - `choice_stage.cntn.up["work"]` holds the value function for choosing to work
   - `choice_stage.cntn.up["retire"]` holds the value function for choosing to retire

### Example: Pension Model with Discrete Choice

A typical pension model might include:

1. **Worker Stage**: Handles consumption/savings decisions if working
2. **Retiree Stage**: Handles consumption decisions if retired
3. **Choice Stage**: A discrete choice stage that decides between working and retiring

The backward connections would be:
- Worker stage's `arvl.up` → Choice stage's `cntn.up["work"]`
- Retiree stage's `arvl.up` → Choice stage's `cntn.up["retire"]`

The choice stage then computes the maximum value between the two alternatives and stores it in `dcsn.up` and `arvl.up`.

### Multi-Path Model Construction

For complex models with multiple paths (e.g., career choices, asset selection):

1. **Multiple Terminal Stages**: A period can have multiple terminal stages, each representing an endpoint of a different path.

2. **Multiple Initial Stages**: Similarly, a period can have multiple initial stages representing different entry points.

3. **Sub-model Merging**: Separate sub-models (e.g., worker and retiree models) can be merged at specific connection points, allowing modular development of complex economic models.

This branching architecture provides the flexibility to model a wide range of economic problems involving discrete choices, regime switching, and path-dependent dynamics while maintaining the clean separation of concerns and modular structure of the ModCraft framework. 