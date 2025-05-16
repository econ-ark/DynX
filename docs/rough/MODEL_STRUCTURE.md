Below is the **lightly edited** summary document, **plus** a short section on **joining Stages** into a **Period** via backward connections—where each Stage’s `cntn.up` can hold **multiple** branches from different upstream Stage arrivals:

---

# StageCraft Model Structure

This document provides a visual overview of the StageCraft model structure, attribute organization, and access patterns for its standard three-node circuit (perches), reflecting the state after configuration and compilation. It also introduces how **multiple stages** can be joined in a **Period**, connecting them in a backward sense such that each stage’s continuation perch (`cntn`) can store multiple incoming `.up` references (branches) from preceding stages.

## **Key Stage Attributes**

A **Stage** object orchestrates the model's computation and holds its core components. Here are its primary attributes:

- **name** (`str`): A user-defined name for the stage instance.  
- **status_flags** (`Dict[str, bool]`): A dictionary tracking the computational status (e.g., `initialized`, `compiled`, `solved`).  
- **perches** (`Dict[str, Perch]`): A dictionary containing the **Perch** objects (nodes) of the circuit, keyed by their names (e.g., `'arvl'`, `'dcsn'`, `'cntn'`). Perches hold state variables and computed data (`.up`, `.down`).  
- **forward_graph** (`networkx.DiGraph`): A directed graph representing the forward flow of computation. Edges connect perches and hold **Mover** objects. Each Mover contains its own `.model` (`FunctionalProblem`) attribute with the parameters, math, and num components needed to perform the operation between the connected perches (e.g., for simulation).  
- **backward_graph** (`networkx.DiGraph`): A directed graph representing the backward flow of computation. Edges connect perches and hold **Mover** objects. Each Mover contains its own `.model` (`FunctionalProblem`) attribute with the parameters, math, and num components needed to perform the operation between the connected perches (e.g., for solving).  
- **model** (`FunctionalProblem`): An aggregated representation of the entire stage's model, combining parameters, settings, mathematical definitions (`.math`), and numerical components (`.num`) from perches and movers after loading and compilation.  
- **init_rep** (`Module | Object`): A reference to the plugin component responsible for parsing the configuration and defining the initial mathematical structure (`.math`).  
- **num_rep** (`Module | Object`): A reference to the plugin component responsible for generating the numerical implementation (`.num`) from the mathematical structure (`.math`).  

## **Overview: Stage vs. Plugin Responsibilities**

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

## **Key Attribute Ownership Summary**

| Attribute/Component          | Primary Owner / Origin      | Description                                                                     |
|------------------------------|-----------------------------|---------------------------------------------------------------------------------|
| **Stage.name**               | Stage                       | Name of the stage instance.                                                     |
| **Stage.status_flags**       | Stage                       | Overall status dictionary.                                                      |
| **Stage.perches** (dict)     | Stage                       | Dictionary holding Perch objects.                                               |
| **Stage.forward_graph**, <br/>**.backward_graph**          | Stage                       | networkx graphs defining mover connections.                                     |
| **Stage.init_rep**, <br/>**Stage.num_rep**                 | Stage (references)         | References to plugin objects/modules.                                           |
| `model.math`, <br/>`.parameters`, <br/>`.settings`         | Plugin (`init_rep`)        | Mathematical definitions, params, settings (loaded from config).                |
| `model.num`                  | Plugin (`num_rep`)          | Numerical functions, grids, shocks (compiled after build).                      |
| `perch.up`, `perch.down`     | Stage (runtime)            | Computed value functions, policies, distributions (populated by solve/sim).     |
| **Mover .comp** (Operator)   | Stage (set via whisperer)   | The actual callable operator for the mover.                                     |

In essence, the Stage provides the standardized "circuit board," while the Plugin provides the specific "components" (mathematical definitions and numerical implementations) that are placed onto that board according to the user's configuration.

## **Model Structure Diagram (Post-Compilation Example)**

```
Stage
├── name: str    # e.g., "ConsumptionSavingsIID"
├── status_flags: Dict[str, bool]   # Flags updated during lifecycle
│   └── { "initialized": True, "compiled": True, "solvable": ..., 
│         "solved": ..., "simulated": ..., "portable": ..., 
│         "all_perches_initialized": True, "all_perches_compiled": True, 
│         "all_movers_initialized": True, "all_movers_compiled": True }
├── perches: Dict[str, Perch]
│   ├── "arvl": Perch
│   │   ├── name: "arvl"
│   │   ├── up: None   # Data populated by solve/simulate
│   │   ├── down: None # Data populated by solve/simulate
│   │   └── model (FunctionalProblem)
│   │       ├── parameters: Dict[...]
│   │       ├── settings: Dict[...]
│   │       ├── math:
│   │       │   └── state: Dict[...]     # Perch-specific state def
│   │       └── num:
│   │           ├── state_space: {'arvl': {'grids': {'a': array}}}
│   │           └── arvl: Dict[...]      # Perch-specific num components
│   ├── "dcsn": Perch(...)
│   │   └── model (FunctionalProblem)
│   │       ├── ... (similar structure) ...
│   │       └── num:
│   │           ├── state_space: {'dcsn': {'grids': {'m': array}}}
│   │           └── dcsn: Dict[...]
│   └── "cntn": Perch(...)
│       └── model (FunctionalProblem)
│           ├── ... (similar structure) ...
│           └── num:
│               ├── state_space: {'cntn': {'grids': {'a_nxt': array}}}
│               └── cntn: Dict[...]
├── forward_graph: nx.DiGraph   # Movers attached to edges
│   └── Edges & Attached Movers:
│       ├── ("arvl", "dcsn"): Mover(...)
│       │   └── model (FunctionalProblem)
│       └── ("dcsn", "cntn"): Mover(...)
│           └── model (FunctionalProblem)
├── backward_graph: nx.DiGraph  # Movers attached to edges
│   └── Edges & Attached Movers:
│       ├── ("dcsn", "arvl"): Mover(...)
│       └── ("cntn", "dcsn"): Mover(...)
├── model (FunctionalProblem)   # Aggregated stage model
│   ├── parameters: Dict[...]
│   ├── settings: Dict[...]
│   ├── math: {...}
│   └── num: {...}
├── methods: Dict
├── whisperer: Any | None
├── init_rep: Module | Object    # Provided at creation
├── num_rep: Module | Object     # Provided at creation
└── __model_representations: Dict # Internal (private)
```

## **Status Flags**

The stage tracks its state using `status_flags`. Here’s what each core flag indicates:

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
  True once the Stage’s movers have attached computational methods (`.comp`), making it self-contained for serialization or reuse.

## **Model Initialization Flow**

The typical flow:

1. **Create Stage**  
2. **Load Config** (`stage.load_config()` → sets `initialized`)  
3. **Build Model** (`stage.build_computational_model()` → sets `compiled`)  
4. **Set Initial Values** (User provides data → sets `solvable`)  
5. **Solve** (`stage.solve()` → sets `solved`)  
6. **Simulate** (`stage.simulate()` → sets `simulated`)

## **Common Access Patterns**

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

## **Joining Multiple Stages into a “Period” (Backward Linking)**

When **multiple** stages are combined into a single **Period**, each stage’s **`arvl.up`** is **unique** (one element), but the **`cntn.up`** can hold **multiple** entries—one for **each** incoming edge from an upstream Stage. For example:

- Stage **A** → Stage **B** (backward sense) implies “B’s `cntn.up[...]` references A’s `arvl.up`.”  
- If **B** also depends on Stage **C**, then `B.cntn.up["A"] = None` and `B.cntn.up["C"] = None` placeholders at config time. During a backward pass, these references unify with the actual `A.arvl.up` and `C.arvl.up`.  
- The user’s aggregator or solver in Stage B merges those multiple `.up[...]` entries in `cntn`.

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

This structure lets multiple Stage arrivals feed into a single Stage’s continuation node. Each stage remains an independent CDC sub-graph, but **the Period** merges them by specifying how the `cntn.up[...]` references are assigned to different upstream `arvl.up`s.

---

**End of Document**