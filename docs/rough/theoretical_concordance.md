# 6. Theoretical to Computational Concordance

This section provides a mapping between theoretical elements in MDP and CDC formulations and their concrete implementations in the ModCraft framework code, using the three-move symbols from the formal MDP theory.

## 6.1 Three-Move MDP Structure

The ModCraft framework directly implements the "three-move" structure for MDPs with post-states. In this structure, each stage j consists of:

1. **Arrival State** (x_{a,j}): The state in which the system arrives at stage j
2. **Decision State** (x_{v,j}): The state after shock realization where decisions are made
3. **Continuation State** (x_{e,j}): The post-decision state after actions are chosen

The transitions between these states are governed by:
- **Arrival to Decision**: x_{v,j} = g_{a→v}(x_{a,j}, W_j) where W_j is the shock
- **Decision to Continuation**: x_{e,j} = g_{v→e}(x_{v,j}, π_j) where π_j is the action
- **Continuation to Next Arrival**: x_{a,j+1} = g_{e→a}(x_{e,j})

The value functions associated with these states are:
- **Arrival Value Function**: 𝒜(x_a)
- **Decision Value Function**: 𝒱(x_v)
- **Continuation Value Function**: ℰ(x_e)

The functional operators that implement backward induction are:
- **Arrival Operator** (𝒯^a): Maps decision values to arrival values
- **Decision Operator** (𝒯^v): Maps continuation values to decision values 
- **Continuation Operator** (𝒯^e): Maps next period's arrival values to continuation values

The composite operator 𝒯 = 𝒯^a ∘ 𝒯^v ∘ 𝒯^e is equivalent to the Bellman operator for MDPs with post-states.

## 6.2 Concordance Table

| Theoretical Element | Mathematical Notation | ModCraft Implementation | Implementation Details |
|---------------------|------------------------|-------------------------|------------------------|
| **Arrival State** | x_{a,j} | `perch_arvl.model.num.state_space` | State space grid in arrival perch |
| **Decision State** | x_{v,j} | `perch_dcsn.model.num.state_space` | State space grid in decision perch |
| **Continuation State** | x_{e,j} | `perch_cntn.model.num.state_space` | State space grid in continuation perch |
| **Action/Control** | π_j | `dcsn.up` (policy) | Policy function stored in decision perch |
| **Arrival Value Function** | 𝒜(x_a) | `arvl.up` | Stored in arrival perch's `.up` attribute |
| **Decision Value Function** | 𝒱(x_v) | `dcsn.up` | Stored in decision perch's `.up` attribute |
| **Continuation Value Function** | ℰ(x_e) | `cntn.up` | Stored in continuation perch's `.up` attribute |
| **Arrival to Decision Transition** | g_{a→v}(x_{a,j}, W_j) | `mover_arvl_to_dcsn.model.math.transition_equations` | Transition equation in arrival-to-decision mover |
| **Decision to Continuation Transition** | g_{v→e}(x_{v,j}, π_j) | `mover_dcsn_to_cntn.model.math.transition_equations` | Transition equation in decision-to-continuation mover |
| **Continuation to Arrival Transition** | g_{e→a}(x_{e,j}) | `mover_cntn_to_arvl.model.math.transition_equations` | Transition equation in continuation-to-arrival mover |
| **Reward Function** | U(x_{v,j}, π_j) | `model.math.functions.reward` | Defined in model math spec |
| **Discount Factor** | β | `model.parameters.beta` | Stored in model parameters |
| **Shock Process** | W_j | `model.num.shocks` | Defined in numerical representation |
| **Arrival Operator** | 𝒯^a | `mover_dcsn_to_arvl.backward()` | Backward method in arrival mover |
| **Decision Operator** | 𝒯^v | `mover_cntn_to_dcsn.backward()` | Backward method in decision mover |
| **Continuation Operator** | 𝒯^e | `mover_arvl_to_cntn.backward()` | Backward method in continuation mover |
| **Composite CDC Operator** | 𝒯 = 𝒯^a ∘ 𝒯^v ∘ 𝒯^e | `stage.solve_backward()` | Full backward induction in stage |
| **Distribution Function** | F(x) | `perch.down` | Distribution stored in perch's `.down` attribute |

## 6.3 Implementation Workflow with Three-Move Structure

The ModCraft workflow implements the three-move structure as follows:

1. **Mathematical Model Definition** (`stage.load_config()` using `init_rep`)
   - Defines the state spaces for each perch (x_{a,j}, x_{v,j}, x_{e,j})
   - Specifies transition functions (g_{a→v}, g_{v→e}, g_{e→a})
   - Defines reward function U(x_{v,j}, π_j) and other mathematical elements

2. **Numerical Implementation** (`stage.build_computational_model()` using `num_rep`)
   - Creates concrete numerical representations of state spaces and transitions
   - Implements shock processes W_j as discrete or continuous distributions
   - Compiles mathematical functions into callable numerical functions

3. **Backward Induction** (`stage.solve()`)
   - Implements the composite operator 𝒯 = 𝒯^a ∘ 𝒯^v ∘ 𝒯^e
   - Solves for 𝒜(x_a), 𝒱(x_v), and ℰ(x_e) through backward operations
   - Stores value functions in the respective perches' `.up` attributes

4. **Forward Simulation** (`stage.simulate()`)
   - Uses the policy function π_j and transitions to simulate state evolution
   - Applies the transitions g_{a→v}, g_{v→e}, g_{e→a} in sequence
   - Populates perches' `.down` attributes with state distributions

## 6.4 Package Structure and Key Functions

| Module/Class | Purpose | Key Methods | Relation to Three-Move Theory |
|--------------|---------|-------------|------------------------------|
| `Stage` | Core container for CDC structure | `load_config()`, `build_computational_model()`, `solve()`, `simulate()` | Implements a complete stage j with all three moves |
| `ArrivalPerch` | Arrival state container | `get_up()`, `set_up()` | Stores x_{a,j} state space and 𝒜(x_a) value function |
| `DecisionPerch` | Decision state container | `get_up()`, `set_up()` | Stores x_{v,j} state space and 𝒱(x_v) value function |
| `ContinuationPerch` | Continuation state container | `get_up()`, `set_up()` | Stores x_{e,j} state space and ℰ(x_e) value function |
| `ArrivalToDecisionMover` | A→D transition | `forward()`, `backward()` | Implements g_{a→v} transition and shock integration |
| `DecisionToContinuationMover` | D→C transition | `forward()`, `backward()` | Implements g_{v→e} transition and optimization |
| `ContinuationToArrivalMover` | C→A transition | `forward()`, `backward()` | Implements g_{e→a} transition between periods |
| `FunctionalProblem` | Model representation | Contains `.math`, `.num` | Holds mathematical and numerical representations of the MDP |
| `Period` | Multi-stage container | `add_stage()`, `connect_stages()` | Links multiple stages j into a complete model | 