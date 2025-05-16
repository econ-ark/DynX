# StageCraft Theory
> *Mathematical framework for the factored Bellman equation and CDA structure*

This document outlines the mathematical framework underlying StageCraft, with a focus on the [factored Bellman equation](concepts.md#factored-bellman-operator), its CDA (Continuation-Decision-Arrival) decomposition and how the decomposition leads to the computational structure of StageCraft stages.

## Contents
1. [Mathematical Framework](#mathematical-theory)
   - Primal Bellman Equation
   - Factored Bellman Equation
   - Forward Operations

3. [Perches and Movers](#perches-and-movers)
   - Perches
   - Backward Movers
   - Forward Movers
4. [Summary](#summary)

## Mathematical Framework

In order to achieve modularity, stages are constructed to compute a **[factored Bellman equation](concepts.md#factored-bellman-operator)**. The factored Bellman equation is a transformation of the standard Bellman equation (**primal Bellman equation**) that represents each stage's problem as a functional operation on a **[continuation value function](concepts.md#functional-objects)**. The continuation value function has lower complexity than the standard value function. Moreover, unlike the standard value, the continuation value is only a function of endogenous state variables rather than the state variables and "future shocks".

While the idea of a continuation value function has been informally known in the literature, the formal theory of factored Bellman equations is developed by Sargent and Stachurski (2025).

### Primal Bellman Equation


Let's start by letting the state-space (**decision state-space**) of a stochastic recursive problem be $\mathscr{X}\_{v}$. We will use $x_{v,j}$, with $x_{v,j}\in \mathscr{X}\_{v}$ to denote the state (**decision state**) at stage $j$. Letting $\mathscr{V}\_{j}$ be the extended real-valued value function (**decision value function**) for stage $j$, consider the  primal  Bellman equation in primitive form

$$\mathscr{V}\_{j}(x_{v,j}) = \max_{a \in A(x_{v,j})} \{ r(x_{v,j}, a) + \beta \mathbb{E} \mathscr{V}\_{j+1}(x_{v,j+1}) \}$$

where $x_{v,j+1}= f(x_{v,j},a, W_{j+1})$ is the successor stage state as a function of the current state $x_{v,j}$, the action $a$, and the exogenous shock $W_{j+1}$ that is realized at the beginning of stage $j+1$ (the successor stage). Moreover, $r$ is the reward function, $\beta\in (0,1)$ is the discount factor and $A$ is the feasible correspondence. 

> This formulation is not modular because both the current exogenous shock and the successor shock are included in the problem representation.

### Factored Bellman Equation

Let us now assume that the [transition function](concepts.md#transition-functions) $f$ can be factored into two measurable functions: $g_{av,j+1}$ and $g_{ve,j}$ such that $f(x_{v,j},a, W_{j+1}) = g_{av,j+1}(g_{ve,j}(x_{v,j},a),W_{j+1})$. 

The transition function $g_{ve,j}(x_{v,j},a)$ takes the stage $j$ decision state $x_{v,j}$ and action $a$ and returns what is reffered to in the literature as the post-decision state, post-state or **continuation state** -- we adopt the term **continuation state**. Letting $\mathscr{X}\_{e}$ denote the continuation [state-space](concepts.md#euclidean-state-space), denote the continuation state by $x_{e,j}$, where $x_{e,j}\in \mathscr{X}\_{e}$ and  $x_{e,j} = g_{ve,j}(x_{v,j},a)$. (We have dropped the stage subscript $j$ on the state-space, but in general, state-spaces between stages may differ.)

The transition function $g_{av,j+1}$ takes the continuation state (from its preceeding state) and the shock $W_{j+1}$ and returns the decision state for stage $j+1$. However, letting $x_{a,j+1} = g_{av,j+1}(x_{e,j},W_{j+1})$, select a continuation value function, $\mathscr{E}$, such that  $\mathscr{E}(x_{e,j}) = \mathbb{E} \mathscr{V}\_{j+1}(x_{a,j+1})$. The primal Bellman equation can written as

$$
\mathscr{V}\_{j}(x_{v,j}) = \max_{a \in A(x_{v,j})} \{ r(x_{v,j}, a) + \beta \mathscr{E}(g_{ve,j}(x_{v,j},a)) \}
$$

For the stage problem to be modular, $\mathscr{E}$ can be an arbitrary function of the continuation state $x_{e,j}$. In the context of a recursive problem, the assignment $\mathscr{E}\_{j} \leftarrow  \mathbb{E} \mathscr{V}\_{j+1}( g_{av,j+1}(\cdot,W_{j+1}))$ is made to connect stage $j$ and its successor stage $j+1$. Note that for modularity, the expectation must be computed in the successor state (since it contains the successor stage's shock). 

If we compute the expectation $\mathbb{E} \mathscr{V}\_{j+1}(x_{a,j+1})$ in the succesor stage, then the successor stage's "output" is a single (possibly high dimensional) function. To express this more formally, we define the successor stage's **arrival value function**, $\mathscr{A}\_{j+1}$, as $\mathscr{A}\_{j+1} = \mathbb{E} \mathscr{V}\_{j+1}(g_{av,j+1}(\cdot,W_{j+1}))$ and let $x_{a,j+1}$ be the "arrival state" at stage $j+1$. Thus, we have $\mathscr{A}\_{j+1}(x_{a,j+1}) = \mathbb{E} \mathscr{V}\_{j+1}(g_{av,j+1}(x_{a,j+1},W_{j+1}))$.

The succesor stage is then connected to the stage $j$ by the assignments $\mathscr{E}\_{j}\leftarrow \mathscr{A}\_{j+1}$, $x_{e,j}\leftarrow x_{a,j+1}$ and $\mathscr{X}\_{e,j} \leftarrow \mathscr{X}\_{j+1,a}$. In turn defining the arrival stage, arrival state-space and arrival value function for stage $j$, we are led to the factored Bellman equation:

$$
\mathscr{A}\_{j}(x_{a,j}) = \mathbb{E}\max_{a \in A(x_{v,j})} \{ r(x_{v,j}, a) + \beta \mathscr{E}(g_{ve,j}(x_{v,j},a)) \} = \mathbb{E} \mathscr{V}\_{j}(x_{v,j}),
$$

where $x_{v,j} = g_{av,j}(x_{a,j},W_j)$ is the decision state. 

### Backward Operators

We will use $\mathscr{T}^{F}$ to denote the [factored Bellman operator](concepts.md#factored-bellman-operator) and write the factored Bellman operation as $\mathscr{A}\_{j} = \mathscr{T}^{F}\mathscr{E}\_{j}$. Note the factored Bellman operator $\mathscr{T}^{F}$ is composed of two non-trivial numerical computational operators: the maximization operator, which we deonote by $\mathscr{T}^{ev}$ and the expectation operator, which we denote by $\mathscr{T}^{va}$. The maximization operator takes the continuation value $\mathscr{E}\_{j}$ and transforms it to a value function $\mathscr{V}\_{j}$. The expectation operator then furnishes the arrival value function $\mathscr{A}\_{j}$. The factored Bellman operator is the composition of these two operators:

$$
\mathscr{T}^{F} = \mathscr{T}^{va} \circ \mathscr{T}^{ev}
$$

We can write these two operations as to decompose the factored Bellman equation

$$
\mathscr{T}^{va} \mathscr{V}\_{j}(x_{a,j}) = \mathbb{E} \mathscr{V}\_{j}(g_{av,j}(x_{a,j},W_j)), \quad x_{a,j}\in \mathscr{X}\_{a}
$$

$$
\mathscr{T}^{ev} \mathscr{E}\_{j}(x_{v,j}) = \max_{a \in A(x_{v,j})} \{ r(x_{v,j}, a) + \beta \mathscr{E}\_{j}(g_{ve,j}(x_{v,j},a)) \}, \quad x_{v,j}\in \mathscr{X}\_{v}
$$

under the assignments $\mathscr{V}\_{j} \leftarrow \mathscr{T}^{ev} \mathscr{E}\_{j}$ and $\mathscr{A}\_{j} \leftarrow \mathscr{T}^{va} \mathscr{V}\_{j}$, where $\mathscr{E}\_{j}$ is a suitably well-defined arbitrary input function. 

### Forward Operators

Within a stage, there are three distribution operations alongside each value function. Under the [transition functions](concepts.md#transition-functions) and optimal policy, these distributions evolve as follows:

$$
\mu_{a,j} \leftarrow \mu_{e,j-1} \quad \text{// Input from predecessor stage}
$$

$$
\mu_{v,j} = \mu_{a,j} \circ h_{av}^{-1}, \quad h_{av} = g_{av}(\cdot, \cdot) \quad \text{// Arrival to Decision}
$$

$$
\mu_{e,j} = \mu_{v,j} \circ h_{ve}^{-1}, \quad h_{ve} = g_{ve}(\pi(\cdot), \cdot) \quad \text{// Decision to Continuation}
$$

where $\mu_{a,j} \circ h_{av}^{-1}$ and $\mu_{v,j} \circ h_{ve}^{-1}$ are push-forward operations under measurable (not necessarily invertible) transition functions.

These distribution operations represent the simulation of a population or the evolution of the stochastic states under the recursive solution.

## Perches and Movers

### Perches

The evaluation of the backward operators above leads to three value functions: the arrival value function, the decision value function, and the continuation value function. However, when we solve a recursive problem backwards, one may not explicitly compute these three value functions or evaluate **canonical Bellman operators**. For instance, one may use the endogenous grid method where arrival, decision and continuation marginal or shadow value functions are computed. Similarly simulating forward, objects representing distributions may also be instantiated using a variety of **distributional statistics**  (such as densities, histograms, etc.).  depending on the methods used in the application.

We collect the **[functional objects](concepts.md#functional-objects)** associated with each of the states (arrival, decision, continuation) as **[perches](concepts.md#perches)**. The perch objects related to the backward solution will be called **solution objects** and the objects of the population distribution will be called **distribution objects**. The functional objects and solution objects at each perch may or may not contain the canonical functional objects we desribed above. 

As such, once a stage has been solved and simulated:

- The [arrival perch](concepts.md#arrival-perch) (`arvl`) has a solution object (`arvl.sol`) and a distribution object (`arvl.dist`)
- The [decision perch](concepts.md#decision-perch) (`dcsn`) has a solution object (`dcsn.sol`) and a distribution object (`dcsn.dist`)
- The [continuation perch](concepts.md#continuation-perch) (`cntn`) has a solution object (`cntn.sol`) and a distribution object (`cntn.dist`)

### Movers

#### Backward Movers

Similarly, a particular solution algorithm may not explicitly compute the maximization operation or expectation operation as above. Thus we will refer to the computational objects associated with the operation which takes:
- The continuation perch to the decision perch as the continuation-to-decision [mover](concepts.md#backward-movers) (`cntn_to_dcsn`)
- The decision perch to the arrival perch as the decision-to-arrival [mover](concepts.md#backward-movers) (`dcsn_to_arvl`)

#### Forward Movers

The forward operations taking each `.dist` operation from one perch to its successor perch will be called [forward movers](concepts.md#forward-movers):
- Arrival-to-decision mover (`arvl_to_dcsn`)
- Decision-to-continuation mover (`dcsn_to_cntn`)


> Each [mover](concepts.md#movers) not only contains the mathematical and computational information implement its operation, but also performs the assignment of the output to the appropriate perch. Thus, the continuation-to-decision mover represents the evaluation $\mathscr{T}^{ev} \mathscr{E}\_{j}$ and the assignment $\mathscr{V}\_{j} \leftarrow \mathscr{T}^{ev} \mathscr{E}\_{j}$. Evaluating the operations are related to the `mover.model` attribute and `mover.comp` attribute of the mover. Outgoing assignment is captured by the `mover.target_name` `mover.target_key` attributes of the mover. Incoming assignment is captured by the `mover.source_name` `mover.source_key` attributes of the mover. An implication is that inter-stage connections are represented by the mover objects (we detail inter state movers when we review **[periods](concepts.md#period)** and **[model circuits](concepts.md#modelcircuit)**.)

The following diagram illustrates each perch (`arvl`, `dcsn`, `cntn`) along with its canonical **functional objects** and each forward mover (`arvl_to_dcsn`, `dcsn_to_cntn`) and backward mover (`dcsn_to_arvl`, `cntn_to_dcsn`) with its canonical **functional operation**.

<p align="center">
  <img src="/assets/stage_cycle_basic.png" alt="Stage Cycle Basic Diagram" width="600"/>
</p>

## Summary

### State Spaces and Transitions

The [state spaces](concepts.md#euclidean-state-space) and [transitions](concepts.md#transition-functions) form the foundation of the CDA structure:

**State Spaces**
| State Type | Description | Powell (2011) Equivalent |
|------------|-------------|-------------------------|
| Arrival ($x_{a,j}$) | Beginning of stage | Post-decision state ($S_{j-1}^x$) |
| Decision ($x_{v,j}$) | After shock realization | Pre-decision state ($S_j$) |
| Continuation ($x_{e,j}$) | After decision made | Post-decision state ($S_j^x$) |

**Transition Functions**

$$
x_{v,j} = g_{a\rightarrow v}(x_{a,j}, W_j) \quad \text{// Arrival to Decision}
$$

$$
x_{e,j} = g_{v\rightarrow e}(x_{v,j}, \pi_j) \quad \text{// Decision to Continuation}
$$

$$
x_{a,j+1} = g_{e\rightarrow a}(x_{e,j}) \quad \text{// Continuation to next Arrival}
$$

### Value Functions and Operators

The [factored Bellman operator](concepts.md#factored-bellman-operator) $\mathscr{T}^{F}$ decomposes into three operators:

$$
\mathscr{T}^{F} = \mathscr{T}^{va} \circ \mathscr{T}^{ev}
$$

These [operators](concepts.md#functional-operators) act on their respective value functions:

$$
\mathscr{T}^{va} \mathscr{V}\_{j}(x_{a,j}) = \mathbb{E} \mathscr{V}\_{j}(g_{av,j}(x_{a,j},W_j)), \quad x_{a,j}\in \mathscr{X}\_{a,j}
$$

$$
\mathscr{T}^{ev} \mathscr{E}\_{j}(x_{v,j}) = \max_{a \in A(x_{v,j})} \{ r(x_{v,j}, a) + \beta \mathscr{E}\_{j}(g_{ve,j}(x_{v,j},a)) \}, \quad x_{v,j}\in \mathscr{X}\_{v,j}
$$

### Perches and Movers

**Perch Types and Contents**
| Perch | Solution Objects (`sol`) | Distribution Objects (`dist`) |
|-------|-------------------------|------------------------------|
| [Arrival](concepts.md#arrival-perch) (`arvl`) | Value function, shadow values, policy etc. | Distributional statistics at stage start |
| [Decision](concepts.md#decision-perch) (`dcsn`) | Value function, shadow values etc. | Distributional statistics after shock realization |
| [Continuation](concepts.md#continuation-perch) (`cntn`) | Value function, shadow values etc. | Distributional statistics after decisions |

**[Forward Movers](concepts.md#forward-movers)** (Distribution Evolution)
| Type | Canonical Operation | Description |
|------|-------------------|-------------|
| Arrival to Decision (`arvl_to_dcsn`) | $\mu_{v,j} = \mu_{a,j} \circ h_{av}^{-1}$ | Maps `arvl.dist` to `dcsn.dist` |
| Decision to Continuation (`dcsn_to_cntn`) | $\mu_{e,j} = \mu_{v,j} \circ h_{ve}^{-1}$ | Maps `dcsn.dist` to `cntn.dist` |

**[Backward Movers](concepts.md#backward-movers)** (Solution Methods)
| Type | Canonical Operation | Description |
|------|-------------------|-------------|
| Decision to Arrival (`dcsn_to_arvl`) | $\mathscr{T}^{va}: \mathscr{V} \mapsto \mathscr{A}$ | Maps `dcsn.sol` to `arvl.sol` |
| Continuation to Decision (`cntn_to_dcsn`) | $\mathscr{T}^{ev}: \mathscr{E} \mapsto \mathscr{V}$ | Maps `cntn.sol` to `dcsn.sol` |

