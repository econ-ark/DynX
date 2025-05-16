# Discrete Housing Choice Model with Renting
Author: Akshay Shanker, UNSW, Sydney 
-----------------------------------

We now turn to the discrete choice model of housing decisions where individuals can purchase a home as durable stock or rent housing services. In addition to facing fixed costs to buying and selling a house, the housing choice grid itself is discrete. The application is a simple extension of Fella (2014). However, with added renting, the policy functions of the individual are non-monotone. As such, this application demonstrates an important case where existing discrete-continuous algorithms in the literature cannot accurately recover the correct approximate policy.

## Model Environment

Consider an agent who draws an infinite sequence of bounded, stationary Markov wage shocks $(y_{t})_{t=0}^{\infty}$. Each period the agent decides whether they will be a renter or home-owner. They then decide how much to consume of non-durable goods $c_{t}$, how much to save in liquid (financial) assets $a_{t+1}$, and how much to save in housing assets $H_{t+1}$ if they are an owner and their housing services $H^{\text{rent}}_{t}$ if they are a renter. Similar to Application 2, housing assets serve a dual role, namely they are a form of investment good but also provide durable consumption services. Moreover, housing can only be purchased in discrete amounts on a finite grid $\mathbb{H}$, and adjusting housing each period incurs a fixed transaction cost $\tau H_{t+1}$, with $\tau \in [0,1)$. We assume no borrowing and so, $a_{t}\geq 0$ must hold.

The agent's maximization problem yields the following value function:

$$V_{0}(a,y,H) = \max_{(a_{t}, \tilde{H}_{t})_{t = 0}^{\infty}}\mathbb{E}\sum_{t=0}^{\infty}\beta^{t}u(c_{t},H_{t+1})$$

### Canonical Bellman Equations in Primal Form 

The Bellman equation is:

$$V(a,y,H) = \max_{c,H^{\prime}}\left\{u(c,H^{\prime}) + \mathbb{E}_{y}V^{\prime}(a^{\prime},y^{\prime},H^{\prime})\right\}$$

such that $a^{\prime}$ satisfies the feasibility condition $a^{\prime}\geq 0$ and the budget constraint holds. Since this is an infinite horizon problem, recall the fixed point to the Bellman equation yields the value function, and the maximizing correspondence yields the policy functions.

For the home owner who decides to move house (adjuster), the budget constraint is:

$$a_{t+1}+ c_{t} = (1+r)a_{t} + y_{t} + d^{adj}_{t}H_{t} - (1+d^{adj}_{t}\tau) H_{t+1} := w_{t}^{\text{own}}$$

where $d^{adj}_{t} = \mathbb{1}_{H_{t+1}\not= H_{t}}$. For the renter:

$$a_{t+1}+ c_{t} = (1+r)a_{t} + y_{t} + H_{t} - P^{r}S_{t} := w^{\text{rent}}_{t}$$

with the per-period utility function defined by:

$$u(c,H) = \alpha\log(c) + (1-\alpha)\log(\kappa(H+\iota)), \qquad \qquad \kappa>0, \iota>0$$

The owner's Bellman equation becomes, conditional on a housing stock choice of $H_{t+1}$:

$$\hat{V}_{t}^{\text{own}}(H_{t+1},y_{t},w_{t}^{\text{own}}) = \max_{c_{t}}\left\{u(c_{t},H_{t+1}) + \mathbb{E}_{y_{t}}V_{t+1}(a_{t+1},y_{t+1},H_{t+1})\right\}$$

such that $a_{t+1} + c_{t} = w^{\text{owner}}_{t}$. For a renter, the consumption choice is given by:

$$\hat{V}_{t}^{\text{rent}}(S_{t},y_{t}, w_{t}^{\text{rent}}) = \max_{c_{t}}\left\{u(c_{t},S_{t}) + \mathbb{E}_{y_{t}}V_{t+1}(a_{t+1},y_{t+1},0)\right\}$$

The housing choice for the owner is then given by Bellman:

$$V_{t}^{\text{own}}(a_{t}, y_{t}, H_{t}) = \max_{H_{t+1}} \hat{V}_{t}^{\text{own}}(H_{t+1},y_{t},w_{t}^{\text{own}})$$

such that $w_{t}^{\text{own}}\geq 0$, and the housing choice for the renter is given by Bellman:

$$V_{t}^{\text{renter}}(w_{t}, y_{t}) = \max_{S_{t}} \hat{V}_{t}^{\text{rent}}(S_{t},y_{t},w_{t}^{\text{rent}})$$

such that $w_{t}^{\text{rent}}\geq 0$. The renter-owner choice yields the period $t$ value function via:

$$V_{t}(a_{t}, y_{t}, H_{t}) = \max_{\text{rent}, \text{own}} \left\{V_{t}^{\text{renter}}((1+r)a + H_{t}, y_{t}), V_{t}^{\text{own}}(a_{t}, y_{t}, H_{t})\right\}$$

## Perch-Level Value Functions in Heptapod-B

In the Heptapod-B framework, we use a three-perch structure with value functions associated with each perch:

| Perch | Value function | Marginal value (w.r.t. liquid assets `a` or cash-on-hand `w`) |
|-------|----------------|---------------------------------------------------------------|
| Arrival ($\mathscr{X}_a$) | $V_a(\cdot)$ | $\Lambda_a(\cdot) \equiv \partial_{a}V_a$ |
| Decision ($\mathscr{X}_v$) | $V_v(\cdot)$ | $\Lambda_v(\cdot) \equiv \partial_{w}V_v$ |
| Continuation ($\mathscr{X}_e$) | $V_e(\cdot)$ | $\Lambda_e(\cdot) \equiv \partial_{a^{\text{nxt}}}V_e$ |

For stages with **branching continuations** (e.g., rent vs own) we write:
$V_e^{\text{own}}$, $V_e^{\text{rent}}$ and the corresponding $\Lambda_e^{(\cdot)}$.

## Stages in the **StageCraft** Framework

We will write the problem in the three-perch (`Arrival → Decision → Continuation`) layout of StageCraft. 
We use $H^{\text{nxt}}$ to denote the housing stock once it has been decided in the period. Thus, at time $t$, we have the assignment $H_{t+1} \leftarrow H^{\text{nxt}}$. Arrival liquid assets are simply $a$ and assets taken into the next period will be $a^{\text{nxt}}$.

State variables are always listed **in the same order** inside every function that uses them.

### 1. Home-Owner Consumption Choice Stage [OWNC]

#### State Spaces and Variables

| Perch | State Variables | Domain |
|-------|----------------|---------|
| $\mathscr{X}_a$ | $(H^{\text{nxt}}, y, w)$ | $H^{\text{nxt}} \in \mathbb{H}$, $y \in \mathbb{Y}$, $w \in \mathbb{R}_+$ |
| $\mathscr{X}_v$ | $(H^{\text{nxt}}, y, w)$ | same as above |
| $\mathscr{X}_e$ | $(a^{\text{nxt}}, y, H^{\text{nxt}})$ | $a^{\text{nxt}} \in \mathbb{R}_+$, $y \in \mathbb{Y}$, $H^{\text{nxt}} \in \mathbb{H}$ |

**Choice Variables**: $c \in [0, w]$ (consumption)

#### Core Functions
- **Utility**: $u(c, H^{\text{nxt}})$
- **Marginal utility**: $u_c(c, H^{\text{nxt}}) = \partial_c u$
- **Inverse marginal utility**: $u_c^{-1}(\ell, H^{\text{nxt}})$

#### Transition Functions
$$
g_{av}(H^{\text{nxt}}, y, w) = (H^{\text{nxt}}, y, w), \qquad
g_{ve}(H^{\text{nxt}}, y, w, c) = (w-c, y, H^{\text{nxt}}).
$$

#### Backward Operators (EGM-style)

$$
\begin{aligned}
V_v(H^{\text{nxt}}, y, w) &= \max_{c \in [0,w]}\{u(c, H^{\text{nxt}}) + \beta V_e(w-c, y, H^{\text{nxt}})\}, \\
\Lambda_e(a^{\text{nxt}}, y, H^{\text{nxt}}) &= \beta \Lambda_e(a^{\text{nxt}}, y, H^{\text{nxt}}), \\
c^{\text{EGM}} &= u_c^{-1}(\Lambda_e, H^{\text{nxt}}), \qquad m^{\text{EGM}} = c^{\text{EGM}} + a^{\text{nxt}}, \\
Q(a^{\text{nxt}}, y, H^{\text{nxt}}) &= u(c^{\text{EGM}}, H^{\text{nxt}}) + \beta V_e(a^{\text{nxt}}, y, H^{\text{nxt}}).
\end{aligned}
$$

#### Factored Bellman Operators

Here we use EGM with FUES (Forward Upper Envelope Search):

* **Continuation to Decision (Backward)**:

  The Bellman equation:
  $$\mathscr{T}^{ev} \mathscr{E}(H^{\text{nxt}}, y, w) = \max_{c \in [0,w]} \{ u(c, H^{\text{nxt}}) + \beta \mathscr{E}(w-c, y, H^{\text{nxt}}) \}$$

  The Carroll-policy operator for the unconstrained region:
  $$\mathbb{C}\Lambda^{e}(a^{\text{nxt}}, y, H^{\text{nxt}}) = u_c^{-1}(\beta \Lambda^{e}(a^{\text{nxt}}, y, H^{\text{nxt}}), H^{\text{nxt}})$$

  The Carroll-state operator:
  $$\mathbb{C}\Lambda^{e}(a^{\text{nxt}}, y, H^{\text{nxt}}) = u_c^{-1}(\beta \Lambda^{e}(a^{\text{nxt}}, y, H^{\text{nxt}}), H^{\text{nxt}}) + a^{\text{nxt}}$$

  The L-operator:
  $$\mathbb{L}\Lambda^{e}(a^{\text{nxt}}, y, H^{\text{nxt}}) = u_c(c, H^{\text{nxt}})$$ 

  The Q-function:
  $$Q(a^{\text{nxt}}, y, H^{\text{nxt}}) = u(c, H^{\text{nxt}}) + \beta \mathscr{E}(a^{\text{nxt}}, y, H^{\text{nxt}})$$

* **Decision to Arrival (Backward)**:
  $$\mathscr{T}^{va} \mathscr{V}(H^{\text{nxt}}, y, w) = \mathscr{V}(H^{\text{nxt}}, y, w)$$ (Identity)

### 2. Renter Consumption Choice Stage [RNTC]

Identical to OWNC with $H^{\text{nxt}}$ replaced by rental services $S$ (and the continuation perch fixes housing at $0$).

#### State Spaces and Variables

| Perch | State Variables | Domain |
|-------|----------------|---------|
| $\mathscr{X}_a$ | $(S, y, w)$ | $S \in \mathbb{H}$, $y \in \mathbb{Y}$, $w \in \mathbb{R}_+$ |
| $\mathscr{X}_v$ | $(S, y, w)$ | same as above |
| $\mathscr{X}_e$ | $(a^{\text{nxt}}, y, S)$ | $a^{\text{nxt}} \in \mathbb{R}_+$, $y \in \mathbb{Y}$, $S \in \mathbb{H}$ |

**Choice Variables**: $c \in [0, w]$ (consumption)

#### Transition Functions
$$
g_{av}(S, y, w) = (S, y, w), \qquad
g_{ve}(S, y, w, c) = (w-c, y, S).
$$

#### Factored Bellman Operators

* **Continuation to Decision (Backward)**:
  $$\mathscr{T}^{ev} \mathscr{E}(S, y, w) = \max_{c \in [0,w]} \{ u(c, S) + \beta \mathscr{E}(w-c, y, 0) \}$$

* **Decision to Arrival (Backward)**:
  $$\mathscr{T}^{va} \mathscr{V}(S, y, w) = \mathscr{V}(S, y, w)$$ (Identity)

### 3. Owner Housing Choice Stage [OWNH]

#### State Spaces and Variables

| Perch | State Variables | Domain |
|-------|----------------|---------|
| $\mathscr{X}_a$ | $(a, y, H)$ | $a \in \mathbb{R}_+$, $y \in \mathbb{Y}$, $H \in \mathbb{H}$ |
| $\mathscr{X}_v$ | $(a, y, H)$ | same as above |
| $\mathscr{X}_e$ | $(H^{\text{nxt}}, y, w^{\text{own}})$ | $H^{\text{nxt}} \in \mathbb{H}$, $y \in \mathbb{Y}$, $w^{\text{own}} \in \mathbb{R}_+$ |

**Choice Variables**: $H^{\text{nxt}} \in \mathbb{H}$ (next-period housing)

#### Transition Functions

* **Arrival to Decision**: $g_{av}(a, y, H) = (a, y, H)$ (Identity)
* **Decision to Continuation**: $g_{ve}(a, y, H, H^{\text{nxt}}) = (H^{\text{nxt}}, y, w^{\text{own}})$

Budget link:  
$$
w^{\text{own}} = (1+r)a + y + d^{\text{adj}}H - (1+d^{\text{adj}}\tau)H^{\text{nxt}}, \quad d^{\text{adj}} = \mathbf{1}_{H^{\text{nxt}} \neq H}.
$$

#### Backward Operator and Factored Bellman Operators

Backward operator:  
$$
V_v(a, y, H) = \max_{H^{\text{nxt}} \in \mathbb{H}} V_e(H^{\text{nxt}}, y, w^{\text{own}}).
$$

* **Continuation to Decision (Backward)**:
  $$\mathscr{T}^{ev} \mathscr{E}(a, y, H) = \max_{H^{\text{nxt}} \in \mathbb{H}} \{ \mathscr{E}(H^{\text{nxt}}, y, w^{\text{own}}) \}$$

  Method: Discrete choice over $H^{\text{nxt}}$ without branching. 

* **Decision to Arrival (Backward)**:
  $$\mathscr{T}^{va} \mathscr{V}(a, y, H) = \mathscr{V}(a, y, H)$$ (Identity)

### 4. Renter Housing Choice Stage [RNTH]

Analogous to OWNH with $(a, y, H) \to (w, y)$ and $(H^{\text{nxt}}, y, w^{\text{own}}) \to (S, y, w^{\text{rent}})$.

#### State Spaces and Variables

| Perch | State Variables | Domain |
|-------|----------------|---------|
| $\mathscr{X}_a$ | $(w, y)$ | $w \in \mathbb{R}_+$, $y \in \mathbb{Y}$ |
| $\mathscr{X}_v$ | $(w, y)$ | same as above |
| $\mathscr{X}_e$ | $(S, y, w^{\text{rent}})$ | $S \in \mathbb{H}$, $y \in \mathbb{Y}$, $w^{\text{rent}} \in \mathbb{R}_+$ |

**Choice Variables**: $S \in \mathbb{H}$ such that $P^rS \leq w$ (rental housing services)

Budget:  
$$w^{\text{rent}} = w - P^{r}S.$$

#### Transition Functions

* **Arrival to Decision**: $g_{av}(w, y) = (w, y)$ (Identity)
* **Decision to Continuation**: $g_{ve}(w, y, S) = (S, y, w-P^rS)$

#### Factored Bellman Operators

* **Continuation to Decision (Backward)**:
  $$\mathscr{T}^{ev} \mathscr{E}(w, y) = \max_{S \in \mathbb{H}, P^rS \leq w} \{ \mathscr{E}(S, y, w - P^rS) \}$$

* **Decision to Arrival (Backward)**:
  $$\mathscr{T}^{va} \mathscr{V}(w, y) = \mathscr{V}(w, y)$$ (Identity)

  Method: Discrete choice over $S$ without branching.

### 5. Tenure Choice Stage [TENU]

#### State Spaces and Variables

| Perch | State Variables | Domain |
|-------|----------------|---------|
| $\mathscr{X}_a$ | $(a, H, y^{\text{pre}})$ | $a \in \mathbb{R}_+$, $H \in \mathbb{H}$, $y^{\text{pre}} \in \mathbb{Y}$ |
| $\mathscr{X}_v$ | $(a, H, y)$ | $a \in \mathbb{R}_+$, $H \in \mathbb{H}$, $y \in \mathbb{Y}$ |
| $\mathscr{X}_e^{\text{own}}$ | $(a, y, H)$ | $a \in \mathbb{R}_+$, $y \in \mathbb{Y}$, $H \in \mathbb{H}$ |
| $\mathscr{X}_e^{\text{rent}}$ | $(w, y)$ | $w \in \mathbb{R}_+$, $y \in \mathbb{Y}$ |

**Choice Variables**: Discrete choice between renting and owning

#### Transition Functions

* **Arrival to Decision**: $g_{av}(a, H, y^{\text{pre}}) = (a, H, f(y^{\text{pre}}, \xi))$
* **Decision to Continuation (Own)**: $g^{\text{own}}_{ve}(a, y, H) = (a, y, H)$ (Identity)
* **Decision to Continuation (Rent)**: $g^{\text{rent}}_{ve}(a, y, H) = ((1+r)a + H, y)$

#### Backward Operators

Two-branch continuation:

$$
V_v(a, H, y) = \max\{V_e^{\text{rent}}((1+r)a+H, y), V_e^{\text{own}}(a, y, H)\}.
$$

Arrival update integrates over wage shock $\xi_t$:

$$
V_a(a, H, y^{\text{pre}}) = \mathbb{E}_\xi[V_v(a, H, f(y^{\text{pre}}, \xi))].
$$

#### Factored Bellman Operators

* **Continuation to Decision (Backward)**:
  $$\mathscr{T}^{ev} \mathscr{E}(a, H, y) = \max \{ \mathscr{E}^{\text{rent}}((1+r)a + H, y), \mathscr{E}^{\text{own}}(a, y, H) \}$$

  Method: Discrete choice with branching to rent or own. The continuation solution will have two entries associated with the two incoming backward edges (RNTH and OWNH). The decision problem will stack these two and pick the optimal of the two over each point in the state space.

* **Decision to Arrival (Backward)**:
  $$\mathscr{T}^{va} \mathscr{V}(a, H, y^{\text{pre}}) = \mathbb{E}_{\xi}[\mathscr{V}(a, H, f(y^{\text{pre}}, \xi))]$$

  This is the point at which we do integration.

## Stage-to-Stage Connections

The following connections link the stages within and across time periods:

### Intra-Period Connections

| From → To | Mapping |
|-----------|---------|
| TENU (own) → OWNH | Identity on $(a, y, H)$ |
| TENU (rent) → RNTH | $(a, H, y) \mapsto ((1+r)a+H, y)$ |
| OWNH → OWNC | $(H^{\text{nxt}}, y, w^{\text{own}}) \mapsto (H^{\text{nxt}}, y, w^{\text{own}})$ |
| RNTH → RNTC | $(S, y, w^{\text{rent}}) \mapsto (S, y, w^{\text{rent}})$ |
| RNTC → OWNC | $(a^{\text{nxt}}, y, S) \mapsto (H^{\text{nxt}} = 0, y, w)$ |

### Inter-Period Connections

| From → To | Mapping |
|-----------|---------|
| **Inter-period** $t$ | OWNC / RNTC → TENU$_{t+1}$ via their continuation states |

Note that in the forward direction, TENU has two incoming edges and two outgoing edges.

The graph therefore covers all assets, housing, wage shocks and tenure transitions needed for a complete Heptapod-B YAML representation—no symbols are left undefined, and every Bellman operator references variables declared in its perch's state space.