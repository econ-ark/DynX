# 7. Examples and Case Studies

This section showcases practical applications of the ModCraft framework through example models that demonstrate its flexibility and power in solving various economic problems.

## 7.1 Consumption-Savings Model

The consumption-savings model is a classic example of dynamic programming in economics and serves as a good introductory example for the ModCraft framework.

**Key Features:**
- Single agent making consumption-savings decisions over time
- Stochastic income process with persistent shocks
- Borrowing constraints and uncertain future income
- Three-move structure with clear state transitions

**Implementation:**
- Arrival state: Asset holdings at beginning of period
- Decision state: Assets plus realized income
- Continuation state: Assets after consumption decision
- Key methods: Backward induction with value function iteration
- Illustrates grid interpolation for continuous state spaces

## 7.2 Lifecycle Model with Pensions

This more advanced model extends the basic consumption-savings framework to include retirement decisions and pension systems.

**Key Features:**
- Multi-period lifecycle with working and retirement phases
- Endogenous labor supply and retirement timing
- Pension benefits based on work history
- Health shocks affecting labor productivity

**Implementation:**
- Multiple interconnected stages for working and retirement phases
- State variables include assets, age, productivity, and pension eligibility
- Demonstrates how periods can connect multiple stages
- Shows implementation of discrete choice (retirement decision)

## 7.3 Branching Structures and Discrete Choice

This example illustrates how to implement models with discrete choices that lead to different future paths (branching structures).

**Key Features:**
- Modeling career choices with different education paths
- Housing purchase decisions with mortgage options
- Path dependency where choices affect future constraints

**Implementation:**
- Uses the branching stage pattern with multiple continuation perches
- Demonstrates how to implement dynamic discrete choice models
- Shows calculation of choice-specific value functions
- Illustrates the connection between different stages in a period

## 7.4 Heterogeneous Agent Macroeconomic Model

This advanced example demonstrates how ModCraft can be used to build heterogeneous agent macroeconomic models with aggregate dynamics.

**Key Features:**
- Population of heterogeneous agents with idiosyncratic shocks
- Aggregation of individual decisions to market-level outcomes
- General equilibrium with endogenous prices
- Transition dynamics between steady states

**Implementation:**
- Uses distribution functions in perches to track population distribution
- Implements Krusell-Smith style approximate aggregation
- Shows how to build multi-level models with micro and macro components
- Demonstrates simulation approaches for heterogeneous populations

## 7.5 Model Extensions and Combinations

ModCraft's modular design enables researchers to combine and extend existing models in novel ways:

**Example Extensions:**
- Adding health dynamics to retirement models
- Introducing housing markets to lifecycle models
- Incorporating family structure and intergenerational transfers
- Modeling entrepreneurship decisions with risky business investments

**Implementation Approach:**
- Start with a basic model template
- Add new state variables to relevant perches
- Extend transition functions to accommodate new dynamics
- Connect additional stages for new decision processes
- Reuse components from existing models through the modular architecture

## 7.6 Practical Implementation Tips

When implementing your own models in ModCraft, consider the following best practices:

1. **Start with mathematical formulation** - Clearly define states, actions, transitions, and objective functions
2. **Design your stage structure** - Identify the natural three-move structure in your problem
3. **Configure grid spaces carefully** - Balance computational efficiency with accuracy
4. **Use vectorized operations** - Optimize computational performance with numpy/JAX operations
5. **Validate with simple cases** - Start with known simplified versions before adding complexity
6. **Leverage existing templates** - Build on pre-configured models for common problem types
7. **Test transition functions** - Verify that state transitions work as expected
8. **Monitor convergence** - Check for proper convergence of value functions
9. **Visualize results** - Use built-in plotting tools to verify model behavior 