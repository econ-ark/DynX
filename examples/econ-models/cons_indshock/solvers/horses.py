"""
1D Endogenous Grid Method (EGM) model horse for lifecycle and similar models.

This module contains the core computational operators for solving consumption-savings
problems using the Endogenous Grid Method (EGM) developed by Christopher Carroll.

The module implements two key operator factories:
1. operator_factory_cntn_to_dcsn: Maps future value functions to current decision variables
2. operator_factory_dcsn_to_arvl: Integrates over uncertainty to produce expected values

These operators form the building blocks for solving dynamic programming problems
with cash-on-hand as the decision variable and assets as the endogenous state.
"""

import numpy as np
from scipy.interpolate import interp1d


def operator_factory_cntn_to_dcsn(mover):
    """Create an operator for the cntn_to_dcsn mover using the EGM method.

    Creates a transformation operator that maps continuation values to decision
    values using the Endogenous Grid Method (EGM) with compiled T-Operators.

    Parameters
    ----------
    mover : Mover
        The cntn_to_dcsn mover with self-contained numerical functions and grids

    Returns
    -------
    callable
        The T_cntn_to_dcsn operator that expects only perch data as input
    """
    # Extract the compiled T-operator functions from the mover
    compiled_funcs = mover.model.num["functions"]
    T_Carroll = compiled_funcs["T_Carroll"]  # m = T_C(lambda_cntn, a_nxt)
    T_Coleman = compiled_funcs["T_Coleman"]  # c = T_Co(lambda_cntn)
    T_Stachurski = compiled_funcs["T_Stachurski"]  # u_c = T_S(lambda_cntn)
    T_Bellman = compiled_funcs["T_Bellman"]  # v = T_B(lambda_cntn, vlu_cntn)
    q_inv_func = compiled_funcs["q_inv_func"]  # c = q_inv_func(lambda_cntn)
    g_e_to_v = compiled_funcs["g_e_to_v"]  # v = g_e_to_v(c, a_nxt)

    # Extract all necessary grids from the mover's numeric model
    m_grid = mover.model.num.state_space.dcsn.grids.m  # Decision grid (m)
    a_cntn = mover.model.num.state_space.cntn.grids.a_nxt  # Continuation grid (a_nxt)

    # Access key parameters
    parameters = mover.model.parameters_dict  # Access via mover.model


    def T_cntn_to_dcsn(perch_data):
        """Transform arrays from continuation perch to decision perch.

        Maps continuation value arrays to policy and value arrays using
        the Endogenous Grid Method (EGM).

        Parameters
        ----------
        perch_data : dict
            Dictionary of arrays from the continuation perch's sol attribute
            with keys:
            - 'vlu_cntn': Value function on continuation grid (np.ndarray)
            - 'lambda_cntn': Marginal value on continuation grid (np.ndarray)
            - 'a_nxt_grid': Continuation grid points (np.ndarray)

        Returns
        -------
        dict
            Results dictionary with policy, value, and marginal value arrays
            on the standard m_grid, with keys:
            - 'policy': Consumption policy on decision grid (np.ndarray)
            - 'lambda_dcsn': Marginal value on decision grid (np.ndarray)
            - 'vlu_dcsn': Value function on decision grid (np.ndarray)
            - 'm_grid': Decision grid points (np.ndarray)
        """
        # Extract perch data arrays and the grid they are defined on
        vlu_cntn = perch_data["vlu_cntn"]
        lambda_cntn = perch_data["lambda_cntn"]

        # Ensure input arrays are of the right type (numpy arrays)
        if not isinstance(vlu_cntn, np.ndarray):
            vlu_cntn = np.array(vlu_cntn)
        if not isinstance(lambda_cntn, np.ndarray):
            lambda_cntn = np.array(lambda_cntn)

        # --- EGM Step 1: Calculate results on the endogenous grid ---
        # Calculate optimal consumption using the inverse FOC (T_Coleman equivalent)
        #policy_egm = q_inv_func(lambda_cntn=lambda_cntn)
        policy_egm = T_Coleman(lambda_cntn=lambda_cntn)

        # Calculate the corresponding endogenous cash-on-hand grid using budget constraint (T_Carroll equivalent)
        #x_dcsn_egm = policy_egm + a_cntn
        x_dcsn_egm = T_Carroll(lambda_cntn=lambda_cntn, a_nxt=a_cntn)

        # Calculate the value function using the Bellman equation (T_Bellman)
        v_dcsn_egm = T_Bellman(lambda_cntn=lambda_cntn, vlu_cntn=vlu_cntn)

        # Calculate the marginal utility of consumption (T_Stachurski equivalent)
        lambda_dcsn_egm = T_Stachurski(lambda_cntn=lambda_cntn)

        # Convert results to numpy arrays if needed (safety check)
        if not isinstance(policy_egm, np.ndarray):
            policy_egm = np.array(policy_egm)
        if not isinstance(x_dcsn_egm, np.ndarray):
            x_dcsn_egm = np.array(x_dcsn_egm)
        if not isinstance(v_dcsn_egm, np.ndarray):
            v_dcsn_egm = np.array(v_dcsn_egm)
        if not isinstance(lambda_dcsn_egm, np.ndarray):
            lambda_dcsn_egm = np.array(lambda_dcsn_egm)

        # --- Filter invalid points from the endogenous grid results ---
        # Filter out non-finite values and points where savings are non-positive
        # Allowing zero savings (a_nxt = m - c >= 0 => m >= c)
        savings_egm = x_dcsn_egm - policy_egm
        valid_indices = (
            np.isfinite(x_dcsn_egm)
            & np.isfinite(policy_egm)
            & np.isfinite(v_dcsn_egm)
            & np.isfinite(lambda_dcsn_egm)
            & (savings_egm >= 0)  # Allow zero savings
        )

        x_dcsn_valid_egm = x_dcsn_egm[valid_indices]
        policy_valid_egm = policy_egm[valid_indices]
        v_dcsn_valid_egm = v_dcsn_egm[valid_indices]
        lambda_dcsn_valid_egm = lambda_dcsn_egm[valid_indices]

        # Sort the valid endogenous grid results by cash-on-hand (m)
        sort_indices = np.argsort(x_dcsn_valid_egm)
        m_sorted_egm = x_dcsn_valid_egm[sort_indices]
        policy_sorted_egm = policy_valid_egm[sort_indices]
        v_sorted_egm = v_dcsn_valid_egm[sort_indices]
        lambda_sorted_egm = lambda_dcsn_valid_egm[sort_indices]

        # --- EGM Step 2: Interpolate results onto the standard m_grid ---
        if len(m_sorted_egm) < 2:
            raise ValueError(
                "Not enough valid points on the endogenous grid to interpolate. Check model parameters or initial conditions."
            )

        # Create linear interpolators with extrapolation for points outside the endogenous grid range
        interp_policy_egm = interp1d(
            m_sorted_egm,
            policy_sorted_egm,
            bounds_error=False,
            fill_value="extrapolate",
            kind="linear",
        )
        interp_value_egm = interp1d(
            m_sorted_egm,
            v_sorted_egm,
            bounds_error=False,
            fill_value="extrapolate",
            kind="linear",
        )
        interp_marg_util_egm = interp1d(
            m_sorted_egm,
            lambda_sorted_egm,
            bounds_error=False,
            fill_value="extrapolate",
            kind="linear",
        )

        # Evaluate the interpolators on the standard decision grid (m_grid)
        policy_dcsn_array = interp_policy_egm(m_grid)
        vlu_dcsn_array = interp_value_egm(m_grid)
        lambda_dcsn_array = interp_marg_util_egm(m_grid)

        # --- Handle points below the minimum endogenous grid point ---
        # For cash-on-hand values lower than the minimum calculated on the
        # endogenous grid (m_sorted_egm[0]), linear extrapolation might lead
        # to invalid results (e.g., c > m). Apply reasonable constraints.
        min_valid_m = m_sorted_egm[0]
        mask = m_grid < min_valid_m

        # Constraint 1: Consumption cannot exceed cash-on-hand (c <= m)
        # Apply a simple rule: consume a large fraction (e.g., 99%) of m
        # This ensures c > 0 and c <= m for m > 0.
        policy_dcsn_array[mask] = np.maximum(
            m_grid[mask] * (0.97), 1e-100
        )  # Ensure positive consumption

        # Constraint 2: Recalculate marginal utility based on constrained consumption
        # Avoid division by zero if policy_dcsn_array is zero
        # Use np.maximum to ensure the base is positive before exponentiation
        lambda_dcsn_array[mask] = np.maximum(policy_dcsn_array[mask], 1e-12) ** (-parameters["gamma"])

        # Constraint 3: Assign a very low value for the value function in this region
        # Represents the poor outcome of having very low resources.
        vlu_dcsn_array[mask] = -1e200  # Assign a large negative number

        # Return the arrays evaluated and corrected on the standard m_grid
        return {
            "policy": policy_dcsn_array,
            "lambda_dcsn": lambda_dcsn_array,
            "vlu_dcsn": vlu_dcsn_array,
        }

    return T_cntn_to_dcsn


def operator_factory_dcsn_to_arvl(mover):
    """Create an operator for the dcsn_to_arvl mover that transforms functions
    from decision perch to arrival perch by integrating over shock distribution.

    Parameters
    ----------
    mover : Mover
        The dcsn_to_arvl mover with self-contained numerical functions and grids

    Returns
    -------
    callable
        The T_dcsn_to_arvl operator that expects only perch data as input
    """
    # Extract parameters from the mover's model
    parameters = mover.model.parameters_dict  # Access via mover.model
    beta = parameters["beta"]  # Assume parameters are present now
    r = parameters["r"]

    # Extract all necessary grids from the mover's numeric model
    a_grid = mover.model.num.state_space.arvl.grids.a
    m_grid = mover.model.num.state_space.dcsn.grids.m

    # Get the shock grid from the mover's numeric model
    shock_info = mover.model.num["shocks"]["income_shock"]
    shock_grid = shock_info['values']
    shock_probs = shock_info['probs']

    # Extract the g_a_to_v function from the mover's numeric model
    g_a_to_v = mover.model.num["functions"]["g_a_to_v"]

    # The operator now only takes perch_data as input
    def T_dcsn_to_arvl(perch_data):
        """
        Functional operator that transforms arrays from decision perch to arrival perch.
        Integrates decision perch value array over shock distribution to produce arrival perch value array.

        Parameters
        ----------
        perch_data : dict
            Dictionary of arrays from the decision perch's sol attribute
            {'vlu_dcsn': np.ndarray, 'lambda_dcsn': np.ndarray, 'm_grid': np.ndarray}

        Returns
        -------
        dict
            Results dictionary with arrival value array, marginal value array, conditional expectations and grid.
            {'vlu_arvl': np.ndarray, 'lambda_arvl': np.ndarray, 'cond_exp': dict, 'a_grid': np.ndarray}
        """
        # Extract perch data arrays and the grid they are defined on
        vlu_dcsn_array = perch_data["vlu_dcsn"]
        lambda_dcsn_array = perch_data["lambda_dcsn"]

        # Create interpolators for the decision functions
        interp_vlu_dcsn = interp1d(
            m_grid,
            vlu_dcsn_array,
            bounds_error=False,
            fill_value="extrapolate",
            kind="linear",
        )

        interp_lambda_dcsn = interp1d(
            m_grid,
            lambda_dcsn_array,
            bounds_error=False,
            fill_value="extrapolate",
            kind="linear",
        )

        # Vectorized implementation of the expected value calculation
        # Create a mesh grid of assets and shocks: A×Y grid of shape (len(a_grid), len(shock_grid))
        a_mesh, shock_mesh = np.meshgrid(a_grid, shock_grid, indexing="ij")
        prob_mesh = np.tile(
            shock_probs, (len(a_grid), 1)
        )  # Replicate probabilities to match grid

        # Calculate cash-on-hand for all asset-shock combinations at once
        m_mesh = g_a_to_v(a=a_mesh, y=shock_mesh)

        # Apply interpolators to all cash-on-hand values
        values_mesh = interp_vlu_dcsn(m_mesh)
        lambda_mesh = interp_lambda_dcsn(m_mesh)

        # replace negative lambda with 1e-200
        lambda_mesh[lambda_mesh < 0] = 1e-200

        # Store conditional expectations (value function conditioned on each shock)
        cond_exp = {"vlu_cond": {}, "lambda_cond": {}}

        # Compute and store conditional expectations for each shock
        for i, y in enumerate(shock_grid):
            cond_exp["vlu_cond"][float(y)] = values_mesh[:, i]
            cond_exp["lambda_cond"][float(y)] = lambda_mesh[:, i]

        # Multiply values/lambda by probabilities and sum across shock dimension (axis=1)
        vlu_arvl = np.sum(values_mesh * prob_mesh, axis=1)
        lambda_arvl = np.sum(lambda_mesh * prob_mesh, axis=1)

        # Return the arrival arrays evaluated on the standard a_grid
        return {
            "vlu_arvl": vlu_arvl,
            "lambda_arvl": lambda_arvl,
            "cond_exp": cond_exp
        }
    
    return T_dcsn_to_arvl
