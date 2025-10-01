# solver_scipy.py
# All comments in English

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def build_variable_index(instance):
    """
    Builds a dictionary mapping each decision variable to a position in the x vector.
    Also prepares integrality information for MILP.
    """
    var_index = {}
    idx = 0
    integrality = []

    # v_{i,n} ∈ {0,1}
    for s in instance.S:
        for i in instance.V_of_s[s]:
            for n in instance.N:
                var_index[("v", i, n)] = idx
                integrality.append(2)  # binary
                idx += 1

    # l_{e}^{(s)} ∈ {0,1}
    for s in instance.S:
        for e in instance.E:
            var_index[("l", e, s)] = idx
            integrality.append(2)  # binary
            idx += 1

    # u_n ∈ [0,1] continuous, z_n ∈ {0,1}
    for n in instance.N:
        var_index[("u", n)] = idx
        integrality.append(0)  # continuous
        idx += 1

        var_index[("z", n)] = idx
        integrality.append(2)  # binary
        idx += 1

    # rho_e ∈ [0,1] continuous, w_e ∈ {0,1}
    for e in instance.E:
        var_index[("rho", e)] = idx
        integrality.append(0)  # continuous
        idx += 1

        var_index[("w", e)] = idx
        integrality.append(2)  # binary
        idx += 1

    return var_index, idx, np.array(integrality)


def solve_lp(instance):
    """
    Solves the MILP using scipy.optimize.milp (HiGHS backend).
    """
    var_index, n_vars, integrality = build_variable_index(instance)

    # Objective vector c
    c = np.zeros(n_vars)
    for n in instance.N:
        c[var_index[("u", n)]] = 1
        c[var_index[("z", n)]] = 1
    for e in instance.E:
        c[var_index[("rho", e)]] = 1
        c[var_index[("w", e)]] = 1

    # Constraints
    A_ub, b_ub = [], []
    A_eq, b_eq = [], []

    # -------------------------------
    # Node CPU capacity
    # -------------------------------
    for n in instance.N:
        row = np.zeros(n_vars)
        for s in instance.S:
            for i in instance.V_of_s[s]:
                row[var_index[("v", i, n)]] = instance.CPU_i[i]
        A_ub.append(row)
        b_ub.append(instance.CPU_cap[n])

    # Definition of u_n
    for n in instance.N:
        row = np.zeros(n_vars)
        for s in instance.S:
            for i in instance.V_of_s[s]:
                row[var_index[("v", i, n)]] = instance.CPU_i[i] / instance.CPU_cap[n]
        row[var_index[("u", n)]] = -1
        A_eq.append(row)
        b_eq.append(0.0)

    # Activation of node: u_n <= z_n
    for n in instance.N:
        row = np.zeros(n_vars)
        row[var_index[("u", n)]] = 1
        row[var_index[("z", n)]] = -1
        A_ub.append(row)
        b_ub.append(0.0)

    # -------------------------------
    # Link bandwidth capacity
    # -------------------------------
    for e in instance.E:
        row = np.zeros(n_vars)
        for s in instance.S:
            row[var_index[("l", e, s)]] = instance.BW_s[s]
        A_ub.append(row)
        b_ub.append(instance.BW_cap[e])

    # Definition of rho_e
    for e in instance.E:
        row = np.zeros(n_vars)
        for s in instance.S:
            row[var_index[("l", e, s)]] = instance.BW_s[s] / instance.BW_cap[e]
        row[var_index[("rho", e)]] = -1
        A_eq.append(row)
        b_eq.append(0.0)

    # Activation of link: rho_e <= w_e
    for e in instance.E:
        row = np.zeros(n_vars)
        row[var_index[("rho", e)]] = 1
        row[var_index[("w", e)]] = -1
        A_ub.append(row)
        b_ub.append(0.0)

    # -------------------------------
    # VNF assignment: sum_n v_{i,n} = 1
    # -------------------------------
    for s in instance.S:
        for i in instance.V_of_s[s]:
            row = np.zeros(n_vars)
            for n in instance.N:
                row[var_index[("v", i, n)]] = 1
            A_eq.append(row)
            b_eq.append(1.0)

    # -------------------------------
    # Anti-colocation: sum_{i in V_s} v_{i,n} <= 1
    # -------------------------------
    for s in instance.S:
        for n in instance.N:
            row = np.zeros(n_vars)
            for i in instance.V_of_s[s]:
                row[var_index[("v", i, n)]] = 1
            A_ub.append(row)
            b_ub.append(1.0)

    # -------------------------------
    # Latency: sum_e lat(e) * l_{e}^{(s)} <= L_s
    # -------------------------------
    for s in instance.S:
        row = np.zeros(n_vars)
        for e in instance.E:
            row[var_index[("l", e, s)]] = instance.lat_e[e]
        A_ub.append(row)
        b_ub.append(instance.L_s[s])

    # -------------------------------
    # Solve MILP
    # -------------------------------
    constraints = []
    if A_ub:
        constraints.append(LinearConstraint(np.array(A_ub), -np.inf, np.array(b_ub)))
    if A_eq:
        constraints.append(LinearConstraint(np.array(A_eq), np.array(b_eq), np.array(b_eq)))

    bounds = Bounds(0, 1)

    res = milp(c=c,
               constraints=constraints,
               integrality=integrality,
               bounds=bounds)

    return res, var_index
