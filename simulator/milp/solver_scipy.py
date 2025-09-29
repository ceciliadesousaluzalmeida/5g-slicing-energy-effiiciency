# solver_scipy.py
# All comments in English

import numpy as np
from scipy.optimize import linprog

def build_variable_index(instance):
    """
    Builds a dictionary mapping each decision variable to a position in the x vector.
    """
    var_index = {}
    idx = 0
    # v_{i,n}
    for s in instance.S:
        for i in instance.V_of_s[s]:
            for n in instance.N:
                var_index[("v", i, n)] = idx; idx += 1
    # l_{e}^{(s)}
    for s in instance.S:
        for e in instance.E:
            var_index[("l", e, s)] = idx; idx += 1
    # u_n, z_n
    for n in instance.N:
        var_index[("u", n)] = idx; idx += 1
        var_index[("z", n)] = idx; idx += 1
    # rho_e, w_e
    for e in instance.E:
        var_index[("rho", e)] = idx; idx += 1
        var_index[("w", e)] = idx; idx += 1

    return var_index, idx


def solve_lp(instance):
    """
    Solves the LP relaxation of the MILP with scipy.optimize.linprog
    """
    var_index, n_vars = build_variable_index(instance)

    # Objective vector c
    c = np.zeros(n_vars)
    for n in instance.N:
        c[var_index[("u", n)]] = 1
        c[var_index[("z", n)]] = 1
    for e in instance.E:
        c[var_index[("rho", e)]] = 1
        c[var_index[("w", e)]] = 1

    # Constraints
    A_ub, b_ub, A_eq, b_eq = [], [], [], []

    # -------------------------------
    # Node CPU capacity
    # -------------------------------
    for n in instance.N:
        row = np.zeros(n_vars)
        for s in instance.S:
            for i in instance.V_of_s[s]:
                row[var_index[("v", i, n)]] = instance.CPU_i[i]
        A_ub.append(row)
        b_ub.append(instance.CPUcap[n])

    # Definition of u_n
    for n in instance.N:
        row = np.zeros(n_vars)
        for s in instance.S:
            for i in instance.V_of_s[s]:
                row[var_index[("v", i, n)]] = instance.CPU_i[i] / instance.CPUcap[n]
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
        b_ub.append(instance.BWcap[e])

    # Definition of rho_e
    for e in instance.E:
        row = np.zeros(n_vars)
        for s in instance.S:
            row[var_index[("l", e, s)]] = instance.BW_s[s] / instance.BWcap[e]
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
            row[var_index[("l", e, s)]] = instance.Lat[e]
        A_ub.append(row)
        b_ub.append(instance.Ls[s])

    # -------------------------------
    # Solve LP relaxation
    # -------------------------------
    A_ub = np.array(A_ub) if A_ub else None
    b_ub = np.array(b_ub) if b_ub else None
    A_eq = np.array(A_eq) if A_eq else None
    b_eq = np.array(b_eq) if b_eq else None

    res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                  A_eq=A_eq, b_eq=b_eq,
                  bounds=(0, 1), method="highs")

    return res, var_index
