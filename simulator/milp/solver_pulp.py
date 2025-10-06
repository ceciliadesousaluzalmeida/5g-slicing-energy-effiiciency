# milp/solver_pulp.py
# All comments in English

import pulp
from pulp import LpProblem, LpMinimize, LpBinary, LpContinuous, lpSum, LpStatus
from dataclasses import dataclass

def _incidence(n, e):
    """Incidence for undirected edge e=(u,v): +1 at u, -1 at v, 0 otherwise."""
    u, v = e
    if n == u: return 1.0
    if n == v: return -1.0
    return 0.0

@dataclass
class PulpSolveResult:
    status_code: int
    status_str: str
    objective: float
    values: dict  # maps tuple-keys -> float values


def solve_pulp(instance, msg=False, time_limit=None):
    """
    Solve the MILP with PuLP + CBC, following the same model:
      - Variables:
          v[i,n] ∈ {0,1}, u[n] ∈ [0,1], z[n] ∈ {0,1},
          f[e,s,(i,j)] ∈ {0,1}, rho[e] ∈ [0,1], w[e] ∈ {0,1}
      - Objective: min sum_n(u+z) + sum_e(rho+w)
      - Constraints:
          CPU, u lower bound, u <= z,
          assignment sum_n v=1,
          anti-colocation,
          flow conservation per VL,
          link capacity, rho definition, rho <= w,
          latency per VL.
    """
    N = list(instance.N)
    E = list(instance.E)
    S = list(instance.S)

    # --- Problem ---
    prob = LpProblem("SlicePlacementEnergy", LpMinimize)

    # --- Variables ---
    # v[i,n]
    v = {(i, n): pulp.LpVariable(f"v_{i}_{n}", lowBound=0, upBound=1, cat=LpBinary)
         for s in S for i in instance.V_of_s[s] for n in N}

    # u[n] continuous, z[n] binary
    u = {n: pulp.LpVariable(f"u_{n}", lowBound=0, upBound=1, cat=LpContinuous) for n in N}
    z = {n: pulp.LpVariable(f"z_{n}", lowBound=0, upBound=1, cat=LpBinary) for n in N}

    # f[e,s,(i,j)] binary
    f = {}
    for s in S:
        vnf_ids = instance.V_of_s[s]
        for q in range(len(vnf_ids) - 1):
            i, j = vnf_ids[q], vnf_ids[q + 1]
            for e in E:
                f[(e, s, (i, j))] = pulp.LpVariable(
                    f"f_{e[0]}_{e[1]}_s{s}_{i}_to_{j}", lowBound=0, upBound=1, cat=LpBinary
                )

    # rho[e] continuous, w[e] binary
    rho = {e: pulp.LpVariable(f"rho_{e[0]}_{e[1]}", lowBound=0, upBound=1, cat=LpContinuous) for e in E}
    w   = {e: pulp.LpVariable(f"w_{e[0]}_{e[1]}",   lowBound=0, upBound=1, cat=LpBinary)     for e in E}

    # --- Objective: min sum_n(u+z) + sum_e(rho+w) ---
    prob += lpSum(u[n] + z[n] for n in N) + lpSum(rho[e] + w[e] for e in E), "EnergyObjective"

    # --- Constraints ---

    # (1) CPU capacity: sum_i CPU_i * v[i,n] <= CPU_cap[n]
    for n in N:
        prob += lpSum(instance.CPU_i[i] * v[(i, n)]
                      for s in S for i in instance.V_of_s[s]) <= instance.CPU_cap[n], f"CPU_cap_{n}"

    # (1b) u[n] >= (sum_i CPU_i v[i,n]) / CPU_cap[n]
    for n in N:
        cap = max(1e-9, float(instance.CPU_cap[n]))
        prob += lpSum((instance.CPU_i[i] / cap) * v[(i, n)]
                      for s in S for i in instance.V_of_s[s]) - u[n] <= 0, f"u_lb_{n}"

    # (1c) u[n] <= z[n]
    for n in N:
        prob += u[n] - z[n] <= 0, f"u_leq_z_{n}"

    # (2) Assignment: sum_n v[i,n] = 1  for each VNF i
    for s in S:
        for i in instance.V_of_s[s]:
            prob += lpSum(v[(i, n)] for n in N) == 1, f"assign_{i}"

    # (3) Anti-colocation: sum_{i in V_s} v[i,n] <= 1  for each slice s and node n
    for s in S:
        for n in N:
            prob += lpSum(v[(i, n)] for i in instance.V_of_s[s]) <= 1, f"antico_s{s}_n{n}"

    # (4) Flow conservation per VL (i->j): sum_e inc(n,e) f[e,s,(i,j)] = v[i,n] - v[j,n]
    for s in S:
        vnf_ids = instance.V_of_s[s]
        for q in range(len(vnf_ids) - 1):
            i, j = vnf_ids[q], vnf_ids[q + 1]
            for n in N:
                prob += lpSum(_incidence(n, e) * f[(e, s, (i, j))] for e in E) - v[(i, n)] + v[(j, n)] == 0, \
                        f"flow_s{s}_{i}_to_{j}_n{n}"

    # (5) Link capacity and rho
    for e in E:
        # Capacity: sum_{s,(i,j)} BW_s * f[e,s,(i,j)] <= BW_cap[e]
        prob += lpSum(instance.BW_s[s] * f[(e, s, (i, j))]
                      for s in S
                      for (i, j) in [(instance.V_of_s[s][qq], instance.V_of_s[s][qq+1])
                                     for qq in range(len(instance.V_of_s[s]) - 1)]
                      ) <= instance.BW_cap[e], f"bwcap_{e}"

        # rho definition: sum BW_s f - BW_cap[e]*rho[e] <= 0
        prob += lpSum(instance.BW_s[s] * f[(e, s, (i, j))]
                      for s in S
                      for (i, j) in [(instance.V_of_s[s][qq], instance.V_of_s[s][qq+1])
                                     for qq in range(len(instance.V_of_s[s]) - 1)]
                      ) - instance.BW_cap[e] * rho[e] <= 0, f"rho_def_{e}"

        # Activation: rho[e] <= w[e]
        prob += rho[e] - w[e] <= 0, f"rho_leq_w_{e}"

    # (6) Latency per VL: sum_e lat(e) * f[e,s,(i,j)] <= L_s
    for s in S:
        vnf_ids = instance.V_of_s[s]
        for q in range(len(vnf_ids) - 1):
            i, j = vnf_ids[q], vnf_ids[q + 1]
            prob += lpSum(instance.lat_e[e] * f[(e, s, (i, j))] for e in E) <= instance.L_s[s], \
                    f"latency_s{s}_{i}_to_{j}"

    # --- Solve with CBC ---
    solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit) if time_limit else pulp.PULP_CBC_CMD(msg=msg)
    prob.solve(solver)

    # --- Collect solution ---
    values = {}
    # v
    for key, var in v.items(): values[("v",) + key] = var.value()
    # u, z
    for n in N: values[("u", n)] = u[n].value()
    for n in N: values[("z", n)] = z[n].value()
    # f
    for key, var in f.items(): values[("f",) + key] = var.value()
    # rho, w
    for e in E: values[("rho", e)] = rho[e].value()
    for e in E: values[("w", e)]   = w[e].value()

    print("Total CPU demand:", sum(instance.CPU_i.values()))
    print("Total CPU cap:", sum(instance.CPU_cap.values()))
    print("Total BW demand per slice:", instance.BW_s)
    print("Total BW cap:", {e: instance.BW_cap[e] for e in instance.E})
    print("Latency budgets:", instance.L_s)


    status_code = pulp.LpStatus[prob.status]
    res = PulpSolveResult(
        status_code=prob.status,
        status_str=status_code,
        objective=pulp.value(prob.objective),
        values=values
    )
    return res
