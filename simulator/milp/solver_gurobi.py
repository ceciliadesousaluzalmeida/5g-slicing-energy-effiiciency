from dataclasses import dataclass
import gurobipy as gp
from gurobipy import GRB

def _incidence(n, e):
    """Incidence for undirected edge e=(u,v): +1 at u, -1 at v, 0 otherwise."""
    u, v = e
    if n == u: return 1.0
    if n == v: return -1.0
    return 0.0

@dataclass
class GurobiSolveResult:
    status_code: int
    status_str: str
    objective: float
    values: dict  # maps tuple-keys -> float values


def solve_gurobi(instance, msg=False, time_limit=None):
    """
    Solve the MILP with Gurobi following the same model used in solver_pulp.py:
      - Variables:
          v[i,n] ∈ {0,1}, u[n] ∈ [0,1], z[n] ∈ {0,1},
          f[e,s,(i,j)] ∈ {0,1}, rho[e] ∈ [0,1], w[e] ∈ {0,1}
      - Objective: min ∑(u+z) + ∑(rho+w)
      - Constraints:
          CPU, u lower bound, u ≤ z,
          assignment, anti-colocation,
          flow conservation per VL,
          link capacity, rho definition, rho ≤ w,
          latency per VL.
    """
    N = list(instance.N)
    E = list(instance.E)
    S = list(instance.S)

    # --- Model ---
    m = gp.Model("SlicePlacementEnergy")
    m.Params.OutputFlag = 1 if msg else 0
    if time_limit:
        m.Params.TimeLimit = time_limit

    # --- Variables ---
    # v[i,n]
    v = {(i, n): m.addVar(vtype=GRB.BINARY, name=f"v_{i}_{n}")
         for s in S for i in instance.V_of_s[s] for n in N}

    # u[n], z[n]
    u = {n: m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"u_{n}") for n in N}
    z = {n: m.addVar(vtype=GRB.BINARY, name=f"z_{n}") for n in N}

    # f[e,s,(i,j)]
    f = {}
    for s in S:
        vnf_ids = instance.V_of_s[s]
        for q in range(len(vnf_ids) - 1):
            i, j = vnf_ids[q], vnf_ids[q + 1]
            for e in E:
                f[(e, s, (i, j))] = m.addVar(vtype=GRB.BINARY,
                                             name=f"f_{e[0]}_{e[1]}_s{s}_{i}_to_{j}")

    # rho[e], w[e]
    rho = {e: m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"rho_{e[0]}_{e[1]}") for e in E}
    w   = {e: m.addVar(vtype=GRB.BINARY, name=f"w_{e[0]}_{e[1]}") for e in E}

    # --- Objective ---
    m.setObjective(
        gp.quicksum(u[n] + z[n] for n in N) +
        gp.quicksum(rho[e] + w[e] for e in E),
        GRB.MINIMIZE
    )

    # --- Constraints ---

    # (1) CPU capacity: ∑ CPU_i * v[i,n] ≤ CPU_cap[n]
    for n in N:
        m.addConstr(
            gp.quicksum(instance.CPU_i[i] * v[(i, n)]
                        for s in S for i in instance.V_of_s[s])
            <= instance.CPU_cap[n],
            name=f"CPU_cap_{n}"
        )

    # (1b) u[n] ≥ (∑ CPU_i v[i,n]) / CPU_cap[n]
    for n in N:
        cap = max(1e-9, float(instance.CPU_cap[n]))
        m.addConstr(
            gp.quicksum((instance.CPU_i[i] / cap) * v[(i, n)]
                        for s in S for i in instance.V_of_s[s])
            - u[n] <= 0,
            name=f"u_lb_{n}"
        )

    # (1c) u[n] ≤ z[n]
    for n in N:
        m.addConstr(u[n] - z[n] <= 0, name=f"u_leq_z_{n}")

    # (2) Assignment: ∑_n v[i,n] = 1
    for s in S:
        for i in instance.V_of_s[s]:
            m.addConstr(
                gp.quicksum(v[(i, n)] for n in N) == 1,
                name=f"assign_{i}"
            )

    # (3) Anti-colocation: ∑_i v[i,n] ≤ 1
    for s in S:
        for n in N:
            m.addConstr(
                gp.quicksum(v[(i, n)] for i in instance.V_of_s[s]) <= 1,
                name=f"antico_s{s}_n{n}"
            )

    # (4) Flow conservation: ∑ inc(n,e) f[e,s,(i,j)] = v[i,n] - v[j,n]
    for s in S:
        vnf_ids = instance.V_of_s[s]
        for q in range(len(vnf_ids) - 1):
            i, j = vnf_ids[q], vnf_ids[q + 1]
            for n in N:
                m.addConstr(
                    gp.quicksum(_incidence(n, e) * f[(e, s, (i, j))] for e in E)
                    - v[(i, n)] + v[(j, n)] == 0,
                    name=f"flow_s{s}_{i}_to_{j}_n{n}"
                )

    # (5) Link capacity and rho
    for e in E:
        m.addConstr(
            gp.quicksum(instance.BW_s[s] * f[(e, s, (i, j))]
                        for s in S
                        for (i, j) in [(instance.V_of_s[s][qq], instance.V_of_s[s][qq+1])
                                       for qq in range(len(instance.V_of_s[s]) - 1)])
            <= instance.BW_cap[e],
            name=f"bwcap_{e}"
        )

        m.addConstr(
            gp.quicksum(instance.BW_s[s] * f[(e, s, (i, j))]
                        for s in S
                        for (i, j) in [(instance.V_of_s[s][qq], instance.V_of_s[s][qq+1])
                                       for qq in range(len(instance.V_of_s[s]) - 1)])
            - instance.BW_cap[e] * rho[e] <= 0,
            name=f"rho_def_{e}"
        )

        m.addConstr(rho[e] - w[e] <= 0, name=f"rho_leq_w_{e}")

    # (6) Latency per VL
    for s in S:
        vnf_ids = instance.V_of_s[s]
        for q in range(len(vnf_ids) - 1):
            i, j = vnf_ids[q], vnf_ids[q + 1]
            m.addConstr(
                gp.quicksum(instance.lat_e[e] * f[(e, s, (i, j))] for e in E)
                <= instance.L_s[s],
                name=f"latency_s{s}_{i}_to_{j}"
            )

       # --- Solve ---
    m.optimize()

    # --- Handle infeasible or unsolved models ---
    if m.status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
        print(f"[MILP] No feasible solution found or model not solved. Status: {m.status} ({m.Status})")
        # Optionally: export IIS for infeasible models
        if m.status == GRB.INFEASIBLE:
            try:
                m.computeIIS()
                m.write("model_infeasible.ilp")
                print("[MILP] IIS written to model_infeasible.ilp")
            except Exception as e:
                print(f"[MILP] Could not write IIS: {e}")
        return GurobiSolveResult(
            status_code=int(m.status),
            status_str=str(m.Status),
            objective=None,
            values={}
        )

    # --- Collect solution ---
    values = {}
    for key, var in v.items():
        values[("v",) + key] = var.X
    for n in N:
        values[("u", n)] = u[n].X
        values[("z", n)] = z[n].X
    for key, var in f.items():
        values[("f",) + key] = var.X
    for e in E:
        values[("rho", e)] = rho[e].X
        values[("w", e)] = w[e].X

    status_code = int(m.status)
    status_str = m.Status
    res = GurobiSolveResult(
        status_code=status_code,
        status_str=str(status_str),
        objective=m.objVal if m.status in [GRB.OPTIMAL, GRB.SUBOPTIMAL] else None,
        values=values
    )
    return res
