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
    MILP with ENTRY/EXIT support.
    Variables:
      - v[i,n] ∈ {0,1}, u[n] ∈ [0,1], z[n] ∈ {0,1}
      - f[e,s,(i,j)] ∈ {0,1}  (VL flows between consecutive VNFs)
      - f_entry[e,s], f_exit[e,s] ∈ {0,1}  (ENTRY→first, last→EXIT)  [optional if entry/exit defined]
      - rho[e] ∈ [0,1], w[e] ∈ {0,1}
    Objective:
      - min ∑(u+z) + ∑(rho+w)
    Constraints:
      - CPU capacity & utilization, assignment, anti-colocation
      - Flow conservation for each VL
      - Link capacity (+ f_entry/f_exit if defined)
      - Latency per VL (+ entry/exit legs if defined)
    """
    N = list(instance.N)
    E = list(instance.E)
    S = list(instance.S)

    # --- Helpers to get ENTRY/EXIT per slice (supports global or per-slice spec) ---
    def get_entry(s):
        # Return node id or None if not defined
        if hasattr(instance, "entry_of_s"):
            return instance.entry_of_s.get(s, None)
        return getattr(instance, "entry_node", None)

    def get_exit(s):
        if hasattr(instance, "exit_of_s"):
            return instance.exit_of_s.get(s, None)
        return getattr(instance, "exit_node", None)

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

    # f[e,s,(i,j)] (between consecutive VNFs)
    f = {}
    for s in S:
        vnf_ids = instance.V_of_s[s]
        for q in range(len(vnf_ids) - 1):
            i, j = vnf_ids[q], vnf_ids[q + 1]
            for e in E:
                f[(e, s, (i, j))] = m.addVar(vtype=GRB.BINARY,
                                             name=f"f_{e[0]}_{e[1]}_s{s}_{i}_to_{j}")

    # New: f_entry[e,s], f_exit[e,s] if entry/exit exist
    f_entry, f_exit = {}, {}
    for s in S:
        s_entry, s_exit = get_entry(s), get_exit(s)
        if s_entry is not None:
            for e in E:
                f_entry[(e, s)] = m.addVar(vtype=GRB.BINARY, name=f"f_entry_{e[0]}_{e[1]}_s{s}")
        if s_exit is not None:
            for e in E:
                f_exit[(e, s)] = m.addVar(vtype=GRB.BINARY, name=f"f_exit_{e[0]}_{e[1]}_s{s}")

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

    # (1) CPU capacity
    for n in N:
        m.addConstr(
            gp.quicksum(instance.CPU_i[i] * v[(i, n)]
                        for s in S for i in instance.V_of_s[s])
            <= instance.CPU_cap[n],
            name=f"CPU_cap_{n}"
        )

    # (1b) u[n] lower bound
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

    # (2) Assignment: each VNF placed exactly once
    for s in S:
        for i in instance.V_of_s[s]:
            m.addConstr(
                gp.quicksum(v[(i, n)] for n in N) == 1,
                name=f"assign_{i}"
            )

    # (3) Anti-colocation: one VNF per node per slice
    for s in S:
        for n in N:
            m.addConstr(
                gp.quicksum(v[(i, n)] for i in instance.V_of_s[s]) <= 1,
                name=f"antico_s{s}_n{n}"
            )

    # (4) Flow conservation for VLs between consecutive VNFs
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

    # (4b) Flow conservation for ENTRY→first VNF (if ENTRY exists)
    for s in S:
        s_entry = get_entry(s)
        if s_entry is None:
            continue
        first_vnf = instance.V_of_s[s][0]
        for n in N:
            m.addConstr(
                gp.quicksum(_incidence(n, e) * f_entry[(e, s)] for e in E)
                == (1.0 if n == s_entry else 0.0) - v[(first_vnf, n)],
                name=f"flow_entry_s{s}_n{n}"
            )

    # (4c) Flow conservation for last VNF→EXIT (if EXIT exists)
    for s in S:
        s_exit = get_exit(s)
        if s_exit is None:
            continue
        last_vnf = instance.V_of_s[s][-1]
        for n in N:
            m.addConstr(
                gp.quicksum(_incidence(n, e) * f_exit[(e, s)] for e in E)
                == v[(last_vnf, n)] - (1.0 if n == s_exit else 0.0),
                name=f"flow_exit_s{s}_n{n}"
            )

    # (5) Link capacity and rho (include VLs + entry/exit legs)
    for e in E:
        # Total carried BW on edge e
        flow_sum_VLs = gp.quicksum(
            instance.BW_s[s] * f[(e, s, (i, j))]
            for s in S
            for (i, j) in [(instance.V_of_s[s][qq], instance.V_of_s[s][qq+1])
                           for qq in range(len(instance.V_of_s[s]) - 1)]
            if (e, s, (i, j)) in f  # safety (always true)
        )

        flow_sum_ENTRY = gp.quicksum(
            instance.BW_s[s] * f_entry[(e, s)]
            for s in S if (e, s) in f_entry
        )

        flow_sum_EXIT = gp.quicksum(
            instance.BW_s[s] * f_exit[(e, s)]
            for s in S if (e, s) in f_exit
        )

        total_edge_flow = flow_sum_VLs + flow_sum_ENTRY + flow_sum_EXIT

        # Capacity
        m.addConstr(total_edge_flow <= instance.BW_cap[e], name=f"bwcap_{e}")

        # rho definition and activation
        m.addConstr(total_edge_flow - instance.BW_cap[e] * rho[e] <= 0, name=f"rho_def_{e}")
        m.addConstr(rho[e] - w[e] <= 0, name=f"rho_leq_w_{e}")

    # (6) Latency per VL (sum of latencies along chosen edges)
    for s in S:
        vnf_ids = instance.V_of_s[s]
        for q in range(len(vnf_ids) - 1):
            i, j = vnf_ids[q], vnf_ids[q + 1]
            m.addConstr(
                gp.quicksum(instance.lat_e[e] * f[(e, s, (i, j))] for e in E)
                <= instance.L_s[s],
                name=f"latency_s{s}_{i}_to_{j}"
            )

    # (6b) Latency for ENTRY→first VNF (if ENTRY exists)
    for s in S:
        if any((e, s) in f_entry for e in E):
            m.addConstr(
                gp.quicksum(instance.lat_e[e] * f_entry[(e, s)] for e in E)
                <= instance.L_s[s],
                name=f"latency_entry_s{s}"
            )

    # (6c) Latency for last VNF→EXIT (if EXIT exists)
    for s in S:
        if any((e, s) in f_exit for e in E):
            m.addConstr(
                gp.quicksum(instance.lat_e[e] * f_exit[(e, s)] for e in E)
                <= instance.L_s[s],
                name=f"latency_exit_s{s}"
            )

    # --- Solve ---
    m.optimize()

    # --- Handle infeasible or unsolved models ---
    if m.status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
        print(f"[MILP] No feasible solution found or model not solved. Status: {m.status} ({m.Status})")
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
    # New: store f_entry / f_exit if they exist
    for key, var in f_entry.items():
        values[("f_entry",) + key] = var.X
    for key, var in f_exit.items():
        values[("f_exit",) + key] = var.X

    status_code = int(m.status)
    status_str = m.Status
    res = GurobiSolveResult(
        status_code=status_code,
        status_str=str(status_str),
        objective=m.objVal if m.status in [GRB.OPTIMAL, GRB.SUBOPTIMAL] else None,
        values=values
    )
    return res
