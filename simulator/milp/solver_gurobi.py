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
    MILP with ENTRY/EXIT support (optional).
    Required fields in `instance`:
      N (iterable nodes), E (iterable edges), S (iterable slices ids)
      V_of_s[s] -> ordered list of VNF ids
      CPU_i[i] -> cpu of VNF i
      CPU_cap[n] -> cpu capacity of node n
      BW_cap[e] -> bandwidth capacity of edge e
      lat_e[e] -> latency of edge e
      L_s[s] -> latency budget per VL of slice s
      BW_s[s] -> bandwidth demand per VL of slice s

    Optional ENTRY/EXIT (use qualquer uma das opções):
      - instance.entry_node / instance.exit_node (globais)
      - instance.entry_of_s[s] / instance.exit_of_s[s] (por slice)
    """
    N = list(instance.N)
    E = list(instance.E)
    S = list(instance.S)

    # --- Helpers for ENTRY/EXIT (NEW) ---
    def get_entry(s):
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

    # f[e,s,(i,j)] (entre VNFs consecutivos)
    f = {}
    for s in S:
        vnf_ids = instance.V_of_s[s]
        for q in range(len(vnf_ids) - 1):
            i, j = vnf_ids[q], vnf_ids[q + 1]
            for e in E:
                f[(e, s, (i, j))] = m.addVar(vtype=GRB.BINARY,
                                             name=f"f_{e[0]}_{e[1]}_s{s}_{i}_to_{j}")

    # (NEW) f_entry[e,s], f_exit[e,s] se ENTRY/EXIT existirem
    f_entry = {}
    f_exit  = {}
    for s in S:
        if get_entry(s) is not None:
            for e in E:
                f_entry[(e, s)] = m.addVar(vtype=GRB.BINARY, name=f"f_entry_{e[0]}_{e[1]}_s{s}")
        if get_exit(s) is not None:
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

    # (2) Assignment: cada VNF alocado exatamente uma vez
    for s in S:
        for i in instance.V_of_s[s]:
            m.addConstr(
                gp.quicksum(v[(i, n)] for n in N) == 1,
                name=f"assign_{i}"
            )

    # (3) Anti-colocation: no máx. 1 VNF do slice por nó
    for s in S:
        for n in N:
            m.addConstr(
                gp.quicksum(v[(i, n)] for i in instance.V_of_s[s]) <= 1,
                name=f"antico_s{s}_n{n}"
            )

    # (4) Flow conservation VLs (entre VNFs consecutivos)
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

    # (4b) Flow ENTRY→first VNF (se existir ENTRY)
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

    # (4c) Flow last VNF→EXIT (se existir EXIT)
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

    # (5) Link capacity + rho (inclui VLs + ENTRY/EXIT se existirem)
    for e in E:
        flow_sum_VLs = gp.quicksum(
            instance.BW_s[s] * f[(e, s, (i, j))]
            for s in S
            for (i, j) in [(instance.V_of_s[s][qq], instance.V_of_s[s][qq+1])
                           for qq in range(len(instance.V_of_s[s]) - 1)]
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

        m.addConstr(total_edge_flow <= instance.BW_cap[e], name=f"bwcap_{e}")
        m.addConstr(total_edge_flow - instance.BW_cap[e] * rho[e] <= 0, name=f"rho_def_{e}")
        m.addConstr(rho[e] - w[e] <= 0, name=f"rho_leq_w_{e}")

    # (6) Latência por VL
    for s in S:
        vnf_ids = instance.V_of_s[s]
        for q in range(len(vnf_ids) - 1):
            i, j = vnf_ids[q], vnf_ids[q + 1]
            m.addConstr(
                gp.quicksum(instance.lat_e[e] * f[(e, s, (i, j))] for e in E)
                <= instance.L_s[s],
                name=f"latency_s{s}_{i}_to_{j}"
            )

    # (6b) Latência ENTRY→first (se existir ENTRY)
    for s in S:
        if any((e, s) in f_entry for e in E):
            m.addConstr(
                gp.quicksum(instance.lat_e[e] * f_entry[(e, s)] for e in E)
                <= instance.L_s[s],
                name=f"latency_entry_s{s}"
            )

    # (6c) Latência last→EXIT (se existir EXIT)
    for s in S:
        if any((e, s) in f_exit for e in E):
            m.addConstr(
                gp.quicksum(instance.lat_e[e] * f_exit[(e, s)] for e in E)
                <= instance.L_s[s],
                name=f"latency_exit_s{s}"
            )

    # --- Solve ---
    m.optimize()

    # --- Handle infeasible/unsolved ---
    if m.status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
        print(f"[MILP] No feasible solution or unsolved. Status: {m.status} ({m.Status})")
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
    # (NEW) store f_entry/f_exit if exist
    for key, var in f_entry.items():
        values[("f_entry",) + key] = var.X
    for key, var in f_exit.items():
        values[("f_exit",) + key] = var.X

    return GurobiSolveResult(
        status_code=int(m.status),
        status_str=str(m.Status),
        objective=m.objVal,
        values=values
    )
