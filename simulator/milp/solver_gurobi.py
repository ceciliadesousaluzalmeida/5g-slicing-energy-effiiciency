from dataclasses import dataclass
import gurobipy as gp
from gurobipy import GRB
import networkx as nx

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


def solve_gurobi(instance,
                 msg=False,
                 time_limit=None,
                 DEG_LIMIT=3,
                 MIP_GAP=0.05,
                 # --- Optional A* integration ---
                 astar_node_cands=None,       # dict: i -> list of allowed nodes
                 astar_edge_cands=None,       # dict: (s,i,j) -> list of allowed edges (u,v)
                 astar_entry_edge_cands=None, # dict: s -> list of allowed edges for ENTRY
                 warm_start=False):
    """
    MILP model for energy-aware slice placement, supporting optional A* pre-filtering.
    If A* candidates are given, restricts variables and flows to those subsets.

    Expected fields in `instance`:
      - N, E, S
      - V_of_s[s]: ordered VNFs per slice s
      - CPU_i[i], CPU_cap[n]
      - BW_cap[e], lat_e[e]
      - BW_sij[(s,i,j)], L_sij[(s,i,j)]
      - (optional) BW_entry[s], L_entry[s], entry_of_s[s] or entry_node
    """

    # -------------------------
    # Helpers ENTRY/EXIT
    # -------------------------
    def get_entry(s):
        if hasattr(instance, "entry_of_s"):
            return instance.entry_of_s.get(s, None)
        return getattr(instance, "entry_node", None)

    # -------------------------
    # Basic sets
    # -------------------------
    N = list(instance.N)
    E_full = list(instance.E)
    S = list(instance.S)

    # -------------------------
    # Graph pruning (low-latency edges)
    # -------------------------
    G = nx.Graph()
    for (u, v) in E_full:
        lat = float(instance.lat_e.get((u, v), instance.lat_e.get((v, u), 1.0)))
        bwc = float(instance.BW_cap.get((u, v), instance.BW_cap.get((v, u), 1.0)))
        G.add_edge(u, v, latency=lat, bandwidth=bwc)

    E_use = set()
    for n in N:
        nbrs = [(G[n][nb]["latency"], (min(n, nb), max(n, nb))) for nb in G.neighbors(n)]
        nbrs.sort(key=lambda x: x[0])
        for _, e in nbrs[:DEG_LIMIT]:
            E_use.add(e)
    E_use = list(E_use)

    if msg:
        print(f"[MILP] Edge pruning: |E_full|={len(E_full)} → |E_use|={len(E_use)}")

    # -------------------------
    # Candidate filtering from A*
    # -------------------------
    # Node candidates per VNF
    N_i = {}
    all_vnfs = [i for s in S for i in instance.V_of_s[s]]
    for i in all_vnfs:
        if astar_node_cands and i in astar_node_cands:
            cand = [n for n in astar_node_cands[i] if n in N]
            N_i[i] = cand if cand else list(N)
        else:
            N_i[i] = list(N)

    # Edge candidates per (s,i,j)
    E_sij = {}
    for s in S:
        vnf_ids = instance.V_of_s[s]
        for q in range(len(vnf_ids)-1):
            i, j = vnf_ids[q], vnf_ids[q+1]
            if astar_edge_cands and (s,i,j) in astar_edge_cands:
                cand = set((min(u,v), max(u,v)) for (u,v) in astar_edge_cands[(s,i,j)])
                E_sij[(s,i,j)] = [e for e in E_use if e in cand]
            else:
                E_sij[(s,i,j)] = list(E_use)

    # Edge candidates for ENTRY
    E_entry = {}
    for s in S:
        if astar_entry_edge_cands and s in astar_entry_edge_cands:
            cand = set((min(u,v), max(u,v)) for (u,v) in astar_entry_edge_cands[s])
            E_entry[s] = [e for e in E_use if e in cand]
        else:
            E_entry[s] = list(E_use)

    # -------------------------
    # Model
    # -------------------------
    m = gp.Model("SlicePlacementEnergy")
    m.Params.OutputFlag = 1 if msg else 0
    if time_limit:
        m.Params.TimeLimit = time_limit
    m.Params.MIPFocus = 1
    m.Params.Heuristics = 0.5
    m.Params.Presolve = 2
    m.Params.Cuts = 3
    m.Params.NumericFocus = 1
    m.Params.IntFeasTol = 1e-5
    m.Params.FeasibilityTol = 1e-6
    m.Params.OptimalityTol = 1e-6
    m.Params.MIPGap = MIP_GAP

    # -------------------------
    # Variables
    # -------------------------
    v = {(i,n): m.addVar(vtype=GRB.BINARY, name=f"v_{i}_{n}")
         for s in S for i in instance.V_of_s[s] for n in N_i[i]}

    u = {n: m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"u_{n}") for n in N}
    z = {n: m.addVar(vtype=GRB.BINARY, name=f"z_{n}") for n in N}

    f = {}
    for s in S:
        vnf_ids = instance.V_of_s[s]
        for q in range(len(vnf_ids)-1):
            i, j = vnf_ids[q], vnf_ids[q+1]
            for e in E_sij[(s,i,j)]:
                f[(e,s,(i,j))] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1,
                                          name=f"f_{e[0]}_{e[1]}_s{s}_{i}_to_{j}")

    f_entry = {}
    for s in S:
        s_entry = get_entry(s)
        if s_entry is not None:
            for e in E_entry[s]:
                f_entry[(e,s)] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1,
                                          name=f"f_entry_{e[0]}_{e[1]}_s{s}")

    rho = {e: m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"rho_{e[0]}_{e[1]}") for e in E_use}
    w   = {e: m.addVar(vtype=GRB.BINARY, name=f"w_{e[0]}_{e[1]}") for e in E_use}

    # -------------------------
    # Warm-start (optional)
    # -------------------------
    if warm_start and hasattr(instance, "best_astar"):
        best_astar = instance.best_astar
        for s in S:
            if s not in best_astar: 
                continue
            place = best_astar[s].get("place", {})
            paths = best_astar[s].get("paths", {})
            entry = best_astar[s].get("entry", [])
            for i,n_best in place.items():
                for n in N_i[i]:
                    if (i,n) in v:
                        v[(i,n)].Start = 1.0 if n==n_best else 0.0
            for (i,j), edges in paths.items():
                norm_edges = set((min(u,v),max(u,v)) for (u,v) in edges)
                for e in E_use:
                    if (e,s,(i,j)) in f:
                        f[(e,s,(i,j))].Start = 1.0 if e in norm_edges else 0.0
            if entry:
                norm_entry = set((min(u,v),max(u,v)) for (u,v) in entry)
                for e in E_use:
                    if (e,s) in f_entry:
                        f_entry[(e,s)].Start = 1.0 if e in norm_entry else 0.0

    # -------------------------
    # Objective
    # -------------------------
    m.setObjective(
        gp.quicksum(u[n] + z[n] for n in N) +
        gp.quicksum(rho[e] + w[e] for e in E_use),
        GRB.MINIMIZE
    )

    # -------------------------
    # Constraints
    # -------------------------
    # (1) CPU capacity
    for n in N:
        m.addConstr(
            gp.quicksum(instance.CPU_i[i] * v[(i,n)]
                        for s in S for i in instance.V_of_s[s] if (i,n) in v)
            <= instance.CPU_cap[n], name=f"CPU_cap_{n}"
        )

    # (1b) u lower bound and link with z
    for n in N:
        cap = max(1e-9, float(instance.CPU_cap[n]))
        m.addConstr(
            gp.quicksum((instance.CPU_i[i]/cap)*v[(i,n)]
                        for s in S for i in instance.V_of_s[s] if (i,n) in v)
            - u[n] <= 0, name=f"u_lb_{n}")
        m.addConstr(u[n] - z[n] <= 0, name=f"u_leq_z_{n}")

    # (2) Unique assignment per VNF
    for s in S:
        for i in instance.V_of_s[s]:
            m.addConstr(
                gp.quicksum(v[(i,n)] for n in N_i[i]) == 1,
                name=f"assign_{i}"
            )

    # (3) Anti-colocation per slice
    for s in S:
        for n in N:
            m.addConstr(
                gp.quicksum(v[(i,n)] for i in instance.V_of_s[s] if (i,n) in v) <= 1,
                name=f"antico_s{s}_n{n}"
            )

    # (4) Flow conservation
    for s in S:
        vnf_ids = instance.V_of_s[s]
        for q in range(len(vnf_ids)-1):
            i, j = vnf_ids[q], vnf_ids[q+1]
            for n in N:
                m.addConstr(
                    gp.quicksum(_incidence(n,e)*f[(e,s,(i,j))] for e in E_use if (e,s,(i,j)) in f)
                    - v.get((i,n),0) + v.get((j,n),0) == 0,
                    name=f"flow_s{s}_{i}_to_{j}_n{n}"
                )

    # (4b) Flow ENTRY → first VNF
    for s in S:
        s_entry = get_entry(s)
        if s_entry is None:
            continue
        first_vnf = instance.V_of_s[s][0]
        for n in N:
            m.addConstr(
                gp.quicksum(_incidence(n,e)*f_entry[(e,s)] for e in E_use if (e,s) in f_entry)
                == (1.0 if n==s_entry else 0.0) - v.get((first_vnf,n),0),
                name=f"flow_entry_s{s}_n{n}"
            )

    # (5) Link capacity and activation
    for e in E_use:
        flow_sum_VLs = gp.quicksum(
            instance.BW_sij[(s,i,j)]*f[(e,s,(i,j))]
            for s in S
            for (i,j) in [(instance.V_of_s[s][k],instance.V_of_s[s][k+1])
                          for k in range(len(instance.V_of_s[s])-1)]
            if (s,i,j) in instance.BW_sij and (e,s,(i,j)) in f
        )
        flow_sum_ENTRY = gp.quicksum(
            instance.BW_entry.get(s,0.0)*f_entry[(e,s)]
            for s in S if (e,s) in f_entry
        )
        total_flow = flow_sum_VLs + flow_sum_ENTRY
        cap_e = float(instance.BW_cap.get(e, instance.BW_cap.get((e[1],e[0]),1e9)))
        m.addConstr(total_flow <= cap_e, name=f"bwcap_{e}")
        m.addConstr(total_flow - cap_e*rho[e] <= 0, name=f"rho_def_{e}")
        m.addConstr(rho[e] - w[e] <= 0, name=f"rho_leq_w_{e}")

    # (6) Latency constraints
    for s in S:
        vnf_ids = instance.V_of_s[s]
        for q in range(len(vnf_ids)-1):
            i,j = vnf_ids[q],vnf_ids[q+1]
            lat_budget = instance.L_sij.get((s,i,j),1e9)
            m.addConstr(
                gp.quicksum(
                    float(instance.lat_e.get(e,instance.lat_e.get((e[1],e[0]),1.0)))*f[(e,s,(i,j))]
                    for e in E_use if (e,s,(i,j)) in f
                ) <= lat_budget,
                name=f"latency_s{s}_{i}_to_{j}"
            )

    for s in S:
        if any((e,s) in f_entry for e in E_use):
            lat_budget = instance.L_entry.get(s,1e9)
            m.addConstr(
                gp.quicksum(
                    float(instance.lat_e.get(e,instance.lat_e.get((e[1],e[0]),1.0)))*f_entry[(e,s)]
                    for e in E_use if (e,s) in f_entry
                ) <= lat_budget,
                name=f"latency_entry_s{s}"
            )

    # -------------------------
    # Solve
    # -------------------------
    m.optimize()

    if m.status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
        print(f"[MILP] No feasible solution or unsolved. Status={m.status}")
        if m.status == GRB.INFEASIBLE:
            try:
                m.computeIIS(); m.write("model_infeasible.ilp")
                print("[MILP] IIS written to model_infeasible.ilp")
            except Exception as e:
                print(f"[MILP] Could not write IIS: {e}")
        return GurobiSolveResult(m.status, str(m.Status), None, {})

    # -------------------------
    # Collect solution
    # -------------------------
    values = {}
    for key,var in v.items(): values[("v",)+key] = var.X
    for n in N: values[("u",n)] = u[n].X; values[("z",n)] = z[n].X
    for key,var in f.items(): values[("f",)+key] = var.X
    for key,var in f_entry.items(): values[("f_entry",)+key] = var.X
    for e in E_use: values[("rho",e)] = rho[e].X; values[("w",e)] = w[e].X

    return GurobiSolveResult(m.status, str(m.Status), m.objVal, values)
