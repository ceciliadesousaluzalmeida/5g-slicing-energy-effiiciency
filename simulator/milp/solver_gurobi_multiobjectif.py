# All comments in English
from dataclasses import dataclass
import gurobipy as gp
from gurobipy import GRB
import networkx as nx


def _incidence(n, e):
    """Incidence for undirected edge e=(u,v): +1 at u, -1 at v, 0 otherwise."""
    u, v = e
    if n == u:
        return 1.0
    if n == v:
        return -1.0
    return 0.0


@dataclass
class GurobiSolveResult:
    status_code: int
    status_str: str
    objective: float
    values: dict  # maps tuple-keys -> float values


def solve_gurobi_multiobj(
    instance,
    msg=False,
    time_limit=None,
    DEG_LIMIT=100,
    MIP_GAP=0.05,
    # --- Multi-objective control ---
    use_multiobj=True,
    energy_priority=2,
    latency_priority=1,
    # --- Soft latency control ---
    use_soft_latency=True,
    latency_slack_penalty=1000.0,
    # --- Anti-colocation control ---
    max_vnfs_per_slice_per_node=1,
    # --- Optional A* integration ---
    astar_node_cands=None,
    astar_edge_cands=None,
    astar_entry_edge_cands=None,
    warm_start=False,
):
    """
    Multi-objective MILP model for energy-aware slice placement, with optional
    A* pre-filtering and Gurobi multi-objective support.
    """

    # -------------------------
    # Helpers ENTRY/EXIT
    # -------------------------
    def get_entry(s):
        """Return the entry node for slice s, if defined in the instance."""
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
        # Sort neighbors by latency and keep only DEG_LIMIT best edges
        nbrs = [(G[n][nb]["latency"], (min(n, nb), max(n, nb))) for nb in G.neighbors(n)]
        nbrs.sort(key=lambda x: x[0])
        for _, e in nbrs[:DEG_LIMIT]:
            E_use.add(e)
    E_use = list(E_use)

    # -------------------------
    # Optional: avoid routing through nodes with zero CPU
    # -------------------------
    # This is useful when instance.CPU_cap encodes *remaining* capacity
    # (e.g., in the sequential MILP wrapper). For the global call, if CPU_cap
    # is the original capacity, then zero_cpu_nodes will be empty.
    zero_cpu_nodes = {n for n in N if instance.CPU_cap.get(n, 0.0) <= 0.0}
    if zero_cpu_nodes and msg:
        print(f"[MILP-MULTI] Nodes with zero CPU (removed from routing): {sorted(zero_cpu_nodes)}")

    E_use = [
        e for e in E_use
        if e[0] not in zero_cpu_nodes and e[1] not in zero_cpu_nodes
    ]

    if msg:
        print(f"[MILP-MULTI] Edge pruning: |E_full|={len(E_full)} → |E_use|={len(E_use)}")

    # -------------------------
    # Candidate filtering from A*
    # -------------------------

    # Node candidates per VNF
    N_i = {}
    all_vnfs = [i for s in S for i in instance.V_of_s[s]]
    for i in all_vnfs:
        if astar_node_cands and i in astar_node_cands:
            # Restrict to nodes that are in N and in A* candidates
            cand = [n for n in astar_node_cands[i] if n in N]
            N_i[i] = cand if cand else list(N)
        else:
            # If no A* hint, allow all nodes
            N_i[i] = list(N)

    # Edge candidates per (s,i,j)
    E_sij = {}
    for s in S:
        vnf_ids = instance.V_of_s[s]
        for q in range(len(vnf_ids) - 1):
            i, j = vnf_ids[q], vnf_ids[q + 1]
            if astar_edge_cands and (s, i, j) in astar_edge_cands:
                cand = set((min(u, v), max(u, v)) for (u, v) in astar_edge_cands[(s, i, j)])
                E_sij[(s, i, j)] = [e for e in E_use if e in cand]
            else:
                E_sij[(s, i, j)] = list(E_use)

    # Edge candidates for ENTRY
    E_entry = {}
    for s in S:
        if astar_entry_edge_cands and s in astar_entry_edge_cands:
            cand = set((min(u, v), max(u, v)) for (u, v) in astar_entry_edge_cands[s])
            E_entry[s] = [e for e in E_use if e in cand]
        else:
            E_entry[s] = list(E_use)

    # -------------------------
    # Model
    # -------------------------
    m = gp.Model("SlicePlacementEnergyMultiObj")
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

    # Placement variables
    v = {
        (i, n): m.addVar(vtype=GRB.BINARY, name=f"v_{i}_{n}")
        for s in S
        for i in instance.V_of_s[s]
        for n in N_i[i]
    }

    # Node load and activation
    u = {n: m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"u_{n}") for n in N}
    z = {n: m.addVar(vtype=GRB.BINARY, name=f"z_{n}") for n in N}

    # Flow variables for each VL (between VNFs)
    f = {}
    for s in S:
        vnf_ids = instance.V_of_s[s]
        for q in range(len(vnf_ids) - 1):
            i, j = vnf_ids[q], vnf_ids[q + 1]
            for e in E_sij[(s, i, j)]:
                f[(e, s, (i, j))] = m.addVar(
                    vtype=GRB.CONTINUOUS,
                    lb=0,
                    ub=1,
                    name=f"f_{e[0]}_{e[1]}_s{s}_{i}_to_{j}",
                )

    # Flow variables for ENTRY
    f_entry = {}
    for s in S:
        s_entry = get_entry(s)
        if s_entry is not None:
            for e in E_entry[s]:
                f_entry[(e, s)] = m.addVar(
                    vtype=GRB.CONTINUOUS,
                    lb=0,
                    ub=1,
                    name=f"f_entry_{e[0]}_{e[1]}_s{s}",
                )

    # Link load and activation
    rho = {
        e: m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"rho_{e[0]}_{e[1]}")
        for e in E_use
    }
    w = {
        e: m.addVar(vtype=GRB.BINARY, name=f"w_{e[0]}_{e[1]}")
        for e in E_use
    }

    # Slack variables for soft latency (VLs and entry)
    slack_lat = {}
    slack_lat_entry = {}

    if use_soft_latency:
        # Slack for latency of each (s,i,j)
        for s in S:
            vnf_ids = instance.V_of_s[s]
            for q in range(len(vnf_ids) - 1):
                i, j = vnf_ids[q], vnf_ids[q + 1]
                slack_lat[(s, i, j)] = m.addVar(
                    vtype=GRB.CONTINUOUS,
                    lb=0.0,
                    name=f"slack_lat_s{s}_{i}_to_{j}",
                )

        # Slack for latency of each ENTRY → first VNF
        for s in S:
            s_entry = get_entry(s)
            if s_entry is not None and any((e, s) in f_entry for e in E_use):
                slack_lat_entry[s] = m.addVar(
                    vtype=GRB.CONTINUOUS,
                    lb=0.0,
                    name=f"slack_lat_entry_s{s}",
                )

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
            entry_path = best_astar[s].get("entry", [])

            # Warm-start placement
            for i, n_best in place.items():
                for n in N_i[i]:
                    if (i, n) in v:
                        v[(i, n)].Start = 1.0 if n == n_best else 0.0

            # Warm-start VL paths
            for (i, j), edges in paths.items():
                norm_edges = set((min(u, v), max(u, v)) for (u, v) in edges)
                for e in E_use:
                    if (e, s, (i, j)) in f:
                        f[(e, s, (i, j))].Start = 1.0 if e in norm_edges else 0.0

            # Warm-start entry path
            if entry_path:
                norm_entry = set((min(u, v), max(u, v)) for (u, v) in entry_path)
                for e in E_use:
                    if (e, s) in f_entry:
                        f_entry[(e, s)].Start = 1.0 if e in norm_entry else 0.0

    # -------------------------
    # Objective(s)
    # -------------------------

    # Energy-like objective: node load + activation, link load + activation
    energy_expr = (
        gp.quicksum(u[n] + z[n] for n in N)
        + gp.quicksum(rho[e] + w[e] for e in E_use)
    )

    # Latency objective: total latency over all flows (VLs + entry)
    latency_terms_vls = []
    for s in S:
        vnf_ids = instance.V_of_s[s]
        for q in range(len(vnf_ids) - 1):
            i, j = vnf_ids[q], vnf_ids[q + 1]
            for e in E_use:
                if (e, s, (i, j)) in f:
                    lat_e = float(instance.lat_e.get(e, instance.lat_e.get((e[1], e[0]), 1.0)))
                    latency_terms_vls.append(lat_e * f[(e, s, (i, j))])

    latency_terms_entry = []
    for s in S:
        for e in E_use:
            if (e, s) in f_entry:
                lat_e = float(instance.lat_e.get(e, instance.lat_e.get((e[1], e[0]), 1.0)))
                latency_terms_entry.append(lat_e * f_entry[(e, s)])

    latency_expr = gp.quicksum(latency_terms_vls) + gp.quicksum(latency_terms_entry)

    # Slack penalty (soft latency)
    if use_soft_latency and (slack_lat or slack_lat_entry):
        slack_penalty = latency_slack_penalty * (
            gp.quicksum(slack_lat[key] for key in slack_lat)
            + gp.quicksum(slack_lat_entry[s] for s in slack_lat_entry)
        )
    else:
        slack_penalty = 0.0

    if not use_multiobj:
        # Single-objective: energy + penalty for latency violations
        m.setObjective(energy_expr + slack_penalty, GRB.MINIMIZE)
    else:
        # Multi-objective:
        #  - Objective 0: minimize energy + slack penalty
        #  - Objective 1: minimize total latency
        m.setObjectiveN(
            energy_expr + slack_penalty,
            index=0,
            priority=energy_priority,
            weight=1.0,
            name="MinEnergyPlusSlack",
        )
        m.setObjectiveN(
            latency_expr,
            index=1,
            priority=latency_priority,
            weight=1.0,
            name="MinLatency",
        )

    # -------------------------
    # Solve
    # -------------------------
    m.optimize()

    status_code = m.status
    status_str = str(m.Status)

    # If not optimal or suboptimal, return an empty result object (never None)
    if status_code not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
        if msg:
            print(f"[MILP-MULTI] No feasible solution or unsolved. Status={status_code}")
        if status_code == GRB.INFEASIBLE:
            try:
                m.computeIIS()
                m.write("model_infeasible_multiobj.ilp")
                if msg:
                    print("[MILP-MULTI] IIS written to model_infeasible_multiobj.ilp")
            except Exception as e:
                if msg:
                    print(f"[MILP-MULTI] Could not write IIS: {e}")

        return GurobiSolveResult(
            status_code=status_code,
            status_str=status_str,
            objective=None,
            values={},
        )

    # -------------------------
    # Collect solution
    # -------------------------
    values = {}

    # Placement variables
    for key, var in v.items():
        values[("v",) + key] = var.X

    # Node load and activation
    for n in N:
        values[("u", n)] = u[n].X
        values[("z", n)] = z[n].X

    # Flow variables for VLs
    for key, var in f.items():
        values[("f",) + key] = var.X

    # Flow variables for entry
    for key, var in f_entry.items():
        values[("f_entry",) + key] = var.X

    # Link load and activation
    for e in E_use:
        values[("rho", e)] = rho[e].X
        values[("w", e)] = w[e].X

    # Slack variables (if soft latency is enabled)
    for key, var in slack_lat.items():
        values[("slack_lat",) + key] = var.X
    for s in slack_lat_entry:
        values[("slack_lat_entry", s)] = slack_lat_entry[s].X

    objective_value = m.objVal

    return GurobiSolveResult(
        status_code=status_code,
        status_str=status_str,
        objective=objective_value,
        values=values,
    )
