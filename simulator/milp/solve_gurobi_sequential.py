from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import gurobipy as gp
from gurobipy import GRB


# ==============================
# Energy / latency weight params
# ==============================

NODE_ENERGY_WEIGHT = 1.0   # weight for normalized node CPU usage
LINK_ENERGY_WEIGHT = 1.0   # weight for normalized link bandwidth usage
LATENCY_PENALTY   = 1e3    # penalty for latency slack xi


@dataclass
class GurobiSolveResult:
    """Simple container for a MILP solve result."""
    status_code: int
    status_str: str
    objective: float
    values: dict  # maps tuple keys -> float values
    solcount: int
    runtime: float
    mip_gap: Optional[float] = None
    iis_file: Optional[str] = None


def _status_to_str(status_code: int) -> str:
    """Map Gurobi status codes to readable strings."""
    mapping = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
        GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
        GRB.NODE_LIMIT: "NODE_LIMIT",
        GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
        GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
    }
    return mapping.get(status_code, f"STATUS_{status_code}")


def _apply_fast_params(
    model: gp.Model,
    *,
    msg: bool,
    time_limit: Optional[float],
    mip_gap: Optional[float],
    mip_focus: int,
    heuristics: float,
    cuts: int,
    presolve: int,
    numeric_focus: int,
    seed: Optional[int],
    threads: Optional[int],
    log_file: Optional[str],
):
    # Fast settings to get a good feasible solution quickly
    model.Params.OutputFlag = 1 if msg else 0

    if log_file:
        model.Params.LogFile = log_file

    if time_limit is not None:
        model.Params.TimeLimit = time_limit

    if mip_gap is not None:
        model.Params.MIPGap = mip_gap

    # 1: find feasible fast, 2: prove optimality, 3: improve bound
    model.Params.MIPFocus = mip_focus

    # Heuristic effort and cuts balance
    model.Params.Heuristics = heuristics
    model.Params.Cuts = cuts

    # Presolve usually helps a lot
    model.Params.Presolve = presolve

    # Numerical stability (increase if you see instability)
    model.Params.NumericFocus = numeric_focus

    if seed is not None:
        model.Params.Seed = seed

    if threads is not None:
        model.Params.Threads = threads


def _get_slice_vl_pairs(instance: Any, s: Any) -> List[Tuple[Any, Any]]:
    """
    Helper: return the list of logical VL pairs (i, j) for slice s
    based on instance.V_of_s and instance.BW_sij keys.
    """
    vl_pairs: List[Tuple[Any, Any]] = []
    V_s = instance.V_of_s[s]
    for i in V_s:
        for j in V_s:
            if (s, i, j) in instance.BW_sij:
                vl_pairs.append((i, j))
    return vl_pairs


def _collect_all_vl_pairs(instance: Any, slice_set: List[Any]) -> Dict[Any, List[Tuple[Any, Any]]]:
    """Collect VL pairs per slice to avoid recomputation."""
    return {s: _get_slice_vl_pairs(instance, s) for s in slice_set}


def build_multi_slice_model(
    instance: Any,
    slice_set: List[Any],
    msg: bool = False,
    time_limit: Optional[float] = None,
    *,
    mip_gap: Optional[float] = 0.02,
    mip_focus: int = 1,
    heuristics: float = 0.6,
    cuts: int = 1,
    presolve: int = 2,
    numeric_focus: int = 1,
    seed: Optional[int] = 0,
    threads: Optional[int] = None,
    log_file: Optional[str] = None,
    iis_on_infeasible: bool = True,
) -> GurobiSolveResult:
    """
    Build and solve a single MILP for a given subset of slices.

    Expected instance fields (same as your single-slice solver):
      - instance.N        : list of physical nodes
      - instance.E        : list of physical directed edges (u, v)
      - instance.V_of_s   : dict s -> list of global VNF ids
      - instance.CPU_i    : dict vnf_id -> cpu demand
      - instance.CPU_cap  : dict node -> cpu capacity
      - instance.BW_cap   : dict (u,v) -> link capacity
      - instance.BW_sij   : dict (s,i,j) -> bandwidth demand for VL i->j
      - instance.L_sij    : dict (s,i,j) -> latency SLA for VL i->j (optional per VL)
      - instance.lat_e    : dict (u,v) -> link latency

    Model:
      - x[s,i,n] binary placement
      - f[s,i,j,u,v] continuous flow for each logical VL (i->j)
      - xi[s] continuous slack for latency constraints (soft SLA)
      - CPU & BW capacities are GLOBAL (across all slices in slice_set)
      - Anti-colocation per slice (VNFs of same slice cannot share a node)

    Objective:
      energy proxy (normalized CPU + normalized BW) + LATENCY_PENALTY * sum(xi[s])
    """

    model = gp.Model(f"multi_slice_{len(slice_set)}")

    _apply_fast_params(
        model,
        msg=msg,
        time_limit=time_limit,
        mip_gap=mip_gap,
        mip_focus=mip_focus,
        heuristics=heuristics,
        cuts=cuts,
        presolve=presolve,
        numeric_focus=numeric_focus,
        seed=seed,
        threads=threads,
        log_file=log_file,
    )

    N = list(instance.N)
    E = list(instance.E)
    vl_pairs_by_s = _collect_all_vl_pairs(instance, slice_set)

    # ==========================
    # Variables
    # ==========================

    # Placement variables x[s,i,n]
    x_index = []
    for s in slice_set:
        for i in instance.V_of_s[s]:
            for n in N:
                x_index.append((s, i, n))
    x = model.addVars(x_index, vtype=GRB.BINARY, name="x")

    # Flow variables f[s,i,j,u,v] for each logical VL and physical edge
    f_index = []
    for s in slice_set:
        for (i, j) in vl_pairs_by_s[s]:
            for (u, v) in E:
                f_index.append((s, i, j, u, v))
    f = model.addVars(f_index, vtype=GRB.CONTINUOUS, lb=0.0, name="f")

    # Latency slack per slice xi[s]
    xi = model.addVars(slice_set, vtype=GRB.CONTINUOUS, lb=0.0, name="xi")

    # ==========================
    # Constraints
    # ==========================

    # 1) Each VNF i of slice s must be placed on exactly one node (hard acceptance)
    for s in slice_set:
        V_s = list(instance.V_of_s[s])
        for i in V_s:
            model.addConstr(
                gp.quicksum(x[s, i, n] for n in N) == 1,
                name=f"place_s{s}_i{i}",
            )

    # 2) Hard anti-colocation per slice: at most one VNF of slice s per node n
    for s in slice_set:
        V_s = list(instance.V_of_s[s])
        for n in N:
            model.addConstr(
                gp.quicksum(x[s, i, n] for i in V_s) <= 1,
                name=f"anti_coloc_s{s}_n{n}",
            )

    # 3) Global CPU capacity per node (sum over slices)
    for n in N:
        cpu_usage = gp.quicksum(
            instance.CPU_i[i] * x[s, i, n]
            for s in slice_set
            for i in instance.V_of_s[s]
        )
        model.addConstr(
            cpu_usage <= instance.CPU_cap[n],
            name=f"cpu_cap_global_n{n}",
        )

    # 4) Global BW capacity per physical edge (sum over slices and VLs)
    for (u, v) in E:
        bw_usage = gp.quicksum(
            f[s, i, j, u, v]
            for s in slice_set
            for (i, j) in vl_pairs_by_s[s]
        )
        model.addConstr(
            bw_usage <= instance.BW_cap[(u, v)],
            name=f"bw_cap_global_{u}_{v}",
        )

    # 5) Flow conservation per slice and per logical VL (i->j)
    #    outflow - inflow = BW_sij * (x[s,i,n] - x[s,j,n])
    for s in slice_set:
        for (i, j) in vl_pairs_by_s[s]:
            bw_ij = instance.BW_sij[(s, i, j)]
            for n in N:
                outflow = gp.quicksum(
                    f[s, i, j, n, v] for (uu, v) in E if uu == n
                )
                inflow = gp.quicksum(
                    f[s, i, j, u, n] for (u, vv) in E if vv == n
                )
                model.addConstr(
                    outflow - inflow == bw_ij * (x[s, i, n] - x[s, j, n]),
                    name=f"flow_cons_s{s}_ij{i}_{j}_n{n}",
                )

    # 6) Latency constraints with slack xi[s]
    #    For each VL (i->j) of slice s:
    #       (1 / BW_sij) * sum_e lat_e[e] * f_sij[e] <= L_sij + xi[s]
    for s in slice_set:
        for (i, j) in vl_pairs_by_s[s]:
            key = (s, i, j)
            if key not in instance.L_sij:
                continue

            bw_ij = instance.BW_sij[key]
            L_ij = instance.L_sij[key]

            lat_expr = (1.0 / bw_ij) * gp.quicksum(
                instance.lat_e[(u, v)] * f[s, i, j, u, v] for (u, v) in E
            )

            model.addConstr(
                lat_expr <= L_ij + xi[s],
                name=f"lat_s{s}_ij{i}_{j}",
            )

    # ==========================
    # Objective (energy proxy)
    # ==========================

    # Node energy proxy: normalized CPU usage per node (global)
    node_energy_terms = []
    for n in N:
        cap_n = instance.CPU_cap[n]
        if cap_n > 0:
            cpu_usage_n = gp.quicksum(
                instance.CPU_i[i] * x[s, i, n]
                for s in slice_set
                for i in instance.V_of_s[s]
            )
            node_energy_terms.append(cpu_usage_n / cap_n)
    E_nodes = gp.quicksum(node_energy_terms)

    # Link energy proxy: normalized bandwidth usage per link (global)
    link_energy_terms = []
    for (u, v) in E:
        cap_uv = instance.BW_cap[(u, v)]
        if cap_uv > 0:
            bw_usage_uv = gp.quicksum(
                f[s, i, j, u, v]
                for s in slice_set
                for (i, j) in vl_pairs_by_s[s]
            )
            link_energy_terms.append(bw_usage_uv / cap_uv)
    E_links = gp.quicksum(link_energy_terms)

    energy_expr = NODE_ENERGY_WEIGHT * E_nodes + LINK_ENERGY_WEIGHT * E_links
    model.setObjective(energy_expr + LATENCY_PENALTY * gp.quicksum(xi[s] for s in slice_set), GRB.MINIMIZE)

    # ==========================
    # Solve + robust return
    # ==========================

    model.optimize()

    status = model.Status
    status_str = _status_to_str(status)
    solcount = int(model.SolCount)
    runtime = float(model.Runtime)
    gap = None
    try:
        gap = float(model.MIPGap)
    except gp.GurobiError:
        gap = None

    # If infeasible, optionally write IIS
    iis_file = None
    if status == GRB.INFEASIBLE and iis_on_infeasible:
        try:
            model.computeIIS()
            iis_file = f"iis_multi_{len(slice_set)}_slices.ilp"
            model.write(iis_file)
        except gp.GurobiError:
            iis_file = None

        return GurobiSolveResult(
            status_code=status,
            status_str=status_str,
            objective=float("inf"),
            values={},
            solcount=solcount,
            runtime=runtime,
            mip_gap=gap,
            iis_file=iis_file,
        )

    # If INF_OR_UNBD, disambiguate
    if status == GRB.INF_OR_UNBD:
        try:
            model.Params.DualReductions = 0
            model.optimize()
            status = model.Status
            status_str = _status_to_str(status)
            solcount = int(model.SolCount)
            runtime = float(model.Runtime)
            try:
                gap = float(model.MIPGap)
            except gp.GurobiError:
                gap = None
        except gp.GurobiError:
            pass

    # Return best incumbent if available (TIME_LIMIT included)
    if solcount == 0:
        return GurobiSolveResult(
            status_code=status,
            status_str=status_str,
            objective=float("inf"),
            values={},
            solcount=solcount,
            runtime=runtime,
            mip_gap=gap,
            iis_file=iis_file,
        )

    values: Dict[Tuple, float] = {}

    # Store x
    for (s, i, n) in x_index:
        values[("x", s, i, n)] = float(x[s, i, n].X)

    # Store f
    for (s, i, j, u, v) in f_index:
        values[("f", s, i, j, u, v)] = float(f[s, i, j, u, v].X)

    # Store xi
    for s in slice_set:
        values[("xi", s)] = float(xi[s].X)

    obj = float(model.ObjVal)

    return GurobiSolveResult(
        status_code=status,
        status_str=status_str,
        objective=obj,
        values=values,
        solcount=solcount,
        runtime=runtime,
        mip_gap=gap,
        iis_file=iis_file,
    )


def solve_gurobi_shrink_until_feasible(
    instance: Any,
    slice_order: Optional[List[Any]] = None,
    msg: bool = False,
    time_limit: Optional[float] = None,
    *,
    mip_gap: Optional[float] = 0.02,
    mip_focus: int = 1,
    heuristics: float = 0.6,
    cuts: int = 1,
    presolve: int = 2,
    numeric_focus: int = 1,
    seed: Optional[int] = 0,
    threads: Optional[int] = None,
    log_file_prefix: Optional[str] = None,
    iis_on_infeasible: bool = True,
) -> Dict[str, Any]:
    """
    Try to solve ALL slices at once.
    If infeasible, remove 1 slice and retry until feasible.

    Removal policy:
      - removes the last slice in slice_order (or instance.S order)

    Returns:
      - accepted_slices: the final feasible subset (maximal under this removal policy)
      - last_result: GurobiSolveResult for the final subset (or None)
      - attempts: list of (subset, status_str, solcount, runtime, objective)
    """

    if slice_order is None:
        current = list(instance.S)
    else:
        current = list(slice_order)

    attempts = []

    while len(current) > 0:
        if msg:
            print(f"[INFO] Trying multi-slice MILP with {len(current)} slices: {current}")

        log_file = None
        if log_file_prefix:
            log_file = f"{log_file_prefix}_multi_{len(current)}.log"

        res = build_multi_slice_model(
            instance=instance,
            slice_set=current,
            msg=msg,
            time_limit=time_limit,
            mip_gap=mip_gap,
            mip_focus=mip_focus,
            heuristics=heuristics,
            cuts=cuts,
            presolve=presolve,
            numeric_focus=numeric_focus,
            seed=seed,
            threads=threads,
            log_file=log_file,
            iis_on_infeasible=iis_on_infeasible,
        )

        attempts.append(
            (
                list(current),
                res.status_str,
                res.solcount,
                res.runtime,
                res.objective,
                res.mip_gap,
                res.iis_file,
            )
        )

        # Feasible if we have an incumbent
        if res.solcount > 0 and res.objective != float("inf"):
            if msg:
                print(f"[SUCCESS] Feasible with {len(current)} slices (status={res.status_str}).")
            return {
                "accepted_slices": list(current),
                "last_result": res,
                "attempts": attempts,
            }

        # Infeasible or no incumbent -> remove one slice and retry
        removed = current.pop()
        if msg:
            print(f"[INFO] No feasible solution, removing slice {removed} and retrying...")

    return {
        "accepted_slices": [],
        "last_result": None,
        "attempts": attempts,
    }
