from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import gurobipy as gp
from gurobipy import GRB

# ==============================
# Energy weight params
# ==============================

NODE_ENERGY_WEIGHT = 1.0   # weight for node activation / node energy proxy
LINK_ENERGY_WEIGHT = 1.0   # weight for link activation / link energy proxy


@dataclass
class GurobiSolveResult:
    """Simple container for a MILP solve result."""
    status_code: int
    status_str: str
    objective: float
    values: Dict[Tuple, float]  # maps tuple keys -> float values
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
    model.Params.OutputFlag = 1 if msg else 0

    if log_file:
        model.Params.LogFile = log_file

    if time_limit is not None:
        model.Params.TimeLimit = time_limit

    if mip_gap is not None:
        model.Params.MIPGap = mip_gap

    model.Params.MIPFocus = mip_focus
    model.Params.Heuristics = heuristics
    model.Params.Cuts = cuts
    model.Params.Presolve = presolve
    model.Params.NumericFocus = numeric_focus

    if seed is not None:
        model.Params.Seed = seed

    if threads is not None:
        model.Params.Threads = threads


def _get_slice_vl_pairs(instance: Any, s: Any) -> List[Tuple[Any, Any]]:
    """Return the list of logical VL pairs (i, j) for slice s based on BW_sij keys."""
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
    Multi-slice MILP without latency slack (hard latency constraints).

    Variables:
      - x[s,i,n] binary placement
      - y[s,i,j,u,v] binary edge-use for each logical VL (i->j) on physical edge (u,v)
      - a[n] binary node activation
      - b[u,v] binary link activation

    Constraints:
      - placement, anti-colocation
      - global CPU capacity tied to a[n]
      - global BW capacity tied to b[u,v]
      - flow conservation
      - hard latency: sum_e lat_e[e]*y <= L_ij

    Objective:
      NODE_ENERGY_WEIGHT*(node activation + normalized CPU load)
    + LINK_ENERGY_WEIGHT*(link activation + normalized BW load)
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

    x_index: List[Tuple[Any, Any, Any]] = []
    for s in slice_set:
        for i in instance.V_of_s[s]:
            for n in N:
                x_index.append((s, i, n))
    x = model.addVars(x_index, vtype=GRB.BINARY, name="x")

    a = model.addVars(N, vtype=GRB.BINARY, name="a")
    b = model.addVars(E, vtype=GRB.BINARY, name="b")

    y_index: List[Tuple[Any, Any, Any, Any, Any]] = []
    for s in slice_set:
        for (i, j) in vl_pairs_by_s[s]:
            for (u, v) in E:
                y_index.append((s, i, j, u, v))
    y = model.addVars(y_index, vtype=GRB.BINARY, name="y")

    # ==========================
    # Constraints
    # ==========================

    # 1) Placement
    for s in slice_set:
        V_s = list(instance.V_of_s[s])
        for i in V_s:
            model.addConstr(
                gp.quicksum(x[s, i, n] for n in N) == 1,
                name=f"place_s{s}_i{i}",
            )

    # 2) Anti-colocation per slice
    for s in slice_set:
        V_s = list(instance.V_of_s[s])
        for n in N:
            model.addConstr(
                gp.quicksum(x[s, i, n] for i in V_s) <= 1,
                name=f"anti_coloc_s{s}_n{n}",
            )

    # 3) Global CPU capacity per node tied to a[n]
    for n in N:
        cpu_usage = gp.quicksum(
            instance.CPU_i[i] * x[s, i, n]
            for s in slice_set
            for i in instance.V_of_s[s]
        )
        model.addConstr(
            cpu_usage <= instance.CPU_cap[n] * a[n],
            name=f"cpu_cap_global_n{n}",
        )

    # 4) Global BW capacity per edge tied to b[u,v]
    for (u, v) in E:
        bw_usage = gp.quicksum(
            instance.BW_sij[(s, i, j)] * y[s, i, j, u, v]
            for s in slice_set
            for (i, j) in vl_pairs_by_s[s]
        )
        model.addConstr(
            bw_usage <= instance.BW_cap[(u, v)] * b[u, v],
            name=f"bw_cap_global_{u}_{v}",
        )

    # 5) Flow conservation (binary y)
    for s in slice_set:
        for (i, j) in vl_pairs_by_s[s]:
            for n in N:
                outflow = gp.quicksum(
                    y[s, i, j, n, vv] for (uu, vv) in E if uu == n
                )
                inflow = gp.quicksum(
                    y[s, i, j, uu, n] for (uu, vv) in E if vv == n
                )
                model.addConstr(
                    outflow - inflow == (x[s, i, n] - x[s, j, n]),
                    name=f"flow_cons_s{s}_ij{i}_{j}_n{n}",
                )

    # 6) Hard latency constraints (no slack)
    for s in slice_set:
        for (i, j) in vl_pairs_by_s[s]:
            key = (s, i, j)
            if key not in instance.L_sij:
                continue
            L_ij = instance.L_sij[key]

            lat_expr = gp.quicksum(
                instance.lat_e[(u, v)] * y[s, i, j, u, v] for (u, v) in E
            )

            model.addConstr(
                lat_expr <= L_ij,
                name=f"lat_hard_s{s}_ij{i}_{j}",
            )

    # 7) Node activation
    for n in N:
        placed_any = gp.quicksum(x[s, i, n] for s in slice_set for i in instance.V_of_s[s])
        model.addConstr(
            placed_any <= len(slice_set) * a[n],
            name=f"node_act_{n}",
        )

    # 8) Link activation
    for (u, v) in E:
        used_any = gp.quicksum(
            y[s, i, j, u, v] for s in slice_set for (i, j) in vl_pairs_by_s[s]
        )
        model.addConstr(
            used_any <= len(slice_set) * b[u, v],
            name=f"link_act_{u}_{v}",
        )

    # ==========================
    # Objective (energy proxy)
    # ==========================

    node_cost = gp.quicksum(
        a[n] + (1.0 / instance.CPU_cap[n]) * gp.quicksum(
            instance.CPU_i[i] * x[s, i, n]
            for s in slice_set
            for i in instance.V_of_s[s]
        )
        for n in N if instance.CPU_cap[n] > 0
    )

    link_cost = gp.quicksum(
        b[u, v] + (1.0 / instance.BW_cap[(u, v)]) * gp.quicksum(
            instance.BW_sij[(s, i, j)] * y[s, i, j, u, v]
            for s in slice_set
            for (i, j) in vl_pairs_by_s[s]
        )
        for (u, v) in E if instance.BW_cap[(u, v)] > 0
    )

    model.setObjective(
        NODE_ENERGY_WEIGHT * node_cost + LINK_ENERGY_WEIGHT * link_cost,
        GRB.MINIMIZE,
    )

    # ==========================
    # Solve + return
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

    for (s, i, n) in x_index:
        values[("x", s, i, n)] = float(x[s, i, n].X)

    for (s, i, j, u, v) in y_index:
        values[("f", s, i, j, u, v)] = float(y[s, i, j, u, v].X)

    for n in N:
        values[("a", n)] = float(a[n].X)

    for (u, v) in E:
        values[("b", u, v)] = float(b[u, v].X)

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
    """

    current = list(instance.S) if slice_order is None else list(slice_order)
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

        if res.solcount > 0 and res.objective != float("inf"):
            if msg:
                print(f"[SUCCESS] Feasible with {len(current)} slices (status={res.status_str}).")
            return {
                "accepted_slices": list(current),
                "last_result": res,
                "attempts": attempts,
            }

        removed = current.pop()
        if msg:
            print(f"[INFO] No feasible solution, removing slice {removed} and retrying...")

    return {
        "accepted_slices": [],
        "last_result": None,
        "attempts": attempts,
    }