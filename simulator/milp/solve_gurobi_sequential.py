from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import gurobipy as gp
from gurobipy import GRB

# ==============================
# Energy weight params
# ==============================

NODE_ENERGY_WEIGHT = 1.0
LINK_ENERGY_WEIGHT = 1.0


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


def _has_integer_vars(model: gp.Model) -> bool:
    # English: MIPGap is only defined for MIP models (has integer/binary vars).
    for v in model.getVars():
        if v.VType != GRB.CONTINUOUS:
            return True
    return False


def _safe_mip_gap(model: gp.Model) -> Optional[float]:
    # English: Safe access to MIPGap; it may be unavailable depending on version/status/multiobj.
    if not _has_integer_vars(model):
        return None
    if getattr(model, "SolCount", 0) == 0:
        return None

    # English: Only attempt to read gap for typical MIP statuses with an incumbent.
    if getattr(model, "Status", None) not in (
        GRB.OPTIMAL,
        GRB.SUBOPTIMAL,
        GRB.TIME_LIMIT,
        GRB.INTERRUPTED,
        GRB.SOLUTION_LIMIT,
        GRB.NODE_LIMIT,
        GRB.ITERATION_LIMIT,
    ):
        return None

    try:
        return float(model.MIPGap)
    except Exception:
        return None



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


# ------------------------------
# ENTRY helpers
# ------------------------------
def _get_entry_node(instance: Any, s: Any) -> Optional[Any]:
    """Return the physical ENTRY node for slice s, if available."""
    # English: Support common attribute names to avoid breaking your pipeline.
    if hasattr(instance, "entry_of_s"):
        return instance.entry_of_s.get(s)
    if hasattr(instance, "ENTRY_of_s"):
        return instance.ENTRY_of_s.get(s)
    if hasattr(instance, "entry_node_of_s"):
        return instance.entry_node_of_s.get(s)
    return None


def _get_entry_bw(instance: Any, s: Any, first_vnf: Any) -> Optional[float]:
    """
    Return bandwidth demand for ENTRY->first_vnf.
    Priority:
      1) instance.BW_sij[(s,'ENTRY',first_vnf)] if exists
      2) instance.BW_entry_s[s] if exists
      3) fallback to the bandwidth of the first logical VL in the slice (if any)
    """
    if (s, "ENTRY", first_vnf) in getattr(instance, "BW_sij", {}):
        return float(instance.BW_sij[(s, "ENTRY", first_vnf)])

    if hasattr(instance, "BW_entry_s") and s in instance.BW_entry_s:
        return float(instance.BW_entry_s[s])

    # English: Fallback to "first" VL bandwidth if your dataset doesn't define entry BW explicitly.
    V_s = list(instance.V_of_s[s])
    for i in V_s:
        for j in V_s:
            if (s, i, j) in instance.BW_sij:
                return float(instance.BW_sij[(s, i, j)])

    return None


def _slice_requires_entry(instance: Any, s: Any) -> bool:
    m = getattr(instance, "entry_required_s", None)
    if isinstance(m, dict):
        return bool(m.get(s, False))
    return False


def _get_slice_vl_pairs(instance: Any, s: Any) -> List[Tuple[Any, Any]]:
    vl_pairs: List[Tuple[Any, Any]] = []
    V_s = list(instance.V_of_s[s])

    # Standard VLs between VNFs
    for i in V_s:
        for j in V_s:
            if (s, i, j) in instance.BW_sij:
                vl_pairs.append((i, j))

    # Add ENTRY->first only if this slice requires ENTRY
    if _slice_requires_entry(instance, s):
        entry_node = _get_entry_node(instance, s)
        if entry_node is None:
            raise ValueError(f"Slice {s} requires ENTRY but no entry node defined in instance.")

        if len(V_s) == 0:
            raise ValueError(f"Slice {s} requires ENTRY but has no VNFs.")

        first_vnf = V_s[0]
        bw_entry = _get_entry_bw(instance, s, first_vnf)
        if bw_entry is None:
            raise ValueError(
                f"Missing bandwidth for ENTRY->first_vnf in slice {s}. "
                f"Provide BW_sij[(s,'ENTRY',{first_vnf})] or BW_entry_s[{s}] (or at least one VL BW to fallback)."
            )

        instance.BW_sij[(s, "ENTRY", first_vnf)] = float(bw_entry)
        vl_pairs.append(("ENTRY", first_vnf))

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
    Multi-slice MILP with hard latency constraints.

    Adds support for ENTRY->first_vnf routing:
      - creates y[s,'ENTRY',first,u,v]
      - enforces flow conservation with fixed supply at the physical entry node
      - consumes BW and activates links consistently

    NEW:
      - Adds acceptance variable z[s] to maximize number of accepted slices.
      - Placement/flow/latency constraints are conditioned on z[s].
      - Objective is lexicographic: maximize sum(z) then minimize energy.
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

    # English: z[s] = 1 if slice s is accepted, else 0.
    z = model.addVars(slice_set, vtype=GRB.BINARY, name="z")

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

    # 1) Placement (conditional on z[s])
    for s in slice_set:
        V_s = list(instance.V_of_s[s])
        for i in V_s:
            model.addConstr(
                gp.quicksum(x[s, i, n] for n in N) == z[s],
                name=f"place_s{s}_i{i}",
            )

    # 2) Anti-colocation per slice (active only if slice accepted)
    for s in slice_set:
        V_s = list(instance.V_of_s[s])
        for n in N:
            model.addConstr(
                gp.quicksum(x[s, i, n] for i in V_s) <= z[s],
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

    # 4) Global BW capacity per edge tied to b[u,v] (includes ENTRY VL)
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

    # 5) Flow conservation (binary y), conditional on z[s]
    for s in slice_set:
        entry_node = _get_entry_node(instance, s)

        for (i, j) in vl_pairs_by_s[s]:
            for n in N:
                outflow = gp.quicksum(y[s, i, j, n, vv] for (uu, vv) in E if uu == n)
                inflow = gp.quicksum(y[s, i, j, uu, n] for (uu, vv) in E if vv == n)

                if i == "ENTRY":
                    # English: ENTRY has a fixed physical location => supply = z[s] at entry_node.
                    if entry_node is None:
                        raise ValueError(f"Slice {s} has ENTRY VL but no entry node defined in instance.")
                    supply = 1.0 if n == entry_node else 0.0
                    rhs = (supply * z[s]) - x[s, j, n]
                else:
                    # English: standard VNF->VNF flow: source at x[s,i,n], sink at x[s,j,n]
                    rhs = x[s, i, n] - x[s, j, n]

                model.addConstr(
                    outflow - inflow == rhs,
                    name=f"flow_cons_s{s}_ij{i}_{j}_n{n}",
                )

    # 6) Hard latency constraints (conditional on z[s]) using Big-M
    for s in slice_set:
        for (i, j) in vl_pairs_by_s[s]:
            key = (s, i, j)
            if key not in instance.L_sij:
                continue
            L_ij = float(instance.L_sij[key])
            lat_expr = gp.quicksum(instance.lat_e[(u, v)] * y[s, i, j, u, v] for (u, v) in E)

            # English: Big-M relaxes latency when z[s]=0. Safe upper bound is sum of all edge latencies.
            M_lat = float(sum(instance.lat_e[(u, v)] for (u, v) in E))
            model.addConstr(
                lat_expr <= L_ij + M_lat * (1 - z[s]),
                name=f"lat_hard_s{s}_ij{i}_{j}",
            )

    # 7) Node activation
    for n in N:
        placed_any = gp.quicksum(x[s, i, n] for s in slice_set for i in instance.V_of_s[s])
        model.addConstr(placed_any <= len(slice_set) * a[n], name=f"node_act_{n}")

    # 8) Link activation (includes ENTRY VL)
    for (u, v) in E:
        used_any = gp.quicksum(
            y[s, i, j, u, v] for s in slice_set for (i, j) in vl_pairs_by_s[s]
        )
        model.addConstr(used_any <= len(slice_set) * b[u, v], name=f"link_act_{u}_{v}")

    # 9) Prevent routing if slice not accepted (tightening)
    for s in slice_set:
        for (i, j) in vl_pairs_by_s[s]:
            for (u, v) in E:
                model.addConstr(
                    y[s, i, j, u, v] <= z[s],
                    name=f"route_only_if_accept_s{s}_{i}_{j}_{u}_{v}",
                )

    # ==========================
    # Objective (lexicographic)
    # ==========================

    node_cost = gp.quicksum(
        a[n]
        + (1.0 / instance.CPU_cap[n]) * gp.quicksum(
            instance.CPU_i[i] * x[s, i, n]
            for s in slice_set
            for i in instance.V_of_s[s]
        )
        for n in N if instance.CPU_cap[n] > 0
    )

    link_cost = gp.quicksum(
        b[u, v]
        + (1.0 / instance.BW_cap[(u, v)]) * gp.quicksum(
            instance.BW_sij[(s, i, j)] * y[s, i, j, u, v]
            for s in slice_set
            for (i, j) in vl_pairs_by_s[s]
        )
        for (u, v) in E if instance.BW_cap[(u, v)] > 0
    )

    total_energy = NODE_ENERGY_WEIGHT * node_cost + LINK_ENERGY_WEIGHT * link_cost

    # English: Use a single global MINIMIZE sense for compatibility with older gurobipy.
    model.ModelSense = GRB.MINIMIZE

    # English: Objective 0 (highest priority): maximize acceptance == minimize (-sum(z)).
    model.setObjectiveN(
        -gp.quicksum(z[s] for s in slice_set),
        0,      # index
        2,      # priority
        1.0,    # weight
        0.0,    # abstol
        0.0,    # reltol
    )

    # English: Objective 1 (lower priority): minimize energy among max-accept solutions.
    model.setObjectiveN(
        total_energy,
        1,      # index
        1,      # priority
        1.0,    # weight
        0.0,    # abstol
        0.0,    # reltol
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
        gap = _safe_mip_gap(model)
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
                gap = _safe_mip_gap(model)
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

    for s in slice_set:
        values[("z", s)] = float(z[s].X)

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


def build_multi_slice_model_raw(
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
) -> Tuple[gp.Model, Dict[str, Any]]:
    """
    Build the MILP but do NOT optimize.
    Returns (model, vp) where vp contains vars and indices.
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
    z = model.addVars(slice_set, vtype=GRB.BINARY, name="z")

    x_index: List[Tuple[Any, Any, Any]] = [(s, i, n) for s in slice_set for i in instance.V_of_s[s] for n in N]
    x = model.addVars(x_index, vtype=GRB.BINARY, name="x")

    a = model.addVars(N, vtype=GRB.BINARY, name="a")
    b = model.addVars(E, vtype=GRB.BINARY, name="b")

    y_index: List[Tuple[Any, Any, Any, Any, Any]] = [
        (s, i, j, u, v)
        for s in slice_set
        for (i, j) in vl_pairs_by_s[s]
        for (u, v) in E
    ]
    y = model.addVars(y_index, vtype=GRB.BINARY, name="y")

    # ==========================
    # Constraints
    # ==========================

    # 1) Placement conditioned on z[s]
    for s in slice_set:
        for i in list(instance.V_of_s[s]):
            model.addConstr(gp.quicksum(x[s, i, n] for n in N) == z[s], name=f"place_s{s}_i{i}")

    # 2) Anti-colocation per slice
    for s in slice_set:
        V_s = list(instance.V_of_s[s])
        for n in N:
            model.addConstr(gp.quicksum(x[s, i, n] for i in V_s) <= z[s], name=f"anti_coloc_s{s}_n{n}")

    # 3) CPU capacity tied to a[n]
    for n in N:
        cpu_usage = gp.quicksum(
            instance.CPU_i[i] * x[s, i, n] for s in slice_set for i in instance.V_of_s[s]
        )
        model.addConstr(cpu_usage <= instance.CPU_cap[n] * a[n], name=f"cpu_cap_global_n{n}")

    # 4) BW capacity tied to b[u,v]
    for (u, v) in E:
        bw_usage = gp.quicksum(
            instance.BW_sij[(s, i, j)] * y[s, i, j, u, v]
            for s in slice_set
            for (i, j) in vl_pairs_by_s[s]
        )
        model.addConstr(bw_usage <= instance.BW_cap[(u, v)] * b[u, v], name=f"bw_cap_global_{u}_{v}")

    # 5) Flow conservation
    for s in slice_set:
        entry_node = _get_entry_node(instance, s)

        for (i, j) in vl_pairs_by_s[s]:
            for n in N:
                outflow = gp.quicksum(y[s, i, j, n, vv] for (uu, vv) in E if uu == n)
                inflow  = gp.quicksum(y[s, i, j, uu, n] for (uu, vv) in E if vv == n)

                if i == "ENTRY":
                    if entry_node is None:
                        raise ValueError(f"Slice {s} has ENTRY VL but no entry node defined in instance.")
                    supply = 1.0 if n == entry_node else 0.0
                    rhs = (supply * z[s]) - x[s, j, n]
                else:
                    rhs = x[s, i, n] - x[s, j, n]

                model.addConstr(outflow - inflow == rhs, name=f"flow_cons_s{s}_ij{i}_{j}_n{n}")

    # 6) Hard latency (Big-M when z=0)
    for s in slice_set:
        for (i, j) in vl_pairs_by_s[s]:
            key = (s, i, j)
            if key not in instance.L_sij:
                continue

            L_ij = float(instance.L_sij[key])
            lat_expr = gp.quicksum(instance.lat_e[(u, v)] * y[s, i, j, u, v] for (u, v) in E)

            M_lat = float(sum(instance.lat_e[(u, v)] for (u, v) in E))
            model.addConstr(lat_expr <= L_ij + M_lat * (1 - z[s]), name=f"lat_hard_s{s}_ij{i}_{j}")

    # 7) Node activation
    for n in N:
        placed_any = gp.quicksum(x[s, i, n] for s in slice_set for i in instance.V_of_s[s])
        model.addConstr(placed_any <= len(slice_set) * a[n], name=f"node_act_{n}")

    # 8) Link activation
    for (u, v) in E:
        used_any = gp.quicksum(y[s, i, j, u, v] for s in slice_set for (i, j) in vl_pairs_by_s[s])
        model.addConstr(used_any <= len(slice_set) * b[u, v], name=f"link_act_{u}_{v}")

    # 9) No routing if not accepted
    for s in slice_set:
        for (i, j) in vl_pairs_by_s[s]:
            for (u, v) in E:
                model.addConstr(y[s, i, j, u, v] <= z[s], name=f"route_only_if_accept_s{s}_{i}_{j}_{u}_{v}")

    # ==========================
    # Energy expression
    # ==========================
    node_cost = gp.quicksum(
        a[n]
        + (1.0 / instance.CPU_cap[n]) * gp.quicksum(
            instance.CPU_i[i] * x[s, i, n] for s in slice_set for i in instance.V_of_s[s]
        )
        for n in N if instance.CPU_cap[n] > 0
    )

    link_cost = gp.quicksum(
        b[u, v]
        + (1.0 / instance.BW_cap[(u, v)]) * gp.quicksum(
            instance.BW_sij[(s, i, j)] * y[s, i, j, u, v] for s in slice_set for (i, j) in vl_pairs_by_s[s]
        )
        for (u, v) in E if instance.BW_cap[(u, v)] > 0
    )

    total_energy = NODE_ENERGY_WEIGHT * node_cost + LINK_ENERGY_WEIGHT * link_cost

    vp = {
        "N": N,
        "E": E,
        "vl_pairs_by_s": vl_pairs_by_s,
        "z": z,
        "x": x,
        "y": y,
        "a": a,
        "b": b,
        "x_index": x_index,
        "y_index": y_index,
        "total_energy": total_energy,
    }

    model.update()
    return model, vp


def solve_two_phase_max_accept_then_min_energy(
    instance,
    slice_set,
    *,
    msg=False,
    time_limit_phase1=None,
    time_limit_phase2=None,
    mip_gap=0.02,
    seed=0,
    threads=None,
    log_file_phase1=None,
    log_file_phase2=None,
):
    # Build RAW model (no optimize yet)
    model, vp = build_multi_slice_model_raw(
        instance=instance,
        slice_set=slice_set,
        msg=msg,
        time_limit=time_limit_phase1,
        mip_gap=mip_gap,
        seed=seed,
        threads=threads,
        log_file=log_file_phase1,
    )

    z = vp["z"]
    total_energy = vp["total_energy"]

    # -----------------------
    # Phase 1: maximize acceptance
    # -----------------------
    model.ModelSense = GRB.MAXIMIZE
    model.setObjective(gp.quicksum(z[s] for s in slice_set))
    model.optimize()

    status1 = model.Status
    status1_str = _status_to_str(status1)

    if model.SolCount == 0 or status1 in (GRB.INFEASIBLE, GRB.INF_OR_UNBD, GRB.UNBOUNDED):
        return {
            "phase1_status": status1_str,
            "phase2_status": None,
            "max_accepted": 0,
            "accepted_slices": [],
            "rejected_slices": list(slice_set),
            "last_result": None,
        }

    Z_star = int(round(sum(z[s].X for s in slice_set)))

    print("[PHASE1] status:", _status_to_str(model.Status))
    print("[PHASE1] SolCount:", model.SolCount)
    print("[PHASE1] ObjVal (accepted):", model.ObjVal if model.SolCount else None)


    # -----------------------
    # Phase 2: fix acceptance, minimize energy
    # -----------------------
    if time_limit_phase2 is not None:
        model.Params.TimeLimit = time_limit_phase2
    if log_file_phase2:
        model.Params.LogFile = log_file_phase2

    model.addConstr(gp.quicksum(z[s] for s in slice_set) == Z_star, name="fix_max_acceptance")
    model.ModelSense = GRB.MINIMIZE
    model.setObjective(total_energy)
    model.optimize()

    status2 = model.Status
    status2_str = _status_to_str(status2)

    if model.SolCount == 0:
        return {
            "phase1_status": status1_str,
            "phase2_status": status2_str,
            "max_accepted": Z_star,
            "accepted_slices": [],
            "rejected_slices": list(slice_set),
            "last_result": None,
        }

    # Extract values in the SAME format your adapters/exports expect
    values: Dict[Tuple, float] = {}

    for (s, i, n) in vp["x_index"]:
        values[("x", s, i, n)] = float(vp["x"][s, i, n].X)

    for (s, i, j, u, v) in vp["y_index"]:
        values[("f", s, i, j, u, v)] = float(vp["y"][s, i, j, u, v].X)

    for n in vp["N"]:
        values[("a", n)] = float(vp["a"][n].X)

    for (u, v) in vp["E"]:
        values[("b", u, v)] = float(vp["b"][u, v].X)

    for s in slice_set:
        values[("z", s)] = float(vp["z"][s].X)

    accepted_slices = [s for s in slice_set if values.get(("z", s), 0.0) > 0.5]
    rejected_slices = [s for s in slice_set if s not in accepted_slices]

    res = GurobiSolveResult(
        status_code=status2,
        status_str=status2_str,
        objective=float(model.ObjVal),
        values=values,
        solcount=int(model.SolCount),
        runtime=float(model.Runtime),
        mip_gap=_safe_mip_gap(model),
        iis_file=None,
    )

    return {
        "phase1_status": status1_str,
        "phase2_status": status2_str,
        "max_accepted": Z_star,
        "accepted_slices": accepted_slices,
        "rejected_slices": rejected_slices,
        "last_result": res,
    }
