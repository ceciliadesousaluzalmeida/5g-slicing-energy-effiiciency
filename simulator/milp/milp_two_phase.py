from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import gurobipy as gp
from gurobipy import GRB

NODE_ENERGY_WEIGHT = 1.0
LINK_ENERGY_WEIGHT = 1.0


@dataclass
class GurobiSolveResult:
    status_code: int
    status_str: str
    objective: float
    values: Dict[Tuple, float]
    solcount: int
    runtime: float
    mip_gap: Optional[float] = None
    iis_file: Optional[str] = None


def _status_to_str(status_code: int) -> str:
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
    for v in model.getVars():
        if v.VType != GRB.CONTINUOUS:
            return True
    return False


def _safe_mip_gap(model: gp.Model) -> Optional[float]:
    if not _has_integer_vars(model):
        return None
    if getattr(model, "SolCount", 0) == 0:
        return None
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


def _get_entry_node(instance: Any, s: Any) -> Optional[Any]:
    if hasattr(instance, "entry_of_s"):
        return instance.entry_of_s.get(s)
    if hasattr(instance, "ENTRY_of_s"):
        return instance.ENTRY_of_s.get(s)
    if hasattr(instance, "entry_node_of_s"):
        return instance.entry_node_of_s.get(s)
    if hasattr(instance, "entry_node"):
        return getattr(instance, "entry_node")
    return None


def _get_entry_bw(instance: Any, s: Any, first_vnf: Any) -> Optional[float]:
    BW_sij = getattr(instance, "BW_sij", {})
    if (s, "ENTRY", first_vnf) in BW_sij:
        return float(BW_sij[(s, "ENTRY", first_vnf)])
    if hasattr(instance, "BW_entry_s") and s in instance.BW_entry_s:
        return float(instance.BW_entry_s[s])

    V_s = list(instance.V_of_s[s])
    for i in V_s:
        for j in V_s:
            if (s, i, j) in BW_sij:
                return float(BW_sij[(s, i, j)])
    return None


def _get_slice_vl_pairs(instance: Any, s: Any) -> List[Tuple[Any, Any]]:
    vl_pairs: List[Tuple[Any, Any]] = []
    V_s = list(instance.V_of_s[s])
    BW_sij = getattr(instance, "BW_sij", {})

    for i in V_s:
        for j in V_s:
            if (s, i, j) in BW_sij:
                vl_pairs.append((i, j))

    entry_node = _get_entry_node(instance, s)
    if entry_node is not None and len(V_s) > 0:
        first_vnf = V_s[0]
        bw_entry = _get_entry_bw(instance, s, first_vnf)
        if bw_entry is None:
            raise ValueError(
                f"Missing bandwidth for ENTRY->first_vnf in slice {s}. "
                f"Provide BW_sij[(s,'ENTRY',{first_vnf})] or BW_entry_s[{s}] "
                f"(or at least one internal VL BW to fallback)."
            )
        instance.BW_sij[(s, "ENTRY", first_vnf)] = float(bw_entry)
        vl_pairs.append(("ENTRY", first_vnf))

    return vl_pairs


def _collect_all_vl_pairs(instance: Any, slice_set: List[Any]) -> Dict[Any, List[Tuple[Any, Any]]]:
    return {s: _get_slice_vl_pairs(instance, s) for s in slice_set}


def build_multi_slice_model_with_accept(
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

    z = model.addVars(slice_set, vtype=GRB.BINARY, name="z")

    x_index: List[Tuple[Any, Any, Any]] = [(s, i, n) for s in slice_set for i in instance.V_of_s[s] for n in N]
    x = model.addVars(x_index, vtype=GRB.BINARY, name="x")

    a = model.addVars(N, vtype=GRB.BINARY, name="a")
    b = model.addVars(E, vtype=GRB.BINARY, name="b")

    y_index: List[Tuple[Any, Any, Any, Any, Any]] = [
        (s, i, j, u, v) for s in slice_set for (i, j) in vl_pairs_by_s[s] for (u, v) in E
    ]
    y = model.addVars(y_index, vtype=GRB.BINARY, name="y")

    for s in slice_set:
        for i in list(instance.V_of_s[s]):
            model.addConstr(gp.quicksum(x[s, i, n] for n in N) == z[s], name=f"place_s{s}_i{i}")

    for s in slice_set:
        V_s = list(instance.V_of_s[s])
        for n in N:
            model.addConstr(gp.quicksum(x[s, i, n] for i in V_s) <= z[s], name=f"anti_coloc_s{s}_n{n}")

    for n in N:
        cpu_usage = gp.quicksum(instance.CPU_i[i] * x[s, i, n] for s in slice_set for i in instance.V_of_s[s])
        model.addConstr(cpu_usage <= instance.CPU_cap[n] * a[n], name=f"cpu_cap_global_n{n}")

    for (u, v) in E:
        bw_usage = gp.quicksum(
            instance.BW_sij[(s, i, j)] * y[s, i, j, u, v] for s in slice_set for (i, j) in vl_pairs_by_s[s]
        )
        model.addConstr(bw_usage <= instance.BW_cap[(u, v)] * b[u, v], name=f"bw_cap_global_{u}_{v}")

    for s in slice_set:
        entry_node = _get_entry_node(instance, s)
        for (i, j) in vl_pairs_by_s[s]:
            for n in N:
                outflow = gp.quicksum(y[s, i, j, n, vv] for (uu, vv) in E if uu == n)
                inflow = gp.quicksum(y[s, i, j, uu, n] for (uu, vv) in E if vv == n)

                if i == "ENTRY":
                    if entry_node is None:
                        raise ValueError(f"Slice {s} has ENTRY VL but no entry node defined in instance.")
                    supply = 1.0 if n == entry_node else 0.0
                    rhs = (supply * z[s]) - x[s, j, n]
                else:
                    rhs = x[s, i, n] - x[s, j, n]

                model.addConstr(outflow - inflow == rhs, name=f"flow_cons_s{s}_ij{i}_{j}_n{n}")

    for s in slice_set:
        for (i, j) in vl_pairs_by_s[s]:
            key = (s, i, j)
            if key not in instance.L_sij:
                continue
            L_ij = float(instance.L_sij[key])
            lat_expr = gp.quicksum(instance.lat_e[(u, v)] * y[s, i, j, u, v] for (u, v) in E)
            M_lat = float(sum(instance.lat_e[(u, v)] for (u, v) in E))
            model.addConstr(lat_expr <= L_ij + M_lat * (1 - z[s]), name=f"lat_hard_s{s}_ij{i}_{j}")

    for n in N:
        placed_any = gp.quicksum(x[s, i, n] for s in slice_set for i in instance.V_of_s[s])
        model.addConstr(placed_any <= len(slice_set) * a[n], name=f"node_act_{n}")

    for (u, v) in E:
        used_any = gp.quicksum(y[s, i, j, u, v] for s in slice_set for (i, j) in vl_pairs_by_s[s])
        model.addConstr(used_any <= len(slice_set) * b[u, v], name=f"link_act_{u}_{v}")

    for s in slice_set:
        for (i, j) in vl_pairs_by_s[s]:
            for (u, v) in E:
                model.addConstr(y[s, i, j, u, v] <= z[s], name=f"route_only_if_accept_s{s}_{i}_{j}_{u}_{v}")

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

    vars_pack = {
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
    return model, vars_pack


def solve_two_phase_max_accept_then_min_energy(
    instance: Any,
    *,
    slice_order: Optional[List[Any]] = None,
    msg: bool = False,
    time_limit: Optional[float] = None,
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
    slice_set = list(instance.S) if slice_order is None else list(slice_order)

    log_file = None
    if log_file_prefix:
        log_file = f"{log_file_prefix}_two_phase_{len(slice_set)}.log"

    model, vp = build_multi_slice_model_with_accept(
        instance=instance,
        slice_set=slice_set,
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

    z = vp["z"]
    total_energy = vp["total_energy"]

    model.ModelSense = GRB.MAXIMIZE
    model.setObjective(gp.quicksum(z[s] for s in slice_set))
    model.optimize()

    status1 = model.Status
    if model.SolCount == 0 or status1 in (GRB.INFEASIBLE, GRB.INF_OR_UNBD, GRB.UNBOUNDED):
        iis_file = None
        if status1 == GRB.INFEASIBLE and iis_on_infeasible:
            try:
                model.computeIIS()
                iis_file = f"iis_two_phase_{len(slice_set)}.ilp"
                model.write(iis_file)
            except gp.GurobiError:
                iis_file = None
        return {
            "accepted_slices": [],
            "rejected_slices": list(slice_set),
            "phase1_status": _status_to_str(status1),
            "phase2_status": None,
            "max_accepted": 0,
            "objective_energy": None,
            "runtime_phase1": float(model.Runtime),
            "runtime_phase2": None,
            "mip_gap_phase1": _safe_mip_gap(model),
            "mip_gap_phase2": None,
            "iis_file": iis_file,
        }

    Z_star = int(round(sum(z[s].X for s in slice_set)))

    model.addConstr(gp.quicksum(z[s] for s in slice_set) == Z_star, name="fix_max_acceptance")

    model.ModelSense = GRB.MINIMIZE
    model.setObjective(total_energy)
    model.optimize()

    status2 = model.Status
    if model.SolCount == 0:
        return {
            "accepted_slices": [],
            "rejected_slices": list(slice_set),
            "phase1_status": _status_to_str(status1),
            "phase2_status": _status_to_str(status2),
            "max_accepted": Z_star,
            "objective_energy": None,
            "runtime_phase1": None,
            "runtime_phase2": float(model.Runtime),
            "mip_gap_phase1": None,
            "mip_gap_phase2": _safe_mip_gap(model),
            "iis_file": None,
        }

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
        values[("z", s)] = float(z[s].X)

    accepted_slices = [s for s in slice_set if values.get(("z", s), 0.0) > 0.5]
    rejected_slices = [s for s in slice_set if s not in accepted_slices]

    return {
        "accepted_slices": accepted_slices,
        "rejected_slices": rejected_slices,
        "phase1_status": _status_to_str(status1),
        "phase2_status": _status_to_str(status2),
        "max_accepted": Z_star,
        "objective_energy": float(model.ObjVal),
        "runtime_phase1": None,
        "runtime_phase2": float(model.Runtime),
        "mip_gap_phase1": None,
        "mip_gap_phase2": _safe_mip_gap(model),
        "values": values,
        "log_file": log_file,
    }
