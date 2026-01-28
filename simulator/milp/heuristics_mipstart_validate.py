from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import gurobipy as gp
from gurobipy import GRB


def _normalize_token(x: Any) -> str:
    s = str(x).strip().lower()
    s = s.replace(" ", "")
    s = re.sub(r"[^a-z0-9_(),\-\[\]]", "", s)
    return s


def _resolve_vnf_key_for_milp(i_milp: Any, placed: Dict[Any, Any], s: Any) -> Optional[Any]:
    if i_milp in placed:
        return i_milp
    si = str(i_milp)
    if si in placed:
        return si
    ni = _normalize_token(i_milp)
    if ni in placed:
        return ni
    if si and _normalize_token(si) in placed:
        return _normalize_token(si)

    for c in [(s, i_milp), (s, si), (s, ni)]:
        if c in placed:
            return c

    if isinstance(i_milp, str):
        m = re.match(r"^vnf(\d+)_(\d+)$", i_milp)
        if m:
            s2 = int(m.group(1))
            k2 = int(m.group(2))
            for kk in [(s2, k2), (s2, str(k2)), (str(s2), k2), (str(s2), str(k2))]:
                if kk in placed:
                    return kk
            for kk in [k2, str(k2), _normalize_token(k2)]:
                if kk in placed:
                    return kk
    return None


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


def _eval_constr_with_start(model: gp.Model, constr: gp.Constr) -> Tuple[float, str, float, float]:
    row = model.getRow(constr)
    lhs = 0.0
    for k in range(row.size()):
        v = row.getVar(k)
        c = row.getCoeff(k)
        lhs += float(c) * float(getattr(v, "Start", 0.0) or 0.0)

    sense = constr.Sense
    rhs = float(constr.RHS)

    viol = 0.0
    if sense == GRB.EQUAL:
        viol = abs(lhs - rhs)
    elif sense == GRB.LESS_EQUAL:
        viol = max(0.0, lhs - rhs)
    elif sense == GRB.GREATER_EQUAL:
        viol = max(0.0, rhs - lhs)

    return viol, sense, lhs, rhs


def validate_mip_start(model: gp.Model, *, tol: float = 1e-6, max_violations: int = 50) -> Dict[str, Any]:
    model.update()

    violations = []
    by_family = {}

    for c in model.getConstrs():
        viol, sense, lhs, rhs = _eval_constr_with_start(model, c)
        if viol > tol:
            name = c.ConstrName
            fam = name.split("_")[0] if name else "unknown"
            violations.append((viol, name, fam, sense, lhs, rhs))
            by_family[fam] = by_family.get(fam, 0) + 1

    violations.sort(key=lambda x: x[0], reverse=True)
    top = violations[:max_violations]
    ok = len(violations) == 0

    reason = "VALID (all MILP constraints satisfied by heuristic MIP start)." if ok else "INVALID (heuristic MIP start violates MILP constraints)."
    if not ok:
        main_fams = sorted(by_family.items(), key=lambda x: x[1], reverse=True)[:10]
        fam_txt = ", ".join([f"{k}={v}" for k, v in main_fams])
        reason += f" Most frequent violated constraint families: {fam_txt}."

    return {
        "ok": ok,
        "n_violations": len(violations),
        "by_family": by_family,
        "top": top,
        "reason": reason,
    }


def apply_mip_start_from_heuristic(
    model: gp.Model,
    vars_pack: Dict[str, Any],
    instance: Any,
    result_list: List[Any],
    *,
    force_anti_colocation: bool = True,
    route_if_missing: bool = True,
) -> None:
    N = vars_pack["N"]
    E = vars_pack["E"]
    vl_pairs_by_s = vars_pack["vl_pairs_by_s"]
    x = vars_pack["x"]
    y = vars_pack["y"]
    z = vars_pack["z"]
    a = vars_pack["a"]
    b = vars_pack["b"]

    slice_set = list(instance.S)

    for s in slice_set:
        z[s].Start = 0.0
    for k in x.keys():
        x[k].Start = 0.0
    for k in y.keys():
        y[k].Start = 0.0
    for n in N:
        a[n].Start = 0.0
    for e in E:
        b[e].Start = 0.0

    Gm = nx.DiGraph()
    for (u, v) in E:
        Gm.add_edge(u, v, latency=float(instance.lat_e.get((u, v), 1.0)))

    bw_residual = {e: float(instance.BW_cap.get(e, 0.0)) for e in E}
    used_nodes = set()
    used_edges = set()

    def _path_latency(path_nodes: List[Any]) -> float:
        total = 0.0
        for uu, vv in zip(path_nodes[:-1], path_nodes[1:]):
            total += float(instance.lat_e.get((uu, vv), 1.0))
        return total

    def _commit_path(s_id: Any, i: Any, j: Any, path_nodes: List[Any], demand: float) -> bool:
        for u, v in zip(path_nodes[:-1], path_nodes[1:]):
            key = (s_id, i, j, u, v)
            if key not in y:
                return False
            if bw_residual.get((u, v), 0.0) + 1e-9 < demand:
                return False
        for u, v in zip(path_nodes[:-1], path_nodes[1:]):
            y[s_id, i, j, u, v].Start = 1.0
            used_edges.add((u, v))
            bw_residual[(u, v)] = bw_residual.get((u, v), 0.0) - demand
        return True

    for s in slice_set:
        res = result_list[s] if s < len(result_list) else None
        if res is None:
            continue

        placed = getattr(res, "placed_vnfs", None)
        placed = placed if isinstance(placed, dict) else {}

        routed = getattr(res, "routed_vls", None)
        routed = routed if isinstance(routed, dict) else {}

        used_nodes_in_slice = set()
        ok_place = True

        for i in list(instance.V_of_s[s]):
            key_in_heur = _resolve_vnf_key_for_milp(i, placed, s)
            if key_in_heur is None:
                ok_place = False
                break
            node = placed.get(key_in_heur, None)
            if node is None:
                ok_place = False
                break
            if force_anti_colocation and node in used_nodes_in_slice:
                ok_place = False
                break
            if (s, i, node) not in x:
                ok_place = False
                break
            x[s, i, node].Start = 1.0
            used_nodes.add(node)
            used_nodes_in_slice.add(node)

        if not ok_place:
            z[s].Start = 0.0
            continue

        z[s].Start = 1.0

        entry_node = _get_entry_node(instance, s)

        slice_ok = True
        for (i, j) in vl_pairs_by_s.get(s, []):
            if i == "ENTRY":
                if entry_node is None:
                    slice_ok = False
                    break
                src_node = entry_node
                dst_node = None
                keyj = _resolve_vnf_key_for_milp(j, placed, s)
                if keyj is None:
                    slice_ok = False
                    break
                dst_node = placed.get(keyj, None)
                if dst_node is None:
                    slice_ok = False
                    break
            else:
                srci = _resolve_vnf_key_for_milp(i, placed, s)
                dstj = _resolve_vnf_key_for_milp(j, placed, s)
                if srci is None or dstj is None:
                    slice_ok = False
                    break
                src_node = placed.get(srci, None)
                dst_node = placed.get(dstj, None)
                if src_node is None or dst_node is None:
                    slice_ok = False
                    break

            if src_node == dst_node:
                continue

            demand = float(instance.BW_sij.get((s, i, j), 0.0))

            path_nodes = None
            if (i, j) in routed and isinstance(routed[(i, j)], list) and len(routed[(i, j)]) >= 2:
                path_nodes = routed[(i, j)]
            elif route_if_missing:
                H = nx.DiGraph()
                for (u, v) in E:
                    if bw_residual.get((u, v), 0.0) + 1e-9 >= demand:
                        H.add_edge(u, v, latency=float(instance.lat_e.get((u, v), 1.0)))
                try:
                    path_nodes = nx.shortest_path(H, src_node, dst_node, weight="latency")
                except Exception:
                    path_nodes = None

            if path_nodes is None:
                slice_ok = False
                break
            if path_nodes[0] != src_node or path_nodes[-1] != dst_node:
                slice_ok = False
                break

            L_ij = instance.L_sij.get((s, i, j), None)
            if L_ij is not None:
                if float(_path_latency(path_nodes)) > float(L_ij) + 1e-9:
                    slice_ok = False
                    break

            if not _commit_path(s, i, j, path_nodes, demand):
                slice_ok = False
                break

        if not slice_ok:
            z[s].Start = 0.0
            for i in list(instance.V_of_s[s]):
                for n in N:
                    if (s, i, n) in x:
                        x[s, i, n].Start = 0.0
            for (i, j) in vl_pairs_by_s.get(s, []):
                for (u, v) in E:
                    if (s, i, j, u, v) in y:
                        y[s, i, j, u, v].Start = 0.0

    for n in used_nodes:
        if n in a:
            a[n].Start = 1.0

    for (u, v) in used_edges:
        if (u, v) in b:
            b[u, v].Start = 1.0

    model.update()
