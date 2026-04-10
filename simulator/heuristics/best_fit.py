from typing import Any, Dict, Tuple

import pandas as pd

try:
    from sage.all import Graph as SageGraph
except Exception:
    SageGraph = None

import networkx as nx


ENTRY_VNF_ID = "ENTRY"


def _edge_key(u, v):
    return (u, v) if u <= v else (v, u)


def _as_vnf_id(v):
    return str(v).strip()


def _vl_key(vl):
    return (_as_vnf_id(vl["from"]), _as_vnf_id(vl["to"]))


def _to_sage_graph_if_needed(G):
    """
    Convert a NetworkX undirected graph to a Sage Graph once.
    Edge label stores latency as float.
    """
    if SageGraph is None:
        raise RuntimeError("Sage is not available (cannot import sage.all).")

    if isinstance(G, SageGraph):
        return G

    if isinstance(G, nx.DiGraph):
        raise ValueError("This Best-Fit expects an undirected graph (nx.Graph).")

    if not isinstance(G, nx.Graph):
        raise TypeError("G must be a Sage Graph or a NetworkX undirected graph.")

    sG = SageGraph()
    sG.add_vertices(list(G.nodes))
    for u, v, data in G.edges(data=True):
        lat = float(data.get("latency", 1.0))
        sG.add_edge(u, v, lat)

    return sG


def _canonicalize_link_capacity(link_capacity_base):
    """
    Canonicalize undirected capacities into (min,max) keys.
    """
    cap = {}
    for (u, v), bw in link_capacity_base.items():
        ek = _edge_key(u, v)
        bw = float(bw)
        if ek in cap and abs(cap[ek] - bw) > 1e-9:
            raise ValueError(f"Conflicting capacities for edge {ek}.")
        cap[ek] = bw
    return cap


def _iter_sage_edges_with_labels(sG):
    """
    Iterate over Sage graph edges with numeric labels.
    """
    for edge in sG.edges(sort=False):
        if len(edge) == 3:
            u, v, label = edge
        else:
            u, v = edge
            label = 1.0
        yield u, v, float(label)


def shortest_feasible_path_with_capacity_sage(sG, u, v, link_capacity_canon, bandwidth):
    """
    Compute the shortest feasible path by latency on the subgraph of edges
    whose residual capacity is at least 'bandwidth'.
    """
    if u == v:
        return [u], 0.0

    feasible = SageGraph()
    feasible.add_vertices(list(sG.vertices()))

    for a, b, lat in _iter_sage_edges_with_labels(sG):
        cap = float(link_capacity_canon.get(_edge_key(a, b), 0.0))
        if cap + 1e-12 >= float(bandwidth):
            feasible.add_edge(a, b, lat)

    try:
        path = feasible.shortest_path(u, v, by_weight=True)
    except Exception:
        return None, None

    if not path:
        return None, None

    lat = 0.0
    for a, b in zip(path[:-1], path[1:]):
        edge_label = feasible.edge_label(a, b)
        lat += float(edge_label if edge_label is not None else 1.0)

    return path, float(lat)


def _infer_entry_bandwidth(vl_chain, first_vnf_id):
    """
    Infer the entry bandwidth consistently.

    Priority:
    1) explicit ENTRY -> first_vnf VL
    2) first outgoing VL from the first VNF
    3) fallback to 0.0
    """
    first_vnf_id = _as_vnf_id(first_vnf_id)

    for vl in vl_chain:
        v_from, v_to = _vl_key(vl)
        if v_from == ENTRY_VNF_ID and v_to == first_vnf_id:
            return float(vl["bandwidth"])

    for vl in vl_chain:
        v_from, _ = _vl_key(vl)
        if v_from == first_vnf_id:
            return float(vl["bandwidth"])

    return 0.0


def _route_newly_routable_internal_vls(
    sG,
    vl_list,
    temp_placed,
    temp_routed,
    temp_link_capacity,
):
    """
    Route internal VLs whose endpoints are already placed and not yet routed.
    Colocated VLs are stored as a trivial route [node].
    """
    delta_lat = 0.0

    for v_from, v_to, bw in vl_list:
        if (v_from, v_to) in temp_routed:
            continue

        src_node = temp_placed.get(v_from)
        dst_node = temp_placed.get(v_to)

        if src_node is None or dst_node is None:
            continue

        if src_node == dst_node:
            temp_routed[(v_from, v_to)] = [src_node]
            continue

        path, lat = shortest_feasible_path_with_capacity_sage(
            sG, src_node, dst_node, temp_link_capacity, bw
        )
        if path is None:
            return False, 0.0

        for a, b in zip(path[:-1], path[1:]):
            ek = _edge_key(a, b)
            temp_link_capacity[ek] -= bw
            if temp_link_capacity[ek] < -1e-9:
                return False, 0.0

        temp_routed[(v_from, v_to)] = path
        delta_lat += lat

    return True, float(delta_lat)


def _route_entry_if_needed(
    sG,
    entry,
    vnf_ids,
    vl_chain,
    temp_placed,
    temp_routed,
    temp_link_capacity,
):
    """
    Route ENTRY -> first VNF once the first VNF is placed.
    """
    if entry is None or not vnf_ids:
        return True, 0.0

    first_id = vnf_ids[0]
    key = (ENTRY_VNF_ID, first_id)

    if key in temp_routed:
        return True, 0.0

    if first_id not in temp_placed:
        return True, 0.0

    dst_node = temp_placed[first_id]
    bw_entry = _infer_entry_bandwidth(vl_chain, first_id)

    if entry == dst_node:
        temp_routed[key] = [entry]
        return True, 0.0

    path, lat = shortest_feasible_path_with_capacity_sage(
        sG, entry, dst_node, temp_link_capacity, bw_entry
    )
    if path is None:
        return False, 0.0

    for a, b in zip(path[:-1], path[1:]):
        ek = _edge_key(a, b)
        temp_link_capacity[ek] -= bw_entry
        if temp_link_capacity[ek] < -1e-9:
            return False, 0.0

    temp_routed[key] = path
    return True, float(lat)


class BFState:
    def __init__(self, placed_vnfs=None, routed_vls=None, g_cost=0.0):
        self.placed_vnfs = placed_vnfs or {}
        self.routed_vls = routed_vls or {}
        self.g_cost = float(g_cost)


def run_best_fit(G, slices, node_capacity_base, link_capacity_base, csv_path=None):
    """
    True Best-Fit for CPU.

    Selection criteria:
    1) minimize residual CPU after placement
    2) tie-break by smaller incremental latency
    3) tie-break by deterministic node order
    """
    sG = _to_sage_graph_if_needed(G)

    node_capacity_global = {n: float(c) for n, c in node_capacity_base.items()}
    link_capacity_global = _canonicalize_link_capacity(link_capacity_base)

    results = []
    full_results = []

    node_order = sorted(sG.vertices())

    for i, slice_data in enumerate(slices, start=1):
        if len(slice_data) == 2:
            vnf_chain, vl_chain = slice_data
            entry = None
        elif len(slice_data) == 3:
            vnf_chain, vl_chain, entry = slice_data
        else:
            raise ValueError(f"Unexpected slice format: {slice_data}")

        vnf_ids = [_as_vnf_id(v["id"]) for v in vnf_chain]
        vnf_cpu = {_as_vnf_id(v["id"]): float(v["cpu"]) for v in vnf_chain}
        vnf_slice = {_as_vnf_id(v["id"]): v["slice"] for v in vnf_chain}

        vl_list = []
        for vl in vl_chain:
            v_from, v_to = _vl_key(vl)
            vl_list.append((v_from, v_to, float(vl["bandwidth"])))

        placed_vnfs = {}
        routed_vls = {}
        g_cost = 0.0

        local_node_capacity = node_capacity_global.copy()
        local_link_capacity = link_capacity_global.copy()

        node_used_slices = {n: set() for n in node_order}

        success = True

        for vnf_id in vnf_ids:
            cpu_needed = vnf_cpu[vnf_id]
            slice_id = vnf_slice[vnf_id]

            best = None
            best_residual = None
            best_delta_lat = None

            candidate_nodes = [
                n
                for n in node_order
                if local_node_capacity.get(n, 0.0) + 1e-12 >= cpu_needed
                and slice_id not in node_used_slices.get(n, set())
            ]

            for node in candidate_nodes:
                residual = float(local_node_capacity[node] - cpu_needed)

                temp_placed = placed_vnfs.copy()
                temp_routed = routed_vls.copy()
                temp_node_capacity = local_node_capacity.copy()
                temp_link_capacity = local_link_capacity.copy()

                temp_node_capacity[node] -= cpu_needed
                temp_placed[vnf_id] = node

                ok_internal, delta_internal = _route_newly_routable_internal_vls(
                    sG=sG,
                    vl_list=vl_list,
                    temp_placed=temp_placed,
                    temp_routed=temp_routed,
                    temp_link_capacity=temp_link_capacity,
                )
                if not ok_internal:
                    continue

                ok_entry, delta_entry = _route_entry_if_needed(
                    sG=sG,
                    entry=entry,
                    vnf_ids=vnf_ids,
                    vl_chain=vl_chain,
                    temp_placed=temp_placed,
                    temp_routed=temp_routed,
                    temp_link_capacity=temp_link_capacity,
                )
                if not ok_entry:
                    continue

                delta_lat = float(delta_internal + delta_entry)

                if best is None:
                    best = (
                        temp_placed,
                        temp_routed,
                        temp_node_capacity,
                        temp_link_capacity,
                        node,
                        residual,
                        delta_lat,
                    )
                    best_residual = residual
                    best_delta_lat = delta_lat
                else:
                    if residual < best_residual - 1e-12:
                        best = (
                            temp_placed,
                            temp_routed,
                            temp_node_capacity,
                            temp_link_capacity,
                            node,
                            residual,
                            delta_lat,
                        )
                        best_residual = residual
                        best_delta_lat = delta_lat
                    elif abs(residual - best_residual) <= 1e-12:
                        if delta_lat < best_delta_lat - 1e-12:
                            best = (
                                temp_placed,
                                temp_routed,
                                temp_node_capacity,
                                temp_link_capacity,
                                node,
                                residual,
                                delta_lat,
                            )
                            best_residual = residual
                            best_delta_lat = delta_lat
                        elif abs(delta_lat - best_delta_lat) <= 1e-12 and node < best[4]:
                            best = (
                                temp_placed,
                                temp_routed,
                                temp_node_capacity,
                                temp_link_capacity,
                                node,
                                residual,
                                delta_lat,
                            )
                            best_residual = residual
                            best_delta_lat = delta_lat

            if best is None:
                success = False
                break

            (
                placed_vnfs,
                routed_vls,
                local_node_capacity,
                local_link_capacity,
                chosen_node,
                _,
                chosen_delta,
            ) = best

            g_cost += chosen_delta
            node_used_slices.setdefault(chosen_node, set()).add(slice_id)

        if success:
            node_capacity_global = local_node_capacity
            link_capacity_global = local_link_capacity
            results.append({"slice": i, "accepted": True, "g_cost": g_cost})
            full_results.append(BFState(placed_vnfs, routed_vls, g_cost))
        else:
            results.append({"slice": i, "accepted": False, "g_cost": None})
            full_results.append(None)

    df = pd.DataFrame(results)
    if csv_path:
        df.to_csv(csv_path, index=False)

    return df, full_results