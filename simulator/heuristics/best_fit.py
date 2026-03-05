from copy import deepcopy
import pandas as pd

try:
    from sage.all import Graph as SageGraph
except Exception:
    SageGraph = None

import networkx as nx


def _edge_key(u, v):
    return (u, v) if u <= v else (v, u)


def _as_vnf_id(v):
    return str(v).strip()


def _vl_key(vl):
    return (_as_vnf_id(vl["from"]), _as_vnf_id(vl["to"]))


def _to_sage_graph_if_needed(G):
    """
    Convert a NetworkX undirected graph to a Sage Graph once.
    Store latency as edge label to enable weighted shortest paths.
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


def shortest_path_with_capacity_sage(sG, u, v, link_capacity_canon, bandwidth):
    """
    Sage weighted shortest path (edge label = latency) + capacity check.
    """
    if u == v:
        return [u], 0.0

    try:
        path = sG.shortest_path(u, v, by_weight=True)
    except Exception:
        return None, None

    if not path:
        return None, None

    for a, b in zip(path[:-1], path[1:]):
        cap = link_capacity_canon.get(_edge_key(a, b))
        if cap is None or cap < bandwidth:
            return None, None

    lat = 0.0
    for a, b in zip(path[:-1], path[1:]):
        lat += float(sG.edge_label(a, b))

    return path, float(lat)


class BFState:
    def __init__(self, placed_vnfs=None, routed_vls=None, g_cost=0.0):
        self.placed_vnfs = placed_vnfs or {}
        self.routed_vls = routed_vls or {}
        self.g_cost = float(g_cost)


def run_best_fit(G, slices, node_capacity_base, link_capacity_base, csv_path=None):
    """
    True Best-Fit for CPU (tightest residual CPU).
    Tie-break: smallest incremental latency from newly-routable VLs.
    """
    sG = _to_sage_graph_if_needed(G)

    node_capacity_global = dict(node_capacity_base)
    link_capacity_global = _canonicalize_link_capacity(link_capacity_base)

    results = []
    full_results = []

    for i, slice_data in enumerate(slices, start=1):
        if len(slice_data) == 2:
            vnf_chain, vl_chain = slice_data
            entry = None
        elif len(slice_data) == 3:
            vnf_chain, vl_chain, entry = slice_data
        else:
            raise ValueError(f"Unexpected slice format: {slice_data}")

        # Normalize VNFs and VLs for fast access
        vnf_ids = [_as_vnf_id(v["id"]) for v in vnf_chain]
        vnf_cpu = {_as_vnf_id(v["id"]): float(v["cpu"]) for v in vnf_chain}
        vnf_slice = {_as_vnf_id(v["id"]): v["slice"] for v in vnf_chain}

        vl_list = []
        bw_by_vl = {}
        for vl in vl_chain:
            k = _vl_key(vl)
            bw = float(vl["bandwidth"])
            vl_list.append((k[0], k[1], bw))
            bw_by_vl[(k[0], k[1])] = bw

        # Local working copies per slice
        placed_vnfs = {}
        routed_vls = {}
        g_cost = 0.0

        local_node_capacity = node_capacity_global.copy()
        local_link_capacity = link_capacity_global.copy()

        # Anti-affinity O(1)
        node_used_slices = {n: set() for n in sG.vertices()}

        success = True

        # Place VNFs in chain order
        for vnf_id in vnf_ids:
            cpu_needed = vnf_cpu[vnf_id]
            slice_id = vnf_slice[vnf_id]

            # Candidate nodes with enough CPU
            candidate_nodes = [
                n for n in sG.vertices()
                if local_node_capacity.get(n, 0.0) >= cpu_needed and slice_id not in node_used_slices.get(n, set())
            ]

            best = None
            best_residual = None
            best_delta_lat = None

            for node in candidate_nodes:
                residual = local_node_capacity[node] - cpu_needed

                # Simulate placement
                temp_placed = placed_vnfs.copy()
                temp_routed = routed_vls.copy()
                temp_node_capacity = local_node_capacity.copy()
                temp_link_capacity = local_link_capacity.copy()

                temp_node_capacity[node] -= cpu_needed
                temp_placed[vnf_id] = node

                delta_lat = 0.0
                routing_ok = True

                # Route newly-routable VLs (endpoints both placed)
                for (v_from, v_to, bw) in vl_list:
                    if (v_from, v_to) in temp_routed:
                        continue

                    src_node = temp_placed.get(v_from)
                    dst_node = temp_placed.get(v_to)
                    if src_node is None or dst_node is None or src_node == dst_node:
                        continue

                    path, lat = shortest_path_with_capacity_sage(
                        sG, src_node, dst_node, temp_link_capacity, bw
                    )
                    if path is None:
                        routing_ok = False
                        break

                    for a, b in zip(path[:-1], path[1:]):
                        ek = _edge_key(a, b)
                        temp_link_capacity[ek] -= bw
                        if temp_link_capacity[ek] < -1e-9:
                            routing_ok = False
                            break
                    if not routing_ok:
                        break

                    temp_routed[(v_from, v_to)] = path
                    delta_lat += lat

                if not routing_ok:
                    continue

                # Optional entry routing (kept minimal and safe)
                if entry is not None and vnf_ids:
                    first_id = vnf_ids[0]
                    if ("ENTRY", first_id) not in temp_routed and first_id in temp_placed:
                        bw_entry = bw_by_vl.get((vnf_ids[0], vnf_ids[1]), 0.0) if len(vnf_ids) >= 2 else 0.0
                        path, lat = shortest_path_with_capacity_sage(
                            sG, entry, temp_placed[first_id], temp_link_capacity, bw_entry
                        )
                        if path is None:
                            routing_ok = False
                        else:
                            for a, b in zip(path[:-1], path[1:]):
                                ek = _edge_key(a, b)
                                temp_link_capacity[ek] -= bw_entry
                                if temp_link_capacity[ek] < -1e-9:
                                    routing_ok = False
                                    break
                            if routing_ok:
                                temp_routed[("ENTRY", first_id)] = path
                                delta_lat += lat

                if not routing_ok:
                    continue

                # Best-Fit selection:
                # 1) minimize residual CPU
                # 2) tie-break by delta latency
                if best is None:
                    best = (temp_placed, temp_routed, temp_node_capacity, temp_link_capacity, node, residual, delta_lat)
                    best_residual = residual
                    best_delta_lat = delta_lat
                else:
                    if residual < best_residual - 1e-12:
                        best = (temp_placed, temp_routed, temp_node_capacity, temp_link_capacity, node, residual, delta_lat)
                        best_residual = residual
                        best_delta_lat = delta_lat
                    elif abs(residual - best_residual) <= 1e-12 and delta_lat < best_delta_lat - 1e-12:
                        best = (temp_placed, temp_routed, temp_node_capacity, temp_link_capacity, node, residual, delta_lat)
                        best_residual = residual
                        best_delta_lat = delta_lat

            if best is None:
                success = False
                break

            placed_vnfs, routed_vls, local_node_capacity, local_link_capacity, chosen_node, _, chosen_delta = best
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