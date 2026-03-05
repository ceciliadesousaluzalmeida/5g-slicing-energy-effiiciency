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
    Edge label stores latency (float).
    """
    if SageGraph is None:
        raise RuntimeError("Sage is not available (cannot import sage.all).")

    if isinstance(G, SageGraph):
        return G

    if isinstance(G, nx.DiGraph):
        raise ValueError("This First-Fit expects an undirected graph (nx.Graph).")

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


class FFState:
    def __init__(self, placed_vnfs=None, routed_vls=None, g_cost=0.0):
        self.placed_vnfs = placed_vnfs or {}
        self.routed_vls = routed_vls or {}
        self.g_cost = float(g_cost)


def run_first_fit(G, slices, node_capacity_base, link_capacity_base, *, csv_path=None):
    """
    True First-Fit.
    For each VNF in chain order:
      - scan nodes in a fixed order
      - choose the first node that keeps placement + newly-routable VLs feasible
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

        # Normalize for faster loops
        vnf_ids = [_as_vnf_id(v["id"]) for v in vnf_chain]
        vnf_cpu = {_as_vnf_id(v["id"]): float(v["cpu"]) for v in vnf_chain}
        vnf_slice = {_as_vnf_id(v["id"]): v["slice"] for v in vnf_chain}

        vl_list = []
        for vl in vl_chain:
            (v_from, v_to) = _vl_key(vl)
            vl_list.append((v_from, v_to, float(vl["bandwidth"])))

        placed_vnfs = {}
        routed_vls = {}
        g_cost = 0.0
        success = True

        local_node_capacity = node_capacity_global.copy()
        local_link_capacity = link_capacity_global.copy()

        # Anti-affinity O(1)
        node_used_slices = {n: set() for n in sG.vertices()}

        # Fixed node order for determinism
        node_order = sorted(sG.vertices())

        for vnf_id in vnf_ids:
            cpu_needed = vnf_cpu[vnf_id]
            slice_id = vnf_slice[vnf_id]

            placed = False

            # First-Fit scan
            for node in node_order:
                avail_cpu = local_node_capacity.get(node, 0.0)
                if avail_cpu < cpu_needed:
                    continue
                if slice_id in node_used_slices.get(node, set()):
                    continue

                # Simulate placement in temporary copies
                temp_placed = placed_vnfs.copy()
                temp_routed = routed_vls.copy()
                temp_node_capacity = local_node_capacity.copy()
                temp_link_capacity = local_link_capacity.copy()

                temp_node_capacity[node] -= cpu_needed
                temp_placed[vnf_id] = node

                routing_ok = True
                delta_lat = 0.0

                # Route newly-routable internal VLs
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

                # Optional entry->first VNF routing (kept minimal)
                if entry is not None and vnf_ids:
                    first_id = vnf_ids[0]
                    if ("ENTRY", first_id) not in temp_routed and first_id in temp_placed:
                        # If you have a specific bandwidth model for entry, plug it here.
                        bw_entry = 0.0
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

                # Commit immediately (First-Fit)
                placed_vnfs = temp_placed
                routed_vls = temp_routed
                local_node_capacity = temp_node_capacity
                local_link_capacity = temp_link_capacity
                g_cost += delta_lat

                node_used_slices.setdefault(node, set()).add(slice_id)
                placed = True
                break

            if not placed:
                success = False
                break

        if success:
            node_capacity_global = local_node_capacity
            link_capacity_global = local_link_capacity
            results.append({"slice": i, "accepted": True, "g_cost": g_cost})
            full_results.append(FFState(placed_vnfs, routed_vls, g_cost))
        else:
            results.append({"slice": i, "accepted": False, "g_cost": None})
            full_results.append(None)

    df = pd.DataFrame(results)
    if csv_path is not None:
        df.to_csv(str(csv_path), index=False)

    return df, full_results