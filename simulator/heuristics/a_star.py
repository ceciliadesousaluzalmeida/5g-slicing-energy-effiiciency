from heapq import heappush, heappop
from copy import deepcopy

import networkx as nx
import pandas as pd


# ------------------------- helpers: normalization -------------------------
def _as_vnf_id(v):
    return str(v).strip()


def _vl_key_from_to(v_from, v_to):
    return (_as_vnf_id(v_from), _as_vnf_id(v_to))


def _vl_key(vl):
    return _vl_key_from_to(vl["from"], vl["to"])


# ------------------------- helpers: canonical undirected edges -------------------------
def _edge_key(u, v):
    return (u, v) if u <= v else (v, u)


def _assert_path_endpoints(path, src_node, dst_node):
    if not path:
        raise ValueError("VL path is empty.")
    if path[0] != src_node or path[-1] != dst_node:
        raise ValueError(
            f"VL path endpoints do not match: expected ({src_node}->{dst_node}), "
            f"got ({path[0]}->{path[-1]})"
        )


# ------------------------- helper: feasible shortest path -------------------------
def shortest_path_with_capacity(G, u, v, link_capacity_canon, bandwidth, edge_latency=None):
    if u == v:
        return [u], 0.0

    try:
        path = nx.shortest_path(G, u, v, weight="latency")
    except nx.NetworkXNoPath:
        return None, None

    for a, b in zip(path[:-1], path[1:]):
        cap = link_capacity_canon.get(_edge_key(a, b))
        if cap is None or cap < bandwidth:
            return None, None

    if edge_latency is None:
        latency = 0.0
        for a, b in zip(path[:-1], path[1:]):
            latency += float(G[a][b].get("latency", 1.0))
    else:
        latency = 0.0
        for a, b in zip(path[:-1], path[1:]):
            latency += edge_latency.get(_edge_key(a, b), 1.0)

    return path, float(latency)


# ------------------------- A* DATA STRUCTURES -------------------------
class AStarState:
    __slots__ = (
        "placed_vnfs",
        "routed_vls",
        "g_cost",
        "node_capacity",
        "link_capacity",
        "node_used_slices",
    )

    def __init__(
        self,
        placed_vnfs=None,
        routed_vls=None,
        g_cost=0.0,
        node_capacity=None,
        link_capacity=None,
        node_used_slices=None,
    ):
        self.placed_vnfs = placed_vnfs or {}         # {vnf_id(str): node}
        self.routed_vls = routed_vls or {}           # {(from_id,to_id): [path nodes]}
        self.g_cost = float(g_cost)
        self.node_capacity = node_capacity or {}     # {node: remaining_cpu}
        self.link_capacity = link_capacity or {}     # {(u,v) canon: remaining_bw}
        self.node_used_slices = node_used_slices or {}  # {node: set(slice_ids)}

    def is_goal(self, vnf_chain_len, vl_list_norm):
        if len(self.placed_vnfs) != vnf_chain_len:
            return False

        # All VLs that require routing must exist and be endpoint-consistent
        for (v_from, v_to, bw) in vl_list_norm:
            src_node = self.placed_vnfs.get(v_from)
            dst_node = self.placed_vnfs.get(v_to)
            if src_node is None or dst_node is None:
                return False
            if src_node == dst_node:
                continue
            path = self.routed_vls.get((v_from, v_to))
            if not path or path[0] != src_node or path[-1] != dst_node:
                return False

        return True


# ---------------------------- A* ORCHESTRATOR ----------------------------
def run_astar(G, slices, node_capacity_base, link_capacity_base, csv_path=None):
    astar_results = []

    # Canonicalize global capacities once
    node_capacity_global = dict(node_capacity_base)

    link_capacity_global = {}
    for (u, v), cap in link_capacity_base.items():
        ek = _edge_key(u, v)
        if ek in link_capacity_global:
            # If both directions exist, keep consistent value
            if abs(link_capacity_global[ek] - float(cap)) > 1e-9:
                raise ValueError(f"Conflicting capacities for edge {ek}")
        else:
            link_capacity_global[ek] = float(cap)

    # Precompute canonical edge latencies
    edge_latency = {}
    for u, v, data in G.edges(data=True):
        edge_latency[_edge_key(u, v)] = float(data.get("latency", 1.0))

    # ---------- Heuristic ----------
    def heuristic_for_slice(state, vl_list_norm):
        h = 0.0
        for (v_from, v_to, bw) in vl_list_norm:
            if (v_from, v_to) in state.routed_vls:
                continue
            src_node = state.placed_vnfs.get(v_from)
            dst_node = state.placed_vnfs.get(v_to)
            if src_node is None or dst_node is None or src_node == dst_node:
                continue
            try:
                h += nx.shortest_path_length(G, src_node, dst_node, weight="latency")
            except nx.NetworkXNoPath:
                h += 10_000.0
        return float(h)

    # ---------- Expand ----------
    def expand_state(state, vnf_chain_ids, vnf_cpu_by_id, vnf_slice_by_id, vl_list_norm):
        expansions = []

        # Choose next VNF in chain order (not sorting ids)
        next_vnf_id = None
        for vid in vnf_chain_ids:
            if vid not in state.placed_vnfs:
                next_vnf_id = vid
                break
        if next_vnf_id is None:
            return expansions

        cpu_need = float(vnf_cpu_by_id[next_vnf_id])
        slice_id = vnf_slice_by_id[next_vnf_id]

        # Prefer nodes with more remaining CPU
        sorted_nodes = sorted(G.nodes, key=lambda n: state.node_capacity.get(n, 0.0), reverse=True)

        for node in sorted_nodes:
            avail_cpu = state.node_capacity.get(node, 0.0)
            if avail_cpu < cpu_need:
                continue

            # Anti-affinity O(1): node cannot already host this slice
            used = state.node_used_slices.get(node)
            if used is not None and slice_id in used:
                continue

            # Shallow copies are enough (flat dicts)
            new_placed = state.placed_vnfs.copy()
            new_routed = state.routed_vls.copy()
            new_node_capacity = state.node_capacity.copy()
            new_link_capacity = state.link_capacity.copy()
            new_node_used_slices = {k: v.copy() for k, v in state.node_used_slices.items()}

            new_cost = state.g_cost

            # Place
            new_node_capacity[node] = avail_cpu - cpu_need
            new_placed[next_vnf_id] = node
            new_node_used_slices.setdefault(node, set()).add(slice_id)

            # Route what becomes routable
            routing_ok = True

            for (v_from, v_to, bw) in vl_list_norm:
                if (v_from, v_to) in new_routed:
                    continue

                src_node = new_placed.get(v_from)
                dst_node = new_placed.get(v_to)
                if src_node is None or dst_node is None:
                    continue
                if src_node == dst_node:
                    continue

                path, lat = shortest_path_with_capacity(
                    G, src_node, dst_node, new_link_capacity, bw, edge_latency=edge_latency
                )
                if path is None:
                    routing_ok = False
                    break

                # Apply BW
                for u, v in zip(path[:-1], path[1:]):
                    ek = _edge_key(u, v)
                    new_link_capacity[ek] -= bw
                    if new_link_capacity[ek] < -1e-9:
                        routing_ok = False
                        break
                if not routing_ok:
                    break

                new_routed[(v_from, v_to)] = path
                new_cost += lat

            if routing_ok:
                expansions.append(
                    AStarState(
                        placed_vnfs=new_placed,
                        routed_vls=new_routed,
                        g_cost=new_cost,
                        node_capacity=new_node_capacity,
                        link_capacity=new_link_capacity,
                        node_used_slices=new_node_used_slices,
                    )
                )

        return expansions

    # ---------- Solve one slice ----------
    def solve_one_slice(vnf_chain, vl_chain):
        # Precompute per-slice structures to kill `next(...)` in hot loops
        vnf_chain_ids = [_as_vnf_id(v["id"]) for v in vnf_chain]
        vnf_cpu_by_id = { _as_vnf_id(v["id"]): float(v["cpu"]) for v in vnf_chain }
        vnf_slice_by_id = { _as_vnf_id(v["id"]): v["slice"] for v in vnf_chain }

        # Normalize VLs into a compact list [(from,to,bw), ...]
        vl_list_norm = []
        for vl in vl_chain:
            k = _vl_key(vl)
            vl_list_norm.append((k[0], k[1], float(vl["bandwidth"])))

        init_state = AStarState(
            placed_vnfs={},
            routed_vls={},
            g_cost=0.0,
            node_capacity=node_capacity_global.copy(),
            link_capacity=link_capacity_global.copy(),
            node_used_slices={},
        )

        heap = []
        counter = 0
        heappush(heap, (0.0, counter, init_state))
        best_seen = {}

        while heap:
            f, _, state = heappop(heap)

            # Optional dominance pruning
            key = (len(state.placed_vnfs), len(state.routed_vls))
            prev = best_seen.get(key)
            if prev is not None and state.g_cost >= prev:
                continue
            best_seen[key] = state.g_cost

            if state.is_goal(len(vnf_chain_ids), vl_list_norm):
                return state

            for child in expand_state(state, vnf_chain_ids, vnf_cpu_by_id, vnf_slice_by_id, vl_list_norm):
                h = heuristic_for_slice(child, vl_list_norm)
                counter += 1
                heappush(heap, (child.g_cost + h, counter, child))

        return None

    # ---------- Main loop ----------
    for slice_data in slices:
        if len(slice_data) == 2:
            vnf_chain, vl_chain = slice_data
        elif len(slice_data) == 3:
            vnf_chain, vl_chain, _entry = slice_data  # ignored in this baseline A*
        else:
            raise ValueError(f"Unexpected slice format: {slice_data}")

        result_state = solve_one_slice(vnf_chain, vl_chain)
        astar_results.append(result_state)

        if result_state is None:
            continue

        # Commit CPU
        for vnf_id, node in result_state.placed_vnfs.items():
            cpu = None
            for v in vnf_chain:
                if _as_vnf_id(v["id"]) == vnf_id:
                    cpu = float(v["cpu"])
                    break
            if cpu is None:
                raise KeyError(f"CPU not found for VNF {vnf_id}")
            node_capacity_global[node] -= cpu

        # Commit bandwidth using canonical edges
        bw_by_key = { _vl_key(vl): float(vl["bandwidth"]) for vl in vl_chain }

        for (v_from, v_to), path in result_state.routed_vls.items():
            bw = bw_by_key.get((v_from, v_to))
            if bw is None:
                raise KeyError(f"Bandwidth not found for VL ({v_from},{v_to})")

            src_node = result_state.placed_vnfs[v_from]
            dst_node = result_state.placed_vnfs[v_to]
            _assert_path_endpoints(path, src_node, dst_node)

            for u, v in zip(path[:-1], path[1:]):
                ek = _edge_key(u, v)
                link_capacity_global[ek] -= bw
                if link_capacity_global[ek] < -1e-9:
                    raise ValueError(f"Link capacity underflow on {ek}")

    df_results = pd.DataFrame(
        [{"slice": i, "accepted": (r is not None), "g_cost": (r.g_cost if r else None)}
         for i, r in enumerate(astar_results, start=1)]
    )
    if csv_path:
        df_results.to_csv(csv_path, index=False)

    return df_results, astar_results
