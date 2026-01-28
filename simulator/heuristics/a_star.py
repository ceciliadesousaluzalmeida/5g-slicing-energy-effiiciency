from copy import deepcopy
from queue import PriorityQueue

import networkx as nx
import pandas as pd


# ------------------------- helpers: normalization -------------------------
def _as_vnf_id(v):
    # Always normalize VNF identifiers to string
    return str(v).strip()


def _vl_key_from_to(v_from, v_to):
    # Normalize (from,to) key used across the whole pipeline
    return (_as_vnf_id(v_from), _as_vnf_id(v_to))


def _vl_key(vl):
    # Use normalized (from,to) as unique key
    return _vl_key_from_to(vl["from"], vl["to"])


# ------------------------- helpers: capacities -------------------------
def _get_link_cap(capdict, u, v):
    if (u, v) in capdict:
        return capdict[(u, v)]
    if (v, u) in capdict:
        return capdict[(v, u)]
    return None


def _dec_link_cap(capdict, u, v, amount):
    if (u, v) in capdict:
        capdict[(u, v)] -= amount
    elif (v, u) in capdict:
        capdict[(v, u)] -= amount
    else:
        raise KeyError(f"Edge ({u},{v}) not found in link capacities.")


def _assert_path_endpoints(path, src_node, dst_node):
    # Ensure path endpoints match the placed nodes
    if not path:
        raise ValueError("VL path is empty.")
    if path[0] != src_node or path[-1] != dst_node:
        raise ValueError(
            f"VL path endpoints do not match: expected ({src_node}->{dst_node}), "
            f"got ({path[0]}->{path[-1]})"
        )


# ------------------------- helper: feasible shortest path -------------------------
def shortest_path_with_capacity(G, u, v, link_capacity, bandwidth):
    if u == v:
        return [u], 0.0

    try:
        path = nx.shortest_path(G, u, v, weight="latency")
    except nx.NetworkXNoPath:
        return None, None

    for a, b in zip(path[:-1], path[1:]):
        cap = _get_link_cap(link_capacity, a, b)
        if cap is None or cap < bandwidth:
            return None, None

    latency = sum(G[a][b].get("latency", 1.0) for a, b in zip(path[:-1], path[1:]))
    return path, float(latency)


# ------------------------- A* DATA STRUCTURES -------------------------
class AStarState:
    def __init__(self, placed_vnfs=None, routed_vls=None, g_cost=0.0,
                 node_capacity=None, link_capacity=None):
        self.placed_vnfs = placed_vnfs or {}      # {vnf_id(str): node(int)}
        self.routed_vls  = routed_vls or {}       # {(from_id,to_id): [path nodes]}
        self.g_cost      = float(g_cost)
        self.node_capacity = node_capacity or {}
        self.link_capacity = link_capacity or {}

    def is_goal(self, vnf_chain, vl_chain, entry=None):
        # Goal: all VNFs placed + all VLs routed with correct endpoints
        if len(self.placed_vnfs) != len(vnf_chain):
            return False

        for vl in vl_chain:
            k = _vl_key(vl)
            v_from, v_to = k

            src_node = self.placed_vnfs.get(v_from)
            dst_node = self.placed_vnfs.get(v_to)
            if src_node is None or dst_node is None:
                return False
            if src_node == dst_node:
                continue

            if k not in self.routed_vls:
                return False

            path = self.routed_vls[k]
            if not path:
                return False
            if path[0] != src_node or path[-1] != dst_node:
                return False

        return True

    def __lt__(self, other):
        return self.g_cost < other.g_cost


# ---------------------------- A* ORCHESTRATOR ----------------------------
def run_astar(G, slices, node_capacity_base, link_capacity_base, csv_path=None):
    astar_results = []
    node_capacity_global = deepcopy(node_capacity_base)
    link_capacity_global = deepcopy(link_capacity_base)

    # ---------- Heuristic ----------
    def heuristic_for_slice(state, vnf_chain, vl_chain, entry=None):
        h = 0.0
        for vl in vl_chain:
            k = _vl_key(vl)
            if k in state.routed_vls:
                continue

            v_from, v_to = k
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
    def expand_state(state, vnf_chain, vl_chain, entry=None):
        expansions = []

        unplaced_ids = [_as_vnf_id(v["id"]) for v in vnf_chain if _as_vnf_id(v["id"]) not in state.placed_vnfs]
        if not unplaced_ids:
            return expansions

        next_vnf_id = sorted(unplaced_ids)[0]
        vnf_obj = next(v for v in vnf_chain if _as_vnf_id(v["id"]) == next_vnf_id)

        sorted_nodes = sorted(G.nodes, key=lambda n: state.node_capacity.get(n, 0), reverse=True)

        for node in sorted_nodes:
            avail_cpu = state.node_capacity.get(node, 0)
            cpu_need = vnf_obj["cpu"]
            if avail_cpu < cpu_need:
                continue

            # Anti-affinity: no two VNFs of same slice on same node
            same_slice_on_node = any(
                next(v for v in vnf_chain if _as_vnf_id(v["id"]) == pid)["slice"] == vnf_obj["slice"]
                for pid, n_ in state.placed_vnfs.items() if n_ == node
            )
            if same_slice_on_node:
                continue

            new_placed = state.placed_vnfs.copy()
            new_routed = state.routed_vls.copy()
            new_node_capacity = deepcopy(state.node_capacity)
            new_link_capacity = deepcopy(state.link_capacity)
            new_cost = float(state.g_cost)

            new_node_capacity[node] -= cpu_need
            new_placed[next_vnf_id] = node

            routing_ok = True

            for vl in vl_chain:
                k = _vl_key(vl)
                if k in new_routed:
                    continue

                v_from, v_to = k
                src_node = new_placed.get(v_from)
                dst_node = new_placed.get(v_to)
                if src_node is None or dst_node is None:
                    continue
                if src_node == dst_node:
                    # No routing needed for co-located VNFs
                    continue

                path, lat = shortest_path_with_capacity(G, src_node, dst_node, new_link_capacity, vl["bandwidth"])
                if path is None:
                    routing_ok = False
                    break

                _assert_path_endpoints(path, src_node, dst_node)

                for u, v in zip(path[:-1], path[1:]):
                    _dec_link_cap(new_link_capacity, u, v, vl["bandwidth"])

                # Store ONLY the path for compatibility with validator + mip-start
                new_routed[k] = path
                new_cost += float(lat)

            if routing_ok:
                expansions.append(
                    AStarState(new_placed, new_routed, new_cost, new_node_capacity, new_link_capacity)
                )

        return expansions

    # ---------- Solve one slice ----------
    def solve_one_slice(vnf_chain, vl_chain, entry=None):
        init_state = AStarState(
            {}, {}, 0.0,
            deepcopy(node_capacity_global),
            deepcopy(link_capacity_global)
        )

        pq = PriorityQueue()
        counter = 0
        pq.put((0.0, counter, init_state))
        visited = 0

        while not pq.empty():
            _, _, state = pq.get()
            visited += 1

            if state.is_goal(vnf_chain, vl_chain, entry):
                return state

            for child in expand_state(state, vnf_chain, vl_chain, entry):
                h = heuristic_for_slice(child, vnf_chain, vl_chain, entry)
                counter += 1
                pq.put((child.g_cost + h, counter, child))

        return None

    # ---------- Main loop ----------
    for i, slice_data in enumerate(slices, start=1):
        if len(slice_data) == 2:
            vnf_chain, vl_chain = slice_data
            entry = None
        elif len(slice_data) == 3:
            vnf_chain, vl_chain, entry = slice_data
        else:
            raise ValueError(f"Unexpected slice format: {slice_data}")

        result_state = solve_one_slice(vnf_chain, vl_chain, entry)
        astar_results.append(result_state)

        if result_state is None:
            continue

        # Commit CPU
        for vnf_id, node in result_state.placed_vnfs.items():
            vnf_cpu = next(v["cpu"] for v in vnf_chain if _as_vnf_id(v["id"]) == vnf_id)
            node_capacity_global[node] -= vnf_cpu

        # Commit bandwidth using stored paths
        for (v_from, v_to), path in result_state.routed_vls.items():
            # Find VL bandwidth in vl_chain by normalized ids
            bw = None
            for vl in vl_chain:
                k = _vl_key(vl)
                if k == (v_from, v_to):
                    bw = vl["bandwidth"]
                    break
            if bw is None:
                raise KeyError(f"Bandwidth not found for VL ({v_from},{v_to})")

            src_node = result_state.placed_vnfs[v_from]
            dst_node = result_state.placed_vnfs[v_to]
            _assert_path_endpoints(path, src_node, dst_node)

            for u, v in zip(path[:-1], path[1:]):
                _dec_link_cap(link_capacity_global, u, v, bw)

    summary = [
        {"slice": idx, "accepted": (res is not None), "g_cost": (res.g_cost if res else None)}
        for idx, res in enumerate(astar_results, start=1)
    ]
    df_results = pd.DataFrame(summary)
    if csv_path:
        df_results.to_csv(csv_path, index=False)

    return df_results, astar_results
