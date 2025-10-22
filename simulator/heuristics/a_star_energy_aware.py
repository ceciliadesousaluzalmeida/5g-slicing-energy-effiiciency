# energy_aware_astar_mininfra.py
from copy import deepcopy
import pandas as pd
import networkx as nx
from queue import PriorityQueue

class AStarInfraState:
    def __init__(self, placed_vnfs=None, routed_vls=None, g_cost=0.0,
                 node_capacity=None, link_capacity=None):
        self.placed_vnfs   = placed_vnfs or {}
        self.routed_vls    = routed_vls or {}
        self.g_cost        = float(g_cost)
        self.node_capacity = node_capacity or {}
        self.link_capacity = link_capacity or {}

    def is_goal(self, vnf_chain, vl_chain, entry=None):
        return (len(self.placed_vnfs) == len(vnf_chain) and
                len([1 for vl in vl_chain if (vl["from"], vl["to"]) in self.routed_vls]) == len(vl_chain) and
                (entry is None or ("ENTRY", vnf_chain[0]["id"]) in self.routed_vls))

    def __lt__(self, other):
        return self.g_cost < other.g_cost


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


def shortest_path_with_capacity(G, u, v, link_capacity, bandwidth):
    """Return shortest path (min hops) respecting capacity."""
    try:
        path = nx.shortest_path(G, u, v)  # purely topological (fewest hops)
    except nx.NetworkXNoPath:
        return None

    for a, b in zip(path[:-1], path[1:]):
        cap = _get_link_cap(link_capacity, a, b)
        if cap is None or cap < bandwidth:
            return None
    return path


def energy_aware_astar(G, slices, node_capacity_base, link_capacity_base,
                                w_nodes=1.0, w_links=1.0, csv_path=None):
    """
    Infrastructure-efficient A*: minimize number of active nodes and links.
    Cost function:
        f = w_nodes * N_active + w_links * L_active
    Same structure and logs as your other A* variants.
    """
    astar_results = []
    node_capacity_global = deepcopy(node_capacity_base)
    link_capacity_global = deepcopy(link_capacity_base)

    def count_active_nodes(state):
        return len(set(state.placed_vnfs.values()))

    def count_active_links(state):
        used_links = set()
        for path in state.routed_vls.values():
            for u, v in zip(path[:-1], path[1:]):
                used_links.add(tuple(sorted((u, v))))
        return len(used_links)

    def expand_state(state, vnf_chain, vl_chain, entry=None):
        expansions = []
        unplaced_ids = [v["id"] for v in vnf_chain if v["id"] not in state.placed_vnfs]
        if not unplaced_ids:
            return expansions

        next_vnf_id = sorted(unplaced_ids)[0]
        vnf_obj = next(v for v in vnf_chain if v["id"] == next_vnf_id)
        sorted_nodes = sorted(G.nodes, key=lambda n: state.node_capacity.get(n, 0), reverse=True)

        for node in sorted_nodes:
            avail_cpu = state.node_capacity.get(node, 0)
            cpu_need = vnf_obj["cpu"]
            if avail_cpu < cpu_need:
                continue

            # avoid placing two VNFs of same slice on same node
            same_slice_on_node = any(
                next(v for v in vnf_chain if v["id"] == pid)["slice"] == vnf_obj["slice"]
                for pid, n_ in state.placed_vnfs.items() if n_ == node
            )
            if same_slice_on_node:
                continue

            new_placed = state.placed_vnfs.copy()
            new_routed = state.routed_vls.copy()
            new_node_cap = deepcopy(state.node_capacity)
            new_link_cap = deepcopy(state.link_capacity)
            new_cost = state.g_cost

            new_node_cap[node] -= cpu_need
            new_placed[next_vnf_id] = node

            routing_ok = True
            for vl in vl_chain:
                key = (vl["from"], vl["to"])
                if key in new_routed:
                    continue
                src_node = new_placed.get(vl["from"])
                dst_node = new_placed.get(vl["to"])
                if src_node is None or dst_node is None:
                    continue
                path = shortest_path_with_capacity(G, src_node, dst_node, new_link_cap, vl["bandwidth"])
                if path is None:
                    routing_ok = False
                    break
                for u, v in zip(path[:-1], path[1:]):
                    _dec_link_cap(new_link_cap, u, v, vl["bandwidth"])
                new_routed[key] = path

            if routing_ok and entry is not None and vnf_chain:
                first_id = vnf_chain[0]["id"]
                if ("ENTRY", first_id) not in new_routed:
                    first_node = new_placed.get(first_id)
                    if first_node is not None:
                        bw_first = vl_chain[0]["bandwidth"] if vl_chain else 0.0
                        path = shortest_path_with_capacity(G, entry, first_node, new_link_cap, bw_first)
                        if path is None:
                            routing_ok = False
                        else:
                            for u, v in zip(path[:-1], path[1:]):
                                _dec_link_cap(new_link_cap, u, v, bw_first)
                            new_routed[("ENTRY", first_id)] = path

            if routing_ok:
                used_nodes = count_active_nodes(state)
                used_links = count_active_links(state)
                new_cost = w_nodes * used_nodes + w_links * used_links
                expansions.append(AStarInfraState(new_placed, new_routed, new_cost, new_node_cap, new_link_cap))
        return expansions

    def solve_one_slice(vnf_chain, vl_chain, entry=None):
        init_state = AStarInfraState({}, {}, 0.0, deepcopy(node_capacity_global), deepcopy(link_capacity_global))
        pq = PriorityQueue()
        counter = 0
        pq.put((0.0, counter, init_state))
        visited = 0

        while not pq.empty():
            _, _, state = pq.get()
            visited += 1
            if state.is_goal(vnf_chain, vl_chain, entry):
                print(f"[INFO][A*-Infra] Solution found after {visited} expansions. "
                      f"(nodes={count_active_nodes(state)}, links={count_active_links(state)}, cost={state.g_cost:.3f})")
                return state
            for child in expand_state(state, vnf_chain, vl_chain, entry):
                counter += 1
                pq.put((child.g_cost, counter, child))
        print(f"[WARN][A*-Infra] No feasible solution for slice after {visited} expansions.")
        return None

    for i, slice_data in enumerate(slices, start=1):
        if len(slice_data) == 2:
            vnf_chain, vl_chain = slice_data
            entry = None
        elif len(slice_data) == 3:
            vnf_chain, vl_chain, entry = slice_data
        else:
            raise ValueError(f"Unexpected slice format: {slice_data}")

        print(f"\n[INFO][A*-Infra] === Solving slice {i} (VNFs={len(vnf_chain)}, VLs={len(vl_chain)}) ===")
        result_state = solve_one_slice(vnf_chain, vl_chain, entry)
        astar_results.append(result_state)

        if result_state is None:
            print(f"[WARN][A*-Infra] Slice {i} rejected.\n")
            continue

        # Commit global capacities
        for vnf_id, node in result_state.placed_vnfs.items():
            vnf_cpu = next(v["cpu"] for v in vnf_chain if v["id"] == vnf_id)
            node_capacity_global[node] -= vnf_cpu

        for (src, dst), path in result_state.routed_vls.items():
            if src == "ENTRY":
                continue
            bw = next(vl["bandwidth"] for vl in vl_chain if vl["from"] == src and vl["to"] == dst)
            for u, v in zip(path[:-1], path[1:]):
                _dec_link_cap(link_capacity_global, u, v, bw)

        print(f"[SUMMARY][A*-Infra] Slice {i} accepted. "
              f"(nodes={count_active_nodes(result_state)}, links={count_active_links(result_state)}, cost={result_state.g_cost:.3f})\n")

    summary = [{
        "slice": idx,
        "accepted": res is not None,
        "g_cost": (res.g_cost if res else None),
        "nodes_used": (len(set(res.placed_vnfs.values())) if res else None),
        "links_used": (len({(u, v_) for p in (res.routed_vls.values() if res else []) for u, v_ in zip(p[:-1], p[1:])}) if res else None)
    } for idx, res in enumerate(astar_results, start=1)]

    df_results = pd.DataFrame(summary)
    if csv_path:
        df_results.to_csv(csv_path, index=False)
    return df_results, astar_results
