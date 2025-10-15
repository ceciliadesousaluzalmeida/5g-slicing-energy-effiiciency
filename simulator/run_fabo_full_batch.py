# All comments in English
import networkx as nx
import pandas as pd
from queue import PriorityQueue
from copy import deepcopy
from itertools import count


class FABOState:
    def __init__(self, placed_vnfs=None, routed_vls=None, g_cost=0.0,
                 node_capacity=None, link_capacity=None):
        self.placed_vnfs = placed_vnfs or {}
        self.routed_vls = routed_vls or {}
        self.g_cost = g_cost
        self.node_capacity = node_capacity or {}
        self.link_capacity = link_capacity or {}

    def is_goal(self, vnf_chain, vl_chain, entry=None, exit_=None):
        base_done = (len(self.placed_vnfs) == len(vnf_chain)
                     and len(self.routed_vls) >= len(vl_chain))
        if not base_done:
            return False
        if entry is None and exit_ is None:
            return True
        first_id, last_id = vnf_chain[0]["id"], vnf_chain[-1]["id"]
        return (("ENTRY", first_id) in self.routed_vls) and ((last_id, "EXIT") in self.routed_vls)

    def __lt__(self, other):
        return self.g_cost < other.g_cost


def run_fabo_full_batch(G, slices, node_capacity_base, link_latency, link_capacity_base, csv_path=None):
    # --- Helper functions -------------------------------------------------
    def get_edge_value(dct, a, b, default=0.0):
        if (a, b) in dct:
            return dct[(a, b)]
        if (b, a) in dct:
            return dct[(b, a)]
        return default

    def dec_edge_capacity(dct, a, b, bw):
        if (a, b) in dct:
            dct[(a, b)] -= bw
        elif (b, a) in dct:
            dct[(b, a)] -= bw

    def shortest_path_with_capacity(u, v, link_capacity, bw):
        try:
            path = nx.shortest_path(G, u, v, weight="latency")
        except nx.NetworkXNoPath:
            return None, None
        for x, y in zip(path[:-1], path[1:]):
            if get_edge_value(link_capacity, x, y, 0) < bw:
                return None, None
        lat = sum(get_edge_value(link_latency, x, y, 1.0) for x, y in zip(path[:-1], path[1:]))
        return path, lat

    # --- Global structures ------------------------------------------------
    fabo_results = []
    node_capacity_global = deepcopy(node_capacity_base)
    link_capacity_global = deepcopy(link_capacity_base)
    total_capacity = deepcopy(node_capacity_base)

    # --- Heuristic --------------------------------------------------------
    def fabo_heuristic(state, vnf_chain, vl_chain, entry=None, exit_=None):
        h = 0.0
        for vl in vl_chain:
            if (vl["from"], vl["to"]) in state.routed_vls:
                continue
            s_node = state.placed_vnfs.get(vl["from"])
            d_node = state.placed_vnfs.get(vl["to"])
            if s_node and d_node:
                try:
                    h += nx.shortest_path_length(G, s_node, d_node, weight="latency")
                except nx.NetworkXNoPath:
                    h += 10_000
        if entry and exit_ and vnf_chain:
            f_id, l_id = vnf_chain[0]["id"], vnf_chain[-1]["id"]
            if ("ENTRY", f_id) not in state.routed_vls and f_id in state.placed_vnfs:
                try:
                    h += nx.shortest_path_length(G, entry, state.placed_vnfs[f_id], weight="latency")
                except nx.NetworkXNoPath:
                    h += 10_000
            if (l_id, "EXIT") not in state.routed_vls and l_id in state.placed_vnfs:
                try:
                    h += nx.shortest_path_length(G, state.placed_vnfs[l_id], exit_, weight="latency")
                except nx.NetworkXNoPath:
                    h += 10_000
        return h

    # --- Expand state -----------------------------------------------------
    def expand_state(state, vnf_chain, vl_chain, entry=None, exit_=None):
        expansions = []
        unplaced = [v["id"] for v in vnf_chain if v["id"] not in state.placed_vnfs]
        if not unplaced:
            return expansions
        next_vnf_id = sorted(unplaced)[0]
        vnf = next(v for v in vnf_chain if v["id"] == next_vnf_id)

        # Node ordering by load-balance heuristic
        cpu_ratio = {n: 1.0 - (state.node_capacity.get(n, 0) / total_capacity.get(n, 1))
                     for n in G.nodes}
        candidate_nodes = sorted(G.nodes, key=lambda n: cpu_ratio[n])

        for node in candidate_nodes:
            avail = state.node_capacity.get(node, 0)
            if avail < vnf["cpu"]:
                print(f"[DEBUG][FABO] Skip {node}: need {vnf['cpu']}, have {avail}")
                continue

            # Anti-affinity
            if any(node == placed_node and vnf["slice"] == next(v2["slice"]
                   for v2 in vnf_chain if v2["id"] == pid)
                   for pid, placed_node in state.placed_vnfs.items()):
                print(f"[DEBUG][FABO] Anti-affinity: {vnf['id']} on {node} denied.")
                continue

            new_state = FABOState(
                placed_vnfs=state.placed_vnfs.copy(),
                routed_vls=state.routed_vls.copy(),
                g_cost=state.g_cost,
                node_capacity=deepcopy(state.node_capacity),
                link_capacity=deepcopy(state.link_capacity)
            )

            new_state.node_capacity[node] -= vnf["cpu"]
            new_state.placed_vnfs[next_vnf_id] = node
            print(f"[INFO][FABO] Placed {next_vnf_id} on node {node} "
                  f"(use={vnf['cpu']}, remaining={new_state.node_capacity[node]}).")

            routing_ok = True

            # Route internal VLs
            for vl in vl_chain:
                key = (vl["from"], vl["to"])
                if key in new_state.routed_vls:
                    continue
                s_node = new_state.placed_vnfs.get(vl["from"])
                d_node = new_state.placed_vnfs.get(vl["to"])
                if not s_node or not d_node:
                    continue
                path, lat = shortest_path_with_capacity(s_node, d_node, new_state.link_capacity, vl["bandwidth"])
                if not path:
                    routing_ok = False
                    print(f"[DEBUG][FABO] Routing failed for {key}.")
                    break
                for a, b in zip(path[:-1], path[1:]):
                    dec_edge_capacity(new_state.link_capacity, a, b, vl["bandwidth"])
                new_state.routed_vls[key] = path
                new_state.g_cost += lat

            if not routing_ok:
                continue

            # Entry/Exit routing
            if entry and exit_ and vnf_chain:
                f_id, l_id = vnf_chain[0]["id"], vnf_chain[-1]["id"]
                if ("ENTRY", f_id) not in new_state.routed_vls and f_id in new_state.placed_vnfs:
                    path, lat = shortest_path_with_capacity(entry, new_state.placed_vnfs[f_id],
                                                            new_state.link_capacity, vl_chain[0]["bandwidth"])
                    if path:
                        for u, v in zip(path[:-1], path[1:]):
                            dec_edge_capacity(new_state.link_capacity, u, v, vl_chain[0]["bandwidth"])
                        new_state.routed_vls[("ENTRY", f_id)] = path
                        new_state.g_cost += lat
                if (l_id, "EXIT") not in new_state.routed_vls and l_id in new_state.placed_vnfs:
                    path, lat = shortest_path_with_capacity(new_state.placed_vnfs[l_id], exit_,
                                                            new_state.link_capacity, vl_chain[-1]["bandwidth"])
                    if path:
                        for u, v in zip(path[:-1], path[1:]):
                            dec_edge_capacity(new_state.link_capacity, u, v, vl_chain[-1]["bandwidth"])
                        new_state.routed_vls[(l_id, "EXIT")] = path
                        new_state.g_cost += lat

            expansions.append(new_state)

        return expansions

    # --- Main loop per slice ---------------------------------------------
    summary = []
    for i, slice_data in enumerate(slices, start=1):
        if len(slice_data) == 2:
            vnf_chain, vl_chain = slice_data
            entry, exit_ = None, None
        else:
            vnf_chain, vl_chain, entry, exit_ = slice_data

        print(f"\n[INFO][FABO] === Solving slice {i} with {len(vnf_chain)} VNFs and {len(vl_chain)} VLs ===")

        init_state = FABOState(
            placed_vnfs={},
            routed_vls={},
            g_cost=0.0,
            node_capacity=deepcopy(node_capacity_global),
            link_capacity=deepcopy(link_capacity_global)
        )

        pq = PriorityQueue()
        counter = count()
        pq.put((0.0, next(counter), init_state))
        visited = 0

        result = None
        while not pq.empty():
            _, __, state = pq.get()
            visited += 1
            if state.is_goal(vnf_chain, vl_chain, entry, exit_):
                print(f"[INFO][FABO] Found feasible solution after {visited} states.")
                result = state
                break
            for ns in expand_state(state, vnf_chain, vl_chain, entry, exit_):
                f_score = ns.g_cost + fabo_heuristic(ns, vnf_chain, vl_chain, entry, exit_)
                pq.put((f_score, next(counter), ns))

        fabo_results.append(result)

        if result:
            # Commit global resources
            for vnf_id, node in result.placed_vnfs.items():
                cpu = next(v["cpu"] for v in vnf_chain if v["id"] == vnf_id)
                node_capacity_global[node] -= cpu
                if node_capacity_global[node] < 0:
                    print(f"[ERROR][FABO] Global CPU overcapacity on node {node}.")

            for (src, dst), path in result.routed_vls.items():
                if src == "ENTRY" or dst == "EXIT":
                    continue
                bw = next(vl["bandwidth"] for vl in vl_chain if vl["from"] == src and vl["to"] == dst)
                for u, v in zip(path[:-1], path[1:]):
                    dec_edge_capacity(link_capacity_global, u, v, bw)

            print(f"[SUMMARY][FABO] Slice {i} accepted. "
                  f"min_node_cpu={min(node_capacity_global.values())}, "
                  f"links_low_bw={sum(1 for v in link_capacity_global.values() if v <= 0)}")
        else:
            print(f"[SUMMARY][FABO] Slice {i} rejected.")

        summary.append({
            "slice": i,
            "accepted": result is not None,
            "g_cost": (result.g_cost if result else None)
        })

    df_results = pd.DataFrame(summary)
    if csv_path:
        try:
            df_results.to_csv(csv_path, index=False)
            print(f"[INFO][FABO] Results written to {csv_path}")
        except Exception as e:
            print(f"[WARN][FABO] Could not write CSV: {e}")

    return df_results, fabo_results
