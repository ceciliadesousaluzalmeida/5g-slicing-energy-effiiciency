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

    def is_goal(self, vnf_chain, vl_chain, entry=None):
        return (len(self.placed_vnfs) == len(vnf_chain)
                and len(self.routed_vls) >= len(vl_chain))

    def __lt__(self, other):
        return self.g_cost < other.g_cost


def run_fabo_full_batch(G, slices, node_capacity_base, link_latency, link_capacity_base, csv_path=None):
    # --- Helper functions -------------------------------------------------
    def get_edge_value(dct, a, b, default=0.0):
        return dct.get((a, b), dct.get((b, a), default))

    def dec_edge_capacity(dct, a, b, bw):
        if (a, b) in dct:
            dct[(a, b)] -= bw
        elif (b, a) in dct:
            dct[(b, a)] -= bw

    # --- Best bandwidth path ----------------------------------------------
    def best_bandwidth_path(u, v, link_capacity, bw, k=3):
        """
        Returns feasible path with maximum min-bandwidth among k shortest (by latency).
        Includes safety checks for disconnected or trivial paths.
        """
        # Safety check
        if u is None or v is None or u == v:
            return None, None

        try:
            paths = list(nx.shortest_simple_paths(G, u, v, weight="latency"))
        except nx.NetworkXNoPath:
            return None, None

        best_path, best_score, best_latency = None, -1, None

        for path in paths[:k]:
            # Skip trivial or invalid paths
            if len(path) < 2:
                continue

            # Get bandwidths safely
            bw_values = [
                get_edge_value(link_capacity, a, b, 0)
                for a, b in zip(path[:-1], path[1:])
            ]
            if not bw_values:  # empty list
                continue

            min_bw = min(bw_values)
            if min_bw < bw:
                continue

            latency = sum(
                get_edge_value(link_latency, a, b, 1.0)
                for a, b in zip(path[:-1], path[1:])
            )

            if min_bw > best_score or (min_bw == best_score and (best_latency is None or latency < best_latency)):
                best_path, best_score, best_latency = path, min_bw, latency

        return (best_path, best_latency) if best_path else (None, None)


    # --- Global structures ------------------------------------------------
    fabo_results = []
    node_capacity_global = deepcopy(node_capacity_base)
    link_capacity_global = deepcopy(link_capacity_base)
    total_capacity = deepcopy(node_capacity_base)

    # --- Heuristic --------------------------------------------------------
    def fabo_heuristic(state, vnf_chain, vl_chain, entry=None):
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
        if entry and vnf_chain:
            f_id = vnf_chain[0]["id"]
            if ("ENTRY", f_id) not in state.routed_vls and f_id in state.placed_vnfs:
                try:
                    h += nx.shortest_path_length(G, entry, state.placed_vnfs[f_id], weight="latency")
                except nx.NetworkXNoPath:
                    h += 10_000
        return h

    # --- Expand state -----------------------------------------------------
    def expand_state(state, vnf_chain, vl_chain, entry=None):
        expansions = []
        unplaced = [v["id"] for v in vnf_chain if v["id"] not in state.placed_vnfs]
        if not unplaced:
            return expansions
        next_vnf_id = sorted(unplaced)[0]
        vnf = next(v for v in vnf_chain if v["id"] == next_vnf_id)

        # Node ordering by load-balance (prefer less loaded nodes)
        cpu_ratio = {n: 1.0 - (state.node_capacity.get(n, 0) / total_capacity.get(n, 1))
                     for n in G.nodes}
        candidate_nodes = sorted(G.nodes, key=lambda n: cpu_ratio[n])

        for node in candidate_nodes:
            avail = state.node_capacity.get(node, 0)
            if avail < vnf["cpu"]:
                continue

            # Anti-affinity
            if any(node == placed_node and vnf["slice"] == next(v2["slice"]
                   for v2 in vnf_chain if v2["id"] == pid)
                   for pid, placed_node in state.placed_vnfs.items()):
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

            routing_ok = True

            # Route internal VLs using best-bandwidth path
            for vl in vl_chain:
                key = (vl["from"], vl["to"])
                if key in new_state.routed_vls:
                    continue
                s_node = new_state.placed_vnfs.get(vl["from"])
                d_node = new_state.placed_vnfs.get(vl["to"])
                if not s_node or not d_node:
                    continue
                path, lat = best_bandwidth_path(s_node, d_node, new_state.link_capacity, vl["bandwidth"])
                if not path:
                    routing_ok = False
                    break
                for a, b in zip(path[:-1], path[1:]):
                    dec_edge_capacity(new_state.link_capacity, a, b, vl["bandwidth"])
                new_state.routed_vls[key] = path
                new_state.g_cost += lat

            if not routing_ok:
                continue

            # Entry → first VNF
            if entry and vnf_chain:
                f_id = vnf_chain[0]["id"]
                if ("ENTRY", f_id) not in new_state.routed_vls and f_id in new_state.placed_vnfs:
                    path, lat = best_bandwidth_path(entry, new_state.placed_vnfs[f_id],
                                                    new_state.link_capacity,
                                                    vl_chain[0]["bandwidth"] if vl_chain else 0)
                    if path:
                        for u, v in zip(path[:-1], path[1:]):
                            dec_edge_capacity(new_state.link_capacity, u, v,
                                              vl_chain[0]["bandwidth"] if vl_chain else 0)
                        new_state.routed_vls[("ENTRY", f_id)] = path
                        new_state.g_cost += lat
                    else:
                        routing_ok = False

            if routing_ok:
                expansions.append(new_state)

        return expansions

    # --- Main loop per slice ---------------------------------------------
    summary = []
    for i, slice_data in enumerate(slices, start=1):
        if len(slice_data) == 2:
            vnf_chain, vl_chain = slice_data
            entry = None
        elif len(slice_data) == 3:
            vnf_chain, vl_chain, entry = slice_data
        else:
            raise ValueError(f"Unexpected slice format: {slice_data}")

        print(f"\n[INFO][FABO] === Solving slice {i} ({len(vnf_chain)} VNFs, {len(vl_chain)} VLs) ===")

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
            if state.is_goal(vnf_chain, vl_chain, entry):
                print(f"[INFO][FABO] Found feasible solution after {visited} states.")
                result = state
                break
            for ns in expand_state(state, vnf_chain, vl_chain, entry):
                f_score = ns.g_cost + fabo_heuristic(ns, vnf_chain, vl_chain, entry)
                pq.put((f_score, next(counter), ns))

        fabo_results.append(result)

        if result:
            # Commit global resources
            for vnf_id, node in result.placed_vnfs.items():
                cpu = next(v["cpu"] for v in vnf_chain if v["id"] == vnf_id)
                node_capacity_global[node] -= cpu
            for (src, dst), path in result.routed_vls.items():
                if src == "ENTRY":
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
