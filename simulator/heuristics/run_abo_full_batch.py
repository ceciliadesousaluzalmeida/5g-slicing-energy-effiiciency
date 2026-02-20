# All comments in English
import heapq
from copy import deepcopy
from itertools import islice

import pandas as pd
import networkx as nx


class ABOState:
    def __init__(self, placed_vnfs=None, routed_vls=None, g_cost=0.0):
        self.placed_vnfs = placed_vnfs or {}
        self.routed_vls = routed_vls or {}
        self.g_cost = float(g_cost)

    def is_goal(self, vnf_chain, vl_chain, entry=None):
        if len(self.placed_vnfs) != len(vnf_chain):
            return False

        for vl in vl_chain:
            key = (vl["from"], vl["to"])
            src_node = self.placed_vnfs.get(vl["from"])
            dst_node = self.placed_vnfs.get(vl["to"])
            if src_node is None or dst_node is None:
                return False

            if src_node == dst_node:
                # Co-located: no path required
                continue

            path = self.routed_vls.get(key)
            if not path:
                return False

            # Strong endpoint consistency check
            if path[0] != src_node or path[-1] != dst_node:
                return False

        return True

    def __lt__(self, other):
        return self.g_cost < other.g_cost


def run_abo_full_batch(G, slices, node_capacity_base, link_latency, link_capacity_base, csv_path=None):
    abo_results = []

    # Global residual capacities (committed slice by slice)
    node_capacity_global = dict(node_capacity_base)
    link_capacity_global = dict(link_capacity_base)

    # ---- helpers: undirected capacity semantics --------------------------------
    def get_edge_value(dct, a, b, default=None):
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
        else:
            raise KeyError(f"Edge ({a},{b}) not found in link capacities.")

    # ---- caches ----------------------------------------------------------------
    # Cache K-shortest (by latency) simple paths in the STATIC topology
    # key: (u, v, k) -> [path1, path2, ...]
    k_paths_cache = {}

    # Cache shortest_path_length in STATIC topology
    sp_len_cache = {}

    def shortest_path_len_cached(u, v):
        if u == v:
            return 0.0
        key = (u, v)
        if key in sp_len_cache:
            return sp_len_cache[key]
        try:
            val = nx.shortest_path_length(G, u, v, weight="latency")
        except nx.NetworkXNoPath:
            val = 10_000.0
        sp_len_cache[key] = float(val)
        return float(val)

    def get_k_shortest_paths(u, v, k):
        key = (u, v, k)
        if key in k_paths_cache:
            return k_paths_cache[key]

        try:
            it = nx.shortest_simple_paths(G, u, v, weight="latency")
            paths = list(islice(it, k))
        except nx.NetworkXNoPath:
            paths = []

        k_paths_cache[key] = paths
        return paths

    # ---- helper: choose path with best residual bandwidth among k-shortest -----
    def best_bandwidth_path(u, v, link_capacity, bandwidth, k=3):
        if u is None or v is None:
            return None, None
        if u == v:
            return [u], 0.0

        paths = get_k_shortest_paths(u, v, k)
        if not paths:
            return None, None

        best_path, best_score, best_latency = None, -1.0, None

        for path in paths:
            if not path or len(path) < 2:
                continue

            # Feasibility + min residual bw
            min_bw = float("inf")
            feasible = True
            for a, b in zip(path[:-1], path[1:]):
                cap = get_edge_value(link_capacity, a, b, None)
                if cap is None or cap < bandwidth:
                    feasible = False
                    break
                if cap < min_bw:
                    min_bw = cap

            if not feasible:
                continue

            latency = 0.0
            for a, b in zip(path[:-1], path[1:]):
                latency += float(get_edge_value(link_latency, a, b, 1.0))

            if (min_bw > best_score) or (min_bw == best_score and (best_latency is None or latency < best_latency)):
                best_path, best_score, best_latency = path, float(min_bw), float(latency)

        return (best_path, best_latency) if best_path else (None, None)

    # ---- heuristic --------------------------------------------------------------
    def abo_heuristic(state, vnf_chain, vl_chain, entry=None):
        h = 0.0

        for vl in vl_chain:
            src, dst = vl["from"], vl["to"]
            if (src, dst) in state.routed_vls:
                continue

            src_node = state.placed_vnfs.get(src)
            dst_node = state.placed_vnfs.get(dst)
            if src_node is None or dst_node is None or src_node == dst_node:
                continue

            h += shortest_path_len_cached(src_node, dst_node)

        if entry is not None and vnf_chain:
            first_id = vnf_chain[0]["id"]
            if ("ENTRY", first_id) not in state.routed_vls and first_id in state.placed_vnfs:
                h += shortest_path_len_cached(entry, state.placed_vnfs[first_id])

        return float(h)

    # ---- expand state -----------------------------------------------------------
    def expand_state(state, vnf_chain, vl_chain, node_capacity, link_capacity, entry=None, k_paths=3):
        expansions = []

        unplaced = [v["id"] for v in vnf_chain if v["id"] not in state.placed_vnfs]
        if not unplaced:
            return expansions

        next_vnf = sorted(unplaced)[0]
        vnf_obj = next(v for v in vnf_chain if v["id"] == next_vnf)
        cpu_need = float(vnf_obj["cpu"])

        # Iterate nodes (you can later replace with a smarter ordering)
        for node in sorted(G.nodes):
            avail_cpu = float(node_capacity.get(node, 0.0))
            if avail_cpu < cpu_need:
                continue

            # Anti-affinity check: no two VNFs from same slice on same node
            same_slice_on_node = any(
                placed_node == node and
                vnf_obj["slice"] == next(v["slice"] for v in vnf_chain if v["id"] == other_id)
                for other_id, placed_node in state.placed_vnfs.items()
            )
            if same_slice_on_node:
                continue

            new_placed = state.placed_vnfs.copy()
            new_routed = state.routed_vls.copy()

            # Shallow copies are safe (dict of floats)
            new_node_capacity = node_capacity.copy()
            new_link_capacity = link_capacity.copy()

            new_cost = float(state.g_cost)

            new_node_capacity[node] = float(new_node_capacity.get(node, 0.0)) - cpu_need
            new_placed[next_vnf] = node

            routing_ok = True

            # Internal VL routing
            for vl in vl_chain:
                key = (vl["from"], vl["to"])
                if key in new_routed:
                    continue

                src_node = new_placed.get(vl["from"])
                dst_node = new_placed.get(vl["to"])
                if src_node is None or dst_node is None:
                    continue

                if src_node == dst_node:
                    new_routed[key] = [src_node]
                    continue

                bw = float(vl["bandwidth"])
                path, lat = best_bandwidth_path(src_node, dst_node, new_link_capacity, bw, k=k_paths)
                if not path:
                    routing_ok = False
                    break

                # Reserve bandwidth along the path
                for u, v in zip(path[:-1], path[1:]):
                    cap = get_edge_value(new_link_capacity, u, v, None)
                    if cap is None or cap < bw:
                        routing_ok = False
                        break
                    dec_edge_capacity(new_link_capacity, u, v, bw)

                if not routing_ok:
                    break

                new_routed[key] = path
                new_cost += float(lat)

            if not routing_ok:
                continue

            # Entry → first VNF (if applicable)
            if entry is not None and vnf_chain:
                first_id = vnf_chain[0]["id"]
                if ("ENTRY", first_id) not in new_routed and first_id in new_placed:
                    # Use the first VL bandwidth if you want; otherwise choose a constant or a slice param.
                    entry_bw = float(vl_chain[0]["bandwidth"]) if vl_chain else 0.0
                    if entry_bw > 0.0:
                        path, lat = best_bandwidth_path(entry, new_placed[first_id], new_link_capacity, entry_bw, k=k_paths)
                        if not path:
                            routing_ok = False
                        else:
                            for u, v in zip(path[:-1], path[1:]):
                                cap = get_edge_value(new_link_capacity, u, v, None)
                                if cap is None or cap < entry_bw:
                                    routing_ok = False
                                    break
                                dec_edge_capacity(new_link_capacity, u, v, entry_bw)

                            if routing_ok:
                                new_routed[("ENTRY", first_id)] = path
                                new_cost += float(lat)

            if routing_ok:
                expansions.append((ABOState(new_placed, new_routed, new_cost), new_node_capacity, new_link_capacity))

        return expansions

    # ---- single-slice search ----------------------------------------------------
    def run_one_slice(vnf_chain, vl_chain, entry=None, node_capacity=None, link_capacity=None, k_paths=3):
        init_state = ABOState()

        heap = []
        counter = 0
        heapq.heappush(heap, (0.0, counter, init_state, node_capacity, link_capacity))
        visited = 0

        while heap:
            _, _, state, cur_node_cap, cur_link_cap = heapq.heappop(heap)
            visited += 1

            if state.is_goal(vnf_chain, vl_chain, entry):
                print(f"[INFO][ABO] Found feasible solution after {visited} states.")
                return state, cur_node_cap, cur_link_cap

            for child_state, child_node_cap, child_link_cap in expand_state(
                state, vnf_chain, vl_chain, cur_node_cap, cur_link_cap, entry, k_paths=k_paths
            ):
                h = abo_heuristic(child_state, vnf_chain, vl_chain, entry)
                counter += 1
                heapq.heappush(heap, (child_state.g_cost + h, counter, child_state, child_node_cap, child_link_cap))

        print("[WARN][ABO] No feasible solution for this slice.")
        return None, None, None

    # ---- main loop over slices --------------------------------------------------
    for i, slice_data in enumerate(slices, start=1):
        if len(slice_data) == 2:
            vnf_chain, vl_chain = slice_data
            entry = None
        elif len(slice_data) == 3:
            vnf_chain, vl_chain, entry = slice_data
        else:
            raise ValueError(f"Unexpected slice format: {slice_data}")

        print(f"\n[INFO][ABO] === Solving slice {i} ({len(vnf_chain)} VNFs, {len(vl_chain)} VLs) ===")

        # Start slice search from current global residuals (shallow copy is enough)
        node_capacity = node_capacity_global.copy()
        link_capacity = link_capacity_global.copy()

        # You can tune k_paths here
        result, final_node_cap, final_link_cap = run_one_slice(
            vnf_chain, vl_chain, entry, node_capacity, link_capacity, k_paths=3
        )
        abo_results.append(result)

        if result:
            # Commit resources globally using the final residuals returned by the search
            node_capacity_global = final_node_cap
            link_capacity_global = final_link_cap

            print(
                f"[SUMMARY][ABO] Slice {i} accepted. "
                f"min_node_cpu={min(node_capacity_global.values())}, "
                f"links_low_bw={sum(1 for v in link_capacity_global.values() if v <= 0)}"
            )
        else:
            print(f"[SUMMARY][ABO] Slice {i} rejected.")

    # ---- summary DataFrame ------------------------------------------------------
    summary = [{"slice": i + 1, "accepted": r is not None, "g_cost": (r.g_cost if r else None)}
               for i, r in enumerate(abo_results)]
    df_results = pd.DataFrame(summary)

    if csv_path:
        try:
            df_results.to_csv(csv_path, index=False)
            print(f"[INFO][ABO] Results written to {csv_path}")
        except Exception as e:
            print(f"[WARN][ABO] Could not write CSV: {e}")

    return df_results, abo_results
