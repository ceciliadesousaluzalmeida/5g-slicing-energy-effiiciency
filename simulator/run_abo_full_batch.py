# All comments in English
from queue import PriorityQueue
import pandas as pd
import networkx as nx
from copy import deepcopy


class ABOState:
    def __init__(self, placed_vnfs=None, routed_vls=None, g_cost=0.0):
        self.placed_vnfs = placed_vnfs or {}
        self.routed_vls = routed_vls or {}
        self.g_cost = g_cost

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


def run_abo_full_batch(G, slices, node_capacity_base, link_latency, link_capacity_base, csv_path=None):
    abo_results = []
    node_capacity_global = deepcopy(node_capacity_base)
    link_capacity_global = deepcopy(link_capacity_base)

    # ---- helper: feasible shortest path with capacity ----
    def shortest_path_with_capacity(G, u, v, link_capacity, bandwidth):
        try:
            path = nx.shortest_path(G, u, v, weight="latency")
        except nx.NetworkXNoPath:
            return None, None
        for a, b in zip(path[:-1], path[1:]):
            cap = link_capacity.get((a, b), link_capacity.get((b, a), 0))
            if cap < bandwidth:
                return None, None
        latency = sum(link_latency.get((a, b), link_latency.get((b, a), 1.0))
                      for a, b in zip(path[:-1], path[1:]))
        return path, latency

    # ---- heuristic (similar to A*) ----
    def abo_heuristic(state, vnf_chain, vl_chain, entry=None, exit_=None):
        h = 0.0
        for vl in vl_chain:
            src, dst = vl["from"], vl["to"]
            if (src, dst) in state.routed_vls:
                continue
            src_node = state.placed_vnfs.get(src)
            dst_node = state.placed_vnfs.get(dst)
            if src_node is not None and dst_node is not None:
                try:
                    h += nx.shortest_path_length(G, src_node, dst_node, weight="latency")
                except nx.NetworkXNoPath:
                    h += 10_000
        if entry is not None and exit_ is not None and vnf_chain:
            first_id, last_id = vnf_chain[0]["id"], vnf_chain[-1]["id"]
            if ("ENTRY", first_id) not in state.routed_vls and first_id in state.placed_vnfs:
                try:
                    h += nx.shortest_path_length(G, entry, state.placed_vnfs[first_id], weight="latency")
                except nx.NetworkXNoPath:
                    h += 10_000
            if (last_id, "EXIT") not in state.routed_vls and last_id in state.placed_vnfs:
                try:
                    h += nx.shortest_path_length(G, state.placed_vnfs[last_id], exit_, weight="latency")
                except nx.NetworkXNoPath:
                    h += 10_000
        return h

    # ---- expand state ----
    def expand_state(state, vnf_chain, vl_chain, node_capacity, link_capacity, entry=None, exit_=None):
        expansions = []
        unplaced = [v["id"] for v in vnf_chain if v["id"] not in state.placed_vnfs]
        if not unplaced:
            return expansions
        next_vnf = sorted(unplaced)[0]
        vnf_obj = next(v for v in vnf_chain if v["id"] == next_vnf)

        for node in sorted(G.nodes):
            avail_cpu = node_capacity.get(node, 0)
            cpu_need = vnf_obj["cpu"]

            if avail_cpu < cpu_need:
                print(f"[DEBUG][ABO] Skip node {node} for {vnf_obj['id']}: required={cpu_need}, available={avail_cpu}")
                continue

            # Anti-affinity check
            if any(placed_node == node and vnf_obj["slice"] == next(v["slice"] for v in vnf_chain if v["id"] == other_id)
                   for other_id, placed_node in state.placed_vnfs.items()):
                print(f"[DEBUG][ABO] Anti-affinity: {vnf_obj['id']} cannot go on node {node}.")
                continue

            new_placed = state.placed_vnfs.copy()
            new_routed = state.routed_vls.copy()
            new_node_capacity = deepcopy(node_capacity)
            new_link_capacity = deepcopy(link_capacity)
            new_cost = state.g_cost

            new_node_capacity[node] -= cpu_need
            new_placed[next_vnf] = node
            print(f"[INFO][ABO] Placed {next_vnf} on node {node} "
                  f"(use={cpu_need}, remaining={new_node_capacity[node]}).")

            routing_ok = True
            # Route internal VLs
            for vl in vl_chain:
                key = (vl["from"], vl["to"])
                if key in new_routed:
                    continue
                src_node = new_placed.get(vl["from"])
                dst_node = new_placed.get(vl["to"])
                if src_node is None or dst_node is None:
                    continue
                path, lat = shortest_path_with_capacity(G, src_node, dst_node, new_link_capacity, vl["bandwidth"])
                if path is None:
                    print(f"[DEBUG][ABO] No feasible path for VL {key}.")
                    routing_ok = False
                    break
                for u, v in zip(path[:-1], path[1:]):
                    bw = vl["bandwidth"]
                    if (u, v) in new_link_capacity:
                        new_link_capacity[(u, v)] -= bw
                    else:
                        new_link_capacity[(v, u)] -= bw
                new_routed[key] = path
                new_cost += lat

            if not routing_ok:
                print(f"[DEBUG][ABO] Discarding state after {next_vnf} on node {node} (routing failed).")
                continue

            # Entry/Exit routing
            if entry is not None and exit_ is not None and vnf_chain:
                first_id, last_id = vnf_chain[0]["id"], vnf_chain[-1]["id"]

                # ENTRY → first
                if ("ENTRY", first_id) not in new_routed and first_id in new_placed:
                    path, lat = shortest_path_with_capacity(G, entry, new_placed[first_id],
                                                            new_link_capacity, vl_chain[0]["bandwidth"])
                    if path:
                        for u, v in zip(path[:-1], path[1:]):
                            bw = vl_chain[0]["bandwidth"]
                            if (u, v) in new_link_capacity:
                                new_link_capacity[(u, v)] -= bw
                            else:
                                new_link_capacity[(v, u)] -= bw
                        new_routed[("ENTRY", first_id)] = path
                        new_cost += lat
                    else:
                        routing_ok = False

                if not routing_ok:
                    continue

                # last → EXIT
                if (last_id, "EXIT") not in new_routed and last_id in new_placed:
                    path, lat = shortest_path_with_capacity(G, new_placed[last_id], exit_,
                                                            new_link_capacity, vl_chain[-1]["bandwidth"])
                    if path:
                        for u, v in zip(path[:-1], path[1:]):
                            bw = vl_chain[-1]["bandwidth"]
                            if (u, v) in new_link_capacity:
                                new_link_capacity[(u, v)] -= bw
                            else:
                                new_link_capacity[(v, u)] -= bw
                        new_routed[(last_id, "EXIT")] = path
                        new_cost += lat
                    else:
                        routing_ok = False

                if not routing_ok:
                    print(f"[DEBUG][ABO] Failed ENTRY/EXIT routing for state {next_vnf} on {node}.")
                    continue

            expansions.append(ABOState(new_placed, new_routed, new_cost))

        return expansions

    # ---- single-slice search ----
    def run_one_slice(vnf_chain, vl_chain, entry=None, exit_=None, node_capacity=None, link_capacity=None):
        init_state = ABOState()
        pq = PriorityQueue()
        counter = 0
        pq.put((0.0, counter, init_state))
        visited = 0

        while not pq.empty():
            _, _, state = pq.get()
            visited += 1
            if state.is_goal(vnf_chain, vl_chain, entry, exit_):
                print(f"[INFO][ABO] Found feasible solution after {visited} states.")
                return state
            for child in expand_state(state, vnf_chain, vl_chain, node_capacity, link_capacity, entry, exit_):
                h = abo_heuristic(child, vnf_chain, vl_chain, entry, exit_)
                counter += 1
                pq.put((child.g_cost + h, counter, child))
        print("[WARN][ABO] No feasible solution for this slice.")
        return None

    # ---- main loop over slices ----
    for i, slice_data in enumerate(slices, start=1):
        if len(slice_data) == 2:
            vnf_chain, vl_chain = slice_data
            entry, exit_ = None, None
        else:
            vnf_chain, vl_chain, entry, exit_ = slice_data

        print(f"\n[INFO][ABO] === Solving slice {i} with {len(vnf_chain)} VNFs and {len(vl_chain)} VLs ===")
        node_capacity = deepcopy(node_capacity_global)
        link_capacity = deepcopy(link_capacity_global)
        result = run_one_slice(vnf_chain, vl_chain, entry, exit_, node_capacity, link_capacity)
        abo_results.append(result)

        if result:
            # Commit resources globally
            for vnf_id, node in result.placed_vnfs.items():
                cpu_needed = next(v["cpu"] for v in vnf_chain if v["id"] == vnf_id)
                node_capacity_global[node] -= cpu_needed
                if node_capacity_global[node] < 0:
                    print(f"[ERROR][ABO] Global overcapacity on node {node}.")
            for (src, dst), path in result.routed_vls.items():
                if src == "ENTRY" or dst == "EXIT":
                    continue
                bw = next(vl["bandwidth"] for vl in vl_chain if vl["from"] == src and vl["to"] == dst)
                for u, v in zip(path[:-1], path[1:]):
                    if (u, v) in link_capacity_global:
                        link_capacity_global[(u, v)] -= bw
                    elif (v, u) in link_capacity_global:
                        link_capacity_global[(v, u)] -= bw

            print(f"[SUMMARY][ABO] Slice {i} accepted. "
                  f"min_node_cpu={min(node_capacity_global.values())}, "
                  f"links_low_bw={sum(1 for v in link_capacity_global.values() if v <= 0)}")
        else:
            print(f"[SUMMARY][ABO] Slice {i} rejected.")

    # ---- final dataframe ----
    summary = [{"slice": i+1, "accepted": r is not None, "g_cost": r.g_cost if r else None}
               for i, r in enumerate(abo_results)]
    df_results = pd.DataFrame(summary)
    if csv_path:
        try:
            df_results.to_csv(csv_path, index=False)
            print(f"[INFO][ABO] Results written to {csv_path}")
        except Exception as e:
            print(f"[WARN][ABO] Could not write CSV: {e}")

    return df_results, abo_results
