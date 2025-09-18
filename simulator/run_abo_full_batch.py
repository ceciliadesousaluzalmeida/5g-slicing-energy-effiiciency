from queue import PriorityQueue
import pandas as pd
import networkx as nx
from copy import deepcopy

class ABOState:
    def __init__(self, placed_vnfs=None, routed_vls=None, g_cost=0):
        self.placed_vnfs = placed_vnfs or {}
        self.routed_vls = routed_vls or {}
        self.g_cost = g_cost

    def is_goal(self, vnf_chain, vl_chain):
        return (len(self.placed_vnfs) == len(vnf_chain)
                and len(self.routed_vls) == len(vl_chain))

    def __lt__(self, other):
        return self.g_cost < other.g_cost


def run_abo_full_batch(G, slices, node_capacity_base, link_latency, link_capacity_base, csv_path=None):
    abo_results = []
    node_capacity_global = deepcopy(node_capacity_base)
    link_capacity_global = deepcopy(link_capacity_base)

    def abo_heuristic(state, vnf_chain, vl_chain):
        """Estimate remaining latency cost for unrouted VLs."""
        h = 0
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
        return h

    def expand_state(state, vnf_chain, vl_chain, node_capacity, link_capacity):
        expansions = []
        unplaced = [v["id"] for v in vnf_chain if v["id"] not in state.placed_vnfs]
        if not unplaced:
            return expansions
        next_vnf = sorted(unplaced)[0]
        vnf_obj = next(v for v in vnf_chain if v["id"] == next_vnf)

        for node in sorted(G.nodes):
            avail_cpu = node_capacity.get(node, 0)
            cpu_need = vnf_obj["cpu"]

            # CPU check
            if avail_cpu < cpu_need:
                print(f"[DEBUG][ABO] Skip node {node} for {vnf_obj['id']}: "
                      f"required={cpu_need}, available={avail_cpu}")
                continue

            # Anti-affinity: same slice not allowed on same node
            already_on_node = any(
                placed_node == node and vnf_obj["slice"] == placed_vnf["slice"]
                for placed_vnf_id, placed_node in state.placed_vnfs.items()
                for placed_vnf in vnf_chain if placed_vnf["id"] == placed_vnf_id
            )
            if already_on_node:
                print(f"[DEBUG][ABO] Anti-affinity: {vnf_obj['id']} cannot go on node {node}.")
                continue

            new_placed = state.placed_vnfs.copy()
            new_routed = state.routed_vls.copy()
            new_node_capacity = deepcopy(node_capacity)
            new_link_capacity = deepcopy(link_capacity)
            new_cost = state.g_cost

            # Place VNF
            new_node_capacity[node] -= cpu_need
            if new_node_capacity[node] < 0:
                print(f"[ERROR][ABO] Negative CPU on node {node} after placing {next_vnf}.")
                continue
            new_placed[next_vnf] = node
            print(f"[INFO][ABO] Placed {next_vnf} on node {node} "
                  f"(use={cpu_need}, remaining={new_node_capacity[node]}).")

            # Try to route new VLs
            routing_ok = True
            for vl in vl_chain:
                key = (vl["from"], vl["to"])
                if key in new_routed:
                    continue
                src_node = new_placed.get(vl["from"])
                dst_node = new_placed.get(vl["to"])
                if src_node is None or dst_node is None:
                    continue
                try:
                    path = nx.shortest_path(G, src_node, dst_node, weight="latency")
                    path_latency = sum(link_latency.get((path[i], path[i+1]), 9999)
                                       for i in range(len(path)-1))
                    if path_latency > vl["latency"]:
                        print(f"[DEBUG][ABO] Latency constraint failed for VL {key}: "
                              f"{path_latency} > {vl['latency']}")
                        routing_ok = False
                        break
                    if any(new_link_capacity.get((path[i], path[i+1]), 0) < vl["bandwidth"]
                           and new_link_capacity.get((path[i+1], path[i]), 0) < vl["bandwidth"]
                           for i in range(len(path)-1)):
                        print(f"[DEBUG][ABO] Bandwidth constraint failed for VL {key}.")
                        routing_ok = False
                        break
                    # Deduct bandwidth
                    for u, v in zip(path[:-1], path[1:]):
                        if (u, v) in new_link_capacity:
                            new_link_capacity[(u, v)] -= vl["bandwidth"]
                        elif (v, u) in new_link_capacity:
                            new_link_capacity[(v, u)] -= vl["bandwidth"]
                    new_routed[key] = path
                    new_cost += vl["bandwidth"] + path_latency
                except nx.NetworkXNoPath:
                    print(f"[DEBUG][ABO] No path for VL {key} between {src_node} and {dst_node}.")
                    routing_ok = False
                    break

            if routing_ok:
                expansions.append(ABOState(new_placed, new_routed, new_cost))
            else:
                print(f"[DEBUG][ABO] Discard state after placing {next_vnf} on node {node}.")

        return expansions

    def run_one_slice(vnf_chain, vl_chain):
        init_state = ABOState()
        queue = PriorityQueue()
        counter = 0
        queue.put((0, counter, init_state))
        visited = 0

        while not queue.empty():
            _, _, state = queue.get()
            visited += 1
            if state.is_goal(vnf_chain, vl_chain):
                print(f"[INFO][ABO] Found solution after expanding {visited} states.")
                return state
            for child in expand_state(state, vnf_chain, vl_chain, node_capacity, link_capacity):
                h = abo_heuristic(child, vnf_chain, vl_chain)
                counter += 1
                queue.put((child.g_cost + h, counter, child))
        print("[WARN][ABO] No feasible solution for slice.")
        return None

    # ---- Main loop over slices ----
    for i, (vnf_chain, vl_chain) in enumerate(slices, start=1):
        print(f"\n[INFO][ABO] === Solving slice {i} with {len(vnf_chain)} VNFs and {len(vl_chain)} VLs ===")
        node_capacity = deepcopy(node_capacity_global)
        link_capacity = deepcopy(link_capacity_global)
        result = run_one_slice(vnf_chain, vl_chain)
        abo_results.append(result)

        if result:
            # Commit resources globally
            for vnf_id, node in result.placed_vnfs.items():
                cpu_needed = next(v["cpu"] for v in vnf_chain if v["id"] == vnf_id)
                node_capacity_global[node] -= cpu_needed
                if node_capacity_global[node] < 0:
                    print(f"[ERROR][ABO] Global overcapacity on node {node} after committing {vnf_id}.")
            for (src, dst), path in result.routed_vls.items():
                bw = next(vl["bandwidth"] for vl in vl_chain if vl["from"] == src and vl["to"] == dst)
                for u, v in zip(path[:-1], path[1:]):
                    if (u, v) in link_capacity_global:
                        link_capacity_global[(u, v)] -= bw
                    elif (v, u) in link_capacity_global:
                        link_capacity_global[(v, u)] -= bw

            print(f"[SUMMARY][ABO] After slice {i}: "
                  f"min_node_cpu={min(node_capacity_global.values())}, "
                  f"links_low_bw={sum(1 for v in link_capacity_global.values() if v <= 0)}")

    # Final summary DataFrame
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
