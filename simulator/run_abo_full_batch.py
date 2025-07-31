from queue import PriorityQueue
import pandas as pd
import networkx as nx

class ABOState:
    def __init__(self, placed_vnfs=None, routed_vls=None, g_cost=0):
        self.placed_vnfs = placed_vnfs or {}
        self.routed_vls = routed_vls or {}
        self.g_cost = g_cost

    def is_goal(self, vnf_chain, vl_chain):
        return len(self.placed_vnfs) == len(vnf_chain) and len(self.routed_vls) == len(vl_chain)

    def __lt__(self, other):
        return self.g_cost < other.g_cost

def run_abo_full_batch(G, slices, node_capacity_base, link_latency, link_capacity_base, csv_path=None):
    abo_results = []

    for i, (vnf_chain, vl_chain) in enumerate(slices):
        node_capacity = node_capacity_base.copy()
        link_capacity = link_capacity_base.copy()

        def abo_heuristic(state):
                total_estimated_latency = 0
                for vl in vl_chain:
                    src = vl["from"]
                    dst = vl["to"]
                    if (src, dst) in state.routed_vls:
                        continue
                    src_node = state.placed_vnfs.get(src)
                    dst_node = state.placed_vnfs.get(dst)
                    if src_node is not None and dst_node is not None:
                        try:
                            total_estimated_latency += nx.shortest_path_length(G, src_node, dst_node, weight="latency")
                        except nx.NetworkXNoPath:
                            total_estimated_latency += 9999
                return total_estimated_latency

        def expand_state(state):
            expansions = []
            next_vnf = next((v["id"] for v in vnf_chain if v["id"] not in state.placed_vnfs), None)
            if next_vnf is None:
                print("✅ All VNFs have been placed.")
                return expansions

            vnf_obj = next(v for v in vnf_chain if v["id"] == next_vnf)

            for node in G.nodes:
                print(f"\n🔍 Trying to place {next_vnf} on Node {node}")

                if node_capacity[node] < vnf_obj["cpu"]:
                    print("❌ Not enough CPU")
                    continue

                already_on_node = any(
                    placed_node == node and
                    vnf_obj["slice"] == placed_vnf["slice"]
                    for placed_vnf_id, placed_node in state.placed_vnfs.items()
                    for placed_vnf in vnf_chain
                    if placed_vnf["id"] == placed_vnf_id
                )
                if already_on_node:
                    print("❌ A VNF from the same slice is already placed on this node")
                    continue

                new_placed = state.placed_vnfs.copy()
                new_placed[next_vnf] = node
                new_routed = state.routed_vls.copy()
                g_cost = state.g_cost
                routing_success = True

                for vl in vl_chain:
                    src, dst = vl["from"], vl["to"]
                    if src in new_placed and dst in new_placed and (src, dst) not in new_routed:
                        src_node = new_placed[src]
                        dst_node = new_placed[dst]

                        try:
                            path = nx.shortest_path(G, src_node, dst_node, weight="latency")
                            path_latency = sum(link_latency.get((path[i], path[i+1]), 9999) for i in range(len(path)-1))
                            print(f"→ Routing VL {src} → {dst}: path {path}, latency = {path_latency:.2f} ms (max allowed: {vl['latency']} ms)")

                            if path_latency > vl["latency"]:
                                print("❌ Latency constraint violated")
                                routing_success = False
                                break

                            if any(link_capacity.get((path[i], path[i+1]), 0) < vl["bandwidth"] for i in range(len(path)-1)):
                                print("❌ Insufficient bandwidth on path")
                                routing_success = False
                                break

                            for i in range(len(path)-1):
                                link_capacity[(path[i], path[i+1])] -= vl["bandwidth"]

                            new_routed[(src, dst)] = path
                            g_cost += vl["bandwidth"]

                        except nx.NetworkXNoPath:
                            print(f"❌ No path found between {src_node} and {dst_node}")
                            routing_success = False
                            break

                if routing_success:
                    print(f"✅ Successfully placed {next_vnf} on Node {node} and routed all links")
                    new_state = ABOState(new_placed, new_routed, g_cost)
                    expansions.append(new_state)

            print(f"🌱 {len(expansions)} states generated from {next_vnf}")
            return expansions

        def run():
            initial_state = ABOState()
            queue = PriorityQueue()
            queue.put((0, initial_state))

            while not queue.empty():
                _, state = queue.get()
                if state.is_goal(vnf_chain, vl_chain):
                    return state
                for new_state in expand_state(state):
                    f_score = new_state.g_cost + abo_heuristic(new_state)
                    queue.put((f_score, new_state))

            print(f"[Slice {i+1}] ❌ Rejected: no valid placement found.")
            print("Remaining node capacities:")
            for n in G.nodes:
                print(f"  Node {n}: {node_capacity[n]} CPU units available")

            print("Remaining link capacities:")
            for (u, v) in G.edges:
                cap = link_capacity.get((u, v), 'N/A')
                print(f"  Link ({u}, {v}): {cap} Mbps available")
            return None

        result = run()
        abo_results.append(result)

    summary = [{"slice": i+1, "accepted": r is not None, "g_cost": r.g_cost if r else None} for i, r in enumerate(abo_results)]
    df_results = pd.DataFrame(summary)

    if csv_path:
        df_results.to_csv(csv_path, index=False)

    return df_results, abo_results
