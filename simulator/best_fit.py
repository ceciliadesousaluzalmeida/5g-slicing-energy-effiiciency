import networkx as nx
import pandas as pd
from copy import deepcopy

class BFState:
    def __init__(self, placed_vnfs=None, routed_vls=None, g_cost=0):
        self.placed_vnfs = placed_vnfs or {}
        self.routed_vls = routed_vls or {}
        self.g_cost = g_cost

    def is_goal(self, vnf_chain, vl_chain):
        return len(self.placed_vnfs) == len(vnf_chain) and len(self.routed_vls) == len(vl_chain)


def run_best_fit(G, slices, node_capacity_base, link_capacity_base, link_latency, csv_path=None):
    results = []
    full_results = []

    node_capacity_global = deepcopy(node_capacity_base)
    link_capacity_global = deepcopy(link_capacity_base)

    for i, (vnf_chain, vl_chain) in enumerate(slices, start=1):
        print(f"\n[INFO][BF] === Solving slice {i} with {len(vnf_chain)} VNFs and {len(vl_chain)} VLs ===")
        placed_vnfs = {}
        routed_vls = {}
        g_cost = 0
        success = True

        local_node_capacity = deepcopy(node_capacity_global)
        local_link_capacity = deepcopy(link_capacity_global)

        for vnf in vnf_chain:
            vnf_id = vnf["id"]
            cpu_needed = vnf["cpu"]
            slice_id = vnf["slice"]

            # Candidate nodes: sorted by "best fit" (lowest remaining capacity after placement)
            candidate_nodes = sorted(
                [n for n in G.nodes if local_node_capacity.get(n, 0) >= cpu_needed],
                key=lambda n: (local_node_capacity[n] - cpu_needed, n)
            )

            placed = False
            for node in candidate_nodes:
                # Anti-affinity check
                if any(
                    placed_node == node and slice_id == other_vnf["slice"]
                    for other_id, placed_node in placed_vnfs.items()
                    for other_vnf in vnf_chain if other_vnf["id"] == other_id
                ):
                    print(f"[DEBUG][BF] Anti-affinity: {vnf_id} cannot go on node {node}.")
                    continue

                # Tentative placement
                temp_placed = placed_vnfs.copy()
                temp_placed[vnf_id] = node
                temp_routed = routed_vls.copy()
                temp_link_capacity = deepcopy(local_link_capacity)
                temp_g_cost = g_cost
                routing_ok = True

                # Try to route all feasible VLs
                for vl in vl_chain:
                    src, dst = vl["from"], vl["to"]
                    if src in temp_placed and dst in temp_placed and (src, dst) not in temp_routed:
                        src_node = temp_placed[src]
                        dst_node = temp_placed[dst]
                        try:
                            path = nx.shortest_path(G, src_node, dst_node, weight="latency")
                            latency = sum(link_latency.get((u, v), link_latency.get((v, u), 9999))
                                          for u, v in zip(path[:-1], path[1:]))
                            if latency > vl["latency"]:
                                print(f"[DEBUG][BF] Latency SLA failed for VL {src}->{dst}: "
                                      f"{latency} > {vl['latency']}")
                                routing_ok = False
                                break
                            for u, v in zip(path[:-1], path[1:]):
                                cap = temp_link_capacity.get((u, v), temp_link_capacity.get((v, u), 0))
                                if cap < vl["bandwidth"]:
                                    print(f"[DEBUG][BF] Bandwidth SLA failed for VL {src}->{dst}.")
                                    routing_ok = False
                                    break
                            if not routing_ok:
                                break
                            for u, v in zip(path[:-1], path[1:]):
                                if (u, v) in temp_link_capacity:
                                    temp_link_capacity[(u, v)] -= vl["bandwidth"]
                                else:
                                    temp_link_capacity[(v, u)] -= vl["bandwidth"]
                            temp_routed[(src, dst)] = path
                            temp_g_cost += latency
                        except nx.NetworkXNoPath:
                            print(f"[DEBUG][BF] No path for VL {src}->{dst}.")
                            routing_ok = False
                            break

                if routing_ok:
                    placed_vnfs = temp_placed
                    routed_vls = temp_routed
                    local_link_capacity = temp_link_capacity
                    local_node_capacity[node] -= cpu_needed
                    g_cost = temp_g_cost
                    placed = True
                    print(f"[INFO][BF] Placed {vnf_id} on node {node} "
                          f"(use={cpu_needed}, remaining={local_node_capacity[node]}).")
                    break

            if not placed:
                print(f"[WARN][BF] Failed to place VNF {vnf_id}, slice {i} rejected.")
                success = False
                break

        if success:
            # Commit resources globally
            node_capacity_global = local_node_capacity
            link_capacity_global = local_link_capacity
            results.append({"slice": i, "accepted": True, "g_cost": g_cost})
            full_results.append(BFState(placed_vnfs, routed_vls, g_cost))
            print(f"[SUMMARY][BF] Slice {i} accepted. "
                  f"Remaining min_node_cpu={min(node_capacity_global.values())}, "
                  f"links_low_bw={sum(1 for v in link_capacity_global.values() if v <= 0)}")
        else:
            results.append({"slice": i, "accepted": False, "g_cost": None})
            full_results.append(None)
            print(f"[SUMMARY][BF] Slice {i} rejected.")

    df = pd.DataFrame(results)
    if csv_path:
        try:
            df.to_csv(csv_path, index=False)
            print(f"[INFO][BF] Results written to {csv_path}")
        except Exception as e:
            print(f"[WARN][BF] Could not write CSV: {e}")

    return df, full_results
