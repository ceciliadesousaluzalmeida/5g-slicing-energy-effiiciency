import networkx as nx
import pandas as pd
from copy import deepcopy

def run_best_fit(G, slices, node_capacity_base, link_capacity_base, link_latency, csv_path=None):
    results = []
    full_results = []

    node_capacity_global = deepcopy(node_capacity_base)
    link_capacity_global = deepcopy(link_capacity_base)

    for i, (vnf_chain, vl_chain) in enumerate(slices):
        print(f"\n🔄 Processing Slice {i + 1}")
        placed_vnfs = {}
        routed_vls = {}
        g_cost = 0
        success = True

        for vnf in vnf_chain:
            vnf_id = vnf["id"]
            cpu_needed = vnf["cpu"]
            slice_id = vnf["slice"]

            candidate_nodes = sorted(
                [n for n in G.nodes if node_capacity_global[n] >= cpu_needed],
                key=lambda n: (node_capacity_global[n] - cpu_needed, n)
            )

            placed = False
            for node in candidate_nodes:
                if any(
                    placed_node == node and slice_id == other_vnf["slice"]
                    for other_id, placed_node in placed_vnfs.items()
                    for other_vnf in vnf_chain if other_vnf["id"] == other_id
                ):
                    continue

                temp_placed = placed_vnfs.copy()
                temp_placed[vnf_id] = node
                temp_routed = routed_vls.copy()
                temp_link_capacity = deepcopy(link_capacity_global)
                temp_g_cost = g_cost
                routing_ok = True

                for vl in vl_chain:
                    src, dst = vl["from"], vl["to"]
                    if src in temp_placed and dst in temp_placed and (src, dst) not in temp_routed:
                        src_node = temp_placed[src]
                        dst_node = temp_placed[dst]

                        try:
                            path = nx.shortest_path(G, src_node, dst_node, weight="latency")
                            latency = sum(link_latency.get((u, v), link_latency.get((v, u), 0))
                                          for u, v in zip(path[:-1], path[1:]))
                            if latency > vl["latency"]:
                                routing_ok = False
                                break
                            for u, v in zip(path[:-1], path[1:]):
                                cap = temp_link_capacity.get((u, v), temp_link_capacity.get((v, u), 0))
                                if cap < vl["bandwidth"]:
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
                            routing_ok = False
                            break

                if routing_ok:
                    placed_vnfs = temp_placed
                    routed_vls = temp_routed
                    link_capacity_global = temp_link_capacity
                    node_capacity_global[node] -= cpu_needed
                    g_cost = temp_g_cost
                    placed = True
                    break

            if not placed:
                print(f"❌ Failed to place VNF {vnf_id}")
                success = False
                break

        results.append({
            "slice": i + 1,
            "accepted": success,
            "g_cost": g_cost if success else None
        })

        if success:
            full_results.append(
                type("BestFitResult", (), {
                    "placed_vnfs": placed_vnfs,
                    "routed_vls": routed_vls,
                    "g_cost": g_cost
                })()
            )
        else:
            full_results.append(None)

    df = pd.DataFrame(results)
    if csv_path:
        df.to_csv(csv_path, index=False)

    return df, full_results
