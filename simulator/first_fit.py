import networkx as nx
from copy import deepcopy
import pandas as pd

class FFState:
    def __init__(self, placed_vnfs=None, routed_vls=None):
        self.placed_vnfs = placed_vnfs or {}
        self.routed_vls = routed_vls or {}

    def is_goal(self, vnf_chain, vl_chain):
        return len(self.placed_vnfs) == len(vnf_chain) and len(self.routed_vls) == len(vl_chain)


def run_first_fit(G, slices, node_capacity_base, link_capacity_base, link_latency, csv_path=None):
    results = []
    node_capacity_global = deepcopy(node_capacity_base)
    link_capacity_global = deepcopy(link_capacity_base)

    for i, (vnf_chain, vl_chain) in enumerate(slices, start=1):
        print(f"\n[INFO][FF] === Solving slice {i} with {len(vnf_chain)} VNFs and {len(vl_chain)} VLs ===")
        placed_vnfs = {}
        routed_vls = {}

        # Work on local copies first
        local_node_capacity = deepcopy(node_capacity_global)
        local_link_capacity = deepcopy(link_capacity_global)
        success = True

        for vnf in vnf_chain:
            vnf_id = vnf["id"]
            vnf_cpu = vnf["cpu"]
            placed = False

            for node in sorted(G.nodes):
                avail_cpu = local_node_capacity.get(node, 0)

                # CPU check
                if avail_cpu < vnf_cpu:
                    print(f"[DEBUG][FF] Skip node {node} for {vnf_id}: required={vnf_cpu}, available={avail_cpu}")
                    continue

                # Anti-affinity
                if any(node == placed_node and vnf["slice"] == other_vnf["slice"]
                       for other_vnf_id, placed_node in placed_vnfs.items()
                       for other_vnf in vnf_chain if other_vnf["id"] == other_vnf_id):
                    print(f"[DEBUG][FF] Anti-affinity: {vnf_id} cannot be placed on node {node}.")
                    continue

                # Place VNF
                placed_vnfs[vnf_id] = node
                local_node_capacity[node] -= vnf_cpu
                placed = True
                print(f"[INFO][FF] Placed {vnf_id} on node {node} "
                      f"(use={vnf_cpu}, remaining={local_node_capacity[node]}).")
                break

            if not placed:
                print(f"[WARN][FF] Could not place {vnf_id}, slice {i} rejected.")
                success = False
                break

            # Try to route VLs made feasible by this placement
            for vl in vl_chain:
                src, dst = vl["from"], vl["to"]
                if src in placed_vnfs and dst in placed_vnfs and (src, dst) not in routed_vls:
                    src_node = placed_vnfs[src]
                    dst_node = placed_vnfs[dst]
                    try:
                        path = nx.shortest_path(G, src_node, dst_node, weight="latency")
                        total_latency = sum(link_latency.get((path[j], path[j+1]), 9999) for j in range(len(path)-1))

                        if total_latency > vl["latency"]:
                            print(f"[DEBUG][FF] Latency SLA failed for VL {src}->{dst}: "
                                  f"{total_latency} > {vl['latency']}")
                            success = False
                            break

                        if any(local_link_capacity.get((path[j], path[j+1]), 0) < vl["bandwidth"] and
                               local_link_capacity.get((path[j+1], path[j]), 0) < vl["bandwidth"]
                               for j in range(len(path)-1)):
                            print(f"[DEBUG][FF] Bandwidth SLA failed for VL {src}->{dst}.")
                            success = False
                            break

                        # Deduct bandwidth locally
                        for j in range(len(path)-1):
                            u, v = path[j], path[j+1]
                            if (u, v) in local_link_capacity:
                                local_link_capacity[(u, v)] -= vl["bandwidth"]
                            elif (v, u) in local_link_capacity:
                                local_link_capacity[(v, u)] -= vl["bandwidth"]

                        routed_vls[(src, dst)] = path
                        print(f"[INFO][FF] Routed VL {src}->{dst} via {path} "
                              f"(latency={total_latency}, bw={vl['bandwidth']}).")

                    except nx.NetworkXNoPath:
                        print(f"[DEBUG][FF] No path available for VL {src}->{dst}.")
                        success = False
                        break

            if not success:
                break

        if success:
            # Commit resources globally
            node_capacity_global = local_node_capacity
            link_capacity_global = local_link_capacity
            results.append(FFState(placed_vnfs, routed_vls))
            print(f"[SUMMARY][FF] Slice {i} accepted. "
                  f"Remaining min_node_cpu={min(node_capacity_global.values())}, "
                  f"links_low_bw={sum(1 for v in link_capacity_global.values() if v <= 0)}")
        else:
            results.append(None)
            print(f"[SUMMARY][FF] Slice {i} rejected.")

    # Build summary table
    summary = [{
        "slice": i,
        "accepted": r is not None
    } for i, r in enumerate(results, start=1)]
    df = pd.DataFrame(summary)

    if csv_path:
        try:
            df.to_csv(csv_path, index=False)
            print(f"[INFO][FF] Results written to {csv_path}")
        except Exception as e:
            print(f"[WARN][FF] Could not write CSV: {e}")

    return df, results
