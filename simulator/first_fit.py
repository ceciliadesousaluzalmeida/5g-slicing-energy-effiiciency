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

    for i, (vnf_chain, vl_chain) in enumerate(slices):
        print(f"🔄 Starting Slice {i+1}")
        placed_vnfs = {}
        routed_vls = {}
        success = True

        for vnf in vnf_chain:
            vnf_id = vnf["id"]
            placed = False

            for node in sorted(G.nodes):
                if node_capacity_global[node] < vnf["cpu"]:
                    continue

                if any(node == placed_node and vnf["slice"] == other_vnf["slice"]
                       for other_vnf_id, placed_node in placed_vnfs.items()
                       for other_vnf in vnf_chain if other_vnf["id"] == other_vnf_id):
                    continue

                placed_vnfs[vnf_id] = node
                node_capacity_global[node] -= vnf["cpu"]
                placed = True
                print(f"✅ Placed {vnf_id} on Node {node}")
                break

            if not placed:
                print(f"❌ Could not place {vnf_id}")
                success = False
                break

            for vl in vl_chain:
                src, dst = vl["from"], vl["to"]
                if src in placed_vnfs and dst in placed_vnfs and (src, dst) not in routed_vls:
                    src_node = placed_vnfs[src]
                    dst_node = placed_vnfs[dst]
                    try:
                        path = nx.shortest_path(G, src_node, dst_node, weight="latency")
                        latency = sum(link_latency.get((path[i], path[i+1]), 9999) for i in range(len(path)-1))
                        if latency > vl["latency"]:
                            print(f"❌ Latency too high for VL {src}->{dst}")
                            success = False
                            break
                        if any(link_capacity_global.get((path[i], path[i+1]), 0) < vl["bandwidth"] for i in range(len(path)-1)):
                            print(f"❌ Not enough bandwidth for VL {src}->{dst}")
                            success = False
                            break
                        for i in range(len(path)-1):
                            link_capacity_global[(path[i], path[i+1])] -= vl["bandwidth"]
                        routed_vls[(src, dst)] = path
                        print(f"→ Routed VL {src}->{dst} via {path}")
                    except nx.NetworkXNoPath:
                        print(f"❌ No path for VL {src}->{dst}")
                        success = False
                        break

            if not success:
                break

        if success:
            results.append(FFState(placed_vnfs, routed_vls))
        else:
            results.append(None)

    summary = [{
        "slice": i+1,
        "accepted": r is not None
    } for i, r in enumerate(results)]
    df = pd.DataFrame(summary)
    if csv_path:
        df.to_csv(csv_path, index=False)
    return df, results
