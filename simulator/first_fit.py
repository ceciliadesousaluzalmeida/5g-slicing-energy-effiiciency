import pandas as pd
import networkx as nx

class FFState:
    def __init__(self, placed_vnfs=None, routed_vls=None, g_cost=0):
        self.placed_vnfs = placed_vnfs or {}
        self.routed_vls = routed_vls or {}
        self.g_cost = g_cost

    def is_goal(self, vnf_chain):
        return len(self.placed_vnfs) == len(vnf_chain)

def run_first_fit(G, slices, node_capacity_base):
    ff_results = []
    node_capacity = node_capacity_base.copy()
    sorted_nodes = sorted(G.nodes)

    for i, (vnf_chain, vl_chain) in enumerate(slices):
        placed_vnfs = {}
        success = True

        # First Fit placement
        for vnf in vnf_chain:
            vnf_id = vnf["id"]
            cpu_req = vnf["cpu"]
            allocated = False

            for node in sorted_nodes:
                if node_capacity[node] >= cpu_req:
                    placed_vnfs[vnf_id] = node
                    node_capacity[node] -= cpu_req
                    allocated = True
                    break

            if not allocated:
                success = False
                break

        # Minimum latency routing (shortest path by latency)
        if success:
            routed_vls = {}
            total_latency = 0

            for vl in vl_chain:
                src, dst = vl["from"], vl["to"]
                src_node = placed_vnfs[src]
                dst_node = placed_vnfs[dst]

                try:
                    path = nx.shortest_path(G, src_node, dst_node, weight="latency")
                    latency = sum(G[u][v]["latency"] for u, v in zip(path[:-1], path[1:]))
                    routed_vls[(src, dst)] = path
                    total_latency += latency
                except nx.NetworkXNoPath:
                    print(f"[FF] No path between nodes {src_node} and {dst_node} — slice {i+1}")
                    success = False
                    break

        if success:
            result = FFState(
                placed_vnfs=placed_vnfs,
                routed_vls=routed_vls,
                g_cost=total_latency
            )
        else:
            result = None

        ff_results.append(result)

    summary = [{"slice": i + 1, "accepted": r is not None, "g_cost": r.g_cost if r else None} for i, r in enumerate(ff_results)]
    df_results = pd.DataFrame(summary)

    return df_results, ff_results, node_capacity
