import pandas as pd

class FFState:
    def __init__(self, placed_vnfs=None, g_cost=0):
        self.placed_vnfs = placed_vnfs or {}
        self.g_cost = g_cost

    def is_goal(self, vnf_chain):
        return len(self.placed_vnfs) == len(vnf_chain)

def run_first_fit(G, slices, node_capacity_base):
    ff_results = []
    node_capacity = node_capacity_base.copy()  

    for i, (vnf_chain, _) in enumerate(slices):
        placed_vnfs = {}
        success = True

        for vnf in vnf_chain:
            vnf_id = vnf["id"]
            cpu_req = vnf["cpu"]
            allocated = False

            for node in G.nodes:
                if node_capacity[node] >= cpu_req:
                    placed_vnfs[vnf_id] = node
                    node_capacity[node] -= cpu_req
                    allocated = True
                    break

            if not allocated:
                success = False
                break

        if success:
            result = FFState(placed_vnfs=placed_vnfs, g_cost=0)
        else:
            result = None

        ff_results.append(result)

    summary = [{"slice": i+1, "accepted": r is not None, "g_cost": r.g_cost if r else None} for i, r in enumerate(ff_results)]
    df_results = pd.DataFrame(summary)

    return df_results, ff_results, node_capacity  