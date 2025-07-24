from queue import PriorityQueue
import pandas as pd
import networkx as nx
from tqdm import tqdm

class FABOState:
    def __init__(self, placed_vnfs=None, routed_vls=None, g_cost=0, node_capacity=None, link_capacity=None):
        self.placed_vnfs = placed_vnfs or {}
        self.routed_vls = routed_vls or {}
        self.g_cost = g_cost
        self.node_capacity = node_capacity or {}
        self.link_capacity = link_capacity or {}

    def is_goal(self, vnf_chain, vl_chain):
        return len(self.placed_vnfs) == len(vnf_chain) and len(self.routed_vls) == len(vl_chain)

    def __lt__(self, other):
        return self.g_cost < other.g_cost

def run_fabo_full_batch(G, slices, node_capacity_base, link_latency, link_capacity_base, csv_path=None):
    fabo_results = []
    total_capacity = node_capacity_base.copy()

    for i, (vnf_chain, vl_chain) in enumerate(tqdm(slices, desc="Running FABO", unit="slice")):
        if all(node_capacity_base[n] < min(v["cpu"] for v in vnf_chain) for n in G.nodes):
            print(f"[Warning] Slice {i+1} skipped: no node has enough CPU to host the smallest VNF.")
            fabo_results.append(None)
            continue

        def fabo_heuristic(state):
            return sum(vl["bandwidth"] for vl in vl_chain if (vl["from"], vl["to"]) not in state.routed_vls)

        def expand_state(state):
            expansions = []
            next_vnf = next((v["id"] for v in vnf_chain if v["id"] not in state.placed_vnfs), None)
            if next_vnf is None:
                return expansions

            vnf_obj = next(v for v in vnf_chain if v["id"] == next_vnf)

            cpu_used_ratio = {
                n: 1 - state.node_capacity[n] / total_capacity[n] if total_capacity[n] > 0 else 1
                for n in G.nodes
            }
            candidate_nodes = sorted(G.nodes, key=lambda n: cpu_used_ratio[n])[:5]

            for node in candidate_nodes:
                if state.node_capacity[node] < vnf_obj["cpu"]:
                    continue

                already_on_node = any(
                    placed_node == node and
                    vnf_obj["slice"] == placed_vnf["slice"]
                    for placed_vnf_id, placed_node in state.placed_vnfs.items()
                    for placed_vnf in vnf_chain
                    if placed_vnf["id"] == placed_vnf_id
                )
                if already_on_node:
                    continue

                new_placed = state.placed_vnfs.copy()
                new_placed[next_vnf] = node
                new_routed = state.routed_vls.copy()
                new_node_capacity = state.node_capacity.copy()
                new_link_capacity = state.link_capacity.copy()
                g_cost = state.g_cost

                for vl in vl_chain:
                    src, dst = vl["from"], vl["to"]
                    if src in new_placed and dst in new_placed and (src, dst) not in new_routed:
                        src_node = new_placed[src]
                        dst_node = new_placed[dst]
                        try:
                            path = nx.shortest_path(G, src_node, dst_node, weight="latency")
                            path_latency = sum(link_latency[(path[i], path[i+1])] for i in range(len(path)-1))
                            if path_latency > vl["latency"]:
                                continue
                            for i in range(len(path)-1):
                                u, v = path[i], path[i+1]
                                if new_link_capacity[(u, v)] < vl["bandwidth"]:
                                    break
                            else:
                                penalty = sum(
                                    1 + (1 - new_link_capacity[(path[i], path[i+1])] / link_capacity_base[(path[i], path[i+1])])
                                    for i in range(len(path)-1)
                                )
                                for i in range(len(path)-1):
                                    new_link_capacity[(path[i], path[i+1])] -= vl["bandwidth"]
                                new_routed[(src, dst)] = path
                                g_cost += vl["bandwidth"] * penalty
                        except nx.NetworkXNoPath:
                            continue

                new_node_capacity[node] -= vnf_obj["cpu"]
                new_state = FABOState(
                    placed_vnfs=new_placed,
                    routed_vls=new_routed,
                    g_cost=g_cost,
                    node_capacity=new_node_capacity,
                    link_capacity=new_link_capacity
                )
                expansions.append(new_state)

            return expansions

        def run():
            initial_state = FABOState(
                placed_vnfs={},
                routed_vls={},
                g_cost=0,
                node_capacity=node_capacity_base.copy(),
                link_capacity=link_capacity_base.copy()
            )
            queue = PriorityQueue()
            queue.put((0, initial_state))

            while not queue.empty():
                _, state = queue.get()
                if state.is_goal(vnf_chain, vl_chain):
                    return state
                for new_state in expand_state(state):
                    f_score = new_state.g_cost + fabo_heuristic(new_state)
                    queue.put((f_score, new_state))
            return None

        result = run()
        fabo_results.append(result)

    summary = [{"slice": i+1, "accepted": r is not None, "g_cost": r.g_cost if r else None} for i, r in enumerate(fabo_results)]
    df_results = pd.DataFrame(summary)

    if csv_path:
        df_results.to_csv(csv_path, index=False)

    return df_results, fabo_results
