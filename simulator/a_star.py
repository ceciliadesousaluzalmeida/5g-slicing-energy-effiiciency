from queue import PriorityQueue
import pandas as pd
import networkx as nx

class AStarState:
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

def run_astar_full_batch(G, slices, node_capacity_base, link_latency, link_capacity_base, csv_path=None):
    astar_results = []

    for i, (vnf_chain, vl_chain) in enumerate(slices):
        def astar_heuristic(state):
            return sum(
                vl["bandwidth"]
                for vl in vl_chain
                if (vl["from"], vl["to"]) not in state.routed_vls
            )

        def expand_state(state):
            expansions = []
            next_vnf = next((v["id"] for v in vnf_chain if v["id"] not in state.placed_vnfs), None)
            if next_vnf is None:
                return expansions

            vnf_obj = next(v for v in vnf_chain if v["id"] == next_vnf)

            for node in G.nodes:
                if state.node_capacity[node] >= vnf_obj["cpu"]:
                    new_placed = state.placed_vnfs.copy()
                    new_placed[next_vnf] = node
                    new_routed = state.routed_vls.copy()
                    new_node_capacity = state.node_capacity.copy()
                    new_link_capacity = state.link_capacity.copy()
                    g_cost = state.g_cost

                    # Attempt to route virtual links
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
                                if all(new_link_capacity[(path[i], path[i+1])] >= vl["bandwidth"] for i in range(len(path)-1)):
                                    for i in range(len(path)-1):
                                        new_link_capacity[(path[i], path[i+1])] -= vl["bandwidth"]
                                    new_routed[(src, dst)] = path
                                    g_cost += vl["bandwidth"]
                            except nx.NetworkXNoPath:
                                continue

                    new_node_capacity[node] -= vnf_obj["cpu"]
                    new_state = AStarState(new_placed, new_routed, g_cost, new_node_capacity, new_link_capacity)
                    expansions.append(new_state)

            return expansions

        def run():
            initial_state = AStarState(
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
                    f_score = new_state.g_cost + astar_heuristic(new_state)
                    queue.put((f_score, new_state))
            return None

        result = run()
        astar_results.append(result)

    summary = [{"slice": i+1, "accepted": r is not None, "g_cost": r.g_cost if r else None} for i, r in enumerate(astar_results)]
    df_results = pd.DataFrame(summary)

    if csv_path:
        df_results.to_csv(csv_path, index=False)

    return df_results, astar_results
