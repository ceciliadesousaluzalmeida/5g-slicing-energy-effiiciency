from pathlib import Path
import networkx as nx
from queue import PriorityQueue
from copy import deepcopy
import pandas as pd

class AStarBestFitState:
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

def run_astar_best_fit(G, slices, node_capacity_base, link_capacity_base, link_latency, csv_path=None):
    results = []

    for i, (vnf_chain, vl_chain) in enumerate(slices):
        def heuristic(state):
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
            if not next_vnf:
                return expansions

            vnf_obj = next(v for v in vnf_chain if v["id"] == next_vnf)

            # Best Fit: sort by remaining CPU (smallest positive first)
            candidate_nodes = sorted(
                [n for n in G.nodes if state.node_capacity[n] >= vnf_obj["cpu"]],
                key=lambda n: state.node_capacity[n]
            )

            for node in candidate_nodes:
                # Avoid placing two VNFs from same slice on same node
                same_slice = any(
                    placed_node == node and vnf_obj["slice"] == placed_vnf["slice"]
                    for placed_vnf_id, placed_node in state.placed_vnfs.items()
                    for placed_vnf in vnf_chain
                    if placed_vnf["id"] == placed_vnf_id
                )
                if same_slice:
                    continue

                new_placed = state.placed_vnfs.copy()
                new_placed[next_vnf] = node
                new_routed = state.routed_vls.copy()
                new_node_capacity = deepcopy(state.node_capacity)
                new_link_capacity = deepcopy(state.link_capacity)
                g_cost = state.g_cost
                success = True

                for vl in vl_chain:
                    src, dst = vl["from"], vl["to"]
                    if src in new_placed and dst in new_placed and (src, dst) not in new_routed:
                        src_node, dst_node = new_placed[src], new_placed[dst]
                        try:
                            path = nx.shortest_path(G, src_node, dst_node, weight="latency")
                            for u, v in zip(path[:-1], path[1:]):
                                cap = new_link_capacity.get((u, v)) or new_link_capacity.get((v, u))
                                if cap is None or cap < vl["bandwidth"]:
                                    success = False
                                    break
                            if not success:
                                break
                            for u, v in zip(path[:-1], path[1:]):
                                if (u, v) in new_link_capacity:
                                    new_link_capacity[(u, v)] -= vl["bandwidth"]
                                else:
                                    new_link_capacity[(v, u)] -= vl["bandwidth"]
                            new_routed[(src, dst)] = path
                            g_cost += sum(link_latency.get((u, v), link_latency.get((v, u), 0)) for u, v in zip(path[:-1], path[1:]))
                        except nx.NetworkXNoPath:
                            success = False
                            break

                if success:
                    new_node_capacity[node] -= vnf_obj["cpu"]
                    new_state = AStarBestFitState(new_placed, new_routed, g_cost, new_node_capacity, new_link_capacity)
                    expansions.append(new_state)

            return expansions

        def run():
            initial_state = AStarBestFitState(
                placed_vnfs={},
                routed_vls={},
                g_cost=0,
                node_capacity=deepcopy(node_capacity_base),
                link_capacity=deepcopy(link_capacity_base)
            )
            queue = PriorityQueue()
            queue.put((0, initial_state))

            while not queue.empty():
                _, state = queue.get()
                if state.is_goal(vnf_chain, vl_chain):
                    return state
                for new_state in expand_state(state):
                    f_score = new_state.g_cost + heuristic(new_state)
                    queue.put((f_score, new_state))
            return None

        result = run()
        results.append(result)

    df = pd.DataFrame([{
        "slice": i + 1,
        "accepted": r is not None,
        "g_cost": r.g_cost if r else None
    } for i, r in enumerate(results)])

    if csv_path:
        df.to_csv(csv_path, index=False)

    return df, results
