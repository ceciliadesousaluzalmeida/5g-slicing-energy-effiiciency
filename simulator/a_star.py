from copy import deepcopy
import pandas as pd
import networkx as nx
from queue import PriorityQueue

class AStarState:
    def __init__(self, placed_vnfs=None, routed_vls=None, g_cost=0, node_capacity=None, link_capacity=None):
        self.placed_vnfs = placed_vnfs or {}
        self.routed_vls = routed_vls or {}
        self.g_cost = g_cost
        self.node_capacity = node_capacity or {}
        self.link_capacity = link_capacity or {}

    def is_goal(self, vnf_chain, vl_chain):
        return (
            len(self.placed_vnfs) == len(vnf_chain) and
            len(self.routed_vls) == len(vl_chain)
        )

    def __lt__(self, other):
        return self.g_cost < other.g_cost

def run_astar(G, slices, node_capacity_base, link_capacity_base, csv_path=None):
    astar_results = []
    node_capacity_global = deepcopy(node_capacity_base)
    link_capacity_global = deepcopy(link_capacity_base)

    for i, (vnf_chain, vl_chain) in enumerate(slices):
        def heuristic(state):
            total = 0
            for vl in vl_chain:
                src = vl["from"]
                dst = vl["to"]
                if (src, dst) in state.routed_vls:
                    continue
                src_node = state.placed_vnfs.get(src)
                dst_node = state.placed_vnfs.get(dst)
                if src_node is not None and dst_node is not None:
                    try:
                        total += nx.shortest_path_length(G, src_node, dst_node, weight="latency")
                    except nx.NetworkXNoPath:
                        total += 9999
            return total

        def expand_state(state):
            expansions = []
            next_vnf = next((v["id"] for v in vnf_chain if v["id"] not in state.placed_vnfs), None)
            if next_vnf is None:
                return expansions

            vnf_obj = next(v for v in vnf_chain if v["id"] == next_vnf)

            sorted_nodes = sorted(
                G.nodes,
                key=lambda n: state.node_capacity.get(n, 0),
                reverse=True
            )

            for node in sorted_nodes:
                if state.node_capacity.get(node, 0) < vnf_obj["cpu"]:
                    continue

                already_placed = any(
                    placed_node == node and
                    vnf_obj["slice"] == v["slice"]
                    for vnf_id, placed_node in state.placed_vnfs.items()
                    for v in vnf_chain if v["id"] == vnf_id
                )
                if already_placed:
                    continue

                new_placed = state.placed_vnfs.copy()
                new_placed[next_vnf] = node
                new_routed = state.routed_vls.copy()
                new_node_capacity = deepcopy(state.node_capacity)
                new_link_capacity = deepcopy(state.link_capacity)
                g_cost = state.g_cost
                routing_success = True

                for vl in vl_chain:
                    src, dst = vl["from"], vl["to"]
                    if src in new_placed and dst in new_placed and (src, dst) not in new_routed:
                        src_node = new_placed[src]
                        dst_node = new_placed[dst]
                        try:
                            path = nx.shortest_path(G, src_node, dst_node, weight="latency")
                            for u, v in zip(path[:-1], path[1:]):
                                cap = new_link_capacity.get((u, v)) or new_link_capacity.get((v, u))
                                if cap is None or cap < vl["bandwidth"]:
                                    routing_success = False
                                    break
                            if not routing_success:
                                break
                            for u, v in zip(path[:-1], path[1:]):
                                if (u, v) in new_link_capacity:
                                    new_link_capacity[(u, v)] -= vl["bandwidth"]
                                else:
                                    new_link_capacity[(v, u)] -= vl["bandwidth"]
                            new_routed[(src, dst)] = path
                            g_cost += sum(G[u][v]["latency"] for u, v in zip(path[:-1], path[1:]))
                        except nx.NetworkXNoPath:
                            routing_success = False
                            break

                if routing_success:
                    new_node_capacity[node] -= vnf_obj["cpu"]
                    new_state = AStarState(new_placed, new_routed, g_cost, new_node_capacity, new_link_capacity)
                    expansions.append(new_state)

            return expansions

        def run():
            initial_state = AStarState(
                placed_vnfs={},
                routed_vls={},
                g_cost=0,
                node_capacity=deepcopy(node_capacity_global),
                link_capacity=deepcopy(link_capacity_global)
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
        astar_results.append(result)

        if result:
            for vnf_id, node in result.placed_vnfs.items():
                slice_id = int(vnf_id.split("_")[0][-1])
                vnf_idx = int(vnf_id.split("_")[1])
                cpu_needed = slices[slice_id][0][vnf_idx]["cpu"]
                node_capacity_global[node] -= cpu_needed

    summary = [{"slice": i + 1, "accepted": r is not None, "g_cost": r.g_cost if r else None} for i, r in enumerate(astar_results)]
    df_results = pd.DataFrame(summary)
    return df_results, astar_results
