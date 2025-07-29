from queue import PriorityQueue
import pandas as pd
import networkx as nx
from copy import deepcopy

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

def run_astar(G, slices, node_capacity_base, link_latency, link_capacity_base, csv_path=None):
    astar_results = []

    # Global mutable capacities
    global_node_capacity = node_capacity_base.copy()
    global_link_capacity = link_capacity_base.copy()

    for i, (vnf_chain, vl_chain) in enumerate(slices):

        def heuristic(state):
            return sum(vl["bandwidth"] for vl in vl_chain if (vl["from"], vl["to"]) not in state.routed_vls)

        def expand_state(state):
            expansions = []
            next_vnf = next((v["id"] for v in vnf_chain if v["id"] not in state.placed_vnfs), None)
            if next_vnf is None:
                return expansions

            vnf_obj = next(v for v in vnf_chain if v["id"] == next_vnf)

            for node in G.nodes:
                if state.node_capacity.get(node, 0) < vnf_obj["cpu"]:
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
                            path_latency = sum(
                                link_latency[(min(path[i], path[i + 1]), max(path[i], path[i + 1]))]
                                for i in range(len(path) - 1)
                            )

                            if path_latency > vl["latency"]:
                                routing_success = False
                                break

                            if all(
                                new_link_capacity[(min(path[i], path[i + 1]), max(path[i], path[i + 1]))] >= vl["bandwidth"]
                                for i in range(len(path) - 1)
                            ):
                                for i in range(len(path) - 1):
                                    edge = (min(path[i], path[i + 1]), max(path[i], path[i + 1]))
                                    new_link_capacity[edge] -= vl["bandwidth"]
                                new_routed[(src, dst)] = path
                                g_cost += vl["bandwidth"]
                            else:
                                routing_success = False
                                break
                        except nx.NetworkXNoPath:
                            routing_success = False
                            break

                if routing_success:
                    new_node_capacity[node] -= vnf_obj["cpu"]
                    if new_node_capacity[node] < 0:
                        continue
                    new_state = AStarState(new_placed, new_routed, g_cost, new_node_capacity, new_link_capacity)
                    expansions.append(new_state)

            return expansions

        def run(i):
            initial_state = AStarState(
                placed_vnfs={},
                routed_vls={},
                g_cost=0,
                node_capacity=global_node_capacity.copy(),
                link_capacity=global_link_capacity.copy()
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

            print(f"[Slice {i+1}] Rejected: no valid placement found.", flush=True)
            return None

        result = run(i)
        astar_results.append(result)

        # Update global capacities after successful placement
        if result:
            for vnf_id, node in result.placed_vnfs.items():
                cpu = next(
                    vnf["cpu"]
                    for vnf in vnf_chain
                    if vnf["id"] == vnf_id
                )
                global_node_capacity[node] -= cpu

            for (src, dst), path in result.routed_vls.items():
                bandwidth = next(
                    vl["bandwidth"]
                    for vl in vl_chain
                    if vl["from"] == src and vl["to"] == dst
                )
                for i in range(len(path) - 1):
                    edge = (min(path[i], path[i + 1]), max(path[i], path[i + 1]))
                    global_link_capacity[edge] -= bandwidth

    summary = [{"slice": i + 1, "accepted": r is not None, "g_cost": r.g_cost if r else None} for i, r in enumerate(astar_results)]
    df_results = pd.DataFrame(summary)

    if csv_path:
        df_results.to_csv(csv_path, index=False)

    return df_results, astar_results
