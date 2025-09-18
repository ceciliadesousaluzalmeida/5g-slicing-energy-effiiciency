import networkx as nx
import pandas as pd
from queue import PriorityQueue
from copy import deepcopy
from itertools import count

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


def run_fabo_full_batch(
    G: nx.Graph,
    slices,
    node_capacity_base: dict,
    link_latency: dict,
    link_capacity_base: dict,
    csv_path: str = None,
):
    # --- Helpers -------------------------------------------------------------

    def get_edge_value(dct: dict, a, b, default=0):
        if (a, b) in dct:
            return dct[(a, b)]
        if not G.is_directed() and (b, a) in dct:
            return dct[(b, a)]
        return default

    def dec_edge_capacity(dct: dict, a, b, amount: float):
        cur = get_edge_value(dct, a, b, 0)
        dct[(a, b)] = cur - amount
        if not G.is_directed():
            cur_rev = get_edge_value(dct, b, a, 0)
            dct[(b, a)] = cur_rev - amount

    def enough_bw_on_path(dct: dict, path, bw: float) -> bool:
        for u, v in zip(path, path[1:]):
            if get_edge_value(dct, u, v, 0) < bw:
                return False
        return True

    def dijkstra_by_latency(src_node, dst_node):
        def w(u, v, data):
            return get_edge_value(link_latency, u, v, 9999)
        return nx.shortest_path(G, source=src_node, target=dst_node, weight=w)

    def path_latency_total(path):
        return sum(get_edge_value(link_latency, path[i], path[i+1], 9999) for i in range(len(path)-1))

    # --- Global structures ---------------------------------------------------
    fabo_results = []
    node_capacity_global = deepcopy(node_capacity_base)
    link_capacity_global = deepcopy(link_capacity_base)
    total_capacity = deepcopy(node_capacity_base)

    # --- Expander ------------------------------------------------------------
    def expand_state(state: FABOState, vnf_chain, vl_chain):
        expansions = []
        next_vnf_id = next((v["id"] for v in vnf_chain if v["id"] not in state.placed_vnfs), None)
        if next_vnf_id is None:
            return expansions

        vnf_obj = next(v for v in vnf_chain if v["id"] == next_vnf_id)
        vnf_cpu = vnf_obj["cpu"]
        vnf_slice = vnf_obj["slice"]

        cpu_used_ratio = {
            n: 1.0 - (state.node_capacity.get(n, 0) / total_capacity.get(n, 1)) if total_capacity.get(n, 0) > 0 else 1.0
            for n in G.nodes
        }
        candidate_nodes = sorted(G.nodes, key=lambda n: cpu_used_ratio[n])

        for node in candidate_nodes:
            avail_cpu = state.node_capacity.get(node, 0)

            if avail_cpu < vnf_cpu:
                print(f"[DEBUG][FABO] Skip node {node} for {vnf_obj['id']}: "
                      f"required={vnf_cpu}, available={avail_cpu}")
                continue

            already_on_node_same_slice = any(
                placed_node == node and
                vnf_slice == next(v["slice"] for v in vnf_chain if v["id"] == placed_id)
                for placed_id, placed_node in state.placed_vnfs.items()
            )
            if already_on_node_same_slice:
                print(f"[DEBUG][FABO] Anti-affinity: {vnf_obj['id']} cannot be placed on node {node}.")
                continue

            new_placed = dict(state.placed_vnfs)
            new_routed = dict(state.routed_vls)
            new_node_capacity = deepcopy(state.node_capacity)
            new_link_capacity = deepcopy(state.link_capacity)
            g_cost = state.g_cost

            new_node_capacity[node] -= vnf_cpu
            if new_node_capacity[node] < 0:
                print(f"[ERROR][FABO] Negative CPU after placing {next_vnf_id} on node {node}.")
                continue

            new_placed[next_vnf_id] = node
            print(f"[INFO][FABO] Placed {next_vnf_id} on node {node} "
                  f"(use={vnf_cpu}, remaining={new_node_capacity[node]}).")

            routing_success = True
            for vl in vl_chain:
                key = (vl["from"], vl["to"])
                if key in new_routed:
                    continue
                if vl["from"] in new_placed and vl["to"] in new_placed:
                    src_node = new_placed[vl["from"]]
                    dst_node = new_placed[vl["to"]]
                    try:
                        path = dijkstra_by_latency(src_node, dst_node)
                    except nx.NetworkXNoPath:
                        print(f"[DEBUG][FABO] No path for VL {key}.")
                        routing_success = False
                        break

                    total_lat = path_latency_total(path)
                    if total_lat > vl["latency"]:
                        print(f"[DEBUG][FABO] Latency SLA failed for VL {key}: "
                              f"{total_lat} > {vl['latency']}")
                        routing_success = False
                        break

                    bw = vl["bandwidth"]
                    if not enough_bw_on_path(new_link_capacity, path, bw):
                        print(f"[DEBUG][FABO] Bandwidth SLA failed for VL {key}.")
                        routing_success = False
                        break

                    penalty = 0.0
                    for a, b in zip(path, path[1:]):
                        cap_now = get_edge_value(new_link_capacity, a, b, 0)
                        cap_base = max(1, get_edge_value(link_capacity_base, a, b, 1))
                        penalty += 1.0 + (1.0 - (cap_now / cap_base))
                    for a, b in zip(path, path[1:]):
                        dec_edge_capacity(new_link_capacity, a, b, bw)

                    new_routed[key] = path
                    g_cost += bw * penalty

            if routing_success:
                expansions.append(FABOState(
                    placed_vnfs=new_placed,
                    routed_vls=new_routed,
                    g_cost=g_cost,
                    node_capacity=new_node_capacity,
                    link_capacity=new_link_capacity
                ))
            else:
                print(f"[DEBUG][FABO] Discard state after placing {next_vnf_id} on node {node}.")

        return expansions

    # --- Heuristic -----------------------------------------------------------
    def fabo_heuristic(state: FABOState, vnf_chain, vl_chain):
        total_estimated_latency = 0.0
        for vl in vl_chain:
            src, dst = vl["from"], vl["to"]
            if (src, dst) in state.routed_vls:
                continue
            src_node = state.placed_vnfs.get(src)
            dst_node = state.placed_vnfs.get(dst)
            if src_node is not None and dst_node is not None:
                try:
                    def w(u, v, data):
                        return get_edge_value(link_latency, u, v, 9999)
                    total_estimated_latency += nx.shortest_path_length(G, src_node, dst_node, weight=w)
                except nx.NetworkXNoPath:
                    total_estimated_latency += 9999.0
        return total_estimated_latency

    # --- Main loop -----------------------------------------------------------
    summary_rows = []
    for i, (vnf_chain, vl_chain) in enumerate(slices, start=1):
        print(f"\n[INFO][FABO] === Solving slice {i} with {len(vnf_chain)} VNFs and {len(vl_chain)} VLs ===")

        def run_single_slice():
            initial_state = FABOState(
                placed_vnfs={},
                routed_vls={},
                g_cost=0.0,
                node_capacity=deepcopy(node_capacity_global),
                link_capacity=deepcopy(link_capacity_global)
            )
            pq = PriorityQueue()
            tiebreak = count()
            pq.put((0.0, next(tiebreak), initial_state))

            visited = 0
            while not pq.empty():
                _, __, state = pq.get()
                visited += 1
                if state.is_goal(vnf_chain, vl_chain):
                    print(f"[INFO][FABO] Found solution after expanding {visited} states.")
                    return state
                for ns in expand_state(state, vnf_chain, vl_chain):
                    f_score = ns.g_cost + fabo_heuristic(ns, vnf_chain, vl_chain)
                    pq.put((f_score, next(tiebreak), ns))
            print("[WARN][FABO] No feasible solution for this slice.")
            return None

        result = run_single_slice()
        fabo_results.append(result)

        if result:
            used_per_node = {}
            for vnf_id, node in result.placed_vnfs.items():
                cpu = next(v["cpu"] for v in vnf_chain if v["id"] == vnf_id)
                used_per_node[node] = used_per_node.get(node, 0) + cpu

            for node, cpu_used in used_per_node.items():
                node_capacity_global[node] -= cpu_used
                if node_capacity_global[node] < 0:
                    print(f"[ERROR][FABO] Global overcapacity on node {node} after slice {i}.")

            for (src, dst), path in result.routed_vls.items():
                bw = next(vl["bandwidth"] for vl in vl_chain if vl["from"] == src and vl["to"] == dst)
                for a, b in zip(path, path[1:]):
                    link_capacity_global[(a, b)] = get_edge_value(link_capacity_global, a, b, 0) - bw
                    if link_capacity_global[(a, b)] < 0:
                        print(f"[ERROR][FABO] Global link overcapacity on {(a, b)} after slice {i}.")
                    if not G.is_directed():
                        link_capacity_global[(b, a)] = get_edge_value(link_capacity_global, b, a, 0) - bw
                        if link_capacity_global[(b, a)] < 0:
                            print(f"[ERROR][FABO] Global link overcapacity on {(b, a)} after slice {i}.")

            print(f"[SUMMARY][FABO] After slice {i}: "
                  f"min_node_cpu={min(node_capacity_global.values())}, "
                  f"links_low_bw={sum(1 for v in link_capacity_global.values() if v <= 0)}")

        summary_rows.append({
            "slice": i,
            "accepted": result is not None,
            "g_cost": (result.g_cost if result else None)
        })

    df_results = pd.DataFrame(summary_rows)
    if csv_path:
        try:
            df_results.to_csv(csv_path, index=False)
            print(f"[INFO][FABO] Results written to {csv_path}")
        except Exception as e:
            print(f"[WARN][FABO] Could not write CSV: {e}")

    return df_results, fabo_results
