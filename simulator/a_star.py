from copy import deepcopy
import pandas as pd
import networkx as nx
from queue import PriorityQueue

# ------------------------- A* DATA STRUCTURES -------------------------
class AStarState:
    def __init__(self, placed_vnfs=None, routed_vls=None, g_cost=0,
                 node_capacity=None, link_capacity=None):
        # placed_vnfs: dict[vnf_id -> node]
        # routed_vls:  dict[(src_vnf_id, dst_vnf_id) -> path(list of nodes)]
        self.placed_vnfs   = placed_vnfs or {}
        self.routed_vls    = routed_vls or {}
        self.g_cost        = g_cost
        self.node_capacity = node_capacity or {}
        self.link_capacity = link_capacity or {}

    def is_goal(self, vnf_chain, vl_chain):
        # Goal = all VNFs placed AND all VLs routed
        return (len(self.placed_vnfs) == len(vnf_chain)
                and len(self.routed_vls) == len(vl_chain))

    def __lt__(self, other):
        # PriorityQueue needs a strict ordering
        return self.g_cost < other.g_cost


# ---------------------------- A* ORCHESTRATOR ----------------------------
def run_astar(G, slices, node_capacity_base, link_capacity_base, csv_path=None):
    """
    A* joint VNF placement + VL routing (per-slice), committing resources slice-by-slice.

    Parameters
    ----------
    G : networkx.Graph
        Substrate graph with edge attr 'latency'. (Capacity dicts provided externally)
    slices : list of tuples
        Each item is (vnf_chain, vl_chain)
        - vnf_chain: list of dicts with keys: id, slice, cpu
        - vl_chain:  list of dicts with keys: from, to, bandwidth
    node_capacity_base : dict[node -> cpu_available]
    link_capacity_base : dict[(u,v) -> bw_available]  # undirected may be given once; we read symmetric
    csv_path : str or None
        If provided, writes summary CSV.

    Returns
    -------
    df_results : pandas.DataFrame
    astar_results : list[AStarState or None]
    """
    astar_results = []

    # Global capacities that will be COMMITTED after each accepted slice
    node_capacity_global = deepcopy(node_capacity_base)
    link_capacity_global = deepcopy(link_capacity_base)

    # --------------- helper: A* heuristic (latency lower bound) ---------------
    def heuristic_for_slice(state, vnf_chain, vl_chain):
        """
        Admissible-ish heuristic: sum of shortest path latencies between already-placed VNF endpoints
        whose VLs are not yet routed. If nodes are not both placed, ignore (optimistic).
        """
        h = 0
        for vl in vl_chain:
            src = vl["from"]
            dst = vl["to"]
            if (src, dst) in state.routed_vls:
                continue
            src_node = state.placed_vnfs.get(src)
            dst_node = state.placed_vnfs.get(dst)
            if src_node is not None and dst_node is not None:
                try:
                    # Use latency edge attribute; fall back to unweighted if missing
                    h += nx.shortest_path_length(G, src_node, dst_node, weight="latency")
                except nx.NetworkXNoPath:
                    # Hefty penalty to discourage this branch
                    h += 10_000
        return h

    # --------------- helper: expand state by placing the next VNF ---------------
    def expand_state(state, vnf_chain, vl_chain):
        """
        Generate all feasible next states by placing the next unplaced VNF into some node, then
        routing any now-fully-instantiated VLs. Enforces CPU capacity, link capacity and anti-affinity.
        """
        expansions = []

        # Pick next VNF deterministically: first unplaced by id order to keep search stable
        unplaced_ids = [v["id"] for v in vnf_chain if v["id"] not in state.placed_vnfs]
        if not unplaced_ids:
            return expansions
        next_vnf_id = sorted(unplaced_ids)[0]
        vnf_obj = next(v for v in vnf_chain if v["id"] == next_vnf_id)

        # Nodes sorted by remaining CPU desc to try "promising" fits first
        sorted_nodes = sorted(G.nodes, key=lambda n: state.node_capacity.get(n, 0), reverse=True)

        for node in sorted_nodes:
            avail_cpu = state.node_capacity.get(node, 0)
            cpu_need  = vnf_obj["cpu"]

            # --- CPU gating before allocation ---
            if avail_cpu < cpu_need:
                # Debug log in English as required
                print(f"[DEBUG][A*] Skip node {node} for {vnf_obj['id']}: "
                      f"required CPU={cpu_need}, available={avail_cpu}")
                continue

            # --- Anti-affinity: no two VNFs of the same slice on the same node ---
            same_slice_on_node = False
            for placed_id, placed_node in state.placed_vnfs.items():
                if placed_node != node:
                    continue
                # Find slice of already placed VNF
                placed_slice = next(v for v in vnf_chain if v["id"] == placed_id)["slice"]
                if placed_slice == vnf_obj["slice"]:
                    same_slice_on_node = True
                    break
            if same_slice_on_node:
                print(f"[DEBUG][A*] Anti-affinity: {vnf_obj['id']} cannot be placed on node {node} "
                      f"(another VNF of slice {vnf_obj['slice']} already there).")
                continue

            # --- Create next state (deep copies to avoid reference leaks) ---
            new_placed        = state.placed_vnfs.copy()
            new_routed        = state.routed_vls.copy()
            new_node_capacity = deepcopy(state.node_capacity)
            new_link_capacity = deepcopy(state.link_capacity)
            new_cost          = state.g_cost

            # --- Place VNF and deduct CPU ---
            new_node_capacity[node] -= cpu_need
            if new_node_capacity[node] < 0:
                # Safety guard; should never happen due to pre-check
                print(f"[ERROR][A*] Negative CPU after placement on node {node}: {new_node_capacity[node]}")
                continue
            new_placed[next_vnf_id] = node
            print(f"[INFO][A*] Placed {next_vnf_id} on node {node} (use={cpu_need}, "
                  f"remaining={new_node_capacity[node]}).")

            # --- Try to route any VLs whose endpoints are now both placed ---
            routing_ok = True
            for vl in vl_chain:
                key = (vl["from"], vl["to"])
                if key in new_routed:
                    continue
                src_node = new_placed.get(vl["from"])
                dst_node = new_placed.get(vl["to"])
                if src_node is None or dst_node is None:
                    continue  # cannot route yet

                try:
                    path = nx.shortest_path(G, src_node, dst_node, weight="latency")
                except nx.NetworkXNoPath:
                    print(f"[DEBUG][A*] No path between {src_node} and {dst_node} for VL {key}.")
                    routing_ok = False
                    break

                # Check link capacities along the path
                for u, v in zip(path[:-1], path[1:]):
                    # Support undirected capacity dicts by checking both orientations
                    cap = (new_link_capacity.get((u, v))
                           if (u, v) in new_link_capacity
                           else new_link_capacity.get((v, u)))
                    if cap is None or cap < vl["bandwidth"]:
                        print(f"[DEBUG][A*] Insufficient link capacity on edge ({u},{v}) "
                              f"for VL {key}: need={vl['bandwidth']}, have={cap}.")
                        routing_ok = False
                        break
                if not routing_ok:
                    break

                # Deduct bandwidth and commit this VL
                for u, v in zip(path[:-1], path[1:]):
                    if (u, v) in new_link_capacity:
                        new_link_capacity[(u, v)] -= vl["bandwidth"]
                    else:
                        new_link_capacity[(v, u)] -= vl["bandwidth"]
                new_routed[key] = path
                # Accumulate latency cost
                new_cost += sum(G[u][v].get("latency", 1) for u, v in zip(path[:-1], path[1:]))

            if not routing_ok:
                # Skip this expansion; routing failed for at least one VL
                print(f"[DEBUG][A*] Discard state: routing failed after placing {next_vnf_id} on node {node}.")
                continue

            # State is feasible; add to frontier
            expansions.append(
                AStarState(new_placed, new_routed, new_cost, new_node_capacity, new_link_capacity)
            )

        return expansions

    # --------------- A* per-slice search ---------------
    def solve_one_slice(vnf_chain, vl_chain):
        """Return a feasible/low-cost AStarState or None."""
        init_state = AStarState(
            placed_vnfs={},
            routed_vls={},
            g_cost=0,
            node_capacity=deepcopy(node_capacity_global),   # start from GLOBAL snapshot
            link_capacity=deepcopy(link_capacity_global)
        )
        pq = PriorityQueue()
        # (f = g + h, tie-breaker serial, state)
        counter = 0
        pq.put((0, counter, init_state))

        visited = 0
        while not pq.empty():
            _, _, state = pq.get()
            visited += 1

            if state.is_goal(vnf_chain, vl_chain):
                print(f"[INFO][A*] Found feasible solution for slice after expanding {visited} states.")
                return state

            for child in expand_state(state, vnf_chain, vl_chain):
                h = heuristic_for_slice(child, vnf_chain, vl_chain)
                counter += 1
                pq.put((child.g_cost + h, counter, child))

        print("[WARN][A*] No feasible solution for this slice.")
        return None

    # --------------- Main loop over slices ---------------
    for i, (vnf_chain, vl_chain) in enumerate(slices, start=1):
        print(f"\n[INFO][A*] === Solving slice {i} with {len(vnf_chain)} VNFs and {len(vl_chain)} VLs ===")
        result_state = solve_one_slice(vnf_chain, vl_chain)
        astar_results.append(result_state)

        if result_state is None:
            continue  # slice rejected; do not change global capacities

        # -------- Commit resource usage from the accepted solution to GLOBAL --------
        # Commit CPU
        for vnf_id, node in result_state.placed_vnfs.items():
            # Find the cpu of this VNF from the chain
            vnf_cpu = next(v["cpu"] for v in vnf_chain if v["id"] == vnf_id)
            node_capacity_global[node] -= vnf_cpu
            if node_capacity_global[node] < 0:
                # Hard error: global over-commit should never happen
                print(f"[ERROR][A*] Global CPU over-commit on node {node}: "
                      f"{node_capacity_global[node]} after committing {vnf_id} (cpu={vnf_cpu}).")

        # Commit bandwidth
        for (src, dst), path in result_state.routed_vls.items():
            bw = next(vl["bandwidth"] for vl in vl_chain if vl["from"] == src and vl["to"] == dst)
            for u, v in zip(path[:-1], path[1:]):
                if (u, v) in link_capacity_global:
                    link_capacity_global[(u, v)] -= bw
                elif (v, u) in link_capacity_global:
                    link_capacity_global[(v, u)] -= bw
                else:
                    print(f"[ERROR][A*] Edge ({u},{v}) not found in global link capacities during commit.")

        # Optional: summary snapshot for diagnostics
        print(f"[SUMMARY][A*] After slice {i}: "
              f"min_node_cpu={min(node_capacity_global.values())}, "
              f"num_links_low_bw={sum(1 for v in link_capacity_global.values() if v <= 0)}")

    # --------------- Final sanity check (diagnostic only) ---------------
    for n, cap in node_capacity_global.items():
        if cap < 0:
            print(f"[ALERT][A*] Overcapacity on node {n}: used exceeds limit by {-cap} units.")

    # --------------- Build summary dataframe ---------------
    summary = []
    for idx, res in enumerate(astar_results, start=1):
        summary.append({
            "slice": idx,
            "accepted": res is not None,
            "g_cost": res.g_cost if res else None
        })
    df_results = pd.DataFrame(summary)
    if csv_path:
        try:
            df_results.to_csv(csv_path, index=False)
            print(f"[INFO][A*] Results written to {csv_path}")
        except Exception as e:
            print(f"[WARN][A*] Could not write CSV: {e}")

    return df_results, astar_results
