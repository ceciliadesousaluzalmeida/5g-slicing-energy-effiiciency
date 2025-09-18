import networkx as nx
import pandas as pd
from queue import PriorityQueue
from tqdm import tqdm
from copy import deepcopy
from itertools import count


class FABOState:
    def __init__(self, placed_vnfs=None, routed_vls=None, g_cost=0, node_capacity=None, link_capacity=None):
        # placed_vnfs: dict[vnf_id -> node]
        # routed_vls: dict[(src_vnf_id, dst_vnf_id) -> path(list of nodes)]
        self.placed_vnfs = placed_vnfs or {}
        self.routed_vls = routed_vls or {}
        self.g_cost = g_cost
        self.node_capacity = node_capacity or {}
        self.link_capacity = link_capacity or {}

    def is_goal(self, vnf_chain, vl_chain):
        # Goal when all VNFs placed AND all VLs (with both endpoints placed) are routed
        return len(self.placed_vnfs) == len(vnf_chain) and len(self.routed_vls) == len(vl_chain)

    def __lt__(self, other):
        # Tie-breaker for PriorityQueue; we keep it, but callers also pass a counter to be safe
        return self.g_cost < other.g_cost


def run_fabo_full_batch(
    G: nx.Graph,
    slices,
    node_capacity_base: dict,
    link_latency: dict,
    link_capacity_base: dict,
    csv_path: str = None,
    fail_fast: bool = True,
):
    """
    Args:
        G: networkx Graph/DiGraph
        slices: iterable of (vnf_chain, vl_chain)
            vnf_chain: list of dicts with keys: {"id", "cpu", "slice"}
            vl_chain: list of dicts with keys: {"from", "to", "latency", "bandwidth"}
        node_capacity_base: dict[node -> CPU capacity]
        link_latency: dict[(u, v) -> latency] (directional if G.is_directed())
        link_capacity_base: dict[(u, v) -> bandwidth capacity]
        csv_path: optional output CSV for summary
        fail_fast: if True, raise RuntimeError on capacity violations after each slice

    Returns:
        df_results: pandas DataFrame with summary
        fabo_results: list of FABOState or None for each slice
    """

    # --- Helpers -------------------------------------------------------------

    def get_edge_value(dct: dict, a, b, default=0):
        """Read an edge value regardless of direction (mirrors if graph is undirected)."""
        if (a, b) in dct:
            return dct[(a, b)]
        if not G.is_directed() and (b, a) in dct:
            return dct[(b, a)]
        return default

    def dec_edge_capacity(dct: dict, a, b, amount: float):
        """Decrement edge capacity for (a,b) and mirror if graph is undirected."""
        cur = get_edge_value(dct, a, b, 0)
        dct[(a, b)] = cur - amount
        if not G.is_directed():
            cur_rev = get_edge_value(dct, b, a, 0)
            dct[(b, a)] = cur_rev - amount

    def enough_bw_on_path(dct: dict, path, bw: float) -> bool:
        """Check if all edges on path have >= bw available (considering direction)."""
        for u, v in zip(path, path[1:]):
            if get_edge_value(dct, u, v, 0) < bw:
                return False
        return True

    def dijkstra_by_latency(src_node, dst_node):
        """Shortest path by latency using link_latency dict (no need for G edge attributes)."""
        def w(u, v, data):
            return get_edge_value(link_latency, u, v, 9999)
        # networkx allows a callable weight(u, v, data)
        return nx.shortest_path(G, source=src_node, target=dst_node, weight=w)

    def path_latency_total(path):
        """Compute total latency from link_latency dict."""
        return sum(get_edge_value(link_latency, path[i], path[i+1], 9999) for i in range(len(path)-1))

    # --- Global structures (mutable across slices) ---------------------------

    fabo_results = []
    node_capacity_global = deepcopy(node_capacity_base)
    link_capacity_global = deepcopy(link_capacity_base)
    total_capacity = deepcopy(node_capacity_base)  # used for fairness ratio

    # --- Expander ------------------------------------------------------------

    def expand_state(state: FABOState, vnf_chain, vl_chain):
        """Generate next states by placing the next VNF (and routing newly eligible VLs)."""
        expansions = []

        # Pick next VNF not yet placed (keep chain order)
        next_vnf_id = next((v["id"] for v in vnf_chain if v["id"] not in state.placed_vnfs), None)
        if next_vnf_id is None:
            return expansions

        vnf_obj = next(v for v in vnf_chain if v["id"] == next_vnf_id)
        vnf_cpu = vnf_obj["cpu"]
        vnf_slice = vnf_obj["slice"]

        # Fairness: prefer nodes with lower used ratio = (1 - remaining/base)
        cpu_used_ratio = {
            n: 1.0 - (state.node_capacity.get(n, 0) / total_capacity.get(n, 1)) if total_capacity.get(n, 0) > 0 else 1.0
            for n in G.nodes
        }
        candidate_nodes = sorted(G.nodes, key=lambda n: cpu_used_ratio[n])

        for node in candidate_nodes:
            # Basic CPU check
            if state.node_capacity.get(node, 0) < vnf_cpu:
                continue

            # Enforce: no two VNFs from the same slice on the same node
            already_on_node_same_slice = any(
                placed_node == node and
                vnf_slice == next(v["slice"] for v in vnf_chain if v["id"] == placed_id)
                for placed_id, placed_node in state.placed_vnfs.items()
            )
            if already_on_node_same_slice:
                continue

            # Copy structures (state isolation)
            new_placed = dict(state.placed_vnfs)
            new_routed = dict(state.routed_vls)
            new_node_capacity = deepcopy(state.node_capacity)
            new_link_capacity = deepcopy(state.link_capacity)
            g_cost = state.g_cost

            # Debit CPU early (fail-fast for routing attempts)
            new_node_capacity[node] -= vnf_cpu
            if new_node_capacity[node] < 0:
                # Should not happen due to check, but keep it safe
                continue

            # Place VNF
            new_placed[next_vnf_id] = node

            # Route any VL whose endpoints are now both placed and not routed yet
            routing_success = True
            for vl in vl_chain:
                src, dst = vl["from"], vl["to"]
                if (src, dst) in new_routed:
                    continue
                if src in new_placed and dst in new_placed:
                    src_node = new_placed[src]
                    dst_node = new_placed[dst]
                    try:
                        path = dijkstra_by_latency(src_node, dst_node)
                    except nx.NetworkXNoPath:
                        routing_success = False
                        break

                    # Latency SLA
                    total_lat = path_latency_total(path)
                    if total_lat > vl["latency"]:
                        routing_success = False
                        break

                    # Bandwidth feasibility
                    bw = vl["bandwidth"]
                    if not enough_bw_on_path(new_link_capacity, path, bw):
                        routing_success = False
                        break

                    # Compute congestion penalty and debit BW
                    penalty = 0.0
                    for a, b in zip(path, path[1:]):
                        cap_now = get_edge_value(new_link_capacity, a, b, 0)
                        cap_base = max(1, get_edge_value(link_capacity_base, a, b, 1))
                        penalty += 1.0 + (1.0 - (cap_now / cap_base))
                    for a, b in zip(path, path[1:]):
                        dec_edge_capacity(new_link_capacity, a, b, bw)

                    # Register routed VL and cost
                    new_routed[(src, dst)] = path
                    g_cost += bw * penalty

            if routing_success:
                expansions.append(FABOState(
                    placed_vnfs=new_placed,
                    routed_vls=new_routed,
                    g_cost=g_cost,
                    node_capacity=new_node_capacity,
                    link_capacity=new_link_capacity
                ))

        return expansions

    # --- Heuristic -----------------------------------------------------------

    def fabo_heuristic(state: FABOState, vnf_chain, vl_chain):
        """Estimate remaining latency based on shortest-path lengths for unrouted VLs with both endpoints placed."""
        total_estimated_latency = 0.0
        for vl in vl_chain:
            src, dst = vl["from"], vl["to"]
            if (src, dst) in state.routed_vls:
                continue
            src_node = state.placed_vnfs.get(src)
            dst_node = state.placed_vnfs.get(dst)
            if src_node is not None and dst_node is not None:
                try:
                    # Use the same latency-based path metric as the expander
                    def w(u, v, data):
                        return get_edge_value(link_latency, u, v, 9999)
                    total_estimated_latency += nx.shortest_path_length(G, src_node, dst_node, weight=w)
                except nx.NetworkXNoPath:
                    total_estimated_latency += 9999.0
        return total_estimated_latency

    # --- Main loop per slice -------------------------------------------------

    summary_rows = []
    for i, (vnf_chain, vl_chain) in enumerate(tqdm(slices, desc="Running FABO", unit="slice")):

        def run_single_slice():
            initial_state = FABOState(
                placed_vnfs={},
                routed_vls={},
                g_cost=0.0,
                node_capacity=deepcopy(node_capacity_global),
                link_capacity=deepcopy(link_capacity_global)
            )

            # PriorityQueue items: (f_score, tie_counter, state)
            pq = PriorityQueue()
            tiebreak = count()
            pq.put((0.0, next(tiebreak), initial_state))

            while not pq.empty():
                _, __, state = pq.get()

                if state.is_goal(vnf_chain, vl_chain):
                    return state

                for ns in expand_state(state, vnf_chain, vl_chain):
                    f_score = ns.g_cost + fabo_heuristic(ns, vnf_chain, vl_chain)
                    pq.put((f_score, next(tiebreak), ns))

            return None

        result = run_single_slice()
        fabo_results.append(result)

        # Update global capacities with strict validation
        if result:
            # Accumulate CPU usage per node for this slice
            used_per_node = {}
            for vnf_id, node in result.placed_vnfs.items():
                cpu = next(v["cpu"] for v in vnf_chain if v["id"] == vnf_id)
                used_per_node[node] = used_per_node.get(node, 0) + cpu

            # Apply CPU usage and validate
            for node, cpu_used in used_per_node.items():
                node_capacity_global[node] = node_capacity_global.get(node, 0) - cpu_used
                if fail_fast and node_capacity_global[node] < 0:
                    raise RuntimeError(
                        f"[CPU overflow] Node {node} negative after slice #{i+1} "
                        f"(remaining={node_capacity_global[node]})."
                    )

            # Apply BW usage and validate (both directions if undirected)
            for (src, dst), path in result.routed_vls.items():
                vl = next(v for v in vl_chain if v["from"] == src and v["to"] == dst)
                bw = vl["bandwidth"]
                for a, b in zip(path, path[1:]):
                    # forward
                    link_capacity_global[(a, b)] = get_edge_value(link_capacity_global, a, b, 0) - bw
                    if fail_fast and link_capacity_global[(a, b)] < 0:
                        raise RuntimeError(
                            f"[BW overflow] Link {(a, b)} negative after slice #{i+1}."
                        )
                    # mirror if needed
                    if not G.is_directed():
                        link_capacity_global[(b, a)] = get_edge_value(link_capacity_global, b, a, 0) - bw
                        if fail_fast and link_capacity_global[(b, a)] < 0:
                            raise RuntimeError(
                                f"[BW overflow] Link {(b, a)} negative after slice #{i+1}."
                            )

        summary_rows.append({
            "slice": i + 1,
            "accepted": result is not None,
            "g_cost": (result.g_cost if result else None)
        })

    # Final sanity checks (optional but useful)
    for n in node_capacity_base:
        used = node_capacity_base[n] - node_capacity_global.get(n, 0)
        if fail_fast and used < 0:
            raise RuntimeError(f"[CPU accounting] Node {n} negative 'used' ({used}).")
        if fail_fast and used > node_capacity_base[n]:
            raise RuntimeError(f"[CPU overflow] Node {n} used {used} > base {node_capacity_base[n]}.")

    df_results = pd.DataFrame(summary_rows)
    if csv_path:
        df_results.to_csv(csv_path, index=False)

    return df_results, fabo_results
