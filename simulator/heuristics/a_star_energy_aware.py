# energy_aware_astar.py
from copy import deepcopy
import pandas as pd
import networkx as nx
from queue import PriorityQueue
from dataclasses import dataclass


# ------------------------- Energy model params -------------------------
@dataclass
class EnergyParams:
    """
    Simple linear incremental energy model for links and per-hop forwarding on nodes.
    All rates are per Mbps carried (W/Mbps). Tweak to your measurements.
    """
    link_dynamic_w_per_mbps: float = 0.002   # e.g., 2 mW per Mbps on a link
    node_forward_w_per_mbps: float = 0.001   # e.g., 1 mW per Mbps for per-hop forwarding
    # If you need more realism (idle/static costs, non-linearities), inject them later.


# ------------------------- A* DATA STRUCTURES -------------------------
class AStarEnergyState:
    def __init__(self, placed_vnfs=None, routed_vls=None, g_cost=0.0,
                 node_capacity=None, link_capacity=None,
                 latency_sum=0.0, energy_sum=0.0):
        self.placed_vnfs   = placed_vnfs or {}
        self.routed_vls    = routed_vls or {}
        self.g_cost        = float(g_cost)      # combined objective so far
        self.node_capacity = node_capacity or {}
        self.link_capacity = link_capacity or {}
        self.latency_sum   = float(latency_sum) # accumulated latency only
        self.energy_sum    = float(energy_sum)  # accumulated energy only

    def is_goal(self, vnf_chain, vl_chain, entry=None):
        """Goal: all VNFs placed and all VLs routed."""
        return (len(self.placed_vnfs) == len(vnf_chain) and
                len([1 for vl in vl_chain if (vl["from"], vl["to"]) in self.routed_vls]) == len(vl_chain) and
                (entry is None or ("ENTRY", vnf_chain[0]["id"]) in self.routed_vls))

    def __lt__(self, other):
        # PriorityQueue uses this when f-costs tie
        return self.g_cost < other.g_cost


# ------------------------- helpers: capacities -------------------------
def _get_link_cap(capdict, u, v):
    if (u, v) in capdict:
        return capdict[(u, v)]
    if (v, u) in capdict:
        return capdict[(v, u)]
    return None

def _dec_link_cap(capdict, u, v, amount):
    if (u, v) in capdict:
        capdict[(u, v)] -= amount
    elif (v, u) in capdict:
        capdict[(v, u)] -= amount
    else:
        raise KeyError(f"Edge ({u},{v}) not found in link capacities.")


# ------------------------- per-edge energy -------------------------
def _edge_incremental_energy(u, v, bandwidth_mbps, energy_params: EnergyParams):
    """
    Incremental energy to push `bandwidth_mbps` over edge (u,v).
    Only dynamic parts are counted here; static costs can be modeled externally if needed.
    """
    link_e = energy_params.link_dynamic_w_per_mbps * bandwidth_mbps
    # Count per-hop forwarding energy on the *arrival* node (v)
    node_e = energy_params.node_forward_w_per_mbps * bandwidth_mbps
    return link_e + node_e


# ------------------------- energy+latency shortest path with capacity -------------------------
def energy_latency_shortest_path_with_capacity(G, u, v, link_capacity, bandwidth, w_lat, w_energy, energy_params: EnergyParams):
    """
    Dijkstra on residual graph with composite edge cost:
      cost(u->x) = w_lat * latency(u,x) + w_energy * energy(u,x)
    Capacity constraint: only traverse edges with residual >= bandwidth.
    Returns:
      path, latency_sum, energy_sum  (None, None, None if infeasible)
    """
    import math
    # Distances tracked for composite cost and for separate metrics
    dist = {u: 0.0}
    lat_acc = {u: 0.0}
    en_acc = {u: 0.0}
    prev = {}

    # Min-heap of (composite_cost, node)
    from heapq import heappush, heappop
    heap = []
    heappush(heap, (0.0, u))

    visited = set()

    def has_cap(a, b):
        cap = _get_link_cap(link_capacity, a, b)
        return (cap is not None) and (cap >= bandwidth - 1e-12)

    while heap:
        fcur, cur = heappop(heap)
        if cur in visited:
            continue
        visited.add(cur)
        if cur == v:
            # Reconstruct path
            path = [v]
            x = v
            while x in prev:
                x = prev[x]
                path.append(x)
            path.reverse()
            return path, lat_acc[v], en_acc[v]

        for nbr in G.neighbors(cur):
            if not has_cap(cur, nbr):
                continue
            # edge metrics
            elat = float(G[cur][nbr].get("latency", 1.0))
            eeng = _edge_incremental_energy(cur, nbr, bandwidth, energy_params)
            cand_lat = lat_acc[cur] + elat
            cand_eng = en_acc[cur] + eeng
            cand_f   = w_lat * cand_lat + w_energy * cand_eng

            if nbr not in dist or cand_f + 1e-12 < dist[nbr]:
                dist[nbr] = cand_f
                prev[nbr] = cur
                lat_acc[nbr] = cand_lat
                en_acc[nbr]  = cand_eng
                heappush(heap, (cand_f, nbr))

    # No path
    print(f"[DEBUG][A*-E] No feasible energy-latency path between {u} and {v}.")
    return None, None, None


# ---------------------------- ENERGY-AWARE A* ORCHESTRATOR ----------------------------
def energy_aware_astar(G, slices, node_capacity_base, link_capacity_base,
                       w_lat=0.7, w_energy=0.3,
                       energy_params: EnergyParams = None,
                       csv_path=None):
    """
    Same structure and returns as your run_astar, but A* expands states using a combined cost:
        f = w_lat * (latency_so_far + h_lat) + w_energy * (energy_so_far + h_energy[=0])
    - State.g_cost stores the combined objective.
    - State.latency_sum and State.energy_sum keep the raw sums for logging/summary.

    Args:
      G: networkx graph with edge attribute "latency"
      slices: list of slices; each item is (vnf_chain, vl_chain) or (vnf_chain, vl_chain, entry)
      node_capacity_base: dict node -> cpu
      link_capacity_base: dict (u,v) -> Mbps residual capacity
      w_lat, w_energy: weights for the combined objective
      energy_params: EnergyParams (defaults to simple linear coefficients)
      csv_path: optional CSV output path for the summary

    Returns:
      df_results, astar_results   (DataFrame, list of AStarEnergyState or None)
    """
    if energy_params is None:
        energy_params = EnergyParams()

    astar_results = []
    node_capacity_global = deepcopy(node_capacity_base)
    link_capacity_global = deepcopy(link_capacity_base)

    # ---------- Heuristic (admissible): use only latency lower bound; energy LB = 0 ----------
    def heuristic_for_slice(state, vnf_chain, vl_chain, entry=None):
        # Keep the exact same style as original, but multiply by w_lat (since g_cost is combined)
        h_lat = 0.0
        for vl in vl_chain:
            key = (vl["from"], vl["to"])
            if key in state.routed_vls:
                continue
            src_node = state.placed_vnfs.get(vl["from"])
            dst_node = state.placed_vnfs.get(vl["to"])
            if src_node is not None and dst_node is not None:
                try:
                    h_lat += nx.shortest_path_length(G, src_node, dst_node, weight="latency")
                except nx.NetworkXNoPath:
                    h_lat += 10_000.0

        if entry is not None and vnf_chain:
            first_id = vnf_chain[0]["id"]
            first_node = state.placed_vnfs.get(first_id)
            if first_node is not None and ("ENTRY", first_id) not in state.routed_vls:
                try:
                    h_lat += nx.shortest_path_length(G, entry, first_node, weight="latency")
                except nx.NetworkXNoPath:
                    h_lat += 10_000.0

        # Energy lower bound kept as 0 to preserve admissibility
        return float(w_lat * h_lat)

    # ---------- Expand ----------
    def expand_state(state, vnf_chain, vl_chain, entry=None):
        expansions = []
        unplaced_ids = [v["id"] for v in vnf_chain if v["id"] not in state.placed_vnfs]
        if not unplaced_ids:
            return expansions

        next_vnf_id = sorted(unplaced_ids)[0]
        vnf_obj = next(v for v in vnf_chain if v["id"] == next_vnf_id)
        sorted_nodes = sorted(G.nodes, key=lambda n: state.node_capacity.get(n, 0), reverse=True)

        for node in sorted_nodes:
            avail_cpu = state.node_capacity.get(node, 0)
            cpu_need  = vnf_obj["cpu"]
            if avail_cpu < cpu_need:
                continue

            # Anti-affinity: do not co-locate VNFs of the same slice on the same node
            same_slice_on_node = any(
                next(v for v in vnf_chain if v["id"] == pid)["slice"] == vnf_obj["slice"]
                for pid, n_ in state.placed_vnfs.items() if n_ == node
            )
            if same_slice_on_node:
                continue

            # Clone state
            new_placed = state.placed_vnfs.copy()
            new_routed = state.routed_vls.copy()
            new_node_capacity = deepcopy(state.node_capacity)
            new_link_capacity = deepcopy(state.link_capacity)
            new_combined_cost = state.g_cost
            new_lat_sum = state.latency_sum
            new_eng_sum = state.energy_sum

            # Place VNF
            new_node_capacity[node] -= cpu_need
            new_placed[next_vnf_id] = node

            routing_ok = True
            # Route all VLs that became routable after this placement
            for vl in vl_chain:
                key = (vl["from"], vl["to"])
                if key in new_routed:
                    continue
                src_node = new_placed.get(vl["from"])
                dst_node = new_placed.get(vl["to"])
                if src_node is None or dst_node is None:
                    continue

                path, lat, eng = energy_latency_shortest_path_with_capacity(
                    G, src_node, dst_node,
                    new_link_capacity, vl["bandwidth"],
                    w_lat=w_lat, w_energy=w_energy,
                    energy_params=energy_params
                )
                if path is None:
                    routing_ok = False
                    # Debug similar to original
                    print(f"[DEBUG][A*-E] VL({vl['from']}->{vl['to']}) infeasible after placing VNF {next_vnf_id} on node {node}.")
                    break

                # Reserve bandwidth for the path
                for u, v_ in zip(path[:-1], path[1:]):
                    _dec_link_cap(new_link_capacity, u, v_, vl["bandwidth"])
                new_routed[key] = path

                # Update metrics
                new_lat_sum += lat
                new_eng_sum += eng
                new_combined_cost += w_lat * lat + w_energy * eng

            # Connect ENTRY -> first VNF if applicable and still not routed
            if routing_ok and entry is not None and vnf_chain:
                first_id = vnf_chain[0]["id"]
                if ("ENTRY", first_id) not in new_routed:
                    first_node = new_placed.get(first_id)
                    if first_node is not None:
                        bw_first = vl_chain[0]["bandwidth"] if vl_chain else 0.0
                        path, lat, eng = energy_latency_shortest_path_with_capacity(
                            G, entry, first_node,
                            new_link_capacity, bw_first,
                            w_lat=w_lat, w_energy=w_energy,
                            energy_params=energy_params
                        )
                        if path is None:
                            routing_ok = False
                            print(f"[DEBUG][A*-E] ENTRY->{first_id} infeasible after placing VNF {next_vnf_id} on node {node}.")
                        else:
                            for u, v_ in zip(path[:-1], path[1:]):
                                _dec_link_cap(new_link_capacity, u, v_, bw_first)
                            new_routed[("ENTRY", first_id)] = path
                            new_lat_sum += lat
                            new_eng_sum += eng
                            new_combined_cost += w_lat * lat + w_energy * eng

            if routing_ok:
                expansions.append(
                    AStarEnergyState(new_placed, new_routed, new_combined_cost,
                                     new_node_capacity, new_link_capacity,
                                     latency_sum=new_lat_sum, energy_sum=new_eng_sum)
                )
        return expansions

    # ---------- Solve one slice ----------
    def solve_one_slice(vnf_chain, vl_chain, entry=None):
        init_state = AStarEnergyState({}, {}, 0.0,
                                      deepcopy(node_capacity_global),
                                      deepcopy(link_capacity_global),
                                      latency_sum=0.0, energy_sum=0.0)
        pq = PriorityQueue()
        counter = 0
        pq.put((0.0, counter, init_state))
        visited = 0

        print(f"[INFO][A*-E] Weights: w_lat={w_lat:.3f}, w_energy={w_energy:.3f}")

        while not pq.empty():
            _, _, state = pq.get()
            visited += 1

            if state.is_goal(vnf_chain, vl_chain, entry):
                print(f"[INFO][A*-E] Solution found after {visited} expansions. "
                      f"(lat={state.latency_sum:.3f}, energy={state.energy_sum:.6f}, cost={state.g_cost:.3f})")
                return state

            for child in expand_state(state, vnf_chain, vl_chain, entry):
                h = heuristic_for_slice(child, vnf_chain, vl_chain, entry)
                counter += 1
                pq.put((child.g_cost + h, counter, child))

        print(f"[WARN][A*-E] No feasible solution for slice after {visited} expansions.")
        return None

    # ---------- Main loop ----------
    for i, slice_data in enumerate(slices, start=1):
        if len(slice_data) == 2:
            vnf_chain, vl_chain = slice_data
            entry = None
        elif len(slice_data) == 3:
            vnf_chain, vl_chain, entry = slice_data
        else:
            raise ValueError(f"Unexpected slice format: {slice_data}")

        print(f"\n[INFO][A*-E] === Solving slice {i} (VNFs={len(vnf_chain)}, VLs={len(vl_chain)}) ===")
        result_state = solve_one_slice(vnf_chain, vl_chain, entry)
        astar_results.append(result_state)

        if result_state is None:
            print(f"[WARN][A*-E] Slice {i} rejected.\n")
            continue

        # Commit CPU
        for vnf_id, node in result_state.placed_vnfs.items():
            vnf_cpu = next(v["cpu"] for v in vnf_chain if v["id"] == vnf_id)
            node_capacity_global[node] -= vnf_cpu

        # Commit bandwidth (same pattern as your original)
        for (src, dst), path in result_state.routed_vls.items():
            if src == "ENTRY":
                continue
            bw = next(vl["bandwidth"] for vl in vl_chain if vl["from"] == src and vl["to"] == dst)
            for u, v in zip(path[:-1], path[1:]):
                _dec_link_cap(link_capacity_global, u, v, bw)

        print(f"[SUMMARY][A*-E] Slice {i} accepted. "
              f"(lat_total={result_state.latency_sum:.3f}, energy_total={result_state.energy_sum:.6f}, "
              f"combined_cost={result_state.g_cost:.3f})\n")

    summary = [{
        "slice": idx,
        "accepted": res is not None,
        "g_cost": (res.g_cost if res else None),
        "latency_sum": (res.latency_sum if res else None),
        "energy_sum": (res.energy_sum if res else None),
    } for idx, res in enumerate(astar_results, start=1)]

    df_results = pd.DataFrame(summary)
    if csv_path:
        df_results.to_csv(csv_path, index=False)
    return df_results, astar_results
