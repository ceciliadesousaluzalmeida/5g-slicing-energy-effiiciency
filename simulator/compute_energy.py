from collections import defaultdict
from typing import Dict, Tuple, List, Any

def compute_energy_per_slice(
    results: List, 
    slices: List[Tuple[List[Dict], List[Dict]]], 
    node_capacity: Dict[int, int], 
    a: int = 5, 
    b: int = 2
) -> Tuple[List[int], List[Dict[int, int]]]:
    slice_energy = []
    slice_cpu_usage = []

    for slice_id, result in enumerate(results):
        if result:
            cpu_usage = {n: 0 for n in node_capacity}
            for vnf_id, node in result.placed_vnfs.items():
                vnf_index = int(vnf_id.split('_')[1])
                cpu = slices[slice_id][0][vnf_index]["cpu"]
                cpu_usage[node] += cpu
            energy = sum(a + b * cpu for cpu in cpu_usage.values() if cpu > 0)
        else:
            cpu_usage = {n: 0 for n in node_capacity}
            energy = None

        slice_energy.append(energy)
        slice_cpu_usage.append(cpu_usage)

    return slice_energy, slice_cpu_usage


def _is_milp_result(res: Any) -> bool:
    # Heurísticas não têm .values com chaves ("f", e, s, (i,j))
    return hasattr(res, "values") and isinstance(res.values, dict)

def _extract_milp_f_matrix(milp_res, instance) -> Dict[int, Dict[Tuple[int,int], float]]:
    """
    Build, for each slice s, the aggregated per-edge usage:
    F_s_e[e] = sum_{(i,j) in chain(s)} f[(e, s, (i, j))]
    """
    F_s_e: Dict[int, Dict[Tuple[int,int], float]] = {s: defaultdict(float) for s in instance.S}
    for key, val in milp_res.values.items():
        # key format for f: ("f", e, s, (i,j))
        if not (isinstance(key, tuple) and len(key) == 4 and key[0] == "f"):
            continue
        e, s, ij = key[1], key[2], key[3]
        if val > 1e-9:
            F_s_e[s][e] += val  # val is 0/1 here
    return F_s_e

def compute_milp_bandwidth_latency(milp_res, instance):
    """
    Compute total bandwidth and latency usage for MILP results.
    Based on variables f[(e,s,(i,j))] from Gurobi.
    """
    values = milp_res.values
    per_link_bw = {e: 0.0 for e in instance.E}
    per_slice_bw = {s: 0.0 for s in instance.S}
    per_slice_lat = {s: 0.0 for s in instance.S}

    for key, val in values.items():
        # Only handle f variables that are active
        if not (isinstance(key, tuple) and len(key) == 4 and key[0] == "f"):
            continue
        e, s, (i, j) = key[1], key[2], key[3]
        if val > 0.5:
            bw = instance.BW_s[s]
            per_link_bw[e] += bw
            per_slice_bw[s] += bw
            per_slice_lat[s] += instance.lat_e[e]

    total_bw = sum(per_link_bw.values())
    total_lat = sum(per_slice_lat.values())

    return total_bw, total_lat, per_slice_bw, per_slice_lat



def compute_total_bandwidth(results: List, slices: List[Tuple[List, List]], instance=None, link_capacity: Dict[Tuple[int,int], float]=None) -> List[float]:
    """
    Backward-compatible wrapper:
      - For MILP results: returns bandwidth footprint per slice (traffic * hops)
      - For heuristics: sums bandwidth * path_length across VLs
    """
    totals: List[float] = []
    # If this is a single MILP result covering all slices:
    if len(results) == 1 and _is_milp_result(results[0]) and instance is not None:
        bw_fp, _, _ = compute_milp_bandwidth_latency(results[0], instance)
        # Ensure ordering by slice index the same as 'slices' list
        for s_idx in range(len(slices)):
            totals.append(bw_fp.get(s_idx, None))
        return totals

    # Heuristic path-based fallback (A*, ABO, FABO)
    for slice_id, result in enumerate(results):
        if not result:
            totals.append(None)
            continue
        used = 0.0
        vl_chain = slices[slice_id][1]
        # Sum bandwidth * number of hops for each routed VL
        for vl in vl_chain:
            path = result.routed_vls.get((vl["from"], vl["to"])) if hasattr(result, "routed_vls") else None
            if path:
                used += vl["bandwidth"] * max(0, len(path) - 1)  # bandwidth-footprint for this VL
        totals.append(used)
    return totals

def compute_total_latency(results: List, link_latency: Dict[Tuple[int, int], float], instance=None) -> List[float]:
    """
    Backward-compatible wrapper:
      - For MILP results: sums lat_e[e] * F_s_e[e] across all VLs of the slice
      - For heuristics: sum of per-link latencies along each routed VL path
    """
    totals: List[float] = []
    if len(results) == 1 and _is_milp_result(results[0]) and instance is not None:
        _, lat, _ = compute_milp_bandwidth_latency(results[0], instance)
        for s_idx in range(len(instance.S)):
            totals.append(lat.get(s_idx, None))
        return totals

    # Heuristic path-based fallback
    for res in results:
        if not res:
            totals.append(None)
            continue
        s_lat = 0.0
        if hasattr(res, "routed_vls"):
            for path in res.routed_vls.values():
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    key = (u, v) if (u, v) in link_latency else (v, u)
                    s_lat += link_latency.get(key, 0.0)
        totals.append(s_lat)
    return totals

def compute_energy_per_node(results, slices, node_capacity, a=5, b=2):
    node_energy = {node: 0 for node in node_capacity}

    for slice_id, result in enumerate(results):
        if result:
            for vnf_id, node in result.placed_vnfs.items():
                vnf_index = int(vnf_id.split('_')[1])
                cpu = slices[slice_id][0][vnf_index]["cpu"]
                node_energy[node] += cpu

    for node in node_energy:
        if node_energy[node] > 0:
            cpu_used = node_energy[node]
            node_energy[node] = a + b * cpu_used
        else:
            node_energy[node] = 0

    total_energy = sum(node_energy.values())
    return node_energy, total_energy



def compute_energy(placed_vnfs: dict, routed_vls: dict, slices: list) -> float:
    """
    Computes the total energy consumption based on CPU usage and routing hops.
    
    Parameters:
    - placed_vnfs: dict mapping VNF ids to node ids where they were placed
    - routed_vls: dict mapping (src_vnf_id, dst_vnf_id) to list of edges used in routing
    - slices: list of tuples (vnf_chain, vl_chain), where:
        - vnf_chain is a list of VNFs with attributes: {"id": ..., "cpu": ...}
        - vl_chain is a list of VLs with attributes: {"from": ..., "to": ...}
    
    Returns:
    - energy_total: float, the sum of CPU usage and total number of hops used in routing
    """
    
    # Map VNF id to its CPU requirement
    vnf_cpu_map = {
        vnf["id"]: vnf["cpu"]
        for vnf_chain, _ in slices
        for vnf in vnf_chain
    }

    # Compute total CPU energy (sum of CPU used for all placed VNFs)
    energy_cpu = sum(
        vnf_cpu_map[vnf_id] for vnf_id in placed_vnfs if vnf_id in vnf_cpu_map
    )

    # Compute routing energy as total number of hops (edges used)
    energy_routing = sum(len(path) for path in routed_vls.values())

    # Total energy = CPU + Routing (can be weighted if needed)
    energy_total = energy_cpu + energy_routing

    return energy_total


def compute_routing_energy_weighted(G, routed_vls: dict) -> float:
    routing_energy = 0.0

    for path in routed_vls.values():
        # Se o path for uma lista de nós, converte para lista de arestas
        if all(isinstance(node, int) for node in path):
            path = list(zip(path[:-1], path[1:]))

        for u, v in path:
            if G.has_edge(u, v):
                latency = G[u][v].get("latency", 1)
                bandwidth = G[u][v].get("bandwidth", 1)
            elif G.has_edge(v, u):
                latency = G[v][u].get("latency", 1)
                bandwidth = G[v][u].get("bandwidth", 1)
            else:
                continue

            if bandwidth == 0:
                continue

            routing_energy += latency / bandwidth

    return routing_energy



def compute_total_energy(cpu_allocations: dict, routed_vls: dict, slices: list, G) -> float:
    """
    Computes total energy: CPU + routing energy
    """
    vnf_cpu_map = {
        vnf["id"]: vnf["cpu"]
        for vnf_chain, _ in slices
        for vnf in vnf_chain
    }

    cpu_energy = sum(
        vnf_cpu_map[vnf_id] for vnf_id in cpu_allocations if vnf_id in vnf_cpu_map
    )

    routing_energy = compute_routing_energy_weighted(G, routed_vls)

    return cpu_energy + routing_energy




def compute_total_energy_with_routing(
    result,
    slices,
    node_capacity,
    G,
    a=5,
    b=2
) -> float:
    if not result:
        return None

    # CPU energy
    node_energy = {node: 0 for node in node_capacity}
    slice_id = None

    for idx, (vnf_chain, _) in enumerate(slices):
        if all(vnf["id"] in result.placed_vnfs for vnf in vnf_chain):
            slice_id = idx
            break

    if slice_id is None:
        return None

    for vnf_id, node in result.placed_vnfs.items():
        vnf_index = int(vnf_id.split('_')[1])
        cpu = slices[slice_id][0][vnf_index]["cpu"]
        node_energy[node] += cpu

    energy_cpu = sum(a + b * usage for usage in node_energy.values() if usage > 0)

    # Routing energy
    energy_routing = compute_routing_energy_weighted(G, result.routed_vls)

    return energy_cpu + energy_routing


# All comments in English
# All comments in English
def compute_energy_new(result_list, slices, node_capacity_base, link_capacity_base):
    """
    Generic normalized energy model:
    - Node energy = 1 (if active) + utilization fraction
    - Link energy = utilization fraction
    Works with slices = list of (vnfs, vls).
    """
    node_usage = {n: 0 for n in node_capacity_base}
    link_usage = {e: 0 for e in link_capacity_base}

    for state in result_list:
        if state is None:
            continue

        # VNFs: placed_vnfs = {vnf_id: node_id}
        for vnf_id, node in getattr(state, "placed_vnfs", {}).items():
            # find vnf info
            for vnfs, _ in slices:
                for v in vnfs:
                    if v["id"] == vnf_id:
                        node_usage[node] += v["cpu"]

        # VLs: routed_vls = {vl_id: path_edges}
        for vl_id, path in getattr(state, "routed_vls", {}).items():
            # find vl info
            for _, vls in slices:
                for vl in vls:
                    # identify by from->to string
                    if f"{vl['from']}->{vl['to']}" == vl_id:
                        bw_req = vl["bandwidth"]
                        for e in path:
                            link_usage[e] += bw_req

    # Normalized node energy
    total_energy = 0.0
    for n, used in node_usage.items():
        cap = node_capacity_base[n]
        if used > 0:
            total_energy += 1.0
            total_energy += used / cap

    # Normalized link energy
    for e, used in link_usage.items():
        cap = link_capacity_base[e]
        if used > 0:
            total_energy += 1.0
            total_energy += used / cap

    return total_energy
