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

def compute_energy_new(result_list, slices, node_capacity_base, link_capacity_base):
    """
    Generic normalized energy model:
    - Node energy = 1 (if active) + utilization fraction
    - Link energy = utilization fraction
    Works with slices = list of (vnfs, vls) or (vnfs, vls, entry, exit_).
    """
    node_usage = {n: 0 for n in node_capacity_base}
    link_usage = {e: 0 for e in link_capacity_base}

    for state in result_list:
        if state is None:
            continue

        # --- VNFs: placed_vnfs = {vnf_id: node_id} ---
        for vnf_id, node in getattr(state, "placed_vnfs", {}).items():
            for slice_data in slices:
                # Support both 2-tuple and 4-tuple slice structures
                vnfs = slice_data[0] if len(slice_data) >= 1 else []
                for v in vnfs:
                    if v["id"] == vnf_id:
                        node_usage[node] += v["cpu"]

        # --- VLs: routed_vls = {vl_id or (src,to): path_edges} ---
        for vl_key, path in getattr(state, "routed_vls", {}).items():
            for slice_data in slices:
                vls = slice_data[1] if len(slice_data) >= 2 else []
                for vl in vls:
                    # Support both string and tuple identifiers
                    if (
                        isinstance(vl_key, tuple)
                        and len(vl_key) == 2
                        and vl_key == (vl["from"], vl["to"])
                    ) or (
                        isinstance(vl_key, str)
                        and vl_key == f"{vl['from']}->{vl['to']}"
                    ):
                        bw_req = vl["bandwidth"]
                        # Add bandwidth usage on each edge
                        for i in range(len(path) - 1):
                            e = (path[i], path[i + 1])
                            e = e if e in link_capacity_base else (e[1], e[0])
                            if e in link_usage:
                                link_usage[e] += bw_req

    # --- Normalized node energy ---
    total_energy = 0.0
    for n, used in node_usage.items():
        cap = node_capacity_base[n]
        if used > 0:
            total_energy += 1.0  # base activation
            total_energy += used / cap  # proportional usage

    # --- Normalized link energy ---
    for e, used in link_usage.items():
        cap = link_capacity_base[e]
        if used > 0:
            total_energy += 1.0
            total_energy += used / cap

    return total_energy



def _is_vl_routed(result, slice_idx, src, dst):
    """
    Check if a VL (src->dst) is routed in 'result', accepting different key styles:
    - (src, dst)
    - (dst, src)                # reversed order
    - (slice_idx, (src, dst))
    - (slice_idx, (dst, src))
    Accepts either a path (list of nodes) or a list of edges (u,v).
    Works with both CBC and Gurobi adapters.
    """
    if not result or not getattr(result, "routed_vls", None):
        return False

    # direct heuristic keys
    if (src, dst) in result.routed_vls:
        val = result.routed_vls[(src, dst)]
        return bool(val)
    if (dst, src) in result.routed_vls:
        val = result.routed_vls[(dst, src)]
        return bool(val)

    # MILP adapter style
    key1 = (slice_idx, (src, dst))
    key2 = (slice_idx, (dst, src))
    if key1 in result.routed_vls:
        return bool(result.routed_vls[key1])
    if key2 in result.routed_vls:
        return bool(result.routed_vls[key2])

    return False



def count_accepted_slices(results, slices, eps=1e-6, verbose=False):
    """
    Count accepted slices:
      - All VNFs placed
      - All VLs routed

    Works with:
      - slices = [(vnf_chain, vl_chain)] or [(vnf_chain, vl_chain, entry, exit_)]
      - heuristics & MILP-iterative: list with one result per slice (or None)
      - MILP aggregated adapter: single result with routed_vls possibly keyed by (s_idx,(src,dst))
      - MILP raw: single result with .values (fallback support)
    """
    if not results:
        return 0

    # --- Helpers ---------------------------------------------------------

    def _get_vnfs_vls(slice_data):
        """Return (vnfs, vls) for 2- or 4-tuple slice formats."""
        if isinstance(slice_data, (list, tuple)):
            if len(slice_data) >= 2:
                return slice_data[0], slice_data[1]
            elif len(slice_data) == 1:
                return slice_data[0], []
        return [], []

    def _is_entry_exit_leg(key):
        """Detect ENTRY/EXIT special legs (we ignore them for acceptance criteria)."""
        if isinstance(key, tuple):
            return (len(key) == 2) and ("ENTRY" in key or "EXIT" in key)
        if isinstance(key, str):
            return key.startswith("ENTRY->") or key.endswith("->EXIT")
        return False

    def _vl_present(routed_vls, s_idx, src, dst):
        """
        Check multiple key conventions in routed_vls:
         - (src, dst), (dst, src)
         - (s_idx, (src, dst)), (s_idx, (dst, src))
         - "src->dst"
        ENTRY/EXIT legs are ignored.
        """
        # direct tuple keys
        if (src, dst) in routed_vls or (dst, src) in routed_vls:
            return True

        # per-slice keyed tuples
        key1 = (s_idx, (src, dst))
        key2 = (s_idx, (dst, src))
        if key1 in routed_vls or key2 in routed_vls:
            return True

        # string keys
        key_s = f"{src}->{dst}"
        key_s_rev = f"{dst}->{src}"
        if key_s in routed_vls or key_s_rev in routed_vls:
            return True

        return False

    # --------------------------------------------------------------------
    # Detect result flavor
    r0 = results[0]

    # Case C: MILP raw (.values)
    if hasattr(r0, "values") and isinstance(r0.values, dict):
        var_dict = r0.values
        accepted = 0
        for s_idx in range(len(slices)):
            vnfs, vls = _get_vnfs_vls(slices[s_idx])

            # VNFs placed?
            vnfs_ok = True
            for v in vnfs:
                i = v["id"]
                assigned = any(
                    isinstance(k, tuple) and len(k) >= 3 and k[0] == "v" and k[1] == i and val > 0.5
                    for k, val in var_dict.items()
                )
                if not assigned:
                    vnfs_ok = False
                    if verbose:
                        print(f"[MILP raw] slice {s_idx}: VNF {i} not allocated")
                    break

            # VLs routed?
            vls_ok = True
            if vnfs_ok:
                for vl in vls:
                    src, dst = vl["from"], vl["to"]
                    routed = any(
                        isinstance(k, tuple)
                        and len(k) >= 4
                        and k[0] == "f"
                        and k[2] == s_idx
                        and set(k[3]) == {src, dst}
                        and val > 0.5
                        for k, val in var_dict.items()
                    )
                    if not routed:
                        vls_ok = False
                        if verbose:
                            print(f"[MILP raw] slice {s_idx}: VL ({src}->{dst}) not routed")
                        break

            if vnfs_ok and vls_ok:
                accepted += 1
                if verbose:
                    print(f"[MILP raw] slice {s_idx}: ACCEPTED ✓")

        if verbose:
            print(f"Total slices accepted (MILP raw): {accepted}/{len(slices)}")
        return accepted

    # Case B: MILP aggregated adapter (single result), where routed_vls keys may include s_idx
    is_milp_aggregated = (
        len(results) == 1
        and hasattr(r0, "routed_vls")
        and isinstance(getattr(r0, "routed_vls", {}), dict)
        and any(
            (isinstance(k, tuple) and len(k) >= 2 and isinstance(k[0], int))  # e.g., (s_idx, (src,dst))
            or (isinstance(k, tuple) and len(k) == 2)                          # or plain (src,dst)
            or isinstance(k, str)                                              # or "src->dst"
            for k in r0.routed_vls.keys()
        )
    )

    if is_milp_aggregated:
        accepted = 0
        placed_vnfs = getattr(r0, "placed_vnfs", {})
        routed_vls = {
            k: v for k, v in getattr(r0, "routed_vls", {}).items()
            if not _is_entry_exit_leg(k)  # ignore ENTRY/EXIT legs
        }

        for s_idx in range(len(slices)):
            vnfs, vls = _get_vnfs_vls(slices[s_idx])

            vnfs_ok = all(v["id"] in placed_vnfs for v in vnfs)
            if not vnfs_ok and verbose:
                missing = [v["id"] for v in vnfs if v["id"] not in placed_vnfs]
                print(f"[MILP adapter] slice {s_idx}: missing VNFs {missing}")

            vls_ok = True
            if vnfs_ok:
                for vl in vls:
                    src, dst = vl["from"], vl["to"]
                    if not _vl_present(routed_vls, s_idx, src, dst):
                        vls_ok = False
                        if verbose:
                            print(f"[MILP adapter] slice {s_idx}: VL ({src}->{dst}) not routed")
                        break

            if vnfs_ok and vls_ok:
                accepted += 1
                if verbose:
                    print(f"[MILP adapter] slice {s_idx}: ACCEPTED ✓")

        if verbose:
            print(f"Total slices accepted (MILP adapter): {accepted}/{len(slices)}")
        return accepted

    # Case A: heurísticas e MILP iterativo (um resultado por slice)
    accepted = 0
    # results pode ser menor que len(slices) se parou no meio; por isso iteramos sobre índices
    for s_idx in range(len(slices)):
        result = results[s_idx] if s_idx < len(results) else None
        if not result or not hasattr(result, "placed_vnfs") or not hasattr(result, "routed_vls"):
            if verbose:
                print(f"[Heuristic/Iter MILP] slice {s_idx}: no result or missing attributes")
            continue

        vnfs, vls = _get_vnfs_vls(slices[s_idx])

        # VNFs
        vnfs_ok = all(v["id"] in result.placed_vnfs for v in vnfs)
        if not vnfs_ok:
            if verbose:
                missing = [v["id"] for v in vnfs if v["id"] not in result.placed_vnfs]
                print(f"[Heuristic/Iter MILP] slice {s_idx}: missing VNFs {missing}")
            continue

        # VLs (ignora ENTRY/EXIT)
        routed_vls = {
            k: v for k, v in getattr(result, "routed_vls", {}).items()
            if not _is_entry_exit_leg(k)
        }
        vls_ok = all(
            _vl_present(routed_vls, s_idx, vl["from"], vl["to"])
            for vl in vls
        )
        if not vls_ok and verbose:
            missing = [
                (vl["from"], vl["to"]) for vl in vls
                if not _vl_present(routed_vls, s_idx, vl["from"], vl["to"])
            ]
            print(f"[Heuristic/Iter MILP] slice {s_idx}: missing VLs {missing}")

        if vnfs_ok and vls_ok:
            accepted += 1
            if verbose:
                print(f"[Heuristic/Iter MILP] slice {s_idx}: ACCEPTED ✓")

    if verbose:
        print(f"Total slices accepted (heuristics/iterative): {accepted}/{len(slices)}")
    return accepted
