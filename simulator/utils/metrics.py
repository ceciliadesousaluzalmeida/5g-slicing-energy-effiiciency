from collections import defaultdict
from turtle import pd
from turtle import pd
from typing import Dict, Tuple, List, Any, Optional


ENTRY_VNF_ID = "__ENTRY__"


# =========================================================
# Generic helpers
# =========================================================
def _split_slice(slice_data):
    """
    Return (vnfs, vls, entry) from supported slice formats:
      - (vnfs, vls)
      - (vnfs, vls, entry)
    """
    if not isinstance(slice_data, (list, tuple)):
        return [], [], None

    if len(slice_data) == 2:
        return slice_data[0], slice_data[1], None

    if len(slice_data) >= 3:
        return slice_data[0], slice_data[1], slice_data[2]

    return [], [], None


def _first_vnf_id(vnfs):
    if not vnfs:
        return None
    return vnfs[0]["id"]


def _infer_entry_bandwidth(vnfs, vls):
    """
    Infer the ingress bandwidth from the first outgoing VL of the first VNF.
    """
    first_id = _first_vnf_id(vnfs)
    if first_id is None:
        return 0.0

    for vl in vls:
        if vl["from"] == first_id:
            return float(vl["bandwidth"])
    return 0.0


def _infer_entry_latency(vnfs, vls):
    """
    Infer the ingress latency budget from the first outgoing VL of the first VNF.
    """
    first_id = _first_vnf_id(vnfs)
    if first_id is None:
        return 0.0

    for vl in vls:
        if vl["from"] == first_id:
            return float(vl.get("latency", 0.0))
    return 0.0


def _effective_vls_for_metrics(slice_data):
    """
    Return the effective VL list used for metrics.
    If the slice has an entry node, prepend ENTRY -> first_vnf.
    """
    vnfs, vls, entry = _split_slice(slice_data)
    effective = list(vls)

    if entry is not None and vnfs:
        effective.insert(
            0,
            {
                "from": ENTRY_VNF_ID,
                "to": _first_vnf_id(vnfs),
                "bandwidth": _infer_entry_bandwidth(vnfs, vls),
                "latency": _infer_entry_latency(vnfs, vls),
            },
        )

    return effective


def _build_vnf_cpu_map(slices):
    """
    Build a global VNF -> CPU lookup from slices.
    """
    cpu_map = {}
    for slice_data in slices:
        vnfs, _, _ = _split_slice(slice_data)
        for v in vnfs:
            cpu_map[v["id"]] = float(v["cpu"])
    return cpu_map


def _get_routed_path(result, slice_idx, src, dst):
    """
    Retrieve a routed path from different possible key conventions.
    """
    if result is None or not hasattr(result, "routed_vls"):
        return None

    routed_vls = getattr(result, "routed_vls", {})

    candidates = [
        (src, dst),
        (dst, src),
        (slice_idx, (src, dst)),
        (slice_idx, (dst, src)),
        f"{src}->{dst}",
        f"{dst}->{src}",
    ]

    for key in candidates:
        if key in routed_vls:
            return routed_vls[key]

    return None


def _is_milp_result(res: Any) -> bool:
    return hasattr(res, "values") and isinstance(res.values, dict)


def _canon_edge(u, v):
    return (u, v) if (u, v) <= (v, u) else (v, u)

def _get_vlink_bw(sl, src, dst):

    for vl in _effective_vls_for_metrics(sl):

        if vl["from"] == src and vl["to"] == dst:
            return float(vl["bandwidth"])

    return None
# =========================================================
# MILP metrics
# =========================================================
def compute_milp_bandwidth_latency(milp_res, instance):
    """
    Compute per-slice and per-link bandwidth/latency from MILP results.

    Expected MILP flow key format:
      ("f", s, i, j, u, v)
    """
    var_dict = getattr(milp_res, "values", {})

    per_slice_bw = {s: 0.0 for s in instance.S}
    per_slice_lat = {s: 0.0 for s in instance.S}
    per_link_bw = defaultdict(float)
    per_link_lat = defaultdict(float)

    for key, val in var_dict.items():
        if not isinstance(key, tuple):
            continue

        if len(key) == 6 and key[0] == "f" and val > 0.5:
            _, s, i, j, u, v = key

            bw = float(getattr(instance, "BW_sij", {}).get((s, i, j), 0.0))

            lat = 0.0
            if (u, v) in getattr(instance, "lat_e", {}):
                lat = float(instance.lat_e[(u, v)])
            elif (v, u) in getattr(instance, "lat_e", {}):
                lat = float(instance.lat_e[(v, u)])

            per_slice_bw[s] += bw
            per_slice_lat[s] += lat
            per_link_bw[(u, v)] += bw
            per_link_lat[(u, v)] += lat

    return per_slice_bw, per_slice_lat, dict(per_link_bw), dict(per_link_lat)


# =========================================================
# Bandwidth / latency wrappers
# =========================================================
def compute_total_bandwidth(
    results: List,
    slices: List,
    instance=None,
    link_capacity: Dict[Tuple[int, int], float] = None,
) -> List[Optional[float]]:
    """
    Return bandwidth footprint per slice.

    For MILP:
      - sum of BW_sij over all used directed arcs
    For heuristics:
      - sum over routed VLs of bandwidth * hop_count
      - includes ENTRY -> first_vnf when entry exists
    """
    totals: List[Optional[float]] = []

    if len(results) == 1 and _is_milp_result(results[0]) and instance is not None:
        per_slice_bw, _, _, _ = compute_milp_bandwidth_latency(results[0], instance)
        for s_idx in range(len(slices)):
            totals.append(per_slice_bw.get(s_idx, None))
        return totals

    for slice_idx, result in enumerate(results):
        if not result:
            totals.append(None)
            continue

        used = 0.0
        effective_vls = _effective_vls_for_metrics(slices[slice_idx])

        for vl in effective_vls:
            src, dst = vl["from"], vl["to"]
            path = _get_routed_path(result, slice_idx, src, dst)
            if path:
                used += float(vl["bandwidth"]) * max(0, len(path) - 1)

        totals.append(used)

    return totals


def compute_total_latency(
    results: List,
    slices: List,
    link_latency: Dict[Tuple[int, int], float],
    instance=None,
) -> List[Optional[float]]:
    """
    Return total routing latency per slice.

    For MILP:
      - sum of lat_e over all used directed arcs
    For heuristics:
      - sum of per-edge latencies across routed paths
      - includes ENTRY -> first_vnf when entry exists
    """
    totals: List[Optional[float]] = []

    if len(results) == 1 and _is_milp_result(results[0]) and instance is not None:
        _, per_slice_lat, _, _ = compute_milp_bandwidth_latency(results[0], instance)

        for s_idx in range(len(instance.S)):
            totals.append(per_slice_lat.get(s_idx, None))

        return totals

    for slice_idx, result in enumerate(results):
        if not result:
            totals.append(None)
            continue

        if slice_idx >= len(slices):
            totals.append(None)
            continue

        s_lat = 0.0
        effective_vls = _effective_vls_for_metrics(slices[slice_idx])

        for vl in effective_vls:
            src, dst = vl["from"], vl["to"]
            path = _get_routed_path(result, slice_idx, src, dst)

            if not path:
                continue

            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                key = (u, v) if (u, v) in link_latency else (v, u)
                s_lat += float(link_latency.get(key, 0.0))

        totals.append(s_lat)

    return totals


# =========================================================
# CPU / node energy metrics
# =========================================================
def compute_energy_per_slice(
    results: List,
    slices: List,
    node_capacity: Dict[int, int],
    a: int = 5,
    b: int = 2,
) -> Tuple[List[Optional[float]], List[Dict[int, float]]]:
    """
    Per-slice node energy:
      energy = sum(a + b * cpu_used_on_node) for active nodes in that slice
    """
    slice_energy = []
    slice_cpu_usage = []

    cpu_map = _build_vnf_cpu_map(slices)

    for result in results:
        if result:
            cpu_usage = {n: 0.0 for n in node_capacity}
            for vnf_id, node in getattr(result, "placed_vnfs", {}).items():
                if vnf_id == ENTRY_VNF_ID:
                    continue
                cpu_usage[node] += float(cpu_map.get(vnf_id, 0.0))

            energy = sum(a + b * cpu for cpu in cpu_usage.values() if cpu > 0)
        else:
            cpu_usage = {n: 0.0 for n in node_capacity}
            energy = None

        slice_energy.append(energy)
        slice_cpu_usage.append(cpu_usage)

    return slice_energy, slice_cpu_usage


def compute_energy_per_node(results, slices, node_capacity, a=5, b=2):
    """
    Aggregate node energy across all slices.
    """
    node_cpu = {node: 0.0 for node in node_capacity}
    cpu_map = _build_vnf_cpu_map(slices)

    for result in results:
        if not result:
            continue

        for vnf_id, node in getattr(result, "placed_vnfs", {}).items():
            if vnf_id == ENTRY_VNF_ID:
                continue
            node_cpu[node] += float(cpu_map.get(vnf_id, 0.0))

    node_energy = {}
    for node, cpu_used in node_cpu.items():
        if cpu_used > 0:
            node_energy[node] = a + b * cpu_used
        else:
            node_energy[node] = 0.0

    total_energy = sum(node_energy.values())
    return node_energy, total_energy


# =========================================================
# Legacy-style energy metrics
# =========================================================
def compute_energy(placed_vnfs: dict, routed_vls: dict, slices: list) -> float:
    """
    Legacy total energy:
      - CPU = sum of placed VNF cpu
      - Routing = total number of routed edges
    """
    vnf_cpu_map = _build_vnf_cpu_map(slices)

    energy_cpu = sum(
        float(vnf_cpu_map[vnf_id])
        for vnf_id in placed_vnfs
        if vnf_id in vnf_cpu_map and vnf_id != ENTRY_VNF_ID
    )

    energy_routing = 0.0
    for path in routed_vls.values():
        if isinstance(path, list) and len(path) >= 2:
            energy_routing += max(0, len(path) - 1)

    return energy_cpu + energy_routing


def compute_routing_energy_weighted(G, routed_vls: dict) -> float:
    """
    Weighted routing energy = sum(latency / bandwidth) over used edges.
    """
    routing_energy = 0.0

    for path in routed_vls.values():
        if not isinstance(path, list) or len(path) < 2:
            continue

        # Convert node-path to edge traversal
        if all(not isinstance(x, tuple) for x in path):
            edges = list(zip(path[:-1], path[1:]))
        else:
            edges = path

        for u, v in edges:
            if G.has_edge(u, v):
                latency = float(G[u][v].get("latency", 1.0))
                bandwidth = float(G[u][v].get("bandwidth", 1.0))
            elif G.has_edge(v, u):
                latency = float(G[v][u].get("latency", 1.0))
                bandwidth = float(G[v][u].get("bandwidth", 1.0))
            else:
                continue

            if bandwidth > 0:
                routing_energy += latency / bandwidth

    return routing_energy


def compute_total_energy(cpu_allocations: dict, routed_vls: dict, slices: list, G) -> float:
    """
    Legacy total energy:
      CPU energy + weighted routing energy
    """
    vnf_cpu_map = _build_vnf_cpu_map(slices)

    cpu_energy = sum(
        float(vnf_cpu_map[vnf_id])
        for vnf_id in cpu_allocations
        if vnf_id in vnf_cpu_map and vnf_id != ENTRY_VNF_ID
    )

    routing_energy = compute_routing_energy_weighted(G, routed_vls)
    return cpu_energy + routing_energy


def compute_total_energy_with_routing(
    result,
    slices,
    node_capacity,
    G,
    a=5,
    b=2,
) -> Optional[float]:
    """
    Per-result total energy:
      node energy + weighted routing energy
    """
    if not result:
        return None

    cpu_map = _build_vnf_cpu_map(slices)
    node_cpu = {node: 0.0 for node in node_capacity}

    for vnf_id, node in getattr(result, "placed_vnfs", {}).items():
        if vnf_id == ENTRY_VNF_ID:
            continue
        node_cpu[node] += float(cpu_map.get(vnf_id, 0.0))

    energy_cpu = sum(a + b * usage for usage in node_cpu.values() if usage > 0)
    energy_routing = compute_routing_energy_weighted(G, getattr(result, "routed_vls", {}))

    return energy_cpu + energy_routing


# =========================================================
# Normalized energy model
# =========================================================
def compute_energy_new(result_list, slices, node_capacity_base, link_capacity_base):
    node_usage = {n: 0.0 for n in node_capacity_base}
    link_usage = {e: 0.0 for e in link_capacity_base}

    cpu_map = _build_vnf_cpu_map(slices)

    for s_idx, state in enumerate(result_list):
        if state is None:
            continue

        for vnf_id, node in getattr(state, "placed_vnfs", {}).items():
            if vnf_id == ENTRY_VNF_ID:
                continue

            cpu = None

            if isinstance(vnf_id, tuple) and len(vnf_id) == 2:
                cpu = cpu_map.get(vnf_id[1])

            if cpu is None:
                cpu = cpu_map.get(vnf_id)

            if cpu is not None:
                node_usage[node] += float(cpu)

        effective_vls = _effective_vls_for_metrics(slices[s_idx]) if s_idx < len(slices) else []

        bw_by_vl = {
            (vl["from"], vl["to"]): float(vl.get("bandwidth", 0.0))
            for vl in effective_vls
        }

        for vl_key, path in getattr(state, "routed_vls", {}).items():
            if not path or len(path) < 2:
                continue

            if isinstance(vl_key, tuple) and len(vl_key) == 3:
                _, src, dst = vl_key
            elif isinstance(vl_key, tuple) and len(vl_key) == 2:
                src, dst = vl_key
            else:
                continue

            bw = bw_by_vl.get((src, dst), bw_by_vl.get((dst, src), 0.0))

            for u, v in zip(path[:-1], path[1:]):
                e = (u, v) if (u, v) in link_usage else (v, u)
                if e in link_usage:
                    link_usage[e] += bw

    total_energy = 0.0

    for n, used in node_usage.items():
        cap = float(node_capacity_base[n])
        if used > 0 and cap > 0:
            total_energy += 1.0 + used / cap

    for e, used in link_usage.items():
        cap = float(link_capacity_base[e])
        if used > 0 and cap > 0:
            total_energy += 1.0 + used / cap

    return total_energy

# =========================================================
# Acceptance
# =========================================================
def _is_entry_exit_leg(key):
    if isinstance(key, tuple):
        return (len(key) == 2) and ("ENTRY" in key or "EXIT" in key or ENTRY_VNF_ID in key)
    if isinstance(key, str):
        return key.startswith("ENTRY->") or key.endswith("->EXIT") or key.startswith(f"{ENTRY_VNF_ID}->")
    return False


def _vl_present(routed_vls, s_idx, src, dst):
    """
    Check multiple key conventions in routed_vls.
    """
    candidates = [
        (src, dst),
        (dst, src),
        (s_idx, (src, dst)),
        (s_idx, (dst, src)),
        f"{src}->{dst}",
        f"{dst}->{src}",
    ]
    for key in candidates:
        if key in routed_vls:
            return True
    return False


def count_accepted_slices(results, slices, eps=1e-6, verbose=False):
    """
    Count accepted slices:
      - all real VNFs placed
      - all real VLs routed

    ENTRY legs are ignored in the acceptance criterion.
    """
    if not results:
        return 0

    r0 = results[0]

    # -----------------------------------------------------
    # Case 1: MILP raw or adapted MILP result
    # -----------------------------------------------------
    if len(results) == 1 and _is_milp_result(r0):
        accepted = 0
        values = getattr(r0, "values", {})

        # Placement map from ("x", s, i, n)
        placed_by_slice = defaultdict(set)
        routed_by_slice = defaultdict(set)

        for key, val in values.items():
            if not isinstance(key, tuple):
                continue

            if len(key) == 4 and key[0] == "x" and val > 0.5:
                _, s, i, _n = key
                placed_by_slice[s].add(i)

            if len(key) == 6 and key[0] == "f" and val > 0.5:
                _, s, i, j, _u, _v = key
                if i != "ENTRY" and j != "EXIT":
                    routed_by_slice[s].add((i, j))

        for s_idx, slice_data in enumerate(slices):
            vnfs, vls, _entry = _split_slice(slice_data)

            vnfs_ok = all(v["id"] in placed_by_slice[s_idx] for v in vnfs)
            vls_ok = all((vl["from"], vl["to"]) in routed_by_slice[s_idx] for vl in vls)

            if vnfs_ok and vls_ok:
                accepted += 1
                if verbose:
                    print(f"[MILP] slice {s_idx}: ACCEPTED ✓")
            elif verbose:
                print(f"[MILP] slice {s_idx}: REJECTED")

        if verbose:
            print(f"Total slices accepted (MILP): {accepted}/{len(slices)}")
        return accepted

    # -----------------------------------------------------
    # Case 2: heuristics / one-result-per-slice
    # -----------------------------------------------------
    accepted = 0

    for s_idx, slice_data in enumerate(slices):
        result = results[s_idx] if s_idx < len(results) else None
        if not result or not hasattr(result, "placed_vnfs") or not hasattr(result, "routed_vls"):
            if verbose:
                print(f"[Heuristic] slice {s_idx}: no result")
            continue

        vnfs, vls, _entry = _split_slice(slice_data)

        vnfs_ok = all(v["id"] in result.placed_vnfs for v in vnfs)
        if not vnfs_ok:
            if verbose:
                missing = [v["id"] for v in vnfs if v["id"] not in result.placed_vnfs]
                print(f"[Heuristic] slice {s_idx}: missing VNFs {missing}")
            continue

        routed_vls = {
            k: v for k, v in getattr(result, "routed_vls", {}).items()
            if not _is_entry_exit_leg(k)
        }

        vls_ok = all(_vl_present(routed_vls, s_idx, vl["from"], vl["to"]) for vl in vls)

        if vnfs_ok and vls_ok:
            accepted += 1
            if verbose:
                print(f"[Heuristic] slice {s_idx}: ACCEPTED ✓")
        elif verbose:
            missing = [
                (vl["from"], vl["to"])
                for vl in vls
                if not _vl_present(routed_vls, s_idx, vl["from"], vl["to"])
            ]
            print(f"[Heuristic] slice {s_idx}: missing VLs {missing}")

    if verbose:
        print(f"Total slices accepted (heuristics): {accepted}/{len(slices)}")

    return accepted


def compute_latency_margin(results, slices, link_latency):
    rows = []

    for s_idx, slice_data in enumerate(slices):
        result = results[s_idx] if s_idx < len(results) else None
        if not result:
            continue

        effective_vls = _effective_vls_for_metrics(slice_data)

        for vl in effective_vls:
            src, dst = vl["from"], vl["to"]
            sla = float(vl.get("latency", 0.0))

            path = _get_routed_path(result, s_idx, src, dst)
            if not path:
                continue

            observed = 0.0
            for u, v in zip(path[:-1], path[1:]):
                key = (u, v) if (u, v) in link_latency else (v, u)
                observed += float(link_latency.get(key, 0.0))

            margin = None
            violation = None

            if sla > 0:
                margin = (sla - observed) / sla
                violation = observed > sla

            rows.append({
                "slice": s_idx,
                "src": src,
                "dst": dst,
                "latency_sla": sla,
                "latency_observed": observed,
                "latency_margin": margin,
                "latency_violation": violation,
                "path_hops": max(0, len(path) - 1),
            })

    return pd.DataFrame(rows)