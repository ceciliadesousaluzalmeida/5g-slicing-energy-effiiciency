import os
import time
import random
import re
from copy import deepcopy
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd


# ============================
# Helpers
# ============================

def safe_filename(text: str) -> str:
    # All comments in English
    text = str(text).lower()
    text = text.replace(" ", "_")
    text = re.sub(r"[^a-z0-9_\-]", "", text)
    return text


def _edge_key(u, v):
    # All comments in English
    return (u, v) if u <= v else (v, u)


def _normalize_token(x):
    # All comments in English
    s = str(x).strip().lower()
    s = s.replace(" ", "")
    s = re.sub(r"[^a-z0-9_(),\-\[\]]", "", s)
    return s


def _route_key_to_parts(vl_key):
    # All comments in English
    if isinstance(vl_key, tuple) and len(vl_key) == 3:
        s, i, j = vl_key
        return s, i, j
    return None, None, None


# ============================
# Slice parsing helpers
# ============================

def set_entry_for_slice(slice_obj, entry_node):
    # All comments in English
    if isinstance(slice_obj, dict):
        new_sl = deepcopy(slice_obj)
        new_sl["entry"] = entry_node
        return new_sl

    if isinstance(slice_obj, tuple):
        items = list(slice_obj)
        if len(items) >= 3:
            items[2] = entry_node
            return tuple(items)
        return tuple(list(items) + [entry_node])

    if isinstance(slice_obj, list):
        items = list(slice_obj)
        if len(items) >= 3:
            items[2] = entry_node
            return items
        return items + [entry_node]

    return slice_obj


def get_entry_from_slice(slice_obj):
    # All comments in English
    if isinstance(slice_obj, dict):
        return slice_obj.get("entry")

    if isinstance(slice_obj, (tuple, list)) and len(slice_obj) >= 3:
        return slice_obj[2]

    return None


def assign_fixed_entry_per_slice(slices, entry_node=6):
    # All comments in English
    new_slices = []
    for sl in slices:
        new_slices.append(set_entry_for_slice(sl, entry_node))
    return new_slices


def _get_slice_id(slice_obj, default_idx):
    # All comments in English
    if isinstance(slice_obj, dict):
        return slice_obj.get("id", default_idx)
    return default_idx


def _extract_vnfs_from_slice(slice_obj):
    # All comments in English
    if isinstance(slice_obj, dict):
        for k in ["vnfs", "vnf_list", "functions", "chain", "vnf_chain", "vnf_sequence"]:
            vnfs = slice_obj.get(k)
            if isinstance(vnfs, list):
                return vnfs
        return []

    if isinstance(slice_obj, (tuple, list)):
        if len(slice_obj) >= 1 and isinstance(slice_obj[0], list):
            return slice_obj[0]
        for item in slice_obj:
            if isinstance(item, dict) and any(
                k in item for k in ["vnfs", "vnf_list", "functions", "chain", "vnf_chain"]
            ):
                return _extract_vnfs_from_slice(item)
        return []

    return []


def _extract_vls_from_slice(slice_obj):
    # All comments in English
    if isinstance(slice_obj, dict):
        for k in ["vls", "vl_chain", "virtual_links", "links", "edges"]:
            vls = slice_obj.get(k)
            if isinstance(vls, list):
                return vls
        return []

    if isinstance(slice_obj, (tuple, list)):
        if len(slice_obj) >= 2 and isinstance(slice_obj[1], list):
            return slice_obj[1]
        for item in slice_obj:
            if isinstance(item, dict) and any(
                k in item for k in ["vls", "vl_chain", "virtual_links", "links", "edges"]
            ):
                return _extract_vls_from_slice(item)
        return []

    return []


def _get_vnf_id(v):
    # All comments in English
    if isinstance(v, dict):
        return v.get("id") or v.get("vnf_id") or v.get("name") or v.get("label")
    if isinstance(v, (str, int)):
        return v
    for attr in ["id", "vnf_id", "name", "label"]:
        if hasattr(v, attr):
            return getattr(v, attr)
    return None


def _get_vnf_cpu(v):
    # All comments in English
    if isinstance(v, dict):
        return (
            v.get("cpu")
            or v.get("cpu_demand")
            or v.get("demand_cpu")
            or v.get("cpu_req")
        )
    if isinstance(v, (int, float)):
        return float(v)
    for attr in ["cpu", "cpu_demand", "demand_cpu", "cpu_req"]:
        if hasattr(v, attr):
            return getattr(v, attr)
    return None


def _vl_from_to(vl):
    # All comments in English
    if isinstance(vl, dict):
        return vl.get("from"), vl.get("to")
    return None, None


def _vl_bw(vl):
    # All comments in English
    if isinstance(vl, dict):
        return vl.get("bandwidth")
    return None


def _get_first_vnf_id_from_slice(slice_obj):
    # All comments in English
    vnfs = _extract_vnfs_from_slice(slice_obj)
    if not vnfs:
        return None
    return _get_vnf_id(vnfs[0])


def _infer_entry_bandwidth_from_slice(slice_obj):
    # All comments in English
    entry = get_entry_from_slice(slice_obj)
    if entry is None:
        return None

    first_vnf = _get_first_vnf_id_from_slice(slice_obj)
    vls = _extract_vls_from_slice(slice_obj)

    if first_vnf is None:
        return None

    for vl in vls:
        src, dst = _vl_from_to(vl)
        if str(src) == "ENTRY" and str(dst) == str(first_vnf):
            bw = _vl_bw(vl)
            return float(bw) if bw is not None else None

    for vl in vls:
        src, dst = _vl_from_to(vl)
        if str(src) == str(first_vnf):
            bw = _vl_bw(vl)
            return float(bw) if bw is not None else None

    if isinstance(slice_obj, dict) and "entry_bandwidth" in slice_obj:
        try:
            return float(slice_obj["entry_bandwidth"])
        except Exception:
            return None

    return None


def build_slice_metadata(slices):
    # All comments in English
    slice_cpu_map = {}
    slice_vl_bw_map = {}
    slice_vnf_ids = {}
    slice_entries = {}

    for s_idx, sl in enumerate(slices):
        s_id = _get_slice_id(sl, s_idx)
        vnfs = _extract_vnfs_from_slice(sl)
        vls = _extract_vls_from_slice(sl)

        slice_vnf_ids[s_id] = []
        slice_entries[s_id] = get_entry_from_slice(sl)

        for v in vnfs:
            vid = _get_vnf_id(v)
            cpu = _get_vnf_cpu(v)
            if vid is not None:
                slice_vnf_ids[s_id].append(vid)
                try:
                    slice_cpu_map[(s_id, vid)] = float(cpu)
                except Exception:
                    pass

        for vl in vls:
            src, dst = _vl_from_to(vl)
            bw = _vl_bw(vl)
            if src is None or dst is None or bw is None:
                continue
            try:
                slice_vl_bw_map[(s_id, src, dst)] = float(bw)
            except Exception:
                continue

        entry_bw = _infer_entry_bandwidth_from_slice(sl)
        first_vnf = _get_first_vnf_id_from_slice(sl)
        if slice_entries[s_id] is not None and first_vnf is not None and entry_bw is not None:
            slice_vl_bw_map[(s_id, "ENTRY", first_vnf)] = float(entry_bw)

    return {
        "slice_cpu_map": slice_cpu_map,
        "slice_vl_bw_map": slice_vl_bw_map,
        "slice_vnf_ids": slice_vnf_ids,
        "slice_entries": slice_entries,
    }


# ============================
# Result normalization
# ============================

class NormalizedResultView:
    """
    Canonical normalized view used only for:
      - exports
      - normalized energy model

    Canonical format:
      placed_vnfs: {(s, vnf_id) -> node}
      routed_vls : {(s, i, j) -> [path_nodes]}
    """

    def __init__(self, placed_vnfs=None, routed_vls=None, g_cost=None, original=None):
        self.placed_vnfs = placed_vnfs or {}
        self.routed_vls = routed_vls or {}
        self.g_cost = g_cost
        self.original = original

    def __repr__(self):
        return (
            f"<NormalizedResultView | "
            f"{len(self.placed_vnfs)} placements, {len(self.routed_vls)} routed_vls>"
        )


def _normalize_single_result(result, default_slice_id, valid_slice_ids):
    # All comments in English
    placed_norm = {}
    routed_norm = {}

    placed_raw = getattr(result, "placed_vnfs", {}) or {}
    routed_raw = getattr(result, "routed_vls", {}) or {}

    for k, node in placed_raw.items():
        if isinstance(k, tuple) and len(k) == 2 and k[0] in valid_slice_ids:
            placed_norm[(k[0], k[1])] = node
        else:
            if default_slice_id is None:
                continue
            placed_norm[(default_slice_id, k)] = node

    for k, path in routed_raw.items():
        if isinstance(k, tuple) and len(k) == 3 and k[0] in valid_slice_ids:
            routed_norm[(k[0], k[1], k[2])] = path
        elif isinstance(k, tuple) and len(k) == 2:
            if default_slice_id is None:
                continue
            i, j = k
            routed_norm[(default_slice_id, i, j)] = path

    return NormalizedResultView(
        placed_vnfs=placed_norm,
        routed_vls=routed_norm,
        g_cost=getattr(result, "g_cost", None),
        original=result,
    )


def normalize_result_list(method_name, result_list, slices):
    # All comments in English
    valid_slice_ids = set(_get_slice_id(sl, idx) for idx, sl in enumerate(slices))

    if not result_list:
        return []

    normalized = []

    if len(result_list) == len(slices):
        for idx, res in enumerate(result_list):
            s_id = _get_slice_id(slices[idx], idx)
            normalized.append(_normalize_single_result(res, s_id, valid_slice_ids))
        return normalized

    if len(result_list) == 1:
        normalized.append(_normalize_single_result(result_list[0], None, valid_slice_ids))
        return normalized

    for idx, res in enumerate(result_list):
        s_id = idx if idx < len(slices) else None
        normalized.append(_normalize_single_result(res, s_id, valid_slice_ids))

    return normalized


def build_per_slice_results_from_normalized(normalized_results, slices):
    # All comments in English
    """
    Convert normalized results into one pseudo-raw result per slice.

    Output format per slice:
      result.placed_vnfs = {vnf_id -> node}
      result.routed_vls  = {(src, dst) -> path_nodes}
    """
    per_slice = []

    for s_idx, _slice in enumerate(slices):
        placed_vnfs = {}
        routed_vls = {}

        for res in normalized_results:
            for key, node in getattr(res, "placed_vnfs", {}).items():
                if not (isinstance(key, tuple) and len(key) == 2):
                    continue

                s_id, vnf_id = key
                if s_id == s_idx:
                    placed_vnfs[vnf_id] = node

            for key, path in getattr(res, "routed_vls", {}).items():
                if not (isinstance(key, tuple) and len(key) == 3):
                    continue

                s_id, src, dst = key
                if s_id == s_idx:
                    routed_vls[(src, dst)] = path

        per_slice.append(
            SimpleNamespace(
                placed_vnfs=placed_vnfs,
                routed_vls=routed_vls,
            )
        )

    return per_slice


# ============================
# Exports
# ============================

def export_slice_entries_to_rows(
    slices,
    num_slices,
    num_vnfs_per_slice,
    seed,
    timestamp_str,
):
    # All comments in English
    rows = []

    for s_idx, sl in enumerate(slices):
        rows.append({
            "timestamp": timestamp_str,
            "num_slices": num_slices,
            "num_vnfs_per_slice": num_vnfs_per_slice,
            "seed": seed,
            "slice_idx": s_idx,
            "slice_id": _get_slice_id(sl, s_idx),
            "entry_node": get_entry_from_slice(sl),
        })

    return rows


def export_routes_to_rows(
    method_name,
    normalized_results,
    num_slices,
    num_vnfs_per_slice,
    seed,
    timestamp_str,
):
    # All comments in English
    rows = []

    for res_idx, res in enumerate(normalized_results):
        for vl_key, path_nodes in res.routed_vls.items():
            s_id, vnf_src, vnf_dst = _route_key_to_parts(vl_key)

            if not path_nodes or len(path_nodes) < 2:
                continue

            path_edges = list(zip(path_nodes[:-1], path_nodes[1:]))

            rows.append({
                "timestamp": timestamp_str,
                "method": method_name,
                "num_slices": num_slices,
                "num_vnfs_per_slice": num_vnfs_per_slice,
                "seed": seed,
                "result_idx": res_idx,
                "slice": s_id,
                "vnf_src": vnf_src,
                "vnf_dst": vnf_dst,
                "path_nodes": "->".join(map(str, path_nodes)),
                "path_edges": ";".join([f"{u}-{v}" for (u, v) in path_edges]),
                "num_hops": len(path_edges),
            })

    return rows


def export_node_hosting_to_rows(
    method_name,
    normalized_results,
    slice_cpu_map,
    num_slices,
    num_vnfs_per_slice,
    seed,
    timestamp_str,
    node_capacity_base,
):
    # All comments in English
    rows_hosting = []
    rows_cpu = []

    hosted = {}
    cpu_used = {}
    unknown_cpu = {}

    for res in normalized_results:
        for key, node in res.placed_vnfs.items():
            if not (isinstance(key, tuple) and len(key) == 2):
                continue

            s_id, vnf_id = key
            hosted.setdefault(node, set()).add(f"{s_id}:{vnf_id}")

            if (s_id, vnf_id) in slice_cpu_map:
                cpu_used[node] = cpu_used.get(node, 0.0) + float(slice_cpu_map[(s_id, vnf_id)])
            else:
                unknown_cpu[node] = unknown_cpu.get(node, 0) + 1

    for node, vnf_set in hosted.items():
        rows_hosting.append({
            "timestamp": timestamp_str,
            "method": method_name,
            "num_slices": num_slices,
            "num_vnfs_per_slice": num_vnfs_per_slice,
            "seed": seed,
            "node": node,
            "hosted_vnfs": ",".join(sorted(vnf_set)),
            "num_hosted_vnfs": len(vnf_set),
        })

    for node in node_capacity_base.keys():
        used = cpu_used.get(node, 0.0)
        cap = float(node_capacity_base[node]) if node in node_capacity_base else np.nan

        rows_cpu.append({
            "timestamp": timestamp_str,
            "method": method_name,
            "num_slices": num_slices,
            "num_vnfs_per_slice": num_vnfs_per_slice,
            "seed": seed,
            "node": node,
            "cpu_used": used,
            "cpu_capacity": cap,
            "cpu_utilization": used / cap if cap > 0 else np.nan,
            "num_hosted_vnfs": len(hosted.get(node, set())),
            "num_unknown_cpu_vnfs": int(unknown_cpu.get(node, 0)),
        })

    return rows_hosting, rows_cpu


def export_link_bw_load_to_rows(
    method_name,
    normalized_results,
    slice_vl_bw_map,
    num_slices,
    num_vnfs_per_slice,
    seed,
    timestamp_str,
    link_capacity_base,
):
    # All comments in English
    rows = []
    bw_used = {}

    for res in normalized_results:
        for vl_key, path_nodes in res.routed_vls.items():
            s_id, vnf_src, vnf_dst = _route_key_to_parts(vl_key)

            if s_id is None or not path_nodes or len(path_nodes) < 2:
                continue

            bw = slice_vl_bw_map.get((s_id, vnf_src, vnf_dst))
            if bw is None:
                continue

            for u, v in zip(path_nodes[:-1], path_nodes[1:]):
                bw_used[(u, v)] = bw_used.get((u, v), 0.0) + float(bw)

    for (u, v), used in bw_used.items():
        cap = float(link_capacity_base.get((u, v), float("nan")))
        rows.append({
            "timestamp": timestamp_str,
            "method": method_name,
            "num_slices": num_slices,
            "num_vnfs_per_slice": num_vnfs_per_slice,
            "seed": seed,
            "u": u,
            "v": v,
            "bw_used": used,
            "bw_capacity": cap,
            "bw_utilization": used / cap if cap and cap > 0 else float("nan"),
        })

    return rows


def export_milp_solution_to_rows(
    out,
    method_name,
    num_slices,
    num_vnfs_per_slice,
    seed,
    timestamp_str,
):
    # All comments in English
    if out is None or out.get("last_result") is None:
        return []

    res = out["last_result"]
    vals = res.values
    rows = []

    for key, val in vals.items():
        if not key or val <= 0.5:
            continue
        if key[0] == "x" and len(key) == 4:
            _, s, vnf_id, node = key
            rows.append({
                "timestamp": timestamp_str,
                "method": method_name,
                "num_slices": num_slices,
                "num_vnfs_per_slice": num_vnfs_per_slice,
                "seed": seed,
                "type": "placement",
                "slice": s,
                "vnf_src": vnf_id,
                "vnf_dst": None,
                "node": node,
                "u": None,
                "v": None,
                "path_nodes": None,
                "path_edges": None,
                "value": float(val),
            })

    for key, val in vals.items():
        if not key or val <= 1e-9:
            continue
        if key[0] == "f" and len(key) == 6:
            _, s, i, j, u, v = key
            rows.append({
                "timestamp": timestamp_str,
                "method": method_name,
                "num_slices": num_slices,
                "num_vnfs_per_slice": num_vnfs_per_slice,
                "seed": seed,
                "type": "flow_edge",
                "slice": s,
                "vnf_src": i,
                "vnf_dst": j,
                "node": None,
                "u": u,
                "v": v,
                "path_nodes": None,
                "path_edges": None,
                "value": float(val),
            })

    for key, val in vals.items():
        if not key:
            continue
        if key[0] == "xi" and len(key) == 2:
            _, s = key
            rows.append({
                "timestamp": timestamp_str,
                "method": method_name,
                "num_slices": num_slices,
                "num_vnfs_per_slice": num_vnfs_per_slice,
                "seed": seed,
                "type": "xi",
                "slice": s,
                "vnf_src": None,
                "vnf_dst": None,
                "node": None,
                "u": None,
                "v": None,
                "path_nodes": None,
                "path_edges": None,
                "value": float(val),
            })

    return rows


# ============================
# Method execution helpers
# ============================

def run_heuristic_method(method_name, func, args, slices):
    # All comments in English
    start = time.time()

    try:
        out = func(*args)

        if not isinstance(out, (tuple, list)) or len(out) < 2:
            raise TypeError(
                f"{method_name} must return at least two values like (_, results), "
                f"got {type(out).__name__}"
            )

        res_list = out[1]

        if res_list is None:
            res_list = []

        if not isinstance(res_list, list):
            res_list = [res_list]

        if len(res_list) not in (1, len(slices)):
            print(
                f"[WARN] {method_name} returned {len(res_list)} results for {len(slices)} slices. "
                f"Normalization will try a conservative fallback."
            )

        runtime = time.time() - start
        return {
            "ok": True,
            "raw_results": res_list,
            "normalized_results": normalize_result_list(method_name, res_list, slices),
            "runtime_sec": runtime,
            "error": None,
        }

    except Exception as e:
        runtime = time.time() - start
        return {
            "ok": False,
            "raw_results": [],
            "normalized_results": [],
            "runtime_sec": runtime,
            "error": str(e),
        }


# ============================
# Main
# ============================

def main():
    # All comments in English

    GLOBAL_SEED = 42
    FIXED_ENTRY_NODE = 6

    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    os.environ["PYTHONHASHSEED"] = str(GLOBAL_SEED)

    from milp.create_instance import create_instance
    from milp.solve_gurobi_sequential import solve_two_phase_max_accept_then_min_energy
    from milp.adapter import MILPResultAdapterGurobi

    from utils.topology import topologie_finlande
    from utils.generate_slices import generate_random_slices

    from heuristics.a_star import run_astar
    from heuristics.run_abo_full_batch import run_abo_full_batch
    from heuristics.run_fabo_full_batch import run_fabo_full_batch
    from heuristics.best_fit import run_best_fit
    from heuristics.first_fit import run_first_fit
    from heuristics.a_star_energy_aware import energy_aware_astar

    from utils.metrics import (
        compute_energy_new,
        compute_total_bandwidth,
        compute_total_latency,
        count_accepted_slices,
    )

    MILP_TIME_LIMIT = 30

    MAX_MILP_SLICES = 10000
    MAX_MILP_VNFS_TOTAL = 36000

    param_grid = {
        "num_slices": [4, 8],
        "num_vnfs_per_slice": [2, 3],
        "seed": [1],
    }

    vnf_profiles = [
        {"cpu": 1, "throughput": 15, "latency": 30},
        {"cpu": 1, "throughput": 20, "latency": 60},
        {"cpu": 2, "throughput": 25, "latency": 90},
        {"cpu": 3, "throughput": 30, "latency": 100},
        {"cpu": 4, "throughput": 35, "latency": 125},
        {"cpu": 6, "throughput": 50, "latency": 135},
    ]

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join("./results", safe_filename(timestamp))
    os.makedirs(results_dir, exist_ok=True)
    print(f"[INFO] Results will be saved under: {results_dir}")

    G = topologie_finlande()

    node_capacity_base = {n: G.nodes[n]["cpu"] for n in G.nodes}
    link_capacity_base = {(u, v): G[u][v]["bandwidth"] for u, v in G.edges}
    link_capacity_base.update({(v, u): G[u][v]["bandwidth"] for u, v in G.edges})

    link_latency = {(u, v): G[u][v]["latency"] for u, v in G.edges}
    link_latency.update({(v, u): G[u][v]["latency"] for u, v in G.edges})

    max_slices = max(param_grid["num_slices"])

    records_metrics = []
    records_routes = []
    records_milp_raw = []
    records_node_hosting = []
    records_node_cpu = []
    records_link_bw = []
    records_slice_entries = []

    fixed_slice_pools = {}

    for num_vnfs in param_grid["num_vnfs_per_slice"]:
        pool_seed = 1000 + num_vnfs
        random.seed(pool_seed)
        np.random.seed(pool_seed)

        print(
            f"[INFO] Pre-generating FIXED slice pool: max_slices={max_slices}, "
            f"VNFs={num_vnfs}, pool_seed={pool_seed}"
        )

        fixed_slice_pools[num_vnfs] = generate_random_slices(
            G,
            vnf_profiles,
            num_slices=max_slices,
            num_vnfs_per_slice=num_vnfs,
            entry=None,
        )

    for num_vnfs in param_grid["num_vnfs_per_slice"]:
        base_slice_pool = fixed_slice_pools[num_vnfs]

        for seed in param_grid["seed"]:
            print(
                f"\n[INFO] === Seed={seed} | fixed entry per slice={FIXED_ENTRY_NODE} "
                f"| VNFs per slice={num_vnfs} ==="
            )

            for num_slices in param_grid["num_slices"]:
                total_vnfs = num_slices * num_vnfs
                ts_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                print(
                    f"\n[INFO] === Running Load Test: {num_slices} slices × "
                    f"{num_vnfs} VNFs (total={total_vnfs}), seed={seed}, "
                    f"fixed-entry-per-slice={FIXED_ENTRY_NODE} ==="
                )

                slices_fixed = deepcopy(base_slice_pool[:num_slices])
                slices = assign_fixed_entry_per_slice(
                    slices_fixed,
                    entry_node=FIXED_ENTRY_NODE,
                )

                metadata = build_slice_metadata(slices)
                slice_cpu_map = metadata["slice_cpu_map"]
                slice_vl_bw_map = metadata["slice_vl_bw_map"]
                slice_entries = metadata["slice_entries"]

                print(f"[DEBUG] seed={seed} | fixed entry per slice={FIXED_ENTRY_NODE}")
                print(f"[DEBUG] first slice sample: {slices[0]}")
                for idx, sl in enumerate(slices[:3]):
                    print(f"[DEBUG] slice {idx} entry={get_entry_from_slice(sl)}")

                records_slice_entries.extend(
                    export_slice_entries_to_rows(
                        slices=slices,
                        num_slices=num_slices,
                        num_vnfs_per_slice=num_vnfs,
                        seed=seed,
                        timestamp_str=ts_now,
                    )
                )

                raw_method_results = {}
                normalized_method_results = {}
                method_times = {}

                heuristic_specs = [
                    ("A*", run_astar, (G, slices, node_capacity_base, link_capacity_base)),
                    ("ABO", run_abo_full_batch, (G, slices, node_capacity_base, link_latency, link_capacity_base)),
                    ("FABO", run_fabo_full_batch, (G, slices, node_capacity_base, link_latency, link_capacity_base)),
                    ("Best Fit", run_best_fit, (G, slices, node_capacity_base, link_capacity_base)),
                    ("First Fit", run_first_fit, (G, slices, node_capacity_base, link_capacity_base)),
                    ("Energy-Aware A*", energy_aware_astar, (G, slices, node_capacity_base, link_capacity_base)),
                ]

                for method_name, func, args in heuristic_specs:
                    out = run_heuristic_method(method_name, func, args, slices)

                    if not out["ok"]:
                        print(f"[ERROR] {method_name} failed: {out['error']}")
                        raw_method_results[method_name] = []
                        normalized_method_results[method_name] = []
                        method_times[method_name] = out["runtime_sec"]
                        continue

                    raw_method_results[method_name] = out["raw_results"]
                    normalized_method_results[method_name] = out["normalized_results"]
                    method_times[method_name] = out["runtime_sec"]

                milp_instance = None

                if num_slices <= MAX_MILP_SLICES and total_vnfs <= MAX_MILP_VNFS_TOTAL:
                    try:
                        print("[INFO][MILP] Running Gurobi (max-accept)...")
                        start = time.time()

                        milp_instance = create_instance(G, slices)

                        milp_instance.entry_node = None
                        milp_instance.entry_node_s = {
                            s: get_entry_from_slice(slices[s]) for s in range(len(slices))
                        }
                        milp_instance.entry_required_s = {
                            s: get_entry_from_slice(slices[s]) is not None for s in range(len(slices))
                        }

                        out = solve_two_phase_max_accept_then_min_energy(
                            instance=milp_instance,
                            slice_set=list(milp_instance.S),
                            msg=False,
                            time_limit_phase1=MILP_TIME_LIMIT,
                            time_limit_phase2=MILP_TIME_LIMIT,
                        )

                        if out.get("last_result") is not None:
                            raw_method_results["MILP_Gurobi"] = [out["last_result"]]

                            adapter = MILPResultAdapterGurobi(out["last_result"], milp_instance)
                            normalized_method_results["MILP_Gurobi"] = normalize_result_list(
                                "MILP_Gurobi",
                                [adapter],
                                slices,
                            )

                            acc = len(out.get("accepted_slices", []))
                            rej = len(out.get("rejected_slices", []))
                            print(f"[INFO][MILP] Accepted={acc}/{len(slices)} (Rejected={rej})")

                            records_milp_raw.extend(
                                export_milp_solution_to_rows(
                                    out=out,
                                    method_name="MILP_Gurobi",
                                    num_slices=num_slices,
                                    num_vnfs_per_slice=num_vnfs,
                                    seed=seed,
                                    timestamp_str=ts_now,
                                )
                            )
                        else:
                            raw_method_results["MILP_Gurobi"] = []
                            normalized_method_results["MILP_Gurobi"] = []

                        method_times["MILP_Gurobi"] = time.time() - start

                    except Exception as e:
                        print(f"[ERROR][MILP] Failed: {e}")
                        raw_method_results["MILP_Gurobi"] = []
                        normalized_method_results["MILP_Gurobi"] = []
                        method_times["MILP_Gurobi"] = None
                        milp_instance = None

                # ----------------------------
                # Exports from normalized data
                # ----------------------------
                for method_name, normalized_results in normalized_method_results.items():
                    if not normalized_results:
                        continue

                    records_routes.extend(
                        export_routes_to_rows(
                            method_name=method_name,
                            normalized_results=normalized_results,
                            num_slices=num_slices,
                            num_vnfs_per_slice=num_vnfs,
                            seed=seed,
                            timestamp_str=ts_now,
                        )
                    )

                    records_link_bw.extend(
                        export_link_bw_load_to_rows(
                            method_name=method_name,
                            normalized_results=normalized_results,
                            slice_vl_bw_map=slice_vl_bw_map,
                            num_slices=num_slices,
                            num_vnfs_per_slice=num_vnfs,
                            seed=seed,
                            timestamp_str=ts_now,
                            link_capacity_base=link_capacity_base,
                        )
                    )

                    hosting_rows, cpu_rows = export_node_hosting_to_rows(
                        method_name=method_name,
                        normalized_results=normalized_results,
                        slice_cpu_map=slice_cpu_map,
                        num_slices=num_slices,
                        num_vnfs_per_slice=num_vnfs,
                        seed=seed,
                        timestamp_str=ts_now,
                        node_capacity_base=node_capacity_base,
                    )

                    records_node_hosting.extend(hosting_rows)
                    records_node_cpu.extend(cpu_rows)

                # ----------------------------
                # Metrics
                # ----------------------------
                entry_nodes_used = [
                    slice_entries[_get_slice_id(sl, idx)]
                    for idx, sl in enumerate(slices)
                ]

                for method_name in raw_method_results.keys():
                    raw_results = raw_method_results.get(method_name, [])
                    normalized_results = normalized_method_results.get(method_name, [])

                    if not raw_results and not normalized_results:
                        continue

                    try:
                        if method_name == "MILP_Gurobi":
                            metric_results = raw_results

                            accepted = count_accepted_slices(
                                metric_results,
                                slices,
                                verbose=False,
                            )

                            bw_list = compute_total_bandwidth(
                                metric_results,
                                slices,
                                instance=milp_instance,
                                link_capacity=link_capacity_base,
                            )
                            total_bw = sum(x for x in bw_list if x is not None)

                            lat_list = compute_total_latency(
                                metric_results,
                                link_latency,
                                instance=milp_instance,
                            )
                            total_lat = sum(x for x in lat_list if x is not None)

                        else:
                            if len(raw_results) == len(slices):
                                metric_results = raw_results
                                print(
                                    f"[DEBUG] {method_name}: using raw per-slice results for metrics"
                                )
                            else:
                                metric_results = build_per_slice_results_from_normalized(
                                    normalized_results,
                                    slices,
                                )
                                print(
                                    f"[DEBUG] {method_name}: raw output is not per-slice "
                                    f"(len={len(raw_results)}), using normalized->per-slice adapter"
                                )

                            accepted = count_accepted_slices(
                                metric_results,
                                slices,
                                verbose=False,
                            )

                            bw_list = compute_total_bandwidth(
                                metric_results,
                                slices,
                                instance=None,
                                link_capacity=link_capacity_base,
                            )
                            total_bw = sum(x for x in bw_list if x is not None)

                            lat_list = compute_total_latency(
                                metric_results,
                                link_latency,
                                instance=None,
                            )
                            total_lat = sum(x for x in lat_list if x is not None)

                        total_energy = (
                            compute_energy_new(
                                normalized_results,
                                slices,
                                node_capacity_base,
                                link_capacity_base,
                            )
                            if normalized_results
                            else None
                        )

                        records_metrics.append({
                            "timestamp": ts_now,
                            "num_slices": num_slices,
                            "num_vnfs_per_slice": num_vnfs,
                            "total_vnfs": total_vnfs,
                            "seed": seed,
                            "entry_node": FIXED_ENTRY_NODE,
                            "entry_nodes_used": ",".join(map(str, entry_nodes_used)),
                            "method": method_name,
                            "accepted": accepted,
                            "total_energy": total_energy,
                            "total_bandwidth": total_bw,
                            "total_latency": total_lat,
                            "runtime_sec": method_times.get(method_name),
                        })

                    except Exception as e:
                        print(
                            f"[ERROR] Metrics failed for {method_name}, "
                            f"slices={num_slices}, vnfs={num_vnfs}, seed={seed}: {e}"
                        )

    df_metrics = pd.DataFrame(records_metrics)
    df_routes = pd.DataFrame(records_routes)
    df_milp_raw = pd.DataFrame(records_milp_raw)
    df_node_hosting = pd.DataFrame(records_node_hosting)
    df_node_cpu = pd.DataFrame(records_node_cpu)
    df_link_bw = pd.DataFrame(records_link_bw)
    df_slice_entries = pd.DataFrame(records_slice_entries)

    metrics_path = os.path.join(results_dir, "scalability_results.csv")
    routes_path = os.path.join(results_dir, "routes_all_methods.csv")
    milp_raw_path = os.path.join(results_dir, "milp_raw_vars.csv")
    node_hosting_path = os.path.join(results_dir, "node_vnfs_all_methods.csv")
    node_cpu_path = os.path.join(results_dir, "node_cpu_load_all_methods.csv")
    link_bw_path = os.path.join(results_dir, "link_bw_load_all_methods.csv")
    slice_entries_path = os.path.join(results_dir, "slice_entries.csv")

    df_metrics.to_csv(metrics_path, index=False)
    df_routes.to_csv(routes_path, index=False)
    df_milp_raw.to_csv(milp_raw_path, index=False)
    df_node_hosting.to_csv(node_hosting_path, index=False)
    df_node_cpu.to_csv(node_cpu_path, index=False)
    df_link_bw.to_csv(link_bw_path, index=False)
    df_slice_entries.to_csv(slice_entries_path, index=False)

    print(f"[INFO] Slice entries CSV saved to: {slice_entries_path} (rows={len(df_slice_entries)})")
    print(f"[INFO] Link BW load CSV saved to: {link_bw_path} (rows={len(df_link_bw)})")
    print(f"[INFO] Node hosting CSV saved to: {node_hosting_path} (rows={len(df_node_hosting)})")
    print(f"[INFO] Node CPU load CSV saved to: {node_cpu_path} (rows={len(df_node_cpu)})")
    print(f"\n[INFO] Metrics CSV saved to: {metrics_path} (rows={len(df_metrics)})")
    print(f"[INFO] Routes CSV saved to: {routes_path} (rows={len(df_routes)})")
    print(f"[INFO] MILP raw CSV saved to: {milp_raw_path} (rows={len(df_milp_raw)})")
    print(f"[INFO] Results dir: {results_dir}")


if __name__ == "__main__":
    main()