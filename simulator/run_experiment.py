
import os
import time
import random
import re
from copy import deepcopy
from datetime import datetime

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


def _route_key_to_parts(vl_key):
    # All comments in English
    # Supports keys like (s, i, j) or (i, j)
    if isinstance(vl_key, tuple) and len(vl_key) == 3:
        s, i, j = vl_key
        return s, i, j
    if isinstance(vl_key, tuple) and len(vl_key) == 2:
        i, j = vl_key
        return None, i, j
    return None, None, None

import re
import numpy as np

import re
import numpy as np

def _normalize_token(x):
    # All comments in English
    s = str(x).strip().lower()
    s = s.replace(" ", "")
    s = re.sub(r"[^a-z0-9_(),\-\[\]]", "", s)
    return s


def build_vnf_cpu_alias_map(slices, num_vnfs_per_slice=None):
    # All comments in English
    # Returns a dict: alias_key -> cpu
    alias_map = {}

    def _set_alias(key, cpu):
        if key is None or cpu is None:
            return
        try:
            cpu_f = float(cpu)
        except Exception:
            return
        alias_map[key] = cpu_f
        alias_map[_normalize_token(key)] = cpu_f

    def _maybe_cpu(obj):
        # All comments in English
        if obj is None:
            return None
        if isinstance(obj, (int, float)):
            return float(obj)
        if isinstance(obj, dict):
            return obj.get("cpu") or obj.get("cpu_demand") or obj.get("demand_cpu") or obj.get("cpu_req")
        # object attributes
        for attr in ["cpu", "cpu_demand", "demand_cpu", "cpu_req"]:
            if hasattr(obj, attr):
                val = getattr(obj, attr)
                if isinstance(val, (int, float)):
                    return float(val)
        return None

    def _maybe_id(obj):
        # All comments in English
        if obj is None:
            return None
        if isinstance(obj, (int, str)):
            return obj
        if isinstance(obj, dict):
            return obj.get("id") or obj.get("vnf_id") or obj.get("name") or obj.get("label")
        for attr in ["id", "vnf_id", "name", "label"]:
            if hasattr(obj, attr):
                return getattr(obj, attr)
        return None

    def _extract_vnfs_from_slice(sl):
        # All comments in English
        # Case 1: dict slice
        if isinstance(sl, dict):
            for k in ["vnfs", "vnf_list", "functions", "chain", "vnf_chain", "vnf_sequence"]:
                vnfs = sl.get(k)
                if isinstance(vnfs, list) and vnfs:
                    return vnfs
            # mapping vnf->cpu
            for k in ["vnf_cpu", "vnf_cpu_map", "cpu_by_vnf", "cpu_demands", "vnf_demands", "demands"]:
                m = sl.get(k)
                if isinstance(m, dict) and m:
                    # Return dict items as pseudo-vnfs
                    return [(vid, cpu) for vid, cpu in m.items()]
            return []

        # Case 2: tuple/list slice
        if isinstance(sl, (tuple, list)):
            # Heuristic: find the first element that looks like a list of VNFs
            for item in sl:
                if isinstance(item, list) and item:
                    return item
                if isinstance(item, tuple) and item and all(isinstance(x, (tuple, list, dict)) for x in item):
                    return list(item)
                if isinstance(item, dict) and item:
                    # sometimes vnfs stored as dict
                    if any(k in item for k in ["vnfs", "vnf_list", "functions", "chain"]):
                        return _extract_vnfs_from_slice(item)
            return []

        return []

    def _slice_id(sl, default):
        # All comments in English
        if isinstance(sl, dict):
            return sl.get("id", default)
        if isinstance(sl, (tuple, list)):
            # common pattern: (slice_id, ...)
            if len(sl) >= 1 and isinstance(sl[0], (int, str)):
                return sl[0]
        return default

    for s_idx, sl in enumerate(slices):
        s_id = _slice_id(sl, s_idx)
        vnfs = _extract_vnfs_from_slice(sl)

        # vnfs can be list of dict/obj OR list of (id,cpu)
        for k_idx, v in enumerate(vnfs):
            # If v is (id,cpu)
            if isinstance(v, (tuple, list)) and len(v) >= 2 and isinstance(v[0], (int, str)) and isinstance(v[1], (int, float)):
                vid = v[0]
                cpu = v[1]
            else:
                vid = _maybe_id(v)
                cpu = _maybe_cpu(v)

                # If still missing, try tuple/list v format like (id, profile, cpu)
                if cpu is None and isinstance(v, (tuple, list)):
                    # find first numeric as cpu
                    for it in v:
                        if isinstance(it, (int, float)):
                            cpu = float(it)
                            break
                    if vid is None and len(v) >= 1:
                        vid = _maybe_id(v[0])

            # 1) direct id aliases
            if vid is not None:
                _set_alias(vid, cpu)
                _set_alias(str(vid), cpu)

            # 2) (slice, position) aliases
            for s_key in [s_idx, s_id]:
                _set_alias((s_key, k_idx), cpu)
                _set_alias(f"vnf{s_key}_{k_idx}", cpu)
                _set_alias(f"{s_key}_{k_idx}", cpu)
                _set_alias(f"({s_key},{k_idx})", cpu)

            # 3) global integer aliases
            if num_vnfs_per_slice is not None:
                try:
                    global_id = int(s_idx) * int(num_vnfs_per_slice) + int(k_idx)
                    _set_alias(global_id, cpu)
                    _set_alias(str(global_id), cpu)
                except Exception:
                    pass

    return alias_map

def export_node_hosting_to_rows(
    method_name,
    result_list,
    slices,
    num_slices,
    num_vnfs_per_slice,
    seed,
    timestamp_str,
    node_capacity_base,
    debug=False,
):
    # All comments in English
    alias_map = build_vnf_cpu_alias_map(slices, num_vnfs_per_slice=num_vnfs_per_slice)

    rows_hosting = []
    rows_cpu = []

    hosted = {}          # node -> set(vnf_id_str)
    cpu_used = {}        # node -> float
    unknown_cpu = {}     # node -> count
    unknown_examples = []

    for res in result_list:
        if not hasattr(res, "placed_vnfs") or res.placed_vnfs is None:
            continue

        for vnf_id, node in res.placed_vnfs.items():
            vnf_id_str = str(vnf_id)
            hosted.setdefault(node, set()).add(vnf_id_str)

            # Try a sequence of lookups (raw, normalized, tuple-normalized)
            candidates = [vnf_id, vnf_id_str, _normalize_token(vnf_id_str)]

            if isinstance(vnf_id, tuple):
                candidates.append(vnf_id)
                candidates.append(_normalize_token(vnf_id))

            found = False
            for c in candidates:
                if c in alias_map:
                    cpu_used[node] = cpu_used.get(node, 0.0) + float(alias_map[c])
                    found = True
                    break

            if not found:
                unknown_cpu[node] = unknown_cpu.get(node, 0) + 1
                if debug and len(unknown_examples) < 12:
                    unknown_examples.append((vnf_id, vnf_id_str, _normalize_token(vnf_id_str)))

    if debug and unknown_examples:
        print("[DEBUG] Unknown VNF IDs (sample):")
        for ex in unknown_examples:
            print("   ", ex)
        print("[DEBUG] alias_map size:", len(alias_map))
        print("[DEBUG] alias_map sample:", list(alias_map.items())[:8])

    # Hosting rows
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

    # CPU rows (include nodes with 0 VNFs)
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

def export_routes_to_rows(
    method_name,
    result_list,
    num_slices,
    num_vnfs_per_slice,
    seed,
    timestamp_str,
):
    # All comments in English
    rows = []
    for res_idx, res in enumerate(result_list):
        if not hasattr(res, "routed_vls") or res.routed_vls is None:
            continue

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


def export_milp_solution_to_rows(
    out,
    instance,
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

    # Placements: ("x", s, vnf_id, node)
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

    # Flows: ("f", s, i, j, u, v)
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

    # Slack: ("xi", s)
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

def export_link_bw_load_to_rows(
    method_name,
    result_list,
    slices,
    num_slices,
    num_vnfs_per_slice,
    seed,
    timestamp_str,
    link_capacity_base,
):
    # All comments in English
    rows = []

    # Compute total BW used per directed physical link (u,v)
    bw_used = {}  # (u,v) -> float

    # Build a vlink bw alias map from slices, similar to CPU aliasing
    # We try common patterns inside each slice/vlink object/dict
    def _get_vlink_bw(sl, src, dst):
        # All comments in English
        if isinstance(sl, dict):
            # Try explicit mapping if exists
            for k in ["vlink_bw", "bw_by_vlink", "bw_demands", "link_demands", "demands_bw"]:
                m = sl.get(k)
                if isinstance(m, dict):
                    if (src, dst) in m:
                        return m[(src, dst)]
                    if (str(src), str(dst)) in m:
                        return m[(str(src), str(dst))]
            # Fallback: if profiles are uniform, you may not have per-vlink BW here
        return None

    # Create a simple slice index -> slice object map
    slice_map = {i: sl for i, sl in enumerate(slices)}
    # Also allow direct id access if slice is dict with "id"
    for i, sl in enumerate(slices):
        if isinstance(sl, dict) and "id" in sl:
            slice_map[sl["id"]] = sl

    for res in result_list:
        if not hasattr(res, "routed_vls") or res.routed_vls is None:
            continue

        for vl_key, path_nodes in res.routed_vls.items():
            s_id, vnf_src, vnf_dst = _route_key_to_parts(vl_key)
            if not path_nodes or len(path_nodes) < 2:
                continue

            # Try to recover BW demand for this vlink
            bw = None
            if s_id in slice_map:
                bw = _get_vlink_bw(slice_map[s_id], vnf_src, vnf_dst)

            # If BW is missing, skip (but you will see it in the report)
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



# ============================
# Main
# ============================

def main():
    # All comments in English

    # --- Global reproducibility seed ---
    GLOBAL_SEED = 42
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    os.environ["PYTHONHASHSEED"] = str(GLOBAL_SEED)

    # --- Imports that depend on your project structure ---
    from milp.create_instance import create_instance
    from milp.solve_gurobi_sequential import solve_gurobi_max_accept
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
        count_accepted_slices,
        compute_energy_new,
        compute_total_bandwidth,
        compute_total_latency,
    )

    # ============================
    # Config (edit here)
    # ============================

    MILP_TIME_LIMIT = 10**9 
    ENTRY = 6

    MAX_MILP_SLICES = 10**9
    MAX_MILP_VNFS_TOTAL = 10**9

    param_grid = {
        "num_slices": [4, 8, 16, 32, 64, 128],
        "num_vnfs_per_slice": [2, 3, 4, 5, 6],
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


    # ============================
    # Directories
    # ============================

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join("./results", safe_filename(timestamp))
    os.makedirs(results_dir, exist_ok=True)
    print(f"[INFO] Results will be saved under: {results_dir}")

    # ============================
    # Topology and capacities
    # ============================

    G = topologie_finlande()
    node_capacity_base = {n: G.nodes[n]["cpu"] for n in G.nodes}
    link_capacity_base = {(u, v): G[u][v]["bandwidth"] for u, v in G.edges}
    link_capacity_base.update({(v, u): G[u][v]["bandwidth"] for u, v in G.edges})

    link_latency = {(u, v): G[u][v]["latency"] for u, v in G.edges}
    link_latency.update({(v, u): G[u][v]["latency"] for u, v in G.edges})

    max_slices = max(param_grid["num_slices"])

    # Records
    records_metrics = []
    records_routes = []
    records_milp_raw = []
    records_node_hosting = []
    records_node_cpu = []
    records_link_bw = []



    # ============================
    # Experiment loop
    # ============================

    for num_vnfs in param_grid["num_vnfs_per_slice"]:
        for seed in param_grid["seed"]:
            random.seed(seed)
            np.random.seed(seed)

            print(
                f"\n[INFO] === Pre-generating slice pool: max_slices={max_slices}, "
                f"VNFs={num_vnfs}, seed={seed} ==="
            )

            slice_pool = generate_random_slices(
                G,
                vnf_profiles,
                num_slices=max_slices,
                num_vnfs_per_slice=num_vnfs,
                entry=ENTRY,
            )

            for num_slices in param_grid["num_slices"]:
                total_vnfs = num_slices * num_vnfs
                ts_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                print(
                    f"\n[INFO] === Running Load Test: {num_slices} slices × "
                    f"{num_vnfs} VNFs (total={total_vnfs}), seed={seed} ==="
                )

                slices = deepcopy(slice_pool[:num_slices])
                method_results = {}
                method_times = {}

                print("\n[DEBUG] slices[0] type:", type(slices[0]))
                if isinstance(slices[0], dict):
                    print("[DEBUG] slices[0] keys:", list(slices[0].keys())[:MAX_MILP_SLICES])
                    for k in ["vnfs", "vnf_list", "functions", "chain", "vnf_cpu", "cpu_demands", "vnf_demands"]:
                        if k in slices[0]:
                            print(f"[DEBUG] slices[0]['{k}'] type:", type(slices[0][k]))
                            print(f"[DEBUG] slices[0]['{k}'] sample:", str(slices[0][k])[:MAX_MILP_VNFS_TOTAL])

                # Show one heuristic placement id format
                tmp_name = "ABO"
                if tmp_name in method_results and method_results[tmp_name]:
                    r0 = method_results[tmp_name][0]
                    print("[DEBUG] placed_vnfs type:", type(r0.placed_vnfs))
                    print("[DEBUG] placed_vnfs sample:", list(r0.placed_vnfs.items())[:5])

               

                # --- Heuristics ---
                for name, func, args in [
                    ("A*", run_astar, (G, slices, node_capacity_base, link_capacity_base)),
                    ("ABO", run_abo_full_batch, (G, slices, node_capacity_base, link_latency, link_capacity_base)),
                    ("FABO", run_fabo_full_batch, (G, slices, node_capacity_base, link_latency, link_capacity_base)),
                    ("Best Fit", run_best_fit, (G, slices, node_capacity_base, link_capacity_base, link_latency)),
                    ("First Fit", run_first_fit, (G, slices, node_capacity_base, link_capacity_base, link_latency)),
                    ("Energy-Aware A*", energy_aware_astar, (G, slices, node_capacity_base, link_capacity_base)),
                ]:
                    start = time.time()
                    try:
                        _, res_list = func(*args)
                        method_results[name] = res_list
                    except Exception as e:
                        print(f"[ERROR] {name} failed: {e}")
                        method_results[name] = []
                    method_times[name] = time.time() - start

                # --- MILP (max-accept with z[s]) ---
                if num_slices <= MAX_MILP_SLICES and total_vnfs <= MAX_MILP_VNFS_TOTAL:
                    try:
                        print("[INFO][MILP] Running Gurobi (max-accept)…")
                        start = time.time()

                        instance = create_instance(G, slices)
                        instance.entry_node = ENTRY  # keep if your create_instance uses it

                        out = solve_gurobi_max_accept(
                            instance,
                            msg=False,
                            time_limit=MILP_TIME_LIMIT,
                        )

                        if out.get("last_result") is not None:
                            adapter = MILPResultAdapterGurobi(out["last_result"], instance)
                            method_results["MILP_Gurobi"] = [adapter]

                            # Optional: print acceptance summary
                            acc = len(out.get("accepted_slices", []))
                            rej = len(out.get("rejected_slices", []))
                            print(f"[INFO][MILP] Accepted={acc}/{len(slices)} (Rejected={rej})")

                            records_milp_raw.extend(
                                export_milp_solution_to_rows(
                                    out=out,
                                    instance=instance,
                                    method_name="MILP_Gurobi",
                                    num_slices=num_slices,
                                    num_vnfs_per_slice=num_vnfs,
                                    seed=seed,
                                    timestamp_str=ts_now,
                                )
                            )
                        else:
                            method_results["MILP_Gurobi"] = []

                        method_times["MILP_Gurobi"] = time.time() - start

                    except Exception as e:
                        print(f"[ERROR][MILP] Failed: {e}")
                        method_results["MILP_Gurobi"] = []
                        method_times["MILP_Gurobi"] = None


                # --- Export routes (ALL methods) ---
                for method_name, result_list in method_results.items():
                    if not result_list:
                        continue
                    records_routes.extend(
                        export_routes_to_rows(
                            method_name=method_name,
                            result_list=result_list,
                            num_slices=num_slices,
                            num_vnfs_per_slice=num_vnfs,
                            seed=seed,
                            timestamp_str=ts_now,
                        )
                    )
                
                for method_name, result_list in method_results.items():
                    if not result_list:
                        continue
                    records_link_bw.extend(
                        export_link_bw_load_to_rows(
                            method_name=method_name,
                            result_list=result_list,
                            slices=slices,
                            num_slices=num_slices,
                            num_vnfs_per_slice=num_vnfs,
                            seed=seed,
                            timestamp_str=ts_now,
                            link_capacity_base=link_capacity_base,
                        )
                    )

                
                # --- Export node hosting + cpu load (ALL methods) ---
                for method_name, result_list in method_results.items():
                    if not result_list:
                        continue

                    hosting_rows, cpu_rows = export_node_hosting_to_rows(
                        method_name=method_name,
                        result_list=result_list,
                        slices=slices,
                        num_slices=num_slices,
                        num_vnfs_per_slice=num_vnfs,
                        seed=seed,
                        timestamp_str=ts_now,
                        node_capacity_base=node_capacity_base,
                    )

                    records_node_hosting.extend(hosting_rows)
                    records_node_cpu.extend(cpu_rows)


                # --- Metrics ---
                for method_name, result_list in method_results.items():
                    if not result_list:
                        continue
                    try:
                        accepted = count_accepted_slices(result_list, slices)
                        total_energy = compute_energy_new(
                            result_list, slices, node_capacity_base, link_capacity_base
                        )
                        total_bw = sum(b for b in compute_total_bandwidth(result_list, slices) if b)
                        total_lat = sum(l for l in compute_total_latency(result_list, link_latency) if l)

                        records_metrics.append({
                            "timestamp": ts_now,
                            "num_slices": num_slices,
                            "num_vnfs_per_slice": num_vnfs,
                            "total_vnfs": total_vnfs,
                            "seed": seed,
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

    # ============================
    # Save CSVs
    # ============================

    df_metrics = pd.DataFrame(records_metrics)
    df_routes = pd.DataFrame(records_routes)
    df_milp_raw = pd.DataFrame(records_milp_raw)

    metrics_path = os.path.join(results_dir, "scalability_results.csv")
    routes_path = os.path.join(results_dir, "routes_all_methods.csv")
    milp_raw_path = os.path.join(results_dir, "milp_raw_vars.csv")

    df_metrics.to_csv(metrics_path, index=False)
    df_routes.to_csv(routes_path, index=False)
    df_milp_raw.to_csv(milp_raw_path, index=False)


    df_node_hosting = pd.DataFrame(records_node_hosting)
    df_node_cpu = pd.DataFrame(records_node_cpu)

    node_hosting_path = os.path.join(results_dir, "node_vnfs_all_methods.csv")
    node_cpu_path = os.path.join(results_dir, "node_cpu_load_all_methods.csv")

    df_node_hosting.to_csv(node_hosting_path, index=False)
    df_node_cpu.to_csv(node_cpu_path, index=False)

    df_link_bw = pd.DataFrame(records_link_bw)
    link_bw_path = os.path.join(results_dir, "link_bw_load_all_methods.csv")
    df_link_bw.to_csv(link_bw_path, index=False)
    print(f"[INFO] Link BW load CSV saved to: {link_bw_path} (rows={len(df_link_bw)})")


    print(f"[INFO] Node hosting CSV saved to: {node_hosting_path} (rows={len(df_node_hosting)})")
    print(f"[INFO] Node CPU load CSV saved to: {node_cpu_path} (rows={len(df_node_cpu)})")


    print(f"\n[INFO] Metrics CSV saved to: {metrics_path} (rows={len(df_metrics)})")
    print(f"[INFO] Routes  CSV saved to: {routes_path} (rows={len(df_routes)})")
    print(f"[INFO] MILP raw CSV saved to: {milp_raw_path} (rows={len(df_milp_raw)})")
    print(f"[INFO] Results dir: {results_dir}")


if __name__ == "__main__":
    main()
