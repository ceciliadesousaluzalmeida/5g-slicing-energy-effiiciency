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


def _normalize_token(x):
    # All comments in English
    s = str(x).strip().lower()
    s = s.replace(" ", "")
    s = re.sub(r"[^a-z0-9_(),\-\[\]]", "", s)
    return s


def _spawn_rng(seed: int, *keys: int) -> np.random.Generator:
    # All comments in English
    # Deterministic child RNG from a base seed + keys
    ss = np.random.SeedSequence([int(seed), *map(int, keys)])
    return np.random.default_rng(ss)


def _sample_slices_from_pool(pool, k: int, rng: np.random.Generator):
    # All comments in English
    # Sample without replacement and shuffle order to change admission dynamics
    if k > len(pool):
        raise ValueError(f"Cannot sample k={k} from pool size {len(pool)}.")
    idx = rng.choice(len(pool), size=k, replace=False)
    sampled = [deepcopy(pool[i]) for i in idx]
    rng.shuffle(sampled)
    return sampled


def _apply_micro_noise_to_tuple_slice_demands(slices, rng, cpu_sigma=0.00, thr_sigma=0.00, lat_sigma=0.00):
    # All comments in English
    # Optional: apply tiny noise to avoid degenerate identical instances
    # Set sigmas to 0.0 to disable (default).
    if cpu_sigma == 0.0 and thr_sigma == 0.0 and lat_sigma == 0.0:
        return slices

    out = []
    for sl in slices:
        if not isinstance(sl, (tuple, list)) or len(sl) < 2:
            out.append(sl)
            continue

        vnfs, vlinks = sl[0], sl[1]
        entry = sl[2] if len(sl) >= 3 else None

        vnfs2 = []
        for v in vnfs:
            if not isinstance(v, dict):
                vnfs2.append(v)
                continue
            v2 = dict(v)
            if "cpu" in v2 and isinstance(v2["cpu"], (int, float)) and cpu_sigma > 0:
                v2["cpu"] = max(0.0, float(v2["cpu"]) * (1.0 + rng.normal(0.0, cpu_sigma)))
            if "throughput" in v2 and isinstance(v2["throughput"], (int, float)) and thr_sigma > 0:
                v2["throughput"] = max(0.0, float(v2["throughput"]) * (1.0 + rng.normal(0.0, thr_sigma)))
            if "latency" in v2 and isinstance(v2["latency"], (int, float)) and lat_sigma > 0:
                v2["latency"] = max(0.0, float(v2["latency"]) * (1.0 + rng.normal(0.0, lat_sigma)))
            vnfs2.append(v2)

        vlinks2 = []
        for e in vlinks:
            if not isinstance(e, dict):
                vlinks2.append(e)
                continue
            e2 = dict(e)
            if "bandwidth" in e2 and isinstance(e2["bandwidth"], (int, float)) and thr_sigma > 0:
                e2["bandwidth"] = max(0.0, float(e2["bandwidth"]) * (1.0 + rng.normal(0.0, thr_sigma)))
            if "latency" in e2 and isinstance(e2["latency"], (int, float)) and lat_sigma > 0:
                e2["latency"] = max(0.0, float(e2["latency"]) * (1.0 + rng.normal(0.0, lat_sigma)))
            vlinks2.append(e2)

        out.append((vnfs2, vlinks2, entry))
    return out


# ============================
# Export helpers
# ============================

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
        # Your generator returns (vnfs, vlinks, entry)
        if isinstance(sl, (tuple, list)) and len(sl) >= 1 and isinstance(sl[0], list):
            return sl[0]
        if isinstance(sl, dict):
            for k in ["vnfs", "vnf_list", "functions", "chain", "vnf_chain", "vnf_sequence"]:
                vnfs = sl.get(k)
                if isinstance(vnfs, list) and vnfs:
                    return vnfs
            for k in ["vnf_cpu", "vnf_cpu_map", "cpu_by_vnf", "cpu_demands", "vnf_demands", "demands"]:
                m = sl.get(k)
                if isinstance(m, dict) and m:
                    return [(vid, cpu) for vid, cpu in m.items()]
            return []
        return []

    def _slice_id(sl, default):
        # All comments in English
        # Your slices are positional; slice index is enough
        if isinstance(sl, dict):
            return sl.get("id", default)
        return default

    for s_idx, sl in enumerate(slices):
        s_id = _slice_id(sl, s_idx)
        vnfs = _extract_vnfs_from_slice(sl)

        for k_idx, v in enumerate(vnfs):
            if isinstance(v, (tuple, list)) and len(v) >= 2 and isinstance(v[0], (int, str)) and isinstance(v[1], (int, float)):
                vid = v[0]
                cpu = v[1]
            else:
                vid = _maybe_id(v)
                cpu = _maybe_cpu(v)

                if cpu is None and isinstance(v, (tuple, list)):
                    for it in v:
                        if isinstance(it, (int, float)):
                            cpu = float(it)
                            break
                    if vid is None and len(v) >= 1:
                        vid = _maybe_id(v[0])

            if vid is not None:
                _set_alias(vid, cpu)
                _set_alias(str(vid), cpu)

            for s_key in [s_idx, s_id]:
                _set_alias((s_key, k_idx), cpu)
                _set_alias(f"vnf{s_key}_{k_idx}", cpu)
                _set_alias(f"{s_key}_{k_idx}", cpu)
                _set_alias(f"({s_key},{k_idx})", cpu)

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
    replicate,
    subseed,
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

            candidates = [vnf_id, vnf_id_str, _normalize_token(vnf_id_str)]

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

    for node, vnf_set in hosted.items():
        rows_hosting.append({
            "timestamp": timestamp_str,
            "method": method_name,
            "num_slices": num_slices,
            "num_vnfs_per_slice": num_vnfs_per_slice,
            "seed": seed,
            "replicate": replicate,
            "subseed": subseed,
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
            "replicate": replicate,
            "subseed": subseed,
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
    replicate,
    subseed,
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
                "replicate": replicate,
                "subseed": subseed,
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
    replicate,
    subseed,
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
                "replicate": replicate,
                "subseed": subseed,
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
                "replicate": replicate,
                "subseed": subseed,
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
                "replicate": replicate,
                "subseed": subseed,
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
    replicate,
    subseed,
    timestamp_str,
    link_capacity_base,
):
    # All comments in English
    rows = []
    bw_used = {}  # (u,v) -> float

    # Your slice is (vnfs, vlinks, entry)
    def _get_vlink_bw(sl, src, dst):
        # All comments in English
        if isinstance(sl, (tuple, list)) and len(sl) >= 2:
            vlinks = sl[1]
            if isinstance(vlinks, list):
                for e in vlinks:
                    if not isinstance(e, dict):
                        continue
                    if e.get("from") == src and e.get("to") == dst:
                        return e.get("bandwidth")
        if isinstance(sl, dict):
            for k in ["vlink_bw", "bw_by_vlink", "bw_demands", "link_demands", "demands_bw"]:
                m = sl.get(k)
                if isinstance(m, dict):
                    if (src, dst) in m:
                        return m[(src, dst)]
                    if (str(src), str(dst)) in m:
                        return m[(str(src), str(dst))]
        return None

    slice_map = {i: sl for i, sl in enumerate(slices)}
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

            bw = None
            if s_id in slice_map:
                bw = _get_vlink_bw(slice_map[s_id], vnf_src, vnf_dst)

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
            "replicate": replicate,
            "subseed": subseed,
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

    # --- Global reproducibility anchor ---
    GLOBAL_SEED = 42
    os.environ["PYTHONHASHSEED"] = str(GLOBAL_SEED)

    # --- Imports that depend on your project structure ---
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
        count_accepted_slices,
        compute_energy_new,
        compute_total_bandwidth,
        compute_total_latency,
    )

    # ============================
    # Config (edit here)
    # ============================

    MILP_TIME_LIMIT = 3600  # seconds per phase
    ENTRY = 6

    # Enable/disable MILP by thresholds
    MAX_MILP_SLICES = 10**9
    MAX_MILP_VNFS_TOTAL = 10**9

    # Replicates to force within-seed variability (and thus std)
    param_grid = {
        "num_slices": [4, 8, 16, 32, 64],
        "num_vnfs_per_slice": [2, 3, 4, 5, 6],
        "seed": [1, 2, 3, 4, 5],
        "replicate": list(range(1, 6)),  # 5 replicates per seed
    }

    # Optional: tiny noise (keep 0.0 for strict comparability)
    MICRO_NOISE = {
        "cpu_sigma": 0.00,
        "thr_sigma": 0.00,
        "lat_sigma": 0.00,
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

    # ===========================
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

            print(
                f"\n[INFO] === Pre-generating slice pool: max_slices={max_slices}, "
                f"VNFs={num_vnfs}, seed={seed} ==="
            )

            # Create a deterministic RNG for pool generation
            pool_rng = _spawn_rng(seed, num_vnfs, 999)

            # If your generator relies on python/random or numpy.random, lock them here
            pool_subseed_py = int(pool_rng.integers(0, 2**31 - 1))
            pool_subseed_np = int(pool_rng.integers(0, 2**31 - 1))
            random.seed(pool_subseed_py)
            np.random.seed(pool_subseed_np)

            slice_pool = generate_random_slices(
                G,
                vnf_profiles,
                num_slices=max_slices,
                num_vnfs_per_slice=num_vnfs,
                entry=ENTRY,
            )

            for num_slices in param_grid["num_slices"]:
                total_vnfs = num_slices * num_vnfs

                for rep in param_grid["replicate"]:
                    ts_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    print(
                        f"\n[INFO] === Running Load Test: {num_slices} slices × "
                        f"{num_vnfs} VNFs (total={total_vnfs}), seed={seed}, rep={rep} ==="
                    )

                    # Deterministic run-level RNG
                    run_rng = _spawn_rng(seed, num_vnfs, num_slices, rep)
                    run_subseed = int(run_rng.integers(0, 2**31 - 1))

                    # Drive stochastic tie-breakers consistently per run (if any)
                    random.seed(run_subseed)
                    np.random.seed(run_subseed)

                    # IMPORTANT: sample from pool instead of taking prefix
                    slices = _sample_slices_from_pool(slice_pool, num_slices, run_rng)

                    # Optional: micro noise (disabled by default)
                    slices = _apply_micro_noise_to_tuple_slice_demands(
                        slices,
                        run_rng,
                        cpu_sigma=MICRO_NOISE["cpu_sigma"],
                        thr_sigma=MICRO_NOISE["thr_sigma"],
                        lat_sigma=MICRO_NOISE["lat_sigma"],
                    )

                    method_results = {}
                    method_times = {}

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

                    # --- MILP (max-accept then min-energy) ---
                    if num_slices <= MAX_MILP_SLICES and total_vnfs <= MAX_MILP_VNFS_TOTAL:
                        try:
                            print("[INFO][MILP] Running Gurobi (max-accept)…")
                            start = time.time()

                            instance = create_instance(G, slices)
                            instance.entry_node = ENTRY
                            instance.entry_required_s = {s: False for s in instance.S}

                            out = solve_two_phase_max_accept_then_min_energy(
                                instance=instance,
                                slice_set=list(instance.S),
                                msg=False,
                                time_limit_phase1=MILP_TIME_LIMIT,
                                time_limit_phase2=MILP_TIME_LIMIT,
                            )

                            if out.get("last_result") is not None:
                                adapter = MILPResultAdapterGurobi(out["last_result"], instance)
                                method_results["MILP_Gurobi"] = [adapter]

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
                                        replicate=rep,
                                        subseed=run_subseed,
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
                                replicate=rep,
                                subseed=run_subseed,
                                timestamp_str=ts_now,
                            )
                        )

                    # --- Export link BW load (ALL methods) ---
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
                                replicate=rep,
                                subseed=run_subseed,
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
                            replicate=rep,
                            subseed=run_subseed,
                            timestamp_str=ts_now,
                            node_capacity_base=node_capacity_base,
                            debug=False,
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
                                "replicate": rep,
                                "subseed": run_subseed,
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
                                f"slices={num_slices}, vnfs={num_vnfs}, seed={seed}, rep={rep}: {e}"
                            )

    # ============================
    # Save CSVs
    # ============================

    df_metrics = pd.DataFrame(records_metrics)
    df_routes = pd.DataFrame(records_routes)
    df_milp_raw = pd.DataFrame(records_milp_raw)
    df_node_hosting = pd.DataFrame(records_node_hosting)
    df_node_cpu = pd.DataFrame(records_node_cpu)
    df_link_bw = pd.DataFrame(records_link_bw)

    metrics_path = os.path.join(results_dir, "scalability_results.csv")
    routes_path = os.path.join(results_dir, "routes_all_methods.csv")
    milp_raw_path = os.path.join(results_dir, "milp_raw_vars.csv")
    node_hosting_path = os.path.join(results_dir, "node_vnfs_all_methods.csv")
    node_cpu_path = os.path.join(results_dir, "node_cpu_load_all_methods.csv")
    link_bw_path = os.path.join(results_dir, "link_bw_load_all_methods.csv")

    df_metrics.to_csv(metrics_path, index=False)
    df_routes.to_csv(routes_path, index=False)
    df_milp_raw.to_csv(milp_raw_path, index=False)
    df_node_hosting.to_csv(node_hosting_path, index=False)
    df_node_cpu.to_csv(node_cpu_path, index=False)
    df_link_bw.to_csv(link_bw_path, index=False)

    print(f"\n[INFO] Metrics CSV saved to: {metrics_path} (rows={len(df_metrics)})")
    print(f"[INFO] Routes  CSV saved to: {routes_path} (rows={len(df_routes)})")
    print(f"[INFO] MILP raw CSV saved to: {milp_raw_path} (rows={len(df_milp_raw)})")
    print(f"[INFO] Node hosting CSV saved to: {node_hosting_path} (rows={len(df_node_hosting)})")
    print(f"[INFO] Node CPU load CSV saved to: {node_cpu_path} (rows={len(df_node_cpu)})")
    print(f"[INFO] Link BW load CSV saved to: {link_bw_path} (rows={len(df_link_bw)})")
    print(f"[INFO] Results dir: {results_dir}")


if __name__ == "__main__":
    main()