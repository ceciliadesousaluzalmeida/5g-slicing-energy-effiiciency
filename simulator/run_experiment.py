
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
    from milp.solve_gurobi_sequential import solve_gurobi_shrink_until_feasible
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

    MILP_TIME_LIMIT = 30
    ENTRY = 6

    MAX_MILP_SLICES = 4
    MAX_MILP_VNFS_TOTAL = 8

    param_grid = {
        "num_slices": [2, 3, 4],
        "num_vnfs_per_slice": [2],
        "seed": [1],
    }

    vnf_profiles = [
        {"cpu": 2, "throughput": 40, "latency": 120},
        {"cpu": 4, "throughput": 50, "latency": 180},
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

                # --- Heuristics ---
                for name, func, args in [
                    ("A*", run_astar, (G, slices, node_capacity_base, link_capacity_base)),
                    ("ABO", run_abo_full_batch, (G, slices, node_capacity_base, link_latency, link_capacity_base)),
                    ("FABO", run_fabo_full_batch, (G, slices, node_capacity_base, link_latency, link_capacity_base)),
                    ("Best Fit", run_best_fit, (G, slices, node_capacity_base, link_capacity_base, link_latency)),
                    ("First Fit", run_first_fit, (G, slices, node_capacity_base, link_capacity_base, link_latency)),
                    ("Energy-Aware A*", energy_aware_astar, (G, slices, node_capacity_base, link_capacity_base, 0.6, 0.4)),
                ]:
                    start = time.time()
                    try:
                        _, res_list = func(*args)
                        method_results[name] = res_list
                    except Exception as e:
                        print(f"[ERROR] {name} failed: {e}")
                        method_results[name] = []
                    method_times[name] = time.time() - start

                # --- MILP (shrink-until-feasible) ---
                if num_slices <= MAX_MILP_SLICES and total_vnfs <= MAX_MILP_VNFS_TOTAL:
                    try:
                        print("[INFO][MILP] Running Gurobi (shrink-until-feasible)…")
                        start = time.time()

                        instance = create_instance(G, slices)
                        instance.entry_node = ENTRY

                        out = solve_gurobi_shrink_until_feasible(
                            instance,
                            msg=False,
                            time_limit=MILP_TIME_LIMIT,
                        )

                        if out.get("last_result") is not None:
                            adapter = MILPResultAdapterGurobi(out["last_result"], instance)
                            method_results["MILP_Gurobi"] = [adapter]

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

    print(f"\n[INFO] Metrics CSV saved to: {metrics_path} (rows={len(df_metrics)})")
    print(f"[INFO] Routes  CSV saved to: {routes_path} (rows={len(df_routes)})")
    print(f"[INFO] MILP raw CSV saved to: {milp_raw_path} (rows={len(df_milp_raw)})")
    print(f"[INFO] Results dir: {results_dir}")


if __name__ == "__main__":
    main()
