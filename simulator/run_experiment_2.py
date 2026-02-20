import os
import time
import random
import re
from copy import deepcopy
from datetime import datetime
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd


# ============================
# Helpers
# ============================

def safe_filename(text: str) -> str:
    text = str(text).lower().replace(" ", "_")
    return re.sub(r"[^a-z0-9_\-]", "", text)


def export_routes_to_rows(method_name, result_list, num_slices, num_vnfs_per_slice, seed, timestamp_str):
    rows = []
    for res_idx, res in enumerate(result_list):
        if not hasattr(res, "routed_vls") or res.routed_vls is None:
            continue

        for vl_key, path_nodes in res.routed_vls.items():
            if not path_nodes or len(path_nodes) < 2:
                continue

            path_edges = list(zip(path_nodes[:-1], path_nodes[1:]))

            rows.append(
                {
                    "timestamp": timestamp_str,
                    "method": method_name,
                    "num_slices": num_slices,
                    "num_vnfs_per_slice": num_vnfs_per_slice,
                    "seed": seed,
                    "result_idx": res_idx,
                    "vl_key": str(vl_key),
                    "path_nodes": "->".join(map(str, path_nodes)),
                    "path_edges": ";".join([f"{u}-{v}" for (u, v) in path_edges]),
                    "num_hops": len(path_edges),
                }
            )
    return rows


def export_milp_solution_to_rows(out, instance, method_name, num_slices, num_vnfs_per_slice, seed, timestamp_str):
    if out is None or out.get("last_result") is None:
        return []

    res = out["last_result"]
    vals = res.values
    rows = []

    for key, val in vals.items():
        if not key:
            continue

        if key[0] == "x" and len(key) == 4 and val > 0.5:
            _, s, vnf_id, node = key
            rows.append(
                {
                    "timestamp": timestamp_str,
                    "method": method_name,
                    "num_slices": num_slices,
                    "num_vnfs_per_slice": num_vnfs_per_slice,
                    "seed": seed,
                    "type": "placement",
                    "slice": s,
                    "vnf_src": vnf_id,
                    "node": node,
                    "value": float(val),
                }
            )

        if key[0] == "f" and len(key) == 6 and val > 1e-9:
            _, s, i, j, u, v = key
            rows.append(
                {
                    "timestamp": timestamp_str,
                    "method": method_name,
                    "num_slices": num_slices,
                    "num_vnfs_per_slice": num_vnfs_per_slice,
                    "seed": seed,
                    "type": "flow_edge",
                    "slice": s,
                    "vnf_src": i,
                    "vnf_dst": j,
                    "u": u,
                    "v": v,
                    "value": float(val),
                }
            )

        if key[0] == "xi" and len(key) == 2 and val > 0.5:
            _, s = key
            rows.append(
                {
                    "timestamp": timestamp_str,
                    "method": method_name,
                    "num_slices": num_slices,
                    "num_vnfs_per_slice": num_vnfs_per_slice,
                    "seed": seed,
                    "type": "xi",
                    "slice": s,
                    "value": float(val),
                }
            )

    return rows


# ============================
# Main
# ============================

def main():
    GLOBAL_SEED = 42
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    os.environ["PYTHONHASHSEED"] = str(GLOBAL_SEED)

    # --- project imports ---
    from utils.topology import topologie_finlande
    from utils.generate_slices import generate_random_slices

    # PURE A*
    from heuristics.a_star import run_astar

    # MILP
    from milp.create_instance import create_instance
    from milp.solve_gurobi_sequential import solve_two_phase_max_accept_then_min_energy
    from milp.adapter import MILPResultAdapterGurobi

    # metrics
    from utils.metrics import (
        count_accepted_slices,
        compute_energy_new,
        compute_total_bandwidth,
        compute_total_latency,
    )

    # ============================
    # Config
    # ============================
    ENTRY = 6
    MILP_TIME_LIMIT = 3600  # seconds per phase
    QUIET = True  # set False if you want to see prints

    param_grid = {
        "num_slices": [32],
        "num_vnfs_per_slice": [2, 3, 4, 5, 6],
        "seed": [1],
    }

    vnf_profiles = []
    for cpu in range(1, 7):
        for factor in [10, 15, 20, 25, 30]:
            throughput = cpu * factor
            latency = int(15 + cpu * factor * 0.8)
            vnf_profiles.append({"cpu": cpu, "throughput": throughput, "latency": latency})

    # ============================
    # Output dir
    # ============================
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = Path("./results") / safe_filename(timestamp)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Results dir: {results_dir}")

    # ============================
    # Topology
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

            print(f"\n[INFO] Pre-generate pool: max_slices={max_slices}, vnfs={num_vnfs}, seed={seed}")
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

                print(f"\n[INFO] Run: slices={num_slices}, vnfs/slice={num_vnfs}, total_vnfs={total_vnfs}, seed={seed}")
                slices = deepcopy(slice_pool[:num_slices])

                method_results = {}
                method_times = {}

                # ----------------------------
                # A* (pure)
                # ----------------------------
                print("[INFO] Running A* (pure)...")
                start = time.time()
                try:
                    if QUIET:
                        with redirect_stdout(open(os.devnull, "w")):
                            _, res_list = run_astar(G, slices, node_capacity_base, link_capacity_base, csv_path=None)
                    else:
                        _, res_list = run_astar(G, slices, node_capacity_base, link_capacity_base, csv_path=None)

                    method_results["A*"] = res_list
                except Exception as e:
                    print(f"[ERROR] A* failed: {e}")
                    method_results["A*"] = []
                method_times["A*"] = time.time() - start

                # ----------------------------
                # MILP
                # ----------------------------
                print("[INFO] Running MILP (Gurobi two-phase)...")
                start = time.time()
                try:
                    instance = create_instance(G, slices)
                    instance.entry_node = ENTRY
                    instance.entry_required_s = {s: False for s in instance.S}

                    if QUIET:
                        with redirect_stdout(open(os.devnull, "w")):
                            out = solve_two_phase_max_accept_then_min_energy(
                                instance=instance,
                                slice_set=list(instance.S),
                                msg=False,
                                time_limit_phase1=MILP_TIME_LIMIT,
                                time_limit_phase2=MILP_TIME_LIMIT,
                            )
                    else:
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
                except Exception as e:
                    print(f"[ERROR] MILP failed: {e}")
                    method_results["MILP_Gurobi"] = []
                method_times["MILP_Gurobi"] = time.time() - start

                # ----------------------------
                # Exports + Metrics
                # ----------------------------
                for method_name, result_list in method_results.items():
                    if not result_list:
                        continue

                    # routes
                    try:
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
                    except Exception as e:
                        print(f"[WARN] Route export failed for {method_name}: {e}")

                    # metrics
                    try:
                        accepted = count_accepted_slices(result_list, slices)
                        total_energy = compute_energy_new(result_list, slices, node_capacity_base, link_capacity_base)
                        total_bw = sum(b for b in compute_total_bandwidth(result_list, slices) if b)
                        total_lat = sum(l for l in compute_total_latency(result_list, link_latency) if l)

                        records_metrics.append(
                            {
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
                            }
                        )
                    except Exception as e:
                        print(f"[ERROR] Metrics failed for {method_name}: {e}")

    # ============================
    # Save CSVs
    # ============================
    df_metrics = pd.DataFrame(records_metrics)
    df_routes = pd.DataFrame(records_routes)
    df_milp_raw = pd.DataFrame(records_milp_raw)

    metrics_path = results_dir / "compare_astar_vs_milp_metrics.csv"
    routes_path = results_dir / "compare_astar_vs_milp_routes.csv"
    milp_raw_path = results_dir / "milp_raw_vars.csv"

    df_metrics.to_csv(metrics_path, index=False)
    df_routes.to_csv(routes_path, index=False)
    df_milp_raw.to_csv(milp_raw_path, index=False)

    print(f"\n[INFO] Saved: {metrics_path} (rows={len(df_metrics)})")
    print(f"[INFO] Saved: {routes_path} (rows={len(df_routes)})")
    print(f"[INFO] Saved: {milp_raw_path} (rows={len(df_milp_raw)})")


if __name__ == "__main__":
    main()
