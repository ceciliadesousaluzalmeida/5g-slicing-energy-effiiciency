import os
import time
import random
import re
from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd


def safe_filename(text: str) -> str:
    text = str(text).lower()
    text = text.replace(" ", "_")
    text = re.sub(r"[^a-z0-9_\-]", "", text)
    return text


def _route_key_to_parts(vl_key):
    if isinstance(vl_key, tuple) and len(vl_key) == 3:
        s, i, j = vl_key
        return s, i, j
    if isinstance(vl_key, tuple) and len(vl_key) == 2:
        i, j = vl_key
        return None, i, j
    return None, None, None


def export_routes_to_rows(method_name, result_list, num_slices, num_vnfs_per_slice, seed, timestamp_str):
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

        if key[0] == "f" and len(key) == 6 and val > 1e-9:
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

        if key[0] == "z" and len(key) == 2:
            _, s = key
            rows.append({
                "timestamp": timestamp_str,
                "method": method_name,
                "num_slices": num_slices,
                "num_vnfs_per_slice": num_vnfs_per_slice,
                "seed": seed,
                "type": "z",
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


def main():
    GLOBAL_SEED = 42
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    os.environ["PYTHONHASHSEED"] = str(GLOBAL_SEED)
    

    from milp.create_instance import create_instance
    from milp.solve_gurobi_sequential import solve_gurobi_two_phase_max_accept_then_min_energy
    from milp.adapter import MILPResultAdapterGurobi

    from utils.topology import topologie_finlande
    from utils.generate_slices import generate_random_slices

    from utils.metrics import (
        count_accepted_slices,
        compute_energy_new,
        compute_total_bandwidth,
        compute_total_latency,
    )

    MILP_TIME_LIMIT = 1800  # 30 minutes (seconds)
    ENTRY = 6

    param_grid = {
        "num_slices": [4, 8, 16, 32, 64, 128],
        "num_vnfs_per_slice": [2, 3, 4],
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
                slices = deepcopy(slice_pool[:num_slices])

                print(
                    f"\n[INFO] === MILP only: {num_slices} slices × {num_vnfs} VNFs "
                    f"(total={total_vnfs}), seed={seed} ==="
                )

                runtime = np.nan
                accepted = 0
                total_energy = np.nan
                total_bw = np.nan
                total_lat = np.nan
                log_file = os.path.join(
                            results_dir,
                            f"gurobi_"
                            f"slices{num_slices}_"
                            f"vnfs{num_vnfs}_"
                            f"seed{seed}.log"
                        )


                try:
                    print("[INFO][MILP] Running Gurobi (two-phase max-accept -> min-energy)…")
                    t0 = time.time()

                    instance = create_instance(G, slices)

                    # Keep it safe: not all instance classes have entry_node
                    try:
                        setattr(instance, "entry_node", ENTRY)
                    except Exception:
                        pass

                    out = solve_gurobi_two_phase_max_accept_then_min_energy(
                        instance,
                        msg=False,
                        time_limit=MILP_TIME_LIMIT,
                    )

                    runtime = time.time() - t0

                    if out.get("last_result") is None:
                        raise RuntimeError("MILP returned no last_result")

                    adapter = MILPResultAdapterGurobi(out["last_result"], instance)
                    result_list = [adapter]

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

                    records_routes.extend(
                        export_routes_to_rows(
                            method_name="MILP_Gurobi",
                            result_list=result_list,
                            num_slices=num_slices,
                            num_vnfs_per_slice=num_vnfs,
                            seed=seed,
                            timestamp_str=ts_now,
                        )
                    )

                    accepted = count_accepted_slices(result_list, slices)
                    total_energy = compute_energy_new(result_list, slices, node_capacity_base, link_capacity_base)
                    total_bw = sum(b for b in compute_total_bandwidth(result_list, slices) if b)
                    total_lat = sum(l for l in compute_total_latency(result_list, link_latency) if l)

                except Exception as e:
                    print(f"[ERROR][MILP] Failed: {e}")

                acceptance_rate = (accepted / num_slices) if num_slices > 0 else np.nan

                records_metrics.append({
                    "timestamp": ts_now,
                    "num_slices": num_slices,
                    "num_vnfs_per_slice": num_vnfs,
                    "total_vnfs": total_vnfs,
                    "seed": seed,
                    "method": "MILP_Gurobi",
                    "accepted": accepted,
                    "acceptance_rate": acceptance_rate,
                    "total_energy": total_energy,
                    "total_bandwidth": total_bw,
                    "total_latency": total_lat,
                    "runtime_sec": runtime,
                })

    df_metrics = pd.DataFrame(records_metrics)
    df_routes = pd.DataFrame(records_routes)
    df_milp_raw = pd.DataFrame(records_milp_raw)

    metrics_path = os.path.join(results_dir, "scalability_results.csv")
    routes_path = os.path.join(results_dir, "routes_all_methods.csv")
    milp_raw_path = os.path.join(results_dir, "milp_raw_vars.csv")
    out = solve_gurobi_two_phase_max_accept_then_min_energy(
        instance,
        msg=True,                
        time_limit=MILP_TIME_LIMIT,
        log_file_prefix=log_file, 
    )



    df_metrics.to_csv(metrics_path, index=False)
    df_routes.to_csv(routes_path, index=False)
    df_milp_raw.to_csv(milp_raw_path, index=False)

    print(f"\n[INFO] Metrics CSV saved to: {metrics_path} (rows={len(df_metrics)})")
    print(f"[INFO] Routes  CSV saved to: {routes_path} (rows={len(df_routes)})")
    print(f"[INFO] MILP raw CSV saved to: {milp_raw_path} (rows={len(df_milp_raw)})")
    print(f"[INFO] Results dir: {results_dir}")


if __name__ == "__main__":
    main()
