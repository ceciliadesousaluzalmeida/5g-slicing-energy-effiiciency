import os
import re
import random
from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd

from experiments.config import (
    GLOBAL_SEED,
    MILP_TIME_LIMIT,
    MAX_MILP_SLICES,
    MAX_MILP_VNFS_TOTAL,
    PARAM_GRID,
    VNF_PROFILES,
)

from experiments.runners import run_heuristics, run_milp_if_allowed
from experiments.metrics_pipeline import build_metrics_records
from experiments.exporters import build_export_rows, export_milp_solution_to_rows

from utils.topology import topologie_finlande
from utils.generate_slices import generate_random_slices


def safe_filename(text: str) -> str:
    # Convert text into a filesystem-safe name.
    text = str(text).lower()
    text = text.replace(" ", "_")
    return re.sub(r"[^a-z0-9_\-]", "", text)


def setup_seed(seed: int = GLOBAL_SEED):
    # Set global random seeds for reproducibility.
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_topology():
    # Load topology and extract capacities/latencies.
    G = topologie_finlande()

    node_capacity_base = {
        n: G.nodes[n]["cpu"]
        for n in G.nodes
    }

    link_capacity_base = {
        (u, v): G[u][v]["bandwidth"]
        for u, v in G.edges
    }
    link_capacity_base.update({
        (v, u): G[u][v]["bandwidth"]
        for u, v in G.edges
    })

    link_latency = {
        (u, v): G[u][v]["latency"]
        for u, v in G.edges
    }
    link_latency.update({
        (v, u): G[u][v]["latency"]
        for u, v in G.edges
    })

    return G, node_capacity_base, link_capacity_base, link_latency


def choose_entry_nodes_per_slice(G, num_slices, seed):
    # Choose one deterministic random entry node per slice.
    rng = random.Random(seed)
    nodes = sorted(list(G.nodes))
    return [rng.choice(nodes) for _ in range(num_slices)]


def set_entry_for_slice(slice_obj, entry_node):
    # Add or replace the entry node in a slice object.
    if isinstance(slice_obj, dict):
        new_sl = deepcopy(slice_obj)
        new_sl["entry"] = entry_node
        return new_sl

    if isinstance(slice_obj, tuple):
        items = list(slice_obj)
        if len(items) >= 3:
            items[2] = entry_node
        else:
            items.append(entry_node)
        return tuple(items)

    if isinstance(slice_obj, list):
        items = list(slice_obj)
        if len(items) >= 3:
            items[2] = entry_node
        else:
            items.append(entry_node)
        return items

    return slice_obj


def set_entry_for_slices(slices, entry_nodes):
    # Apply one entry node per slice.
    return [
        set_entry_for_slice(sl, entry_node)
        for sl, entry_node in zip(slices, entry_nodes)
    ]


def generate_slice_pools(G):
    # Pre-generate one fixed slice pool for each VNF-chain size.
    max_slices = max(PARAM_GRID["num_slices"])
    fixed_slice_pools = {}

    for num_vnfs in PARAM_GRID["num_vnfs_per_slice"]:
        pool_seed = 1000 + num_vnfs

        random.seed(pool_seed)
        np.random.seed(pool_seed)

        print(
            f"[INFO] Pre-generating slice pool: "
            f"max_slices={max_slices}, VNFs={num_vnfs}, seed={pool_seed}"
        )

        fixed_slice_pools[num_vnfs] = generate_random_slices(
            G,
            VNF_PROFILES,
            num_slices=max_slices,
            num_vnfs_per_slice=num_vnfs,
            entry=None,
        )

    return fixed_slice_pools


def prepare_slices(slice_pool, num_slices, G, seed):
    # Select slices and assign one random entry node per slice.
    slices_fixed = deepcopy(slice_pool[:num_slices])

    entry_nodes = choose_entry_nodes_per_slice(
        G=G,
        num_slices=num_slices,
        seed=seed,
    )

    slices_with_entries = set_entry_for_slices(
        slices=slices_fixed,
        entry_nodes=entry_nodes,
    )

    return slices_with_entries, entry_nodes


def save_all_csvs(results_dir, records):
    # Save all experiment records as CSV files.
    paths = {
        "metrics": os.path.join(results_dir, "scalability_results.csv"),
        "routes": os.path.join(results_dir, "routes_all_methods.csv"),
        "milp_raw": os.path.join(results_dir, "milp_raw_vars.csv"),
        "node_hosting": os.path.join(results_dir, "node_vnfs_all_methods.csv"),
        "node_cpu": os.path.join(results_dir, "node_cpu_load_all_methods.csv"),
        "link_bw": os.path.join(results_dir, "link_bw_load_all_methods.csv"),
        "slice_entries": os.path.join(results_dir, "slice_entries.csv"),
    }

    for key, path in paths.items():
        df = pd.DataFrame(records[key])
        df.to_csv(path, index=False)
        print(f"[INFO] Saved {key}: {path} rows={len(df)}")


def main():
    setup_seed()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join("./results", safe_filename(timestamp))
    os.makedirs(results_dir, exist_ok=True)

    print(f"[INFO] Results will be saved under: {results_dir}")

    G, node_capacity_base, link_capacity_base, link_latency = load_topology()
    slice_pools = generate_slice_pools(G)

    records = {
        "metrics": [],
        "routes": [],
        "milp_raw": [],
        "node_hosting": [],
        "node_cpu": [],
        "link_bw": [],
        "slice_entries": [],
    }

    for num_vnfs in PARAM_GRID["num_vnfs_per_slice"]:
        base_slice_pool = slice_pools[num_vnfs]

        for seed in PARAM_GRID["seed"]:
            print(
                f"\n[INFO] Seed={seed} | "
                f"VNFs per slice={num_vnfs}"
            )

            for num_slices in PARAM_GRID["num_slices"]:
                total_vnfs = num_slices * num_vnfs
                ts_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                print(
                    f"\n[INFO] Running: {num_slices} slices × "
                    f"{num_vnfs} VNFs | total_vnfs={total_vnfs}"
                )

                slices, entry_nodes = prepare_slices(
                    slice_pool=base_slice_pool,
                    num_slices=num_slices,
                    G=G,
                    seed=seed,
                )

                print(
                    f"[INFO] Entry nodes: "
                    f"{entry_nodes[:10]}"
                    f"{'...' if len(entry_nodes) > 10 else ''}"
                )

                for slice_idx, (slice_data, entry_node) in enumerate(zip(slices, entry_nodes)):
                    vnfs, vls, _entry = slice_data

                    records["slice_entries"].append({
                        "timestamp": ts_now,
                        "num_slices": num_slices,
                        "num_vnfs_per_slice": num_vnfs,
                        "total_vnfs": total_vnfs,
                        "seed": seed,
                        "slice_idx": slice_idx,
                        "entry_node": entry_node,

                        # VNF information
                        "vnf_ids": ";".join(str(v["id"]) for v in vnfs),
                        "vnf_cpu": ";".join(str(v.get("cpu", "")) for v in vnfs),

                        # Virtual links
                        "vls": ";".join(
                            f"{vl['from']}->{vl['to']}"
                            for vl in vls
                        ),
                        "vl_bandwidths": ";".join(
                            str(vl.get("bandwidth", ""))
                            for vl in vls
                        ),
                        "vl_latencies": ";".join(
                            str(vl.get("latency", ""))
                            for vl in vls
                        ),
                        "slice_cpu_total": sum(v.get("cpu", 0) for v in vnfs),
                        "slice_bandwidth_total": sum(vl.get("bandwidth", 0) for vl in vls),
                        "slice_latency_budget": sum(vl.get("latency", 0) for vl in vls),
                    })

                method_results, method_times = run_heuristics(
                    G=G,
                    slices=slices,
                    node_capacity_base=node_capacity_base,
                    link_capacity_base=link_capacity_base,
                    link_latency=link_latency,
                )

                milp_results, milp_times, milp_out, milp_instance = run_milp_if_allowed(
                    G=G,
                    slices=slices,
                    entry_node=None,
                    num_slices=num_slices,
                    total_vnfs=total_vnfs,
                    max_milp_slices=MAX_MILP_SLICES,
                    max_milp_vnfs_total=MAX_MILP_VNFS_TOTAL,
                    milp_time_limit=MILP_TIME_LIMIT,
                )

                method_results.update(milp_results)
                method_times.update(milp_times)

                records["metrics"].extend(
                    build_metrics_records(
                        method_results=method_results,
                        method_times=method_times,
                        slices=slices,
                        node_capacity_base=node_capacity_base,
                        link_capacity_base=link_capacity_base,
                        link_latency=link_latency,
                        num_slices=num_slices,
                        num_vnfs=num_vnfs,
                        total_vnfs=total_vnfs,
                        seed=seed,
                        entry_node="per_slice_random",
                        timestamp_str=ts_now,
                        milp_out=milp_out,
                        milp_instance=milp_instance,
                    )
                )

                export_rows = build_export_rows(
                    method_results=method_results,
                    slices=slices,
                    num_slices=num_slices,
                    num_vnfs_per_slice=num_vnfs,
                    seed=seed,
                    timestamp_str=ts_now,
                    node_capacity_base=node_capacity_base,
                    link_capacity_base=link_capacity_base,
                )

                records["routes"].extend(export_rows["routes"])
                records["node_hosting"].extend(export_rows["node_hosting"])
                records["node_cpu"].extend(export_rows["node_cpu"])
                records["link_bw"].extend(export_rows["link_bw"])

                if milp_out is not None and milp_instance is not None:
                    records["milp_raw"].extend(
                        export_milp_solution_to_rows(
                            out=milp_out,
                            instance=milp_instance,
                            method_name="MILP_Gurobi",
                            num_slices=num_slices,
                            num_vnfs_per_slice=num_vnfs,
                            seed=seed,
                            timestamp_str=ts_now,
                        )
                    )

    save_all_csvs(results_dir, records)
    print(f"[INFO] Done. Results dir: {results_dir}")


if __name__ == "__main__":
    main()