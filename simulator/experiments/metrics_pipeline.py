from utils.metrics import (
    count_accepted_slices,
    compute_energy_new,
    compute_total_bandwidth,
    compute_total_latency,
    compute_milp_bandwidth_latency,
)


def build_metrics_records(
    method_results,
    method_times,
    slices,
    node_capacity_base,
    link_capacity_base,
    link_latency,
    num_slices,
    num_vnfs,
    total_vnfs,
    seed,
    entry_node,
    timestamp_str,
    milp_out=None,
    milp_instance=None,
):
    rows = []

    for method_name, result_list in method_results.items():
        if not result_list:
            continue

        if method_name == "MILP_Gurobi" and milp_out is not None:
            accepted = len(milp_out.get("accepted_slices", []))
        else:
            accepted = count_accepted_slices(result_list, slices)

        total_energy = compute_energy_new(
            result_list,
            slices,
            node_capacity_base,
            link_capacity_base,
        )

        if (
            method_name == "MILP_Gurobi"
            and milp_out is not None
            and milp_instance is not None
            and milp_out.get("last_result") is not None
        ):
            per_slice_bw, per_slice_lat, _, _ = compute_milp_bandwidth_latency(
                milp_out["last_result"],
                milp_instance,
            )

            total_bw = sum(per_slice_bw.values())
            total_lat = sum(per_slice_lat.values())

        else:
            total_bw = sum(
                b for b in compute_total_bandwidth(result_list, slices)
                if b is not None
            )

            total_lat = sum(
                l for l in compute_total_latency(result_list, slices, link_latency)
                if l is not None
            )

        rows.append({
            "timestamp": timestamp_str,
            "num_slices": num_slices,
            "num_vnfs_per_slice": num_vnfs,
            "total_vnfs": total_vnfs,
            "seed": seed,
            "entry_node": entry_node,
            "method": method_name,
            "accepted": accepted,
            "total_energy": total_energy,
            "total_bandwidth": total_bw,
            "total_latency": total_lat,
            "runtime_sec": method_times.get(method_name),
        })

    return rows