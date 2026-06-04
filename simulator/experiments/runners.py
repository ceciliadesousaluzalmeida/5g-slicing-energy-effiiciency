import time

from heuristics.a_star import run_astar
from heuristics.run_abo_full_batch import run_abo_full_batch
from heuristics.run_fabo_full_batch import run_fabo_full_batch
from heuristics.best_fit import run_best_fit
from heuristics.first_fit import run_first_fit
from heuristics.a_star_energy_aware import energy_aware_astar


def run_heuristics(G, slices, node_capacity_base, link_capacity_base, link_latency):
    # Run all heuristic methods and return their results and runtimes.
    methods = [
        ("A*", run_astar, (G, slices, node_capacity_base, link_capacity_base)),
        ("ABO", run_abo_full_batch, (G, slices, node_capacity_base, link_latency, link_capacity_base)),
        ("FABO", run_fabo_full_batch, (G, slices, node_capacity_base, link_latency, link_capacity_base)),
        ("Best Fit", run_best_fit, (G, slices, node_capacity_base, link_capacity_base)),
        ("First Fit", run_first_fit, (G, slices, node_capacity_base, link_capacity_base)),
        ("Energy-Aware A*", energy_aware_astar, (G, slices, node_capacity_base, link_capacity_base)),
    ]

    method_results = {}
    method_times = {}

    for name, func, args in methods:
        start = time.time()
        try:
            _, res_list = func(*args)
            method_results[name] = res_list
        except Exception as e:
            print(f"[ERROR] {name} failed: {e}")
            method_results[name] = []
        method_times[name] = time.time() - start

    return method_results, method_times


def run_milp_if_allowed(
    G,
    slices,
    entry_node,
    num_slices,
    total_vnfs,
    max_milp_slices,
    max_milp_vnfs_total,
    milp_time_limit,
):
    # Run MILP only when the instance is within the configured limits.
    if num_slices > max_milp_slices or total_vnfs > max_milp_vnfs_total:
        return {}, {}, None, None

    from milp.create_instance import create_instance
    from milp.solve_gurobi_sequential import solve_two_phase_max_accept_then_min_energy
    from milp.adapter import MILPResultAdapterGurobi

    import time

    method_results = {}
    method_times = {}

    try:
        start = time.time()

        instance = create_instance(G, slices)
        instance.entry_node = entry_node
        instance.entry_required_s = {s: False for s in instance.S}

        out = solve_two_phase_max_accept_then_min_energy(
            instance=instance,
            slice_set=list(instance.S),
            msg=False,
            time_limit_phase1=milp_time_limit,
            time_limit_phase2=milp_time_limit,
        )

        if out.get("last_result") is not None:
            adapter = MILPResultAdapterGurobi(out["last_result"], instance)
            method_results["MILP_Gurobi"] = [adapter]
        else:
            method_results["MILP_Gurobi"] = []

        method_times["MILP_Gurobi"] = time.time() - start

        return method_results, method_times, out, instance

    except Exception as e:
        print(f"[ERROR][MILP] Failed: {e}")

        method_results["MILP_Gurobi"] = []
        method_times["MILP_Gurobi"] = None

        return method_results, method_times, None, None