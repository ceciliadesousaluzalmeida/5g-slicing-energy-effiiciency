def sanity_check_milp_gurobi(res, instance, slices, eps=1e-6):
    """
    Sanity check for MILP (Gurobi) results.
    Prints node CPU usage, slice acceptance, and routing correctness.
    Compatible with GurobiSolveResult and slices containing (vnfs, vls[, entry, exit_]).
    """

    print("=== Sanity check MILP (Gurobi) ===")
    if res.objective is not None:
        print(f"Objective value: {res.objective:.3f}")
    else:
        print("Objective value: None (no optimal solution)")
    print(f"Status: {res.status_str}")

    var_dict = getattr(res, "values", {})  # expected: dict with ('v', i, n), ('f', e, s, (i,j)), etc.

    # --- Helper to extract vnfs and vls from flexible slice tuples ---
    def _get_vnfs_vls(slice_data):
        if isinstance(slice_data, (list, tuple)):
            if len(slice_data) >= 2:
                return slice_data[0], slice_data[1]
            elif len(slice_data) == 1:
                return slice_data[0], []
        return [], []

    # --- CPU usage per node ---
    cpu_used = {n: 0 for n in instance.N}
    for key, value in var_dict.items():
        if not isinstance(key, tuple):
            continue
        if key[0] == "v" and value > 0.5:
            _, i, n = key
            cpu_used[n] += instance.CPU_i[i]

    print("\nNode CPU usage:")
    for n in instance.N:
        used = cpu_used[n]
        cap = instance.CPU_cap[n]
        warn = "⚠️" if used > cap + eps else ""
        print(f"  Node {n}: used {used:.2f} / cap {cap:.2f} {warn}")

    # --- Slice acceptance check ---
    accepted = 0
    for s in instance.S:
        vnfs, vls = _get_vnfs_vls(slices[s])
        ok = True
        print(f"\n[Slice {s}]")

        # --- VNFs ---
        for v in vnfs:
            i = v["id"]
            assigned_nodes = [
                n for n in instance.N
                if ("v", i, n) in var_dict and var_dict[("v", i, n)] > 0.5
            ]
            if assigned_nodes:
                print(f"  VNF {i} → Node(s) {assigned_nodes}")
            else:
                print(f"  VNF {i} ✗ NOT allocated")
                ok = False

        # --- VLs ---
        for vl in vls:
            i, j = vl["from"], vl["to"]
            used_edges = []
            for key, value in var_dict.items():
                if not isinstance(key, tuple):
                    continue
                if key[0] == "f" and value > 0.5:
                    _, e, s_idx, (src, dst) = key
                    if s_idx == s and {src, dst} == {i, j}:
                        used_edges.append(e)
            if used_edges:
                print(f"  VL ({i}->{j}) ✓ routed via {used_edges}")
            else:
                print(f"  VL ({i}->{j}) ✗ NOT routed")
                ok = False

        # --- Entry/Exit edges (optional) ---
        entry = getattr(instance, "entry_node", None)
        exit_ = getattr(instance, "exit_node", None)
        if entry is not None:
            print(f"  ENTRY node: {entry}")
        if exit_ is not None:
            print(f"  EXIT node: {exit_}")

        # --- Status per slice ---
        if ok:
            print(f"  → Slice {s} ACCEPTED ✓")
            accepted += 1
        else:
            print(f"  → Slice {s} REJECTED ✗")

    print(f"\nTotal slices accepted: {accepted}/{len(instance.S)}")
