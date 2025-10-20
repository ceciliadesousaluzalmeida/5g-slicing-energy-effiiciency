from types import SimpleNamespace

def create_instance(G, slices):
    """
    Build a MILP-compatible instance from graph G and slices.
    Supports slices of forms:
      (vnf_chain, vl_chain)
      (vnf_chain, vl_chain, entry)
      (vnf_chain, vl_chain, entry, exit_)
    """
    instance = SimpleNamespace()

    # --- Basic sets ---
    instance.N = list(G.nodes)
    instance.E = list(G.edges)
    instance.S = list(range(len(slices)))

    # --- Link attributes ---
    instance.lat_e = {(u, v): G[u][v].get("latency", 1.0) for u, v in G.edges}
    instance.BW_cap = {(u, v): G[u][v].get("bandwidth", 100.0) for u, v in G.edges}

    # --- Node capacities ---
    instance.CPU_cap = {n: G.nodes[n].get("cpu", 100.0) for n in G.nodes}

    # --- Slice-level attributes ---
    instance.V_of_s = {}
    instance.CPU_i = {}
    instance.BW_s = {}
    instance.L_s = {}

    # Optional per-slice entry/exit dictionaries
    instance.entry_of_s = {}
    instance.exit_of_s = {}

    for s_idx, slice_data in enumerate(slices):
        # Normalize tuple forms
        if len(slice_data) == 2:
            vnf_chain, vl_chain = slice_data
            entry = None
            exit_ = None
        elif len(slice_data) == 3:
            vnf_chain, vl_chain, entry = slice_data
            exit_ = None
        elif len(slice_data) == 4:
            vnf_chain, vl_chain, entry, exit_ = slice_data
        else:
            raise ValueError(f"Unexpected slice structure ({len(slice_data)} elements): {slice_data}")

        # Register VNFs
        vnf_ids = [vnf["id"] for vnf in vnf_chain]
        instance.V_of_s[s_idx] = vnf_ids

        # CPU requirements
        for vnf in vnf_chain:
            instance.CPU_i[vnf["id"]] = vnf["cpu"]

        # Average bandwidth and max latency for the slice
        if vl_chain:
            bw_avg = sum(vl["bandwidth"] for vl in vl_chain) / len(vl_chain)
            lat_max = max(vl["latency"] for vl in vl_chain)
        else:
            bw_avg = 0.0
            lat_max = 0.0

        instance.BW_s[s_idx] = bw_avg
        instance.L_s[s_idx] = lat_max

        # Optional ENTRY/EXIT registration
        if entry is not None:
            instance.entry_of_s[s_idx] = entry
        if exit_ is not None:
            instance.exit_of_s[s_idx] = exit_

    # Default ENTRY/EXIT (if none provided at all)
    if not instance.entry_of_s:
        instance.entry_node = 2
    if not instance.exit_of_s:
        instance.exit_node = None

    return instance
