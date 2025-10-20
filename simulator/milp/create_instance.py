from types import SimpleNamespace

def create_instance(G, slices):
    """
    Build a MILP-compatible instance with per-VL (i->j) bandwidth and latency.
    Supports slices of forms:
      (vnf_chain, vl_chain)
      (vnf_chain, vl_chain, entry)
    EXIT was removed.
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
    instance.BW_sij = {}   # bandwidth per (s, i, j)
    instance.L_sij = {}    # latency per (s, i, j)
    instance.BW_entry = {} # bandwidth from ENTRY->first VNF
    instance.L_entry = {}  # latency from ENTRY->first VNF
    instance.entry_of_s = {}

    # --- Build per-slice data ---
    for s_idx, slice_data in enumerate(slices):
        if len(slice_data) == 2:
            vnf_chain, vl_chain = slice_data
            entry = None
        elif len(slice_data) == 3:
            vnf_chain, vl_chain, entry = slice_data
        else:
            raise ValueError(f"Unexpected slice structure ({len(slice_data)} elements): {slice_data}")

        # Register VNF list
        vnf_ids = [vnf["id"] for vnf in vnf_chain]
        instance.V_of_s[s_idx] = vnf_ids

        # CPU per VNF
        for vnf in vnf_chain:
            instance.CPU_i[vnf["id"]] = vnf["cpu"]

        # Per-VL attributes
        for q in range(len(vnf_ids) - 1):
            i, j = vnf_ids[q], vnf_ids[q + 1]
            vl = vl_chain[q]
            instance.BW_sij[(s_idx, i, j)] = float(vl["bandwidth"])
            instance.L_sij[(s_idx, i, j)] = float(vl["latency"])

        # ENTRY (optional)
        if entry is not None:
            instance.entry_of_s[s_idx] = entry
            if vl_chain:
                instance.BW_entry[s_idx] = float(vl_chain[0]["bandwidth"])
                instance.L_entry[s_idx] = float(vl_chain[0]["latency"])
            else:
                instance.BW_entry[s_idx] = 0.0
                instance.L_entry[s_idx] = 0.0

    # Default ENTRY node if none given
    if not instance.entry_of_s:
        instance.entry_node = 2

    return instance
