# === Build MILP instance that supports (vnf_chain, vl_chain) or (vnf_chain, vl_chain, entry, exit_) ===
from types import SimpleNamespace

def create_instance(G, slices):
    """
    Build a minimal MILP-compatible instance object from graph G and slices.
    Supports slices with or without ENTRY/EXIT info.
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

    # --- Slice data structures ---
    instance.V_of_s = {}
    instance.CPU_i = {}
    instance.BW_s = {}
    instance.L_s = {}

    # Optional per-slice entry/exit dictionaries
    instance.entry_of_s = {}
    instance.exit_of_s = {}

    for s_idx, slice_data in enumerate(slices):
        # Accept either 2-tuple or 4-tuple
        if len(slice_data) == 2:
            vnf_chain, vl_chain = slice_data
            entry = None
            exit_ = None
        elif len(slice_data) == 4:
            vnf_chain, vl_chain, entry, exit_ = slice_data
            instance.entry_of_s[s_idx] = entry
            instance.exit_of_s[s_idx] = exit_
        else:
            raise ValueError(f"Unexpected slice structure: {len(slice_data)} elements")

        # Register VNFs
        vnf_ids = [vnf["id"] for vnf in vnf_chain]
        instance.V_of_s[s_idx] = vnf_ids

        # CPU requirements
        for vnf in vnf_chain:
            instance.CPU_i[vnf["id"]] = vnf["cpu"]

        # Bandwidth = média dos VLs (ou 0 se não houver)
        if vl_chain:
            bw_avg = sum(vl["bandwidth"] for vl in vl_chain) / len(vl_chain)
            lat_max = max(vl["latency"] for vl in vl_chain)
        else:
            bw_avg = 0.0
            lat_max = 0.0

        instance.BW_s[s_idx] = bw_avg
        instance.L_s[s_idx] = lat_max

    # Caso não haja ENTRY/EXIT por slice, define globais padrão
    instance.entry_node = 2
    instance.exit_node = 9

    return instance
