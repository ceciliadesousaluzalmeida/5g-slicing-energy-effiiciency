from .formulation import MILPInstance

def build_instance_from_topology(G, slices):
    # Sets
    N = list(G.nodes())
    E = list(G.edges())
    S = [f"s{s_id}" for s_id in range(len(slices))]

    # VNFs per slice
    V_of_s = {}
    CPU_i = {}
    for s_id, (vnfs, vls) in enumerate(slices):
        sid = f"s{s_id}"
        V_of_s[sid] = []
        for v in vnfs:
            V_of_s[sid].append(v["id"])
            CPU_i[v["id"]] = v["cpu"]

    # Node CPU capacities
    CPUcap = {n: G.nodes[n].get("cpu", 10) for n in N}

    # Bandwidth demands per slice (simplify: max throughput among VNFs)
    BW_s = {}
    for s_id, (vnfs, vls) in enumerate(slices):
        sid = f"s{s_id}"
        BW_s[sid] = max(v["throughput"] for v in vnfs)

    # Link capacities
    BWcap = {}
    Lat = {}
    for e in E:
        BWcap[e] = G.edges[e].get("bandwidth", 100)
        Lat[e] = G.edges[e].get("latency", 1)

    # Latency budgets (simplify: max latency among VNFs of the slice)
    Ls = {}
    for s_id, (vnfs, vls) in enumerate(slices):
        sid = f"s{s_id}"
        Ls[sid] = max(v["latency"] for v in vnfs)

    return MILPInstance(G, slices)
