def generate_random_slices(G, vnf_profiles, num_slices, num_vnfs_per_slice, entry=None, exit_=None):
    """
    Generate slices with entry/exit nodes fixed (2 -> 9).
    Each slice carries its source/destination for routing.
    """
    slices = []
    for s in range(num_slices):
        vnfs = []
        for i in range(num_vnfs_per_slice):
            vnf = dict(vnf_profiles[i])
            vnf["id"] = f"vnf{s}_{i}"
            vnf["slice"] = s
            vnfs.append(vnf)

        vlinks = [
            {"from": vnfs[i]["id"], "to": vnfs[i+1]["id"],
             "bandwidth": vnfs[i]["throughput"], "latency": vnfs[i]["latency"]}
            for i in range(num_vnfs_per_slice - 1)
        ]

        slices.append((vnfs, vlinks, entry, exit_))
    return slices
