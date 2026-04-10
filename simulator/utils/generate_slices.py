import random

def generate_random_slices(G, vnf_profiles, num_slices, num_vnfs_per_slice, entry=None):
    """
    Generate slices with a per-slice entry node.
    If entry is provided, all slices use that fixed entry.
    If entry is None, each slice gets a random entry node.
    """
    slices = []
    all_nodes = list(G.nodes())

    for s in range(num_slices):
        current_entry = entry if entry is not None else random.choice(all_nodes)

        vnfs = []
        for i in range(num_vnfs_per_slice):
            vnf = dict(vnf_profiles[i])
            vnf["id"] = f"vnf{s}_{i}"
            vnf["slice"] = s
            vnfs.append(vnf)

        vlinks = [
            {
                "from": vnfs[i]["id"],
                "to": vnfs[i + 1]["id"],
                "bandwidth": vnfs[i]["throughput"],
                "latency": vnfs[i]["latency"],
            }
            for i in range(num_vnfs_per_slice - 1)
        ]

        slices.append((vnfs, vlinks, current_entry))

    return slices