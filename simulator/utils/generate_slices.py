import random


def generate_random_slices(G, vnf_profiles, num_slices, num_vnfs_per_slice, entry=None):
    """
    Generate slices with randomized VNF profiles.
    Entry can be assigned later by the experiment runner.
    """
    slices = []

    for s in range(num_slices):
        vnfs = []

        for i in range(num_vnfs_per_slice):
            profile = random.choice(vnf_profiles)

            vnf = dict(profile)
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

        slices.append((vnfs, vlinks, entry))

    return slices