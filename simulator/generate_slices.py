import random

vnf_profiles = [
    {"cpu": 2, "throughput": 140},
    {"cpu": 4, "throughput": 150},
    {"cpu": 6, "throughput": 160},
    {"cpu": 8, "throughput": 175}
]

def generate_random_slices(G, vnf_profiles, num_slices, num_vnfs_per_slice=3):
    slices = []
    for i in range(num_slices):
        vnfs = []
        vls = []

        for j in range(num_vnfs_per_slice):
            profile = random.choice(vnf_profiles)
            vnf_id = f"vnf{i}_{j}"
            vnfs.append({
                "id": vnf_id,
                "cpu": profile["cpu"],
                "throughput": profile["throughput"],
                "slice": i
            })

        for j in range(num_vnfs_per_slice - 1):
            vls.append({
                "from": vnfs[j]["id"],
                "to": vnfs[j + 1]["id"],
                "bandwidth": vnfs[j]["throughput"],
                "latency": random.randint(120, 250) 
            })

        slices.append((vnfs, vls))
    return slices
