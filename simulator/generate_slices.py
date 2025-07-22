vnf_profiles = {
    "vnf1": {"cpu": 2, "throughput": 100},
    "vnf2": {"cpu": 3, "throughput": 80},
    "vnf3": {"cpu": 1, "throughput": 120}
}

def generate_random_slices(G, vnf_profiles, num_slices):
    import random
    slices = []
    for i in range(num_slices):
        profile = random.choice(list(vnf_profiles.keys()))
        vnf = vnf_profiles[profile]
        src, dst = random.sample(list(G.nodes), 2)
        vnfs = [{"id": f"vnf{i}_{j}", "cpu": vnf["cpu"]} for j in range(3)]
        vls = [
            {"from": vnfs[0]["id"], "to": vnfs[1]["id"], "bandwidth": vnf["throughput"], "latency": 25},
            {"from": vnfs[1]["id"], "to": vnfs[2]["id"], "bandwidth": vnf["throughput"], "latency": 25}
        ]
        slices.append((vnfs, vls))
    return slices
