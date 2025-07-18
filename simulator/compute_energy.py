def compute_energy(abo_results, slices, node_capacity, a=5, b=2):
    cpu_usage = {n: 0 for n in node_capacity}
    for result in abo_results:
        if result:
            for vnf_id, node in result.placed_vnfs.items():
                slice_id = int(vnf_id.split('_')[0].replace('vnf', ''))
                cpu = slices[slice_id][0][int(vnf_id.split('_')[1])]["cpu"]
                cpu_usage[node] += cpu
    energy = {node: a + b * cpu for node, cpu in cpu_usage.items() if cpu > 0}
    return cpu_usage, energy