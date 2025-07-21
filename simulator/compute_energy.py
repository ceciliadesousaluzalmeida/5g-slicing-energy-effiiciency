def compute_energy_per_slice(results, slices, node_capacity, a=5, b=2):
    slice_energy = []
    for slice_id, result in enumerate(results):
        if result:
            cpu_usage = {n: 0 for n in node_capacity}
            for vnf_id, node in result.placed_vnfs.items():
                vnf_index = int(vnf_id.split('_')[1])
                cpu = slices[slice_id][0][vnf_index]["cpu"]
                cpu_usage[node] += cpu
          
            energy = sum(a + b * cpu for cpu in cpu_usage.values() if cpu > 0)
        else:
            energy = None
        slice_energy.append(energy)
    return slice_energy
