def compute_energy_per_slice(results, slices, node_capacity, a=5, b=2):
    slice_energy = []
    slice_cpu_usage = []

    for slice_id, result in enumerate(results):
        if result:
            cpu_usage = {n: 0 for n in node_capacity}
            for vnf_id, node in result.placed_vnfs.items():
                vnf_index = int(vnf_id.split('_')[1])
                cpu = slices[slice_id][0][vnf_index]["cpu"]
                cpu_usage[node] += cpu

            energy = sum(a + b * cpu for cpu in cpu_usage.values() if cpu > 0)
        else:
            cpu_usage = {n: 0 for n in node_capacity}
            energy = None

        slice_energy.append(energy)
        slice_cpu_usage.append(cpu_usage)

    return slice_energy, slice_cpu_usage  


def compute_total_bandwidth(results, slices):
    total_bandwidth = []
    for slice_id, result in enumerate(results):
        if result:
            used = 0
            vl_chain = slices[slice_id][1]
            for vl in vl_chain:
                if (vl["from"], vl["to"]) in result.routed_vls:
                    used += vl["bandwidth"]
            total_bandwidth.append(used)
        else:
            total_bandwidth.append(None)
    return total_bandwidth


def compute_total_latency(results, link_latency):
    total_latency = []
    for result in results:
        if result:
            slice_latency = 0
            for path in result.routed_vls.values():
                path_latency = sum(link_latency[(path[i], path[i+1])] for i in range(len(path)-1))
                slice_latency += path_latency
            total_latency.append(slice_latency)
        else:
            total_latency.append(None)
    return total_latency

