def allocate_vnfs_first_fit(path, num_vnfs, node_capacity_map):
    """
    Try to allocate 'num_vnfs' across the nodes in the path using First Fit strategy.
    Reduces capacity directly in node_capacity_map if successful.
    Returns list of nodes where VNFs were placed or None if failed.
    """
    allocation = []
    for node in path:
        if node_capacity_map[node] > 0:
            allocation.append(node)
            node_capacity_map[node] -= 1
            if len(allocation) == num_vnfs:
                return allocation
    return None
