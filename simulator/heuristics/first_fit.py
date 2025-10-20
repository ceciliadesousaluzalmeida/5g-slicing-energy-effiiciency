# All comments in English
import networkx as nx
import pandas as pd
from copy import deepcopy

class FFState:
    def __init__(self, placed_vnfs=None, routed_vls=None, g_cost=0.0):
        self.placed_vnfs = placed_vnfs or {}
        self.routed_vls = routed_vls or {}
        self.g_cost = g_cost

    def is_goal(self, vnf_chain, vl_chain, entry=None):
        """Goal: all VNFs placed and all VLs routed."""
        return (len(self.placed_vnfs) == len(vnf_chain)
                and len(self.routed_vls) >= len(vl_chain))


def run_first_fit(G, slices, node_capacity_base, link_capacity_base, link_latency, csv_path=None):
    results = []
    full_results = []

    node_capacity_global = deepcopy(node_capacity_base)
    link_capacity_global = deepcopy(link_capacity_base)

    # --------------- helper: feasible shortest path ---------------
    def shortest_path_with_capacity(G, u, v, link_capacity, bandwidth):
        try:
            path = nx.shortest_path(G, u, v, weight="latency")
        except nx.NetworkXNoPath:
            return None, None
        for a, b in zip(path[:-1], path[1:]):
            cap = link_capacity.get((a, b), link_capacity.get((b, a), 0))
            if cap < bandwidth:
                return None, None
        latency = sum(link_latency.get((a, b), link_latency.get((b, a), 1.0))
                      for a, b in zip(path[:-1], path[1:]))
        return path, latency

    # --------------- main loop over slices ---------------
    for i, slice_data in enumerate(slices, start=1):
        if len(slice_data) == 2:
            vnf_chain, vl_chain = slice_data
            entry = None
        elif len(slice_data) == 3:
            vnf_chain, vl_chain, entry = slice_data
        else:
            raise ValueError(f"Unexpected slice format: {slice_data}")

        print(f"\n[INFO][FF] === Solving slice {i} with {len(vnf_chain)} VNFs and {len(vl_chain)} VLs ===")

        placed_vnfs = {}
        routed_vls = {}
        g_cost = 0.0
        success = True

        local_node_capacity = deepcopy(node_capacity_global)
        local_link_capacity = deepcopy(link_capacity_global)

        for vnf in vnf_chain:
            vnf_id = vnf["id"]
            cpu_need = vnf["cpu"]
            slice_id = vnf["slice"]

            placed = False
            for node in sorted(G.nodes):
                avail_cpu = local_node_capacity.get(node, 0)
                if avail_cpu < cpu_need:
                    continue

                # Anti-affinity: same slice cannot share node
                if any(node == placed_node and slice_id == next(v["slice"] for v in vnf_chain if v["id"] == other_id)
                       for other_id, placed_node in placed_vnfs.items()):
                    continue

                temp_placed = placed_vnfs.copy()
                temp_routed = routed_vls.copy()
                temp_node_capacity = deepcopy(local_node_capacity)
                temp_link_capacity = deepcopy(local_link_capacity)
                temp_g_cost = g_cost
                routing_ok = True

                temp_node_capacity[node] -= cpu_need
                temp_placed[vnf_id] = node

                # Route internal VLs
                for vl in vl_chain:
                    src, dst = vl["from"], vl["to"]
                    if src in temp_placed and dst in temp_placed and (src, dst) not in temp_routed:
                        src_node, dst_node = temp_placed[src], temp_placed[dst]
                        path, lat = shortest_path_with_capacity(G, src_node, dst_node,
                                                                temp_link_capacity, vl["bandwidth"])
                        if path is None:
                            routing_ok = False
                            break
                        for u, v in zip(path[:-1], path[1:]):
                            bw = vl["bandwidth"]
                            if (u, v) in temp_link_capacity:
                                temp_link_capacity[(u, v)] -= bw
                            else:
                                temp_link_capacity[(v, u)] -= bw
                        temp_routed[(src, dst)] = path
                        temp_g_cost += lat

                if not routing_ok:
                    continue

                # ENTRY → first VNF if defined
                if entry is not None and vnf_chain:
                    first_id = vnf_chain[0]["id"]
                    if ("ENTRY", first_id) not in temp_routed and first_id in temp_placed:
                        path, lat = shortest_path_with_capacity(G, entry, temp_placed[first_id],
                                                                temp_link_capacity,
                                                                vl_chain[0]["bandwidth"] if vl_chain else 0)
                        if path is None:
                            routing_ok = False
                        else:
                            for u, v in zip(path[:-1], path[1:]):
                                bw = vl_chain[0]["bandwidth"] if vl_chain else 0
                                if (u, v) in temp_link_capacity:
                                    temp_link_capacity[(u, v)] -= bw
                                else:
                                    temp_link_capacity[(v, u)] -= bw
                            temp_routed[("ENTRY", first_id)] = path
                            temp_g_cost += lat

                if not routing_ok:
                    continue

                # Commit local placement
                placed_vnfs = temp_placed
                routed_vls = temp_routed
                local_node_capacity = temp_node_capacity
                local_link_capacity = temp_link_capacity
                g_cost = temp_g_cost
                placed = True
                print(f"[INFO][FF] Placed {vnf_id} on node {node} "
                      f"(use={cpu_need}, remaining={local_node_capacity[node]}).")
                break

            if not placed:
                print(f"[WARN][FF] Failed to place VNF {vnf_id}, slice {i} rejected.")
                success = False
                break

        if success:
            node_capacity_global = local_node_capacity
            link_capacity_global = local_link_capacity
            results.append({"slice": i, "accepted": True, "g_cost": g_cost})
            full_results.append(FFState(placed_vnfs, routed_vls, g_cost))
            print(f"[SUMMARY][FF] Slice {i} accepted. "
                  f"Remaining min_node_cpu={min(node_capacity_global.values())}, "
                  f"links_low_bw={sum(1 for v in link_capacity_global.values() if v <= 0)}")
        else:
            results.append({"slice": i, "accepted": False, "g_cost": None})
            full_results.append(None)
            print(f"[SUMMARY][FF] Slice {i} rejected.")

    df = pd.DataFrame(results)
    if csv_path:
        try:
            df.to_csv(csv_path, index=False)
            print(f"[INFO][FF] Results written to {csv_path}")
        except Exception as e:
            print(f"[WARN][FF] Could not write CSV: {e}")

    return df, full_results
