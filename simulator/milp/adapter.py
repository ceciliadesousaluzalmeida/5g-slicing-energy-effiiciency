
class MILPResultAdapterGurobi:
    def __init__(self, gurobi_result, instance):
        vals = gurobi_result.values
        self.placed_vnfs = {}
        self.routed_vls = {}

        # --- VNFs placement (argmax rule) ---
        for s in instance.S:
            for i in instance.V_of_s[s]:
                best_n, best_v = None, -1
                for n in instance.N:
                    vval = float(vals.get(("v", i, n), 0.0) or 0.0)
                    if vval > best_v:
                        best_v, best_n = vval, n
                if best_n is not None:
                    self.placed_vnfs[i] = best_n

        # --- Virtual Links routing reconstruction ---
        for s in instance.S:
            vnf_ids = instance.V_of_s[s]
            for q in range(len(vnf_ids) - 1):
                i, j = vnf_ids[q], vnf_ids[q + 1]
                edges = []
                for e in instance.E:
                    if (vals.get(("f", e, s, (i, j)), 0.0) or 0.0) > 0.5:
                        edges.append(e)

                # Try to reconstruct a node path from the set of edges
                path_nodes = None
                if edges:
                    from collections import defaultdict
                    adj = defaultdict(list)
                    for u, v in edges:
                        adj[u].append(v)
                        adj[v].append(u)

                    # Detect endpoints (degree = 1) or fallback to first edge
                    endpoints = [x for x in adj if len(adj[x]) == 1]
                    start = endpoints[0] if endpoints else edges[0][0]

                    # Simple depth-first traversal to reconstruct the order
                    visited = set()
                    order = [start]
                    cur = start
                    while True:
                        nxts = [x for x in adj[cur] if (cur, x) not in visited and (x, cur) not in visited]
                        if not nxts:
                            break
                        nxt = nxts[0]
                        visited.add((cur, nxt))
                        visited.add((nxt, cur))
                        order.append(nxt)
                        cur = nxt
                    path_nodes = order

                # Save under both key forms (by slice and by VNF pair)
                key_milp = (s, (i, j))
                self.routed_vls[key_milp] = path_nodes if path_nodes else edges
                self.routed_vls[(i, j)] = path_nodes if path_nodes else edges

    def __repr__(self):
        return f"<MILPResultAdapterGurobi placed={len(self.placed_vnfs)} vls={len(self.routed_vls)}>"
