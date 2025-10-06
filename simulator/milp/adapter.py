# milp/adapter.py
# All comments in English

class MILPResultAdapterCBC:
    def __init__(self, pulp_result, instance):
        vals = pulp_result.values
        self.placed_vnfs = {}
        self.routed_vls = {}

        # VNFs (argmax)
        for s in instance.S:
            for i in instance.V_of_s[s]:
                best_n, best_v = None, -1
                for n in instance.N:
                    vval = (vals.get(("v", i, n), 0.0) or 0.0)
                    if vval > best_v:
                        best_v, best_n = vval, n
                if best_n is not None:
                    self.placed_vnfs[i] = best_n

        # VLs: coletar edges onde f=1 e construir caminho como lista de nós
        for s in instance.S:
            vnf_ids = instance.V_of_s[s]
            for q in range(len(vnf_ids) - 1):
                i, j = vnf_ids[q], vnf_ids[q + 1]
                edges = []
                for e in instance.E:
                    if (vals.get(("f", e, s, (i, j)), 0.0) or 0.0) > 0.5:
                        edges.append(e)

                # try to turn edge list into a node path
                path_nodes = None
                if edges:
                    # build adjacency from edges
                    from collections import defaultdict, deque
                    adj = defaultdict(list)
                    for u, v in edges:
                        adj[u].append(v)
                        adj[v].append(u)
                    # try to find an Euler-like simple path (start at a node with degree 1)
                    endpoints = [x for x in adj if len(adj[x]) == 1]
                    start = endpoints[0] if endpoints else edges[0][0]
                    visited = set()
                    order = [start]
                    cur = start
                    prev = None
                    while True:
                        nxts = [x for x in adj[cur] if (cur, x) not in visited and (x, cur) not in visited]
                        if not nxts:
                            break
                        nxt = nxts[0]
                        visited.add((cur, nxt))
                        visited.add((nxt, cur))
                        order.append(nxt)
                        prev, cur = cur, nxt
                    path_nodes = order

                # salve com as duas chaves compatíveis
                key_milp = (s, (i, j))
                self.routed_vls[key_milp] = path_nodes if path_nodes else edges
                # chave plana por VNF (para heurísticas/compute_*):
                self.routed_vls[(i, j)] = path_nodes if path_nodes else edges

    
    def __repr__(self):
        return f"<MILPResultAdapterCBC placed={len(self.placed_vnfs)} vls={len(self.routed_vls)}>"
