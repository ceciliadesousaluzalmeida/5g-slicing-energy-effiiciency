# All comments in English

from collections import defaultdict, deque


class MILPResultAdapterGurobi:
    """
    Adapter to make the MILP (Gurobi) results compatible with heuristic-style plotting.

    Supports two formats:
      1) "single/multi-slice shrink solver" format:
         - placement keys: ("x", s, vnf_id, node)
         - flow keys     : ("f", s, i, j, u, v)
         - slack keys    : ("xi", s)

      2) legacy format (if you still use it somewhere):
         - placement keys: ("v", vnf_id, node)
         - flow keys     : ("f", edge, s, (i, j)) and optional f_entry/f_exit

    Reconstructs:
      - placed_vnfs: {vnf_id -> node}
      - routed_vls : {(src, dst) -> path(list of nodes)}  (paths reconstructed from active edges)
    """

    def __init__(self, solve_result, instance):
        self.solve_result = solve_result
        self.instance = instance
        self.placed_vnfs = {}
        self.routed_vls = {}

        if not solve_result or not getattr(solve_result, "values", None):
            print("[WARN][MILPAdapter] Empty MILP result — nothing to adapt.")
            return

        self._parse_vnfs()
        self._parse_routes()

    # ---------------------------------------------------------------------
    def _parse_vnfs(self):
        """Extract placement decisions from supported key formats."""
        vals = self.solve_result.values

        # Preferred format: ("x", s, vnf_id, node)
        for key, val in vals.items():
            if not key or val <= 0.5:
                continue
            if key[0] == "x" and len(key) == 4:
                _, s, vnf_id, node = key
                self.placed_vnfs[vnf_id] = node

        # Legacy fallback: ("v", vnf_id, node)
        if not self.placed_vnfs:
            for key, val in vals.items():
                if not key or val <= 0.5:
                    continue
                if key[0] == "v" and len(key) == 3:
                    _, vnf_id, node = key
                    self.placed_vnfs[vnf_id] = node

    # ---------------------------------------------------------------------
    def _parse_routes(self):
        """
        Reconstruct all routed virtual links (VLs) from flow variables.

        Preferred format:
          ("f", s, i, j, u, v) with val > 0.5

        Legacy fallback:
          ("f", e, s, (i, j)) with val > 0.5
          plus optional ("f_entry", e, s) and ("f_exit", e, s)
        """
        vals = self.solve_result.values
        inst = self.instance

        edges_by_vl = defaultdict(list)

        # Preferred format: ("f", s, i, j, u, v)
        for key, val in vals.items():
            if not key or val <= 0.5:
                continue
            if key[0] == "f" and len(key) == 6:
                _, s, i, j, u, v = key
                edges_by_vl[(i, j)].append((u, v))

        if edges_by_vl:
            for vl_key, edges in edges_by_vl.items():
                path = self._reconstruct_path_from_edges(edges)
                if path:
                    self.routed_vls[vl_key] = path
            return

        # Legacy core VLs: ("f", e, s, (i, j))
        edges_by_vl_legacy = defaultdict(list)
        for key, val in vals.items():
            if not key or val <= 0.5:
                continue
            if key[0] == "f" and len(key) == 4:
                _, e, s, ij = key
                if isinstance(ij, tuple) and len(ij) == 2:
                    i, j = ij
                    edges_by_vl_legacy[(i, j)].append(e)

        # Legacy entry/exit legs if present
        # NOTE: these were previously stored as keys ("ENTRY", first_vnf) and (last_vnf, "EXIT")
        for key, val in vals.items():
            if not key or val <= 0.5:
                continue

            if key[0] == "f_entry" and len(key) == 3:
                _, e, s = key
                first_vnf = inst.V_of_s[s][0]
                edges_by_vl_legacy[("ENTRY", first_vnf)].append(e)

            if key[0] == "f_exit" and len(key) == 3:
                _, e, s = key
                last_vnf = inst.V_of_s[s][-1]
                edges_by_vl_legacy[(last_vnf, "EXIT")].append(e)

        for vl_key, edges in edges_by_vl_legacy.items():
            path = self._reconstruct_path_from_edges(edges)
            if path:
                self.routed_vls[vl_key] = path

    # ---------------------------------------------------------------------
    def _reconstruct_path_from_edges(self, edges):
        """
        Reconstruct a node path from a set of directed edges.

        This assumes the active edges form a simple path.
        Returns:
          - list of nodes [n1, n2, ..., nk] if reconstruction succeeds
          - None otherwise
        """
        if not edges:
            return None

        # Build adjacency and indegree/outdegree
        out_adj = defaultdict(list)
        indeg = defaultdict(int)
        outdeg = defaultdict(int)

        for (u, v) in edges:
            out_adj[u].append(v)
            outdeg[u] += 1
            indeg[v] += 1
            # ensure nodes exist in dicts
            indeg[u] += 0
            outdeg[v] += 0

        # Find a start node: outdeg=1 and indeg=0 if possible
        start_candidates = [n for n in outdeg.keys() if outdeg[n] > 0 and indeg[n] == 0]
        start = start_candidates[0] if start_candidates else None

        # Fallback: pick any node that has outgoing edges
        if start is None:
            start = next(iter(out_adj.keys()))

        # Walk greedily (assumes a path)
        path = [start]
        visited_edges = set()
        current = start

        while True:
            next_nodes = out_adj.get(current, [])
            if not next_nodes:
                break

            # If branching happens, pick the first (should not happen in a well-formed path)
            nxt = next_nodes[0]
            edge = (current, nxt)
            if edge in visited_edges:
                # cycle detected
                break
            visited_edges.add(edge)

            path.append(nxt)
            current = nxt

        # Basic sanity: path length should be at least 2
        if len(path) < 2:
            return None
        return path

    # ---------------------------------------------------------------------
    def __repr__(self):
        return f"<MILPResultAdapterGurobi | {len(self.placed_vnfs)} VNFs, {len(self.routed_vls)} VLs>"


# All comments in English

class MILPSequentialAdapter:
    """
    Adapter to make the SEQUENTIAL MILP results compatible with
    heuristic-style plotting and metric functions.

    Exposes:
      placed_vnfs: {vnf_id -> node}
      routed_vls: {(i, j) -> [n1, n2, ..., nk]}
    """

    def __init__(self, seq_result, instance):
        self.seq_result = seq_result
        self.instance = instance
        self.placed_vnfs = {}
        self.routed_vls = {}

        if (
            not seq_result
            or "slice_results" not in seq_result
            or not seq_result["slice_results"]
        ):
            print("[WARN][MILPSequentialAdapter] Empty sequential MILP result")
            return

        self._parse_vnfs()
        self._parse_routes()

    def _parse_vnfs(self):
        """Extract placement from keys ('x', s, i, n)."""
        for s, res_s in self.seq_result["slice_results"].items():
            for key, val in res_s.values.items():
                if key[0] == "x" and val > 0.5:
                    _, s_id, vnf_id, node = key
                    self.placed_vnfs[vnf_id] = node

    def _parse_routes(self):
        """Reconstruct routed paths from ('f', s, i, j, u, v)."""
        vals = self.solve_result.values
        edges_by_vl = defaultdict(list)
        for s, res_s in self.seq_result["slice_results"].items():
            edges_by_vl = defaultdict(list)

            for key, val in vals.items():
                if not key or val <= 0.5:
                    continue
                if key[0] == "f" and len(key) == 6:
                    _, s, i, j, u, v = key
                    # IMPORTANT: keep slice id in the VL key
                    edges_by_vl[(s, i, j)].append((u, v))

            if edges_by_vl:
                for vl_key, edges in edges_by_vl.items():
                    path = self._reconstruct_path_from_edges(edges)
                    if path:
                        self.routed_vls[vl_key] = path
                return

    def _reconstruct_path_from_edges(self, edges):
        """Reconstruct a node path from a set of directed edges."""
        if not edges:
            return None

        out_adj = defaultdict(list)
        indeg = defaultdict(int)
        outdeg = defaultdict(int)

        for (u, v) in edges:
            out_adj[u].append(v)
            outdeg[u] += 1
            indeg[v] += 1
            indeg[u] += 0
            outdeg[v] += 0

        start_candidates = [n for n in outdeg.keys() if outdeg[n] > 0 and indeg[n] == 0]
        start = start_candidates[0] if start_candidates else None
        if start is None:
            start = next(iter(out_adj.keys()))

        path = [start]
        visited_edges = set()
        current = start

        while True:
            next_nodes = out_adj.get(current, [])
            if not next_nodes:
                break

            nxt = next_nodes[0]
            edge = (current, nxt)
            if edge in visited_edges:
                break
            visited_edges.add(edge)

            path.append(nxt)
            current = nxt

        if len(path) < 2:
            return None
        return path
