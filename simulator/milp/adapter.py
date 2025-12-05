# All comments in English

class MILPResultAdapterGurobi:
    """
    Adapter to make the MILP (Gurobi) results compatible with heuristic-style plotting.
    Reconstructs:
      - placed_vnfs: {vnf_id -> node}
      - routed_vls: {(src_vnf, dst_vnf) -> path(list of nodes)}
    Also includes entry/exit legs if present in the MILP model.
    """

    def __init__(self, solve_result, instance):
        self.solve_result = solve_result
        self.instance = instance
        self.placed_vnfs = {}
        self.routed_vls = {}

        if not solve_result or not solve_result.values:
            print("[WARN][MILPAdapter] Empty MILP result — nothing to adapt.")
            return

        self._parse_vnfs()
        self._parse_routes()

    # ---------------------------------------------------------------------
    def _parse_vnfs(self):
        """Extract placement decisions v[i,n] = 1."""
        vals = self.solve_result.values
        for key, val in vals.items():
            if key[0] == "v" and val > 0.5:
                _, vnf_id, node = key
                self.placed_vnfs[vnf_id] = node

    # ---------------------------------------------------------------------
    def _parse_routes(self):
        """
        Reconstruct all routed virtual links (VLs) and entry/exit legs
        from f[(e,s,(i,j))], f_entry[(e,s)], and f_exit[(e,s)].
        """
        vals = self.solve_result.values
        inst = self.instance

        # --- 1️⃣ Core VLs between VNFs ---
        for key, val in vals.items():
            if not key or val <= 0.5:
                continue
            if key[0] == "f" and len(key) == 4:
                _, e, s, (i, j) = key
                self._add_edge_to_path((i, j), e)

        # --- 2️⃣ ENTRY→first VNF ---
        for key, val in vals.items():
            if not key or val <= 0.5:
                continue
            if key[0] == "f_entry" and len(key) == 3:
                _, e, s = key
                entry_node = (inst.entry_of_s[s] if hasattr(inst, "entry_of_s") and s in inst.entry_of_s
                              else getattr(inst, "entry_node", None))
                first_vnf = inst.V_of_s[s][0]
                self._add_edge_to_path(("ENTRY", first_vnf), e)

        # --- 3️⃣ Last VNF→EXIT ---
        for key, val in vals.items():
            if not key or val <= 0.5:
                continue
            if key[0] == "f_exit" and len(key) == 3:
                _, e, s = key
                exit_node = (inst.exit_of_s[s] if hasattr(inst, "exit_of_s") and s in inst.exit_of_s
                             else getattr(inst, "exit_node", None))
                last_vnf = inst.V_of_s[s][-1]
                self._add_edge_to_path((last_vnf, "EXIT"), e)

    # ---------------------------------------------------------------------
    def _add_edge_to_path(self, key, edge):
        """
        Build path progressively per VL key, preserving order.
        """
        if key not in self.routed_vls:
            self.routed_vls[key] = []
        # avoid duplicates
        if not self.routed_vls[key] or self.routed_vls[key][-1] != edge[0]:
            self.routed_vls[key].append(edge[0])
        self.routed_vls[key].append(edge[1])

    # ---------------------------------------------------------------------
    def __repr__(self):
        return f"<MILPResultAdapterGurobi | {len(self.placed_vnfs)} VNFs, {len(self.routed_vls)} VLs>"


# All comments in English

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
        for s, res_s in self.seq_result["slice_results"].items():
            for key, val in res_s.values.items():
                if key[0] == "f" and val > 0.5:
                    _, s_id, i, j, u, v = key
                    self._add_edge_to_path((i, j), (u, v))

    def _add_edge_to_path(self, key, edge):
        """Append edge (u,v) to path for logical VL (i,j)."""
        if key not in self.routed_vls:
            self.routed_vls[key] = []
        if not self.routed_vls[key] or self.routed_vls[key][-1] != edge[0]:
            self.routed_vls[key].append(edge[0])
        self.routed_vls[key].append(edge[1])
