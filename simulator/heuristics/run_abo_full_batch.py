import heapq
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    from sage.all import Graph  # type: ignore
except Exception:
    Graph = None  # type: ignore


# ------------------------- small helpers -------------------------
def _as_vnf_id(v: Any) -> str:
    # Normalize VNF identifiers.
    return str(v).strip()


def _vl_key_from_to(v_from: Any, v_to: Any) -> Tuple[str, str]:
    # Canonical VL key with normalized endpoints.
    return (_as_vnf_id(v_from), _as_vnf_id(v_to))


def _vl_key(vl: Dict[str, Any]) -> Tuple[str, str]:
    # Use a single consistent key everywhere.
    return _vl_key_from_to(vl["from"], vl["to"])


def _assert_path_endpoints(path: List[Any], src_node: Any, dst_node: Any) -> None:
    # Strong validation to avoid silent bugs.
    if not path or len(path) < 2:
        raise ValueError("VL path is missing or trivial.")
    if path[0] != src_node or path[-1] != dst_node:
        raise ValueError(
            f"VL path endpoints mismatch: expected {src_node}->{dst_node}, got {path[0]}->{path[-1]}"
        )


def _edge_key(u: Any, v: Any) -> Tuple[Any, Any]:
    # Canonical undirected edge key.
    return (u, v) if u <= v else (v, u)


def _canonize_edge_dict(dct: Dict[Tuple[Any, Any], float]) -> Dict[Tuple[Any, Any], float]:
    # Convert (u,v)/(v,u) keys to canonical (min,max) once.
    out: Dict[Tuple[Any, Any], float] = {}
    for (u, v), val in dct.items():
        out[_edge_key(u, v)] = float(val)
    return out


# ------------------------- Sage adapter -------------------------
class SageGraphAdapter:
    """
    Minimal adapter around a Sage graph.
    We cache nodes and neighbors in Python to avoid Sage iterator overhead in hot loops.
    """

    def __init__(self, sage_graph: Any):
        self.G = sage_graph

    @classmethod
    def from_networkx_like(cls, nx_graph: Any) -> "SageGraphAdapter":
        if Graph is None:
            raise RuntimeError("SageMath is not available (cannot import sage.all.Graph).")

        g = Graph(multiedges=False, loops=False)
        g.add_vertices(list(nx_graph.nodes))
        for u, v in nx_graph.edges:
            g.add_edge(u, v)
        return cls(g)

    def nodes(self) -> List[Any]:
        return list(self.G.vertices())

    def neighbors(self, u: Any) -> List[Any]:
        return list(self.G.neighbors(u))


# ------------------------- ABO state -------------------------
class ABOState:
    __slots__ = ("placed_vnfs", "routed_vls", "g_cost", "node_capacity", "link_capacity")

    def __init__(
        self,
        placed_vnfs: Optional[Dict[str, Any]] = None,
        routed_vls: Optional[Dict[Tuple[str, str], List[Any]]] = None,
        g_cost: float = 0.0,
        node_capacity: Optional[Dict[Any, float]] = None,
        link_capacity: Optional[Dict[Tuple[Any, Any], float]] = None,
    ):
        self.placed_vnfs = placed_vnfs or {}
        self.routed_vls = routed_vls or {}
        self.g_cost = float(g_cost)
        self.node_capacity = node_capacity or {}
        self.link_capacity = link_capacity or {}

    def is_goal_chain(self, vnf_ids: List[str], entry: Optional[Any]) -> bool:
        """
        For a chain with VNFs [v0,v1,...], goal means all VNFs placed
        and all chain VLs routed (v0->v1, v1->v2, ...).
        If entry is provided, we also expect ("ENTRY", v0) to be routed unless co-located logic applies.
        """
        if len(self.placed_vnfs) != len(vnf_ids):
            return False

        for i in range(1, len(vnf_ids)):
            a, b = vnf_ids[i - 1], vnf_ids[i]
            na = self.placed_vnfs.get(a)
            nb = self.placed_vnfs.get(b)
            if na is None or nb is None:
                return False
            if na == nb:
                continue
            p = self.routed_vls.get((a, b))
            if not p or p[0] != na or p[-1] != nb:
                return False

        if entry is not None and vnf_ids:
            v0 = vnf_ids[0]
            n0 = self.placed_vnfs.get(v0)
            if n0 is None:
                return False
            if entry != n0:
                p = self.routed_vls.get(("ENTRY", v0))
                if not p or p[0] != entry or p[-1] != n0:
                    return False

        return True

    def __lt__(self, other: "ABOState") -> bool:
        return self.g_cost < other.g_cost


# ------------------------- main ABO (Sage + FAST runtime knobs) -------------------------
def run_abo_full_batch(
    G: Any,
    slices: List[Any],
    node_capacity_base: Dict[Any, float],
    link_latency: Dict[Tuple[Any, Any], float],
    link_capacity_base: Dict[Tuple[Any, Any], float],
    csv_path: Optional[str] = None,
    verbose: bool = False,
    beam_width: int = 8,
    max_states_per_slice: int = 600,
    *args,
    **kwargs,
):
    """
        ABO-SAGE-FAST (aligned with FABO-SAGE-FAST decisions):
      - Uses Sage graph + Python-cached adjacency
      - Widest-path with latency tie-break for routing
      - CHAIN OPTIMIZATION: route only the last VL when placing the next VNF
      - beam_width: limit candidate nodes per expansion (major runtime control)
      - max_states_per_slice: hard cap expansions per slice (guarantees runtime)

    Important:
      - Assumes each slice is a VNF chain (v0->v1->...->vk).
      - If slice_data includes entry, we route ("ENTRY" -> v0) after placing v0.
    """

    # ---- Build Sage adapter (or reuse) ----
    if isinstance(G, SageGraphAdapter):
        SG = G
    elif Graph is not None and isinstance(G, Graph):
        SG = SageGraphAdapter(G)
    else:
        SG = SageGraphAdapter.from_networkx_like(G)

    # ---- Canonicalize latency/capacity dicts once ----
    lat_canon = _canonize_edge_dict(link_latency)
    cap_global = _canonize_edge_dict(link_capacity_base)

    # ---- Cache adjacency in Python once ----
    nodes = SG.nodes()
    adj2: Dict[Any, List[Tuple[Any, Tuple[Any, Any]]]] = {}
    for u in nodes:
        neigh = SG.neighbors(u)
        # Precompute (neighbor, canonical_edge_key) to avoid _edge_key calls in hot loops.
        adj2[u] = [(v, _edge_key(u, v)) for v in neigh]

    # ---- Helpers ----
    def dec_edge_capacity(dct: Dict[Tuple[Any, Any], float], ek: Tuple[Any, Any], bw: float) -> None:
        # Decrease remaining bandwidth on a canonical edge with underflow check.
        dct[ek] -= bw
        if dct[ek] < -1e-9:
            raise ValueError(f"Link capacity underflow on {ek}: remaining={dct[ek]}, dec={bw}")

    def best_bandwidth_path(u: Any, v: Any, link_capacity: Dict[Tuple[Any, Any], float], bw: float):
        """
                Widest path (maximize bottleneck), tie-break by minimum latency.
        Complexity ~ O(E log V) per call.
        """
        if u is None or v is None:
            return None, None
        if u == v:
            return [u], 0.0

        bwf = float(bw)

        _adj2 = adj2
        _cap = link_capacity
        _lat = lat_canon
        heappush = heapq.heappush
        heappop = heapq.heappop

        best_bn = {u: float("inf")}
        best_lt = {u: 0.0}
        parent = {u: None}

        heap = [(-float("inf"), 0.0, u)]

        while heap:
            neg_bn, cur_lat, x = heappop(heap)
            cur_bn = -neg_bn

            old_bn = best_bn.get(x, -1.0)
            if cur_bn < old_bn:
                continue
            if cur_bn == old_bn and cur_lat > best_lt.get(x, float("inf")):
                continue

            if x == v:
                break

            for y, ek in _adj2[x]:
                cap_xy = _cap.get(ek, 0.0)
                if cap_xy <= 0.0:
                    continue

                # new_bn = min(cur_bn, cap_xy) avoiding min() overhead.
                if cur_bn == float("inf"):
                    new_bn = cap_xy
                else:
                    new_bn = cap_xy if cap_xy < cur_bn else cur_bn

                if new_bn < bwf:
                    continue

                new_lat = cur_lat + _lat.get(ek, 1.0)

                ob = best_bn.get(y, -1.0)
                ol = best_lt.get(y, float("inf"))

                if (new_bn > ob) or (new_bn == ob and new_lat < ol):
                    best_bn[y] = new_bn
                    best_lt[y] = new_lat
                    parent[y] = x
                    heappush(heap, (-new_bn, new_lat, y))

        if v not in best_bn or best_bn[v] < bwf:
            return None, None

        # Reconstruct path.
        path = []
        cur = v
        while cur is not None:
            path.append(cur)
            cur = parent.get(cur)
        path.reverse()

        if not path or path[0] != u or path[-1] != v:
            return None, None

        return path, float(best_lt[v])

    # ---- Global remaining resources ----
    abo_results: List[Optional[ABOState]] = []
    node_capacity_global = node_capacity_base.copy()

    summary_rows = []

    # ------------------------- solve each slice -------------------------
    for i, slice_data in enumerate(slices, start=1):
        if len(slice_data) == 2:
            vnf_chain, vl_chain = slice_data
            entry = None
        elif len(slice_data) == 3:
            vnf_chain, vl_chain, entry = slice_data
        else:
            raise ValueError(f"Unexpected slice format: {slice_data}")

        # Precompute chain structures once per slice.
        vnf_ids = [_as_vnf_id(v["id"]) for v in vnf_chain]
        vnf_by_id = {vid: v for vid, v in zip(vnf_ids, vnf_chain)}
        vnf_cpu = {vid: float(vnf_by_id[vid].get("cpu", 0.0)) for vid in vnf_ids}
        vnf_slice = {vid: vnf_by_id[vid].get("slice") for vid in vnf_ids}

        # Chain VLs are exactly consecutive pairs.
        chain_vls: List[Tuple[str, str, float]] = []
        bw_by_chain: Dict[Tuple[str, str], float] = {}
        for idx in range(1, len(vnf_ids)):
            a, b = vnf_ids[idx - 1], vnf_ids[idx]
            bw = None
            for vl in vl_chain:
                if _vl_key(vl) == (a, b):
                    bw = float(vl.get("bandwidth", 0.0))
                    break
            if bw is None:
                bw = 0.0
            chain_vls.append((a, b, bw))
            bw_by_chain[(a, b)] = bw

        # Entry bandwidth choice aligned with original approach (reuse first VL bw).
        entry_bw = float(vl_chain[0].get("bandwidth", 0.0)) if (entry is not None and vl_chain) else 0.0

        if verbose:
            print(f"\n[INFO][ABO-SAGE-FAST] === Solving slice {i} ({len(vnf_ids)} VNFs, {len(chain_vls)} VLs) ===")

        init_state = ABOState(
            placed_vnfs={},
            routed_vls={},
            g_cost=0.0,
            node_capacity=node_capacity_global.copy(),
            link_capacity=cap_global.copy(),
        )

        heap: List[Tuple[float, int, ABOState]] = []
        tie = 0
        heapq.heappush(heap, (0.0, tie, init_state))

        visited = 0
        result: Optional[ABOState] = None

        # Dominance pruning by progress (placed_count, routed_count) -> best g_cost.
        best_seen: Dict[Tuple[int, int], float] = {}

        while heap and visited < int(max_states_per_slice):
            _, __, state = heapq.heappop(heap)
            visited += 1

            prog_key = (len(state.placed_vnfs), len(state.routed_vls))
            prev_best = best_seen.get(prog_key)
            if prev_best is not None and state.g_cost >= prev_best:
                continue
            best_seen[prog_key] = state.g_cost

            if state.is_goal_chain(vnf_ids, entry):
                result = state
                break

            # Choose next VNF in chain order.
            next_idx = len(state.placed_vnfs)
            if next_idx >= len(vnf_ids):
                continue

            next_vid = vnf_ids[next_idx]
            cpu_need = vnf_cpu[next_vid]
            slice_id = vnf_slice[next_vid]

            # Cheap node ordering for runtime: prefer more available CPU (descending).
            nc = state.node_capacity
            candidate_nodes = sorted(nodes, key=lambda n: nc.get(n, 0.0), reverse=True)
            if beam_width and beam_width > 0:
                candidate_nodes = candidate_nodes[: int(beam_width)]

            for node in candidate_nodes:
                avail = float(nc.get(node, 0.0))
                if avail < cpu_need:
                    continue

                # Anti-affinity within the slice (no 2 VNFs of same slice on same node).
                target_slice = slice_id
                same_slice_on_node = any(
                    placed_node == node and vnf_slice.get(pid) == target_slice
                    for pid, placed_node in state.placed_vnfs.items()
                )
                if same_slice_on_node:
                    continue

                # Shallow copies (fast).
                new_placed = state.placed_vnfs.copy()
                new_routed = state.routed_vls.copy()
                new_node_cap = state.node_capacity.copy()
                new_link_cap = state.link_capacity.copy()
                new_cost = float(state.g_cost)

                # Place VNF.
                new_node_cap[node] = avail - cpu_need
                new_placed[next_vid] = node

                routing_ok = True

                # -------------------- CHAIN OPT: route only the last VL --------------------
                # When placing VNF at index next_idx, only VL (prev -> current) becomes newly determined.
                if next_idx > 0:
                    prev_vid = vnf_ids[next_idx - 1]
                    a, b = prev_vid, next_vid
                    k = (a, b)

                    na = new_placed[a]
                    nb = new_placed[b]

                    if na != nb and k not in new_routed:
                        bw = float(bw_by_chain.get(k, 0.0))
                        path, lat = best_bandwidth_path(na, nb, new_link_cap, bw)
                        if not path or len(path) < 2:
                            routing_ok = False
                        else:
                            _assert_path_endpoints(path, na, nb)
                            for u, v in zip(path[:-1], path[1:]):
                                ek = _edge_key(u, v)
                                if new_link_cap.get(ek, 0.0) < bw:
                                    routing_ok = False
                                    break
                                dec_edge_capacity(new_link_cap, ek, bw)
                            if routing_ok:
                                new_routed[k] = path
                                new_cost += float(lat)

                if not routing_ok:
                    continue

                # -------------------- Entry -> v0 when v0 is placed --------------------
                if entry is not None and next_idx == 0 and entry_bw > 0.0:
                    v0 = next_vid
                    n0 = new_placed[v0]
                    if entry != n0:
                        path, lat = best_bandwidth_path(entry, n0, new_link_cap, entry_bw)
                        if not path or len(path) < 2:
                            continue
                        _assert_path_endpoints(path, entry, n0)
                        for u, v in zip(path[:-1], path[1:]):
                            ek = _edge_key(u, v)
                            if new_link_cap.get(ek, 0.0) < entry_bw:
                                routing_ok = False
                                break
                            dec_edge_capacity(new_link_cap, ek, entry_bw)
                        if not routing_ok:
                            continue
                        new_routed[("ENTRY", v0)] = path
                        new_cost += float(lat)

                child = ABOState(
                    placed_vnfs=new_placed,
                    routed_vls=new_routed,
                    g_cost=new_cost,
                    node_capacity=new_node_cap,
                    link_capacity=new_link_cap,
                )

                # Heuristic is intentionally kept super cheap for runtime: 0.0
                # This makes it closer to uniform-cost search on g_cost.
                f_score = child.g_cost

                tie += 1
                heapq.heappush(heap, (f_score, tie, child))

        abo_results.append(result)

        if verbose:
            if result is not None:
                print(f"[INFO][ABO-SAGE-FAST] Found solution after {visited} states.")
            else:
                print(
                    f"[WARN][ABO-SAGE-FAST] No solution within max_states_per_slice={max_states_per_slice} "
                    f"(visited={visited})."
                )

        # ---- Commit globals if accepted ----
        if result is not None:
            # Commit CPU on nodes.
            for vnf_id, node in result.placed_vnfs.items():
                node_capacity_global[node] -= float(vnf_cpu[vnf_id])
                if node_capacity_global[node] < -1e-9:
                    raise ValueError(
                        f"Node capacity underflow on {node} in slice {i}: remaining={node_capacity_global[node]}"
                    )

            # Commit BW along routed paths.
            # For chain VLs: use bw_by_chain
            # For entry route: use entry_bw
            for (src, dst), path in result.routed_vls.items():
                if src == "ENTRY":
                    bw = float(entry_bw)
                    src_node = entry
                    dst_node = result.placed_vnfs[dst]
                    _assert_path_endpoints(path, src_node, dst_node)
                else:
                    bw = float(bw_by_chain.get((src, dst), 0.0))
                    src_node = result.placed_vnfs[src]
                    dst_node = result.placed_vnfs[dst]
                    if src_node != dst_node:
                        _assert_path_endpoints(path, src_node, dst_node)

                if bw > 0.0 and len(path) >= 2:
                    for u, v in zip(path[:-1], path[1:]):
                        ek = _edge_key(u, v)
                        dec_edge_capacity(cap_global, ek, bw)

        summary_rows.append({"slice": i, "accepted": result is not None, "g_cost": (result.g_cost if result else None)})

    df_results = pd.DataFrame(summary_rows)
    if csv_path:
        df_results.to_csv(csv_path, index=False)
        if verbose:
            print(f"[INFO][ABO-SAGE-FAST] Results written to {csv_path}")

    return df_results, abo_results