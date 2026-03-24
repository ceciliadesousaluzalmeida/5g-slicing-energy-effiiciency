# All comments in English
import heapq
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    from sage.all import Graph  # type: ignore
except Exception:
    Graph = None  # type: ignore


ENTRY_VNF_ID = "ENTRY"

_GRAPH_ADAPTER_CACHE: Dict[int, Any] = {}
_GRAPH_LATENCY_CACHE: Dict[int, Dict[Tuple[Any, Any], float]] = {}


# ------------------------- small helpers -------------------------
def _as_vnf_id(v: Any) -> str:
    return str(v).strip()


def _vl_key_from_to(v_from: Any, v_to: Any) -> Tuple[str, str]:
    return (_as_vnf_id(v_from), _as_vnf_id(v_to))


def _vl_key(vl: Dict[str, Any]) -> Tuple[str, str]:
    return _vl_key_from_to(vl["from"], vl["to"])


def _assert_path_endpoints(path: List[Any], src_node: Any, dst_node: Any) -> None:
    if not path or len(path) < 2:
        raise ValueError("VL path is missing or trivial.")
    if path[0] != src_node or path[-1] != dst_node:
        raise ValueError(
            f"VL path endpoints mismatch: expected {src_node}->{dst_node}, got {path[0]}->{path[-1]}"
        )


def _edge_key(u: Any, v: Any) -> Tuple[Any, Any]:
    return (u, v) if u <= v else (v, u)


def _canonize_edge_dict(dct: Dict[Tuple[Any, Any], float]) -> Dict[Tuple[Any, Any], float]:
    out: Dict[Tuple[Any, Any], float] = {}
    for edge, val in dct.items():
        if not isinstance(edge, tuple) or len(edge) < 2:
            raise ValueError(f"Unsupported edge key format: {edge}")
        u, v = edge[0], edge[1]
        out[_edge_key(u, v)] = float(val)
    return out


def _normalize_slice_item(item: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Optional[Any]]:
    if not isinstance(item, (tuple, list)):
        raise ValueError(f"Unsupported slice format: {item}")

    if len(item) == 2:
        vnf_chain, vl_chain = item
        entry = None
    elif len(item) == 3:
        vnf_chain, vl_chain, entry = item
    elif len(item) >= 4:
        vnf_chain = item[-3]
        vl_chain = item[-2]
        entry = item[-1]
    else:
        raise ValueError(f"Unsupported slice format: {item}")

    return list(vnf_chain), list(vl_chain), entry


def _infer_link_latency_from_graph(G: Any) -> Dict[Tuple[Any, Any], float]:
    gid = id(G)
    cached = _GRAPH_LATENCY_CACHE.get(gid)
    if cached is not None:
        return cached

    out: Dict[Tuple[Any, Any], float] = {}
    for u, v, data in G.edges(data=True):
        lat = float(data.get("latency", data.get("distance", 1.0)))
        out[(u, v)] = lat
        out[(v, u)] = lat

    _GRAPH_LATENCY_CACHE[gid] = out
    return out


# ------------------------- Sage adapter -------------------------
class SageGraphAdapter:
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


def _get_cached_graph_adapter(G: Any) -> "SageGraphAdapter":
    gid = id(G)
    cached = _GRAPH_ADAPTER_CACHE.get(gid)
    if cached is not None:
        return cached

    if isinstance(G, SageGraphAdapter):
        adapter = G
    elif Graph is not None and isinstance(G, Graph):
        adapter = SageGraphAdapter(G)
    else:
        adapter = SageGraphAdapter.from_networkx_like(G)

    _GRAPH_ADAPTER_CACHE[gid] = adapter
    return adapter


# ------------------------- ABO state -------------------------
class ABOState:
    __slots__ = (
        "placed_vnfs",
        "routed_vls",
        "g_cost",
        "h_cost",
        "f_cost",
        "bw_score",
        "node_capacity",
        "link_capacity",
    )

    def __init__(
        self,
        placed_vnfs: Optional[Dict[str, Any]] = None,
        routed_vls: Optional[Dict[Tuple[str, str], List[Any]]] = None,
        g_cost: float = 0.0,
        h_cost: float = 0.0,
        bw_score: float = 0.0,
        node_capacity: Optional[Dict[Any, float]] = None,
        link_capacity: Optional[Dict[Tuple[Any, Any], float]] = None,
    ):
        self.placed_vnfs = placed_vnfs or {}
        self.routed_vls = routed_vls or {}
        self.g_cost = float(g_cost)
        self.h_cost = float(h_cost)
        self.f_cost = float(g_cost + h_cost)
        self.bw_score = float(bw_score)
        self.node_capacity = node_capacity or {}
        self.link_capacity = link_capacity or {}

    def is_goal_chain(self, vnf_ids: List[str], entry: Optional[Any]) -> bool:
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
                p = self.routed_vls.get((ENTRY_VNF_ID, v0))
                if not p or p[0] != entry or p[-1] != n0:
                    return False

        return True

    def __lt__(self, other: "ABOState") -> bool:
        if self.f_cost != other.f_cost:
            return self.f_cost < other.f_cost
        if self.bw_score != other.bw_score:
            return self.bw_score > other.bw_score
        return self.g_cost < other.g_cost


# ------------------------- main ABO -------------------------
def run_abo_full_batch(
    G: Any,
    slices: List[Any],
    node_capacity_base: Dict[Any, float],
    arg4: Dict[Tuple[Any, Any], float],
    arg5: Optional[Dict[Tuple[Any, Any], float]] = None,
    csv_path: Optional[str] = None,
    verbose: bool = False,
    beam_width: int = 6,
    max_states_per_slice: int = 3000,
    *args,
    **kwargs,
):
    """
    A* version of ABO with residual bandwidth priority.

    Priority order:
    1) smaller f(n) = g(n) + h(n)
    2) larger residual-bandwidth score
    3) smaller g(n)

    Supported call patterns:
    1) run_abo_full_batch(G, slices, node_capacity_base, link_capacity_base)
    2) run_abo_full_batch(G, slices, node_capacity_base, link_latency, link_capacity_base, ...)
    """

    if arg5 is None:
        link_capacity_base = arg4
        link_latency = _infer_link_latency_from_graph(G)
    else:
        link_latency = arg4
        link_capacity_base = arg5

    beam_width = int(kwargs.get("beam_width", beam_width))
    max_states_per_slice = int(kwargs.get("max_states_per_slice", max_states_per_slice))
    verbose = bool(kwargs.get("verbose", verbose))
    csv_path = kwargs.get("csv_path", csv_path)

    SG = _get_cached_graph_adapter(G)

    lat_canon = _canonize_edge_dict(link_latency)
    cap_global = _canonize_edge_dict(link_capacity_base)

    nodes = SG.nodes()
    adj2: Dict[Any, List[Tuple[Any, Tuple[Any, Any]]]] = {}
    for u in nodes:
        neigh = SG.neighbors(u)
        adj2[u] = [(v, _edge_key(u, v)) for v in neigh]

    positive_lats = [v for v in lat_canon.values() if v > 0.0]
    min_positive_latency = min(positive_lats) if positive_lats else 0.0

    def dec_edge_capacity(dct: Dict[Tuple[Any, Any], float], ek: Tuple[Any, Any], bw: float) -> None:
        dct[ek] -= bw
        if dct[ek] < -1e-9:
            raise ValueError(f"Link capacity underflow on {ek}: remaining={dct[ek]}, dec={bw}")

    def best_bandwidth_path(
        src: Any,
        dst: Any,
        link_capacity: Dict[Tuple[Any, Any], float],
        bw: float,
    ) -> Tuple[Optional[List[Any]], Optional[float], Optional[float]]:
        if src is None or dst is None:
            return None, None, None

        if src == dst:
            return [src], 0.0, float("inf")

        bwf = float(bw)

        _adj2 = adj2
        _cap_get = link_capacity.get
        _lat_get = lat_canon.get
        heappush = heapq.heappush
        heappop = heapq.heappop

        best_bn = {src: float("inf")}
        best_lt = {src: 0.0}
        parent = {src: None}

        heap: List[Tuple[float, float, Any]] = [(-float("inf"), 0.0, src)]

        while heap:
            neg_bn, cur_lat, x = heappop(heap)
            cur_bn = -neg_bn

            old_bn = best_bn.get(x, -1.0)
            if cur_bn < old_bn:
                continue
            if cur_bn == old_bn and cur_lat > best_lt.get(x, float("inf")):
                continue

            if x == dst:
                break

            for y, ek in _adj2[x]:
                cap_xy = _cap_get(ek, 0.0)
                if cap_xy <= 0.0:
                    continue

                new_bn = cap_xy if cur_bn == float("inf") else (cap_xy if cap_xy < cur_bn else cur_bn)
                if new_bn < bwf:
                    continue

                new_lat = cur_lat + _lat_get(ek, 1.0)

                ob = best_bn.get(y, -1.0)
                ol = best_lt.get(y, float("inf"))

                if (new_bn > ob) or (new_bn == ob and new_lat < ol):
                    best_bn[y] = new_bn
                    best_lt[y] = new_lat
                    parent[y] = x
                    heappush(heap, (-new_bn, new_lat, y))

        if dst not in best_bn or best_bn[dst] < bwf:
            return None, None, None

        path = []
        cur = dst
        while cur is not None:
            path.append(cur)
            cur = parent.get(cur)
        path.reverse()

        if not path or path[0] != src or path[-1] != dst:
            return None, None, None

        return path, float(best_lt[dst]), float(best_bn[dst])

    def node_bandwidth_potential(node: Any, link_capacity: Dict[Tuple[Any, Any], float]) -> float:
        total = 0.0
        for _, ek in adj2[node]:
            total += float(link_capacity.get(ek, 0.0))
        return total

    def remaining_chain_routes_count(
        vnf_ids: List[str],
        routed_vls: Dict[Tuple[str, str], List[Any]],
    ) -> int:
        routed_chain = 0
        for a, b in routed_vls.keys():
            if a != ENTRY_VNF_ID:
                routed_chain += 1
        total_chain = max(0, len(vnf_ids) - 1)
        return max(0, total_chain - routed_chain)

    def heuristic_remaining_latency(
        vnf_ids: List[str],
        routed_vls: Dict[Tuple[str, str], List[Any]],
    ) -> float:
        remaining_routes = remaining_chain_routes_count(vnf_ids, routed_vls)
        return float(remaining_routes * min_positive_latency)

    abo_results: List[Optional[ABOState]] = []
    node_capacity_global = {n: float(cap) for n, cap in node_capacity_base.items()}
    summary_rows = []

    for i, slice_data in enumerate(slices, start=1):
        vnf_chain, vl_chain, entry = _normalize_slice_item(slice_data)

        vnf_ids = [_as_vnf_id(v["id"]) for v in vnf_chain]
        vnf_by_id = {vid: v for vid, v in zip(vnf_ids, vnf_chain)}
        vnf_cpu = {vid: float(vnf_by_id[vid].get("cpu", 0.0)) for vid in vnf_ids}
        vnf_slice = {vid: vnf_by_id[vid].get("slice") for vid in vnf_ids}

        bw_by_chain: Dict[Tuple[str, str], float] = {}
        for idx in range(1, len(vnf_ids)):
            a, b = vnf_ids[idx - 1], vnf_ids[idx]
            bw = 0.0
            for vl in vl_chain:
                if _vl_key(vl) == (a, b):
                    bw = float(vl.get("bandwidth", 0.0))
                    break
            bw_by_chain[(a, b)] = bw

        entry_bw = float(vl_chain[0].get("bandwidth", 0.0)) if (entry is not None and vl_chain) else 0.0

        if verbose:
            print(f"\n[INFO][ABO-A*] === Solving slice {i} ({len(vnf_ids)} VNFs) ===")

        init_h = heuristic_remaining_latency(vnf_ids, {})
        init_state = ABOState(
            placed_vnfs={},
            routed_vls={},
            g_cost=0.0,
            h_cost=init_h,
            bw_score=0.0,
            node_capacity=node_capacity_global.copy(),
            link_capacity=cap_global.copy(),
        )

        heap: List[Tuple[float, float, float, int, ABOState]] = []
        tie = 0
        heapq.heappush(heap, (init_state.f_cost, -init_state.bw_score, init_state.g_cost, tie, init_state))

        visited = 0
        result: Optional[ABOState] = None

        best_seen: Dict[Tuple[int, int], Tuple[float, float]] = {}

        while heap and visited < max_states_per_slice:
            _, __, ___, ____, state = heapq.heappop(heap)
            visited += 1

            prog_key = (len(state.placed_vnfs), len(state.routed_vls))
            prev_best = best_seen.get(prog_key)
            cur_key = (state.f_cost, -state.bw_score)

            if prev_best is not None:
                prev_f, prev_neg_bw = prev_best
                if cur_key >= (prev_f, prev_neg_bw):
                    continue
            best_seen[prog_key] = cur_key

            if state.is_goal_chain(vnf_ids, entry):
                result = state
                break

            next_idx = len(state.placed_vnfs)
            if next_idx >= len(vnf_ids):
                continue

            next_vid = vnf_ids[next_idx]
            cpu_need = vnf_cpu[next_vid]
            target_slice = vnf_slice[next_vid]

            nc = state.node_capacity

            used_nodes_same_slice = set()
            for pid, placed_node in state.placed_vnfs.items():
                if vnf_slice.get(pid) == target_slice:
                    used_nodes_same_slice.add(placed_node)

            feasible_nodes = []
            for node in nodes:
                avail = float(nc.get(node, 0.0))
                if avail < cpu_need:
                    continue
                if node in used_nodes_same_slice:
                    continue

                bw_potential = node_bandwidth_potential(node, state.link_capacity)
                feasible_nodes.append((node, bw_potential, avail))

            feasible_nodes.sort(key=lambda x: (x[1], x[2]), reverse=True)

            if beam_width and beam_width > 0:
                feasible_nodes = feasible_nodes[:beam_width]

            for node, node_bw_potential, avail in feasible_nodes:
                new_cost = float(state.g_cost)

                new_placed = state.placed_vnfs.copy()
                new_node_cap = state.node_capacity.copy()

                new_node_cap[node] = avail - cpu_need
                new_placed[next_vid] = node

                routing_ok = True
                new_routed = state.routed_vls
                new_link_cap = state.link_capacity
                local_bw_score = float(state.bw_score)

                if next_idx > 0:
                    prev_vid = vnf_ids[next_idx - 1]
                    a, b = prev_vid, next_vid
                    na = new_placed[a]
                    nb = new_placed[b]

                    if na != nb and (a, b) not in state.routed_vls:
                        bw = float(bw_by_chain.get((a, b), 0.0))
                        path, lat, bottleneck_before = best_bandwidth_path(na, nb, state.link_capacity, bw)

                        if not path or len(path) < 2:
                            routing_ok = False
                        else:
                            _assert_path_endpoints(path, na, nb)

                            new_routed = state.routed_vls.copy()
                            new_link_cap = state.link_capacity.copy()

                            path_residual_after = float("inf")

                            for u, v in zip(path[:-1], path[1:]):
                                ek = _edge_key(u, v)
                                if new_link_cap.get(ek, 0.0) < bw:
                                    routing_ok = False
                                    break
                                dec_edge_capacity(new_link_cap, ek, bw)
                                if new_link_cap[ek] < path_residual_after:
                                    path_residual_after = new_link_cap[ek]

                            if routing_ok:
                                new_routed[(a, b)] = path
                                new_cost += float(lat)
                                if bottleneck_before is not None:
                                    local_bw_score += float(bottleneck_before)
                                if path_residual_after != float("inf"):
                                    local_bw_score += float(path_residual_after)

                if not routing_ok:
                    continue

                if entry is not None and next_idx == 0:
                    v0 = next_vid
                    n0 = new_placed[v0]

                    if entry != n0:
                        path, lat, bottleneck_before = best_bandwidth_path(entry, n0, new_link_cap, entry_bw)

                        if not path or len(path) < 2:
                            continue

                        _assert_path_endpoints(path, entry, n0)

                        if entry_bw > 0.0:
                            if new_link_cap is state.link_capacity:
                                new_link_cap = state.link_capacity.copy()
                            if new_routed is state.routed_vls:
                                new_routed = state.routed_vls.copy()

                            path_residual_after = float("inf")

                            for u, v in zip(path[:-1], path[1:]):
                                ek = _edge_key(u, v)
                                if new_link_cap.get(ek, 0.0) < entry_bw:
                                    routing_ok = False
                                    break
                                dec_edge_capacity(new_link_cap, ek, entry_bw)
                                if new_link_cap[ek] < path_residual_after:
                                    path_residual_after = new_link_cap[ek]

                            if not routing_ok:
                                continue

                            if bottleneck_before is not None:
                                local_bw_score += float(bottleneck_before)
                            if path_residual_after != float("inf"):
                                local_bw_score += float(path_residual_after)

                        if new_routed is state.routed_vls:
                            new_routed = state.routed_vls.copy()

                        new_routed[(ENTRY_VNF_ID, v0)] = path
                        new_cost += float(lat)

                local_bw_score += 0.01 * float(node_bw_potential)

                new_h = heuristic_remaining_latency(vnf_ids, new_routed)

                child = ABOState(
                    placed_vnfs=new_placed,
                    routed_vls=new_routed,
                    g_cost=new_cost,
                    h_cost=new_h,
                    bw_score=local_bw_score,
                    node_capacity=new_node_cap,
                    link_capacity=new_link_cap,
                )

                tie += 1
                heapq.heappush(
                    heap,
                    (child.f_cost, -child.bw_score, child.g_cost, tie, child),
                )

        abo_results.append(result)

        if verbose:
            if result is not None:
                print(f"[INFO][ABO-A*] Found solution after {visited} states.")
            else:
                print(
                    f"[WARN][ABO-A*] No solution within max_states_per_slice={max_states_per_slice} "
                    f"(visited={visited})."
                )

        if result is not None:
            for vnf_id, node in result.placed_vnfs.items():
                node_capacity_global[node] -= float(vnf_cpu[vnf_id])
                if node_capacity_global[node] < -1e-9:
                    raise ValueError(
                        f"Node capacity underflow on {node} in slice {i}: remaining={node_capacity_global[node]}"
                    )

            for (src, dst), path in result.routed_vls.items():
                if src == ENTRY_VNF_ID:
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

        summary_rows.append(
            {
                "slice": i,
                "accepted": result is not None,
                "g_cost": (result.g_cost if result else None),
                "h_cost": (result.h_cost if result else None),
                "f_cost": (result.f_cost if result else None),
                "bw_score": (result.bw_score if result else None),
            }
        )

    df_results = None
    if csv_path or verbose:
        df_results = pd.DataFrame(summary_rows)

    if csv_path and df_results is not None:
        df_results.to_csv(csv_path, index=False)
        if verbose:
            print(f"[INFO][ABO-A*] Results written to {csv_path}")

    if df_results is None:
        df_results = pd.DataFrame(summary_rows)

    return df_results, abo_results