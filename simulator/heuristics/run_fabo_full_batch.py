# run_fabo_full_batch.py
# All comments are in English.

import heapq
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    from sage.all import Graph  # type: ignore
except Exception:
    Graph = None  # type: ignore


ENTRY_VNF_ID = "__ENTRY__"


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
    for (u, v), val in dct.items():
        out[_edge_key(u, v)] = float(val)
    return out


def _infer_entry_bandwidth(vnf_chain: List[Dict[str, Any]], bw_by_chain: Dict[Tuple[str, str], float]) -> float:
    """
    Infer ingress bandwidth demand from the slice description.
    Priority:
      1) first VNF throughput
      2) first chain VL bandwidth
      3) 0.0
    """
    if not vnf_chain:
        return 0.0

    first_vnf = vnf_chain[0]

    thr = first_vnf.get("throughput")
    if thr is not None:
        return float(thr)

    if len(vnf_chain) >= 2:
        a = _as_vnf_id(vnf_chain[0]["id"])
        b = _as_vnf_id(vnf_chain[1]["id"])
        return float(bw_by_chain.get((a, b), 0.0))

    return 0.0


# ------------------------- Sage adapter -------------------------
class SageGraphAdapter:
    """
    Minimal adapter around a Sage graph.
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


# ------------------------- state -------------------------
class FABOState:
    __slots__ = (
        "placed_vnfs",
        "routed_vls",
        "g_cost",
        "h_cost",
        "f_cost",
        "node_capacity",
        "link_capacity",
        "node_use_count",
        "last_bottleneck",
        "sum_node_capacity",
        "sum_link_capacity",
    )

    def __init__(
        self,
        placed_vnfs: Optional[Dict[str, Any]] = None,
        routed_vls: Optional[Dict[Tuple[str, str], List[Any]]] = None,
        g_cost: float = 0.0,
        h_cost: float = 0.0,
        node_capacity: Optional[Dict[Any, float]] = None,
        link_capacity: Optional[Dict[Tuple[Any, Any], float]] = None,
        node_use_count: Optional[Dict[Any, int]] = None,
        last_bottleneck: float = float("inf"),
        sum_node_capacity: float = 0.0,
        sum_link_capacity: float = 0.0,
    ):
        self.placed_vnfs = placed_vnfs or {}
        self.routed_vls = routed_vls or {}
        self.g_cost = float(g_cost)
        self.h_cost = float(h_cost)
        self.f_cost = float(g_cost + h_cost)
        self.node_capacity = node_capacity or {}
        self.link_capacity = link_capacity or {}
        self.node_use_count = node_use_count or {}
        self.last_bottleneck = float(last_bottleneck)
        self.sum_node_capacity = float(sum_node_capacity)
        self.sum_link_capacity = float(sum_link_capacity)

    def is_goal_chain(self, vnf_ids: List[str]) -> bool:
        if len(self.placed_vnfs) != len(vnf_ids):
            return False

        get_placed = self.placed_vnfs.get
        get_routed = self.routed_vls.get

        for i in range(1, len(vnf_ids)):
            a = vnf_ids[i - 1]
            b = vnf_ids[i]
            na = get_placed(a)
            nb = get_placed(b)
            if na is None or nb is None:
                return False
            if na == nb:
                continue
            p = get_routed((a, b))
            if not p or p[0] != na or p[-1] != nb:
                return False
        return True

    def __lt__(self, other: "FABOState") -> bool:
        return (self.f_cost, self.h_cost, self.g_cost) < (other.f_cost, other.h_cost, other.g_cost)


# ------------------------- main algorithm -------------------------
def run_fabo_full_batch(
    G: Any,
    slices: List[Any],
    node_capacity_base: Dict[Any, float],
    link_latency: Dict[Tuple[Any, Any], float],
    link_capacity_base: Dict[Tuple[Any, Any], float],
    csv_path: Optional[str] = None,
    verbose: bool = False,
    beam_width: Optional[int] = 6,
    max_states_per_slice: int = 3000,
    alpha_latency: float = 1.0,
    beta_node_fairness: float = 2.0,
    gamma_node_reuse: float = 0.5,
    delta_bandwidth: float = 1.0,
    epsilon_bw: float = 1e-9,
    *args,
    **kwargs,
):
    """
    Optimized A*-based FABO with:
      - node fairness only
      - widest-path bandwidth-aware routing
      - explicit g(n), h(n), and f(n)=g(n)+h(n)
      - reduced routing calls via top-k candidate filtering
      - lighter heuristic
      - stronger dominance pruning
      - explicit entry-point support for the first VNF
    """

    # ------------------------- graph preparation -------------------------
    if isinstance(G, SageGraphAdapter):
        SG = G
    elif Graph is not None and isinstance(G, Graph):
        SG = SageGraphAdapter(G)
    else:
        SG = SageGraphAdapter.from_networkx_like(G)

    lat_canon = _canonize_edge_dict(link_latency)
    cap_global = _canonize_edge_dict(link_capacity_base)

    nodes = SG.nodes()
    node_base_cap = {n: float(node_capacity_base[n]) for n in nodes}

    adj2: Dict[Any, List[Tuple[Any, Tuple[Any, Any]]]] = {}
    for u in nodes:
        adj2[u] = [(v, _edge_key(u, v)) for v in SG.neighbors(u)]

    if not lat_canon:
        raise ValueError("link_latency is empty.")

    if not cap_global:
        raise ValueError("link_capacity_base is empty.")

    min_edge_latency = min(lat_canon.values())
    max_base_link_capacity = max(cap_global.values())

    # ------------------------- helpers -------------------------
    def dec_edge_capacity(dct: Dict[Tuple[Any, Any], float], ek: Tuple[Any, Any], bw: float) -> None:
        dct[ek] -= bw
        if dct[ek] < -1e-9:
            raise ValueError(f"Link capacity underflow on {ek}: remaining={dct[ek]}, dec={bw}")

    def node_utilization_from_remaining(node: Any, remaining_cpu: float) -> float:
        base = node_base_cap[node]
        if base <= 0.0:
            return 1.0
        used_ratio = 1.0 - (remaining_cpu / base)
        if used_ratio < 0.0:
            return 0.0
        if used_ratio > 1.0:
            return 1.0
        return used_ratio

    def node_fairness_penalty(node: Any, remaining_after: float, use_count_after: int) -> float:
        util_after = node_utilization_from_remaining(node, remaining_after)
        return beta_node_fairness * (util_after * util_after) + gamma_node_reuse * use_count_after

    def bottleneck_penalty(bottleneck_after_reservation: float) -> float:
        safe_bn = bottleneck_after_reservation if bottleneck_after_reservation > epsilon_bw else epsilon_bw
        ratio = safe_bn / max_base_link_capacity
        if ratio > 1.0:
            ratio = 1.0
        return delta_bandwidth * (1.0 - ratio)

    def best_bandwidth_path(
        src: Any,
        dst: Any,
        link_capacity: Dict[Tuple[Any, Any], float],
        bw_demand: float,
    ) -> Tuple[Optional[List[Any]], Optional[float], Optional[float]]:
        """
        Widest path with latency tie-break.
        Returns:
          - path
          - latency
          - bottleneck before reservation
        """
        if src is None or dst is None:
            return None, None, None

        if src == dst:
            return [src], 0.0, float("inf")

        bwf = bw_demand
        _adj2 = adj2
        _cap = link_capacity
        _lat = lat_canon
        heappush = heapq.heappush
        heappop = heapq.heappop

        best_bn: Dict[Any, float] = {src: float("inf")}
        best_lt: Dict[Any, float] = {src: 0.0}
        parent: Dict[Any, Any] = {src: None}

        best_bn_get = best_bn.get
        best_lt_get = best_lt.get
        parent_get = parent.get

        heap: List[Tuple[float, float, Any]] = [(-float("inf"), 0.0, src)]

        while heap:
            neg_bn, cur_lat, x = heappop(heap)
            cur_bn = -neg_bn

            old_bn = best_bn_get(x, -1.0)
            if cur_bn < old_bn:
                continue
            if cur_bn == old_bn and cur_lat > best_lt_get(x, float("inf")):
                continue

            if x == dst:
                break

            for y, ek in _adj2[x]:
                cap_xy = _cap.get(ek, 0.0)
                if cap_xy <= 0.0:
                    continue

                new_bn = cap_xy if cur_bn == float("inf") or cap_xy < cur_bn else cur_bn
                if new_bn < bwf:
                    continue

                new_lat = cur_lat + _lat.get(ek, 1.0)

                old_y_bn = best_bn_get(y, -1.0)
                if new_bn > old_y_bn:
                    best_bn[y] = new_bn
                    best_lt[y] = new_lat
                    parent[y] = x
                    heappush(heap, (-new_bn, new_lat, y))
                elif new_bn == old_y_bn:
                    old_y_lt = best_lt_get(y, float("inf"))
                    if new_lat < old_y_lt:
                        best_lt[y] = new_lat
                        parent[y] = x
                        heappush(heap, (-new_bn, new_lat, y))

        dst_bn = best_bn_get(dst)
        if dst_bn is None or dst_bn < bwf:
            return None, None, None

        path: List[Any] = []
        cur = dst
        while cur is not None:
            path.append(cur)
            cur = parent_get(cur)
        path.reverse()

        if not path or path[0] != src or path[-1] != dst:
            return None, None, None

        return path, best_lt[dst], dst_bn

    # ------------------------- heuristic -------------------------
    def optimistic_min_node_penalty(
        cpu_need: float,
        node_capacity: Dict[Any, float],
        node_use_count: Dict[Any, int],
    ) -> float:
        best = float("inf")
        get_node_cap = node_capacity.get
        get_node_use = node_use_count.get

        for node in nodes:
            rem = get_node_cap(node, 0.0)
            if rem < cpu_need:
                continue
            use_after = get_node_use(node, 0) + 1
            pen = node_fairness_penalty(node, rem - cpu_need, use_after)
            if pen < best:
                best = pen

        return 0.0 if best == float("inf") else best

    def heuristic_cost(
        state: FABOState,
        vnf_ids: List[str],
        vnf_cpu: Dict[str, float],
    ) -> float:
        """
        Cheap explicit heuristic:
          - optimistic future node placement penalty
          - relaxed routing latency
          - optimistic bandwidth term based on best-case bottleneck
        """
        placed_count = len(state.placed_vnfs)
        remaining_vnfs = len(vnf_ids) - placed_count
        if remaining_vnfs <= 0:
            return 0.0

        h = 0.0

        for idx in range(placed_count, len(vnf_ids)):
            vid = vnf_ids[idx]
            cpu_need = vnf_cpu[vid]
            h += optimistic_min_node_penalty(cpu_need, state.node_capacity, state.node_use_count)

        remaining_edges = len(vnf_ids) - 1 - len(state.routed_vls)
        if remaining_edges > 0:
            h += alpha_latency * min_edge_latency * remaining_edges
            h += bottleneck_penalty(max_base_link_capacity) * remaining_edges

        return h

    # ------------------------- global solve -------------------------
    results: List[Optional[FABOState]] = []
    node_capacity_global = {n: float(node_capacity_base[n]) for n in nodes}
    summary_rows = []

    for i, slice_data in enumerate(slices, start=1):
        if len(slice_data) == 2:
            vnf_chain, vl_chain = slice_data
            entry = None
        elif len(slice_data) == 3:
            vnf_chain, vl_chain, entry = slice_data
        else:
            raise ValueError(f"Unexpected slice format: {slice_data}")

        vnf_ids = [_as_vnf_id(v["id"]) for v in vnf_chain]
        vnf_by_id = {vid: v for vid, v in zip(vnf_ids, vnf_chain)}
        vnf_cpu = {vid: float(vnf_by_id[vid].get("cpu", 0.0)) for vid in vnf_ids}
        vnf_slice = {vid: vnf_by_id[vid].get("slice") for vid in vnf_ids}

        bw_by_chain: Dict[Tuple[str, str], float] = {}
        for idx in range(1, len(vnf_ids)):
            a = vnf_ids[idx - 1]
            b = vnf_ids[idx]
            bw = None
            for vl in vl_chain:
                if _vl_key(vl) == (a, b):
                    bw = float(vl.get("bandwidth", 0.0))
                    break
            if bw is None:
                bw = 0.0
            bw_by_chain[(a, b)] = bw

        entry_key: Optional[Tuple[str, str]] = None
        entry_bw = 0.0
        if entry is not None and len(vnf_ids) > 0:
            first_vid = vnf_ids[0]
            entry_key = (ENTRY_VNF_ID, first_vid)
            entry_bw = _infer_entry_bandwidth(vnf_chain, bw_by_chain)
            bw_by_chain[entry_key] = entry_bw

        if verbose:
            print(f"\n[INFO][FABO] Solving slice {i}")

        init_node_use_count = {n: 0 for n in nodes}

        init_state = FABOState(
            placed_vnfs={},
            routed_vls={},
            g_cost=0.0,
            h_cost=0.0,
            node_capacity=node_capacity_global.copy(),
            link_capacity=cap_global.copy(),
            node_use_count=init_node_use_count,
            last_bottleneck=float("inf"),
            sum_node_capacity=float(sum(node_capacity_global.values())),
            sum_link_capacity=float(sum(cap_global.values())),
        )
        init_state.h_cost = heuristic_cost(init_state, vnf_ids, vnf_cpu)
        init_state.f_cost = init_state.g_cost + init_state.h_cost

        heap: List[Tuple[float, float, float, int, FABOState]] = []
        tie = 0
        heapq.heappush(
            heap,
            (
                init_state.f_cost,
                init_state.h_cost,
                -init_state.last_bottleneck,
                tie,
                init_state,
            ),
        )

        visited = 0
        result: Optional[FABOState] = None

        best_seen: Dict[Tuple[Any, ...], Tuple[float, float, float]] = {}

        while heap and visited < int(max_states_per_slice):
            _, _, _, _, state = heapq.heappop(heap)
            visited += 1

            prefix_len = len(state.placed_vnfs)
            prefix_signature = tuple(state.placed_vnfs.get(vid, None) for vid in vnf_ids[:prefix_len])

            prev = best_seen.get(prefix_signature)
            state_signature = (state.g_cost, state.sum_node_capacity, state.sum_link_capacity)

            if prev is not None:
                prev_g, prev_node_sum, prev_link_sum = prev
                if (
                    state.g_cost >= prev_g
                    and state.sum_node_capacity <= prev_node_sum
                    and state.sum_link_capacity <= prev_link_sum
                ):
                    continue

                if (
                    state.g_cost < prev_g
                    or state.sum_node_capacity > prev_node_sum
                    or state.sum_link_capacity > prev_link_sum
                ):
                    best_seen[prefix_signature] = state_signature
            else:
                best_seen[prefix_signature] = state_signature

            if state.is_goal_chain(vnf_ids):
                result = state
                break

            next_idx = len(state.placed_vnfs)
            if next_idx >= len(vnf_ids):
                continue

            next_vid = vnf_ids[next_idx]
            cpu_need = vnf_cpu[next_vid]
            slice_id = vnf_slice[next_vid]

            state_node_cap = state.node_capacity
            state_node_use = state.node_use_count
            get_state_node_cap = state_node_cap.get
            get_state_node_use = state_node_use.get

            feasible_candidates: List[Tuple[float, Any, float, int]] = []

            for node in nodes:
                avail = get_state_node_cap(node, 0.0)
                if avail < cpu_need:
                    continue

                same_slice_on_node = False
                for pid, placed_node in state.placed_vnfs.items():
                    if placed_node == node and vnf_slice.get(pid) == slice_id:
                        same_slice_on_node = True
                        break
                if same_slice_on_node:
                    continue

                rem_after = avail - cpu_need
                use_after = get_state_node_use(node, 0) + 1
                local_pen = node_fairness_penalty(node, rem_after, use_after)
                feasible_candidates.append((local_pen, node, rem_after, use_after))

            feasible_candidates.sort(key=lambda x: x[0])

            if beam_width is not None and beam_width > 0 and len(feasible_candidates) > beam_width:
                feasible_candidates = feasible_candidates[: int(beam_width)]

            prev_vid = vnf_ids[next_idx - 1] if next_idx > 0 else None
            prev_node = state.placed_vnfs.get(prev_vid) if prev_vid is not None else None

            for placement_penalty, node, rem_after, use_after in feasible_candidates:
                new_placed = state.placed_vnfs.copy()
                new_routed = state.routed_vls.copy()
                new_node_cap = state.node_capacity.copy()
                new_link_cap = state.link_capacity.copy()
                new_node_use_count = state.node_use_count.copy()

                incremental_cost = placement_penalty
                last_bottleneck = float("inf")
                new_sum_node_capacity = state.sum_node_capacity - cpu_need
                new_sum_link_capacity = state.sum_link_capacity

                # Place current VNF
                new_node_cap[node] = rem_after
                new_placed[next_vid] = node
                new_node_use_count[node] = use_after

                routing_ok = True

                # -------------------- entry -> first VNF --------------------
                if next_idx == 0 and entry is not None and entry_key is not None:
                    if entry != node:
                        path, lat, bottleneck_before = best_bandwidth_path(
                            entry,
                            node,
                            new_link_cap,
                            entry_bw,
                        )

                        if not path or lat is None or bottleneck_before is None or len(path) < 2:
                            routing_ok = False
                        else:
                            bottleneck_after = bottleneck_before - entry_bw
                            if bottleneck_after < -1e-9:
                                routing_ok = False
                            else:
                                incremental_cost += alpha_latency * lat
                                incremental_cost += bottleneck_penalty(
                                    bottleneck_after if bottleneck_after > 0.0 else 0.0
                                )
                                last_bottleneck = bottleneck_after if bottleneck_after > 0.0 else 0.0

                                for u, v in zip(path[:-1], path[1:]):
                                    ek = _edge_key(u, v)
                                    if new_link_cap.get(ek, 0.0) < entry_bw:
                                        routing_ok = False
                                        break
                                    dec_edge_capacity(new_link_cap, ek, entry_bw)
                                    new_sum_link_capacity -= entry_bw

                                if routing_ok:
                                    new_routed[entry_key] = path

                # -------------------- previous VNF -> current VNF --------------------
                if routing_ok and next_idx > 0 and prev_vid is not None:
                    k = (prev_vid, next_vid)
                    cur_node = node

                    if prev_node != cur_node and k not in new_routed:
                        bw_demand = bw_by_chain.get(k, 0.0)

                        path, lat, bottleneck_before = best_bandwidth_path(
                            prev_node,
                            cur_node,
                            new_link_cap,
                            bw_demand,
                        )

                        if not path or lat is None or bottleneck_before is None or len(path) < 2:
                            routing_ok = False
                        else:
                            bottleneck_after = bottleneck_before - bw_demand
                            if bottleneck_after < -1e-9:
                                routing_ok = False
                            else:
                                incremental_cost += alpha_latency * lat
                                incremental_cost += bottleneck_penalty(
                                    bottleneck_after if bottleneck_after > 0.0 else 0.0
                                )
                                last_bottleneck = bottleneck_after if bottleneck_after > 0.0 else 0.0

                                for u, v in zip(path[:-1], path[1:]):
                                    ek = _edge_key(u, v)
                                    if new_link_cap.get(ek, 0.0) < bw_demand:
                                        routing_ok = False
                                        break
                                    dec_edge_capacity(new_link_cap, ek, bw_demand)
                                    new_sum_link_capacity -= bw_demand

                                if routing_ok:
                                    new_routed[k] = path

                if not routing_ok:
                    continue

                child = FABOState(
                    placed_vnfs=new_placed,
                    routed_vls=new_routed,
                    g_cost=state.g_cost + incremental_cost,
                    h_cost=0.0,
                    node_capacity=new_node_cap,
                    link_capacity=new_link_cap,
                    node_use_count=new_node_use_count,
                    last_bottleneck=last_bottleneck,
                    sum_node_capacity=new_sum_node_capacity,
                    sum_link_capacity=new_sum_link_capacity,
                )
                child.h_cost = heuristic_cost(child, vnf_ids, vnf_cpu)
                child.f_cost = child.g_cost + child.h_cost

                tie += 1
                heapq.heappush(
                    heap,
                    (
                        child.f_cost,
                        child.h_cost,
                        -child.last_bottleneck,
                        tie,
                        child,
                    ),
                )

        results.append(result)

        if verbose:
            if result is not None:
                print(f"[INFO][FABO] Found solution after {visited} expanded states.")
            else:
                print(f"[WARN][FABO] No solution within max_states_per_slice={max_states_per_slice}.")

        if result is not None:
            for vid, node in result.placed_vnfs.items():
                node_capacity_global[node] -= vnf_cpu[vid]
                if node_capacity_global[node] < -1e-9:
                    raise ValueError(
                        f"Node capacity underflow on {node} in slice {i}: remaining={node_capacity_global[node]}"
                    )

            for (src, dst), path in result.routed_vls.items():
                bw = bw_by_chain.get((src, dst), 0.0)

                if src == ENTRY_VNF_ID:
                    src_node = entry
                    dst_node = result.placed_vnfs[dst]
                else:
                    src_node = result.placed_vnfs[src]
                    dst_node = result.placed_vnfs[dst]

                _assert_path_endpoints(path, src_node, dst_node)

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
                "expanded_states": visited,
            }
        )

    df_results = pd.DataFrame(summary_rows)

    if csv_path:
        df_results.to_csv(csv_path, index=False)
        if verbose:
            print(f"[INFO][FABO] Results written to {csv_path}")

    return df_results, results