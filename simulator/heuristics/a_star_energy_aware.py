from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import heapq
import math
import pandas as pd
import networkx as nx


ENTRY_VNF_ID = "ENTRY"


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


# ------------------------- canonical helpers -------------------------
def _edge_key(u: Any, v: Any) -> Tuple[Any, Any]:
    # Canonical undirected edge key.
    return (u, v) if u <= v else (v, u)


def _validate_canonical_link_keys(
    link_capacity_base: Dict[Tuple[Any, Any], float]
) -> Dict[Tuple[Any, Any], float]:
    # Merge (u,v)/(v,u) into canonical (min,max) with conflict detection.
    merged: Dict[Tuple[Any, Any], float] = {}
    for (u, v), cap in link_capacity_base.items():
        ek = _edge_key(u, v)
        capf = float(cap)
        if ek in merged and abs(merged[ek] - capf) > 1e-9:
            raise ValueError(
                f"Non-canonical or conflicting capacities for edge {ek}: "
                f"seen={merged[ek]} vs new={capf} from key {(u, v)}"
            )
        merged[ek] = capf
    return merged


def _dec_link_cap(
    capdict: Dict[Tuple[Any, Any], float],
    u: Any,
    v: Any,
    amount: float,
) -> None:
    # Decrease remaining bandwidth on canonical edge with underflow check.
    ek = _edge_key(u, v)
    if ek not in capdict:
        raise KeyError(f"Edge {ek} not found in link capacities.")
    capdict[ek] -= float(amount)
    if capdict[ek] < -1e-9:
        raise ValueError(
            f"Link capacity underflow on {ek}: remaining={capdict[ek]}, dec={amount}"
        )


def _build_adj2(G: nx.Graph) -> Dict[Any, List[Tuple[Any, Tuple[Any, Any]]]]:
    # Precompute adjacency with canonical edge keys.
    adj2: Dict[Any, List[Tuple[Any, Tuple[Any, Any]]]] = {}
    for u in G.nodes:
        neigh = []
        for v in G.neighbors(u):
            neigh.append((v, _edge_key(u, v)))
        adj2[u] = neigh
    return adj2


def _extract_entry_bandwidth(
    vl_chain: List[Dict[str, Any]],
    first_vnf_id: str,
) -> float:
    # Extract ENTRY -> first_vnf bandwidth if present, otherwise reuse first chain VL bandwidth.
    for vl in vl_chain:
        if _vl_key(vl) == (ENTRY_VNF_ID, first_vnf_id):
            return float(vl.get("bandwidth", 0.0))
    if vl_chain:
        return float(vl_chain[0].get("bandwidth", 0.0))
    return 0.0


# ------------------------- energy params -------------------------
@dataclass(frozen=True)
class EnergyParams:
    node_baseline_w: float = 1.0
    link_baseline_w: float = 1.0
    node_dynamic_w: float = 0.0
    link_dynamic_w: float = 0.0


def _min_node_dynamic_unit_cost(
    node_capacity_total: Dict[Any, float],
    energy_params: EnergyParams,
) -> float:
    # Minimum dynamic node-energy cost per CPU unit among all valid nodes.
    vals = [
        float(energy_params.node_dynamic_w) / float(cap)
        for cap in node_capacity_total.values()
        if float(cap) > 0.0
    ]
    return float(min(vals)) if vals else 0.0


def _min_link_dynamic_unit_cost(
    link_capacity_total: Dict[Tuple[Any, Any], float],
    energy_params: EnergyParams,
) -> float:
    # Minimum dynamic link-energy cost per BW unit among all valid links.
    vals = [
        float(energy_params.link_dynamic_w) / float(cap)
        for cap in link_capacity_total.values()
        if float(cap) > 0.0
    ]
    return float(min(vals)) if vals else 0.0


# ------------------------- latency support (optional) -------------------------
@dataclass
class LatencyInfo:
    link_latencies: Optional[Dict[Tuple[Any, Any], float]] = None
    vl_sla: Optional[Dict[Tuple[Any, Any], float]] = None

    def path_latency(self, path: List[Any], bw_demand_mbps: float) -> float:
        # Path latency = sum of per-link latencies + optional serialization delay.
        if not self.link_latencies or not path or len(path) <= 1:
            return 0.0

        total = 0.0
        for u, v in zip(path[:-1], path[1:]):
            total += float(self.link_latencies.get(_edge_key(u, v), 0.0))

        if bw_demand_mbps > 0:
            tx = (1500 * 8) / (bw_demand_mbps * 1e6)
            total += tx * (len(path) - 1)

        return float(total)

    def check_path(
        self,
        path: List[Any],
        bw_demand_mbps: float,
        vl_key: Optional[Tuple[Any, Any]] = None,
    ) -> Tuple[bool, float]:
        # Check latency SLA (if any).
        lat = self.path_latency(path, bw_demand_mbps)
        if self.vl_sla is None or vl_key is None:
            return True, lat
        max_lat = self.vl_sla.get(vl_key, None)
        if max_lat is None:
            return True, lat
        return (lat <= float(max_lat)), float(lat)


# ------------------------- A* state -------------------------
class AStarInfraState:
    __slots__ = (
        "placed_vnfs",
        "routed_vls",
        "node_capacity",
        "link_capacity",
        "active_nodes",
        "active_links",
        "node_cpu_used",
        "link_bw_used",
        "energy",
        "h_cost",
        "f_cost",
    )

    def __init__(
        self,
        placed_vnfs: Optional[Dict[Any, Any]] = None,
        routed_vls: Optional[Dict[Tuple[Any, Any], List[Any]]] = None,
        node_capacity: Optional[Dict[Any, float]] = None,
        link_capacity: Optional[Dict[Tuple[Any, Any], float]] = None,
        active_nodes: Optional[Set[Any]] = None,
        active_links: Optional[Set[Tuple[Any, Any]]] = None,
        node_cpu_used: Optional[Dict[Any, float]] = None,
        link_bw_used: Optional[Dict[Tuple[Any, Any], float]] = None,
        energy: float = 0.0,
        h_cost: float = 0.0,
    ):
        self.placed_vnfs = placed_vnfs or {}
        self.routed_vls = routed_vls or {}
        self.node_capacity = node_capacity or {}
        self.link_capacity = link_capacity or {}
        self.active_nodes = active_nodes or set()
        self.active_links = active_links or set()
        self.node_cpu_used = node_cpu_used or {}
        self.link_bw_used = link_bw_used or {}
        self.energy = float(energy)
        self.h_cost = float(h_cost)
        self.f_cost = self._calculate_f_cost()

    def _calculate_f_cost(self):
        # Lexicographic score: total estimated energy, then fewer new activations.
        return (
            self.energy + self.h_cost,
            0.01 * len(self.active_nodes),
            0.001 * len(self.active_links),
        )

    def __lt__(self, other: "AStarInfraState") -> bool:
        return self.f_cost < other.f_cost

    def is_goal_chain(self, vnf_ids: List[str], entry: Optional[Any]) -> bool:
        # Goal = all VNFs placed and all required VLs correctly routed.
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


# ------------------------- lower-bound heuristic -------------------------
def estimate_remaining_energy_admissible(
    state: AStarInfraState,
    vnf_ids: List[str],
    vnf_cpu: Dict[str, float],
    vnf_slice: Dict[str, Any],
    bw_by_chain: Dict[Tuple[str, str], float],
    entry: Optional[Any],
    entry_bw: float,
    node_capacity_total: Dict[Any, float],
    link_capacity_total: Dict[Tuple[Any, Any], float],
    energy_params: EnergyParams,
) -> float:
    """
    Admissible lower bound:
    1) minimal dynamic CPU cost for remaining VNFs,
    2) minimal unavoidable new node activations due to anti-affinity,
    3) minimal dynamic BW cost for remaining unrouted VLs.
    """
    placed_count = len(state.placed_vnfs)
    remaining_vnfs = vnf_ids[placed_count:]

    if not remaining_vnfs:
        return 0.0

    h = 0.0

    min_node_unit = _min_node_dynamic_unit_cost(node_capacity_total, energy_params)
    min_link_unit = _min_link_dynamic_unit_cost(link_capacity_total, energy_params)

    for vid in remaining_vnfs:
        cpu_need = float(vnf_cpu.get(vid, 0.0))
        h += min_node_unit * cpu_need

    current_slice_nodes = {
        node
        for pid, node in state.placed_vnfs.items()
        if vnf_slice.get(pid) == vnf_slice.get(remaining_vnfs[0])
    }

    feasible_active_nodes = 0
    for n in state.active_nodes:
        if n in current_slice_nodes:
            continue
        if float(state.node_capacity.get(n, 0.0)) > 0.0:
            feasible_active_nodes += 1

    unavoidable_new_nodes = max(0, len(remaining_vnfs) - feasible_active_nodes)
    h += float(energy_params.node_baseline_w) * float(unavoidable_new_nodes)

    for idx in range(max(1, placed_count), len(vnf_ids)):
        a = vnf_ids[idx - 1]
        b = vnf_ids[idx]
        if (a, b) not in state.routed_vls:
            h += min_link_unit * float(bw_by_chain.get((a, b), 0.0))

    if entry is not None and vnf_ids:
        first_vnf = vnf_ids[0]
        first_node = state.placed_vnfs.get(first_vnf)
        if (
            first_node is not None
            and first_node != entry
            and (ENTRY_VNF_ID, first_vnf) not in state.routed_vls
        ):
            h += min_link_unit * float(entry_bw)

    return float(h)


# ------------------------- incremental energy -------------------------
def _vnf_increment(
    state: AStarInfraState,
    node: Any,
    cpu_demand: float,
    energy_params: EnergyParams,
    node_capacity_total: Dict[Any, float],
) -> float:
    # Incremental energy for placing a VNF on a node.
    inc = 0.0

    if node not in state.active_nodes:
        inc += float(energy_params.node_baseline_w)

    cap_total = node_capacity_total.get(node, None)
    if cap_total is None or cap_total <= 0:
        raise ValueError(f"Missing/invalid total CPU capacity for node {node}.")

    inc += float(energy_params.node_dynamic_w) * (
        float(cpu_demand) / float(cap_total)
    )
    return float(inc)


def _path_increment_and_apply(
    state: AStarInfraState,
    path: List[Any],
    bw_demand: float,
    energy_params: EnergyParams,
    link_capacity_total: Dict[Tuple[Any, Any], float],
    new_link_cap: Dict[Tuple[Any, Any], float],
    new_active_nodes: Set[Any],
    new_active_links: Set[Tuple[Any, Any]],
    new_link_bw_used: Dict[Tuple[Any, Any], float],
) -> float:
    # Reserve bandwidth along path and compute exact incremental energy.
    if not path or len(path) <= 1:
        return 0.0
    if bw_demand < 0:
        raise ValueError(f"Invalid bw_demand (negative): {bw_demand}")

    inc = 0.0
    bwf = float(bw_demand)

    nb = float(energy_params.node_baseline_w)
    lb = float(energy_params.link_baseline_w)
    ld = float(energy_params.link_dynamic_w)

    # Internal path nodes may become active due to routing.
    for n in path:
        if n not in new_active_nodes:
            new_active_nodes.add(n)
            if n not in state.active_nodes:
                inc += nb

    for u, v in zip(path[:-1], path[1:]):
        ek = _edge_key(u, v)

        rem = new_link_cap.get(ek, None)
        if rem is None:
            raise KeyError(f"Edge {ek} not found in link capacities.")
        if rem < bwf:
            raise ValueError(
                f"Infeasible path: edge {ek} remaining bw {rem} < demand {bwf}."
            )

        new_link_cap[ek] = float(rem) - bwf
        if new_link_cap[ek] < -1e-9:
            raise ValueError(
                f"Internal capacity underflow on {ek}: remaining={new_link_cap[ek]}, dec={bwf}"
            )

        if ek not in new_active_links:
            new_active_links.add(ek)
            if ek not in state.active_links:
                inc += lb

        bw_cap_total = link_capacity_total.get(ek, None)
        if bw_cap_total is None or bw_cap_total <= 0:
            raise ValueError(f"Missing/invalid total BW capacity for link {ek}.")

        inc += ld * (bwf / float(bw_cap_total))
        new_link_bw_used[ek] = float(new_link_bw_used.get(ek, 0.0)) + bwf

    return float(inc)


# ------------------------- fast path energy estimate -------------------------
def _estimate_path_increment_only(
    s: Any,
    t: Any,
    bw_demand: float,
    state: AStarInfraState,
    energy_params: EnergyParams,
    link_capacity_total: Dict[Tuple[Any, Any], float],
    adj2: Dict[Any, List[Tuple[Any, Tuple[Any, Any]]]],
    latency_info: Optional[LatencyInfo] = None,
    vl_key: Optional[Tuple[Any, Any]] = None,
    max_hops: Optional[int] = None,
) -> float:
    # Return the incremental energy of the best feasible path, or +inf if none exists.
    path = _find_min_energy_path_dijkstra(
        s=s,
        t=t,
        bw_demand=bw_demand,
        state=state,
        energy_params=energy_params,
        link_capacity_total=link_capacity_total,
        adj2=adj2,
        latency_info=latency_info,
        vl_key=vl_key,
        max_hops=max_hops,
    )
    if path is None:
        return float("inf")

    if len(path) <= 1:
        return 0.0

    inc = 0.0
    bwf = float(bw_demand)

    for n in path:
        if n not in state.active_nodes:
            inc += float(energy_params.node_baseline_w)

    for u, v in zip(path[:-1], path[1:]):
        ek = _edge_key(u, v)
        if ek not in state.active_links:
            inc += float(energy_params.link_baseline_w)

        bw_cap_total = link_capacity_total.get(ek, None)
        if bw_cap_total is None or bw_cap_total <= 0:
            return float("inf")
        inc += float(energy_params.link_dynamic_w) * (bwf / float(bw_cap_total))

    return float(inc)


# ------------------------- energy-aware Dijkstra -------------------------
def _find_min_energy_path_dijkstra(
    s: Any,
    t: Any,
    bw_demand: float,
    state: AStarInfraState,
    energy_params: EnergyParams,
    link_capacity_total: Dict[Tuple[Any, Any], float],
    adj2: Dict[Any, List[Tuple[Any, Tuple[Any, Any]]]],
    latency_info: Optional[LatencyInfo] = None,
    vl_key: Optional[Tuple[Any, Any]] = None,
    max_hops: Optional[int] = None,
) -> Optional[List[Any]]:
    """
    Minimum incremental-energy path under residual bandwidth constraints.
    """
    if s == t:
        return [s]
    if bw_demand < 0:
        raise ValueError(f"Invalid bw_demand (negative): {bw_demand}")

    bwf = float(bw_demand)

    dist: Dict[Any, float] = {s: 0.0}
    parent: Dict[Any, Any] = {s: None}
    hops: Dict[Any, int] = {s: 0}

    heap = [(0.0, 0, s)]

    active_nodes = state.active_nodes
    active_links = state.active_links
    rem_cap = state.link_capacity

    nb = float(energy_params.node_baseline_w)
    lb = float(energy_params.link_baseline_w)
    ld = float(energy_params.link_dynamic_w)

    while heap:
        cur_cost, cur_hops, u = heapq.heappop(heap)
        if cur_cost != dist.get(u, math.inf):
            continue

        if u == t:
            break

        if max_hops is not None and cur_hops >= int(max_hops):
            continue

        for v, ek in adj2[u]:
            rem = rem_cap.get(ek, None)
            if rem is None or rem < bwf:
                continue

            nh = cur_hops + 1
            if max_hops is not None and nh > int(max_hops):
                continue

            step = 0.0

            if v not in active_nodes:
                step += nb

            if ek not in active_links:
                step += lb

            bw_total = link_capacity_total.get(ek, None)
            if bw_total is None or bw_total <= 0:
                continue
            step += ld * (bwf / float(bw_total))

            new_cost = cur_cost + step
            old = dist.get(v, math.inf)

            if new_cost < old:
                dist[v] = new_cost
                parent[v] = u
                hops[v] = nh
                heapq.heappush(heap, (new_cost, nh, v))

    if t not in dist:
        return None

    path: List[Any] = []
    cur = t
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)
    path.reverse()

    if not path or path[0] != s or path[-1] != t:
        return None

    if latency_info is not None:
        ok, _ = latency_info.check_path(path, bwf, vl_key)
        if not ok:
            return None

    return path


# ------------------------- main solver -------------------------
def energy_aware_astar(
    G: nx.Graph,
    slices: List[Any],
    node_capacity_base: Dict[Any, float],
    link_capacity_base: Dict[Tuple[Any, Any], float],
    energy_params: EnergyParams = EnergyParams(),
    latency_info: Optional[LatencyInfo] = None,
    csv_path: Optional[str] = None,
    max_hops: Optional[int] = None,
    verbose: bool = False,
    beam_width: int = 0,
    max_states_per_slice: int = 1200,
):
    if latency_info is not None and not isinstance(latency_info, LatencyInfo):
        raise TypeError(
            f"latency_info must be LatencyInfo or None, got {type(latency_info).__name__}: {latency_info}"
        )

    node_capacity_total = {n: float(c) for n, c in node_capacity_base.items()}
    link_capacity_base_canon = _validate_canonical_link_keys(link_capacity_base)
    link_capacity_total = dict(link_capacity_base_canon)

    # Global residual capacities.
    node_capacity_global = node_capacity_total.copy()
    link_capacity_global = link_capacity_total.copy()

    # Global active infrastructure persists across accepted slices.
    active_nodes_global: Set[Any] = set()
    active_links_global: Set[Tuple[Any, Any]] = set()

    adj2 = _build_adj2(G)
    nodes_sorted = sorted(G.nodes, key=str)

    astar_results: List[Optional[AStarInfraState]] = []
    summary_rows: List[Dict[str, Any]] = []

    def solve_one_slice_chain(
        vnf_chain: List[Dict[str, Any]],
        vl_chain: List[Dict[str, Any]],
        entry: Optional[Any],
    ) -> Optional[AStarInfraState]:
        vnf_ids = [_as_vnf_id(v["id"]) for v in vnf_chain]
        vnf_by_id = {vid: v for vid, v in zip(vnf_ids, vnf_chain)}
        vnf_cpu = {vid: float(vnf_by_id[vid].get("cpu", 0.0)) for vid in vnf_ids}
        vnf_slice = {vid: vnf_by_id[vid].get("slice") for vid in vnf_ids}

        bw_by_chain: Dict[Tuple[str, str], float] = {}
        for idx in range(1, len(vnf_ids)):
            a, b = vnf_ids[idx - 1], vnf_ids[idx]
            bw = None
            for vl in vl_chain:
                if _vl_key(vl) == (a, b):
                    bw = float(vl.get("bandwidth", 0.0))
                    break
            bw_by_chain[(a, b)] = float(0.0 if bw is None else bw)

        entry_bw = 0.0
        if entry is not None and vnf_ids:
            entry_bw = _extract_entry_bandwidth(vl_chain, vnf_ids[0])

        init_state = AStarInfraState(
            placed_vnfs={},
            routed_vls={},
            node_capacity=node_capacity_global.copy(),
            link_capacity=link_capacity_global.copy(),
            active_nodes=set(active_nodes_global),
            active_links=set(active_links_global),
            node_cpu_used={},
            link_bw_used={},
            energy=0.0,
            h_cost=0.0,
        )
        init_state.h_cost = estimate_remaining_energy_admissible(
            state=init_state,
            vnf_ids=vnf_ids,
            vnf_cpu=vnf_cpu,
            vnf_slice=vnf_slice,
            bw_by_chain=bw_by_chain,
            entry=entry,
            entry_bw=entry_bw,
            node_capacity_total=node_capacity_total,
            link_capacity_total=link_capacity_total,
            energy_params=energy_params,
        )
        init_state.f_cost = init_state._calculate_f_cost()

        heap: List[Tuple[Tuple[float, float, float], int, AStarInfraState]] = []
        tie = 0
        heapq.heappush(heap, (init_state.f_cost, tie, init_state))

        visited = 0
        best_solution: Optional[AStarInfraState] = None
        best_energy = float("inf")

        # Safer dominance: same placement + same routed paths only.
        best_seen: Dict[
            Tuple[
                Tuple[Tuple[Any, Any], ...],
                Tuple[Tuple[Tuple[Any, Any], Tuple[Any, ...]], ...],
            ],
            float,
        ] = {}

        def _candidate_nodes(
            state: AStarInfraState,
            next_vid: str,
            next_idx: int,
        ) -> List[Any]:
            # Candidate ordering based on estimated incremental energy.
            cpu_need = float(vnf_cpu[next_vid])
            slice_id = vnf_slice[next_vid]

            scored: List[Tuple[Tuple[float, float, float, str], Any]] = []

            for node in nodes_sorted:
                avail = float(state.node_capacity.get(node, 0.0))
                if avail < cpu_need:
                    continue

                same_slice_on_node = any(
                    vnf_slice.get(pid) == slice_id and placed_node == node
                    for pid, placed_node in state.placed_vnfs.items()
                )
                if same_slice_on_node:
                    continue

                try:
                    place_inc = _vnf_increment(
                        state=state,
                        node=node,
                        cpu_demand=cpu_need,
                        energy_params=energy_params,
                        node_capacity_total=node_capacity_total,
                    )
                except Exception:
                    continue

                route_est = 0.0

                # Estimate predecessor -> current routing cost.
                if next_idx > 0:
                    prev_vid = vnf_ids[next_idx - 1]
                    prev_node = state.placed_vnfs.get(prev_vid)
                    if prev_node is None:
                        continue
                    if prev_node != node:
                        route_est = _estimate_path_increment_only(
                            s=prev_node,
                            t=node,
                            bw_demand=float(bw_by_chain.get((prev_vid, next_vid), 0.0)),
                            state=state,
                            energy_params=energy_params,
                            link_capacity_total=link_capacity_total,
                            adj2=adj2,
                            latency_info=latency_info,
                            vl_key=(prev_vid, next_vid),
                            max_hops=max_hops,
                        )
                        if math.isinf(route_est):
                            continue

                # Estimate entry -> first VNF routing cost.
                entry_est = 0.0
                if entry is not None and next_idx == 0 and entry != node:
                    entry_est = _estimate_path_increment_only(
                        s=entry,
                        t=node,
                        bw_demand=float(entry_bw),
                        state=state,
                        energy_params=energy_params,
                        link_capacity_total=link_capacity_total,
                        adj2=adj2,
                        latency_info=latency_info,
                        vl_key=(ENTRY_VNF_ID, next_vid),
                        max_hops=max_hops,
                    )
                    if math.isinf(entry_est):
                        continue

                total_est = float(place_inc + route_est + entry_est)

                scored.append(
                    (
                        (
                            total_est,
                            0.0 if node in state.active_nodes else 1.0,
                            -float(avail),
                            str(node),
                        ),
                        node,
                    )
                )

            scored.sort(key=lambda x: x[0])
            ordered = [node for _, node in scored]

            if beam_width and beam_width > 0:
                return ordered[: int(beam_width)]
            return ordered

        while heap and visited < int(max_states_per_slice):
            _, __, state = heapq.heappop(heap)
            visited += 1

            route_sig = tuple(
                sorted((k, tuple(path)) for k, path in state.routed_vls.items())
            )
            prog_key = (
                tuple(sorted(state.placed_vnfs.items())),
                route_sig,
            )

            prev_best = best_seen.get(prog_key)
            if prev_best is not None and state.energy >= prev_best:
                continue
            best_seen[prog_key] = float(state.energy)

            if state.energy >= best_energy:
                continue

            if state.is_goal_chain(vnf_ids, entry):
                if state.energy < best_energy:
                    best_energy = float(state.energy)
                    best_solution = state
                continue

            next_idx = len(state.placed_vnfs)
            if next_idx >= len(vnf_ids):
                continue

            next_vid = vnf_ids[next_idx]
            cpu_need = float(vnf_cpu[next_vid])
            slice_id = vnf_slice[next_vid]

            for node in _candidate_nodes(state, next_vid, next_idx):
                avail = float(state.node_capacity.get(node, 0.0))
                if avail < cpu_need:
                    continue

                same_slice_on_node = any(
                    vnf_slice.get(pid) == slice_id and placed_node == node
                    for pid, placed_node in state.placed_vnfs.items()
                )
                if same_slice_on_node:
                    continue

                new_placed = dict(state.placed_vnfs)
                new_routed = dict(state.routed_vls)
                new_node_cap = state.node_capacity.copy()
                new_link_cap = state.link_capacity.copy()
                new_active_nodes = set(state.active_nodes)
                new_active_links = set(state.active_links)
                new_node_cpu_used = dict(state.node_cpu_used)
                new_link_bw_used = dict(state.link_bw_used)
                new_energy = float(state.energy)

                # Place VNF.
                new_node_cap[node] = avail - cpu_need
                new_placed[next_vid] = node
                new_node_cpu_used[node] = (
                    float(new_node_cpu_used.get(node, 0.0)) + cpu_need
                )

                new_energy += _vnf_increment(
                    state=state,
                    node=node,
                    cpu_demand=cpu_need,
                    energy_params=energy_params,
                    node_capacity_total=node_capacity_total,
                )
                new_active_nodes.add(node)

                tmp_state = AStarInfraState(
                    placed_vnfs=new_placed,
                    routed_vls=new_routed,
                    node_capacity=new_node_cap,
                    link_capacity=new_link_cap,
                    active_nodes=new_active_nodes,
                    active_links=new_active_links,
                    node_cpu_used=new_node_cpu_used,
                    link_bw_used=new_link_bw_used,
                    energy=new_energy,
                    h_cost=0.0,
                )

                # Route predecessor -> current.
                if next_idx > 0:
                    prev_vid = vnf_ids[next_idx - 1]
                    k = (prev_vid, next_vid)
                    na = new_placed[prev_vid]
                    nb = new_placed[next_vid]

                    if na != nb and k not in new_routed:
                        bw = float(bw_by_chain.get(k, 0.0))
                        path = _find_min_energy_path_dijkstra(
                            s=na,
                            t=nb,
                            bw_demand=bw,
                            state=tmp_state,
                            energy_params=energy_params,
                            link_capacity_total=link_capacity_total,
                            adj2=adj2,
                            latency_info=latency_info,
                            vl_key=k,
                            max_hops=max_hops,
                        )
                        if path is None:
                            continue

                        inc = _path_increment_and_apply(
                            state=tmp_state,
                            path=path,
                            bw_demand=bw,
                            energy_params=energy_params,
                            link_capacity_total=link_capacity_total,
                            new_link_cap=new_link_cap,
                            new_active_nodes=new_active_nodes,
                            new_active_links=new_active_links,
                            new_link_bw_used=new_link_bw_used,
                        )
                        new_energy += float(inc)
                        new_routed[k] = path

                        tmp_state = AStarInfraState(
                            placed_vnfs=new_placed,
                            routed_vls=new_routed,
                            node_capacity=new_node_cap,
                            link_capacity=new_link_cap,
                            active_nodes=new_active_nodes,
                            active_links=new_active_links,
                            node_cpu_used=new_node_cpu_used,
                            link_bw_used=new_link_bw_used,
                            energy=new_energy,
                            h_cost=0.0,
                        )

                # Route ENTRY -> first VNF.
                if entry is not None and next_idx == 0:
                    v0 = next_vid
                    n0 = new_placed[v0]

                    if entry != n0 and (ENTRY_VNF_ID, v0) not in new_routed:
                        path = _find_min_energy_path_dijkstra(
                            s=entry,
                            t=n0,
                            bw_demand=float(entry_bw),
                            state=tmp_state,
                            energy_params=energy_params,
                            link_capacity_total=link_capacity_total,
                            adj2=adj2,
                            latency_info=latency_info,
                            vl_key=(ENTRY_VNF_ID, v0),
                            max_hops=max_hops,
                        )
                        if path is None:
                            continue

                        inc = _path_increment_and_apply(
                            state=tmp_state,
                            path=path,
                            bw_demand=float(entry_bw),
                            energy_params=energy_params,
                            link_capacity_total=link_capacity_total,
                            new_link_cap=new_link_cap,
                            new_active_nodes=new_active_nodes,
                            new_active_links=new_active_links,
                            new_link_bw_used=new_link_bw_used,
                        )
                        new_energy += float(inc)
                        new_routed[(ENTRY_VNF_ID, v0)] = path

                child = AStarInfraState(
                    placed_vnfs=new_placed,
                    routed_vls=new_routed,
                    node_capacity=new_node_cap,
                    link_capacity=new_link_cap,
                    active_nodes=new_active_nodes,
                    active_links=new_active_links,
                    node_cpu_used=new_node_cpu_used,
                    link_bw_used=new_link_bw_used,
                    energy=new_energy,
                    h_cost=0.0,
                )

                child.h_cost = estimate_remaining_energy_admissible(
                    state=child,
                    vnf_ids=vnf_ids,
                    vnf_cpu=vnf_cpu,
                    vnf_slice=vnf_slice,
                    bw_by_chain=bw_by_chain,
                    entry=entry,
                    entry_bw=entry_bw,
                    node_capacity_total=node_capacity_total,
                    link_capacity_total=link_capacity_total,
                    energy_params=energy_params,
                )
                child.f_cost = child._calculate_f_cost()

                if child.energy + child.h_cost < best_energy:
                    tie += 1
                    heapq.heappush(heap, (child.f_cost, tie, child))

        if best_solution is not None and verbose:
            print(
                f"[INFO][A*-EnergyAware] Solution found: energy={best_solution.energy:.6f}, "
                f"active_nodes={len(best_solution.active_nodes)}, active_links={len(best_solution.active_links)}, "
                f"expansions={visited}"
            )

        if best_solution is None and verbose:
            print(
                f"[WARN][A*-EnergyAware] No feasible solution "
                f"(expansions={visited}, max_states_per_slice={max_states_per_slice})."
            )

        return best_solution

    # ------------------------- outer loop over slices -------------------------
    for i, slice_data in enumerate(slices, start=1):
        if len(slice_data) == 2:
            vnf_chain, vl_chain = slice_data
            entry = None
        elif len(slice_data) == 3:
            vnf_chain, vl_chain, entry = slice_data
        else:
            raise ValueError(f"Unexpected slice format: {slice_data}")

        if verbose:
            print(
                f"\n[INFO][A*-EnergyAware] === Solving slice {i} "
                f"(VNFs={len(vnf_chain)}, VLs={len(vl_chain)}) ==="
            )

        res = solve_one_slice_chain(vnf_chain, vl_chain, entry)
        astar_results.append(res)

        if res is None:
            summary_rows.append(
                {
                    "slice": i,
                    "accepted": False,
                    "energy": None,
                    "nodes_used": None,
                    "links_used": None,
                }
            )
            if verbose:
                print(f"[SUMMARY][A*-EnergyAware] Slice {i} rejected.")
            continue

        cpu_by_id = {_as_vnf_id(v["id"]): float(v.get("cpu", 0.0)) for v in vnf_chain}
        for vnf_id, node in res.placed_vnfs.items():
            node_capacity_global[node] -= float(cpu_by_id[vnf_id])
            if node_capacity_global[node] < -1e-9:
                raise ValueError(
                    f"Node capacity underflow on {node} in slice {i}: "
                    f"remaining={node_capacity_global[node]}, dec={cpu_by_id[vnf_id]}"
                )

        bw_by_key = {
            (_as_vnf_id(vl["from"]), _as_vnf_id(vl["to"])): float(vl.get("bandwidth", 0.0))
            for vl in vl_chain
        }

        first_vnf_id = _as_vnf_id(vnf_chain[0]["id"]) if vnf_chain else None
        if entry is not None and first_vnf_id is not None:
            bw_by_key[(ENTRY_VNF_ID, first_vnf_id)] = _extract_entry_bandwidth(
                vl_chain, first_vnf_id
            )

        for (src, dst), path in res.routed_vls.items():
            if (src, dst) not in bw_by_key:
                raise KeyError(
                    f"BW not found for routed VL {(src, dst)} in slice {i}. "
                    f"Available VL keys: {list(bw_by_key.keys())}"
                )

            bw = float(bw_by_key[(src, dst)])
            for u, v in zip(path[:-1], path[1:]):
                _dec_link_cap(link_capacity_global, u, v, bw)

        # Persist active infrastructure globally.
        active_nodes_global.update(res.active_nodes)
        active_links_global.update(res.active_links)

        summary_rows.append(
            {
                "slice": i,
                "accepted": True,
                "energy": float(res.energy),
                "nodes_used": len(res.active_nodes),
                "links_used": len(res.active_links),
            }
        )

        if verbose:
            print(
                f"[SUMMARY][A*-EnergyAware] Slice {i} accepted: "
                f"energy={res.energy:.6f}, nodes={len(res.active_nodes)}, "
                f"links={len(res.active_links)}"
            )

            if latency_info is not None and latency_info.vl_sla:
                for vl in vl_chain:
                    k = (_as_vnf_id(vl["from"]), _as_vnf_id(vl["to"]))
                    if k in res.routed_vls and k in latency_info.vl_sla:
                        p = res.routed_vls[k]
                        ok, lat = latency_info.check_path(
                            p, float(vl.get("bandwidth", 0.0)), k
                        )
                        print(
                            f"[DEBUG] VL {k}: latency={lat:.6f} "
                            f"(SLA={latency_info.vl_sla[k]:.6f}) ok={ok}"
                        )

                if (
                    first_vnf_id is not None
                    and (ENTRY_VNF_ID, first_vnf_id) in res.routed_vls
                ):
                    p = res.routed_vls[(ENTRY_VNF_ID, first_vnf_id)]
                    ok, lat = latency_info.check_path(
                        p,
                        float(bw_by_key[(ENTRY_VNF_ID, first_vnf_id)]),
                        (ENTRY_VNF_ID, first_vnf_id),
                    )
                    print(
                        f"[DEBUG] VL {(ENTRY_VNF_ID, first_vnf_id)}: latency={lat:.6f} ok={ok}"
                    )

            if link_capacity_global:
                min_edge = min(link_capacity_global, key=lambda e: link_capacity_global[e])
                print(
                    f"[CHECK] After slice {i}: min remaining link cap = "
                    f"{link_capacity_global[min_edge]:.6f} on edge {min_edge}"
                )

    df = pd.DataFrame(summary_rows)
    if csv_path:
        df.to_csv(csv_path, index=False)
        if verbose:
            print(f"[INFO][A*-EnergyAware] Results saved to {csv_path}")

    return df, astar_results