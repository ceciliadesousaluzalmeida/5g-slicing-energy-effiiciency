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
    link_capacity_base: Dict[Tuple[Any, Any], float],
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
    node_dynamic_w: float = 1.0
    link_dynamic_w: float = 1.0


# ------------------------- latency support -------------------------
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
        # Check latency SLA if available.
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
        # Use energy only as the A* score.
        return (self.energy + self.h_cost,)

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


# ------------------------- heuristic -------------------------
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
    # Keep h(n)=0 while debugging energy alignment.
    return 0.0


# ------------------------- incremental energy -------------------------
def _vnf_increment(
    state: AStarInfraState,
    node: Any,
    cpu_demand: float,
    energy_params: EnergyParams,
    node_capacity_total: Dict[Any, float],
) -> float:
    # Compute the incremental node energy after placing one VNF on a node.
    cap_total = float(node_capacity_total.get(node, 0.0))

    if cap_total <= 0.0:
        raise ValueError(f"Missing/invalid total CPU capacity for node {node}.")

    cpu = float(cpu_demand)

    if cpu < 0.0:
        raise ValueError(f"Invalid cpu_demand: {cpu_demand}")

    old_cpu = float(state.node_cpu_used.get(node, 0.0))
    new_cpu = old_cpu + cpu

    if new_cpu > cap_total + 1e-9:
        raise ValueError(
            f"Node capacity exceeded on {node}: used_after={new_cpu}, cap={cap_total}"
        )

    inc = 0.0

    if node not in state.active_nodes:
        inc += float(energy_params.node_baseline_w)

    old_dynamic = float(energy_params.node_dynamic_w) * (old_cpu / cap_total)
    new_dynamic = float(energy_params.node_dynamic_w) * (new_cpu / cap_total)

    inc += new_dynamic - old_dynamic

    return float(inc)


def _path_increment_and_apply(
    state: AStarInfraState,
    path: List[Any],
    bwf: float,
    energy_params: EnergyParams,
    link_capacity_total: Dict[Tuple[Any, Any], float],
) -> float:
    # Apply bandwidth usage on path and return the incremental energy.
    if not path or len(path) < 2:
        return 0.0

    inc = 0.0
    bwf = float(bwf)

    for u, v in zip(path[:-1], path[1:]):
        ek = _edge_key(u, v)

        bw_total = float(link_capacity_total.get(ek, 0.0))

        if bw_total <= 0.0:
            raise ValueError(f"Missing/invalid bandwidth capacity for edge {ek}")

        old_bw = float(state.link_bw_used.get(ek, 0.0))
        new_bw = old_bw + bwf

        if new_bw > bw_total + 1e-9:
            raise ValueError(
                f"Bandwidth capacity exceeded on edge {ek}: "
                f"used_after={new_bw}, cap={bw_total}"
            )

        if ek not in state.active_links:
            inc += float(energy_params.link_baseline_w)
            state.active_links.add(ek)

        old_dynamic = float(energy_params.link_dynamic_w) * (old_bw / bw_total)
        new_dynamic = float(energy_params.link_dynamic_w) * (new_bw / bw_total)

        inc += new_dynamic - old_dynamic

        state.link_bw_used[ek] = new_bw

    return float(inc)


def _estimate_path_increment_only(
    state: AStarInfraState,
    path: List[Any],
    bwf: float,
    energy_params: EnergyParams,
    link_capacity_total: Dict[Tuple[Any, Any], float],
) -> float:
    # Estimate path incremental energy without changing the state.
    if not path or len(path) < 2:
        return 0.0

    inc = 0.0
    bwf = float(bwf)

    for u, v in zip(path[:-1], path[1:]):
        ek = _edge_key(u, v)

        bw_total = float(link_capacity_total.get(ek, 0.0))

        if bw_total <= 0.0:
            return math.inf

        old_bw = float(state.link_bw_used.get(ek, 0.0))
        new_bw = old_bw + bwf

        if new_bw > bw_total + 1e-9:
            return math.inf

        if ek not in state.active_links:
            inc += float(energy_params.link_baseline_w)

        old_dynamic = float(energy_params.link_dynamic_w) * (old_bw / bw_total)
        new_dynamic = float(energy_params.link_dynamic_w) * (new_bw / bw_total)

        inc += new_dynamic - old_dynamic

    return float(inc)


# ------------------------- energy-aware Dijkstra -------------------------
def _find_min_energy_path_dijkstra(
    G: nx.Graph,
    adj2: Dict[Any, List[Tuple[Any, Tuple[Any, Any]]]],
    src: Any,
    dst: Any,
    bwf: float,
    state: AStarInfraState,
    energy_params: EnergyParams,
    link_capacity_total: Dict[Tuple[Any, Any], float],
    latency_info: Optional[LatencyInfo] = None,
    vl_key: Optional[Tuple[Any, Any]] = None,
    max_hops: Optional[int] = None,
    bw_load_penalty: float = 0.0,
) -> Optional[List[Any]]:
    # Find a feasible path minimizing incremental energy plus optional bandwidth-load penalty.
    if src == dst:
        return [src]

    bwf = float(bwf)

    dist: Dict[Any, float] = {src: 0.0}
    prev: Dict[Any, Any] = {}
    hops: Dict[Any, int] = {src: 0}

    heap: List[Tuple[float, int, Any]] = [(0.0, 0, src)]
    visited: Set[Any] = set()

    while heap:
        current_cost, current_hops, u = heapq.heappop(heap)

        if u in visited:
            continue

        visited.add(u)

        if u == dst:
            break

        for v, ek in adj2.get(u, []):
            if v in visited:
                continue

            next_hops = current_hops + 1

            if max_hops is not None and next_hops > int(max_hops):
                continue

            remaining_bw = float(state.link_capacity.get(ek, 0.0))
            total_bw = float(link_capacity_total.get(ek, 0.0))

            if total_bw <= 0.0:
                continue

            if remaining_bw + 1e-9 < bwf:
                continue

            old_bw = float(state.link_bw_used.get(ek, 0.0))
            new_bw = old_bw + bwf

            if new_bw > total_bw + 1e-9:
                continue

            step_cost = 0.0

            if ek not in state.active_links:
                step_cost += float(energy_params.link_baseline_w)
            

            old_dynamic = float(energy_params.link_dynamic_w) * (old_bw / total_bw)
            new_dynamic = float(energy_params.link_dynamic_w) * (new_bw / total_bw)

            step_cost += new_dynamic - old_dynamic

            util_after = new_bw / total_bw
            step_cost += float(bw_load_penalty) * util_after

            new_cost = current_cost + step_cost

            if new_cost + 1e-12 < dist.get(v, math.inf):
                dist[v] = new_cost
                prev[v] = u
                hops[v] = next_hops
                heapq.heappush(heap, (new_cost, next_hops, v))

    if dst not in dist:
        return None

    path = [dst]
    cur = dst

    while cur != src:
        cur = prev.get(cur)

        if cur is None:
            return None

        path.append(cur)

    path.reverse()

    if latency_info is not None:
        ok, _ = latency_info.check_path(path, bwf, vl_key)

        if not ok:
            return None

    return path


# ------------------------- candidate nodes -------------------------
def _candidate_nodes(
    state: AStarInfraState,
    vnf_id: str,
    cpu_need: float,
    vnf_slice: Dict[str, Any],
    node_capacity_total: Dict[Any, float],
    energy_params: EnergyParams,
    nodes_sorted: List[Any],
    cpu_load_penalty: float = 0.0,
) -> List[Any]:
    # Return feasible candidate nodes sorted by incremental energy and CPU utilization.
    candidates = []

    current_slice = vnf_slice.get(vnf_id)

    used_nodes_same_slice = {
        node
        for placed_vnf, node in state.placed_vnfs.items()
        if vnf_slice.get(placed_vnf) == current_slice
    }

    for node in nodes_sorted:
        remaining_cpu = float(state.node_capacity.get(node, 0.0))

        if remaining_cpu + 1e-9 < float(cpu_need):
            continue

        if node in used_nodes_same_slice:
            continue

        cap_total = float(node_capacity_total.get(node, 0.0))

        if cap_total <= 0.0:
            continue

        old_cpu = float(state.node_cpu_used.get(node, 0.0))
        new_cpu = old_cpu + float(cpu_need)

        if new_cpu > cap_total + 1e-9:
            continue

        inc_energy = _vnf_increment(
            state=state,
            node=node,
            cpu_demand=cpu_need,
            energy_params=energy_params,
            node_capacity_total=node_capacity_total,
        )

        cpu_util_after = new_cpu / cap_total
        score = inc_energy + float(cpu_load_penalty) * cpu_util_after

        candidates.append(
            (
                score,
                inc_energy,
                cpu_util_after,
                -remaining_cpu,
                node,
            )
        )

    candidates.sort()

    return [node for *_unused, node in candidates]


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
    cpu_load_penalty: float = 0.0,
    bw_load_penalty: float = 0.02,
):
    if latency_info is not None and not isinstance(latency_info, LatencyInfo):
        raise TypeError(
            f"latency_info must be LatencyInfo or None, got {type(latency_info).__name__}: {latency_info}"
        )

    node_capacity_total = {n: float(c) for n, c in node_capacity_base.items()}
    link_capacity_total = _validate_canonical_link_keys(link_capacity_base)

    node_capacity_global = node_capacity_total.copy()
    link_capacity_global = link_capacity_total.copy()

    node_cpu_used_global: Dict[Any, float] = {n: 0.0 for n in node_capacity_total}
    link_bw_used_global: Dict[Tuple[Any, Any], float] = {
        e: 0.0 for e in link_capacity_total
    }

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

        vnf_cpu = {
            vid: float(vnf_by_id[vid].get("cpu", 0.0))
            for vid in vnf_ids
        }

        vnf_slice = {
            vid: vnf_by_id[vid].get("slice")
            for vid in vnf_ids
        }

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
            node_cpu_used=dict(node_cpu_used_global),
            link_bw_used=dict(link_bw_used_global),
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

        heap: List[Tuple[Tuple[float], int, AStarInfraState]] = []
        tie = 0

        heapq.heappush(heap, (init_state.f_cost, tie, init_state))

        visited = 0
        best_solution: Optional[AStarInfraState] = None
        best_energy = float("inf")

        best_seen: Dict[
            Tuple[
                Tuple[Tuple[Any, Any], ...],
                Tuple[Tuple[Tuple[Any, Any], Tuple[Any, ...]], ...],
            ],
            float,
        ] = {}

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

            candidate_nodes = _candidate_nodes(
                state=state,
                vnf_id=next_vid,
                cpu_need=cpu_need,
                vnf_slice=vnf_slice,
                node_capacity_total=node_capacity_total,
                energy_params=energy_params,
                nodes_sorted=nodes_sorted,
                cpu_load_penalty=cpu_load_penalty,
            )

            if beam_width and beam_width > 0:
                candidate_nodes = candidate_nodes[: int(beam_width)]

            for node in candidate_nodes:
                avail = float(state.node_capacity.get(node, 0.0))

                if avail + 1e-9 < cpu_need:
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

                inc_node = _vnf_increment(
                    state=state,
                    node=node,
                    cpu_demand=cpu_need,
                    energy_params=energy_params,
                    node_capacity_total=node_capacity_total,
                )

                new_energy += inc_node
                new_node_cap[node] = avail - cpu_need
                new_node_cpu_used[node] = (
                    float(new_node_cpu_used.get(node, 0.0)) + cpu_need
                )
                new_active_nodes.add(node)
                new_placed[next_vid] = node

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

                # Route previous VNF -> current VNF.
                if next_idx > 0:
                    prev_vid = vnf_ids[next_idx - 1]
                    k = (prev_vid, next_vid)

                    na = new_placed[prev_vid]
                    nb = new_placed[next_vid]

                    if na != nb and k not in new_routed:
                        bw = float(bw_by_chain.get(k, 0.0))

                        path = _find_min_energy_path_dijkstra(
                            G=G,
                            adj2=adj2,
                            src=na,
                            dst=nb,
                            bwf=bw,
                            state=tmp_state,
                            energy_params=energy_params,
                            link_capacity_total=link_capacity_total,
                            latency_info=latency_info,
                            vl_key=k,
                            max_hops=max_hops,
                            bw_load_penalty=bw_load_penalty,
                        )

                        if path is None:
                            continue

                        inc_path = _path_increment_and_apply(
                            state=tmp_state,
                            path=path,
                            bwf=bw,
                            energy_params=energy_params,
                            link_capacity_total=link_capacity_total,
                        )

                        for u, v in zip(path[:-1], path[1:]):
                            _dec_link_cap(new_link_cap, u, v, bw)

                        new_energy += float(inc_path)
                        new_routed[k] = path
                        tmp_state.energy = new_energy

                # Route ENTRY -> first VNF.
                if entry is not None and next_idx == 0:
                    v0 = next_vid
                    n0 = new_placed[v0]

                    if entry != n0 and (ENTRY_VNF_ID, v0) not in new_routed:
                        k_entry = (ENTRY_VNF_ID, v0)

                        path = _find_min_energy_path_dijkstra(
                            G=G,
                            adj2=adj2,
                            src=entry,
                            dst=n0,
                            bwf=float(entry_bw),
                            state=tmp_state,
                            energy_params=energy_params,
                            link_capacity_total=link_capacity_total,
                            latency_info=latency_info,
                            vl_key=k_entry,
                            max_hops=max_hops,
                            bw_load_penalty=bw_load_penalty,
                        )

                        if path is None:
                            continue

                        inc_path = _path_increment_and_apply(
                            state=tmp_state,
                            path=path,
                            bwf=float(entry_bw),
                            energy_params=energy_params,
                            link_capacity_total=link_capacity_total,
                        )

                        for u, v in zip(path[:-1], path[1:]):
                            _dec_link_cap(new_link_cap, u, v, float(entry_bw))

                        new_energy += float(inc_path)
                        new_routed[k_entry] = path
                        tmp_state.energy = new_energy

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
                f"[INFO][A*-EnergyAware] Solution found: "
                f"energy={best_solution.energy:.6f}, "
                f"active_nodes={len(best_solution.active_nodes)}, "
                f"active_links={len(best_solution.active_links)}, "
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
                f"(VNFs={len(vnf_chain)}, VLs={len(vl_chain)}, entry={entry}) ==="
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

        # Commit the accepted slice state globally.
        node_capacity_global = res.node_capacity.copy()
        link_capacity_global = res.link_capacity.copy()
        node_cpu_used_global = dict(res.node_cpu_used)
        link_bw_used_global = dict(res.link_bw_used)
        active_nodes_global = set(res.active_nodes)
        active_links_global = set(res.active_links)

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
                f"energy={res.energy:.6f}, "
                f"nodes={len(res.active_nodes)}, "
                f"links={len(res.active_links)}"
            )

            if link_capacity_global:
                min_edge = min(link_capacity_global, key=lambda e: link_capacity_global[e])
                print(
                    f"[CHECK] After slice {i}: min remaining link cap = "
                    f"{link_capacity_global[min_edge]:.6f} on edge {min_edge}"
                )

            min_node = min(node_capacity_global, key=lambda n: node_capacity_global[n])
            print(
                f"[CHECK] After slice {i}: min remaining node cap = "
                f"{node_capacity_global[min_node]:.6f} on node {min_node}"
            )

    df = pd.DataFrame(summary_rows)

    if csv_path:
        df.to_csv(csv_path, index=False)

        if verbose:
            print(f"[INFO][A*-EnergyAware] Results saved to {csv_path}")

    return df, astar_results