# energy_aware_astar_mininfra.py
# NOTE: Code comments are in English per your preference.

from copy import deepcopy
from dataclasses import dataclass
from queue import PriorityQueue
from typing import Dict, List, Tuple, Optional, Set, Any

import math
import pandas as pd
import networkx as nx


# ------------------------- Canonical helpers (undirected) -------------------------
def _edge_key(u, v) -> Tuple[Any, Any]:
    """Return canonical undirected edge key."""
    return tuple(sorted((u, v)))


def _get_link_cap(capdict: Dict[Tuple[Any, Any], float], u, v) -> Optional[float]:
    """Get remaining bandwidth on undirected edge."""
    return capdict.get(_edge_key(u, v), None)


def _dec_link_cap(capdict: Dict[Tuple[Any, Any], float], u, v, amount: float) -> None:
    """
    Decrease remaining bandwidth on undirected edge.

    English: Hard-fail if the decrement would underflow (go negative).
    This prevents silent capacity violations during global commit.
    """
    ek = _edge_key(u, v)
    if ek not in capdict:
        raise KeyError(f"Edge {ek} not found in link capacities.")
    capdict[ek] -= amount
    if capdict[ek] < -1e-9:
        raise ValueError(f"Link capacity underflow on {ek}: remaining={capdict[ek]}, dec={amount}")


def _validate_canonical_link_keys(link_capacity_base: Dict[Tuple[Any, Any], float]) -> Dict[Tuple[Any, Any], float]:
    """
    English: Ensure all link keys are canonical and merge duplicates safely.
    If duplicates exist (e.g., (2,6) and (6,2)), it will merge only if capacities match.
    """
    merged: Dict[Tuple[Any, Any], float] = {}
    for (u, v), cap in link_capacity_base.items():
        ek = _edge_key(u, v)
        if ek in merged and abs(merged[ek] - cap) > 1e-9:
            raise ValueError(
                f"Non-canonical or conflicting capacities for edge {ek}: "
                f"seen={merged[ek]} vs new={cap} from key {(u, v)}"
            )
        merged[ek] = float(cap)
    return merged


# ------------------------- Energy model -------------------------
@dataclass(frozen=True)
class EnergyParams:
    """
    MILP-compatible *incremental* energy model (consistent and non-duplicated):
      - node_baseline_w: cost paid once when a node becomes active
      - link_baseline_w: cost paid once when a link becomes active
      - node_dynamic_w: variable cost for CPU placement, proportional to cpu_demand / cpu_cap_total
      - link_dynamic_w: variable cost for bandwidth routing, proportional to bw_demand / bw_cap_total
    """
    node_baseline_w: float = 1.0
    link_baseline_w: float = 1.0
    node_dynamic_w: float = 1.0
    link_dynamic_w: float = 1.0


# ------------------------- Optional latency constraints -------------------------
@dataclass
class LatencyInfo:
    """
    Optional hard latency constraints for VLs.
    - link_latencies uses canonical undirected keys: {(u,v): latency}
    - vl_sla uses VL keys: {(from_vnf,to_vnf): max_latency}
    """
    link_latencies: Optional[Dict[Tuple[Any, Any], float]] = None
    vl_sla: Optional[Dict[Tuple[Any, Any], float]] = None

    def path_latency(self, path: List[Any], bw_demand_mbps: float) -> float:
        """Compute total path latency (prop + simple transmission approximation)."""
        if not self.link_latencies or not path or len(path) == 1:
            return 0.0

        total = 0.0
        for u, v in zip(path[:-1], path[1:]):
            total += self.link_latencies.get(_edge_key(u, v), 0.0)

        # Simple transmission delay approximation per hop (optional)
        # packet=1500B -> 1500*8 bits, bw in Mbps -> bw*1e6 bps
        if bw_demand_mbps > 0:
            tx = (1500 * 8) / (bw_demand_mbps * 1e6)
            total += tx * (len(path) - 1)

        return total

    def check_path(
        self,
        path: List[Any],
        bw_demand_mbps: float,
        vl_key: Optional[Tuple[Any, Any]] = None
    ) -> Tuple[bool, float]:
        """Check if a path satisfies SLA for a given VL key (hard constraint)."""
        lat = self.path_latency(path, bw_demand_mbps)
        if self.vl_sla is None or vl_key is None:
            return True, lat
        max_lat = self.vl_sla.get(vl_key, None)
        if max_lat is None:
            return True, lat
        return (lat <= max_lat), lat


# ------------------------- A* State -------------------------
class AStarInfraState:
    """
    State stores:
      - placements: vnf_id -> node
      - routes: (from_id,to_id) -> path list of nodes
      - remaining capacities (node/link)
      - active infra sets (nodes/links)
      - resource usage tracking (cpu_used/bw_used) for reporting (not required for energy)
      - energy: accumulated incremental energy (no double counting)
    """
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
        """f = g + h with small tie-breakers."""
        # Tie-breaker: fewer active nodes/links
        return (
            self.energy + self.h_cost,
            0.01 * len(self.active_nodes),
            0.001 * len(self.active_links),
        )

    def __lt__(self, other):
        return self.f_cost < other.f_cost

    def is_goal(self, vnf_chain, vl_chain, entry=None) -> bool:
        if len(self.placed_vnfs) != len(vnf_chain):
            return False
        want = set((vl["from"], vl["to"]) for vl in vl_chain)
        if not want.issubset(set(self.routed_vls.keys())):
            return False
        if entry is not None and vnf_chain:
            first_id = vnf_chain[0]["id"]
            if ("ENTRY", first_id) not in self.routed_vls:
                return False
        return True


# ------------------------- Heuristic (admissible & safe) -------------------------
def estimate_remaining_energy_admissible(
    state: AStarInfraState,
    vnf_chain: List[Dict[str, Any]],
    vl_chain: List[Dict[str, Any]],
    energy_params: EnergyParams,
    node_capacity_total: Dict[Any, float],
    link_capacity_total: Dict[Tuple[Any, Any], float],
) -> float:
    """
    Admissible heuristic: returns 0.0 (never overestimates).
    You can later strengthen it carefully, but 0 is consistent and safe.
    """
    return 0.0


# ------------------------- Energy increment utilities (NO DOUBLE COUNTING) -------------------------
def _vnf_increment(
    state: AStarInfraState,
    node: Any,
    cpu_demand: float,
    energy_params: EnergyParams,
    node_capacity_total: Dict[Any, float],
) -> float:
    """Incremental energy for placing a VNF on a node."""
    inc = 0.0
    if node not in state.active_nodes:
        inc += energy_params.node_baseline_w

    cap_total = node_capacity_total.get(node, None)
    if cap_total is None or cap_total <= 0:
        raise ValueError(f"Missing/invalid total CPU capacity for node {node}.")

    inc += energy_params.node_dynamic_w * (cpu_demand / cap_total)
    return inc


def _path_increment_and_apply(
    state: AStarInfraState,
    path: List[Any],
    bw_demand: float,
    energy_params: EnergyParams,
    link_capacity_total: Dict[Tuple[Any, Any], float],
    # Mutated copies (child state buffers):
    new_link_cap: Dict[Tuple[Any, Any], float],
    new_active_nodes: Set[Any],
    new_active_links: Set[Tuple[Any, Any]],
    new_link_bw_used: Dict[Tuple[Any, Any], float],
) -> float:
    """
    Compute incremental energy for routing a VL along `path` and apply:
      - remaining capacity decrement on each edge
      - active links/nodes updates
      - bw_used updates

    English: This function hard-checks capacity against the *child state's* remaining capacities
    and will raise if infeasible.
    """
    if not path or len(path) == 1:
        return 0.0

    if bw_demand < 0:
        raise ValueError(f"Invalid bw_demand (negative): {bw_demand}")

    inc = 0.0

    # Activate nodes along the path (baseline once)
    for n in path:
        if n not in new_active_nodes:
            new_active_nodes.add(n)
            if n not in state.active_nodes:
                inc += energy_params.node_baseline_w

    # Process edges
    for u, v in zip(path[:-1], path[1:]):
        ek = _edge_key(u, v)

        # Capacity check (hard)
        rem = new_link_cap.get(ek, None)
        if rem is None:
            raise KeyError(f"Edge {ek} not found in link capacities.")
        if rem < bw_demand:
            raise ValueError(f"Infeasible path: edge {ek} remaining bw {rem} < demand {bw_demand}.")

        # Consume capacity
        new_link_cap[ek] -= bw_demand
        if new_link_cap[ek] < -1e-9:
            raise ValueError(f"Internal capacity underflow on {ek}: remaining={new_link_cap[ek]}, dec={bw_demand}")

        # Activate link baseline once
        if ek not in new_active_links:
            new_active_links.add(ek)
            if ek not in state.active_links:
                inc += energy_params.link_baseline_w

        # Link dynamic
        bw_cap_total = link_capacity_total.get(ek, None)
        if bw_cap_total is None or bw_cap_total <= 0:
            raise ValueError(f"Missing/invalid total BW capacity for link {ek}.")
        inc += energy_params.link_dynamic_w * (bw_demand / bw_cap_total)

        # Tracking
        new_link_bw_used[ek] = new_link_bw_used.get(ek, 0.0) + bw_demand

    return inc


# ------------------------- Routing: k-shortest simple paths + exact incremental cost -------------------------
def _find_min_energy_path_ksp(
    G: nx.Graph,
    s: Any,
    t: Any,
    bw_demand: float,
    state: AStarInfraState,
    energy_params: EnergyParams,
    link_capacity_total: Dict[Tuple[Any, Any], float],
    latency_info: Optional[LatencyInfo] = None,
    vl_key: Optional[Tuple[Any, Any]] = None,
    k: int = 30,
    max_hops: Optional[int] = None,
) -> Optional[List[Any]]:
    """
    Find a feasible path that minimizes *exact* incremental energy given current state,
    by enumerating up to k shortest simple paths (by hop count) and evaluating exact incremental energy.
    """
    if s == t:
        return [s]

    if bw_demand < 0:
        raise ValueError(f"Invalid bw_demand (negative): {bw_demand}")

    # Generator of simple paths in increasing hop-length (unweighted) order.
    try:
        gen = nx.shortest_simple_paths(G, s, t)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None

    best_path = None
    best_cost = math.inf
    checked = 0

    for path in gen:
        checked += 1
        if checked > k:
            break

        if max_hops is not None and (len(path) - 1) > max_hops:
            continue

        # Capacity feasibility
        feasible = True
        for u, v in zip(path[:-1], path[1:]):
            rem = state.link_capacity.get(_edge_key(u, v), None)
            if rem is None or rem < bw_demand:
                feasible = False
                break
        if not feasible:
            continue

        # Latency feasibility (hard)
        if latency_info is not None:
            ok, _lat = latency_info.check_path(path, bw_demand, vl_key)
            if not ok:
                continue

        # Exact incremental energy evaluation WITHOUT mutating the state
        tmp_link_cap = deepcopy(state.link_capacity)
        tmp_active_nodes = set(state.active_nodes)
        tmp_active_links = set(state.active_links)
        tmp_link_bw_used = dict(state.link_bw_used)

        try:
            inc = _path_increment_and_apply(
                state=state,
                path=path,
                bw_demand=bw_demand,
                energy_params=energy_params,
                link_capacity_total=link_capacity_total,
                new_link_cap=tmp_link_cap,
                new_active_nodes=tmp_active_nodes,
                new_active_links=tmp_active_links,
                new_link_bw_used=tmp_link_bw_used,
            )
        except Exception:
            continue

        if inc < best_cost:
            best_cost = inc
            best_path = path

    return best_path


# ------------------------- Main solver -------------------------
def energy_aware_astar(
    G: nx.Graph,
    slices: List[Any],
    node_capacity_base: Dict[Any, float],
    link_capacity_base: Dict[Tuple[Any, Any], float],
    energy_params: EnergyParams = EnergyParams(),
    latency_info: Optional[LatencyInfo] = None,
    csv_path: Optional[str] = None,
    k_paths: int = 30,
    max_hops: Optional[int] = None,
):
    """
    Energy-aware A* that is:
      - consistent (canonical edge keys everywhere)
      - MILP-style baseline + variable terms
      - no double counting (routing cost computed once per path)
      - hard constraints for capacities and optional latency SLA
      - sequential slice admission (commit after each accepted slice)

    Fixes in this version:
      1) Global commit now hard-checks capacity underflow.
      2) Commit validates that every routed VL has a matching bandwidth in vl_chain.
      3) Canonical link keys are validated/merged at input.
    """

    if latency_info is not None and not isinstance(latency_info, LatencyInfo):
        raise TypeError(f"latency_info must be LatencyInfo or None, got {type(latency_info).__name__}: {latency_info}")

    # Total capacities are the base capacities (assumed constant)
    node_capacity_total = {n: float(c) for n, c in node_capacity_base.items()}
    link_capacity_base_canon = _validate_canonical_link_keys(link_capacity_base)
    link_capacity_total = dict(link_capacity_base_canon)

    # Global remaining capacities (committed sequentially)
    node_capacity_global = deepcopy(node_capacity_total)
    link_capacity_global = deepcopy(link_capacity_total)

    astar_results: List[Optional[AStarInfraState]] = []

    def expand_state(state: AStarInfraState, vnf_chain, vl_chain, entry=None) -> List[AStarInfraState]:
        expansions: List[AStarInfraState] = []

        # Choose next VNF in *chain order* (not by id sorting)
        next_vnf = None
        for v in vnf_chain:
            if v["id"] not in state.placed_vnfs:
                next_vnf = v
                break
        if next_vnf is None:
            return expansions

        next_vnf_id = next_vnf["id"]
        cpu_need = float(next_vnf.get("cpu", 0.0))

        # Candidate nodes: prefer already active nodes, then highest remaining CPU
        node_order = sorted(
            G.nodes,
            key=lambda n: ((n not in state.active_nodes), -state.node_capacity.get(n, 0.0))
        )

        # Pre-index VNFs for anti-affinity checks
        vnf_by_id = {v["id"]: v for v in vnf_chain}

        for node in node_order:
            avail_cpu = state.node_capacity.get(node, 0.0)
            if avail_cpu < cpu_need:
                continue

            # Anti-affinity: VNFs of same slice cannot share same node
            same_slice_on_node = any(
                vnf_by_id[pid]["slice"] == next_vnf["slice"]
                for pid, n_ in state.placed_vnfs.items()
                if n_ == node
            )
            if same_slice_on_node:
                continue

            # Clone structures
            new_placed = dict(state.placed_vnfs)
            new_routed = dict(state.routed_vls)
            new_node_cap = deepcopy(state.node_capacity)
            new_link_cap = deepcopy(state.link_capacity)
            new_active_nodes = set(state.active_nodes)
            new_active_links = set(state.active_links)
            new_node_cpu_used = dict(state.node_cpu_used)
            new_link_bw_used = dict(state.link_bw_used)
            new_energy = state.energy

            # Place VNF (apply CPU capacity + energy)
            new_node_cap[node] -= cpu_need
            new_placed[next_vnf_id] = node
            new_node_cpu_used[node] = new_node_cpu_used.get(node, 0.0) + cpu_need

            # Activate node and add incremental VNF energy
            if node not in new_active_nodes:
                new_active_nodes.add(node)
            new_energy += _vnf_increment(
                state=state,
                node=node,
                cpu_demand=cpu_need,
                energy_params=energy_params,
                node_capacity_total=node_capacity_total,
            )

            # Build a temporary state (for routing evaluation)
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

            # Route VLs whose endpoints are now placed
            routing_ok = True

            for vl in vl_chain:
                vl_key = (vl["from"], vl["to"])
                if vl_key in new_routed:
                    continue

                src_node = new_placed.get(vl["from"])
                dst_node = new_placed.get(vl["to"])
                if src_node is None or dst_node is None:
                    continue

                bw = float(vl.get("bandwidth", 0.0))

                path = _find_min_energy_path_ksp(
                    G=G,
                    s=src_node,
                    t=dst_node,
                    bw_demand=bw,
                    state=tmp_state,
                    energy_params=energy_params,
                    link_capacity_total=link_capacity_total,
                    latency_info=latency_info,
                    vl_key=vl_key,
                    k=k_paths,
                    max_hops=max_hops,
                )
                if path is None:
                    routing_ok = False
                    break

                # Apply routing effects *once* (no double counting)
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
                new_energy += inc
                new_routed[vl_key] = path

                # Update tmp_state for subsequent VLs
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

            # ENTRY -> first VNF (optional)
            if routing_ok and entry is not None and vnf_chain:
                first_id = vnf_chain[0]["id"]
                if ("ENTRY", first_id) not in new_routed:
                    first_node = new_placed.get(first_id)
                    if first_node is None:
                        routing_ok = False
                    else:
                        bw_first = float(vl_chain[0].get("bandwidth", 0.0)) if vl_chain else 0.0
                        path = _find_min_energy_path_ksp(
                            G=G,
                            s=entry,
                            t=first_node,
                            bw_demand=bw_first,
                            state=tmp_state,
                            energy_params=energy_params,
                            link_capacity_total=link_capacity_total,
                            latency_info=latency_info,
                            vl_key=("ENTRY", first_id),
                            k=k_paths,
                            max_hops=max_hops,
                        )
                        if path is None:
                            routing_ok = False
                        else:
                            inc = _path_increment_and_apply(
                                state=tmp_state,
                                path=path,
                                bw_demand=bw_first,
                                energy_params=energy_params,
                                link_capacity_total=link_capacity_total,
                                new_link_cap=new_link_cap,
                                new_active_nodes=new_active_nodes,
                                new_active_links=new_active_links,
                                new_link_bw_used=new_link_bw_used,
                            )
                            new_energy += inc
                            new_routed[("ENTRY", first_id)] = path

            if not routing_ok:
                continue

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
            expansions.append(child)

        return expansions

    def solve_one_slice(vnf_chain, vl_chain, entry=None) -> Optional[AStarInfraState]:
        init_state = AStarInfraState(
            placed_vnfs={},
            routed_vls={},
            node_capacity=deepcopy(node_capacity_global),
            link_capacity=deepcopy(link_capacity_global),
            active_nodes=set(),
            active_links=set(),
            node_cpu_used={},
            link_bw_used={},
            energy=0.0,
            h_cost=0.0,
        )
        init_state.h_cost = estimate_remaining_energy_admissible(
            init_state, vnf_chain, vl_chain, energy_params, node_capacity_total, link_capacity_total
        )
        init_state.f_cost = init_state._calculate_f_cost()

        pq = PriorityQueue()
        counter = 0
        pq.put((init_state.f_cost, counter, init_state))

        visited = 0
        best_solution = None
        best_energy = float("inf")

        while not pq.empty():
            _, _, state = pq.get()
            visited += 1

            if state.energy >= best_energy:
                continue

            if state.is_goal(vnf_chain, vl_chain, entry):
                if state.energy < best_energy:
                    best_energy = state.energy
                    best_solution = state
                continue

            for child in expand_state(state, vnf_chain, vl_chain, entry):
                child.h_cost = estimate_remaining_energy_admissible(
                    child, vnf_chain, vl_chain, energy_params, node_capacity_total, link_capacity_total
                )
                child.f_cost = child._calculate_f_cost()

                if child.energy + child.h_cost < best_energy:
                    counter += 1
                    pq.put((child.f_cost, counter, child))

        if best_solution is not None:
            print(
                f"[INFO][A*-EnergyAware] Solution found: energy={best_solution.energy:.6f}, "
                f"active_nodes={len(best_solution.active_nodes)}, active_links={len(best_solution.active_links)}, "
                f"expansions={visited}"
            )
            return best_solution

        print(f"[WARN][A*-EnergyAware] No feasible solution (expansions={visited}).")
        return None

    # ------------------------- Run slices sequentially & commit capacities -------------------------
    for i, slice_data in enumerate(slices, start=1):
        if len(slice_data) == 2:
            vnf_chain, vl_chain = slice_data
            entry = None
        elif len(slice_data) == 3:
            vnf_chain, vl_chain, entry = slice_data
        else:
            raise ValueError(f"Unexpected slice format: {slice_data}")

        print(f"\n[INFO][A*-EnergyAware] === Solving slice {i} (VNFs={len(vnf_chain)}, VLs={len(vl_chain)}) ===")
        if latency_info and latency_info.vl_sla:
            print(f"[INFO][A*-EnergyAware] Latency SLA enabled for {len(latency_info.vl_sla)} VLs")

        res = solve_one_slice(vnf_chain, vl_chain, entry)
        astar_results.append(res)

        if res is None:
            print(f"[SUMMARY][A*-EnergyAware] Slice {i} rejected.")
            continue

        # Commit global node capacities (optional: add underflow checks if desired)
        cpu_by_id = {v["id"]: float(v.get("cpu", 0.0)) for v in vnf_chain}
        for vnf_id, node in res.placed_vnfs.items():
            node_capacity_global[node] -= cpu_by_id[vnf_id]
            if node_capacity_global[node] < -1e-9:
                raise ValueError(
                    f"Node capacity underflow on {node} in slice {i}: remaining={node_capacity_global[node]}, "
                    f"dec={cpu_by_id[vnf_id]}"
                )

        # Commit global link capacities (skip ENTRY)
        bw_by_key = {(vl["from"], vl["to"]): float(vl.get("bandwidth", 0.0)) for vl in vl_chain}

        for (src, dst), path in res.routed_vls.items():
            if src == "ENTRY":
                continue

            if (src, dst) not in bw_by_key:
                raise KeyError(
                    f"BW not found for routed VL {(src, dst)} in slice {i}. Available VL keys: {list(bw_by_key.keys())}"
                )

            bw = bw_by_key[(src, dst)]
            for u, v in zip(path[:-1], path[1:]):
                _dec_link_cap(link_capacity_global, u, v, bw)

        print(
            f"[SUMMARY][A*-EnergyAware] Slice {i} accepted: "
            f"energy={res.energy:.6f}, nodes={len(res.active_nodes)}, links={len(res.active_links)}"
        )
        print(f"[DEBUG] Placements: {res.placed_vnfs}")

        if latency_info is not None and latency_info.vl_sla:
            for vl in vl_chain:
                k = (vl["from"], vl["to"])
                if k in res.routed_vls and k in latency_info.vl_sla:
                    p = res.routed_vls[k]
                    ok, lat = latency_info.check_path(p, float(vl.get("bandwidth", 0.0)), k)
                    print(f"[DEBUG] VL {k}: latency={lat:.6f} (SLA={latency_info.vl_sla[k]:.6f}) ok={ok}")

        # English: Optional sanity print (min remaining link cap after each accepted slice)
        if link_capacity_global:
            min_edge = min(link_capacity_global, key=lambda e: link_capacity_global[e])
            print(f"[CHECK] After slice {i}: min remaining link cap = {link_capacity_global[min_edge]:.6f} on edge {min_edge}")

    # ------------------------- Summary dataframe -------------------------
    summary = []
    for idx, st in enumerate(astar_results, start=1):
        if st is None:
            summary.append(
                {"slice": idx, "accepted": False, "nodes_used": None, "links_used": None, "energy": None, "f_cost": None}
            )
        else:
            summary.append(
                {
                    "slice": idx,
                    "accepted": True,
                    "nodes_used": len(st.active_nodes),
                    "links_used": len(st.active_links),
                    "energy": st.energy,
                    "f_cost": st.f_cost[0],
                }
            )

    df = pd.DataFrame(summary)
    if csv_path:
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Results saved to {csv_path}")

    return df, astar_results
