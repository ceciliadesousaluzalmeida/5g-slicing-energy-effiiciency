# energy_aware_astar_mininfra.py
from copy import deepcopy
import pandas as pd
import networkx as nx
from queue import PriorityQueue
from dataclasses import dataclass
import math
from typing import Dict, List, Tuple, Optional, Set, Any

# ------------------------- Energy model -------------------------
@dataclass
class EnergyParams:
    """
    Energy model compatible with MILP formulation:
      - link_dynamic_w_per_mbps: per-Mbps dynamic cost on links (normalized BW usage)
      - node_forward_w_per_mbps: per-Mbps forwarding cost on nodes
      - link_baseline_w: fixed activation cost when a link becomes active (b_uv * 1)
      - node_baseline_w: fixed activation cost when a node becomes active (a_n * 1)
      - host_baseline_w: one-time cost when placing a VNF on a previously inactive host
      - min_incremental_per_hop: optimistic per-hop lower bound for pruning
    """
    link_dynamic_w_per_mbps: float = 1.0  # Now matches MILP: b_uv * BW_used/BW_cap
    node_forward_w_per_mbps: float = 1.0  # Matches MILP: a_n * CPU_used/CPU_cap
    link_baseline_w: float = 1.0          # Fixed cost: b_uv * 1
    node_baseline_w: float = 1.0          # Fixed cost: a_n * 1
    host_baseline_w: float = 0.0          # Could be merged with node_baseline_w
    min_incremental_per_hop: float = 0.0015


class AStarInfraState:
    def __init__(self, placed_vnfs=None, routed_vls=None, g_cost=(0,0,0.0),
                 node_capacity=None, link_capacity=None, energy=0.0,
                 active_nodes=None, active_links=None,
                 node_cpu_used=None, link_bw_used=None):
        self.placed_vnfs   = placed_vnfs or {}              # {vnf_id: node}
        self.routed_vls    = routed_vls or {}               # {(from,to): path}
        self.g_cost        = g_cost                         # (nodes_used, links_used, energy)
        self.node_capacity = node_capacity or {}            # {node: cpu_left}
        self.link_capacity = link_capacity or {}            # {(u,v): bw_left} undirected
        self.energy        = float(energy)                  # accumulated energy
        
        # NEW: Track active infrastructure and resource usage for MILP-compatible costs
        self.active_nodes  = active_nodes or set()          # Nodes with baseline cost already paid
        self.active_links  = active_links or set()          # Links with baseline cost already paid
        self.node_cpu_used = node_cpu_used or {}            # {node: total_cpu_used} for normalized cost
        self.link_bw_used  = link_bw_used or {}             # {(u,v): total_bw_used} for normalized cost

    def is_goal(self, vnf_chain, vl_chain, entry=None):
        # All VNFs placed
        if len(self.placed_vnfs) != len(vnf_chain):
            return False
        # All VLs routed
        want = set((vl["from"], vl["to"]) for vl in vl_chain)
        if not want.issubset(set(self.routed_vls.keys())):
            return False
        # ENTRY->first VNF (if applicable)
        if entry is not None and vnf_chain:
            first_id = vnf_chain[0]["id"]
            if ("ENTRY", first_id) not in self.routed_vls:
                return False
        return True

    def __lt__(self, other):
        # Lexicographic comparison: (nodes_used, links_used, energy)
        return self.g_cost < other.g_cost


# ------------------------- Helpers for link caps (undirected) -------------------------
def _edge_key(u, v):
    return tuple(sorted((u, v)))

def _get_link_cap(capdict, u, v):
    key = (u, v) if (u, v) in capdict else ((v, u) if (v, u) in capdict else None)
    return capdict.get(key) if key is not None else None

def _dec_link_cap(capdict, u, v, amount):
    if (u, v) in capdict:
        capdict[(u, v)] -= amount
    elif (v, u) in capdict:
        capdict[(v, u)] -= amount
    else:
        raise KeyError(f"Edge ({u},{v}) not found in link capacities.")


# ------------------------- Energy calculation helpers (MILP-compatible) -------------------------
def _calculate_milp_energy_increment(state, new_node=None, new_link=None,
                                     cpu_demand=0, bw_demand=0,
                                     new_active_nodes=None, new_active_links=None,
                                     energy_params=None):
    """
    Calculate energy increment using MILP formula:
      For nodes: a_n * (1 + CPU_used/CPU_cap)
      For links: b_uv * (1 + BW_used/BW_cap)
    """
    increment = 0.0
    new_active_nodes = new_active_nodes or set()
    new_active_links = new_active_links or set()
    
    # Node energy (if placing a VNF)
    if new_node is not None and cpu_demand > 0:
        # Baseline cost: a_n * 1 (only if node becomes active)
        if new_node not in state.active_nodes and new_node in new_active_nodes:
            increment += energy_params.node_baseline_w
        
        # Variable cost: a_n * (CPU_used/CPU_cap)
        # Need to estimate total capacity (used + remaining)
        current_cpu_used = state.node_cpu_used.get(new_node, 0.0)
        remaining_cpu = state.node_capacity.get(new_node, 0.0)
        # Estimate total capacity = used + remaining/0.8 (assuming 80% max usage)
        total_capacity = current_cpu_used + (remaining_cpu / 0.8)
        
        if total_capacity > 0 and (new_node in state.active_nodes or new_node in new_active_nodes):
            # Only pay variable cost if node is active
            cpu_usage_ratio = cpu_demand / total_capacity
            increment += energy_params.node_forward_w_per_mbps * cpu_usage_ratio
    
    # Link energy (if using a link)
    if new_link is not None and bw_demand > 0:
        u, v = new_link
        link_key = (u, v) if (u, v) in state.link_capacity else (v, u)
        
        # Baseline cost: b_uv * 1 (only if link becomes active)
        if link_key not in state.active_links and link_key in new_active_links:
            increment += energy_params.link_baseline_w
        
        # Variable cost: b_uv * (BW_used/BW_cap)
        current_bw_used = state.link_bw_used.get(link_key, 0.0)
        remaining_bw = state.link_capacity.get(link_key, 0.0)
        total_capacity = current_bw_used + (remaining_bw / 0.8)
        
        if total_capacity > 0 and (link_key in state.active_links or link_key in new_active_links):
            # Only pay variable cost if link is active
            bw_usage_ratio = bw_demand / total_capacity
            increment += energy_params.link_dynamic_w_per_mbps * bw_usage_ratio
    
    return increment


# ------------------------- Energy-aware DFS path finder (MILP-compatible) -------------------------
def find_min_energy_path_dfs(
    G, s, t, demand_mbps, link_capacity, energy_params,
    active_nodes, active_links, state,
    max_hops=None
):
    """
    Energy-driven DFS with MILP-compatible cost calculation.
    Returns (best_path, path_energy_increment) or (None, None) if infeasible.
    
    state: current AStarInfraState for accessing node_cpu_used, link_bw_used
    """
    if s == t:
        return [s], 0.0

    best_path, best_cost = None, math.inf
    visited = set()

    def heuristic_hops(u):
        try:
            return nx.shortest_path_length(G, u, t)
        except nx.NetworkXNoPath:
            return math.inf

    def step_energy_milp(u, v, used_nodes, used_links):
        """MILP-compatible energy calculation for step u->v"""
        link_key = (u, v) if (u, v) in link_capacity else (v, u)
        
        # New active sets for this path branch
        new_active_nodes = set(used_nodes)
        new_active_links = set(used_links)
        
        # Add v to active nodes if not already active
        if v not in active_nodes and v not in new_active_nodes:
            new_active_nodes.add(v)
        
        # Add link to active links if not already active
        if link_key not in active_links and link_key not in new_active_links:
            new_active_links.add(link_key)
        
        # Calculate MILP energy increment
        return _calculate_milp_energy_increment(
            state=state,
            new_node=v,
            new_link=(u, v),
            bw_demand=demand_mbps,
            new_active_nodes=new_active_nodes,
            new_active_links=new_active_links,
            energy_params=energy_params
        )

    def lower_bound(u, acc_cost):
        rem_h = heuristic_hops(u)
        if rem_h == math.inf:
            return math.inf
        lb = acc_cost + (energy_params.min_incremental_per_hop * demand_mbps * rem_h)
        return lb

    def dfs(u, acc_cost, path, used_nodes, used_links, depth):
        nonlocal best_path, best_cost
        if acc_cost >= best_cost:
            return
        if max_hops is not None and depth > max_hops:
            return
        # Lower bound pruning
        if lower_bound(u, acc_cost) >= best_cost:
            return
        if u == t:
            best_cost = acc_cost
            best_path = list(path)
            return

        visited.add(u)

        # Expand neighbors
        nbrs = sorted(G.neighbors(u), key=lambda v: (
            (0 if ((u, v) in active_links or (v, u) in active_links) else 1),
            G.degree(v)
        ))

        for v in nbrs:
            if v in visited:
                continue
            # Capacity check
            cap = _get_link_cap(link_capacity, u, v)
            if cap is None or cap < demand_mbps:
                continue

            # Energy increment for (u->v) with MILP formula
            inc = step_energy_milp(u, v, used_nodes, used_links)

            # Apply step (temporary): mark link used and node used
            link_key = (u, v) if (u, v) in link_capacity else (v, u)
            used_links.add(link_key)
            used_nodes.add(v)

            dfs(v, acc_cost + inc, path + [v], used_nodes, used_links, depth + 1)

            # Rollback
            used_links.remove(link_key)
            used_nodes.remove(v)

        visited.remove(u)

    dfs(s, 0.0, [s], set(), set(), 0)
    return best_path, (best_cost if best_path is not None else None)


# ------------------------- Latency constraint helpers -------------------------
@dataclass
class LatencyInfo:
    """
    Optional latency constraints for VLs.
    If provided, paths must satisfy these hard constraints.
    """
    link_latencies: Dict[Tuple, float] = None  # {(u,v): latency} for physical links
    vl_sla: Dict[Tuple, float] = None         # {(from_vnf,to_vnf): max_latency} for VLs
    
    def check_path_latency(self, path, bw_demand, vl_key=None):
        """Check if a path satisfies latency SLA (hard constraint)"""
        if not self.link_latencies or not path:
            return True, 0.0
        
        total_latency = 0.0
        for u, v in zip(path[:-1], path[1:]):
            link_key = _edge_key(u, v)
            total_latency += self.link_latencies.get(link_key, 0.0)
        
        # Add simplified transmission delay
        transmission_delay = (1500 * 8) / (bw_demand * 1e6) if bw_demand > 0 else 0.0
        total_latency += transmission_delay * (len(path) - 1)
        
        # Check against SLA if provided
        if vl_key and self.vl_sla:
            max_latency = self.vl_sla.get(vl_key)
            if max_latency is not None and total_latency > max_latency:
                return False, total_latency
        
        return True, total_latency


# ------------------------- Modified DFS with latency constraints -------------------------
def find_min_energy_path_dfs_with_latency(
    G, s, t, demand_mbps, link_capacity, energy_params,
    active_nodes, active_links, state, latency_info=None, vl_key=None,
    max_hops=None
):
    """
    Energy-driven DFS with MILP costs and optional latency constraints.
    """
    if s == t:
        return [s], 0.0

    best_path, best_cost = None, math.inf
    visited = set()

    def heuristic_hops(u):
        try:
            return nx.shortest_path_length(G, u, t)
        except nx.NetworkXNoPath:
            return math.inf

    def step_energy_milp(u, v, used_nodes, used_links):
        """Same as before"""
        link_key = (u, v) if (u, v) in link_capacity else (v, u)
        new_active_nodes = set(used_nodes)
        new_active_links = set(used_links)
        
        if v not in active_nodes and v not in new_active_nodes:
            new_active_nodes.add(v)
        if link_key not in active_links and link_key not in new_active_links:
            new_active_links.add(link_key)
        
        return _calculate_milp_energy_increment(
            state=state,
            new_node=v,
            new_link=(u, v),
            bw_demand=demand_mbps,
            new_active_nodes=new_active_nodes,
            new_active_links=new_active_links,
            energy_params=energy_params
        )

    def lower_bound(u, acc_cost):
        rem_h = heuristic_hops(u)
        if rem_h == math.inf:
            return math.inf
        lb = acc_cost + (energy_params.min_incremental_per_hop * demand_mbps * rem_h)
        return lb

    def dfs(u, acc_cost, path, used_nodes, used_links, current_latency, depth):
        nonlocal best_path, best_cost
        if acc_cost >= best_cost:
            return
        if max_hops is not None and depth > max_hops:
            return
        # Lower bound pruning
        if lower_bound(u, acc_cost) >= best_cost:
            return
        
        # Latency constraint check (if we have SLA)
        if latency_info and vl_key and latency_info.vl_sla:
            max_latency = latency_info.vl_sla.get(vl_key)
            if max_latency is not None and current_latency > max_latency:
                return
        
        if u == t:
            # Final latency check
            if latency_info:
                feasible, final_latency = latency_info.check_path_latency(path, demand_mbps, vl_key)
                if not feasible:
                    return
            best_cost = acc_cost
            best_path = list(path)
            return

        visited.add(u)

        nbrs = sorted(G.neighbors(u), key=lambda v: (
            (0 if ((u, v) in active_links or (v, u) in active_links) else 1),
            G.degree(v)
        ))

        for v in nbrs:
            if v in visited:
                continue
            # Capacity check
            cap = _get_link_cap(link_capacity, u, v)
            if cap is None or cap < demand_mbps:
                continue
            
            # Calculate link latency if needed
            link_latency = 0.0
            if latency_info and latency_info.link_latencies:
                link_key = _edge_key(u, v)
                link_latency = latency_info.link_latencies.get(link_key, 0.0)
            
            # Check intermediate latency
            new_latency = current_latency + link_latency
            if latency_info and vl_key and latency_info.vl_sla:
                max_latency = latency_info.vl_sla.get(vl_key)
                if max_latency is not None and new_latency > max_latency:
                    continue

            # Energy increment
            inc = step_energy_milp(u, v, used_nodes, used_links)

            # Apply step
            link_key = (u, v) if (u, v) in link_capacity else (v, u)
            used_links.add(link_key)
            used_nodes.add(v)

            dfs(v, acc_cost + inc, path + [v], used_nodes, used_links, new_latency, depth + 1)

            # Rollback
            used_links.remove(link_key)
            used_nodes.remove(v)

        visited.remove(u)

    dfs(s, 0.0, [s], set(), set(), 0.0, 0)
    return best_path, (best_cost if best_path is not None else None)


# ------------------------- Main solver (updated for MILP compatibility) -------------------------
def energy_aware_astar(
    G, slices, node_capacity_base, link_capacity_base,
    w_nodes=1.0, w_links=1.0, energy_params=EnergyParams(),
    latency_info=None, csv_path=None
):
    """
    Infrastructure- & energy-aware search with MILP-compatible costs.
    
    Changes for MILP compatibility:
    1. Energy calculation uses: a_n*(1 + CPU_used/CPU_cap) + b_uv*(1 + BW_used/BW_cap)
    2. No latency penalty in objective (hard constraint if provided)
    3. Track active infrastructure and resource usage for normalized costs
    """
    astar_results = []
    node_capacity_global = deepcopy(node_capacity_base)
    link_capacity_global = deepcopy(link_capacity_base)

    def state_active_nodes(state):
        nodes = set(state.placed_vnfs.values())
        for path in state.routed_vls.values():
            for v in path:
                nodes.add(v)
        return nodes

    def state_active_links(state):
        used = set()
        for path in state.routed_vls.values():
            for u, v in zip(path[:-1], path[1:]):
                key = (u, v) if (u, v) in state.link_capacity else ((v, u) if (v, u) in state.link_capacity else _edge_key(u, v))
                used.add(key)
        return used

    def lex_cost_tuple(state):
        nodes_used = len(state.active_nodes)  # Use tracked active nodes
        links_used = len(state.active_links)  # Use tracked active links
        return (w_nodes * nodes_used, w_links * links_used, state.energy)

    def expand_state(state, vnf_chain, vl_chain, entry=None):
        expansions = []

        # Identify next VNF to place
        unplaced_ids = [v["id"] for v in vnf_chain if v["id"] not in state.placed_vnfs]
        if not unplaced_ids:
            return expansions
        next_vnf_id = sorted(unplaced_ids)[0]
        vnf_obj = next(v for v in vnf_chain if v["id"] == next_vnf_id)

        # Candidate nodes
        active_nodes_now = state_active_nodes(state)
        node_order = sorted(
            G.nodes,
            key=lambda n: ((n not in active_nodes_now), -state.node_capacity.get(n, 0))
        )

        for node in node_order:
            cpu_need = vnf_obj["cpu"]
            avail_cpu = state.node_capacity.get(node, 0)
            if avail_cpu < cpu_need:
                continue

            # Anti-affinity
            same_slice_on_node = any(
                next(v for v in vnf_chain if v["id"] == pid)["slice"] == vnf_obj["slice"]
                for pid, n_ in state.placed_vnfs.items() if n_ == node
            )
            if same_slice_on_node:
                continue

            # Clone state
            new_placed = dict(state.placed_vnfs)
            new_routed = dict(state.routed_vls)
            new_node_cap = deepcopy(state.node_capacity)
            new_link_cap = deepcopy(state.link_capacity)
            new_energy  = state.energy
            
            # NEW: Track resource usage and active infrastructure
            new_active_nodes = set(state.active_nodes)
            new_active_links = set(state.active_links)
            new_node_cpu_used = dict(state.node_cpu_used)
            new_link_bw_used = dict(state.link_bw_used)

            # Place VNF
            new_node_cap[node] -= cpu_need
            new_placed[next_vnf_id] = node
            
            # Update CPU usage tracking
            new_node_cpu_used[node] = new_node_cpu_used.get(node, 0.0) + cpu_need
            
            # Host baseline energy if node was not active before
            if node not in state.active_nodes:
                new_energy += energy_params.host_baseline_w
            
            # MILP node energy cost: a_n * (1 + CPU_used/CPU_cap)
            # Baseline cost if node becomes active
            if node not in state.active_nodes:
                new_active_nodes.add(node)
                new_energy += energy_params.node_baseline_w  # a_n * 1
            
            # Variable CPU cost (only if node is active)
            if node in new_active_nodes:
                # Estimate total capacity
                remaining_cpu = new_node_cap.get(node, 0.0)
                total_capacity = new_node_cpu_used[node] + (remaining_cpu / 0.8)
                if total_capacity > 0:
                    cpu_usage_ratio = cpu_need / total_capacity
                    new_energy += energy_params.node_forward_w_per_mbps * cpu_usage_ratio

            # Route all VLs whose endpoints are now known
            routing_ok = True
            
            # Precompute active sets for this state
            temp_state = AStarInfraState(
                placed_vnfs=new_placed,
                routed_vls=new_routed,
                node_capacity=new_node_cap,
                link_capacity=new_link_cap,
                active_nodes=new_active_nodes,
                active_links=new_active_links,
                node_cpu_used=new_node_cpu_used,
                link_bw_used=new_link_bw_used
            )
            
            cur_active_nodes = state_active_nodes(temp_state)
            cur_active_links = state_active_links(temp_state)

            for vl in vl_chain:
                key = (vl["from"], vl["to"])
                if key in new_routed:
                    continue
                src_node = new_placed.get(vl["from"])
                dst_node = new_placed.get(vl["to"])
                if src_node is None or dst_node is None:
                    continue

                # Choose DFS function based on latency constraints
                if latency_info:
                    path, path_energy = find_min_energy_path_dfs_with_latency(
                        G, src_node, dst_node, vl["bandwidth"],
                        new_link_cap, energy_params,
                        active_nodes=cur_active_nodes,
                        active_links=cur_active_links,
                        state=temp_state,
                        latency_info=latency_info,
                        vl_key=key,
                        max_hops=None
                    )
                else:
                    path, path_energy = find_min_energy_path_dfs(
                        G, src_node, dst_node, vl["bandwidth"],
                        new_link_cap, energy_params,
                        active_nodes=cur_active_nodes,
                        active_links=cur_active_links,
                        state=temp_state,
                        max_hops=None
                    )
                
                if path is None:
                    routing_ok = False
                    break

                # Apply link capacities and record path
                for u, v in zip(path[:-1], path[1:]):
                    link_key = _dec_link_cap(new_link_cap, u, v, vl["bandwidth"])
                    # Update link bandwidth usage tracking
                    canonical_key = _edge_key(u, v)
                    new_link_bw_used[canonical_key] = new_link_bw_used.get(canonical_key, 0.0) + vl["bandwidth"]
                    
                    # Update active links and MILP link energy
                    if canonical_key not in state.active_links:
                        new_active_links.add(canonical_key)
                        # Baseline link cost: b_uv * 1
                        new_energy += energy_params.link_baseline_w
                    
                    # Variable link cost: b_uv * (BW_used/BW_cap)
                    if canonical_key in new_active_links:
                        remaining_bw = new_link_cap.get(canonical_key, 0.0)
                        total_capacity = new_link_bw_used[canonical_key] + (remaining_bw / 0.8)
                        if total_capacity > 0:
                            bw_usage_ratio = vl["bandwidth"] / total_capacity
                            new_energy += energy_params.link_dynamic_w_per_mbps * bw_usage_ratio
                    
                    # Update active nodes for forwarding
                    if v not in state.active_nodes:
                        new_active_nodes.add(v)
                        # Node forwarding baseline already handled above

                new_routed[key] = path
                new_energy += path_energy  # Add any additional energy from DFS

            # ENTRY → first VNF path
            if routing_ok and entry is not None and vnf_chain:
                first_id = vnf_chain[0]["id"]
                if ("ENTRY", first_id) not in new_routed:
                    first_node = new_placed.get(first_id)
                    if first_node is not None:
                        bw_first = vl_chain[0]["bandwidth"] if vl_chain else 0.0
                        
                        if latency_info:
                            path, path_energy = find_min_energy_path_dfs_with_latency(
                                G, entry, first_node, bw_first,
                                new_link_cap, energy_params,
                                active_nodes=cur_active_nodes,
                                active_links=cur_active_links,
                                state=temp_state,
                                latency_info=latency_info,
                                vl_key=("ENTRY", first_id),
                                max_hops=None
                            )
                        else:
                            path, path_energy = find_min_energy_path_dfs(
                                G, entry, first_node, bw_first,
                                new_link_cap, energy_params,
                                active_nodes=cur_active_nodes,
                                active_links=cur_active_links,
                                state=temp_state,
                                max_hops=None
                            )
                        
                        if path is None:
                            routing_ok = False
                        else:
                            for u, v in zip(path[:-1], path[1:]):
                                _dec_link_cap(new_link_cap, u, v, bw_first)
                                canonical_key = _edge_key(u, v)
                                new_link_bw_used[canonical_key] = new_link_bw_used.get(canonical_key, 0.0) + bw_first
                                
                                if canonical_key not in state.active_links:
                                    new_active_links.add(canonical_key)
                                    new_energy += energy_params.link_baseline_w
                                
                                if canonical_key in new_active_links:
                                    remaining_bw = new_link_cap.get(canonical_key, 0.0)
                                    total_capacity = new_link_bw_used[canonical_key] + (remaining_bw / 0.8)
                                    if total_capacity > 0:
                                        bw_usage_ratio = bw_first / total_capacity
                                        new_energy += energy_params.link_dynamic_w_per_mbps * bw_usage_ratio
                                
                                if v not in state.active_nodes:
                                    new_active_nodes.add(v)
                            
                            new_routed[("ENTRY", first_id)] = path
                            new_energy += path_energy

            if not routing_ok:
                continue

            # Build child state
            child = AStarInfraState(
                placed_vnfs=new_placed,
                routed_vls=new_routed,
                g_cost=(0, 0, 0.0),
                node_capacity=new_node_cap,
                link_capacity=new_link_cap,
                energy=new_energy,
                active_nodes=new_active_nodes,
                active_links=new_active_links,
                node_cpu_used=new_node_cpu_used,
                link_bw_used=new_link_bw_used
            )
            child.g_cost = lex_cost_tuple(child)
            expansions.append(child)

        return expansions

    def solve_one_slice(vnf_chain, vl_chain, entry=None):
        init_state = AStarInfraState(
            placed_vnfs={},
            routed_vls={},
            g_cost=(0, 0, 0.0),
            node_capacity=deepcopy(node_capacity_global),
            link_capacity=deepcopy(link_capacity_global),
            energy=0.0,
            active_nodes=set(),
            active_links=set(),
            node_cpu_used={},
            link_bw_used={}
        )
        init_state.g_cost = (0, 0, 0.0)

        pq = PriorityQueue()
        counter = 0
        pq.put((init_state.g_cost, counter, init_state))
        visited = 0

        while not pq.empty():
            _, _, state = pq.get()
            visited += 1
            if state.is_goal(vnf_chain, vl_chain, entry):
                nodes_used = len(state.active_nodes)
                links_used = len(state.active_links)
                print(f"[INFO][A*-Energy Aware] Solution after {visited} expansions. "
                      f"(nodes={nodes_used}, links={links_used}, energy={state.energy:.4f})")
                if latency_info:
                    print(f"[INFO][A*-Energy Aware] Latency constraints: {'SATISFIED' if latency_info else 'N/A'}")
                return state

            for child in expand_state(state, vnf_chain, vl_chain, entry):
                counter += 1
                pq.put((child.g_cost, counter, child))

        print(f"[WARN][A*-Energy Aware] No feasible solution for slice after {visited} expansions.")
        return None

    # ------------------------- Run over slices & commit capacities -------------------------
    for i, slice_data in enumerate(slices, start=1):
        if len(slice_data) == 2:
            vnf_chain, vl_chain = slice_data
            entry = None
        elif len(slice_data) == 3:
            vnf_chain, vl_chain, entry = slice_data
        else:
            raise ValueError(f"Unexpected slice format: {slice_data}")

        print(f"\n[INFO][A*-Energy Aware] === Solving slice {i} (VNFs={len(vnf_chain)}, VLs={len(vl_chain)}) ===")
        if latency_info:
            print(f"[INFO][A*-Energy Aware] Latency constraints enabled for {len(latency_info.vl_sla) if latency_info.vl_sla else 0} VLs")
        
        result_state = solve_one_slice(vnf_chain, vl_chain, entry)
        astar_results.append(result_state)

        if result_state is None:
            print(f"[WARN][A*-Energy Aware] Slice {i} rejected.\n")
            continue

        # Commit global node capacities
        for vnf_id, node in result_state.placed_vnfs.items():
            vnf_cpu = next(v["cpu"] for v in vnf_chain if v["id"] == vnf_id)
            node_capacity_global[node] -= vnf_cpu

        # Commit global link capacities
        for (src, dst), path in result_state.routed_vls.items():
            if src == "ENTRY":
                continue
            bw = next(vl["bandwidth"] for vl in vl_chain if vl["from"] == src and vl["to"] == dst)
            for u, v in zip(path[:-1], path[1:]):
                _dec_link_cap(link_capacity_global, u, v, bw)

        nodes_used = len(result_state.active_nodes)
        links_used = len(result_state.active_links)

        print(f"[SUMMARY][A*-Energy Aware] Slice {i} accepted. "
              f"(nodes={nodes_used}, links={links_used}, energy={result_state.energy:.4f})\n")
        
        # Print VNF placements for debugging
        print(f"[DEBUG] VNF placements: {result_state.placed_vnfs}")
        if latency_info:
            # Check and print latency satisfaction
            for vl in vl_chain:
                key = (vl["from"], vl["to"])
                if key in result_state.routed_vls:
                    path = result_state.routed_vls[key]
                    feasible, actual_latency = latency_info.check_path_latency(
                        path, vl["bandwidth"], key
                    )
                    if feasible and latency_info.vl_sla and key in latency_info.vl_sla:
                        print(f"  VL {key}: latency {actual_latency:.6f} <= SLA {latency_info.vl_sla[key]:.6f}")
                    elif not feasible:
                        print(f"  VL {key}: LATENCY VIOLATION! {actual_latency:.6f} > SLA {latency_info.vl_sla[key]:.6f}")

    summary = []
    for idx, res in enumerate(astar_results, start=1):
        if res is None:
            summary.append({"slice": idx, "accepted": False, "nodes_used": None, "links_used": None, "energy": None})
        else:
            summary.append({
                "slice": idx,
                "accepted": True,
                "nodes_used": len(res.active_nodes),
                "links_used": len(res.active_links),
                "energy": res.energy
            })

    df_results = pd.DataFrame(summary)
    if csv_path:
        df_results.to_csv(csv_path, index=False)
    return df_results, astar_results

