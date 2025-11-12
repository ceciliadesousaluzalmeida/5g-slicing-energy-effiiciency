# energy_aware_astar_mininfra.py
from copy import deepcopy
import pandas as pd
import networkx as nx
from queue import PriorityQueue
from dataclasses import dataclass
import math

# ------------------------- Energy model -------------------------
@dataclass
class EnergyParams:
    """
    Incremental energy model:
      - link_dynamic_w_per_mbps: per-Mbps transport cost on links
      - node_forward_w_per_mbps: per-Mbps per-hop forwarding on nodes
      - link_baseline_w: one-time cost when a link becomes active
      - node_baseline_w: one-time cost when a node becomes active
      - host_baseline_w: one-time cost when placing a VNF on a previously inactive host
      - min_incremental_per_hop: optimistic per-hop lower bound used for pruning in DFS
    """
    link_dynamic_w_per_mbps: float = 0.002
    node_forward_w_per_mbps: float = 0.001
    link_baseline_w: float = 0.05
    node_baseline_w: float = 0.02
    host_baseline_w: float = 0.02
    min_incremental_per_hop: float = 0.0015  # lower bound helper (safe optimistic)

class AStarInfraState:
    def __init__(self, placed_vnfs=None, routed_vls=None, g_cost=(0,0,0.0),
                 node_capacity=None, link_capacity=None, energy=0.0):
        self.placed_vnfs   = placed_vnfs or {}              # {vnf_id: node}
        self.routed_vls    = routed_vls or {}               # {(from,to): path}
        self.g_cost        = g_cost                         # (nodes_used, links_used, energy)
        self.node_capacity = node_capacity or {}            # {node: cpu_left}
        self.link_capacity = link_capacity or {}            # {(u,v): bw_left} undirected
        self.energy        = float(energy)                  # accumulated energy

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
    return (u, v) if (u, v) else None  # placeholder to avoid lints

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


# ------------------------- Energy-aware DFS path finder -------------------------
def find_min_energy_path_dfs(
    G, s, t, demand_mbps, link_capacity, energy_params,
    active_nodes, active_links, max_hops=None
):
    """
    Energy-driven DFS with branch-and-bound.
    Returns (best_path, path_energy_increment) or (None, None) if infeasible.
    - active_nodes/active_links: sets currently active in the state (baseline already paid).
    - Baselines are charged only when a node/link is activated for the first time along the path,
      and not in active_*.
    - Capacity is read from 'link_capacity' (undirected dict).
    """
    if s == t:
        return [s], 0.0

    best_path, best_cost = None, math.inf
    visited = set()

    # Simple lower bound: remaining hops * (min_incremental_per_hop * demand)
    def heuristic_hops(u):
        try:
            # Fall back to graph distance in hops
            return nx.shortest_path_length(G, u, t)
        except nx.NetworkXNoPath:
            return math.inf

    def step_energy(u, v, used_nodes, used_links):
        # Link part: dynamic + baseline if link freshly activated
        link_key = (u, v) if (u, v) in link_capacity else (v, u)
        dyn = energy_params.link_dynamic_w_per_mbps * demand_mbps
        base_link = 0.0
        if link_key not in active_links and link_key not in used_links:
            base_link = energy_params.link_baseline_w

        # Node forwarding on 'v' (arrival node): dynamic + baseline if freshly activated
        fwd_dyn = energy_params.node_forward_w_per_mbps * demand_mbps
        base_node = 0.0
        if v not in active_nodes and v not in used_nodes:
            base_node = energy_params.node_baseline_w

        return dyn + fwd_dyn + base_link + base_node

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

        # Expand neighbors: prefer active links first, then lower degree (arbitrary tie-break)
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

            # Energy increment for (u->v)
            inc = step_energy(u, v, used_nodes, used_links)

            # Apply step (temporary): mark link used and node used for baseline accounting
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


# ------------------------- Main solver -------------------------
def energy_aware_astar(
    G, slices, node_capacity_base, link_capacity_base,
    w_nodes=1.0, w_links=1.0, energy_params=EnergyParams(), csv_path=None
):
    """
    Infrastructure- & energy-aware search:
      - Lexicographic state cost: (nodes_active, links_active, energy)
      - Routing uses DFS branch-and-bound minimizing energy increment.
    """
    astar_results = []
    node_capacity_global = deepcopy(node_capacity_base)   # committed after each accepted slice
    link_capacity_global = deepcopy(link_capacity_base)

    def state_active_nodes(state):
        # Nodes hosting VNFs or used along routed paths
        nodes = set(state.placed_vnfs.values())
        for path in state.routed_vls.values():
            for v in path:
                nodes.add(v)
        return nodes

    def state_active_links(state):
        # Undirected edge set used by any routed path
        used = set()
        for path in state.routed_vls.values():
            for u, v in zip(path[:-1], path[1:]):
                key = (u, v) if (u, v) in state.link_capacity else ((v, u) if (v, u) in state.link_capacity else tuple(sorted((u, v))))
                used.add(key)
        return used

    def lex_cost_tuple(state):
        nodes_used = len(state_active_nodes(state))
        links_used = len(state_active_links(state))
        return (w_nodes * nodes_used, w_links * links_used, state.energy)

    def expand_state(state, vnf_chain, vl_chain, entry=None):
        expansions = []

        # Identify next VNF to place (smallest id not yet placed – deterministic order)
        unplaced_ids = [v["id"] for v in vnf_chain if v["id"] not in state.placed_vnfs]
        if not unplaced_ids:
            return expansions
        next_vnf_id = sorted(unplaced_ids)[0]
        vnf_obj = next(v for v in vnf_chain if v["id"] == next_vnf_id)

        # Candidate nodes: prefer those with more CPU left; already-active nodes first
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

            # Anti-affinity: avoid placing two VNFs of the same slice on the same node
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

            # Place VNF
            new_node_cap[node] -= cpu_need
            new_placed[next_vnf_id] = node

            # Host baseline energy if node was not active before (host activation)
            if node not in active_nodes_now:
                new_energy += energy_params.host_baseline_w

            # Route all VLs whose endpoints are now known
            routing_ok = True
            # Precompute active sets for baseline accounting during routing
            cur_active_nodes = state_active_nodes(AStarInfraState(new_placed, new_routed, state.g_cost, new_node_cap, new_link_cap))
            cur_active_links = state_active_links(AStarInfraState(new_placed, new_routed, state.g_cost, new_node_cap, new_link_cap))

            for vl in vl_chain:
                key = (vl["from"], vl["to"])
                if key in new_routed:
                    continue
                src_node = new_placed.get(vl["from"])
                dst_node = new_placed.get(vl["to"])
                if src_node is None or dst_node is None:
                    continue

                # DFS energy-min route
                path, path_energy = find_min_energy_path_dfs(
                    G, src_node, dst_node, vl["bandwidth"],
                    new_link_cap, energy_params,
                    active_nodes=cur_active_nodes,
                    active_links=cur_active_links,
                    max_hops=None  # set if you want IDDFS cap
                )
                if path is None:
                    routing_ok = False
                    break

                # Apply link capacities and record path
                for u, v in zip(path[:-1], path[1:]):
                    _dec_link_cap(new_link_cap, u, v, vl["bandwidth"])
                    # Update active sets so next VL benefits from already-ON infra
                    key_uv = (u, v) if (u, v) in new_link_cap else (v, u)
                    cur_active_links.add(key_uv)
                    cur_active_nodes.add(v)

                new_routed[key] = path
                new_energy += path_energy

            # ENTRY → first VNF path (if required and not yet routed)
            if routing_ok and entry is not None and vnf_chain:
                first_id = vnf_chain[0]["id"]
                if ("ENTRY", first_id) not in new_routed:
                    first_node = new_placed.get(first_id)
                    if first_node is not None:
                        # Bandwidth for ENTRY link: use first VL's bandwidth if available; else 0
                        bw_first = vl_chain[0]["bandwidth"] if vl_chain else 0.0
                        path, path_energy = find_min_energy_path_dfs(
                            G, entry, first_node, bw_first,
                            new_link_cap, energy_params,
                            active_nodes=cur_active_nodes,
                            active_links=cur_active_links,
                            max_hops=None
                        )
                        if path is None:
                            routing_ok = False
                        else:
                            for u, v in zip(path[:-1], path[1:]):
                                _dec_link_cap(new_link_cap, u, v, bw_first)
                                key_uv = (u, v) if (u, v) in new_link_cap else (v, u)
                                cur_active_links.add(key_uv)
                                cur_active_nodes.add(v)
                            new_routed[("ENTRY", first_id)] = path
                            new_energy += path_energy

            if not routing_ok:
                continue

            # Build child state and compute its lexicographic cost
            child = AStarInfraState(
                placed_vnfs=new_placed,
                routed_vls=new_routed,
                g_cost=(0, 0, 0.0),  # placeholder; set below
                node_capacity=new_node_cap,
                link_capacity=new_link_cap,
                energy=new_energy
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
            energy=0.0
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
                nodes_used = len(state_active_nodes(state))
                links_used = len(state_active_links(state))
                print(f"[INFO][A*-Energy/Infra] Solution after {visited} expansions. "
                      f"(nodes={nodes_used}, links={links_used}, energy={state.energy:.4f})")
                return state

            for child in expand_state(state, vnf_chain, vl_chain, entry):
                counter += 1
                pq.put((child.g_cost, counter, child))

        print(f"[WARN][A*-Energy/Infra] No feasible solution for slice after {visited} expansions.")
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

        print(f"\n[INFO][A*-Energy/Infra] === Solving slice {i} (VNFs={len(vnf_chain)}, VLs={len(vl_chain)}) ===")
        result_state = solve_one_slice(vnf_chain, vl_chain, entry)
        astar_results.append(result_state)

        if result_state is None:
            print(f"[WARN][A*-Energy/Infra] Slice {i} rejected.\n")
            continue

        # Commit global node capacities
        for vnf_id, node in result_state.placed_vnfs.items():
            vnf_cpu = next(v["cpu"] for v in vnf_chain if v["id"] == vnf_id)
            node_capacity_global[node] -= vnf_cpu

        # Commit global link capacities
        for (src, dst), path in result_state.routed_vls.items():
            # Skip ENTRY link if you don't want to consume global caps for it
            if src == "ENTRY":
                continue
            bw = next(vl["bandwidth"] for vl in vl_chain if vl["from"] == src and vl["to"] == dst)
            for u, v in zip(path[:-1], path[1:]):
                _dec_link_cap(link_capacity_global, u, v, bw)

        nodes_used = len(set(result_state.placed_vnfs.values()) |
                         {v for p in result_state.routed_vls.values() for v in p})
        links_used = len({(u, v_) if (u, v_) in link_capacity_global else (v_, u)
                          for p in result_state.routed_vls.values() for u, v_ in zip(p[:-1], p[1:])})

        print(f"[SUMMARY][A*-Energy/Infra] Slice {i} accepted. "
              f"(nodes={nodes_used}, links={links_used}, energy={result_state.energy:.4f})\n")

    summary = []
    for idx, res in enumerate(astar_results, start=1):
        if res is None:
            summary.append({"slice": idx, "accepted": False, "nodes_used": None, "links_used": None, "energy": None})
        else:
            nodes_used = len(set(res.placed_vnfs.values()) |
                             {v for p in res.routed_vls.values() for v in p})
            links_used = len({(u, v_) if (u, v_) in link_capacity_base else (v_, u)
                              for p in res.routed_vls.values() for u, v_ in zip(p[:-1], p[1:])})
            summary.append({
                "slice": idx,
                "accepted": True,
                "nodes_used": nodes_used,
                "links_used": links_used,
                "energy": res.energy
            })

    df_results = pd.DataFrame(summary)
    if csv_path:
        df_results.to_csv(csv_path, index=False)
    return df_results, astar_results
