from heapq import heappush, heappop

import pandas as pd

try:
    from sage.all import Graph as SageGraph
except Exception:
    SageGraph = None

import networkx as nx


ENTRY_VNF_ID = "__ENTRY__"


# ------------------------- helpers: normalization -------------------------
def _as_vnf_id(v):
    return str(v).strip()


def _vl_key_from_to(v_from, v_to):
    return (_as_vnf_id(v_from), _as_vnf_id(v_to))


def _vl_key(vl):
    return _vl_key_from_to(vl["from"], vl["to"])


# ------------------------- helpers: entry handling -------------------------
def _first_vnf_id(vnf_chain):
    if not vnf_chain:
        raise ValueError("vnf_chain is empty.")
    return _as_vnf_id(vnf_chain[0]["id"])


def _infer_entry_bandwidth(vnf_chain, vl_chain):
    """
    English: Infer the ingress bandwidth from the first outgoing VL of the first VNF.
    If there is no outgoing VL, default to 0.0.
    """
    if not vnf_chain:
        return 0.0

    first_id = _first_vnf_id(vnf_chain)
    for vl in vl_chain:
        if _as_vnf_id(vl["from"]) == first_id:
            return float(vl["bandwidth"])
    return 0.0


def _augment_vl_chain_with_entry(vnf_chain, vl_chain, entry):
    """
    English: Add a virtual ingress VL from ENTRY_VNF_ID to the first VNF
    so the slice must effectively start at the given topology entry node.
    """
    if entry is None:
        return list(vl_chain)

    first_id = _first_vnf_id(vnf_chain)
    ingress_bw = _infer_entry_bandwidth(vnf_chain, vl_chain)

    augmented = list(vl_chain)
    augmented.insert(
        0,
        {
            "from": ENTRY_VNF_ID,
            "to": first_id,
            "bandwidth": ingress_bw,
        },
    )
    return augmented


# ------------------------- helpers: canonical undirected edges -------------------------
def _edge_key(u, v):
    return (u, v) if u <= v else (v, u)


def _assert_path_endpoints(path, src_node, dst_node):
    if not path:
        raise ValueError("VL path is empty.")
    if path[0] != src_node or path[-1] != dst_node:
        raise ValueError(
            f"VL path endpoints do not match: expected ({src_node}->{dst_node}), "
            f"got ({path[0]}->{path[-1]})"
        )


# ------------------------- graph adapter (NX -> Sage) -------------------------
def _to_sage_graph_if_needed(G):
    """
    Convert a NetworkX undirected graph into a Sage Graph once.
    Edge label becomes the latency (float), so Sage weighted shortest path uses it directly.
    """
    if SageGraph is None:
        raise RuntimeError("Sage is not available in this environment (cannot import sage.all).")

    if isinstance(G, SageGraph):
        return G

    if not isinstance(G, (nx.Graph, nx.DiGraph)):
        raise TypeError("G must be a Sage Graph or a NetworkX graph.")

    if isinstance(G, nx.DiGraph):
        raise ValueError("This implementation expects an undirected graph (use nx.Graph).")

    sG = SageGraph()
    sG.add_vertices(list(G.nodes))

    for u, v, data in G.edges(data=True):
        lat = float(data.get("latency", 1.0))
        sG.add_edge(u, v, lat)

    return sG


def _build_edge_latency_from_sage(sG):
    """
    Build a canonical dict edge_latency[(min,max)] = latency.
    """
    edge_latency = {}
    for u, v, lat in sG.edge_iterator(labels=True):
        edge_latency[_edge_key(u, v)] = float(lat)
    return edge_latency


# ------------------------- helper: feasible shortest path -------------------------
def shortest_path_with_capacity_sage(sG, u, v, link_capacity_canon, bandwidth):
    """
    Weighted shortest path in Sage (latency = edge label), then capacity check.
    """
    if u == v:
        return [u], 0.0

    try:
        path = sG.shortest_path(u, v, by_weight=True)
    except Exception:
        return None, None

    if not path:
        return None, None

    for a, b in zip(path[:-1], path[1:]):
        cap = link_capacity_canon.get(_edge_key(a, b))
        if cap is None or cap < bandwidth:
            return None, None

    latency = 0.0
    for a, b in zip(path[:-1], path[1:]):
        latency += float(sG.edge_label(a, b))

    return path, float(latency)


# ------------------------- A* DATA STRUCTURES -------------------------
class AStarState:
    __slots__ = (
        "placed_vnfs",
        "routed_vls",
        "g_cost",
        "node_capacity",
        "link_capacity",
        "node_used_slices",
    )

    def __init__(
        self,
        placed_vnfs=None,
        routed_vls=None,
        g_cost=0.0,
        node_capacity=None,
        link_capacity=None,
        node_used_slices=None,
    ):
        self.placed_vnfs = placed_vnfs or {}            # {vnf_id(str): node}
        self.routed_vls = routed_vls or {}              # {(from_id,to_id): [path nodes]}
        self.g_cost = float(g_cost)
        self.node_capacity = node_capacity or {}        # {node: remaining_cpu}
        self.link_capacity = link_capacity or {}        # {(u,v) canon: remaining_bw}
        self.node_used_slices = node_used_slices or {}  # {node: set(slice_ids)}

    def is_goal(self, real_vnf_ids, vl_list_norm):
        placed_real = sum(1 for vid in real_vnf_ids if vid in self.placed_vnfs)
        if placed_real != len(real_vnf_ids):
            return False

        for (v_from, v_to, _bw) in vl_list_norm:
            src_node = self.placed_vnfs.get(v_from)
            dst_node = self.placed_vnfs.get(v_to)
            if src_node is None or dst_node is None:
                return False
            if src_node == dst_node:
                continue
            path = self.routed_vls.get((v_from, v_to))
            if not path or path[0] != src_node or path[-1] != dst_node:
                return False

        return True


# ---------------------------- A* ORCHESTRATOR (Sage) ----------------------------
def run_astar(G, slices, node_capacity_base, link_capacity_base, csv_path=None):
    """
    A* using Sage weighted shortest paths (latency as edge label).
    Accepts either a Sage Graph or a NetworkX undirected graph (converted once).

    Entry-aware behavior:
    - If a slice is given as (vnf_chain, vl_chain, entry),
      a virtual ingress VL ENTRY_VNF_ID -> first_vnf is added.
    - ENTRY_VNF_ID is pre-placed on the topology node 'entry'.
    """
    astar_results = []

    sG = _to_sage_graph_if_needed(G)
    _build_edge_latency_from_sage(sG)

    node_capacity_global = dict(node_capacity_base)

    link_capacity_global = {}
    for (u, v), cap in link_capacity_base.items():
        ek = _edge_key(u, v)
        if ek in link_capacity_global:
            if abs(link_capacity_global[ek] - float(cap)) > 1e-9:
                raise ValueError(f"Conflicting capacities for edge {ek}")
        else:
            link_capacity_global[ek] = float(cap)

    # ---------- Heuristic ----------
    def heuristic_for_slice(state, vl_list_norm):
        h = 0.0
        for (v_from, v_to, _bw) in vl_list_norm:
            if (v_from, v_to) in state.routed_vls:
                continue
            src_node = state.placed_vnfs.get(v_from)
            dst_node = state.placed_vnfs.get(v_to)
            if src_node is None or dst_node is None or src_node == dst_node:
                continue
            try:
                h += float(sG.shortest_path_length(src_node, dst_node, by_weight=True))
            except Exception:
                h += 10_000.0
        return float(h)

    # ---------- Expand ----------
    def expand_state(state, vnf_chain_ids, vnf_cpu_by_id, vnf_slice_by_id, vl_list_norm):
        expansions = []

        next_vnf_id = None
        for vid in vnf_chain_ids:
            if vid not in state.placed_vnfs:
                next_vnf_id = vid
                break
        if next_vnf_id is None:
            return expansions

        cpu_need = float(vnf_cpu_by_id[next_vnf_id])
        slice_id = vnf_slice_by_id[next_vnf_id]

        sorted_nodes = sorted(
            sG.vertices(),
            key=lambda n: state.node_capacity.get(n, 0.0),
            reverse=True,
        )

        for node in sorted_nodes:
            avail_cpu = state.node_capacity.get(node, 0.0)
            if avail_cpu < cpu_need:
                continue

            used = state.node_used_slices.get(node)
            if used is not None and slice_id in used:
                continue

            new_placed = state.placed_vnfs.copy()
            new_routed = state.routed_vls.copy()
            new_node_capacity = state.node_capacity.copy()
            new_link_capacity = state.link_capacity.copy()
            new_node_used_slices = {k: v.copy() for k, v in state.node_used_slices.items()}

            new_cost = state.g_cost

            new_node_capacity[node] = avail_cpu - cpu_need
            new_placed[next_vnf_id] = node
            new_node_used_slices.setdefault(node, set()).add(slice_id)

            routing_ok = True

            for (v_from, v_to, bw) in vl_list_norm:
                if (v_from, v_to) in new_routed:
                    continue

                src_node = new_placed.get(v_from)
                dst_node = new_placed.get(v_to)
                if src_node is None or dst_node is None:
                    continue
                if src_node == dst_node:
                    continue

                path, lat = shortest_path_with_capacity_sage(
                    sG, src_node, dst_node, new_link_capacity, bw
                )
                if path is None:
                    routing_ok = False
                    break

                for u, v in zip(path[:-1], path[1:]):
                    ek = _edge_key(u, v)
                    new_link_capacity[ek] -= bw
                    if new_link_capacity[ek] < -1e-9:
                        routing_ok = False
                        break
                if not routing_ok:
                    break

                new_routed[(v_from, v_to)] = path
                new_cost += lat

            if routing_ok:
                expansions.append(
                    AStarState(
                        placed_vnfs=new_placed,
                        routed_vls=new_routed,
                        g_cost=new_cost,
                        node_capacity=new_node_capacity,
                        link_capacity=new_link_capacity,
                        node_used_slices=new_node_used_slices,
                    )
                )

        return expansions

    # ---------- Solve one slice ----------
    def solve_one_slice(vnf_chain, vl_chain, entry):
        if entry is not None and entry not in set(sG.vertices()):
            raise ValueError(f"Entry node {entry!r} is not present in the topology.")

        vnf_chain_ids = [_as_vnf_id(v["id"]) for v in vnf_chain]
        real_vnf_ids = set(vnf_chain_ids)

        vnf_cpu_by_id = {_as_vnf_id(v["id"]): float(v["cpu"]) for v in vnf_chain}
        vnf_slice_by_id = {_as_vnf_id(v["id"]): v["slice"] for v in vnf_chain}

        effective_vl_chain = _augment_vl_chain_with_entry(vnf_chain, vl_chain, entry)

        vl_list_norm = []
        for vl in effective_vl_chain:
            k = _vl_key(vl)
            vl_list_norm.append((k[0], k[1], float(vl["bandwidth"])))

        init_placed = {}
        if entry is not None:
            init_placed[ENTRY_VNF_ID] = entry

        init_state = AStarState(
            placed_vnfs=init_placed,
            routed_vls={},
            g_cost=0.0,
            node_capacity=node_capacity_global.copy(),
            link_capacity=link_capacity_global.copy(),
            node_used_slices={},
        )

        heap = []
        counter = 0
        heappush(heap, (0.0, counter, init_state))
        best_seen = {}

        while heap:
            _f, _c, state = heappop(heap)

            placed_real = sum(1 for vid in real_vnf_ids if vid in state.placed_vnfs)
            key = (placed_real, len(state.routed_vls))
            prev = best_seen.get(key)
            if prev is not None and state.g_cost >= prev:
                continue
            best_seen[key] = state.g_cost

            if state.is_goal(real_vnf_ids, vl_list_norm):
                return state, effective_vl_chain

            for child in expand_state(state, vnf_chain_ids, vnf_cpu_by_id, vnf_slice_by_id, vl_list_norm):
                h = heuristic_for_slice(child, vl_list_norm)
                counter += 1
                heappush(heap, (child.g_cost + h, counter, child))

        return None, effective_vl_chain

    # ---------- Main loop ----------
    effective_vl_chains = []

    for slice_data in slices:
        if len(slice_data) == 2:
            vnf_chain, vl_chain = slice_data
            entry = None
        elif len(slice_data) == 3:
            vnf_chain, vl_chain, entry = slice_data
        else:
            raise ValueError(f"Unexpected slice format: {slice_data}")

        result_state, effective_vl_chain = solve_one_slice(vnf_chain, vl_chain, entry)
        astar_results.append(result_state)
        effective_vl_chains.append(effective_vl_chain)

        if result_state is None:
            continue

        cpu_by_id = {_as_vnf_id(v["id"]): float(v["cpu"]) for v in vnf_chain}
        for vnf_id, node in result_state.placed_vnfs.items():
            if vnf_id == ENTRY_VNF_ID:
                continue
            node_capacity_global[node] -= float(cpu_by_id[vnf_id])

        bw_by_key = {_vl_key(vl): float(vl["bandwidth"]) for vl in effective_vl_chain}
        for (v_from, v_to), path in result_state.routed_vls.items():
            bw = bw_by_key.get((v_from, v_to))
            if bw is None:
                raise KeyError(f"Bandwidth not found for VL ({v_from},{v_to})")

            src_node = result_state.placed_vnfs[v_from]
            dst_node = result_state.placed_vnfs[v_to]
            _assert_path_endpoints(path, src_node, dst_node)

            for u, v in zip(path[:-1], path[1:]):
                ek = _edge_key(u, v)
                link_capacity_global[ek] -= bw
                if link_capacity_global[ek] < -1e-9:
                    raise ValueError(f"Link capacity underflow on {ek}")

    df_results = pd.DataFrame(
        [
            {
                "slice": i,
                "accepted": (r is not None),
                "g_cost": (r.g_cost if r else None),
            }
            for i, r in enumerate(astar_results, start=1)
        ]
    )

    if csv_path:
        df_results.to_csv(csv_path, index=False)

    return df_results, astar_results