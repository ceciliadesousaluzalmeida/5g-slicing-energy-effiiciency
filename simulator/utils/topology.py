import networkx as nx
import matplotlib.pyplot as plt
import math
import random

def generate_complete_graph(num_nodes, seed=42):
    import networkx as nx, random
    G = nx.complete_graph(num_nodes, create_using=nx.DiGraph)
    random.seed(seed)
    for u, v in G.edges():
        G[u][v]['latency'] = random.randint(1, 10)
        G[u][v]['capacity'] = random.randint(50, 150)
    return G

def draw_graph(G): 
    pos = nx.spring_layout(G, seed=42)

    node_labels = {
        node: f"{node}\n{G.nodes[node]['cpu']} CPU"
        for node in G.nodes
    }

    plt.figure(figsize=(10, 8))
    nx.draw(G, pos,labels=node_labels, node_color="lightgreen", node_size=800, edge_color='gray', font_weight='bold', with_labels=True, arrows=False)

    edge_labels = {
        (u, v): f"{G[u][v]['latency']:.3f} ms\n{G[u][v]['bandwidth']} Mbps"
        for u, v in G.edges
    }

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='blue')

    plt.title("Complete Graph")
    plt.axis("off")
    plt.show()


import networkx as nx
import random
import networkx as nx

def topologie_finlande():
    G = nx.Graph()

    # Heterogeneous CPU capacities
    core_nodes = {1, 6, 11}
    for node in range(1, 13):
        if node in core_nodes:
            G.add_node(node, cpu=64, memory=64)
        else:
            G.add_node(node, cpu=16, memory=16)

    edges = [
        (1, 2, 37),
        (1, 3, 183),
        (2, 4, 87),
        (2, 6, 110),
        (3, 4, 62),
        (3, 11, 69),
        (3, 12, 73),
        (4, 5, 84),
        (5, 6, 83),
        (5, 8, 44),
        (6, 7, 133),
        (7, 8, 50),
        (7, 9, 18),
        (7, 10, 16),
        (8, 9, 28),
        (8, 11, 20),
        (9, 10, 74),
        (10, 11, 54),
        (10, 12, 34)
    ]

    for u, v, distance_km in edges:
        latency_ms = max(1, round(distance_km * 0.005, 3))  # realistic optical latency
        # More generous bandwidth capacity
        if distance_km < 50:
            capacity = 1000  # ~100 Gbps
        elif distance_km < 150:
            capacity = 500   # ~50 Gbps
        else:
            capacity = 250   # ~25 Gbps
        G.add_edge(u, v, latency=latency_ms, bandwidth=capacity, capacity=capacity)

    return G



import networkx as nx

def topology_bayern():
    
    G = nx.Graph()

    # Add nodes with default CPU capacity
    for n in range(1, 7):
        G.add_node(n, cpu=10)  # you can adjust CPU capacity here

    # Add edges with default bandwidth and latency
    edges = [
        (1, 2), (1, 3),
        (2, 3), (2, 4),
        (3, 4), (3, 5),
        (4, 5), (4, 6),
        (5, 6)
    ]

    for (u, v) in edges:
        G.add_edge(u, v, bandwidth=100, latency=1)

    return G




import networkx as nx

import math
import random
import networkx as nx

import math
import random
import networkx as nx

def topology_brazil(
    path="utils/topologies/topology_brazil.gml",
    seed: int = 42,
    bw_scale: float = 1.15,
    fiber_km_per_ms: float = 200.0,
    jitter_ms: float = 0.05,
    bw_min_max=(300, 20000),
    noise_std_bw: float = 0.08
):
    """
    RNP Brazil topology (Topology Zoo) with realistic capacities and discrete CPU levels.
    - 27 nodes total
      → 2 nodes with 128 CPU
      → 4 nodes with 64 CPU
      → 8 nodes with 32 CPU
      → remaining with 16 CPU
    """

    random.seed(seed)

    # --- Load raw graph ---
    G_raw = nx.parse_gml(open(path, encoding="utf-8").read())

    # Keep only internal nodes if present
    internal_nodes = [n for n, d in G_raw.nodes(data=True) if d.get("Internal", 1) == 1]
    G_raw = G_raw.subgraph(internal_nodes).copy()

    # --- Coordinates if available ---
    def get_xy(n, d):
        lon = d.get("Longitude", d.get("longitude", d.get("long", None)))
        lat = d.get("Latitude", d.get("latitude", d.get("lat", None)))
        try:
            return float(lon), float(lat)
        except (TypeError, ValueError):
            return None
    coords = {n: get_xy(n, d) for n, d in G_raw.nodes(data=True)}
    has_geo = all(v is not None for v in coords.values())

    def haversine_km(p1, p2):
        lon1, lat1 = map(math.radians, p1)
        lon2, lat2 = map(math.radians, p2)
        dlon, dlat = lon2 - lon1, lat2 - lat1
        a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
        return 6371.0 * (2 * math.asin(math.sqrt(a)))

    if not has_geo:
        pos = nx.spring_layout(G_raw, seed=seed)

    # --- Edge betweenness for link capacity tiers ---
    ebc = nx.edge_betweenness_centrality(G_raw, normalized=True)

    # --- Create new graph ---
    G = nx.Graph()
    G.add_nodes_from(G_raw.nodes(data=True))

    # --- Assign discrete CPU levels based on closeness centrality ---
    clos = nx.closeness_centrality(G_raw)
    sorted_nodes = sorted(clos, key=clos.get, reverse=True)  # top central first
    n_nodes = len(sorted_nodes)

    cpu_values = [128]*2 + [64]*4 + [32]*8
    remaining = n_nodes - len(cpu_values)
    cpu_values += [16]*remaining
    if len(cpu_values) != n_nodes:
        print("[WARN] CPU vector length mismatch; adjusting to node count.")
        cpu_values = (cpu_values + [16]*n_nodes)[:n_nodes]

    for n, cpu in zip(sorted_nodes, cpu_values):
        G.nodes[n]["cpu"] = cpu

    # --- Assign edges: bandwidth + latency ---
    bw_lo, bw_hi = bw_min_max
    for u, v, data in G_raw.edges(data=True):
        bw_raw_bps = data.get("LinkSpeedRaw", None)
        bw_mbps_from_raw = None
        if bw_raw_bps is not None:
            try:
                bw_mbps_from_raw = float(bw_raw_bps) / 1e6
            except (TypeError, ValueError):
                bw_mbps_from_raw = None

        tier = ebc.get((u, v), ebc.get((v, u), 0.0))
        bw_from_tier = bw_lo + (bw_hi - bw_lo) * (0.2 + 0.8 * (tier ** 0.6))
        bw = bw_mbps_from_raw if bw_mbps_from_raw else bw_from_tier
        bw *= (1.0 + random.gauss(0.0, noise_std_bw))
        bw *= bw_scale
        bw = max(bw_lo * 0.8, min(bw, bw_hi * 1.05))

        if has_geo:
            d_km = haversine_km(coords[u], coords[v])
        else:
            dx = pos[u][0] - pos[v][0]
            dy = pos[u][1] - pos[v][1]
            d_km = math.hypot(dx, dy) * 1000
        prop_ms = max(0.05, d_km / fiber_km_per_ms)
        lat_ms = prop_ms + max(0.0, random.gauss(0.0, jitter_ms))

        G.add_edge(u, v, bandwidth=float(bw), latency=float(lat_ms), distance_km=float(d_km))

    # --- Relabel to integers for consistency ---
    mapping = {n: i for i, n in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)

    # --- Summary ---
    avg_cpu = sum(nx.get_node_attributes(G, "cpu").values()) / G.number_of_nodes()
    avg_bw  = sum(nx.get_edge_attributes(G, "bandwidth").values()) / G.number_of_edges()
    avg_lat = sum(nx.get_edge_attributes(G, "latency").values()) / G.number_of_edges()
    print(f"Loaded RNP Brazil topology with {G.number_of_nodes()} nodes and {G.number_of_edges()} links.")
    print(f"CPU distribution: 2×128, 4×64, 8×32, {G.number_of_nodes()-14}×16")
    print(f"Average CPU: {avg_cpu:.1f} | Avg BW: {avg_bw:.0f} Mbps | Avg latency: {avg_lat:.2f} ms")

    return G
