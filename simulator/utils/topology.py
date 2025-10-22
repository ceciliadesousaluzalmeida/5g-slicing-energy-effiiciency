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

def topology_brazil(path="utils/topologies/topology_brazil.gml"):
    """
    Load the RNP Brazil topology (Topology Zoo) and return a simplified, enriched NetworkX graph.

    - Removes non-internal nodes (e.g., RedCLARA, Internet Comercial)
    - Converts edge attributes to bandwidth (Mbps) and latency (ms)
    - Assigns CPU capacities to nodes (scaled by node degree)
    """

    # --- Load the topology file ---
    G = nx.parse_gml(open(path, encoding="utf-8").read())

    # --- Keep only internal nodes ---
    internal_nodes = [n for n, data in G.nodes(data=True) if data.get("Internal", 1) == 1]
    G = G.subgraph(internal_nodes).copy()

    # --- Normalize edge attributes ---
    for (u, v, data) in G.edges(data=True):
        bw_raw = data.get("LinkSpeedRaw", 1e9)  # default 1 Gbps if missing
        data["bandwidth"] = bw_raw / 1e6        # convert to Mbps
        data["latency"] = 5.0                   # approximate latency (ms)

    # --- Assign CPU capacity based on node degree ---
    degrees = dict(G.degree())
    max_deg = max(degrees.values()) if degrees else 1
    for n in G.nodes:
        # Scale CPU between 50 and 200 units depending on degree
        degree_ratio = degrees[n] / max_deg
        G.nodes[n]["cpu"] = int(50 + 150 * degree_ratio)

    # --- Relabel nodes to integers ---
    mapping = {n: i for i, n in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)

    # --- Print summary ---
    print(f"Loaded RNP Brazil topology with {G.number_of_nodes()} nodes and {G.number_of_edges()} links.")
    print("Average node CPU:", round(sum(nx.get_node_attributes(G, 'cpu').values()) / G.number_of_nodes(), 2))
    print("Average link bandwidth (Mbps):", round(sum(nx.get_edge_attributes(G, 'bandwidth').values()) / G.number_of_edges(), 2))

    return G

