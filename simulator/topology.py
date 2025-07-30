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

def topologie_finlande():
    G = nx.Graph()

    for node in range(1, 13):
        G.add_node(node, cpu=10, memory=10)

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
        latency_ms = round(distance_km * 0.5, 3)
        capacity = random.randint(50, 150)
        G.add_edge(u, v, latency=latency_ms, bandwidth=capacity, capacity=capacity)

    return G
