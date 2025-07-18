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
    pos = nx.circular_layout(G)

    plt.figure(figsize=(10, 10))
    nx.draw(G, pos, node_size=300, node_color="lightgreen", with_labels=True, arrows=False)
    plt.title("Complete Graph")
    plt.axis("off")
    plt.show()

