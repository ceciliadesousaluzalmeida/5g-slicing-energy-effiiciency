from queue import PriorityQueue
import networkx as nx


G = nx.DiGraph()

def heuristic(u, v):
    # Simple heuristic: assumes the cost is proportional to the minimum latency among successors
    return min(G[u][n]['latency'] for n in G.successors(u)) if list(G.successors(u)) else 0

# Define cost function based on latency
def cost(u, v):
    return G[u][v]['latency']

# A* algorithm considering link capacity and latency constraints
def astar_with_sla(graph, source, target, sla_min_capacity, sla_latency_max):
    from queue import PriorityQueue
    visited = set()
    queue = PriorityQueue()
    queue.put((0, 0, [source]))  # (priority, accumulated_latency, path)

    while not queue.empty():
        _, total_latency, path = queue.get()
        current = path[-1]

        # If the destination is reached and latency is within SLA, return the path
        if current == target and total_latency <= sla_latency_max:
            return path, total_latency

        if current in visited:
            continue
        visited.add(current)

        for neighbor in graph.successors(current):
            edge = graph[current][neighbor]
            # Only consider links that satisfy the minimum capacity requirement
            if edge['capacity'] >= sla_min_capacity:
                new_latency = total_latency + edge['latency']
                # Only continue exploring if accumulated latency is within SLA
                if new_latency <= sla_latency_max:
                    priority = new_latency + heuristic(neighbor, target)
                    queue.put((priority, new_latency, path + [neighbor]))

    return None, None  # No valid path found