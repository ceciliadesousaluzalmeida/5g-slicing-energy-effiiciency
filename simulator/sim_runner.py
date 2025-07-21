from run_abo_full_batch import run_abo_full_batch
from topology import generate_complete_graph
from generate_slices import generate_random_slices, vnf_profiles 

import networkx as nx

def sim_runner():
      

    # Generate graph
    G = generate_complete_graph(32)

    # Generate slices
    slices = generate_random_slices(G, vnf_profiles)

    # Define resources
    node_capacity = {n: 10 for n in G.nodes}
    link_latency = {(u, v): G[u][v]['latency'] for u, v in G.edges}
    link_capacity = {(u, v): G[u][v]['capacity'] for u, v in G.edges}

    # Run ABO and export results
    df, results = run_abo_full_batch(
        G,
        slices,
        node_capacity,
        link_latency,
        link_capacity,
        csv_path="results.csv"
    )

    # Stop energy tracking
   

    print(f"Execution completed. {df['accepted'].sum()} slices accepted.")
    print("Results saved to results.csv")
if __name__ == "__main__":
    sim_runner()
