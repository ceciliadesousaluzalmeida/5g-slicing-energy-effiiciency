import cProfile
import pstats

from heuristics.a_star import run_astar  # ajuste
from heuristics.run_abo_full_batch import run_abo_full_batch
from heuristics.run_fabo_full_batch import run_fabo_full_batch
from heuristics.best_fit import run_best_fit
from heuristics.a_star_energy_aware import energy_aware_astar
from copy import deepcopy


from utils.generate_slices import generate_random_slices
from utils.topology import topologie_finlande

G = topologie_finlande()
node_capacity_base = {n: G.nodes[n]["cpu"] for n in G.nodes}
link_capacity_base = {(u, v): G[u][v]["bandwidth"] for u, v in G.edges}
link_capacity_base.update({(v, u): G[u][v]["bandwidth"] for u, v in G.edges})

link_latency = {(u, v): G[u][v]["latency"] for u, v in G.edges}
link_latency.update({(v, u): G[u][v]["latency"] for u, v in G.edges})



vnf_profiles = [
        {"cpu": 1, "throughput": 15, "latency": 30},
        {"cpu": 2, "throughput": 30, "latency": 20},
    ]

 # ajuste

def main():
    slices = generate_random_slices(G, num_slices=4, num_vnfs_per_slice=2, vnf_profiles=vnf_profiles)
    run_fabo_full_batch(G, slices, node_capacity_base, link_latency, link_capacity_base)
    #run_abo_full_batch(G, slices, node_capacity_base, link_latency, link_capacity_base)
    #energy_aware_astar(G, slices, node_capacity_base, link_capacity_base)
    #run_astar(G, slices, node_capacity_base, link_capacity_base)
    
 # ajuste

if __name__ == "__main__":
    cProfile.run("main()", "only_astar.pstats")
    p = pstats.Stats("only_astar.pstats")
    p.strip_dirs().sort_stats("cumulative").print_stats(40)
