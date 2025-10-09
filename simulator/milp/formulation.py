
from collections import defaultdict

class MILPInstance:
    """
    Data container for the energy-aware slice placement MILP.
    Compatible with solver_gurobi.py.
    """
    def __init__(self, G, slices):
        # --- Physical topology ---
        self.N = list(G.nodes())
        self.E = list(G.edges())  # assumed undirected edges
        # --- Slice definitions ---
        self.S = list(range(len(slices)))
        self.V_of_s = {s: [vnf["id"] for vnf in slices[s][0]] for s in self.S}

        # --- Parameters per VNF and slice ---
        self.CPU_i = {vnf["id"]: vnf["cpu"] for s in slices for vnf in s[0]}
        self.BW_s = {s: min(vnf["throughput"] for vnf in slices[s][0]) for s in self.S}
        self.L_s = {s: max(vnf["latency"] for vnf in slices[s][0]) for s in self.S}

        # --- Node and link capacities ---
        self.CPU_cap = {n: G.nodes[n]["cpu"] for n in self.N}
        self.BW_cap = {e: G.edges[e]["bandwidth"] for e in self.E}
        self.lat_e = {e: G.edges[e]["latency"] for e in self.E}

        # --- Optional energy coefficients (defaults if not in graph) ---
        self.alpha = {n: G.nodes[n].get("alpha", 1.0) for n in self.N}   # dynamic node coeff
        self.beta = {n: G.nodes[n].get("beta", 1.0) for n in self.N}     # static node coeff
        self.gamma = {e: G.edges[e].get("gamma", 1.0) for e in self.E}   # dynamic link coeff
        self.beta_link = {e: G.edges[e].get("beta_link", 1.0) for e in self.E}  # static link coeff

    def __repr__(self):
        return (f"<MILPInstance | nodes={len(self.N)} edges={len(self.E)} "
                f"slices={len(self.S)} vnfs={len(self.CPU_i)}>")

