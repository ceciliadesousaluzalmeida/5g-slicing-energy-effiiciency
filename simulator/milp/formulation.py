from collections import defaultdict

class MILPInstance:
    def __init__(self, G, slices):
        self.N = list(G.nodes())
        self.E = list(G.edges())  
        self.S = list(range(len(slices)))
        self.V_of_s = {s: [vnf["id"] for vnf in slices[s][0]] for s in self.S}
        self.CPU_i = {vnf["id"]: vnf["cpu"] for s in slices for vnf in s[0]}
        self.BW_s = {s: min(vnf["throughput"] for vnf in slices[s][0]) for s in self.S}
        self.L_s = {s: max(vnf["latency"] for vnf in slices[s][0]) for s in self.S}

        self.CPU_cap = {n: G.nodes[n]["cpu"] for n in self.N}
        self.BW_cap = {e: G.edges[e]["bandwidth"] for e in self.E}
        self.lat_e = {e: G.edges[e]["latency"] for e in self.E}

    def build_lp(self):
        c = []        
        A_eq, b_eq = [], []
        A_ub, b_ub = [], []
        bounds = []
        var_index = {}

        idx = 0

    
        for i in self.CPU_i:
            for n in self.N:
                var_index[("v", i, n)] = idx
                c.append(0)   
                bounds.append((0, 1))
                idx += 1

 
        for n in self.N:
            var_index[("u", n)] = idx
            c.append(1)   
            bounds.append((0, 1))
            idx += 1

            var_index[("z", n)] = idx
            c.append(1)  
            bounds.append((0, 1))
            idx += 1

       
        for s in self.S:
            vnf_ids = self.V_of_s[s]
            for q in range(len(vnf_ids) - 1):
                i, j = vnf_ids[q], vnf_ids[q + 1]
                for e in self.E:
                    var_index[("f", e, s, (i, j))] = idx
                    c.append(0)  
                    bounds.append((0, 1))
                    idx += 1

        for e in self.E:
            var_index[("rho", e)] = idx
            c.append(1)   
            bounds.append((0, 1))
            idx += 1

            var_index[("w", e)] = idx
            c.append(1)  
            bounds.append((0, 1))
            idx += 1

      
        for n in self.N:
            row = [0] * idx
            for i in self.CPU_i:
                row[var_index[("v", i, n)]] = self.CPU_i[i]
            row[var_index[("u", n)]] = -self.CPU_cap[n]
            A_ub.append(row)
            b_ub.append(0)

            # u_n <= z_n
            row = [0] * idx
            row[var_index[("u", n)]] = 1
            row[var_index[("z", n)]] = -1
            A_ub.append(row)
            b_ub.append(0)

        for i in self.CPU_i:
            row = [0] * idx
            for n in self.N:
                row[var_index[("v", i, n)]] = 1
            A_eq.append(row)
            b_eq.append(1)

     
        for s in self.S:
            vnf_ids = self.V_of_s[s]
            for q in range(len(vnf_ids) - 1):
                i, j = vnf_ids[q], vnf_ids[q + 1]
                for n in self.N:
                    row = [0] * idx
                    for e in self.E:
                        u, v = e
                        if u == n:
                            row[var_index[("f", e, s, (i, j))]] += 1
                        if v == n:
                            row[var_index[("f", e, s, (i, j))]] -= 1
                    row[var_index[("v", i, n)]] += 1
                    row[var_index[("v", j, n)]] -= 1
                    A_eq.append(row)
                    b_eq.append(0)

    
        for e in self.E:
            row = [0] * idx
            for s in self.S:
                vnf_ids = self.V_of_s[s]
                for q in range(len(vnf_ids) - 1):
                    i, j = vnf_ids[q], vnf_ids[q + 1]
                    row[var_index[("f", e, s, (i, j))]] = self.BW_s[s]
            row[var_index[("rho", e)]] = -self.BW_cap[e]
            A_ub.append(row)
            b_ub.append(0)

            # rho_e <= w_e
            row = [0] * idx
            row[var_index[("rho", e)]] = 1
            row[var_index[("w", e)]] = -1
            A_ub.append(row)
            b_ub.append(0)

        
        for s in self.S:
            vnf_ids = self.V_of_s[s]
            for q in range(len(vnf_ids) - 1):
                i, j = vnf_ids[q], vnf_ids[q + 1]
                row = [0] * idx
                for e in self.E:
                    row[var_index[("f", e, s, (i, j))]] = self.lat_e[e]
                A_ub.append(row)
                b_ub.append(self.L_s[s])

        return c, A_ub, b_ub, A_eq, b_eq, bounds, var_index
