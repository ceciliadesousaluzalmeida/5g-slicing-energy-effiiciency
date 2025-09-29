class MILPInstance:
    """
    Defines sets and parameters of the MILP instance.
    """
    def __init__(self, N, E, S, V_of_s, CPU_i, CPUcap, BW_s, BWcap, Lat, Ls):
        self.N = N                  # list of nodes
        self.E = E                  # list of edges
        self.S = S                  # list of slices
        self.V_of_s = V_of_s        # dict: slice -> list of VNFs
        self.CPU_i = CPU_i          # dict: vnf -> cpu demand
        self.CPUcap = CPUcap        # dict: node -> cpu capacity
        self.BW_s = BW_s            # dict: slice -> bandwidth demand
        self.BWcap = BWcap          # dict: edge -> bandwidth capacity
        self.Lat = Lat              # dict: edge -> latency
        self.Ls = Ls                # dict: slice -> latency budget
