# tests.py
# All comments in English

from formulation import MILPInstance
from simulator.milp.solver_pulp import solve_lp

def build_tiny_instance():
    # 2 nodes, 1 link, 1 slice with 1 VNF
    N = ["n1", "n2"]
    E = [("n1", "n2")]
    S = ["s1"]
    V_of_s = {"s1": ["v1"]}
    CPU_i = {"v1": 2}
    CPUcap = {"n1": 4, "n2": 4}
    BW_s = {"s1": 3}
    BWcap = {("n1", "n2"): 5}
    Lat = {("n1", "n2"): 10}
    Ls = {"s1": 20}

    return MILPInstance(N, E, S, V_of_s, CPU_i, CPUcap, BW_s, BWcap, Lat, Ls)


if __name__ == "__main__":
    instance = build_tiny_instance()
    res, var_index = solve_lp(instance)
    print("Status:", res.message)
    print("Objective value:", res.fun)
    print("Solution vector:", res.x)
