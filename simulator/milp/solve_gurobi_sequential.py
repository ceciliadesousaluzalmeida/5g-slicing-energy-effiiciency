from dataclasses import dataclass
import gurobipy as gp
from gurobipy import GRB


# ==============================
# Energy / latency weight params
# ==============================

NODE_ENERGY_WEIGHT = 1.0   # weight for normalized node CPU usage
LINK_ENERGY_WEIGHT = 1.0   # weight for normalized link bandwidth usage
LATENCY_PENALTY   = 1e3    # penalty for latency slack xi


@dataclass
class GurobiSolveResult:
    """Simple container for a single MILP solve result."""
    status_code: int
    status_str: str
    objective: float
    values: dict  # maps tuple keys -> float values


def _get_slice_vl_pairs(instance, s):
    """
    Helper: return the list of logical VL pairs (i, j) for slice s
    based on instance.V_of_s and instance.BW_sij keys.
    """
    vl_pairs = []
    V_s = instance.V_of_s[s]
    for i in V_s:
        for j in V_s:
            if (s, i, j) in instance.BW_sij:
                vl_pairs.append((i, j))
    return vl_pairs


def build_single_slice_model(
    instance,
    s,
    remaining_cpu,
    remaining_bw,
    msg: bool = False,
    time_limit=None,
) -> GurobiSolveResult:
    """
    Build and solve a MILP for a single slice s with remaining capacities.

    Compatible with instances created by `create_instance(G, slices)`:
      - instance.N        : list of physical nodes
      - instance.E        : list of physical edges (u, v)
      - instance.S        : list of slice indices
      - instance.V_of_s   : dict s -> list of global VNF ids
      - instance.CPU_i    : dict vnf_id -> cpu demand
      - instance.CPU_cap  : dict node -> cpu capacity
      - instance.BW_cap   : dict (u,v) -> link capacity
      - instance.BW_sij   : dict (s,i,j) -> bandwidth demand for VL i->j
      - instance.L_sij    : dict (s,i,j) -> latency SLA for VL i->j
      - instance.lat_e    : dict (u,v) -> link latency

    Current version:
      - Placement with hard anti-colocation.
      - Flow per logical VL (i->j), subject to BW capacities.
      - Latency per VL with a single slack xi per slice.
      - Objective: proxy of energy (CPU + BW) + latency penalty * xi.
    """

    model = gp.Model(f"single_slice_{s}")
    if not msg:
        model.Params.OutputFlag = 0
    if time_limit is not None:
        model.Params.TimeLimit = time_limit

    # Sets from instance
    N = list(instance.N)          # physical nodes
    E = list(instance.E)          # physical edges (u, v)
    V_s = list(instance.V_of_s[s])  # VNFs of slice s
    vl_pairs = _get_slice_vl_pairs(instance, s)  # logical VLs (i, j)

    # ==========================
    # Variables
    # ==========================

    # x[i,n] ∈ {0,1}: placement of VNF i (global id) on node n
    x = model.addVars(V_s, N, vtype=GRB.BINARY, name="x")

    # f[i,j,u,v] ≥ 0: flow for logical VL (i->j) on physical edge (u,v)
    # Only define for VL pairs that exist in instance.BW_sij
    f = model.addVars(
        [(i, j, u, v) for (i, j) in vl_pairs for (u, v) in E],
        vtype=GRB.CONTINUOUS,
        lb=0.0,
        name="f"
    )

    # Single latency slack for the whole slice (soft SLA)
    xi = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"xi_s{s}")

    # ==========================
    # Constraints
    # ==========================

    # 1) Each VNF i must be placed on exactly one node
    for i in V_s:
        model.addConstr(
            gp.quicksum(x[i, n] for n in N) == 1,
            name=f"place_s{s}_i{i}",
        )

    # 2) Hard anti-colocation: at most one VNF of this slice per node
    for n in N:
        model.addConstr(
            gp.quicksum(x[i, n] for i in V_s) <= 1,
            name=f"anti_coloc_s{s}_n{n}",
        )

    # 3) CPU capacity with remaining_cpu
    #    CPU demand per VNF is stored in instance.CPU_i[vnf_id]
    for n in N:
        cpu_usage = gp.quicksum(
            instance.CPU_i[i] * x[i, n] for i in V_s
        )
        model.addConstr(
            cpu_usage <= remaining_cpu[n],
            name=f"cpu_cap_s{s}_n{n}",
        )

    # 4) Bandwidth capacity per physical edge with remaining_bw
    for (u, v) in E:
        bw_usage = gp.quicksum(
            f[i, j, u, v] for (i, j) in vl_pairs
        )
        model.addConstr(
            bw_usage <= remaining_bw[(u, v)],
            name=f"bw_cap_s{s}_{u}_{v}",
        )

    # 5) Flow conservation per logical VL (i->j)
    #    outflow - inflow = BW_sij * (x[i,n] - x[j,n])
    for (i, j) in vl_pairs:
        bw_ij = instance.BW_sij[(s, i, j)]
        for n in N:
            outflow = gp.quicksum(
                f[i, j, n, v] for (u, v) in E if u == n
            )
            inflow = gp.quicksum(
                f[i, j, u, n] for (u, v) in E if v == n
            )
            model.addConstr(
                outflow - inflow == bw_ij * (x[i, n] - x[j, n]),
                name=f"flow_cons_s{s}_ij{i}_{j}_n{n}",
            )

    # 6) Latency constraints with slack xi
    #    For each VL (i->j):
    #       (1 / BW_sij) * sum_e lat_e[e] * f_ij[e] <= L_sij + xi
    for (i, j) in vl_pairs:
        key = (s, i, j)
        bw_ij = instance.BW_sij[key]
        if key in instance.L_sij:
            L_ij = instance.L_sij[key]
        else:
            # If no specific latency bound is given, skip the constraint
            continue

        lat_expr = (1.0 / bw_ij) * gp.quicksum(
            instance.lat_e[(u, v)] * f[i, j, u, v] for (u, v) in E
        )

        model.addConstr(
            lat_expr <= L_ij + xi,
            name=f"lat_s{s}_ij{i}_{j}",
        )

    # ==========================
    # Energy-like objective
    # ==========================

    # Node energy proxy: normalized CPU usage
    node_energy_terms = []
    for n in N:
        cpu_usage_n = gp.quicksum(instance.CPU_i[i] * x[i, n] for i in V_s)
        cap_n = instance.CPU_cap[n]
        if cap_n > 0:
            node_energy_terms.append(cpu_usage_n / cap_n)

    E_nodes = gp.quicksum(node_energy_terms)

    # Link energy proxy: normalized bandwidth usage
    link_energy_terms = []
    for (u, v) in E:
        bw_usage_uv = gp.quicksum(
            f[i, j, u, v] for (i, j) in vl_pairs
        )
        cap_uv = instance.BW_cap[(u, v)]
        if cap_uv > 0:
            link_energy_terms.append(bw_usage_uv / cap_uv)

    E_links = gp.quicksum(link_energy_terms)

    energy_expr = NODE_ENERGY_WEIGHT * E_nodes + LINK_ENERGY_WEIGHT * E_links

    # Final objective: energy + latency penalty * xi
    model.setObjective(energy_expr + LATENCY_PENALTY * xi, GRB.MINIMIZE)

    # ==========================
    # Solve
    # ==========================

    model.optimize()

    status = model.Status
    status_str = model.Status

    if status != GRB.OPTIMAL:
        return GurobiSolveResult(
            status_code=status,
            status_str=status_str,
            objective=float("inf"),
            values={},
        )

    values = {}

    # Store placement x
    for i in V_s:
        for n in N:
            values[("x", s, i, n)] = x[i, n].X

    # Store flows f
    for (i, j) in vl_pairs:
        for (u, v) in E:
            key = (i, j, u, v)
            if key in f:
                values[("f", s, i, j, u, v)] = f[key].X

    # Store slack xi if needed
    values[("xi", s)] = xi.X

    obj = model.ObjVal

    return GurobiSolveResult(
        status_code=status,
        status_str=status_str,
        objective=obj,
        values=values,
    )


def update_remaining_capacities(instance, s, res, remaining_cpu, remaining_bw):
    """
    Update remaining CPU and BW from the solution of slice s.
    """

    # Update CPU
    for n in instance.N:
        cpu_used = 0.0
        for i in instance.V_of_s[s]:
            key = ("x", s, i, n)
            if key in res.values:
                cpu_used += instance.CPU_i[i] * res.values[key]
        remaining_cpu[n] -= cpu_used

    # Update BW
    vl_pairs = _get_slice_vl_pairs(instance, s)
    for (u, v) in instance.E:
        bw_used = 0.0
        for (i, j) in vl_pairs:
            key = ("f", s, i, j, u, v)
            if key in res.values:
                bw_used += res.values[key]
        remaining_bw[(u, v)] -= bw_used


def allocate_slices_sequential(instance, slice_order=None, msg: bool = False, time_limit=None):
    """
    Sequentially allocate slices with a per-slice MILP.
    Stop at the first infeasible slice.

    Compatible with instances created by `create_instance(G, slices)`.

    - Respects CPU and BW capacities.
    - Enforces hard anti-colocation per slice.
    - Accounts for latency (soft via slack xi) and an energy-like objective.
    """

    if slice_order is None:
        slice_order = list(instance.S)

    # Initialize remaining capacities based on instance.CPU_cap and BW_cap
    remaining_cpu = {n: instance.CPU_cap[n] for n in instance.N}
    remaining_bw = {(u, v): instance.BW_cap[(u, v)] for (u, v) in instance.E}

    accepted_slices = []
    slice_results = {}
    first_infeasible_slice = None

    for s in slice_order:
        if msg:
            print(f"[INFO] Solving MILP for slice {s} with remaining capacities...")

        res = build_single_slice_model(
            instance,
            s,
            remaining_cpu,
            remaining_bw,
            msg=msg,
            time_limit=time_limit,
        )

        if res.status_code != GRB.OPTIMAL:
            # First infeasible slice, stop here
            first_infeasible_slice = s
            if msg:
                print(f"[INFO] Slice {s} infeasible, stopping sequential allocation.")
            break

        # Slice s accepted
        accepted_slices.append(s)
        slice_results[s] = res

        # Update remaining capacities based on this solution
        update_remaining_capacities(instance, s, res, remaining_cpu, remaining_bw)

    return {
        "accepted_slices": accepted_slices,
        "slice_results": slice_results,
        "first_infeasible_slice": first_infeasible_slice,
        "remaining_cpu": remaining_cpu,
        "remaining_bw": remaining_bw,
    }


def solve_gurobi_sequential(instance, slice_order=None, msg: bool = False, time_limit=None):
    """
    Public entry point used by notebooks and scripts.

    This wrapper keeps the same naming style as your other solvers and
    simply calls allocate_slices_sequential.
    """
    return allocate_slices_sequential(
        instance,
        slice_order=slice_order,
        msg=msg,
        time_limit=time_limit,
    )
