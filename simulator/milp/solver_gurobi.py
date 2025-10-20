from dataclasses import dataclass
import gurobipy as gp
from gurobipy import GRB
import networkx as nx

def _incidence(n, e):
    """Incidence for undirected edge e=(u,v): +1 at u, -1 at v, 0 otherwise."""
    u, v = e
    if n == u: return 1.0
    if n == v: return -1.0
    return 0.0

@dataclass
class GurobiSolveResult:
    status_code: int
    status_str: str
    objective: float
    values: dict  # maps tuple-keys -> float values


def solve_gurobi(instance, msg=False, time_limit=None,
                 DEG_LIMIT=3,        # <= arestas por nó (por latência) para usar no MILP
                 MIP_GAP=0.05):      # gap alvo (5%)
    """
    MILP com suporte a ENTRY opcional e malha esparsa.
    Campos esperados em `instance`:
      - N, E, S
      - V_of_s[s]: lista ordenada de VNFs do slice s
      - CPU_i[i], CPU_cap[n]
      - BW_cap[e], lat_e[e]
      - BW_s[s], L_s[s]
      - (opcional) entry_of_s[s] ou entry_node

    Melhorias de escalabilidade:
      - Usa subgrafo E_use com no máx. DEG_LIMIT arestas de menor latência por nó
      - Variáveis de fluxo contínuas em [0,1] (f e f_entry)
      - Parâmetros do Gurobi ajustados
    """
    # -------------------------
    # Helpers ENTRY/EXIT
    # -------------------------
    def get_entry(s):
        if hasattr(instance, "entry_of_s"):
            return instance.entry_of_s.get(s, None)
        return getattr(instance, "entry_node", None)

    def get_exit(_s):
        # Caso queira voltar a usar EXIT, adicione aqui e crie f_exit como no entry.
        return getattr(instance, "exit_node", None) if hasattr(instance, "exit_node") else None

    # -------------------------
    # Conjuntos básicos
    # -------------------------
    N = list(instance.N)
    E_full = list(instance.E)
    S = list(instance.S)

    # -------------------------
    # Constrói grafo e faz pruning das arestas
    # -------------------------
    G = nx.Graph()
    for (u, v) in E_full:
        lat = float(instance.lat_e.get((u, v), instance.lat_e.get((v, u), 1.0)))
        bwc = float(instance.BW_cap.get((u, v), instance.BW_cap.get((v, u), 1.0)))
        G.add_edge(u, v, latency=lat, bandwidth=bwc)

    # pega até DEG_LIMIT arestas de menor latência por nó
    E_use = set()
    for n in N:
        nbrs = []
        for nb in G.neighbors(n):
            lat = G[n][nb].get("latency", 1.0)
            # armazena lado *normalizado* (u,v) com u < v para evitar duplicatas
            u, v = (n, nb) if n < nb else (nb, n)
            nbrs.append((lat, (u, v)))
        nbrs.sort(key=lambda x: x[0])
        for _, e in nbrs[:DEG_LIMIT]:
            E_use.add(e)
    E_use = list(E_use)
    if msg:
        print(f"[MILP] Edge pruning: |E_full|={len(E_full)} → |E_use|={len(E_use)} (DEG_LIMIT={DEG_LIMIT})")

    # -------------------------
    # Modelo
    # -------------------------
    m = gp.Model("SlicePlacementEnergy")
    m.Params.OutputFlag = 1 if msg else 0
    if time_limit:
        m.Params.TimeLimit = time_limit

    # Tuning sugerido
    m.Params.MIPFocus = 1
    m.Params.Heuristics = 0.5
    m.Params.Presolve = 2
    m.Params.Cuts = 3
    m.Params.NumericFocus = 1
    m.Params.IntFeasTol = 1e-5
    m.Params.FeasibilityTol = 1e-6
    m.Params.OptimalityTol = 1e-6
    m.Params.MIPGap = MIP_GAP

    # -------------------------
    # Variáveis
    # -------------------------
    # v[i,n] binária
    v = {(i, n): m.addVar(vtype=GRB.BINARY, name=f"v_{i}_{n}")
         for s in S for i in instance.V_of_s[s] for n in N}

    # u[n] ∈ [0,1], z[n] binária
    u = {n: m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"u_{n}") for n in N}
    z = {n: m.addVar(vtype=GRB.BINARY, name=f"z_{n}") for n in N}

    # f[e,s,(i,j)] contínua em [0,1] nas arestas do subgrafo
    f = {}
    for s in S:
        vnf_ids = instance.V_of_s[s]
        for q in range(len(vnf_ids) - 1):
            i, j = vnf_ids[q], vnf_ids[q + 1]
            for e in E_use:
                f[(e, s, (i, j))] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1,
                                             name=f"f_{e[0]}_{e[1]}_s{s}_{i}_to_{j}")

    # f_entry[e,s] contínua em [0,1] nas arestas do subgrafo (se houver ENTRY)
    f_entry = {}
    for s in S:
        s_entry = get_entry(s)
        if s_entry is not None:
            for e in E_use:
                f_entry[(e, s)] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1,
                                           name=f"f_entry_{e[0]}_{e[1]}_s{s}")

    # rho[e], w[e] por aresta do subgrafo
    rho = {e: m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"rho_{e[0]}_{e[1]}") for e in E_use}
    w   = {e: m.addVar(vtype=GRB.BINARY, name=f"w_{e[0]}_{e[1]}") for e in E_use}

    # -------------------------
    # Objetivo
    # -------------------------
    m.setObjective(
        gp.quicksum(u[n] + z[n] for n in N) +
        gp.quicksum(rho[e] + w[e] for e in E_use),
        GRB.MINIMIZE
    )

    # -------------------------
    # Restrições
    # -------------------------
    # (1) CPU capacity
    for n in N:
        m.addConstr(
            gp.quicksum(instance.CPU_i[i] * v[(i, n)]
                        for s in S for i in instance.V_of_s[s])
            <= instance.CPU_cap[n],
            name=f"CPU_cap_{n}"
        )
    # (1b) u[n] lower bound
    for n in N:
        cap = max(1e-9, float(instance.CPU_cap[n]))
        m.addConstr(
            gp.quicksum((instance.CPU_i[i] / cap) * v[(i, n)]
                        for s in S for i in instance.V_of_s[s])
            - u[n] <= 0,
            name=f"u_lb_{n}"
        )
    # (1c) u[n] ≤ z[n]
    for n in N:
        m.addConstr(u[n] - z[n] <= 0, name=f"u_leq_z_{n}")

    # (2) Assignment: cada VNF alocado exatamente uma vez
    for s in S:
        for i in instance.V_of_s[s]:
            m.addConstr(
                gp.quicksum(v[(i, n)] for n in N) == 1,
                name=f"assign_{i}"
            )

    # (3) Anti-colocation: no máx. 1 VNF do slice por nó
    for s in S:
        for n in N:
            m.addConstr(
                gp.quicksum(v[(i, n)] for i in instance.V_of_s[s]) <= 1,
                name=f"antico_s{s}_n{n}"
            )

    # (4) Conservação de fluxo para VLs entre VNFs consecutivos
    for s in S:
        vnf_ids = instance.V_of_s[s]
        for q in range(len(vnf_ids) - 1):
            i, j = vnf_ids[q], vnf_ids[q + 1]
            for n in N:
                m.addConstr(
                    gp.quicksum(_incidence(n, e) * f[(e, s, (i, j))] for e in E_use)
                    - v[(i, n)] + v[(j, n)] == 0,
                    name=f"flow_s{s}_{i}_to_{j}_n{n}"
                )

    # (4b) Conservação de fluxo ENTRY→primeiro VNF (se houver ENTRY)
    for s in S:
        s_entry = get_entry(s)
        if s_entry is None:
            continue
        first_vnf = instance.V_of_s[s][0]
        for n in N:
            m.addConstr(
                gp.quicksum(_incidence(n, e) * f_entry[(e, s)] for e in E_use)
                == (1.0 if n == s_entry else 0.0) - v[(first_vnf, n)],
                name=f"flow_entry_s{s}_n{n}"
            )

    # (5) Link capacity and rho (includes VLs + ENTRY)
    for e in E_use:
        # Soma total de fluxo em cada aresta
        flow_sum_VLs = gp.quicksum(
            instance.BW_sij[(s, i, j)] * f[(e, s, (i, j))]
            for s in S
            for (i, j) in [(instance.V_of_s[s][qq], instance.V_of_s[s][qq + 1])
                        for qq in range(len(instance.V_of_s[s]) - 1)]
            if (s, i, j) in instance.BW_sij and (e, s, (i, j)) in f
        )

        flow_sum_ENTRY = gp.quicksum(
            instance.BW_entry.get(s, 0.0) * f_entry[(e, s)]
            for s in S if (e, s) in f_entry
        )

        total_edge_flow = flow_sum_VLs + flow_sum_ENTRY

        cap_e = float(instance.BW_cap.get(e, instance.BW_cap.get((e[1], e[0]), 1e9)))
        m.addConstr(total_edge_flow <= cap_e, name=f"bwcap_{e}")
        m.addConstr(total_edge_flow - cap_e * rho[e] <= 0, name=f"rho_def_{e}")
        m.addConstr(rho[e] - w[e] <= 0, name=f"rho_leq_w_{e}")


    # (6) Latency per virtual link (uses L_sij instead of L_s)
    for s in S:
        vnf_ids = instance.V_of_s[s]
        for q in range(len(vnf_ids) - 1):
            i, j = vnf_ids[q], vnf_ids[q + 1]
            lat_budget = instance.L_sij.get((s, i, j), 1e9)
            m.addConstr(
                gp.quicksum(
                    float(instance.lat_e.get(e, instance.lat_e.get((e[1], e[0]), 1.0))) *
                    f[(e, s, (i, j))]
                    for e in E_use if (e, s, (i, j)) in f
                )
                <= lat_budget,
                name=f"latency_s{s}_{i}_to_{j}"
            )

# (6b) Latency ENTRY→first VNF (if defined)
    for s in S:
        if any((e, s) in f_entry for e in E_use):
            lat_budget = instance.L_entry.get(s, 1e9)
            m.addConstr(
                gp.quicksum(
                    float(instance.lat_e.get(e, instance.lat_e.get((e[1], e[0]), 1.0))) *
                    f_entry[(e, s)]
                    for e in E_use if (e, s) in f_entry
                )
                <= lat_budget,
                name=f"latency_entry_s{s}"
            )


    # -------------------------
    # Solve
    # -------------------------
    m.optimize()

    if m.status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
        print(f"[MILP] No feasible solution or unsolved. Status: {m.status} ({m.Status})")
        if m.status == GRB.INFEASIBLE:
            try:
                m.computeIIS()
                m.write("model_infeasible.ilp")
                print("[MILP] IIS written to model_infeasible.ilp")
            except Exception as e:
                print(f"[MILP] Could not write IIS: {e}")
        return GurobiSolveResult(
            status_code=int(m.status),
            status_str=str(m.Status),
            objective=None,
            values={}
        )

    # -------------------------
    # Coleta da solução
    # -------------------------
    values = {}
    for key, var in v.items():         values[("v",) + key] = var.X
    for n in N:                        values[("u", n)] = u[n].X; values[("z", n)] = z[n].X
    for key, var in f.items():         values[("f",) + key] = var.X
    for e in E_use:                    values[("rho", e)] = rho[e].X; values[("w", e)] = w[e].X
    for key, var in f_entry.items():   values[("f_entry",) + key] = var.X

    return GurobiSolveResult(
        status_code=int(m.status),
        status_str=str(m.Status),
        objective=m.objVal,
        values=values
    )
