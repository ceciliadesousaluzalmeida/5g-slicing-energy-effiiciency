from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gurobipy as gp
from gurobipy import GRB

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from milp.milp_two_phase import build_multi_slice_model_with_accept
from milp.heuristics_mipstart_validate import apply_mip_start_from_heuristic, validate_mip_start




def now_tag() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def safe_name(x: str) -> str:
    return (
        x.replace(" ", "_")
        .replace("*", "star")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


class Tee:
    # English: Mirror stdout/stderr into a log file.
    def __init__(self, path: str, stream):
        self.path = path
        self.stream = stream
        self.f = open(path, "w", encoding="utf-8")

    def write(self, data):
        self.stream.write(data)
        self.f.write(data)

    def flush(self):
        self.stream.flush()
        self.f.flush()

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass


def status_to_str(code: int) -> str:
    mapping = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
        GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
        GRB.NODE_LIMIT: "NODE_LIMIT",
        GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
        GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
    }
    return mapping.get(code, f"STATUS_{code}")


def safe_mip_gap(model: gp.Model) -> Optional[float]:
    try:
        if getattr(model, "SolCount", 0) == 0:
            return None
        return float(model.MIPGap)
    except Exception:
        return None


def heuristic_accept_count(instance: Any, result_list: List[Any]) -> int:
    accepted = 0
    for s in list(instance.S):
        res = result_list[s] if s < len(result_list) else None
        if res is None:
            continue
        if hasattr(res, "accepted"):
            try:
                if bool(getattr(res, "accepted")):
                    accepted += 1
                continue
            except Exception:
                pass
        pv = getattr(res, "placed_vnfs", None)
        if isinstance(pv, dict) and len(pv) > 0:
            accepted += 1
    return accepted


def validate_endpoints_and_hops(
    G,
    instance: Any,
    result_list: List[Any],
    *,
    strict_directed: bool = True,
) -> Dict[str, Any]:
    """
    English:
      For each slice and each VL (i,j) that exists in instance.BW_sij:
        - ensure both VNFs are placed
        - ensure routed path exists when src != dst
        - ensure endpoints match placement (or ENTRY node)
        - ensure each hop exists in G (directed if strict_directed)
    """
    violations = []

    # Edge set for fast check
    if not G.is_directed():
        edge_ok = set(G.edges()) | {(v, u) for (u, v) in G.edges()}
    else:
        edge_ok = set(G.edges())
        if not strict_directed:
            edge_ok = edge_ok | {(v, u) for (u, v) in edge_ok}

    def get_entry_node(s):
        if hasattr(instance, "entry_of_s"):
            return instance.entry_of_s.get(s)
        if hasattr(instance, "ENTRY_of_s"):
            return instance.ENTRY_of_s.get(s)
        if hasattr(instance, "entry_node_of_s"):
            return instance.entry_node_of_s.get(s)
        if hasattr(instance, "entry_node"):
            return getattr(instance, "entry_node")
        return None

    for s in list(instance.S):
        res = result_list[s] if s < len(result_list) else None
        if res is None:
            continue

        placed = getattr(res, "placed_vnfs", None)
        routed = getattr(res, "routed_vls", None)
        placed = placed if isinstance(placed, dict) else {}
        routed = routed if isinstance(routed, dict) else {}

        V_s = list(instance.V_of_s[s])

        vl_pairs = []
        for i in ["ENTRY"] + V_s:
            for j in V_s:
                if (s, i, j) in instance.BW_sij:
                    vl_pairs.append((i, j))

        entry_node = get_entry_node(s)

        for (i, j) in vl_pairs:
            if i == "ENTRY":
                if entry_node is None:
                    violations.append({
                        "slice": s,
                        "type": "missing_entry_node",
                        "vl": (i, j),
                        "details": "instance has ENTRY VL but no entry node is defined."
                    })
                    continue
                src_node = entry_node
            else:
                src_node = placed.get(i, None)
                if src_node is None:
                    violations.append({
                        "slice": s,
                        "type": "missing_placement_src",
                        "vl": (i, j),
                        "details": f"missing placement for VNF {i}."
                    })
                    continue

            dst_node = placed.get(j, None)
            if dst_node is None:
                violations.append({
                    "slice": s,
                    "type": "missing_placement_dst",
                    "vl": (i, j),
                    "details": f"missing placement for VNF {j}."
                })
                continue

            # Locate path
            path = None
            if (i, j) in routed:
                path = routed[(i, j)]

            if src_node == dst_node:
                if path is None:
                    continue
                if isinstance(path, list) and (len(path) == 0 or (len(path) == 1 and path[0] == src_node)):
                    continue
                violations.append({
                    "slice": s,
                    "type": "unexpected_path_same_node",
                    "vl": (i, j),
                    "details": f"src==dst=={src_node} but path={path}."
                })
                continue

            if path is None or not isinstance(path, list) or len(path) < 2:
                violations.append({
                    "slice": s,
                    "type": "missing_or_trivial_path",
                    "vl": (i, j),
                    "details": f"src={src_node}, dst={dst_node}, path={path}."
                })
                continue

            if path[0] != src_node or path[-1] != dst_node:
                violations.append({
                    "slice": s,
                    "type": "endpoint_mismatch",
                    "vl": (i, j),
                    "details": f"Expected ({src_node}->{dst_node}), got ({path[0]}->{path[-1]}), path={path}."
                })
                continue

            bad_hops = []
            for u, v in zip(path[:-1], path[1:]):
                if (u, v) not in edge_ok:
                    bad_hops.append((u, v))
            if bad_hops:
                violations.append({
                    "slice": s,
                    "type": "invalid_hops",
                    "vl": (i, j),
                    "details": f"Invalid hops={bad_hops}, strict_directed={strict_directed}."
                })

    ok = len(violations) == 0
    return {
        "ok": ok,
        "n_violations": len(violations),
        "violations": violations[:200],
        "reason": "VALID (paths match endpoints and hops)." if ok else "INVALID (endpoint/hop violations found).",
    }


def write_txt(path: str, lines: List[str]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_validation_once(
    *,
    heuristic_name: str,
    seed: int,
    num_slices: int,
    num_vnfs_per_slice: int,
    entry_node: int,
    milp_time_limit: int,
    milp_gap: float,
    msg: bool,
    results_dir: str,
) -> Dict[str, Any]:
    from utils.topology import topologie_finlande
    from utils.generate_slices import generate_random_slices
    from milp.create_instance import create_instance

    from heuristics.a_star import run_astar
    from heuristics.run_abo_full_batch import run_abo_full_batch
    from heuristics.run_fabo_full_batch import run_fabo_full_batch
    from heuristics.a_star_energy_aware import energy_aware_astar
    from heuristics.best_fit import run_best_fit
    from heuristics.first_fit import run_first_fit

    vnf_profiles = [
        {"cpu": 1, "throughput": 15, "latency": 30},
        {"cpu": 1, "throughput": 20, "latency": 60},
        {"cpu": 2, "throughput": 25, "latency": 90},
        {"cpu": 3, "throughput": 30, "latency": 100},
        {"cpu": 4, "throughput": 35, "latency": 125},
        {"cpu": 6, "throughput": 50, "latency": 135},
    ]

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    G = topologie_finlande()

    node_capacity_base = {n: G.nodes[n]["cpu"] for n in G.nodes}
    link_capacity_base = {(u, v): G[u][v]["bandwidth"] for u, v in G.edges}
    link_capacity_base.update({(v, u): G[u][v]["bandwidth"] for u, v in G.edges})

    link_latency = {(u, v): G[u][v]["latency"] for u, v in G.edges}
    link_latency.update({(v, u): G[u][v]["latency"] for u, v in G.edges})

    slice_pool = generate_random_slices(
        G,
        vnf_profiles,
        num_slices=num_slices,
        num_vnfs_per_slice=num_vnfs_per_slice,
        entry=entry_node,
    )
    slices = deepcopy(slice_pool[:num_slices])

    heuristics_map = {
        "A*": (run_astar, (G, slices, node_capacity_base, link_capacity_base)),
        "ABO": (run_abo_full_batch, (G, slices, node_capacity_base, link_latency, link_capacity_base)),
        "FABO": (run_fabo_full_batch, (G, slices, node_capacity_base, link_latency, link_capacity_base)),
        "EAA": (energy_aware_astar, (G, slices, node_capacity_base, link_capacity_base)),
        "BF": (run_best_fit, (G, slices, node_capacity_base, link_capacity_base, link_latency)),
        "FF": (run_first_fit, (G, slices, node_capacity_base, link_capacity_base, link_latency)),
    }
    if heuristic_name not in heuristics_map:
        raise ValueError(f"Unknown heuristic_name={heuristic_name}. Options={list(heuristics_map.keys())}")

    instance = create_instance(G, slices)
    if hasattr(instance, "entry_node"):
        instance.entry_node = entry_node

    func, args = heuristics_map[heuristic_name]

    t0 = time.time()
    _, result_list = func(*args)
    heur_runtime = time.time() - t0
    heur_accepted = heuristic_accept_count(instance, result_list)

    # Endpoint/hops validation
    endpoint_check = validate_endpoints_and_hops(
        G=G,
        instance=instance,
        result_list=result_list,
        strict_directed=G.is_directed(),
    )

    # Build MILP model (your MILP) and apply MIP start
    tag = now_tag()
    safe_h = safe_name(heuristic_name)
    base = os.path.join(results_dir, f"validation_{safe_h}_s{num_slices}_v{num_vnfs_per_slice}_seed{seed}_{tag}")
    gurobi_log = base + ".gurobi.log"
    report_txt = base + ".report.txt"
    report_json = base + ".report.json"

    model, vp = build_multi_slice_model_with_accept(
        instance=instance,
        slice_set=list(instance.S),
        msg=msg,
        time_limit=milp_time_limit,
        mip_gap=milp_gap,
        mip_focus=1,
        heuristics=0.6,
        cuts=1,
        presolve=2,
        numeric_focus=1,
        seed=seed,
        threads=None,
        log_file=gurobi_log,
    )

    apply_mip_start_from_heuristic(
        model=model,
        vars_pack=vp,
        instance=instance,
        result_list=result_list,
        force_anti_colocation=True,
        route_if_missing=True,
    )

    mipstart_validation = validate_mip_start(model, tol=1e-6, max_violations=50)

    # Optional: actually optimize the MILP starting from the heuristic start
    t1 = time.time()
    model.optimize()
    milp_runtime = time.time() - t1

    status = status_to_str(model.Status)
    solcount = int(model.SolCount)
    gap = safe_mip_gap(model)

    # Count accepted from MILP solution (z vars)
    milp_accepted = 0
    if solcount > 0 and model.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED):
        z = vp["z"]
        for s in list(instance.S):
            try:
                if z[s].X > 0.5:
                    milp_accepted += 1
            except Exception:
                pass

    # Write TXT report
    lines = []
    lines.append("=== Heuristic-to-YourMILP Validation Report ===")
    lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Heuristic: {heuristic_name}")
    lines.append(f"Seed: {seed}")
    lines.append(f"Num slices: {num_slices}")
    lines.append(f"VNFs per slice: {num_vnfs_per_slice}")
    lines.append(f"Entry node: {entry_node}")
    lines.append("")
    lines.append(f"Heuristic runtime (s): {heur_runtime:.6f}")
    lines.append(f"Heuristic accepted: {heur_accepted}/{num_slices}")
    lines.append("")
    lines.append(f"Endpoint/Hops check: {'VALID' if endpoint_check['ok'] else 'INVALID'}")
    lines.append(f"Why: {endpoint_check['reason']}")
    lines.append(f"Endpoint violations: {endpoint_check['n_violations']}")
    if not endpoint_check["ok"]:
        for v in endpoint_check["violations"][:10]:
            lines.append(f"  - slice={v['slice']} type={v['type']} vl={v['vl']} details={v['details']}")
    lines.append("")
    lines.append(f"MIP start validation: {'VALID' if mipstart_validation['ok'] else 'INVALID'}")
    lines.append(f"Why: {mipstart_validation['reason']}")
    lines.append(f"Violated constraints: {mipstart_validation['n_violations']}")
    if not mipstart_validation["ok"] and mipstart_validation["top"]:
        lines.append("Top violations (violation | name | sense | lhs | rhs):")
        for viol, name, fam, sense, lhs, rhs in mipstart_validation["top"][:20]:
            lines.append(f"  - {viol:.3e} | {name} | {sense} | lhs={lhs:.6f} rhs={rhs:.6f}")
    lines.append("")
    lines.append(f"Gurobi log: {gurobi_log}")
    lines.append(f"MILP status: {status}, SolCount={solcount}, MIPGap={gap}")
    lines.append(f"MILP accepted: {milp_accepted}/{num_slices}")
    lines.append(f"MILP runtime (wall): {milp_runtime:.4f}s")

    write_txt(report_txt, lines)

    # Write JSON report (structured)
    payload = {
        "timestamp": datetime.now().isoformat(),
        "heuristic": heuristic_name,
        "seed": seed,
        "num_slices": num_slices,
        "vnfs_per_slice": num_vnfs_per_slice,
        "entry_node": entry_node,
        "heur_runtime_s": heur_runtime,
        "heur_accepted": heur_accepted,
        "endpoint_check": endpoint_check,
        "mipstart_validation": mipstart_validation,
        "milp": {
            "status": status,
            "solcount": solcount,
            "mip_gap": gap,
            "accepted": milp_accepted,
            "runtime_wall_s": milp_runtime,
        },
        "paths": {
            "report_txt": report_txt,
            "report_json": report_json,
            "gurobi_log": gurobi_log,
        },
    }
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return {
        "heuristic": heuristic_name,
        "seed": seed,
        "num_slices": num_slices,
        "vnfs_per_slice": num_vnfs_per_slice,
        "entry_node": entry_node,
        "heur_runtime_s": float(heur_runtime),
        "heur_accepted": int(heur_accepted),
        "endpoint_ok": bool(endpoint_check["ok"]),
        "endpoint_violations": int(endpoint_check["n_violations"]),
        "mipstart_ok": bool(mipstart_validation["ok"]),
        "mipstart_violations": int(mipstart_validation["n_violations"]),
        "milp_status": status,
        "milp_accepted": int(milp_accepted),
        "milp_solcount": int(solcount),
        "milp_gap": gap,
        "report_txt": report_txt,
        "report_json": report_json,
        "gurobi_log": gurobi_log,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, default="./results")
    p.add_argument("--heuristic", type=str, default="A*")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--num_slices", type=int, default=64)
    p.add_argument("--vnfs_per_slice", type=int, default=2)
    p.add_argument("--entry_node", type=int, default=6)
    p.add_argument("--milp_time_limit", type=int, default=1800)
    p.add_argument("--milp_gap", type=float, default=0.02)
    p.add_argument("--msg", action="store_true")
    args = p.parse_args()

    tag = now_tag()
    ensure_dir(args.results_dir)

    run_dir = os.path.join(
        args.results_dir,
        f"validate_{safe_name(args.heuristic)}_s{args.num_slices}_v{args.vnfs_per_slice}_seed{args.seed}_{tag}",
    )
    ensure_dir(run_dir)

    stdout_log = os.path.join(run_dir, "stdout.log")
    stderr_log = os.path.join(run_dir, "stderr.log")

    tee_out = Tee(stdout_log, sys.stdout)
    tee_err = Tee(stderr_log, sys.stderr)
    sys.stdout = tee_out
    sys.stderr = tee_err

    try:
        print(f"[INFO] Results dir: {run_dir}")
        print(f"[INFO] Heuristic: {args.heuristic}")
        print(f"[INFO] seed={args.seed} slices={args.num_slices} vnfs_per_slice={args.vnfs_per_slice} entry={args.entry_node}")
        print(f"[INFO] milp_time_limit={args.milp_time_limit} milp_gap={args.milp_gap} msg={args.msg}")
        print("")

        summary = run_validation_once(
            heuristic_name=args.heuristic,
            seed=args.seed,
            num_slices=args.num_slices,
            num_vnfs_per_slice=args.vnfs_per_slice,
            entry_node=args.entry_node,
            milp_time_limit=args.milp_time_limit,
            milp_gap=args.milp_gap,
            msg=bool(args.msg),
            results_dir=run_dir,
        )

        # Write a small summary JSON in run_dir root
        with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print("\n[SUMMARY]")
        for k, v in summary.items():
            print(f"{k}: {v}")

        print(f"\n[INFO] stdout log: {stdout_log}")
        print(f"[INFO] stderr log: {stderr_log}")
        print(f"[INFO] report txt: {summary['report_txt']}")
        print(f"[INFO] report json: {summary['report_json']}")
        print(f"[INFO] gurobi log: {summary['gurobi_log']}")

    finally:
        try:
            sys.stdout = tee_out.stream
            sys.stderr = tee_err.stream
        except Exception:
            pass
        try:
            tee_out.close()
            tee_err.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
