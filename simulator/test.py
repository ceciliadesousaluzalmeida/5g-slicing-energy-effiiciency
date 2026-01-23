from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def _split_nodes(s: str) -> List[str]:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return []
    s = str(s).strip()
    if s == "" or s.lower() == "nan":
        return []
    return [x.strip() for x in s.split("->") if x.strip() != ""]


def _split_edges(s: str) -> List[Tuple[str, str]]:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return []
    s = str(s).strip()
    if s == "" or s.lower() == "nan":
        return []
    # Supports separators like "|", ";", ",", " " (fallback)
    for sep in ["|", ";", ",", " "]:
        if sep in s:
            parts = [p.strip() for p in s.split(sep) if p.strip() != ""]
            return [_parse_edge(p) for p in parts]
    return [_parse_edge(s)]


def _parse_edge(token: str) -> Tuple[str, str]:
    token = token.strip()
    if "-" not in token:
        return (token, token)
    a, b = token.split("-", 1)
    return (a.strip(), b.strip())


def validate_routes(routes_csv: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(routes_csv)

    # Required columns in your file
    required = [
        "timestamp", "method", "num_slices", "num_vnfs_per_slice", "seed",
        "result_idx", "slice", "vnf_src", "vnf_dst", "path_nodes", "path_edges", "num_hops"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in routes CSV: {missing}. Available: {list(df.columns)}")

    violations: List[Dict[str, object]] = []

    for _, r in df.iterrows():
        method = str(r["method"])
        seed = str(r["seed"])
        scenario = f"slices={r['num_slices']}_vnfs={r['num_vnfs_per_slice']}_seed={seed}"
        ridx = r["result_idx"]

        nodes = _split_nodes(r["path_nodes"])
        edges = _split_edges(r["path_edges"])

        # 1) num_hops check
        try:
            hops = int(r["num_hops"])
        except Exception:
            hops = None

        if hops is not None:
            if hops != len(edges):
                violations.append({
                    "type": "HOPS_MISMATCH_EDGES",
                    "method": method,
                    "scenario": scenario,
                    "entity": f"result_idx={ridx}",
                    "detail": "num_hops != number of parsed edges in path_edges.",
                    "value": hops,
                    "limit": len(edges),
                })

        # 2) nodes vs edges length relation: for simple paths, len(nodes) should be len(edges)+1
        if nodes and edges:
            if len(nodes) != len(edges) + 1:
                violations.append({
                    "type": "NODES_EDGES_LENGTH_INCONSISTENT",
                    "method": method,
                    "scenario": scenario,
                    "entity": f"result_idx={ridx}",
                    "detail": "len(path_nodes) should equal len(path_edges)+1 for a simple hop-by-hop path.",
                    "value": len(nodes),
                    "limit": len(edges) + 1,
                })

        # 3) edge sequence must match node sequence
        if len(nodes) >= 2 and edges:
            expected_edges = list(zip(nodes[:-1], nodes[1:]))
            if expected_edges != edges:
                violations.append({
                    "type": "EDGE_SEQUENCE_MISMATCH",
                    "method": method,
                    "scenario": scenario,
                    "entity": f"result_idx={ridx}",
                    "detail": "Edges implied by path_nodes do not match path_edges sequence.",
                    "value": str(edges),
                    "limit": str(expected_edges),
                })

        # 4) empty path suspicious cases
        # If you have vnf_src and vnf_dst, a valid route normally has at least 1 hop unless colocated node-level routing is allowed
        if (not nodes) and (not edges):
            violations.append({
                "type": "EMPTY_PATH",
                "method": method,
                "scenario": scenario,
                "entity": f"result_idx={ridx}",
                "detail": "Both path_nodes and path_edges are empty.",
                "value": None,
                "limit": None,
            })

        # 5) self-loops or repeated nodes (cycle) – mark as suspicious
        if any(a == b for a, b in edges):
            violations.append({
                "type": "SELF_LOOP_EDGE",
                "method": method,
                "scenario": scenario,
                "entity": f"result_idx={ridx}",
                "detail": "Path contains an edge a-a (self-loop).",
                "value": str(edges),
                "limit": None,
            })

        if len(nodes) != len(set(nodes)) and len(nodes) > 0:
            violations.append({
                "type": "CYCLE_OR_REPEATED_NODE",
                "method": method,
                "scenario": scenario,
                "entity": f"result_idx={ridx}",
                "detail": "path_nodes contains repeated nodes (cycle or revisiting).",
                "value": str(nodes),
                "limit": None,
            })

    violations_df = pd.DataFrame(violations)
    if len(violations_df) == 0:
        summary_df = pd.DataFrame(columns=["method", "type", "count"])
    else:
        summary_df = (
            violations_df.groupby(["method", "type"])
            .size()
            .reset_index(name="count")
            .sort_values(["count", "method", "type"], ascending=[False, True, True])
        )
    return violations_df, summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate routes_all_methods.csv path consistency.")
    parser.add_argument("--dir", type=str, required=True, help="Folder containing routes_all_methods.csv")
    args = parser.parse_args()

    folder = Path(args.dir).expanduser().resolve()
    routes_csv = folder / "routes_all_methods.csv"

    violations_df, summary_df = validate_routes(routes_csv)

    out_v = folder / "route_violations_report.csv"
    out_s = folder / "route_violations_summary.csv"
    violations_df.to_csv(out_v, index=False)
    summary_df.to_csv(out_s, index=False)

    print(f"Saved: {out_v}")
    print(f"Saved: {out_s}")
    print("\nTop summary:")
    print(summary_df.head(30).to_string(index=False) if len(summary_df) else "No route violations found.")


if __name__ == "__main__":
    main()

