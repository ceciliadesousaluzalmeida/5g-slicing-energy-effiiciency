import re
from copy import deepcopy

import numpy as np


ENTRY_VNF_ID = "__ENTRY__"


def _normalize_token(x):
    # Normalize IDs for robust lookup.
    s = str(x).strip().lower()
    s = s.replace(" ", "")
    s = re.sub(r"[^a-z0-9_(),\-\[\]]", "", s)
    return s


def _route_key_to_parts(vl_key):
    # Supports keys like (s, i, j) or (i, j).
    if isinstance(vl_key, tuple) and len(vl_key) == 3:
        s, i, j = vl_key
        return s, i, j

    if isinstance(vl_key, tuple) and len(vl_key) == 2:
        i, j = vl_key
        return None, i, j

    return None, None, None


def build_vnf_cpu_alias_map(slices, num_vnfs_per_slice=None):
    # Build a robust VNF CPU lookup map.
    alias_map = {}

    def _set_alias(key, cpu):
        if key is None or cpu is None:
            return
        try:
            cpu_f = float(cpu)
        except Exception:
            return
        alias_map[key] = cpu_f
        alias_map[_normalize_token(key)] = cpu_f

    def _maybe_cpu(obj):
        if obj is None:
            return None
        if isinstance(obj, (int, float)):
            return float(obj)
        if isinstance(obj, dict):
            return (
                obj.get("cpu")
                or obj.get("cpu_demand")
                or obj.get("demand_cpu")
                or obj.get("cpu_req")
            )
        for attr in ["cpu", "cpu_demand", "demand_cpu", "cpu_req"]:
            if hasattr(obj, attr):
                val = getattr(obj, attr)
                if isinstance(val, (int, float)):
                    return float(val)
        return None

    def _maybe_id(obj):
        if obj is None:
            return None
        if isinstance(obj, (int, str)):
            return obj
        if isinstance(obj, dict):
            return obj.get("id") or obj.get("vnf_id") or obj.get("name") or obj.get("label")
        for attr in ["id", "vnf_id", "name", "label"]:
            if hasattr(obj, attr):
                return getattr(obj, attr)
        return None

    def _extract_vnfs_from_slice(sl):
        if isinstance(sl, dict):
            for k in ["vnfs", "vnf_list", "functions", "chain", "vnf_chain", "vnf_sequence"]:
                vnfs = sl.get(k)
                if isinstance(vnfs, list) and vnfs:
                    return vnfs

            for k in ["vnf_cpu", "vnf_cpu_map", "cpu_by_vnf", "cpu_demands", "vnf_demands", "demands"]:
                m = sl.get(k)
                if isinstance(m, dict) and m:
                    return [(vid, cpu) for vid, cpu in m.items()]

            return []

        if isinstance(sl, (tuple, list)):
            for item in sl:
                if isinstance(item, list) and item:
                    return item
                if isinstance(item, tuple) and item and all(
                    isinstance(x, (tuple, list, dict)) for x in item
                ):
                    return list(item)
                if isinstance(item, dict) and item:
                    if any(k in item for k in ["vnfs", "vnf_list", "functions", "chain"]):
                        return _extract_vnfs_from_slice(item)

            return []

        return []

    def _slice_id(sl, default):
        if isinstance(sl, dict):
            return sl.get("id", default)
        if isinstance(sl, (tuple, list)):
            if len(sl) >= 1 and isinstance(sl[0], (int, str)):
                return sl[0]
        return default

    for s_idx, sl in enumerate(slices):
        s_id = _slice_id(sl, s_idx)
        vnfs = _extract_vnfs_from_slice(sl)

        for k_idx, v in enumerate(vnfs):
            if (
                isinstance(v, (tuple, list))
                and len(v) >= 2
                and isinstance(v[0], (int, str))
                and isinstance(v[1], (int, float))
            ):
                vid = v[0]
                cpu = v[1]
            else:
                vid = _maybe_id(v)
                cpu = _maybe_cpu(v)

                if cpu is None and isinstance(v, (tuple, list)):
                    for it in v:
                        if isinstance(it, (int, float)):
                            cpu = float(it)
                            break
                    if vid is None and len(v) >= 1:
                        vid = _maybe_id(v[0])

            if vid is not None:
                _set_alias(vid, cpu)
                _set_alias(str(vid), cpu)

            for s_key in [s_idx, s_id]:
                _set_alias((s_key, k_idx), cpu)
                _set_alias(f"vnf{s_key}_{k_idx}", cpu)
                _set_alias(f"{s_key}_{k_idx}", cpu)
                _set_alias(f"({s_key},{k_idx})", cpu)

            if num_vnfs_per_slice is not None:
                try:
                    global_id = int(s_idx) * int(num_vnfs_per_slice) + int(k_idx)
                    _set_alias(global_id, cpu)
                    _set_alias(str(global_id), cpu)
                except Exception:
                    pass

    return alias_map


def export_node_hosting_to_rows(
    method_name,
    result_list,
    slices,
    num_slices,
    num_vnfs_per_slice,
    seed,
    timestamp_str,
    node_capacity_base,
    debug=False,
):
    # Export node hosting and CPU usage rows.
    alias_map = build_vnf_cpu_alias_map(
        slices,
        num_vnfs_per_slice=num_vnfs_per_slice,
    )

    rows_hosting = []
    rows_cpu = []

    hosted = {}
    cpu_used = {}
    unknown_cpu = {}
    unknown_examples = []

    for res in result_list:
        if not hasattr(res, "placed_vnfs") or res.placed_vnfs is None:
            continue

        for vnf_id, node in res.placed_vnfs.items():
            vnf_id_str = str(vnf_id)
            hosted.setdefault(node, set()).add(vnf_id_str)

            candidates = [vnf_id, vnf_id_str, _normalize_token(vnf_id_str)]

            if isinstance(vnf_id, tuple):
                candidates.append(vnf_id)
                candidates.append(_normalize_token(vnf_id))

            found = False

            for c in candidates:
                if c in alias_map:
                    cpu_used[node] = cpu_used.get(node, 0.0) + float(alias_map[c])
                    found = True
                    break

            if not found:
                unknown_cpu[node] = unknown_cpu.get(node, 0) + 1
                if debug and len(unknown_examples) < 12:
                    unknown_examples.append(
                        (vnf_id, vnf_id_str, _normalize_token(vnf_id_str))
                    )

    if debug and unknown_examples:
        print("[DEBUG] Unknown VNF IDs sample:")
        for ex in unknown_examples:
            print("   ", ex)

    for node, vnf_set in hosted.items():
        rows_hosting.append({
            "timestamp": timestamp_str,
            "method": method_name,
            "num_slices": num_slices,
            "num_vnfs_per_slice": num_vnfs_per_slice,
            "seed": seed,
            "node": node,
            "hosted_vnfs": ",".join(sorted(vnf_set)),
            "num_hosted_vnfs": len(vnf_set),
        })

    for node in node_capacity_base.keys():
        used = cpu_used.get(node, 0.0)
        cap = float(node_capacity_base[node]) if node in node_capacity_base else np.nan

        rows_cpu.append({
            "timestamp": timestamp_str,
            "method": method_name,
            "num_slices": num_slices,
            "num_vnfs_per_slice": num_vnfs_per_slice,
            "seed": seed,
            "node": node,
            "cpu_used": used,
            "cpu_capacity": cap,
            "cpu_utilization": used / cap if cap > 0 else np.nan,
            "num_hosted_vnfs": len(hosted.get(node, set())),
            "num_unknown_cpu_vnfs": int(unknown_cpu.get(node, 0)),
        })

    return rows_hosting, rows_cpu


def export_routes_to_rows(
    method_name,
    result_list,
    num_slices,
    num_vnfs_per_slice,
    seed,
    timestamp_str,
):
    # Export routed virtual links as path rows.
    rows = []

    for res_idx, res in enumerate(result_list):
        if not hasattr(res, "routed_vls") or res.routed_vls is None:
            continue

        for vl_key, path_nodes in res.routed_vls.items():
            s_id, vnf_src, vnf_dst = _route_key_to_parts(vl_key)

            if not path_nodes or len(path_nodes) < 2:
                continue

            path_edges = list(zip(path_nodes[:-1], path_nodes[1:]))

            rows.append({
                "timestamp": timestamp_str,
                "method": method_name,
                "num_slices": num_slices,
                "num_vnfs_per_slice": num_vnfs_per_slice,
                "seed": seed,
                "result_idx": res_idx,
                "slice": s_id,
                "vnf_src": vnf_src,
                "vnf_dst": vnf_dst,
                "path_nodes": "->".join(map(str, path_nodes)),
                "path_edges": ";".join([f"{u}-{v}" for (u, v) in path_edges]),
                "num_hops": len(path_edges),
            })

    return rows


def export_milp_solution_to_rows(
    out,
    instance,
    method_name,
    num_slices,
    num_vnfs_per_slice,
    seed,
    timestamp_str,
):
    # Export MILP raw selected variables.
    if out is None or out.get("last_result") is None:
        return []

    res = out["last_result"]
    vals = res.values
    rows = []

    for key, val in vals.items():
        if not key or val <= 0.5:
            continue

        if key[0] == "x" and len(key) == 4:
            _, s, vnf_id, node = key
            rows.append({
                "timestamp": timestamp_str,
                "method": method_name,
                "num_slices": num_slices,
                "num_vnfs_per_slice": num_vnfs_per_slice,
                "seed": seed,
                "type": "placement",
                "slice": s,
                "vnf_src": vnf_id,
                "vnf_dst": None,
                "node": node,
                "u": None,
                "v": None,
                "path_nodes": None,
                "path_edges": None,
                "value": float(val),
            })

    for key, val in vals.items():
        if not key or val <= 1e-9:
            continue

        if key[0] == "f" and len(key) == 6:
            _, s, i, j, u, v = key
            rows.append({
                "timestamp": timestamp_str,
                "method": method_name,
                "num_slices": num_slices,
                "num_vnfs_per_slice": num_vnfs_per_slice,
                "seed": seed,
                "type": "flow_edge",
                "slice": s,
                "vnf_src": i,
                "vnf_dst": j,
                "node": None,
                "u": u,
                "v": v,
                "path_nodes": None,
                "path_edges": None,
                "value": float(val),
            })

    for key, val in vals.items():
        if not key:
            continue

        if key[0] == "xi" and len(key) == 2:
            _, s = key
            rows.append({
                "timestamp": timestamp_str,
                "method": method_name,
                "num_slices": num_slices,
                "num_vnfs_per_slice": num_vnfs_per_slice,
                "seed": seed,
                "type": "xi",
                "slice": s,
                "vnf_src": None,
                "vnf_dst": None,
                "node": None,
                "u": None,
                "v": None,
                "path_nodes": None,
                "path_edges": None,
                "value": float(val),
            })

    return rows


def _split_slice(slice_data):
    # Return (vnfs, vls, entry) from supported slice formats.
    if not isinstance(slice_data, (list, tuple)):
        return [], [], None

    if len(slice_data) == 2:
        return slice_data[0], slice_data[1], None

    if len(slice_data) >= 3:
        return slice_data[0], slice_data[1], slice_data[2]

    return [], [], None


def _first_vnf_id(vnfs):
    # Return the first VNF id of a slice.
    if not vnfs:
        return None
    return vnfs[0]["id"]


def _infer_entry_bandwidth(vnfs, vls):
    # Infer ENTRY bandwidth from the first outgoing VL of the first VNF.
    first_id = _first_vnf_id(vnfs)

    if first_id is None:
        return 0.0

    for vl in vls:
        if vl["from"] == first_id:
            return float(vl.get("bandwidth", 0.0))

    return 0.0


def _effective_vls_for_metrics(slice_data):
    # Return real VLs plus synthetic ENTRY -> first_vnf if entry exists.
    vnfs, vls, entry = _split_slice(slice_data)
    effective = list(vls)

    if entry is not None and vnfs:
        effective.insert(
            0,
            {
                "from": ENTRY_VNF_ID,
                "to": _first_vnf_id(vnfs),
                "bandwidth": _infer_entry_bandwidth(vnfs, vls),
            },
        )

    return effective


def _get_vlink_bw(slice_data, src, dst):
    # Retrieve bandwidth demand for a VL, including ENTRY leg.
    for vl in _effective_vls_for_metrics(slice_data):
        if str(vl["from"]) == str(src) and str(vl["to"]) == str(dst):
            return float(vl.get("bandwidth", 0.0))

    return None


def export_link_bw_load_to_rows(
    method_name,
    result_list,
    slices,
    num_slices,
    num_vnfs_per_slice,
    seed,
    timestamp_str,
    link_capacity_base,
):
    # Export bandwidth usage per physical link.
    rows = []
    bw_used = {}

    for res_idx, res in enumerate(result_list):
        if not hasattr(res, "routed_vls") or res.routed_vls is None:
            continue

        slice_idx_default = res_idx

        for vl_key, path_nodes in res.routed_vls.items():
            s_id, vnf_src, vnf_dst = _route_key_to_parts(vl_key)

            if not path_nodes or len(path_nodes) < 2:
                continue

            if s_id is None:
                s_id = slice_idx_default

            if s_id is None or int(s_id) >= len(slices):
                continue

            bw = _get_vlink_bw(slices[int(s_id)], vnf_src, vnf_dst)

            if bw is None:
                continue

            for u, v in zip(path_nodes[:-1], path_nodes[1:]):
                edge = (u, v) if (u, v) in link_capacity_base else (v, u)

                if edge not in link_capacity_base:
                    continue

                bw_used[edge] = bw_used.get(edge, 0.0) + float(bw)

    for (u, v), used in bw_used.items():
        cap = float(link_capacity_base.get((u, v), float("nan")))

        rows.append({
            "timestamp": timestamp_str,
            "method": method_name,
            "num_slices": num_slices,
            "num_vnfs_per_slice": num_vnfs_per_slice,
            "seed": seed,
            "u": u,
            "v": v,
            "bw_used": used,
            "bw_capacity": cap,
            "bw_utilization": used / cap if cap and cap > 0 else float("nan"),
        })

    return rows


def build_export_rows(
    method_results,
    slices,
    num_slices,
    num_vnfs_per_slice,
    seed,
    timestamp_str,
    node_capacity_base,
    link_capacity_base,
):
    # Build all export rows for heuristic and adapted results.
    records_routes = []
    records_node_hosting = []
    records_node_cpu = []
    records_link_bw = []

    for method_name, result_list in method_results.items():
        if not result_list:
            continue

        records_routes.extend(
            export_routes_to_rows(
                method_name=method_name,
                result_list=result_list,
                num_slices=num_slices,
                num_vnfs_per_slice=num_vnfs_per_slice,
                seed=seed,
                timestamp_str=timestamp_str,
            )
        )

        records_link_bw.extend(
            export_link_bw_load_to_rows(
                method_name=method_name,
                result_list=result_list,
                slices=slices,
                num_slices=num_slices,
                num_vnfs_per_slice=num_vnfs_per_slice,
                seed=seed,
                timestamp_str=timestamp_str,
                link_capacity_base=link_capacity_base,
            )
        )

        hosting_rows, cpu_rows = export_node_hosting_to_rows(
            method_name=method_name,
            result_list=result_list,
            slices=slices,
            num_slices=num_slices,
            num_vnfs_per_slice=num_vnfs_per_slice,
            seed=seed,
            timestamp_str=timestamp_str,
            node_capacity_base=node_capacity_base,
        )

        records_node_hosting.extend(hosting_rows)
        records_node_cpu.extend(cpu_rows)

    return {
        "routes": records_routes,
        "node_hosting": records_node_hosting,
        "node_cpu": records_node_cpu,
        "link_bw": records_link_bw,
    }