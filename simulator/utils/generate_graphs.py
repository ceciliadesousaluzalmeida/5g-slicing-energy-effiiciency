import matplotlib.pyplot as plt
import networkx as nx
import os

# ------------------------------------------------------------
# Heuristic solution visualization
# ------------------------------------------------------------
def plot_solution_heuristic(G, result, title="Heuristic Solution", pos=None):
    """
    Visualize the solution of a heuristic algorithm.
    - Active nodes in green if hosting at least one VNF
    - Each routed virtual link path drawn in a distinct color
    - Entry/exit legs drawn as dashed black lines
    - Node labels show VNFs placed
    - Path details printed in console
    """
    if pos is None:
        pos = nx.spring_layout(G, seed=42)

    # Collect VNFs per node
    vnfs_per_node = {n: [] for n in G.nodes()}
    if hasattr(result, "placed_vnfs"):
        for vnf, n in result.placed_vnfs.items():
            vnfs_per_node[n].append(vnf)

    # Active nodes in green
    active_nodes = {n for n, lst in vnfs_per_node.items() if lst}
    node_colors = ["#8FD694" if n in active_nodes else "#D3D3D3" for n in G.nodes()]

    plt.figure(figsize=(10, 8))
    nx.draw(
        G, pos,
        node_color=node_colors,
        node_size=950,
        with_labels=True,
        font_size=10,
        edge_color="#B0B0B0",
        width=1.2
    )

    # No routed links
    if not result or not hasattr(result, "routed_vls") or not result.routed_vls:
        print("No routed virtual links found in result.")
        plt.title(title, fontsize=14, pad=15)
        plt.tight_layout()
        plt.show()
        return

    colors = plt.cm.get_cmap("tab10", 10)
    for idx, ((src, dst), path) in enumerate(result.routed_vls.items()):
        color = colors(idx)
        edges_in_path = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        mid_idx = len(path) // 2
        mid_node = path[mid_idx]

        # Entry/Exit paths in dashed black
        if src == "ENTRY" or dst == "EXIT":
            nx.draw_networkx_edges(
                G, pos, edgelist=edges_in_path,
                edge_color="black", style="dashed", width=2.2, alpha=0.9
            )
            label = "ENTRY→first" if src == "ENTRY" else "last→EXIT"
            plt.text(
                pos[mid_node][0], pos[mid_node][1] + 0.1,
                label, color="black", fontsize=9, ha='center'
            )
        else:
            nx.draw_networkx_edges(
                G, pos, edgelist=edges_in_path,
                edge_color=[color], width=3.2, alpha=0.9
            )
            plt.text(
                pos[mid_node][0], pos[mid_node][1] + 0.06,
                f"{src}->{dst}", color=color, fontsize=9, ha='center'
            )

        print(f"Path for VL {src}->{dst}: {path}")

    plt.title(title, fontsize=14, pad=15)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# Combined routes visualization
# ------------------------------------------------------------
def plot_all_routes(G, results, title="All Routes", results_dir="./results", label_offset=0.08):
    """
    Plot all routed paths from a list of heuristic results and save the figure in results_dir.
    Labels are stacked per edge and offset perpendicularly to avoid overlaps.
    """
    import os
    import math
    import matplotlib.pyplot as plt
    import networkx as nx

    # --- Helpers (all local to keep function self-contained) ---
    def _midpoint(pu, pv):
        return ((pu[0] + pv[0]) / 2.0, (pu[1] + pv[1]) / 2.0)

    def _unit_perp(pu, pv):
        """Return a unit vector perpendicular to (pv - pu)."""
        dx, dy = pv[0] - pu[0], pv[1] - pu[1]
        n = math.hypot(dx, dy) or 1.0
        return (-dy / n, dx / n)

    def _ekey(u, v):
        """Undirected edge key for stacking labels on the same link."""
        return (u, v) if u <= v else (v, u)

    os.makedirs(results_dir, exist_ok=True)
    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(11, 9))
    nx.draw(
        G, pos,
        node_color="#D3D3D3", edge_color="#A0A0A0",
        node_size=900, width=0.8, with_labels=True,
        font_size=10
    )

    colors = plt.cm.get_cmap("tab20", 20)
    color_idx = 0

    # --- Draw colored paths and collect label requests bucketed by edge ---
    edge_labels = {}  # (u,v)_undirected -> list of dicts(label,color,mid_edge)
    for s_idx, res in enumerate(results):
        if not res or not hasattr(res, "routed_vls"):
            continue

        for (src, dst), path in res.routed_vls.items():
            if not path or len(path) < 2:
                continue

            # Draw the path
            edges_in_path = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
            color = colors(color_idx % 20)
            nx.draw_networkx_edges(
                G, pos, edgelist=edges_in_path,
                edge_color=[color], width=2.8, alpha=0.9
            )
            color_idx += 1

            # Choose the central edge of the path for labeling
            mid_idx = (len(edges_in_path) - 1) // 2
            u_mid, v_mid = edges_in_path[mid_idx]

            k = _ekey(u_mid, v_mid)
            edge_labels.setdefault(k, []).append({
                "text": f"S{s_idx + 1}:{src}->{dst}",
                "color": color,
                "edge": (u_mid, v_mid)
            })

    # --- Place labels: stack per edge, offset perpendicular to the edge vector ---
    for (u, v), items in edge_labels.items():
        pu, pv = pos[u], pos[v]
        mx, my = _midpoint(pu, pv)
        px, py = _unit_perp(pu, pv)

        n = len(items)
        # symmetric offsets: e.g., n=1 -> [0], n=2 -> [-0.5, +0.5], n=3 -> [-1,0,+1]
        if n == 1:
            offsets = [0]
        else:
            # center the stack and keep spacing consistent
            base = list(range(-(n // 2), n // 2 + 1))
            offsets = base[:n]  # trim in case of odd/even mismatch

        for off, it in zip(offsets, items):
            ox = mx + off * label_offset * px
            oy = my + off * label_offset * py
            plt.text(
                ox, oy, it["text"],
                color=it["color"], fontsize=10, ha="center", va="center",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5)
            )

    plt.title(title, fontsize=14, pad=15)
    plt.tight_layout()

    filename = title.lower().replace(" ", "_") + ".png"
    output_path = os.path.join(results_dir, filename)
    plt.savefig(output_path, dpi=400)
    plt.close()
    print(f"[INFO] Saved route plot to {output_path}")


# ------------------------------------------------------------
# CPU usage comparison
# ------------------------------------------------------------
def plot_cpu_usage(G, slices, method_results, results_dir):
    node_capacity = {n: G.nodes[n]["cpu"] for n in G.nodes}
    nodes = list(node_capacity.keys())

    fig, ax = plt.subplots(figsize=(11, 6))
    bar_width = 0.12
    colors = plt.cm.get_cmap("Set2", len(method_results))
    methods = list(method_results.keys())

    for idx, (method_name, result_list) in enumerate(method_results.items()):
        if not result_list:
            continue

        used_cpu = {n: 0 for n in nodes}

        # MILP case
        if method_name.startswith("MILP"):
            milp_res = result_list[0]
            if hasattr(milp_res, "placed_vnfs"):
                for vnf_id, n in milp_res.placed_vnfs.items():
                    try:
                        vnf_cpu = next(vnf["cpu"] for s in slices for vnf in s[0] if vnf["id"] == vnf_id)
                        used_cpu[n] += vnf_cpu
                    except StopIteration:
                        continue
        else:
            for r in result_list:
                if r is None or not hasattr(r, "placed_vnfs"):
                    continue
                for vnf_id, n in r.placed_vnfs.items():
                    try:
                        vnf_cpu = next(vnf["cpu"] for s in slices for vnf in s[0] if vnf["id"] == vnf_id)
                        used_cpu[n] += vnf_cpu
                    except StopIteration:
                        continue

        # Plot bars
        usage = [used_cpu[n] for n in nodes]
        offset = (idx - len(methods) / 2) * bar_width
        positions = [i + offset for i in range(len(nodes))]
        ax.bar(
            positions, usage,
            width=bar_width,
            label=method_name,
            color=colors(idx % len(methods)),
            edgecolor="black",
            linewidth=0.5
        )

    # Labels and layout
    ax.set_title("CPU Utilization per Node", fontsize=14, pad=10)
    ax.set_xlabel("Nodes", fontsize=12)
    ax.set_ylabel("CPU (units)", fontsize=12)
    ax.set_xticks(range(len(nodes)))
    ax.set_xticklabels(nodes, fontsize=10)
    ax.legend(fontsize=10, ncol=2)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()

    output_path = os.path.join(results_dir, "cpu_utilization.png")
    plt.savefig(output_path, dpi=400)
    plt.close()
    print(f"[INFO] Saved CPU utilization chart to {output_path}")
