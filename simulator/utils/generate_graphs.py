import matplotlib.pyplot as plt
import networkx as nx
import os

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

    vnfs_per_node = {n: [] for n in G.nodes()}
    if hasattr(result, "placed_vnfs"):
        for vnf, n in result.placed_vnfs.items():
            vnfs_per_node[n].append(vnf)

    active_nodes = {n for n, lst in vnfs_per_node.items() if lst}
    node_colors = ["lightgreen" if n in active_nodes else "lightgray" for n in G.nodes()]

    plt.figure(figsize=(9, 7))
    nx.draw(G, pos, node_color=node_colors, node_size=800, with_labels=True, font_size=9)

    if not result or not hasattr(result, "routed_vls") or not result.routed_vls:
        print("No routed virtual links found in result.")
        plt.title(title)
        plt.show()
        return

    colors = plt.cm.tab10.colors
    for idx, ((src, dst), path) in enumerate(result.routed_vls.items()):
        color = colors[idx % len(colors)]
        edges_in_path = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        mid_idx = len(path) // 2
        mid_node = path[mid_idx]

        # Entry/Exit paths in dashed black
        if src == "ENTRY" or dst == "EXIT":
            nx.draw_networkx_edges(G, pos, edgelist=edges_in_path, edge_color="black",
                                   style="dashed", width=2, alpha=0.9)
            label = "ENTRY→first" if src == "ENTRY" else "last→EXIT"
            plt.text(pos[mid_node][0], pos[mid_node][1] + 0.08, label, color="black", fontsize=7)
        else:
            nx.draw_networkx_edges(G, pos, edgelist=edges_in_path,
                                   edge_color=[color], width=3, alpha=0.9)
            plt.text(pos[mid_node][0], pos[mid_node][1] + 0.05,
                     f"{src}->{dst}", color=color, fontsize=8)

        print(f"Path for VL {src}->{dst}: {path}")

    plt.title(title)
    plt.show()

def plot_all_routes(G, results, title="All Routed Paths (All Slices)"):
    import matplotlib.pyplot as plt
    import networkx as nx
    import random

    # --- Layout ---
    pos = nx.spring_layout(G, seed=42, k=0.8)
    plt.figure(figsize=(12, 9))

    # === Base topology (black background) ===
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color="white",
        edgecolors="black",
        node_size=850,
        linewidths=1.8,
        alpha=1.0,
    )
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color="black",
        width=1.2,
        style="solid",
        alpha=0.4,
    )
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold", font_color="black")

    # === Color palette (strong contrast) ===
    cmap = (
        list(plt.cm.tab10.colors)
        + list(plt.cm.Set2.colors)
        + list(plt.cm.Dark2.colors)
        + list(plt.cm.Paired.colors)
    )

    # === Plot each slice route ===
    color_idx = 0
    used_labels = set()
    label_offset_factor = 0.05

    for s_idx, res in enumerate(results):
        if not res or not hasattr(res, "routed_vls"):
            continue

        for (src, dst), path in res.routed_vls.items():
            edges_in_path = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
            color = cmap[color_idx % len(cmap)]
            color_idx += 1

            # Draw route edges over black background
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=edges_in_path,
                edge_color=[color],
                width=3.0,
                alpha=0.9,
            )

            # --- Label positioning ---
            mid_idx = len(path) // 2
            mid_node = path[mid_idx]
            x, y = pos[mid_node]
            offset_x = random.uniform(-0.04, 0.04)
            offset_y = label_offset_factor * ((color_idx % 5) - 2)
            label_text = f"S{s_idx + 1}: {src}->{dst}"

            if label_text not in used_labels:
                plt.text(
                    x + offset_x,
                    y + offset_y,
                    label_text,
                    color=color,
                    fontsize=8,
                    fontweight="bold",
                    bbox=dict(
                        facecolor="white",
                        alpha=0.8,
                        edgecolor="none",
                        boxstyle="round,pad=0.15",
                    ),
                )
                used_labels.add(label_text)

            print(f"[Slice {s_idx+1}] Path {src}->{dst}: {path}")

    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_cpu_usage(G, slices, method_results, results_dir):
    node_capacity = {n: G.nodes[n]["cpu"] for n in G.nodes}
    nodes = list(node_capacity.keys())

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.12
    colors = plt.cm.Set2.colors
    methods = list(method_results.keys())

    for idx, (method_name, result_list) in enumerate(method_results.items()):
        if not result_list:
            continue

        used_cpu = {n: 0 for n in nodes}

        # --- MILP special case ---
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

        # --- Plot bars ---
        usage = [used_cpu[n] for n in nodes]
        offset = (idx - len(methods) / 2) * bar_width
        positions = [i + offset for i in range(len(nodes))]
        ax.bar(positions, usage, width=bar_width, label=method_name, color=colors[idx % len(colors)])

    # --- Labels and layout ---
    ax.set_title("CPU Utilization per Node")
    ax.set_xlabel("Nodes")
    ax.set_ylabel("CPU (units)")
    ax.set_xticks(range(len(nodes)))
    ax.set_xticklabels(nodes)
    ax.legend()
    plt.grid(axis="y")

    # --- Save instead of show ---
    output_path = os.path.join(results_dir, "cpu_utilization.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved CPU utilization chart to {output_path}")

