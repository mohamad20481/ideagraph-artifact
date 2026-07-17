"""
graph_explorer.py - Interactive knowledge graph visualization for IdeaGraph.

Uses Plotly for interactive 2D graph with:
  - Nodes = papers (sized by citation count, colored by cluster)
  - Edges = citation/reference relationships
  - Hover tooltips with paper details
  - Cluster highlighting
  - Frontier node markers
  - Idea overlay (show where generated ideas connect to papers)
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# Cluster colors (up to 12 distinct)
CLUSTER_COLORS = [
    "#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6",
    "#1abc9c", "#e67e22", "#34495e", "#16a085", "#c0392b",
    "#8e44ad", "#2980b9",
]


def _force_layout(nodes: List[str], edges: List[Tuple[str, str]],
                  iterations: int = 50) -> Dict[str, Tuple[float, float]]:
    """
    Fruchterman-Reingold force-directed layout, optimized for the
    graph-explorer use case (≤ ~300 nodes).

    Optimizations vs. the naive version:
      - Index nodes as parallel float lists (px, py, dx, dy) instead of
        dict-of-tuples — dict lookups dominated the inner O(N²) loop.
      - Adaptive iteration cap: large graphs stabilise visually in fewer
        passes, so we scale iterations down to keep wall time bounded.
      - Hoist `k*k` and `1/k` constants out of the hot loop.
      - Edge endpoints resolved to indices ONCE outside the iteration loop
        instead of doing dict lookups (`pos[a]`, `pos[b]`) per iteration.
    """
    n = len(nodes)
    if n == 0:
        return {}
    # Stable index mapping; positions stored as parallel lists for fast access.
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    px = [random.uniform(-1, 1) for _ in range(n)]
    py = [random.uniform(-1, 1) for _ in range(n)]

    if n <= 1:
        return {nodes[0]: (px[0], py[0])} if n == 1 else {}

    # Adaptive iteration cap: with n nodes, repulsion is O(n²) per iter,
    # so a hard product cap keeps wall time bounded for large graphs.
    # 50 nodes → 50 iters; 100 → ~30; 200 → ~15.
    iterations = max(15, min(iterations, 75_000 // (n * n) if n * n > 0 else iterations))

    # Pre-resolve edge endpoints to indices once (not per iteration).
    edge_idx_pairs: List[Tuple[int, int]] = []
    for a, b in edges:
        ai = node_to_idx.get(a)
        bi = node_to_idx.get(b)
        if ai is not None and bi is not None and ai != bi:
            edge_idx_pairs.append((ai, bi))

    area = 4.0
    k = math.sqrt(area / n)  # optimal distance
    k_sq = k * k
    inv_k = 1.0 / k
    sqrt = math.sqrt  # local reference (slightly faster than attr lookup)

    dx_buf = [0.0] * n
    dy_buf = [0.0] * n

    for it in range(iterations):
        # Reset displacement buffers in-place (avoid allocating new lists).
        for i in range(n):
            dx_buf[i] = 0.0
            dy_buf[i] = 0.0

        # Repulsion (all pairs).
        for i in range(n):
            pix = px[i]
            piy = py[i]
            for j in range(i + 1, n):
                ddx = pix - px[j]
                ddy = piy - py[j]
                dist_sq = ddx * ddx + ddy * ddy
                if dist_sq < 1e-4:
                    dist_sq = 1e-4
                dist = sqrt(dist_sq)
                # force = k_sq / dist → applied as (ddx/dist) * force
                # = ddx * k_sq / dist_sq. Skip the explicit normalization step.
                factor = k_sq / dist_sq
                fx = ddx * factor
                fy = ddy * factor
                dx_buf[i] += fx
                dy_buf[i] += fy
                dx_buf[j] -= fx
                dy_buf[j] -= fy

        # Attraction (edges).
        for ai, bi in edge_idx_pairs:
            ddx = px[ai] - px[bi]
            ddy = py[ai] - py[bi]
            dist_sq = ddx * ddx + ddy * ddy
            if dist_sq < 1e-4:
                dist_sq = 1e-4
            dist = sqrt(dist_sq)
            # force = dist * dist / k = dist_sq / k → applied as (ddx/dist) * force
            # = ddx * dist / k.
            factor = dist * inv_k
            fx = ddx * factor
            fy = ddy * factor
            dx_buf[ai] -= fx
            dy_buf[ai] -= fy
            dx_buf[bi] += fx
            dy_buf[bi] += fy

        # Apply with cooling.
        temp = 1.0 - (it / iterations)
        max_move = temp * 0.1
        for i in range(n):
            dxi = dx_buf[i]
            dyi = dy_buf[i]
            dist = sqrt(dxi * dxi + dyi * dyi)
            if dist < 1e-4:
                continue
            move = max_move if dist > max_move else dist
            scale = move / dist
            px[i] += dxi * scale
            py[i] += dyi * scale

    return {nodes[i]: (px[i], py[i]) for i in range(n)}


def build_dag_figure(dag_summary: Dict[str, Any],
                     ideas: List[Dict] = None,
                     width: int = 800, height: int = 600) -> Optional["go.Figure"]:
    """
    Build an interactive Plotly network graph from the DAG summary.

    Args:
        dag_summary: from results["dag_summary"]
        ideas: list of generated ideas (overlaid as star nodes)
    """
    if not HAS_PLOTLY:
        return None

    nodes_data = dag_summary.get("nodes", [])
    edges_data = dag_summary.get("edges", [])
    clusters = dag_summary.get("clusters", {})

    if not nodes_data:
        return None

    # Build node index
    node_ids = [n.get("id", n.get("paper_id", f"n{i}")) for i, n in enumerate(nodes_data)]
    node_map = {nid: i for i, nid in enumerate(node_ids)}

    # Layout
    edge_pairs = []
    for e in edges_data:
        src = e.get("source", e.get("from", ""))
        tgt = e.get("target", e.get("to", ""))
        if src in node_map and tgt in node_map:
            edge_pairs.append((src, tgt))

    positions = _force_layout(node_ids, edge_pairs, iterations=60)

    # Edge traces
    edge_x, edge_y = [], []
    for src, tgt in edge_pairs:
        x0, y0 = positions.get(src, (0, 0))
        x1, y1 = positions.get(tgt, (0, 0))
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.5, color="rgba(150, 150, 150, 0.3)"),
        hoverinfo="none",
    )

    # Node traces
    node_x = [positions.get(nid, (0, 0))[0] for nid in node_ids]
    node_y = [positions.get(nid, (0, 0))[1] for nid in node_ids]

    # Node attributes
    node_sizes = []
    node_colors = []
    node_texts = []
    hover_texts = []

    for i, n in enumerate(nodes_data):
        title = n.get("title", "Unknown")[:60]
        year = n.get("year", "?")
        citations = n.get("citation_count", n.get("citationCount", 0))
        cluster_id = n.get("cluster_id", 0)
        is_frontier = n.get("is_frontier", False)

        # Size by citations (log scale)
        size = max(8, min(30, 8 + math.log(max(citations, 1)) * 3))
        node_sizes.append(size)

        # Color by cluster
        color_idx = cluster_id % len(CLUSTER_COLORS) if cluster_id is not None else 0
        node_colors.append(CLUSTER_COLORS[color_idx])

        # Text (short label for large nodes)
        if citations > 50:
            node_texts.append(title[:20])
        else:
            node_texts.append("")

        # Hover
        frontier_tag = " [FRONTIER]" if is_frontier else ""
        hover_texts.append(
            f"<b>{title}</b><br>"
            f"Year: {year}<br>"
            f"Citations: {citations}<br>"
            f"Cluster: {cluster_id}{frontier_tag}"
        )

    # Frontier markers (border)
    frontier_lines = []
    for i, n in enumerate(nodes_data):
        if n.get("is_frontier"):
            frontier_lines.append(2)
        else:
            frontier_lines.append(0)

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=node_texts, textposition="top center",
        textfont=dict(size=8, color="white"),
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=frontier_lines, color="white"),
            opacity=0.9,
        ),
        hovertext=hover_texts,
        hoverinfo="text",
    )

    traces = [edge_trace, node_trace]

    # Overlay ideas as star markers
    if ideas:
        idea_x, idea_y = [], []
        idea_hover = []
        for idea in ideas[:15]:
            # Place ideas near random existing nodes
            if node_ids:
                anchor = random.choice(node_ids)
                ax, ay = positions.get(anchor, (0, 0))
                idea_x.append(ax + random.uniform(-0.15, 0.15))
                idea_y.append(ay + random.uniform(-0.15, 0.15))
            else:
                idea_x.append(random.uniform(-1, 1))
                idea_y.append(random.uniform(-1, 1))

            q = idea.get("quality_score", 0)
            idea_hover.append(
                f"<b>IDEA: {idea.get('title', '?')[:50]}</b><br>"
                f"Quality: {q:.2f}<br>"
                f"Type: {(idea.get('methodology_type') or '?').replace('_', ' ').title()}"
            )

        idea_trace = go.Scatter(
            x=idea_x, y=idea_y, mode="markers",
            marker=dict(
                size=14, color="#f1c40f",
                symbol="star", line=dict(width=1, color="white"),
            ),
            hovertext=idea_hover, hoverinfo="text",
            name="Generated Ideas",
        )
        traces.append(idea_trace)

    fig = go.Figure(data=traces)

    # Cluster legend
    unique_clusters = sorted(set(n.get("cluster_id", 0) for n in nodes_data if n.get("cluster_id") is not None))
    for cid in unique_clusters[:10]:
        cluster_meta = clusters.get(str(cid), clusters.get(cid, {}))
        theme = cluster_meta.get("theme", f"Cluster {cid}")[:30] if isinstance(cluster_meta, dict) else f"Cluster {cid}"
        color_idx = cid % len(CLUSTER_COLORS)
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=10, color=CLUSTER_COLORS[color_idx]),
            name=theme, showlegend=True,
        ))

    if ideas:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=12, color="#f1c40f", symbol="star"),
            name="Generated Ideas", showlegend=True,
        ))

    fig.update_layout(
        title="Knowledge DAG Explorer",
        template="plotly_dark",
        width=width, height=height,
        showlegend=True,
        legend=dict(
            orientation="h", y=-0.1,
            font=dict(size=10),
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=40, b=60),
        hovermode="closest",
    )

    return fig


def build_idea_connection_graph(ideas: List[Dict], width: int = 700, height: int = 500) -> Optional["go.Figure"]:
    """
    Build a graph showing relationships between generated ideas.
    Connects ideas with similar methodology types or overlapping keywords.
    """
    if not HAS_PLOTLY or len(ideas) < 2:
        return None

    n = min(len(ideas), 20)
    selected = sorted(ideas, key=lambda x: x.get("quality_score", 0), reverse=True)[:n]

    # Build edges: connect ideas with same methodology type
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if selected[i].get("methodology_type") == selected[j].get("methodology_type"):
                edges.append((i, j))
            elif selected[i].get("novelty_level") == selected[j].get("novelty_level"):
                if random.random() < 0.3:
                    edges.append((i, j))

    # Layout
    node_ids = list(range(n))
    edge_pairs = [(str(a), str(b)) for a, b in edges]
    positions = _force_layout([str(i) for i in node_ids], edge_pairs, iterations=40)

    # Edges
    edge_x, edge_y = [], []
    for a, b in edges:
        x0, y0 = positions.get(str(a), (0, 0))
        x1, y1 = positions.get(str(b), (0, 0))
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=1, color="rgba(52, 152, 219, 0.3)"),
        hoverinfo="none",
    ))

    # Nodes
    node_x = [positions.get(str(i), (0, 0))[0] for i in range(n)]
    node_y = [positions.get(str(i), (0, 0))[1] for i in range(n)]
    node_sizes = [max(12, idea.get("quality_score", 0.5) * 30) for idea in selected]
    node_colors = [idea.get("quality_score", 0.5) for idea in selected]
    hover = [
        f"<b>{idea.get('title', '?')[:50]}</b><br>"
        f"Quality: {idea.get('quality_score', 0):.2f}<br>"
        f"Type: {(idea.get('methodology_type') or '?').replace('_', ' ').title()}"
        for idea in selected
    ]

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers",
        marker=dict(
            size=node_sizes, color=node_colors,
            colorscale="RdYlGn", cmin=0, cmax=1,
            line=dict(width=1, color="white"),
            colorbar=dict(title="Quality"),
        ),
        hovertext=hover, hoverinfo="text",
    ))

    fig.update_layout(
        title="Idea Connection Graph",
        template="plotly_dark",
        width=width, height=height,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False,
    )

    return fig
