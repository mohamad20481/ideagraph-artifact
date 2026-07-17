"""
analytics.py - Analytics computation and chart generation for IdeaGraph.

Uses plotly for interactive visualizations.
"""

from __future__ import annotations
import heapq
from typing import Any, Dict, List
from collections import Counter

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

import db

METHODOLOGY_LABELS = [
    "Empirical Study", "Theoretical Analysis", "System Design",
    "Dataset Creation", "Survey/Meta-Analysis", "Tool/Library",
    "Interdisciplinary Bridge",
]
NOVELTY_LABELS = ["Incremental", "Moderate", "Substantial"]


def compute_analytics(user_id: int) -> Dict[str, Any]:
    """Aggregate stats across all saved results for a user."""
    all_ideas = db.get_all_user_ideas(user_id)
    results_list = db.get_user_results(user_id)

    if not all_ideas:
        return {"total_ideas": 0, "total_runs": len(results_list)}

    # Quality scores
    quality_scores = [i.get("quality_score", 0) for i in all_ideas if i.get("quality_score")]

    # Domain stats
    domain_stats = {}
    for idea in all_ideas:
        topic = idea.get("_topic", "Unknown")[:50]
        if topic not in domain_stats:
            domain_stats[topic] = {"count": 0, "total_quality": 0, "max_quality": 0}
        domain_stats[topic]["count"] += 1
        q = idea.get("quality_score", 0)
        domain_stats[topic]["total_quality"] += q
        domain_stats[topic]["max_quality"] = max(domain_stats[topic]["max_quality"], q)
    for d in domain_stats.values():
        d["avg_quality"] = d["total_quality"] / d["count"] if d["count"] else 0

    # Methodology distribution
    method_counts = Counter(i.get("methodology_type", "unknown") for i in all_ideas)

    # Novelty distribution
    novelty_counts = Counter(i.get("novelty_level", "unknown") for i in all_ideas)

    # Top ideas
    top_ideas = heapq.nlargest(15, all_ideas, key=lambda x: x.get("quality_score", 0))

    # Strategy distribution
    strategy_counts = Counter(i.get("source_strategy", "?") for i in all_ideas)

    return {
        "total_ideas": len(all_ideas),
        "total_runs": len(results_list),
        "quality_scores": quality_scores,
        "quality_mean": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
        "quality_max": max(quality_scores) if quality_scores else 0,
        "quality_min": min(quality_scores) if quality_scores else 0,
        "domain_stats": domain_stats,
        "method_counts": dict(method_counts),
        "novelty_counts": dict(novelty_counts),
        "strategy_counts": dict(strategy_counts),
        "top_ideas": top_ideas,
    }


def build_quality_histogram(quality_scores: List[float]) -> "go.Figure":
    """Interactive histogram of idea quality scores."""
    fig = go.Figure(data=[go.Histogram(
        x=quality_scores, nbinsx=20,
        marker_color="#4CAF50", marker_line_color="#2E7D32", marker_line_width=1,
    )])
    fig.update_layout(
        title="Quality Score Distribution",
        xaxis_title="Quality Score", yaxis_title="Count",
        template="plotly_dark", height=350, margin=dict(t=40, b=40),
    )
    return fig


def build_domain_comparison(domain_stats: Dict[str, Dict]) -> "go.Figure":
    """Bar chart comparing domains by idea count and avg quality."""
    domains = list(domain_stats.keys())
    counts = [d["count"] for d in domain_stats.values()]
    avg_quals = [d["avg_quality"] for d in domain_stats.values()]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Idea Count", x=domains, y=counts,
        marker_color="#2196F3", yaxis="y",
    ))
    fig.add_trace(go.Scatter(
        name="Avg Quality", x=domains, y=avg_quals,
        mode="lines+markers", marker_color="#FF9800",
        yaxis="y2",
    ))
    fig.update_layout(
        title="Domain Comparison",
        yaxis=dict(title="Idea Count", side="left"),
        yaxis2=dict(title="Avg Quality", side="right", overlaying="y", range=[0, 1]),
        template="plotly_dark", height=350, margin=dict(t=40, b=80),
        legend=dict(x=0, y=1.1, orientation="h"),
    )
    return fig


def build_methodology_heatmap(all_ideas: List[Dict]) -> "go.Figure":
    """Heatmap of idea counts by methodology × novelty."""
    from models.idea import (
        METHODOLOGY_TYPES, NOVELTY_LEVELS,
        METHODOLOGY_TYPE_TO_IDX, NOVELTY_LEVEL_TO_IDX,
    )

    grid = [[0] * len(NOVELTY_LEVELS) for _ in METHODOLOGY_TYPES]
    for idea in all_ideas:
        r = METHODOLOGY_TYPE_TO_IDX.get(idea.get("methodology_type", ""))
        c = NOVELTY_LEVEL_TO_IDX.get(idea.get("novelty_level", ""))
        if r is not None and c is not None:
            grid[r][c] += 1

    fig = go.Figure(data=go.Heatmap(
        z=grid, x=NOVELTY_LABELS, y=METHODOLOGY_LABELS,
        colorscale="Viridis", text=grid, texttemplate="%{text}",
        textfont={"size": 14},
    ))
    fig.update_layout(
        title="Ideas by Methodology × Novelty",
        template="plotly_dark", height=400, margin=dict(t=40, b=40, l=160),
    )
    return fig


def build_strategy_pie(strategy_counts: Dict[str, int]) -> "go.Figure":
    """Pie chart of generation strategy distribution."""
    labels = {"A": "Frontier Extension", "B": "Cross-Cluster", "C": "Gap-Filling",
              "consensus": "Consensus", "?": "Unknown"}
    fig = go.Figure(data=[go.Pie(
        labels=[labels.get(k, k) for k in strategy_counts.keys()],
        values=list(strategy_counts.values()),
        hole=0.4,
    )])
    fig.update_layout(
        title="Generation Strategy Distribution",
        template="plotly_dark", height=350, margin=dict(t=40, b=40),
    )
    return fig


# ============================================================================
# New Analytics Charts (v2)
# ============================================================================

def build_probe_failure_chart(all_ideas: List[Dict]) -> "go.Figure":
    """Stacked bar chart showing which probes fail most often."""
    if not HAS_PLOTLY:
        return None
    probes = ["code", "dataset", "constraint", "novelty"]
    pass_counts = {p: 0 for p in probes}
    fail_counts = {p: 0 for p in probes}

    for idea in all_ideas:
        scores = idea.get("probe_scores", {})
        for p in probes:
            score = scores.get(p, 0.5)
            if isinstance(score, (int, float)):
                if score >= 0.4:
                    pass_counts[p] += 1
                else:
                    fail_counts[p] += 1

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Pass", x=probes, y=[pass_counts[p] for p in probes], marker_color="#2ecc71"))
    fig.add_trace(go.Bar(name="Fail", x=probes, y=[fail_counts[p] for p in probes], marker_color="#e74c3c"))
    fig.update_layout(
        barmode="stack", title="Probe Pass/Fail Breakdown",
        template="plotly_dark", height=350, margin=dict(t=40, b=40),
    )
    return fig


def build_quality_over_iterations(iteration_events: List[Dict]) -> "go.Figure":
    """Line chart showing quality trend over iterations."""
    if not HAS_PLOTLY or not iteration_events:
        return None
    iters = [e.get("iteration", i + 1) for i, e in enumerate(iteration_events)]
    means = [e.get("quality_mean", 0) for e in iteration_events]
    maxes = [e.get("quality_max", 0) for e in iteration_events]
    coverages = [e.get("coverage", 0) for e in iteration_events]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=iters, y=means, mode="lines+markers", name="Mean Quality", line=dict(color="#3498db", width=3)))
    fig.add_trace(go.Scatter(x=iters, y=maxes, mode="lines+markers", name="Max Quality", line=dict(color="#2ecc71", width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=iters, y=coverages, mode="lines+markers", name="Coverage", line=dict(color="#e67e22", width=2), yaxis="y2"))
    fig.update_layout(
        title="Quality & Coverage Over Iterations",
        template="plotly_dark", height=350, margin=dict(t=40, b=40),
        yaxis=dict(title="Quality (0-1)"),
        yaxis2=dict(title="Coverage (0-1)", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.1),
    )
    return fig


def build_cost_breakdown(stats: Dict) -> "go.Figure":
    """Pie chart of cost breakdown by category."""
    if not HAS_PLOTLY:
        return None
    labels = ["Ideation", "Experiment", "Code", "Execution", "Analysis", "Paper", "Review"]
    # Estimate costs from stats (proportional to iterations and stages)
    total = stats.get("estimated_cost_usd", 0.01)
    values = [total * f for f in [0.30, 0.10, 0.15, 0.05, 0.10, 0.15, 0.15]]

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.4)])
    fig.update_layout(
        title=f"Estimated Cost Breakdown (${total:.3f} total)",
        template="plotly_dark", height=350, margin=dict(t=40, b=40),
    )
    return fig


def build_novelty_distribution(all_ideas: List[Dict]) -> "go.Figure":
    """Histogram of novelty levels across ideas."""
    if not HAS_PLOTLY:
        return None
    levels = [idea.get("novelty_level", "unknown") for idea in all_ideas]
    counts = Counter(levels)

    colors = {"incremental": "#3498db", "moderate": "#e67e22", "substantial": "#2ecc71", "unknown": "#95a5a6"}
    fig = go.Figure(data=[go.Bar(
        x=list(counts.keys()),
        y=list(counts.values()),
        marker_color=[colors.get(k, "#95a5a6") for k in counts.keys()],
    )])
    fig.update_layout(
        title="Novelty Level Distribution",
        template="plotly_dark", height=300, margin=dict(t=40, b=40),
    )
    return fig


def build_idea_radar(idea: Dict, title: str = "") -> "go.Figure":
    """10-dimension radar chart showing idea strengths across all probe dimensions."""
    if not HAS_PLOTLY:
        return None

    scores = idea.get("probe_scores", {})
    # Full 10-dimension radar (matches execution_critic's 10 probes)
    dim_map = [
        ("Code", "code"), ("Dataset", "dataset"), ("Compute", "constraint"),
        ("Novelty", "novelty"), ("Specificity", "specificity"),
        ("Significance", "significance"), ("Clarity", "clarity"),
        ("Testability", "testability"), ("Scalability", "scalability"),
        ("Risk Balance", "risk_balance"),
    ]
    categories = [d[0] for d in dim_map]
    values = [scores.get(d[1], 0.5) for d in dim_map]

    # Close the radar polygon
    values.append(values[0])
    categories.append(categories[0])

    fig = go.Figure(data=go.Scatterpolar(
        r=values, theta=categories, fill="toself",
        line=dict(color="#3498db", width=2),
        fillcolor="rgba(52, 152, 219, 0.25)",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=title or idea.get("title", "Idea")[:40],
        template="plotly_dark", height=350, margin=dict(t=40, b=20),
        showlegend=False,
    )
    return fig


def build_ideas_comparison_radar(ideas: List[Dict]) -> "go.Figure":
    """Overlaid radar charts comparing multiple ideas."""
    if not HAS_PLOTLY or not ideas:
        return None

    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]
    categories = ["Code", "Dataset", "Constraint", "Novelty", "Quality"]

    fig = go.Figure()
    for i, idea in enumerate(ideas[:5]):
        scores = idea.get("probe_scores", {})
        values = [
            scores.get("code", 0.5), scores.get("dataset", 0.5),
            scores.get("constraint", 0.5), scores.get("novelty", 0.5),
            idea.get("quality_score", 0.5),
        ]
        values.append(values[0])
        fig.add_trace(go.Scatterpolar(
            r=values, theta=categories + [categories[0]],
            fill="toself", name=idea.get("title", f"Idea {i+1}")[:30],
            line=dict(color=colors[i % len(colors)], width=2),
            fillcolor=f"rgba({int(colors[i % len(colors)][1:3], 16)},{int(colors[i % len(colors)][3:5], 16)},{int(colors[i % len(colors)][5:7], 16)},0.1)",
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Idea Comparison (Radar)",
        template="plotly_dark", height=400, margin=dict(t=50, b=20),
    )
    return fig


def build_quality_ranking_bar(ideas: List[Dict], top_n: int = 15) -> "go.Figure":
    """Horizontal bar chart ranking ideas by quality with color-coded probes."""
    if not HAS_PLOTLY or not ideas:
        return None

    sorted_ideas = sorted(ideas, key=lambda x: x.get("quality_score", 0), reverse=True)[:top_n]

    titles = [f"{i.get('title', '?')[:35]}" for i in sorted_ideas]
    qualities = [i.get("quality_score", 0) for i in sorted_ideas]
    colors = ["#27ae60" if q >= 0.7 else "#f39c12" if q >= 0.4 else "#e74c3c" for q in qualities]

    fig = go.Figure(data=go.Bar(
        x=qualities, y=titles, orientation="h",
        marker_color=colors, text=[f"{q:.2f}" for q in qualities],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"Top {top_n} Ideas by Quality",
        template="plotly_dark", height=max(300, top_n * 30),
        margin=dict(l=250, t=40, b=20),
        xaxis=dict(range=[0, 1.1], title="Quality Score"),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def build_strategy_success_heatmap(all_ideas: List[Dict]) -> "go.Figure":
    """Heatmap of success rate by strategy × novelty level."""
    if not HAS_PLOTLY:
        return None
    from models.idea import NOVELTY_LEVELS, NOVELTY_LEVEL_TO_IDX
    strategies = ["A", "B", "C"]
    strategy_to_idx = {"A": 0, "B": 1, "C": 2}

    grid = [[0.0] * len(NOVELTY_LEVELS) for _ in strategies]
    counts = [[0] * len(NOVELTY_LEVELS) for _ in strategies]

    for idea in all_ideas:
        r = strategy_to_idx.get(idea.get("source_strategy", "?"))
        c = NOVELTY_LEVEL_TO_IDX.get(idea.get("novelty_level", ""))
        if r is not None and c is not None:
            q = idea.get("quality_score", 0)
            grid[r][c] += q
            counts[r][c] += 1

    # Average
    for r in range(len(strategies)):
        for c in range(len(NOVELTY_LEVELS)):
            if counts[r][c] > 0:
                grid[r][c] = round(grid[r][c] / counts[r][c], 3)

    strat_labels = ["A: Frontier", "B: Bridge", "C: Gap-Fill"]
    fig = go.Figure(data=go.Heatmap(
        z=grid, x=NOVELTY_LABELS, y=strat_labels,
        colorscale="RdYlGn", text=[[f"{v:.2f}" for v in row] for row in grid],
        texttemplate="%{text}", textfont={"size": 14},
        zmin=0, zmax=1,
    ))
    fig.update_layout(
        title="Average Quality: Strategy × Novelty",
        template="plotly_dark", height=300, margin=dict(t=40, b=40, l=120),
    )
    return fig
