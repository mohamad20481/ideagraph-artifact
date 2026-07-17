"""
idea_evolution.py - Track how ideas evolve across pipeline runs.

Provides:
  - Version history for ideas (comparing across runs)
  - Quality trajectory visualization
  - Diff view between idea versions
  - Evolution timeline (Plotly)
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

import db


def get_idea_history(user_id: int) -> Dict[str, List[Dict]]:
    """
    Build idea evolution history from all saved runs.
    Groups ideas by similar titles across runs.
    Returns {normalized_title: [versions sorted by date]}.
    """
    # Single batched query (1 round-trip) instead of 1 + N (one per result).
    # Use db_cache wrapper so repeated tab-renders don't refetch.
    try:
        import db_cache  # type: ignore
        results_list = db_cache.get_user_results_full(user_id)
    except Exception:
        results_list = db.get_user_results_full(user_id)
    if not results_list:
        return {}

    # Collect all ideas across all runs
    all_ideas = []
    for r in results_list:
        run_date = (r.get("created_at") or "")[:16]
        topic = r.get("topic", "")
        run_id = r["id"]
        for idea in r.get("ideas", []):
            idea_copy = dict(idea)
            idea_copy["_run_date"] = run_date
            idea_copy["_run_topic"] = topic
            idea_copy["_run_id"] = run_id
            all_ideas.append(idea_copy)

    # Group by normalized title (lowercase, stripped)
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for idea in all_ideas:
        title = (idea.get("title") or "").strip().lower()
        if not title or len(title) < 5:
            continue
        # Fuzzy grouping: first 40 chars as key
        key = title[:40]
        groups[key].append(idea)

    # Sort each group by date
    for key in groups:
        groups[key].sort(key=lambda x: x.get("_run_date", ""))

    # Filter to groups with actual evolution (>= 1 entry)
    return {k: v for k, v in groups.items() if len(v) >= 1}


def compute_quality_trajectory(user_id: int) -> Dict[str, Any]:
    """
    Compute quality metrics trajectory across all runs.
    Returns data for timeline chart.
    """
    # Single batched query instead of 1 + N (one per result).
    results_list = db.get_user_results_full(user_id)
    if not results_list:
        return {"dates": [], "qualities": [], "coverages": [], "ideas_counts": []}

    dates = []
    qualities = []
    coverages = []
    ideas_counts = []
    topics = []

    for r in sorted(results_list, key=lambda x: x.get("created_at", "")):
        date = (r.get("created_at") or "")[:10]
        ideas = r.get("ideas", [])
        coverage = r.get("coverage", 0)
        stats = r.get("stats", {})
        mean_q = stats.get("quality_mean", 0)
        if not mean_q and ideas:
            qs = [i.get("quality_score", 0) for i in ideas]
            mean_q = sum(qs) / len(qs) if qs else 0

        dates.append(date)
        qualities.append(mean_q)
        coverages.append(coverage)
        ideas_counts.append(len(ideas))
        topics.append((r.get("topic") or "")[:30])

    return {
        "dates": dates,
        "qualities": qualities,
        "coverages": coverages,
        "ideas_counts": ideas_counts,
        "topics": topics,
    }


def build_quality_trajectory_chart(user_id: int) -> Optional["go.Figure"]:
    """Build a Plotly timeline of quality evolution across runs."""
    if not HAS_PLOTLY:
        return None

    data = compute_quality_trajectory(user_id)
    if not data["dates"] or len(data["dates"]) < 2:
        return None

    fig = go.Figure()

    # Quality line
    fig.add_trace(go.Scatter(
        x=data["dates"], y=data["qualities"],
        mode="lines+markers", name="Mean Quality",
        line=dict(color="#3498db", width=3),
        marker=dict(size=8),
        hovertext=[f"Topic: {t}" for t in data["topics"]],
    ))

    # Coverage line
    fig.add_trace(go.Scatter(
        x=data["dates"], y=data["coverages"],
        mode="lines+markers", name="Coverage",
        line=dict(color="#2ecc71", width=2, dash="dash"),
        marker=dict(size=6),
        yaxis="y2",
    ))

    # Ideas count as bar
    fig.add_trace(go.Bar(
        x=data["dates"], y=data["ideas_counts"],
        name="Ideas Count",
        marker_color="rgba(243, 156, 18, 0.4)",
        yaxis="y3",
    ))

    fig.update_layout(
        title="Research Quality Evolution Over Time",
        template="plotly_dark",
        height=400,
        xaxis=dict(title="Date"),
        yaxis=dict(title="Quality (0-1)", range=[0, 1], side="left"),
        yaxis2=dict(title="Coverage", overlaying="y", side="right", range=[0, 1]),
        yaxis3=dict(overlaying="y", visible=False),
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=50, r=50, t=50, b=40),
    )

    return fig


def build_idea_versions_chart(versions: List[Dict]) -> Optional["go.Figure"]:
    """Build a chart showing quality evolution of a specific idea across versions."""
    if not HAS_PLOTLY or len(versions) < 2:
        return None

    dates = [v.get("_run_date", "")[:10] for v in versions]
    qualities = [v.get("quality_score", 0) for v in versions]
    titles = [v.get("title", "?")[:40] for v in versions]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(versions) + 1)),
        y=qualities,
        mode="lines+markers+text",
        text=[f"v{i+1}" for i in range(len(versions))],
        textposition="top center",
        line=dict(color="#e74c3c", width=3),
        marker=dict(size=12, color=qualities, colorscale="RdYlGn", cmin=0, cmax=1),
        hovertext=[f"{t}<br>Date: {d}<br>Quality: {q:.3f}" for t, d, q in zip(titles, dates, qualities)],
        hoverinfo="text",
    ))

    fig.update_layout(
        title=f"Idea Evolution: {titles[0][:40]}",
        template="plotly_dark",
        height=300,
        xaxis=dict(title="Version"),
        yaxis=dict(title="Quality", range=[0, 1]),
        margin=dict(l=50, r=20, t=50, b=40),
    )
    return fig


def diff_idea_versions(old: Dict, new: Dict) -> Dict[str, Dict[str, str]]:
    """Compute a diff between two idea versions."""
    fields = ["title", "motivation", "method", "hypothesis",
              "resources", "expected_outcome", "risk_assessment"]
    diff = {}
    for field in fields:
        old_val = str(old.get(field, ""))[:300]
        new_val = str(new.get(field, ""))[:300]
        if old_val != new_val:
            diff[field] = {"old": old_val, "new": new_val}
    # Quality change
    old_q = old.get("quality_score", 0)
    new_q = new.get("quality_score", 0)
    if old_q != new_q:
        diff["quality_score"] = {
            "old": f"{old_q:.3f}",
            "new": f"{new_q:.3f}",
            "delta": f"{new_q - old_q:+.3f}",
        }
    return diff
