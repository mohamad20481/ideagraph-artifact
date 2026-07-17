"""
idea_simulator.py - Visual simulation for research ideas.

Given any idea, generate 4 interactive visualizations that let users SEE
what executing it would look like:

  1. MethodFlow      - auto-extracted stage diagram (input → method → output)
  2. OutcomeSim      - Monte Carlo simulation of expected results (1000 trials)
  3. Timeline        - Gantt chart of project phases with milestones
  4. ResourceGauge   - GPU-hours / cost / data spend over time

All deterministic from idea fields — no extra LLM calls. Pure NumPy + Plotly.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# ─────────────────────────────────────────────────────────────────────────────
# 1. METHOD FLOW DIAGRAM
# ─────────────────────────────────────────────────────────────────────────────

# Common ML pipeline stage keywords → display node label
_STAGE_KEYWORDS = [
    (("dataset", "data", "corpus", "benchmark", "imagenet", "zinc"),    "📦 Data"),
    (("preprocess", "tokeniz", "normaliz", "augment", "clean"),          "🧹 Preprocess"),
    (("embed", "encod", "tokenizer", "feature extraction"),              "🔤 Embed"),
    (("train", "fit", "optimize", "gradient", "loss", "backprop"),       "🏋️ Train"),
    (("attention", "transformer", "neural network", "model", "gnn",
       "cnn", "rnn", "lstm", "bert"),                                     "🧠 Model"),
    (("evaluat", "valid", "test", "benchmark"),                          "📊 Evaluate"),
    (("compar", "baseline", "ablat"),                                    "⚖️ Compare"),
    (("analyz", "interpret", "visualiz", "metric"),                      "🔍 Analyze"),
    (("deploy", "production", "inference", "serv"),                      "🚀 Deploy"),
    (("publish", "paper", "report", "writ"),                             "📝 Publish"),
]


def extract_stages(idea: Dict[str, Any]) -> List[str]:
    """
    Extract a sequence of pipeline stages from an idea's method/hypothesis text.
    Returns 3-7 ordered stage labels detected via keyword matching.
    Always starts with 📦 Data and ends with 📊 Evaluate.
    """
    text = " ".join([
        idea.get("method", ""), idea.get("hypothesis", ""),
        idea.get("expected_outcome", ""),
    ]).lower()

    detected: List[Tuple[int, str]] = []  # (first occurrence position, label)
    seen = set()
    for keywords, label in _STAGE_KEYWORDS:
        for kw in keywords:
            pos = text.find(kw)
            if pos >= 0 and label not in seen:
                detected.append((pos, label))
                seen.add(label)
                break

    # Sort by first-mention position to preserve narrative order
    detected.sort(key=lambda x: x[0])
    stages = [label for _, label in detected]

    # Ensure we always have a sensible flow
    if not stages:
        stages = ["📦 Data", "🧠 Model", "📊 Evaluate"]

    # Force canonical order: Data first, Evaluate last
    if "📦 Data" in stages:
        stages.remove("📦 Data")
    stages.insert(0, "📦 Data")

    if "📊 Evaluate" in stages:
        stages.remove("📊 Evaluate")
    stages.append("📊 Evaluate")

    # Cap at 7 stages
    return stages[:7]


def build_method_flow(idea: Dict[str, Any]):
    """
    Build a horizontal flowchart of the idea's method as a Plotly figure.
    Returns None if Plotly unavailable.
    """
    if not HAS_PLOTLY:
        return None

    stages = extract_stages(idea)
    n = len(stages)
    if n == 0:
        return None

    # Layout: equally-spaced horizontal nodes
    xs = list(range(n))
    ys = [0] * n

    fig = go.Figure()

    # Draw arrows (lines + arrowheads) between consecutive stages
    for i in range(n - 1):
        fig.add_annotation(
            x=xs[i + 1] - 0.05, y=ys[i + 1],
            ax=xs[i] + 0.05, ay=ys[i],
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.4, arrowwidth=2,
            arrowcolor="#0ea5e9",
        )

    # Stage nodes as text-marker scatter
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="markers+text",
        marker=dict(
            size=80,
            color="#f0f9ff",
            line=dict(color="#0ea5e9", width=2),
            symbol="square",
        ),
        text=stages,
        textposition="middle center",
        textfont=dict(size=14, color="#0c4a6e", family="Arial"),
        hoverinfo="text",
        hovertext=[f"Stage {i+1}: {s}" for i, s in enumerate(stages)],
    ))

    fig.update_layout(
        title=dict(text="🔄 Method Flow", x=0.5, font=dict(size=14)),
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False,
        plot_bgcolor="white",
        xaxis=dict(visible=False, range=[-0.5, n - 0.5]),
        yaxis=dict(visible=False, range=[-1, 1]),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. MONTE CARLO OUTCOME SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def simulate_outcomes(idea: Dict[str, Any], n_trials: int = 1000) -> Dict[str, Any]:
    """
    Run a deterministic Monte Carlo simulation of the idea's expected results.

    The mean is anchored on the idea's quality score; variance reflects
    novelty (high novelty = wider distribution = more uncertain).
    Returns:
      {
        "trials": [float, ...],           # raw simulated outcomes (0-100)
        "mean": float, "std": float,
        "p10": float, "p50": float, "p90": float,
        "success_pct": float,             # % trials > baseline
        "baseline": float,
      }
    """
    # Deterministic seed from idea title — same idea always simulates same way
    seed_hash = hashlib.md5(
        idea.get("title", "x").encode("utf-8"), usedforsecurity=False,
    ).hexdigest()
    rng = random.Random(int(seed_hash[:8], 16))

    quality = idea.get("quality_score", 0.5)
    novelty = idea.get("probe_scores", {}).get("novelty", 0.5)
    feasibility = (
        idea.get("probe_scores", {}).get("code", 0.5)
        + idea.get("probe_scores", {}).get("dataset", 0.5)
    ) / 2.0

    # Mean improvement % over baseline (anchored on quality)
    mean = 5.0 + 25.0 * quality              # 5%-30% range
    # Standard deviation grows with novelty (high novelty = high uncertainty)
    std = 2.0 + 8.0 * novelty                # 2-10 range
    # Baseline performance assumes feasibility makes a workable system
    baseline = 50.0 + 20.0 * feasibility     # 50-70 baseline

    trials = [max(0.0, min(100.0, rng.gauss(mean + baseline, std)))
              for _ in range(n_trials)]
    trials.sort()

    p10 = trials[int(n_trials * 0.1)]
    p50 = trials[int(n_trials * 0.5)]
    p90 = trials[int(n_trials * 0.9)]
    success_pct = sum(1 for t in trials if t > baseline) / n_trials * 100

    return {
        "trials": trials,
        "mean": sum(trials) / len(trials),
        "std": (sum((t - sum(trials) / len(trials)) ** 2 for t in trials) / len(trials)) ** 0.5,
        "p10": p10, "p50": p50, "p90": p90,
        "success_pct": success_pct,
        "baseline": baseline,
        "metric_unit": "%",
        "n_trials": n_trials,
    }


def build_outcome_distribution(idea: Dict[str, Any]):
    """Plotly histogram of Monte Carlo outcomes with percentile markers."""
    if not HAS_PLOTLY:
        return None
    sim = simulate_outcomes(idea)
    trials = sim["trials"]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=trials, nbinsx=30,
        marker=dict(color="#0ea5e9", line=dict(color="#0284c7", width=1)),
        opacity=0.85,
        name="Outcomes",
    ))

    # Vertical lines at p10, p50, p90, baseline
    for pct, color, label in [
        (sim["baseline"], "#94a3b8", "Baseline"),
        (sim["p10"], "#f59e0b", "P10 (worst-case)"),
        (sim["p50"], "#10b981", "P50 (median)"),
        (sim["p90"], "#0ea5e9", "P90 (best-case)"),
    ]:
        fig.add_vline(
            x=pct, line=dict(color=color, width=2, dash="dash"),
            annotation=dict(text=label, font=dict(size=10, color=color),
                            yanchor="top"),
        )

    fig.update_layout(
        title=dict(text=f"🎲 Monte Carlo Outcomes (n={sim['n_trials']})",
                   x=0.5, font=dict(size=14)),
        xaxis_title="Performance Metric (%)",
        yaxis_title="Frequency",
        height=320, margin=dict(l=40, r=20, t=50, b=40),
        showlegend=False,
        plot_bgcolor="white",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. PROJECT TIMELINE (Gantt chart)
# ─────────────────────────────────────────────────────────────────────────────

def build_timeline(idea: Dict[str, Any], total_weeks: Optional[int] = None):
    """
    Build a Gantt chart of project phases with weekly milestones.
    Auto-detects total weeks from resources field, defaults to 12.
    """
    if not HAS_PLOTLY:
        return None

    if total_weeks is None:
        # Try to extract from resources field
        resources = idea.get("resources", "").lower()
        m = re.search(r"(\d+)\s*(?:week|weeks)", resources)
        total_weeks = int(m.group(1)) if m else 12
        total_weeks = max(2, min(52, total_weeks))

    # Phase distribution: ~25% setup, 40% development, 25% experiments, 10% paper
    phases = [
        ("📚 Literature & Setup", 0,
         max(1, int(total_weeks * 0.20)), "#bae6fd"),
        ("🔨 Method Implementation", max(1, int(total_weeks * 0.20)),
         max(2, int(total_weeks * 0.55)), "#7dd3fc"),
        ("🧪 Experiments & Ablations", max(2, int(total_weeks * 0.55)),
         max(3, int(total_weeks * 0.85)), "#0ea5e9"),
        ("📝 Paper Writing", max(3, int(total_weeks * 0.85)),
         total_weeks, "#0284c7"),
    ]

    fig = go.Figure()
    for phase_name, start, end, color in phases:
        fig.add_trace(go.Bar(
            y=[phase_name], x=[end - start], base=start,
            orientation="h",
            marker=dict(color=color, line=dict(color="#0c4a6e", width=1)),
            text=f"Weeks {start+1}–{end}",
            textposition="inside",
            textfont=dict(size=11, color="white"),
            hoverinfo="text",
            hovertext=f"{phase_name}<br>Weeks {start+1}–{end} ({end-start} weeks)",
            name=phase_name,
        ))

    # Milestone markers
    milestones = [
        (max(1, int(total_weeks * 0.20)), "🏁 Setup done"),
        (max(2, int(total_weeks * 0.55)), "🎯 Method working"),
        (max(3, int(total_weeks * 0.85)), "📊 Results in"),
        (total_weeks, "🎓 Paper submitted"),
    ]
    for week, label in milestones:
        fig.add_annotation(
            x=week, y=-0.5, text=label,
            showarrow=True, arrowhead=2, arrowcolor="#f59e0b",
            font=dict(size=10, color="#92400e"),
            ax=0, ay=30,
        )

    fig.update_layout(
        title=dict(text=f"📅 Project Timeline ({total_weeks} weeks)",
                   x=0.5, font=dict(size=14)),
        xaxis=dict(title="Week", range=[-0.5, total_weeks + 0.5],
                   showgrid=True, gridcolor="#e0f2fe"),
        yaxis=dict(autorange="reversed"),
        height=280, margin=dict(l=160, r=30, t=50, b=60),
        showlegend=False,
        plot_bgcolor="white",
        barmode="overlay",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. RESOURCE CONSUMPTION GAUGES
# ─────────────────────────────────────────────────────────────────────────────

def estimate_resources(idea: Dict[str, Any]) -> Dict[str, Any]:
    """Heuristic estimate of resource consumption from the idea's resources text."""
    text = (idea.get("resources", "") + " " + idea.get("method", "")).lower()
    quality = idea.get("quality_score", 0.5)

    # Try to extract GPU-hours
    gpu_hours = 0
    m = re.search(r"(\d+(?:,\d{3})*)\s*(?:gpu[- ]?hours?|hours?)", text)
    if m:
        gpu_hours = int(m.group(1).replace(",", ""))
    else:
        # Heuristic from method complexity
        if any(w in text for w in ("transformer", "billion", "large", "foundation")):
            gpu_hours = 500
        elif any(w in text for w in ("a100", "h100", "tpu")):
            gpu_hours = 200
        elif any(w in text for w in ("gpu", "cuda")):
            gpu_hours = 50
        else:
            gpu_hours = 20

    # Cost estimate ($1.50/GPU-hour for A100 cloud)
    cost_usd = gpu_hours * 1.5

    # Data size
    data_gb = 0
    m = re.search(r"(\d+(?:\.\d+)?)\s*gb", text)
    if m:
        data_gb = float(m.group(1))
    else:
        if any(w in text for w in ("imagenet", "openwebtext", "c4", "common crawl")):
            data_gb = 200
        elif any(w in text for w in ("benchmark", "dataset")):
            data_gb = 10
        else:
            data_gb = 1

    # Time
    time_weeks = 12
    m = re.search(r"(\d+)\s*(?:week|weeks)", text)
    if m:
        time_weeks = int(m.group(1))

    return {
        "gpu_hours": gpu_hours,
        "cost_usd": round(cost_usd, 2),
        "data_gb": round(data_gb, 1),
        "time_weeks": time_weeks,
        "feasibility_score": (
            idea.get("probe_scores", {}).get("constraint", 0.5) * 100
        ),
    }


def build_resource_gauges(idea: Dict[str, Any]):
    """Build a row of 4 gauge indicators for resources."""
    if not HAS_PLOTLY:
        return None

    res = estimate_resources(idea)

    # Each gauge: (value, max, label, suffix, threshold_steps)
    gauges = [
        ("💻 GPU-Hours", res["gpu_hours"], 1000, "h",
         [(0, 100, "#bbf7d0"), (100, 500, "#fde68a"), (500, 1000, "#fecaca")]),
        ("💰 Cost (USD)", res["cost_usd"], 2000, "$",
         [(0, 200, "#bbf7d0"), (200, 1000, "#fde68a"), (1000, 2000, "#fecaca")]),
        ("💾 Data (GB)", res["data_gb"], 500, "GB",
         [(0, 50, "#bbf7d0"), (50, 200, "#fde68a"), (200, 500, "#fecaca")]),
        ("⏱️ Time (weeks)", res["time_weeks"], 52, "w",
         [(0, 12, "#bbf7d0"), (12, 30, "#fde68a"), (30, 52, "#fecaca")]),
    ]

    fig = go.Figure()
    for i, (label, value, max_val, suffix, steps) in enumerate(gauges):
        row = i // 2
        col = i % 2
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=value,
            domain={
                "x": [col * 0.52, col * 0.52 + 0.45],
                "y": [0.55 - row * 0.55, 1.0 - row * 0.55],
            },
            title={"text": label, "font": {"size": 12}},
            number={"suffix": suffix, "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, max_val]},
                "bar": {"color": "#0ea5e9"},
                "steps": [{"range": [s[0], s[1]], "color": s[2]} for s in steps],
                "threshold": {
                    "line": {"color": "#dc2626", "width": 3},
                    "thickness": 0.7,
                    "value": value,
                },
            },
        ))

    fig.update_layout(
        height=320, margin=dict(l=20, r=20, t=20, b=20),
        title=dict(text="📊 Resource Consumption", x=0.5, font=dict(size=14)),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# UNIFIED SIMULATION (returns all 4 figures + summary)
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(idea: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the full visual simulation suite.
    Returns dict with 4 Plotly figures + summary stats.
    """
    return {
        "method_flow": build_method_flow(idea),
        "outcome_dist": build_outcome_distribution(idea),
        "timeline": build_timeline(idea),
        "resources": build_resource_gauges(idea),
        "stages": extract_stages(idea),
        "outcome_stats": simulate_outcomes(idea, n_trials=200),  # less for summary
        "resource_stats": estimate_resources(idea),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. WHAT-IF: simulate with adjusted knobs
# ─────────────────────────────────────────────────────────────────────────────

def simulate_with_adjustments(
    idea: Dict[str, Any],
    compute_multiplier: float = 1.0,    # 0.25 = quarter compute, 4.0 = 4x compute
    data_quality_boost: float = 0.0,    # -0.3 to +0.3 added to dataset score
    novelty_bet: float = 0.0,           # -0.3 to +0.3 added to novelty (raises variance)
    n_trials: int = 1000,
) -> Dict[str, Any]:
    """
    Re-run Monte Carlo with adjusted parameters. Models the effect of:
    - Adding/removing compute → shifts the mean (more compute = better)
    - Better/worse data → shifts the mean
    - Higher novelty bet → widens the variance (high risk, high reward)
    """
    adjusted = dict(idea)
    base_quality = idea.get("quality_score", 0.5)

    # Compute boost: log-scale (diminishing returns)
    compute_effect = math.log2(max(0.1, compute_multiplier)) * 0.05
    # Data quality directly improves quality score
    new_quality = max(0.0, min(1.0, base_quality + data_quality_boost + compute_effect))
    adjusted["quality_score"] = new_quality

    # Novelty bet: shift novelty score
    new_probe = dict(idea.get("probe_scores", {}))
    new_probe["novelty"] = max(0.0, min(1.0, new_probe.get("novelty", 0.5) + novelty_bet))
    adjusted["probe_scores"] = new_probe

    sim = simulate_outcomes(adjusted, n_trials=n_trials)
    sim["adjustments"] = {
        "compute_multiplier": compute_multiplier,
        "data_quality_boost": data_quality_boost,
        "novelty_bet": novelty_bet,
        "effective_quality": new_quality,
    }
    return sim


def build_what_if_chart(
    baseline_idea: Dict[str, Any],
    adjustments: Dict[str, float],
):
    """
    Show baseline vs adjusted outcome distributions overlaid on one histogram.
    """
    if not HAS_PLOTLY:
        return None

    base = simulate_outcomes(baseline_idea, n_trials=500)
    adj = simulate_with_adjustments(
        baseline_idea,
        compute_multiplier=adjustments.get("compute_multiplier", 1.0),
        data_quality_boost=adjustments.get("data_quality_boost", 0.0),
        novelty_bet=adjustments.get("novelty_bet", 0.0),
        n_trials=500,
    )

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=base["trials"], nbinsx=25,
        marker=dict(color="#94a3b8", line=dict(color="#64748b", width=1)),
        opacity=0.55, name=f"Baseline (P50={base['p50']:.0f}%)",
    ))
    fig.add_trace(go.Histogram(
        x=adj["trials"], nbinsx=25,
        marker=dict(color="#0ea5e9", line=dict(color="#0284c7", width=1)),
        opacity=0.75, name=f"Adjusted (P50={adj['p50']:.0f}%)",
    ))
    delta = adj["p50"] - base["p50"]
    fig.update_layout(
        title=dict(
            text=f"🎚️ What-If: Δ median = {delta:+.1f}%",
            x=0.5, font=dict(size=14),
        ),
        xaxis_title="Performance Metric (%)",
        yaxis_title="Frequency",
        barmode="overlay",
        height=320,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", y=-0.15),
        plot_bgcolor="white",
    )
    return fig, {"baseline": base, "adjusted": adj, "delta_p50": delta}


# ─────────────────────────────────────────────────────────────────────────────
# 6. MULTI-IDEA OUTCOME COMPARISON (overlay 2-4 distributions)
# ─────────────────────────────────────────────────────────────────────────────

def build_multi_outcome_overlay(ideas: List[Dict[str, Any]]):
    """Overlay outcome distributions for multiple ideas on a single chart."""
    if not HAS_PLOTLY or not ideas:
        return None

    palette = ["#0ea5e9", "#10b981", "#f59e0b", "#a855f7", "#ef4444"]
    fig = go.Figure()

    summaries = []
    for i, idea in enumerate(ideas[:5]):
        sim = simulate_outcomes(idea, n_trials=400)
        title = idea.get("title", f"Idea {i+1}")[:35]
        color = palette[i % len(palette)]
        fig.add_trace(go.Histogram(
            x=sim["trials"], nbinsx=20,
            marker=dict(color=color, line=dict(color=color, width=1)),
            opacity=0.55,
            name=f"{title} (P50={sim['p50']:.0f}%)",
        ))
        summaries.append({
            "title": title, "p10": sim["p10"], "p50": sim["p50"], "p90": sim["p90"],
            "success_pct": sim["success_pct"], "color": color,
        })

    fig.update_layout(
        title=dict(text="📊 Multi-Idea Outcome Comparison",
                   x=0.5, font=dict(size=14)),
        xaxis_title="Performance Metric (%)",
        yaxis_title="Frequency",
        barmode="overlay",
        height=350,
        margin=dict(l=40, r=20, t=50, b=80),
        legend=dict(orientation="h", y=-0.25),
        plot_bgcolor="white",
    )
    return fig, summaries


# ─────────────────────────────────────────────────────────────────────────────
# 7. SENSITIVITY TORNADO CHART
# ─────────────────────────────────────────────────────────────────────────────

def compute_sensitivity(idea: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Tornado-style sensitivity: for each factor, how much does the median
    outcome change when we move that factor by ±1 standard deviation?
    Returns sorted list of {factor, low_p50, high_p50, range}.
    """
    base = simulate_outcomes(idea, n_trials=200)["p50"]

    factors = []

    # Compute (×0.5 vs ×2)
    low = simulate_with_adjustments(idea, compute_multiplier=0.5, n_trials=200)["p50"]
    high = simulate_with_adjustments(idea, compute_multiplier=2.0, n_trials=200)["p50"]
    factors.append({
        "factor": "💻 Compute",
        "low_p50": low, "high_p50": high, "base": base,
        "range": abs(high - low),
        "low_label": "½× compute", "high_label": "2× compute",
    })

    # Data quality (-0.2 vs +0.2)
    low = simulate_with_adjustments(idea, data_quality_boost=-0.2, n_trials=200)["p50"]
    high = simulate_with_adjustments(idea, data_quality_boost=+0.2, n_trials=200)["p50"]
    factors.append({
        "factor": "💾 Data Quality",
        "low_p50": low, "high_p50": high, "base": base,
        "range": abs(high - low),
        "low_label": "Worse data", "high_label": "Better data",
    })

    # Novelty bet (-0.2 vs +0.2 → wider variance)
    low = simulate_with_adjustments(idea, novelty_bet=-0.2, n_trials=200)["p50"]
    high = simulate_with_adjustments(idea, novelty_bet=+0.2, n_trials=200)["p50"]
    factors.append({
        "factor": "🎯 Novelty Bet",
        "low_p50": low, "high_p50": high, "base": base,
        "range": abs(high - low),
        "low_label": "Safer", "high_label": "Riskier",
    })

    factors.sort(key=lambda x: x["range"], reverse=True)
    return factors


def build_sensitivity_tornado(idea: Dict[str, Any]):
    """Horizontal tornado chart showing factor sensitivity."""
    if not HAS_PLOTLY:
        return None

    factors = compute_sensitivity(idea)
    base = factors[0]["base"] if factors else 50

    fig = go.Figure()
    for f in factors:
        # Bar from low to high, centered conceptually around base
        low_dx = f["base"] - f["low_p50"]    # how far below base
        high_dx = f["high_p50"] - f["base"]  # how far above base

        # Negative direction (low value)
        fig.add_trace(go.Bar(
            y=[f["factor"]], x=[-low_dx], orientation="h",
            marker=dict(color="#fca5a5", line=dict(color="#ef4444", width=1)),
            hovertext=f"{f['low_label']}: {f['low_p50']:.1f}%",
            hoverinfo="text",
            text=f"{f['low_label']}: {f['low_p50']:.1f}%",
            textposition="inside",
            textfont=dict(color="white", size=11),
            showlegend=False,
        ))
        # Positive direction (high value)
        fig.add_trace(go.Bar(
            y=[f["factor"]], x=[high_dx], orientation="h",
            marker=dict(color="#86efac", line=dict(color="#10b981", width=1)),
            hovertext=f"{f['high_label']}: {f['high_p50']:.1f}%",
            hoverinfo="text",
            text=f"{f['high_label']}: {f['high_p50']:.1f}%",
            textposition="inside",
            textfont=dict(color="white", size=11),
            showlegend=False,
        ))

    fig.add_vline(x=0, line=dict(color="#0c4a6e", width=2))
    fig.update_layout(
        title=dict(
            text=f"🌀 Sensitivity Analysis (baseline P50 = {base:.1f}%)",
            x=0.5, font=dict(size=14),
        ),
        xaxis_title="Δ from baseline median",
        height=240,
        margin=dict(l=120, r=20, t=50, b=40),
        barmode="relative",
        plot_bgcolor="white",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 8. PARETO FRONTIER (cost vs quality across all ideas)
# ─────────────────────────────────────────────────────────────────────────────

def compute_pareto_frontier(ideas: List[Dict[str, Any]]) -> List[bool]:
    """
    Identify which ideas are Pareto-optimal in (cost↓, quality↑) space.
    Returns parallel list of bools matching `ideas`.
    """
    points = []
    for i, idea in enumerate(ideas):
        cost = estimate_resources(idea)["cost_usd"]
        quality = idea.get("quality_score", 0.0)
        points.append((i, cost, quality))

    on_frontier = [True] * len(ideas)
    for i, c_i, q_i in points:
        for j, c_j, q_j in points:
            if i == j:
                continue
            # j dominates i if j is cheaper AND higher quality (with some tolerance)
            if c_j <= c_i and q_j > q_i + 1e-6:
                on_frontier[i] = False
                break
            if c_j < c_i and q_j >= q_i:
                on_frontier[i] = False
                break
    return on_frontier


def build_pareto_scatter(ideas: List[Dict[str, Any]]):
    """Scatter plot of cost vs quality, highlighting Pareto-optimal ideas."""
    if not HAS_PLOTLY or not ideas:
        return None

    on_frontier = compute_pareto_frontier(ideas)

    costs, qualities, titles, colors, sizes = [], [], [], [], []
    for idea, opt in zip(ideas, on_frontier):
        c = estimate_resources(idea)["cost_usd"]
        q = idea.get("quality_score", 0.0)
        costs.append(c)
        qualities.append(q * 100)  # show as percent
        titles.append(idea.get("title", "?")[:50])
        colors.append("#10b981" if opt else "#94a3b8")
        sizes.append(18 if opt else 10)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=costs, y=qualities,
        mode="markers+text",
        marker=dict(
            color=colors, size=sizes,
            line=dict(color="#0c4a6e", width=1),
            symbol=["star" if o else "circle" for o in on_frontier],
        ),
        text=[t[:25] for t in titles],
        textposition="top center",
        textfont=dict(size=9),
        hovertext=[
            f"{t}<br>Cost: ${c:.0f}<br>Quality: {q:.1f}%<br>"
            f"{'⭐ Pareto-optimal' if o else 'Dominated'}"
            for t, c, q, o in zip(titles, costs, qualities, on_frontier)
        ],
        hoverinfo="text",
    ))

    # Connect Pareto frontier with a line
    frontier_pts = sorted(
        [(c, q) for c, q, o in zip(costs, qualities, on_frontier) if o]
    )
    if len(frontier_pts) >= 2:
        fig.add_trace(go.Scatter(
            x=[p[0] for p in frontier_pts],
            y=[p[1] for p in frontier_pts],
            mode="lines",
            line=dict(color="#10b981", width=2, dash="dot"),
            hoverinfo="skip", showlegend=False,
        ))

    fig.update_layout(
        title=dict(
            text=f"⭐ Pareto Frontier "
                 f"({sum(on_frontier)}/{len(ideas)} optimal)",
            x=0.5, font=dict(size=14),
        ),
        xaxis_title="💰 Estimated Cost (USD)",
        yaxis_title="📊 Quality Score (%)",
        height=380,
        margin=dict(l=50, r=20, t=50, b=50),
        showlegend=False,
        plot_bgcolor="white",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 9. RISK WATERFALL — FMEA → outcome reduction
# ─────────────────────────────────────────────────────────────────────────────

def build_risk_waterfall(idea: Dict[str, Any]):
    """
    Show how each FMEA failure mode reduces the expected success rate.
    Baseline → mode 1 hit → mode 2 hit → ... → adjusted success rate.
    """
    if not HAS_PLOTLY:
        return None

    sim = simulate_outcomes(idea, n_trials=200)
    base_success = sim["success_pct"]

    fmea = idea.get("_fmea") or {}
    failure_modes = fmea.get("failure_modes", [])

    # If no FMEA, generate heuristic ones
    if not failure_modes:
        try:
            from idea_enhancer import generate_fmea_heuristic
            failure_modes = generate_fmea_heuristic(idea)
        except Exception:
            failure_modes = []

    if not failure_modes:
        # Build a synthetic single-bar chart
        fig = go.Figure(go.Bar(
            x=["Baseline"], y=[base_success], marker=dict(color="#10b981"),
            text=[f"{base_success:.0f}%"], textposition="outside",
        ))
        fig.update_layout(
            title="⛓️ Risk Waterfall (no FMEA data)",
            yaxis_title="Success Probability (%)",
            height=280, plot_bgcolor="white",
        )
        return fig

    # Each failure mode reduces success by severity*detectability/100
    cur = base_success
    labels = ["Baseline"]
    values = [base_success]
    measures = ["absolute"]
    colors = ["#10b981"]
    for fm in failure_modes[:5]:
        rpn = fm.get("risk_priority", fm.get("severity", 1) * fm.get("detectability", 1))
        # Each point of RPN reduces success by ~1.5%, capped per-mode at 15%
        reduction = -min(15.0, rpn * 1.5)
        labels.append(fm.get("mode", "?")[:30])
        values.append(reduction)
        measures.append("relative")
        cur += reduction
        colors.append("#ef4444")

    labels.append("Adjusted")
    values.append(max(0, cur))
    measures.append("total")
    colors.append("#0ea5e9")

    fig = go.Figure(go.Waterfall(
        x=labels, y=values, measure=measures,
        increasing=dict(marker=dict(color="#10b981")),
        decreasing=dict(marker=dict(color="#ef4444")),
        totals=dict(marker=dict(color="#0ea5e9")),
        text=[f"{v:+.1f}%" if m == "relative" else f"{v:.1f}%"
              for v, m in zip(values, measures)],
        textposition="outside",
        connector=dict(line=dict(color="#94a3b8", dash="dot")),
    ))
    fig.update_layout(
        title=dict(text="⛓️ Risk Waterfall: FMEA Impact on Success",
                   x=0.5, font=dict(size=14)),
        yaxis_title="Success Probability (%)",
        height=320,
        margin=dict(l=40, r=20, t=50, b=80),
        plot_bgcolor="white",
        xaxis=dict(tickangle=-20),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 10. ANIMATED EXECUTION PLAYBACK (CI/CD-style stage-by-stage progress)
# ─────────────────────────────────────────────────────────────────────────────

def build_execution_playback(idea: Dict[str, Any]):
    """
    Animated bar chart showing each stage 'completing' over time.
    Uses Plotly frames to step through stages — user clicks Play.
    """
    if not HAS_PLOTLY:
        return None

    stages = extract_stages(idea)
    n = len(stages)
    if n == 0:
        return None

    # Per-stage duration (weeks) — heuristic split based on stage type
    durations = []
    for s in stages:
        if "Data" in s:        durations.append(2)
        elif "Preprocess" in s: durations.append(1)
        elif "Train" in s or "Model" in s: durations.append(4)
        elif "Evaluate" in s:  durations.append(2)
        elif "Analyze" in s:   durations.append(2)
        else:                  durations.append(1)
    total_w = sum(durations)

    # Build animation frames: each frame adds one more completed stage
    frames = []
    for completed in range(n + 1):
        statuses = ["✅" if i < completed else "⏳" for i in range(n)]
        # Bar widths scale with duration; "completed" bars are green, future are gray
        colors = ["#10b981" if i < completed else "#cbd5e1" for i in range(n)]
        frames.append(go.Frame(
            data=[go.Bar(
                y=[f"{s} {st}" for s, st in zip(stages, statuses)],
                x=durations, orientation="h",
                marker=dict(color=colors, line=dict(color="#0c4a6e", width=1)),
                text=[f"{d}w" for d in durations],
                textposition="inside",
                textfont=dict(color="white", size=11),
            )],
            name=str(completed),
            layout=dict(
                title=dict(
                    text=f"▶️ Execution Replay — Stage {completed}/{n} "
                         f"({sum(durations[:completed])}/{total_w} weeks)",
                    x=0.5, font=dict(size=14),
                ),
            ),
        ))

    # Initial frame: nothing started
    fig = go.Figure(
        data=[go.Bar(
            y=[f"{s} ⏳" for s in stages],
            x=durations, orientation="h",
            marker=dict(color=["#cbd5e1"] * n, line=dict(color="#0c4a6e", width=1)),
            text=[f"{d}w" for d in durations],
            textposition="inside",
            textfont=dict(color="white", size=11),
        )],
        frames=frames,
    )

    fig.update_layout(
        title=dict(text=f"▶️ Execution Replay — Stage 0/{n} (0/{total_w} weeks)",
                   x=0.5, font=dict(size=14)),
        xaxis=dict(title="Weeks", showgrid=True, gridcolor="#e0f2fe"),
        yaxis=dict(autorange="reversed"),
        height=300,
        margin=dict(l=140, r=20, t=70, b=70),
        plot_bgcolor="white",
        showlegend=False,
        updatemenus=[{
            "type": "buttons",
            "x": 0.5, "y": -0.20, "xanchor": "center",
            "showactive": False,
            "buttons": [
                {
                    "label": "▶ Play",
                    "method": "animate",
                    "args": [None, {
                        "frame": {"duration": 700, "redraw": True},
                        "fromcurrent": True,
                        "transition": {"duration": 250},
                    }],
                },
                {
                    "label": "⏸ Pause",
                    "method": "animate",
                    "args": [[None], {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate", "transition": {"duration": 0},
                    }],
                },
            ],
        }],
        sliders=[{
            "active": 0,
            "x": 0.1, "y": -0.06, "len": 0.8,
            "steps": [{
                "label": str(i), "method": "animate",
                "args": [[str(i)], {"frame": {"duration": 0, "redraw": True},
                                     "mode": "immediate"}],
            } for i in range(n + 1)],
        }],
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 11. CONFIDENCE CONE OVER TIME (uncertainty narrows as project progresses)
# ─────────────────────────────────────────────────────────────────────────────

def build_confidence_cone(idea: Dict[str, Any]):
    """
    Show outcome uncertainty narrowing as the project progresses.
    Y-axis: predicted performance %; X-axis: weeks.
    P10/P50/P90 cone widens early, narrows near completion.
    """
    if not HAS_PLOTLY:
        return None

    res = estimate_resources(idea)
    total_weeks = max(2, res["time_weeks"])

    sim = simulate_outcomes(idea, n_trials=300)
    final_p50 = sim["p50"]
    final_p10 = sim["p10"]
    final_p90 = sim["p90"]
    baseline = sim["baseline"]

    # Generate cone: at week 0 we know almost nothing (huge spread);
    # at week=total_weeks we know the final estimate.
    weeks = list(range(0, total_weeks + 1))
    spread_factor = [1.0 - (w / total_weeks) ** 1.5 for w in weeks]  # decays non-linearly
    p10_curve = [
        baseline + (final_p10 - baseline) * (w / total_weeks)
        - 15 * spread_factor[i] for i, w in enumerate(weeks)
    ]
    p50_curve = [
        baseline + (final_p50 - baseline) * (w / total_weeks)
        for w in weeks
    ]
    p90_curve = [
        baseline + (final_p90 - baseline) * (w / total_weeks)
        + 15 * spread_factor[i] for i, w in enumerate(weeks)
    ]

    fig = go.Figure()

    # P90 upper bound
    fig.add_trace(go.Scatter(
        x=weeks, y=p90_curve, mode="lines",
        line=dict(color="rgba(14,165,233,0.0)", width=0),
        showlegend=False, hoverinfo="skip",
    ))
    # P10 lower bound + fill to P90
    fig.add_trace(go.Scatter(
        x=weeks, y=p10_curve, mode="lines",
        fill="tonexty", fillcolor="rgba(14,165,233,0.2)",
        line=dict(color="rgba(14,165,233,0.0)", width=0),
        name="P10–P90 confidence band",
        hovertemplate="Week %{x}<br>P10: %{y:.1f}%<extra></extra>",
    ))
    # Median line
    fig.add_trace(go.Scatter(
        x=weeks, y=p50_curve, mode="lines+markers",
        line=dict(color="#0ea5e9", width=3),
        marker=dict(size=6, color="#0ea5e9"),
        name="Predicted median",
        hovertemplate="Week %{x}<br>Median: %{y:.1f}%<extra></extra>",
    ))
    # Baseline reference
    fig.add_hline(
        y=baseline, line=dict(color="#94a3b8", width=1, dash="dash"),
        annotation=dict(text=f"Baseline ({baseline:.0f}%)",
                        font=dict(color="#64748b"), xanchor="left"),
    )

    fig.update_layout(
        title=dict(text="🌡️ Confidence Cone — Uncertainty Narrows Over Time",
                   x=0.5, font=dict(size=14)),
        xaxis_title="Week",
        yaxis_title="Predicted Performance (%)",
        height=320,
        margin=dict(l=50, r=20, t=50, b=40),
        plot_bgcolor="white",
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 12. CUMULATIVE BUDGET BURN-DOWN
# ─────────────────────────────────────────────────────────────────────────────

def build_budget_burndown(idea: Dict[str, Any]):
    """
    Stacked area chart showing GPU-hours and USD spend accumulating week-by-week.
    """
    if not HAS_PLOTLY:
        return None

    res = estimate_resources(idea)
    total_weeks = max(2, res["time_weeks"])
    total_gpu = res["gpu_hours"]
    total_cost = res["cost_usd"]

    # Phase distribution: setup 20%, dev 35%, exp 30%, paper 15%
    # GPU usage is heaviest during dev + experiments
    weekly_gpu_share = []
    for w in range(total_weeks):
        frac = (w + 1) / total_weeks
        if frac < 0.20:        share = 0.05  # setup is light
        elif frac < 0.55:      share = 0.4 / (total_weeks * 0.35)  # dev: 40% over 35% of time
        elif frac < 0.85:      share = 0.5 / (total_weeks * 0.30)  # exp: 50% over 30% of time
        else:                  share = 0.05 / (total_weeks * 0.15) # paper: light
        weekly_gpu_share.append(share)
    # Normalize
    s = sum(weekly_gpu_share)
    weekly_gpu_share = [x / s for x in weekly_gpu_share]

    weeks = list(range(1, total_weeks + 1))
    cum_gpu = []
    cum_cost = []
    running_g = 0
    running_c = 0
    for share in weekly_gpu_share:
        running_g += total_gpu * share
        running_c += total_cost * share
        cum_gpu.append(running_g)
        cum_cost.append(running_c)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=weeks, y=cum_gpu,
        mode="lines+markers", name="GPU-hours",
        line=dict(color="#0ea5e9", width=3),
        fill="tozeroy", fillcolor="rgba(14,165,233,0.15)",
        hovertemplate="Week %{x}<br>%{y:.0f} GPU-hrs<extra></extra>",
        yaxis="y",
    ))
    fig.add_trace(go.Scatter(
        x=weeks, y=cum_cost,
        mode="lines+markers", name="Cost (USD)",
        line=dict(color="#10b981", width=3, dash="dot"),
        hovertemplate="Week %{x}<br>$%{y:.0f}<extra></extra>",
        yaxis="y2",
    ))
    fig.update_layout(
        title=dict(text=f"💰 Budget Burn-Down — ${total_cost:.0f} / "
                        f"{total_gpu} GPU-hrs over {total_weeks}w",
                   x=0.5, font=dict(size=14)),
        xaxis_title="Week",
        yaxis=dict(title="GPU-hours (cumulative)", side="left"),
        yaxis2=dict(title="Cost USD (cumulative)", side="right",
                    overlaying="y", showgrid=False),
        height=320,
        margin=dict(l=50, r=50, t=50, b=40),
        legend=dict(orientation="h", y=-0.2),
        plot_bgcolor="white",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 13. 3D IDEA SPACE SCATTER (quality × cost × novelty)
# ─────────────────────────────────────────────────────────────────────────────

def build_3d_idea_space(ideas: List[Dict[str, Any]], highlight_idx: int = -1):
    """3D scatter: x=cost, y=quality, z=novelty. Highlight one idea optionally."""
    if not HAS_PLOTLY or not ideas:
        return None

    xs, ys, zs, titles, sizes, colors = [], [], [], [], [], []
    for i, idea in enumerate(ideas):
        cost = estimate_resources(idea)["cost_usd"]
        quality = idea.get("quality_score", 0.0) * 100
        novelty = idea.get("probe_scores", {}).get("novelty", 0.5) * 100
        xs.append(cost)
        ys.append(quality)
        zs.append(novelty)
        titles.append(idea.get("title", "?")[:50])
        if i == highlight_idx:
            sizes.append(16)
            colors.append("#f59e0b")
        else:
            sizes.append(7)
            colors.append("#0ea5e9")

    fig = go.Figure(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="markers",
        marker=dict(
            size=sizes, color=colors, opacity=0.85,
            line=dict(color="#0c4a6e", width=1),
            symbol=["diamond" if i == highlight_idx else "circle"
                    for i in range(len(ideas))],
        ),
        text=titles,
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Cost: $%{x:.0f}<br>"
            "Quality: %{y:.1f}%<br>"
            "Novelty: %{z:.1f}%<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=dict(text="🌐 3D Idea Space — Cost × Quality × Novelty",
                   x=0.5, font=dict(size=14)),
        scene=dict(
            xaxis_title="💰 Cost (USD)",
            yaxis_title="📊 Quality (%)",
            zaxis_title="🎯 Novelty (%)",
            xaxis=dict(backgroundcolor="#f0f9ff", gridcolor="#bae6fd"),
            yaxis=dict(backgroundcolor="#f0f9ff", gridcolor="#bae6fd"),
            zaxis=dict(backgroundcolor="#f0f9ff", gridcolor="#bae6fd"),
        ),
        height=480,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 14. PROBE SCORE SUNBURST (radial breakdown of 10 dimensions)
# ─────────────────────────────────────────────────────────────────────────────

def build_probe_sunburst(idea: Dict[str, Any]):
    """Radial sunburst of all 10 probe dimensions, grouped by category."""
    if not HAS_PLOTLY:
        return None

    scores = idea.get("probe_scores") or {}
    if not scores:
        return None

    # Group probes into 3 categories: feasibility, quality, novelty/significance
    groups = {
        "🔧 Feasibility": ["code", "dataset", "constraint", "scalability"],
        "✨ Quality": ["specificity", "clarity", "testability", "risk_balance"],
        "🌟 Impact": ["novelty", "significance"],
    }

    labels = ["Idea"]   # root
    parents = [""]
    values = [0]
    colors = ["#0c4a6e"]

    group_colors = {
        "🔧 Feasibility": "#0ea5e9",
        "✨ Quality": "#10b981",
        "🌟 Impact": "#f59e0b",
    }

    for group_name, probes in groups.items():
        # Group total = sum of child scores (so the ring is sized correctly)
        group_score = sum(scores.get(p, 0) for p in probes)
        labels.append(group_name)
        parents.append("Idea")
        values.append(group_score)
        colors.append(group_colors[group_name])

        # Each probe is a leaf
        for p in probes:
            v = scores.get(p, 0) or 0
            if isinstance(v, (int, float)):
                labels.append(p.replace("_", " ").title())
                parents.append(group_name)
                values.append(round(v, 3))
                # Leaf color: green/amber/red by score
                if v >= 0.7:    colors.append("#86efac")
                elif v >= 0.4:  colors.append("#fde68a")
                else:           colors.append("#fca5a5")

    fig = go.Figure(go.Sunburst(
        labels=labels, parents=parents, values=values,
        marker=dict(colors=colors, line=dict(color="white", width=2)),
        branchvalues="total",
        hovertemplate="<b>%{label}</b><br>Score: %{value:.2f}<extra></extra>",
        insidetextorientation="radial",
        textfont=dict(size=11),
    ))
    fig.update_layout(
        title=dict(text="🎯 Probe Score Breakdown (10 dimensions)",
                   x=0.5, font=dict(size=14)),
        height=400,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 15. CARBON FOOTPRINT — green metrics for the research
# ─────────────────────────────────────────────────────────────────────────────

# A100 PCIe @ ~250W TDP × 1.3 PUE for cloud DC overhead
_A100_KW_PER_HOUR = 0.25 * 1.3  # 0.325 kWh per GPU-hour
# US grid avg ~370 g CO2 / kWh (2024 EIA estimate)
_CO2_KG_PER_KWH = 0.370


def estimate_carbon(idea: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute carbon footprint of running this idea.
    Returns: kWh, kg_co2, equivalents, green_score (0-100, higher is greener).
    """
    res = estimate_resources(idea)
    gpu_hours = res["gpu_hours"]
    kwh = gpu_hours * _A100_KW_PER_HOUR
    kg_co2 = kwh * _CO2_KG_PER_KWH

    # Real-world equivalents (engaging hooks)
    # Avg gasoline car emits ~404 g CO2/mile (EPA 2024)
    miles_driven = kg_co2 / 0.404
    # Tree absorbs ~21 kg CO2/year on average
    trees_year = kg_co2 / 21.0
    # Avg US household uses ~30 kWh/day
    household_days = kwh / 30.0

    # Green score: lower emissions = higher score
    # 1 ton CO2 → score 0; <10 kg CO2 → score 100
    if kg_co2 <= 10:
        green = 100
    elif kg_co2 >= 1000:
        green = 0
    else:
        green = round(100 * (1 - math.log10(kg_co2 / 10) / math.log10(100)))

    return {
        "gpu_hours": gpu_hours,
        "kwh": round(kwh, 1),
        "kg_co2": round(kg_co2, 2),
        "miles_driven_eq": round(miles_driven, 1),
        "trees_year_eq": round(trees_year, 2),
        "household_days_eq": round(household_days, 1),
        "green_score": int(green),
    }


def build_carbon_footprint(idea: Dict[str, Any]):
    """Multi-panel card: green-score gauge + 3 equivalence indicators."""
    if not HAS_PLOTLY:
        return None

    c = estimate_carbon(idea)
    score_color = ("#10b981" if c["green_score"] >= 70 else
                   "#f59e0b" if c["green_score"] >= 40 else "#ef4444")

    fig = go.Figure()
    # Big green-score gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=c["green_score"],
        domain={"x": [0, 0.5], "y": [0, 1]},
        title={"text": "🌱 Green Score", "font": {"size": 14}},
        number={"suffix": "/100", "font": {"size": 32, "color": score_color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": score_color, "thickness": 0.7},
            "steps": [
                {"range": [0, 40], "color": "#fecaca"},
                {"range": [40, 70], "color": "#fde68a"},
                {"range": [70, 100], "color": "#bbf7d0"},
            ],
            "bgcolor": "#f0f9ff",
        },
    ))
    # CO2 number
    fig.add_trace(go.Indicator(
        mode="number",
        value=c["kg_co2"],
        domain={"x": [0.5, 0.75], "y": [0.5, 1]},
        title={"text": "💨 CO₂ Emissions", "font": {"size": 12}},
        number={"suffix": " kg", "font": {"size": 22, "color": "#0c4a6e"}},
    ))
    # kWh number
    fig.add_trace(go.Indicator(
        mode="number",
        value=c["kwh"],
        domain={"x": [0.75, 1.0], "y": [0.5, 1]},
        title={"text": "⚡ Energy", "font": {"size": 12}},
        number={"suffix": " kWh", "font": {"size": 22, "color": "#0c4a6e"}},
    ))
    # Miles equivalent
    fig.add_trace(go.Indicator(
        mode="number",
        value=c["miles_driven_eq"],
        domain={"x": [0.5, 0.75], "y": [0, 0.5]},
        title={"text": "🚗 Equiv. miles driven", "font": {"size": 11}},
        number={"font": {"size": 18, "color": "#64748b"}},
    ))
    # Trees equivalent
    fig.add_trace(go.Indicator(
        mode="number",
        value=c["trees_year_eq"],
        domain={"x": [0.75, 1.0], "y": [0, 0.5]},
        title={"text": "🌳 Trees needed (1y)", "font": {"size": 11}},
        number={"font": {"size": 18, "color": "#64748b"}},
    ))

    fig.update_layout(
        title=dict(text="🌍 Carbon Footprint Estimate",
                   x=0.5, font=dict(size=14)),
        height=300, margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 16. CITATION IMPACT FORECAST (Poisson over 5 years)
# ─────────────────────────────────────────────────────────────────────────────

def forecast_citations(idea: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict citation count over 5 years using a Poisson-like model.
    Higher quality + novelty + significance → more citations.
    """
    quality = idea.get("quality_score", 0.5)
    probe = idea.get("probe_scores", {})
    novelty = probe.get("novelty", 0.5)
    significance = probe.get("significance", 0.5)
    clarity = probe.get("clarity", 0.5)

    # Year-1 citations: typical conference paper gets 2-50 in year 1
    # Calibrate so a 0.5/0.5/0.5/0.5 idea gets ~5; 0.9/0.9/0.9/0.9 gets ~50
    year1_lambda = 2.0 + 8.0 * quality + 12.0 * novelty + 8.0 * significance + 4.0 * clarity
    # Citations follow exponential decay then plateau (rough heuristic)
    multipliers = [1.0, 2.2, 3.0, 3.5, 3.8]  # cumulative as fraction of asymptote
    asymptote = year1_lambda * 4.0  # total expected citations
    cumulative = [round(asymptote * m / multipliers[-1]) for m in multipliers]

    # h-index contribution: rough heuristic — a paper with N cites contributes
    # to h-index when N >= h. We assume h is around sqrt(asymptote).
    h_contrib = max(1, int(math.sqrt(asymptote)))

    return {
        "year1_lambda": round(year1_lambda, 1),
        "asymptote": round(asymptote, 0),
        "cumulative": cumulative,
        "h_index_contrib": h_contrib,
        "quality_tier": (
            "viral" if asymptote >= 100 else
            "strong" if asymptote >= 30 else
            "moderate" if asymptote >= 10 else "minor"
        ),
    }


def build_citation_forecast(idea: Dict[str, Any]):
    """Bar chart of cumulative citations over 5 years."""
    if not HAS_PLOTLY:
        return None

    f = forecast_citations(idea)
    years = list(range(1, 6))

    # Per-year (delta) citations for the bar chart
    yearly = [f["cumulative"][0]]
    for i in range(1, 5):
        yearly.append(f["cumulative"][i] - f["cumulative"][i - 1])

    # Color bars by intensity (sky blue gradient)
    colors = ["#bae6fd", "#7dd3fc", "#38bdf8", "#0ea5e9", "#0284c7"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"Year {y}" for y in years], y=yearly,
        marker=dict(color=colors, line=dict(color="#0c4a6e", width=1)),
        text=[f"+{v}" for v in yearly], textposition="outside",
        textfont=dict(size=11, color="#0c4a6e"),
        name="Per-year citations",
    ))
    # Cumulative line on secondary axis
    fig.add_trace(go.Scatter(
        x=[f"Year {y}" for y in years], y=f["cumulative"],
        mode="lines+markers", name="Cumulative",
        line=dict(color="#10b981", width=2, dash="dot"),
        marker=dict(size=8, color="#10b981"),
        yaxis="y2",
        hovertemplate="%{x}<br>Total: %{y}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text=f"📈 Citation Forecast — ~{f['asymptote']:.0f} total over 5y "
                 f"({f['quality_tier']})",
            x=0.5, font=dict(size=14),
        ),
        xaxis_title="",
        yaxis=dict(title="Citations / year"),
        yaxis2=dict(title="Cumulative", overlaying="y", side="right",
                    showgrid=False),
        height=300,
        margin=dict(l=50, r=50, t=50, b=40),
        legend=dict(orientation="h", y=-0.20),
        plot_bgcolor="white",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 17. IDEA SIMILARITY NETWORK
# ─────────────────────────────────────────────────────────────────────────────

def _idea_similarity(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    """Token Jaccard between method+title fields. 0 = unrelated, 1 = identical."""
    text_a = (a.get("title", "") + " " + a.get("method", "")).lower()
    text_b = (b.get("title", "") + " " + b.get("method", "")).lower()
    tokens_a = set(re.findall(r"\b[a-z]{4,}\b", text_a))
    tokens_b = set(re.findall(r"\b[a-z]{4,}\b", text_b))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def build_similarity_network(ideas: List[Dict[str, Any]],
                              highlight_idx: int = -1):
    """Force-directed graph of ideas; edges = method similarity above threshold."""
    if not HAS_PLOTLY or len(ideas) < 2:
        return None

    n = len(ideas)
    # Compute pairwise similarities
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            s = _idea_similarity(ideas[i], ideas[j])
            if s >= 0.10:  # only show meaningfully connected pairs
                sims.append((i, j, s))

    # Position nodes on a circle (deterministic, no networkx dep needed)
    angles = [2 * math.pi * i / n for i in range(n)]
    radius = 1.0
    xs = [radius * math.cos(a) for a in angles]
    ys = [radius * math.sin(a) for a in angles]

    fig = go.Figure()

    # Edges first (so they render under nodes)
    for i, j, s in sims:
        opacity = 0.2 + 0.8 * s  # stronger sim = more visible
        width = 0.5 + 4 * s
        fig.add_trace(go.Scatter(
            x=[xs[i], xs[j]], y=[ys[i], ys[j]],
            mode="lines",
            line=dict(width=width, color=f"rgba(14,165,233,{opacity:.2f})"),
            hoverinfo="text",
            hovertext=f"{ideas[i].get('title', '?')[:30]} ↔ {ideas[j].get('title', '?')[:30]}<br>Similarity: {s:.0%}",
            showlegend=False,
        ))

    # Nodes
    sizes = [
        24 if i == highlight_idx else
        12 + 16 * idea.get("quality_score", 0.5)
        for i, idea in enumerate(ideas)
    ]
    colors = [
        "#f59e0b" if i == highlight_idx else "#0ea5e9"
        for i in range(n)
    ]
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="markers+text",
        marker=dict(
            size=sizes, color=colors,
            line=dict(color="#0c4a6e", width=1.5),
            opacity=0.9,
        ),
        text=[f"{i+1}" for i in range(n)],
        textfont=dict(size=10, color="white"),
        textposition="middle center",
        hovertext=[
            f"<b>{idea.get('title', '?')[:50]}</b><br>"
            f"Q: {idea.get('quality_score', 0):.2f}"
            for idea in ideas
        ],
        hoverinfo="text",
        showlegend=False,
    ))

    fig.update_layout(
        title=dict(
            text=f"🕸️ Idea Similarity Network ({len(sims)} connections)",
            x=0.5, font=dict(size=14),
        ),
        height=400,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False),
        plot_bgcolor="#f0f9ff",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 18. CUMULATIVE SUCCESS FUNNEL
# ─────────────────────────────────────────────────────────────────────────────

def build_success_funnel(idea: Dict[str, Any]):
    """
    Funnel showing P(reach each stage successfully). Each stage compounds.
    Models: data_pass → method_works → results_significant → paper_accepted.
    """
    if not HAS_PLOTLY:
        return None

    quality = idea.get("quality_score", 0.5)
    probe = idea.get("probe_scores", {})

    # Per-stage success probabilities (heuristic from probe scores)
    p_data = 0.5 + 0.5 * probe.get("dataset", 0.5)               # 0.5-1.0
    p_code = 0.4 + 0.6 * probe.get("code", 0.5)                  # 0.4-1.0
    p_results = 0.3 + 0.7 * probe.get("specificity", 0.5)        # 0.3-1.0
    p_signif = 0.3 + 0.7 * probe.get("significance", 0.5)        # 0.3-1.0
    p_accept = 0.2 + 0.8 * probe.get("novelty", 0.5)             # 0.2-1.0

    stages = [
        ("📦 Data acquired", 100.0),
        ("🧪 Method runs", 100 * p_data * p_code),
        ("📊 Results obtained", 100 * p_data * p_code * p_results),
        ("✨ Results significant", 100 * p_data * p_code * p_results * p_signif),
        ("🎓 Paper accepted", 100 * p_data * p_code * p_results * p_signif * p_accept),
    ]

    labels = [s[0] for s in stages]
    values = [s[1] for s in stages]

    fig = go.Figure(go.Funnel(
        y=labels, x=values,
        textinfo="value+percent initial",
        textfont=dict(size=12, color="white"),
        marker=dict(
            color=["#0ea5e9", "#0284c7", "#0369a1", "#075985", "#0c4a6e"],
            line=dict(color="white", width=2),
        ),
        connector=dict(line=dict(color="#bae6fd", dash="dot", width=2)),
        hovertemplate="<b>%{label}</b><br>P(reach this stage): %{x:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text=f"🎯 Success Funnel — P(paper accepted) = {values[-1]:.1f}%",
            x=0.5, font=dict(size=14),
        ),
        height=320, margin=dict(l=140, r=20, t=50, b=20),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 19. STAGE CRITICALITY HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

def build_stage_criticality(idea: Dict[str, Any]):
    """
    Bar chart showing which stage's failure has the biggest downstream impact.
    Computes the "if this stage fails completely, success drops by X%" delta.
    """
    if not HAS_PLOTLY:
        return None

    probe = idea.get("probe_scores", {})

    # Compute baseline P(success)
    base_p = (
        (0.5 + 0.5 * probe.get("dataset", 0.5))
        * (0.4 + 0.6 * probe.get("code", 0.5))
        * (0.3 + 0.7 * probe.get("specificity", 0.5))
        * (0.3 + 0.7 * probe.get("significance", 0.5))
        * (0.2 + 0.8 * probe.get("novelty", 0.5))
    )

    stages = [
        ("📦 Data quality",    "dataset"),
        ("💻 Code/method",     "code"),
        ("📐 Specificity",     "specificity"),
        ("🌟 Significance",    "significance"),
        ("🎯 Novelty",         "novelty"),
    ]

    # For each stage, set its score to 0 and recompute success
    impacts = []
    for stage_label, key in stages:
        modified = dict(probe)
        modified[key] = 0.0
        if key == "dataset":
            stage_p = 0.5  # min P_data
        else:
            # Recompute with this stage at zero
            modified_p = (
                (0.5 + 0.5 * modified.get("dataset", 0.5))
                * (0.4 + 0.6 * modified.get("code", 0.5))
                * (0.3 + 0.7 * modified.get("specificity", 0.5))
                * (0.3 + 0.7 * modified.get("significance", 0.5))
                * (0.2 + 0.8 * modified.get("novelty", 0.5))
            )
            stage_p = modified_p

        # Just use full recompute always
        modified_p = (
            (0.5 + 0.5 * modified.get("dataset", 0.5))
            * (0.4 + 0.6 * modified.get("code", 0.5))
            * (0.3 + 0.7 * modified.get("specificity", 0.5))
            * (0.3 + 0.7 * modified.get("significance", 0.5))
            * (0.2 + 0.8 * modified.get("novelty", 0.5))
        )
        impact = max(0, (base_p - modified_p) * 100)
        impacts.append((stage_label, impact))

    impacts.sort(key=lambda x: x[1], reverse=True)
    labels = [x[0] for x in impacts]
    values = [x[1] for x in impacts]

    # Color by criticality: most-impactful = red, least = green
    max_imp = max(values) if values else 1
    colors = []
    for v in values:
        ratio = v / max_imp if max_imp > 0 else 0
        if ratio >= 0.7:    colors.append("#ef4444")
        elif ratio >= 0.4:  colors.append("#f59e0b")
        else:               colors.append("#10b981")

    fig = go.Figure(go.Bar(
        y=labels, x=values, orientation="h",
        marker=dict(color=colors, line=dict(color="#0c4a6e", width=1)),
        text=[f"{v:.1f}% drop" for v in values],
        textposition="outside",
        textfont=dict(size=11, color="#0c4a6e"),
        hovertemplate="<b>%{y}</b><br>Removing this stage drops success by %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text="🎚️ Stage Criticality (which failure hurts most?)",
            x=0.5, font=dict(size=14),
        ),
        xaxis_title="Δ Success Probability (%)",
        height=280,
        margin=dict(l=140, r=80, t=50, b=40),
        plot_bgcolor="white",
        yaxis=dict(autorange="reversed"),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 20. NOVELTY CONSTELLATION — 2D map of idea position in concept space
# ─────────────────────────────────────────────────────────────────────────────

def _hash_to_2d(text: str) -> Tuple[float, float]:
    """Deterministic 2D position from text using two hash bytes (no embeddings dep)."""
    h = hashlib.md5(text.lower().encode("utf-8"), usedforsecurity=False).digest()
    # Take 2 pairs of bytes, map to [-1, 1]
    x = (h[0] / 127.5) - 1.0
    y = (h[1] / 127.5) - 1.0
    return (x, y)


def build_novelty_constellation(
    idea: Dict[str, Any],
    other_ideas: List[Dict[str, Any]] = None,
    dag_papers: List[Dict[str, Any]] = None,
):
    """
    2D star map. Literature papers = small blue dots, other ideas = circles,
    YOUR idea = bright star with a glowing halo. Position via deterministic
    pseudo-embedding (hash-based). Distance = conceptual novelty.
    """
    if not HAS_PLOTLY:
        return None

    other_ideas = other_ideas or []
    dag_papers = dag_papers or []

    fig = go.Figure()

    # Background: literature papers (small dim dots)
    if dag_papers:
        ps = [_hash_to_2d(p.get("title", "?")) for p in dag_papers]
        fig.add_trace(go.Scatter(
            x=[p[0] for p in ps], y=[p[1] for p in ps],
            mode="markers",
            marker=dict(size=4, color="#bae6fd", opacity=0.5,
                        line=dict(color="#7dd3fc", width=0.5)),
            text=[p.get("title", "?")[:60] for p in dag_papers],
            hoverinfo="text",
            name="Literature",
        ))

    # Other ideas (medium circles)
    if other_ideas:
        os_pos = [_hash_to_2d(i.get("title", "?")) for i in other_ideas]
        os_q = [i.get("quality_score", 0.5) for i in other_ideas]
        fig.add_trace(go.Scatter(
            x=[p[0] for p in os_pos], y=[p[1] for p in os_pos],
            mode="markers",
            marker=dict(
                size=[10 + 8 * q for q in os_q],
                color="#0ea5e9", opacity=0.6,
                line=dict(color="#0284c7", width=1),
                symbol="circle",
            ),
            text=[i.get("title", "?")[:60] for i in other_ideas],
            hoverinfo="text",
            name="Other ideas",
        ))

    # The star — YOUR idea
    your_x, your_y = _hash_to_2d(idea.get("title", ""))
    novelty = idea.get("probe_scores", {}).get("novelty", 0.5)

    # Glowing halo: 3 layered scatter points with decreasing opacity
    for halo_size, halo_alpha in [(45, 0.15), (35, 0.25), (25, 0.4)]:
        fig.add_trace(go.Scatter(
            x=[your_x], y=[your_y],
            mode="markers",
            marker=dict(
                size=halo_size, color=f"rgba(245,158,11,{halo_alpha})",
                line=dict(color=f"rgba(245,158,11,0)", width=0),
            ),
            hoverinfo="skip", showlegend=False,
        ))
    # Bright star core
    fig.add_trace(go.Scatter(
        x=[your_x], y=[your_y],
        mode="markers+text",
        marker=dict(size=22, color="#f59e0b",
                    line=dict(color="white", width=2),
                    symbol="star"),
        text=[f"⭐ {idea.get('title', 'You')[:30]}"],
        textposition="top center",
        textfont=dict(size=11, color="#92400e", family="Arial Black"),
        name="Your idea",
        hovertext=f"<b>{idea.get('title', '?')[:60]}</b><br>"
                  f"Novelty score: {novelty:.0%}",
        hoverinfo="text",
    ))

    # Compute novelty distance: how far is your idea from nearest neighbor?
    nearest_dist = float("inf")
    for other in (dag_papers + other_ideas):
        ox, oy = _hash_to_2d(other.get("title", ""))
        d = math.hypot(your_x - ox, your_y - oy)
        if d < nearest_dist:
            nearest_dist = d
    nearest_label = (
        "🏝️ Lone Island (very novel)" if nearest_dist > 0.5 else
        "🌌 Frontier (novel)" if nearest_dist > 0.2 else
        "🏘️ Crowded (incremental)"
    )

    fig.update_layout(
        title=dict(
            text=f"🌟 Novelty Constellation — {nearest_label}",
            x=0.5, font=dict(size=14),
        ),
        xaxis=dict(visible=False, range=[-1.3, 1.3]),
        yaxis=dict(visible=False, range=[-1.3, 1.3], scaleanchor="x"),
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor="#0c1e3a",
        showlegend=True,
        legend=dict(orientation="h", y=-0.05, font=dict(color="white", size=10)),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 21. IDEA DNA FINGERPRINT — unique visual signature
# ─────────────────────────────────────────────────────────────────────────────

def _idea_dna_bands(idea: Dict[str, Any]) -> List[Tuple[str, float]]:
    """Generate 16 colored bands encoding the idea's attributes."""
    h = hashlib.sha256(
        (idea.get("title", "") + idea.get("method", "")).encode("utf-8"),
    ).digest()
    quality = idea.get("quality_score", 0.5)
    probe = idea.get("probe_scores") or {}

    # Get 16 bands: 6 from probe scores + 10 from hash
    palette = ["#0ea5e9", "#10b981", "#f59e0b", "#a855f7",
               "#ef4444", "#3b82f6", "#84cc16", "#06b6d4",
               "#ec4899", "#14b8a6", "#f97316", "#8b5cf6"]
    bands = []
    keys = ["code", "dataset", "novelty", "specificity",
            "significance", "testability", "clarity", "scalability"]
    for i, k in enumerate(keys):
        v = probe.get(k, 0.5) if isinstance(probe.get(k), (int, float)) else 0.5
        bands.append((palette[i % len(palette)], v))
    # Fill remaining 8 bands from hash bytes (deterministic)
    for i in range(8):
        v = h[i] / 255.0
        bands.append((palette[(i + 8) % len(palette)], v))
    return bands


def build_dna_fingerprint(idea: Dict[str, Any]):
    """
    Render the idea as a unique vertical DNA-like band pattern.
    Each idea produces a deterministic, shareable fingerprint.
    """
    if not HAS_PLOTLY:
        return None

    bands = _idea_dna_bands(idea)
    n = len(bands)

    fig = go.Figure()

    # Each band is a vertical bar; height encodes intensity (0..1)
    xs = list(range(n))
    heights = [b[1] for b in bands]
    colors = [b[0] for b in bands]

    fig.add_trace(go.Bar(
        x=xs, y=heights,
        marker=dict(color=colors, line=dict(color="white", width=0)),
        hovertemplate="Band %{x}<br>Intensity: %{y:.2f}<extra></extra>",
        showlegend=False,
        width=0.95,
    ))
    # Mirror bars below the axis for symmetry
    fig.add_trace(go.Bar(
        x=xs, y=[-h for h in heights],
        marker=dict(color=colors, line=dict(color="white", width=0)),
        hoverinfo="skip", showlegend=False,
        width=0.95,
    ))

    # Compute a short "fingerprint string" for sharing
    fp_hash = hashlib.md5(
        (idea.get("title", "") + idea.get("method", "")).encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()[:12].upper()

    fig.update_layout(
        title=dict(
            text=f"🧬 Idea DNA — Fingerprint #{fp_hash}",
            x=0.5, font=dict(size=14),
        ),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, range=[-1.1, 1.1]),
        height=160,
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor="#0c1e3a",
        bargap=0.05,
        barmode="overlay",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 22. TIME MACHINE — recency of each component
# ─────────────────────────────────────────────────────────────────────────────

# Year each technique became "well-known" — rough heuristic.
# Not exhaustive; covers most common ML terms.
_TECHNIQUE_ERAS = {
    "linear regression": 1900, "decision tree": 1980, "k-means": 1957,
    "svm": 1995, "random forest": 2001, "boosting": 1996, "xgboost": 2014,
    "cnn": 2012, "lstm": 1997, "rnn": 1986, "gru": 2014,
    "attention": 2014, "transformer": 2017, "bert": 2018, "gpt": 2018,
    "self-supervised": 2018, "contrastive learning": 2020,
    "diffusion": 2020, "diffusion model": 2020,
    "foundation model": 2022, "llm": 2022, "instruction tuning": 2022,
    "rlhf": 2022, "chain-of-thought": 2022, "tool use": 2023,
    "agent": 2023, "agentic": 2023, "multi-agent": 2023,
    "graph neural network": 2017, "gnn": 2017, "message passing": 2017,
    "graph attention": 2018, "node2vec": 2016, "deepwalk": 2014,
    "reinforcement learning": 1990, "policy gradient": 2000,
    "actor critic": 2000, "ppo": 2017, "dqn": 2013,
    "q-learning": 1989, "td-learning": 1988,
    "vae": 2013, "gan": 2014, "wasserstein": 2017,
    "embedding": 2013, "word2vec": 2013, "glove": 2014,
    "vit": 2020, "vision transformer": 2020, "clip": 2021,
    "stable diffusion": 2022, "lora": 2021,
    "fine-tuning": 2018, "few-shot": 2020, "zero-shot": 2020,
    "prompt": 2020, "prompt engineering": 2022,
    "knowledge graph": 2012, "embedding": 2013,
    "federated learning": 2017, "differential privacy": 2006,
    "neural architecture search": 2017, "automl": 2018,
    "meta-learning": 2017, "maml": 2017,
    "moe": 2022, "mixture of experts": 2022,
    "mamba": 2024, "state space model": 2023,
}


def detect_techniques(idea: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Find which techniques are mentioned + their first-appearance year."""
    text = (idea.get("title", "") + " " + idea.get("method", "")
            + " " + idea.get("hypothesis", "")).lower()
    found = []
    seen = set()
    for term, year in _TECHNIQUE_ERAS.items():
        if term in text and term not in seen:
            found.append({"term": term, "year": year})
            seen.add(term)
    found.sort(key=lambda x: x["year"])
    return found


def build_time_machine(idea: Dict[str, Any]):
    """
    Timeline showing how dated/cutting-edge each component of the method is.
    Plots each detected technique as a marker on a year axis.
    """
    if not HAS_PLOTLY:
        return None

    techniques = detect_techniques(idea)
    if not techniques:
        return None

    current_year = 2026
    years = [t["year"] for t in techniques]
    labels = [t["term"].title() for t in techniques]

    # Color by recency: red = stale (>10y old), amber = mature, green = recent
    colors = []
    for y in years:
        age = current_year - y
        if age <= 3:    colors.append("#10b981")  # cutting-edge
        elif age <= 8:  colors.append("#0ea5e9")  # mainstream
        elif age <= 15: colors.append("#f59e0b")  # mature
        else:           colors.append("#94a3b8")  # vintage

    fig = go.Figure()

    # Horizontal "timeline" line
    fig.add_trace(go.Scatter(
        x=[min(years) - 2, current_year + 1],
        y=[0, 0],
        mode="lines",
        line=dict(color="#cbd5e1", width=2),
        showlegend=False, hoverinfo="skip",
    ))

    # Markers for each technique
    fig.add_trace(go.Scatter(
        x=years, y=[0] * len(years),
        mode="markers+text",
        marker=dict(size=18, color=colors,
                    line=dict(color="white", width=2),
                    symbol="circle"),
        text=labels,
        textposition=["top center" if i % 2 == 0 else "bottom center"
                      for i in range(len(labels))],
        textfont=dict(size=10, color="#0c4a6e"),
        hovertemplate="<b>%{text}</b><br>Introduced: %{x}<extra></extra>",
        showlegend=False,
    ))

    # Mark current year
    fig.add_vline(
        x=current_year, line=dict(color="#ef4444", width=2, dash="dash"),
        annotation=dict(text="⏰ Now", font=dict(color="#ef4444"),
                        yanchor="bottom"),
    )

    # Average age summary
    avg_age = current_year - sum(years) / len(years)
    novelty_label = (
        "🚀 Cutting-edge" if avg_age <= 3 else
        "✨ Modern" if avg_age <= 8 else
        "📚 Mature" if avg_age <= 15 else "🏛️ Vintage"
    )

    fig.update_layout(
        title=dict(
            text=f"⏳ Time Machine — avg age {avg_age:.0f}y ({novelty_label})",
            x=0.5, font=dict(size=14),
        ),
        xaxis_title="Year introduced",
        yaxis=dict(visible=False, range=[-1, 1]),
        height=240,
        margin=dict(l=20, r=20, t=50, b=40),
        plot_bgcolor="white",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 23. REVIEWER CHAT SIMULATOR — predicted feedback as chat thread
# ─────────────────────────────────────────────────────────────────────────────

def simulate_reviewer_chat(idea: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Predict what 3 reviewers would say about this idea, derived from
    probe scores + heuristics. No LLM call — purely template-driven.
    """
    probe = idea.get("probe_scores", {})
    quality = idea.get("quality_score", 0.5)

    chat = []

    # Reviewer 1: Methodologist — focuses on rigor
    spec = probe.get("specificity", 0.5)
    test = probe.get("testability", 0.5)
    if spec >= 0.7 and test >= 0.7:
        r1 = ("The methodology is well-specified with a clear, falsifiable "
              "hypothesis. I can see exactly how this would be tested. "
              "Strong recommend.")
        sentiment = "positive"
    elif spec < 0.4 or test < 0.4:
        r1 = ("The method needs more concrete details before I can evaluate it. "
              "Specifically, what hyperparameters? What dataset version? "
              "How will you measure success quantitatively?")
        sentiment = "negative"
    else:
        r1 = ("Reasonable methodology but I'd like to see a clearer ablation "
              "plan. What's the contribution beyond existing baselines?")
        sentiment = "neutral"
    chat.append({"role": "Reviewer 1 (Methodologist) 👨‍🔬", "msg": r1, "sentiment": sentiment})

    # Reviewer 2: Theorist — focuses on novelty + significance
    novelty = probe.get("novelty", 0.5)
    sig = probe.get("significance", 0.5)
    if novelty >= 0.7 and sig >= 0.7:
        r2 = ("This is genuinely novel and addresses a problem the field "
              "actually cares about. Could be a NeurIPS-quality paper if "
              "executed well.")
        sentiment = "positive"
    elif novelty < 0.4:
        r2 = ("Honestly? This feels like an incremental tweak on existing "
              "work. What's the core insight that hasn't been published yet?")
        sentiment = "negative"
    else:
        r2 = ("Decent novelty, but I'm not convinced the impact is high. "
              "Who exactly benefits from solving this?")
        sentiment = "neutral"
    chat.append({"role": "Reviewer 2 (Theorist) 👩‍🎓", "msg": r2, "sentiment": sentiment})

    # Reviewer 3: Practitioner — focuses on feasibility
    code = probe.get("code", 0.5)
    dataset = probe.get("dataset", 0.5)
    constraint = probe.get("constraint", 0.5)
    if code >= 0.6 and dataset >= 0.6 and constraint >= 0.5:
        r3 = ("Feasibility looks good. Standard datasets, implementable methods, "
              "reasonable compute. I could reproduce this in 2-3 weeks.")
        sentiment = "positive"
    elif constraint < 0.4:
        r3 = ("Compute requirements look unrealistic for an academic lab. "
              "Can you scale this down to fit on 4 GPUs over a month?")
        sentiment = "negative"
    elif dataset < 0.4:
        r3 = ("I'm worried about data availability. Where exactly is this "
              "dataset? Can you verify it's accessible?")
        sentiment = "negative"
    else:
        r3 = ("Implementation looks workable but I'd want to see a working "
              "code prototype before fully buying in. Got a Colab to share?")
        sentiment = "neutral"
    chat.append({"role": "Reviewer 3 (Practitioner) 👨‍💻", "msg": r3, "sentiment": sentiment})

    # Meta-review (overall verdict)
    n_pos = sum(1 for c in chat if c["sentiment"] == "positive")
    n_neg = sum(1 for c in chat if c["sentiment"] == "negative")
    if n_pos >= 2:
        verdict = "✅ Likely Accept"
        verdict_color = "#10b981"
    elif n_neg >= 2:
        verdict = "❌ Likely Reject"
        verdict_color = "#ef4444"
    else:
        verdict = "⚖️ Borderline — needs revision"
        verdict_color = "#f59e0b"

    return [{"role": "📋 Meta-Reviewer", "msg": verdict,
             "sentiment": "verdict", "color": verdict_color}] + chat


# ─────────────────────────────────────────────────────────────────────────────
# 24. IDEA TAROT — Past / Present / Future cards
# ─────────────────────────────────────────────────────────────────────────────

def generate_tarot(idea: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Generate 3 tarot-style cards for the idea: Past (origin), Present (state),
    Future (predicted outcome). Auto-derived from probe scores + metadata.
    """
    quality = idea.get("quality_score", 0.5)
    probe = idea.get("probe_scores", {})
    novelty = probe.get("novelty", 0.5)
    feasibility = (probe.get("code", 0.5) + probe.get("dataset", 0.5)) / 2

    method_type = (idea.get("methodology_type", "") or "").replace("_", " ").title()
    novelty_lvl = (idea.get("novelty_level", "") or "").capitalize()

    # PAST — what came before this idea
    if novelty >= 0.7:
        past = {
            "name": "🌑 The Pioneer",
            "meaning": "An untrodden path",
            "narrative": (
                f"This idea emerged where no clear precedent existed. "
                f"You're charting a {novelty_lvl.lower() or 'new'} direction "
                f"in {method_type.lower() or 'research'}."
            ),
        }
    elif novelty >= 0.4:
        past = {
            "name": "🌗 The Synthesizer",
            "meaning": "Bridging existing work",
            "narrative": (
                f"This idea stands on the shoulders of prior work, "
                f"combining established techniques in a new arrangement."
            ),
        }
    else:
        past = {
            "name": "🌕 The Iterator",
            "meaning": "An incremental improvement",
            "narrative": (
                "This builds carefully on an established foundation. "
                "Lower risk, but the contribution must be unambiguous."
            ),
        }

    # PRESENT — current state of the idea
    if quality >= 0.7 and feasibility >= 0.6:
        present = {
            "name": "⚡ The Forge",
            "meaning": "Ready to be built",
            "narrative": (
                f"This idea is in strong shape (quality {quality:.2f}). "
                f"All the pieces — method, data, compute — are in place. "
                f"The next move is to start building."
            ),
        }
    elif quality >= 0.5:
        present = {
            "name": "🔨 The Workshop",
            "meaning": "Refinement needed",
            "narrative": (
                f"The core is sound, but quality {quality:.2f} suggests "
                f"polish is needed. Strengthen the weakest probe before "
                f"committing to execution."
            ),
        }
    else:
        present = {
            "name": "🌫️ The Mist",
            "meaning": "Still forming",
            "narrative": (
                f"Quality {quality:.2f} is below the iron threshold. "
                f"This idea needs more substance — concrete methods, "
                f"specific datasets, falsifiable hypothesis."
            ),
        }

    # FUTURE — predicted outcome
    sig = probe.get("significance", 0.5)
    overall = (quality + novelty + sig) / 3
    if overall >= 0.7:
        future = {
            "name": "👑 The Crown",
            "meaning": "High potential payoff",
            "narrative": (
                "If executed well, this idea could earn meaningful citations "
                "and meaningfully shift its sub-field. Pursue it."
            ),
        }
    elif overall >= 0.5:
        future = {
            "name": "🎯 The Bullseye",
            "meaning": "Solid but unspectacular",
            "narrative": (
                "Likely to publish at a respectable venue and earn modest "
                "citations. A reliable, professional contribution."
            ),
        }
    else:
        future = {
            "name": "⚠️ The Crossroads",
            "meaning": "Uncertain outcome",
            "narrative": (
                "Many paths from here. Either pivot the core idea, "
                "or commit to substantial revision before attempting execution."
            ),
        }

    return [past, present, future]


# ─────────────────────────────────────────────────────────────────────────────
# 25. IDEA POKÉMON CARD — collectible stat block (highly shareable)
# ─────────────────────────────────────────────────────────────────────────────

# Type system: each idea gets one of these primary types based on methodology
_POKE_TYPES = {
    "empirical_study":         {"name": "Empirical",   "color": "#0ea5e9", "icon": "📊"},
    "theoretical_analysis":    {"name": "Theoretical", "color": "#a855f7", "icon": "📐"},
    "system_design":           {"name": "Engineering", "color": "#f59e0b", "icon": "⚙️"},
    "dataset_creation":        {"name": "Curatorial",  "color": "#10b981", "icon": "📚"},
    "survey_meta_analysis":    {"name": "Synthetic",   "color": "#ec4899", "icon": "🔍"},
    "tool_library":            {"name": "Builder",     "color": "#8b5cf6", "icon": "🔧"},
    "interdisciplinary_bridge":{"name": "Bridging",    "color": "#06b6d4", "icon": "🌉"},
}


def build_pokemon_card(idea: Dict[str, Any]) -> str:
    """
    Render a collectible stat-card-style HTML block for the idea.
    Pokemon-card vibes — big shareable image-like layout.
    Returns HTML string for st.markdown.
    """
    title = idea.get("title", "Untitled Idea")[:60]
    quality = idea.get("quality_score", 0.5)
    probe = idea.get("probe_scores") or {}
    method_type = idea.get("methodology_type") or "empirical_study"
    novelty_lvl = (idea.get("novelty_level") or "moderate").capitalize()

    type_info = _POKE_TYPES.get(method_type, _POKE_TYPES["empirical_study"])

    # Stats (Pokemon-like, scale 0-99 for visual)
    hp = int(40 + 60 * quality)
    attack = int(30 + 70 * probe.get("novelty", 0.5))      # creative power
    defense = int(30 + 70 * probe.get("risk_balance", 0.5))  # robustness
    speed = int(30 + 70 * probe.get("code", 0.5))          # implementability
    special = int(30 + 70 * probe.get("significance", 0.5))  # impact potential

    # Total power score
    total = hp + attack + defense + speed + special
    rarity = ("⭐⭐⭐⭐⭐ Legendary" if total >= 380 else
              "⭐⭐⭐⭐ Epic"        if total >= 320 else
              "⭐⭐⭐ Rare"          if total >= 260 else
              "⭐⭐ Uncommon"         if total >= 200 else
              "⭐ Common")

    # "Move" name based on dominant probe
    moves = {
        "novelty":      ("Paradigm Shift", probe.get("novelty", 0.5)),
        "specificity":  ("Precision Strike", probe.get("specificity", 0.5)),
        "significance": ("Field Impact",   probe.get("significance", 0.5)),
        "code":         ("Quick Build",    probe.get("code", 0.5)),
        "scalability":  ("Mass Deploy",    probe.get("scalability", 0.5)),
    }
    # max returns (key, (label, power)); unpack carefully
    move_name, move_tuple = max(moves.items(), key=lambda kv: kv[1][1])
    move_label, move_power = move_tuple
    move_dmg = int(30 + 70 * move_power)

    # Card ID (deterministic)
    card_id = hashlib.md5(title.encode("utf-8"), usedforsecurity=False).hexdigest()[:6].upper()

    # Generate stat bar HTML
    def _stat_bar(label: str, value: int, color: str) -> str:
        pct = min(100, value)
        return (
            f'<div style="display:flex;align-items:center;margin:3px 0;font-size:11px">'
            f'<span style="width:60px;color:#475569;font-weight:600">{label}</span>'
            f'<span style="width:30px;color:#0c4a6e;font-weight:700;text-align:right;'
            f'margin-right:8px">{value}</span>'
            f'<div style="flex:1;background:#e0f2fe;border-radius:3px;height:8px;overflow:hidden">'
            f'<div style="background:{color};height:100%;width:{pct}%;border-radius:3px"></div>'
            f'</div></div>'
        )

    return (
        f'<div style="max-width:340px;margin:8px auto;'
        f'background:linear-gradient(135deg,#fef3c7 0%,#fde68a 50%,#fcd34d 100%);'
        f'border:3px solid #92400e;border-radius:14px;padding:14px;'
        f'box-shadow:0 8px 16px rgba(146,64,14,0.25);font-family:Arial,sans-serif">'
        # Header bar
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'background:linear-gradient(90deg,{type_info["color"]},{type_info["color"]}cc);'
        f'color:white;padding:6px 10px;border-radius:8px;margin-bottom:8px">'
        f'<span style="font-size:14px;font-weight:800">{title}</span>'
        f'<span style="font-size:13px;font-weight:700">HP {hp}</span>'
        f'</div>'
        # Type + image area (idea title doubles as art)
        f'<div style="background:#0c1e3a;border-radius:8px;padding:24px 12px;text-align:center;'
        f'margin-bottom:8px;border:2px solid #92400e">'
        f'<div style="font-size:48px;margin-bottom:4px">{type_info["icon"]}</div>'
        f'<div style="font-size:11px;color:#fcd34d;font-weight:700;letter-spacing:0.05em;'
        f'text-transform:uppercase">{type_info["name"]} Type</div>'
        f'<div style="font-size:10px;color:#94a3b8">Novelty: {novelty_lvl}</div>'
        f'</div>'
        # Move
        f'<div style="background:#fed7aa;border-radius:6px;padding:6px 10px;'
        f'margin-bottom:8px;display:flex;justify-content:space-between;align-items:center">'
        f'<span style="font-size:12px;font-weight:700;color:#9a3412">⚡ {move_label}</span>'
        f'<span style="font-size:14px;font-weight:800;color:#9a3412">{move_dmg}</span>'
        f'</div>'
        # Stats
        f'<div style="background:white;border-radius:6px;padding:8px 10px">'
        f'{_stat_bar("⚔️ ATK", attack, "#ef4444")}'
        f'{_stat_bar("🛡️ DEF", defense, "#10b981")}'
        f'{_stat_bar("⚡ SPD", speed, "#0ea5e9")}'
        f'{_stat_bar("✨ SP", special, "#a855f7")}'
        f'</div>'
        # Footer
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'margin-top:8px;font-size:10px;color:#92400e">'
        f'<span>#{card_id}</span>'
        f'<span style="font-weight:700">{rarity}</span>'
        f'<span>TOTAL {total}</span>'
        f'</div>'
        f'</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# 26. IDEA WEATHER FORECAST — 7-week project weather pattern
# ─────────────────────────────────────────────────────────────────────────────

def build_weather_forecast(idea: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate a 7-week weather forecast representing project execution risk.
    Each week gets a weather pattern based on which phase + risk profile.
    """
    res = estimate_resources(idea)
    probe = idea.get("probe_scores", {})
    quality = idea.get("quality_score", 0.5)

    total_weeks = max(7, min(res["time_weeks"], 16))
    # Compute 7 weeks distributed evenly across the project
    sample_weeks = [int(i * total_weeks / 7) + 1 for i in range(7)]

    # Risk profile by phase
    forecasts = []
    for w in sample_weeks:
        frac = w / total_weeks
        # Phase weather: setup is easy, dev is risky, exp is variable, paper is calm
        if frac < 0.20:    # setup
            risk = 0.2
            phase = "Setup"
        elif frac < 0.55:  # dev — risky
            risk = 0.6 - 0.4 * probe.get("code", 0.5)
            phase = "Dev"
        elif frac < 0.85:  # experiments — variable
            risk = 0.5 - 0.3 * probe.get("specificity", 0.5)
            phase = "Experiments"
        else:              # paper writing — calm
            risk = 0.15
            phase = "Writing"

        # Pick weather based on risk + quality
        adjusted_risk = max(0.0, min(1.0, risk - 0.3 * quality + 0.2 * (1 - probe.get("risk_balance", 0.5))))

        if adjusted_risk < 0.20:
            weather = {"icon": "☀️", "label": "Sunny", "color": "#fbbf24",
                       "advice": "Smooth sailing"}
        elif adjusted_risk < 0.40:
            weather = {"icon": "🌤️", "label": "Partly Cloudy", "color": "#a3e635",
                       "advice": "Mostly clear, minor clouds"}
        elif adjusted_risk < 0.60:
            weather = {"icon": "☁️", "label": "Overcast", "color": "#94a3b8",
                       "advice": "Plan checkpoints"}
        elif adjusted_risk < 0.80:
            weather = {"icon": "🌧️", "label": "Rainy", "color": "#3b82f6",
                       "advice": "Expect setbacks"}
        else:
            weather = {"icon": "⛈️", "label": "Stormy", "color": "#7c3aed",
                       "advice": "High risk — be ready to pivot"}

        forecasts.append({
            "week": w,
            "phase": phase,
            "risk": round(adjusted_risk, 2),
            **weather,
        })
    return forecasts


def weather_to_html(forecasts: List[Dict[str, Any]]) -> str:
    """Render the 7-week forecast as a horizontal weather bar."""
    parts = ['<div style="display:flex;gap:6px;margin:12px 0;flex-wrap:wrap;'
             'justify-content:center">']
    for f in forecasts:
        parts.append(
            f'<div style="flex:1;min-width:80px;max-width:120px;background:white;'
            f'border:2px solid {f["color"]};border-radius:10px;padding:8px 4px;'
            f'text-align:center;box-shadow:0 2px 6px rgba(0,0,0,0.06)">'
            f'<div style="font-size:11px;color:#64748b;font-weight:600;'
            f'text-transform:uppercase;letter-spacing:0.04em">Week {f["week"]}</div>'
            f'<div style="font-size:32px;line-height:1.2;margin:4px 0">{f["icon"]}</div>'
            f'<div style="font-size:11px;font-weight:700;color:{f["color"]}">{f["label"]}</div>'
            f'<div style="font-size:9px;color:#64748b;margin-top:2px">{f["phase"]}</div>'
            f'<div style="font-size:8px;color:#94a3b8;margin-top:2px">{f["advice"]}</div>'
            f'</div>'
        )
    parts.append('</div>')
    return "".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# 27. TWIN UNIVERSE — parallel simulations across alternative paths
# ─────────────────────────────────────────────────────────────────────────────

# Hand-picked alternative-path scenarios
_TWIN_UNIVERSES = [
    {
        "name": "🪐 Universe A: Same-but-Cheaper",
        "description": "Same idea on ¼ compute, public dataset only",
        "compute_multiplier": 0.25,
        "data_quality_boost": 0.0,
        "novelty_bet": 0.0,
        "color": "#0ea5e9",
    },
    {
        "name": "🌟 Universe B: Premium Resources",
        "description": "Same idea with 4× compute and curated data",
        "compute_multiplier": 4.0,
        "data_quality_boost": 0.2,
        "novelty_bet": 0.0,
        "color": "#10b981",
    },
    {
        "name": "💎 Universe C: High-Risk Bet",
        "description": "Same idea pushed into radical territory",
        "compute_multiplier": 1.0,
        "data_quality_boost": 0.0,
        "novelty_bet": 0.3,
        "color": "#a855f7",
    },
    {
        "name": "🎯 Universe D: Conservative Path",
        "description": "Same idea, safer assumptions, proven techniques",
        "compute_multiplier": 0.5,
        "data_quality_boost": 0.1,
        "novelty_bet": -0.2,
        "color": "#f59e0b",
    },
]


def build_twin_universe(idea: Dict[str, Any]):
    """
    Run 4 parallel simulations of the same idea under different conditions.
    Plot all 4 P10/P50/P90 ranges side by side as candlestick-style markers.
    """
    if not HAS_PLOTLY:
        return None

    base_sim = simulate_outcomes(idea, n_trials=300)

    fig = go.Figure()

    # Baseline marker
    fig.add_trace(go.Scatter(
        x=["Baseline"], y=[base_sim["p50"]],
        mode="markers",
        marker=dict(size=18, color="#64748b", symbol="diamond",
                    line=dict(color="white", width=2)),
        error_y=dict(
            type="data", symmetric=False,
            array=[base_sim["p90"] - base_sim["p50"]],
            arrayminus=[base_sim["p50"] - base_sim["p10"]],
            color="#64748b", thickness=2, width=8,
        ),
        name="Baseline",
        hovertemplate="P10: %{customdata[0]:.1f}%<br>"
                      "P50: %{y:.1f}%<br>"
                      "P90: %{customdata[1]:.1f}%<extra>Baseline</extra>",
        customdata=[[base_sim["p10"], base_sim["p90"]]],
        showlegend=False,
    ))

    # Each twin universe
    for u in _TWIN_UNIVERSES:
        sim = simulate_with_adjustments(
            idea,
            compute_multiplier=u["compute_multiplier"],
            data_quality_boost=u["data_quality_boost"],
            novelty_bet=u["novelty_bet"],
            n_trials=300,
        )
        fig.add_trace(go.Scatter(
            x=[u["name"]], y=[sim["p50"]],
            mode="markers",
            marker=dict(size=18, color=u["color"], symbol="circle",
                        line=dict(color="white", width=2)),
            error_y=dict(
                type="data", symmetric=False,
                array=[sim["p90"] - sim["p50"]],
                arrayminus=[sim["p50"] - sim["p10"]],
                color=u["color"], thickness=2, width=8,
            ),
            hovertemplate=(f"<b>{u['name']}</b><br>{u['description']}<br>"
                           f"P10: %{{customdata[0]:.1f}}%<br>"
                           f"P50: %{{y:.1f}}%<br>"
                           f"P90: %{{customdata[1]:.1f}}%<extra></extra>"),
            customdata=[[sim["p10"], sim["p90"]]],
            showlegend=False,
        ))

    fig.update_layout(
        title=dict(text="🌌 Twin Universe — Outcomes Across Alternative Paths",
                   x=0.5, font=dict(size=14)),
        yaxis_title="Predicted performance (%)",
        xaxis=dict(tickangle=-15),
        height=380,
        margin=dict(l=50, r=20, t=50, b=120),
        plot_bgcolor="#fafbfd",
    )
    return fig


def twin_universe_summary(idea: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the 4 universes with their P50 and verdict."""
    base = simulate_outcomes(idea, n_trials=200)["p50"]
    out = []
    for u in _TWIN_UNIVERSES:
        sim = simulate_with_adjustments(
            idea,
            compute_multiplier=u["compute_multiplier"],
            data_quality_boost=u["data_quality_boost"],
            novelty_bet=u["novelty_bet"],
            n_trials=200,
        )
        delta = sim["p50"] - base
        verdict = ("🚀 Better" if delta > 3 else
                   "⚖️ Similar" if abs(delta) <= 3 else
                   "📉 Worse")
        out.append({
            "name": u["name"],
            "description": u["description"],
            "color": u["color"],
            "p50": round(sim["p50"], 1),
            "delta": round(delta, 1),
            "verdict": verdict,
        })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 28. ORIGIN STORY — narrative myth of the idea
# ─────────────────────────────────────────────────────────────────────────────

def generate_origin_story(idea: Dict[str, Any]) -> str:
    """
    Generate a 3-paragraph narrative myth describing the idea's origin,
    journey, and destiny. Pure templating — no LLM needed.
    """
    title = idea.get("title", "this idea")
    method_type = (idea.get("methodology_type") or "").replace("_", " ")
    novelty_lvl = (idea.get("novelty_level") or "moderate").lower()
    quality = idea.get("quality_score", 0.5)
    probe = idea.get("probe_scores", {})
    novelty = probe.get("novelty", 0.5)
    sig = probe.get("significance", 0.5)
    code = probe.get("code", 0.5)

    # ── Paragraph 1: ORIGIN ─────────────────────────────────────────────────
    if novelty >= 0.7:
        p1 = (
            f"In a research landscape crowded with familiar paths, "
            f"<b>{title}</b> emerged from the spaces between established disciplines. "
            f"It began as a question nobody had quite asked yet: a {novelty_lvl} "
            f"departure from the {method_type or 'standard'} canon."
        )
    elif novelty >= 0.4:
        p1 = (
            f"<b>{title}</b> was born from an observation that two seemingly "
            f"separate ideas might, in fact, be two faces of the same problem. "
            f"It draws on {method_type or 'established'} traditions while "
            f"recombining their tools in a {novelty_lvl} arrangement."
        )
    else:
        p1 = (
            f"<b>{title}</b> took shape through patient iteration on a "
            f"well-understood foundation. Rather than chase the radical, "
            f"it commits to refining what already works — pushing the "
            f"{method_type or 'canonical'} approach a {novelty_lvl} step further."
        )

    # ── Paragraph 2: JOURNEY ────────────────────────────────────────────────
    if quality >= 0.7 and code >= 0.6:
        p2 = (
            f"The path forward is unusually clear. The methodology has crystallized "
            f"with quality {quality:.2f}; the implementation is tractable, the data "
            f"reachable, the experiments well-defined. What remains is the disciplined "
            f"work of execution."
        )
    elif quality >= 0.5:
        p2 = (
            f"There are still rough edges to smooth (quality {quality:.2f}). "
            f"The core insight is sound, but specific weak points — visible in "
            f"the probe scores — need attention before commitment. A round of "
            f"refinement on the weakest probe will likely move this from <i>idea</i> "
            f"to <i>blueprint</i>."
        )
    else:
        p2 = (
            f"At quality {quality:.2f}, the idea is still formative. Gaps in "
            f"specificity, feasibility, or coherence make execution premature. "
            f"The next chapter is one of refinement: substantiate the method, "
            f"name the dataset, define the metrics, then revisit."
        )

    # ── Paragraph 3: DESTINY ────────────────────────────────────────────────
    overall = (quality + novelty + sig) / 3
    if overall >= 0.7:
        p3 = (
            f"<b>If executed with care, this idea has the markings of a paper that gets cited.</b> "
            f"It addresses a problem the field is paying attention to, and the proposed "
            f"approach is novel enough to stand out. The destiny here is impact — but "
            f"only if the execution matches the ambition."
        )
    elif overall >= 0.5:
        p3 = (
            f"The expected destination is a respectable conference paper, useful to a "
            f"specific community even if it doesn't reshape the field. Reliable, "
            f"professional, citable. There is honor in solid mid-tier work."
        )
    else:
        p3 = (
            f"As written, this idea risks falling between the cracks: not novel "
            f"enough to attract attention, not refined enough to be cited as a "
            f"reference implementation. The honest path forward is to either "
            f"<b>pivot the core claim</b> or <b>commit to substantial revision</b> before "
            f"investing resources."
        )

    return f'<p style="margin-bottom:10px">{p1}</p>' \
           f'<p style="margin-bottom:10px">{p2}</p>' \
           f'<p>{p3}</p>'


# ─────────────────────────────────────────────────────────────────────────────
# 29. PROBABILITY CLOUD — 2D heatmap of (compute × data) → success
# ─────────────────────────────────────────────────────────────────────────────

def build_probability_cloud(idea: Dict[str, Any]):
    """
    2D heatmap showing P(success) across the joint space of compute × data quality.
    Lets users see the entire What-If grid at once.
    """
    if not HAS_PLOTLY:
        return None

    # Grid: compute multipliers × data quality boosts
    compute_mults = [0.25, 0.5, 1.0, 2.0, 4.0]
    data_boosts = [-0.2, -0.1, 0.0, 0.1, 0.2]

    z = []  # rows = data, cols = compute
    for db in data_boosts:
        row = []
        for cm in compute_mults:
            sim = simulate_with_adjustments(
                idea, compute_multiplier=cm,
                data_quality_boost=db, n_trials=80,
            )
            row.append(round(sim["p50"], 1))
        z.append(row)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=[f"{c}×" for c in compute_mults],
        y=[f"{db:+.1f}" for db in data_boosts],
        colorscale=[
            [0.0, "#ef4444"], [0.25, "#f59e0b"], [0.5, "#fde68a"],
            [0.75, "#86efac"], [1.0, "#10b981"],
        ],
        text=[[f"{v:.0f}%" for v in row] for row in z],
        texttemplate="%{text}",
        textfont=dict(size=11, color="#0c4a6e"),
        hovertemplate=(
            "Compute: %{x}<br>"
            "Data boost: %{y}<br>"
            "Predicted P50: %{z:.1f}%<extra></extra>"
        ),
        colorbar=dict(title="P50 (%)"),
    ))

    fig.update_layout(
        title=dict(text="🌫️ Probability Cloud — outcome across (compute × data)",
                   x=0.5, font=dict(size=14)),
        xaxis_title="Compute multiplier (× baseline)",
        yaxis_title="Data quality boost (Δ)",
        height=350,
        margin=dict(l=80, r=20, t=50, b=50),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 30. MOVIE TRAILER PITCH — Hollywood-style elevator pitch
# ─────────────────────────────────────────────────────────────────────────────

# Verbs that hook for cinematic taglines
_TRAILER_HOOKS = [
    "where machines reason",
    "where data sings",
    "where novelty breaks the rules",
    "where every probe matters",
    "where the field hits its ceiling",
    "where one idea changes everything",
    "where the impossible becomes mundane",
]


def generate_movie_trailer(idea: Dict[str, Any]) -> Dict[str, Any]:
    """Hollywood-style movie trailer for the idea: tagline, cast, rating, pitch."""
    title = idea.get("title", "Untitled")[:60]
    quality = idea.get("quality_score", 0.5)
    probe = idea.get("probe_scores", {})
    novelty = probe.get("novelty", 0.5)
    sig = probe.get("significance", 0.5)
    method_type = (idea.get("methodology_type") or "").replace("_", " ").title()

    # Deterministic hook
    hook = _TRAILER_HOOKS[abs(hash(title)) % len(_TRAILER_HOOKS)]

    # Genre by methodology type
    genre = {
        "empirical_study":          "🎯 Documentary Thriller",
        "theoretical_analysis":     "🧠 Mind-Bending Drama",
        "system_design":            "⚙️ Engineering Epic",
        "dataset_creation":         "📚 Origin Story",
        "survey_meta_analysis":     "🔍 Detective Mystery",
        "tool_library":             "🔧 Builder Saga",
        "interdisciplinary_bridge": "🌉 Buddy Adventure",
    }.get(idea.get("methodology_type"), "🎯 Documentary Thriller")

    # Tagline
    tagline_options = [
        f"In a world {hook}... one idea dares to ask the question.",
        f"They said it couldn't be done. They were wrong.",
        f"This summer, science meets ambition.",
        f"From the team that brought you {method_type or 'research'}, a new chapter begins.",
        f"Some ideas are incremental. This one is {novelty_lvl(novelty)}.",
    ]
    tagline = tagline_options[abs(hash(title + "tagline")) % len(tagline_options)]

    # 3-line pitch
    if quality >= 0.7:
        pitch = (
            f"When the field hits a wall, a single research team dares to think differently. "
            f"Combining {method_type or 'cutting-edge'} methods with a {novelty_lvl(novelty)} twist, "
            f"they set out to {(idea.get('expected_outcome') or 'change everything')[:80].lower()}. "
            f"This is the story of how it all begins."
        )
    elif quality >= 0.4:
        pitch = (
            f"It's not a sure thing. The path is uncertain, the methods unproven, "
            f"the data a question mark. But the hypothesis is too compelling to ignore: "
            f"{(idea.get('hypothesis') or 'something fundamental might be true')[:80].lower()}. "
            f"What happens next will redefine the boundaries of what's possible."
        )
    else:
        pitch = (
            f"Sometimes you don't know what you've got until you sketch it out. "
            f"This is a research idea in its rawest form — full of promise, "
            f"full of gaps, full of possibility. Coming next: substantial revision, "
            f"or a complete pivot. Either way, the journey starts here."
        )

    # Imaginary cast (researcher archetypes)
    cast = [
        {"role": "🎬 Director", "name": "Dr. " + _hash_name(title, "director"),
         "credit": method_type + " auteur"},
        {"role": "🌟 Lead Researcher", "name": _hash_name(title, "lead"),
         "credit": "Quality champion"},
        {"role": "🧪 Co-Investigator", "name": _hash_name(title, "co"),
         "credit": "Methodology specialist"},
        {"role": "💡 Theory Advisor", "name": "Prof. " + _hash_name(title, "theory"),
         "credit": "Cited 1000+ times"},
    ]

    # Rating
    if quality >= 0.7 and novelty >= 0.7 and sig >= 0.7:
        rating = "🏆 Critics' Pick — Festival Must-See"
        stars = 5
    elif quality >= 0.6:
        rating = "✅ Recommended — Strong Premise"
        stars = 4
    elif quality >= 0.4:
        rating = "⚖️ Mixed Reviews — Limited Run"
        stars = 3
    else:
        rating = "⚠️ Pre-Production — Needs Major Rewrites"
        stars = 2

    return {
        "title": title,
        "genre": genre,
        "tagline": tagline,
        "pitch": pitch,
        "cast": cast,
        "rating": rating,
        "stars": stars,
        "release": "📅 In Theaters: Submission Window 2026",
    }


def novelty_lvl(score: float) -> str:
    """Map novelty score to descriptor."""
    if score >= 0.8: return "revolutionary"
    if score >= 0.6: return "fresh"
    if score >= 0.4: return "thoughtful"
    return "incremental"


def _hash_name(seed: str, salt: str) -> str:
    """Generate a fake but deterministic researcher name."""
    first_names = ["Aria", "Cyrus", "Lina", "Marcus", "Nova", "Priya",
                   "Ravi", "Sofia", "Theo", "Yumi", "Zane", "Maya"]
    last_names = ["Chen", "Patel", "Okonkwo", "Volkov", "Reyes", "Tanaka",
                  "Almeida", "Singh", "Nakamura", "Hassan", "Kim", "Rivera"]
    h = int(hashlib.md5((seed + salt).encode("utf-8"), usedforsecurity=False).hexdigest(), 16)
    return f"{first_names[h % len(first_names)]} {last_names[(h // 7) % len(last_names)]}"


def trailer_to_html(t: Dict[str, Any]) -> str:
    """Render the movie trailer as a poster-style HTML block."""
    cast_html = "".join(
        f'<div style="margin:3px 0;font-size:11px;color:#fcd34d">'
        f'<b>{c["role"]}</b> · {c["name"]} <span style="color:#94a3b8">({c["credit"]})</span>'
        f'</div>'
        for c in t["cast"]
    )
    stars_html = "★" * t["stars"] + "☆" * (5 - t["stars"])
    return (
        f'<div style="background:linear-gradient(135deg,#0c1e3a 0%,#1e3a8a 50%,#7c3aed 100%);'
        f'border-radius:14px;padding:24px;color:white;'
        f'box-shadow:0 8px 24px rgba(124,58,237,0.3);'
        f'border:2px solid #fbbf24">'
        # Genre tag
        f'<div style="display:inline-block;background:rgba(251,191,36,0.2);'
        f'border:1px solid #fbbf24;color:#fbbf24;font-size:10px;font-weight:700;'
        f'padding:3px 10px;border-radius:12px;letter-spacing:0.08em;'
        f'text-transform:uppercase;margin-bottom:12px">{t["genre"]}</div>'
        # Title
        f'<h2 style="color:#fff;font-size:24px;margin:8px 0;line-height:1.2;'
        f'font-weight:800;letter-spacing:-0.5px">{t["title"]}</h2>'
        # Stars
        f'<div style="font-size:18px;color:#fbbf24;margin:6px 0">{stars_html}</div>'
        # Tagline
        f'<div style="font-size:14px;font-style:italic;color:#fde68a;'
        f'margin:8px 0 16px 0;line-height:1.5">"{t["tagline"]}"</div>'
        # Pitch
        f'<div style="background:rgba(255,255,255,0.05);border-radius:8px;'
        f'padding:12px 14px;margin:12px 0;font-size:13px;line-height:1.6;'
        f'color:#e0f2fe">{t["pitch"]}</div>'
        # Cast
        f'<div style="margin:14px 0">'
        f'<div style="font-size:10px;font-weight:700;color:#fbbf24;'
        f'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px">CAST</div>'
        f'{cast_html}</div>'
        # Rating
        f'<div style="margin-top:16px;padding-top:12px;border-top:1px solid rgba(255,255,255,0.15);'
        f'display:flex;justify-content:space-between;align-items:center;font-size:11px">'
        f'<span style="color:#fbbf24;font-weight:700">{t["rating"]}</span>'
        f'<span style="color:#94a3b8">{t["release"]}</span>'
        f'</div>'
        f'</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# 31. RPG QUEST LOG — next steps as quests with XP rewards
# ─────────────────────────────────────────────────────────────────────────────

def generate_quest_log(idea: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate a list of RPG-style quests representing concrete next steps.
    Each quest has difficulty, XP reward, and unlock condition.
    """
    quality = idea.get("quality_score", 0.5)
    probe = idea.get("probe_scores", {})
    quests = []

    # Main quest line driven by weakest probes
    weak_dims = sorted(
        [(k, v) for k, v in probe.items() if isinstance(v, (int, float))],
        key=lambda kv: kv[1],
    )

    # Always: "Run experiment" main quest
    quests.append({
        "type": "🌟 Main Quest",
        "title": "Validate the core hypothesis",
        "objective": f"Run a minimal experiment to test: {(idea.get('hypothesis') or 'the central claim')[:60]}",
        "difficulty": "★★★",
        "xp": 250,
        "status": "active" if quality >= 0.5 else "locked",
        "unlock_at": None if quality >= 0.5 else "Quality must reach 0.5 first",
    })

    # Side quests for top-3 weakest probes
    quest_templates = {
        "code": ("🔧 Fortify the Forge",
                 "Implement a minimal proof-of-concept in <100 lines of code."),
        "dataset": ("📚 Gather the Tomes",
                     "Identify 2-3 specific public datasets with download links."),
        "constraint": ("⚖️ Set the Boundaries",
                       "Estimate exact GPU-hours, RAM, and storage requirements."),
        "scalability": ("📈 Stress Test the Realm",
                        "Plan a 10× scaling experiment to verify generalization."),
        "specificity": ("📐 Sharpen the Blade",
                        "Rewrite method with concrete algorithm names + hyperparameters."),
        "significance": ("🌍 Map the Territory",
                          "Articulate exactly who benefits when this works."),
        "clarity": ("📖 Translate the Scrolls",
                     "Rewrite the abstract for a non-expert reader."),
        "testability": ("🎯 Define Victory",
                         "Write a falsifiable hypothesis with quantitative success criteria."),
        "novelty": ("🌠 Chart the Frontier",
                     "Find 5 most-similar prior papers; explain how this differs."),
        "risk_balance": ("🛡️ Forge the Shield",
                          "Build a 5-mode FMEA table with concrete mitigations."),
    }
    for k, score in weak_dims[:3]:
        if k in quest_templates and score < 0.7:
            t, obj = quest_templates[k]
            quests.append({
                "type": "📜 Side Quest",
                "title": t,
                "objective": obj,
                "difficulty": "★★" if score >= 0.4 else "★",
                "xp": 100 if score >= 0.4 else 50,
                "status": "active",
                "unlock_at": None,
            })

    # Bonus quest: paper writing (always last)
    if quality >= 0.7:
        quests.append({
            "type": "🏆 Endgame Quest",
            "title": "Write the paper",
            "objective": "Draft the 8-page submission for your target venue.",
            "difficulty": "★★★★",
            "xp": 500,
            "status": "locked",
            "unlock_at": "Complete validation experiment first",
        })

    return quests


def quest_log_to_html(quests: List[Dict[str, Any]]) -> str:
    """Render the quest log as a stylized RPG-style scroll."""
    items = []
    for q in quests:
        is_locked = q["status"] == "locked"
        opacity = "0.55" if is_locked else "1.0"
        bg_grad = ("linear-gradient(135deg,#374151,#1f2937)" if is_locked else
                   {"🌟 Main Quest": "linear-gradient(135deg,#f59e0b,#dc2626)",
                    "📜 Side Quest": "linear-gradient(135deg,#3b82f6,#1e40af)",
                    "🏆 Endgame Quest": "linear-gradient(135deg,#a855f7,#7c2d92)"}.get(
                       q["type"], "linear-gradient(135deg,#0ea5e9,#0c4a6e)"))
        lock_badge = ('<span style="background:rgba(0,0,0,0.4);color:#cbd5e1;'
                      'padding:2px 8px;border-radius:8px;font-size:10px;'
                      'font-weight:700">🔒 LOCKED</span>' if is_locked else
                      '<span style="background:rgba(16,185,129,0.3);color:#34d399;'
                      'padding:2px 8px;border-radius:8px;font-size:10px;'
                      'font-weight:700">⚡ ACTIVE</span>')
        items.append(
            f'<div style="background:{bg_grad};color:white;border-radius:10px;'
            f'padding:12px 16px;margin:6px 0;opacity:{opacity};'
            f'border:1px solid rgba(255,255,255,0.15)">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'margin-bottom:6px">'
            f'<div style="font-size:11px;font-weight:700;letter-spacing:0.05em;'
            f'text-transform:uppercase;color:#fde68a">{q["type"]}</div>'
            f'{lock_badge}'
            f'</div>'
            f'<div style="font-size:14px;font-weight:700;margin-bottom:4px">{q["title"]}</div>'
            f'<div style="font-size:12px;line-height:1.5;color:#e0f2fe;'
            f'margin-bottom:8px">{q["objective"]}</div>'
            f'<div style="display:flex;justify-content:space-between;font-size:11px;'
            f'color:#fbbf24;font-weight:700">'
            f'<span>Difficulty: {q["difficulty"]}</span>'
            f'<span>+{q["xp"]} XP</span>'
            f'</div>'
            + (f'<div style="font-size:10px;color:#94a3b8;margin-top:4px;'
               f'font-style:italic">⚠️ {q["unlock_at"]}</div>' if q.get("unlock_at") else "")
            + f'</div>'
        )
    return ''.join(items)


# ─────────────────────────────────────────────────────────────────────────────
# 32. CONFERENCE MATCH — auto-match idea to ML conferences
# ─────────────────────────────────────────────────────────────────────────────

# ML conferences with deadlines + acceptance criteria
_CONFERENCES = [
    {
        "name": "NeurIPS",
        "full": "Conference on Neural Information Processing Systems",
        "tier": "A*", "deadline": "May 22",
        "acceptance_rate": 25, "min_quality": 0.65,
        "weight_novelty": 0.4, "weight_significance": 0.3, "weight_specificity": 0.3,
        "color": "#7c3aed",
    },
    {
        "name": "ICML",
        "full": "International Conference on Machine Learning",
        "tier": "A*", "deadline": "Jan 31",
        "acceptance_rate": 27, "min_quality": 0.60,
        "weight_novelty": 0.35, "weight_significance": 0.30, "weight_testability": 0.35,
        "color": "#0ea5e9",
    },
    {
        "name": "ICLR",
        "full": "International Conference on Learning Representations",
        "tier": "A*", "deadline": "Sep 27",
        "acceptance_rate": 32, "min_quality": 0.55,
        "weight_novelty": 0.40, "weight_clarity": 0.25, "weight_specificity": 0.35,
        "color": "#10b981",
    },
    {
        "name": "AAAI",
        "full": "Association for the Advancement of Artificial Intelligence",
        "tier": "A", "deadline": "Aug 15",
        "acceptance_rate": 24, "min_quality": 0.50,
        "weight_significance": 0.4, "weight_specificity": 0.3, "weight_testability": 0.3,
        "color": "#f59e0b",
    },
    {
        "name": "Workshop",
        "full": "Domain workshop (NeurIPS / ICML satellite)",
        "tier": "B", "deadline": "Rolling",
        "acceptance_rate": 60, "min_quality": 0.30,
        "weight_specificity": 0.5, "weight_clarity": 0.5,
        "color": "#94a3b8",
    },
]


def match_conferences(idea: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Score each conference for fit with this idea, sorted by best match.
    """
    quality = idea.get("quality_score", 0.5)
    probe = idea.get("probe_scores", {})

    matches = []
    for conf in _CONFERENCES:
        # Compute weighted match score from relevant probes
        score = 0.0
        weight_sum = 0.0
        for key, val in conf.items():
            if key.startswith("weight_"):
                probe_key = key.replace("weight_", "")
                pscore = probe.get(probe_key, 0.5)
                if isinstance(pscore, (int, float)):
                    score += val * pscore
                    weight_sum += val
        normalized = (score / weight_sum) if weight_sum > 0 else 0.5

        # Multiply by quality factor
        match_score = round(normalized * 0.6 + quality * 0.4, 2)

        # Decision
        if match_score >= conf["min_quality"] + 0.10:
            verdict = "🎯 Strong Fit"
            verdict_color = "#10b981"
        elif match_score >= conf["min_quality"]:
            verdict = "⚖️ Borderline"
            verdict_color = "#f59e0b"
        else:
            verdict = "❌ Below Bar"
            verdict_color = "#ef4444"

        matches.append({
            "name": conf["name"],
            "full": conf["full"],
            "tier": conf["tier"],
            "deadline": conf["deadline"],
            "acceptance_rate": conf["acceptance_rate"],
            "match_score": match_score,
            "verdict": verdict,
            "verdict_color": verdict_color,
            "color": conf["color"],
        })

    matches.sort(key=lambda x: x["match_score"], reverse=True)
    return matches


def conference_match_to_html(matches: List[Dict[str, Any]]) -> str:
    """Render conference matches as a list of styled cards."""
    items = []
    for i, m in enumerate(matches):
        rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
        items.append(
            f'<div style="background:white;border-left:5px solid {m["color"]};'
            f'border:1px solid #e0f2fe;border-radius:10px;padding:12px 16px;'
            f'margin:6px 0;box-shadow:0 1px 3px rgba(0,0,0,0.04);'
            f'display:flex;justify-content:space-between;align-items:center">'
            f'<div style="flex:1">'
            f'<div style="font-size:14px;font-weight:700;color:#0c4a6e">'
            f'{rank_emoji} {m["name"]} '
            f'<span style="background:{m["color"]};color:white;font-size:10px;'
            f'padding:2px 8px;border-radius:8px;font-weight:700;margin-left:6px">'
            f'Tier {m["tier"]}</span></div>'
            f'<div style="font-size:11px;color:#64748b;margin:2px 0">'
            f'{m["full"]}</div>'
            f'<div style="font-size:11px;color:#64748b">'
            f'📅 {m["deadline"]} · Acceptance {m["acceptance_rate"]}%</div>'
            f'</div>'
            f'<div style="text-align:right;margin-left:12px">'
            f'<div style="background:{m["verdict_color"]};color:white;'
            f'font-size:11px;font-weight:700;padding:4px 10px;border-radius:8px">'
            f'{m["verdict"]}</div>'
            f'<div style="font-size:10px;color:#64748b;margin-top:3px">'
            f'Match: <b>{m["match_score"]:.2f}</b></div>'
            f'</div>'
            f'</div>'
        )
    return ''.join(items)


# ─────────────────────────────────────────────────────────────────────────────
# 33. IDEA MOSAIC — fractal stained-glass art
# ─────────────────────────────────────────────────────────────────────────────

def build_idea_mosaic(idea: Dict[str, Any]):
    """
    Stained-glass mosaic generated deterministically from the idea.
    Each tile color is chosen from idea attributes; layout uses hash bytes.
    """
    if not HAS_PLOTLY:
        return None

    title = idea.get("title", "Idea")
    h = hashlib.sha256((title + idea.get("method", "")).encode("utf-8")).digest()

    quality = idea.get("quality_score", 0.5)
    probe = idea.get("probe_scores") or {}
    novelty = probe.get("novelty", 0.5)

    # 6×6 grid (36 tiles), each colored from a palette modulated by hash
    n_rows, n_cols = 6, 6
    palettes = {
        "high_q":   ["#0c4a6e", "#0369a1", "#0284c7", "#0ea5e9", "#38bdf8",
                     "#7dd3fc", "#bae6fd", "#e0f2fe"],
        "mid_q":    ["#3a3a4a", "#5a5a6a", "#7a7a8a", "#9a9aaa", "#babacc",
                     "#dadaee", "#f0f0f8", "#a855f7"],
        "low_q":    ["#7c2d12", "#9a3412", "#c2410c", "#dc2626", "#ef4444",
                     "#f87171", "#fca5a5", "#fecaca"],
        "novel":    ["#581c87", "#7c3aed", "#a855f7", "#c084fc", "#f59e0b",
                     "#fbbf24", "#fde68a", "#10b981"],
    }
    if novelty >= 0.65:
        palette = palettes["novel"]
    elif quality >= 0.65:
        palette = palettes["high_q"]
    elif quality >= 0.4:
        palette = palettes["mid_q"]
    else:
        palette = palettes["low_q"]

    fig = go.Figure()

    # Generate 36 tiles with deterministic colors and slight size variations
    for i in range(n_rows):
        for j in range(n_cols):
            idx = i * n_cols + j
            color = palette[h[idx % len(h)] % len(palette)]
            # Slight overlap for stained-glass feel
            offset_x = (h[(idx * 2) % len(h)] / 255 - 0.5) * 0.15
            offset_y = (h[(idx * 2 + 1) % len(h)] / 255 - 0.5) * 0.15
            x_center = j + offset_x
            y_center = (n_rows - 1 - i) + offset_y
            # Tile as a square scatter marker
            fig.add_trace(go.Scatter(
                x=[x_center], y=[y_center],
                mode="markers",
                marker=dict(
                    size=72, symbol="square",
                    color=color,
                    line=dict(color="#0c1e3a", width=2),
                    opacity=0.85,
                ),
                hoverinfo="skip", showlegend=False,
            ))

    # Center medallion: idea title ID
    medal_id = hashlib.md5(
        title.encode("utf-8"), usedforsecurity=False,
    ).hexdigest()[:6].upper()
    fig.add_trace(go.Scatter(
        x=[(n_cols - 1) / 2], y=[(n_rows - 1) / 2],
        mode="markers+text",
        marker=dict(size=70, color="rgba(0,0,0,0.65)",
                    line=dict(color="#fbbf24", width=2)),
        text=f"#{medal_id}",
        textfont=dict(size=11, color="#fbbf24", family="Arial Black"),
        textposition="middle center",
        hoverinfo="text",
        hovertext=f"Idea: {title[:60]}",
        showlegend=False,
    ))

    fig.update_layout(
        title=dict(text=f"🎨 Idea Mosaic — Stained Glass #{medal_id}",
                   x=0.5, font=dict(size=14)),
        xaxis=dict(visible=False, range=[-0.7, n_cols - 0.3], scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False, range=[-0.7, n_rows - 0.3]),
        height=440,
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor="#0c1e3a",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 34. ACCEPTANCE SPEECH — best-paper award speech preview
# ─────────────────────────────────────────────────────────────────────────────

def generate_acceptance_speech(idea: Dict[str, Any]) -> str:
    """
    Generate a 30-second 'best paper award' acceptance speech for the idea.
    Pure templating — humor + motivation.
    """
    title = idea.get("title", "this work")
    quality = idea.get("quality_score", 0.5)
    probe = idea.get("probe_scores", {})
    novelty = probe.get("novelty", 0.5)
    method_type = (idea.get("methodology_type") or "research").replace("_", " ")

    # Different opening based on novelty
    if novelty >= 0.7:
        opening = (
            f"Wow. Honestly, I didn't prepare a speech. When we started "
            f"working on <b>{title}</b>, the consensus was that this was "
            f"a long shot. Here we are."
        )
    elif novelty >= 0.4:
        opening = (
            f"Thank you. <b>{title}</b> began as a conversation between "
            f"two papers nobody thought belonged in the same room."
        )
    else:
        opening = (
            f"Thank you so much. <b>{title}</b> is the result of patient, "
            f"incremental work — and I think there's a lesson in that."
        )

    # Acknowledgments (auto-fake co-authors from hashing)
    coa1 = _hash_name(title, "coa1")
    coa2 = _hash_name(title, "coa2")
    advisor = _hash_name(title, "advisor")

    body = (
        f"This work would not exist without my co-authors. {coa1}, "
        f"who insisted on the {method_type} angle when I wanted to give up. "
        f"{coa2}, who wrote 4am code for three weeks straight. "
        f"And of course Prof. {advisor}, who refused to accept any version "
        f"of the abstract that mentioned the word \"interesting\"."
    )

    # Closing depends on quality
    if quality >= 0.7:
        closing = (
            f"To everyone working on hard problems with no clear answer: "
            f"keep going. The paper that gets published is rarely the first version. "
            f"It's the one you stuck with through 30 rejections and one acceptance. "
            f"Thank you."
        )
    elif quality >= 0.4:
        closing = (
            f"And to my students: I lied about how easy this would be. "
            f"I'm sorry. But I'd lie again. Thank you."
        )
    else:
        closing = (
            f"And finally, to the reviewers who saw potential where others didn't: "
            f"thank you for the second chance. We tried not to let you down."
        )

    return (
        f'<p style="margin-bottom:12px">{opening}</p>'
        f'<p style="margin-bottom:12px">{body}</p>'
        f'<p>{closing}</p>'
    )


def tarot_to_html(cards: List[Dict[str, str]]) -> str:
    """Render the 3 tarot cards as a stylized HTML block (for st.markdown)."""
    parts = ['<div style="display:flex;gap:12px;justify-content:center;'
             'margin:12px 0;flex-wrap:wrap">']
    titles = ["🕰️ Past", "⚡ Present", "🔮 Future"]
    bg_grads = [
        "linear-gradient(135deg,#0c1e3a 0%,#1e3a8a 100%)",
        "linear-gradient(135deg,#7c2d12 0%,#c2410c 100%)",
        "linear-gradient(135deg,#581c87 0%,#9333ea 100%)",
    ]
    for title, bg, card in zip(titles, bg_grads, cards):
        parts.append(
            f'<div style="flex:1;min-width:160px;max-width:240px;'
            f'background:{bg};color:white;border-radius:12px;'
            f'padding:14px 16px;box-shadow:0 4px 12px rgba(0,0,0,0.15);'
            f'border:2px solid rgba(255,255,255,0.2)">'
            f'<div style="font-size:11px;font-weight:600;opacity:0.7;'
            f'text-transform:uppercase;letter-spacing:0.08em">{title}</div>'
            f'<div style="font-size:16px;font-weight:800;margin:4px 0">{card["name"]}</div>'
            f'<div style="font-size:11px;opacity:0.8;margin-bottom:8px;'
            f'font-style:italic">— {card["meaning"]}</div>'
            f'<div style="font-size:12px;line-height:1.5;opacity:0.95">'
            f'{card["narrative"]}</div>'
            f'</div>'
        )
    parts.append('</div>')
    return "".join(parts)
