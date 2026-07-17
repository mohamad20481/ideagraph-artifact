"""
pipeline_simulator.py — admin-facing visual walkthrough of how IdeaGraph
processes a research topic end-to-end.

Renders a deterministic, no-LLM "demo run" so admins can show investors,
new users, or themselves the actual machinery: how a topic becomes a
Knowledge DAG, how the QD archive routes ideas to cells by methodology
× novelty, how probes score them, how the new execution-revision loop
re-weights quality, and how Pareto replacement decides what survives.

Public API:
    render_pipeline_simulator(st_module)  — full Streamlit panel
    architecture_figure()                 — Plotly graph_objects.Figure
    fake_pipeline_run(topic, seed=42)     — deterministic data generator
"""
from __future__ import annotations

import hashlib
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Stage descriptions — sourced from agents/, pipeline.py, intelligence.py
# Kept in sync with the actual code by quoting public class/function names.
# ─────────────────────────────────────────────────────────────────────────────

PIPELINE_STAGES: List[Dict[str, str]] = [
    {
        "icon": "🧠",
        "name": "Knowledge DAG",
        "agent": "KnowledgeArchitect",
        "purpose": "Build a directed-acyclic graph of papers connected by "
                    "forward citation, backward citation, and lateral concept "
                    "links from the topic seed.",
        "inputs": "topic string",
        "outputs": "DAG: 30–80 nodes × 3 edge types",
        "color": "#0ea5e9",
    },
    {
        "icon": "🎯",
        "name": "Diversity Routing",
        "agent": "DiversityManager",
        "purpose": "Pick the next target QD cell (methodology × novelty) and "
                    "the source strategy (A/B/C) using Thompson sampling over "
                    "past success rates.",
        "inputs": "current archive, iteration",
        "outputs": "(target_cell, strategy)",
        "color": "#a855f7",
    },
    {
        "icon": "💡",
        "name": "Idea Generation",
        "agent": "IdeationAgent",
        "purpose": "Produce a 7-field research idea (title, motivation, method, "
                    "hypothesis, resources, expected outcome, risks) routed to "
                    "the target cell. Uses literature context, exemplars, and "
                    "the chosen strategy.",
        "inputs": "DAG nodes + (cell, strategy)",
        "outputs": "Idea dataclass",
        "color": "#10b981",
    },
    {
        "icon": "🔎",
        "name": "Semantic De-Duplication",
        "agent": "SemanticNoveltyChecker",
        "purpose": "Reject ideas whose method text overlaps too closely with "
                    "anything already archived (cosine similarity > 0.55 on "
                    "method embeddings).",
        "inputs": "candidate idea + archived methods",
        "outputs": "is_novel: bool",
        "color": "#f59e0b",
    },
    {
        "icon": "🧪",
        "name": "10-Probe Critique",
        "agent": "ExecutionCritic",
        "purpose": "One LLM call scores the idea on 10 dimensions: code, "
                    "dataset, constraint, novelty (must ≥ 0.4 to pass) plus "
                    "specificity, significance, clarity, testability, "
                    "scalability, risk_balance.",
        "inputs": "idea",
        "outputs": "probe_scores dict + quality (weighted avg)",
        "color": "#ec4899",
    },
    {
        "icon": "🔁",
        "name": "Execution Revision",
        "agent": "execution_revisor (NEW)",
        "purpose": "Tiny-experiment LLM proxy (1k samples × 1 seed, smaller "
                    "model). Bayesian-blends the resulting feasibility signal "
                    "with the probe quality. Closes the −0.34 feasibility gap.",
        "inputs": "probe-passing idea",
        "outputs": "blended quality + trust weight + 95% CI",
        "color": "#dc2626",
    },
    {
        "icon": "🗂️",
        "name": "QD Archive Insert",
        "agent": "QDArchive",
        "purpose": "Pareto replacement: keep the new idea iff it dominates "
                    "the cell incumbent on quality OR a probe-score axis. "
                    "21-cell grid: 7 methodologies × 3 novelty levels.",
        "inputs": "idea + cell incumbent",
        "outputs": "updated archive",
        "color": "#0c4a6e",
    },
    {
        "icon": "⚖️",
        "name": "Debate Tournament",
        "agent": "DebateOrchestrator",
        "purpose": "Top-K survivors are paired round-robin; opposing agents "
                    "argue strengths/weaknesses; a judge agent ranks them. "
                    "Optional, gated by DEBATE_ENABLED.",
        "inputs": "top-K archive ideas",
        "outputs": "ranked ideas with debate_score",
        "color": "#7c3aed",
    },
]


METHODOLOGY_TYPES: List[str] = [
    "empirical_study", "theoretical_analysis", "system_design",
    "dataset_creation", "survey_meta_analysis", "tool_library",
    "interdisciplinary_bridge",
]
NOVELTY_LEVELS: List[str] = ["incremental", "moderate", "substantial"]


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic fake-pipeline-run generator (no LLM, no DB, ~1ms)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FakeIdea:
    title: str
    method_type: str
    novelty: str
    probe_scores: Dict[str, float]
    quality: float
    archived: bool = False
    rejected_reason: str = ""
    execution_signal: Optional[float] = None
    execution_trust: Optional[float] = None
    blended_quality: Optional[float] = None
    iteration: int = 0


@dataclass
class FakeDAGNode:
    """A synthetic paper node in the Knowledge DAG."""
    id: str
    title: str
    year: int
    cluster: int        # 0..n_clusters-1, used for color
    citations: int      # plausible citation count, drives node size
    is_seed: bool = False     # original topic-derived node
    is_frontier: bool = False  # interesting expansion point


@dataclass
class FakeDAGEdge:
    """An edge in the Knowledge DAG. kind ∈ {forward, backward, lateral}."""
    source: str
    target: str
    kind: str


@dataclass
class FakeRun:
    topic: str
    seed: int
    dag_nodes: int
    dag_edges: int
    ideas: List[FakeIdea] = field(default_factory=list)
    archive: Dict[Tuple[int, int], FakeIdea] = field(default_factory=dict)
    # The real KnowledgeArchitect builds a DAG with forward/backward/lateral
    # edges. We synthesize one with the same structure here so the admin
    # can SEE what stage 1 actually produces, not just count it.
    dag_node_objs: List["FakeDAGNode"] = field(default_factory=list)
    dag_edge_objs: List["FakeDAGEdge"] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        return len(self.archive) / 21.0

    @property
    def mean_quality(self) -> float:
        if not self.archive:
            return 0.0
        qs = [i.blended_quality if i.blended_quality is not None else i.quality
               for i in self.archive.values()]
        return sum(qs) / len(qs)


# Seed catalog of plausible-looking idea titles per methodology.
# Real ideas are richer; this is just enough to make the demo concrete.
_TITLE_BANK: Dict[str, List[str]] = {
    "empirical_study": [
        "Benchmarking {x} across 5 architectures",
        "Empirical study of {x} robustness under domain shift",
        "Scaling laws for {x} on cross-domain evaluation",
        "Statistical power analysis of {x} replication studies",
    ],
    "theoretical_analysis": [
        "A capacity-theoretic analysis of {x}",
        "Generalization bounds for {x} under sparsity",
        "Information-theoretic limits of {x}",
        "Convergence guarantees for {x} optimization",
    ],
    "system_design": [
        "An end-to-end pipeline for {x} at scale",
        "A streaming architecture for online {x}",
        "Differential-privacy-preserving {x} infrastructure",
        "Federated {x} with byzantine-robust aggregation",
    ],
    "dataset_creation": [
        "A counterfactual benchmark for {x}",
        "Crowdsourced annotations for fine-grained {x}",
        "Multi-modal corpus for {x} probing",
        "Adversarial test set for {x} robustness",
    ],
    "survey_meta_analysis": [
        "A systematic review of {x} methodologies (2018–2025)",
        "Meta-analysis of effect sizes in {x}",
        "Taxonomy of failure modes in {x}",
        "Reproducibility audit of recent {x} claims",
    ],
    "tool_library": [
        "Open-source toolkit for {x} interpretation",
        "Lightweight library for {x} debugging",
        "{x}-Bench: a unified evaluation framework",
        "Auto-tuning helper for {x} hyperparameters",
    ],
    "interdisciplinary_bridge": [
        "Bridging {x} with cognitive neuroscience",
        "Causal-inference perspectives on {x}",
        "Applying {x} to materials discovery",
        "{x} for early-stage clinical decision support",
    ],
}


def _rng_for(topic: str, seed: int) -> random.Random:
    h = hashlib.sha256(f"{topic}|{seed}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


def _fake_probe_scores(rng: random.Random, base_q: float) -> Dict[str, float]:
    """Generate a plausible probe-scores dict whose weighted-mean ≈ base_q."""
    keys = ["code", "dataset", "constraint", "novelty",
            "specificity", "significance", "clarity",
            "testability", "scalability", "risk_balance"]
    return {k: max(0.0, min(1.0, rng.gauss(base_q, 0.10))) for k in keys}


def fake_pipeline_run(
    topic: str,
    seed: int = 42,
    n_ideas: int = 16,
    enable_revision: bool = True,
    saturation_fn: Optional[Any] = None,
) -> FakeRun:
    """Deterministic synthetic pipeline run. Pure Python, no I/O.

    `saturation_fn(cell)` is an optional federation hook used by
    federated_diversity.simulate_population. When provided, the
    methodology/novelty cell-selection step biases AWAY from cells the
    function reports as globally saturated. Default None = independent
    mode (the homogenization scenario).
    """
    rng = _rng_for(topic, seed)
    topic_short = (topic or "this domain").strip().split()
    topic_kw = " ".join(topic_short[:3]) if topic_short else "this domain"

    n_nodes = rng.randint(40, 70)
    run = FakeRun(
        topic=topic,
        seed=seed,
        dag_nodes=n_nodes,
        dag_edges=0,  # filled in below from the actual edges generated
    )
    # Build the synthetic Knowledge DAG so the admin can visualize what the
    # KnowledgeArchitect stage produces, not just count its outputs.
    _build_fake_dag(run, rng, topic_kw, n_nodes)

    for i in range(n_ideas):
        # Pick methodology weighted toward empirical/system/theoretical
        m_weights = [3.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0]
        n_weights = [2.0, 4.0, 2.0]
        # Federation hook: bias AWAY from globally-saturated cells. We
        # construct the joint (mi, ni) weight matrix by multiplying the
        # marginal weights, then attenuating each cell by (1 - penalty).
        # This is the population-scale homogenization fix from the paper.
        if saturation_fn is not None:
            joint_weights = []
            joint_keys = []
            for mi in range(7):
                for ni in range(3):
                    pen = float(saturation_fn((mi, ni)) or 0.0)
                    pen = max(0.0, min(1.0, pen))
                    w = m_weights[mi] * n_weights[ni] * (1.0 - pen)
                    joint_weights.append(w)
                    joint_keys.append((mi, ni))
            # Avoid all-zero collapse if every cell is saturated
            if sum(joint_weights) < 1e-9:
                joint_weights = [m_weights[mi] * n_weights[ni]
                                  for mi, ni in joint_keys]
            method_idx, nov_idx = rng.choices(joint_keys, weights=joint_weights)[0]
        else:
            method_idx = rng.choices(range(7), weights=m_weights)[0]
            nov_idx = rng.choices([0, 1, 2], weights=n_weights)[0]
        method = METHODOLOGY_TYPES[method_idx]
        # Novelty roughly normal around moderate
        novelty = NOVELTY_LEVELS[nov_idx]

        # Most ideas are "decent" (0.4-0.8); a few are weak; a few brilliant
        base_q = max(0.05, min(0.95, rng.gauss(0.55, 0.18)))
        probes = _fake_probe_scores(rng, base_q)

        title_template = rng.choice(_TITLE_BANK[method])
        title = title_template.format(x=topic_kw)

        idea = FakeIdea(
            title=title,
            method_type=method,
            novelty=novelty,
            probe_scores=probes,
            quality=base_q,
            iteration=i,
        )

        # Simulate the de-dup gate (~12% rejection rate)
        if rng.random() < 0.12:
            idea.rejected_reason = "semantically duplicate (>0.55 cosine)"
            run.ideas.append(idea)
            continue

        # Simulate the probe-pass gate
        passes = (probes["code"] >= 0.4 and probes["dataset"] >= 0.4
                  and probes["constraint"] >= 0.4 and probes["novelty"] >= 0.4
                  and base_q >= 0.30)
        if not passes:
            idea.rejected_reason = "failed core-4 probe gate"
            run.ideas.append(idea)
            continue

        # Execution-aware revision (Bayesian blend with mock signal)
        if enable_revision:
            # exec_signal correlates with quality but adds noise — sometimes
            # surprising us positively, sometimes negatively. This is the
            # whole point of the loop.
            exec_signal = max(0.0, min(1.0, rng.gauss(base_q - 0.05, 0.18)))
            trust = 0.34  # 1k×1 default per the calibration in the module
            blended = (1 - trust) * base_q + trust * exec_signal
            idea.execution_signal = exec_signal
            idea.execution_trust = trust
            idea.blended_quality = blended
            effective_q = blended
        else:
            effective_q = base_q

        # Pareto-style cell replacement: keep iff better than incumbent
        cell_key = (method_idx, nov_idx)
        existing = run.archive.get(cell_key)
        existing_q = (
            existing.blended_quality if existing and existing.blended_quality is not None
            else (existing.quality if existing else -1)
        )
        if existing is None or effective_q > existing_q:
            idea.archived = True
            run.archive[cell_key] = idea
            if existing is not None:
                existing.archived = False
                existing.rejected_reason = (
                    f"replaced by better idea in cell ({method_idx},{nov_idx})"
                )
        else:
            idea.rejected_reason = (
                f"cell ({method_idx},{nov_idx}) already holds higher q={existing_q:.2f}"
            )

        run.ideas.append(idea)

    return run


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic Knowledge DAG generator
# ─────────────────────────────────────────────────────────────────────────────

# Plausible-looking paper-title fragments per cluster theme. The real
# KnowledgeArchitect pulls these from Semantic Scholar; we synthesize.
_PAPER_PHRASES: List[str] = [
    "{x}: a foundational survey",
    "On the limits of {x}",
    "Scaling {x} with attention",
    "Self-supervised {x} pretraining",
    "Compositional {x} via graphs",
    "Robust {x} under distribution shift",
    "Causal perspectives on {x}",
    "Interpretable {x} at scale",
    "Cross-domain transfer for {x}",
    "Few-shot {x} with meta-learning",
    "Federated {x} with privacy guarantees",
    "Probing the geometry of {x}",
    "A unified framework for {x}",
    "{x} at the data-quality frontier",
    "Adversarial {x}: attacks and defenses",
    "Low-resource {x}: a meta-analysis",
    "Multi-modal {x}: bridging text and images",
    "Reproducibility issues in {x}",
    "Calibration in modern {x} systems",
    "Lessons from a decade of {x}",
    "An empirical study of {x} optimizers",
    "Bias and fairness in {x}",
    "Active learning for {x}",
    "Continual {x} without catastrophic forgetting",
    "Efficient {x} for edge devices",
]


def _build_fake_dag(run: FakeRun, rng: random.Random,
                     topic_kw: str, n_nodes: int) -> None:
    """Synthesize a deterministic Knowledge DAG and attach it to `run`.

    Structure mirrors what KnowledgeArchitect produces:
      - 4–6 thematic clusters
      - Forward edges (newer → older citation)
      - Backward edges (older paper → its citing successor)
      - Lateral edges (cross-cluster concept links)
      - A handful of frontier nodes (interesting expansion points)
    """
    n_clusters = rng.randint(4, 6)
    base_year = rng.randint(2014, 2018)
    nodes: List[FakeDAGNode] = []
    title_pool = list(_PAPER_PHRASES)
    rng.shuffle(title_pool)

    for i in range(n_nodes):
        cluster = i % n_clusters if i < n_clusters else rng.randint(0, n_clusters - 1)
        year = base_year + rng.randint(0, 11)  # spans roughly a decade
        # Citations follow a rough power-law: most papers low, a few high
        cites = int(rng.lognormvariate(2.5, 1.0))
        cites = max(1, min(cites, 1500))
        title_template = title_pool[i % len(title_pool)]
        title = title_template.format(x=topic_kw)
        is_seed = i < n_clusters  # one seed per cluster
        nodes.append(FakeDAGNode(
            id=f"P{i:03d}",
            title=title[:80],
            year=year,
            cluster=cluster,
            citations=cites,
            is_seed=is_seed,
            is_frontier=False,
        ))

    # Mark ~10% of high-citation nodes as frontier expansions
    frontier_pool = sorted(nodes, key=lambda n: n.citations, reverse=True)
    for node in frontier_pool[:max(2, n_nodes // 10)]:
        node.is_frontier = True

    # Build edges with a target mix of ~50% forward / ~25% backward / ~25%
    # lateral, matching what KnowledgeArchitect actually produces. We do
    # this by *targeted* sampling: most edges are sampled from same-cluster
    # node pairs (citation chains), and a smaller number are cross-cluster.
    edges: List[FakeDAGEdge] = []
    seen = set()
    by_cluster: Dict[int, List[FakeDAGNode]] = {}
    for node in nodes:
        by_cluster.setdefault(node.cluster, []).append(node)

    target_edges = int(n_nodes * rng.uniform(2.2, 3.4))
    target_forward = int(target_edges * 0.50)
    target_backward = int(target_edges * 0.25)
    target_lateral = target_edges - target_forward - target_backward

    def _add_edge(src: str, tgt: str, kind: str) -> bool:
        if src == tgt:
            return False
        key = (src, tgt)
        if key in seen or (tgt, src) in seen:
            return False
        edges.append(FakeDAGEdge(source=src, target=tgt, kind=kind))
        seen.add(key)
        return True

    # Forward (newer → older within same cluster)
    forward_attempts = 0
    fwd_count = 0
    while fwd_count < target_forward and forward_attempts < target_forward * 8:
        forward_attempts += 1
        cluster = rng.choice(list(by_cluster.keys()))
        members = by_cluster[cluster]
        if len(members) < 2:
            continue
        a, b = rng.sample(members, 2)
        # newer (higher year) cites older (lower year) — that's "forward"
        if a.year < b.year:
            a, b = b, a
        if _add_edge(a.id, b.id, "forward"):
            fwd_count += 1

    # Backward (older paper → its later successor, same cluster)
    bwd_attempts = 0
    bwd_count = 0
    while bwd_count < target_backward and bwd_attempts < target_backward * 8:
        bwd_attempts += 1
        cluster = rng.choice(list(by_cluster.keys()))
        members = by_cluster[cluster]
        if len(members) < 2:
            continue
        a, b = rng.sample(members, 2)
        if a.year > b.year:
            a, b = b, a  # ensure a is older
        if _add_edge(a.id, b.id, "backward"):
            bwd_count += 1

    # Lateral (cross-cluster concept link)
    lat_attempts = 0
    lat_count = 0
    cluster_keys = list(by_cluster.keys())
    while lat_count < target_lateral and lat_attempts < target_lateral * 8:
        lat_attempts += 1
        if len(cluster_keys) < 2:
            break
        c1, c2 = rng.sample(cluster_keys, 2)
        a = rng.choice(by_cluster[c1])
        b = rng.choice(by_cluster[c2])
        if _add_edge(a.id, b.id, "lateral"):
            lat_count += 1

    run.dag_node_objs = nodes
    run.dag_edge_objs = edges
    run.dag_edges = len(edges)


# ─────────────────────────────────────────────────────────────────────────────
# Plotly figures
# ─────────────────────────────────────────────────────────────────────────────

def architecture_figure():
    """Pipeline architecture as a horizontal flowchart with annotations."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    n = len(PIPELINE_STAGES)
    xs = list(range(n))

    fig = go.Figure()

    # Connecting arrows (drawn first so the boxes overlay them)
    for i in range(n - 1):
        fig.add_annotation(
            x=xs[i + 1] - 0.12, y=0,
            ax=xs[i] + 0.12, ay=0,
            xref="x", yref="y", axref="x", ayref="y",
            arrowhead=3, arrowsize=1.4, arrowwidth=2,
            arrowcolor="rgba(100,116,139,0.55)",
            showarrow=True,
        )

    # Stage nodes
    fig.add_trace(go.Scatter(
        x=xs, y=[0] * n,
        mode="markers+text",
        marker=dict(
            size=68,
            color=[s["color"] for s in PIPELINE_STAGES],
            line=dict(width=3, color="white"),
            symbol="circle",
        ),
        text=[s["icon"] for s in PIPELINE_STAGES],
        textfont=dict(size=22),
        textposition="middle center",
        hovertemplate=[
            f"<b>{s['name']}</b><br>"
            f"<i>{s['agent']}</i><br><br>"
            f"{s['purpose']}<br><br>"
            f"<b>Inputs:</b> {s['inputs']}<br>"
            f"<b>Outputs:</b> {s['outputs']}<extra></extra>"
            for s in PIPELINE_STAGES
        ],
        showlegend=False,
    ))

    # Stage names below
    for i, s in enumerate(PIPELINE_STAGES):
        fig.add_annotation(
            x=xs[i], y=-0.55,
            text=f"<b>{s['name']}</b><br><span style='font-size:9px;color:#64748b'>"
                 f"{s['agent']}</span>",
            showarrow=False,
            font=dict(size=11, color="#0c4a6e"),
            xref="x", yref="y",
        )

    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=20, b=80),
        xaxis=dict(
            range=[-0.6, n - 0.4],
            showgrid=False, showline=False, showticklabels=False, zeroline=False,
        ),
        yaxis=dict(
            range=[-1.3, 1.3],
            showgrid=False, showline=False, showticklabels=False, zeroline=False,
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor="white", font_size=12, namelength=-1),
    )
    return fig


def dag_figure(run: FakeRun):
    """Render the synthetic Knowledge DAG as an interactive force-directed graph.

    Reuses the same Fruchterman-Reingold layout used by graph_explorer.py for
    real DAGs. Nodes are colored by cluster, sized by log(citations), and
    seed/frontier nodes get distinguishing rings. Edges are colored by kind:
    forward citation (sky), backward citation (purple), lateral concept (amber).
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    if not run.dag_node_objs:
        return None

    try:
        from graph_explorer import _force_layout
    except ImportError:
        _force_layout = None

    # Prefer the real production layout for visual consistency with tab_dag.
    node_ids = [n.id for n in run.dag_node_objs]
    edges_for_layout = [(e.source, e.target) for e in run.dag_edge_objs]

    if _force_layout is not None:
        positions = _force_layout(node_ids, edges_for_layout, iterations=60)
    else:
        # Fallback: deterministic circular-ish layout (very rare path)
        positions = {
            nid: (math.cos(2 * math.pi * i / max(1, len(node_ids))),
                  math.sin(2 * math.pi * i / max(1, len(node_ids))))
            for i, nid in enumerate(node_ids)
        }

    cluster_colors = [
        "#0ea5e9", "#a855f7", "#10b981", "#f59e0b", "#ec4899",
        "#7c3aed", "#0d9488", "#e11d48",
    ]

    edge_styles = {
        "forward":  {"color": "rgba(14,165,233,0.45)",  "width": 1.4},
        "backward": {"color": "rgba(168,85,247,0.45)",  "width": 1.4},
        "lateral":  {"color": "rgba(245,158,11,0.55)",  "width": 1.6},
    }

    fig = go.Figure()

    # ── Edges (one trace per kind so the legend explains the coloring) ──────
    for kind, style in edge_styles.items():
        xs: List[Any] = []
        ys: List[Any] = []
        for e in run.dag_edge_objs:
            if e.kind != kind:
                continue
            x0, y0 = positions.get(e.source, (0.0, 0.0))
            x1, y1 = positions.get(e.target, (0.0, 0.0))
            xs.extend([x0, x1, None])
            ys.extend([y0, y1, None])
        if xs:
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(color=style["color"], width=style["width"]),
                hoverinfo="skip", showlegend=True,
                name=f"{kind} edges",
            ))

    # ── Nodes ───────────────────────────────────────────────────────────────
    nx_pts = [positions.get(n.id, (0.0, 0.0))[0] for n in run.dag_node_objs]
    ny_pts = [positions.get(n.id, (0.0, 0.0))[1] for n in run.dag_node_objs]
    sizes = [10 + 4 * math.log1p(n.citations) for n in run.dag_node_objs]
    colors = [cluster_colors[n.cluster % len(cluster_colors)]
              for n in run.dag_node_objs]
    line_widths = [3 if n.is_seed else (2 if n.is_frontier else 1)
                    for n in run.dag_node_objs]
    line_colors = ["#fbbf24" if n.is_seed else
                    ("#f43f5e" if n.is_frontier else "rgba(255,255,255,0.6)")
                    for n in run.dag_node_objs]

    hovers = [
        f"<b>{n.title}</b><br>"
        f"ID: {n.id} · Year: {n.year} · Cluster: {n.cluster}<br>"
        f"Citations: {n.citations:,}<br>"
        f"{'⭐ Seed paper' if n.is_seed else ''}"
        f"{'  ' if n.is_seed and n.is_frontier else ''}"
        f"{'🚩 Frontier (expansion target)' if n.is_frontier else ''}"
        for n in run.dag_node_objs
    ]

    fig.add_trace(go.Scatter(
        x=nx_pts, y=ny_pts, mode="markers",
        marker=dict(
            size=sizes, color=colors,
            line=dict(width=line_widths, color=line_colors),
            opacity=0.92,
        ),
        text=hovers,
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
        name="papers",
    ))

    fig.update_layout(
        height=520,
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="rgba(248,250,252,1)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, bgcolor="rgba(255,255,255,0.7)",
        ),
    )
    return fig


def archive_heatmap_figure(run: FakeRun):
    """7×3 grid showing which cells are filled and their quality."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    z = [[None] * 3 for _ in range(7)]
    text = [[""] * 3 for _ in range(7)]
    for (mi, ni), idea in run.archive.items():
        q = idea.blended_quality if idea.blended_quality is not None else idea.quality
        z[mi][ni] = q
        text[mi][ni] = f"{q:.2f}"

    fig = go.Figure(data=go.Heatmap(
        z=z, text=text, texttemplate="%{text}",
        x=[n.title() for n in NOVELTY_LEVELS],
        y=[m.replace("_", " ").title() for m in METHODOLOGY_TYPES],
        colorscale=[[0, "#fef2f2"], [0.5, "#fde68a"], [1, "#10b981"]],
        zmin=0, zmax=1,
        showscale=True,
        colorbar=dict(title="Quality", thickness=14, len=0.7),
        hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>Quality: %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        height=380,
        margin=dict(l=140, r=20, t=20, b=20),
        xaxis=dict(side="top", title=""),
        yaxis=dict(autorange="reversed", title=""),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def funnel_figure(run: FakeRun):
    """Show how many ideas survive each gate."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    total = len(run.ideas)
    deduped = sum(1 for i in run.ideas if "duplicate" not in i.rejected_reason)
    probe_passed = sum(1 for i in run.ideas
                        if not i.rejected_reason or "replaced" in i.rejected_reason
                        or "already holds" in i.rejected_reason)
    archived = sum(1 for i in run.ideas if i.archived)

    fig = go.Figure(go.Funnel(
        y=["Generated", "Survived de-dup", "Passed probes", "Archived"],
        x=[total, deduped, probe_passed, archived],
        marker=dict(color=["#0ea5e9", "#a855f7", "#f59e0b", "#10b981"]),
        textinfo="value+percent initial",
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit panel
# ─────────────────────────────────────────────────────────────────────────────

def render_pipeline_simulator(st_module) -> None:
    """Full Streamlit panel: architecture diagram, run controls, walkthrough."""
    st = st_module

    st.markdown("### 🎬 Pipeline Simulator")
    st.caption(
        "Educational walkthrough — shows how IdeaGraph turns a topic into a "
        "QD-archived set of research ideas. Runs deterministically with no "
        "LLM calls (instant). Hover any stage in the architecture for details."
    )

    # ── Architecture overview ────────────────────────────────────────────────
    arch = architecture_figure()
    if arch is not None:
        st.plotly_chart(arch, use_container_width=True,
                        key="pipesim_arch_chart")

    with st.expander("📋 Stage-by-stage reference", expanded=False):
        for i, s in enumerate(PIPELINE_STAGES, 1):
            st.markdown(
                f"**{i}. {s['icon']} {s['name']}** — "
                f"`{s['agent']}`  \n"
                f"{s['purpose']}  \n"
                f"<small style='color:#64748b'>"
                f"<b>In:</b> {s['inputs']} &nbsp; · &nbsp; "
                f"<b>Out:</b> {s['outputs']}</small>",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Demo run controls ────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
    topic = c1.text_input(
        "Topic", value="graph neural networks for drug discovery",
        key="pipesim_topic",
    )
    seed = c2.number_input("Seed", value=42, min_value=0, max_value=999_999,
                            step=1, key="pipesim_seed")
    n_ideas = c3.number_input("# ideas", value=16, min_value=4, max_value=40,
                                step=2, key="pipesim_n")
    revision_on = c4.toggle("Exec revision",
                              value=True, key="pipesim_rev",
                              help="Closes the probe → archive feedback loop "
                                    "with Bayesian-blended tiny experiments.")

    if st.button("▶ Simulate a run", type="primary",
                  use_container_width=True, key="pipesim_run"):
        st.session_state["_pipesim_run"] = fake_pipeline_run(
            topic, seed=int(seed), n_ideas=int(n_ideas),
            enable_revision=bool(revision_on),
        )

    run: Optional[FakeRun] = st.session_state.get("_pipesim_run")
    if run is None:
        st.info("Click **▶ Simulate a run** to see the pipeline produce a "
                 "fresh archive in real time.")
        return

    # ── Summary metrics ──────────────────────────────────────────────────────
    archived = sum(1 for i in run.ideas if i.archived)
    rejected = len(run.ideas) - archived
    revised = sum(1 for i in run.ideas if i.archived
                   and i.execution_signal is not None)

    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Topic", run.topic[:24] + ("…" if len(run.topic) > 24 else ""))
    s2.metric("DAG", f"{run.dag_nodes} nodes",
                f"{run.dag_edges} edges")
    s3.metric("Archived", archived, f"of {len(run.ideas)} generated")
    s4.metric("Coverage", f"{run.coverage*100:.0f}%",
                f"{len(run.archive)}/21 cells")
    s5.metric("Mean q", f"{run.mean_quality:.2f}",
                f"{revised} revised" if revised else None)

    st.markdown("---")

    # ── Knowledge DAG visualisation ─────────────────────────────────────────
    # The first stage of the pipeline (KnowledgeArchitect) produces this DAG.
    # Showing it as a graph — not just a count — makes the simulator concrete:
    # admins can SEE clusters, frontier nodes, and edge types.
    st.markdown("**🧠 Knowledge DAG** — the literature graph stage 1 builds")
    dag_fig = dag_figure(run)
    if dag_fig is not None:
        st.plotly_chart(dag_fig, use_container_width=True,
                        key="pipesim_dag_chart")

        if run.dag_node_objs:
            n_seeds = sum(1 for n in run.dag_node_objs if n.is_seed)
            n_frontier = sum(1 for n in run.dag_node_objs if n.is_frontier)
            n_clusters = len({n.cluster for n in run.dag_node_objs})
            n_fwd = sum(1 for e in run.dag_edge_objs if e.kind == "forward")
            n_bwd = sum(1 for e in run.dag_edge_objs if e.kind == "backward")
            n_lat = sum(1 for e in run.dag_edge_objs if e.kind == "lateral")
            st.caption(
                f"**{n_clusters} clusters** · "
                f"⭐ {n_seeds} seed papers · "
                f"🚩 {n_frontier} frontier nodes (highest-citation expansion targets) · "
                f"edges: {n_fwd} forward · {n_bwd} backward · {n_lat} lateral. "
                "Hover any node for paper details. Nodes sized by log(citations); "
                "color = cluster; gold ring = seed; rose ring = frontier."
            )
    else:
        st.caption("DAG figure unavailable (Plotly missing).")

    st.markdown("---")

    # ── Funnel + Heatmap ─────────────────────────────────────────────────────
    fcol, hcol = st.columns([1, 1.2])
    with fcol:
        st.markdown("**Survival funnel**")
        funnel = funnel_figure(run)
        if funnel is not None:
            st.plotly_chart(funnel, use_container_width=True,
                              key="pipesim_funnel")
    with hcol:
        st.markdown("**QD archive (methodology × novelty)**")
        heat = archive_heatmap_figure(run)
        if heat is not None:
            st.plotly_chart(heat, use_container_width=True,
                              key="pipesim_heatmap")

    # ── Per-idea ledger with reasons ────────────────────────────────────────
    st.markdown("---")
    st.markdown("**Idea ledger** — every candidate, what happened, and why.")

    # Sort: archived first by quality desc, then rejected
    archived_sorted = sorted(
        [i for i in run.ideas if i.archived],
        key=lambda x: (x.blended_quality if x.blended_quality is not None
                       else x.quality),
        reverse=True,
    )
    rejected_sorted = sorted(
        [i for i in run.ideas if not i.archived],
        key=lambda x: x.iteration,
    )

    for idea in archived_sorted + rejected_sorted:
        if idea.archived:
            badge = "🟢 archived"
            color = "#10b981"
        elif "duplicate" in idea.rejected_reason:
            badge = "🟡 dedup"
            color = "#f59e0b"
        elif "core-4" in idea.rejected_reason:
            badge = "🔴 probe-fail"
            color = "#ef4444"
        else:
            badge = "⚪ replaced"
            color = "#64748b"

        q = idea.blended_quality if idea.blended_quality is not None else idea.quality
        rev_extra = ""
        if idea.execution_signal is not None and idea.archived:
            delta = (idea.blended_quality - idea.quality
                      if idea.blended_quality is not None else 0)
            arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
            rev_extra = (
                f"  ·  exec={idea.execution_signal:.2f} "
                f"trust={idea.execution_trust*100:.0f}% "
                f"{arrow}{abs(delta):.2f}"
            )

        st.markdown(
            f"<div style='border-left:3px solid {color};padding:6px 12px;"
            f"margin:4px 0;background:#f8fafc;border-radius:4px'>"
            f"<span style='font-size:11px;color:{color};font-weight:700;"
            f"text-transform:uppercase;letter-spacing:0.04em'>{badge}</span> "
            f"<b style='color:#0c4a6e'>{idea.title}</b>"
            f"<br><span style='font-size:11px;color:#64748b'>"
            f"{idea.method_type.replace('_',' ')} × {idea.novelty} · "
            f"q={q:.2f}{rev_extra}"
            f"{(' · ' + idea.rejected_reason) if idea.rejected_reason else ''}"
            f"</span></div>",
            unsafe_allow_html=True,
        )
