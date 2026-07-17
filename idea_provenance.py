"""
idea_provenance.py — provenance tracing for ideas in IdeaGraph.

Internally IdeaGraph runs three ideation strategies (frontier extension,
cross-cluster bridging, gap-filling) plus regeneration and execution-aware
revision. The final output normally just shows the title + method; this
module unifies what's recoverable about *where each idea came from* and
exposes:

  - A `ProvenanceRecord` dataclass capturing strategy, seed papers, target
    cell, parent lineage, revision history, and a quality journey.
  - `extract_provenance(idea, ...)` that backfills from existing `Idea`
    fields (works on legacy ideas that never opted into provenance) and
    consumes any explicit provenance the pipeline attaches.
  - Plotly-based provenance graph + an HTML card for inline UI.
  - A tiny **behavioral-study harness** for the within-subjects experiment
    the proposed CHI/FAccT paper would run: present ideas with vs without
    provenance, collect trust ratings, compute trust-calibration vs
    ground-truth quality.

Public API:
    ProvenanceRecord
    STRATEGY_LABELS                                 → Dict[code, label]
    extract_provenance(idea, dag_summary=None)      → ProvenanceRecord
    build_provenance_figure(record)                  → plotly Figure
    render_provenance_card_html(record)              → HTML fragment
    behavioral_assignment(ideas, seed=None)          → Dict[idea_idx, "with"|"without"]
    summarize_behavioral_study(ratings)              → study summary dict
"""
from __future__ import annotations

import hashlib
import html as _html
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Strategy labels — keep human-readable next to the single-letter codes
# the pipeline emits internally.
# ─────────────────────────────────────────────────────────────────────────────

STRATEGY_LABELS: Dict[str, Dict[str, str]] = {
    "A": {
        "label": "Frontier Extension",
        "icon": "🌟",
        "description": (
            "Picks a *frontier paper* — one with no known successors in the "
            "DAG — and proposes the next research step from there. Tends to "
            "produce ideas grounded in a specific recent paper."
        ),
    },
    "B": {
        "label": "Cross-Cluster Bridging",
        "icon": "🌉",
        "description": (
            "Combines two distinct clusters in the Knowledge DAG (e.g. one "
            "on graph methods, one on chemistry) and proposes the bridge. "
            "Tends to produce interdisciplinary ideas."
        ),
    },
    "C": {
        "label": "Gap-Filling",
        "icon": "🧩",
        "description": (
            "Targets an empty or under-explored cell in the QD archive "
            "(specific methodology × novelty combination). Tends to "
            "produce ideas in directions the population hasn't tried yet."
        ),
    },
    "R": {
        "label": "Regeneration (parent-derived)",
        "icon": "🔄",
        "description": (
            "Derived from an existing idea via the Regenerate tab "
            "(refine / extend / pivot / contrast / cross-domain / mutate). "
            "The lineage_note records the specific transformation."
        ),
    },
    "?": {
        "label": "Unknown",
        "icon": "❓",
        "description": "Strategy was not recorded for this idea.",
    },
}


@dataclass
class ProvenanceRecord:
    """Everything we know about where an idea came from."""

    idea_title: str
    source_strategy: str                         # "A" / "B" / "C" / "R" / "?"
    strategy_label: str
    strategy_icon: str
    strategy_description: str

    # Inputs (the "ingredients")
    seed_papers: List[Dict[str, Any]] = field(default_factory=list)
    target_cell: Optional[Tuple[int, int]] = None
    methodology_type: Optional[str] = None
    novelty_level: Optional[str] = None

    # Lineage (when this idea descends from another)
    parent_title: Optional[str] = None
    generation: int = 0
    lineage_note: Optional[str] = None
    regen_mode: Optional[str] = None             # set when source_strategy == "R"

    # Probe & revision history
    probe_scores: Dict[str, float] = field(default_factory=dict)
    revision_history: List[Dict[str, Any]] = field(default_factory=list)
    quality_journey: List[Dict[str, Any]] = field(default_factory=list)

    # Coverage of provenance information (0..1) — how much we actually know
    completeness: float = 0.0
    # Sources used to construct the record (audit trail)
    sources_used: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "idea_title": self.idea_title,
            "source_strategy": self.source_strategy,
            "strategy_label": self.strategy_label,
            "strategy_icon": self.strategy_icon,
            "strategy_description": self.strategy_description,
            "seed_papers": list(self.seed_papers),
            "target_cell": (list(self.target_cell)
                              if self.target_cell else None),
            "methodology_type": self.methodology_type,
            "novelty_level": self.novelty_level,
            "parent_title": self.parent_title,
            "generation": self.generation,
            "lineage_note": self.lineage_note,
            "regen_mode": self.regen_mode,
            "probe_scores": dict(self.probe_scores),
            "revision_history": list(self.revision_history),
            "quality_journey": list(self.quality_journey),
            "completeness": self.completeness,
            "sources_used": list(self.sources_used),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Extraction — backfills from whatever Idea fields are populated
# ─────────────────────────────────────────────────────────────────────────────

def _idea_to_dict(idea: Any) -> Dict[str, Any]:
    if hasattr(idea, "to_dict"):
        return idea.to_dict()
    if isinstance(idea, dict):
        return dict(idea)
    return {}


def _strategy_meta(code: str) -> Dict[str, str]:
    return STRATEGY_LABELS.get((code or "?").upper(),
                                  STRATEGY_LABELS["?"])


def _build_quality_journey(d: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Sequence of (stage_name, score) snapshots — what we know happened."""
    out: List[Dict[str, Any]] = []
    # Stage 1: probe-only quality (pre-revision)
    pq = d.get("probe_quality")
    if pq is not None:
        out.append({"stage": "probe", "value": float(pq),
                     "note": "Probe-only quality (pre-execution-revision)"})
    # Stage 2: execution-aware blended (after Bayesian update with tiny exp)
    es = d.get("execution_signal")
    et = d.get("execution_trust")
    delta = d.get("execution_delta")
    if es is not None:
        note = f"Execution signal {float(es):.2f}"
        if et is not None:
            note += f" · trust {float(et)*100:.0f}%"
        out.append({"stage": "execution_signal", "value": float(es),
                     "note": note})
    # Stage 3: final quality_score (what the archive ranks by)
    qs = d.get("quality_score")
    if qs is not None:
        note = "Final quality_score (archive ranking)"
        if delta is not None:
            note += f" · Δ {float(delta):+.2f} from probe"
        out.append({"stage": "final", "value": float(qs), "note": note})
    return out


def _build_revision_history(d: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Best-effort reconstruction of the revision events, in chronological
    order: generation → probe → execution_revision → (regeneration only
    appears for regenerated ideas, where it replaces 'generation')."""
    history: List[Dict[str, Any]] = []
    code = (d.get("source_strategy") or "").upper()

    # 1. The birth event — either a regeneration (R) or a normal generation
    if code == "R":
        meta = d.get("execution_meta") or {}
        history.append({
            "stage": "regeneration",
            "agent": "idea_regenerator",
            "summary": (
                f"Mode: {meta.get('regen_mode', '?')} · "
                f"{meta.get('lineage_note', '(no lineage note)')}"
            ),
        })
    elif d.get("title"):
        sm = _strategy_meta(code or "?")
        history.append({
            "stage": "generation",
            "agent": "IdeationAgent",
            "summary": (
                f"{sm['icon']} Strategy {code or '?'} ({sm['label']})"
            ),
        })

    # 2. Probe critique (if probe_scores were populated)
    if d.get("probe_scores"):
        weak = sorted(
            [(k, float(v)) for k, v in d["probe_scores"].items()
             if isinstance(v, (int, float))],
            key=lambda kv: kv[1],
        )[:3]
        if weak:
            summary = ("Scored on 10 dimensions; weakest: "
                       + ", ".join(f"{k} ({v:.2f})" for k, v in weak))
        else:
            summary = "Scored on 10 dimensions."
        history.append({
            "stage": "probe_critique",
            "agent": "ExecutionCritic",
            "summary": summary,
        })

    # 3. Execution-aware revision (if the loop ran)
    if d.get("execution_signal") is not None:
        meta = d.get("execution_meta") or {}
        # sample_size may be a string fallback like '?' — guard before format
        ss_raw = meta.get("sample_size")
        try:
            ss_str = f"{int(ss_raw):,}"
        except (TypeError, ValueError):
            ss_str = str(ss_raw if ss_raw is not None else "?")
        metric_name = meta.get("metric_name", "")
        ci = meta.get("confidence_interval")
        ci_str = (f" · 95% CI [{ci[0]:.2f}, {ci[1]:.2f}]"
                   if ci and len(ci) == 2
                   and isinstance(ci[0], (int, float))
                   and isinstance(ci[1], (int, float)) else "")
        delta = d.get("execution_delta")
        delta_str = (f" · Δ {float(delta):+.3f}"
                       if delta is not None else "")
        history.append({
            "stage": "execution_revision",
            "agent": "execution_revisor",
            "summary": (
                f"Tiny-experiment proxy ({ss_str} samples × "
                f"{meta.get('n_seeds', '?')} seed). "
                f"Metric: {metric_name or '(none)'}{ci_str}{delta_str}"
            ),
        })

    return history


def _completeness(record: ProvenanceRecord) -> float:
    """Fraction of provenance buckets we managed to fill."""
    buckets = [
        record.source_strategy and record.source_strategy != "?",
        bool(record.seed_papers),
        record.target_cell is not None,
        record.methodology_type is not None,
        record.novelty_level is not None,
        bool(record.probe_scores),
        bool(record.revision_history),
        bool(record.quality_journey),
    ]
    return sum(1 for b in buckets if b) / len(buckets)


def extract_provenance(idea: Any,
                         dag_summary: Optional[Dict[str, Any]] = None,
                         ) -> ProvenanceRecord:
    """Extract a `ProvenanceRecord` from an Idea, backfilling fields when
    the pipeline didn't attach explicit provenance.

    `dag_summary` (optional) is the dict returned alongside the run; we
    use it to look up paper titles/years/clusters for any seed paper IDs
    explicitly stored on the idea.
    """
    d = _idea_to_dict(idea)
    sources_used: List[str] = []

    # 1. Strategy
    code = (d.get("source_strategy") or "?").upper()
    meta = _strategy_meta(code)
    if d.get("source_strategy"):
        sources_used.append("idea.source_strategy")

    # 2. Explicit provenance dict (when the pipeline attached one)
    explicit = d.get("provenance") or {}
    if explicit:
        sources_used.append("idea.provenance")

    # 3. Seed papers — explicit beats backfill
    seed_papers: List[Dict[str, Any]] = []
    explicit_seeds = explicit.get("seed_papers") or []
    if explicit_seeds:
        for sp in explicit_seeds[:6]:
            if isinstance(sp, dict):
                seed_papers.append({
                    "id": str(sp.get("id", "")),
                    "title": str(sp.get("title", "(unknown)"))[:120],
                    "year": sp.get("year"),
                    "role": str(sp.get("role", "seed")),
                })
    elif explicit.get("seed_paper_ids") and dag_summary:
        # Backfill from DAG node lookup
        nodes = dag_summary.get("nodes") or []
        by_id = {str(n.get("id")): n for n in nodes if isinstance(n, dict)}
        for sid in explicit["seed_paper_ids"][:6]:
            n = by_id.get(str(sid))
            if n:
                seed_papers.append({
                    "id": str(n.get("id", sid)),
                    "title": str(n.get("title", "(unknown)"))[:120],
                    "year": n.get("year"),
                    "role": "seed",
                })
        sources_used.append("dag_summary lookup")

    # 4. Target cell, methodology, novelty
    methodology_type = d.get("methodology_type")
    novelty_level = d.get("novelty_level")
    target_cell = explicit.get("target_cell")
    if target_cell and isinstance(target_cell, (list, tuple)) and len(target_cell) == 2:
        target_cell = (int(target_cell[0]), int(target_cell[1]))
    elif methodology_type and novelty_level:
        try:
            from models.idea import (
                METHODOLOGY_TYPE_TO_IDX, NOVELTY_LEVEL_TO_IDX,
            )
            target_cell = (
                METHODOLOGY_TYPE_TO_IDX.get(methodology_type, 0),
                NOVELTY_LEVEL_TO_IDX.get(novelty_level, 0),
            )
            sources_used.append("methodology+novelty → cell")
        except Exception:
            target_cell = None

    # 5. Lineage from regeneration / refinement
    parent_title = d.get("parent_title")
    generation = int(d.get("generation") or 0)
    exec_meta = d.get("execution_meta") or {}
    lineage_note = exec_meta.get("lineage_note")
    regen_mode = exec_meta.get("regen_mode")

    # 6. Quality journey + revision history
    quality_journey = _build_quality_journey(d)
    revision_history = _build_revision_history(d)

    record = ProvenanceRecord(
        idea_title=str(d.get("title", "Untitled"))[:200],
        source_strategy=code,
        strategy_label=meta["label"],
        strategy_icon=meta["icon"],
        strategy_description=meta["description"],
        seed_papers=seed_papers,
        target_cell=target_cell,
        methodology_type=methodology_type,
        novelty_level=novelty_level,
        parent_title=parent_title,
        generation=generation,
        lineage_note=lineage_note,
        regen_mode=regen_mode,
        probe_scores={
            k: float(v) for k, v in (d.get("probe_scores") or {}).items()
            if isinstance(v, (int, float))
        },
        revision_history=revision_history,
        quality_journey=quality_journey,
        sources_used=sources_used,
    )
    record.completeness = _completeness(record)
    return record


# ─────────────────────────────────────────────────────────────────────────────
# HTML card — inline display alongside an idea
# ─────────────────────────────────────────────────────────────────────────────

def render_provenance_card_html(record: ProvenanceRecord) -> str:
    """Compact, escaped HTML card showing the provenance in one block."""
    e = _html.escape

    seeds_html = ""
    if record.seed_papers:
        items = "".join(
            f'<li><b>{e(str(s.get("title","?"))[:80])}</b>'
            f' <span style="color:#64748b">({e(str(s.get("year","?")))})</span> '
            f'<span style="color:#64748b">— {e(str(s.get("role","seed")))}</span>'
            f'</li>'
            for s in record.seed_papers[:5]
        )
        seeds_html = (
            f'<div style="margin-top:8px"><b>📄 Seed papers</b>'
            f'<ul style="margin:4px 0 0 18px;font-size:12px">{items}</ul></div>'
        )

    cell_html = ""
    if record.target_cell:
        cell_html = (
            f'<div style="margin-top:6px;font-size:12px;color:#475569">'
            f'<b>🎯 Target cell:</b> '
            f'{e(record.methodology_type or "?").replace("_"," ")} × '
            f'{e(record.novelty_level or "?")}</div>'
        )

    lineage_html = ""
    if record.parent_title:
        regen = (f' (regeneration mode: <code>{e(record.regen_mode)}</code>)'
                  if record.regen_mode else "")
        note = (f'<br><i style="color:#475569">{e(record.lineage_note)}</i>'
                 if record.lineage_note else "")
        lineage_html = (
            f'<div style="margin-top:6px;background:#fef3c7;padding:6px 10px;'
            f'border-radius:6px;font-size:12px;color:#713f12">'
            f'<b>🌳 Derived from:</b> {e(record.parent_title)}{regen}'
            f' · gen {record.generation}{note}</div>'
        )

    history_html = ""
    if record.revision_history:
        items = "".join(
            f'<li><b>{e(h["stage"])}</b> <code>({e(h["agent"])})</code><br>'
            f'<span style="color:#64748b">{e(h["summary"])}</span></li>'
            for h in record.revision_history
        )
        history_html = (
            f'<div style="margin-top:8px"><b>🛠️ Revision history</b>'
            f'<ol style="margin:4px 0 0 18px;font-size:12px">{items}</ol></div>'
        )

    journey_html = ""
    if record.quality_journey:
        rows = "".join(
            f'<div style="display:flex;gap:8px;align-items:center;margin:2px 0">'
            f'<div style="width:130px;font-size:11px;color:#475569">'
            f'{e(j["stage"])}</div>'
            f'<div style="flex:1;background:#e2e8f0;border-radius:4px;height:10px;'
            f'position:relative;overflow:hidden">'
            f'<div style="position:absolute;left:0;top:0;bottom:0;'
            f'width:{j["value"]*100:.0f}%;background:#0ea5e9"></div></div>'
            f'<div style="width:42px;text-align:right;font-size:11px;'
            f'font-variant-numeric:tabular-nums;color:#0c4a6e">'
            f'<b>{j["value"]:.2f}</b></div></div>'
            for j in record.quality_journey
        )
        journey_html = (
            f'<div style="margin-top:8px"><b>📈 Quality journey</b>'
            f'<div style="margin:4px 0 0 0;font-size:11px">{rows}</div></div>'
        )

    completeness_pct = record.completeness * 100
    completeness_color = ("#10b981" if completeness_pct >= 75
                            else "#f59e0b" if completeness_pct >= 40
                            else "#ef4444")

    return (
        f'<div style="background:#f8fafc;border:1px solid #e2e8f0;'
        f'border-left:4px solid #0ea5e9;border-radius:8px;'
        f'padding:12px 16px;margin:8px 0;font-family:system-ui,sans-serif">'
        f'<div style="display:flex;justify-content:space-between;'
        f'align-items:flex-start;gap:12px">'
        f'<div>'
        f'<div style="font-size:11px;color:#0ea5e9;font-weight:700;'
        f'text-transform:uppercase;letter-spacing:0.06em">Provenance</div>'
        f'<div style="font-size:13px;color:#0c4a6e;margin:1px 0;'
        f'font-style:italic">{e(record.idea_title)}</div>'
        f'<div style="font-size:14px;font-weight:700;color:#0c4a6e;margin:2px 0">'
        f'{record.strategy_icon} {e(record.strategy_label)}</div>'
        f'<div style="font-size:12px;color:#475569">'
        f'{e(record.strategy_description)}</div>'
        f'</div>'
        f'<div style="text-align:right">'
        f'<div style="font-size:10px;color:#64748b">Completeness</div>'
        f'<div style="font-size:18px;font-weight:700;color:{completeness_color}">'
        f'{completeness_pct:.0f}%</div></div>'
        f'</div>'
        f'{cell_html}{lineage_html}{seeds_html}{history_html}{journey_html}'
        f'</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Plotly tree of where the idea came from
# ─────────────────────────────────────────────────────────────────────────────

def build_provenance_figure(record: ProvenanceRecord):
    """A simple two-column tree: ingredients (left) → idea (right)."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    nodes_x: List[float] = []
    nodes_y: List[float] = []
    nodes_text: List[str] = []
    nodes_color: List[str] = []
    nodes_size: List[int] = []
    edges_x: List[Any] = []
    edges_y: List[Any] = []

    # Right: the final idea
    idea_x = 0.92
    idea_y = 0.5
    nodes_x.append(idea_x)
    nodes_y.append(idea_y)
    nodes_text.append(f"<b>💡 {record.idea_title[:48]}</b>")
    nodes_color.append("#0ea5e9")
    nodes_size.append(34)

    # Left column: inputs
    ingredients: List[Tuple[str, str, str]] = []  # (label, icon, color)
    if record.source_strategy and record.source_strategy != "?":
        ingredients.append((
            f"Strategy {record.source_strategy}: {record.strategy_label}",
            record.strategy_icon, "#a855f7",
        ))
    if record.parent_title:
        ingredients.append((
            f"Parent: {record.parent_title[:36]}",
            "🌳", "#f59e0b",
        ))
    for sp in record.seed_papers[:4]:
        ingredients.append((
            f"{sp.get('title','?')[:36]} ({sp.get('year','?')})",
            "📄", "#10b981",
        ))
    if record.methodology_type and record.novelty_level:
        ingredients.append((
            f"Cell: {record.methodology_type.replace('_',' ')} × "
            f"{record.novelty_level}",
            "🎯", "#ec4899",
        ))

    if not ingredients:
        ingredients.append(("(no inputs recorded)", "❓", "#94a3b8"))

    n = len(ingredients)
    for i, (label, icon, color) in enumerate(ingredients):
        y = 1.0 - (i + 1) / (n + 1)
        nodes_x.append(0.08)
        nodes_y.append(y)
        nodes_text.append(f"{icon} {label}")
        nodes_color.append(color)
        nodes_size.append(22)
        edges_x.extend([0.08, idea_x, None])
        edges_y.extend([y, idea_y, None])

    fig = go.Figure()
    # Edges
    fig.add_trace(go.Scatter(
        x=edges_x, y=edges_y, mode="lines",
        line=dict(color="rgba(100,116,139,0.45)", width=1.5),
        hoverinfo="skip", showlegend=False,
    ))
    # Nodes
    fig.add_trace(go.Scatter(
        x=nodes_x, y=nodes_y, mode="markers+text",
        marker=dict(size=nodes_size, color=nodes_color,
                      line=dict(width=2, color="white")),
        text=nodes_text,
        textposition=["middle right" if x < 0.5 else "middle left"
                       for x in nodes_x],
        textfont=dict(size=11, color="#0c4a6e"),
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(
        height=max(280, 80 + 60 * n),
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="rgba(248,250,252,1)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[-0.05, 1.20], showgrid=False, zeroline=False,
                     showticklabels=False),
        yaxis=dict(range=[-0.05, 1.05], showgrid=False, zeroline=False,
                     showticklabels=False),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Behavioral-study harness (within-subjects A/B)
# ─────────────────────────────────────────────────────────────────────────────

def behavioral_assignment(n_ideas: int,
                            seed: Optional[int] = None,
                            balanced: bool = True) -> List[str]:
    """Assign each of `n_ideas` to a condition: 'with' or 'without' provenance.

    Within-subjects design. Defaults to a *balanced* assignment (half
    'with', half 'without', shuffled), which is what a real behavioral
    study should use to avoid spurious imbalance from seed luck. Set
    `balanced=False` for the naive independent-Bernoulli flavor.
    """
    rng = random.Random(seed)
    if balanced:
        n_with = n_ideas // 2
        n_without = n_ideas - n_with
        slots = ["with"] * n_with + ["without"] * n_without
        rng.shuffle(slots)
        return slots
    return [rng.choice(["with", "without"]) for _ in range(n_ideas)]


def _pearson(xs: List[float], ys: List[float]) -> float:
    """Simple Pearson correlation; returns 0 if undefined."""
    n = min(len(xs), len(ys))
    if n < 2:
        return 0.0
    mx = sum(xs[:n]) / n
    my = sum(ys[:n]) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    dx = math.sqrt(sum((xs[i] - mx) ** 2 for i in range(n)))
    dy = math.sqrt(sum((ys[i] - my) ** 2 for i in range(n)))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def summarize_behavioral_study(
    ratings: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate within-subjects ratings into a study summary.

    Each rating dict must have: condition ('with'|'without'),
    trust_rating (1..5 int), quality_score (0..1).

    Returns dict with mean trust per condition, Δ, calibration (Pearson r
    of trust × quality_score per condition), and N per condition.
    """
    with_p = [r for r in ratings if r.get("condition") == "with"]
    without_p = [r for r in ratings if r.get("condition") == "without"]

    def _mean_trust(rs):
        if not rs:
            return 0.0
        return sum(float(r.get("trust_rating", 0)) for r in rs) / len(rs)

    def _calibration(rs):
        return _pearson(
            [float(r.get("trust_rating", 0)) for r in rs],
            [float(r.get("quality_score", 0)) for r in rs],
        )

    return {
        "n_total": len(ratings),
        "n_with": len(with_p),
        "n_without": len(without_p),
        "mean_trust_with": _mean_trust(with_p),
        "mean_trust_without": _mean_trust(without_p),
        "trust_delta": _mean_trust(with_p) - _mean_trust(without_p),
        "calibration_with": _calibration(with_p),
        "calibration_without": _calibration(without_p),
        "calibration_delta": _calibration(with_p) - _calibration(without_p),
    }


# Sentinel + helper for attaching provenance from the pipeline (optional API)
def attach_provenance(idea: Any, **fields: Any) -> None:
    """Attach explicit provenance to an Idea instance.

    Any kwargs become entries on the `provenance` dict — this is the
    contract pipeline code uses when it actually KNOWS which strategy
    (and which seed papers) generated the idea. Backfill still works
    for ideas that never call this.
    """
    if not hasattr(idea, "provenance"):
        return
    current = getattr(idea, "provenance") or {}
    if not isinstance(current, dict):
        current = {}
    current.update(fields)
    try:
        idea.provenance = current
    except AttributeError:
        # slots=True with provenance not declared — bail silently
        pass
