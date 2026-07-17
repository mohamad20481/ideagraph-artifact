"""
idea_sorting.py — publication-friendly sort and group operations for Idea dicts.

The Ideas tab originally had 5 sort options (Quality ↓/↑, Novelty,
Methodology, Strategy). When preparing the archive for a paper, a
talk, or a thesis appendix you typically want richer ordering than
that — things like Pareto-efficient subsets, diversity-interleaved
playlists, top-per-cell stratified samples, or lineage-grouped trees.

This module is **pure-Python, dependency-free, side-effect-free**. It
operates on plain idea dicts so it works whether the source is
session_state, the DB, a shared link, or a JSON dump.

Public API:
    SORT_MODES                                     → dict[mode_id, label/description]
    GROUP_MODES                                    → dict[mode_id, label]
    DIRECTIONAL_MODES                              → set of mode_ids that respect descending
    sort_ideas(ideas, mode, descending=True)       → new sorted list
    group_ideas(ideas, group_by)                   → list[(section_label, ideas[])]
    pareto_front(ideas)                            → ideas split into front + dominated
    diversity_interleave(ideas)                    → MMR order (max-min Jaccard distance)
"""
from __future__ import annotations

import random
import re
from typing import Any, Callable, Dict, List, Optional, Tuple


# ── Helpers ─────────────────────────────────────────────────────────────────

_NOVELTY_RANK = {"substantial": 3, "moderate": 2, "incremental": 1}

_TOKEN_RE = re.compile(r"[A-Za-z]{3,}")


def _q(i: Dict[str, Any]) -> float:
    """Quality score, defaulted to 0."""
    try:
        return float(i.get("quality_score") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _n_rank(i: Dict[str, Any]) -> int:
    return _NOVELTY_RANK.get((i.get("novelty_level") or "").lower(), 0)


def _tokens(text: str) -> set:
    return set(t.lower() for t in _TOKEN_RE.findall(text or ""))


def _idea_tokens(i: Dict[str, Any]) -> set:
    """Title + method tokens — proxy for semantic content."""
    return _tokens(
        f"{i.get('title','')} {i.get('method','')}",
    )


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _critic_originality(i: Dict[str, Any]) -> float:
    """Pull the adversarial critic's originality score from execution_meta,
    if attack_and_revise() ran on this idea."""
    meta = i.get("execution_meta") or {}
    if not isinstance(meta, dict):
        return 0.0
    crit = meta.get("novelty_critique") or {}
    if not isinstance(crit, dict):
        return 0.0
    try:
        return float(crit.get("originality_score") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _corpus_novelty(i: Dict[str, Any]) -> float:
    """Pull the corpus-anchored novelty score (mode Q), if it was scored."""
    meta = i.get("execution_meta") or {}
    if not isinstance(meta, dict):
        return 0.0
    cn = meta.get("corpus_novelty") or {}
    if not isinstance(cn, dict):
        return 0.0
    try:
        return float(cn.get("score") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _composite(i: Dict[str, Any]) -> float:
    """Quality × normalized novelty rank."""
    q = _q(i)
    n = _n_rank(i)
    return q * (n / 3.0 if n else 0.5)


def _generation(i: Dict[str, Any]) -> int:
    try:
        return int(i.get("generation") or 0)
    except (TypeError, ValueError):
        return 0


def _smart_blend(i: Dict[str, Any]) -> float:
    """Weighted blend of all available novelty + quality signals.

    Each signal contributes only if it has a non-zero value, and the
    weights re-normalize across whichever signals are present — so
    ideas missing the optional probes (critic, corpus) aren't unfairly
    penalized.
    """
    signals = [
        (_q(i),                 0.35, True),                    # quality
        (_n_rank(i) / 3.0,      0.20, _n_rank(i) > 0),          # novelty level
        (_critic_originality(i), 0.20, _critic_originality(i) > 0),
        (_corpus_novelty(i),    0.25, _corpus_novelty(i) > 0),
    ]
    active = [(v, w) for v, w, present in signals if present]
    if not active:
        return 0.0
    total_w = sum(w for _, w in active)
    return sum(v * w for v, w in active) / total_w


def _qd_grid_position(i: Dict[str, Any]) -> Tuple[int, int]:
    """The (methodology_index, novelty_index) cell coordinate from the
    QD archive grid. Returns (-1, -1) for ideas without both fields so
    they sort to the start in ascending order, end in descending.
    """
    try:
        from models.idea import METHODOLOGY_TYPES, NOVELTY_LEVELS
        m = i.get("methodology_type")
        n = i.get("novelty_level")
        m_idx = METHODOLOGY_TYPES.index(m) if m in METHODOLOGY_TYPES else -1
        n_idx = NOVELTY_LEVELS.index(n) if n in NOVELTY_LEVELS else -1
        return (m_idx, n_idx)
    except Exception:
        return (-1, -1)


def _has_lineage(i: Dict[str, Any], title_set: set) -> bool:
    """True iff this idea is a parent or a child within the archive."""
    return bool(
        (i.get("parent_title") and i.get("parent_title") in title_set)
    )


def _title_length(i: Dict[str, Any]) -> int:
    """Character count of the title — useful for paper-layout planning."""
    return len(i.get("title") or "")


def _cross_pollination(idea: Dict[str, Any],
                          all_ideas: List[Dict[str, Any]]) -> int:
    """Centrality in the parent-child graph: count children + ancestors
    within the archive. Surfaces 'hub' ideas that anchor refinement
    chains."""
    title = idea.get("title", "")
    if not title:
        return 0
    titles = {i.get("title", "") for i in all_ideas if i.get("title")}
    # Children: how many other ideas claim this idea as parent.
    n_children = sum(
        1 for o in all_ideas
        if o is not idea and o.get("parent_title") == title
    )
    # Has a parent in the archive (contributes 1 to centrality).
    has_parent = 1 if (
        idea.get("parent_title") and idea.get("parent_title") in titles
    ) else 0
    return n_children + has_parent


def _surprise_score(idea: Dict[str, Any],
                       all_ideas: List[Dict[str, Any]]) -> float:
    """Absolute distance from the median quality of ideas in the same
    methodology × novelty cell. High surprise = anomaly within its
    peer group."""
    meth = idea.get("methodology_type") or ""
    nov = idea.get("novelty_level") or ""
    if not (meth and nov):
        return 0.0
    peers = [
        _q(o) for o in all_ideas
        if o.get("methodology_type") == meth
        and o.get("novelty_level") == nov
    ]
    if len(peers) < 2:
        return 0.0
    sorted_peers = sorted(peers)
    n = len(sorted_peers)
    if n % 2 == 1:
        median = sorted_peers[n // 2]
    else:
        median = (sorted_peers[n // 2 - 1] + sorted_peers[n // 2]) / 2.0
    return abs(_q(idea) - median)


def _probe_stability(i: Dict[str, Any]) -> float:
    """Inverse standard deviation across probe scores — higher = more
    'consistent' (the probes agree on quality). 0 for ideas without
    multi-probe data."""
    probes = i.get("probe_scores") or {}
    if not isinstance(probes, dict):
        return 0.0
    vals = []
    for v in probes.values():
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            continue
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    std = var ** 0.5
    # Map low std → high score; cap at 1.0.
    return max(0.0, 1.0 - min(std, 1.0)) * mean


# ── Pareto front (Quality × Novelty) ───────────────────────────────────────

def pareto_front(ideas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return ideas reordered so Pareto-efficient (non-dominated by any
    other idea on Quality × Novelty) come first, sorted internally by
    quality desc. Dominated ideas follow, also by quality desc.

    Domination: A dominates B iff Q(A) >= Q(B) AND N(A) >= N(B) AND
    (Q(A) > Q(B) OR N(A) > N(B)).
    """
    if not ideas:
        return []
    front: List[Dict[str, Any]] = []
    for cand in ideas:
        qc, nc = _q(cand), _n_rank(cand)
        dominated = False
        for other in ideas:
            if other is cand:
                continue
            qo, no = _q(other), _n_rank(other)
            if qo >= qc and no >= nc and (qo > qc or no > nc):
                dominated = True
                break
        if not dominated:
            front.append(cand)
    front.sort(key=_q, reverse=True)
    rest = [i for i in ideas if i not in front]
    rest.sort(key=_q, reverse=True)
    return front + rest


def pareto_front_only(ideas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Same as pareto_front but DROPS the dominated tail — for charts /
    appendices where you only want the efficient frontier."""
    full = pareto_front(ideas)
    if not ideas:
        return []
    front_size = 0
    for cand in full:
        qc, nc = _q(cand), _n_rank(cand)
        dominated = any(
            (other is not cand) and (_q(other) >= qc and _n_rank(other) >= nc)
            and (_q(other) > qc or _n_rank(other) > nc)
            for other in ideas
        )
        if dominated:
            break
        front_size += 1
    return full[:front_size]


# ── Diversity interleave (MMR — max-min Jaccard distance) ──────────────────

def diversity_interleave(ideas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Reorder so consecutive ideas maximize diversity.

    First idea = highest quality. Each subsequent idea = the remaining
    idea farthest (highest min-Jaccard-distance) from the set of
    already-picked ideas. Tie-break by quality.

    O(N²) — fine for archives up to ~200 ideas. Above that, prefer
    pareto_front + Top-per-X to thin first.
    """
    if not ideas:
        return []
    pool = list(ideas)
    pool.sort(key=_q, reverse=True)
    out = [pool.pop(0)]
    picked_tokens = [_idea_tokens(out[0])]
    while pool:
        best = None
        best_dist = -1.0
        best_q = -1.0
        for cand in pool:
            ct = _idea_tokens(cand)
            min_sim = min((_jaccard(ct, pt) for pt in picked_tokens),
                            default=0.0)
            dist = 1.0 - min_sim
            cq = _q(cand)
            if (dist > best_dist) or (dist == best_dist and cq > best_q):
                best = cand
                best_dist = dist
                best_q = cq
        pool.remove(best)
        out.append(best)
        picked_tokens.append(_idea_tokens(best))
    return out


# ── Strategy round-robin ───────────────────────────────────────────────────

def strategy_round_robin(ideas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group by source_strategy code, sort each group by quality desc,
    then interleave round-robin across groups. Surfaces strategy
    breadth instead of letting one strategy dominate the head."""
    if not ideas:
        return []
    by_strat: Dict[str, List[Dict[str, Any]]] = {}
    for i in ideas:
        s = (i.get("source_strategy") or "?")
        by_strat.setdefault(s, []).append(i)
    for s in by_strat:
        by_strat[s].sort(key=_q, reverse=True)
    keys = sorted(by_strat.keys())
    out: List[Dict[str, Any]] = []
    while any(by_strat[k] for k in keys):
        for k in keys:
            if by_strat[k]:
                out.append(by_strat[k].pop(0))
    return out


# ── Top-per-X stratification ───────────────────────────────────────────────

def top_per_methodology(ideas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep the highest-quality idea from each methodology_type only."""
    best: Dict[str, Dict[str, Any]] = {}
    for i in ideas:
        m = i.get("methodology_type") or "(unspecified)"
        if m not in best or _q(best[m]) < _q(i):
            best[m] = i
    out = list(best.values())
    out.sort(key=_q, reverse=True)
    return out


def top_per_strategy(ideas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep the highest-quality idea per source_strategy only."""
    best: Dict[str, Dict[str, Any]] = {}
    for i in ideas:
        s = (i.get("source_strategy") or "?")
        if s not in best or _q(best[s]) < _q(i):
            best[s] = i
    out = list(best.values())
    out.sort(key=_q, reverse=True)
    return out


# ── Lineage-grouped (parents → children) ──────────────────────────────────

def refinement_chains_only(ideas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only ideas that participate in a parent-child chain — either
    they have a parent in the archive, or another idea in the archive
    names them as parent. Then walk the chains depth-first."""
    if not ideas:
        return []
    titles = {i.get("title", "") for i in ideas if i.get("title")}
    parents_referenced = {
        i.get("parent_title") for i in ideas
        if i.get("parent_title") and i.get("parent_title") in titles
    }
    chained = [
        i for i in ideas
        if (i.get("parent_title") and i.get("parent_title") in titles)
        or i.get("title", "") in parents_referenced
    ]
    return lineage_grouped(chained)


def lineage_grouped(ideas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Walk the parent_title → children tree. Roots first (sorted by
    quality desc), each followed by its descendants depth-first. Ideas
    that point at a non-existent parent are treated as roots."""
    if not ideas:
        return []
    by_title: Dict[str, Dict[str, Any]] = {}
    for i in ideas:
        t = i.get("title")
        if t and t not in by_title:
            by_title[t] = i
    children: Dict[str, List[Dict[str, Any]]] = {}
    for i in ideas:
        parent = i.get("parent_title")
        if parent and parent in by_title:
            children.setdefault(parent, []).append(i)
    roots = [
        i for i in ideas
        if not i.get("parent_title") or i.get("parent_title") not in by_title
    ]
    roots.sort(key=_q, reverse=True)
    out: List[Dict[str, Any]] = []
    visited = set()

    def walk(node: Dict[str, Any]) -> None:
        node_id = id(node)
        if node_id in visited:
            return
        visited.add(node_id)
        out.append(node)
        kids = children.get(node.get("title", ""), [])
        kids = sorted(kids, key=_q, reverse=True)
        for child in kids:
            walk(child)

    for r in roots:
        walk(r)
    # Catch any ideas not visited (cycle / parent-of-self / orphan).
    for i in ideas:
        if id(i) not in visited:
            out.append(i)
    return out


# ── Sort dispatcher ─────────────────────────────────────────────────────────

# Mode key → (display label, default direction, description, requires_meta)
SORT_MODES: Dict[str, Dict[str, Any]] = {
    "quality": {
        "label": "📊 Quality score",
        "default_desc": True,
        "directional": True,
        "description": "Probe quality score (the default).",
    },
    "novelty_level": {
        "label": "🌟 Novelty level",
        "default_desc": True,
        "directional": True,
        "description": "substantial → moderate → incremental.",
    },
    "composite": {
        "label": "⭐ Composite (Quality × Novelty)",
        "default_desc": True,
        "directional": True,
        "description": "Quality scaled by novelty rank — balances both.",
    },
    "originality_critic": {
        "label": "🎯 Originality (critic score)",
        "default_desc": True,
        "directional": True,
        "description": (
            "Adversarial critic's originality score. Available only "
            "for ideas processed by attack_and_revise()."
        ),
    },
    "corpus_novelty": {
        "label": "🛰️ Corpus novelty score",
        "default_desc": True,
        "directional": True,
        "description": (
            "Distance-to-nearest-corpus-entry from the corpus-anchored "
            "mode (Q). Available only for ideas scored that way."
        ),
    },
    "pareto": {
        "label": "⚖️ Pareto front (Quality × Novelty)",
        "default_desc": True,
        "directional": False,
        "description": (
            "Non-dominated ideas first (efficient frontier), "
            "dominated tail after."
        ),
    },
    "pareto_only": {
        "label": "⚖️ Pareto front ONLY (drops dominated)",
        "default_desc": True,
        "directional": False,
        "description": (
            "Just the efficient frontier — for paper figures / "
            "appendices."
        ),
    },
    "diversity": {
        "label": "🎲 Diversity-interleaved (MMR)",
        "default_desc": True,
        "directional": False,
        "description": (
            "First idea = top quality; each next maximizes Jaccard "
            "distance from already-picked ideas."
        ),
    },
    "strategy_rr": {
        "label": "🔀 Strategy round-robin",
        "default_desc": True,
        "directional": False,
        "description": (
            "Interleave by source_strategy — one A, one B, one C, "
            "back to A…"
        ),
    },
    "top_per_method": {
        "label": "🏆 Top per methodology",
        "default_desc": True,
        "directional": False,
        "description": (
            "One idea per methodology_type — the highest-quality one."
        ),
    },
    "top_per_strategy": {
        "label": "🏅 Top per strategy",
        "default_desc": True,
        "directional": False,
        "description": (
            "One idea per source_strategy code — the highest-quality one."
        ),
    },
    "lineage": {
        "label": "🌳 Lineage-grouped (parents → children)",
        "default_desc": True,
        "directional": False,
        "description": (
            "Walk the parent → child tree depth-first; show "
            "refinement chains together."
        ),
    },
    "generation": {
        "label": "🧬 Generation depth",
        "default_desc": True,
        "directional": True,
        "description": (
            "Most-refined first (generation=N → … → 0). Reverse for "
            "originals-first."
        ),
    },
    "methodology": {
        "label": "📐 Methodology (alphabetical)",
        "default_desc": False,
        "directional": True,
        "description": "Group ideas by methodology_type.",
    },
    "strategy": {
        "label": "🔤 Strategy code (A–Z)",
        "default_desc": False,
        "directional": True,
        "description": "Sort by single-letter source_strategy code.",
    },
    "title": {
        "label": "📛 Title (A–Z)",
        "default_desc": False,
        "directional": True,
        "description": "Alphabetical by title — for appendix tables.",
    },
    "recent": {
        "label": "🆕 Recently added",
        "default_desc": True,
        "directional": True,
        "description": (
            "Preserves the archive's insertion order; newest first "
            "by default."
        ),
    },
    "random": {
        "label": "🎰 Random (seeded)",
        "default_desc": False,
        "directional": False,
        "description": (
            "Deterministic shuffle with seed=42. Useful for "
            "blind-review prep."
        ),
    },
    "smart_blend": {
        "label": "🧠 Smart blend (Q + N + critic + corpus)",
        "default_desc": True,
        "directional": True,
        "description": (
            "Weighted blend of quality, novelty rank, originality "
            "critic, and corpus novelty — auto-renormalizes across "
            "whichever signals are present per-idea."
        ),
    },
    "qd_grid": {
        "label": "🗺️ QD grid coordinate (methodology × novelty)",
        "default_desc": False,
        "directional": True,
        "description": (
            "Formal QD archive order: walk methodology rows × novelty "
            "columns. Mirrors the paper's grid figure exactly."
        ),
    },
    "refinement_chains": {
        "label": "🔗 Refinement chains only (no orphans)",
        "default_desc": True,
        "directional": False,
        "description": (
            "Drop ideas with no parent and no children in the archive; "
            "show only those that participate in a refinement chain."
        ),
    },
    "probe_stability": {
        "label": "📐 Probe stability (most consistent first)",
        "default_desc": True,
        "directional": True,
        "description": (
            "Inverse stddev across probe_scores × mean — favors ideas "
            "where all probes agree on quality. 0 for ideas without "
            "multi-probe scoring."
        ),
    },
    "title_length": {
        "label": "📏 Title length",
        "default_desc": False,
        "directional": True,
        "description": (
            "Sort by title character count — useful when laying out "
            "ideas in a paper table or appendix."
        ),
    },
    "cross_pollination": {
        "label": "🕸️ Cross-pollination (hub ideas)",
        "default_desc": True,
        "directional": True,
        "description": (
            "Centrality in the parent-child graph: count of children "
            "in the archive + 1 if has a parent. Surfaces hub ideas "
            "that anchor refinement chains."
        ),
    },
    "surprise": {
        "label": "💥 Surprise (anomaly vs peer cell)",
        "default_desc": True,
        "directional": True,
        "description": (
            "Absolute distance from the median quality of ideas in the "
            "same (methodology × novelty) cell. Surfaces outliers — "
            "ideas that are unusually good or bad within their peer group."
        ),
    },
    "bayesian_surprise": {
        "label": "🔮 Bayesian Surprise (epistemic shift × plausibility)",
        "default_desc": True,
        "directional": True,
        "description": (
            "Information-theoretic novelty: how much the LLM's belief "
            "about the field shifts when conditioned on the hypothesis, "
            "multiplied by an LLM-rated plausibility. High score = "
            "moves the worldview AND is believable. Replaces 'novelty "
            "bias' that rewards exotic-sounding-but-implausible ideas. "
            "Requires running 'Compute Bayesian Surprise' first — "
            "ideas without the meta field rank 0."
        ),
    },
    "epistemic_shift": {
        "label": "🌊 Epistemic Shift (raw surprise, no plausibility weight)",
        "default_desc": True,
        "directional": True,
        "description": (
            "Same prior/posterior KL surrogate as 🔮 Bayesian Surprise "
            "but WITHOUT the plausibility multiplier. Use this when "
            "you want to surface paradigm-breakers regardless of "
            "feasibility (e.g. for blue-sky brainstorming sessions)."
        ),
    },
    "debate_fitness": {
        "label": "⚔️ Debate Fitness (σ of proposer−critic margin)",
        "default_desc": True,
        "directional": True,
        "description": (
            "Adversarial-game fitness from Extension 1: T rounds of "
            "Proposer-vs-Critic dialogue, fitness = σ(τ · margin). "
            "Range [0, 1]; 0.5 = tied debate. Surfaces ideas that "
            "survive direct attack on assumptions / missing baselines / "
            "scalability. Run 'Compute Debate Fitness' first; unscored "
            "ideas default to 0.5."
        ),
    },
    "debate_margin": {
        "label": "🏹 Debate Margin (raw Σ proposer − Σ critic)",
        "default_desc": True,
        "directional": True,
        "description": (
            "Same debate game but ranks by the unbounded raw margin "
            "instead of the σ-compressed fitness. Use when you want "
            "to see HOW MUCH the proposer dominated, not just whether."
        ),
    },
}

# Sorts whose direction toggle is meaningful (others are categorical /
# non-monotonic and ignore the toggle).
DIRECTIONAL_MODES = {
    k for k, v in SORT_MODES.items() if v.get("directional", False)
}


def _key_fn_for(mode: str) -> Optional[Callable[[Dict[str, Any]], Any]]:
    if mode == "quality":
        return _q
    if mode == "novelty_level":
        return _n_rank
    if mode == "composite":
        return _composite
    if mode == "originality_critic":
        return _critic_originality
    if mode == "corpus_novelty":
        return _corpus_novelty
    if mode == "generation":
        return _generation
    if mode == "methodology":
        return lambda i: (i.get("methodology_type") or "")
    if mode == "strategy":
        return lambda i: (i.get("source_strategy") or "")
    if mode == "title":
        return lambda i: (i.get("title") or "").lower()
    if mode == "smart_blend":
        return _smart_blend
    if mode == "qd_grid":
        return _qd_grid_position
    if mode == "probe_stability":
        return _probe_stability
    if mode == "title_length":
        return _title_length
    if mode == "bayesian_surprise":
        # Read cached score from execution_meta.bayesian_surprise. If
        # surprise hasn't been computed for the idea yet, returns 0.0 —
        # the UI flow expects you to click "Compute Bayesian Surprise"
        # before sorting by it.
        try:
            from bayesian_surprise import bayesian_score_key
            return bayesian_score_key
        except Exception:
            return lambda i: 0.0
    if mode == "epistemic_shift":
        try:
            from bayesian_surprise import surprise_key
            return surprise_key
        except Exception:
            return lambda i: 0.0
    if mode == "debate_fitness":
        try:
            from debate_fitness import debate_fitness_key
            return debate_fitness_key
        except Exception:
            return lambda i: 0.5
    if mode == "debate_margin":
        try:
            from debate_fitness import debate_margin_key
            return debate_margin_key
        except Exception:
            return lambda i: 0.0
    # `cross_pollination` and `surprise` need access to the full list to
    # compute centrality / cell-median — handled in the main dispatcher,
    # not via a per-idea key function.
    return None


def sort_ideas(
    ideas: List[Dict[str, Any]],
    mode: str,
    descending: bool = True,
) -> List[Dict[str, Any]]:
    """Return a new list of ideas sorted by `mode`.

    For directional modes (quality, title, generation, etc.) the
    `descending` argument flips the order. For non-directional modes
    (pareto, diversity, lineage, …) the argument is ignored — the
    algorithm has a single natural ordering.
    """
    if not ideas:
        return []
    if mode not in SORT_MODES:
        raise ValueError(
            f"unknown sort mode {mode!r}; must be one of {sorted(SORT_MODES)}"
        )

    # Algorithmic modes
    if mode == "pareto":
        return pareto_front(ideas)
    if mode == "pareto_only":
        return pareto_front_only(ideas)
    if mode == "diversity":
        return diversity_interleave(ideas)
    if mode == "strategy_rr":
        return strategy_round_robin(ideas)
    if mode == "top_per_method":
        return top_per_methodology(ideas)
    if mode == "top_per_strategy":
        return top_per_strategy(ideas)
    if mode == "lineage":
        return lineage_grouped(ideas)
    if mode == "refinement_chains":
        return refinement_chains_only(ideas)
    if mode == "cross_pollination":
        # Score each idea against the full list, then sort.
        scored = [(i, _cross_pollination(i, ideas)) for i in ideas]
        scored.sort(key=lambda x: x[1], reverse=bool(descending))
        return [i for i, _ in scored]
    if mode == "surprise":
        scored = [(i, _surprise_score(i, ideas)) for i in ideas]
        scored.sort(key=lambda x: x[1], reverse=bool(descending))
        return [i for i, _ in scored]
    if mode == "random":
        rng = random.Random(42)
        out = list(ideas)
        rng.shuffle(out)
        return out
    if mode == "recent":
        # Insertion order; reverse if descending (= newest first).
        return list(reversed(ideas)) if descending else list(ideas)

    # Key-function modes
    key_fn = _key_fn_for(mode)
    if key_fn is None:
        return list(ideas)
    return sorted(ideas, key=key_fn, reverse=bool(descending))


# ── Grouping for sectioned displays ─────────────────────────────────────────

GROUP_MODES: Dict[str, str] = {
    "none":          "— No grouping —",
    "methodology":   "📐 By methodology",
    "strategy":      "🔤 By source strategy",
    "novelty_level": "🌟 By novelty level",
    "generation":    "🧬 By generation depth",
    "quality_band":  "📊 By quality band (≥0.7 / 0.4–0.7 / <0.4)",
}


def _quality_band(i: Dict[str, Any]) -> str:
    q = _q(i)
    if q >= 0.7:
        return "🟢 High (q ≥ 0.7)"
    if q >= 0.4:
        return "🟡 Mid (0.4 ≤ q < 0.7)"
    return "🔴 Low (q < 0.4)"


def _group_key(i: Dict[str, Any], group_by: str) -> str:
    if group_by == "methodology":
        m = i.get("methodology_type") or "(unspecified)"
        return m.replace("_", " ").title()
    if group_by == "strategy":
        return f"Strategy {i.get('source_strategy') or '?'}"
    if group_by == "novelty_level":
        return (i.get("novelty_level") or "(unspecified)").title()
    if group_by == "generation":
        g = _generation(i)
        return f"Generation {g}"
    if group_by == "quality_band":
        return _quality_band(i)
    return ""


def group_ideas(
    ideas: List[Dict[str, Any]],
    group_by: str,
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """Bucket `ideas` into (section_label, ideas_in_section) tuples.

    Sections are emitted in the order their first idea appears in
    `ideas` — so the caller controls section order by sorting first.
    `group_by='none'` returns a single ("", ideas) tuple.
    """
    if group_by not in GROUP_MODES:
        raise ValueError(
            f"unknown group_by mode {group_by!r}; "
            f"must be one of {sorted(GROUP_MODES)}"
        )
    if group_by == "none" or not ideas:
        return [("", list(ideas))]
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    order: List[str] = []
    for i in ideas:
        k = _group_key(i, group_by)
        if k not in buckets:
            buckets[k] = []
            order.append(k)
        buckets[k].append(i)
    return [(k, buckets[k]) for k in order]
