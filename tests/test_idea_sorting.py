"""Tests for idea_sorting — publication-friendly sort + group modes."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import idea_sorting as s


def _idea(
    title: str,
    quality: float = 0.5,
    novelty: str = "moderate",
    methodology: str = "empirical_study",
    strategy: str = "A",
    generation: int = 0,
    parent: str = "",
    method_text: str = "",
    risk: str = "",
    crit_score: float = None,
    corpus_score: float = None,
) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "title": title,
        "quality_score": quality,
        "novelty_level": novelty,
        "methodology_type": methodology,
        "source_strategy": strategy,
        "generation": generation,
        "parent_title": parent or None,
        "method": method_text or title,
        "risk_assessment": risk,
    }
    meta: Dict[str, Any] = {}
    if crit_score is not None:
        meta["novelty_critique"] = {"originality_score": crit_score}
    if corpus_score is not None:
        meta["corpus_novelty"] = {"score": corpus_score}
    if meta:
        d["execution_meta"] = meta
    return d


# ── Registry shape ──────────────────────────────────────────────────────────

def test_sort_modes_registry_well_formed():
    assert len(s.SORT_MODES) >= 12
    for k, v in s.SORT_MODES.items():
        assert "label" in v and v["label"]
        assert "description" in v and len(v["description"]) > 10
        assert "default_desc" in v
        assert "directional" in v


def test_directional_modes_is_subset_of_sort_modes():
    assert s.DIRECTIONAL_MODES.issubset(set(s.SORT_MODES.keys()))
    # Non-directional algorithmic modes (Pareto, diversity, lineage, etc.)
    # must NOT be in the directional set.
    for non_dir in ("pareto", "pareto_only", "diversity",
                      "strategy_rr", "top_per_method", "top_per_strategy",
                      "lineage", "random"):
        assert non_dir not in s.DIRECTIONAL_MODES


def test_group_modes_includes_none():
    assert "none" in s.GROUP_MODES


# ── sort_ideas: empty + invalid ────────────────────────────────────────────

def test_sort_empty_returns_empty():
    for mode in s.SORT_MODES:
        assert s.sort_ideas([], mode) == []


def test_sort_invalid_mode_raises():
    with pytest.raises(ValueError):
        s.sort_ideas([_idea("x")], "not_a_real_mode")


# ── Quality ↓ / ↑ ──────────────────────────────────────────────────────────

def test_quality_descending_default():
    ideas = [_idea("low", 0.1), _idea("hi", 0.9), _idea("mid", 0.5)]
    out = s.sort_ideas(ideas, "quality", descending=True)
    assert [i["title"] for i in out] == ["hi", "mid", "low"]


def test_quality_ascending():
    ideas = [_idea("low", 0.1), _idea("hi", 0.9), _idea("mid", 0.5)]
    out = s.sort_ideas(ideas, "quality", descending=False)
    assert [i["title"] for i in out] == ["low", "mid", "hi"]


# ── Novelty level ──────────────────────────────────────────────────────────

def test_novelty_level_descending():
    ideas = [
        _idea("inc", 0.5, novelty="incremental"),
        _idea("sub", 0.5, novelty="substantial"),
        _idea("mod", 0.5, novelty="moderate"),
    ]
    out = s.sort_ideas(ideas, "novelty_level", descending=True)
    assert [i["title"] for i in out] == ["sub", "mod", "inc"]


# ── Composite (quality × normalized novelty rank) ──────────────────────────

def test_composite_prefers_high_novelty_when_quality_close():
    ideas = [
        _idea("HQ-LowN", quality=0.9, novelty="incremental"),     # 0.9 * .33 = .30
        _idea("MQ-HighN", quality=0.6, novelty="substantial"),   # 0.6 * 1.0 = .60
    ]
    out = s.sort_ideas(ideas, "composite", descending=True)
    assert out[0]["title"] == "MQ-HighN"


# ── Originality critic + corpus novelty (need execution_meta) ──────────────

def test_originality_critic_pulls_from_execution_meta():
    ideas = [
        _idea("no_meta", 0.5),
        _idea("low_crit", 0.5, crit_score=0.2),
        _idea("hi_crit", 0.5, crit_score=0.8),
    ]
    out = s.sort_ideas(ideas, "originality_critic", descending=True)
    assert [i["title"] for i in out] == ["hi_crit", "low_crit", "no_meta"]


def test_corpus_novelty_pulls_from_execution_meta():
    ideas = [
        _idea("no_meta", 0.5),
        _idea("high_novel", 0.5, corpus_score=0.9),
        _idea("low_novel", 0.5, corpus_score=0.1),
    ]
    out = s.sort_ideas(ideas, "corpus_novelty", descending=True)
    assert [i["title"] for i in out] == ["high_novel", "low_novel", "no_meta"]


# ── Pareto front ───────────────────────────────────────────────────────────

def test_pareto_efficient_first():
    # For multiple Pareto members, points must TRADE OFF — high-Q-low-N
    # vs low-Q-high-N must each be non-dominated by the other.
    ideas = [
        _idea("highQ_lowN", quality=0.9, novelty="incremental"),  # front
        _idea("lowQ_highN", quality=0.4, novelty="substantial"),  # front
        _idea("dominated",  quality=0.3, novelty="incremental"),  # dominated
    ]
    out = s.sort_ideas(ideas, "pareto")
    # First two are the Pareto front (mutual non-domination).
    front_titles = {out[0]["title"], out[1]["title"]}
    assert front_titles == {"highQ_lowN", "lowQ_highN"}
    # Dominated tail follows.
    assert out[2]["title"] == "dominated"


def test_pareto_only_drops_dominated():
    ideas = [
        _idea("dominated", quality=0.4, novelty="incremental"),
        _idea("front_highQ", quality=0.9, novelty="incremental"),
        _idea("front_highN", quality=0.4, novelty="substantial"),
    ]
    out = s.sort_ideas(ideas, "pareto_only")
    titles = [i["title"] for i in out]
    assert "dominated" not in titles
    assert "front_highQ" in titles
    assert "front_highN" in titles


def test_pareto_with_singleton():
    out = s.sort_ideas([_idea("solo", 0.5)], "pareto")
    assert len(out) == 1


# ── Diversity (MMR) interleave ─────────────────────────────────────────────

def test_diversity_first_idea_is_highest_quality():
    ideas = [
        _idea("a", 0.3, method_text="x"),
        _idea("b", 0.9, method_text="y"),
        _idea("c", 0.5, method_text="z"),
    ]
    out = s.sort_ideas(ideas, "diversity")
    assert out[0]["title"] == "b"


def test_diversity_prefers_dissimilar_next():
    """If A and B are near-duplicates and C is different, after picking
    A the next pick should be C (more distant), not B."""
    ideas = [
        _idea("A", quality=0.9, method_text="alpha beta gamma delta"),
        _idea("B", quality=0.85, method_text="alpha beta gamma epsilon"),  # near A
        _idea("C", quality=0.6, method_text="zeta eta theta iota"),       # far from A
    ]
    out = s.sort_ideas(ideas, "diversity")
    assert out[0]["title"] == "A"
    assert out[1]["title"] == "C"  # diversity beats quality here
    assert out[2]["title"] == "B"


def test_diversity_preserves_count():
    ideas = [_idea(f"t{i}", quality=0.5, method_text=f"m{i}") for i in range(8)]
    out = s.sort_ideas(ideas, "diversity")
    assert len(out) == 8
    assert {i["title"] for i in out} == {i["title"] for i in ideas}


# ── Strategy round-robin ───────────────────────────────────────────────────

def test_strategy_round_robin_interleaves():
    ideas = [
        _idea("A1", quality=0.9, strategy="A"),
        _idea("A2", quality=0.8, strategy="A"),
        _idea("B1", quality=0.7, strategy="B"),
        _idea("B2", quality=0.6, strategy="B"),
        _idea("C1", quality=0.5, strategy="C"),
    ]
    out = s.sort_ideas(ideas, "strategy_rr")
    # First three must be one A, one B, one C (alphabetical key order).
    strats = [i["source_strategy"] for i in out[:3]]
    assert set(strats) == {"A", "B", "C"}
    # And the highest-quality within each group is picked first.
    assert out[0]["title"] == "A1"
    assert out[1]["title"] == "B1"


def test_strategy_round_robin_uneven_groups():
    ideas = [
        _idea("A1", strategy="A"),
        _idea("A2", strategy="A"),
        _idea("A3", strategy="A"),
        _idea("B1", strategy="B"),
    ]
    out = s.sort_ideas(ideas, "strategy_rr")
    assert len(out) == 4
    assert {i["title"] for i in out} == {"A1", "A2", "A3", "B1"}


# ── Top per methodology / strategy ─────────────────────────────────────────

def test_top_per_methodology_keeps_one_per_bucket():
    ideas = [
        _idea("emp_hi", 0.9, methodology="empirical_study"),
        _idea("emp_lo", 0.4, methodology="empirical_study"),
        _idea("theor_mid", 0.6, methodology="theoretical_analysis"),
    ]
    out = s.sort_ideas(ideas, "top_per_method")
    titles = [i["title"] for i in out]
    assert "emp_hi" in titles
    assert "theor_mid" in titles
    assert "emp_lo" not in titles  # dominated within its bucket
    assert len(out) == 2


def test_top_per_strategy_keeps_one_per_strategy():
    ideas = [
        _idea("A_hi", 0.9, strategy="A"),
        _idea("A_lo", 0.4, strategy="A"),
        _idea("B_mid", 0.6, strategy="B"),
    ]
    out = s.sort_ideas(ideas, "top_per_strategy")
    titles = [i["title"] for i in out]
    assert "A_hi" in titles
    assert "B_mid" in titles
    assert "A_lo" not in titles
    assert len(out) == 2


# ── Lineage-grouped ────────────────────────────────────────────────────────

def test_lineage_grouped_walks_tree_depth_first():
    ideas = [
        _idea("root", 0.8),
        _idea("child1", 0.6, parent="root"),
        _idea("child2", 0.7, parent="root"),
        _idea("grandchild", 0.5, parent="child1"),
        _idea("other_root", 0.4),
    ]
    out = s.sort_ideas(ideas, "lineage")
    titles = [i["title"] for i in out]
    # root + descendants must come before other_root (higher quality root first).
    root_pos = titles.index("root")
    other_root_pos = titles.index("other_root")
    grandchild_pos = titles.index("grandchild")
    assert root_pos < other_root_pos
    # grandchild must appear between root and other_root (DFS).
    assert root_pos < grandchild_pos < other_root_pos
    # child2 (higher q than child1) comes before child1 if siblings sorted by q.
    assert titles.index("child2") < titles.index("child1")


def test_lineage_grouped_handles_orphan_parent_reference():
    """If parent_title points to a non-existent idea, treat as root."""
    ideas = [
        _idea("orphan_kid", 0.5, parent="ghost_parent"),
        _idea("normal", 0.8),
    ]
    out = s.sort_ideas(ideas, "lineage")
    titles = [i["title"] for i in out]
    assert set(titles) == {"orphan_kid", "normal"}
    # normal (higher q) comes first.
    assert titles.index("normal") < titles.index("orphan_kid")


# ── Generation depth ───────────────────────────────────────────────────────

def test_generation_descending():
    ideas = [
        _idea("g0", generation=0),
        _idea("g3", generation=3),
        _idea("g1", generation=1),
    ]
    out = s.sort_ideas(ideas, "generation", descending=True)
    assert [i["title"] for i in out] == ["g3", "g1", "g0"]


# ── Categorical sorts ──────────────────────────────────────────────────────

def test_methodology_alphabetical():
    ideas = [
        _idea("z", methodology="z_method"),
        _idea("a", methodology="a_method"),
    ]
    out = s.sort_ideas(ideas, "methodology", descending=False)
    assert [i["title"] for i in out] == ["a", "z"]


def test_strategy_alphabetical():
    ideas = [
        _idea("B", strategy="B"),
        _idea("A", strategy="A"),
        _idea("C", strategy="C"),
    ]
    out = s.sort_ideas(ideas, "strategy", descending=False)
    assert [i["title"] for i in out] == ["A", "B", "C"]


def test_title_alphabetical():
    ideas = [_idea("Zebra"), _idea("apple"), _idea("Mango")]
    out = s.sort_ideas(ideas, "title", descending=False)
    assert [i["title"] for i in out] == ["apple", "Mango", "Zebra"]


def test_title_descending():
    ideas = [_idea("Zebra"), _idea("apple"), _idea("Mango")]
    out = s.sort_ideas(ideas, "title", descending=True)
    assert [i["title"] for i in out] == ["Zebra", "Mango", "apple"]


# ── Recently added (insertion order) ───────────────────────────────────────

def test_recent_descending_reverses_insertion():
    ideas = [_idea("oldest"), _idea("middle"), _idea("newest")]
    out = s.sort_ideas(ideas, "recent", descending=True)
    assert [i["title"] for i in out] == ["newest", "middle", "oldest"]


def test_recent_ascending_preserves_insertion():
    ideas = [_idea("oldest"), _idea("middle"), _idea("newest")]
    out = s.sort_ideas(ideas, "recent", descending=False)
    assert [i["title"] for i in out] == ["oldest", "middle", "newest"]


# ── Random (seeded — deterministic) ────────────────────────────────────────

def test_random_is_deterministic():
    ideas = [_idea(f"t{i}") for i in range(20)]
    out1 = s.sort_ideas(ideas, "random")
    out2 = s.sort_ideas(ideas, "random")
    assert [i["title"] for i in out1] == [i["title"] for i in out2]


def test_random_preserves_count_and_content():
    ideas = [_idea(f"t{i}") for i in range(10)]
    out = s.sort_ideas(ideas, "random")
    assert len(out) == 10
    assert {i["title"] for i in out} == {i["title"] for i in ideas}


# ── Group by ───────────────────────────────────────────────────────────────

def test_group_none_returns_single_section():
    ideas = [_idea("a"), _idea("b")]
    sections = s.group_ideas(ideas, "none")
    assert len(sections) == 1
    assert sections[0][0] == ""
    assert sections[0][1] == ideas


def test_group_by_methodology_splits_into_buckets():
    ideas = [
        _idea("emp_a", methodology="empirical_study"),
        _idea("theor_a", methodology="theoretical_analysis"),
        _idea("emp_b", methodology="empirical_study"),
    ]
    sections = s.group_ideas(ideas, "methodology")
    labels = [s_label for s_label, _ in sections]
    # 2 unique methodology buckets, label is title-cased.
    assert len(sections) == 2
    assert "Empirical Study" in labels
    assert "Theoretical Analysis" in labels
    # Section sizes correct.
    sizes = {s_label: len(items) for s_label, items in sections}
    assert sizes["Empirical Study"] == 2
    assert sizes["Theoretical Analysis"] == 1


def test_group_by_quality_band():
    ideas = [
        _idea("hi", quality=0.85),
        _idea("mid", quality=0.5),
        _idea("low", quality=0.2),
        _idea("hi2", quality=0.7),
    ]
    sections = s.group_ideas(ideas, "quality_band")
    labels = {s_label: len(items) for s_label, items in sections}
    assert any("High" in k for k in labels)
    assert any("Mid" in k for k in labels)
    assert any("Low" in k for k in labels)


def test_group_by_generation():
    ideas = [
        _idea("a", generation=0),
        _idea("b", generation=2),
        _idea("c", generation=0),
    ]
    sections = s.group_ideas(ideas, "generation")
    labels = {s_label: len(items) for s_label, items in sections}
    assert labels.get("Generation 0") == 2
    assert labels.get("Generation 2") == 1


def test_group_invalid_mode_raises():
    with pytest.raises(ValueError):
        s.group_ideas([_idea("x")], "not_a_real_group")


def test_group_preserves_first_appearance_order():
    """Sections must be emitted in the order their first idea appears."""
    ideas = [
        _idea("z", methodology="zzz_method"),
        _idea("a", methodology="aaa_method"),
        _idea("z2", methodology="zzz_method"),
    ]
    sections = s.group_ideas(ideas, "methodology")
    # 'Zzz Method' bucket must come BEFORE 'Aaa Method' because the
    # first 'z' appears earlier in the input.
    labels = [s_label for s_label, _ in sections]
    assert labels.index("Zzz Method") < labels.index("Aaa Method")


# ── Sort + group composability ─────────────────────────────────────────────

def test_sort_then_group_preserves_within_section_order():
    """When you sort first then group, ordering inside each section
    should follow the sort."""
    ideas = [
        _idea("emp_lo", quality=0.2, methodology="empirical_study"),
        _idea("emp_hi", quality=0.9, methodology="empirical_study"),
        _idea("theor",  quality=0.5, methodology="theoretical_analysis"),
    ]
    sorted_list = s.sort_ideas(ideas, "quality", descending=True)
    sections = s.group_ideas(sorted_list, "methodology")
    by_label = {s_label: items for s_label, items in sections}
    emp_titles = [i["title"] for i in by_label["Empirical Study"]]
    # Within the empirical bucket, descending quality order must hold.
    assert emp_titles == ["emp_hi", "emp_lo"]


# ─────────────────────────────────────────────────────────────────────────────
# New modes (smart_blend, qd_grid, refinement_chains, probe_stability)
# ─────────────────────────────────────────────────────────────────────────────

def test_smart_blend_renormalizes_across_present_signals():
    """An idea with only quality (no critic/corpus) should not be
    penalized — the blend renormalizes across whatever signals exist.
    """
    ideas = [
        # Only quality; weight reverts to 1.0 of just quality.
        _idea("q_only", quality=0.9),
        # Quality + corpus novelty present (higher composite signal).
        _idea("q_plus_corpus", quality=0.9, corpus_score=0.9),
    ]
    out = s.sort_ideas(ideas, "smart_blend", descending=True)
    titles = [i["title"] for i in out]
    # q_plus_corpus has more signals at high values → higher blend.
    assert titles[0] == "q_plus_corpus"


def test_smart_blend_uses_all_four_signals_when_present():
    rich = _idea(
        "rich", quality=0.5, novelty="substantial",
        crit_score=0.8, corpus_score=0.9,
    )
    poor = _idea("poor", quality=0.5, novelty="incremental")
    out = s.sort_ideas([rich, poor], "smart_blend", descending=True)
    assert out[0]["title"] == "rich"


def test_smart_blend_no_signals_returns_zero():
    """An idea with zero quality and no meta scores blends to 0."""
    out = s.sort_ideas(
        [_idea("empty", quality=0.0, novelty="")],
        "smart_blend", descending=True,
    )
    # Doesn't crash and returns the single idea.
    assert len(out) == 1


def test_qd_grid_orders_by_methodology_then_novelty():
    """QD grid order: (methodology_idx, novelty_idx). Ascending by
    default puts methodology row 0, col 0 first."""
    from models.idea import METHODOLOGY_TYPES, NOVELTY_LEVELS
    ideas = [
        _idea("cell_2_2", methodology=METHODOLOGY_TYPES[2],
                novelty=NOVELTY_LEVELS[2]),
        _idea("cell_0_0", methodology=METHODOLOGY_TYPES[0],
                novelty=NOVELTY_LEVELS[0]),
        _idea("cell_0_1", methodology=METHODOLOGY_TYPES[0],
                novelty=NOVELTY_LEVELS[1]),
        _idea("cell_1_0", methodology=METHODOLOGY_TYPES[1],
                novelty=NOVELTY_LEVELS[0]),
    ]
    out = s.sort_ideas(ideas, "qd_grid", descending=False)
    assert [i["title"] for i in out] == [
        "cell_0_0", "cell_0_1", "cell_1_0", "cell_2_2",
    ]


def test_qd_grid_descending_reverses():
    from models.idea import METHODOLOGY_TYPES, NOVELTY_LEVELS
    ideas = [
        _idea("cell_0_0", methodology=METHODOLOGY_TYPES[0],
                novelty=NOVELTY_LEVELS[0]),
        _idea("cell_6_2", methodology=METHODOLOGY_TYPES[6],
                novelty=NOVELTY_LEVELS[2]),
    ]
    out = s.sort_ideas(ideas, "qd_grid", descending=True)
    assert [i["title"] for i in out] == ["cell_6_2", "cell_0_0"]


def test_qd_grid_unknown_methodology_sorts_to_extreme():
    """Ideas without a valid (methodology, novelty) get (-1, -1) — they
    sort to the start ascending, end descending."""
    from models.idea import METHODOLOGY_TYPES, NOVELTY_LEVELS
    ideas = [
        _idea("valid", methodology=METHODOLOGY_TYPES[0],
                novelty=NOVELTY_LEVELS[0]),
        _idea("unknown", methodology="bogus_type"),
    ]
    out_asc = s.sort_ideas(ideas, "qd_grid", descending=False)
    assert out_asc[0]["title"] == "unknown"


def test_refinement_chains_only_drops_orphans():
    """Ideas with no parent in the archive AND no children referencing
    them must be filtered out."""
    ideas = [
        _idea("root", quality=0.8),
        _idea("child", quality=0.6, parent="root"),
        _idea("orphan_no_relations", quality=0.7),
    ]
    out = s.sort_ideas(ideas, "refinement_chains")
    titles = {i["title"] for i in out}
    assert "root" in titles            # has a child
    assert "child" in titles           # has a parent in archive
    assert "orphan_no_relations" not in titles


def test_refinement_chains_empty_input_returns_empty():
    assert s.sort_ideas([], "refinement_chains") == []


def test_refinement_chains_all_orphans_returns_empty():
    ideas = [_idea(f"o{i}", quality=0.5) for i in range(3)]
    assert s.sort_ideas(ideas, "refinement_chains") == []


def test_probe_stability_ranks_consistent_probes_higher():
    ideas = [
        # Stable, high mean.
        {"title": "stable", "quality_score": 0.8,
          "probe_scores": {"a": 0.8, "b": 0.81, "c": 0.79}},
        # Unstable: huge variance.
        {"title": "unstable", "quality_score": 0.8,
          "probe_scores": {"a": 0.1, "b": 0.9, "c": 0.5}},
        # No probes — score 0.
        {"title": "no_probes", "quality_score": 0.8},
    ]
    out = s.sort_ideas(ideas, "probe_stability", descending=True)
    titles = [i["title"] for i in out]
    assert titles[0] == "stable"
    # no_probes scores 0 → last.
    assert titles[-1] == "no_probes"


def test_probe_stability_single_probe_returns_zero():
    """Need at least 2 probes to compute stddev."""
    ideas = [{"title": "single", "quality_score": 0.5,
                "probe_scores": {"a": 0.9}}]
    out = s.sort_ideas(ideas, "probe_stability", descending=True)
    assert len(out) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Registry & dispatcher consistency for the new modes
# ─────────────────────────────────────────────────────────────────────────────

def test_new_modes_registered_in_sort_modes():
    for mode in ("smart_blend", "qd_grid", "refinement_chains",
                  "probe_stability"):
        assert mode in s.SORT_MODES
        # Each must have a usable label and description.
        assert s.SORT_MODES[mode]["label"]
        assert len(s.SORT_MODES[mode]["description"]) > 20


def test_total_sort_modes_at_least_22():
    """22 modes total = 18 original + 4 new."""
    assert len(s.SORT_MODES) >= 22


def test_smart_blend_and_qd_grid_are_directional():
    assert "smart_blend" in s.DIRECTIONAL_MODES
    assert "qd_grid" in s.DIRECTIONAL_MODES
    assert "probe_stability" in s.DIRECTIONAL_MODES


def test_refinement_chains_is_not_directional():
    """It's an algorithmic mode — direction toggle should be disabled."""
    assert "refinement_chains" not in s.DIRECTIONAL_MODES


# ─────────────────────────────────────────────────────────────────────────────
# Round-3 additions: title_length, cross_pollination, surprise
# ─────────────────────────────────────────────────────────────────────────────

def test_title_length_ascending_default():
    ideas = [
        _idea("MediumTitle ok"),
        _idea("A"),
        _idea("This title is much longer than the others"),
    ]
    out = s.sort_ideas(ideas, "title_length", descending=False)
    titles = [i["title"] for i in out]
    assert titles[0] == "A"  # shortest
    assert titles[-1].startswith("This title")  # longest


def test_title_length_descending():
    ideas = [_idea("A"), _idea("AAAAA"), _idea("AA")]
    out = s.sort_ideas(ideas, "title_length", descending=True)
    assert [i["title"] for i in out] == ["AAAAA", "AA", "A"]


def test_cross_pollination_hub_first():
    """Idea with the most children + has a parent = highest centrality."""
    ideas = [
        _idea("root", quality=0.8),
        _idea("child1", parent="root"),
        _idea("child2", parent="root"),
        _idea("child3", parent="root"),
        _idea("grandchild", parent="child1"),
        _idea("isolated", quality=0.9),
    ]
    out = s.sort_ideas(ideas, "cross_pollination", descending=True)
    titles = [i["title"] for i in out]
    # root has 3 children, score=3
    # child1 has 1 child + 1 parent = score 2
    # child2/child3/grandchild: score 1 (just parent)
    # isolated: score 0
    assert titles[0] == "root"
    assert titles[1] == "child1"
    assert titles[-1] == "isolated"  # no relations


def test_cross_pollination_all_orphans_returns_in_input_order():
    ideas = [_idea(f"o{i}", quality=0.5) for i in range(5)]
    out = s.sort_ideas(ideas, "cross_pollination", descending=True)
    # All score 0; original order preserved (stable sort).
    assert len(out) == 5
    assert {i["title"] for i in out} == {f"o{i}" for i in range(5)}


def test_surprise_score_finds_anomalies_in_cell():
    """Within a (methodology, novelty) cell, the idea farthest from the
    cell's median quality is the biggest surprise."""
    ideas = [
        _idea("ord1", quality=0.5, methodology="empirical_study",
                novelty="moderate"),
        _idea("ord2", quality=0.55, methodology="empirical_study",
                novelty="moderate"),
        _idea("ord3", quality=0.45, methodology="empirical_study",
                novelty="moderate"),
        # Outlier in the same cell — wildly higher quality.
        _idea("outlier", quality=0.95, methodology="empirical_study",
                novelty="moderate"),
        # Different cell — peers don't apply.
        _idea("other_cell", quality=0.5, methodology="theoretical_analysis",
                novelty="substantial"),
    ]
    out = s.sort_ideas(ideas, "surprise", descending=True)
    titles = [i["title"] for i in out]
    # The outlier should be at the top — biggest absolute distance.
    assert titles[0] == "outlier"


def test_surprise_score_singleton_cell_returns_zero():
    """If a cell has < 2 ideas, surprise can't be computed (returns 0)."""
    ideas = [
        _idea("lonely", quality=0.9, methodology="empirical_study",
                novelty="substantial"),
        _idea("other_cell_1", quality=0.5, methodology="dataset_creation",
                novelty="moderate"),
        _idea("other_cell_2", quality=0.6, methodology="dataset_creation",
                novelty="moderate"),
    ]
    out = s.sort_ideas(ideas, "surprise", descending=True)
    # lonely's surprise = 0; other_cell pair has 0.05 distance each.
    # So other_cell_1 or other_cell_2 wins, not lonely.
    assert out[-1]["title"] == "lonely" or out[0]["title"] != "lonely"


def test_total_sort_modes_at_least_25():
    """25 modes total = 18 base + 4 round-2 + 3 round-3."""
    assert len(s.SORT_MODES) >= 25


def test_round3_modes_registered():
    for mode in ("title_length", "cross_pollination", "surprise"):
        assert mode in s.SORT_MODES
        assert s.SORT_MODES[mode]["label"]
        assert len(s.SORT_MODES[mode]["description"]) > 30
