"""Tests for iqd_controls.py (Extension 5 — Interactive Quality-Diversity).

Covers:
  - IQDState dataclass + JSON round-trip
  - cell_for_idea / build_archive (best per cell)
  - Freeze / prune mutations + their interactions
  - Parent selection queue (max 2, dedup, replace-oldest)
  - apply_iqd_generation_filter
  - directed_crossover: happy path, no-client, bad JSON, sanitize
    invalid methodology/novelty, lineage stamping
  - render_iqd_panel smoke
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import iqd_controls as iqd


# ── IQDState ────────────────────────────────────────────────────────────────

def test_iqdstate_defaults_empty():
    s = iqd.IQDState()
    assert s.frozen_cells == set()
    assert s.pruned_cells == set()
    assert s.selected_parents == []


def test_iqdstate_json_roundtrip():
    s = iqd.IQDState()
    s.frozen_cells.add((1, 2))
    s.pruned_cells.add((3, 1))
    s.selected_parents = [5, 7]
    payload = s.to_jsonable()
    rebuilt = iqd.IQDState.from_jsonable(payload)
    assert rebuilt.frozen_cells == {(1, 2)}
    assert rebuilt.pruned_cells == {(3, 1)}
    assert rebuilt.selected_parents == [5, 7]


def test_iqdstate_from_jsonable_tolerates_garbage():
    assert iqd.IQDState.from_jsonable("garbage").frozen_cells == set()
    assert iqd.IQDState.from_jsonable({
        "frozen_cells": [["a", "b"], [1]],  # bad shapes
        "pruned_cells": [(0, 0)],
        "selected_parents": ["x", 3],
    }).pruned_cells == {(0, 0)}


# ── cell_for_idea ──────────────────────────────────────────────────────────

def test_cell_for_idea_known_values():
    idea = {
        "methodology_type": "empirical_study",
        "novelty_level": "moderate",
    }
    assert iqd.cell_for_idea(idea) == (0, 1)


def test_cell_for_idea_unknown_maps_to_off_grid():
    assert iqd.cell_for_idea({"methodology_type": "weird"}) == (-1, -1)
    assert iqd.cell_for_idea({}) == (-1, -1)
    assert iqd.cell_for_idea(None) == (-1, -1)


# ── build_archive ─────────────────────────────────────────────────────────

def test_build_archive_picks_highest_quality_per_cell():
    ideas = [
        {"title": "low", "methodology_type": "system_design",
         "novelty_level": "moderate", "quality_score": 0.3},
        {"title": "high", "methodology_type": "system_design",
         "novelty_level": "moderate", "quality_score": 0.8},
        {"title": "other_cell", "methodology_type": "system_design",
         "novelty_level": "substantial", "quality_score": 0.5},
    ]
    archive = iqd.build_archive(ideas)
    assert archive[(2, 1)]["title"] == "high"
    assert archive[(2, 2)]["title"] == "other_cell"
    assert len(archive) == 2


def test_build_archive_excludes_off_grid_ideas():
    ideas = [
        {"title": "off", "methodology_type": "bogus", "novelty_level": "moderate",
         "quality_score": 0.9},
        {"title": "on", "methodology_type": "system_design",
         "novelty_level": "moderate", "quality_score": 0.5},
    ]
    archive = iqd.build_archive(ideas)
    assert len(archive) == 1
    assert (2, 1) in archive


def test_build_archive_empty_input():
    assert iqd.build_archive([]) == {}
    assert iqd.build_archive(None) == {}


# ── Freeze / prune mutations ───────────────────────────────────────────────

def test_freeze_unfreeze():
    s = iqd.IQDState()
    iqd.freeze_cell(s, (0, 0))
    assert iqd.is_frozen(s, (0, 0))
    iqd.unfreeze_cell(s, (0, 0))
    assert not iqd.is_frozen(s, (0, 0))


def test_prune_unprune():
    s = iqd.IQDState()
    iqd.prune_cell(s, (1, 1))
    assert iqd.is_pruned(s, (1, 1))
    iqd.unprune_cell(s, (1, 1))
    assert not iqd.is_pruned(s, (1, 1))


def test_freezing_unprunes():
    """If a cell was pruned and then frozen, the prune should clear —
    freezing implies the user values that cell."""
    s = iqd.IQDState()
    iqd.prune_cell(s, (0, 0))
    iqd.freeze_cell(s, (0, 0))
    assert iqd.is_frozen(s, (0, 0))
    assert not iqd.is_pruned(s, (0, 0))


def test_pruning_unfreezes():
    s = iqd.IQDState()
    iqd.freeze_cell(s, (0, 0))
    iqd.prune_cell(s, (0, 0))
    assert iqd.is_pruned(s, (0, 0))
    assert not iqd.is_frozen(s, (0, 0))


# ── Parent-selection queue ──────────────────────────────────────────────────

def test_select_parent_appends():
    s = iqd.IQDState()
    iqd.select_parent(s, 3)
    assert s.selected_parents == [3]


def test_select_parent_dedupes():
    s = iqd.IQDState()
    iqd.select_parent(s, 3)
    iqd.select_parent(s, 3)
    assert s.selected_parents == [3]


def test_select_parent_max_two_replaces_oldest():
    s = iqd.IQDState()
    iqd.select_parent(s, 1)
    iqd.select_parent(s, 2)
    iqd.select_parent(s, 3)  # should evict 1, keep [2, 3]
    assert s.selected_parents == [2, 3]


def test_clear_parents():
    s = iqd.IQDState()
    iqd.select_parent(s, 1)
    iqd.select_parent(s, 2)
    iqd.clear_parents(s)
    assert s.selected_parents == []


# ── apply_iqd_generation_filter ────────────────────────────────────────────

def test_filter_removes_pruned_cells():
    s = iqd.IQDState()
    iqd.prune_cell(s, (0, 0))
    iqd.prune_cell(s, (2, 1))
    candidates = [(0, 0), (1, 1), (2, 1), (3, 2)]
    filtered = iqd.apply_iqd_generation_filter(s, candidates)
    assert (0, 0) not in filtered
    assert (2, 1) not in filtered
    assert (1, 1) in filtered
    assert (3, 2) in filtered


def test_filter_no_prunes_is_identity():
    s = iqd.IQDState()
    candidates = [(0, 0), (1, 1)]
    assert iqd.apply_iqd_generation_filter(s, candidates) == candidates


# ── directed_crossover ────────────────────────────────────────────────────

def _mock_llm_returning(json_text):
    cli = MagicMock()
    resp = MagicMock()
    resp.success = True
    resp.text = json_text
    cli.call.return_value = resp
    return cli


_VALID_JSON = """
{
  "title": "Hybrid efficient transformer",
  "motivation": "Bridge sparse routing with theoretical guarantees",
  "method": "Sparse-routed attention with provable bounds",
  "hypothesis": "Theory + sparse routing match dense quality at 1/4 cost",
  "resources": "1× A100, 24h, public benchmarks",
  "expected_outcome": "Match dense at 25% FLOPs",
  "risk_assessment": "Theoretical bound may not be tight",
  "methodology_type": "system_design",
  "novelty_level": "substantial"
}
"""


def test_directed_crossover_happy_path():
    parent_a = {
        "title": "Sparse Routing",
        "motivation": "fewer ops",
        "method": "k=2 routing",
        "hypothesis": "sparse routing works",
        "risk_assessment": "may underfit",
        "methodology_type": "system_design",
        "novelty_level": "moderate",
        "generation": 2,
    }
    parent_b = {
        "title": "Convergence Theory",
        "motivation": "prove things",
        "method": "spectral analysis",
        "hypothesis": "bounds tight",
        "risk_assessment": "assumptions strong",
        "methodology_type": "theoretical_analysis",
        "novelty_level": "substantial",
        "generation": 3,
    }
    cli = _mock_llm_returning(_VALID_JSON)
    child = iqd.directed_crossover(
        parent_a, parent_b,
        constraint="must run on single A100 in <24h",
        llm_client=cli,
    )
    assert child is not None
    assert child["title"] == "Hybrid efficient transformer"
    assert child["source_strategy"] == "I"
    assert child["generation"] == 4   # max(2, 3) + 1
    assert child["methodology_type"] == "system_design"
    assert child["novelty_level"] == "substantial"
    meta = child["execution_meta"]["iqd_crossover"]
    assert meta["parent_a_title"] == "Sparse Routing"
    assert meta["parent_b_title"] == "Convergence Theory"
    assert "single A100" in meta["user_constraint"]
    assert "Sparse Routing" in child["parent_title"]


def test_directed_crossover_no_client_returns_none():
    out = iqd.directed_crossover(
        {"title": "A"}, {"title": "B"},
        constraint="x",
        llm_client=None,
    )
    assert out is None


def test_directed_crossover_bad_json_returns_none():
    cli = _mock_llm_returning("not json at all, just prose")
    out = iqd.directed_crossover(
        {"title": "A", "methodology_type": "system_design",
         "novelty_level": "moderate"},
        {"title": "B", "methodology_type": "system_design",
         "novelty_level": "moderate"},
        constraint="x",
        llm_client=cli,
    )
    assert out is None


def test_directed_crossover_strips_markdown_fences():
    cli = _mock_llm_returning("```json\n" + _VALID_JSON + "\n```")
    child = iqd.directed_crossover(
        {"title": "A", "methodology_type": "system_design",
         "novelty_level": "moderate"},
        {"title": "B", "methodology_type": "system_design",
         "novelty_level": "moderate"},
        constraint="x",
        llm_client=cli,
    )
    assert child is not None
    assert child["title"] == "Hybrid efficient transformer"


def test_directed_crossover_sanitizes_invalid_methodology():
    """If LLM returns an unknown methodology_type, fall back to parent A's."""
    bad_json = _VALID_JSON.replace(
        '"methodology_type": "system_design"',
        '"methodology_type": "completely_made_up"',
    )
    cli = _mock_llm_returning(bad_json)
    parent_a = {"title": "A", "methodology_type": "theoretical_analysis",
                "novelty_level": "moderate"}
    parent_b = {"title": "B", "methodology_type": "system_design",
                "novelty_level": "moderate"}
    child = iqd.directed_crossover(parent_a, parent_b, "x", llm_client=cli)
    assert child["methodology_type"] == "theoretical_analysis"


def test_directed_crossover_sanitizes_invalid_novelty():
    bad_json = _VALID_JSON.replace(
        '"novelty_level": "substantial"',
        '"novelty_level": "alien"',
    )
    cli = _mock_llm_returning(bad_json)
    child = iqd.directed_crossover(
        {"title": "A", "methodology_type": "system_design",
         "novelty_level": "moderate"},
        {"title": "B", "methodology_type": "system_design",
         "novelty_level": "moderate"},
        constraint="x",
        llm_client=cli,
    )
    assert child["novelty_level"] == "moderate"


def test_directed_crossover_call_failure_returns_none():
    cli = MagicMock()
    cli.call.side_effect = RuntimeError("network down")
    out = iqd.directed_crossover(
        {"title": "A"}, {"title": "B"}, "x", llm_client=cli,
    )
    assert out is None


# ── Streamlit render_iqd_panel smoke ───────────────────────────────────────

def _make_st_stub():
    stub = MagicMock()

    def _make_col_mock():
        c = MagicMock()
        c.button = MagicMock(return_value=False)
        c.markdown = MagicMock()
        c.metric = MagicMock()
        return c

    stub.columns.side_effect = lambda spec, **kw: (
        [_make_col_mock() for _ in range(spec)]
        if isinstance(spec, int)
        else [_make_col_mock() for _ in spec]
    )
    for attr in ("expander", "container", "spinner"):
        cm = getattr(stub, attr)
        cm.return_value.__enter__ = MagicMock(return_value=stub)
        cm.return_value.__exit__ = MagicMock(return_value=None)
    stub.button.return_value = False
    stub.text_input.return_value = ""
    stub.session_state = {}
    return stub


def test_render_panel_empty_ideas_shows_info():
    st = _make_st_stub()
    iqd.render_iqd_panel(st, [], archive_id="x")
    assert st.info.called


def test_render_panel_with_ideas_renders_grid():
    st = _make_st_stub()
    ideas = [
        {"title": f"Idea {i}",
         "methodology_type": "system_design",
         "novelty_level": "moderate",
         "quality_score": 0.5 + i * 0.05}
        for i in range(3)
    ]
    iqd.render_iqd_panel(st, ideas, archive_id="x")
    # Should have rendered the "🗺️ Archive map" header.
    md_calls = [
        c.args[0] for c in st.markdown.call_args_list
        if c.args and isinstance(c.args[0], str)
    ]
    assert any("Archive map" in m for m in md_calls)


def test_render_panel_initializes_iqd_state_in_session():
    st = _make_st_stub()
    ideas = [
        {"title": "x", "methodology_type": "system_design",
         "novelty_level": "moderate", "quality_score": 0.5},
    ]
    iqd.render_iqd_panel(st, ideas, archive_id="abc")
    assert "_iqd_state:abc" in st.session_state
    assert isinstance(st.session_state["_iqd_state:abc"], iqd.IQDState)


def test_render_panel_shows_synth_form_only_when_two_parents():
    st = _make_st_stub()
    ideas = [
        {"title": "A", "methodology_type": "system_design",
         "novelty_level": "moderate", "quality_score": 0.5},
        {"title": "B", "methodology_type": "system_design",
         "novelty_level": "substantial", "quality_score": 0.6},
    ]
    # Pre-seed state with 2 parents picked.
    state = iqd.IQDState()
    state.selected_parents = [0, 1]
    st.session_state["_iqd_state:test"] = state
    iqd.render_iqd_panel(st, ideas, archive_id="test")
    md_calls = [
        c.args[0] for c in st.markdown.call_args_list
        if c.args and isinstance(c.args[0], str)
    ]
    assert any("Synthesize child idea" in m for m in md_calls)
