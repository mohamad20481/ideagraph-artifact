"""Tests for idea_provenance.py — provenance tracing + behavioral study harness."""
from __future__ import annotations

from typing import Any, Dict

import pytest

from idea_provenance import (
    STRATEGY_LABELS,
    ProvenanceRecord,
    extract_provenance,
    render_provenance_card_html,
    build_provenance_figure,
    behavioral_assignment,
    summarize_behavioral_study,
    attach_provenance,
    _pearson,
)
from models.idea import Idea


# ── Fixtures ─────────────────────────────────────────────────────────────

def _idea(**overrides) -> Idea:
    base = dict(
        title="Diffusion world models",
        motivation="m", method="Use latent diffusion as a world model",
        hypothesis="diffusion world models predict 25% better",
        resources="8 A100", expected_outcome="SOTA on Habitat",
        risk_assessment="compute heavy",
    )
    base.update({k: v for k, v in overrides.items() if k in base})
    idea = Idea(**base)
    for k, v in overrides.items():
        if k not in base and hasattr(idea, k):
            setattr(idea, k, v)
    return idea


def _full_idea() -> Idea:
    i = _idea()
    i.source_strategy = "B"
    i.methodology_type = "empirical_study"
    i.novelty_level = "substantial"
    i.quality_score = 0.85
    i.probe_quality = 0.72
    i.execution_signal = 0.78
    i.execution_trust = 0.34
    i.execution_delta = 0.04
    i.execution_meta = {"metric_name": "AUROC", "sample_size": 1000,
                          "n_seeds": 1, "confidence_interval": [0.71, 0.84]}
    i.probe_scores = {"novelty": 0.9, "specificity": 0.7,
                       "code": 0.8, "dataset": 0.7}
    return i


# ── Strategy catalog ─────────────────────────────────────────────────────

class TestStrategyCatalog:
    def test_all_known_strategies_present(self):
        # The pipeline emits A/B/C; regenerator emits R; "?" is the unknown bucket
        for code in ("A", "B", "C", "R", "?"):
            assert code in STRATEGY_LABELS

    def test_each_strategy_has_label_icon_description(self):
        for code, meta in STRATEGY_LABELS.items():
            assert meta["label"] and meta["icon"] and meta["description"]


# ── Extraction ───────────────────────────────────────────────────────────

class TestExtraction:
    def test_full_idea_yields_full_record(self):
        i = _full_idea()
        attach_provenance(i, seed_papers=[
            {"id": "P001", "title": "LDM", "year": 2022, "role": "frontier"},
        ])
        r = extract_provenance(i)
        assert r.source_strategy == "B"
        assert r.strategy_label == "Cross-Cluster Bridging"
        assert r.target_cell is not None
        assert len(r.seed_papers) == 1
        assert r.completeness > 0.85

    def test_legacy_idea_still_extractable(self):
        # Idea with no explicit provenance, only minimal fields
        legacy = _idea()
        legacy.source_strategy = "A"
        legacy.methodology_type = "theoretical_analysis"
        legacy.novelty_level = "moderate"
        legacy.quality_score = 0.6
        legacy.probe_scores = {"novelty": 0.6}
        r = extract_provenance(legacy)
        assert r.source_strategy == "A"
        assert r.target_cell is not None
        assert r.completeness > 0.40

    def test_unknown_strategy_falls_back(self):
        i = _idea()
        # Don't set source_strategy
        r = extract_provenance(i)
        assert r.source_strategy == "?"
        assert r.strategy_label == "Unknown"

    def test_regeneration_lineage_recovered(self):
        i = _full_idea()
        i.source_strategy = "R"
        i.parent_title = "Parent idea title"
        i.generation = 1
        i.execution_meta = {"regen_mode": "extend",
                             "lineage_note": "Builds on parent."}
        r = extract_provenance(i)
        assert r.parent_title == "Parent idea title"
        assert r.generation == 1
        assert r.regen_mode == "extend"
        assert r.lineage_note == "Builds on parent."

    def test_quality_journey_built_when_revision_present(self):
        r = extract_provenance(_full_idea())
        assert len(r.quality_journey) >= 2
        # Each stage has a value in [0, 1]
        for stage in r.quality_journey:
            assert 0.0 <= stage["value"] <= 1.0
            assert stage["stage"] and stage["note"]

    def test_revision_history_chronological(self):
        # Generation should come before probe before execution_revision
        r = extract_provenance(_full_idea())
        stages = [h["stage"] for h in r.revision_history]
        # Generation must appear first in the chronological order
        assert stages[0] == "generation"
        # And probe_critique must come before execution_revision
        if "probe_critique" in stages and "execution_revision" in stages:
            assert (stages.index("probe_critique")
                     < stages.index("execution_revision"))

    def test_dag_summary_lookup_resolves_ids(self):
        # Pipeline can attach seed_paper_ids; extractor resolves via dag_summary
        i = _idea()
        attach_provenance(i, seed_paper_ids=["P001", "P003"])
        dag = {
            "nodes": [
                {"id": "P001", "title": "Seminal paper", "year": 2018},
                {"id": "P003", "title": "Later work", "year": 2022},
            ],
        }
        r = extract_provenance(i, dag_summary=dag)
        titles = [s["title"] for s in r.seed_papers]
        assert "Seminal paper" in titles
        assert "Later work" in titles


# ── HTML rendering / XSS ─────────────────────────────────────────────────

class TestHTMLRendering:
    def test_renders_for_full_idea(self):
        r = extract_provenance(_full_idea())
        html = render_provenance_card_html(r)
        assert isinstance(html, str)
        assert "Provenance" in html
        assert "Cross-Cluster Bridging" in html

    def test_xss_in_title_escaped(self):
        evil = _idea()
        evil.title = "<script>alert(1)</script>"
        r = extract_provenance(evil)
        html = render_provenance_card_html(r)
        assert "<script>alert" not in html
        assert "&lt;script&gt;" in html

    def test_xss_in_lineage_note_escaped(self):
        i = _idea()
        i.execution_meta = {"lineage_note": "<img src=x onerror=alert(1)>"}
        i.parent_title = "<b>Parent</b>"
        r = extract_provenance(i)
        html = render_provenance_card_html(r)
        assert "<img src=x" not in html
        assert "<b>Parent</b>" not in html  # raw bold tag should not survive

    def test_completeness_color_changes_with_value(self):
        # High completeness → green; low → red
        i_full = _full_idea()
        attach_provenance(i_full, seed_papers=[
            {"id": "P1", "title": "x", "year": 2020, "role": "seed"},
        ])
        html_full = render_provenance_card_html(extract_provenance(i_full))
        html_empty = render_provenance_card_html(
            extract_provenance(_idea())
        )
        assert "10b981" in html_full or "rgb(16,185" in html_full.lower()
        assert "ef4444" in html_empty or "rgb(239,68" in html_empty.lower()


# ── Plotly figure ────────────────────────────────────────────────────────

class TestProvenanceFigure:
    def test_returns_figure(self):
        r = extract_provenance(_full_idea())
        fig = build_provenance_figure(r)
        assert fig is None or hasattr(fig, "to_dict")

    def test_two_traces_edges_and_nodes(self):
        r = extract_provenance(_full_idea())
        fig = build_provenance_figure(r)
        if fig is None:
            return
        # First trace = edges, second = nodes
        assert len(fig.data) == 2
        assert fig.data[0].mode == "lines"
        assert "markers" in fig.data[1].mode

    def test_idea_node_present(self):
        r = extract_provenance(_full_idea())
        fig = build_provenance_figure(r)
        if fig is None:
            return
        # The final-idea node text should contain the title (truncated)
        node_texts = list(fig.data[1].text)
        assert any("Diffusion" in t for t in node_texts)

    def test_no_inputs_still_renders(self):
        # Bare idea with no strategy / seeds / parent
        r = extract_provenance(_idea())
        fig = build_provenance_figure(r)
        if fig is None:
            return
        # Should still produce a figure with at least the idea node
        assert len(fig.data[1].x) >= 2  # idea + "no inputs recorded" placeholder


# ── Behavioral study harness ─────────────────────────────────────────────

class TestBehavioralStudy:
    def test_balanced_assignment_is_balanced(self):
        # 10 ideas → 5 with, 5 without
        a = behavioral_assignment(10, seed=42, balanced=True)
        assert a.count("with") == 5
        assert a.count("without") == 5

    def test_balanced_handles_odd_n(self):
        # 9 ideas → 4 with, 5 without
        a = behavioral_assignment(9, seed=42, balanced=True)
        assert abs(a.count("with") - a.count("without")) == 1

    def test_unbalanced_mode_is_independent_bernoulli(self):
        # Without balance, the count distribution can be skewed
        a = behavioral_assignment(20, seed=42, balanced=False)
        # All entries are valid conditions
        assert all(c in ("with", "without") for c in a)

    def test_deterministic_given_seed(self):
        a1 = behavioral_assignment(8, seed=99, balanced=True)
        a2 = behavioral_assignment(8, seed=99, balanced=True)
        assert a1 == a2

    def test_pearson_corner_cases(self):
        # Constant series → undefined → 0
        assert _pearson([1, 1, 1], [1, 2, 3]) == 0.0
        # Single point → 0
        assert _pearson([1], [2]) == 0.0
        # Perfect positive correlation → 1
        assert _pearson([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)

    def test_summary_with_no_ratings(self):
        s = summarize_behavioral_study([])
        assert s["n_total"] == 0
        assert s["n_with"] == 0
        assert s["mean_trust_with"] == 0.0
        assert s["trust_delta"] == 0.0

    def test_summary_only_with_provenance(self):
        ratings = [
            {"condition": "with", "trust_rating": 5, "quality_score": 0.9},
            {"condition": "with", "trust_rating": 3, "quality_score": 0.5},
        ]
        s = summarize_behavioral_study(ratings)
        assert s["n_with"] == 2
        assert s["n_without"] == 0
        assert s["mean_trust_with"] == 4.0

    def test_summary_calibration_when_perfect(self):
        # Ratings perfectly aligned with quality should give calibration ≈ 1
        ratings = [
            {"condition": "with", "trust_rating": 5, "quality_score": 0.9},
            {"condition": "with", "trust_rating": 4, "quality_score": 0.7},
            {"condition": "with", "trust_rating": 3, "quality_score": 0.5},
            {"condition": "with", "trust_rating": 2, "quality_score": 0.3},
        ]
        s = summarize_behavioral_study(ratings)
        assert s["calibration_with"] > 0.99

    def test_summary_calibration_anti_correlated(self):
        # Ratings inverted: higher trust for worse ideas → negative calibration
        ratings = [
            {"condition": "without", "trust_rating": 5, "quality_score": 0.2},
            {"condition": "without", "trust_rating": 4, "quality_score": 0.4},
            {"condition": "without", "trust_rating": 2, "quality_score": 0.7},
            {"condition": "without", "trust_rating": 1, "quality_score": 0.9},
        ]
        s = summarize_behavioral_study(ratings)
        assert s["calibration_without"] < -0.95


# ── attach_provenance ────────────────────────────────────────────────────

class TestAttachProvenance:
    def test_sets_provenance_dict_on_idea(self):
        i = _idea()
        attach_provenance(i, target_cell=(0, 1),
                           seed_papers=[{"id": "X"}])
        assert i.provenance is not None
        assert i.provenance["target_cell"] == (0, 1)
        assert i.provenance["seed_papers"] == [{"id": "X"}]

    def test_merges_into_existing_provenance(self):
        i = _idea()
        i.provenance = {"existing": "value"}
        attach_provenance(i, new_field="added")
        assert i.provenance["existing"] == "value"
        assert i.provenance["new_field"] == "added"

    def test_handles_corrupted_provenance(self):
        # If something has stuffed a non-dict into provenance, attach
        # should overwrite cleanly rather than crash
        i = _idea()
        i.provenance = "not a dict"
        attach_provenance(i, target_cell=(0, 1))
        assert isinstance(i.provenance, dict)
        assert i.provenance["target_cell"] == (0, 1)


# ── App-wiring smoke check ────────────────────────────────────────────────

class TestAppWiring:
    def test_app_imports_idea_provenance(self):
        with open("app.py", encoding="utf-8") as f:
            src = f.read()
        assert "from idea_provenance import" in src
        assert "tab_provenance" in src
        assert '"Provenance"' in src

    def test_idea_dataclass_has_provenance_field(self):
        i = _idea()
        assert hasattr(i, "provenance")
        d = i.to_dict()
        assert "provenance" in d
