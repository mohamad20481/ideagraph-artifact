"""Tests for the third wave of novelty modules: analogy, failure_mode, extremum."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import analogy_ideation as ana
import failure_mode_ideation as fm
import extremum_ideation as ex
from models.idea import METHODOLOGY_TYPES, NOVELTY_LEVELS


# ── Mock LLM client ─────────────────────────────────────────────────────────

@dataclass
class _MockResp:
    success: bool
    text: str


class _SeqClient:
    """Deterministic LLM stand-in: returns a queue of pre-baked responses."""

    def __init__(self, responses: List[Any]):
        self._queue = list(responses)
        self.call_count = 0
        self.last_system = None
        self.last_user = None

    def call(self, system: str, user: str, **kw) -> _MockResp:
        self.call_count += 1
        self.last_system = system
        self.last_user = user
        if not self._queue:
            return _MockResp(False, "")
        next_item = self._queue.pop(0)
        if isinstance(next_item, _MockResp):
            return next_item
        if isinstance(next_item, dict):
            return _MockResp(True, json.dumps(next_item))
        if isinstance(next_item, str):
            return _MockResp(True, next_item)
        return _MockResp(False, "")


def _valid_idea_payload(extra: dict, strategy: str) -> dict:
    base = {
        "title": "Test Title",
        "motivation": "Why this matters.",
        "method": "Concrete method here.",
        "hypothesis": "Falsifiable claim.",
        "resources": "1 GPU-week.",
        "expected_outcome": "Measurable result.",
        "risk_assessment": "Known risks.",
        "source_strategy": strategy,
        "methodology_type": METHODOLOGY_TYPES[0],
        "novelty_level": NOVELTY_LEVELS[1],
    }
    base.update(extra)
    return base


# ─────────────────────────────────────────────────────────────────────────────
# analogy_ideation
# ─────────────────────────────────────────────────────────────────────────────

def test_analogy_extract_empty_topic_raises():
    with pytest.raises(ValueError):
        ana.extract_analogy("", "biology")


def test_analogy_extract_empty_domain_raises():
    with pytest.raises(ValueError):
        ana.extract_analogy("ML topic", "")


def test_analogy_extract_no_client_returns_none():
    assert ana.extract_analogy("ML topic", "biology", claude_client=None) is None


def test_analogy_extract_happy_path_builds_bridge():
    client = _SeqClient([{
        "source_structure": "immune cell clonal selection",
        "target_counterpart": "neural network weight subnetworks",
        "morphism": "cells ↔ subnetworks; antigen presentation ↔ gradient signal",
        "invariant": "selection pressure preserves task-relevant variants",
        "risk_of_break": "fails when target lacks population diversity",
        "confidence": 0.78,
    }])
    bridge = ana.extract_analogy("neural architecture search", "biology",
                                    claude_client=client)
    assert bridge is not None
    assert bridge.source_domain == "biology"
    assert "immune cell" in bridge.source_structure
    assert 0.0 <= bridge.confidence <= 1.0
    assert bridge.confidence == pytest.approx(0.78)


def test_analogy_extract_missing_required_fields_returns_none():
    client = _SeqClient([{
        "source_structure": "",  # empty → invalid
        "target_counterpart": "x", "morphism": "y",
    }])
    assert ana.extract_analogy("topic", "biology", claude_client=client) is None


def test_analogy_extract_confidence_coercion_clamps_to_unit():
    client = _SeqClient([{
        "source_structure": "s", "target_counterpart": "t", "morphism": "m",
        "confidence": 5.0,  # out of range — clamp to 1.0
    }])
    bridge = ana.extract_analogy("topic", "biology", claude_client=client)
    assert bridge is not None
    assert bridge.confidence == 1.0


def test_analogy_generate_from_bridge_builds_idea_with_M_strategy():
    bridge = ana.AnalogyBridge(
        source_domain="biology", source_structure="s",
        target_counterpart="t", morphism="m", invariant="i",
    )
    client = _SeqClient([
        _valid_idea_payload({"transplanted_mechanism": "clonal selection"}, "M"),
    ])
    idea = ana.generate_from_analogy("topic", bridge, claude_client=client)
    assert idea is not None
    assert idea.source_strategy == "M"
    assert idea.execution_meta is not None
    assert idea.execution_meta["regen_mode"] == "analogy"
    assert idea.execution_meta["analogy_bridge"]["source_domain"] == "biology"
    assert "clonal" in idea.execution_meta["transplanted_mechanism"]


def test_analogy_generate_invalid_method_type_coerced_to_none():
    bridge = ana.AnalogyBridge(source_domain="d", source_structure="s",
                                 target_counterpart="t", morphism="m")
    payload = _valid_idea_payload({}, "M")
    payload["methodology_type"] = "not-a-real-type"
    payload["novelty_level"] = "not-a-real-level"
    client = _SeqClient([payload])
    idea = ana.generate_from_analogy("topic", bridge, claude_client=client)
    assert idea is not None
    assert idea.methodology_type is None
    assert idea.novelty_level is None


def test_analogy_batch_uses_n_domains_and_calls_2n_times():
    """Each domain → 1 extract call + 1 generate call = 2 LLM calls per idea."""
    bridge_resp = {
        "source_structure": "s", "target_counterpart": "t",
        "morphism": "m", "invariant": "i", "confidence": 0.5,
    }
    idea_resp = _valid_idea_payload({"transplanted_mechanism": "tm"}, "M")
    # 3 domains → 3*(bridge + idea) = 6 responses
    client = _SeqClient([
        bridge_resp, idea_resp, bridge_resp, idea_resp, bridge_resp, idea_resp,
    ])
    out = ana.analogy_batch("topic", claude_client=client, n=3,
                              domains=ana.DEFAULT_DOMAINS[:3], seed=42)
    assert len(out) == 3
    assert client.call_count == 6
    assert all(i.source_strategy == "M" for i in out)


def test_analogy_batch_empty_topic_raises():
    with pytest.raises(ValueError):
        ana.analogy_batch("", n=2)


def test_analogy_batch_zero_n_returns_empty():
    assert ana.analogy_batch("topic", n=0, claude_client=None) == []


# ─────────────────────────────────────────────────────────────────────────────
# failure_mode_ideation
# ─────────────────────────────────────────────────────────────────────────────

def test_failure_enumerate_empty_topic_raises():
    with pytest.raises(ValueError):
        fm.enumerate_failure_modes("")


def test_failure_enumerate_no_client_returns_empty_list():
    assert fm.enumerate_failure_modes("topic", claude_client=None) == []


def test_failure_enumerate_zero_n_returns_empty():
    assert fm.enumerate_failure_modes("topic", n=0, claude_client=None) == []


def test_failure_enumerate_sorts_by_severity_desc():
    client = _SeqClient([{
        "failure_modes": [
            {"name": "weak_baseline", "mechanism": "x", "common_signs": "y",
             "immunity_strategy": "use strong baselines", "severity": 0.3},
            {"name": "data_leakage", "mechanism": "x", "common_signs": "y",
             "immunity_strategy": "split before any preprocessing",
             "severity": 0.9},
            {"name": "p_hacking", "mechanism": "x", "common_signs": "y",
             "immunity_strategy": "pre-register hypotheses", "severity": 0.6},
        ]
    }])
    modes = fm.enumerate_failure_modes("topic", claude_client=client, n=3)
    assert [m.name for m in modes] == ["data_leakage", "p_hacking", "weak_baseline"]
    assert modes[0].severity == 0.9
    assert all(0.0 <= m.severity <= 1.0 for m in modes)


def test_failure_enumerate_drops_entries_without_name_or_immunity():
    client = _SeqClient([{
        "failure_modes": [
            {"name": "", "immunity_strategy": "x"},               # bad
            {"name": "x", "immunity_strategy": ""},                 # bad
            {"name": "good", "immunity_strategy": "strategy"},     # ok
        ]
    }])
    modes = fm.enumerate_failure_modes("topic", claude_client=client)
    assert len(modes) == 1
    assert modes[0].name == "good"


def test_failure_generate_immune_idea_has_Y_strategy_and_metadata():
    mode = fm.FailureMode(name="data_leakage", mechanism="m",
                            common_signs="cs", immunity_strategy="split before all")
    client = _SeqClient([
        _valid_idea_payload({"immunity_mechanism": "held-out preprocessing"}, "Y"),
    ])
    idea = fm.generate_immune_idea("topic", mode, claude_client=client)
    assert idea is not None
    assert idea.source_strategy == "Y"
    assert idea.execution_meta["regen_mode"] == "failure_mode"
    assert idea.execution_meta["failure_mode_targeted"]["name"] == "data_leakage"
    assert "held-out" in idea.execution_meta["immunity_mechanism"]


def test_failure_generate_empty_topic_raises():
    mode = fm.FailureMode(name="x", immunity_strategy="y")
    with pytest.raises(ValueError):
        fm.generate_immune_idea("", mode, claude_client=None)


def test_failure_batch_chain_2n_calls():
    mode_resp = {"failure_modes": [
        {"name": f"mode_{i}", "mechanism": "m", "common_signs": "cs",
         "immunity_strategy": "ims", "severity": 0.5}
        for i in range(3)
    ]}
    idea_resp = _valid_idea_payload({"immunity_mechanism": "im"}, "Y")
    # 1 enumerate call + 3 generate calls
    client = _SeqClient([mode_resp, idea_resp, idea_resp, idea_resp])
    out = fm.failure_mode_batch("topic", claude_client=client, n=3)
    assert len(out) == 3
    assert client.call_count == 4
    assert all(i.source_strategy == "Y" for i in out)


def test_failure_batch_empty_when_enumerate_fails():
    client = _SeqClient([_MockResp(False, "")])
    assert fm.failure_mode_batch("topic", claude_client=client, n=3) == []


# ─────────────────────────────────────────────────────────────────────────────
# extremum_ideation
# ─────────────────────────────────────────────────────────────────────────────

def test_extremum_propose_invalid_axis_raises():
    with pytest.raises(ValueError):
        ex.propose_extreme("topic", "not_an_axis", "minimal", claude_client=None)


def test_extremum_propose_invalid_direction_raises():
    with pytest.raises(ValueError):
        ex.propose_extreme("topic", "compute", "weird_direction",
                              claude_client=None)


def test_extremum_propose_empty_topic_raises():
    with pytest.raises(ValueError):
        ex.propose_extreme("", "compute", "minimal", claude_client=None)


def test_extremum_propose_no_client_returns_none():
    assert ex.propose_extreme("topic", "compute", "minimal",
                                  claude_client=None) is None


def test_extremum_propose_happy_path_returns_regime():
    client = _SeqClient([{
        "magnitude": "single CPU core, 4GB RAM, no GPU",
        "why_hard": "modern methods assume GPUs",
        "what_changes": "every algorithm choice must be O(1) in batch",
        "only_here": "deployable on cheap edge devices in low-income regions",
    }])
    regime = ex.propose_extreme("LLM inference", "compute", "minimal",
                                    claude_client=client)
    assert regime is not None
    assert regime.axis == "compute"
    assert regime.direction == "minimal"
    assert "CPU" in regime.magnitude
    assert regime.label == "compute:minimal"


def test_extremum_propose_missing_magnitude_returns_none():
    client = _SeqClient([{
        "magnitude": "",  # required
        "why_hard": "x", "what_changes": "y", "only_here": "z",
    }])
    assert ex.propose_extreme("topic", "compute", "minimal",
                                  claude_client=client) is None


def test_extremum_generate_at_extreme_builds_idea_with_T_strategy():
    regime = ex.Regime(axis="compute", direction="minimal",
                         magnitude="1 CPU", why_hard="x", what_changes="y",
                         only_here="z")
    client = _SeqClient([
        _valid_idea_payload({"regime_exploit": "uses int8 quantization"}, "T"),
    ])
    idea = ex.generate_at_extreme("topic", regime, claude_client=client)
    assert idea is not None
    assert idea.source_strategy == "T"
    assert idea.execution_meta["regen_mode"] == "extremum"
    assert idea.execution_meta["regime"]["axis"] == "compute"
    assert "int8" in idea.execution_meta["regime_exploit"]


def test_extremum_batch_with_custom_pairs():
    regime_resp = {"magnitude": "m", "why_hard": "w",
                    "what_changes": "wc", "only_here": "oh"}
    idea_resp = _valid_idea_payload({"regime_exploit": "x"}, "T")
    client = _SeqClient([regime_resp, idea_resp, regime_resp, idea_resp])
    out = ex.extremum_batch("topic", claude_client=client, n=2,
                                pairs=[("compute", "minimal"),
                                       ("data", "maximal")])
    assert len(out) == 2
    assert all(i.source_strategy == "T" for i in out)
    assert client.call_count == 4


def test_extremum_batch_rejects_invalid_pair():
    with pytest.raises(ValueError):
        ex.extremum_batch("topic", n=1,
                            pairs=[("nonsense_axis", "minimal")],
                            claude_client=None)


def test_extremum_batch_empty_topic_raises():
    with pytest.raises(ValueError):
        ex.extremum_batch("", n=2)


def test_extremum_batch_zero_n_returns_empty():
    assert ex.extremum_batch("topic", n=0, claude_client=None) == []


def test_extremum_axes_all_have_minimal_and_maximal():
    for axis, dirs in ex.AXES.items():
        assert "minimal" in dirs, f"axis {axis} missing 'minimal'"
        assert "maximal" in dirs, f"axis {axis} missing 'maximal'"


# ─────────────────────────────────────────────────────────────────────────────
# Source-strategy code uniqueness and app wiring
# ─────────────────────────────────────────────────────────────────────────────

def test_new_strategy_codes_distinct_from_existing():
    """M, Y, T must not collide with existing codes A/B/C/E/F/G/H/K/L/N/P/R/U/X."""
    new = {"M", "Y", "T"}
    existing = {"A", "B", "C", "E", "F", "G", "H", "K", "L",
                "N", "P", "R", "U", "X"}
    assert new.isdisjoint(existing)


def test_app_novelty_lab_has_13_modes():
    """Smoke check: the radio in the Novelty Lab lists all 13 mode keys."""
    app_text = (ROOT / "app.py").read_text(encoding="utf-8")
    # Look for the radio options list right after _novelty_mode = st.radio(
    idx = app_text.find("_novelty_mode = st.radio")
    assert idx >= 0
    snippet = app_text[idx:idx + 2000]
    for mode in ["adversarial", "contradiction", "ensemble", "constraint",
                  "future_back", "frontier", "genetic", "heretic", "persona",
                  "counterfactual", "analogy", "failure_mode", "extremum"]:
        assert f'"{mode}"' in snippet, f"mode {mode!r} missing from radio"


def test_source_strategy_codes_present_in_module_files():
    """Each new module must actually emit its declared strategy code."""
    assert 'source_strategy="M"' in \
        (ROOT / "analogy_ideation.py").read_text(encoding="utf-8")
    assert 'source_strategy="Y"' in \
        (ROOT / "failure_mode_ideation.py").read_text(encoding="utf-8")
    assert 'source_strategy="T"' in \
        (ROOT / "extremum_ideation.py").read_text(encoding="utf-8")
