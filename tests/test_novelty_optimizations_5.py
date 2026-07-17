"""Tests for the fifth wave of novelty modules: composable_primitive, stakeholder_pareto."""
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

import composable_primitive_ideation as cp
import stakeholder_pareto_ideation as sp
from models.idea import METHODOLOGY_TYPES, NOVELTY_LEVELS


@dataclass
class _MockResp:
    success: bool
    text: str


class _SeqClient:
    def __init__(self, responses: List[Any]):
        self._queue = list(responses)
        self.call_count = 0

    def call(self, system: str, user: str, **kw) -> _MockResp:
        self.call_count += 1
        if not self._queue:
            return _MockResp(False, "")
        item = self._queue.pop(0)
        if isinstance(item, _MockResp):
            return item
        if isinstance(item, dict):
            return _MockResp(True, json.dumps(item))
        if isinstance(item, str):
            return _MockResp(True, item)
        return _MockResp(False, "")


def _idea_payload(extra: dict, strategy: str) -> dict:
    base = {
        "title": "T",
        "motivation": "m", "method": "method", "hypothesis": "h",
        "resources": "r", "expected_outcome": "e", "risk_assessment": "ra",
        "source_strategy": strategy,
        "methodology_type": METHODOLOGY_TYPES[0],
        "novelty_level": NOVELTY_LEVELS[1],
    }
    base.update(extra)
    return base


# ─────────────────────────────────────────────────────────────────────────────
# composable_primitive_ideation (strategy D)
# ─────────────────────────────────────────────────────────────────────────────

def test_cp_identify_empty_topic_raises():
    with pytest.raises(ValueError):
        cp.identify_primitive_slot("", "evaluation_harness")


def test_cp_identify_invalid_kind_raises():
    with pytest.raises(ValueError):
        cp.identify_primitive_slot("topic", "not_a_kind")


def test_cp_identify_no_client_returns_none():
    assert cp.identify_primitive_slot("topic", "evaluation_harness",
                                          claude_client=None) is None


def test_cp_identify_happy_path():
    client = _SeqClient([{
        "name": "DialectalASR-Probe v0",
        "description": "100h held-out probe",
        "why_unfilled": "field uses MSA-only benchmarks",
        "downstream_users": "ASR researchers, dialect projects",
        "adoption_proxy": "GitHub forks + downstream citations",
        "blast_radius": 0.7,
    }])
    slot = cp.identify_primitive_slot("ASR", "evaluation_harness",
                                          claude_client=client)
    assert slot is not None
    assert slot.kind == "evaluation_harness"
    assert "DialectalASR" in slot.name
    assert slot.blast_radius == pytest.approx(0.7)


def test_cp_identify_missing_name_or_proxy_returns_none():
    client = _SeqClient([{
        "name": "",  # required
        "adoption_proxy": "x",
    }])
    assert cp.identify_primitive_slot("topic", "evaluation_harness",
                                          claude_client=client) is None


def test_cp_identify_blast_radius_clamped():
    client = _SeqClient([{
        "name": "n", "adoption_proxy": "p", "blast_radius": 10.0,
    }])
    slot = cp.identify_primitive_slot("topic", "evaluation_harness",
                                          claude_client=client)
    assert slot is not None
    assert slot.blast_radius == 1.0


def test_cp_design_primitive_has_D_strategy_and_metadata():
    slot = cp.PrimitiveSlot(
        kind="evaluation_harness", name="X-Probe v0",
        description="d", adoption_proxy="forks",
    )
    client = _SeqClient([
        _idea_payload({
            "api_surface": "def run_probe(model, lang) -> Dict[str, float]",
            "non_goals": "not a training framework; eval only",
        }, "D"),
    ])
    idea = cp.design_primitive("topic", slot, claude_client=client)
    assert idea is not None
    assert idea.source_strategy == "D"
    assert idea.execution_meta["regen_mode"] == "composable_primitive"
    assert idea.execution_meta["primitive_slot"]["name"] == "X-Probe v0"
    assert "def run_probe" in idea.execution_meta["api_surface"]
    assert "eval only" in idea.execution_meta["non_goals"]


def test_cp_design_invalid_enums_coerced_to_none():
    slot = cp.PrimitiveSlot(kind="evaluation_harness", name="n",
                                adoption_proxy="p")
    payload = _idea_payload({}, "D")
    payload["methodology_type"] = "garbage"
    payload["novelty_level"] = "garbage"
    client = _SeqClient([payload])
    idea = cp.design_primitive("topic", slot, claude_client=client)
    assert idea is not None
    assert idea.methodology_type is None
    assert idea.novelty_level is None


def test_cp_design_empty_topic_raises():
    slot = cp.PrimitiveSlot(kind="evaluation_harness", name="n",
                                adoption_proxy="p")
    with pytest.raises(ValueError):
        cp.design_primitive("", slot, claude_client=None)


def test_cp_batch_chain_call_count():
    slot_resp = {
        "name": "n", "description": "d", "why_unfilled": "w",
        "downstream_users": "du", "adoption_proxy": "ap",
        "blast_radius": 0.5,
    }
    idea_resp = _idea_payload({"api_surface": "api", "non_goals": "ng"}, "D")
    client = _SeqClient([slot_resp, idea_resp,
                              slot_resp, idea_resp,
                              slot_resp, idea_resp])
    out = cp.composable_primitive_batch(
        "topic", claude_client=client, n=3,
        kinds=["evaluation_harness", "diagnostic_probe",
                "compositional_block"],
    )
    assert len(out) == 3
    assert client.call_count == 6
    assert all(i.source_strategy == "D" for i in out)


def test_cp_batch_invalid_kind_override_raises():
    with pytest.raises(ValueError):
        cp.composable_primitive_batch("topic", n=1, kinds=["nope"],
                                          claude_client=None)


def test_cp_batch_empty_topic_raises():
    with pytest.raises(ValueError):
        cp.composable_primitive_batch("", n=1)


def test_cp_batch_zero_n_returns_empty():
    assert cp.composable_primitive_batch("topic", n=0,
                                              claude_client=None) == []


# ─────────────────────────────────────────────────────────────────────────────
# stakeholder_pareto_ideation (strategy S)
# ─────────────────────────────────────────────────────────────────────────────

def test_sp_cast_empty_topic_raises():
    with pytest.raises(ValueError):
        sp.cast_stakeholders("", ["researcher", "end_user"])


def test_sp_cast_empty_roles_raises():
    with pytest.raises(ValueError):
        sp.cast_stakeholders("topic", [])


def test_sp_cast_invalid_role_raises():
    with pytest.raises(ValueError):
        sp.cast_stakeholders("topic", ["researcher", "not_a_role"])


def test_sp_cast_no_client_returns_empty():
    assert sp.cast_stakeholders("topic", ["researcher"],
                                    claude_client=None) == []


def test_sp_cast_filters_bad_entries():
    client = _SeqClient([{
        "stakeholders": [
            {"role": "researcher", "name": "ML researcher",
             "metric": "publication count",
             "win_condition": "publish paper", "risk_if_ignored": "no buy-in"},
            {"role": "fake_role", "name": "x", "metric": "y"},   # invalid role
            {"role": "end_user", "name": "", "metric": "m"},      # empty name
            {"role": "end_user", "name": "clinicians",
             "metric": "minutes saved per shift",
             "win_condition": "less time", "risk_if_ignored": "no use"},
        ]
    }])
    cast = sp.cast_stakeholders(
        "topic", ["researcher", "end_user"], claude_client=client,
    )
    assert len(cast) == 2
    assert {c.role for c in cast} == {"researcher", "end_user"}


def test_sp_design_empty_topic_raises():
    s1 = sp.Stakeholder(role="researcher", name="r", metric="m")
    s2 = sp.Stakeholder(role="end_user", name="u", metric="m")
    with pytest.raises(ValueError):
        sp.design_pareto_idea("", [s1, s2], claude_client=None)


def test_sp_design_too_few_stakeholders_raises():
    s1 = sp.Stakeholder(role="researcher", name="r", metric="m")
    with pytest.raises(ValueError):
        sp.design_pareto_idea("topic", [s1], claude_client=None)


def test_sp_design_pareto_idea_has_S_strategy():
    cast = [
        sp.Stakeholder(role="researcher", name="ML researcher", metric="papers"),
        sp.Stakeholder(role="end_user", name="clinicians",
                         metric="minutes saved"),
        sp.Stakeholder(role="regulator", name="FDA reviewer",
                         metric="audit pass rate"),
    ]
    client = _SeqClient([
        _idea_payload({
            "tradeoffs_named": "accuracy is traded for auditability",
            "per_stakeholder_metric": {
                "researcher": "papers",
                "end_user": "minutes saved",
                "regulator": "audit pass rate",
            },
        }, "S"),
    ])
    idea = sp.design_pareto_idea("topic", cast, claude_client=client)
    assert idea is not None
    assert idea.source_strategy == "S"
    assert idea.execution_meta["regen_mode"] == "stakeholder_pareto"
    assert len(idea.execution_meta["stakeholders"]) == 3
    assert isinstance(idea.execution_meta["per_stakeholder_metric"], dict)
    assert idea.execution_meta["per_stakeholder_metric"]["regulator"] == \
        "audit pass rate"


def test_sp_design_per_stakeholder_metric_string_fallback():
    cast = [
        sp.Stakeholder(role="researcher", name="r", metric="m"),
        sp.Stakeholder(role="end_user", name="u", metric="m"),
    ]
    payload = _idea_payload({
        "tradeoffs_named": "t",
        "per_stakeholder_metric": "researcher → papers; end_user → minutes",
    }, "S")
    client = _SeqClient([payload])
    idea = sp.design_pareto_idea("topic", cast, claude_client=client)
    assert idea is not None
    assert isinstance(idea.execution_meta["per_stakeholder_metric"], str)
    assert "papers" in idea.execution_meta["per_stakeholder_metric"]


def test_sp_batch_chain_call_count():
    cast_resp = {
        "stakeholders": [
            {"role": "researcher", "name": "n", "metric": "m",
             "win_condition": "w", "risk_if_ignored": "r"},
            {"role": "end_user", "name": "u", "metric": "m",
             "win_condition": "w", "risk_if_ignored": "r"},
            {"role": "regulator", "name": "g", "metric": "m",
             "win_condition": "w", "risk_if_ignored": "r"},
        ]
    }
    idea_resp = _idea_payload({"tradeoffs_named": "t",
                                  "per_stakeholder_metric": {"researcher": "m"}},
                                 "S")
    # 2 runs × (1 cast + 1 design) = 4 calls
    client = _SeqClient([cast_resp, idea_resp, cast_resp, idea_resp])
    out = sp.stakeholder_pareto_batch(
        "topic", claude_client=client, n=2, cast_size=3,
        roles=["researcher", "end_user", "regulator", "funder"],
        seed=42,
    )
    assert len(out) == 2
    assert client.call_count == 4
    assert all(i.source_strategy == "S" for i in out)


def test_sp_batch_cast_size_too_small_raises():
    with pytest.raises(ValueError):
        sp.stakeholder_pareto_batch("topic", n=1, cast_size=1,
                                          claude_client=None)


def test_sp_batch_cast_size_exceeds_pool_raises():
    with pytest.raises(ValueError):
        sp.stakeholder_pareto_batch(
            "topic", n=1, cast_size=5,
            roles=["researcher", "end_user"],
            claude_client=None,
        )


def test_sp_batch_invalid_role_in_pool_raises():
    with pytest.raises(ValueError):
        sp.stakeholder_pareto_batch(
            "topic", n=1, cast_size=2,
            roles=["researcher", "not_a_role"],
            claude_client=None,
        )


def test_sp_batch_empty_topic_raises():
    with pytest.raises(ValueError):
        sp.stakeholder_pareto_batch("", n=1)


def test_sp_batch_zero_n_returns_empty():
    assert sp.stakeholder_pareto_batch("topic", n=0,
                                            claude_client=None) == []


# ─────────────────────────────────────────────────────────────────────────────
# Source-strategy code uniqueness + app wiring
# ─────────────────────────────────────────────────────────────────────────────

def test_new_strategy_codes_distinct_from_existing():
    """D and S must not collide with codes already used."""
    new = {"D", "S"}
    existing = {"A", "B", "C", "E", "F", "G", "H", "I", "K", "L", "M",
                 "N", "P", "R", "T", "U", "W", "X", "Y", "Z"}
    assert new.isdisjoint(existing)


def test_app_novelty_lab_has_18_modes():
    """The radio in the Novelty Lab must list all 18 mode keys."""
    app_text = (ROOT / "app.py").read_text(encoding="utf-8")
    idx = app_text.find("_novelty_mode = st.radio")
    assert idx >= 0
    snippet = app_text[idx:idx + 3000]
    expected = ["adversarial", "contradiction", "ensemble", "constraint",
                  "future_back", "frontier", "genetic", "heretic", "persona",
                  "counterfactual", "analogy", "failure_mode", "extremum",
                  "inversion", "null_result", "underserved_cohort",
                  "composable_primitive", "stakeholder_pareto"]
    assert len(expected) == 18
    for mode in expected:
        assert f'"{mode}"' in snippet, f"mode {mode!r} missing from radio"


def test_source_strategy_codes_present_in_module_files():
    assert 'source_strategy="D"' in \
        (ROOT / "composable_primitive_ideation.py").read_text(encoding="utf-8")
    assert 'source_strategy="S"' in \
        (ROOT / "stakeholder_pareto_ideation.py").read_text(encoding="utf-8")
