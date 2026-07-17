"""Tests for the fourth wave of novelty modules: inversion, null_result, underserved_cohort."""
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

import inversion_ideation as iv
import null_result_ideation as nr
import underserved_cohort_ideation as uc
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
        "title": "Test Title",
        "motivation": "Why.",
        "method": "Concrete method.",
        "hypothesis": "Falsifiable.",
        "resources": "1 GPU-week.",
        "expected_outcome": "Measurable.",
        "risk_assessment": "Risks.",
        "source_strategy": strategy,
        "methodology_type": METHODOLOGY_TYPES[0],
        "novelty_level": NOVELTY_LEVELS[1],
    }
    base.update(extra)
    return base


# ─────────────────────────────────────────────────────────────────────────────
# inversion_ideation
# ─────────────────────────────────────────────────────────────────────────────

def test_inversion_propose_empty_topic_raises():
    with pytest.raises(ValueError):
        iv.propose_answer("", "underdog_wins")


def test_inversion_propose_invalid_tone_raises():
    with pytest.raises(ValueError):
        iv.propose_answer("topic", "not_a_tone")


def test_inversion_propose_no_client_returns_none():
    assert iv.propose_answer("topic", "underdog_wins",
                                claude_client=None) is None


def test_inversion_propose_happy_path():
    client = _SeqClient([{
        "headline": "a 100M-param model beats GPT-4 on dialectal Arabic NER",
        "why_surprising": "scale-dominance is canonical for low-resource NLP",
        "why_plausible": "data quality > parameter count in narrow domain",
        "measurable_claim": "F1 >= 0.8 with <= 100M params on cohort X",
        "plausibility": 0.45,
    }])
    ans = iv.propose_answer("low-resource NLP", "underdog_wins",
                                claude_client=client)
    assert ans is not None
    assert ans.tone == "underdog_wins"
    assert ans.plausibility == pytest.approx(0.45)
    assert "100M" in ans.headline


def test_inversion_propose_missing_required_returns_none():
    client = _SeqClient([{
        "headline": "",  # required
        "measurable_claim": "x",
    }])
    assert iv.propose_answer("topic", "underdog_wins",
                                claude_client=client) is None


def test_inversion_propose_plausibility_clamped():
    client = _SeqClient([{
        "headline": "h", "measurable_claim": "m", "plausibility": -5.0,
    }])
    ans = iv.propose_answer("topic", "underdog_wins", claude_client=client)
    assert ans is not None
    assert ans.plausibility == 0.0


def test_inversion_derive_question_has_I_strategy_and_metadata():
    ans = iv.CandidateAnswer(
        tone="underdog_wins", headline="h", why_surprising="s",
        why_plausible="p", measurable_claim="m",
    )
    client = _SeqClient([
        _idea_payload({"derived_question": "Can a small model match GPT-4?"},
                        "I"),
    ])
    idea = iv.derive_question("topic", ans, claude_client=client)
    assert idea is not None
    assert idea.source_strategy == "I"
    assert idea.execution_meta["regen_mode"] == "inversion"
    assert idea.execution_meta["candidate_answer"]["tone"] == "underdog_wins"
    assert "small model" in idea.execution_meta["derived_question"]


def test_inversion_derive_invalid_enums_coerced_to_none():
    ans = iv.CandidateAnswer(tone="underdog_wins", headline="h",
                                 measurable_claim="m")
    payload = _idea_payload({}, "I")
    payload["methodology_type"] = "garbage"
    payload["novelty_level"] = "garbage"
    client = _SeqClient([payload])
    idea = iv.derive_question("topic", ans, claude_client=client)
    assert idea is not None
    assert idea.methodology_type is None
    assert idea.novelty_level is None


def test_inversion_batch_chain_call_count():
    ans = {"headline": "h", "measurable_claim": "m", "plausibility": 0.5,
            "why_surprising": "s", "why_plausible": "p"}
    idea = _idea_payload({"derived_question": "Q"}, "I")
    client = _SeqClient([ans, idea, ans, idea, ans, idea])
    out = iv.inversion_batch("topic", claude_client=client, n=3,
                                  tones=["underdog_wins", "favourite_fails",
                                         "tradeoff_inversion"])
    assert len(out) == 3
    assert client.call_count == 6
    assert all(i.source_strategy == "I" for i in out)


def test_inversion_batch_invalid_tone_in_override_raises():
    with pytest.raises(ValueError):
        iv.inversion_batch("topic", n=1, tones=["bogus_tone"],
                              claude_client=None)


def test_inversion_batch_empty_topic_raises():
    with pytest.raises(ValueError):
        iv.inversion_batch("", n=1)


def test_inversion_batch_zero_n_returns_empty():
    assert iv.inversion_batch("topic", n=0, claude_client=None) == []


# ─────────────────────────────────────────────────────────────────────────────
# null_result_ideation
# ─────────────────────────────────────────────────────────────────────────────

def test_null_propose_empty_topic_raises():
    with pytest.raises(ValueError):
        nr.propose_null_target("", "transfer_failure")


def test_null_propose_invalid_kind_raises():
    with pytest.raises(ValueError):
        nr.propose_null_target("topic", "not_a_kind")


def test_null_propose_no_client_returns_none():
    assert nr.propose_null_target("topic", "transfer_failure",
                                       claude_client=None) is None


def test_null_propose_happy_path():
    client = _SeqClient([{
        "claim_to_be_negated": "RLHF transfers from English to all dialects",
        "why_widely_assumed": "early multilingual benchmarks suggested it",
        "why_doubt_now": "alignment regressions on dialectal Arabic",
        "population": "speakers of Maghrebi Arabic dialects",
        "equivalence_margin": "|Δhelpfulness| < 0.05 on 95% CI",
        "stakes": 0.85,
    }])
    target = nr.propose_null_target("RLHF transfer", "transfer_failure",
                                          claude_client=client)
    assert target is not None
    assert target.kind == "transfer_failure"
    assert target.stakes == pytest.approx(0.85)
    assert "Maghrebi" in target.population


def test_null_propose_missing_claim_or_margin_returns_none():
    client = _SeqClient([{
        "claim_to_be_negated": "",  # required
        "equivalence_margin": "x",
    }])
    assert nr.propose_null_target("topic", "transfer_failure",
                                       claude_client=client) is None


def test_null_design_experiment_has_Z_strategy():
    target = nr.NullTarget(
        kind="transfer_failure", claim_to_be_negated="claim",
        population="pop", equivalence_margin="<5%",
    )
    client = _SeqClient([
        _idea_payload({
            "null_acceptance_criteria": "CI contained in [-5%, +5%]",
            "power_analysis_summary": "N=2400 detects 5% at 80% power",
        }, "Z"),
    ])
    idea = nr.design_null_experiment("topic", target, claude_client=client)
    assert idea is not None
    assert idea.source_strategy == "Z"
    assert idea.execution_meta["regen_mode"] == "null_result"
    assert idea.execution_meta["null_target"]["kind"] == "transfer_failure"
    assert "N=2400" in idea.execution_meta["power_analysis_summary"]
    assert "CI" in idea.execution_meta["null_acceptance_criteria"]


def test_null_design_empty_topic_raises():
    target = nr.NullTarget(kind="transfer_failure",
                                claim_to_be_negated="c",
                                equivalence_margin="m")
    with pytest.raises(ValueError):
        nr.design_null_experiment("", target, claude_client=None)


def test_null_batch_chain_call_count():
    target_resp = {
        "claim_to_be_negated": "c", "why_widely_assumed": "w",
        "why_doubt_now": "d", "population": "p",
        "equivalence_margin": "<5%", "stakes": 0.5,
    }
    idea_resp = _idea_payload({
        "null_acceptance_criteria": "CI",
        "power_analysis_summary": "N=100",
    }, "Z")
    client = _SeqClient([target_resp, idea_resp,
                              target_resp, idea_resp,
                              target_resp, idea_resp])
    out = nr.null_result_batch("topic", claude_client=client, n=3,
                                     kinds=["transfer_failure",
                                            "scaling_plateau",
                                            "cohort_invalidity"])
    assert len(out) == 3
    assert client.call_count == 6
    assert all(i.source_strategy == "Z" for i in out)


def test_null_batch_invalid_kind_in_override_raises():
    with pytest.raises(ValueError):
        nr.null_result_batch("topic", n=1, kinds=["nope"],
                                   claude_client=None)


def test_null_batch_empty_topic_raises():
    with pytest.raises(ValueError):
        nr.null_result_batch("", n=1)


def test_null_batch_zero_n_returns_empty():
    assert nr.null_result_batch("topic", n=0, claude_client=None) == []


# ─────────────────────────────────────────────────────────────────────────────
# underserved_cohort_ideation
# ─────────────────────────────────────────────────────────────────────────────

def test_cohort_identify_empty_topic_raises():
    with pytest.raises(ValueError):
        uc.identify_cohort("", "linguistic")


def test_cohort_identify_invalid_dimension_raises():
    with pytest.raises(ValueError):
        uc.identify_cohort("topic", "not_a_dimension")


def test_cohort_identify_no_client_returns_none():
    assert uc.identify_cohort("topic", "linguistic",
                                  claude_client=None) is None


def test_cohort_identify_happy_path():
    client = _SeqClient([{
        "name": "Levantine Arabic speakers in NW Syria",
        "description": "~2M speakers, intermittent connectivity",
        "why_underserved": "canonical ASR trained on MSA + Egyptian dialect",
        "canonical_failure_mode": "WER above 50% on dialectal input",
        "success_metric": "task success on dialectal voice queries",
        "overlooked_factor": 0.9,
    }])
    cohort = uc.identify_cohort("ASR", "linguistic", claude_client=client)
    assert cohort is not None
    assert cohort.dimension == "linguistic"
    assert "Levantine" in cohort.name
    assert cohort.overlooked_factor == pytest.approx(0.9)


def test_cohort_identify_missing_name_or_metric_returns_none():
    client = _SeqClient([{
        "name": "",  # required
        "success_metric": "x",
    }])
    assert uc.identify_cohort("topic", "linguistic",
                                  claude_client=client) is None


def test_cohort_design_has_W_strategy_and_metadata():
    cohort = uc.Cohort(
        dimension="linguistic", name="X speakers",
        description="d", why_underserved="w", success_metric="m",
    )
    client = _SeqClient([
        _idea_payload({
            "cohort_representation": "100h held-out dialectal speech",
        }, "W"),
    ])
    idea = uc.design_for_cohort("topic", cohort, claude_client=client)
    assert idea is not None
    assert idea.source_strategy == "W"
    assert idea.execution_meta["regen_mode"] == "underserved_cohort"
    assert idea.execution_meta["cohort"]["dimension"] == "linguistic"
    assert "dialectal" in idea.execution_meta["cohort_representation"]


def test_cohort_design_empty_topic_raises():
    cohort = uc.Cohort(dimension="linguistic", name="n", success_metric="m")
    with pytest.raises(ValueError):
        uc.design_for_cohort("", cohort, claude_client=None)


def test_cohort_batch_chain_call_count():
    cohort_resp = {
        "name": "n", "description": "d", "why_underserved": "w",
        "canonical_failure_mode": "cf", "success_metric": "sm",
        "overlooked_factor": 0.5,
    }
    idea_resp = _idea_payload({"cohort_representation": "rep"}, "W")
    client = _SeqClient([cohort_resp, idea_resp,
                              cohort_resp, idea_resp])
    out = uc.underserved_cohort_batch(
        "topic", claude_client=client, n=2,
        dimensions=["linguistic", "infrastructural"],
    )
    assert len(out) == 2
    assert client.call_count == 4
    assert all(i.source_strategy == "W" for i in out)


def test_cohort_batch_invalid_dimension_in_override_raises():
    with pytest.raises(ValueError):
        uc.underserved_cohort_batch("topic", n=1, dimensions=["nope"],
                                          claude_client=None)


def test_cohort_batch_empty_topic_raises():
    with pytest.raises(ValueError):
        uc.underserved_cohort_batch("", n=1)


def test_cohort_batch_zero_n_returns_empty():
    assert uc.underserved_cohort_batch("topic", n=0,
                                              claude_client=None) == []


# ─────────────────────────────────────────────────────────────────────────────
# Source-strategy code uniqueness + app wiring
# ─────────────────────────────────────────────────────────────────────────────

def test_new_strategy_codes_distinct_from_existing():
    """I, Z, W must not collide with codes used by other modes."""
    new = {"I", "Z", "W"}
    existing = {"A", "B", "C", "E", "F", "G", "H", "K", "L", "M",
                 "N", "P", "R", "T", "U", "X", "Y"}
    assert new.isdisjoint(existing)


def test_app_novelty_lab_has_16_modes():
    """Smoke check: the radio in the Novelty Lab lists all 16 mode keys."""
    app_text = (ROOT / "app.py").read_text(encoding="utf-8")
    idx = app_text.find("_novelty_mode = st.radio")
    assert idx >= 0
    snippet = app_text[idx:idx + 2500]
    expected = ["adversarial", "contradiction", "ensemble", "constraint",
                  "future_back", "frontier", "genetic", "heretic", "persona",
                  "counterfactual", "analogy", "failure_mode", "extremum",
                  "inversion", "null_result", "underserved_cohort"]
    for mode in expected:
        assert f'"{mode}"' in snippet, f"mode {mode!r} missing from radio"
    assert len(expected) == 16


def test_source_strategy_codes_present_in_module_files():
    assert 'source_strategy="I"' in \
        (ROOT / "inversion_ideation.py").read_text(encoding="utf-8")
    assert 'source_strategy="Z"' in \
        (ROOT / "null_result_ideation.py").read_text(encoding="utf-8")
    assert 'source_strategy="W"' in \
        (ROOT / "underserved_cohort_ideation.py").read_text(encoding="utf-8")
