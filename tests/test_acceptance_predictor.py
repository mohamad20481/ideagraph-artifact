"""Tests for acceptance_predictor.py — reviewer-aware acceptance scoring."""
from __future__ import annotations

import json
from typing import Any, Dict

import pytest

from acceptance_predictor import (
    VENUE_PROFILES,
    AcceptanceResult,
    score_idea,
    compare_venues,
    rank_ideas,
    heuristic_score,
    llm_score,
    _decision_from_prob,
    _confidence_from_prob,
    _sigmoid,
    _coerce_unit,
    _extract_features,
)
from models.idea import Idea


# ─── Fixtures ────────────────────────────────────────────────────────────

def _strong_idea() -> Idea:
    i = Idea(
        title="Diffusion-based world models for embodied AI",
        motivation="Sim-to-real for embodied agents.",
        method="Conditional latent diffusion as world model; MPC over predictions.",
        hypothesis="Diffusion world models predict 25% better than autoregressive baselines.",
        resources="8 A100 × 14 days; Habitat, ProcTHOR.",
        expected_outcome="SOTA on Habitat-Sim2Real benchmark",
        risk_assessment="compute-heavy; latency at runtime",
    )
    i.methodology_type = "empirical_study"
    i.novelty_level = "substantial"
    i.quality_score = 0.85
    i.probe_scores = {
        "novelty": 0.85, "significance": 0.90, "specificity": 0.80,
        "clarity": 0.85, "testability": 0.80, "scalability": 0.75,
        "risk_balance": 0.75, "code": 0.80, "dataset": 0.75,
        "constraint": 0.65,
    }
    return i


def _weak_idea() -> Idea:
    i = Idea(
        title="A new survey of generic ML approaches",
        motivation="There are many ML approaches.",
        method="Search papers; categorize.",
        hypothesis="Patterns may emerge.",
        resources="1 person × 1 week",
        expected_outcome="maybe a categorization",
        risk_assessment="unclear",
    )
    i.methodology_type = "survey_meta_analysis"
    i.novelty_level = "incremental"
    i.quality_score = 0.30
    i.probe_scores = {
        "novelty": 0.20, "significance": 0.30, "specificity": 0.30,
        "clarity": 0.50, "testability": 0.30, "scalability": 0.30,
        "risk_balance": 0.40, "code": 0.30, "dataset": 0.40,
        "constraint": 0.50,
    }
    return i


# ─── Math primitives ─────────────────────────────────────────────────────

class TestMathPrimitives:
    def test_sigmoid_extremes(self):
        assert _sigmoid(-50) < 0.001
        assert _sigmoid(50) > 0.999
        assert _sigmoid(0) == pytest.approx(0.5)

    def test_decision_thresholds(self):
        assert _decision_from_prob(0.95) == "accept"
        assert _decision_from_prob(0.40) == "borderline"
        assert _decision_from_prob(0.10) == "reject"

    def test_confidence_zero_at_borderline(self):
        # Confidence should be zero exactly at p=0.5 (max uncertainty)
        assert _confidence_from_prob(0.50) == 0.0
        # And one at the extremes
        assert _confidence_from_prob(0.0) == 1.0
        assert _confidence_from_prob(1.0) == 1.0

    def test_coerce_unit_clamps(self):
        assert _coerce_unit(-5) == 0.0
        assert _coerce_unit(5) == 1.0
        assert _coerce_unit("not a number", 0.7) == 0.7


# ─── Venue profile sanity ────────────────────────────────────────────────

class TestVenueProfiles:
    def test_all_required_venues_present(self):
        # The UI assumes these exist; renaming them is a breaking change
        for v in ("NeurIPS", "ICML", "ICLR", "AAAI", "AISTATS",
                  "ACL", "EMNLP", "CVPR", "KDD", "IJCAI",
                  "ML4H", "Workshop"):
            assert v in VENUE_PROFILES, f"missing venue profile {v}"

    def test_each_profile_has_required_fields(self):
        required = {"tier", "acceptance_rate", "description",
                    "weights", "methodology_preferences",
                    "novelty_preferences", "bias"}
        for v, p in VENUE_PROFILES.items():
            assert required.issubset(p.keys()), f"{v} missing keys"
            assert 0.0 < p["acceptance_rate"] < 1.0
            assert p["description"]

    def test_acceptance_rates_sane(self):
        # No venue should have an absurd rate
        for v, p in VENUE_PROFILES.items():
            assert 0.05 <= p["acceptance_rate"] <= 0.70, \
                f"{v}: implausible acceptance_rate {p['acceptance_rate']}"

    def test_top_tier_more_selective_than_workshop(self):
        # NeurIPS should have stricter weights / higher acceptance bar
        # than the generic Workshop venue
        assert (VENUE_PROFILES["NeurIPS"]["acceptance_rate"]
                < VENUE_PROFILES["Workshop"]["acceptance_rate"])


# ─── Feature extraction ──────────────────────────────────────────────────

class TestFeatureExtraction:
    def test_features_in_unit_interval(self):
        idea = _strong_idea()
        feats = _extract_features(idea, VENUE_PROFILES["NeurIPS"])
        for k, v in feats.items():
            assert 0.0 <= v <= 1.0, f"{k} = {v} out of [0,1]"

    def test_missing_probes_default_to_05(self):
        idea = Idea(
            title="x", motivation="x", method="x", hypothesis="x",
            resources="x", expected_outcome="x", risk_assessment="x",
        )
        feats = _extract_features(idea, VENUE_PROFILES["NeurIPS"])
        assert feats["novelty"] == 0.5
        assert feats["specificity"] == 0.5

    def test_methodology_match_uses_venue_profile(self):
        idea = _strong_idea()
        # NeurIPS prefers empirical_study highly
        f_neur = _extract_features(idea, VENUE_PROFILES["NeurIPS"])
        # AISTATS prefers empirical_study less than theoretical
        f_aistats = _extract_features(idea, VENUE_PROFILES["AISTATS"])
        # NeurIPS values empirical_study more than AISTATS does
        assert f_neur["methodology_match"] >= f_aistats["methodology_match"]

    def test_execution_signal_picked_up_when_present(self):
        idea = _strong_idea()
        idea.execution_signal = 0.78
        feats = _extract_features(idea, VENUE_PROFILES["NeurIPS"])
        assert feats["execution_signal"] == pytest.approx(0.78)


# ─── Heuristic scorer ────────────────────────────────────────────────────

class TestHeuristicScorer:
    def test_strong_idea_predicted_accept_at_top_tier(self):
        r = heuristic_score(_strong_idea(), "NeurIPS")
        assert r.decision == "accept"
        assert r.accept_prob > 0.7

    def test_weak_idea_predicted_reject_at_top_tier(self):
        r = heuristic_score(_weak_idea(), "NeurIPS")
        assert r.decision == "reject"
        assert r.accept_prob < 0.30

    def test_workshop_more_lenient_than_top_tier(self):
        idea = _weak_idea()
        # Same idea: Workshop should give a higher prob than NeurIPS
        p_neur = heuristic_score(idea, "NeurIPS").accept_prob
        p_work = heuristic_score(idea, "Workshop").accept_prob
        assert p_work > p_neur

    def test_strong_beats_weak_at_every_venue(self):
        strong = _strong_idea()
        weak = _weak_idea()
        for v in VENUE_PROFILES:
            ps = score_idea(strong, v).accept_prob
            pw = score_idea(weak, v).accept_prob
            assert ps > pw, f"{v}: strong={ps:.2f} not > weak={pw:.2f}"

    def test_unknown_venue_returns_error(self):
        r = heuristic_score(_strong_idea(), "MadeUpConf")
        assert r.error and "unknown venue" in r.error.lower()
        assert r.decision == "reject"

    def test_strengths_and_weaknesses_populated(self):
        r = heuristic_score(_strong_idea(), "NeurIPS")
        assert isinstance(r.top_strengths, list)
        assert isinstance(r.top_weaknesses, list)
        assert len(r.top_strengths) >= 1

    def test_feature_contributions_match_weight_keys(self):
        r = heuristic_score(_strong_idea(), "NeurIPS")
        weight_keys = set(VENUE_PROFILES["NeurIPS"]["weights"].keys())
        contrib_keys = set(r.feature_contributions.keys())
        assert contrib_keys == weight_keys

    def test_used_llm_false_for_heuristic(self):
        r = heuristic_score(_strong_idea(), "NeurIPS")
        assert r.used_llm is False

    def test_idea_dict_input_supported(self):
        # The function should accept plain dicts as well as Idea instances
        idea_dict = _strong_idea().to_dict()
        r = heuristic_score(idea_dict, "NeurIPS")
        assert r.accept_prob > 0.5


# ─── Calibration: bias term targets the venue acceptance rate ────────────

class TestCalibration:
    def test_neutral_idea_hits_acceptance_rate(self):
        # A *truly* neutral idea: all probes 0.5, methodology_type and
        # novelty_level left None so methodology_match/novelty_match
        # default to 0.5 too. This is what the bias term is calibrated
        # against, so the predicted prob should land near the venue's
        # published acceptance rate.
        idea = Idea(
            title="generic", motivation="x", method="x", hypothesis="x",
            resources="x", expected_outcome="x", risk_assessment="x",
        )
        # methodology_type / novelty_level deliberately left None
        idea.quality_score = 0.5
        idea.probe_scores = {k: 0.5 for k in (
            "novelty", "significance", "specificity", "clarity",
            "testability", "scalability", "risk_balance",
            "code", "dataset", "constraint",
        )}
        for v in ("NeurIPS", "ICML", "ICLR", "AAAI", "ACL",
                  "CVPR", "KDD", "Workshop"):
            target = VENUE_PROFILES[v]["acceptance_rate"]
            p = heuristic_score(idea, v).accept_prob
            assert abs(p - target) < 0.05, (
                f"{v}: predicted {p:.2f} vs target {target:.2f}"
            )


# ─── Public dispatchers ──────────────────────────────────────────────────

class TestDispatchers:
    def test_score_idea_default_mode_is_heuristic(self):
        r = score_idea(_strong_idea())
        assert r.used_llm is False
        assert r.venue == "NeurIPS"

    def test_score_idea_unknown_mode_treated_as_heuristic(self):
        # Defensive: unknown mode shouldn't crash, just heuristic-fall-through
        r = score_idea(_strong_idea(), mode="bogus")
        assert isinstance(r, AcceptanceResult)
        assert r.used_llm is False

    def test_compare_venues_returns_sorted_descending(self):
        out = compare_venues(_strong_idea())
        assert len(out) == len(VENUE_PROFILES)
        for i in range(len(out) - 1):
            assert out[i].accept_prob >= out[i + 1].accept_prob

    def test_compare_venues_subset(self):
        out = compare_venues(_strong_idea(),
                              venues=["NeurIPS", "Workshop"])
        assert {r.venue for r in out} == {"NeurIPS", "Workshop"}

    def test_rank_ideas_returns_pairs(self):
        ideas = [_strong_idea(), _weak_idea()]
        out = rank_ideas(ideas, venue="NeurIPS")
        assert len(out) == 2
        # Strong idea must rank above weak idea
        first_idea, first_result = out[0]
        assert first_idea.title == _strong_idea().title
        assert first_result.accept_prob > out[1][1].accept_prob


# ─── LLM scorer (mocked) ─────────────────────────────────────────────────

class _FakeOK:
    def __init__(self, text):
        self.success = True
        self.text = text
        self.error = None
        self.cost_usd = 0.001


class _FakeFail:
    success = False
    text = ""
    error = "rate limit"
    cost_usd = 0.0


class _Client:
    def __init__(self, response):
        self._r = response
        self.calls: list = []

    def call(self, system, user, **kw):
        self.calls.append({"system": system, "user": user, **kw})
        return self._r


_VALID_LLM_RESPONSE = {
    "accept_prob": 0.72,
    "decision": "accept",
    "confidence": 0.85,
    "strengths": ["Clear experimental design",
                   "Novel use of diffusion priors",
                   "Strong baselines"],
    "weaknesses": ["Compute cost may exceed reviewer expectations",
                    "Limited theoretical analysis",
                    "Single-domain evaluation"],
    "one_line_verdict": "Accept; would defend in discussion.",
}


class TestLLMScorer:
    def test_parses_well_formed_response(self):
        client = _Client(_FakeOK(json.dumps(_VALID_LLM_RESPONSE)))
        r = llm_score(_strong_idea(), "NeurIPS", client)
        assert r.used_llm is True
        assert r.accept_prob == pytest.approx(0.72)
        assert r.decision == "accept"
        assert len(r.top_strengths) == 3
        assert len(r.top_weaknesses) == 3

    def test_falls_back_when_call_fails(self):
        client = _Client(_FakeFail())
        r = llm_score(_strong_idea(), "NeurIPS", client)
        assert r.used_llm is False
        assert r.error and "rate limit" in r.error.lower()

    def test_falls_back_on_garbage_json(self):
        client = _Client(_FakeOK("not json"))
        r = llm_score(_strong_idea(), "NeurIPS", client)
        assert r.used_llm is False
        assert r.error and "unparseable" in r.error.lower()

    def test_falls_back_on_fenced_json(self):
        # The LLM sometimes wraps responses in ```json ... ```
        wrapped = "```json\n" + json.dumps(_VALID_LLM_RESPONSE) + "\n```"
        client = _Client(_FakeOK(wrapped))
        r = llm_score(_strong_idea(), "NeurIPS", client)
        assert r.used_llm is True
        assert r.decision == "accept"

    def test_no_client_falls_back_to_heuristic(self):
        r = llm_score(_strong_idea(), "NeurIPS", claude_client=None)
        assert r.used_llm is False
        assert r.error and "unavailable" in r.error.lower()
        # But the heuristic answer is still a valid prediction
        assert 0.0 <= r.accept_prob <= 1.0

    def test_unknown_venue_via_llm_returns_error(self):
        client = _Client(_FakeOK(json.dumps(_VALID_LLM_RESPONSE)))
        r = llm_score(_strong_idea(), "MadeUpConf", client)
        assert r.error and "unknown venue" in r.error.lower()

    def test_invalid_decision_string_normalized(self):
        # If the LLM returns a free-form decision word, we should map it
        # back to one of accept/borderline/reject based on accept_prob
        bad = dict(_VALID_LLM_RESPONSE)
        bad["decision"] = "weak accept"  # not in our enum
        client = _Client(_FakeOK(json.dumps(bad)))
        r = llm_score(_strong_idea(), "NeurIPS", client)
        assert r.decision in ("accept", "borderline", "reject")


# ─── Integration: app wiring ─────────────────────────────────────────────

class TestAppWiring:
    def test_app_imports_acceptance_predictor(self):
        with open("app.py", encoding="utf-8") as f:
            src = f.read()
        assert "from acceptance_predictor import" in src
        assert "tab_reviewer" in src
        assert '"Reviewer Lens"' in src

    def test_disclaimer_banner_present_in_app(self):
        # Load-bearing: the controversial-tool framing must be visible
        with open("app.py", encoding="utf-8") as f:
            src = f.read()
        # Spot-check a few phrases from the disclaimer
        assert "descriptive, not prescriptive" in src
        assert "misuse" in src
