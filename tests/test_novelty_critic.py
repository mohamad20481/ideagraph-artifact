"""Tests for agents/novelty_critic.py — adversarial originality critic."""
from __future__ import annotations
import json

import pytest

from agents.novelty_critic import (
    NoveltyCritique,
    critique_novelty,
    attack_and_revise,
    _AUTOLOAD,
    _coerce_unit,
    _parse_json,
    _user_prompt,
)
from models.idea import Idea


def _idea(**overrides):
    base = dict(
        title="GNN for ADMET",
        motivation="Drug screening is expensive.",
        method="Use message-passing on molecular graphs over ZINC.",
        hypothesis="GNNs beat fingerprints",
        resources="1 A100",
        expected_outcome="15% AUROC gain",
        risk_assessment="dataset bias",
    )
    base.update({k: v for k, v in overrides.items() if k in base})
    idea = Idea(**base)
    idea.methodology_type = overrides.get("methodology_type", "empirical_study")
    idea.novelty_level = overrides.get("novelty_level", "incremental")
    idea.quality_score = overrides.get("quality_score", 0.65)
    return idea


class _FakeOK:
    def __init__(self, text):
        self.success = True
        self.text = text
        self.error = None
        self.cost_usd = 0
        self.model = "test"


class _FakeFail:
    success = False
    text = ""
    error = "rate limit"
    cost_usd = 0


class _Client:
    def __init__(self, response_or_factory):
        self._r = response_or_factory
        self.calls = []

    def call(self, system, user, **kw):
        self.calls.append({"system": system, "user": user, **kw})
        if callable(self._r):
            return self._r(len(self.calls) - 1)
        return self._r


_VALID = {
    "originality_score": 0.35,
    "overall_verdict": "incremental",
    "critiques": ["Standard MPNN with no architectural novelty",
                   "ZINC+ADMET combo done many times"],
    "similar_prior_work": ["Gilmer 2017 MPNN",
                             "Yang 2019 D-MPNN for ADMET"],
    "pivots": ["Add equivariant features",
                "Test on out-of-distribution scaffolds",
                "Couple with active-learning loop"],
    "confidence": 0.85,
}


# ── Helpers ──────────────────────────────────────────────────────────────

class TestHelpers:
    def test_coerce_unit_clamps_and_nan(self):
        assert _coerce_unit(-1) == 0
        assert _coerce_unit(2) == 1
        assert _coerce_unit(float("nan"), 0.5) == 0.5
        assert _coerce_unit("not a number", 0.7) == 0.7

    def test_parse_json_strips_fences(self):
        wrapped = "```json\n" + json.dumps(_VALID) + "\n```"
        d = _parse_json(wrapped)
        assert d and d["overall_verdict"] == "incremental"

    def test_parse_json_returns_none_on_garbage(self):
        assert _parse_json("not json") is None

    def test_user_prompt_includes_idea_fields(self):
        idea = _idea()
        prompt = _user_prompt(idea)
        assert "GNN for ADMET" in prompt
        assert "molecular graphs" in prompt
        assert "originality_score" in prompt  # schema in the prompt


# ── critique_novelty ─────────────────────────────────────────────────────

class TestCritiqueNovelty:
    def test_no_client_returns_degraded(self):
        r = critique_novelty(_idea(), claude_client=None)
        assert r.used_llm is False
        assert r.error and "not configured" in r.error.lower()
        # Score still in valid range
        assert 0.0 <= r.originality_score <= 1.0

    def test_mocked_response_parses_correctly(self):
        client = _Client(_FakeOK(json.dumps(_VALID)))
        r = critique_novelty(_idea(), claude_client=client)
        assert r.used_llm is True
        assert r.originality_score == pytest.approx(0.35)
        assert r.overall_verdict == "incremental"
        assert len(r.critiques) == 2
        assert len(r.pivots) == 3
        assert len(r.similar_prior_work) == 2

    def test_invalid_verdict_normalized_from_score(self):
        bad = dict(_VALID)
        bad["overall_verdict"] = "ok i guess"
        client = _Client(_FakeOK(json.dumps(bad)))
        r = critique_novelty(_idea(), claude_client=client)
        # score 0.35 → "incremental" per the threshold logic
        assert r.overall_verdict == "incremental"

    def test_call_failure_falls_back(self):
        client = _Client(_FakeFail())
        r = critique_novelty(_idea(), claude_client=client)
        assert r.used_llm is False
        assert r.error and "rate limit" in r.error.lower()

    def test_garbage_response_falls_back(self):
        client = _Client(_FakeOK("totally not json"))
        r = critique_novelty(_idea(), claude_client=client)
        assert r.used_llm is False
        assert r.error and "not valid json" in r.error.lower()

    def test_summary_format(self):
        client = _Client(_FakeOK(json.dumps(_VALID)))
        r = critique_novelty(_idea(), claude_client=client)
        s = r.summary()
        assert "incremental" in s and "0.35" in s


# ── attack_and_revise ────────────────────────────────────────────────────

class TestAttackAndRevise:
    def _chain_client(self):
        """Two-step client: critique on call 1, revision on call 2."""
        def factory(i):
            if i == 0:
                return _FakeOK(json.dumps(_VALID))
            return _FakeOK(json.dumps({
                "title": "Equivariant GNN with active-learning for ADMET",
                "motivation": "Closes OOD gap",
                "method": "SE(3)-equivariant MPNN with uncertainty sampling",
                "hypothesis": "OOD AUROC +10%",
                "resources": "r", "expected_outcome": "e", "risk_assessment": "r",
                "methodology_type": "empirical_study",
                "novelty_level": "moderate",
                "novelty_pivots_applied": ["Add equivariance", "Active learning"],
            }))
        return _Client(factory)

    def test_chain_produces_revised_idea(self):
        client = self._chain_client()
        critique, revised = attack_and_revise(_idea(), claude_client=client)
        assert critique.used_llm is True
        assert revised is not None
        assert revised.source_strategy == "N"
        assert revised.parent_title == "GNN for ADMET"
        assert revised.generation == 1
        assert len(client.calls) == 2

    def test_revised_idea_has_critique_in_meta(self):
        critique, revised = attack_and_revise(
            _idea(), claude_client=self._chain_client(),
        )
        meta = revised.execution_meta or {}
        assert "novelty_critique" in meta
        assert meta["regen_mode"] == "novelty_revision"
        assert "pivots_applied" in meta
        assert len(meta["pivots_applied"]) == 2

    def test_no_client_returns_critique_and_none(self):
        critique, revised = attack_and_revise(_idea(), claude_client=None)
        assert critique.used_llm is False
        assert revised is None

    def test_critic_failure_skips_revision(self):
        # If the critique call fails, revision should NOT proceed
        client = _Client(_FakeFail())
        critique, revised = attack_and_revise(_idea(), claude_client=client)
        assert critique.used_llm is False
        assert revised is None
        # Only one call attempted (the critic) — revision never fired
        assert len(client.calls) == 1


# ── App wiring smoke check ───────────────────────────────────────────────

class TestAppWiring:
    def test_app_imports_novelty_critic(self):
        with open("app.py", encoding="utf-8") as f:
            src = f.read()
        assert "from agents.novelty_critic import" in src
        assert "Novelty Lab" in src
        assert "tab_novelty" in src
