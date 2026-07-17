"""Tests for contradiction_ideation.py — TRIZ-style ideation."""
from __future__ import annotations
import json

import pytest

from contradiction_ideation import (
    Contradiction,
    extract_contradictions,
    generate_from_contradiction,
    generate_from_contradictions_batch,
    _parse_json,
    _coerce_unit,
)


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
    error = "fail"
    cost_usd = 0


class _SequencedClient:
    """Returns a different response on each call. Use this when batch flows
    need to extract → generate → generate without exhausting one response."""

    def __init__(self, responses):
        self._r = responses
        self.calls = []

    def call(self, system, user, **kw):
        idx = len(self.calls)
        self.calls.append({"system": system[:50], "user": user[:50], **kw})
        if idx < len(self._r):
            return self._r[idx]
        return _FakeFail()


_EXTRACT_RESPONSE = {
    "contradictions": [
        {"statement": "We want expressiveness AND efficiency",
         "forces_a": "expressiveness → many params",
         "forces_b": "efficiency → few params",
         "why_it_matters": "Edge deployment needs both",
         "resolution_hint": "Conditional computation",
         "severity": 0.9},
        {"statement": "Global context AND local detail",
         "forces_a": "global → wide RF",
         "forces_b": "local → narrow RF",
         "why_it_matters": "Graph learning needs both",
         "resolution_hint": "Hierarchical",
         "severity": 0.7},
    ]
}


def _idea_response(title="Mixture-of-Experts GNN"):
    return {
        "title": title,
        "motivation": "Resolves expressiveness/efficiency tension",
        "method": "MoE routes k of N experts per node based on local context",
        "hypothesis": "Same accuracy with 4× fewer active params",
        "resources": "r", "expected_outcome": "e", "risk_assessment": "r",
        "methodology_type": "system_design",
        "novelty_level": "moderate",
        "resolution_mechanism": "Conditional capacity routing",
    }


# ── Helpers ──────────────────────────────────────────────────────────────

class TestHelpers:
    def test_parse_json_strips_fences(self):
        wrapped = "```json\n" + json.dumps(_EXTRACT_RESPONSE) + "\n```"
        d = _parse_json(wrapped)
        assert d and "contradictions" in d

    def test_coerce_unit_clamp(self):
        assert _coerce_unit(-1.0) == 0.0
        assert _coerce_unit(2.0) == 1.0
        assert _coerce_unit("nope", 0.5) == 0.5


# ── Contradiction dataclass ──────────────────────────────────────────────

class TestContradictionDataclass:
    def test_to_dict_round_trip(self):
        c = Contradiction(
            statement="A AND B", forces_a="x", forces_b="y",
            why_it_matters="z", resolution_hint="hint", severity=0.8,
        )
        d = c.to_dict()
        assert d["statement"] == "A AND B"
        assert d["severity"] == 0.8


# ── extract_contradictions ───────────────────────────────────────────────

class TestExtract:
    def test_empty_topic_raises(self):
        with pytest.raises(ValueError, match="non-empty topic"):
            extract_contradictions("", claude_client=None)

    def test_whitespace_topic_raises(self):
        with pytest.raises(ValueError):
            extract_contradictions("   ", claude_client=None)

    def test_no_client_returns_empty(self):
        assert extract_contradictions("topic", claude_client=None) == []

    def test_n_zero_returns_empty(self):
        client = _SequencedClient([_FakeOK(json.dumps(_EXTRACT_RESPONSE))])
        assert extract_contradictions("topic", claude_client=client, n=0) == []

    def test_mocked_extraction_parses(self):
        client = _SequencedClient([_FakeOK(json.dumps(_EXTRACT_RESPONSE))])
        out = extract_contradictions("topic", claude_client=client, n=2)
        assert len(out) == 2
        # Each contradiction has all the required fields
        for c in out:
            assert c.statement and 0.0 <= c.severity <= 1.0

    def test_sorted_by_severity_desc(self):
        client = _SequencedClient([_FakeOK(json.dumps(_EXTRACT_RESPONSE))])
        out = extract_contradictions("topic", claude_client=client, n=2)
        assert out[0].severity == 0.9
        assert out[1].severity == 0.7

    def test_garbage_response_returns_empty(self):
        client = _SequencedClient([_FakeOK("not json")])
        assert extract_contradictions("topic", claude_client=client) == []

    def test_call_failure_returns_empty(self):
        client = _SequencedClient([_FakeFail()])
        assert extract_contradictions("topic", claude_client=client) == []

    def test_invalid_severity_coerced(self):
        bad = {"contradictions": [
            {"statement": "A AND B", "forces_a": "x", "forces_b": "y",
             "why_it_matters": "z", "resolution_hint": "h",
             "severity": "very high"},  # not a number
        ]}
        client = _SequencedClient([_FakeOK(json.dumps(bad))])
        out = extract_contradictions("topic", claude_client=client)
        assert out[0].severity == 0.5  # falls back to default


# ── generate_from_contradiction ──────────────────────────────────────────

class TestGenerate:
    def _contradiction(self):
        return Contradiction(
            statement="A AND B but conflict",
            forces_a="x", forces_b="y",
            why_it_matters="z", resolution_hint="h",
            severity=0.9,
        )

    def test_empty_topic_raises(self):
        with pytest.raises(ValueError):
            generate_from_contradiction(
                "", self._contradiction(), claude_client=None,
            )

    def test_no_client_returns_none(self):
        out = generate_from_contradiction(
            "topic", self._contradiction(), claude_client=None,
        )
        assert out is None

    def test_mocked_produces_idea(self):
        client = _SequencedClient([_FakeOK(json.dumps(_idea_response()))])
        idea = generate_from_contradiction(
            "topic", self._contradiction(), claude_client=client,
        )
        assert idea is not None
        assert idea.source_strategy == "C"
        assert idea.generation == 0
        assert idea.parent_title is None

    def test_idea_meta_carries_contradiction(self):
        client = _SequencedClient([_FakeOK(json.dumps(_idea_response()))])
        idea = generate_from_contradiction(
            "topic", self._contradiction(), claude_client=client,
        )
        meta = idea.execution_meta or {}
        assert meta["regen_mode"] == "contradiction_driven"
        assert "contradiction_resolved" in meta
        # The contradiction must be preserved in the idea's lineage
        cr = meta["contradiction_resolved"]
        assert cr["statement"] == "A AND B but conflict"
        assert cr["severity"] == 0.9
        # Resolution mechanism is captured
        assert meta["resolution_mechanism"] == "Conditional capacity routing"

    def test_garbage_response_returns_none(self):
        client = _SequencedClient([_FakeOK("not json")])
        out = generate_from_contradiction(
            "topic", self._contradiction(), claude_client=client,
        )
        assert out is None

    def test_missing_required_field_returns_none(self):
        bad = dict(_idea_response())
        del bad["method"]
        client = _SequencedClient([_FakeOK(json.dumps(bad))])
        out = generate_from_contradiction(
            "topic", self._contradiction(), claude_client=client,
        )
        assert out is None

    def test_invalid_methodology_normalized_to_none(self):
        bad = dict(_idea_response())
        bad["methodology_type"] = "totally_fake_method"
        client = _SequencedClient([_FakeOK(json.dumps(bad))])
        idea = generate_from_contradiction(
            "topic", self._contradiction(), claude_client=client,
        )
        assert idea is not None
        assert idea.methodology_type is None


# ── Batch ────────────────────────────────────────────────────────────────

class TestBatch:
    def test_batch_produces_one_idea_per_contradiction(self):
        client = _SequencedClient([
            _FakeOK(json.dumps(_EXTRACT_RESPONSE)),     # extract
            _FakeOK(json.dumps(_idea_response("idea 1"))),  # gen 1
            _FakeOK(json.dumps(_idea_response("idea 2"))),  # gen 2
        ])
        out = generate_from_contradictions_batch(
            "topic", claude_client=client, n=2,
        )
        assert len(out) == 2
        assert all(i.source_strategy == "C" for i in out)
        # Three calls total: 1 extraction + 2 generations
        assert len(client.calls) == 3

    def test_batch_with_empty_extraction_returns_empty(self):
        client = _SequencedClient([_FakeOK("not json")])
        out = generate_from_contradictions_batch(
            "topic", claude_client=client, n=2,
        )
        assert out == []
        # Only 1 call — no generation attempts after extraction failed
        assert len(client.calls) == 1

    def test_batch_skips_failed_generations(self):
        # Extract OK, first gen OK, second gen returns garbage → dropped
        client = _SequencedClient([
            _FakeOK(json.dumps(_EXTRACT_RESPONSE)),
            _FakeOK(json.dumps(_idea_response("good"))),
            _FakeOK("not json"),
        ])
        out = generate_from_contradictions_batch(
            "topic", claude_client=client, n=2,
        )
        # Only one survives
        assert len(out) == 1
        assert out[0].title == "good"


# ── App wiring ───────────────────────────────────────────────────────────

class TestAppWiring:
    def test_app_imports_contradiction_ideation(self):
        with open("app.py", encoding="utf-8") as f:
            src = f.read()
        assert "from contradiction_ideation import" in src
        assert "tab_novelty" in src
