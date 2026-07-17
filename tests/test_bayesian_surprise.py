"""Tests for bayesian_surprise.py (Extension 3 — epistemic novelty).

All LLM calls are mocked. Tests verify:
  - KL surrogate math (cosine-distance + JSD) on synthetic vectors
  - LLM-driven pipeline: prior/posterior sampling + plausibility
  - persist_to_meta stamps result onto idea.execution_meta
  - sort_modes integration: bayesian_surprise + epistemic_shift modes
  - Disk-cache round-trip + hash-key determinism
  - Graceful degradation: no LLM, empty hypothesis, bad plausibility text
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bayesian_surprise as bs


# ── Helpers ─────────────────────────────────────────────────────────────────

def _mock_client(responses):
    """Build a client whose .call() returns ClaudeResponse-like mocks
    in the supplied order. Reset cycle when responses run out."""
    cli = MagicMock()
    idx = {"i": 0}

    def _call(system, user, max_tokens=200, temperature=0.7, **kw):
        i = idx["i"] % len(responses)
        idx["i"] += 1
        resp = MagicMock()
        resp.success = True
        resp.text = responses[i]
        return resp

    cli.call.side_effect = _call
    return cli


# ── Surprise math ───────────────────────────────────────────────────────────

def test_surprise_from_identical_vectors_is_zero():
    v = {"transformer": 0.7, "attention": 0.7}
    # Re-normalize so it's unit length.
    import math
    n = math.sqrt(sum(x * x for x in v.values()))
    v = {k: x / n for k, x in v.items()}
    assert bs.surprise_from_vectors(v, v) == 0.0


def test_surprise_from_orthogonal_vectors_is_one():
    v1 = {"a": 1.0}
    v2 = {"b": 1.0}
    assert bs.surprise_from_vectors(v1, v2) == pytest.approx(1.0, abs=1e-6)


def test_surprise_from_partial_overlap_is_in_between():
    import math
    v1 = {"a": 1.0 / math.sqrt(2), "b": 1.0 / math.sqrt(2)}
    v2 = {"b": 1.0 / math.sqrt(2), "c": 1.0 / math.sqrt(2)}
    s = bs.surprise_from_vectors(v1, v2)
    # cos = 0.5 → surprise = 0.5
    assert s == pytest.approx(0.5, abs=1e-6)


def test_surprise_handles_empty_inputs():
    assert bs.surprise_from_vectors({}, {"a": 1.0}) == 0.0
    assert bs.surprise_from_vectors({"a": 1.0}, {}) == 0.0
    assert bs.surprise_from_vectors({}, {}) == 0.0


def test_jsd_identical_distributions_is_zero():
    v = {"a": 0.5, "b": 0.5}
    assert bs.jensen_shannon_divergence(v, v) == pytest.approx(0.0, abs=1e-6)


def test_jsd_disjoint_distributions_is_one():
    v1 = {"a": 1.0}
    v2 = {"b": 1.0}
    # JSD on fully disjoint = log(2); normalized → 1.0
    assert bs.jensen_shannon_divergence(v1, v2) == pytest.approx(1.0, abs=1e-6)


def test_jsd_handles_empty_inputs():
    assert bs.jensen_shannon_divergence({}, {"a": 1.0}) == 0.0
    assert bs.jensen_shannon_divergence({"a": 1.0}, {}) == 0.0


# ── Aggregate token vector ─────────────────────────────────────────────────

def test_aggregate_token_vector_unit_norm():
    """Output should be L2 unit length."""
    import math
    out = bs._aggregate_token_vector([
        "attention mechanism transformer",
        "transformer attention sparse",
    ])
    norm = math.sqrt(sum(v * v for v in out.values()))
    assert norm == pytest.approx(1.0, abs=1e-4)


def test_aggregate_token_vector_empty_returns_empty():
    assert bs._aggregate_token_vector([]) == {}
    assert bs._aggregate_token_vector([""]) == {}


# ── Plausibility parsing ────────────────────────────────────────────────────

def test_parse_plausibility_extracts_first_float():
    assert bs._parse_plausibility("0.85") == 0.85
    assert bs._parse_plausibility("0.85 because…") == 0.85
    assert bs._parse_plausibility("The rating is 0.7.") == 0.7
    assert bs._parse_plausibility("1") == 1.0


def test_parse_plausibility_clamps_out_of_range():
    assert bs._parse_plausibility("1.5") == 1.0
    assert bs._parse_plausibility("-0.3") == pytest.approx(0.3)  # sign stripped


def test_parse_plausibility_default_on_garbage():
    assert bs._parse_plausibility("") == 0.5
    assert bs._parse_plausibility("not a number") == 0.5


# ── End-to-end compute_bayesian_surprise ────────────────────────────────────

def test_compute_surprise_with_mock_client():
    """Happy path: prior + posterior both sampled, plausibility rated."""
    # 3 samples each direction (default) + 1 plausibility call = 7 calls total.
    # Cycle through 7 responses.
    cli = _mock_client([
        # 3 prior responses (similar field description)
        "Transformers dominate NLP with attention.",
        "Attention is the workhorse of modern NLP.",
        "Most NLP uses transformer-based attention.",
        # 3 posterior responses (shifted by H = "sparse attention")
        "Sparse attention reshapes the efficiency-quality tradeoff.",
        "If sparse attention scales, current dense models become obsolete.",
        "Sparse attention opens efficient long-context modeling.",
        # 1 plausibility
        "0.8",
    ])
    result = bs.compute_bayesian_surprise(
        hypothesis="sparse attention can replace dense attention",
        topic="efficient NLP",
        llm_client=cli,
        n_samples=3,
        rate_plausibility=True,
    )
    assert result.n_samples == 3
    assert 0.0 < result.surprise <= 1.0
    assert 0.0 <= result.jsd <= 1.0
    assert result.plausibility == 0.8
    # bayesian_score = surprise * plausibility
    assert result.bayesian_score == pytest.approx(
        result.surprise * 0.8, abs=1e-6,
    )
    assert result.cache_key


def test_compute_surprise_no_client_returns_zero_struct():
    """Missing client should not raise — return zero-filled struct."""
    result = bs.compute_bayesian_surprise(
        hypothesis="something",
        topic="x",
        llm_client=None,
        n_samples=3,
    )
    assert result.surprise == 0.0
    assert result.plausibility == 0.5  # default neutral
    assert result.bayesian_score == 0.0
    assert result.n_samples == 0


def test_compute_surprise_empty_hypothesis_returns_zero():
    result = bs.compute_bayesian_surprise(
        hypothesis="",
        topic="x",
        llm_client=_mock_client(["irrelevant"]),
    )
    assert result.surprise == 0.0
    assert result.bayesian_score == 0.0


def test_compute_surprise_without_plausibility_skips_extra_call():
    """rate_plausibility=False should NOT issue the extra LLM call."""
    cli = _mock_client(["prior"] * 3 + ["posterior"] * 3)
    result = bs.compute_bayesian_surprise(
        hypothesis="H",
        topic="x",
        llm_client=cli,
        n_samples=3,
        rate_plausibility=False,
    )
    # 3 prior + 3 posterior = 6 calls, no plausibility.
    assert cli.call.call_count == 6
    assert result.plausibility == 0.5  # neutral default


def test_compute_surprise_handles_provider_failure():
    """When LLM returns success=False, the aggregation falls back to empty
    vectors — surprise = 0, no crash."""
    cli = MagicMock()

    def _failing_call(**kw):
        resp = MagicMock()
        resp.success = False
        resp.text = ""
        return resp

    cli.call.side_effect = _failing_call
    result = bs.compute_bayesian_surprise(
        hypothesis="H", topic="x", llm_client=cli, n_samples=2,
    )
    assert result.surprise == 0.0
    assert result.jsd == 0.0


# ── compute_surprise_for_idea (stamps meta) ─────────────────────────────────

def test_compute_surprise_for_idea_stamps_execution_meta():
    cli = _mock_client(
        ["prior"] * 3 + ["posterior with novel angle"] * 3 + ["0.7"]
    )
    idea = {
        "title": "T",
        "motivation": "x",
        "hypothesis": "sparse routing improves long-context efficiency",
    }
    payload = bs.compute_surprise_for_idea(
        idea, topic="efficient NLP", llm_client=cli, n_samples=3,
    )
    assert "bayesian_score" in payload
    assert idea["execution_meta"]["bayesian_surprise"] == payload


def test_compute_surprise_for_idea_falls_back_to_motivation():
    """When hypothesis is empty, falls back to motivation."""
    cli = _mock_client(["a"] * 3 + ["b"] * 3 + ["0.5"])
    idea = {"title": "T", "motivation": "the motivation text", "hypothesis": ""}
    bs.compute_surprise_for_idea(
        idea, topic="x", llm_client=cli, n_samples=3,
    )
    # Just verify we didn't crash and meta was stamped.
    assert "bayesian_surprise" in idea["execution_meta"]


def test_compute_surprise_for_idea_no_persist_when_flag_off():
    cli = _mock_client(["a"] * 3 + ["b"] * 3 + ["0.5"])
    idea = {"title": "T", "hypothesis": "h"}
    bs.compute_surprise_for_idea(
        idea, topic="x", llm_client=cli, n_samples=3,
        persist_to_meta=False,
    )
    assert "execution_meta" not in idea or "bayesian_surprise" not in (
        idea.get("execution_meta") or {}
    )


# ── Sort key helpers ───────────────────────────────────────────────────────

def test_bayesian_score_key_reads_meta():
    idea = {"execution_meta": {"bayesian_surprise": {"bayesian_score": 0.42}}}
    assert bs.bayesian_score_key(idea) == 0.42


def test_bayesian_score_key_missing_returns_zero():
    assert bs.bayesian_score_key({}) == 0.0
    assert bs.bayesian_score_key({"execution_meta": {}}) == 0.0
    assert bs.bayesian_score_key({"execution_meta": {
        "bayesian_surprise": {}
    }}) == 0.0


def test_surprise_key_reads_meta_without_plausibility_weight():
    idea = {"execution_meta": {"bayesian_surprise": {
        "surprise": 0.7, "plausibility": 0.4, "bayesian_score": 0.28,
    }}}
    # surprise_key returns raw surprise (0.7), not the bayesian_score (0.28).
    assert bs.surprise_key(idea) == 0.7


# ── idea_sorting.py integration ────────────────────────────────────────────

def test_sort_by_bayesian_surprise_orders_by_cached_score():
    import idea_sorting
    ideas = [
        {"title": "A", "execution_meta": {"bayesian_surprise": {
            "bayesian_score": 0.1
        }}},
        {"title": "B", "execution_meta": {"bayesian_surprise": {
            "bayesian_score": 0.9
        }}},
        {"title": "C", "execution_meta": {"bayesian_surprise": {
            "bayesian_score": 0.5
        }}},
    ]
    out = idea_sorting.sort_ideas(ideas, "bayesian_surprise", descending=True)
    assert [i["title"] for i in out] == ["B", "C", "A"]


def test_sort_by_epistemic_shift_ignores_plausibility():
    """epistemic_shift mode uses raw surprise, not the multiplied score."""
    import idea_sorting
    ideas = [
        # A has lower bayesian_score but higher raw surprise.
        {"title": "A", "execution_meta": {"bayesian_surprise": {
            "surprise": 0.9, "plausibility": 0.1, "bayesian_score": 0.09,
        }}},
        {"title": "B", "execution_meta": {"bayesian_surprise": {
            "surprise": 0.4, "plausibility": 0.9, "bayesian_score": 0.36,
        }}},
    ]
    out = idea_sorting.sort_ideas(ideas, "epistemic_shift", descending=True)
    # A wins on raw surprise even though its bayesian_score is lower.
    assert [i["title"] for i in out] == ["A", "B"]


def test_new_modes_appear_in_sort_modes_registry():
    import idea_sorting
    assert "bayesian_surprise" in idea_sorting.SORT_MODES
    assert "epistemic_shift" in idea_sorting.SORT_MODES
    # Both should be directional (high-to-low toggle is meaningful).
    assert "bayesian_surprise" in idea_sorting.DIRECTIONAL_MODES
    assert "epistemic_shift" in idea_sorting.DIRECTIONAL_MODES


# ── Hash-key + disk cache ──────────────────────────────────────────────────

def test_hash_key_deterministic_same_inputs():
    k1 = bs._hash_key("hypothesis A", "topic X", 3)
    k2 = bs._hash_key("hypothesis A", "topic X", 3)
    assert k1 == k2


def test_hash_key_changes_with_each_input_dimension():
    base = bs._hash_key("H", "T", 3)
    assert bs._hash_key("H'", "T", 3) != base
    assert bs._hash_key("H", "T'", 3) != base
    assert bs._hash_key("H", "T", 4) != base


def test_disk_cache_roundtrip(tmp_path):
    """First call computes + writes; second call reads from disk without
    issuing any LLM calls."""
    cli = _mock_client(["a"] * 3 + ["b"] * 3 + ["0.6"])
    idea = {"title": "T", "hypothesis": "sparse routing"}
    cache_dir = str(tmp_path / "surprise_cache")
    payload1 = bs.cached_compute_surprise_for_idea(
        idea, topic="x", llm_client=cli, n_samples=3,
        cache_dir=cache_dir,
    )
    n_calls_first = cli.call.call_count
    assert n_calls_first == 7

    # Second call on a fresh idea dict with same hypothesis+topic+N
    # should hit cache → ZERO additional LLM calls.
    idea2 = {"title": "T2", "hypothesis": "sparse routing"}
    payload2 = bs.cached_compute_surprise_for_idea(
        idea2, topic="x", llm_client=cli, n_samples=3,
        cache_dir=cache_dir,
    )
    assert cli.call.call_count == n_calls_first  # no new calls
    assert payload2["bayesian_score"] == payload1["bayesian_score"]
    assert idea2["execution_meta"]["bayesian_surprise"] == payload1


def test_disk_cache_separate_keys_for_different_hypotheses(tmp_path):
    cli = _mock_client(["x"] * 100)
    cache_dir = str(tmp_path / "cache_keys")
    idea_a = {"hypothesis": "A"}
    idea_b = {"hypothesis": "B"}
    bs.cached_compute_surprise_for_idea(
        idea_a, topic="t", llm_client=cli, n_samples=2, cache_dir=cache_dir,
    )
    n_calls_a = cli.call.call_count
    bs.cached_compute_surprise_for_idea(
        idea_b, topic="t", llm_client=cli, n_samples=2, cache_dir=cache_dir,
    )
    # B should NOT hit A's cache.
    assert cli.call.call_count > n_calls_a
