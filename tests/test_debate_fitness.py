"""Tests for debate_fitness.py (Extension 1 lite — adversarial fitness).

All LLM calls are mocked. Tests cover:
  - σ(τ · margin) math
  - Judge-reply parser (JSON, fences, prose fallback)
  - End-to-end compute_debate_fitness over T rounds
  - persist_to_meta stamps execution_meta.debate_fitness
  - sort_modes integration (debate_fitness, debate_margin)
  - Disk-cache round-trip + hash-key determinism
  - Graceful degradation: no client, bad judge JSON
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

import debate_fitness as df


# ── Sigmoid + margin math ──────────────────────────────────────────────────

def test_sigmoid_at_zero_is_half():
    assert df._sigmoid(0.0) == pytest.approx(0.5)


def test_sigmoid_monotonic_increasing():
    assert df._sigmoid(-3) < df._sigmoid(-1) < df._sigmoid(0) < df._sigmoid(1) < df._sigmoid(3)


def test_sigmoid_bounded():
    assert 0.0 <= df._sigmoid(100.0) <= 1.0
    assert 0.0 <= df._sigmoid(-100.0) <= 1.0


def test_fitness_tied_debate_is_half():
    # Equal proposer and critic scores → margin = 0 → σ(0) = 0.5
    f = df.fitness_from_margin([0.6, 0.7], [0.6, 0.7])
    assert f == pytest.approx(0.5)


def test_fitness_proposer_dominates_above_half():
    f = df.fitness_from_margin([0.9, 0.9], [0.2, 0.2], tau=2.0)
    assert f > 0.5
    assert f <= 1.0


def test_fitness_critic_dominates_below_half():
    f = df.fitness_from_margin([0.2, 0.2], [0.9, 0.9], tau=2.0)
    assert f < 0.5
    assert f >= 0.0


def test_fitness_empty_inputs_returns_half():
    assert df.fitness_from_margin([], []) == pytest.approx(0.5)


def test_fitness_higher_tau_sharper_decisions():
    f_soft = df.fitness_from_margin([0.7], [0.5], tau=1.0)
    f_sharp = df.fitness_from_margin([0.7], [0.5], tau=4.0)
    # Same positive margin (+0.2) — higher tau means closer to 1.0.
    assert f_sharp > f_soft


# ── Judge reply parsing ────────────────────────────────────────────────────

def test_parse_judge_clean_json():
    raw = '{"proposer_score": 0.7, "critic_score": 0.3, "rationale": "x"}'
    out = df._parse_judge_reply(raw)
    assert out["proposer_score"] == 0.7
    assert out["critic_score"] == 0.3


def test_parse_judge_strips_markdown_fences():
    raw = '```json\n{"proposer_score": 0.8, "critic_score": 0.2}\n```'
    out = df._parse_judge_reply(raw)
    assert out["proposer_score"] == 0.8


def test_parse_judge_clamps_out_of_range():
    raw = '{"proposer_score": 1.5, "critic_score": -0.5}'
    out = df._parse_judge_reply(raw)
    assert out["proposer_score"] == 1.0
    assert out["critic_score"] == 0.0


def test_parse_judge_fallback_to_prose_floats():
    """If JSON parsing fails, pull the first two floats from the text."""
    raw = "Proposer: 0.65 Critic: 0.35 (no JSON)"
    out = df._parse_judge_reply(raw)
    assert out["proposer_score"] == 0.65
    assert out["critic_score"] == 0.35


def test_parse_judge_empty_string_returns_neutral():
    out = df._parse_judge_reply("")
    assert out["proposer_score"] == 0.5
    assert out["critic_score"] == 0.5


def test_parse_judge_total_garbage_returns_neutral():
    out = df._parse_judge_reply("not a number at all")
    assert out["proposer_score"] == 0.5
    assert out["critic_score"] == 0.5


# ── End-to-end compute_debate_fitness ──────────────────────────────────────

def _mock_client(responses):
    """Build a client that returns the supplied responses in order."""
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


def test_compute_debate_fitness_t1_happy_path():
    """T=1 → 3 calls (proposer, critic, judge). Verify fitness > 0.5
    when judge favors proposer."""
    cli = _mock_client([
        "proposer says X",      # proposer turn
        "critic says Y",        # critic turn
        '{"proposer_score": 0.8, "critic_score": 0.3}',  # judge
    ])
    idea = {
        "title": "T",
        "hypothesis": "sparse routing scales",
        "motivation": "fewer ops",
    }
    result = df.compute_debate_fitness(idea, llm_client=cli, n_rounds=1, tau=2.0)
    assert result.n_rounds == 1
    assert result.proposer_total == 0.8
    assert result.critic_total == 0.3
    assert result.margin == pytest.approx(0.5)
    # σ(2 * 0.5) = σ(1) ≈ 0.73
    assert result.fitness == pytest.approx(df._sigmoid(1.0), abs=1e-6)
    assert len(result.rounds) == 1


def test_compute_debate_fitness_t2_invokes_six_calls():
    """T=2 → 6 calls total (proposer, critic, judge ×2)."""
    cli = _mock_client([
        "p1", "c1", '{"proposer_score": 0.6, "critic_score": 0.4}',
        "p2", "c2", '{"proposer_score": 0.7, "critic_score": 0.5}',
    ])
    idea = {"title": "T", "hypothesis": "H"}
    df.compute_debate_fitness(idea, llm_client=cli, n_rounds=2)
    assert cli.call.call_count == 6


def test_compute_debate_fitness_no_client_returns_neutral():
    result = df.compute_debate_fitness(
        {"hypothesis": "x"}, llm_client=None, n_rounds=2,
    )
    assert result.fitness == 0.5
    assert result.n_rounds == 0
    assert result.cache_key  # still populated


def test_compute_debate_fitness_not_dict_returns_neutral():
    result = df.compute_debate_fitness(
        "not an idea", llm_client=_mock_client(["x"]), n_rounds=1,
    )
    assert result.fitness == 0.5
    assert result.n_rounds == 0


def test_compute_debate_fitness_handles_provider_failures():
    """When LLM returns success=False, judge falls back to 0.5/0.5 →
    margin = 0 → fitness = 0.5. No crash."""
    cli = MagicMock()
    failing = MagicMock()
    failing.success = False
    failing.text = ""
    cli.call.return_value = failing
    result = df.compute_debate_fitness(
        {"hypothesis": "x"}, llm_client=cli, n_rounds=2,
    )
    assert result.n_rounds == 2
    assert result.proposer_total == 1.0  # 0.5 × 2 rounds
    assert result.critic_total == 1.0
    assert result.fitness == pytest.approx(0.5)


def test_compute_fitness_for_idea_stamps_meta():
    cli = _mock_client([
        "p", "c", '{"proposer_score": 0.6, "critic_score": 0.4}',
    ])
    idea = {"title": "T", "hypothesis": "H"}
    payload = df.compute_fitness_for_idea(idea, llm_client=cli, n_rounds=1)
    assert "fitness" in payload
    assert idea["execution_meta"]["debate_fitness"] == payload


def test_compute_fitness_for_idea_no_persist_when_flag_off():
    cli = _mock_client([
        "p", "c", '{"proposer_score": 0.5, "critic_score": 0.5}',
    ])
    idea = {"title": "T", "hypothesis": "H"}
    df.compute_fitness_for_idea(
        idea, llm_client=cli, n_rounds=1, persist_to_meta=False,
    )
    assert "execution_meta" not in idea or \
           "debate_fitness" not in (idea.get("execution_meta") or {})


# ── Sort-key helpers ──────────────────────────────────────────────────────

def test_debate_fitness_key_reads_meta():
    idea = {"execution_meta": {"debate_fitness": {"fitness": 0.82}}}
    assert df.debate_fitness_key(idea) == 0.82


def test_debate_fitness_key_missing_returns_neutral():
    """Unscored ideas default to 0.5 (neutral) so they don't get
    artificially deflated/inflated when mixed with scored ideas."""
    assert df.debate_fitness_key({}) == 0.5
    assert df.debate_fitness_key({"execution_meta": {}}) == 0.5
    assert df.debate_fitness_key({
        "execution_meta": {"debate_fitness": {}}
    }) == 0.5


def test_debate_margin_key_reads_meta():
    idea = {"execution_meta": {"debate_fitness": {"margin": -0.4}}}
    assert df.debate_margin_key(idea) == -0.4


def test_debate_margin_key_missing_returns_zero():
    assert df.debate_margin_key({}) == 0.0


# ── idea_sorting.py integration ────────────────────────────────────────────

def test_sort_by_debate_fitness_orders_by_cached_score():
    import idea_sorting
    ideas = [
        {"title": "A", "execution_meta": {"debate_fitness": {"fitness": 0.3}}},
        {"title": "B", "execution_meta": {"debate_fitness": {"fitness": 0.9}}},
        {"title": "C", "execution_meta": {"debate_fitness": {"fitness": 0.6}}},
    ]
    out = idea_sorting.sort_ideas(ideas, "debate_fitness", descending=True)
    assert [i["title"] for i in out] == ["B", "C", "A"]


def test_sort_by_debate_margin_uses_raw_margin():
    import idea_sorting
    ideas = [
        # Same fitness (σ is bounded) but different raw margins.
        {"title": "A", "execution_meta": {"debate_fitness":
            {"fitness": 0.99, "margin": 5.0}}},
        {"title": "B", "execution_meta": {"debate_fitness":
            {"fitness": 0.99, "margin": 2.0}}},
    ]
    out = idea_sorting.sort_ideas(ideas, "debate_margin", descending=True)
    assert [i["title"] for i in out] == ["A", "B"]


def test_new_debate_modes_in_registry():
    import idea_sorting
    assert "debate_fitness" in idea_sorting.SORT_MODES
    assert "debate_margin" in idea_sorting.SORT_MODES
    assert "debate_fitness" in idea_sorting.DIRECTIONAL_MODES
    assert "debate_margin" in idea_sorting.DIRECTIONAL_MODES


# ── Disk cache ─────────────────────────────────────────────────────────────

def test_hash_key_deterministic():
    idea = {"title": "T", "hypothesis": "H"}
    assert df._hash_key(idea, 2, 2.0) == df._hash_key(idea, 2, 2.0)


def test_hash_key_changes_with_each_dimension():
    idea = {"title": "T", "hypothesis": "H"}
    base = df._hash_key(idea, 2, 2.0)
    assert df._hash_key({"title": "T", "hypothesis": "H'"}, 2, 2.0) != base
    assert df._hash_key(idea, 3, 2.0) != base
    assert df._hash_key(idea, 2, 1.5) != base


def test_disk_cache_roundtrip(tmp_path):
    """First call computes; second call hits cache → zero new LLM calls."""
    cli = _mock_client([
        "p", "c", '{"proposer_score": 0.7, "critic_score": 0.3}',
    ])
    idea = {"title": "T", "hypothesis": "H"}
    cache_dir = str(tmp_path / "dcache")
    payload1 = df.cached_compute_fitness_for_idea(
        idea, llm_client=cli, n_rounds=1, tau=2.0,
        cache_dir=cache_dir,
    )
    assert cli.call.call_count == 3

    idea2 = {"title": "T", "hypothesis": "H"}  # same identity
    payload2 = df.cached_compute_fitness_for_idea(
        idea2, llm_client=cli, n_rounds=1, tau=2.0,
        cache_dir=cache_dir,
    )
    # No new calls — cache hit.
    assert cli.call.call_count == 3
    assert payload2["fitness"] == payload1["fitness"]
    assert idea2["execution_meta"]["debate_fitness"] == payload1


def test_disk_cache_separate_keys_for_different_hypotheses(tmp_path):
    cli = _mock_client(["x"] * 100)
    cache_dir = str(tmp_path / "dkeys")
    df.cached_compute_fitness_for_idea(
        {"title": "T", "hypothesis": "A"}, llm_client=cli, n_rounds=1,
        cache_dir=cache_dir,
    )
    n_a = cli.call.call_count
    df.cached_compute_fitness_for_idea(
        {"title": "T", "hypothesis": "B"}, llm_client=cli, n_rounds=1,
        cache_dir=cache_dir,
    )
    assert cli.call.call_count > n_a
