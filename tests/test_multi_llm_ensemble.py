"""Tests for multi_llm_ensemble.py — parallel cross-provider ideation."""
from __future__ import annotations
import json

import pytest

import config
from multi_llm_ensemble import (
    EnsembleResult,
    available_providers,
    ensemble_generate,
    _idea_tokens,
    _jaccard,
    _diversity_filter,
    _ProviderClient,
    _ProviderResponse,
    _dict_to_idea,
    _parse_json,
)
from models.idea import Idea


def _mk(title, method, methodology_type="empirical_study",
         novelty_level="moderate"):
    i = Idea(
        title=title, motivation="m", method=method, hypothesis="h",
        resources="r", expected_outcome="e", risk_assessment="r",
    )
    i.methodology_type = methodology_type
    i.novelty_level = novelty_level
    return i


# ── Jaccard primitives ───────────────────────────────────────────────────

class TestJaccard:
    def test_identical_sets_one(self):
        s = {"a", "b", "c"}
        assert _jaccard(s, s) == 1.0

    def test_disjoint_zero(self):
        assert _jaccard({"a"}, {"b"}) == 0.0

    def test_empty_returns_zero(self):
        assert _jaccard(set(), set()) == 0.0
        assert _jaccard({"a"}, set()) == 0.0

    def test_overlap(self):
        # 2 common / 4 union = 0.5
        assert _jaccard({"a", "b", "c"}, {"b", "c", "d"}) == pytest.approx(0.5)


class TestIdeaTokens:
    def test_extracts_content_words(self):
        idea = _mk("Graph Neural Network", "message-passing on molecules")
        toks = _idea_tokens(idea)
        # Stopwords removed; short words removed; lowercased
        assert "graph" in toks and "neural" in toks
        assert "the" not in toks and "on" not in toks

    def test_handles_dict_input(self):
        d = {"title": "GNN attention",
             "method": "use attention with graphs",
             "hypothesis": "h"}
        toks = _idea_tokens(d)
        assert "attention" in toks


# ── Diversity filter ─────────────────────────────────────────────────────

class TestDiversityFilter:
    def test_keeps_distinct_ideas(self):
        ideas = [
            (_mk("Graph attention networks for chemistry",
                 "use graph attention transformers for molecular property prediction"),
             "deepseek"),
            (_mk("Normalizing flows for molecule generation",
                 "train flow models on protein binding pocket conditioning"),
             "kimi"),
            (_mk("Geometric deep learning equivariant approach",
                 "SE3 equivariant networks with rotation invariant features"),
             "anthropic"),
        ]
        kept, rejected = _diversity_filter(ideas, similarity_threshold=0.5)
        assert len(kept) == 3
        assert len(rejected) == 0

    def test_filters_near_duplicates(self):
        ideas = [
            (_mk("Graph attention networks for drug",
                 "message-passing neural network with multi-head attention molecular graphs"),
             "deepseek"),
            (_mk("Attention-based MPNN drug",
                 "message-passing neural network with multi-head attention molecular graphs"),
             "kimi"),  # near-duplicate
        ]
        kept, rejected = _diversity_filter(ideas, similarity_threshold=0.4)
        assert len(kept) == 1
        assert len(rejected) == 1
        assert rejected[0]["similarity"] > 0.4

    def test_rejected_pair_payload_shape(self):
        ideas = [
            (_mk("AAA aaa", "alpha alpha alpha alpha alpha"), "p1"),
            (_mk("AAA aaa", "alpha alpha alpha alpha alpha"), "p2"),
        ]
        _, rejected = _diversity_filter(ideas, similarity_threshold=0.5)
        assert len(rejected) == 1
        r = rejected[0]
        assert r["kept_title"] and r["rejected_title"]
        assert r["rejected_provider"] == "p2"
        assert 0 <= r["similarity"] <= 1

    def test_keep_order_is_first_in_first_kept(self):
        # Earlier idea in the list takes priority on near-duplicate
        ideas = [
            (_mk("First", "alpha beta gamma delta"), "p1"),
            (_mk("Second", "alpha beta gamma delta"), "p2"),
        ]
        kept, rejected = _diversity_filter(ideas, similarity_threshold=0.5)
        assert kept[0][0].title == "First"
        assert rejected[0]["rejected_title"] == "Second"


# ── Provider client adapter ──────────────────────────────────────────────

class TestProviderClient:
    def test_construction_stores_provider_and_model(self):
        c = _ProviderClient("kimi", "moonshot-v1-32k")
        assert c.provider == "kimi" and c.model == "moonshot-v1-32k"

    def test_call_with_unconfigured_provider_returns_failure(self):
        # Patch config.PROVIDER's keys to be empty
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(config, "DEEPSEEK_API_KEY", "")
            mp.setattr(config, "KIMI_API_KEY", "")
            mp.setattr(config, "ANTHROPIC_API_KEY", "")
            mp.setattr(config, "OPENAI_API_KEY", "")
            mp.setattr(config, "GEMINI_API_KEY", "")
            mp.setattr(config, "AZURE_API_KEY", "")
            mp.setattr(config, "GROQ_API_KEY", "")
            c = _ProviderClient("deepseek", "deepseek-chat")
            r = c.call("system", "user", max_tokens=10)
            # Without a valid client, real .chat call should fail cleanly.
            # We assert the function returns a _ProviderResponse without
            # raising; success may be False.
            assert isinstance(r, _ProviderResponse)


# ── Dict → Idea ─────────────────────────────────────────────────────────

class TestDictToIdea:
    def test_valid_dict_produces_idea(self):
        d = {
            "title": "Test", "motivation": "m", "method": "x",
            "hypothesis": "h", "resources": "r",
            "expected_outcome": "e", "risk_assessment": "r",
            "methodology_type": "system_design",
            "novelty_level": "moderate",
        }
        idea = _dict_to_idea(d, "kimi", "moonshot-v1-32k", "topic X")
        assert idea is not None
        assert idea.source_strategy == "E"
        assert idea.generation == 0
        meta = idea.execution_meta or {}
        assert meta["ensemble_provider"] == "kimi"
        assert meta["ensemble_model"] == "moonshot-v1-32k"
        assert meta["topic"] == "topic X"
        assert meta["regen_mode"] == "multi_llm_ensemble"

    def test_missing_required_returns_none(self):
        assert _dict_to_idea({"title": "x"}, "p", "m", "topic") is None

    def test_invalid_methodology_normalized(self):
        d = {"title": "T", "method": "x", "hypothesis": "h",
             "motivation": "m", "resources": "r", "expected_outcome": "e",
             "risk_assessment": "r",
             "methodology_type": "not_real_method",
             "novelty_level": "moderate"}
        idea = _dict_to_idea(d, "p", "m", "topic")
        assert idea.methodology_type is None


# ── available_providers / config gating ──────────────────────────────────

class TestAvailableProviders:
    def test_returns_only_providers_with_keys(self):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(config, "DEEPSEEK_API_KEY", "sk-real-deepseek")
            mp.setattr(config, "KIMI_API_KEY", "sk-real-kimi")
            mp.setattr(config, "ANTHROPIC_API_KEY", "")
            mp.setattr(config, "OPENAI_API_KEY", "")
            mp.setattr(config, "GEMINI_API_KEY", "")
            mp.setattr(config, "AZURE_API_KEY", "")
            mp.setattr(config, "GROQ_API_KEY", "")
            out = available_providers()
            assert set(out) == {"deepseek", "kimi"}

    def test_placeholder_keys_excluded(self):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(config, "DEEPSEEK_API_KEY", "sk-xxx-placeholder")
            mp.setattr(config, "KIMI_API_KEY", "your-key-here")
            mp.setattr(config, "ANTHROPIC_API_KEY", "")
            mp.setattr(config, "OPENAI_API_KEY", "")
            mp.setattr(config, "GEMINI_API_KEY", "")
            mp.setattr(config, "AZURE_API_KEY", "")
            mp.setattr(config, "GROQ_API_KEY", "")
            assert available_providers() == []


# ── ensemble_generate ───────────────────────────────────────────────────

class TestEnsembleGenerate:
    def test_empty_topic_raises(self):
        with pytest.raises(ValueError, match="non-empty topic"):
            ensemble_generate("", providers=["deepseek"])

    def test_no_providers_returns_empty_result(self):
        result = ensemble_generate("topic", providers=[], n_per_provider=1)
        assert isinstance(result, EnsembleResult)
        assert result.kept_ideas == []
        assert result.all_ideas == []
        assert result.topic == "topic"

    def test_n_zero_returns_empty(self):
        result = ensemble_generate(
            "topic", providers=["deepseek"], n_per_provider=0,
        )
        assert result.kept_ideas == []

    def test_result_topic_stripped(self):
        result = ensemble_generate("  spaced  ", providers=[])
        assert result.topic == "spaced"


# ── App wiring ──────────────────────────────────────────────────────────

class TestAppWiring:
    def test_app_imports_ensemble(self):
        with open("app.py", encoding="utf-8") as f:
            src = f.read()
        assert "from multi_llm_ensemble import" in src
        assert "Multi-LLM Ensemble" in src
