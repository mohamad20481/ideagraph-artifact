"""Tests for speed_optimizer.py — routing, concurrency, shortcuts, presets, matrix."""
import pytest
from speed_optimizer import (
    route_for_stage, STAGE_TIERS, TIER_MODELS,
    AdaptiveConcurrency, get_adaptive_concurrency,
    quick_probe_shortcut,
    Preset, BUILTIN_PRESETS,
    build_comparison_matrix, COMPARISON_DIMENSIONS,
)


class TestStageRouting:
    def test_known_stages_have_tiers(self):
        for stage in ("probe", "ideation", "review", "default"):
            assert stage in STAGE_TIERS

    def test_probe_routes_to_cheap_tier(self):
        # Tier should be cheap
        assert STAGE_TIERS["probe"] == "cheap"

    def test_route_returns_tuple(self):
        result = route_for_stage("ideation")
        assert isinstance(result, tuple) and len(result) == 2
        provider, model = result
        assert isinstance(provider, str) and isinstance(model, str)

    def test_unknown_stage_falls_back_to_default(self):
        result = route_for_stage("totally_made_up_stage")
        # Should not crash; should return some valid (provider, model)
        assert result is not None and len(result) == 2

    def test_tier_models_have_all_tiers(self):
        for tier in ("cheap", "balanced", "premium"):
            assert tier in TIER_MODELS
            assert len(TIER_MODELS[tier]) >= 2


class TestAdaptiveConcurrency:
    def test_initial_state(self):
        ac = AdaptiveConcurrency(default_workers=4)
        assert ac.recommended_workers == 4

    def test_high_error_rate_reduces_workers(self):
        ac = AdaptiveConcurrency(min_workers=1, max_workers=8, default_workers=6)
        # Inject many errors
        for _ in range(20):
            ac.record_call(duration_s=10, success=False)
        assert ac.recommended_workers == ac.min_workers

    def test_fast_healthy_calls_scale_up(self):
        ac = AdaptiveConcurrency(min_workers=1, max_workers=8, default_workers=2)
        for _ in range(10):
            ac.record_call(duration_s=2.0, success=True)
        # Should scale up at least once
        assert ac.recommended_workers >= 3

    def test_slow_calls_back_off(self):
        ac = AdaptiveConcurrency(min_workers=1, max_workers=8, default_workers=6)
        for _ in range(10):
            ac.record_call(duration_s=60, success=True)
        assert ac.recommended_workers < 6

    def test_stats_returns_dict(self):
        ac = AdaptiveConcurrency()
        ac.record_call(5.0, True)
        s = ac.stats()
        assert "current_workers" in s
        assert "avg_latency_s" in s
        assert "error_rate" in s

    def test_singleton_returns_same_instance(self):
        a = get_adaptive_concurrency()
        b = get_adaptive_concurrency()
        assert a is b


class TestProbeShortcut:
    def test_short_method_rejected(self):
        result = quick_probe_shortcut({"title": "X", "method": "x" * 10, "hypothesis": "h" * 20})
        assert result is not None
        assert result["quality"] < 0.3
        assert result["_shortcut"] is True

    def test_missing_hypothesis_rejected(self):
        result = quick_probe_shortcut({
            "title": "Reasonable Title",
            "method": "A long enough method description with multiple terms and content",
            "hypothesis": "x",
        })
        assert result is not None
        assert "Hypothesis" in result["feedback"]

    def test_decent_idea_proceeds_to_real_probe(self):
        result = quick_probe_shortcut({
            "title": "Graph Neural Networks for Drug Discovery",
            "method": "Use message-passing neural networks on molecular graphs to predict ADMET properties",
            "hypothesis": "GNNs outperform fingerprint-based ML on standard benchmarks",
        })
        assert result is None  # don't shortcut — let real probe decide


class TestPresets:
    def test_builtin_presets_exist(self):
        assert len(BUILTIN_PRESETS) >= 3
        for p in BUILTIN_PRESETS:
            assert p.name and p.topic
            assert 0 <= p.creativity <= 1
            assert p.iterations > 0
            assert p.budget_usd > 0

    def test_preset_to_dict_roundtrip(self):
        original = Preset(name="Test", topic="ML",
                           creativity=0.85, time_weeks=24, risk="high")
        d = original.to_dict()
        restored = Preset.from_dict(d)
        assert restored.name == "Test"
        assert restored.creativity == 0.85
        assert restored.risk == "high"

    def test_preset_from_dict_ignores_unknown_keys(self):
        d = {"name": "X", "topic": "y", "completely_unknown_field": 42}
        p = Preset.from_dict(d)
        assert p.name == "X"
        # Unknown field should be silently dropped without crashing


class TestComparisonMatrix:
    def test_returns_required_fields(self):
        ideas = [
            {"title": "A", "quality_score": 0.8, "probe_scores": {"code": 0.7}},
            {"title": "B", "quality_score": 0.5, "probe_scores": {"code": 0.5}},
        ]
        m = build_comparison_matrix(ideas)
        assert "headers" in m and "rows" in m and "winners" in m and "summary" in m

    def test_picks_higher_quality_as_winner(self):
        ideas = [
            {"title": "Low", "quality_score": 0.3, "probe_scores": {"code": 0.3, "novelty": 0.3}},
            {"title": "High", "quality_score": 0.9, "probe_scores": {"code": 0.9, "novelty": 0.8}},
        ]
        m = build_comparison_matrix(ideas)
        # "High" should win on most dimensions
        assert m["summary"]["wins_per_idea"][1] > m["summary"]["wins_per_idea"][0]

    def test_caps_at_5_ideas(self):
        ideas = [{"title": f"Idea {i}", "quality_score": 0.5} for i in range(10)]
        m = build_comparison_matrix(ideas)
        assert m["summary"]["total_ideas"] == 5

    def test_empty_input(self):
        m = build_comparison_matrix([])
        assert m["headers"] == [] and m["rows"] == []

    def test_handles_missing_probe_scores(self):
        ideas = [
            {"title": "A", "quality_score": 0.5},  # no probe_scores
            {"title": "B", "quality_score": 0.6},
        ]
        m = build_comparison_matrix(ideas)
        # Should not crash
        assert len(m["rows"]) == len(COMPARISON_DIMENSIONS)
