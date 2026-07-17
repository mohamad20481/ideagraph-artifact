"""Tests for agents.execution_revisor — tiny-experiment proxy + Bayesian credit."""
from __future__ import annotations
import json

import pytest

from agents.execution_revisor import (
    revise,
    bayesian_blend,
    blend_score,
    RevisionResult,
    clear_cache,
    cache_size,
    DEFAULT_SAMPLE_SIZE,
    DEFAULT_N_SEEDS,
    DEFAULT_TARGET_SAMPLE_SIZE,
)
from models.idea import Idea


# ── Test fixtures ────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clear_cache():
    clear_cache()
    yield
    clear_cache()


def _make_idea(**overrides):
    base = dict(
        title="Graph Neural Networks for Drug Discovery",
        motivation="Drug screening at scale is expensive.",
        method="Train message-passing networks on molecular graphs over ZINC.",
        hypothesis="GNNs outperform fingerprint baselines on ADMET tasks.",
        resources="1 A100, 24 hours, 200K molecules",
        expected_outcome="15% AUROC improvement over Morgan fingerprints",
        risk_assessment="Dataset bias risk; overfitting at low data regimes.",
    )
    base.update(overrides)
    idea = Idea(**base)
    idea.quality_score = overrides.get("quality_score", 0.65)
    idea.probe_passed = True
    idea.probe_scores = {"code": 0.7, "dataset": 0.6, "novelty": 0.7,
                          "specificity": 0.5, "scalability": 0.5}
    return idea


class _FakeOK:
    """Stand-in for ClaudeResponse with success=True."""
    def __init__(self, text, cost_usd=0.001):
        self.success = True
        self.text = text
        self.cost_usd = cost_usd
        self.error = None
        self.input_tokens = 100
        self.output_tokens = 60


class _FakeFail:
    """Stand-in for ClaudeResponse with success=False."""
    success = False
    text = ""
    error = "rate limit"
    cost_usd = 0.0


class _Client:
    def __init__(self, response):
        self._r = response
        self.calls = []

    def call(self, system, user, **kwargs):
        self.calls.append({"system": system, "user": user, **kwargs})
        return self._r


# ── Bayesian math ────────────────────────────────────────────────────────

class TestBayesianBlend:
    def test_returns_tuple(self):
        m, t = bayesian_blend(0.7, 0.4)
        assert isinstance(m, float) and isinstance(t, float)
        assert 0.0 <= m <= 1.0 and 0.0 <= t <= 1.0

    def test_tiny_proxy_has_modest_trust(self):
        # 1k samples × 1 seed should produce visible but minority trust
        m, t = bayesian_blend(0.7, 0.4, n_samples=1000, n_seeds=1)
        assert 0.10 < t < 0.50, f"trust at 1k×1 = {t:.3f}, expected 0.10-0.50"

    def test_full_scale_gets_near_full_trust(self):
        # 100k × 5 seeds should be near-fully trusted
        m, t = bayesian_blend(0.7, 0.4, n_samples=100_000, n_seeds=5)
        assert t > 0.95, f"trust at 100k×5 = {t:.3f}, expected >0.95"

    def test_trust_increases_with_samples(self):
        _, t1k = bayesian_blend(0.7, 0.4, n_samples=1_000, n_seeds=1)
        _, t10k = bayesian_blend(0.7, 0.4, n_samples=10_000, n_seeds=1)
        _, t100k = bayesian_blend(0.7, 0.4, n_samples=100_000, n_seeds=1)
        assert t1k < t10k < t100k

    def test_trust_increases_with_seeds(self):
        _, t1 = bayesian_blend(0.7, 0.4, n_samples=10_000, n_seeds=1)
        _, t5 = bayesian_blend(0.7, 0.4, n_samples=10_000, n_seeds=5)
        assert t5 > t1

    def test_blend_lies_between_inputs(self):
        # The posterior must always be between the two priors
        m, _ = bayesian_blend(0.8, 0.3, n_samples=10_000, n_seeds=1)
        assert 0.3 <= m <= 0.8

    def test_clamps_inputs_to_unit_interval(self):
        m1, _ = bayesian_blend(-0.5, 1.5)
        m2, _ = bayesian_blend(0.0, 1.0)
        # Out-of-range inputs get clipped, so result equals well-formed call
        assert m1 == m2


class TestLinearBlend:
    def test_endpoints(self):
        # trust=0 → all probe; trust=1 → all exec
        assert blend_score(0.7, 0.3, trust=0.0) == pytest.approx(0.7)
        assert blend_score(0.7, 0.3, trust=1.0) == pytest.approx(0.3)

    def test_midpoint(self):
        assert blend_score(0.8, 0.2, trust=0.5) == pytest.approx(0.5)

    def test_clamps_trust(self):
        # Out-of-range trust gets clipped to [0,1]
        assert blend_score(0.7, 0.3, trust=2.0) == pytest.approx(0.3)
        assert blend_score(0.7, 0.3, trust=-1.0) == pytest.approx(0.7)


# ── revise() with no LLM client (degraded path) ──────────────────────────

class TestDegradedPath:
    def test_returns_successful_result_without_client(self):
        idea = _make_idea(quality_score=0.65)
        r = revise(idea, claude_client=None, use_cache=False)
        assert r.success
        assert r.used_llm is False
        assert "degraded" in (r.error or "").lower()

    def test_degraded_blend_is_finite_and_bounded(self):
        idea = _make_idea(quality_score=0.65)
        r = revise(idea, claude_client=None, use_cache=False)
        assert 0.0 <= r.execution_signal <= 1.0
        assert 0.0 <= r.blended_quality <= 1.0
        assert 0.0 <= r.trust_weight <= 1.0

    def test_weak_execution_probes_lower_signal(self):
        # Two ideas with same probe quality but different execution-relevant
        # probe weakness should produce different exec signals
        good = _make_idea(quality_score=0.65)
        good.probe_scores = {"code": 0.8, "dataset": 0.8, "scalability": 0.8,
                              "specificity": 0.8, "constraint": 0.8}
        weak = _make_idea(quality_score=0.65)
        weak.probe_scores = {"code": 0.3, "dataset": 0.3, "scalability": 0.3,
                              "specificity": 0.3, "constraint": 0.3}
        rg = revise(good, claude_client=None, use_cache=False)
        rw = revise(weak, claude_client=None, use_cache=False)
        assert rg.execution_signal > rw.execution_signal


# ── revise() with mocked LLM ─────────────────────────────────────────────

class TestLLMPath:
    def _good_response(self, exec_signal=0.55):
        return _FakeOK(json.dumps({
            "metric_name": "AUROC",
            "predicted_metric": 0.78,
            "ci_low": 0.71, "ci_high": 0.84,
            "execution_signal": exec_signal,
            "failure_modes": ["data leakage", "small batch noise"],
            "would_scale": True,
            "rationale": "plausible at small scale",
        }))

    def test_parses_well_formed_response(self):
        idea = _make_idea(quality_score=0.65)
        client = _Client(self._good_response())
        r = revise(idea, claude_client=client, use_cache=False)
        assert r.success and r.used_llm
        assert r.metric_name == "AUROC"
        assert r.predicted_metric == pytest.approx(0.78)
        assert r.confidence_interval == (0.71, 0.84)
        assert r.execution_signal == pytest.approx(0.55)
        assert r.failure_modes == ["data leakage", "small batch noise"]

    def test_passes_sample_config_to_user_prompt(self):
        idea = _make_idea()
        client = _Client(self._good_response())
        revise(idea, claude_client=client, n_samples=5_000, n_seeds=3,
                use_cache=False)
        user = client.calls[0]["user"]
        assert "5000 samples" in user.replace(",", "") or "5,000 samples" in user
        assert "3 random seed" in user

    def test_fenced_response_is_parsed(self):
        idea = _make_idea()
        wrapped = "```json\n" + json.dumps({
            "metric_name": "F1", "predicted_metric": 0.6,
            "ci_low": 0.55, "ci_high": 0.65, "execution_signal": 0.5,
            "failure_modes": ["x"], "would_scale": False, "rationale": "ok",
        }) + "\n```"
        client = _Client(_FakeOK(wrapped))
        r = revise(idea, claude_client=client, use_cache=False)
        assert r.success and r.metric_name == "F1"

    def test_falls_back_when_call_fails(self):
        idea = _make_idea()
        client = _Client(_FakeFail())
        r = revise(idea, claude_client=client, use_cache=False)
        assert r.success
        assert r.used_llm is False
        assert "rate limit" in (r.error or "").lower() or "failed" in (r.error or "").lower()

    def test_falls_back_when_response_is_garbage(self):
        idea = _make_idea()
        client = _Client(_FakeOK("totally not json"))
        r = revise(idea, claude_client=client, use_cache=False)
        assert r.success and r.used_llm is False


# ── Caching ──────────────────────────────────────────────────────────────

class TestCaching:
    def test_cache_returns_same_object_on_second_call(self):
        idea = _make_idea()
        client = _Client(_FakeOK(json.dumps({
            "metric_name": "AUROC", "predicted_metric": 0.7,
            "ci_low": 0.65, "ci_high": 0.75,
            "execution_signal": 0.6, "failure_modes": [],
            "would_scale": True, "rationale": "x",
        })))
        clear_cache()
        r1 = revise(idea, claude_client=client, use_cache=True)
        r2 = revise(idea, claude_client=client, use_cache=True)
        assert r1 is r2  # cache hit returns same instance
        assert len(client.calls) == 1

    def test_use_cache_false_bypasses(self):
        idea = _make_idea()
        client = _Client(_FakeOK(json.dumps({
            "metric_name": "AUROC", "predicted_metric": 0.7,
            "ci_low": 0.65, "ci_high": 0.75,
            "execution_signal": 0.6, "failure_modes": [],
            "would_scale": True, "rationale": "x",
        })))
        clear_cache()
        revise(idea, claude_client=client, use_cache=False)
        revise(idea, claude_client=client, use_cache=False)
        assert len(client.calls) == 2

    def test_different_sample_sizes_dont_collide(self):
        idea = _make_idea()
        client = _Client(_FakeOK(json.dumps({
            "metric_name": "AUROC", "predicted_metric": 0.7,
            "ci_low": 0.65, "ci_high": 0.75,
            "execution_signal": 0.6, "failure_modes": [],
            "would_scale": True, "rationale": "x",
        })))
        clear_cache()
        revise(idea, claude_client=client, n_samples=1_000)
        revise(idea, claude_client=client, n_samples=10_000)
        assert len(client.calls) == 2  # different cache keys


# ── RevisionResult shape ─────────────────────────────────────────────────

class TestRevisionResult:
    def test_to_dict_round_trip(self):
        r = RevisionResult(probe_quality=0.7, execution_signal=0.4,
                            blended_quality=0.6, trust_weight=0.3,
                            metric_name="AUROC", predicted_metric=0.78,
                            confidence_interval=(0.7, 0.85))
        d = r.to_dict()
        assert d["probe_quality"] == 0.7
        assert d["confidence_interval"] == [0.7, 0.85]

    def test_summary_contains_arrow_and_numbers(self):
        r = RevisionResult(probe_quality=0.7, execution_signal=0.5,
                            blended_quality=0.62, trust_weight=0.3, delta=-0.08)
        s = r.summary()
        assert "0.70" in s and "0.62" in s and "↓" in s


# ── Idea dataclass integration ───────────────────────────────────────────

class TestIdeaIntegration:
    def test_new_fields_default_to_none(self):
        idea = _make_idea()
        assert idea.execution_signal is None
        assert idea.execution_trust is None
        assert idea.execution_delta is None
        assert idea.execution_meta is None
        assert idea.probe_quality is None

    def test_to_dict_includes_new_fields(self):
        idea = _make_idea()
        idea.execution_signal = 0.55
        idea.execution_trust = 0.3
        idea.probe_quality = 0.65
        d = idea.to_dict()
        assert d["execution_signal"] == 0.55
        assert d["execution_trust"] == 0.3
        assert d["probe_quality"] == 0.65

    def test_revise_accepts_idea_dataclass(self):
        idea = _make_idea(quality_score=0.65)
        r = revise(idea, claude_client=None, use_cache=False)
        assert r.success and r.probe_quality == pytest.approx(0.65)


# ── Integration into pipeline (gated by feature flag) ────────────────────

class TestPipelineGating:
    def test_feature_flag_exists(self):
        import config as _config
        assert hasattr(_config, "ENABLE_EXECUTION_REVISION")
        assert hasattr(_config, "EXECUTION_REVISION_SAMPLE_SIZE")
        assert hasattr(_config, "EXECUTION_REVISION_N_SEEDS")

    def test_sample_size_default_matches_module(self):
        import config as _config
        # The module default and config default should agree, otherwise the
        # UI's "trust weight" preview will be misleading
        assert _config.EXECUTION_REVISION_SAMPLE_SIZE == DEFAULT_SAMPLE_SIZE
        assert _config.EXECUTION_REVISION_N_SEEDS == DEFAULT_N_SEEDS

    def test_pipeline_imports_revisor_module(self):
        # Module is importable from the pipeline's vantage point
        from agents.execution_revisor import revise as _r  # noqa: F401
