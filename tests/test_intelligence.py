"""
Tests for intelligence.py — the smart layer.
"""
import pytest

from intelligence import (
    SemanticNoveltyChecker,
    explain_quality,
    pareto_dominates,
    should_replace_pareto,
    FailurePatternTracker,
    advise_strategy,
    CrossRunMemory,
    IdeaRecommender,
    compute_maturity,
    predict_run_cost,
    ClaimVerifier,
    MATURITY_EMOJI,
)


class TestSemanticNoveltyChecker:
    def test_identical_methods_rejected(self):
        c = SemanticNoveltyChecker(threshold=0.5)
        c.register("Idea A", "use graph neural networks to predict molecular properties")
        is_novel, _, sim = c.check("use graph neural networks to predict molecular properties")
        assert not is_novel
        assert sim >= 0.5

    def test_different_methods_accepted(self):
        c = SemanticNoveltyChecker(threshold=0.5)
        c.register("Idea A", "use graph neural networks to predict molecular properties")
        is_novel, _, sim = c.check("reinforcement learning for robotic arm control in simulation")
        assert is_novel
        assert sim < 0.3

    def test_similar_but_distinct_accepted(self):
        c = SemanticNoveltyChecker(threshold=0.55)
        c.register("Idea A", "graph attention networks for drug toxicity prediction on ZINC dataset")
        is_novel, similar, sim = c.check(
            "graph transformer architecture for protein-ligand binding affinity on PDBBind dataset"
        )
        # Different enough method despite same domain
        assert is_novel or sim < 0.55

    def test_returns_most_similar_title(self):
        c = SemanticNoveltyChecker(threshold=0.5)
        c.register("Alpha", "method alpha using technique one")
        c.register("Beta", "method beta using technique two")
        _, similar, _ = c.check("method alpha using technique one modified")
        assert similar == "Alpha"

    def test_clear_resets(self):
        c = SemanticNoveltyChecker(threshold=0.5)
        c.register("A", "graph neural networks molecular prediction")
        c.clear()
        is_novel, _, _ = c.check("graph neural networks molecular prediction")
        assert is_novel  # no archived ideas after clear


class TestQualityExplainer:
    def test_low_quality_explanation(self):
        text = explain_quality({"code": 0.2, "dataset": 0.3, "novelty": 0.5}, "Test")
        assert "needs" in text.lower() or "revision" in text.lower()
        assert "code" in text.lower()

    def test_high_quality_explanation(self):
        text = explain_quality({"code": 0.9, "dataset": 0.8, "novelty": 0.7}, "Test")
        assert "solid" in text.lower()

    def test_empty_scores(self):
        text = explain_quality({})
        assert "No probe" in text

    def test_mentions_weakest_dimension(self):
        text = explain_quality({"code": 0.1, "dataset": 0.9, "novelty": 0.9}, "Test")
        assert "code" in text.lower()


class TestParetoArchive:
    def test_pareto_dominates_all_better(self):
        assert pareto_dominates(
            {"quality": 0.8, "novelty": 0.7, "feasibility": 0.6},
            {"quality": 0.5, "novelty": 0.5, "feasibility": 0.5},
        )

    def test_pareto_not_dominates_if_one_worse(self):
        assert not pareto_dominates(
            {"quality": 0.8, "novelty": 0.3, "feasibility": 0.6},
            {"quality": 0.5, "novelty": 0.5, "feasibility": 0.5},
        )

    def test_pareto_not_dominates_if_equal(self):
        assert not pareto_dominates(
            {"quality": 0.5, "novelty": 0.5, "feasibility": 0.5},
            {"quality": 0.5, "novelty": 0.5, "feasibility": 0.5},
        )

    def test_should_replace_pareto_with_idea_objects(self):
        from unittest.mock import MagicMock
        new = MagicMock()
        new.quality_score = 0.7
        new.probe_scores = {"novelty": 0.8, "code": 0.6, "dataset": 0.7, "constraint": 0.5}
        old = MagicMock()
        old.quality_score = 0.5
        old.probe_scores = {"novelty": 0.4, "code": 0.5, "dataset": 0.5, "constraint": 0.5}
        assert should_replace_pareto(new, old)


class TestFailurePatternTracker:
    def test_no_hint_before_threshold(self):
        t = FailurePatternTracker()
        t.record_failure(0, 0, "code")
        t.record_failure(0, 0, "code")
        assert t.get_mitigations(0, 0) == ""

    def test_hint_after_threshold(self):
        t = FailurePatternTracker()
        for _ in range(3):
            t.record_failure(0, 0, "dataset")
        hint = t.get_mitigations(0, 0)
        assert "benchmark" in hint.lower() or "publicly" in hint.lower()

    def test_success_reduces_count(self):
        t = FailurePatternTracker()
        for _ in range(3):
            t.record_failure(1, 1, "novelty")
        t.record_success(1, 1)
        assert t.get_mitigations(1, 1) == ""

    def test_different_cells_independent(self):
        t = FailurePatternTracker()
        for _ in range(3):
            t.record_failure(0, 0, "code")
        assert t.get_mitigations(0, 0) != ""
        assert t.get_mitigations(1, 1) == ""


class TestStrategyAdvisor:
    def test_sparse_dag(self):
        advice = advise_strategy(2, 1, 1, 0)
        assert "sparse" in advice.lower() or "Strategy A" in advice

    def test_rich_dag(self):
        advice = advise_strategy(20, 4, 35, 3)
        assert "B" in advice or "C" in advice


class TestCrossRunMemory:
    def test_extract_lessons_empty(self):
        mem = CrossRunMemory()
        lessons = mem.extract_lessons_from_run({"ideas": []})
        assert any("no ideas" in l.lower() for l in lessons)

    def test_extract_lessons_with_ideas(self):
        mem = CrossRunMemory()
        lessons = mem.extract_lessons_from_run({
            "ideas": [
                {"methodology_type": "empirical_study", "quality_score": 0.7,
                 "probe_scores": {"code": 0.8, "dataset": 0.3}},
                {"methodology_type": "empirical_study", "quality_score": 0.6,
                 "probe_scores": {"code": 0.7, "dataset": 0.2}},
            ],
            "coverage": 0.3,
        })
        assert len(lessons) >= 2
        assert any("empirical" in l.lower() for l in lessons)
        assert any("dataset" in l.lower() for l in lessons)

    def test_prompt_context_empty_when_no_lessons(self):
        mem = CrossRunMemory()
        assert mem.get_context_for_prompt() == ""

    def test_prompt_context_with_lessons(self):
        mem = CrossRunMemory()
        mem._lessons = ["Use empirical studies", "Avoid vague methods"]
        ctx = mem.get_context_for_prompt()
        assert "empirical" in ctx
        assert "Lessons" in ctx


# ─────────────────────────────────────────────────────────────────────────────
# Idea Recommender
# ─────────────────────────────────────────────────────────────────────────────

class TestIdeaRecommender:
    def test_finds_similar_ideas(self):
        r = IdeaRecommender()
        target = {"title": "GNN for drugs", "method": "graph neural networks molecular property prediction"}
        candidates = [
            {"title": "GNN toxicity", "method": "graph neural networks toxicity prediction ZINC", "quality_score": 0.6},
            {"title": "RL robotics", "method": "reinforcement learning robotic arm control", "quality_score": 0.7},
            {"title": "NLP sentiment", "method": "transformer sentiment analysis reviews", "quality_score": 0.5},
        ]
        recs = r.recommend(target, candidates, n=3)
        assert len(recs) >= 1
        assert recs[0]["title"] == "GNN toxicity"  # most similar

    def test_excludes_self(self):
        r = IdeaRecommender()
        target = {"title": "My Idea", "method": "some method"}
        candidates = [{"title": "My Idea", "method": "some method", "quality_score": 0.5}]
        assert r.recommend(target, candidates) == []

    def test_returns_similarity_and_reason(self):
        r = IdeaRecommender()
        target = {"title": "GNN", "method": "graph neural networks prediction"}
        candidates = [{"title": "GNN2", "method": "graph neural networks classification", "quality_score": 0.5}]
        recs = r.recommend(target, candidates)
        if recs:
            assert "_similarity" in recs[0]
            assert "_recommendation_reason" in recs[0]


# ─────────────────────────────────────────────────────────────────────────────
# Maturity Scoring
# ─────────────────────────────────────────────────────────────────────────────

class TestMaturityScoring:
    def test_sketch_level(self):
        m = compute_maturity({"title": "Just a title"})
        assert m["level"] == 0
        assert m["label"] == "sketch"

    def test_proposal_level(self):
        m = compute_maturity({
            "title": "Full Idea",
            "motivation": "This is important because of reasons that matter a lot",
            "method": "We use a sophisticated transformer architecture with multi-head attention and positional encoding",
            "hypothesis": "We hypothesize that this approach will outperform baselines significantly",
            "resources": "4x A100 GPUs, 1TB storage",
            "expected_outcome": "We expect 5% improvement on benchmark ZINC with statistical significance",
        })
        assert m["level"] == 2
        assert m["label"] == "proposal"

    def test_validated_level(self):
        m = compute_maturity({
            "title": "Validated Idea", "motivation": "Important research problem here",
            "method": "Detailed method with specific algorithms and datasets and metrics",
            "hypothesis": "Testable hypothesis about performance improvement",
            "resources": "Standard compute", "expected_outcome": "Measurable improvement expected",
            "quality_score": 0.65, "probe_passed": True,
        })
        assert m["level"] == 3
        assert m["label"] == "validated"

    def test_progress_increases_with_completeness(self):
        sketch = compute_maturity({"title": "x"})
        full = compute_maturity({
            "title": "x", "motivation": "y" * 30, "method": "z" * 60,
            "hypothesis": "h" * 30, "resources": "r" * 20, "expected_outcome": "o" * 30,
            "quality_score": 0.7, "probe_passed": True,
        })
        assert full["progress_pct"] > sketch["progress_pct"]

    def test_emoji_exists_for_all_levels(self):
        for level in range(6):
            assert level in MATURITY_EMOJI


# ─────────────────────────────────────────────────────────────────────────────
# Cost Predictor
# ─────────────────────────────────────────────────────────────────────────────

class TestCostPredictor:
    def test_returns_required_fields(self):
        p = predict_run_cost(2.0, 10, debate_enabled=False, provider="deepseek")
        assert "estimated_cost_usd" in p
        assert "estimated_minutes" in p
        assert "cost_breakdown" in p
        assert p["estimated_cost_usd"] > 0
        assert p["estimated_minutes"] > 0

    def test_debate_increases_cost(self):
        no_debate = predict_run_cost(5.0, 10, debate_enabled=False)
        with_debate = predict_run_cost(5.0, 10, debate_enabled=True)
        assert with_debate["estimated_cost_usd"] > no_debate["estimated_cost_usd"]

    def test_more_iterations_more_cost(self):
        short = predict_run_cost(5.0, 3)
        long = predict_run_cost(5.0, 20)
        assert long["estimated_cost_usd"] > short["estimated_cost_usd"]

    def test_openai_more_expensive_than_deepseek(self):
        ds = predict_run_cost(5.0, 10, provider="deepseek")
        oai = predict_run_cost(5.0, 10, provider="openai")
        assert oai["estimated_cost_usd"] > ds["estimated_cost_usd"]

    def test_cost_capped_at_budget(self):
        p = predict_run_cost(0.01, 100, provider="openai")  # tiny budget, many iters
        assert p["estimated_cost_usd"] <= 0.01


# ─────────────────────────────────────────────────────────────────────────────
# Claim Verifier
# ─────────────────────────────────────────────────────────────────────────────

class TestClaimVerifier:
    def test_finds_supported_claims(self):
        v = ClaimVerifier()
        idea = {
            "hypothesis": "Graph neural networks outperform traditional methods on molecular datasets.",
            "method": "We use message passing neural networks on ZINC benchmark for property prediction.",
        }
        papers = [
            {"title": "Message Passing Neural Networks for Molecular Property Prediction",
             "abstract": "We show that MPNN outperforms baselines on ZINC and QM9 molecular benchmarks."},
        ]
        results = v.verify_against_dag(idea, papers)
        assert any(r["status"] == "supported" for r in results)

    def test_marks_unverified_when_no_papers(self):
        v = ClaimVerifier()
        idea = {"hypothesis": "Quantum computing will revolutionize drug discovery.", "method": ""}
        results = v.verify_against_dag(idea, [])
        for r in results:
            assert r["status"] == "unverified"

    def test_extracts_claims_from_method(self):
        v = ClaimVerifier()
        claims = v._extract_claims({
            "hypothesis": "This approach improves accuracy by ten percent on standard benchmarks.",
            "method": "We combine attention mechanisms with graph convolutions for better representation learning.",
        })
        assert len(claims) >= 1
        assert all(len(c) > 20 for c in claims)
