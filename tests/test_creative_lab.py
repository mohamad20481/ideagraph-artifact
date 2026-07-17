"""Tests for creative_lab.py — 10 creative features."""
import pytest
from creative_lab import (
    MUTATION_TYPES, _fallback_mutation, generate_blind_review,
    generate_manifesto, spin_roulette, generate_serendipity_mashups,
    build_evolution_replay, compute_personality, predict_trending,
    get_current_olympic_dimension, compute_olympic_score,
    find_collaborators, PERSONALITY_DIMENSIONS,
)

_SAMPLE_IDEA = {
    "title": "GNN for Drug Discovery",
    "method": "Graph neural networks for molecular property prediction",
    "hypothesis": "GNNs outperform traditional ML on molecular tasks",
    "resources": "4x A100, ZINC dataset",
    "expected_outcome": "10% improvement on benchmarks",
    "methodology_type": "empirical_study",
    "novelty_level": "moderate",
    "quality_score": 0.65,
    "probe_scores": {"code": 0.7, "dataset": 0.8, "novelty": 0.6, "specificity": 0.5},
}


class TestMutationLab:
    def test_all_mutation_types_exist(self):
        assert len(MUTATION_TYPES) == 5
        ids = {m["id"] for m in MUTATION_TYPES}
        assert "flip_hypothesis" in ids
        assert "go_wild" in ids

    @pytest.mark.parametrize("mut_type", [m["id"] for m in MUTATION_TYPES])
    def test_fallback_mutation_returns_valid(self, mut_type):
        result = _fallback_mutation(_SAMPLE_IDEA, mut_type)
        assert result is not None
        assert result.get("title") != _SAMPLE_IDEA["title"]
        assert result.get("_mutation_type") == mut_type

    def test_mutation_preserves_original_ref(self):
        result = _fallback_mutation(_SAMPLE_IDEA, "flip_hypothesis")
        assert result["_original_title"] == "GNN for Drug Discovery"


class TestBlindPeerReview:
    def test_returns_required_fields(self):
        review = generate_blind_review(_SAMPLE_IDEA)
        assert "attacks" in review
        assert "questions" in review
        assert "verdict" in review
        assert "strength_found" in review
        assert len(review["attacks"]) >= 1
        assert len(review["questions"]) >= 1


class TestManifesto:
    def test_with_ideas(self):
        ideas = [_SAMPLE_IDEA] * 5
        result = generate_manifesto(ideas, "TestUser")
        assert "manifesto" in result
        assert "style" in result
        assert "themes" in result
        assert len(result["manifesto"]) > 20

    def test_empty_ideas(self):
        result = generate_manifesto([], "TestUser")
        assert "manifesto" in result
        assert result["themes"] == []

    def test_stats_included(self):
        result = generate_manifesto([_SAMPLE_IDEA] * 3, "User")
        assert result["stats"]["ideas"] == 3


class TestRoulette:
    def test_returns_idea(self):
        pool = [{"title": "A", "quality_score": 0.6}, {"title": "B", "quality_score": 0.7}]
        result = spin_roulette(pool)
        assert result is not None
        assert result["title"] in ("A", "B")

    def test_excludes_seen(self):
        pool = [{"title": "A", "quality_score": 0.6}]
        result = spin_roulette(pool, exclude_titles={"A"})
        assert result is None

    def test_empty_pool(self):
        assert spin_roulette([]) is None


class TestSerendipity:
    def test_generates_mashups(self):
        user = [{"title": "ML idea", "method": "neural", "methodology_type": "empirical_study", "quality_score": 0.6}]
        comm = [{"title": "Bio idea", "method": "biology", "methodology_type": "theoretical_analysis", "quality_score": 0.5}]
        mashups = generate_serendipity_mashups(user, comm, n=1)
        assert len(mashups) >= 1
        assert "synergy_hint" in mashups[0]

    def test_empty_inputs(self):
        assert generate_serendipity_mashups([], []) == []


class TestEvolutionReplay:
    def test_builds_replay(self):
        ideas = [
            {"title": "Idea v1", "method": "method A", "hypothesis": "H1", "quality_score": 0.4, "generation": 0},
            {"title": "Idea v2", "method": "method B", "hypothesis": "H2", "quality_score": 0.7,
             "generation": 1, "parent_title": "Idea v1"},
        ]
        replay = build_evolution_replay(ideas)
        assert len(replay) == 1
        assert replay[0]["quality_delta"] == pytest.approx(0.3, abs=0.01)
        assert len(replay[0]["changes"]) >= 1

    def test_no_parents(self):
        ideas = [{"title": "Solo", "method": "m", "quality_score": 0.5}]
        assert build_evolution_replay(ideas) == []


class TestPersonality:
    def test_returns_all_dimensions(self):
        result = compute_personality([_SAMPLE_IDEA] * 5)
        assert len(result["dimensions"]) == len(PERSONALITY_DIMENSIONS)
        assert result["dominant"] in PERSONALITY_DIMENSIONS
        assert result["blind_spot"] in PERSONALITY_DIMENSIONS

    def test_empty_ideas(self):
        result = compute_personality([])
        assert result["dominant"] == "Unknown"


class TestProphecy:
    def test_returns_predictions(self):
        user = [{"title": "federated learning privacy", "method": "federated aggregation"}]
        comm = [{"title": "federated learning security", "method": "federated privacy methods"}]
        preds = predict_trending(user, comm)
        assert isinstance(preds, list)

    def test_empty(self):
        assert predict_trending([], []) == []


class TestOlympics:
    def test_returns_dimension(self):
        dim = get_current_olympic_dimension()
        assert "title" in dim
        assert "icon" in dim
        assert "description" in dim

    def test_score_computation(self):
        score = compute_olympic_score([_SAMPLE_IDEA] * 3)
        assert isinstance(score, float)
        assert score >= 0


class TestCollaboratorFinder:
    def test_finds_complementary(self):
        user_ideas = [{"title": "ML thing", "method": "neural networks", "methodology_type": "empirical_study"}]
        others = [{"username": "bob", "ideas": [
            {"title": "theory thing", "method": "neural networks proof", "methodology_type": "theoretical_analysis"}
        ]}]
        results = find_collaborators(user_ideas, others)
        assert len(results) >= 1
        assert results[0]["username"] == "bob"

    def test_empty(self):
        assert find_collaborators([], []) == []
