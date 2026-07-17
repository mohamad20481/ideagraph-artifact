"""
Tests for growth.py — user growth, retention, and viral features.
"""
import pytest

from growth import (
    get_current_challenge,
    build_evolution_tree,
    generate_proposal_markdown,
    generate_referral_code,
    get_trending_ideas,
    CHALLENGE_TEMPLATES,
)


class TestWeeklyChallenge:
    def test_returns_valid_challenge(self):
        ch = get_current_challenge()
        assert "title" in ch
        assert "description" in ch
        assert "goal_type" in ch
        assert "goal_count" in ch
        assert "xp_reward" in ch
        assert ch["xp_reward"] > 0
        assert ch["goal_count"] > 0

    def test_has_timing_info(self):
        ch = get_current_challenge()
        assert "week_start" in ch
        assert "week_end" in ch
        assert "days_remaining" in ch
        assert 0 <= ch["days_remaining"] <= 6

    def test_deterministic_for_same_week(self):
        ch1 = get_current_challenge()
        ch2 = get_current_challenge()
        assert ch1["id"] == ch2["id"]

    def test_all_templates_have_required_fields(self):
        for t in CHALLENGE_TEMPLATES:
            assert "id" in t
            assert "title" in t
            assert "description" in t
            assert "goal_type" in t
            assert "goal_count" in t
            assert "xp_reward" in t


class TestEvolutionTree:
    def test_builds_tree_from_ideas(self):
        ideas = [
            {"title": "Root Idea", "quality_score": 0.5, "generation": 0},
            {"title": "Child Idea", "quality_score": 0.7, "generation": 1, "parent_title": "Root Idea"},
        ]
        tree = build_evolution_tree(ideas)
        assert tree["total_ideas"] == 2
        assert tree["total_roots"] == 1
        assert len(tree["roots"]) == 1
        assert tree["roots"][0]["title"] == "Root Idea"
        assert len(tree["roots"][0]["children"]) == 1

    def test_handles_no_parents(self):
        ideas = [
            {"title": "A", "quality_score": 0.5},
            {"title": "B", "quality_score": 0.6},
        ]
        tree = build_evolution_tree(ideas)
        assert tree["total_roots"] == 2

    def test_handles_empty_list(self):
        tree = build_evolution_tree([])
        assert tree["total_ideas"] == 0
        assert tree["roots"] == []


class TestProposalExport:
    def test_generates_markdown(self):
        idea = {
            "title": "Novel GNN Architecture",
            "motivation": "Current GNNs struggle with long-range dependencies.",
            "method": "We propose a hierarchical attention mechanism.",
            "hypothesis": "Our method will improve accuracy by 10%.",
            "resources": "4x A100 GPUs, ZINC dataset",
            "expected_outcome": "State-of-the-art on molecular prediction.",
            "risk_assessment": "May not generalize to all graph types.",
            "quality_score": 0.72,
        }
        md = generate_proposal_markdown(idea, topic="GNN Research")
        assert "Novel GNN Architecture" in md
        assert "hierarchical attention" in md
        assert "Timeline" in md
        assert "Budget" in md
        assert "IdeaGraph" in md

    def test_includes_dag_papers(self):
        idea = {"title": "Test", "motivation": "M", "method": "M", "hypothesis": "H",
                "resources": "R", "expected_outcome": "E", "risk_assessment": "R",
                "quality_score": 0.5}
        papers = [
            {"title": "Paper A", "year": "2024", "authors": [{"name": "Smith"}]},
            {"title": "Paper B", "year": "2023", "authors": [{"name": "Jones"}]},
        ]
        md = generate_proposal_markdown(idea, dag_papers=papers)
        assert "Paper A" in md
        assert "Smith" in md
        assert "Related Work" in md

    def test_handles_empty_idea(self):
        md = generate_proposal_markdown({"title": "", "motivation": "", "method": "",
                                          "hypothesis": "", "resources": "",
                                          "expected_outcome": "", "risk_assessment": "",
                                          "quality_score": 0})
        assert "Research Proposal" in md


class TestReferralProgram:
    def test_generates_deterministic_code(self):
        code1 = generate_referral_code(42)
        code2 = generate_referral_code(42)
        assert code1 == code2
        assert len(code1) == 8
        assert code1 == code1.upper()

    def test_different_users_get_different_codes(self):
        assert generate_referral_code(1) != generate_referral_code(2)


class TestTrendingFeed:
    def test_returns_list(self):
        # With no DB data, should return empty list
        trending = get_trending_ideas(limit=10)
        assert isinstance(trending, list)
