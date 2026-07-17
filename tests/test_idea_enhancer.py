"""Tests for idea_enhancer.py — 5 idea generation enhancements."""
import pytest
from idea_enhancer import (
    IdeationKnobs, reproducibility_score, detect_domain, domain_persona_prompt,
    adversarial_prompt, fallback_adversarial, generate_fmea_heuristic,
    fmea_summary, enhance_idea, build_enhancement_prompt_suffix,
    DOMAIN_PERSONAS, _HEURISTIC_FMEA,
)


_SAMPLE = {
    "title": "GNN for Drug Discovery",
    "method": "PyTorch 2.0.1 + CUDA 11.8 with ResNet-50, batch_size=64, lr=1e-4",
    "hypothesis": "GNNs outperform traditional ML",
    "resources": "8x A100-40GB, 336 GPU-hours, ZINC dataset 50GB, seed=42",
    "methodology_type": "empirical_study",
    "novelty_level": "moderate",
}


class TestIdeationKnobs:
    def test_temperature_mapping(self):
        assert IdeationKnobs(creativity_level=0.0).temperature() == 0.4
        assert IdeationKnobs(creativity_level=1.0).temperature() == 1.0
        assert 0.4 < IdeationKnobs(creativity_level=0.5).temperature() < 1.0

    def test_phase_count_by_time(self):
        assert IdeationKnobs(time_budget_weeks=2).time_phase_count() == 2
        assert IdeationKnobs(time_budget_weeks=12).time_phase_count() == 3
        assert IdeationKnobs(time_budget_weeks=52).time_phase_count() == 4

    def test_risk_descriptors(self):
        for r in ("low", "medium", "high"):
            d = IdeationKnobs(risk_tolerance=r).risk_descriptor()
            assert isinstance(d, str) and len(d) > 10

    def test_prompt_context_includes_all_knobs(self):
        ctx = IdeationKnobs(creativity_level=0.8, time_budget_weeks=24,
                            risk_tolerance="high").to_prompt_context()
        assert "80%" in ctx or "0.80" in ctx
        assert "24 weeks" in ctx
        assert "high" in ctx.lower()


class TestReproducibilityScore:
    def test_specific_idea_scores_high(self):
        result = reproducibility_score(_SAMPLE)
        assert result["score"] >= 0.8

    def test_vague_idea_scores_low(self):
        vague = {"resources": "GPU cluster as needed", "method": "use standard PyTorch"}
        result = reproducibility_score(vague)
        assert result["score"] <= 0.2
        assert len(result["missing"]) >= 4

    def test_returns_required_fields(self):
        result = reproducibility_score(_SAMPLE)
        assert "score" in result
        assert "checks" in result
        assert "missing" in result
        assert "vague_phrases" in result


class TestDomainDetection:
    @pytest.mark.parametrize("topic,expected", [
        ("graph neural networks for drug discovery", "graph"),
        ("transformer language models", "nlp"),
        ("image segmentation", "vision"),
        ("reinforcement learning policy", "rl"),
        ("protein folding prediction", "bio"),
        ("robotic manipulation", "robotics"),
        ("compound binding affinity", "drug"),
    ])
    def test_auto_detection(self, topic, expected):
        assert detect_domain(topic) == expected

    def test_default_to_ml(self):
        assert detect_domain("some random topic") == "ml"

    def test_persona_prompt_includes_metrics(self):
        prompt = domain_persona_prompt("rl")
        assert "exploration" in prompt.lower() or "sample efficiency" in prompt.lower()

    def test_all_domains_have_complete_persona(self):
        for d, p in DOMAIN_PERSONAS.items():
            assert "name" in p and "focus" in p and "concerns" in p and "metrics" in p


class TestAdversarialGenerator:
    def test_fallback_inverts_idea(self):
        adv = fallback_adversarial(_SAMPLE)
        assert adv["_adversarial"] is True
        assert "[Contrary]" in adv["title"]
        assert adv["_parent_title"] == _SAMPLE["title"]

    def test_prompt_has_system_and_user(self):
        prompt = adversarial_prompt(_SAMPLE)
        assert "system" in prompt and "user" in prompt
        assert "contrarian" in prompt["system"].lower() or "invert" in prompt["system"].lower()


class TestFMEA:
    def test_heuristic_returns_failure_modes(self):
        modes = generate_fmea_heuristic(_SAMPLE)
        assert len(modes) >= 2
        assert all("mode" in m and "mitigation" in m for m in modes)

    def test_risk_priority_computed(self):
        modes = generate_fmea_heuristic(_SAMPLE)
        for m in modes:
            assert "risk_priority" in m
            assert m["risk_priority"] == m["severity"] * m["detectability"]

    def test_summary_aggregates_correctly(self):
        modes = generate_fmea_heuristic(_SAMPLE)
        summary = fmea_summary(modes)
        assert summary["total"] == len(modes)
        assert summary["max_rpn"] >= 1
        assert summary["avg_severity"] > 0

    def test_empty_summary(self):
        summary = fmea_summary([])
        assert summary["total"] == 0

    def test_methodology_specific_modes(self):
        # empirical_study should have data leakage
        modes = generate_fmea_heuristic({"methodology_type": "empirical_study"})
        assert any("leakage" in m["mode"].lower() for m in modes)
        # theoretical should have assumption issues
        modes = generate_fmea_heuristic({"methodology_type": "theoretical_analysis"})
        assert any("assumption" in m["mode"].lower() for m in modes)


class TestEnhanceIdea:
    def test_attaches_all_metadata(self):
        knobs = IdeationKnobs(enable_adversarial=True, enable_fmea=True, enable_reproducibility=True)
        enhanced = enhance_idea(_SAMPLE, knobs, topic="graph neural networks")
        assert "_reproducibility" in enhanced
        assert "_fmea" in enhanced
        assert "_domain" in enhanced
        assert "_adversarial_twin" in enhanced
        assert "_knobs" in enhanced

    def test_disabled_features_omitted(self):
        knobs = IdeationKnobs(enable_adversarial=False, enable_fmea=False, enable_reproducibility=False)
        enhanced = enhance_idea(_SAMPLE, knobs)
        assert "_reproducibility" not in enhanced
        assert "_fmea" not in enhanced
        assert "_adversarial_twin" not in enhanced

    def test_preserves_original_fields(self):
        enhanced = enhance_idea(_SAMPLE, IdeationKnobs())
        assert enhanced["title"] == _SAMPLE["title"]
        assert enhanced["method"] == _SAMPLE["method"]


class TestPromptSuffix:
    def test_includes_all_components(self):
        knobs = IdeationKnobs(enable_reproducibility=True)
        suffix = build_enhancement_prompt_suffix(knobs, topic="reinforcement learning")
        assert "CONSTRAINTS" in suffix or "Creativity" in suffix
        assert "REPRODUCIBILITY" in suffix
        # Domain persona should be RL
        assert "reinforcement" in suffix.lower() or "rl" in suffix.lower()
