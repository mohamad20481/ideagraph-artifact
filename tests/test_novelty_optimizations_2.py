"""Tests for the three additional novelty modules:
  - heretic_ideation
  - persona_ideation
  - counterfactual_literature

Plus a smoke-check that the Novelty Lab tab now wires all 10 modes.
"""
from __future__ import annotations
import json
import pytest

from models.idea import Idea


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


class _SeqClient:
    def __init__(self, responses):
        self._r = responses
        self.calls = []

    def call(self, system, user, **kw):
        idx = len(self.calls)
        self.calls.append({"system": system[:40], "user": user[:40], **kw})
        if idx < len(self._r):
            return self._r[idx]
        return _FakeFail()


# ═════════════════════════════════════════════════════════════════════════
# heretic_ideation
# ═════════════════════════════════════════════════════════════════════════

_BELIEF_RESPONSE = {
    "beliefs": [
        {"statement": "Scaling laws hold beyond 1T params",
         "evidence_cited": "Chinchilla, GPT scaling papers",
         "why_canonical": "Empirically reinforced over 4 years of training",
         "cracks": "Tested mostly on language, not multi-modal",
         "falsification_hint": "Show a domain where scaling plateaus early",
         "confidence": 0.85},
        {"statement": "Attention is all you need",
         "evidence_cited": "Transformer dominance since 2017",
         "why_canonical": "Replaced RNN/CNN in most benchmarks",
         "cracks": "State-space models matching on long context",
         "falsification_hint": "Show a domain where attention loses to SSM",
         "confidence": 0.70},
    ]
}

_HERETIC_IDEA_RESPONSE = {
    "title": "Demonstrating scaling-law failure in causal multimodal models",
    "motivation": "If scaling fails in this regime, the canonical view is wrong",
    "method": "Train models 100M → 10B on a causal-reasoning benchmark",
    "hypothesis": "Loss plateaus despite increased parameters",
    "resources": "8 H100s × 3 weeks",
    "expected_outcome": "Clear plateau in causal-task accuracy",
    "risk_assessment": "Field-political risk; methodology will be attacked",
    "methodology_type": "empirical_study",
    "novelty_level": "substantial",
    "falsification_mechanism": "Negative scaling result in a specific regime",
}


class TestHeretic:
    def test_empty_topic_raises(self):
        from heretic_ideation import extract_dominant_beliefs, generate_heretic_idea, DominantBelief
        with pytest.raises(ValueError):
            extract_dominant_beliefs("", claude_client=None)
        with pytest.raises(ValueError):
            generate_heretic_idea("", DominantBelief(statement="x"),
                                   claude_client=None)

    def test_no_llm_returns_empty(self):
        from heretic_ideation import (
            extract_dominant_beliefs, generate_heretic_idea,
            generate_heretic_batch, DominantBelief,
        )
        assert extract_dominant_beliefs("topic", claude_client=None) == []
        assert generate_heretic_idea(
            "topic", DominantBelief(statement="x"), claude_client=None,
        ) is None
        assert generate_heretic_batch("topic", claude_client=None) == []

    def test_extract_parses_and_sorts_by_confidence(self):
        from heretic_ideation import extract_dominant_beliefs
        client = _SeqClient([_FakeOK(json.dumps(_BELIEF_RESPONSE))])
        out = extract_dominant_beliefs("topic", claude_client=client, n=2)
        assert len(out) == 2
        # Sorted by confidence desc
        assert out[0].confidence == 0.85
        assert out[1].confidence == 0.70

    def test_invalid_confidence_coerced(self):
        from heretic_ideation import extract_dominant_beliefs
        bad = {"beliefs": [{
            "statement": "X", "evidence_cited": "", "why_canonical": "",
            "cracks": "", "falsification_hint": "",
            "confidence": "very high",
        }]}
        client = _SeqClient([_FakeOK(json.dumps(bad))])
        out = extract_dominant_beliefs("topic", claude_client=client)
        assert out[0].confidence == 0.5  # default

    def test_generate_produces_heretic_idea(self):
        from heretic_ideation import generate_heretic_idea, DominantBelief
        belief = DominantBelief(
            statement="Scaling laws hold", evidence_cited="e",
            why_canonical="w", cracks="c", falsification_hint="h",
            confidence=0.8,
        )
        client = _SeqClient([_FakeOK(json.dumps(_HERETIC_IDEA_RESPONSE))])
        idea = generate_heretic_idea("topic", belief, claude_client=client)
        assert idea is not None
        assert idea.source_strategy == "H"
        meta = idea.execution_meta or {}
        assert meta["regen_mode"] == "heretic"
        assert "belief_targeted" in meta
        # The falsification mechanism made it through
        assert "Negative scaling" in meta["falsification_mechanism"]

    def test_batch_chains_extract_and_generate(self):
        from heretic_ideation import generate_heretic_batch
        responses = [_FakeOK(json.dumps(_BELIEF_RESPONSE))]
        for _ in range(2):
            responses.append(_FakeOK(json.dumps(_HERETIC_IDEA_RESPONSE)))
        client = _SeqClient(responses)
        ideas = generate_heretic_batch("topic", claude_client=client, n=2)
        assert len(ideas) == 2
        assert all(i.source_strategy == "H" for i in ideas)
        # 1 extraction + 2 generations
        assert len(client.calls) == 3


# ═════════════════════════════════════════════════════════════════════════
# persona_ideation
# ═════════════════════════════════════════════════════════════════════════

_PERSONA_IDEA_RESPONSE = {
    "title": "Skeptic's falsification study of MPNNs",
    "motivation": "Take the popular method seriously enough to test it cleanly",
    "method": "Pre-registered comparison with rigorous baselines",
    "hypothesis": "MPNN wins disappear under stronger baselines",
    "resources": "r", "expected_outcome": "e", "risk_assessment": "r",
    "methodology_type": "empirical_study",
    "novelty_level": "incremental",
    "persona_signature": "Skeptic's pre-registered falsification stance",
}


class TestPersona:
    def test_persona_catalog_has_expected_keys(self):
        from persona_ideation import PERSONAS
        for k in ("skeptic", "industry_practitioner", "philosopher",
                  "historian", "naive_outsider", "methodologist",
                  "futurist", "contrarian"):
            assert k in PERSONAS

    def test_empty_topic_raises(self):
        from persona_ideation import generate_under_persona, persona_swap
        with pytest.raises(ValueError):
            generate_under_persona("", "skeptic", claude_client=None)
        with pytest.raises(ValueError):
            persona_swap("", claude_client=None)

    def test_invalid_persona_raises(self):
        from persona_ideation import generate_under_persona
        with pytest.raises(ValueError):
            generate_under_persona("topic", "NOT_A_PERSONA",
                                    claude_client=None)

    def test_no_llm_returns_none(self):
        from persona_ideation import (
            generate_under_persona, persona_swap, PersonaResult,
        )
        assert generate_under_persona(
            "topic", "skeptic", claude_client=None,
        ) is None
        r = persona_swap("topic", claude_client=None)
        assert isinstance(r, PersonaResult)
        assert r.kept_ideas == []

    def test_single_persona_produces_idea(self):
        from persona_ideation import generate_under_persona
        client = _SeqClient([_FakeOK(json.dumps(_PERSONA_IDEA_RESPONSE))])
        idea = generate_under_persona(
            "topic", "skeptic", claude_client=client,
        )
        assert idea is not None
        assert idea.source_strategy == "P"
        meta = idea.execution_meta or {}
        assert meta["regen_mode"] == "persona_swap"
        assert meta["persona_id"] == "skeptic"
        assert "Skeptic" in meta["persona_label"]
        assert meta["persona_signature"]

    def test_swap_runs_in_parallel_and_filters_duplicates(self):
        from persona_ideation import persona_swap

        # Build a recording client that returns near-identical ideas (so
        # the diversity filter should drop some)
        identical = dict(_PERSONA_IDEA_RESPONSE)
        identical["title"] = "Identical title"

        class _Recorder:
            def __init__(self):
                self.n = 0

            def call(self, system, user, **kw):
                self.n += 1
                return _FakeOK(json.dumps(identical))

        result = persona_swap(
            "topic",
            persona_ids=["skeptic", "philosopher", "historian"],
            n_per_persona=1,
            similarity_threshold=0.30,
            claude_client=_Recorder(),
        )
        # All three personas should be in stats
        for p in ("skeptic", "philosopher", "historian"):
            assert p in result.persona_stats
        # Three generated, but identical → diversity filter should reduce
        assert len(result.all_ideas) == 3
        assert len(result.kept_ideas) < len(result.all_ideas)


# ═════════════════════════════════════════════════════════════════════════
# counterfactual_literature
# ═════════════════════════════════════════════════════════════════════════

_CF_ENTRY_RESPONSE = {
    "title": "A capsule-network approach to molecular property prediction",
    "authors": "1998 Stanford bioinformatics group",
    "year": 1998,
    "summary": "Pre-deep-learning idea using shallow capsule structures",
    "why_neglected": "Compute wasn't there; current authors moved to RNNs",
    "what_to_revive": "Shallow capsules for OOD scaffolds",
}

_CF_IDEA_RESPONSE = {
    "title": "Modern capsule-style geometric features for ADMET",
    "motivation": "Revive the abandoned thread with today's compute",
    "method": "Capsule-inspired equivariant pooling",
    "hypothesis": "Better OOD generalization than vanilla GNN",
    "resources": "r", "expected_outcome": "e", "risk_assessment": "r",
    "methodology_type": "empirical_study",
    "novelty_level": "moderate",
    "what_changed": "Compute is cheap now; equivariance theory is mature",
}


class TestCounterfactualLiterature:
    def test_kinds_catalog_complete(self):
        from counterfactual_literature import LITERATURE_KINDS
        for k in ("abandoned_direction", "retracted_finding",
                  "niche_unfollowed", "contrarian_buried",
                  "early_pioneer", "cross_field_orphan"):
            assert k in LITERATURE_KINDS

    def test_empty_topic_raises(self):
        from counterfactual_literature import (
            imagine_counterfactual_literature,
            generate_from_counterfactual, counterfactual_batch,
            CounterfactualEntry,
        )
        with pytest.raises(ValueError):
            imagine_counterfactual_literature("", claude_client=None)
        with pytest.raises(ValueError):
            generate_from_counterfactual(
                "", CounterfactualEntry(title="x"), claude_client=None,
            )
        with pytest.raises(ValueError):
            counterfactual_batch("", claude_client=None)

    def test_invalid_kind_raises(self):
        from counterfactual_literature import imagine_counterfactual_literature
        with pytest.raises(ValueError):
            imagine_counterfactual_literature(
                "topic", kind="bogus_kind", claude_client=None,
            )

    def test_no_llm_returns_safe_defaults(self):
        from counterfactual_literature import (
            imagine_counterfactual_literature,
            generate_from_counterfactual,
            counterfactual_batch, CounterfactualEntry,
        )
        assert imagine_counterfactual_literature(
            "topic", claude_client=None,
        ) is None
        assert generate_from_counterfactual(
            "topic", CounterfactualEntry(title="x"), claude_client=None,
        ) is None
        assert counterfactual_batch("topic", claude_client=None) == []

    def test_imagine_parses_entry(self):
        from counterfactual_literature import imagine_counterfactual_literature
        client = _SeqClient([_FakeOK(json.dumps(_CF_ENTRY_RESPONSE))])
        e = imagine_counterfactual_literature(
            "topic", kind="abandoned_direction", claude_client=client,
        )
        assert e is not None
        assert e.kind == "abandoned_direction"
        assert "capsule" in e.title.lower()
        assert e.year == 1998

    def test_invalid_year_clamped(self):
        from counterfactual_literature import imagine_counterfactual_literature
        bad = dict(_CF_ENTRY_RESPONSE)
        bad["year"] = 9999  # out of range
        client = _SeqClient([_FakeOK(json.dumps(bad))])
        e = imagine_counterfactual_literature(
            "topic", kind="abandoned_direction", claude_client=client,
        )
        assert 1970 <= e.year <= 2025

    def test_generate_from_counterfactual_produces_idea(self):
        from counterfactual_literature import (
            generate_from_counterfactual, CounterfactualEntry,
        )
        e = CounterfactualEntry(
            kind="abandoned_direction",
            title="Old paper", authors="a group", year=1998,
            summary="s", why_neglected="w", what_to_revive="r",
        )
        client = _SeqClient([_FakeOK(json.dumps(_CF_IDEA_RESPONSE))])
        idea = generate_from_counterfactual("topic", e, claude_client=client)
        assert idea is not None
        assert idea.source_strategy == "L"
        meta = idea.execution_meta or {}
        assert meta["regen_mode"] == "counterfactual_literature"
        assert "literature_entry" in meta
        assert meta["what_changed"]

    def test_batch_chains_imagine_then_generate(self):
        from counterfactual_literature import counterfactual_batch
        # 2 kinds → 4 calls (2 imagine, 2 generate)
        responses = [
            _FakeOK(json.dumps(_CF_ENTRY_RESPONSE)),
            _FakeOK(json.dumps(_CF_IDEA_RESPONSE)),
            _FakeOK(json.dumps(_CF_ENTRY_RESPONSE)),
            _FakeOK(json.dumps(_CF_IDEA_RESPONSE)),
        ]
        client = _SeqClient(responses)
        ideas = counterfactual_batch(
            "topic", claude_client=client,
            n=2, kinds=["abandoned_direction", "contrarian_buried"],
        )
        assert len(ideas) == 2
        assert all(i.source_strategy == "L" for i in ideas)
        assert len(client.calls) == 4


# ═════════════════════════════════════════════════════════════════════════
# App wiring
# ═════════════════════════════════════════════════════════════════════════

class TestAppWiring:
    def test_three_new_modules_imported_in_app(self):
        with open("app.py", encoding="utf-8") as f:
            src = f.read()
        assert "from heretic_ideation import" in src
        assert "from persona_ideation import" in src
        assert "from counterfactual_literature import" in src

    def test_novelty_lab_has_10_modes(self):
        with open("app.py", encoding="utf-8") as f:
            src = f.read()
        for mode_key in ("adversarial", "contradiction", "ensemble",
                          "constraint", "future_back", "frontier",
                          "genetic", "heretic", "persona", "counterfactual"):
            assert f'"{mode_key}"' in src, f"missing mode key: {mode_key}"

    def test_new_strategy_codes_distinct(self):
        # H / P / L should appear as source_strategy markers in the modules
        with open("heretic_ideation.py", encoding="utf-8") as f:
            assert '"H"' in f.read()
        with open("persona_ideation.py", encoding="utf-8") as f:
            assert '"P"' in f.read()
        with open("counterfactual_literature.py", encoding="utf-8") as f:
            assert '"L"' in f.read()
