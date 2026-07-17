"""Tests for the four new novelty-optimization modules:

  - constraint_stacking
  - future_back_ideation
  - embedding_exploration
  - genetic_ideation

Plus a smoke-check that the Novelty Lab tab wires all four.
"""
from __future__ import annotations
import json
import random

import pytest

from models.idea import Idea


# ── Shared fixtures ──────────────────────────────────────────────────────

def _mk(title="Test", method="X", motivation="m", hypothesis="h",
         methodology_type="empirical_study", novelty_level="moderate",
         quality_score=0.5):
    i = Idea(title=title, motivation=motivation, method=method,
             hypothesis=hypothesis, resources="r", expected_outcome="e",
             risk_assessment="r")
    i.methodology_type = methodology_type
    i.novelty_level = novelty_level
    i.quality_score = quality_score
    return i


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
    """Returns sequenced responses; falls through to _FakeFail when exhausted."""

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
# constraint_stacking
# ═════════════════════════════════════════════════════════════════════════

class TestConstraintStacking:
    def test_library_has_expected_categories(self):
        from constraint_stacking import CONSTRAINT_LIBRARY
        for cat in ("data", "compute", "method", "output", "audience"):
            assert cat in CONSTRAINT_LIBRARY
            assert len(CONSTRAINT_LIBRARY[cat]) >= 3

    def test_suggest_no_llm_returns_random_library_sample(self):
        from constraint_stacking import suggest_constraints
        out = suggest_constraints("topic", claude_client=None, n=4)
        assert len(out) == 4
        # Two calls with same topic should be deterministic (seeded by hash)
        out2 = suggest_constraints("topic", claude_client=None, n=4)
        assert out == out2

    def test_suggest_empty_topic_raises(self):
        from constraint_stacking import suggest_constraints
        with pytest.raises(ValueError):
            suggest_constraints("", claude_client=None)

    def test_suggest_with_llm_parses(self):
        from constraint_stacking import suggest_constraints
        c = _SeqClient([_FakeOK(json.dumps({
            "constraints": ["No GPU", "Open data only", "Interpretable"],
        }))])
        out = suggest_constraints("topic", claude_client=c, n=3)
        assert out == ["No GPU", "Open data only", "Interpretable"]

    def test_generate_empty_constraints_raises(self):
        from constraint_stacking import generate_with_constraints
        with pytest.raises(ValueError):
            generate_with_constraints("topic", [], claude_client=None)

    def test_generate_no_llm_returns_none(self):
        from constraint_stacking import generate_with_constraints
        assert generate_with_constraints(
            "topic", ["No GPU"], claude_client=None,
        ) is None

    def test_generate_produces_idea_with_meta(self):
        from constraint_stacking import generate_with_constraints
        c = _SeqClient([_FakeOK(json.dumps({
            "title": "CPU-only molecular property prediction",
            "motivation": "Edge deployment requires CPU-only inference",
            "method": "Random forest on Morgan fingerprints",
            "hypothesis": "RF matches GNN on small ADMET tasks",
            "resources": "1 CPU core, 8GB RAM",
            "expected_outcome": "Within 5% AUROC of GNN baselines",
            "risk_assessment": "Overfitting on small features",
            "methodology_type": "empirical_study",
            "novelty_level": "incremental",
            "constraints_satisfied": [
                "No GPU: uses only scikit-learn",
                "Open data: uses ChEMBL public data",
            ],
            "feasibility_note": "Genuinely satisfiable.",
        }))])
        idea = generate_with_constraints(
            "molecular property prediction",
            ["No GPU", "Open data only"],
            claude_client=c,
        )
        assert idea is not None
        assert idea.source_strategy == "K"
        meta = idea.execution_meta or {}
        assert meta["regen_mode"] == "constraint_stacking"
        assert len(meta["constraints"]) == 2
        assert len(meta["constraints_satisfied"]) == 2
        assert meta["feasibility_note"]


# ═════════════════════════════════════════════════════════════════════════
# future_back_ideation
# ═════════════════════════════════════════════════════════════════════════

class TestFutureBack:
    def test_scenarios_complete(self):
        from future_back_ideation import SCENARIOS
        assert "best_case" in SCENARIOS
        assert "worst_case" in SCENARIOS
        assert "surprising" in SCENARIOS
        assert "neutral" in SCENARIOS

    def test_imagine_empty_topic_raises(self):
        from future_back_ideation import imagine_future
        with pytest.raises(ValueError):
            imagine_future("", claude_client=None)

    def test_imagine_invalid_year_raises(self):
        from future_back_ideation import imagine_future
        with pytest.raises(ValueError):
            imagine_future("topic", year=1990, claude_client=None)
        with pytest.raises(ValueError):
            imagine_future("topic", year=2200, claude_client=None)

    def test_imagine_invalid_scenario_raises(self):
        from future_back_ideation import imagine_future
        with pytest.raises(ValueError):
            imagine_future("topic", scenario="bogus", claude_client=None)

    def test_imagine_no_llm_returns_none(self):
        from future_back_ideation import imagine_future
        assert imagine_future("topic", claude_client=None) is None

    def test_imagine_parses_vision(self):
        from future_back_ideation import imagine_future, FutureVision
        c = _SeqClient([_FakeOK(json.dumps({
            "description": "A vivid future where X is solved.",
            "what_is_solved": ["A", "B", "C"],
            "what_is_commonplace": ["D", "E"],
            "what_is_still_hard": ["F"],
            "surprising_capability": "Real-time protein-ligand binding sim",
        }))])
        v = imagine_future("topic", year=2040,
                            scenario="best_case", claude_client=c)
        assert isinstance(v, FutureVision)
        assert v.year == 2040
        assert v.scenario == "best_case"
        assert v.description.startswith("A vivid future")
        assert len(v.what_is_solved) == 3

    def test_back_propagate_no_llm_returns_none(self):
        from future_back_ideation import back_propagate, FutureVision
        v = FutureVision(topic="t", year=2040, scenario="neutral",
                         description="d")
        assert back_propagate("topic", v, claude_client=None) is None

    def test_back_propagate_produces_idea(self):
        from future_back_ideation import back_propagate, FutureVision
        v = FutureVision(topic="t", year=2040, scenario="best_case",
                         description="d", what_is_solved=["X"],
                         what_is_commonplace=["Y"],
                         what_is_still_hard=["Z"],
                         surprising_capability="W")
        c = _SeqClient([_FakeOK(json.dumps({
            "title": "Foundation idea for X",
            "motivation": "Enables X by 2040",
            "method": "Build the foundational system",
            "hypothesis": "Works at moderate scale",
            "resources": "r", "expected_outcome": "e", "risk_assessment": "r",
            "methodology_type": "system_design",
            "novelty_level": "substantial",
            "future_link": "Direct enabler of X",
        }))])
        idea = back_propagate("topic", v, claude_client=c)
        assert idea is not None
        assert idea.source_strategy == "U"
        meta = idea.execution_meta or {}
        assert meta["regen_mode"] == "future_back"
        assert "vision" in meta
        assert meta["future_link"] == "Direct enabler of X"

    def test_batch_chains_visions_and_ideas(self):
        from future_back_ideation import future_back_batch
        # 2 scenarios → 4 LLM calls (2 visions + 2 back-propagations)
        v_response = _FakeOK(json.dumps({
            "description": "vision",
            "what_is_solved": ["x"], "what_is_commonplace": ["y"],
            "what_is_still_hard": ["z"], "surprising_capability": "w",
        }))
        i_response = _FakeOK(json.dumps({
            "title": "idea", "motivation": "m", "method": "x",
            "hypothesis": "h", "resources": "r", "expected_outcome": "e",
            "risk_assessment": "r",
            "methodology_type": "system_design",
            "novelty_level": "substantial",
            "future_link": "link",
        }))
        c = _SeqClient([v_response, i_response, v_response, i_response])
        out = future_back_batch("topic", claude_client=c, n=2,
                                  scenarios=["best_case", "worst_case"])
        assert len(out) == 2
        assert len(c.calls) == 4


# ═════════════════════════════════════════════════════════════════════════
# embedding_exploration
# ═════════════════════════════════════════════════════════════════════════

class TestEmbeddingExploration:
    def _archive(self):
        return [
            _mk("GNN for ADMET",
                "message-passing neural network attention molecular graphs"),
            _mk("Empirical GNN scaling",
                "message-passing benchmark attention scale graphs"),
            _mk("Theoretical capacity",
                "theoretical analysis expressiveness graphs"),
        ]

    def test_token_distribution(self):
        from embedding_exploration import _coverage_distribution
        d = _coverage_distribution(self._archive())
        assert d.get("graphs", 0) == 3
        assert d.get("attention", 0) == 2

    def test_compute_no_llm_returns_analysis_with_counts_only(self):
        from embedding_exploration import compute_frontier_concepts
        a = compute_frontier_concepts(
            self._archive(), "graph neural networks", claude_client=None,
        )
        assert a.n_archive_ideas == 3
        assert len(a.covered_concepts) > 0
        # No LLM, so the under-explored list stays empty
        assert a.underexplored_concepts == []

    def test_compute_empty_topic_raises(self):
        from embedding_exploration import compute_frontier_concepts
        with pytest.raises(ValueError):
            compute_frontier_concepts([], "", claude_client=None)

    def test_compute_with_llm_populates_frontier(self):
        from embedding_exploration import compute_frontier_concepts
        c = _SeqClient([_FakeOK(json.dumps({
            "underexplored_concepts": ["interpretability", "robustness",
                                          "generalization"],
            "frontier_description": "The interpretability gap is wide open.",
            "frontier_seeds": ["What if GNNs were globally interpretable?"],
        }))])
        a = compute_frontier_concepts(
            self._archive(), "graph neural networks", claude_client=c,
        )
        assert len(a.underexplored_concepts) == 3
        assert "interpretability" in a.underexplored_concepts
        assert a.frontier_description.startswith("The interpretability")
        assert len(a.frontier_seeds) == 1

    def test_generate_empty_topic_raises(self):
        from embedding_exploration import (
            generate_at_frontier, FrontierAnalysis,
        )
        with pytest.raises(ValueError):
            generate_at_frontier("", FrontierAnalysis(),
                                  claude_client=None)

    def test_generate_no_frontier_data_returns_none(self):
        # Empty under-explored + empty seeds → nothing to target → None
        from embedding_exploration import (
            generate_at_frontier, FrontierAnalysis,
        )
        empty = FrontierAnalysis(topic="t")
        assert generate_at_frontier("topic", empty, claude_client=None) is None

    def test_generate_produces_idea(self):
        from embedding_exploration import (
            generate_at_frontier, FrontierAnalysis,
        )
        fa = FrontierAnalysis(
            topic="t",
            underexplored_concepts=["interpretability", "robustness"],
            frontier_description="d",
            frontier_seeds=["what if?"],
        )
        c = _SeqClient([_FakeOK(json.dumps({
            "title": "Globally interpretable GNN",
            "motivation": "Targets interpretability + robustness",
            "method": "Concept-bottleneck GNN with proven OOD bounds",
            "hypothesis": "Interpretability does not hurt OOD",
            "resources": "r", "expected_outcome": "e",
            "risk_assessment": "r",
            "methodology_type": "theoretical_analysis",
            "novelty_level": "substantial",
            "frontier_concepts_used": ["interpretability", "robustness"],
        }))])
        idea = generate_at_frontier("topic", fa, claude_client=c)
        assert idea is not None
        assert idea.source_strategy == "X"
        meta = idea.execution_meta or {}
        assert meta["regen_mode"] == "embedding_exploration"
        assert "interpretability" in meta["frontier_concepts_used"]


# ═════════════════════════════════════════════════════════════════════════
# genetic_ideation
# ═════════════════════════════════════════════════════════════════════════

class TestGenetic:
    def test_default_fitness_rewards_high_quality(self):
        from genetic_ideation import _default_fitness
        good = _mk("A", "alpha beta gamma", quality_score=0.9)
        bad = _mk("B", "alpha beta gamma", quality_score=0.2)
        pop = [good, bad]
        assert _default_fitness(good, pop) > _default_fitness(bad, pop)

    def test_default_fitness_handles_empty_pop(self):
        from genetic_ideation import _default_fitness
        good = _mk("A", "x y z", quality_score=0.7)
        # Population of size 1 → no token-rarity comparison, but no crash
        score = _default_fitness(good, [good])
        assert 0.0 <= score <= 1.0

    def test_crossover_no_llm_returns_none(self):
        from genetic_ideation import crossover
        a, b = _mk("A", "x"), _mk("B", "y")
        assert crossover(a, b, claude_client=None) is None

    def test_crossover_produces_offspring(self):
        from genetic_ideation import crossover
        a = _mk("A method", "MPNN")
        b = _mk("B hypothesis", "GNN")
        c = _SeqClient([_FakeOK(json.dumps({
            "title": "MPNN+GNN hybrid",
            "motivation": "m", "method": "x", "hypothesis": "h",
            "resources": "r", "expected_outcome": "e",
            "risk_assessment": "r",
            "methodology_type": "empirical_study",
            "novelty_level": "moderate",
            "lineage_note": "Method from A, hypothesis from B",
        }))])
        offspring = crossover(a, b, claude_client=c, rng=random.Random(42))
        assert offspring is not None
        assert offspring.source_strategy == "G"
        # Generation should be max(parent_gens) + 1
        assert offspring.generation == 1
        meta = offspring.execution_meta or {}
        assert meta["regen_mode"] == "genetic_crossover"
        assert meta["parent_a_title"] == "A method"
        assert meta["parent_b_title"] == "B hypothesis"

    def test_mutate_no_llm_returns_none(self):
        from genetic_ideation import mutate
        assert mutate(_mk(), claude_client=None) is None

    def test_mutate_produces_idea_with_mutation_kind(self):
        from genetic_ideation import mutate
        c = _SeqClient([_FakeOK(json.dumps({
            "title": "Mutated idea",
            "motivation": "m", "method": "x", "hypothesis": "h",
            "resources": "r", "expected_outcome": "e",
            "risk_assessment": "r",
            "methodology_type": "empirical_study",
            "novelty_level": "moderate",
            "mutation_note": "Swapped dataset",
        }))])
        m = mutate(_mk("Original"), claude_client=c,
                    rng=random.Random(42))
        assert m is not None
        assert m.source_strategy == "G"
        meta = m.execution_meta or {}
        assert meta["regen_mode"] == "genetic_mutation"
        assert "mutation_kind" in meta
        assert meta["mutation_note"] == "Swapped dataset"

    def test_evolve_empty_population(self):
        from genetic_ideation import evolve
        r = evolve([], n_generations=2)
        assert r.initial_size == 0

    def test_evolve_zero_generations(self):
        from genetic_ideation import evolve
        pop = [_mk("A"), _mk("B")]
        r = evolve(pop, n_generations=0, claude_client=None)
        assert r.n_generations == 0
        assert len(r.final_population) == 2

    def test_evolve_no_llm_returns_initial_pop(self):
        from genetic_ideation import evolve
        pop = [_mk("A"), _mk("B"), _mk("C")]
        r = evolve(pop, n_generations=3, claude_client=None)
        assert len(r.final_population) == 3

    def test_evolve_with_mock_llm(self):
        from genetic_ideation import evolve, EvolutionResult
        pop = [_mk("A", quality_score=0.7),
               _mk("B", quality_score=0.6),
               _mk("C", quality_score=0.5),
               _mk("D", quality_score=0.4)]

        # Mock returns valid offspring on every call
        def _factory(i):
            return _FakeOK(json.dumps({
                "title": f"Evolved idea {i}",
                "motivation": "m", "method": "x", "hypothesis": "h",
                "resources": "r", "expected_outcome": "e",
                "risk_assessment": "r",
                "methodology_type": "empirical_study",
                "novelty_level": "moderate",
                "lineage_note": "evolved",
                "mutation_note": "evolved",
            }))

        class _Forever:
            def __init__(self): self.n = 0
            def call(self, system, user, **kw):
                self.n += 1
                return _factory(self.n)

        r = evolve(pop, n_generations=2, claude_client=_Forever(),
                    elite_keep=2, seed=42)
        assert isinstance(r, EvolutionResult)
        assert r.n_generations == 2
        assert len(r.final_population) == len(pop)
        # Either crossovers or mutations should have happened
        assert (r.crossover_count + r.mutation_count) > 0
        # Fitness history should have one entry per generation
        assert len(r.fitness_history) == 2


# ═════════════════════════════════════════════════════════════════════════
# App wiring
# ═════════════════════════════════════════════════════════════════════════

class TestAppWiring:
    def test_all_4_modules_imported_in_app(self):
        with open("app.py", encoding="utf-8") as f:
            src = f.read()
        assert "from constraint_stacking import" in src
        assert "from future_back_ideation import" in src
        assert "from embedding_exploration import" in src
        assert "from genetic_ideation import" in src

    def test_novelty_lab_has_7_modes(self):
        with open("app.py", encoding="utf-8") as f:
            src = f.read()
        # Check that the radio options list contains all 7 mode keys
        for mode_key in ("adversarial", "contradiction", "ensemble",
                          "constraint", "future_back", "frontier", "genetic"):
            assert f'"{mode_key}"' in src, f"missing mode key: {mode_key}"

    def test_strategy_codes_for_new_modes_are_distinct(self):
        # Each module uses a different single-letter source_strategy code
        # to make Provenance distinguishable
        codes = {"K": "constraint_stacking",
                  "U": "future_back",
                  "X": "embedding_exploration",
                  "G": "genetic"}
        # Just verify all four are present in the new module sources
        with open("constraint_stacking.py", encoding="utf-8") as f:
            assert '"K"' in f.read() or "'K'" in f.read()
        with open("future_back_ideation.py", encoding="utf-8") as f:
            assert '"U"' in f.read() or "'U'" in f.read()
        with open("embedding_exploration.py", encoding="utf-8") as f:
            assert '"X"' in f.read() or "'X'" in f.read()
        with open("genetic_ideation.py", encoding="utf-8") as f:
            assert '"G"' in f.read() or "'G'" in f.read()
