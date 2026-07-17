"""Tests for idea_regenerator.py — derivative idea generation."""
from __future__ import annotations
import json

import pytest

from idea_regenerator import (
    REGEN_MODES,
    regenerate,
    _build_user_prompt,
    _dict_to_idea,
    _parse_idea_json,
)
from models.idea import Idea, METHODOLOGY_TYPES, NOVELTY_LEVELS


# ── Fixtures ─────────────────────────────────────────────────────────────

def _make_parent(**overrides):
    base = dict(
        title="Graph Neural Networks for Drug Discovery",
        motivation="Drug screening at scale is expensive.",
        method="Train message-passing networks on molecular graphs over ZINC.",
        hypothesis="GNNs outperform fingerprint baselines on standard ADMET tasks.",
        resources="1 A100, 24 hours, 200K molecules",
        expected_outcome="15% AUROC improvement over Morgan fingerprints",
        risk_assessment="Dataset bias risk; overfitting at low data regimes.",
    )
    base.update(overrides)
    parent = Idea(**base)
    parent.methodology_type = "empirical_study"
    parent.novelty_level = "moderate"
    parent.quality_score = 0.7
    parent.probe_scores = {"code": 0.7, "dataset": 0.5, "specificity": 0.4,
                            "scalability": 0.6, "novelty": 0.7}
    return parent


_VALID_RESPONSE = {
    "title": "Refined GNN ADMET pipeline",
    "motivation": "Build on parent.",
    "method": "Use message-passing with explicit attention over functional groups.",
    "hypothesis": "Attention adds 3-5% AUROC.",
    "resources": "1 A100, 36 hours, 500K molecules",
    "expected_outcome": "AUROC 0.83",
    "risk_assessment": "Attention overfit",
    "source_strategy": "R",
    "methodology_type": "empirical_study",
    "novelty_level": "incremental",
    "lineage_note": "Adds attention to parent GNN.",
}


class _FakeOK:
    def __init__(self, text, cost_usd=0.001):
        self.success = True
        self.text = text
        self.cost_usd = cost_usd
        self.error = None


class _Client:
    def __init__(self, response_factory):
        self._factory = response_factory
        self.calls = []

    def call(self, system, user, **kw):
        self.calls.append({"system": system, "user": user, **kw})
        return self._factory(len(self.calls) - 1)


# ── Mode catalog ─────────────────────────────────────────────────────────

class TestModeCatalog:
    def test_seven_modes_defined(self):
        assert set(REGEN_MODES.keys()) == {
            "refine", "extend", "pivot", "contrast", "cross_domain",
            "mutate", "topic_transplant",
        }

    def test_topic_transplant_is_required_marked(self):
        cfg = REGEN_MODES["topic_transplant"]
        assert cfg.get("requires_target_topic") is True
        # Should preserve the methodology — that's the whole point of the mode
        assert cfg.get("preserve_methodology") is True

    def test_no_other_mode_requires_topic(self):
        for mode, cfg in REGEN_MODES.items():
            if mode == "topic_transplant":
                continue
            assert not cfg.get("requires_target_topic"), \
                f"only topic_transplant should require target_topic; {mode} also does"

    def test_each_mode_has_required_metadata(self):
        required = {"label", "tagline", "description", "instruction",
                    "default_temp"}
        for mode, cfg in REGEN_MODES.items():
            assert required.issubset(cfg.keys()), f"{mode} missing keys"
            assert 0.4 <= cfg["default_temp"] <= 1.0
            assert len(cfg["description"]) > 30
            assert len(cfg["instruction"]) > 60

    def test_distinct_taglines(self):
        # Each mode must have a unique tagline so the UI dropdown is clear
        taglines = [cfg["tagline"] for cfg in REGEN_MODES.values()]
        assert len(set(taglines)) == len(taglines)

    def test_refine_preserves_methodology_and_novelty(self):
        # Refine = same direction, fixed weaknesses
        cfg = REGEN_MODES["refine"]
        assert cfg.get("preserve_methodology") is True
        assert cfg.get("preserve_novelty") is True

    def test_pivot_does_not_preserve_methodology(self):
        # Pivot's whole point is changing methodology
        cfg = REGEN_MODES["pivot"]
        assert cfg.get("preserve_methodology") is False


# ── Prompt construction ──────────────────────────────────────────────────

class TestPromptBuilding:
    def test_includes_parent_seven_fields(self):
        parent = _make_parent()
        prompt = _build_user_prompt(parent, "refine")
        for snippet in (parent.title[:30], parent.method[:30],
                         parent.hypothesis[:30]):
            assert snippet in prompt

    def test_includes_mode_label(self):
        parent = _make_parent()
        for mode, cfg in REGEN_MODES.items():
            prompt = _build_user_prompt(parent, mode)
            assert cfg["label"] in prompt

    def test_weak_probes_appear_in_prompt(self):
        parent = _make_parent()
        prompt = _build_user_prompt(
            parent, "refine",
            weak_probes={"specificity": 0.3, "scalability": 0.5,
                          "novelty": 0.85},  # this last one shouldn't appear
        )
        # Below 0.6 threshold should be flagged; above should not
        assert "specificity" in prompt
        assert "0.3" in prompt or "0.30" in prompt
        # 0.85 novelty is above threshold so shouldn't appear in weak list
        # (we still mention 'novelty' elsewhere in the prompt, so check the
        # weak section header context instead)
        assert "Probe weaknesses" in prompt

    def test_refine_constrains_methodology(self):
        parent = _make_parent()
        prompt = _build_user_prompt(parent, "refine")
        # Refine preserves methodology_type, so the constraint must appear
        assert "MUST stay 'empirical_study'" in prompt

    def test_pivot_does_not_constrain_methodology(self):
        parent = _make_parent()
        prompt = _build_user_prompt(parent, "pivot")
        assert "MUST stay 'empirical_study'" not in prompt

    def test_invalid_mode_raises(self):
        parent = _make_parent()
        with pytest.raises(ValueError):
            regenerate(parent, "not_a_mode", n=1, claude_client=None)


# ── Response parsing ─────────────────────────────────────────────────────

class TestResponseParsing:
    def test_parses_well_formed_json(self):
        s = json.dumps(_VALID_RESPONSE)
        d = _parse_idea_json(s)
        assert d["title"] == _VALID_RESPONSE["title"]

    def test_strips_code_fences(self):
        wrapped = "```json\n" + json.dumps(_VALID_RESPONSE) + "\n```"
        d = _parse_idea_json(wrapped)
        assert d and d["title"] == _VALID_RESPONSE["title"]

    def test_returns_none_on_garbage(self):
        assert _parse_idea_json("not json at all") is None

    def test_dict_to_idea_sets_lineage(self):
        parent = _make_parent()
        d = dict(_VALID_RESPONSE)
        d["_regen_mode"] = "refine"
        idea = _dict_to_idea(d, parent)
        assert idea is not None
        assert idea.parent_title == parent.title
        assert idea.generation == parent.generation + 1
        assert idea.source_strategy == "R"
        assert idea.execution_meta["lineage_note"] == d["lineage_note"]
        assert idea.execution_meta["regen_mode"] == "refine"

    def test_dict_to_idea_rejects_missing_required(self):
        # Parser must reject ideas missing core fields
        d = {"motivation": "x"}  # no title/method/hypothesis
        assert _dict_to_idea(d, _make_parent()) is None

    def test_dict_to_idea_normalizes_invalid_methodology(self):
        d = dict(_VALID_RESPONSE)
        d["methodology_type"] = "not_a_real_methodology"
        idea = _dict_to_idea(d, _make_parent())
        assert idea.methodology_type is None  # invalid → coerced to None

    def test_dict_to_idea_keeps_valid_methodology(self):
        d = dict(_VALID_RESPONSE)
        d["methodology_type"] = "system_design"
        idea = _dict_to_idea(d, _make_parent())
        assert idea.methodology_type == "system_design"


# ── End-to-end regeneration ──────────────────────────────────────────────

class TestRegenerate:
    def test_returns_n_ideas_with_mock_llm(self):
        parent = _make_parent()
        client = _Client(lambda i: _FakeOK(json.dumps(_VALID_RESPONSE)))
        out = regenerate(parent, "refine", n=3, claude_client=client)
        assert len(out) == 3
        assert all(isinstance(i, Idea) for i in out)

    def test_temperature_increases_with_each_call(self):
        parent = _make_parent()
        client = _Client(lambda i: _FakeOK(json.dumps(_VALID_RESPONSE)))
        regenerate(parent, "refine", n=3, claude_client=client)
        temps = [c["temperature"] for c in client.calls]
        # Each subsequent call bumps temperature by 0.05 to avoid duplicates
        assert temps[0] < temps[1] < temps[2]

    def test_lineage_inherited_in_output(self):
        parent = _make_parent()
        client = _Client(lambda i: _FakeOK(json.dumps(_VALID_RESPONSE)))
        out = regenerate(parent, "refine", n=2, claude_client=client)
        for idea in out:
            assert idea.parent_title == parent.title
            assert idea.generation == 1
            assert idea.source_strategy == "R"

    def test_no_llm_returns_empty_list(self):
        out = regenerate(_make_parent(), "refine", n=2, claude_client=None)
        assert out == []

    def test_n_zero_returns_empty(self):
        client = _Client(lambda i: _FakeOK(json.dumps(_VALID_RESPONSE)))
        assert regenerate(_make_parent(), "refine", n=0,
                           claude_client=client) == []
        assert client.calls == []  # no calls made

    def test_skips_failed_calls_returns_partial(self):
        parent = _make_parent()
        # First call OK, second returns garbage, third OK
        responses = [
            _FakeOK(json.dumps(_VALID_RESPONSE)),
            _FakeOK("not json"),
            _FakeOK(json.dumps(_VALID_RESPONSE)),
        ]
        client = _Client(lambda i: responses[i])
        out = regenerate(parent, "refine", n=3, claude_client=client)
        # Two of three parsed; the bad one is silently dropped
        assert len(out) == 2

    def test_mode_passed_into_lineage_meta(self):
        parent = _make_parent()
        client = _Client(lambda i: _FakeOK(json.dumps(_VALID_RESPONSE)))
        out = regenerate(parent, "cross_domain", n=1, claude_client=client)
        assert len(out) == 1
        assert out[0].execution_meta["regen_mode"] == "cross_domain"

    def test_uses_default_temp_for_first_call(self):
        parent = _make_parent()
        client = _Client(lambda i: _FakeOK(json.dumps(_VALID_RESPONSE)))
        regenerate(parent, "refine", n=1, claude_client=client)
        assert client.calls[0]["temperature"] == REGEN_MODES["refine"]["default_temp"]


class TestAppWiring:
    def test_app_imports_idea_regenerator(self):
        # Ensure the Streamlit app actually wires the new tab
        with open("app.py", encoding="utf-8") as f:
            src = f.read()
        assert "from idea_regenerator import" in src
        assert "tab_regenerate" in src
        assert '"Regenerate"' in src

    def test_app_passes_target_topic_to_regenerate(self):
        # The UI must thread target_topic through the regenerate() call
        with open("app.py", encoding="utf-8") as f:
            src = f.read()
        assert "target_topic=" in src
        assert "_regen_target_topic" in src


class TestTopicTransplant:
    def test_transplant_requires_target_topic(self):
        from idea_regenerator import regenerate
        parent = _make_parent()
        with pytest.raises(ValueError, match="requires a non-empty target_topic"):
            regenerate(parent, "topic_transplant", n=1, claude_client=None)

    def test_transplant_with_blank_topic_also_raises(self):
        from idea_regenerator import regenerate
        parent = _make_parent()
        # Whitespace-only is treated as empty
        with pytest.raises(ValueError):
            regenerate(parent, "topic_transplant", n=1,
                       claude_client=None, target_topic="   ")

    def test_target_topic_appears_in_prompt(self):
        from idea_regenerator import _build_user_prompt
        parent = _make_parent()
        prompt = _build_user_prompt(
            parent, "topic_transplant",
            target_topic="protein folding",
        )
        assert "NEW target topic" in prompt
        assert "protein folding" in prompt
        # The override section must be unmistakably explicit
        assert ">>>" in prompt or "Apply the regeneration to this domain" in prompt

    def test_target_topic_optional_for_other_modes(self):
        from idea_regenerator import regenerate
        parent = _make_parent()
        # Without LLM client, list is empty regardless — the point is no raise
        out = regenerate(parent, "refine", n=1, claude_client=None)
        assert out == []

    def test_target_topic_threaded_for_other_modes_when_set(self):
        # Optional override: any mode + target_topic should inject the new
        # topic into the prompt
        from idea_regenerator import _build_user_prompt
        parent = _make_parent()
        prompt = _build_user_prompt(
            parent, "refine", target_topic="materials discovery",
        )
        assert "NEW target topic" in prompt
        assert "materials discovery" in prompt

    def test_no_override_section_when_target_topic_blank(self):
        # Without target_topic, there should be NO override section
        from idea_regenerator import _build_user_prompt
        parent = _make_parent()
        prompt = _build_user_prompt(parent, "refine")
        assert "NEW target topic" not in prompt

    def test_target_topic_landed_in_execution_meta(self):
        # The new topic should be recorded on the regenerated idea so the
        # Provenance tab can surface it
        from idea_regenerator import regenerate
        parent = _make_parent()
        client = _Client(lambda i: _FakeOK(json.dumps(_VALID_RESPONSE)))
        ideas = regenerate(
            parent, "topic_transplant", n=1, claude_client=client,
            target_topic="protein folding",
        )
        assert len(ideas) == 1
        meta = ideas[0].execution_meta or {}
        assert meta.get("target_topic") == "protein folding"
        assert meta.get("regen_mode") == "topic_transplant"

    def test_transplant_default_temp_lower_than_other_modes(self):
        # Topic transplant should be more deterministic (low temp) since the
        # mode's whole purpose is faithful structural transfer, not creative
        # variation
        cfg_t = REGEN_MODES["topic_transplant"]
        cfg_extend = REGEN_MODES["extend"]
        cfg_contrast = REGEN_MODES["contrast"]
        assert cfg_t["default_temp"] < cfg_extend["default_temp"]
        assert cfg_t["default_temp"] < cfg_contrast["default_temp"]

    def test_temperature_jitter_not_applied_to_target_topic(self):
        # Across n calls we bump temperature each iteration; verify the
        # target_topic is preserved across all of them
        from idea_regenerator import regenerate
        parent = _make_parent()
        seen_user_prompts = []

        class _RecordingClient:
            def call(self, system, user, **kw):
                seen_user_prompts.append(user)
                return _FakeOK(json.dumps(_VALID_RESPONSE))

        regenerate(
            parent, "topic_transplant", n=3,
            claude_client=_RecordingClient(),
            target_topic="quantum computing",
        )
        # All three calls must include the target topic
        assert len(seen_user_prompts) == 3
        for p in seen_user_prompts:
            assert "quantum computing" in p


class TestFreshTake:
    def _three_existing(self):
        a = _make_parent(title="GNN benchmark for ADMET")
        a.methodology_type = "empirical_study"
        b = _make_parent(title="Empirical study of GNN scaling")
        b.methodology_type = "empirical_study"
        c = _make_parent(title="Theoretical bounds on message-passing capacity")
        c.methodology_type = "theoretical_analysis"
        return [a, b, c]

    def test_empty_topic_raises(self):
        from idea_regenerator import regenerate_fresh
        with pytest.raises(ValueError, match="non-empty topic"):
            regenerate_fresh("", self._three_existing(), n=1, claude_client=None)

    def test_whitespace_topic_raises(self):
        from idea_regenerator import regenerate_fresh
        with pytest.raises(ValueError):
            regenerate_fresh("   ", self._three_existing(), n=1, claude_client=None)

    def test_no_llm_returns_empty(self):
        from idea_regenerator import regenerate_fresh
        out = regenerate_fresh(
            "graph neural networks", self._three_existing(),
            n=2, claude_client=None,
        )
        assert out == []

    def test_n_zero_returns_empty(self):
        from idea_regenerator import regenerate_fresh
        out = regenerate_fresh(
            "topic", self._three_existing(), n=0,
            claude_client=lambda: None,  # never called
        )
        assert out == []

    def test_methodology_distribution(self):
        from idea_regenerator import _existing_methodology_distribution
        dist = _existing_methodology_distribution(self._three_existing())
        assert dist == {"empirical_study": 2, "theoretical_analysis": 1}

    def test_summary_lists_existing_titles(self):
        from idea_regenerator import _summarize_existing_ideas
        summary = _summarize_existing_ideas(self._three_existing())
        for title_frag in ("GNN benchmark", "scaling", "Theoretical bounds"):
            assert title_frag in summary

    def test_prompt_lists_existing_ideas_as_anti_exemplars(self):
        from idea_regenerator import _build_fresh_user_prompt
        prompt = _build_fresh_user_prompt(
            "graph neural networks for drug discovery",
            self._three_existing(),
        )
        # Topic must appear
        assert "graph neural networks for drug discovery" in prompt
        # All three existing titles must appear
        assert "GNN benchmark for ADMET" in prompt
        assert "Empirical study of GNN scaling" in prompt
        assert "Theoretical bounds on message-passing capacity" in prompt
        # Anti-duplication framing
        assert "fundamentally different source/angle" in prompt or \
               "different source" in prompt
        # Methodology guidance must surface unused types
        assert "system_design" in prompt or "tool_library" in prompt
        # Must explicitly note the over-represented methodologies
        assert "empirical_study" in prompt

    def test_prompt_omits_methodology_section_when_avoid_disabled(self):
        from idea_regenerator import _build_fresh_user_prompt
        prompt = _build_fresh_user_prompt(
            "topic", self._three_existing(),
            avoid_methodologies=False,
        )
        assert "Methodology diversity" not in prompt

    def test_returns_n_ideas_with_mock_llm(self):
        from idea_regenerator import regenerate_fresh
        client = _Client(lambda i: _FakeOK(json.dumps({
            "title": f"Cross-domain bridge to chemistry #{i+1}",
            "motivation": "m", "method": "ML transfer",
            "hypothesis": "h", "resources": "r",
            "expected_outcome": "e", "risk_assessment": "r",
            "source_strategy": "F",
            "methodology_type": "interdisciplinary_bridge",
            "novelty_level": "substantial",
            "divergence_note": "Bridge not empirical/theoretical",
        })))
        out = regenerate_fresh("topic X", self._three_existing(),
                                n=3, claude_client=client)
        assert len(out) == 3
        for idea in out:
            # Source strategy must be 'F' for fresh-take
            assert idea.source_strategy == "F"
            # Generation 0 — these are top-level new ideas, not derivatives
            assert idea.generation == 0
            # No parent
            assert idea.parent_title is None
            # Lineage stamp
            meta = idea.execution_meta or {}
            assert meta.get("regen_mode") == "fresh_take"
            assert meta.get("topic") == "topic X"
            assert "Bridge not empirical" in (meta.get("divergence_note") or "")

    def test_temperature_increases_with_each_call(self):
        from idea_regenerator import regenerate_fresh
        client = _Client(lambda i: _FakeOK(json.dumps({
            "title": f"Idea {i}", "motivation": "m", "method": "x",
            "hypothesis": "h", "resources": "r",
            "expected_outcome": "e", "risk_assessment": "r",
            "source_strategy": "F",
            "methodology_type": "system_design",
            "novelty_level": "substantial",
            "divergence_note": "different",
        })))
        regenerate_fresh("topic", self._three_existing(),
                          n=3, claude_client=client)
        temps = [c["temperature"] for c in client.calls]
        # Each subsequent call bumps temperature for diversity
        assert temps[0] < temps[1] < temps[2]

    def test_session_titles_appear_in_subsequent_prompts(self):
        # The 2nd and 3rd calls should also list the titles already
        # produced THIS session, so the LLM doesn't return near-duplicates
        # of its own prior responses
        from idea_regenerator import regenerate_fresh

        seen_user_prompts = []

        class _RecordingClient:
            def __init__(self):
                self.n = 0

            def call(self, system, user, **kw):
                seen_user_prompts.append(user)
                self.n += 1
                return _FakeOK(json.dumps({
                    "title": f"Fresh idea {self.n}", "motivation": "m",
                    "method": "x", "hypothesis": "h", "resources": "r",
                    "expected_outcome": "e", "risk_assessment": "r",
                    "source_strategy": "F",
                    "methodology_type": "tool_library",
                    "novelty_level": "moderate",
                    "divergence_note": "different",
                }))

        regenerate_fresh("topic", self._three_existing(),
                          n=3, claude_client=_RecordingClient())
        # Second prompt must mention the first idea's title
        assert "Fresh idea 1" in seen_user_prompts[1]
        # Third prompt must mention both prior idea titles
        assert "Fresh idea 1" in seen_user_prompts[2]
        assert "Fresh idea 2" in seen_user_prompts[2]
        # First prompt has nothing yet
        assert "Already produced THIS session" not in seen_user_prompts[0]

    def test_garbage_response_dropped(self):
        # Bad JSON in one response should not crash; partial result returned
        from idea_regenerator import regenerate_fresh
        responses = [
            _FakeOK(json.dumps({
                "title": "Good 1", "motivation": "m", "method": "x",
                "hypothesis": "h", "resources": "r",
                "expected_outcome": "e", "risk_assessment": "r",
                "source_strategy": "F",
                "methodology_type": "tool_library",
                "novelty_level": "moderate",
                "divergence_note": "different",
            })),
            _FakeOK("totally not json"),
            _FakeOK(json.dumps({
                "title": "Good 2", "motivation": "m", "method": "x",
                "hypothesis": "h", "resources": "r",
                "expected_outcome": "e", "risk_assessment": "r",
                "source_strategy": "F",
                "methodology_type": "tool_library",
                "novelty_level": "moderate",
                "divergence_note": "different",
            })),
        ]
        client = _Client(lambda i: responses[i])
        out = regenerate_fresh("topic", self._three_existing(),
                                n=3, claude_client=client)
        # Two parsed successfully; the bad one is silently dropped
        assert len(out) == 2

    def test_app_imports_regenerate_fresh(self):
        with open("app.py", encoding="utf-8") as f:
            src = f.read()
        assert "regenerate_fresh" in src
        # Operation toggle must be wired
        assert "fresh_take" in src
        assert "regen_op_mode" in src
