"""Tests for idea_chat — conversational idea refinement (strategy V)."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import idea_chat as ic
from idea_chat import ChatMessage
from models.idea import METHODOLOGY_TYPES, NOVELTY_LEVELS


# ── Mock LLM client ─────────────────────────────────────────────────────────

@dataclass
class _MockResp:
    success: bool
    text: str


class _SeqClient:
    def __init__(self, responses: List[Any]):
        self._queue = list(responses)
        self.call_count = 0
        self.last_system = None
        self.last_user = None

    def call(self, system: str, user: str, **kw) -> _MockResp:
        self.call_count += 1
        self.last_system = system
        self.last_user = user
        if not self._queue:
            return _MockResp(False, "")
        item = self._queue.pop(0)
        if isinstance(item, _MockResp):
            return item
        if isinstance(item, dict):
            return _MockResp(True, json.dumps(item))
        if isinstance(item, str):
            return _MockResp(True, item)
        return _MockResp(False, "")


_BASE_IDEA = {
    "title": "Probing LLM dialect transfer",
    "motivation": "Existing benchmarks ignore dialectal variation.",
    "method": "Fine-tune a small model on dialectal pairs.",
    "hypothesis": "Pair-tuning beats zero-shot on dialect tasks.",
    "resources": "1 GPU-week.",
    "expected_outcome": "+10 F1 over baseline.",
    "risk_assessment": "Dialect data is scarce.",
    "source_strategy": "A",
    "methodology_type": METHODOLOGY_TYPES[0],
    "novelty_level": NOVELTY_LEVELS[1],
    "generation": 0,
}


# ── ChatMessage dataclass ──────────────────────────────────────────────────

def test_chat_message_to_dict_roundtrip():
    m = ChatMessage(role="user", content="hello")
    d = m.to_dict()
    assert d == {"role": "user", "content": "hello"}


# ── _validate_history ──────────────────────────────────────────────────────

def test_validate_history_accepts_empty_list():
    ic._validate_history([])


def test_validate_history_accepts_valid_turns():
    hist = [
        ChatMessage(role="user", content="a"),
        ChatMessage(role="assistant", content="b"),
    ]
    ic._validate_history(hist)


def test_validate_history_rejects_non_chatmessage():
    with pytest.raises(TypeError):
        ic._validate_history([{"role": "user", "content": "x"}])  # type: ignore[list-item]


def test_validate_history_rejects_bad_role():
    with pytest.raises(ValueError):
        ic._validate_history([ChatMessage(role="bot", content="x")])


# ── chat_turn ──────────────────────────────────────────────────────────────

def test_chat_turn_empty_idea_raises():
    with pytest.raises(ValueError):
        ic.chat_turn({}, [], "hello", claude_client=None)


def test_chat_turn_non_dict_idea_raises():
    with pytest.raises(ValueError):
        ic.chat_turn(None, [], "hello", claude_client=None)  # type: ignore[arg-type]


def test_chat_turn_empty_message_raises():
    with pytest.raises(ValueError):
        ic.chat_turn(_BASE_IDEA, [], "   ", claude_client=None)


def test_chat_turn_no_client_returns_none():
    out = ic.chat_turn(_BASE_IDEA, [], "hi", claude_client=None)
    assert out is None


def test_chat_turn_happy_path_first_turn():
    client = _SeqClient(["I think the method is fine. What's the concern?"])
    reply = ic.chat_turn(_BASE_IDEA, [], "Is the method strong?",
                            claude_client=client)
    assert reply == "I think the method is fine. What's the concern?"
    assert client.call_count == 1
    # System prompt mentions improvement (not invention of alternatives).
    assert "improve" in client.last_system.lower()
    # User prompt embeds the idea state and the user message.
    assert "Probing LLM dialect transfer" in client.last_user
    assert "Is the method strong?" in client.last_user


def test_chat_turn_serializes_prior_history_into_user_msg():
    history = [
        ChatMessage(role="user", content="What is the riskiest assumption?"),
        ChatMessage(role="assistant", content="Data scarcity for dialects."),
    ]
    client = _SeqClient(["Consider crowd-sourcing 1k examples first."])
    reply = ic.chat_turn(_BASE_IDEA, history, "Suggest a mitigation.",
                            claude_client=client)
    assert reply == "Consider crowd-sourcing 1k examples first."
    # Both prior turns must appear in the rendered user prompt.
    assert "USER: What is the riskiest assumption?" in client.last_user
    assert "ASSISTANT: Data scarcity for dialects." in client.last_user
    assert "Suggest a mitigation." in client.last_user


def test_chat_turn_returns_none_when_response_fails():
    client = _SeqClient([_MockResp(False, "")])
    assert ic.chat_turn(_BASE_IDEA, [], "hi", claude_client=client) is None


def test_chat_turn_returns_none_when_text_is_empty():
    client = _SeqClient([_MockResp(True, "   ")])
    assert ic.chat_turn(_BASE_IDEA, [], "hi", claude_client=client) is None


def test_chat_turn_handles_client_exception():
    class _BadClient:
        def call(self, **kw):
            raise RuntimeError("boom")
    assert ic.chat_turn(_BASE_IDEA, [], "hi",
                            claude_client=_BadClient()) is None


def test_chat_turn_does_not_mutate_history():
    history = [ChatMessage(role="user", content="prev")]
    client = _SeqClient(["reply"])
    ic.chat_turn(_BASE_IDEA, history, "now", claude_client=client)
    assert len(history) == 1
    assert history[0].content == "prev"


# ── crystallize ────────────────────────────────────────────────────────────

def test_crystallize_empty_history_raises():
    with pytest.raises(ValueError):
        ic.crystallize(_BASE_IDEA, [], claude_client=None)


def test_crystallize_empty_idea_raises():
    with pytest.raises(ValueError):
        ic.crystallize({}, [ChatMessage(role="user", content="x")],
                          claude_client=None)


def test_crystallize_no_client_returns_none():
    hist = [ChatMessage(role="user", content="cheaper please")]
    assert ic.crystallize(_BASE_IDEA, hist, claude_client=None) is None


def test_crystallize_happy_path_builds_V_strategy_idea():
    history = [
        ChatMessage(role="user",
                       content="The method is too expensive."),
        ChatMessage(role="assistant",
                       content="Use a distilled 100M model instead."),
    ]
    payload = {
        "title": "Probing LLM dialect transfer (distilled)",
        "method": "Distill a 100M model on dialectal pairs.",
        "hypothesis": "Distilled pair-tuning still beats zero-shot.",
        "motivation": "Existing benchmarks ignore dialectal variation.",
        "resources": "Single A100 hour.",
        "expected_outcome": "+8 F1 over baseline at 1/10 cost.",
        "risk_assessment": "Distillation may bleed accuracy.",
        "methodology_type": METHODOLOGY_TYPES[0],
        "novelty_level": NOVELTY_LEVELS[1],
        "change_summary": "Swapped fine-tune for distillation to "
                            "reduce compute.",
    }
    client = _SeqClient([payload])
    new_idea = ic.crystallize(_BASE_IDEA, history, claude_client=client)
    assert new_idea is not None
    assert new_idea.source_strategy == "V"
    assert new_idea.generation == 1  # bumped from 0
    assert new_idea.parent_title == "Probing LLM dialect transfer"
    assert new_idea.execution_meta["regen_mode"] == "chat"
    assert new_idea.execution_meta["parent_strategy"] == "A"
    assert "distillation" in new_idea.execution_meta["change_summary"].lower()
    # Chat history is preserved on the new idea.
    saved_hist = new_idea.execution_meta["chat_history"]
    assert len(saved_hist) == 2
    assert saved_hist[0]["role"] == "user"
    assert saved_hist[1]["role"] == "assistant"


def test_crystallize_missing_fields_fall_back_to_original():
    """If the LLM omits a field, the original idea's value is kept."""
    history = [ChatMessage(role="user", content="trim risks section")]
    payload = {
        # No title, motivation, method, hypothesis, etc. — only:
        "risk_assessment": "Dialect data is scarce; mitigate via pretraining.",
        "change_summary": "Sharpened the risks line.",
    }
    client = _SeqClient([payload])
    new_idea = ic.crystallize(_BASE_IDEA, history, claude_client=client)
    assert new_idea is not None
    assert new_idea.title == _BASE_IDEA["title"]
    assert new_idea.method == _BASE_IDEA["method"]
    assert new_idea.hypothesis == _BASE_IDEA["hypothesis"]
    assert "pretraining" in new_idea.risk_assessment


def test_crystallize_invalid_enums_fall_back_to_original_or_none():
    """methodology_type/novelty_level get rolled back if invalid."""
    history = [ChatMessage(role="user", content="x")]
    payload = {
        "title": "T", "method": "M", "hypothesis": "H",
        "methodology_type": "garbage_value",
        "novelty_level": "also_garbage",
        "change_summary": "test",
    }
    client = _SeqClient([payload])
    new_idea = ic.crystallize(_BASE_IDEA, history, claude_client=client)
    assert new_idea is not None
    # Falls back to the ORIGINAL's valid value (not None) since the
    # original had a valid methodology_type.
    assert new_idea.methodology_type == _BASE_IDEA["methodology_type"]
    assert new_idea.novelty_level == _BASE_IDEA["novelty_level"]


def test_crystallize_missing_required_returns_none():
    """If the LLM leaves title/method/hypothesis empty AND the original
    is empty, we return None."""
    empty_idea = dict(_BASE_IDEA)
    empty_idea["title"] = ""
    empty_idea["method"] = ""
    empty_idea["hypothesis"] = ""
    history = [ChatMessage(role="user", content="x")]
    client = _SeqClient([{"change_summary": "no-op"}])
    assert ic.crystallize(empty_idea, history,
                              claude_client=client) is None


def test_crystallize_handles_failed_response():
    history = [ChatMessage(role="user", content="x")]
    client = _SeqClient([_MockResp(False, "")])
    assert ic.crystallize(_BASE_IDEA, history,
                              claude_client=client) is None


def test_crystallize_handles_unparseable_json():
    history = [ChatMessage(role="user", content="x")]
    client = _SeqClient([_MockResp(True, "not json at all{{{")])
    assert ic.crystallize(_BASE_IDEA, history,
                              claude_client=client) is None


def test_crystallize_strips_code_fences():
    history = [ChatMessage(role="user", content="x")]
    fenced = (
        "```json\n"
        '{"title":"T","method":"M","hypothesis":"H",'
        '"change_summary":"s"}\n'
        "```"
    )
    client = _SeqClient([_MockResp(True, fenced)])
    out = ic.crystallize(_BASE_IDEA, history, claude_client=client)
    assert out is not None
    assert out.title == "T"


def test_crystallize_does_not_mutate_history_or_idea():
    history = [ChatMessage(role="user", content="x"),
                ChatMessage(role="assistant", content="y")]
    payload = {"title": "T", "method": "M", "hypothesis": "H",
                "change_summary": "s"}
    client = _SeqClient([payload])
    original_idea_copy = dict(_BASE_IDEA)
    ic.crystallize(_BASE_IDEA, history, claude_client=client)
    assert _BASE_IDEA == original_idea_copy
    assert len(history) == 2


# ── Source-strategy code uniqueness ────────────────────────────────────────

def test_V_strategy_distinct_from_existing_codes():
    """V must not collide with any other strategy code in use."""
    existing = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L",
                "M", "N", "P", "R", "S", "T", "U", "W", "X", "Y", "Z"}
    assert "V" not in existing


def test_V_strategy_present_in_module_file():
    assert 'source_strategy="V"' in \
        (ROOT / "idea_chat.py").read_text(encoding="utf-8")


def test_app_imports_idea_chat_in_ideas_tab():
    """Smoke check: the Ideas tab is wired to import idea_chat."""
    app_text = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "from idea_chat import" in app_text
    assert "ChatMessage as _ChatMessage" in app_text
    assert "Chat to optimize this idea" in app_text


# ─────────────────────────────────────────────────────────────────────────────
# Chat modes
# ─────────────────────────────────────────────────────────────────────────────

def test_chat_modes_registry_has_four_modes():
    assert set(ic.CHAT_MODES.keys()) == {
        "collaborator", "critic", "skeptic", "advocate",
    }


def test_chat_modes_each_have_label_description_system():
    for k, v in ic.CHAT_MODES.items():
        assert "label" in v and v["label"], f"mode {k} missing label"
        assert "description" in v and v["description"], \
            f"mode {k} missing description"
        assert "system" in v and len(v["system"]) > 50, \
            f"mode {k} system prompt too short"


def test_default_mode_is_collaborator():
    assert ic.DEFAULT_MODE == "collaborator"
    assert ic.DEFAULT_MODE in ic.CHAT_MODES


def test_system_for_mode_returns_correct_prompt():
    assert ic._system_for_mode("collaborator") == ic._COLLAB_SYSTEM
    assert ic._system_for_mode("critic") == ic._CRITIC_SYSTEM
    assert ic._system_for_mode("skeptic") == ic._SKEPTIC_SYSTEM
    assert ic._system_for_mode("advocate") == ic._ADVOCATE_SYSTEM


def test_system_for_mode_rejects_unknown_mode():
    with pytest.raises(ValueError):
        ic._system_for_mode("zealot")


def test_chat_turn_uses_critic_mode_system_prompt():
    client = _SeqClient(["The weakest point is the small sample size."])
    reply = ic.chat_turn(
        _BASE_IDEA, [], "Critique this.",
        claude_client=client, mode="critic",
    )
    assert reply
    assert client.last_system == ic._CRITIC_SYSTEM
    assert "adversarial" in client.last_system.lower()


def test_chat_turn_uses_advocate_mode_system_prompt():
    client = _SeqClient(["Strong reason to believe this works…"])
    ic.chat_turn(
        _BASE_IDEA, [], "Defend it.",
        claude_client=client, mode="advocate",
    )
    assert client.last_system == ic._ADVOCATE_SYSTEM


def test_chat_turn_invalid_mode_raises():
    with pytest.raises(ValueError):
        ic.chat_turn(_BASE_IDEA, [], "hi",
                        claude_client=None, mode="not_a_mode")


# ─────────────────────────────────────────────────────────────────────────────
# SUGGESTED_PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

def test_suggested_prompts_well_formed():
    assert len(ic.SUGGESTED_PROMPTS) >= 6
    for p in ic.SUGGESTED_PROMPTS:
        assert "label" in p and p["label"]
        assert "prompt" in p and len(p["prompt"]) > 10
        # Labels should be short.
        assert len(p["label"]) <= 30


# ─────────────────────────────────────────────────────────────────────────────
# regenerate_last
# ─────────────────────────────────────────────────────────────────────────────

def test_regenerate_last_empty_history_raises():
    with pytest.raises(ValueError):
        ic.regenerate_last(_BASE_IDEA, [], claude_client=None)


def test_regenerate_last_ending_with_user_raises():
    history = [ChatMessage(role="user", content="hi")]
    with pytest.raises(ValueError):
        ic.regenerate_last(_BASE_IDEA, history, claude_client=None)


def test_regenerate_last_missing_user_before_assistant_raises():
    """The penultimate message must be a user turn."""
    history = [
        ChatMessage(role="assistant", content="hello"),
        ChatMessage(role="assistant", content="still talking?"),
    ]
    with pytest.raises(ValueError):
        ic.regenerate_last(_BASE_IDEA, history, claude_client=None)


def test_regenerate_last_happy_path_calls_chat_turn():
    history = [
        ChatMessage(role="user", content="What's the riskiest assumption?"),
        ChatMessage(role="assistant", content="Data scarcity."),
    ]
    client = _SeqClient(["A different take: confounded evaluation."])
    out = ic.regenerate_last(_BASE_IDEA, history, claude_client=client)
    assert out == "A different take: confounded evaluation."
    # The user prompt should contain the prior user message but NOT the
    # old assistant message we're regenerating.
    assert "What's the riskiest assumption?" in client.last_user
    assert "Data scarcity." not in client.last_user


def test_regenerate_last_preserves_pre_pair_history():
    history = [
        ChatMessage(role="user", content="first"),
        ChatMessage(role="assistant", content="reply1"),
        ChatMessage(role="user", content="second"),
        ChatMessage(role="assistant", content="reply2"),
    ]
    client = _SeqClient(["reply2-rerolled"])
    out = ic.regenerate_last(_BASE_IDEA, history, claude_client=client)
    assert out == "reply2-rerolled"
    # The earlier user/assistant pair must appear in the new prompt.
    assert "USER: first" in client.last_user
    assert "ASSISTANT: reply1" in client.last_user
    # The latest assistant being regenerated must NOT appear.
    assert "reply2" not in client.last_user.replace("reply2-rerolled", "")


def test_regenerate_last_does_not_mutate_history():
    history = [
        ChatMessage(role="user", content="u"),
        ChatMessage(role="assistant", content="a"),
    ]
    client = _SeqClient(["new"])
    ic.regenerate_last(_BASE_IDEA, history, claude_client=client)
    assert len(history) == 2
    assert history[-1].content == "a"


def test_regenerate_last_respects_mode_param():
    history = [
        ChatMessage(role="user", content="critique"),
        ChatMessage(role="assistant", content="weak."),
    ]
    client = _SeqClient(["even weaker."])
    ic.regenerate_last(_BASE_IDEA, history, claude_client=client,
                            mode="critic")
    assert client.last_system == ic._CRITIC_SYSTEM


# ─────────────────────────────────────────────────────────────────────────────
# estimate_tokens / estimate_turn_tokens
# ─────────────────────────────────────────────────────────────────────────────

def test_estimate_tokens_empty_string_is_zero():
    assert ic.estimate_tokens("") == 0
    assert ic.estimate_tokens(None) == 0  # type: ignore[arg-type]


def test_estimate_tokens_scales_with_length():
    short = ic.estimate_tokens("hello")
    long = ic.estimate_tokens("hello " * 100)
    assert long > short * 10


def test_estimate_turn_tokens_breakdown_sums_to_total():
    tok = ic.estimate_turn_tokens(
        _BASE_IDEA,
        [ChatMessage(role="user", content="a question that uses some tokens")],
        user_message="next message",
        mode="collaborator",
    )
    assert tok["total_input"] == (
        tok["system"] + tok["context"] + tok["history"] + tok["user"]
    )
    assert tok["system"] > 0
    assert tok["context"] > 0
    assert tok["user"] > 0


def test_estimate_turn_tokens_empty_history_and_message():
    tok = ic.estimate_turn_tokens(_BASE_IDEA, [], user_message="",
                                       mode="collaborator")
    assert tok["history"] == 0
    assert tok["user"] == 0
    assert tok["total_input"] == tok["system"] + tok["context"]


# ─────────────────────────────────────────────────────────────────────────────
# truncate_history
# ─────────────────────────────────────────────────────────────────────────────

def test_truncate_history_short_history_unchanged():
    history = [
        ChatMessage(role="user", content="a"),
        ChatMessage(role="assistant", content="b"),
    ]
    out, was = ic.truncate_history(history, max_turns=20, keep_recent=10)
    assert was is False
    assert len(out) == 2


def test_truncate_history_long_history_truncates_and_adds_summary():
    history = [
        ChatMessage(role=("user" if i % 2 == 0 else "assistant"),
                       content=f"msg{i}")
        for i in range(30)
    ]
    out, was = ic.truncate_history(history, max_turns=20, keep_recent=10)
    assert was is True
    # 10 most-recent + 1 synthetic summary = 11
    assert len(out) == 11
    assert out[0].role == "assistant"
    assert "elided" in out[0].content
    # Most recent message is preserved.
    assert out[-1].content == "msg29"


def test_truncate_history_does_not_mutate_input():
    history = [
        ChatMessage(role=("user" if i % 2 == 0 else "assistant"),
                       content=f"m{i}")
        for i in range(25)
    ]
    snapshot = [(m.role, m.content) for m in history]
    ic.truncate_history(history, max_turns=20, keep_recent=10)
    assert [(m.role, m.content) for m in history] == snapshot


def test_truncate_history_invalid_params():
    history = [ChatMessage(role="user", content="x")]
    with pytest.raises(ValueError):
        ic.truncate_history(history, max_turns=0)
    with pytest.raises(ValueError):
        ic.truncate_history(history, max_turns=10, keep_recent=-1)
    with pytest.raises(ValueError):
        ic.truncate_history(history, max_turns=5, keep_recent=10)


# ─────────────────────────────────────────────────────────────────────────────
# diff_ideas
# ─────────────────────────────────────────────────────────────────────────────

def test_diff_ideas_no_changes_returns_empty():
    same = dict(_BASE_IDEA)
    assert ic.diff_ideas(_BASE_IDEA, same) == {}


def test_diff_ideas_detects_changed_fields():
    updated = dict(_BASE_IDEA)
    updated["title"] = "New Title"
    updated["method"] = "Different method"
    d = ic.diff_ideas(_BASE_IDEA, updated)
    assert set(d.keys()) == {"title", "method"}
    assert d["title"]["before"] == _BASE_IDEA["title"]
    assert d["title"]["after"] == "New Title"


def test_diff_ideas_accepts_idea_dataclass():
    from models.idea import Idea
    updated = Idea(
        title=_BASE_IDEA["title"], motivation="changed!",
        method=_BASE_IDEA["method"], hypothesis=_BASE_IDEA["hypothesis"],
        resources="", expected_outcome="", risk_assessment="",
    )
    d = ic.diff_ideas(_BASE_IDEA, updated)
    assert "motivation" in d
    assert d["motivation"]["after"] == "changed!"


def test_diff_ideas_ignores_unset_fields_in_original():
    """Fields missing from original count as empty before-value."""
    original = {"title": "x", "method": "m", "hypothesis": "h"}
    updated = dict(original)
    updated["motivation"] = "now set"
    d = ic.diff_ideas(original, updated)
    assert "motivation" in d
    assert d["motivation"]["before"] == ""


def test_diff_ideas_invalid_original_raises():
    with pytest.raises(ValueError):
        ic.diff_ideas(None, {})  # type: ignore[arg-type]


def test_diff_ideas_invalid_updated_raises():
    with pytest.raises(ValueError):
        ic.diff_ideas(_BASE_IDEA, "not an idea")  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────────────
# export_markdown
# ─────────────────────────────────────────────────────────────────────────────

def test_export_markdown_empty_idea_raises():
    with pytest.raises(ValueError):
        ic.export_markdown({}, [], refined=None)


def test_export_markdown_basic_contains_title_and_mode():
    history = [
        ChatMessage(role="user", content="riskiest?"),
        ChatMessage(role="assistant", content="data scarcity"),
    ]
    md = ic.export_markdown(_BASE_IDEA, history, mode="critic")
    assert "# Chat to optimize" in md
    assert _BASE_IDEA["title"] in md
    assert "2 turn" in md
    assert ic.CHAT_MODES["critic"]["label"] in md
    assert "## Conversation" in md
    assert "### 1. You" in md
    assert "### 2. Assistant" in md
    assert "riskiest?" in md
    assert "data scarcity" in md


def test_export_markdown_includes_refined_section_when_provided():
    from models.idea import Idea
    refined = Idea(
        title="Refined T", motivation=_BASE_IDEA["motivation"],
        method="cheaper method", hypothesis=_BASE_IDEA["hypothesis"],
        resources="", expected_outcome="", risk_assessment="",
        source_strategy="V", generation=1,
        parent_title=_BASE_IDEA["title"],
    )
    refined.execution_meta = {"change_summary": "swapped to cheaper method"}
    history = [ChatMessage(role="user", content="cheaper please")]
    md = ic.export_markdown(_BASE_IDEA, history, refined=refined,
                                 mode="collaborator")
    assert "## Refined idea (crystallized)" in md
    assert "Refined T" in md
    assert "cheaper method" in md
    assert "swapped to cheaper method" in md
    assert "## What changed" in md
    assert "Title" in md  # field-level diff header
    # Unchanged fields not in diff section.
    assert "Hypothesis\n- **Before**" not in md


def test_export_markdown_handles_dict_refined():
    """`refined` accepts a dict too, not just an Idea."""
    refined = dict(_BASE_IDEA)
    refined["method"] = "new method"
    history = [ChatMessage(role="user", content="change method")]
    md = ic.export_markdown(_BASE_IDEA, history, refined=refined)
    assert "new method" in md
    assert "## What changed" in md
    assert "Method" in md


def test_export_markdown_no_refined_no_change_section():
    history = [ChatMessage(role="user", content="q")]
    md = ic.export_markdown(_BASE_IDEA, history)
    assert "## Refined idea" not in md
    assert "## What changed" not in md


def test_export_markdown_empty_history_renders_placeholder():
    md = ic.export_markdown(_BASE_IDEA, [])
    assert "_(no turns yet)_" in md


# ─────────────────────────────────────────────────────────────────────────────
# crystallize is mode-agnostic (separate from chat modes)
# ─────────────────────────────────────────────────────────────────────────────

def test_crystallize_unchanged_by_chat_mode_choice():
    """Crystallize uses its own dedicated system prompt; chat mode does
    not propagate into crystallize calls."""
    history = [
        ChatMessage(role="user", content="cheaper"),
        ChatMessage(role="assistant", content="ok"),
    ]
    payload = {"title": "T", "method": "M", "hypothesis": "H",
                "change_summary": "s"}
    client = _SeqClient([payload])
    ic.crystallize(_BASE_IDEA, history, claude_client=client)
    # Crystallize system prompt is its own — must not be a chat mode.
    assert client.last_system != ic._COLLAB_SYSTEM
    assert client.last_system != ic._CRITIC_SYSTEM
    assert "JSON" in client.last_system
