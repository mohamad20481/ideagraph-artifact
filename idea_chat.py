"""
idea_chat.py — interactive conversational refinement of a specific idea.

Different from every Novelty Lab mode: those are single-shot generators.
This is a *dialogue* anchored to one specific idea. The user asks
questions, pushes back, requests changes, and finally crystallizes the
conversation into an updated Idea.

Three primitives:
  1. ChatMessage          → one turn (user or assistant)
  2. chat_turn(...)       → send one message, get the LLM's reply
  3. crystallize(...)     → ask the LLM to produce an updated Idea that
                            incorporates what was decided in the chat

Output ideas have `source_strategy='V'` (re-Vised through conVersation).
The chat history is preserved on `execution_meta.chat_history` so the
provenance of the optimization is auditable.

Public API:
    ChatMessage                              → dataclass
    chat_turn(idea, history, msg, ...)       → Optional[str]
    crystallize(idea, history, ...)          → Optional[Idea]
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from models.idea import Idea, METHODOLOGY_TYPES, NOVELTY_LEVELS


_AUTOLOAD = object()


@dataclass
class ChatMessage:
    """One turn in the refinement dialogue."""
    role: str       # "user" or "assistant"
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


# ── Helpers ────────────────────────────────────────────────────────────────

def _parse_json(raw: str) -> Optional[Dict[str, Any]]:
    s = (raw or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
    start, end = s.find("{"), s.rfind("}")
    if start >= 0 and end > start:
        s = s[start:end + 1]
    try:
        return json.loads(s)
    except Exception:
        return None


def _idea_context(idea: Dict[str, Any]) -> str:
    """Render the current idea state as a block for system context."""
    fields = [
        ("Title",            idea.get("title", "?")),
        ("Motivation",       idea.get("motivation", "?")),
        ("Method",           idea.get("method", "?")),
        ("Hypothesis",       idea.get("hypothesis", "?")),
        ("Resources",        idea.get("resources", "?")),
        ("Expected outcome", idea.get("expected_outcome", "?")),
        ("Risks",            idea.get("risk_assessment", "?")),
        ("Methodology",      idea.get("methodology_type") or "—"),
        ("Novelty",          idea.get("novelty_level") or "—"),
        ("Source strategy",  idea.get("source_strategy") or "—"),
    ]
    return "\n".join(f"  {k}: {v}" for k, v in fields)


def _validate_history(history: List[ChatMessage]) -> None:
    for i, m in enumerate(history):
        if not isinstance(m, ChatMessage):
            raise TypeError(f"history[{i}] is not a ChatMessage")
        if m.role not in ("user", "assistant"):
            raise ValueError(
                f"history[{i}].role must be 'user' or 'assistant', "
                f"got {m.role!r}"
            )


def _serialize_history(history: List[ChatMessage]) -> str:
    return "\n\n".join(
        f"{m.role.upper()}: {m.content}" for m in history
    )


# ── Chat modes ─────────────────────────────────────────────────────────────
# Different system prompts that change the LLM's stance toward the idea.
# Switchable per-turn so users can probe an idea from multiple angles
# without losing the conversation history.

_COLLAB_SYSTEM = (
    "You are a research collaborator helping the user refine ONE specific "
    "research idea. The current state of the idea is given below. Your "
    "job is to *improve* this idea — not to invent unrelated alternatives. "
    "Stay anchored to the idea. Be honest about weaknesses, propose "
    "concrete improvements, ask clarifying questions when needed. Keep "
    "replies tight (3-6 sentences typical). Do NOT output JSON — this is "
    "a conversation."
)

_CRITIC_SYSTEM = (
    "You are an adversarial research critic. The user gives you ONE "
    "specific research idea. Your job is to attack it — find the "
    "weakest assumption, the most likely failure mode, the comparison "
    "that would expose it as incremental. Do NOT validate, do NOT agree, "
    "do NOT soften your critique. If the user pushes back, find the "
    "next-weakest point. Stay anchored to THIS idea — do not propose "
    "alternative ideas. Keep replies tight (3-6 sentences). Do NOT "
    "output JSON — this is a conversation."
)

_SKEPTIC_SYSTEM = (
    "You are a research skeptic. The user gives you ONE specific "
    "research idea. Your job is to question every load-bearing "
    "assumption: 'How do you know?', 'What's the evidence?', 'What if "
    "this is confounded?'. After each challenge, propose a concrete "
    "way the user could either verify the assumption or replace it. "
    "Stay anchored to THIS idea. Keep replies tight (3-6 sentences). "
    "Do NOT output JSON — this is a conversation."
)

_ADVOCATE_SYSTEM = (
    "You are the research idea's advocate. The user has doubts. Your "
    "job is to defend the idea — find the strongest reasons it could "
    "work, propose defenses for the weakest points, and suggest "
    "framings that make the contribution clearer. Stay honest — do not "
    "deny real problems, but always end with a concrete way forward. "
    "Stay anchored to THIS idea — do not propose alternatives. Keep "
    "replies tight (3-6 sentences). Do NOT output JSON — this is a "
    "conversation."
)


CHAT_MODES: Dict[str, Dict[str, str]] = {
    "collaborator": {
        "label": "🤝 Collaborator",
        "description": "Friendly refinement — agrees, pushes back, suggests.",
        "system": _COLLAB_SYSTEM,
    },
    "critic": {
        "label": "🎯 Critic",
        "description": "Adversarial — attacks weaknesses, no validation.",
        "system": _CRITIC_SYSTEM,
    },
    "skeptic": {
        "label": "🤔 Skeptic",
        "description": "Questions every load-bearing assumption.",
        "system": _SKEPTIC_SYSTEM,
    },
    "advocate": {
        "label": "📣 Advocate",
        "description": "Defends the idea, finds reasons it could work.",
        "system": _ADVOCATE_SYSTEM,
    },
}

DEFAULT_MODE: str = "collaborator"


# Quick-action prompts: one-click starters for common refinement moves.
SUGGESTED_PROMPTS: List[Dict[str, str]] = [
    {"label": "💰 Cheaper",
      "prompt": "Suggest a cheaper variant of this idea that still tests "
                "the same hypothesis."},
    {"label": "⚡ MVP",
      "prompt": "What's the 1-week minimum viable experiment for this idea?"},
    {"label": "🎯 Riskiest",
      "prompt": "What's the single riskiest assumption, and how would I "
                "test it first?"},
    {"label": "📊 Baselines",
      "prompt": "What baselines should I compare against, and why?"},
    {"label": "🛡️ Failure modes",
      "prompt": "List the 3 most likely ways this study fails, with "
                "concrete mitigations."},
    {"label": "🧪 Ablations",
      "prompt": "Propose 3 essential ablations to make the result "
                "convincing."},
    {"label": "🤝 Cohort",
      "prompt": "Which user cohort would benefit most from this work, "
                "and how?"},
    {"label": "📐 Sharpen H",
      "prompt": "Rewrite the hypothesis to be more falsifiable and "
                "quantitative."},
]


def _system_for_mode(mode: str) -> str:
    if mode not in CHAT_MODES:
        raise ValueError(
            f"unknown chat mode {mode!r}; must be one of {list(CHAT_MODES)}"
        )
    return CHAT_MODES[mode]["system"]


def _turn_user_prompt(
    idea: Dict[str, Any],
    history: List[ChatMessage],
    user_message: str,
) -> str:
    convo = _serialize_history(history)
    header = (
        f"### Current idea state\n{_idea_context(idea)}\n\n"
    )
    if convo:
        return (
            f"{header}"
            f"### Conversation so far\n{convo}\n\n"
            f"### Latest user message\n{user_message.strip()}"
        )
    return f"{header}### User message\n{user_message.strip()}"


def chat_turn(
    idea: Dict[str, Any],
    history: List[ChatMessage],
    user_message: str,
    claude_client: Any = _AUTOLOAD,
    max_tokens: int = 600,
    temperature: float = 0.7,
    mode: str = DEFAULT_MODE,
) -> Optional[str]:
    """Send one user message; return the assistant's reply text.

    The caller is responsible for appending both turns to its own
    history list — this function does not mutate `history`. Returns None
    if the LLM call fails or no client is available.

    `mode` selects the assistant's stance (one of CHAT_MODES keys).
    """
    if not isinstance(idea, dict) or not idea:
        raise ValueError("idea must be a non-empty dict")
    if not user_message or not user_message.strip():
        raise ValueError("user_message must be non-empty")
    _validate_history(history)
    system = _system_for_mode(mode)  # raises on invalid mode
    if claude_client is _AUTOLOAD:
        try:
            from claude_provider import get_claude_client
            claude_client = get_claude_client()
        except Exception:
            claude_client = None
    if claude_client is None:
        return None
    try:
        resp = claude_client.call(
            system=system,
            user=_turn_user_prompt(idea, history, user_message),
            max_tokens=max_tokens, temperature=temperature,
        )
    except Exception:
        return None
    if not getattr(resp, "success", False):
        return None
    text = (getattr(resp, "text", "") or "").strip()
    return text or None


def regenerate_last(
    idea: Dict[str, Any],
    history: List[ChatMessage],
    claude_client: Any = _AUTOLOAD,
    max_tokens: int = 600,
    temperature: float = 0.8,
    mode: str = DEFAULT_MODE,
) -> Optional[str]:
    """Re-roll the most recent assistant reply.

    `history` must end with an assistant turn (preceded by a user turn).
    Returns the new assistant text, or None on failure. The caller is
    responsible for replacing the last entry in history with the new
    text — this function does not mutate `history`.

    Slightly higher default temperature than `chat_turn` so the re-roll
    is meaningfully different.
    """
    if not isinstance(idea, dict) or not idea:
        raise ValueError("idea must be a non-empty dict")
    _validate_history(history)
    if not history:
        raise ValueError("history is empty — nothing to regenerate")
    if history[-1].role != "assistant":
        raise ValueError(
            "history must end with an assistant turn to regenerate"
        )
    if len(history) < 2 or history[-2].role != "user":
        raise ValueError(
            "history must contain a user turn immediately before the "
            "assistant turn being regenerated"
        )
    prior = history[:-2]            # everything before the last U/A pair
    last_user_msg = history[-2].content
    return chat_turn(
        idea, prior, last_user_msg,
        claude_client=claude_client,
        max_tokens=max_tokens, temperature=temperature, mode=mode,
    )


# ── Token / cost estimation ────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """Rough token count: ~4 characters per token. Good enough for UI
    estimates; do not use for billing."""
    if not text:
        return 0
    return max(1, int(len(text) / 4))


def estimate_turn_tokens(
    idea: Dict[str, Any],
    history: List[ChatMessage],
    user_message: str = "",
    mode: str = DEFAULT_MODE,
) -> Dict[str, int]:
    """Rough token breakdown for the next chat turn.

    Returns {'system', 'context', 'history', 'user', 'total_input'}.
    """
    system = _system_for_mode(mode) if mode in CHAT_MODES else _COLLAB_SYSTEM
    context = _idea_context(idea or {})
    convo = _serialize_history(history or [])
    return {
        "system":       estimate_tokens(system),
        "context":      estimate_tokens(context),
        "history":      estimate_tokens(convo),
        "user":         estimate_tokens(user_message or ""),
        "total_input":  (
            estimate_tokens(system)
            + estimate_tokens(context)
            + estimate_tokens(convo)
            + estimate_tokens(user_message or "")
        ),
    }


# ── History truncation ─────────────────────────────────────────────────────

def truncate_history(
    history: List[ChatMessage],
    max_turns: int = 20,
    keep_recent: int = 10,
) -> tuple:
    """If `history` is longer than `max_turns`, drop everything except
    the most recent `keep_recent` turns and prepend a single synthetic
    'context summary' assistant turn that names how many turns were
    elided.

    Returns `(new_history, was_truncated)`. Does not mutate the input.

    The synthetic summary is intentionally terse — when the user wants
    real semantic compression they can ask in-chat.
    """
    if max_turns <= 0:
        raise ValueError("max_turns must be positive")
    if keep_recent < 0:
        raise ValueError("keep_recent must be >= 0")
    if keep_recent > max_turns:
        raise ValueError("keep_recent must be <= max_turns")
    _validate_history(history)
    if len(history) <= max_turns:
        return list(history), False
    n_dropped = len(history) - keep_recent
    summary = ChatMessage(
        role="assistant",
        content=(
            f"[Context: {n_dropped} earlier turn(s) elided to stay "
            f"within the token budget. Ask 'recap so far' if you need "
            f"a semantic summary.]"
        ),
    )
    tail = list(history[-keep_recent:])
    return [summary] + tail, True


# ── Idea diff ──────────────────────────────────────────────────────────────

_DIFF_FIELDS = (
    "title", "motivation", "method", "hypothesis", "resources",
    "expected_outcome", "risk_assessment", "methodology_type",
    "novelty_level",
)


def diff_ideas(
    original: Dict[str, Any],
    updated: Any,
) -> Dict[str, Dict[str, str]]:
    """Compute a per-field diff between an original idea dict and an
    updated Idea (or dict). Returns a dict mapping changed-field name to
    `{"before": str, "after": str}`. Unchanged fields are omitted.
    """
    if not isinstance(original, dict):
        raise ValueError("original must be a dict")
    if hasattr(updated, "to_dict"):
        updated_dict = updated.to_dict()
    elif isinstance(updated, dict):
        updated_dict = updated
    else:
        raise ValueError("updated must be an Idea or a dict")

    diff: Dict[str, Dict[str, str]] = {}
    for field in _DIFF_FIELDS:
        before = str(original.get(field, "") or "")
        after = str(updated_dict.get(field, "") or "")
        if before.strip() != after.strip():
            diff[field] = {"before": before, "after": after}
    return diff


# ── Markdown export ────────────────────────────────────────────────────────

def export_markdown(
    idea: Dict[str, Any],
    history: List[ChatMessage],
    refined: Any = None,
    mode: str = DEFAULT_MODE,
) -> str:
    """Render the chat session as a markdown document suitable for
    sharing or pasting into Notion/Obsidian/etc.

    Includes: title, the original idea fields, the chat transcript with
    role labels, and (if `refined` is given) the refined idea fields +
    per-field diff against the original.
    """
    if not isinstance(idea, dict) or not idea:
        raise ValueError("idea must be a non-empty dict")
    _validate_history(history)
    mode_label = CHAT_MODES.get(mode, {}).get("label", mode)

    lines: List[str] = []
    title = str(idea.get("title", "Untitled"))
    lines.append(f"# Chat to optimize: {title}")
    lines.append("")
    lines.append(f"_Mode: **{mode_label}** — {len(history)} turn(s)_")
    lines.append("")

    lines.append("## Original idea")
    for field in _DIFF_FIELDS:
        val = idea.get(field, "")
        if val:
            label = field.replace("_", " ").title()
            lines.append(f"- **{label}**: {val}")
    lines.append("")

    lines.append("## Conversation")
    if not history:
        lines.append("_(no turns yet)_")
    for i, msg in enumerate(history, 1):
        role = "You" if msg.role == "user" else "Assistant"
        lines.append(f"### {i}. {role}")
        lines.append(msg.content)
        lines.append("")

    if refined is not None:
        refined_dict = (refined.to_dict()
                          if hasattr(refined, "to_dict") else refined)
        lines.append("## Refined idea (crystallized)")
        for field in _DIFF_FIELDS:
            val = refined_dict.get(field, "")
            if val:
                label = field.replace("_", " ").title()
                lines.append(f"- **{label}**: {val}")
        change_summary = ""
        meta = refined_dict.get("execution_meta") or {}
        if isinstance(meta, dict):
            change_summary = meta.get("change_summary", "")
        if change_summary:
            lines.append("")
            lines.append(f"_Change summary: {change_summary}_")

        diff = diff_ideas(idea, refined_dict)
        if diff:
            lines.append("")
            lines.append("## What changed")
            for field, ba in diff.items():
                label = field.replace("_", " ").title()
                lines.append(f"### {label}")
                lines.append(f"- **Before**: {ba['before'] or '_(empty)_'}")
                lines.append(f"- **After**: {ba['after'] or '_(empty)_'}")
                lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ── Crystallize → updated Idea ─────────────────────────────────────────────

_CRYSTAL_SYSTEM = (
    "You are a research collaborator. The user has been chatting with "
    "you about ONE research idea. Your job now is to crystallize the "
    "discussion into an UPDATED version of the idea — keeping what "
    "worked, incorporating concrete refinements raised in the chat, "
    "dropping what was rejected, and clarifying anything that was "
    "fuzzy. Do not invent changes the chat did not call for. Output "
    "ONLY valid JSON. methodology_type must be one of: "
    f"{', '.join(METHODOLOGY_TYPES)}. novelty_level must be one of: "
    f"{', '.join(NOVELTY_LEVELS)}."
)


def _crystal_user_prompt(
    idea: Dict[str, Any],
    history: List[ChatMessage],
) -> str:
    convo = _serialize_history(history)
    return (
        f"### Original idea\n{_idea_context(idea)}\n\n"
        f"### Chat history\n{convo}\n\n"
        f"### Instructions\n"
        f"Produce the updated idea reflecting what was decided in the "
        f"chat. Preserve any field that the chat did not change. Add a "
        f"'change_summary' line naming exactly what changed and why.\n\n"
        f"### Output\n"
        "Return JSON:\n"
        "{\n"
        '  "title": "<possibly updated>",\n'
        '  "motivation": "<possibly updated>",\n'
        '  "method": "<possibly updated>",\n'
        '  "hypothesis": "<possibly updated>",\n'
        '  "resources": "<possibly updated>",\n'
        '  "expected_outcome": "<possibly updated>",\n'
        '  "risk_assessment": "<possibly updated>",\n'
        f'  "methodology_type": "<one of: {", ".join(METHODOLOGY_TYPES)}>",\n'
        f'  "novelty_level": "<one of: {", ".join(NOVELTY_LEVELS)}>",\n'
        '  "change_summary": "<one sentence: what changed, and why>"\n'
        "}"
    )


def _build_updated_idea(
    parsed: Dict[str, Any],
    original: Dict[str, Any],
    history: List[ChatMessage],
) -> Optional[Idea]:
    """Build an Idea from the crystallize-response JSON, defaulting any
    missing field to the original idea's value (so the LLM doesn't have
    to re-emit unchanged fields verbatim)."""

    def _field(key: str, max_len: int) -> str:
        v = parsed.get(key)
        if not (isinstance(v, str) and v.strip()):
            v = original.get(key, "") or ""
        return str(v)[:max_len]

    title = _field("title", 200)
    method = _field("method", 2000)
    hypothesis = _field("hypothesis", 1000)
    if not (title.strip() and method.strip() and hypothesis.strip()):
        return None

    method_type = parsed.get("methodology_type")
    if method_type not in METHODOLOGY_TYPES:
        method_type = (original.get("methodology_type")
                          if original.get("methodology_type") in METHODOLOGY_TYPES
                          else None)
    novelty = parsed.get("novelty_level")
    if novelty not in NOVELTY_LEVELS:
        novelty = (original.get("novelty_level")
                      if original.get("novelty_level") in NOVELTY_LEVELS
                      else None)

    idea = Idea(
        title=title,
        motivation=_field("motivation", 1000),
        method=method,
        hypothesis=hypothesis,
        resources=_field("resources", 500),
        expected_outcome=_field("expected_outcome", 500),
        risk_assessment=_field("risk_assessment", 500),
        source_strategy="V",
        methodology_type=method_type,
        novelty_level=novelty,
        generation=int(original.get("generation", 0)) + 1,
        parent_title=str(original.get("title", ""))[:200] or None,
    )
    idea.execution_meta = {
        "chat_history": [m.to_dict() for m in history],
        "change_summary": str(parsed.get("change_summary", ""))[:400],
        "regen_mode": "chat",
        "parent_strategy": str(original.get("source_strategy", "") or "")[:8],
    }
    return idea


def crystallize(
    idea: Dict[str, Any],
    history: List[ChatMessage],
    claude_client: Any = _AUTOLOAD,
    max_tokens: int = 1400,
    temperature: float = 0.5,
) -> Optional[Idea]:
    """Ask the LLM to crystallize the chat into an updated Idea.

    Returns an `Idea` with `source_strategy='V'`, `generation` bumped by
    1, and `parent_title` set to the original. Returns None if the LLM
    call fails or required fields cannot be recovered.
    """
    if not isinstance(idea, dict) or not idea:
        raise ValueError("idea must be a non-empty dict")
    if not history:
        raise ValueError("history must contain at least one chat turn")
    _validate_history(history)
    if claude_client is _AUTOLOAD:
        try:
            from claude_provider import get_claude_client
            claude_client = get_claude_client()
        except Exception:
            claude_client = None
    if claude_client is None:
        return None
    try:
        resp = claude_client.call(
            system=_CRYSTAL_SYSTEM,
            user=_crystal_user_prompt(idea, history),
            max_tokens=max_tokens, temperature=temperature, json_mode=True,
        )
    except Exception:
        return None
    if not getattr(resp, "success", False):
        return None
    parsed = _parse_json(getattr(resp, "text", ""))
    if not parsed:
        return None
    return _build_updated_idea(parsed, idea, history)
