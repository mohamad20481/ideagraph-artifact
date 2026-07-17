"""
result_chat.py — conversational interface for a whole saved IdeaGraph result.

Different from `idea_chat.py` (which dialogs over ONE idea): this dialogs
over an ENTIRE saved result — the topic plus all generated ideas. Useful
for queries like:

  "Which 3 ideas are best for a 2-week timeline?"
  "Group these by methodology type, with one-line rationale each."
  "Find the two ideas that most contradict each other and explain."
  "Summarize the top 5 highest-coverage ideas in 2 sentences each."
  "Which of these would fit a paper for ICML 2026? Why?"

The chat is anchored to a `result_dict` (the same shape `db.load_result`
returns). History lives in `st.session_state` per result-id so users
can switch between results without losing context. A "💾 Save to
archive" button writes the transcript back to db.results_json for
durable storage.

Public API:
    ResultChatMessage                            → dataclass
    build_context(result_dict, ...) -> str       → builds the system prompt
    chat_turn(result_dict, history, user_msg, ...) -> Optional[str]
    render_chat_panel(st, result_dict, result_id) → Streamlit UI
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


_AUTOLOAD = object()


@dataclass
class ResultChatMessage:
    """One turn in the result-level dialogue."""
    role: str       # "user" or "assistant"
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


# ── Context shaping ─────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are an expert research assistant analyzing the results of a "
    "research-idea generation pipeline. You have access to:\n"
    "  • The research topic the user explored\n"
    "  • Every generated idea (title, motivation, method, hypothesis, "
    "expected outcome, risks, methodology type, novelty level)\n\n"
    "Your job is to help the user understand, compare, prioritize, and "
    "synthesize across these ideas. Be specific — cite idea titles when "
    "you reference them. Be honest about gaps and weaknesses. If the "
    "user asks for a ranking or recommendation, justify it with evidence "
    "from the idea content (not generic platitudes). Keep replies tight "
    "(under ~300 words unless the user explicitly asks for more).\n\n"
    "When the user asks about a specific idea by title, ground your "
    "answer in that idea's actual fields, not a summary. When the user "
    "asks for cross-idea analysis, name the specific ideas you're "
    "comparing."
)


def _shape_idea_compact(idea: Dict[str, Any], idx: int) -> str:
    """One-line compact form: '[N] Title — motivation excerpt'."""
    title = idea.get("title") or idea.get("name") or f"Idea {idx + 1}"
    motiv = (idea.get("motivation") or "").strip()
    if len(motiv) > 140:
        motiv = motiv[:137] + "…"
    return f"[{idx + 1}] {title}" + (f" — {motiv}" if motiv else "")


def _shape_idea_full(idea: Dict[str, Any], idx: int) -> str:
    """Full multi-field form. Used for ideas relevant to the current query."""
    title = idea.get("title") or idea.get("name") or f"Idea {idx + 1}"
    fields = [
        ("Title", title),
        ("Motivation", idea.get("motivation", "")),
        ("Method", idea.get("method", "")),
        ("Hypothesis", idea.get("hypothesis", "")),
        ("Expected outcome", idea.get("expected_outcome", "")),
        ("Risks", idea.get("risk_assessment", "")),
        ("Methodology", idea.get("methodology_type") or "—"),
        ("Novelty", idea.get("novelty_level") or "—"),
    ]
    body = "\n".join(
        f"  {k}: {v}" for k, v in fields if v and v != "—"
    )
    return f"[{idx + 1}] {title}\n{body}"


def _extract_ideas(result_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Pull the ideas list from a result. The pipeline emits multiple
    shapes over time — accept either `ideas` (newer) or `final_ideas`
    (older) or `idea_archive` (even older). Returns [] if none found."""
    for key in ("ideas", "final_ideas", "idea_archive", "all_ideas"):
        v = result_dict.get(key)
        if isinstance(v, list) and v:
            return [d for d in v if isinstance(d, dict)]
    return []


def _ideas_mentioned_in_query(
    ideas: List[Dict[str, Any]], query: str,
) -> List[int]:
    """Return indices of ideas whose title appears as a substring in the
    user query (case-insensitive). Used to decide which ideas get the
    full multi-field treatment vs the compact one-liner."""
    q = (query or "").lower()
    hit = []
    for i, idea in enumerate(ideas):
        title = (idea.get("title") or idea.get("name") or "").strip().lower()
        if title and len(title) >= 4 and title in q:
            hit.append(i)
    return hit


def build_context(
    result_dict: Dict[str, Any],
    user_query: str = "",
    max_full_ideas: int = 6,
) -> str:
    """Render the result as an LLM-friendly context block.

    Strategy: emit a 1-line summary for EVERY idea (so the LLM knows
    what's available), plus the full multi-field block for any idea
    whose title is mentioned in the user query (capped at `max_full_ideas`
    to keep the prompt under control).
    """
    topic = (result_dict.get("topic") or "").strip() or "Unknown topic"
    coverage = result_dict.get("coverage")
    ideas = _extract_ideas(result_dict)

    parts = [
        f"Research topic: {topic}",
    ]
    if isinstance(coverage, (int, float)):
        parts.append(f"Pipeline coverage: {coverage:.0%}")
    parts.append(f"Total ideas generated: {len(ideas)}")

    if not ideas:
        parts.append("\n(No ideas were found in this result.)")
        return "\n".join(parts)

    parts.append("\n— All ideas (1-line summary) —")
    for i, idea in enumerate(ideas):
        parts.append(_shape_idea_compact(idea, i))

    focus = _ideas_mentioned_in_query(ideas, user_query)[:max_full_ideas]
    if focus:
        parts.append("\n— Full detail for ideas mentioned in your question —")
        for i in focus:
            parts.append(_shape_idea_full(ideas[i], i))

    return "\n".join(parts)


# ── Chat turn ──────────────────────────────────────────────────────────────

def _validate_history(history: List[ResultChatMessage]) -> None:
    if history is None:
        return
    for m in history:
        if not isinstance(m, ResultChatMessage):
            raise TypeError(
                f"history entries must be ResultChatMessage, got {type(m).__name__}"
            )
        if m.role not in ("user", "assistant"):
            raise ValueError(f"invalid role {m.role!r}")


def _history_block(history: List[ResultChatMessage], max_turns: int = 12) -> str:
    """Render up to the last `max_turns` turns as a transcript block."""
    if not history:
        return ""
    recent = history[-max_turns:]
    lines = []
    for m in recent:
        speaker = "User" if m.role == "user" else "Assistant"
        lines.append(f"{speaker}: {m.content}")
    return "\n\n".join(lines)


def chat_turn(
    result_dict: Dict[str, Any],
    history: List[ResultChatMessage],
    user_message: str,
    claude_client: Any = _AUTOLOAD,
    max_tokens: int = 700,
    temperature: float = 0.6,
) -> Optional[str]:
    """Send one user message; return the assistant's reply text.

    Caller appends both turns to history after this returns. Returns
    None if the LLM call fails or no client is available — caller should
    surface a "couldn't reach the model" message.
    """
    if not isinstance(result_dict, dict) or not result_dict:
        raise ValueError("result_dict must be a non-empty dict")
    if not user_message or not user_message.strip():
        raise ValueError("user_message must be non-empty")
    _validate_history(history)

    if claude_client is _AUTOLOAD:
        try:
            from claude_provider import get_claude_client
            claude_client = get_claude_client()
        except Exception:
            claude_client = None
    if claude_client is None:
        return None

    ctx = build_context(result_dict, user_query=user_message)
    transcript = _history_block(history or [])
    user_prompt = (
        f"=== RESULT CONTEXT ===\n{ctx}\n\n"
        + (f"=== CONVERSATION SO FAR ===\n{transcript}\n\n" if transcript else "")
        + f"=== NEW QUESTION ===\n{user_message.strip()}"
    )

    try:
        resp = claude_client.call(
            system=_SYSTEM_PROMPT,
            user=user_prompt,
            max_tokens=max_tokens, temperature=temperature,
        )
    except Exception:
        return None
    if not getattr(resp, "success", False):
        return None
    text = (getattr(resp, "text", "") or "").strip()
    return text or None


# ── Persistence helpers ────────────────────────────────────────────────────

def history_to_jsonable(
    history: List[ResultChatMessage],
) -> List[Dict[str, str]]:
    """Pack history for db storage (JSON-safe)."""
    return [m.to_dict() for m in (history or [])]


def history_from_jsonable(
    data: Any,
) -> List[ResultChatMessage]:
    """Inverse: hydrate from db storage. Tolerates missing/bad rows."""
    if not isinstance(data, list):
        return []
    out: List[ResultChatMessage] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        role = row.get("role")
        content = row.get("content")
        if role in ("user", "assistant") and isinstance(content, str):
            out.append(ResultChatMessage(role=role, content=content))
    return out


# ── Idea-mention extraction (which ideas were touched in the conversation) ─

def extract_mentioned_idea_indices(
    history: List[ResultChatMessage],
    ideas: List[Dict[str, Any]],
    min_title_len: int = 4,
) -> List[int]:
    """Return sorted unique indices of ideas whose titles appear anywhere
    in the chat transcript (user or assistant turns).

    Titles shorter than `min_title_len` chars are ignored to avoid false
    matches on common words like "GNN" or "RL" that might appear in
    unrelated prose. The result preserves discovery order — first time
    the idea was mentioned wins, which is what the UI displays.
    """
    if not history or not ideas:
        return []
    transcript_lc = "\n".join(
        (m.content or "").lower() for m in history
    )
    seen: List[int] = []
    for i, idea in enumerate(ideas):
        title = (idea.get("title") or idea.get("name") or "").strip().lower()
        if len(title) < min_title_len:
            continue
        if title in transcript_lc and i not in seen:
            seen.append(i)
    return seen


# ── Streamlit UI ───────────────────────────────────────────────────────────

def render_chat_panel(
    st_module,
    result_dict: Optional[Dict[str, Any]],
    result_id: Optional[int] = None,
) -> None:
    """Draw the chat-with-result panel in the main area.

    Two-column layout:
      • Left (wider): conversation transcript + input
      • Right (narrower): clickable index of every idea in the result,
        with a 💬 badge highlighting ideas the chat has already touched,
        and per-idea **Ask** / **View** buttons for one-click navigation.

    Per-result history lives in `st.session_state["_result_chat:<id>"]`
    so switching results in the sidebar preserves each transcript
    independently.
    """
    if not result_dict:
        st_module.info(
            "💬 Load a saved result from the sidebar to chat with it. "
            "You can ask things like *“which 3 ideas best fit a 2-week "
            "timeline?”* or *“find the ideas that most contradict each "
            "other.”*"
        )
        return

    topic = (result_dict.get("topic") or "").strip() or "this result"
    ideas = _extract_ideas(result_dict)

    # ── One-shot jump-to toast (set when a title was just clicked) ──────
    # Pop so it shows exactly once. The Ideas tab will pick up the
    # _jump_to_idea_* keys to auto-expand the matching idea.
    _toast = st_module.session_state.pop("_jump_toast", None)
    if _toast:
        st_module.success(_toast)

    st_module.markdown(
        f"<div style='font-size:13px;color:#475569;margin-bottom:8px'>"
        f"💬 Chatting about <b>{topic[:80]}</b> "
        f"<span style='color:#94a3b8'>· {len(ideas)} ideas</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # History keyed per-result so switching results doesn't bleed history.
    key = f"_result_chat:{result_id or 'current'}"
    history: List[ResultChatMessage] = st_module.session_state.get(key) or []

    # ── Auto-restore from db if nothing in session_state yet ────────────
    # First time the panel opens (or after logout/reload), session_state
    # is empty — pull the saved current session for this user+result so
    # the chat survives reloads and account switches.
    user_id_for_db = st_module.session_state.get("user_id")
    restore_key = f"_result_chat_restored:{result_id}"
    if (not history) and user_id_for_db \
            and not st_module.session_state.get(restore_key):
        try:
            import db as _db_mod
            saved = _db_mod.load_current_chat_session(
                user_id=int(user_id_for_db),
                result_id=int(result_id) if result_id else None,
            )
            if saved and saved.get("history"):
                history = history_from_jsonable(saved["history"])
                st_module.session_state[key] = history
                st_module.session_state[
                    f"_rc_db_session_id:{result_id}"
                ] = saved["id"]
        except Exception:
            pass
        # Mark as attempted so we don't re-query db on every rerun.
        st_module.session_state[restore_key] = True

    # Which ideas were mentioned in any chat turn? The index panel shows
    # a 💬 badge on these so the user can scan what's already been discussed.
    mentioned_indices = set(
        extract_mentioned_idea_indices(history, ideas)
    )

    # ── Two-column layout: chat (left) + idea index (right) ──────────────
    chat_col, idx_col = st_module.columns([3, 2], gap="medium")

    # ════════════════ Idea index (right column) ══════════════════════════
    with idx_col:
        _render_idea_index(
            st_module, ideas, mentioned_indices, result_id,
        )

    # ════════════════ Chat (left column) ═════════════════════════════════
    with chat_col:
        # ── Suggested prompts (cold-start UX) ────────────────────────────
        if not history:
            st_module.caption("Try one of these to get started:")
            sugg_cols = st_module.columns(2)
            suggestions = [
                "📊 Summarize the top 3 ideas in 2 lines each.",
                "⚖️ Which 2 ideas most contradict each other?",
                "⏳ Which 3 ideas fit a 2-week timeline best?",
                "🎯 Rank all ideas by novelty level.",
            ]
            for i, s in enumerate(suggestions):
                with sugg_cols[i % 2]:
                    if st_module.button(
                        s, key=f"rc_sugg_{result_id}_{i}",
                        use_container_width=True,
                    ):
                        st_module.session_state[f"_rc_pending_{result_id}"] = s
                        st_module.rerun()

        # ── Transcript ───────────────────────────────────────────────────
        for m in history:
            if m.role == "user":
                st_module.chat_message("user").markdown(m.content)
            else:
                st_module.chat_message("assistant").markdown(m.content)

        # ── "Just mentioned" chip row (one-click jump for newly named ideas)
        # Show below the transcript if the last assistant turn referenced
        # any ideas — gives a fast click target without scrolling the index.
        if history and history[-1].role == "assistant":
            last_mentions = extract_mentioned_idea_indices(
                history[-1:], ideas,
            )
            if last_mentions:
                _render_mentions_chips(
                    st_module, ideas, last_mentions, result_id,
                )

        # ── Input ────────────────────────────────────────────────────────
        # Pull pending message from suggestion-button or idea-card click.
        pending = st_module.session_state.pop(
            f"_rc_pending_{result_id}", None,
        )
        user_input = (
            pending
            if pending
            else st_module.chat_input(
                "Ask anything about these ideas…",
                key=f"rc_input_{result_id}",
            )
        )

        if user_input:
            history.append(ResultChatMessage(role="user", content=user_input))
            st_module.session_state[key] = history
            st_module.chat_message("user").markdown(user_input)
            with st_module.chat_message("assistant"):
                with st_module.spinner("Thinking…"):
                    reply = chat_turn(result_dict, history[:-1], user_input)
                if reply:
                    st_module.markdown(reply)
                    history.append(
                        ResultChatMessage(role="assistant", content=reply)
                    )
                    st_module.session_state[key] = history
                else:
                    st_module.error(
                        "Couldn't reach the model. Check that an LLM "
                        "provider is configured in the Admin Dashboard → "
                        "🔌 LLM Provider tab."
                    )

            # ── Auto-persist to db after every successful turn ───────
            # Survives reloads, logout, account switches. Errors are
            # swallowed — failing to save chat history shouldn't break
            # the chat experience itself.
            _auto_save_chat(
                st_module, history, result_id, topic,
            )

        # ── Controls ─────────────────────────────────────────────────────
        if history:
            c1, c2, c3 = st_module.columns([1, 1, 1])
            if c1.button(
                "🗑️ Clear chat", key=f"rc_clear_{result_id}",
                use_container_width=True,
                help="Wipe both the in-memory transcript and the auto-saved db row",
            ):
                st_module.session_state[key] = []
                _clear_current_chat(st_module, result_id)
                st_module.rerun()
            if c2.button(
                "💾 Save snapshot",
                key=f"rc_snap_{result_id}",
                use_container_width=True,
                help="Save a named snapshot you can come back to later",
            ):
                st_module.session_state[
                    f"_rc_show_snap_input:{result_id}"
                ] = True
                st_module.rerun()
            if c3.button(
                "📥 Export JSON",
                key=f"rc_export_{result_id}",
                use_container_width=True,
                help="Download the full transcript as a JSON file",
            ):
                blob = json.dumps(
                    {"topic": topic, "history": history_to_jsonable(history)},
                    indent=2, ensure_ascii=False,
                )
                st_module.download_button(
                    "⬇️ Download chat.json",
                    data=blob.encode("utf-8"),
                    file_name=f"chat_result_{result_id or 'current'}.json",
                    mime="application/json",
                    key=f"rc_dl_{result_id}",
                )

            # ── Snapshot name input (revealed by 💾 Save snapshot click) ─
            if st_module.session_state.get(
                f"_rc_show_snap_input:{result_id}"
            ):
                _snap_title = st_module.text_input(
                    "Snapshot name (optional)",
                    placeholder="e.g. 'after first pass', 'top 3 picks'",
                    key=f"rc_snap_title_{result_id}",
                )
                sb1, sb2 = st_module.columns([1, 1])
                if sb1.button(
                    "💾 Save",
                    key=f"rc_snap_save_{result_id}",
                    type="primary",
                    use_container_width=True,
                ):
                    ok = _save_snapshot(
                        st_module, history, result_id, topic, _snap_title,
                    )
                    st_module.session_state[
                        f"_rc_show_snap_input:{result_id}"
                    ] = False
                    if ok:
                        st_module.success(
                            f"✅ Snapshot saved "
                            f"({_snap_title.strip() or 'unnamed'})"
                        )
                    st_module.rerun()
                if sb2.button(
                    "Cancel",
                    key=f"rc_snap_cancel_{result_id}",
                    use_container_width=True,
                ):
                    st_module.session_state[
                        f"_rc_show_snap_input:{result_id}"
                    ] = False
                    st_module.rerun()

        # ── Saved snapshots list (always shown, even with empty current) ─
        _render_saved_snapshots(st_module, result_id, key)


# ── Idea index (right-column navigator) ────────────────────────────────────

def _render_idea_index(
    st_module,
    ideas: List[Dict[str, Any]],
    mentioned_indices: set,
    result_id: Optional[int],
) -> None:
    """Right-column scrollable list of every idea in the result. Each
    card shows the title + a 1-line motivation snippet, a 💬 badge if
    the idea has been referenced in any chat turn so far, and two
    one-click buttons:

      🔍 Ask  → seeds the next message as "Tell me more about <title>"
      👁 View → toggles an inline expander showing the full idea body
    """
    st_module.markdown(
        "<div style='font-size:11px;color:#0369a1;font-weight:700;"
        "text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px'>"
        "📌 Ideas in this result</div>",
        unsafe_allow_html=True,
    )

    if not ideas:
        st_module.caption("No ideas to navigate.")
        return

    n_mentioned = len(mentioned_indices)
    if n_mentioned:
        st_module.caption(
            f"💬 {n_mentioned} of {len(ideas)} mentioned in chat so far"
        )
    else:
        st_module.caption(f"{len(ideas)} ideas · click 🔍 to ask about one")

    # ── Filter: All / Mentioned only ────────────────────────────────────
    show_mentioned_only = False
    if n_mentioned:
        show_mentioned_only = st_module.toggle(
            "Show only mentioned",
            value=False,
            key=f"rc_idx_filter_{result_id}",
        )

    # ── Scrollable card list inside a fixed-height container ────────────
    visible: List[int] = list(range(len(ideas)))
    if show_mentioned_only:
        visible = [i for i in visible if i in mentioned_indices]

    container = st_module.container(height=480)
    with container:
        for i in visible:
            idea = ideas[i]
            title = idea.get("title") or idea.get("name") or f"Idea {i + 1}"
            motiv = (idea.get("motivation") or "").strip()
            snippet = motiv[:100] + ("…" if len(motiv) > 100 else "")
            is_hot = i in mentioned_indices

            # Card-top metadata strip: 💬 badge + colored marker. The TITLE
            # itself lives on the next line as a clickable button so users
            # can jump straight to the Ideas tab via title click.
            badge_html = (
                "<span style='background:#0ea5e9;color:white;"
                "font-size:9px;font-weight:700;padding:1px 6px;"
                "border-radius:8px'>💬 chat</span>"
                if is_hot else ""
            )
            border_color = "#0ea5e9" if is_hot else "#e0f2fe"
            st_module.markdown(
                f"<div style='border-left:3px solid {border_color};"
                f"padding-left:8px;margin-top:8px;'>"
                f"<span style='color:#94a3b8;font-size:11px;"
                f"font-weight:600'>#{i + 1}</span> {badge_html}"
                + (
                    f"<div style='font-size:11px;color:#64748b;"
                    f"margin-top:2px;line-height:1.4'>{snippet}</div>"
                    if snippet else ""
                )
                + "</div>",
                unsafe_allow_html=True,
            )
            # Title-as-button → jump to Ideas tab. Full-width primary
            # action so the title remains the dominant click target on
            # each card (matches what the user asked for: "click on
            # title of result chat to go to idea result").
            _title_display = title if len(title) <= 60 else title[:57] + "…"
            if st_module.button(
                f"📍 {_title_display}",
                key=f"rc_title_{result_id}_{i}",
                use_container_width=True,
                help="Open this idea in the Ideas tab (auto-expanded)",
            ):
                st_module.session_state["_jump_to_idea_title"] = title
                st_module.session_state["_jump_to_idea_idx"] = i
                # Surface a one-shot toast so the user knows to switch tabs.
                st_module.session_state["_jump_toast"] = (
                    f"📍 Jumped to **{title[:60]}** — click the "
                    f"**Ideas** tab above to view it (auto-expanded)."
                )
                st_module.rerun()

            # Action buttons (Ask / View) — secondary actions below title.
            b1, b2 = st_module.columns(2)
            if b1.button(
                "🔍 Ask",
                key=f"rc_ask_{result_id}_{i}",
                use_container_width=True,
                help="Send 'Tell me more about this idea' as the next message",
            ):
                st_module.session_state[f"_rc_pending_{result_id}"] = (
                    f"Tell me more about: **{title}** — why is it "
                    f"interesting, what's the main risk, and how does "
                    f"it compare to the other ideas here?"
                )
                st_module.rerun()
            view_key = f"_rc_view_{result_id}_{i}"
            currently_viewing = bool(
                st_module.session_state.get(view_key, False)
            )
            if b2.button(
                "👁 Hide" if currently_viewing else "👁 View",
                key=f"rc_view_btn_{result_id}_{i}",
                use_container_width=True,
                help="Show / hide the full idea body inline",
            ):
                st_module.session_state[view_key] = not currently_viewing
                st_module.rerun()
            if currently_viewing:
                _render_idea_full_inline(st_module, idea, i)


def _render_idea_full_inline(
    st_module, idea: Dict[str, Any], idx: int,
) -> None:
    """Inline expansion of an idea's full body. Used by the 👁 View toggle
    on each idea card in the index."""
    fields = [
        ("Motivation",       idea.get("motivation")),
        ("Method",           idea.get("method")),
        ("Hypothesis",       idea.get("hypothesis")),
        ("Expected outcome", idea.get("expected_outcome")),
        ("Risks",            idea.get("risk_assessment")),
        ("Resources",        idea.get("resources")),
        ("Methodology",      idea.get("methodology_type")),
        ("Novelty level",    idea.get("novelty_level")),
    ]
    rendered = []
    for label, value in fields:
        if not value:
            continue
        rendered.append(
            f"<div style='font-size:10px;color:#0369a1;font-weight:700;"
            f"text-transform:uppercase;letter-spacing:0.05em;"
            f"margin-top:6px'>{label}</div>"
            f"<div style='font-size:11px;color:#334155;line-height:1.4'>"
            f"{value}</div>"
        )
    st_module.markdown(
        "<div style='background:#f8fafc;border:1px solid #e2e8f0;"
        "border-radius:6px;padding:8px 10px;margin-bottom:8px'>"
        + ("".join(rendered) if rendered else
           "<i>(no fields to display)</i>")
        + "</div>",
        unsafe_allow_html=True,
    )


def _render_mentions_chips(
    st_module,
    ideas: List[Dict[str, Any]],
    mention_indices: List[int],
    result_id: Optional[int],
) -> None:
    """After an assistant turn, surface a row of chips for every idea
    title the assistant mentioned, with a one-click "ask about this"
    button per chip. Avoids forcing the user to hunt the index for
    something the model just pointed at."""
    if not mention_indices:
        return
    st_module.markdown(
        "<div style='font-size:11px;color:#0369a1;font-weight:700;"
        "text-transform:uppercase;letter-spacing:0.05em;margin:8px 0 4px 0'>"
        "💡 Ideas just mentioned</div>",
        unsafe_allow_html=True,
    )
    cols = st_module.columns(min(3, len(mention_indices)))
    for col_i, idx in enumerate(mention_indices[:9]):  # cap at 9 chips
        idea = ideas[idx]
        title = idea.get("title") or idea.get("name") or f"Idea {idx + 1}"
        col = cols[col_i % len(cols)]
        with col:
            if st_module.button(
                f"#{idx + 1} {title[:30]}",
                key=f"rc_chip_{result_id}_{idx}",
                use_container_width=True,
                help="Ask a follow-up about this idea",
            ):
                st_module.session_state[f"_rc_pending_{result_id}"] = (
                    f"Tell me more about: **{title}**"
                )
                st_module.rerun()


# ── DB persistence helpers (used by render_chat_panel) ─────────────────────

def _auto_save_chat(
    st_module,
    history: List[ResultChatMessage],
    result_id: Optional[int],
    topic: str,
) -> None:
    """Upsert the current chat session to db. No-op if anonymous user or
    db unreachable. Stores the session id in session_state so subsequent
    saves UPDATE in place rather than re-querying."""
    user_id = st_module.session_state.get("user_id")
    if not user_id:
        return
    try:
        import db as _db_mod
        sid_key = f"_rc_db_session_id:{result_id}"
        sid = st_module.session_state.get(sid_key)
        new_id = _db_mod.save_chat_session(
            user_id=int(user_id),
            history=history_to_jsonable(history),
            result_id=int(result_id) if result_id else None,
            title=(topic[:80] if topic else ""),
            is_snapshot=False,
            session_id=int(sid) if sid else None,
        )
        st_module.session_state[sid_key] = new_id
    except Exception:
        # Persistence failure shouldn't break the chat. Silent fallback —
        # the user still has the in-memory transcript.
        pass


def _save_snapshot(
    st_module,
    history: List[ResultChatMessage],
    result_id: Optional[int],
    topic: str,
    snap_title: str,
) -> bool:
    """Save a named snapshot row (insert-only — doesn't overwrite). Returns
    True on success."""
    user_id = st_module.session_state.get("user_id")
    if not user_id:
        st_module.error("Sign in to save snapshots.")
        return False
    if not history:
        st_module.warning("Nothing to save — start a conversation first.")
        return False
    try:
        import db as _db_mod
        label = (snap_title or "").strip() or topic[:60] or "Snapshot"
        _db_mod.save_chat_session(
            user_id=int(user_id),
            history=history_to_jsonable(history),
            result_id=int(result_id) if result_id else None,
            title=label,
            is_snapshot=True,
        )
        return True
    except Exception as e:
        st_module.error(f"Couldn't save snapshot: {e}")
        return False


def _clear_current_chat(
    st_module, result_id: Optional[int],
) -> None:
    """Delete the auto-saved current-session row for this result so the
    next reload starts fresh (matches what 🗑️ Clear visually does)."""
    user_id = st_module.session_state.get("user_id")
    if not user_id:
        return
    sid_key = f"_rc_db_session_id:{result_id}"
    sid = st_module.session_state.pop(sid_key, None)
    if not sid:
        return
    try:
        import db as _db_mod
        _db_mod.delete_chat_session(int(sid), int(user_id))
    except Exception:
        pass


def _render_saved_snapshots(
    st_module, result_id: Optional[int], session_state_key: str,
) -> None:
    """List of named snapshots for THIS result, with Load + Delete
    actions per row. Collapsed by default so it doesn't dominate the UI."""
    user_id = st_module.session_state.get("user_id")
    if not user_id:
        return
    try:
        import db as _db_mod
        snaps = _db_mod.list_chat_sessions(
            user_id=int(user_id),
            result_id=int(result_id) if result_id else None,
            snapshots_only=True,
            limit=50,
        )
    except Exception:
        return
    if not snaps:
        return
    with st_module.expander(
        f"📂 Saved snapshots ({len(snaps)})",
        expanded=False,
    ):
        st_module.caption(
            "Each snapshot is a frozen copy of the chat at the time you "
            "clicked 💾 Save. Loading a snapshot replaces the current "
            "transcript — your current chat is auto-saved separately, "
            "so this doesn't lose work."
        )
        for snap in snaps:
            sid = snap["id"]
            label = snap.get("title") or "Snapshot"
            mc = snap.get("message_count") or 0
            updated = (snap.get("updated_at") or "")[:16]
            row_cols = st_module.columns([5, 1, 1])
            row_cols[0].markdown(
                f"**{label[:60]}**  \n"
                f"<span style='color:#94a3b8;font-size:11px'>"
                f"{mc} messages · {updated}</span>",
                unsafe_allow_html=True,
            )
            if row_cols[1].button(
                "📂 Load",
                key=f"rc_snap_load_{result_id}_{sid}",
                use_container_width=True,
                help="Replace current chat with this snapshot",
            ):
                try:
                    import db as _db_mod
                    loaded = _db_mod.load_chat_session(
                        int(sid), int(user_id),
                    )
                    if loaded and loaded.get("history"):
                        st_module.session_state[session_state_key] = (
                            history_from_jsonable(loaded["history"])
                        )
                        st_module.success(f"Loaded snapshot: {label[:40]}")
                        st_module.rerun()
                except Exception as e:
                    st_module.error(f"Load failed: {e}")
            if row_cols[2].button(
                "🗑️",
                key=f"rc_snap_del_{result_id}_{sid}",
                use_container_width=True,
                help="Permanently delete this snapshot",
            ):
                try:
                    import db as _db_mod
                    _db_mod.delete_chat_session(
                        int(sid), int(user_id),
                    )
                    st_module.rerun()
                except Exception as e:
                    st_module.error(f"Delete failed: {e}")
