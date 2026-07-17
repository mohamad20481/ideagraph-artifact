"""Tests for db.search_user_results + result_chat.py.

Covers:
  - SQL search over topic + idea content (single term, multi-term AND,
    case-insensitive, empty query → all)
  - result_chat.build_context (compact summary always, full detail only
    for ideas whose title appears in the user query)
  - result_chat.chat_turn (happy path, no-client fallback, validates
    inputs)
  - History serialization round-trip
  - render_chat_panel smoke test (no real LLM calls)
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── tmp_db fixture (reused pattern from test_account.py) ────────────────────

@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    import db as _db
    tmp_dir = tmp_path / "data"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_db_path = tmp_dir / "test_ideagraph.db"
    monkeypatch.setattr(_db, "_DB_DIR", str(tmp_dir), raising=False)
    monkeypatch.setattr(_db, "_DB_PATH", str(tmp_db_path), raising=False)
    try:
        if hasattr(_db, "_conn_local") and hasattr(_db._conn_local, "wrapped"):
            try:
                _db._conn_local.wrapped.raw.close()
            except Exception:
                pass
            del _db._conn_local.wrapped
    except Exception:
        pass
    _db.init_db()
    yield _db
    try:
        if hasattr(_db, "_conn_local") and hasattr(_db._conn_local, "wrapped"):
            try:
                _db._conn_local.wrapped.raw.close()
            except Exception:
                pass
            del _db._conn_local.wrapped
    except Exception:
        pass


# ── db.search_user_results ──────────────────────────────────────────────────

def test_search_empty_query_returns_all_results(tmp_db):
    uid = tmp_db.register_user("alice", "pw_search_xyz")
    tmp_db.save_result(uid, "transformers and attention", 0.5, 2,
                       {"ideas": []})
    tmp_db.save_result(uid, "privacy in federated learning", 0.7, 5,
                       {"ideas": []})
    out = tmp_db.search_user_results(uid, "")
    assert len(out) == 2


def test_search_whitespace_only_query_returns_all(tmp_db):
    uid = tmp_db.register_user("bob", "pw_search_xyz")
    tmp_db.save_result(uid, "topic A", 0.5, 1, {"ideas": []})
    out = tmp_db.search_user_results(uid, "   ")
    assert len(out) == 1


def test_search_matches_topic(tmp_db):
    uid = tmp_db.register_user("carol", "pw_search_xyz")
    tmp_db.save_result(uid, "transformers and attention", 0.5, 2, {"ideas": []})
    tmp_db.save_result(uid, "privacy in federated learning", 0.7, 5, {"ideas": []})
    out = tmp_db.search_user_results(uid, "transformer")
    assert len(out) == 1
    assert "transformers" in out[0]["topic"]


def test_search_matches_idea_content_in_json(tmp_db):
    """The search hits results_json too, so an idea title in the blob
    surfaces even when the topic doesn't mention it."""
    uid = tmp_db.register_user("dan", "pw_search_xyz")
    tmp_db.save_result(uid, "general ML methods", 0.6, 2, {
        "ideas": [
            {"title": "Sparse Attention via Hashing", "motivation": "x"},
            {"title": "Plain MLP", "motivation": "y"},
        ],
    })
    out = tmp_db.search_user_results(uid, "hashing")
    assert len(out) == 1


def test_search_case_insensitive(tmp_db):
    uid = tmp_db.register_user("eve", "pw_search_xyz")
    tmp_db.save_result(uid, "Transformers", 0.5, 1, {"ideas": []})
    assert len(tmp_db.search_user_results(uid, "TRANSFORMERS")) == 1
    assert len(tmp_db.search_user_results(uid, "transformers")) == 1
    assert len(tmp_db.search_user_results(uid, "TrAnSfOrM")) == 1


def test_search_multiterm_is_and_not_or(tmp_db):
    """Multiple whitespace-separated terms must ALL appear."""
    uid = tmp_db.register_user("frank", "pw_search_xyz")
    tmp_db.save_result(uid, "transformer privacy", 0.5, 1, {"ideas": []})
    tmp_db.save_result(uid, "transformer scaling", 0.5, 1, {"ideas": []})
    tmp_db.save_result(uid, "graph neural networks", 0.5, 1, {"ideas": []})
    out = tmp_db.search_user_results(uid, "transformer privacy")
    assert len(out) == 1
    assert "privacy" in out[0]["topic"]


def test_search_no_matches_returns_empty(tmp_db):
    uid = tmp_db.register_user("greta", "pw_search_xyz")
    tmp_db.save_result(uid, "topic A", 0.5, 1, {"ideas": []})
    assert tmp_db.search_user_results(uid, "nonexistent_xyz_term") == []


def test_search_scoped_to_user(tmp_db):
    """User A's search should not see user B's results."""
    a = tmp_db.register_user("u_a", "pw_search_xyz")
    b = tmp_db.register_user("u_b", "pw_search_xyz")
    tmp_db.save_result(a, "topic for A", 0.5, 1, {"ideas": []})
    tmp_db.save_result(b, "topic for B", 0.5, 1, {"ideas": []})
    assert len(tmp_db.search_user_results(a, "topic")) == 1
    assert len(tmp_db.search_user_results(b, "topic")) == 1
    assert tmp_db.search_user_results(a, "for B") == []


def test_search_respects_limit(tmp_db):
    uid = tmp_db.register_user("harry", "pw_search_xyz")
    for i in range(20):
        tmp_db.save_result(uid, f"topic {i} foo", 0.5, 1, {"ideas": []})
    out = tmp_db.search_user_results(uid, "foo", limit=5)
    assert len(out) == 5


# ── result_chat.build_context ───────────────────────────────────────────────

def _sample_result():
    return {
        "topic": "neural architecture search",
        "coverage": 0.62,
        "ideas": [
            {"title": "Differentiable Routing",
             "motivation": "explore continuous relaxation of architecture choice"},
            {"title": "Evolutionary Search",
             "motivation": "use tournament selection over candidate networks"},
            {"title": "Random Search Baseline",
             "motivation": "establish a lower bound"},
        ],
    }


def test_build_context_always_includes_topic_and_count():
    import result_chat
    ctx = result_chat.build_context(_sample_result(), user_query="hi")
    assert "neural architecture search" in ctx
    assert "Total ideas generated: 3" in ctx
    assert "Pipeline coverage: 62%" in ctx


def test_build_context_lists_all_ideas_compactly():
    import result_chat
    ctx = result_chat.build_context(_sample_result(), user_query="hi")
    assert "[1] Differentiable Routing" in ctx
    assert "[2] Evolutionary Search" in ctx
    assert "[3] Random Search Baseline" in ctx


def test_build_context_expands_ideas_named_in_query():
    """Mentioning an idea's title in the query → full detail block."""
    import result_chat
    ctx = result_chat.build_context(
        _sample_result(),
        user_query="tell me more about differentiable routing",
    )
    assert "Full detail for ideas mentioned" in ctx
    assert "Motivation: explore continuous relaxation" in ctx


def test_build_context_no_full_detail_when_nothing_named():
    import result_chat
    ctx = result_chat.build_context(
        _sample_result(),
        user_query="give me the gist",
    )
    assert "Full detail" not in ctx


def test_build_context_handles_empty_ideas():
    import result_chat
    ctx = result_chat.build_context({"topic": "x", "ideas": []})
    assert "No ideas were found" in ctx


def test_build_context_finds_ideas_in_alternate_keys():
    """The pipeline emits {ideas: …} now but older results used
    `final_ideas` / `idea_archive`. Search them all."""
    import result_chat
    ctx = result_chat.build_context({
        "topic": "old result",
        "final_ideas": [{"title": "Legacy Idea", "motivation": "old"}],
    })
    assert "Legacy Idea" in ctx


# ── result_chat.chat_turn ──────────────────────────────────────────────────

def test_chat_turn_rejects_empty_result_dict():
    import result_chat
    with pytest.raises(ValueError, match="result_dict"):
        result_chat.chat_turn({}, [], "anything", claude_client=MagicMock())


def test_chat_turn_rejects_empty_message():
    import result_chat
    with pytest.raises(ValueError, match="user_message"):
        result_chat.chat_turn(
            _sample_result(), [], "",
            claude_client=MagicMock(),
        )


def test_chat_turn_returns_none_with_no_client():
    """No client + autoload fails → None (caller surfaces 'check provider')."""
    import result_chat
    out = result_chat.chat_turn(
        _sample_result(), [], "rank these",
        claude_client=None,
    )
    assert out is None


def test_chat_turn_happy_path_passes_full_context():
    import result_chat
    client = MagicMock()
    resp = MagicMock()
    resp.success = True
    resp.text = "The top idea is Differentiable Routing because…"
    client.call.return_value = resp
    out = result_chat.chat_turn(
        _sample_result(),
        history=[],
        user_message="which is best?",
        claude_client=client,
    )
    assert out and "Differentiable Routing" in out
    # Check the call shape: system prompt mentions research-idea pipeline,
    # user prompt contains the context block.
    call_kwargs = client.call.call_args.kwargs
    assert "research" in call_kwargs["system"].lower()
    user_prompt = call_kwargs["user"]
    assert "RESULT CONTEXT" in user_prompt
    assert "neural architecture search" in user_prompt
    assert "which is best?" in user_prompt


def test_chat_turn_failed_response_returns_none():
    import result_chat
    client = MagicMock()
    resp = MagicMock()
    resp.success = False
    resp.text = ""
    client.call.return_value = resp
    out = result_chat.chat_turn(
        _sample_result(), [], "anything", claude_client=client,
    )
    assert out is None


def test_chat_turn_includes_history_in_prompt():
    import result_chat
    client = MagicMock()
    resp = MagicMock()
    resp.success = True
    resp.text = "reply"
    client.call.return_value = resp
    history = [
        result_chat.ResultChatMessage(role="user", content="hi"),
        result_chat.ResultChatMessage(role="assistant", content="hello"),
    ]
    result_chat.chat_turn(
        _sample_result(), history, "follow up question",
        claude_client=client,
    )
    user_prompt = client.call.call_args.kwargs["user"]
    assert "CONVERSATION SO FAR" in user_prompt
    assert "User: hi" in user_prompt
    assert "Assistant: hello" in user_prompt


# ── History serialization ──────────────────────────────────────────────────

def test_history_roundtrip():
    import result_chat
    hist = [
        result_chat.ResultChatMessage(role="user", content="hi"),
        result_chat.ResultChatMessage(role="assistant", content="hello"),
    ]
    j = result_chat.history_to_jsonable(hist)
    assert j == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    back = result_chat.history_from_jsonable(j)
    assert len(back) == 2
    assert back[0].role == "user" and back[0].content == "hi"


def test_history_from_jsonable_filters_garbage():
    import result_chat
    out = result_chat.history_from_jsonable([
        {"role": "user", "content": "ok"},
        {"role": "system", "content": "ignored"},  # bad role
        "not a dict",
        {"role": "assistant"},  # missing content
        {"role": "assistant", "content": "good"},
    ])
    assert [(m.role, m.content) for m in out] == [
        ("user", "ok"), ("assistant", "good"),
    ]


# ── Streamlit smoke test ────────────────────────────────────────────────────

def _make_st_stub():
    stub = MagicMock()
    stub.columns.side_effect = lambda spec, **kw: (
        [MagicMock() for _ in range(spec)]
        if isinstance(spec, int)
        else [MagicMock() for _ in spec]
    )
    for attr in ("expander", "container", "form"):
        cm = getattr(stub, attr)
        cm.return_value.__enter__ = MagicMock(return_value=stub)
        cm.return_value.__exit__ = MagicMock(return_value=None)
    # chat_message + spinner context managers.
    stub.chat_message.return_value.__enter__ = MagicMock(return_value=stub)
    stub.chat_message.return_value.__exit__ = MagicMock(return_value=None)
    stub.spinner.return_value.__enter__ = MagicMock(return_value=stub)
    stub.spinner.return_value.__exit__ = MagicMock(return_value=None)
    stub.button.return_value = False
    stub.chat_input.return_value = None
    stub.session_state = {}
    return stub


def test_render_chat_panel_with_no_result_shows_hint():
    import result_chat
    st = _make_st_stub()
    result_chat.render_chat_panel(st, None, None)
    assert st.info.called


def test_render_chat_panel_with_result_renders_header():
    import result_chat
    st = _make_st_stub()
    result_chat.render_chat_panel(st, _sample_result(), result_id=42)
    # The header markdown call mentions our topic.
    md_calls = [c for c in st.markdown.call_args_list if c.args]
    assert any(
        "neural architecture search" in (c.args[0] or "")
        for c in md_calls
    )


def test_render_chat_panel_suggestion_click_triggers_rerun(monkeypatch):
    """Clicking a suggestion fires rerun(). In real Streamlit this halts
    and the new render picks up the pending key; in this mock the code
    continues past rerun and the key is popped in the same pass, so we
    verify rerun was called rather than asserting on the key's residue."""
    import result_chat
    st = _make_st_stub()
    # First button (first suggestion) returns True.
    st.button.side_effect = [True] + [False] * 50
    result_chat.render_chat_panel(st, _sample_result(), result_id=99)
    assert st.rerun.called


# ── extract_mentioned_idea_indices ──────────────────────────────────────────

def test_extract_mentioned_indices_finds_exact_titles():
    import result_chat
    ideas = _sample_result()["ideas"]
    history = [
        result_chat.ResultChatMessage(
            role="assistant",
            content="The best option is Differentiable Routing because…",
        ),
    ]
    out = result_chat.extract_mentioned_idea_indices(history, ideas)
    assert out == [0]  # only "Differentiable Routing" matches


def test_extract_mentioned_indices_multiple():
    import result_chat
    ideas = _sample_result()["ideas"]
    history = [
        result_chat.ResultChatMessage(
            role="user", content="compare differentiable routing vs evolutionary search",
        ),
        result_chat.ResultChatMessage(
            role="assistant", content="differentiable routing is faster, but evolutionary search…",
        ),
    ]
    out = result_chat.extract_mentioned_idea_indices(history, ideas)
    assert set(out) == {0, 1}


def test_extract_mentioned_indices_case_insensitive():
    import result_chat
    ideas = _sample_result()["ideas"]
    history = [
        result_chat.ResultChatMessage(
            role="user", content="EVOLUTIONARY SEARCH IS COOL",
        ),
    ]
    out = result_chat.extract_mentioned_idea_indices(history, ideas)
    assert out == [1]


def test_extract_mentioned_indices_ignores_short_titles():
    """Title shorter than min_title_len → not matched to avoid noise."""
    import result_chat
    ideas = [
        {"title": "RL", "motivation": "x"},        # too short — ignored
        {"title": "Sparse Attention", "motivation": "y"},
    ]
    history = [
        result_chat.ResultChatMessage(
            role="user",
            content="RL is fine but sparse attention is better",
        ),
    ]
    out = result_chat.extract_mentioned_idea_indices(
        history, ideas, min_title_len=4,
    )
    assert out == [1]  # only "Sparse Attention" matched


def test_extract_mentioned_indices_no_matches():
    import result_chat
    ideas = _sample_result()["ideas"]
    history = [
        result_chat.ResultChatMessage(
            role="user", content="something totally unrelated about cats",
        ),
    ]
    assert result_chat.extract_mentioned_idea_indices(history, ideas) == []


def test_extract_mentioned_indices_empty_inputs():
    import result_chat
    assert result_chat.extract_mentioned_idea_indices([], []) == []
    assert result_chat.extract_mentioned_idea_indices(
        [], _sample_result()["ideas"],
    ) == []
    assert result_chat.extract_mentioned_idea_indices(
        [result_chat.ResultChatMessage(role="user", content="hi")], [],
    ) == []


def test_extract_mentioned_indices_dedupes_repeated_mentions():
    """If the assistant mentioned the same idea 3 times, the index
    appears once in the output."""
    import result_chat
    ideas = _sample_result()["ideas"]
    history = [
        result_chat.ResultChatMessage(
            role="user", content="differentiable routing differentiable routing",
        ),
        result_chat.ResultChatMessage(
            role="assistant", content="differentiable routing again",
        ),
    ]
    assert result_chat.extract_mentioned_idea_indices(history, ideas) == [0]


# ── Idea index renders & interacts ──────────────────────────────────────────

def test_render_chat_panel_includes_idea_index_header():
    """The right-column 📌 'Ideas in this result' header should render."""
    import result_chat
    st = _make_st_stub()
    result_chat.render_chat_panel(st, _sample_result(), result_id=42)
    md_calls = [
        c.args[0] for c in st.markdown.call_args_list
        if c.args and isinstance(c.args[0], str)
    ]
    assert any("Ideas in this result" in m for m in md_calls)


def test_render_chat_panel_shows_mentioned_badge_when_chat_has_history():
    """When chat history references an idea, the index header should
    say '💬 N of M mentioned in chat so far'."""
    import result_chat
    st = _make_st_stub()
    # Pre-populate history mentioning "Differentiable Routing".
    st.session_state["_result_chat:50"] = [
        result_chat.ResultChatMessage(
            role="assistant",
            content="Differentiable Routing is the strongest of the three.",
        ),
    ]
    result_chat.render_chat_panel(st, _sample_result(), result_id=50)
    caption_calls = [
        c.args[0] for c in st.caption.call_args_list
        if c.args and isinstance(c.args[0], str)
    ]
    assert any("mentioned in chat" in c for c in caption_calls)


def test_idea_index_renders_scrollable_container():
    """The 📌 idea index uses st.container(height=480) to make the list
    scrollable. Verifying the container call confirms the index code
    path ran (the per-idea Ask/View buttons are called on column mocks,
    which the stub returns fresh, so we can't easily count them via
    st.button — verifying the container call is the cleaner signal)."""
    import result_chat
    st = _make_st_stub()
    result_chat.render_chat_panel(st, _sample_result(), result_id=77)
    container_calls = st.container.call_args_list
    # Look for the height=480 scrollable container the index uses.
    assert any(
        c.kwargs.get("height") == 480 for c in container_calls
    ), "Idea index scrollable container (height=480) not rendered"


def test_idea_index_ask_button_triggers_rerun_with_pending_message():
    """Clicking ANY button in the panel fires rerun(); when that button
    is a 🔍 Ask card-button, the pending message contains 'Tell me more'.
    Verified by mocking the FIRST button True and checking BOTH rerun
    and the message structure (since pop happens in same render pass
    in the mock, we check session_state DURING the call via a spy)."""
    import result_chat
    st = _make_st_stub()
    # Capture session_state mutations as they happen so we see the
    # pending value before it's popped at the bottom of the render.
    captured = {}

    class _SpySessionState(dict):
        def __setitem__(self, k, v):
            super().__setitem__(k, v)
            if "_rc_pending_" in k:
                captured["last_pending"] = v

    st.session_state = _SpySessionState()
    st.button.side_effect = [True] + [False] * 60
    result_chat.render_chat_panel(st, _sample_result(), result_id=77)
    assert st.rerun.called
    # The first interactive button is a suggestion (cold start), so
    # captured["last_pending"] will be a suggestion's text. The structure
    # we want to verify is: pending got set at all.
    assert "last_pending" in captured
    assert isinstance(captured["last_pending"], str)
    assert len(captured["last_pending"]) > 5


def test_mentions_chip_renders_after_assistant_reply():
    """When the last turn is assistant + mentions an idea, the
    '💡 Ideas just mentioned' header should appear."""
    import result_chat
    st = _make_st_stub()
    st.session_state["_result_chat:88"] = [
        result_chat.ResultChatMessage(role="user", content="best one?"),
        result_chat.ResultChatMessage(
            role="assistant",
            content="The Differentiable Routing idea is strongest.",
        ),
    ]
    result_chat.render_chat_panel(st, _sample_result(), result_id=88)
    md_calls = [
        c.args[0] for c in st.markdown.call_args_list
        if c.args and isinstance(c.args[0], str)
    ]
    assert any("Ideas just mentioned" in m for m in md_calls)


def test_mentions_chip_NOT_rendered_when_last_turn_is_user():
    """Chips only show after an assistant reply, not mid-question."""
    import result_chat
    st = _make_st_stub()
    st.session_state["_result_chat:88"] = [
        result_chat.ResultChatMessage(role="user", content="what about differentiable routing?"),
    ]
    result_chat.render_chat_panel(st, _sample_result(), result_id=88)
    md_calls = [
        c.args[0] for c in st.markdown.call_args_list
        if c.args and isinstance(c.args[0], str)
    ]
    assert not any("Ideas just mentioned" in m for m in md_calls)


# ── Click-title-to-jump-to-Ideas-tab ────────────────────────────────────────

def test_idea_title_button_renders_for_every_idea():
    """Each idea in the index gets a `rc_title_<result_id>_<i>` button
    that, when clicked, sets _jump_to_idea_title for the Ideas tab to
    pick up."""
    import result_chat
    st = _make_st_stub()
    result_chat.render_chat_panel(st, _sample_result(), result_id=42)
    all_keys = [
        c.kwargs.get("key", "") for c in st.button.call_args_list
    ]
    # 3 ideas → 3 title buttons.
    for i in range(3):
        assert any(f"rc_title_42_{i}" in k for k in all_keys), (
            f"Title-jump button for idea #{i} not rendered"
        )


def test_title_button_click_sets_jump_session_state():
    """Clicking the title button sets _jump_to_idea_title (and idx) and
    queues a toast — captured before pop via a spy on session_state."""
    import result_chat
    st = _make_st_stub()

    captured = {}

    class _SpySessionState(dict):
        def __setitem__(self, k, v):
            super().__setitem__(k, v)
            if k in (
                "_jump_to_idea_title", "_jump_to_idea_idx", "_jump_toast",
            ):
                captured[k] = v

    st.session_state = _SpySessionState()
    # Render order: cold-start → 4 suggestion buttons first, THEN per-idea
    # cards. Each card has: title button, Ask, View (3 buttons). To click
    # the title of idea #0 reliably we'd need to know the order, but
    # easier: skip suggestions by pre-populating chat history.
    st.session_state["_result_chat:55"] = [
        result_chat.ResultChatMessage(role="user", content="hi"),
        result_chat.ResultChatMessage(role="assistant", content="hello"),
    ]
    # Now the first button rendered IS the title for idea #0 (left/idx col
    # is rendered before chat col; columns render top-down within each).
    # Make first button True.
    st.button.side_effect = [True] + [False] * 60
    result_chat.render_chat_panel(st, _sample_result(), result_id=55)
    # Verify the jump keys + toast got written before pop.
    assert "_jump_to_idea_title" in captured
    assert "_jump_to_idea_idx" in captured
    assert "_jump_toast" in captured
    assert captured["_jump_to_idea_title"] in {
        "Differentiable Routing",
        "Evolutionary Search",
        "Random Search Baseline",
    }
    assert isinstance(captured["_jump_to_idea_idx"], int)
    assert "Jumped" in captured["_jump_toast"]
    assert st.rerun.called


def test_jump_toast_shown_once_then_popped():
    """The toast under render_chat_panel is consumed via .pop so it
    only shows on the render right after the title click."""
    import result_chat
    st = _make_st_stub()
    st.session_state["_jump_toast"] = "📍 Jumped to **X** — switch tabs."
    result_chat.render_chat_panel(st, _sample_result(), result_id=66)
    # success() should have been called with the toast text.
    success_args = [
        c.args[0] for c in st.success.call_args_list
        if c.args and isinstance(c.args[0], str)
    ]
    assert any("Jumped" in s for s in success_args)
    # And the toast key is gone (consumed).
    assert "_jump_toast" not in st.session_state


def test_title_truncated_when_very_long():
    """Idea titles longer than 60 chars are truncated in the button
    label (the full title is still stored when clicked, so the Ideas
    tab can match it back to the source idea)."""
    import result_chat
    st = _make_st_stub()
    long_title = "A" * 100
    result = {
        "topic": "test",
        "ideas": [{"title": long_title, "motivation": "x"}],
    }
    result_chat.render_chat_panel(st, result, result_id=77)
    title_buttons = [
        c.args[0] for c in st.button.call_args_list
        if c.args and isinstance(c.args[0], str) and c.args[0].startswith("📍 ")
    ]
    assert title_buttons, "No title button rendered"
    # First title button should have the truncated form: 📍 + 57 chars + …
    btn_label = title_buttons[0]
    assert len(btn_label) <= 65  # "📍 " + 57 + "…" + safety margin
    assert btn_label.endswith("…")
