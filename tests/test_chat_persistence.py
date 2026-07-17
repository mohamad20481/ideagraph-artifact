"""Tests for chat-session persistence (💬 Chat tab durability).

Covers:
  - db.save_chat_session current-session upsert (same row reused)
  - db.save_chat_session snapshot flag (always inserts)
  - db.load_chat_session / load_current_chat_session
  - db.list_chat_sessions (filters by result_id, snapshots_only)
  - db.delete_chat_session
  - Cross-user isolation
  - result_chat panel auto-persists turns and auto-restores on next open
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── tmp_db fixture (same pattern as test_account.py) ────────────────────────

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


# ── db.save_chat_session — current-session upsert ──────────────────────────

def test_save_current_session_inserts_first_time(tmp_db):
    uid = tmp_db.register_user("alice", "pw_chat_xyz")
    rid = tmp_db.save_result(uid, "test topic", 0.5, 1, {"ideas": []})
    sid = tmp_db.save_chat_session(
        user_id=uid,
        history=[{"role": "user", "content": "hi"}],
        result_id=rid,
    )
    assert sid > 0
    loaded = tmp_db.load_chat_session(sid, uid)
    assert loaded["message_count"] == 1
    assert loaded["history"] == [{"role": "user", "content": "hi"}]
    assert loaded["is_snapshot"] == 0


def test_save_current_session_upserts_in_place(tmp_db):
    """Second save with same context REUSES the same row id."""
    uid = tmp_db.register_user("bob", "pw_chat_xyz")
    rid = tmp_db.save_result(uid, "test", 0.5, 1, {"ideas": []})
    sid1 = tmp_db.save_chat_session(
        user_id=uid,
        history=[{"role": "user", "content": "first"}],
        result_id=rid,
    )
    sid2 = tmp_db.save_chat_session(
        user_id=uid,
        history=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
        ],
        result_id=rid,
    )
    assert sid1 == sid2
    loaded = tmp_db.load_chat_session(sid2, uid)
    assert loaded["message_count"] == 2


def test_different_results_get_different_sessions(tmp_db):
    uid = tmp_db.register_user("carol", "pw_chat_xyz")
    r1 = tmp_db.save_result(uid, "topic A", 0.5, 1, {"ideas": []})
    r2 = tmp_db.save_result(uid, "topic B", 0.5, 1, {"ideas": []})
    s1 = tmp_db.save_chat_session(
        user_id=uid, history=[{"role": "user", "content": "a"}], result_id=r1,
    )
    s2 = tmp_db.save_chat_session(
        user_id=uid, history=[{"role": "user", "content": "b"}], result_id=r2,
    )
    assert s1 != s2


# ── db.save_chat_session — snapshot mode ───────────────────────────────────

def test_snapshot_always_inserts_new_row(tmp_db):
    uid = tmp_db.register_user("dan", "pw_chat_xyz")
    rid = tmp_db.save_result(uid, "test", 0.5, 1, {"ideas": []})
    s1 = tmp_db.save_chat_session(
        user_id=uid,
        history=[{"role": "user", "content": "snap1"}],
        result_id=rid,
        title="first snap",
        is_snapshot=True,
    )
    s2 = tmp_db.save_chat_session(
        user_id=uid,
        history=[{"role": "user", "content": "snap2"}],
        result_id=rid,
        title="second snap",
        is_snapshot=True,
    )
    assert s1 != s2
    assert tmp_db.load_chat_session(s1, uid)["title"] == "first snap"
    assert tmp_db.load_chat_session(s2, uid)["title"] == "second snap"


def test_snapshot_distinguished_from_current(tmp_db):
    """Snapshots and the current session live in different rows even for
    the same result."""
    uid = tmp_db.register_user("eve", "pw_chat_xyz")
    rid = tmp_db.save_result(uid, "test", 0.5, 1, {"ideas": []})
    cur = tmp_db.save_chat_session(
        user_id=uid,
        history=[{"role": "user", "content": "live"}],
        result_id=rid,
    )
    snap = tmp_db.save_chat_session(
        user_id=uid,
        history=[{"role": "user", "content": "frozen"}],
        result_id=rid,
        is_snapshot=True,
    )
    assert cur != snap
    cur_row = tmp_db.load_chat_session(cur, uid)
    snap_row = tmp_db.load_chat_session(snap, uid)
    assert cur_row["is_snapshot"] == 0
    assert snap_row["is_snapshot"] == 1


# ── load_current_chat_session convenience ──────────────────────────────────

def test_load_current_chat_session_returns_latest(tmp_db):
    uid = tmp_db.register_user("frank", "pw_chat_xyz")
    rid = tmp_db.save_result(uid, "test", 0.5, 1, {"ideas": []})
    tmp_db.save_chat_session(
        user_id=uid, history=[{"role": "user", "content": "x"}], result_id=rid,
    )
    loaded = tmp_db.load_current_chat_session(uid, result_id=rid)
    assert loaded is not None
    assert loaded["history"] == [{"role": "user", "content": "x"}]


def test_load_current_chat_session_returns_none_when_nothing_saved(tmp_db):
    uid = tmp_db.register_user("greta", "pw_chat_xyz")
    rid = tmp_db.save_result(uid, "test", 0.5, 1, {"ideas": []})
    assert tmp_db.load_current_chat_session(uid, result_id=rid) is None


def test_load_current_chat_session_ignores_snapshots(tmp_db):
    """A snapshot row for this result shouldn't satisfy a current-session query."""
    uid = tmp_db.register_user("harry", "pw_chat_xyz")
    rid = tmp_db.save_result(uid, "test", 0.5, 1, {"ideas": []})
    tmp_db.save_chat_session(
        user_id=uid, history=[{"role": "user", "content": "snap"}],
        result_id=rid, is_snapshot=True,
    )
    assert tmp_db.load_current_chat_session(uid, result_id=rid) is None


# ── list_chat_sessions ─────────────────────────────────────────────────────

def test_list_chat_sessions_returns_user_scoped(tmp_db):
    uid_a = tmp_db.register_user("user_a", "pw")
    uid_b = tmp_db.register_user("user_b", "pw")
    tmp_db.save_chat_session(
        user_id=uid_a, history=[{"role": "user", "content": "a"}],
    )
    tmp_db.save_chat_session(
        user_id=uid_b, history=[{"role": "user", "content": "b"}],
    )
    assert len(tmp_db.list_chat_sessions(uid_a)) == 1
    assert len(tmp_db.list_chat_sessions(uid_b)) == 1


def test_list_chat_sessions_filters_by_result_id(tmp_db):
    uid = tmp_db.register_user("ivy", "pw")
    r1 = tmp_db.save_result(uid, "A", 0.5, 1, {"ideas": []})
    r2 = tmp_db.save_result(uid, "B", 0.5, 1, {"ideas": []})
    tmp_db.save_chat_session(
        user_id=uid, history=[{"role": "user", "content": "a"}], result_id=r1,
    )
    tmp_db.save_chat_session(
        user_id=uid, history=[{"role": "user", "content": "b"}], result_id=r2,
    )
    a_only = tmp_db.list_chat_sessions(uid, result_id=r1)
    assert len(a_only) == 1
    assert a_only[0]["result_id"] == r1


def test_list_chat_sessions_snapshots_only(tmp_db):
    uid = tmp_db.register_user("jack", "pw")
    rid = tmp_db.save_result(uid, "x", 0.5, 1, {"ideas": []})
    tmp_db.save_chat_session(
        user_id=uid, history=[{"role": "user", "content": "live"}],
        result_id=rid,
    )
    tmp_db.save_chat_session(
        user_id=uid, history=[{"role": "user", "content": "snap"}],
        result_id=rid, is_snapshot=True,
    )
    all_sessions = tmp_db.list_chat_sessions(uid)
    only_snaps = tmp_db.list_chat_sessions(uid, snapshots_only=True)
    assert len(all_sessions) == 2
    assert len(only_snaps) == 1
    assert only_snaps[0]["is_snapshot"] == 1


# ── delete_chat_session ────────────────────────────────────────────────────

def test_delete_chat_session_happy_path(tmp_db):
    uid = tmp_db.register_user("kate", "pw")
    sid = tmp_db.save_chat_session(
        user_id=uid, history=[{"role": "user", "content": "x"}],
    )
    assert tmp_db.delete_chat_session(sid, uid) is True
    assert tmp_db.load_chat_session(sid, uid) is None


def test_delete_chat_session_wrong_user_rejected(tmp_db):
    uid_a = tmp_db.register_user("user_a", "pw")
    uid_b = tmp_db.register_user("user_b", "pw")
    sid = tmp_db.save_chat_session(
        user_id=uid_a, history=[{"role": "user", "content": "x"}],
    )
    assert tmp_db.delete_chat_session(sid, uid_b) is False
    # Still owned by uid_a.
    assert tmp_db.load_chat_session(sid, uid_a) is not None


# ── Cross-user isolation ───────────────────────────────────────────────────

def test_load_chat_session_scoped_to_owner(tmp_db):
    uid_a = tmp_db.register_user("alpha", "pw")
    uid_b = tmp_db.register_user("beta", "pw")
    sid = tmp_db.save_chat_session(
        user_id=uid_a, history=[{"role": "user", "content": "private"}],
    )
    # uid_b can't load uid_a's chat.
    assert tmp_db.load_chat_session(sid, uid_b) is None
    assert tmp_db.load_chat_session(sid, uid_a) is not None


# ── result_chat panel integration ──────────────────────────────────────────

def _make_st_stub_with_session(user_id=42):
    """Stub with a session_state pre-populated with a logged-in user.

    Important: `col.button(...)` (where col is from st.columns) must also
    return False by default. A vanilla MagicMock would return a MagicMock
    instance — which is *truthy*, accidentally firing every column-scoped
    Ask/View button on every render. We build columns whose .button()
    method returns False explicitly to match real Streamlit behavior.
    """
    stub = MagicMock()

    def _make_col_mock():
        c = MagicMock()
        c.button = MagicMock(return_value=False)
        return c

    stub.columns.side_effect = lambda spec, **kw: (
        [_make_col_mock() for _ in range(spec)]
        if isinstance(spec, int)
        else [_make_col_mock() for _ in spec]
    )
    for attr in ("expander", "container", "form"):
        cm = getattr(stub, attr)
        cm.return_value.__enter__ = MagicMock(return_value=stub)
        cm.return_value.__exit__ = MagicMock(return_value=None)
    stub.chat_message.return_value.__enter__ = MagicMock(return_value=stub)
    stub.chat_message.return_value.__exit__ = MagicMock(return_value=None)
    stub.spinner.return_value.__enter__ = MagicMock(return_value=stub)
    stub.spinner.return_value.__exit__ = MagicMock(return_value=None)
    stub.button.return_value = False
    stub.chat_input.return_value = None
    stub.text_input.return_value = ""
    stub.session_state = {"user_id": user_id}
    return stub


def test_panel_auto_restores_from_db_on_first_open(tmp_db):
    """When session_state has no chat history but db does, the panel
    hydrates session_state from db. Patch chat_turn to neuter any
    user_input-driven LLM call so the test only measures restore."""
    import result_chat
    uid = tmp_db.register_user("restorer", "pw")
    rid = tmp_db.save_result(uid, "topic X", 0.5, 1, {"ideas": []})
    tmp_db.save_chat_session(
        user_id=uid,
        history=[
            {"role": "user", "content": "what's up"},
            {"role": "assistant", "content": "all good"},
        ],
        result_id=rid,
    )
    st = _make_st_stub_with_session(user_id=uid)
    # Ensure chat_input returns None (no user input) for this render.
    st.chat_input = MagicMock(return_value=None)
    result = {"topic": "topic X", "ideas": [
        {"title": "An idea", "motivation": "x"},
    ]}
    with patch.object(result_chat, "chat_turn", return_value=None):
        result_chat.render_chat_panel(st, result, result_id=rid)
    restored = st.session_state.get(f"_result_chat:{rid}")
    assert restored is not None
    # Should be the 2 messages from db — no additional turns since
    # chat_turn is patched to no-op.
    assert len(restored) >= 2
    assert restored[0].role == "user"
    assert restored[0].content == "what's up"
    assert restored[1].role == "assistant"
    assert restored[1].content == "all good"


def test_panel_does_not_overwrite_existing_session_state_history(tmp_db):
    """If session_state already has history (from this session), db
    restore must NOT clobber it."""
    import result_chat
    uid = tmp_db.register_user("noclobber", "pw")
    rid = tmp_db.save_result(uid, "x", 0.5, 1, {"ideas": []})
    tmp_db.save_chat_session(
        user_id=uid,
        history=[{"role": "user", "content": "from db"}],
        result_id=rid,
    )
    st = _make_st_stub_with_session(user_id=uid)
    # Pre-seed session_state with different history.
    st.session_state[f"_result_chat:{rid}"] = [
        result_chat.ResultChatMessage(role="user", content="from session"),
    ]
    result = {"topic": "x", "ideas": [{"title": "An idea", "motivation": "x"}]}
    result_chat.render_chat_panel(st, result, result_id=rid)
    current = st.session_state[f"_result_chat:{rid}"]
    assert len(current) == 1
    assert current[0].content == "from session"


def test_panel_skips_restore_for_anonymous_user(tmp_db):
    """Without a user_id in session_state, db restore is skipped silently."""
    import result_chat
    st = _make_st_stub_with_session(user_id=None)
    st.session_state.pop("user_id", None)
    result = {"topic": "x", "ideas": [{"title": "A", "motivation": "y"}]}
    result_chat.render_chat_panel(st, result, result_id=99)
    # No history in session_state and no crash.
    assert not st.session_state.get("_result_chat:99")


def test_auto_save_chat_writes_to_db(tmp_db):
    """The _auto_save_chat helper upserts a current-session row each
    time it's called."""
    import result_chat
    uid = tmp_db.register_user("saver", "pw")
    rid = tmp_db.save_result(uid, "topic Y", 0.5, 1, {"ideas": []})
    st = _make_st_stub_with_session(user_id=uid)
    history = [
        result_chat.ResultChatMessage(role="user", content="hi"),
        result_chat.ResultChatMessage(role="assistant", content="hello"),
    ]
    result_chat._auto_save_chat(st, history, rid, "topic Y")
    sid = st.session_state.get(f"_rc_db_session_id:{rid}")
    assert sid
    loaded = tmp_db.load_chat_session(sid, uid)
    assert loaded["message_count"] == 2
    # Second call upserts (same row id).
    history.append(
        result_chat.ResultChatMessage(role="user", content="more")
    )
    result_chat._auto_save_chat(st, history, rid, "topic Y")
    sid2 = st.session_state.get(f"_rc_db_session_id:{rid}")
    assert sid2 == sid
    loaded2 = tmp_db.load_chat_session(sid, uid)
    assert loaded2["message_count"] == 3


def test_save_snapshot_creates_new_row_each_time(tmp_db):
    """Manual snapshots always insert a fresh row."""
    import result_chat
    uid = tmp_db.register_user("snapper", "pw")
    rid = tmp_db.save_result(uid, "topic Z", 0.5, 1, {"ideas": []})
    st = _make_st_stub_with_session(user_id=uid)
    history = [
        result_chat.ResultChatMessage(role="user", content="hi"),
    ]
    ok1 = result_chat._save_snapshot(
        st, history, rid, "topic Z", snap_title="first",
    )
    history.append(
        result_chat.ResultChatMessage(role="assistant", content="reply")
    )
    ok2 = result_chat._save_snapshot(
        st, history, rid, "topic Z", snap_title="second",
    )
    assert ok1 and ok2
    snaps = tmp_db.list_chat_sessions(uid, snapshots_only=True)
    assert len(snaps) == 2
    titles = sorted(s["title"] for s in snaps)
    assert titles == ["first", "second"]


def test_save_snapshot_rejects_empty_history(tmp_db):
    import result_chat
    uid = tmp_db.register_user("empty", "pw")
    st = _make_st_stub_with_session(user_id=uid)
    ok = result_chat._save_snapshot(
        st, history=[], result_id=None, topic="x", snap_title="empty",
    )
    assert ok is False
    # Verify warning was called.
    assert st.warning.called


def test_save_snapshot_rejects_anonymous_user(tmp_db):
    import result_chat
    st = _make_st_stub_with_session(user_id=None)
    st.session_state.pop("user_id", None)
    history = [
        result_chat.ResultChatMessage(role="user", content="hi"),
    ]
    ok = result_chat._save_snapshot(
        st, history=history, result_id=None, topic="x", snap_title="x",
    )
    assert ok is False
    assert st.error.called


def test_clear_current_chat_deletes_db_row(tmp_db):
    import result_chat
    uid = tmp_db.register_user("clearer", "pw")
    rid = tmp_db.save_result(uid, "x", 0.5, 1, {"ideas": []})
    st = _make_st_stub_with_session(user_id=uid)
    history = [result_chat.ResultChatMessage(role="user", content="hi")]
    result_chat._auto_save_chat(st, history, rid, "x")
    sid = st.session_state.get(f"_rc_db_session_id:{rid}")
    assert tmp_db.load_chat_session(sid, uid) is not None
    result_chat._clear_current_chat(st, rid)
    assert tmp_db.load_chat_session(sid, uid) is None
    # Session id key was popped.
    assert f"_rc_db_session_id:{rid}" not in st.session_state
