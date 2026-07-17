"""Tests for the Manage Account feature.

Covers:
  - db.get_user_profile / change_password / delete_user / export_user_data
  - account_ui.render_account_page (smoke — sections don't crash)
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── DB helpers — exercise against a temp SQLite ─────────────────────────────

@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    """Spin up a fresh DB in a temp dir for each test.

    The db module uses module-level path constants (no env override), so
    we monkeypatch them directly and clear any cached connection on the
    thread-local before init.
    """
    import db as _db
    tmp_dir = tmp_path / "data"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_db_path = tmp_dir / "test_ideagraph.db"
    monkeypatch.setattr(_db, "_DB_DIR", str(tmp_dir), raising=False)
    monkeypatch.setattr(_db, "_DB_PATH", str(tmp_db_path), raising=False)
    # Drop any cached connection on the thread-local so _get_conn rebuilds
    # against the new path.
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
    # Reset the cached connection so the next test starts clean.
    try:
        if hasattr(_db, "_conn_local") and hasattr(_db._conn_local, "wrapped"):
            try:
                _db._conn_local.wrapped.raw.close()
            except Exception:
                pass
            del _db._conn_local.wrapped
    except Exception:
        pass


def test_get_user_profile_returns_fields(tmp_db):
    uid = tmp_db.register_user("alice", "password123")
    assert uid is not None
    profile = tmp_db.get_user_profile(uid)
    assert profile is not None
    assert profile["id"] == uid
    assert profile["username"].lower() == "alice"
    assert profile["created_at"]


def test_get_user_profile_unknown_user_returns_none(tmp_db):
    assert tmp_db.get_user_profile(99999) is None


def test_change_password_happy_path(tmp_db):
    uid = tmp_db.register_user("bob", "old_password_xyz")
    res = tmp_db.change_password(uid, "old_password_xyz", "new_password_abc")
    assert res["ok"] is True
    # Old password should no longer log in.
    assert tmp_db.login_user("bob", "old_password_xyz") is None
    # New password should.
    assert tmp_db.login_user("bob", "new_password_abc") == uid


def test_change_password_wrong_old_password_rejected(tmp_db):
    uid = tmp_db.register_user("carol", "right_pw_xyz")
    res = tmp_db.change_password(uid, "WRONG", "new_pw_abc")
    assert res["ok"] is False
    assert "incorrect" in res["error"].lower()
    # Confirm the password wasn't changed.
    assert tmp_db.login_user("carol", "right_pw_xyz") == uid


def test_change_password_short_new_password_rejected(tmp_db):
    uid = tmp_db.register_user("dan", "old_pw_xyz")
    res = tmp_db.change_password(uid, "old_pw_xyz", "x")
    assert res["ok"] is False
    assert "6 character" in res["error"]


def test_change_password_unknown_user_rejected(tmp_db):
    res = tmp_db.change_password(99999, "anything", "new_password")
    assert res["ok"] is False
    assert "not found" in res["error"].lower()


def test_delete_user_happy_path(tmp_db):
    uid = tmp_db.register_user("eve", "del_pw_xyz")
    res = tmp_db.delete_user(uid, "del_pw_xyz")
    assert res["ok"] is True
    assert res["deleted"] == uid
    # User is gone.
    assert tmp_db.get_user_profile(uid) is None
    assert tmp_db.login_user("eve", "del_pw_xyz") is None


def test_delete_user_wrong_password_rejected(tmp_db):
    uid = tmp_db.register_user("frank", "real_pw_xyz")
    res = tmp_db.delete_user(uid, "WRONG")
    assert res["ok"] is False
    # User still exists.
    assert tmp_db.get_user_profile(uid) is not None


def test_delete_user_unknown_user_rejected(tmp_db):
    res = tmp_db.delete_user(99999, "anything")
    assert res["ok"] is False
    assert "not found" in res["error"].lower()


def test_delete_user_cascades_to_results(tmp_db):
    uid = tmp_db.register_user("greta", "cascade_pw_xyz")
    rid = tmp_db.save_result(uid, "test topic", 0.5, 3, {"ideas": []})
    assert rid > 0
    # Sanity — result is there.
    assert len(tmp_db.get_user_results(uid)) == 1
    # Delete the user.
    res = tmp_db.delete_user(uid, "cascade_pw_xyz")
    assert res["ok"] is True
    # Results are cleaned up.
    assert tmp_db.get_user_results(uid) == []


def test_export_user_data_includes_profile_and_results(tmp_db):
    uid = tmp_db.register_user("harry", "export_pw_xyz")
    tmp_db.save_result(uid, "topic A", 0.6, 5, {"ideas": [1, 2]})
    tmp_db.save_result(uid, "topic B", 0.3, 2, {"ideas": [3]})
    payload = tmp_db.export_user_data(uid)
    assert payload["user"]["username"].lower() == "harry"
    assert payload["user"]["id"] == uid
    assert len(payload["results"]) == 2
    # Password hash must NOT be exported.
    assert "password_hash" not in payload["user"]
    assert "salt" not in payload["user"]
    # Timestamp present.
    assert payload["exported_at"]


def test_export_user_data_unknown_user_returns_empty_user(tmp_db):
    payload = tmp_db.export_user_data(99999)
    assert payload["user"] is None
    assert payload["results"] == []


# ── account_ui — smoke tests with a MagicMock st ────────────────────────────

def _make_st_stub():
    stub = MagicMock()
    # st.columns(N) → list of N MagicMock columns.
    stub.columns.side_effect = lambda spec, **kw: (
        [MagicMock() for _ in range(spec)]
        if isinstance(spec, int)
        else [MagicMock() for _ in spec]
    )
    # Context-manage expander / container / form.
    for attr in ("expander", "container", "form"):
        cm = getattr(stub, attr)
        cm.return_value.__enter__ = MagicMock(return_value=stub)
        cm.return_value.__exit__ = MagicMock(return_value=None)
    # tabs(...) returns a list of context managers.
    def _tabs(labels):
        out = []
        for _ in labels:
            t = MagicMock()
            t.__enter__ = MagicMock(return_value=stub)
            t.__exit__ = MagicMock(return_value=None)
            out.append(t)
        return out
    stub.tabs.side_effect = _tabs
    # Buttons / form_submit_button default to False (no click).
    stub.button.return_value = False
    stub.form_submit_button.return_value = False
    stub.checkbox.return_value = False
    stub.text_input.return_value = ""
    stub.session_state = {}
    return stub


def test_render_account_page_no_user_id_renders_warning():
    import account_ui
    st = _make_st_stub()
    account_ui.render_account_page(st, None)
    assert st.warning.called


def test_render_account_page_with_user_id_renders_all_tabs(tmp_db):
    """Smoke: render the full page; verify no section crashes."""
    import importlib
    import account_ui
    importlib.reload(account_ui)  # pick up the freshly reloaded db
    uid = tmp_db.register_user("smoke_user", "smoke_pw_xyz")
    st = _make_st_stub()
    account_ui.render_account_page(st, uid)
    # All 6 tabs created.
    assert st.tabs.called
    tab_labels = st.tabs.call_args[0][0]
    assert "👤 Profile" in tab_labels
    assert "🔒 Security" in tab_labels
    assert "💳 Plan & billing" in tab_labels
    assert "📈 Usage" in tab_labels
    assert "📦 My data" in tab_labels
    assert "⚠️ Danger zone" in tab_labels


def test_render_account_page_back_button_clears_flag(tmp_db):
    """If user clicks ← Back, the _show_account_page flag is cleared."""
    import importlib
    import account_ui
    importlib.reload(account_ui)
    uid = tmp_db.register_user("back_user", "back_pw_xyz")
    st = _make_st_stub()
    st.session_state["_show_account_page"] = True
    # First .button() call is the Back button — make it return True.
    st.button.side_effect = [True] + [False] * 50
    account_ui.render_account_page(st, uid)
    assert st.session_state.get("_show_account_page") is False
    assert st.rerun.called
