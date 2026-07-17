"""Tests for the always-on session-recovery token (auth_ui.py).

Solves: 'logged out in the middle of a long LLM generation' caused by
Streamlit's per-WebSocket session_state being discarded when the
connection drops + Remember Me wasn't opted into.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _FakeSessionState:
    """Streamlit's session_state supports BOTH attribute access
    (`st.session_state.logged_in = True`) AND dict access
    (`st.session_state.get("user_id")`). This stub does both, so test
    code can drive auth_ui exactly like the real Streamlit runtime."""

    def __init__(self) -> None:
        self._d = {}

    def get(self, key, default=None):
        return self._d.get(key, default)

    def __contains__(self, key):
        return key in self._d

    def __getitem__(self, key):
        return self._d[key]

    def __setitem__(self, key, value):
        self._d[key] = value

    def __delitem__(self, key):
        del self._d[key]

    def __getattr__(self, key):
        # Only called for attrs not in __dict__.
        if key.startswith("_"):
            raise AttributeError(key)
        try:
            return self._d[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        if key == "_d":
            super().__setattr__(key, value)
        else:
            self._d[key] = value

    def clear(self):
        self._d.clear()


def _import_auth_ui_with_tmp_files(tmp_path, monkeypatch):
    """Import auth_ui with both token files redirected into tmp_path.

    auth_ui imports `streamlit as st` at module top-level. We patch
    sys.modules so a lightweight fake is used in case the real one
    isn't available in the test environment.
    """
    import types
    fake_st = types.SimpleNamespace()
    fake_st.session_state = _FakeSessionState()
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)

    # Force a fresh re-import so module-level `_SESSION_RECOVERY_FILE` etc.
    # pick up our patched paths.
    sys.modules.pop("auth_ui", None)
    import auth_ui  # noqa: E402  -- needs the patched streamlit
    monkeypatch.setattr(
        auth_ui,
        "_SESSION_RECOVERY_FILE",
        tmp_path / ".session_recovery.json",
    )
    monkeypatch.setattr(
        auth_ui,
        "_REMEMBER_FILE",
        tmp_path / ".remember_token.json",
    )
    return auth_ui


# ── Save / Load / Clear ─────────────────────────────────────────────────────

def test_save_session_recovery_creates_file(tmp_path, monkeypatch):
    a = _import_auth_ui_with_tmp_files(tmp_path, monkeypatch)
    assert not a._SESSION_RECOVERY_FILE.exists()
    a._save_session_recovery_token(user_id=42, username="alice")
    assert a._SESSION_RECOVERY_FILE.exists()
    data = json.loads(a._SESSION_RECOVERY_FILE.read_text())
    assert data["user_id"] == 42
    assert data["username"] == "alice"
    assert data["token"]
    assert data["token_hash"]
    assert data["expires_at"] > time.time()
    # Token + hash integrity preserved.
    from production_optimization import hash_token
    assert hash_token(data["token"]) == data["token_hash"]


def test_load_session_recovery_returns_valid_token(tmp_path, monkeypatch):
    a = _import_auth_ui_with_tmp_files(tmp_path, monkeypatch)
    a._save_session_recovery_token(user_id=7, username="bob")
    loaded = a._load_session_recovery_token()
    assert loaded["user_id"] == 7
    assert loaded["username"] == "bob"


def test_load_session_recovery_missing_file_returns_empty(
    tmp_path, monkeypatch,
):
    a = _import_auth_ui_with_tmp_files(tmp_path, monkeypatch)
    # No file written.
    assert a._load_session_recovery_token() == {}


def test_load_session_recovery_expired_token_returns_empty(
    tmp_path, monkeypatch,
):
    a = _import_auth_ui_with_tmp_files(tmp_path, monkeypatch)
    # Write an expired token directly.
    payload = {
        "user_id": 1, "username": "u",
        "token": "x", "token_hash": "y",
        "expires_at": time.time() - 10,  # 10s ago
    }
    a._SESSION_RECOVERY_FILE.write_text(json.dumps(payload))
    assert a._load_session_recovery_token() == {}
    # Expired token must be cleaned up.
    assert not a._SESSION_RECOVERY_FILE.exists()


def test_load_session_recovery_tampered_token_returns_empty(
    tmp_path, monkeypatch,
):
    a = _import_auth_ui_with_tmp_files(tmp_path, monkeypatch)
    # Save a real token, then corrupt the token text.
    a._save_session_recovery_token(user_id=1, username="u")
    payload = json.loads(a._SESSION_RECOVERY_FILE.read_text())
    payload["token"] = "tampered_token"
    a._SESSION_RECOVERY_FILE.write_text(json.dumps(payload))
    assert a._load_session_recovery_token() == {}
    # Tampered token must be cleared.
    assert not a._SESSION_RECOVERY_FILE.exists()


def test_load_session_recovery_missing_expires_at_treated_as_expired(
    tmp_path, monkeypatch,
):
    a = _import_auth_ui_with_tmp_files(tmp_path, monkeypatch)
    payload = {
        "user_id": 1, "username": "u",
        "token": "x", "token_hash": "y",
        # no expires_at
    }
    a._SESSION_RECOVERY_FILE.write_text(json.dumps(payload))
    assert a._load_session_recovery_token() == {}


def test_clear_session_recovery_removes_file(tmp_path, monkeypatch):
    a = _import_auth_ui_with_tmp_files(tmp_path, monkeypatch)
    a._save_session_recovery_token(user_id=1, username="u")
    assert a._SESSION_RECOVERY_FILE.exists()
    a._clear_session_recovery_token()
    assert not a._SESSION_RECOVERY_FILE.exists()


def test_clear_session_recovery_no_file_is_no_op(tmp_path, monkeypatch):
    """Calling clear when no file exists must not raise."""
    a = _import_auth_ui_with_tmp_files(tmp_path, monkeypatch)
    a._clear_session_recovery_token()  # no exception


# ── Heartbeat refresh ───────────────────────────────────────────────────────

def test_refresh_skips_when_more_than_half_ttl_remaining(
    tmp_path, monkeypatch,
):
    """Don't burn disk I/O on every render — only refresh once we're past
    the 50% mark of the TTL."""
    a = _import_auth_ui_with_tmp_files(tmp_path, monkeypatch)
    a._save_session_recovery_token(user_id=1, username="u")
    original_expires = json.loads(
        a._SESSION_RECOVERY_FILE.read_text()
    )["expires_at"]
    # Immediately call refresh — should be a no-op since we have ~100% TTL.
    a._refresh_session_recovery_token()
    new_expires = json.loads(a._SESSION_RECOVERY_FILE.read_text())["expires_at"]
    assert new_expires == original_expires


def test_refresh_extends_when_past_halfway(tmp_path, monkeypatch):
    """Once we cross the halfway point, refresh extends the expiry."""
    a = _import_auth_ui_with_tmp_files(tmp_path, monkeypatch)
    a._save_session_recovery_token(user_id=1, username="u")
    # Force the on-disk expires_at to be 30% of TTL away (past halfway).
    payload = json.loads(a._SESSION_RECOVERY_FILE.read_text())
    payload["expires_at"] = (
        time.time() + a._SESSION_RECOVERY_TTL_SECONDS * 0.3
    )
    a._SESSION_RECOVERY_FILE.write_text(json.dumps(payload))
    before = payload["expires_at"]
    a._refresh_session_recovery_token()
    after = json.loads(a._SESSION_RECOVERY_FILE.read_text())["expires_at"]
    assert after > before
    # New expires_at should be ~now + full TTL.
    assert abs(after - (time.time() + a._SESSION_RECOVERY_TTL_SECONDS)) < 5


def test_refresh_no_file_is_no_op(tmp_path, monkeypatch):
    a = _import_auth_ui_with_tmp_files(tmp_path, monkeypatch)
    assert not a._SESSION_RECOVERY_FILE.exists()
    a._refresh_session_recovery_token()
    # Refresh should NOT create the file on its own.
    assert not a._SESSION_RECOVERY_FILE.exists()


def test_refresh_expired_token_is_no_op(tmp_path, monkeypatch):
    """If the file is already expired, refresh must NOT extend it — that
    would let a long-dead session resurrect itself."""
    a = _import_auth_ui_with_tmp_files(tmp_path, monkeypatch)
    payload = {
        "user_id": 1, "username": "u",
        "token": "x", "token_hash": "y",
        "expires_at": time.time() - 10,
    }
    a._SESSION_RECOVERY_FILE.write_text(json.dumps(payload))
    a._refresh_session_recovery_token()
    # File contents unchanged (still expired).
    after = json.loads(a._SESSION_RECOVERY_FILE.read_text())
    assert after["expires_at"] == payload["expires_at"]


# ── Constants ──────────────────────────────────────────────────────────────

def test_ttl_is_4_hours(tmp_path, monkeypatch):
    a = _import_auth_ui_with_tmp_files(tmp_path, monkeypatch)
    assert a._SESSION_RECOVERY_TTL_SECONDS == 4 * 3600


def test_recovery_ttl_shorter_than_remember_me(tmp_path, monkeypatch):
    """Session recovery is meant to be the short-lived 'survive
    WebSocket disconnects' token; Remember Me is the long-lived
    'persist across browser restarts' token. The recovery TTL MUST be
    strictly shorter than the Remember Me TTL — otherwise they're
    redundant."""
    a = _import_auth_ui_with_tmp_files(tmp_path, monkeypatch)
    assert a._SESSION_RECOVERY_TTL_SECONDS < a._REMEMBER_TTL_SECONDS


# ── Integration: login flow now always writes recovery token ──────────────

def test_auth_module_calls_save_in_init_state(tmp_path, monkeypatch):
    """`_init_auth_state` must auto-login from the recovery token when
    Remember Me is absent."""
    a = _import_auth_ui_with_tmp_files(tmp_path, monkeypatch)

    # Reset session state.
    import streamlit as st
    st.session_state.clear()
    # No Remember Me, but valid session recovery.
    a._save_session_recovery_token(user_id=99, username="charlie")
    assert not a._REMEMBER_FILE.exists()

    a._init_auth_state()
    assert st.session_state.get("logged_in") is True
    assert st.session_state.get("user_id") == 99
    assert st.session_state.get("username") == "charlie"


def test_remember_me_takes_precedence_over_recovery(tmp_path, monkeypatch):
    """When both tokens exist, Remember Me should win (longer TTL means
    it's the more 'authoritative' source)."""
    a = _import_auth_ui_with_tmp_files(tmp_path, monkeypatch)
    import streamlit as st
    st.session_state.clear()

    # Save both, but with different user_ids so we can tell which won.
    a._save_session_recovery_token(user_id=99, username="recovery_user")
    a._save_remember_token(user_id=100, username="remember_user")

    a._init_auth_state()
    assert st.session_state.get("user_id") == 100
    assert st.session_state.get("username") == "remember_user"


def test_recovery_token_used_when_remember_me_missing(tmp_path, monkeypatch):
    """Session-recovery is the fallback when Remember Me wasn't opted into."""
    a = _import_auth_ui_with_tmp_files(tmp_path, monkeypatch)
    import streamlit as st
    st.session_state.clear()
    a._save_session_recovery_token(user_id=5, username="dora")

    a._init_auth_state()
    assert st.session_state.get("logged_in") is True
    assert st.session_state.get("user_id") == 5


def test_recovery_token_expired_does_not_auto_login(tmp_path, monkeypatch):
    a = _import_auth_ui_with_tmp_files(tmp_path, monkeypatch)
    import streamlit as st
    st.session_state.clear()
    # Write an expired recovery token.
    payload = {
        "user_id": 5, "username": "dora",
        "token": "x", "token_hash": "y",
        "expires_at": time.time() - 10,
    }
    a._SESSION_RECOVERY_FILE.write_text(json.dumps(payload))

    a._init_auth_state()
    assert st.session_state.get("logged_in") is False
