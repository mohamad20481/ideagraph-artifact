"""
auth_ui.py - Streamlit UI components for user authentication.

Provides login/register page, sidebar user menu, and "Remember Me"
persistent login via a local token file.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import streamlit as st

import db
from production_optimization import (
    generate_session_token,
    hash_token,
    validate_password,
    validate_username,
)

# Remember-me tokens expire after 7 days.
_REMEMBER_TTL_SECONDS = 7 * 24 * 3600

# Session-recovery token: always-on, short TTL. Solves the "logged out
# in the middle of a long LLM generation" problem caused by Streamlit's
# per-WebSocket session_state. When the WebSocket disconnects (Kaspersky
# injection failures, Chrome memory saver, OS network suspension on long
# LLM calls, Streamlit idle timeout), Streamlit creates a fresh session
# with default `logged_in=False`. Without this token the user would have
# to log in again every time. With it, they auto-resume as long as the
# recovery file is < SESSION_RECOVERY_TTL old.
#
# Critically: this is refreshed on EVERY page render while logged in,
# so an active user's session keeps extending indefinitely. Idle users
# are logged out after the TTL.
_SESSION_RECOVERY_TTL_SECONDS = 4 * 3600   # 4 hours

# ── Remember Me + Session Recovery token files ──────────────────────────────
_REMEMBER_FILE = Path(__file__).parent / "data" / ".remember_token.json"
_SESSION_RECOVERY_FILE = (
    Path(__file__).parent / "data" / ".session_recovery.json"
)


def _save_remember_token(user_id: int, username: str) -> None:
    """Save login token to disk for persistent 'Remember Me'.

    Security model:
      - Token is cryptographically random (secrets.token_urlsafe, 256-bit).
      - Only the SHA-256 hash is compared server-side; the plaintext lives
        solely on the user's machine, so DB leaks can't forge sessions.
      - Token carries an expiry (_REMEMBER_TTL_SECONDS).
    """
    import time
    try:
        os.makedirs(_REMEMBER_FILE.parent, exist_ok=True)
        token = generate_session_token()
        token_hash = hash_token(token)
        expires_at = time.time() + _REMEMBER_TTL_SECONDS
        payload = {
            "user_id": user_id,
            "username": username,
            "token": token,
            "token_hash": token_hash,  # server-side comparison key
            "expires_at": expires_at,
        }
        with open(_REMEMBER_FILE, "w") as f:
            json.dump(payload, f)
        try:
            os.chmod(_REMEMBER_FILE, 0o600)  # owner-only on POSIX; best-effort on Windows
        except Exception:
            pass
    except Exception:
        pass


def _load_remember_token() -> dict:
    """Load saved login token from disk. Returns {} if not found or expired."""
    import time
    try:
        if _REMEMBER_FILE.exists():
            with open(_REMEMBER_FILE) as f:
                data = json.load(f)
            # Expiry check — treat missing expires_at (legacy token) as expired.
            expires_at = data.get("expires_at")
            if not expires_at or time.time() > float(expires_at):
                _clear_remember_token()
                return {}
            # Integrity check — verify stored hash matches plaintext token.
            tok = data.get("token", "")
            expected = data.get("token_hash", "")
            if not tok or not expected or hash_token(tok) != expected:
                _clear_remember_token()
                return {}
            if data.get("user_id") and data.get("username"):
                return data
    except Exception:
        pass
    return {}


def _clear_remember_token() -> None:
    """Remove the remember-me token file on logout."""
    try:
        if _REMEMBER_FILE.exists():
            _REMEMBER_FILE.unlink()
    except Exception:
        pass


# ── Session recovery token (always-on, short TTL) ──────────────────────────

def _save_session_recovery_token(user_id: int, username: str) -> None:
    """Write a short-TTL session-recovery token.

    Different from Remember Me: this is written on EVERY successful
    login (not opt-in), with a much shorter TTL. Its purpose is to
    survive Streamlit's WebSocket disconnects + idle timeouts during
    long LLM operations, NOT to persist across browser restarts.
    """
    import time
    try:
        os.makedirs(_SESSION_RECOVERY_FILE.parent, exist_ok=True)
        token = generate_session_token()
        payload = {
            "user_id":    user_id,
            "username":   username,
            "token":      token,
            "token_hash": hash_token(token),
            "expires_at": time.time() + _SESSION_RECOVERY_TTL_SECONDS,
        }
        with open(_SESSION_RECOVERY_FILE, "w") as f:
            json.dump(payload, f)
        try:
            os.chmod(_SESSION_RECOVERY_FILE, 0o600)
        except Exception:
            pass
    except Exception:
        pass


def _load_session_recovery_token() -> dict:
    """Load + validate the session-recovery token. Returns {} if missing,
    expired, or tampered with."""
    import time
    try:
        if not _SESSION_RECOVERY_FILE.exists():
            return {}
        with open(_SESSION_RECOVERY_FILE) as f:
            data = json.load(f)
        expires_at = data.get("expires_at")
        if not expires_at or time.time() > float(expires_at):
            _clear_session_recovery_token()
            return {}
        tok = data.get("token", "")
        expected = data.get("token_hash", "")
        if not tok or not expected or hash_token(tok) != expected:
            _clear_session_recovery_token()
            return {}
        if data.get("user_id") and data.get("username"):
            return data
    except Exception:
        pass
    return {}


def _refresh_session_recovery_token() -> None:
    """Extend the session-recovery token's expiry to now + TTL.

    Called on every render while logged in (heartbeat). An idle session
    only expires after the FULL TTL of no activity; an active session
    never expires.
    """
    import time
    try:
        if not _SESSION_RECOVERY_FILE.exists():
            return
        with open(_SESSION_RECOVERY_FILE) as f:
            data = json.load(f)
        # Only refresh if the token is still valid (not tampered + not
        # expired). Avoid writing on every render if the existing
        # expires_at is already > halfway through the TTL — saves disk I/O.
        expires_at = float(data.get("expires_at") or 0.0)
        now = time.time()
        if expires_at <= now:
            return  # expired — let the next login re-create it
        # Only refresh once we're past 50% of the TTL window.
        if expires_at - now > _SESSION_RECOVERY_TTL_SECONDS * 0.5:
            return
        data["expires_at"] = now + _SESSION_RECOVERY_TTL_SECONDS
        with open(_SESSION_RECOVERY_FILE, "w") as f:
            json.dump(data, f)
    except Exception:
        pass


def _clear_session_recovery_token() -> None:
    try:
        if _SESSION_RECOVERY_FILE.exists():
            _SESSION_RECOVERY_FILE.unlink()
    except Exception:
        pass


def _init_auth_state() -> None:
    """Initialise auth-related session state keys."""
    defaults = {
        "logged_in": False,
        "user_id": None,
        "username": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ── Auto-login: try Remember Me first, then session-recovery ──────────
    # Remember Me has a longer TTL (7d) and survives browser restarts.
    # Session-recovery (4h) is the safety net for WebSocket loss mid-LLM-
    # generation when Remember Me wasn't opted into.
    if not st.session_state.get("logged_in"):
        saved = _load_remember_token() or _load_session_recovery_token()
        if saved:
            st.session_state.logged_in = True
            st.session_state.user_id = saved["user_id"]
            st.session_state.username = saved["username"]


_init_auth_state()


def is_logged_in() -> bool:
    """Check whether a user is currently logged in."""
    return bool(st.session_state.get("logged_in"))


def show_auth_page() -> None:
    """Render the login / register page (main area)."""
    # Centered branded header
    st.markdown(
        '<div style="text-align:center;padding:40px 0 20px 0">'
        '<div style="font-size:48px;margin-bottom:8px">🧠</div>'
        '<div style="font-size:32px;font-weight:700;color:#0c4a6e;letter-spacing:-0.5px">'
        'IdeaGraph</div>'
        '<div style="font-size:15px;color:#0369a1;margin-top:4px">'
        'AI-Powered Research Ideation Platform</div>'
        '<div style="width:60px;height:3px;background:linear-gradient(90deg,#38bdf8,#0ea5e9);'
        'border-radius:2px;margin:16px auto 0 auto"></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    tab_login, tab_register = st.tabs(["Login", "Register"])

    with tab_login:
        st.subheader("Login")
        with st.form("login_form", clear_on_submit=False):
            login_user = st.text_input(
                "Username", key="login_username",
                autocomplete="username",
            )
            login_pass = st.text_input(
                "Password", type="password", key="login_password",
                autocomplete="current-password",
            )
            remember_me = st.checkbox("Remember me", value=True, key="remember_me")
            submitted = st.form_submit_button("Login", type="primary", use_container_width=True)

        if submitted:
            # Rate-limit login attempts per-username to block credential stuffing.
            from production_optimization import get_rate_limiter
            rl_key = f"login:{login_user.strip().lower()}"
            ok_rl, rl_msg = get_rate_limiter().check(user_id=rl_key)
            if not ok_rl:
                st.error(rl_msg)
            elif not login_user.strip() or not login_pass:
                st.warning("Please enter both username and password.")
            else:
                user_id = db.login_user(login_user, login_pass)
                if user_id is not None:
                    st.session_state.logged_in = True
                    st.session_state.user_id = user_id
                    st.session_state.username = login_user.strip()
                    if remember_me:
                        _save_remember_token(user_id, login_user.strip())
                    # ALWAYS write a session-recovery token, regardless
                    # of Remember Me. Short TTL — survives long LLM
                    # generations + WebSocket disconnects, expires
                    # naturally on idle.
                    _save_session_recovery_token(
                        user_id, login_user.strip(),
                    )
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

    with tab_register:
        st.subheader("Create Account")
        with st.form("register_form", clear_on_submit=False):
            reg_user = st.text_input(
                "Username", key="reg_username",
                autocomplete="username",
            )
            reg_pass = st.text_input(
                "Password", type="password", key="reg_password",
                autocomplete="new-password",
            )
            reg_pass2 = st.text_input(
                "Confirm Password", type="password", key="reg_password2",
                autocomplete="new-password",
            )
            submitted = st.form_submit_button("Register", type="primary", use_container_width=True)

        if submitted:
            ok_user, user_msg = validate_username(reg_user)
            ok_pass, pass_msg = validate_password(reg_pass)
            if not ok_user:
                st.warning(user_msg)
            elif not ok_pass:
                st.warning(pass_msg)
            elif reg_pass != reg_pass2:
                st.error("Passwords do not match.")
            else:
                user_id = db.register_user(reg_user, reg_pass)
                if user_id is not None:
                    st.session_state.logged_in = True
                    st.session_state.user_id = user_id
                    st.session_state.username = reg_user.strip()
                    _save_session_recovery_token(
                        user_id, reg_user.strip(),
                    )
                    st.success("Account created! Redirecting...")
                    st.rerun()
                else:
                    st.error("Username already taken. Please choose another.")


def show_user_menu() -> None:
    """Render sidebar user info + logout button.

    Also refreshes the session-recovery token on every render. This is the
    'heartbeat' that keeps an active user's session alive indefinitely; an
    idle user's session expires only after the full TTL of no activity.
    """
    # Heartbeat — extend the recovery token's expiry while the user is
    # actively rendering pages. Cheap: only writes to disk once per ~2h.
    _refresh_session_recovery_token()

    username = st.session_state.get("username", "User")
    st.markdown(f"**{username}**")

    # ── Manage account ───────────────────────────────────────────────────
    # Toggles a session flag that app.py reads to switch the main area
    # into account_ui.render_account_page() (profile / password / billing
    # / usage / data export / delete-account).
    if st.button(
        "⚙️ Manage account",
        key="sidebar_manage_account",
        use_container_width=True,
        type="secondary",
    ):
        st.session_state["_show_account_page"] = True
        st.rerun()

    if st.button("Logout", use_container_width=True, type="secondary"):
        _clear_remember_token()
        _clear_session_recovery_token()
        st.session_state.logged_in = False
        st.session_state.user_id = None
        st.session_state.username = None
        # Clear pipeline state too
        st.session_state.running = False
        st.session_state.done = False
        st.session_state.results = None
        st.session_state.error = None
        st.session_state.progress_log = []
        st.session_state["_show_account_page"] = False
        st.rerun()


def show_saved_results_sidebar() -> None:
    """Render 'My Results' section in sidebar with modern cards + search."""
    user_id = st.session_state.get("user_id")
    if not user_id:
        return

    st.divider()

    total_count = len(db.get_user_results(user_id))

    # Section header with count badge
    st.markdown(
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'margin-bottom:8px">'
        f'<span style="font-size:12px;font-weight:700;color:#0369a1;'
        f'text-transform:uppercase;letter-spacing:0.08em">📚 My Results</span>'
        f'<span style="background:#0ea5e9;color:white;font-size:10px;font-weight:700;'
        f'padding:2px 8px;border-radius:10px">{total_count}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if total_count == 0:
        st.markdown(
            '<div style="background:rgba(255,255,255,0.6);border:1px dashed #bae6fd;'
            'border-radius:8px;padding:14px 12px;text-align:center;color:#64748b;'
            'font-size:12px">'
            '✨ No saved results yet.<br>Run a pipeline to get started!'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    # ── Search box ───────────────────────────────────────────────────────
    # Searches topic + idea content (titles/motivations/methods). Multiple
    # whitespace-separated terms are AND-ed. Empty input = show everything.
    search_q = st.text_input(
        "🔍 Search",
        value=st.session_state.get("_results_search_q", ""),
        key="_results_search_q",
        placeholder="e.g. attention, transformer privacy",
        label_visibility="collapsed",
        help="Searches topic + every idea's text. Multiple words = all must match.",
    )

    if search_q.strip():
        results_list = db.search_user_results(user_id, search_q.strip())
        if results_list:
            st.caption(
                f"🎯 {len(results_list)} of {total_count} match "
                f"`{search_q.strip()}`"
            )
        else:
            st.markdown(
                '<div style="background:#fef2f2;border:1px dashed #fecaca;'
                'border-radius:8px;padding:10px 12px;text-align:center;color:#7f1d1d;'
                'font-size:12px">'
                f"No results match <code>{search_q.strip()}</code>."
                '</div>',
                unsafe_allow_html=True,
            )
            return
    else:
        results_list = db.get_user_results(user_id)

    for r in results_list:
        rid = r["id"]
        topic = r["topic"][:55]
        coverage = r["coverage"]
        ideas_count = r["ideas_count"]
        created = r["created_at"][:10]  # date only

        # Quality color based on coverage
        _q_color = "#10b981" if coverage >= 0.5 else "#f59e0b" if coverage >= 0.25 else "#94a3b8"

        # Modern card
        st.markdown(
            f'<div style="background:white;border:1px solid #e0f2fe;border-left:3px solid {_q_color};'
            f'border-radius:8px;padding:10px 12px;margin-bottom:6px;'
            f'box-shadow:0 1px 2px rgba(14,165,233,0.05)">'
            f'<div style="font-size:12px;font-weight:600;color:#0c4a6e;'
            f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap;margin-bottom:4px">'
            f'{topic}</div>'
            f'<div style="display:flex;gap:6px;align-items:center;font-size:10px">'
            f'<span style="background:#f0f9ff;color:#0369a1;padding:1px 6px;'
            f'border-radius:4px;font-weight:600">{coverage:.0%} cov</span>'
            f'<span style="background:#f0fdf4;color:#166534;padding:1px 6px;'
            f'border-radius:4px;font-weight:600">{ideas_count} ideas</span>'
            f'<span style="color:#94a3b8;margin-left:auto">{created}</span>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Action buttons in a clean row
        col_load, col_del = st.columns([4, 1])
        with col_load:
            if st.button("📂 Load", key=f"load_{rid}", use_container_width=True):
                loaded = db.load_result(rid, user_id)
                if loaded:
                    # Stamp the result id so the Chat tab can key per-result
                    # transcript history off it.
                    try:
                        loaded["_result_id"] = rid
                    except Exception:
                        pass
                    st.session_state.results = loaded
                    st.session_state.done = True
                    st.session_state.running = False
                    st.session_state.error = None
                    st.session_state.progress_log = ["Loaded from saved results."]
                    st.rerun()
                else:
                    st.error("Failed to load result.")
        with col_del:
            if st.button("🗑️", key=f"del_{rid}", help="Delete this result", use_container_width=True):
                db.delete_result(rid, user_id)
                st.rerun()
