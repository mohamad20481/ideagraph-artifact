"""Tests for the stuck-runs fix.

Covers:
  - ConcurrencyGuard.reset_user / clear / snapshot
  - QuotaEnforcer.reset_user / clear / snapshot
  - Idempotent release (clamps at 0)
  - Admin _render_active_runs_panel smoke (renders no-runs + with-runs)
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from production_optimization import ConcurrencyGuard, QuotaEnforcer


# ── ConcurrencyGuard.reset_user / clear / snapshot ─────────────────────────

def test_concurrency_guard_reset_user_zeroes_count_and_returns_n():
    g = ConcurrencyGuard(per_user_max=5)
    g.acquire(user_id=11)
    g.acquire(user_id=11)
    g.acquire(user_id=11)
    assert g.snapshot() == {11: 3}
    n = g.reset_user(11)
    assert n == 3
    assert g.snapshot() == {}
    # Global also decremented by the same amount.
    assert g.stats()["global_active"] == 0


def test_concurrency_guard_reset_user_only_affects_that_user():
    g = ConcurrencyGuard(per_user_max=5)
    g.acquire(user_id=11)
    g.acquire(user_id=11)
    g.acquire(user_id=22)
    g.reset_user(11)
    snap = g.snapshot()
    assert 11 not in snap
    assert snap.get(22) == 1
    # Global = only user 22's slot left.
    assert g.stats()["global_active"] == 1


def test_concurrency_guard_reset_unknown_user_returns_zero():
    g = ConcurrencyGuard()
    assert g.reset_user(99999) == 0


def test_concurrency_guard_clear_zeroes_everything():
    g = ConcurrencyGuard()
    g.acquire(user_id=11)
    g.acquire(user_id=22)
    g.acquire(user_id=22)
    total = g.clear()
    assert total == 3
    assert g.snapshot() == {}
    assert g.stats()["global_active"] == 0


def test_concurrency_guard_snapshot_filters_zeros():
    """A user whose slots were released back to 0 should NOT appear
    in snapshot (defaultdict can accumulate keys with value 0)."""
    g = ConcurrencyGuard()
    g.acquire(user_id=11)
    g.release(user_id=11)
    # After release, _per_user[11] == 0, but it's still a key.
    assert g.snapshot() == {}


def test_concurrency_release_clamps_at_zero():
    """Double-release should leave counter at 0, not negative."""
    g = ConcurrencyGuard()
    g.acquire(user_id=11)
    g.release(user_id=11)
    g.release(user_id=11)  # second release on already-zero
    g.release(user_id=11)  # third release on already-zero
    assert g.snapshot() == {}
    assert g.stats()["global_active"] == 0


def test_concurrency_per_user_max_3_default():
    """Verify the default per_user_max=3 matches the error message users see."""
    g = ConcurrencyGuard(per_user_max=3)
    assert g.acquire(user_id=11)[0] is True
    assert g.acquire(user_id=11)[0] is True
    assert g.acquire(user_id=11)[0] is True
    ok, msg = g.acquire(user_id=11)
    assert ok is False
    assert "3 runs in progress" in msg
    assert "max 3" in msg


def test_concurrency_acquire_succeeds_after_reset_user():
    """The whole point of reset_user: a stuck user can acquire again."""
    g = ConcurrencyGuard(per_user_max=3)
    g.acquire(user_id=11); g.acquire(user_id=11); g.acquire(user_id=11)
    assert g.acquire(user_id=11)[0] is False
    g.reset_user(11)
    assert g.acquire(user_id=11)[0] is True


# ── QuotaEnforcer.reset_user / clear / snapshot ────────────────────────────

def test_quota_enforcer_reset_user_zeroes_reservation():
    q = QuotaEnforcer()
    # Manually populate to bypass db dependency.
    q._reserved[11] = 5
    n = q.reset_user(11)
    assert n == 5
    assert q.snapshot() == {}


def test_quota_enforcer_clear_zeroes_all():
    q = QuotaEnforcer()
    q._reserved[11] = 2
    q._reserved[22] = 3
    total = q.clear()
    assert total == 5
    assert q.snapshot() == {}


def test_quota_enforcer_snapshot_filters_zeros():
    q = QuotaEnforcer()
    q._reserved[11] = 2
    q._reserved[22] = 0  # stale entry
    assert q.snapshot() == {11: 2}


def test_quota_enforcer_release_run_clamps_at_zero():
    """Double release_run shouldn't go negative or fall over."""
    q = QuotaEnforcer()
    q._reserved[11] = 1
    q.release_run(11, success=False)
    q.release_run(11, success=False)
    q.release_run(11, success=False)
    assert q._reserved[11] == 0


def test_quota_enforcer_reset_unknown_user_returns_zero():
    q = QuotaEnforcer()
    assert q.reset_user(99999) == 0


# ── Singleton + cross-thread integration ───────────────────────────────────

def test_get_concurrency_guard_returns_singleton():
    from production_optimization import get_concurrency_guard, _SINGLETONS
    # Clear before test to avoid leakage from other tests.
    _SINGLETONS.pop("concurrency_guard", None)
    g1 = get_concurrency_guard()
    g2 = get_concurrency_guard()
    assert g1 is g2


def test_get_quota_enforcer_returns_singleton():
    from production_optimization import get_quota_enforcer, _SINGLETONS
    _SINGLETONS.pop("quota_enforcer", None)
    q1 = get_quota_enforcer()
    q2 = get_quota_enforcer()
    assert q1 is q2


def test_concurrent_acquires_then_reset_via_singleton():
    """End-to-end via the public singleton accessors — what the admin
    panel actually calls."""
    from production_optimization import (
        get_concurrency_guard, get_quota_enforcer, _SINGLETONS,
    )
    _SINGLETONS.pop("concurrency_guard", None)
    _SINGLETONS.pop("quota_enforcer", None)
    g = get_concurrency_guard()
    q = get_quota_enforcer()
    g.acquire(user_id=11); g.acquire(user_id=11)
    q._reserved[11] = 2
    assert g.snapshot()[11] == 2
    assert q.snapshot()[11] == 2
    # The admin reset path:
    g.reset_user(11)
    q.reset_user(11)
    assert g.snapshot() == {}
    assert q.snapshot() == {}


# ── Admin panel smoke ──────────────────────────────────────────────────────

def _make_st_stub():
    stub = MagicMock()

    def _make_col_mock():
        c = MagicMock()
        c.button = MagicMock(return_value=False)
        c.metric = MagicMock()
        c.markdown = MagicMock()
        return c

    stub.columns.side_effect = lambda spec, **kw: (
        [_make_col_mock() for _ in range(spec)]
        if isinstance(spec, int)
        else [_make_col_mock() for _ in spec]
    )
    stub.button.return_value = False
    stub.session_state = {}
    return stub


def test_active_runs_panel_renders_empty_state():
    """No active runs → success banner + zero metrics."""
    from production_optimization import _SINGLETONS
    _SINGLETONS.pop("concurrency_guard", None)
    _SINGLETONS.pop("quota_enforcer", None)
    import admin_dashboard
    st = _make_st_stub()
    admin_dashboard._render_active_runs_panel(st)
    # The success message about "no active or stuck runs" was shown.
    success_args = [
        c.args[0] for c in st.success.call_args_list
        if c.args and isinstance(c.args[0], str)
    ]
    assert any("No active or stuck runs" in s for s in success_args)


def test_active_runs_panel_renders_with_held_slots():
    """Active runs exist → panel renders user rows (no success banner)."""
    from production_optimization import (
        get_concurrency_guard, _SINGLETONS,
    )
    _SINGLETONS.pop("concurrency_guard", None)
    _SINGLETONS.pop("quota_enforcer", None)
    g = get_concurrency_guard()
    g.acquire(user_id=11)
    g.acquire(user_id=11)

    import admin_dashboard
    st = _make_st_stub()
    admin_dashboard._render_active_runs_panel(st)
    # The "no active" banner should NOT have fired.
    success_args = [
        c.args[0] for c in st.success.call_args_list
        if c.args and isinstance(c.args[0], str)
    ]
    assert not any("No active or stuck runs" in s for s in success_args)
    # The "Users with held slots" header should have rendered.
    md_args = [
        c.args[0] for c in st.markdown.call_args_list
        if c.args and isinstance(c.args[0], str)
    ]
    assert any("Users with held slots" in m for m in md_args)
    # Clean up — don't leak state into other tests.
    g.clear()


def test_active_runs_panel_has_reset_all_button():
    """Verify the 🚨 Reset ALL button is rendered."""
    from production_optimization import _SINGLETONS
    _SINGLETONS.pop("concurrency_guard", None)
    _SINGLETONS.pop("quota_enforcer", None)
    import admin_dashboard
    st = _make_st_stub()
    admin_dashboard._render_active_runs_panel(st)
    btn_keys = [
        c.kwargs.get("key") for c in st.button.call_args_list
    ]
    assert "runs_reset_all" in btn_keys
