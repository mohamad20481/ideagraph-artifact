"""Tests for the run-scoped release callback (the fix to the per-user
counter bleed regression the adversarial verifier caught).

Critical guarantees verified here:
  1. Calling release_once twice (worker finally + main-thread drain)
     decrements the counter ONLY ONCE — no bleed.
  2. Two concurrent runs by the same user get DIFFERENT callbacks; each
     releases independently. After both complete, counter = 0; if only
     one completes, counter = 1.
  3. Global counter has the same single-decrement-per-run property.
  4. Race: two threads calling release_once simultaneously — exactly
     one wins, the other no-ops.
  5. _release_run pulls the callback from session_state and fires it.
  6. The callback survives across success / failure / no-success cases.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── Helpers ─────────────────────────────────────────────────────────────────

def _fresh_guards():
    """Reset the singleton store so each test starts with clean counters."""
    from production_optimization import _SINGLETONS, get_concurrency_guard, get_quota_enforcer
    _SINGLETONS.pop("concurrency_guard", None)
    _SINGLETONS.pop("quota_enforcer", None)
    return get_concurrency_guard(), get_quota_enforcer()


def _make_release_once_via_app(user_id):
    """Pull the real _make_release_once from app.py. Importing app at
    module level loads Streamlit + sidebar — avoid that by stubbing st."""
    # The function _make_release_once only uses `threading` and the
    # production_optimization module — both are import-safe — so we can
    # exec it in isolation. But importing app.py will trigger the full
    # Streamlit init. Use the same minimal stub other tests use.
    import importlib
    import app  # already loaded by previous tests via conftest path setup
    importlib.reload(app)
    return app._make_release_once(user_id)


# ── Test 1: Single-release semantics (the original bug's REGRESSION fix) ───

def test_release_once_decrements_once_when_called_twice():
    """The whole point of the lock+flag: worker AND main both call,
    only the first does the actual work."""
    guard, quota = _fresh_guards()
    guard.acquire(user_id=11)
    quota._reserved[11] = 1
    assert guard.snapshot()[11] == 1
    assert quota.snapshot()[11] == 1

    release_once = _make_release_once_via_app(user_id=11)
    release_once(success=True)
    release_once(success=True)  # second call — should no-op
    release_once(success=False)  # third call — should also no-op

    assert guard.snapshot() == {}, "guard counter bled"
    assert quota.snapshot() == {}, "quota counter bled"


def test_release_once_per_run_does_not_bleed_into_other_runs():
    """The actual regression caught by adversarial verify: with 2
    in-flight runs for user 11, completing one should leave the
    counter at 1, not 0."""
    guard, quota = _fresh_guards()
    # Simulate Run A and Run B both in flight.
    guard.acquire(user_id=11)
    guard.acquire(user_id=11)
    quota._reserved[11] = 2

    release_run_a = _make_release_once_via_app(user_id=11)
    release_run_b = _make_release_once_via_app(user_id=11)
    assert guard.snapshot()[11] == 2

    # Run A completes. BOTH the worker's finally AND the main-thread
    # _release_run for Run A fire release_run_a (the bug was that BOTH
    # calls actually decremented the counter — bleeding into Run B's slot).
    release_run_a(success=True)
    release_run_a(success=True)
    release_run_a(success=False)

    # Run B is still in flight — counter should be exactly 1.
    assert guard.snapshot()[11] == 1, \
        f"counter bled into Run B's slot: {guard.snapshot()}"
    assert quota.snapshot()[11] == 1

    # Run B completes — both call sites fire release_run_b.
    release_run_b(success=True)
    release_run_b(success=True)
    assert guard.snapshot() == {}
    assert quota.snapshot() == {}


def test_release_once_global_counter_does_not_bleed():
    """Same guarantee on the global counter — the cause of the
    'server exceeds global_max' bug."""
    guard, _ = _fresh_guards()
    guard.acquire(user_id=11)
    guard.acquire(user_id=22)
    assert guard.stats()["global_active"] == 2

    release_a = _make_release_once_via_app(user_id=11)
    # Hammer it.
    for _ in range(10):
        release_a(success=True)
    # User 22 still in flight → global counter should be exactly 1.
    assert guard.stats()["global_active"] == 1


def test_release_once_concurrent_threads_race_for_first():
    """If both the worker thread AND the main thread call release_once
    SIMULTANEOUSLY, exactly one wins (the lock serializes them).
    Hammer it with 20 threads to make any race obvious."""
    guard, quota = _fresh_guards()
    guard.acquire(user_id=11)
    quota._reserved[11] = 1
    release_once = _make_release_once_via_app(user_id=11)

    barrier = threading.Barrier(20)

    def _hammer():
        barrier.wait()
        release_once(success=True)

    threads = [threading.Thread(target=_hammer) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Counter decremented EXACTLY once, regardless of 20 concurrent calls.
    assert guard.snapshot() == {}
    assert quota.snapshot() == {}


def test_release_once_handles_no_user_id():
    """User_id=None (anonymous) should not crash — quota release is
    skipped, concurrency release passes user_id=None to guard."""
    guard, _ = _fresh_guards()
    guard.acquire(user_id=None)  # anonymous slot, global only
    release_once = _make_release_once_via_app(user_id=None)
    release_once(success=False)
    # No crash; global counter decremented.
    assert guard.stats()["global_active"] == 0


# ── Test 2: _release_run reads callback from session_state ──────────────────

def test_release_run_invokes_session_state_callback():
    """When _release_run is called, it pops _release_callback and
    invokes it. After firing, the key is gone."""
    _fresh_guards()
    import app
    # Build a callback that just sets a flag so we know it fired.
    called = {"with_success": None}
    def _cb(success: bool):
        called["with_success"] = success
    # Stash a fake st.session_state on the app module.
    with patch.object(app, "st") as mock_st:
        mock_st.session_state = {"_release_callback": _cb}
        app._release_run(success=True)
    assert called["with_success"] is True
    assert "_release_callback" not in mock_st.session_state


def test_release_run_handles_missing_callback():
    """Old session state with no callback (or already-popped) shouldn't
    crash."""
    _fresh_guards()
    import app
    with patch.object(app, "st") as mock_st:
        mock_st.session_state = {}  # no callback
        # Should not raise.
        app._release_run(success=False)


def test_release_run_clears_legacy_flags():
    """Backward-compat: _release_run also pops the old
    _quota_acquired / _concurrency_acquired flags even though it
    doesn't act on them (the callback is the action path)."""
    _fresh_guards()
    import app
    with patch.object(app, "st") as mock_st:
        mock_st.session_state = {
            "_quota_acquired": True,
            "_concurrency_acquired": True,
        }
        app._release_run(success=False)
    assert "_quota_acquired" not in mock_st.session_state
    assert "_concurrency_acquired" not in mock_st.session_state


# ── Test 3: Success vs failure propagates correctly ─────────────────────────

def test_release_once_success_true_commits_monthly_count():
    """release_once(success=True) should trigger
    QuotaEnforcer.release_run(success=True) which calls
    db.increment_run_count. Verified by mocking the db import."""
    _fresh_guards()
    release_once = _make_release_once_via_app(user_id=11)
    with patch("db.increment_run_count") as mock_inc:
        release_once(success=True)
    mock_inc.assert_called_once_with(11)


def test_release_once_success_false_skips_monthly_count():
    """release_once(success=False) → quota release with success=False →
    no db.increment_run_count call (consistent with existing behavior
    for failed runs)."""
    _fresh_guards()
    release_once = _make_release_once_via_app(user_id=11)
    with patch("db.increment_run_count") as mock_inc:
        release_once(success=False)
    mock_inc.assert_not_called()


def test_release_once_first_call_wins_success_argument():
    """If the worker's finally fires first with success=True, but the
    main thread then fires with success=False, the run STILL counts
    (first call wins). Important: prevents a race from silently
    failing to bill a successful run."""
    _fresh_guards()
    release_once = _make_release_once_via_app(user_id=11)
    with patch("db.increment_run_count") as mock_inc:
        release_once(success=True)   # worker says success
        release_once(success=False)  # main thread overrides — but no-op
    mock_inc.assert_called_once_with(11)


# ── Regression: empty-result runs must NOT be billed to monthly quota ──────
# Caught by the v2-adversarial-verify workflow: my first attempt set
# `_success = True` whenever the pipeline didn't raise, which is too
# permissive — pipelines that returned an empty result were historically
# NOT billed (see legacy _drain_queue's payload-non-empty gate). The
# fix gates on `bool(results and results.get("ideas"))` so the worker
# matches that semantic. These tests lock it in.

def test_empty_result_pipeline_does_not_bill_monthly_quota():
    """A pipeline run that returns no ideas should NOT increment the
    monthly quota counter, even though no exception was raised."""
    _fresh_guards()
    import importlib
    import app
    importlib.reload(app)

    release_once = app._make_release_once(user_id=11)
    with patch("db.increment_run_count") as mock_inc:
        # Simulate the worker computing success from an empty result.
        empty_results = {"ideas": []}
        success = bool(empty_results and empty_results.get("ideas"))
        release_once(success=success)
    mock_inc.assert_not_called()


def test_none_result_pipeline_does_not_bill_monthly_quota():
    """`results = None` (pipeline aborted before producing output) → not billed."""
    _fresh_guards()
    import importlib
    import app
    importlib.reload(app)

    release_once = app._make_release_once(user_id=11)
    with patch("db.increment_run_count") as mock_inc:
        results = None
        success = bool(results and results.get("ideas"))  # type: ignore
        release_once(success=success)
    mock_inc.assert_not_called()


def test_non_empty_result_pipeline_IS_billed():
    """The positive side: when the worker computes success on a real
    non-empty result, the run IS billed."""
    _fresh_guards()
    import importlib
    import app
    importlib.reload(app)

    release_once = app._make_release_once(user_id=11)
    with patch("db.increment_run_count") as mock_inc:
        results = {"ideas": [{"title": "an idea"}, {"title": "another"}]}
        success = bool(results and results.get("ideas"))
        release_once(success=success)
    mock_inc.assert_called_once_with(11)


# ── Helper-level tests: _run_produced_ideas is the single source of truth ──

def test_run_produced_ideas_empty_list_false():
    import app
    assert app._run_produced_ideas({"ideas": []}) is False


def test_run_produced_ideas_none_false():
    import app
    assert app._run_produced_ideas(None) is False


def test_run_produced_ideas_missing_key_false():
    import app
    assert app._run_produced_ideas({"topic": "x"}) is False


def test_run_produced_ideas_non_dict_false():
    """Defensive: a string or int as 'results' shouldn't crash."""
    import app
    assert app._run_produced_ideas("not a dict") is False
    assert app._run_produced_ideas(42) is False


def test_run_produced_ideas_non_list_false():
    """If 'ideas' came back as a non-list (upstream bug), don't bill."""
    import app
    assert app._run_produced_ideas({"ideas": 5}) is False
    assert app._run_produced_ideas({"ideas": "string"}) is False
    assert app._run_produced_ideas({"ideas": None}) is False


def test_run_produced_ideas_non_empty_list_true():
    import app
    assert app._run_produced_ideas({"ideas": [{"title": "x"}]}) is True
    assert app._run_produced_ideas(
        {"ideas": [{"a": 1}, {"b": 2}, {"c": 3}]}
    ) is True


# ── INTEGRATION tests: actually drive the worker threads end-to-end ────────
# These guard against the test/code drift the v3-verify workflow flagged:
# the math tests above only verify the gate FORMULA, not that the workers
# call the gate. The integration tests below drive _run_pipeline_thread
# with a mocked pipeline.run, then assert that release_once was called
# with the success flag matching what _run_produced_ideas would compute.


import queue as _stdlib_queue


def _captured_release_once():
    """Build a release_once-shaped MagicMock that records the success arg."""
    cb = MagicMock()
    return cb


def test_pipeline_worker_empty_results_calls_release_once_with_false():
    """End-to-end: mock pipeline.run() to return {'ideas': []};
    verify worker fires release_once(success=False)."""
    _fresh_guards()
    import importlib
    import app
    importlib.reload(app)

    cb = _captured_release_once()
    q = _stdlib_queue.Queue()
    fake_pipeline = MagicMock()
    fake_pipeline.run.return_value = {"ideas": [], "topic": "x"}
    with patch("pipeline.IdeaGraphPipeline", return_value=fake_pipeline):
        app._run_pipeline_thread(
            topic="x", budget=1.0, iterations=1,
            provider="", model="",
            progress_queue=q,
            debate_enabled=False,
            user_id=11,
            runtime_controller=None,
            release_once=cb,
        )
    # Worker's finally must have fired the callback.
    assert cb.call_count == 1
    # And with success=False (empty ideas → not billable).
    assert cb.call_args.kwargs.get("success") is False


def test_pipeline_worker_non_empty_results_calls_release_once_with_true():
    """End-to-end positive: non-empty ideas → success=True → billable."""
    _fresh_guards()
    import importlib
    import app
    importlib.reload(app)

    cb = _captured_release_once()
    q = _stdlib_queue.Queue()
    fake_pipeline = MagicMock()
    fake_pipeline.run.return_value = {
        "ideas": [{"title": "a"}, {"title": "b"}],
        "topic": "x",
        "coverage": 0.5,
    }
    # Mock db.save_result so we don't write to the real db during the test.
    with patch("pipeline.IdeaGraphPipeline", return_value=fake_pipeline), \
         patch("db.save_result"), patch("db_cache.invalidate_user_results"):
        app._run_pipeline_thread(
            topic="x", budget=1.0, iterations=1,
            provider="", model="",
            progress_queue=q,
            debate_enabled=False,
            user_id=11,
            runtime_controller=None,
            release_once=cb,
        )
    assert cb.call_count == 1
    assert cb.call_args.kwargs.get("success") is True


def test_pipeline_worker_exception_calls_release_once_with_false():
    """If pipeline.run raises, success stays False (default) → not billed."""
    _fresh_guards()
    import importlib
    import app
    importlib.reload(app)

    cb = _captured_release_once()
    q = _stdlib_queue.Queue()
    fake_pipeline = MagicMock()
    fake_pipeline.run.side_effect = RuntimeError("kaboom")
    with patch("pipeline.IdeaGraphPipeline", return_value=fake_pipeline):
        app._run_pipeline_thread(
            topic="x", budget=1.0, iterations=1,
            provider="", model="",
            progress_queue=q,
            debate_enabled=False,
            user_id=11,
            runtime_controller=None,
            release_once=cb,
        )
    assert cb.call_count == 1
    assert cb.call_args.kwargs.get("success") is False


def test_scientist_worker_empty_results_calls_release_once_with_false():
    """Mirror of test_pipeline_worker_empty_results_calls_release_once_with_false
    but driving _run_scientist_thread. Closes the asymmetric-coverage gap
    the v4 reviewers flagged: previously only _run_pipeline_thread had
    end-to-end integration coverage."""
    _fresh_guards()
    import importlib
    import app
    importlib.reload(app)

    cb = _captured_release_once()
    q = _stdlib_queue.Queue()
    fake_scientist = MagicMock()
    fake_scientist.run.return_value = {"ideas": [], "topic": "x"}
    with patch("pipeline_v2.AutomatedScientist", return_value=fake_scientist):
        app._run_scientist_thread(
            topic="x", budget=1.0, iterations=1,
            provider="", model="",
            progress_queue=q,
            debate_enabled=False,
            exec_timeout=60, max_sci_iters=1,
            user_id=11,
            release_once=cb,
        )
    assert cb.call_count == 1
    assert cb.call_args.kwargs.get("success") is False


def test_scientist_worker_non_empty_results_calls_release_once_with_true():
    """End-to-end positive: scientist returns ideas → success=True → billable."""
    _fresh_guards()
    import importlib
    import app
    importlib.reload(app)

    cb = _captured_release_once()
    q = _stdlib_queue.Queue()
    fake_scientist = MagicMock()
    fake_scientist.run.return_value = {
        "ideas": [{"title": "a"}, {"title": "b"}],
        "topic": "x",
        "coverage": 0.5,
    }
    with patch("pipeline_v2.AutomatedScientist", return_value=fake_scientist), \
         patch("db.save_result"), patch("db_cache.invalidate_user_results"):
        app._run_scientist_thread(
            topic="x", budget=1.0, iterations=1,
            provider="", model="",
            progress_queue=q,
            debate_enabled=False,
            exec_timeout=60, max_sci_iters=1,
            user_id=11,
            release_once=cb,
        )
    assert cb.call_count == 1
    assert cb.call_args.kwargs.get("success") is True


def test_scientist_worker_exception_calls_release_once_with_false():
    """Scientist raise → success=False (default) → not billed."""
    _fresh_guards()
    import importlib
    import app
    importlib.reload(app)

    cb = _captured_release_once()
    q = _stdlib_queue.Queue()
    fake_scientist = MagicMock()
    fake_scientist.run.side_effect = RuntimeError("scientist boom")
    with patch("pipeline_v2.AutomatedScientist", return_value=fake_scientist):
        app._run_scientist_thread(
            topic="x", budget=1.0, iterations=1,
            provider="", model="",
            progress_queue=q,
            debate_enabled=False,
            exec_timeout=60, max_sci_iters=1,
            user_id=11,
            release_once=cb,
        )
    assert cb.call_count == 1
    assert cb.call_args.kwargs.get("success") is False


def test_no_drift_all_billing_sites_call_run_produced_ideas():
    """Drift detector — checks the actual app.py SOURCE to confirm
    every billing-decision site calls _run_produced_ideas (not the
    inline `bool(results and results.get('ideas'))` formula).

    This is what actually protects against future regression: if
    someone reverts one of the three call sites to the inline form,
    the source grep fails here. The 'check the function returns the
    same value twice' shape this replaces was tautological since
    _run_produced_ideas is pure.

    The three required sites:
      - app.py _run_pipeline_thread   → _success = _run_produced_ideas(...)
      - app.py _run_scientist_thread  → _scientist_success = _run_produced_ideas(...)
      - app.py _drain_queue done      → _release_run(success=_run_produced_ideas(...))
    """
    import re
    app_src = (ROOT / "app.py").read_text(encoding="utf-8")

    # Must find at least 3 distinct call sites of the helper.
    call_sites = re.findall(r"_run_produced_ideas\s*\(", app_src)
    # 1 definition (`def _run_produced_ideas(`) + 3 calls = 4 total occurrences.
    assert len(call_sites) >= 4, (
        f"Expected ≥4 references to _run_produced_ideas in app.py "
        f"(1 def + ≥3 calls), found {len(call_sites)}. Someone may "
        f"have reverted a billing site to inline formula."
    )

    # And the helper definition must still exist.
    assert "def _run_produced_ideas(" in app_src, (
        "Helper _run_produced_ideas definition is missing from app.py — "
        "the drift-protection invariant is gone."
    )

    # And no remaining INLINE billing-formula on a release_once or
    # release_run call (only the helper should be allowed there).
    # We search for the legacy v3 inline form: `success=bool(... and ...get("ideas"))`.
    legacy_inline = re.findall(
        r"success\s*=\s*bool\s*\(\s*\w+\s+and\s+\w+\.get\(['\"]ideas['\"]",
        app_src,
    )
    assert not legacy_inline, (
        "Found legacy inline billing formula in a release_*/release_run "
        "call — should use _run_produced_ideas instead. Matches: "
        f"{legacy_inline}"
    )


# ── Test 4: idempotency across guards module reload safety ─────────────────

def test_release_once_handles_guards_module_reload(monkeypatch):
    """If production_optimization is reloaded between callback creation
    and callback firing (rare but possible during dev), the callback
    should still gracefully no-op without crashing."""
    _fresh_guards()
    release_once = _make_release_once_via_app(user_id=11)
    # Simulate a broken import by monkey-patching get_concurrency_guard
    # to raise.
    import production_optimization
    original = production_optimization.get_concurrency_guard

    def _boom():
        raise RuntimeError("module reloaded")

    monkeypatch.setattr(production_optimization, "get_concurrency_guard", _boom)
    # Should not raise.
    release_once(success=True)
    monkeypatch.setattr(production_optimization, "get_concurrency_guard", original)
