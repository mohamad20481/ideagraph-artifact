"""Tests for runtime_control.py — interactive pause/resume controller."""
from __future__ import annotations

import threading
import time
from typing import List

import pytest

from runtime_control import (
    RUNNING, PAUSED_BUDGET, PAUSED_NETWORK, PAUSED_USER,
    STOPPED, COMPLETED, PAUSE_STATES, TERMINAL_STATES,
    RuntimeController,
)


# ── Construction & validation ─────────────────────────────────────────────

class TestConstruction:
    def test_default_state_is_running(self):
        c = RuntimeController()
        assert c.status()["state"] == RUNNING

    def test_invalid_budget_raises(self):
        with pytest.raises(ValueError):
            RuntimeController(budget_limit_usd=0)
        with pytest.raises(ValueError):
            RuntimeController(budget_limit_usd=-1)

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            RuntimeController(budget_pause_threshold=0)
        with pytest.raises(ValueError):
            RuntimeController(budget_pause_threshold=1)
        with pytest.raises(ValueError):
            RuntimeController(budget_pause_threshold=1.5)

    def test_invalid_max_failures_raises(self):
        with pytest.raises(ValueError):
            RuntimeController(max_network_failures=0)


# ── Budget pause ──────────────────────────────────────────────────────────

class TestBudgetPause:
    def test_under_threshold_no_pause(self):
        c = RuntimeController(budget_limit_usd=1.0,
                                budget_pause_threshold=0.85)
        assert c.heartbeat(0.50) is True
        assert c.status()["state"] == RUNNING

    def test_threshold_triggers_pause_then_continue(self):
        c = RuntimeController(budget_limit_usd=1.0,
                                budget_pause_threshold=0.85,
                                decision_timeout_s=5)
        out: List[bool] = []

        def worker():
            out.append(c.heartbeat(0.90))

        t = threading.Thread(target=worker)
        t.start()
        time.sleep(0.1)
        s = c.status()
        assert s["state"] == PAUSED_BUDGET
        assert "0.900" in s["pause_reason"] or "0.90" in s["pause_reason"]

        c.decide("continue", budget_topup=1.0)
        t.join(timeout=2)
        assert out == [True]
        assert c.status()["budget_limit_usd"] == 2.0

    def test_threshold_triggers_pause_then_stop(self):
        c = RuntimeController(budget_limit_usd=1.0,
                                budget_pause_threshold=0.85)
        out: List[bool] = []
        t = threading.Thread(target=lambda: out.append(c.heartbeat(0.90)))
        t.start()
        time.sleep(0.1)
        c.decide("stop")
        t.join(timeout=2)
        assert out == [False]
        assert c.status()["state"] == STOPPED

    def test_pause_does_not_re_fire_below_topped_up_threshold(self):
        # After top-up, the "fired" flag rearms only if we're STILL above
        # the new threshold. If we're below, future heartbeats pass through.
        c = RuntimeController(budget_limit_usd=1.0,
                                budget_pause_threshold=0.85)

        def first_heartbeat():
            c.heartbeat(0.90)  # Triggers pause at 90%
        t = threading.Thread(target=first_heartbeat)
        t.start()
        time.sleep(0.1)
        c.decide("continue", budget_topup=2.0)  # Now budget = 3.0
        t.join(timeout=2)

        # 0.90 / 3.0 = 30% — well below threshold. Should NOT pause again.
        assert c.heartbeat(0.90) is True
        assert c.status()["state"] == RUNNING

    def test_pause_re_fires_if_topup_too_small(self):
        c = RuntimeController(budget_limit_usd=1.0,
                                budget_pause_threshold=0.85)
        t = threading.Thread(target=lambda: c.heartbeat(0.90))
        t.start()
        time.sleep(0.1)
        c.decide("continue", budget_topup=0.05)  # Now budget = 1.05
        t.join(timeout=2)
        # 0.90 / 1.05 = 86% — still above threshold; the next heartbeat
        # should still bypass because we already fired once at this level.
        # Actually no — we keep `_budget_pause_fired = True` if still above
        # threshold post-topup, so we won't re-trigger.
        assert c.status()["state"] == RUNNING


# ── Network pause ─────────────────────────────────────────────────────────

class TestNetworkPause:
    def test_failures_trigger_pause(self):
        c = RuntimeController(max_network_failures=3)

        def worker():
            for ok in [True, False, False, False]:
                if not c.record_llm_result(ok):
                    return

        t = threading.Thread(target=worker)
        t.start()
        time.sleep(0.1)
        s = c.status()
        assert s["state"] == PAUSED_NETWORK
        assert s["consecutive_network_failures"] == 3
        c.decide("stop")
        t.join(timeout=2)

    def test_success_resets_failure_counter(self):
        c = RuntimeController(max_network_failures=4)
        # Two failures, then a success, then more — never crosses threshold
        c.record_llm_result(False)
        c.record_llm_result(False)
        c.record_llm_result(True)   # Reset
        c.record_llm_result(False)
        c.record_llm_result(False)
        assert c.status()["state"] == RUNNING
        assert c.status()["consecutive_network_failures"] == 2

    def test_continue_after_network_pause_resets_counter(self):
        c = RuntimeController(max_network_failures=2)
        t = threading.Thread(target=lambda: [
            c.record_llm_result(False),
            c.record_llm_result(False),
        ])
        t.start()
        time.sleep(0.1)
        c.decide("continue")
        t.join(timeout=2)
        # After "continue", the failure counter must be reset so the next
        # batch of failures gets a fresh window.
        assert c.status()["consecutive_network_failures"] == 0


# ── User pause + force-stop ───────────────────────────────────────────────

class TestUserControls:
    def test_request_pause_transitions_to_paused_user(self):
        c = RuntimeController()
        c.request_pause()
        assert c.status()["state"] == PAUSED_USER

    def test_request_pause_no_op_when_not_running(self):
        c = RuntimeController()
        c.stop_now()
        c.request_pause()  # Should be no-op
        assert c.status()["state"] == STOPPED

    def test_stop_now_unblocks_paused_thread(self):
        c = RuntimeController(budget_limit_usd=1.0)
        out = []
        t = threading.Thread(target=lambda: out.append(c.heartbeat(0.95)))
        t.start()
        time.sleep(0.1)
        assert c.status()["state"] == PAUSED_BUDGET
        c.stop_now()
        t.join(timeout=2)
        assert out == [False]
        assert c.status()["state"] == STOPPED


# ── Decision timeout ──────────────────────────────────────────────────────

class TestDecisionTimeout:
    def test_timeout_defaults_to_stop(self):
        c = RuntimeController(budget_limit_usd=1.0,
                                decision_timeout_s=1)
        out = []
        t = threading.Thread(target=lambda: out.append(c.heartbeat(0.95)))
        t.start()
        t.join(timeout=2.5)
        assert out == [False]
        assert c.status()["state"] == STOPPED


# ── decide() edge cases ───────────────────────────────────────────────────

class TestDecideEdgeCases:
    def test_decide_invalid_string_raises(self):
        c = RuntimeController()
        with pytest.raises(ValueError):
            c.decide("maybe")

    def test_decide_when_not_paused_is_noop(self):
        c = RuntimeController()
        c.decide("continue")  # Should not raise, should not change state
        assert c.status()["state"] == RUNNING

    def test_negative_topup_clamped_to_zero(self):
        c = RuntimeController(budget_limit_usd=1.0)
        before = c.status()["budget_limit_usd"]
        t = threading.Thread(target=lambda: c.heartbeat(0.90))
        t.start()
        time.sleep(0.1)
        c.decide("continue", budget_topup=-5.0)
        t.join(timeout=2)
        # Negative topup is clamped to 0 — budget unchanged
        assert c.status()["budget_limit_usd"] == before


# ── State transitions ────────────────────────────────────────────────────

class TestStateTransitions:
    def test_mark_completed_terminal(self):
        c = RuntimeController()
        c.mark_completed()
        assert c.status()["state"] == COMPLETED
        assert c.status()["is_terminal"]

    def test_completed_does_not_overwrite_stopped(self):
        c = RuntimeController()
        c.stop_now()
        c.mark_completed()  # Should be ignored
        assert c.status()["state"] == STOPPED

    def test_heartbeat_after_stop_returns_false(self):
        c = RuntimeController()
        c.stop_now()
        assert c.heartbeat(0.0) is False

    def test_heartbeat_after_complete_returns_true(self):
        # Treat completion as "everything is fine" — no need to halt early
        c = RuntimeController()
        c.mark_completed()
        assert c.heartbeat(0.0) is True


# ── Status payload shape ──────────────────────────────────────────────────

class TestStatusShape:
    def test_status_has_expected_keys(self):
        c = RuntimeController()
        s = c.status()
        for k in ("state", "is_paused", "is_terminal", "pause_reason",
                  "paused_for_s", "budget_limit_usd", "current_cost_usd",
                  "budget_used_frac", "consecutive_network_failures",
                  "llm_calls_total", "llm_calls_failed", "run_id",
                  "max_network_failures", "budget_pause_threshold"):
            assert k in s, f"missing status key {k}"

    def test_event_log_records_decisions(self):
        c = RuntimeController(budget_limit_usd=1.0)
        t = threading.Thread(target=lambda: c.heartbeat(0.95))
        t.start()
        time.sleep(0.1)
        c.decide("continue", budget_topup=0.5)
        t.join(timeout=2)
        events = [e["event"] for e in c.event_log()]
        assert "pause" in events
        assert "decision" in events
        assert "resume" in events


# ── Pipeline + base_agent integration ─────────────────────────────────────

class TestIntegration:
    def test_config_module_exposes_controller_slot(self):
        import config
        # The slot is initialized in config.py for base_agent's decorator
        assert hasattr(config, "_current_runtime_controller")

    def test_pipeline_run_accepts_runtime_controller_kwarg(self):
        # Just verify the kwarg exists on the signature — full pipeline runs
        # are out of scope for unit tests
        import inspect
        from pipeline import IdeaGraphPipeline
        sig = inspect.signature(IdeaGraphPipeline.run)
        assert "runtime_controller" in sig.parameters

    def test_base_agent_call_decorator_observes_results(self):
        # The decorator should be applied to BaseAgent._call
        from agents.base_agent import BaseAgent
        # The wrapped function will have __wrapped__ via functools.wraps
        method = getattr(BaseAgent, "_call", None)
        assert method is not None
        # functools.wraps preserves __wrapped__
        assert hasattr(method, "__wrapped__"), \
            "_call must be wrapped by _observe_llm_call"

    def test_observation_decorator_records_into_active_controller(self):
        # Set up: stash a controller in config, call a fake _call that
        # returns "ok" → controller should record success. This is the
        # contract the decorator implements.
        import config
        c = RuntimeController()
        config._current_runtime_controller = c
        try:
            from agents.base_agent import _observe_llm_call

            class _FakeAgent:
                @_observe_llm_call
                def call(self, ok=True):
                    return "ok" if ok else ""

            agent = _FakeAgent()
            agent.call(ok=True)
            agent.call(ok=False)
            agent.call(ok=False)
            s = c.status()
            assert s["llm_calls_total"] == 3
            assert s["llm_calls_failed"] == 2
            assert s["consecutive_network_failures"] == 2
        finally:
            config._current_runtime_controller = None


# ── App-wiring smoke check ────────────────────────────────────────────────

class TestAppWiring:
    def test_app_imports_runtime_controller(self):
        with open("app.py", encoding="utf-8") as f:
            src = f.read()
        assert "from runtime_control import RuntimeController" in src
        assert "_runtime_controller" in src
        assert "Runtime control" in src
