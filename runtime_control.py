"""
runtime_control.py — interactive pause/resume for long-running pipelines.

A `RuntimeController` lives for the duration of one pipeline run. The
pipeline thread calls `heartbeat()` between iterations and
`record_llm_result()` after each LLM call. When budget approaches its
limit or the network appears to be down, the controller transitions
into a paused state and blocks the pipeline thread on a decision event.
The Streamlit UI polls `status()`, surfaces a banner with Continue /
Stop buttons (and an optional budget top-up), then calls `decide()` to
unblock the thread.

Two pause triggers:

  - **Budget pause** — when `current_cost_usd / budget_limit_usd` crosses
    `budget_pause_threshold` (default 0.85), the controller pauses *once*
    so the user can either approve a top-up or stop early.
  - **Network pause** — when consecutive LLM-call failures cross
    `max_network_failures` (default 4), the controller pauses so the
    user can retry once the network recovers, or stop and salvage the
    partial archive.

Plus a `request_pause()` for the user-facing pause button (cooperative).

All state mutations are guarded by `_lock`. The decision event has a
default 5-minute timeout: if the user closes the tab and never decides,
the pipeline gracefully stops with what it has.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Public state constants — keep these as strings (UI surfaces them directly)
RUNNING = "running"
PAUSED_BUDGET = "paused_budget"
PAUSED_NETWORK = "paused_network"
PAUSED_USER = "paused_user"
STOPPED = "stopped"
COMPLETED = "completed"

PAUSE_STATES = (PAUSED_BUDGET, PAUSED_NETWORK, PAUSED_USER)
TERMINAL_STATES = (STOPPED, COMPLETED)


@dataclass
class _LogEntry:
    ts: float
    event: str
    detail: str = ""


class RuntimeController:
    """Controls whether a pipeline keeps running, pauses, or stops.

    All public methods are thread-safe.

    Pipeline-side calls:
        heartbeat(current_cost_usd)  → bool (True = continue)
        record_llm_result(success)   → bool (True = continue)
        mark_completed()
        is_running()                  → bool

    UI-side calls:
        decide(decision, budget_topup=0)
        request_pause()
        stop_now()
        status()                      → Dict[str, Any]
    """

    def __init__(
        self,
        budget_limit_usd: float = 2.0,
        budget_pause_threshold: float = 0.85,
        max_network_failures: int = 4,
        decision_timeout_s: int = 300,
        run_id: str = "",
    ) -> None:
        if budget_limit_usd <= 0:
            raise ValueError("budget_limit_usd must be > 0")
        if not (0.0 < budget_pause_threshold < 1.0):
            raise ValueError("budget_pause_threshold must be in (0, 1)")
        if max_network_failures <= 0:
            raise ValueError("max_network_failures must be > 0")

        self._lock = threading.Lock()
        self._decision_event = threading.Event()
        self._decision_event.set()  # not currently waiting

        self._state: str = RUNNING
        self._run_id: str = run_id
        self._budget_limit_usd: float = float(budget_limit_usd)
        self._budget_pause_threshold: float = float(budget_pause_threshold)
        self._max_network_failures: int = int(max_network_failures)
        self._decision_timeout_s: int = int(decision_timeout_s)

        self._current_cost_usd: float = 0.0
        self._consecutive_network_failures: int = 0
        self._llm_calls_total: int = 0
        self._llm_calls_failed: int = 0

        # Set when a budget pause has fired so we don't pause again at every
        # subsequent heartbeat. When the user tops up, we recompute the
        # threshold off the new limit and rearm.
        self._budget_pause_fired: bool = False

        self._pending_decision: Optional[str] = None  # "continue" | "stop"
        self._pending_topup: float = 0.0
        self._pause_reason: str = ""
        self._paused_at: Optional[float] = None
        self._log: List[_LogEntry] = []

    # ── Internal helpers ────────────────────────────────────────────────

    def _record(self, event: str, detail: str = "") -> None:
        # Caller must hold _lock. Bounded log so it doesn't grow unbounded.
        self._log.append(_LogEntry(ts=time.time(), event=event, detail=detail))
        if len(self._log) > 200:
            self._log = self._log[-150:]

    def _enter_pause(self, state: str, reason: str) -> None:
        # Caller must hold _lock.
        self._state = state
        self._pause_reason = reason
        self._paused_at = time.time()
        self._pending_decision = None
        self._record("pause", reason)
        self._decision_event.clear()

    def _wait_for_decision(self) -> str:
        """Block until the UI calls decide(...) or the timeout fires.

        Returns 'continue' or 'stop' (timeout maps to 'stop' for safety).
        Released without holding the lock.
        """
        ok = self._decision_event.wait(timeout=self._decision_timeout_s)
        with self._lock:
            if not ok and self._pending_decision is None:
                # Timeout: nobody decided. Default to safe stop.
                self._pending_decision = "stop"
                self._record("decision_timeout",
                              f"after {self._decision_timeout_s}s")
            decision = self._pending_decision or "stop"
            topup = self._pending_topup
            if decision == "continue":
                # Apply top-up + clear pause state
                if topup > 0:
                    self._budget_limit_usd += float(topup)
                    self._record("budget_topup", f"+${topup:.2f}")
                # If we paused on network, reset the counter so retries get a
                # fresh window.
                self._consecutive_network_failures = 0
                self._budget_pause_fired = (
                    self._current_cost_usd / self._budget_limit_usd
                    >= self._budget_pause_threshold
                )
                self._state = RUNNING
                self._paused_at = None
                self._pause_reason = ""
                self._record("resume", f"after {self._pending_decision}")
            else:
                self._state = STOPPED
                self._record("stopped_by_user", "")
            # Reset the pending fields for any subsequent pause
            self._pending_decision = None
            self._pending_topup = 0.0
        return decision

    # ── Pipeline-side API ───────────────────────────────────────────────

    def heartbeat(self, current_cost_usd: float = 0.0) -> bool:
        """Pipeline calls this between iterations. Returns True if the
        pipeline should keep running, False if the user decided to stop.
        """
        with self._lock:
            self._current_cost_usd = float(current_cost_usd)
            # Don't fire if we're already paused or stopped — let the existing
            # state path resolve first.
            if self._state in TERMINAL_STATES:
                return self._state == COMPLETED  # treat completed as continue
            if self._state in PAUSE_STATES:
                # Already waiting on a decision — do not re-enter pause
                pass
            elif (
                not self._budget_pause_fired
                and self._budget_limit_usd > 0
                and self._current_cost_usd / self._budget_limit_usd
                >= self._budget_pause_threshold
            ):
                self._budget_pause_fired = True
                self._enter_pause(
                    PAUSED_BUDGET,
                    f"${self._current_cost_usd:.3f} of "
                    f"${self._budget_limit_usd:.2f} "
                    f"({self._current_cost_usd / self._budget_limit_usd:.0%}) "
                    "consumed",
                )

        if self._state in PAUSE_STATES:
            return self._wait_for_decision() == "continue"
        return self._state != STOPPED

    def record_llm_result(self, success: bool) -> bool:
        """Pipeline calls this after every LLM call. Returns True if the
        pipeline should keep going, False if it should stop.
        """
        should_pause = False
        with self._lock:
            self._llm_calls_total += 1
            if self._state in TERMINAL_STATES or self._state in PAUSE_STATES:
                # Don't trigger nested pauses
                pass
            elif success:
                self._consecutive_network_failures = 0
            else:
                self._llm_calls_failed += 1
                self._consecutive_network_failures += 1
                if (self._consecutive_network_failures
                        >= self._max_network_failures):
                    should_pause = True

            if should_pause:
                self._enter_pause(
                    PAUSED_NETWORK,
                    f"{self._consecutive_network_failures} consecutive "
                    "LLM-call failures — network may be down",
                )

        if should_pause:
            return self._wait_for_decision() == "continue"
        return self._state != STOPPED

    def mark_completed(self) -> None:
        """Pipeline calls this when its main loop exits cleanly."""
        with self._lock:
            if self._state not in TERMINAL_STATES:
                self._state = COMPLETED
                self._record("completed", "")

    def is_running(self) -> bool:
        with self._lock:
            return self._state == RUNNING

    # ── UI-side API ─────────────────────────────────────────────────────

    def decide(self, decision: str, budget_topup: float = 0.0) -> None:
        """UI calls this to unblock a paused pipeline.

        decision: 'continue' or 'stop'.
        budget_topup: dollars to add to the budget limit (only if continue).
        """
        if decision not in ("continue", "stop"):
            raise ValueError("decision must be 'continue' or 'stop'")
        topup = max(0.0, float(budget_topup or 0.0))
        with self._lock:
            if self._state not in PAUSE_STATES:
                # No-op if we're not actually paused
                self._record("decide_noop",
                              f"state={self._state}, decision={decision}")
                return
            self._pending_decision = decision
            self._pending_topup = topup
            self._record("decision", f"{decision} topup=${topup:.2f}")
        self._decision_event.set()

    def request_pause(self) -> None:
        """User clicked the manual pause button. Only fires if running."""
        with self._lock:
            if self._state == RUNNING:
                self._enter_pause(PAUSED_USER, "manual pause requested")

    def stop_now(self) -> None:
        """Force-stop. Works whether running or paused."""
        with self._lock:
            if self._state in TERMINAL_STATES:
                return
            was_paused = self._state in PAUSE_STATES
            self._state = STOPPED
            self._record("stop_now", "user stop")
            if was_paused:
                self._pending_decision = "stop"
                self._pending_topup = 0.0
        if self._decision_event:
            self._decision_event.set()

    def status(self) -> Dict[str, Any]:
        """Snapshot of current state for the UI banner."""
        with self._lock:
            paused_for_s = (time.time() - self._paused_at
                             if self._paused_at else 0.0)
            return {
                "state": self._state,
                "is_paused": self._state in PAUSE_STATES,
                "is_terminal": self._state in TERMINAL_STATES,
                "pause_reason": self._pause_reason,
                "paused_for_s": paused_for_s,
                "budget_limit_usd": self._budget_limit_usd,
                "current_cost_usd": self._current_cost_usd,
                "budget_used_frac": (
                    self._current_cost_usd / self._budget_limit_usd
                    if self._budget_limit_usd > 0 else 0.0
                ),
                "consecutive_network_failures":
                    self._consecutive_network_failures,
                "llm_calls_total": self._llm_calls_total,
                "llm_calls_failed": self._llm_calls_failed,
                "run_id": self._run_id,
                "max_network_failures": self._max_network_failures,
                "budget_pause_threshold": self._budget_pause_threshold,
            }

    def event_log(self, max_entries: int = 30) -> List[Dict[str, Any]]:
        """Recent event log for the UI's debug expander."""
        with self._lock:
            entries = list(self._log[-max_entries:])
        return [
            {"ts": e.ts, "event": e.event, "detail": e.detail}
            for e in entries
        ]
