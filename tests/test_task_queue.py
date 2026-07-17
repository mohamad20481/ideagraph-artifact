"""
Tests for task_queue.py — ThreadQueueBackend semantics.

(Skips CeleryBackend tests unless celery is actually installed + a local
redis is reachable.)
"""
import time

import pytest

from task_queue import (
    TIER_PRIORITY,
    ThreadQueueBackend,
    priority_for_tier,
    reset_task_queue,
    timeout_for_tier,
)


@pytest.fixture
def tq():
    """Fresh ThreadQueueBackend per test so state doesn't leak."""
    q = ThreadQueueBackend(max_workers=4, queue_capacity=100)
    yield q
    q.shutdown(wait=False)


class TestBasicSubmission:
    def test_submit_and_wait_returns_result(self, tq):
        tid = tq.submit(lambda x: x * 2, args=(21,))
        assert tq.wait(tid, timeout=5) == 42

    def test_status_transitions_to_done(self, tq):
        tid = tq.submit(lambda: 1 + 1)
        tq.wait(tid, timeout=5)
        assert tq.status(tid)["status"] == "done"

    def test_exceptions_propagate(self, tq):
        def boom():
            raise ValueError("boom")
        tid = tq.submit(boom)
        with pytest.raises(ValueError, match="boom"):
            tq.wait(tid, timeout=5)
        assert tq.status(tid)["status"] == "error"
        assert "boom" in tq.status(tid)["error"]

    def test_unknown_task_status(self, tq):
        assert tq.status("nonexistent")["status"] == "unknown"


class TestPriority:
    def test_higher_priority_runs_first(self, tq):
        # Pool size=1 so only one task runs at a time.
        small = ThreadQueueBackend(max_workers=1, queue_capacity=100)
        try:
            order = []
            lock = __import__("threading").Lock()

            def work(label):
                with lock:
                    order.append(label)
                time.sleep(0.02)

            # Block the worker by submitting a slow first task.
            blocker = small.submit(work, args=("blocker",), priority=100)
            # Now enqueue the others while the worker is busy.
            time.sleep(0.01)
            low = small.submit(work, args=("low",), priority=0)
            high = small.submit(work, args=("high",), priority=50)

            small.wait(blocker, timeout=5)
            small.wait(high, timeout=5)
            small.wait(low, timeout=5)

            # blocker ran first (was submitted first), then high, then low.
            assert order == ["blocker", "high", "low"]
        finally:
            small.shutdown(wait=False)


class TestCapacity:
    def test_queue_capacity_enforced(self):
        tq = ThreadQueueBackend(max_workers=1, queue_capacity=3)
        try:
            # Tie up the worker. Give the dispatcher a moment to pop it from
            # the heap into the pool so our capacity measurements are clean.
            tq.submit(lambda: time.sleep(2))
            time.sleep(0.05)

            # Now fill the queue to capacity.
            for _ in range(3):
                tq.submit(lambda: None)

            # Next submit must fail with a "capacity" RuntimeError.
            with pytest.raises(RuntimeError, match="capacity"):
                tq.submit(lambda: None)
        finally:
            tq.shutdown(wait=False)


class TestTimeout:
    def test_timeout_kills_slow_task(self, tq):
        def slow():
            time.sleep(2)
            return "done"
        tid = tq.submit(slow, timeout_s=0.2)
        with pytest.raises(TimeoutError):
            tq.wait(tid, timeout=3)


class TestCancellation:
    def test_cancel_queued_task(self):
        tq = ThreadQueueBackend(max_workers=1, queue_capacity=100)
        try:
            # Tie up the worker and wait until it's actually running so the
            # dispatcher can't pop the target before we cancel it.
            blocker = tq.submit(lambda: time.sleep(0.5))
            for _ in range(50):
                if tq.status(blocker)["status"] == "running":
                    break
                time.sleep(0.01)
            target = tq.submit(lambda: 42)
            ok = tq.cancel(target)
            assert ok, "cancel should succeed while task is still queued"
            assert tq.status(target)["status"] == "cancelled"
            tq.wait(blocker, timeout=5)
        finally:
            tq.shutdown(wait=False)


class TestStats:
    def test_stats_track_completed(self, tq):
        for _ in range(3):
            tid = tq.submit(lambda: 1)
            tq.wait(tid, timeout=5)
        s = tq.stats()
        assert s["completed"] == 3
        assert s["errored"] == 0
        assert s["queue_depth"] == 0


class TestTierHelpers:
    def test_priority_for_tier(self):
        assert priority_for_tier("free") == 0
        assert priority_for_tier("pro") == 10
        assert priority_for_tier("team") == 20
        assert priority_for_tier("enterprise") == 30
        assert priority_for_tier("unknown") == 0  # defaults to free

    def test_timeout_for_tier(self):
        assert timeout_for_tier("free") == 300.0
        assert timeout_for_tier("enterprise") == 3600.0
        assert timeout_for_tier("free") < timeout_for_tier("enterprise")


class TestSingleton:
    def test_reset_task_queue(self):
        from task_queue import get_task_queue
        q1 = get_task_queue()
        reset_task_queue()
        q2 = get_task_queue()
        assert q1 is not q2
        reset_task_queue()  # cleanup
