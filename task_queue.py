"""
task_queue.py - Pluggable task queue for pipeline runs.

Replaces the "spawn an unbounded daemon thread per request" pattern in
app.py + api.py with a bounded, inspectable, priority-aware task queue.
Backends:

  * ThreadQueueBackend (default) — bounded ThreadPoolExecutor. Zero external
    deps. Good up to ~50 concurrent runs on a single server.
  * CeleryBackend  — real distributed queue via Celery + Redis. Selected by
    setting TASK_QUEUE_BACKEND=celery and CELERY_BROKER_URL. Workers live in
    separate processes, so this scales horizontally (the only path to 1M users).

API (same across backends):

    tq = get_task_queue()
    task_id = tq.submit(
        fn,                      # callable to run
        args=(...), kwargs={...},
        priority=10,             # higher runs first (free=0, pro=10, team=20)
        user_id=42,              # for quota accounting
        timeout_s=900,           # kill runaway jobs
    )

    status = tq.status(task_id)        # queued | running | done | error
    result = tq.wait(task_id)          # blocks, raises on error
    tq.cancel(task_id)

Observability:
    Every submit/start/finish emits a structured log + updates metrics
    (queue_depth, task_duration_seconds, task_errors_total, ...).
"""

from __future__ import annotations

import heapq
import os
import queue
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

try:
    from observability import logger, metrics
except ImportError:
    # Fallback stub so task_queue can still be imported in isolation.
    class _Stub:
        def info(self, *a, **k): pass
        def warn(self, *a, **k): pass
        def error(self, *a, **k): pass
        def exception(self, *a, **k): pass
        def inc(self, *a, **k): pass
        def set(self, *a, **k): pass
        def observe(self, *a, **k): pass
    logger = _Stub()
    metrics = _Stub()


# ─────────────────────────────────────────────────────────────────────────────
# Task record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(order=True)
class _Task:
    # heapq is a min-heap → negate priority so larger = earlier.
    sort_key: tuple = field(init=False, repr=False)
    priority: int = 0
    submitted_at: float = 0.0
    task_id: str = ""
    fn: Optional[Callable] = field(default=None, compare=False)
    args: tuple = field(default=(), compare=False)
    kwargs: Dict[str, Any] = field(default_factory=dict, compare=False)
    user_id: Optional[int] = field(default=None, compare=False)
    timeout_s: Optional[float] = field(default=None, compare=False)

    def __post_init__(self) -> None:
        # Earlier priority (negative) + earlier submitted_at = runs first.
        self.sort_key = (-self.priority, self.submitted_at)


# ─────────────────────────────────────────────────────────────────────────────
# Thread-pool backend (default)
# ─────────────────────────────────────────────────────────────────────────────

class ThreadQueueBackend:
    """
    Bounded ThreadPoolExecutor + priority heap. Single-process; fine for small
    deployments and for local dev/test. Does NOT survive process restart — for
    that, switch to Celery.
    """

    def __init__(
        self, max_workers: int = 10, queue_capacity: int = 1000,
    ) -> None:
        self.max_workers = max_workers
        self.queue_capacity = queue_capacity
        self._pool = ThreadPoolExecutor(max_workers=max_workers,
                                        thread_name_prefix="tq-worker")
        self._heap: List[_Task] = []
        self._statuses: Dict[str, Dict[str, Any]] = {}
        self._futures: Dict[str, Future] = {}
        # Incremental status counters: avoids the O(N) scan of
        # self._statuses.values() in stats() — _statuses grows monotonically
        # for the life of the process (one entry per submitted task ever),
        # so the old `sum(... for s in ...)` was O(total tasks ever) per
        # scrape, and stats() may be called every few seconds by dashboards.
        self._status_counts: Dict[str, int] = {
            "queued": 0, "running": 0, "done": 0, "error": 0, "cancelled": 0,
        }
        self._lock = threading.RLock()
        self._not_empty = threading.Condition(self._lock)
        # Semaphore gates the dispatcher on actual pool capacity so tasks
        # buffer in our heap (where they can be cancelled / rejected by
        # capacity check) rather than in the pool's internal FIFO queue.
        self._slots = threading.Semaphore(max_workers)
        self._dispatcher = threading.Thread(
            target=self._dispatch_loop, name="tq-dispatcher", daemon=True,
        )
        self._shutdown = False
        self._dispatcher.start()

    # ── Public API ────────────────────────────────────────────────────────

    def submit(
        self,
        fn: Callable,
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        priority: int = 0,
        user_id: Optional[int] = None,
        timeout_s: Optional[float] = None,
    ) -> str:
        with self._lock:
            if len(self._heap) >= self.queue_capacity:
                raise RuntimeError(
                    f"Task queue at capacity ({self.queue_capacity} queued). "
                    f"Backpressure — retry in a moment."
                )
            task_id = uuid.uuid4().hex[:12]
            task = _Task(
                priority=priority, submitted_at=time.time(),
                task_id=task_id, fn=fn, args=args,
                kwargs=kwargs or {}, user_id=user_id, timeout_s=timeout_s,
            )
            heapq.heappush(self._heap, task)
            self._statuses[task_id] = {
                "status": "queued", "submitted_at": task.submitted_at,
                "priority": priority, "user_id": user_id,
            }
            self._status_counts["queued"] += 1
            self._not_empty.notify()
            metrics.set("task_queue_depth", float(len(self._heap)))
            metrics.inc("tasks_submitted_total",
                        tags={"priority": str(priority)})
            logger.info("task_submitted", task_id=task_id,
                        priority=priority, user_id=user_id,
                        queue_depth=len(self._heap))
        return task_id

    def status(self, task_id: str) -> Dict[str, Any]:
        with self._lock:
            return dict(self._statuses.get(task_id, {"status": "unknown"}))

    def wait(self, task_id: str, timeout: Optional[float] = None) -> Any:
        fut = self._get_future(task_id, timeout=timeout)
        return fut.result(timeout=timeout)

    def cancel(self, task_id: str) -> bool:
        with self._lock:
            st = self._statuses.get(task_id)
            if not st:
                return False
            if st["status"] == "queued":
                # Mark as cancelled — dispatcher will skip it.
                st["status"] = "cancelled"
                self._status_counts["queued"] -= 1
                self._status_counts["cancelled"] += 1
                logger.info("task_cancelled", task_id=task_id)
                return True
            fut = self._futures.get(task_id)
            if fut and not fut.done():
                return fut.cancel()
            return False

    def stats(self) -> Dict[str, int]:
        # O(1) read of incremental counters; previous version was O(N) per
        # status (4× scans of all tasks ever submitted).
        with self._lock:
            counts = self._status_counts
            return {
                "queue_depth": len(self._heap),
                "running": counts["running"],
                "completed": counts["done"],
                "errored": counts["error"],
            }

    def shutdown(self, wait: bool = True) -> None:
        with self._lock:
            self._shutdown = True
            self._not_empty.notify_all()
        self._pool.shutdown(wait=wait)

    # ── Internal ──────────────────────────────────────────────────────────

    def _get_future(self, task_id: str, timeout: Optional[float]) -> Future:
        deadline = time.time() + (timeout or float("inf"))
        while True:
            with self._lock:
                fut = self._futures.get(task_id)
                if fut is not None:
                    return fut
            if time.time() > deadline:
                raise TimeoutError(f"Timed out waiting for task {task_id} to start")
            time.sleep(0.05)

    def _dispatch_loop(self) -> None:
        """Pop from heap, submit to pool, but only when pool has free slots."""
        while True:
            # Wait for an available pool slot before peeking at the heap,
            # so tasks stay in our heap (cancellable, capacity-counted).
            acquired = self._slots.acquire(timeout=1.0)
            if not acquired:
                if self._shutdown:
                    return
                continue

            with self._not_empty:
                while not self._heap and not self._shutdown:
                    self._not_empty.wait(timeout=1.0)
                if self._shutdown and not self._heap:
                    self._slots.release()
                    return
                if not self._heap:
                    self._slots.release()
                    continue
                task = heapq.heappop(self._heap)
                st = self._statuses.get(task.task_id)
                if not st or st["status"] == "cancelled":
                    self._slots.release()
                    continue

                def _run_then_release(t=task):
                    try:
                        return self._run(t)
                    finally:
                        self._slots.release()

                fut = self._pool.submit(_run_then_release)
                self._futures[task.task_id] = fut
                metrics.set("task_queue_depth", float(len(self._heap)))

    def _run(self, task: _Task) -> Any:
        start = time.time()
        with self._lock:
            st = self._statuses.get(task.task_id, {})
            st["status"] = "running"
            st["started_at"] = start
            self._status_counts["queued"] -= 1
            self._status_counts["running"] += 1
        logger.info("task_started", task_id=task.task_id,
                    user_id=task.user_id, priority=task.priority)
        try:
            # Timeout enforcement: run in sub-thread, join with timeout.
            if task.timeout_s and task.timeout_s > 0:
                result_holder: Dict[str, Any] = {}
                error_holder: Dict[str, BaseException] = {}

                def _target():
                    try:
                        result_holder["v"] = task.fn(*task.args, **task.kwargs)
                    except BaseException as exc:
                        error_holder["e"] = exc

                t = threading.Thread(target=_target, daemon=True)
                t.start()
                t.join(timeout=task.timeout_s)
                if t.is_alive():
                    raise TimeoutError(
                        f"Task {task.task_id} exceeded timeout "
                        f"{task.timeout_s}s (thread still running; will leak until function returns)"
                    )
                if "e" in error_holder:
                    raise error_holder["e"]
                result = result_holder.get("v")
            else:
                result = task.fn(*task.args, **task.kwargs)

            duration = time.time() - start
            with self._lock:
                st["status"] = "done"
                st["finished_at"] = time.time()
                st["duration_s"] = duration
                self._status_counts["running"] -= 1
                self._status_counts["done"] += 1
            metrics.inc("tasks_completed_total")
            metrics.observe("task_duration_seconds", duration,
                            tags={"status": "ok"})
            logger.info("task_done", task_id=task.task_id,
                        duration_s=round(duration, 3))
            return result
        except BaseException as exc:
            duration = time.time() - start
            with self._lock:
                st["status"] = "error"
                st["finished_at"] = time.time()
                st["duration_s"] = duration
                st["error"] = str(exc)
                self._status_counts["running"] -= 1
                self._status_counts["error"] += 1
            metrics.inc("task_errors_total",
                        tags={"error": type(exc).__name__})
            metrics.observe("task_duration_seconds", duration,
                            tags={"status": "error"})
            logger.error("task_error", task_id=task.task_id,
                         error=str(exc), error_type=type(exc).__name__)
            raise


# ─────────────────────────────────────────────────────────────────────────────
# Celery backend (optional)
# ─────────────────────────────────────────────────────────────────────────────

class CeleryBackend:
    """
    Thin wrapper around Celery. Activated when TASK_QUEUE_BACKEND=celery.

    Requires:
      pip install celery[redis]
      redis-server (broker + backend)
      Set CELERY_BROKER_URL=redis://host:6379/0

    Workers are started separately:
      celery -A task_queue.celery_app worker --loglevel=info --concurrency=10
    """

    def __init__(self, broker_url: str, result_backend: Optional[str] = None) -> None:
        try:
            from celery import Celery
        except ImportError as exc:
            raise RuntimeError(
                "CeleryBackend requires 'celery' package. "
                "pip install 'celery[redis]'."
            ) from exc
        self.app = Celery("ideagraph", broker=broker_url,
                          backend=result_backend or broker_url)
        self.app.conf.update(
            task_serializer="json",
            result_serializer="json",
            accept_content=["json"],
            task_acks_late=True,
            worker_prefetch_multiplier=1,
            task_time_limit=1800,
            task_soft_time_limit=1500,
            result_expires=86400,
        )
        self._tasks: Dict[str, Any] = {}

    def register(self, name: str, fn: Callable) -> Callable:
        """Register a function as a Celery task. Call once at module load."""
        task = self.app.task(name=name, bind=False)(fn)
        self._tasks[name] = task
        return task

    def submit(
        self, fn: Callable, args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        priority: int = 0, user_id: Optional[int] = None,
        timeout_s: Optional[float] = None,
        task_name: Optional[str] = None,
    ) -> str:
        name = task_name or getattr(fn, "name", fn.__name__)
        if name not in self._tasks:
            self.register(name, fn)
        task = self._tasks[name]
        async_result = task.apply_async(
            args=args, kwargs=kwargs or {},
            priority=max(0, min(9, 9 - priority // 10)),  # Celery 0-9, higher = faster
            soft_time_limit=timeout_s,
        )
        logger.info("celery_submit", task_id=async_result.id,
                    user_id=user_id, priority=priority)
        return async_result.id

    def status(self, task_id: str) -> Dict[str, Any]:
        res = self.app.AsyncResult(task_id)
        state_map = {"PENDING": "queued", "STARTED": "running",
                     "SUCCESS": "done", "FAILURE": "error",
                     "REVOKED": "cancelled"}
        return {"status": state_map.get(res.state, res.state.lower())}

    def wait(self, task_id: str, timeout: Optional[float] = None) -> Any:
        return self.app.AsyncResult(task_id).get(timeout=timeout)

    def cancel(self, task_id: str) -> bool:
        self.app.control.revoke(task_id, terminate=True)
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Tier → priority helper
# ─────────────────────────────────────────────────────────────────────────────

TIER_PRIORITY = {
    "free": 0, "pro": 10, "team": 20, "enterprise": 30,
}


def priority_for_tier(tier: str) -> int:
    return TIER_PRIORITY.get(tier, 0)


def timeout_for_tier(tier: str) -> float:
    """Enforce max runtime by tier. Prevents a single whale from hogging workers."""
    return {"free": 300.0, "pro": 900.0, "team": 1800.0,
            "enterprise": 3600.0}.get(tier, 300.0)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

_TASK_QUEUE: Optional[Any] = None
_TQ_LOCK = threading.Lock()


def get_task_queue():
    """Lazy-init the configured backend."""
    global _TASK_QUEUE
    with _TQ_LOCK:
        if _TASK_QUEUE is not None:
            return _TASK_QUEUE
        backend = os.getenv("TASK_QUEUE_BACKEND", "thread").lower()
        if backend == "celery":
            broker = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
            backend_url = os.getenv("CELERY_RESULT_BACKEND") or broker
            _TASK_QUEUE = CeleryBackend(broker, backend_url)
        else:
            _TASK_QUEUE = ThreadQueueBackend(
                max_workers=int(os.getenv("TQ_WORKERS", "10")),
                queue_capacity=int(os.getenv("TQ_CAPACITY", "1000")),
            )
        return _TASK_QUEUE


def reset_task_queue() -> None:
    """For tests: tear down and re-create the backend."""
    global _TASK_QUEUE
    with _TQ_LOCK:
        if _TASK_QUEUE is not None:
            try:
                _TASK_QUEUE.shutdown(wait=False)
            except Exception:
                pass
        _TASK_QUEUE = None


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    tq = get_task_queue()

    def work(n: int, label: str = "x") -> str:
        time.sleep(0.05)
        return f"{label}:{n*n}"

    # Submit 5 tasks with mixed priorities.
    ids = []
    for i in range(5):
        tid = tq.submit(work, args=(i,), kwargs={"label": f"t{i}"},
                        priority=i, user_id=i)
        ids.append(tid)

    # Wait for all.
    for tid in ids:
        print(f"  {tid}: {tq.wait(tid, timeout=5)}")

    print("\nstats:", json.dumps(tq.stats(), indent=2))
