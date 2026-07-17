"""Tests for worker_db.py — the SQLite job store for the worker.

Critical contract: `claim_next_pending_run` is **atomic** — two workers
calling it simultaneously must result in EXACTLY ONE claim.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import List

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import worker_db
from db import _get_conn, _lock


@pytest.fixture(autouse=True)
def _ensure_schema():
    worker_db.init_worker_db()
    yield
    # Clean up worker tables between tests so they don't leak state.
    with _lock:
        conn = _get_conn()
        try:
            conn.execute("DELETE FROM worker_run_ideas")
            conn.execute("DELETE FROM worker_runs")
            conn.commit()
        finally:
            conn.close()


# ── create + read + list ────────────────────────────────────────────────────

def test_create_run_returns_uuid_hex():
    rid = worker_db.create_run({"topic": "x"}, ideas_total=5)
    assert isinstance(rid, str)
    assert len(rid) == 32  # uuid4 hex


def test_create_run_rejects_non_dict_params():
    with pytest.raises(TypeError):
        worker_db.create_run("not a dict")  # type: ignore[arg-type]


def test_get_run_returns_inflated_params():
    rid = worker_db.create_run({"topic": "ml", "budget_usd": 2.5})
    row = worker_db.get_run(rid)
    assert row is not None
    assert row["status"] == "pending"
    assert row["params"]["topic"] == "ml"
    assert row["params"]["budget_usd"] == 2.5
    assert row["ideas_done"] == 0
    assert row["error"] is None
    assert row["created_at"]


def test_get_run_unknown_returns_none():
    assert worker_db.get_run("no-such-id") is None
    assert worker_db.get_run("") is None


def test_list_runs_newest_first():
    rid1 = worker_db.create_run({"topic": "a"})
    time.sleep(0.01)
    rid2 = worker_db.create_run({"topic": "b"})
    out = worker_db.list_runs()
    titles = [r["params"]["topic"] for r in out]
    # Newest first.
    assert titles[0] == "b"
    assert titles[1] == "a"


def test_list_runs_filters_by_user():
    """user_id is a real FK to users(id) — pre-create users so the
    constraint passes, then verify the filter works."""
    with _lock:
        c = _get_conn()
        try:
            c.execute(
                "INSERT INTO users (id, username, password_hash, salt) "
                "VALUES (9001, ?, 'x', 'y'), (9002, ?, 'x', 'y')",
                (f"_wd_test_user_{time.time_ns()}_a",
                  f"_wd_test_user_{time.time_ns()}_b"),
            )
            c.commit()
        finally:
            c.close()
    try:
        rid1 = worker_db.create_run({"topic": "u1"}, user_id=9001)
        rid2 = worker_db.create_run({"topic": "u2"}, user_id=9002)
        out = worker_db.list_runs(user_id=9001)
        assert len(out) == 1
        assert out[0]["params"]["topic"] == "u1"
    finally:
        # Cascade will clean the runs too (ON DELETE SET NULL keeps them
        # but the autouse fixture below deletes all rows from
        # worker_runs anyway).
        with _lock:
            c = _get_conn()
            try:
                c.execute("DELETE FROM users WHERE id IN (9001, 9002)")
                c.commit()
            finally:
                c.close()


def test_list_runs_respects_limit():
    for i in range(5):
        worker_db.create_run({"topic": f"t{i}"})
    assert len(worker_db.list_runs(limit=3)) == 3


# ── Atomic claim ────────────────────────────────────────────────────────────

def test_claim_returns_none_when_no_pending():
    assert worker_db.claim_next_pending_run("worker-1") is None


def test_claim_flips_status_to_running():
    rid = worker_db.create_run({"topic": "x"})
    claimed = worker_db.claim_next_pending_run("worker-A")
    assert claimed is not None
    assert claimed["id"] == rid
    assert claimed["status"] == "running"
    assert claimed["lease_token"] == "worker-A"
    assert claimed["attempts"] == 1


def test_claim_increments_attempts_on_repeat():
    """Reclaim after a finished previous attempt — `attempts` keeps
    counting up so the worker can detect resume."""
    rid = worker_db.create_run({"topic": "x"})
    worker_db.claim_next_pending_run("worker-A")
    # Simulate the worker crashing without marking done; set back to pending.
    worker_db.set_run_status(rid, "pending")
    claimed2 = worker_db.claim_next_pending_run("worker-A")
    assert claimed2 is not None
    assert claimed2["attempts"] == 2


def test_claim_picks_oldest_pending_first():
    """FIFO ordering — first-created run claimed first."""
    rid1 = worker_db.create_run({"topic": "oldest"})
    time.sleep(0.02)
    rid2 = worker_db.create_run({"topic": "newer"})
    claimed = worker_db.claim_next_pending_run("worker-1")
    assert claimed["id"] == rid1


def test_two_workers_one_pending_run_exactly_one_wins():
    """The critical concurrency contract: simultaneous claims must
    produce exactly ONE successful claim."""
    rid = worker_db.create_run({"topic": "race"})
    results: List = []
    barrier = threading.Barrier(2)

    def attempt(worker_id: str) -> None:
        barrier.wait()  # both threads enter claim() at the same moment
        results.append(worker_db.claim_next_pending_run(worker_id))

    t1 = threading.Thread(target=attempt, args=("worker-A",))
    t2 = threading.Thread(target=attempt, args=("worker-B",))
    t1.start(); t2.start()
    t1.join(); t2.join()

    wins = [r for r in results if r is not None]
    assert len(wins) == 1, (
        f"Exactly one worker should win, got {len(wins)}: {[r.get('lease_token') for r in wins]}"
    )
    # The losing thread got None.
    losses = [r for r in results if r is None]
    assert len(losses) == 1


def test_claim_reclaims_stale_running_run():
    """If a worker held a run but stopped heartbeating, another worker
    must reclaim it. We simulate by directly writing an old heartbeat."""
    rid = worker_db.create_run({"topic": "stale"})
    worker_db.claim_next_pending_run("worker-A")

    # Force a stale heartbeat by writing one in the past.
    from datetime import datetime, timezone, timedelta
    stale_iso = (
        datetime.now(timezone.utc) - timedelta(seconds=300)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    with _lock:
        c = _get_conn()
        try:
            c.execute(
                "UPDATE worker_runs SET heartbeat_at = ? WHERE id = ?",
                (stale_iso, rid),
            )
            c.commit()
        finally:
            c.close()

    # Worker-B can now reclaim it.
    claimed = worker_db.claim_next_pending_run("worker-B")
    assert claimed is not None
    assert claimed["id"] == rid
    assert claimed["lease_token"] == "worker-B"
    assert claimed["attempts"] == 2  # second claim


def test_claim_does_not_reclaim_fresh_running_run():
    """A run with a recent heartbeat should NOT be reclaimable."""
    rid = worker_db.create_run({"topic": "fresh"})
    worker_db.claim_next_pending_run("worker-A")
    worker_db.heartbeat_run(rid)  # makes it fresh
    # Worker-B should get None — there's no work and the running run
    # isn't stale.
    assert worker_db.claim_next_pending_run("worker-B") is None


# ── Idea append ────────────────────────────────────────────────────────────

def test_append_run_idea_increments_done():
    rid = worker_db.create_run({"topic": "x"})
    n1 = worker_db.append_run_idea(rid, {"title": "i1", "quality_score": 0.5})
    n2 = worker_db.append_run_idea(rid, {"title": "i2", "quality_score": 0.7})
    assert n1 == 1
    assert n2 == 2
    row = worker_db.get_run(rid)
    assert row["ideas_done"] == 2


def test_append_run_idea_persists_full_dict():
    rid = worker_db.create_run({"topic": "x"})
    worker_db.append_run_idea(rid, {
        "title": "my idea", "quality_score": 0.8,
        "methodology_type": "empirical_study",
        "execution_meta": {"some": "nested"},
    })
    ideas = worker_db.list_run_ideas(rid)
    assert len(ideas) == 1
    assert ideas[0]["title"] == "my idea"
    assert ideas[0]["execution_meta"]["some"] == "nested"
    assert ideas[0]["_archived_at"]  # archive timestamp attached


def test_append_run_idea_handles_non_serializable():
    """Anything that can't JSON-encode is stringified (default=str)
    so the worker callback never crashes."""
    class _Weird:
        def __repr__(self): return "<weird>"
    rid = worker_db.create_run({"topic": "x"})
    n = worker_db.append_run_idea(
        rid, {"title": "x", "obj": _Weird()},
    )
    assert n == 1
    ideas = worker_db.list_run_ideas(rid)
    assert ideas[0]["obj"] == "<weird>"


def test_append_run_idea_requires_run_id():
    with pytest.raises(ValueError):
        worker_db.append_run_idea("", {"title": "x"})


def test_list_run_ideas_empty_for_unknown_id():
    assert worker_db.list_run_ideas("missing") == []


def test_list_run_ideas_returns_in_insertion_order():
    rid = worker_db.create_run({"topic": "x"})
    for i in range(5):
        worker_db.append_run_idea(rid, {"title": f"idea-{i}"})
    out = worker_db.list_run_ideas(rid)
    assert [i["title"] for i in out] == [f"idea-{n}" for n in range(5)]


# ── Status + heartbeat ────────────────────────────────────────────────────

def test_set_run_status_done_clears_lease():
    rid = worker_db.create_run({"topic": "x"})
    worker_db.claim_next_pending_run("worker-A")
    worker_db.set_run_status(rid, "done")
    row = worker_db.get_run(rid)
    assert row["status"] == "done"
    assert row["lease_token"] is None


def test_set_run_status_failed_records_error():
    rid = worker_db.create_run({"topic": "x"})
    worker_db.set_run_status(rid, "failed", error="LLM exploded")
    row = worker_db.get_run(rid)
    assert row["status"] == "failed"
    assert row["error"] == "LLM exploded"


def test_set_run_status_rejects_invalid():
    rid = worker_db.create_run({"topic": "x"})
    with pytest.raises(ValueError):
        worker_db.set_run_status(rid, "bogus_status")


def test_heartbeat_only_updates_running_runs():
    """Heartbeat should be a no-op on pending/done/failed rows."""
    rid = worker_db.create_run({"topic": "x"})
    # While pending: heartbeat should not set heartbeat_at.
    worker_db.heartbeat_run(rid)
    row = worker_db.get_run(rid)
    assert row["heartbeat_at"] is None

    worker_db.claim_next_pending_run("worker-A")
    worker_db.heartbeat_run(rid)
    row = worker_db.get_run(rid)
    assert row["heartbeat_at"] is not None


def test_update_run_progress_explicit():
    rid = worker_db.create_run({"topic": "x"}, ideas_total=10)
    worker_db.update_run_progress(rid, ideas_done=4, ideas_total=15)
    row = worker_db.get_run(rid)
    assert row["ideas_done"] == 4
    assert row["ideas_total"] == 15


def test_update_run_progress_no_args_is_noop():
    rid = worker_db.create_run({"topic": "x"})
    worker_db.update_run_progress(rid)  # no kwargs
    # Just must not raise.


# ── Schema constants ───────────────────────────────────────────────────────

def test_lease_stale_seconds_is_reasonable():
    """Stale-lease threshold should be comfortably above the worker's
    heartbeat cadence (10s) so transient GC pauses don't trigger
    spurious reclaims."""
    assert worker_db.LEASE_STALE_SECONDS >= 30
    assert worker_db.LEASE_STALE_SECONDS <= 300
