"""
worker_db.py — SQLite job store for the independent background-worker pipeline.

Lives alongside the existing `db.py` (which has WAL mode + busy_timeout
already configured) and uses its connection helper. We add two tables —
`worker_runs` and `worker_run_ideas` — and keep them ADDITIVE: nothing
about the existing schema, agent memory, debates, papers, etc. is
touched.

Why a separate module:
  - `db.py` is large and concerns user accounts + results + agent memory.
  - The worker job store is a different domain (queue mechanics, leases,
    idempotent claim). Mixing them is asking for accidents.
  - The existing `scientist_runs` table is NOT a job queue — it's a
    record of completed automated-scientist sessions. We're building a
    queue, so we make a new table.

Schema (created lazily on first use; safe to call multiple times):

    worker_runs
      id              TEXT PK   — uuid4 hex, surfaced in st.query_params
      user_id         INTEGER   — optional, nullable for anonymous runs
      status          TEXT      — pending | running | done | failed
      params_json     TEXT      — JSON dict of generation parameters
      ideas_done      INTEGER   — count of ideas committed so far
      ideas_total     INTEGER   — best-effort target (max ideas in this run)
      error           TEXT      — last error message (nullable)
      created_at      TEXT      — ISO 8601 UTC
      updated_at      TEXT      — ISO 8601 UTC
      heartbeat_at    TEXT      — last worker heartbeat (nullable)
      lease_token     TEXT      — uuid of the worker that claimed it
      attempts        INTEGER   — number of times the run was claimed
                                  (incremented on resume)

    worker_run_ideas
      id              INTEGER PK AUTOINCREMENT
      run_id          TEXT FK to worker_runs(id)
      idea_json       TEXT      — JSON-serialized Idea.to_dict()
      archived_at     TEXT      — ISO 8601 UTC

Upgrade path (documented for the future): swap this module for an RQ
worker against a Redis backend with no API change — the helpers
(`create_run`, `claim_next_pending_run`, …) are the contract. RQ gives
us a real broker, pub/sub for live updates, and out-of-process retries.
For now, SQLite + WAL is enough for one VPS + several workers.

Concurrency model:
  - Multiple workers can poll simultaneously. Atomic claim via:
        UPDATE worker_runs
           SET status='running', lease_token=?, ...
         WHERE id=? AND status='pending'
    The conditional WHERE + rowcount check guarantees exactly one
    worker wins.
  - Stale leases (heartbeat older than LEASE_STALE_SECONDS) are
    reclaimable so a crashed worker doesn't strand a run forever.

Public API:
    init_worker_db()                                       -> None
    create_run(params, user_id=None, ideas_total=None)    -> run_id
    get_run(run_id)                                        -> dict | None
    list_runs(user_id=None, limit=50)                      -> list[dict]
    claim_next_pending_run(worker_id)                      -> dict | None
    append_run_idea(run_id, idea)                          -> int
    list_run_ideas(run_id)                                 -> list[dict]
    update_run_progress(run_id, *, ideas_done=None, ideas_total=None) -> None
    set_run_status(run_id, status, error=None)             -> None
    heartbeat_run(run_id)                                  -> None
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from db import _get_conn, _lock


# Stale-lease threshold: if a `running` run hasn't heartbeated in this
# many seconds, another worker may reclaim it. 90s is comfortably above
# the worker's 10s heartbeat cadence — survives transient GC pauses.
LEASE_STALE_SECONDS = 90

# Polling cadence the worker uses; matches what the docstring above hints.
DEFAULT_POLL_INTERVAL_S = 2.0


def _utcnow_iso() -> str:
    """ISO 8601 with explicit UTC suffix and microseconds.

    Microsecond resolution matters: list_runs() orders by created_at,
    and two runs created in quick succession by the Streamlit UI must
    sort deterministically. 1-second resolution loses that order.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# ── Schema ──────────────────────────────────────────────────────────────────

def init_worker_db() -> None:
    """Create the two worker tables if missing. Idempotent.

    Called once at module-import time from worker.py and from app.py so
    both sides agree on the schema even when the existing db.init_db()
    hasn't been called yet on this process.
    """
    with _lock:
        conn = _get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS worker_runs (
                    id            TEXT    PRIMARY KEY,
                    user_id       INTEGER,
                    status        TEXT    NOT NULL DEFAULT 'pending',
                    params_json   TEXT    NOT NULL,
                    ideas_done    INTEGER NOT NULL DEFAULT 0,
                    ideas_total   INTEGER NOT NULL DEFAULT 0,
                    error         TEXT,
                    created_at    TEXT    NOT NULL,
                    updated_at    TEXT    NOT NULL,
                    heartbeat_at  TEXT,
                    lease_token   TEXT,
                    attempts      INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                        ON DELETE SET NULL
                );
                CREATE INDEX IF NOT EXISTS idx_worker_runs_status
                    ON worker_runs(status, created_at);
                CREATE INDEX IF NOT EXISTS idx_worker_runs_user
                    ON worker_runs(user_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS worker_run_ideas (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id      TEXT    NOT NULL,
                    idea_json   TEXT    NOT NULL,
                    archived_at TEXT    NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES worker_runs(id)
                        ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_worker_run_ideas_run
                    ON worker_run_ideas(run_id, id);
            """)
            conn.commit()
        finally:
            conn.close()


# ── Row → dict helpers ──────────────────────────────────────────────────────

def _row_to_dict(row) -> Dict[str, Any]:
    if row is None:
        return None
    d = dict(row)
    # Convenience: inflate params_json
    if d.get("params_json"):
        try:
            d["params"] = json.loads(d["params_json"])
        except Exception:
            d["params"] = {}
    return d


# ── Create / read / list ────────────────────────────────────────────────────

def create_run(
    params: Dict[str, Any],
    user_id: Optional[int] = None,
    ideas_total: Optional[int] = None,
) -> str:
    """Insert a new `pending` run; return its run_id (uuid hex).

    `params` is serialized to JSON — anything not JSON-serializable will
    raise here, not deep inside the worker. That's intentional.
    """
    if not isinstance(params, dict):
        raise TypeError("params must be a dict")
    run_id = uuid.uuid4().hex
    now = _utcnow_iso()
    params_json = json.dumps(params, ensure_ascii=False, default=str)
    with _lock:
        conn = _get_conn()
        try:
            conn.execute(
                """
                INSERT INTO worker_runs
                    (id, user_id, status, params_json, ideas_total,
                     created_at, updated_at)
                VALUES (?, ?, 'pending', ?, ?, ?, ?)
                """,
                (run_id, user_id, params_json,
                 int(ideas_total or 0), now, now),
            )
            conn.commit()
        finally:
            conn.close()
    return run_id


def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    if not run_id:
        return None
    with _lock:
        conn = _get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM worker_runs WHERE id = ?", (run_id,),
            ).fetchone()
            return _row_to_dict(row)
        finally:
            conn.close()


def list_runs(
    user_id: Optional[int] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Return recent runs, newest first. Filter by user_id if given."""
    with _lock:
        conn = _get_conn()
        try:
            if user_id is None:
                rows = conn.execute(
                    "SELECT * FROM worker_runs "
                    "ORDER BY created_at DESC LIMIT ?",
                    (int(limit),),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM worker_runs WHERE user_id = ? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (int(user_id), int(limit)),
                ).fetchall()
            return [_row_to_dict(r) for r in rows]
        finally:
            conn.close()


# ── Worker claim (atomic) + heartbeat + status + idea append ───────────────

def claim_next_pending_run(worker_id: str) -> Optional[Dict[str, Any]]:
    """Atomically claim ONE pending or stale-running run.

    Strategy:
      1. Promote stale `running` rows back to `pending` (their lease has
         expired since the worker that held them stopped heartbeating).
      2. Pick the oldest `pending` row and try to flip its status to
         `running` with a CONDITIONAL UPDATE that includes
         `WHERE status='pending' AND id=?`. If another worker beat us
         to it, rowcount will be 0 and we move on.
      3. On successful claim, bump `attempts` (so the worker can detect
         a resume vs a fresh run via `attempts > 1`).

    Returns the claimed run dict, or None if no work is available.
    """
    now = _utcnow_iso()
    cutoff = (
        datetime.now(timezone.utc).timestamp() - LEASE_STALE_SECONDS
    )

    with _lock:
        conn = _get_conn()
        try:
            # 1. Reclaim stale leases.
            #    `heartbeat_at` is ISO-8601; compare via julianday() math.
            #    SQLite's strftime('%s', ...) gives seconds since epoch.
            conn.execute(
                """
                UPDATE worker_runs
                   SET status='pending', lease_token=NULL
                 WHERE status='running'
                   AND (heartbeat_at IS NULL
                        OR CAST(strftime('%s', heartbeat_at) AS INTEGER) < ?)
                """,
                (int(cutoff),),
            )

            # 2. Pick the oldest pending run.
            #    ISO-8601 strings sort lexicographically, so plain
            #    `ORDER BY created_at` is correct AND preserves the
            #    microsecond resolution that `datetime()` would strip.
            row = conn.execute(
                "SELECT * FROM worker_runs "
                "WHERE status='pending' "
                "ORDER BY created_at ASC "
                "LIMIT 1",
            ).fetchone()
            if row is None:
                conn.commit()
                return None

            # 3. Atomic claim — flip pending → running iff still pending.
            cur = conn.execute(
                """
                UPDATE worker_runs
                   SET status='running',
                       lease_token=?,
                       heartbeat_at=?,
                       updated_at=?,
                       attempts=attempts + 1
                 WHERE id=? AND status='pending'
                """,
                (worker_id, now, now, row["id"]),
            )
            conn.commit()
            if cur.rowcount != 1:
                # Lost the race to another worker.
                return None

            # Re-read after the update to get current state (with attempts++).
            row = conn.execute(
                "SELECT * FROM worker_runs WHERE id=?", (row["id"],),
            ).fetchone()
            return _row_to_dict(row)
        finally:
            conn.close()


def append_run_idea(run_id: str, idea: Dict[str, Any]) -> int:
    """Append one idea to worker_run_ideas + atomically increment
    worker_runs.ideas_done. Returns the new ideas_done count.

    Called from inside the worker's `on_idea_archived` callback — so this
    must be cheap and robust to weird idea dicts. Anything that can't
    serialize as JSON gets stringified."""
    if not run_id:
        raise ValueError("run_id is required")
    idea_json = json.dumps(idea, ensure_ascii=False, default=str)
    now = _utcnow_iso()
    with _lock:
        conn = _get_conn()
        try:
            conn.execute(
                "INSERT INTO worker_run_ideas (run_id, idea_json, archived_at) "
                "VALUES (?, ?, ?)",
                (run_id, idea_json, now),
            )
            conn.execute(
                "UPDATE worker_runs "
                "   SET ideas_done = ideas_done + 1, updated_at = ? "
                " WHERE id = ?",
                (now, run_id),
            )
            row = conn.execute(
                "SELECT ideas_done FROM worker_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            conn.commit()
            return int(row["ideas_done"]) if row else 0
        finally:
            conn.close()


def list_run_ideas(run_id: str) -> List[Dict[str, Any]]:
    """Return all committed ideas for a run, oldest first."""
    if not run_id:
        return []
    with _lock:
        conn = _get_conn()
        try:
            rows = conn.execute(
                "SELECT idea_json, archived_at "
                "  FROM worker_run_ideas "
                " WHERE run_id = ? "
                " ORDER BY id ASC",
                (run_id,),
            ).fetchall()
            out: List[Dict[str, Any]] = []
            for r in rows:
                try:
                    d = json.loads(r["idea_json"])
                except Exception:
                    d = {"raw": r["idea_json"]}
                if isinstance(d, dict):
                    d["_archived_at"] = r["archived_at"]
                out.append(d)
            return out
        finally:
            conn.close()


def update_run_progress(
    run_id: str,
    *,
    ideas_done: Optional[int] = None,
    ideas_total: Optional[int] = None,
) -> None:
    """Set explicit counts on the run row. Use sparingly — most
    increments happen automatically via `append_run_idea()`. This is for
    cases where the worker learns the target mid-run, or wants to
    backfill on resume."""
    if ideas_done is None and ideas_total is None:
        return
    now = _utcnow_iso()
    with _lock:
        conn = _get_conn()
        try:
            sets, args = [], []
            if ideas_done is not None:
                sets.append("ideas_done = ?")
                args.append(int(ideas_done))
            if ideas_total is not None:
                sets.append("ideas_total = ?")
                args.append(int(ideas_total))
            sets.append("updated_at = ?")
            args.append(now)
            args.append(run_id)
            conn.execute(
                f"UPDATE worker_runs SET {', '.join(sets)} WHERE id = ?",
                tuple(args),
            )
            conn.commit()
        finally:
            conn.close()


_VALID_STATUSES = {"pending", "running", "done", "failed"}


def set_run_status(
    run_id: str, status: str, error: Optional[str] = None,
) -> None:
    """Update status (and optionally error). Clears lease on terminal
    states (done/failed) so the row is no longer considered held."""
    if status not in _VALID_STATUSES:
        raise ValueError(
            f"invalid status {status!r}; must be one of {_VALID_STATUSES}"
        )
    now = _utcnow_iso()
    clear_lease = status in ("done", "failed", "pending")
    with _lock:
        conn = _get_conn()
        try:
            if clear_lease:
                conn.execute(
                    "UPDATE worker_runs "
                    "   SET status = ?, error = ?, "
                    "       lease_token = NULL, updated_at = ? "
                    " WHERE id = ?",
                    (status, error, now, run_id),
                )
            else:
                conn.execute(
                    "UPDATE worker_runs "
                    "   SET status = ?, error = ?, updated_at = ? "
                    " WHERE id = ?",
                    (status, error, now, run_id),
                )
            conn.commit()
        finally:
            conn.close()


def heartbeat_run(run_id: str) -> None:
    """Touch heartbeat_at so the run isn't reclaimed as stale.
    Call regularly (~every 10s) from the worker while a run is active."""
    now = _utcnow_iso()
    with _lock:
        conn = _get_conn()
        try:
            conn.execute(
                "UPDATE worker_runs "
                "   SET heartbeat_at = ?, updated_at = ? "
                " WHERE id = ? AND status = 'running'",
                (now, now, run_id),
            )
            conn.commit()
        finally:
            conn.close()
