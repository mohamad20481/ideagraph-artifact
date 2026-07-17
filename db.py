"""
db.py - SQLite database layer for user accounts, saved results, debates, papers, and agent memory.

Tables:
  - users: id, username, password_hash, salt, created_at
  - results: id, user_id (FK), topic, coverage, ideas_count, results_json, created_at
  - agent_memory: id, user_id, agent_role, memory_type, domain, content, relevance_score, ...
  - debates: id, user_id, result_id (FK), tournament_json, winner_idea_title, rounds_count, ...
  - papers_generated: id, user_id, idea_title, debate_id (FK), paper_markdown, created_at
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

_DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_DB_PATH = os.path.join(_DB_DIR, "ideagraph.db")

# ── Thread-local connection pool ─────────────────────────────────────────────
# Previously: one global threading.Lock serialized every DB op, and _get_conn()
# opened+closed a fresh connection on every call. For a Streamlit app with 14
# tabs that each make several db reads per rerun, this turned every rerun into
# a string of serialized connect/close cycles.
#
# Now: each thread gets one persistent SQLite connection cached on a
# threading.local. SQLite WAL mode supports concurrent readers, and
# busy_timeout=5000 handles writer contention at the SQLite layer, so the
# Python-level mutex is redundant. We keep `_lock` as a no-op so existing
# `with _lock:` call sites continue to compile unchanged.

_conn_local = threading.local()


class _PersistentConn:
    """
    Wrapper over sqlite3.Connection that makes .close() a no-op.

    Existing call sites do `conn = _get_conn() ... conn.close()` inside
    try/finally blocks. With thread-local caching we want to reuse the
    same connection, so close() must not actually close it — but it MUST
    roll back any dangling transaction so a failed call doesn't leave
    dirty state visible to the next caller on the same thread.

    Every other attribute delegates to the underlying sqlite3.Connection.
    """
    __slots__ = ("_conn",)

    def __init__(self, conn: sqlite3.Connection) -> None:
        object.__setattr__(self, "_conn", conn)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_conn"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_conn"), name, value)

    def close(self) -> None:
        # No-op close: connection is thread-local and reused across calls.
        # But roll back any dangling transaction so the next caller on this
        # thread inherits a clean state. rollback() is a no-op if there is
        # no open transaction, so this is safe to call unconditionally.
        try:
            object.__getattribute__(self, "_conn").rollback()
        except Exception:
            pass

    def __enter__(self):
        return object.__getattribute__(self, "_conn").__enter__()

    def __exit__(self, exc_type, exc, tb):
        return object.__getattribute__(self, "_conn").__exit__(exc_type, exc, tb)


class _NoOpLock:
    """
    Context manager that does nothing — kept so existing `with _lock:` blocks
    continue to compile. SQLite WAL + busy_timeout handles concurrency at the
    database layer, so the Python-level mutex only serialized ops unnecessarily.
    """
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def acquire(self, *args, **kwargs):
        return True

    def release(self):
        pass


_lock = _NoOpLock()  # API-compatible shim for existing call sites


def _get_conn() -> _PersistentConn:
    """
    Return a thread-local SQLite connection (reused across calls).

    The first call on a thread opens the underlying connection and sets
    pragmas; subsequent calls reuse it. `_PersistentConn.close()` is a
    no-op, so existing try/finally patterns keep working while avoiding
    repeated connect/close overhead.
    """
    wrapped = getattr(_conn_local, "wrapped", None)
    if wrapped is not None:
        return wrapped

    raw = sqlite3.connect(_DB_PATH, check_same_thread=False, timeout=10.0)
    raw.row_factory = sqlite3.Row
    raw.execute("PRAGMA journal_mode=WAL")
    raw.execute("PRAGMA synchronous=NORMAL")   # safe under WAL, much faster
    raw.execute("PRAGMA cache_size=-20000")    # ~20 MB page cache per conn
    raw.execute("PRAGMA mmap_size=30000000")   # 30 MB memory-mapped I/O for read-heavy workloads
    raw.execute("PRAGMA temp_store=MEMORY")
    raw.execute("PRAGMA foreign_keys=ON")
    raw.execute("PRAGMA busy_timeout=5000")    # auto-retry writer contention
    wrapped = _PersistentConn(raw)
    _conn_local.wrapped = wrapped
    return wrapped


def init_db() -> None:
    """Create tables if they don't exist. Safe to call multiple times."""
    os.makedirs(_DB_DIR, exist_ok=True)
    with _lock:
        conn = _get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    username    TEXT    NOT NULL UNIQUE COLLATE NOCASE,
                    password_hash TEXT NOT NULL,
                    salt        TEXT    NOT NULL,
                    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS results (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id      INTEGER NOT NULL,
                    topic        TEXT    NOT NULL,
                    coverage     REAL    NOT NULL DEFAULT 0.0,
                    ideas_count  INTEGER NOT NULL DEFAULT 0,
                    results_json TEXT    NOT NULL,
                    created_at   TEXT    NOT NULL DEFAULT (datetime('now')),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_results_user_id ON results(user_id);

                CREATE TABLE IF NOT EXISTS agent_memory (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id         INTEGER NOT NULL,
                    agent_role      TEXT    NOT NULL,
                    memory_type     TEXT    NOT NULL,
                    domain          TEXT    NOT NULL DEFAULT '',
                    content         TEXT    NOT NULL,
                    relevance_score REAL    NOT NULL DEFAULT 0.5,
                    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
                    last_used       TEXT,
                    use_count       INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_memory_lookup
                    ON agent_memory(user_id, agent_role, domain);

                CREATE TABLE IF NOT EXISTS debates (
                    id                INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id           INTEGER NOT NULL,
                    result_id         INTEGER,
                    tournament_json   TEXT    NOT NULL,
                    winner_idea_title TEXT,
                    rounds_count      INTEGER NOT NULL DEFAULT 0,
                    created_at        TEXT    NOT NULL DEFAULT (datetime('now')),
                    FOREIGN KEY (user_id)    REFERENCES users(id)    ON DELETE CASCADE,
                    FOREIGN KEY (result_id)  REFERENCES results(id)  ON DELETE SET NULL
                );
                CREATE INDEX IF NOT EXISTS idx_debates_user
                    ON debates(user_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS papers_generated (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id         INTEGER NOT NULL,
                    idea_title      TEXT    NOT NULL,
                    debate_id       INTEGER,
                    paper_markdown  TEXT    NOT NULL,
                    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
                    FOREIGN KEY (user_id)   REFERENCES users(id)   ON DELETE CASCADE,
                    FOREIGN KEY (debate_id) REFERENCES debates(id) ON DELETE SET NULL
                );
                CREATE INDEX IF NOT EXISTS idx_papers_user
                    ON papers_generated(user_id, created_at DESC);

                /* Per-result chat sessions (the 💬 Chat tab). One row per
                   distinct (user, result) "current session" + additional
                   rows when the user saves a named snapshot.

                   - `idea_title` is NULL for whole-result chats, set for
                     single-idea chats (idea_chat.py) if/when we persist
                     those too.
                   - `is_snapshot=0` = the auto-saved current session (one
                     per result, upserted in place). `is_snapshot=1` =
                     user-named snapshots (multiple allowed per result).
                */
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id       INTEGER NOT NULL,
                    result_id     INTEGER,
                    idea_title    TEXT,
                    title         TEXT NOT NULL DEFAULT '',
                    history_json  TEXT NOT NULL DEFAULT '[]',
                    message_count INTEGER NOT NULL DEFAULT 0,
                    is_snapshot   INTEGER NOT NULL DEFAULT 0,
                    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at    TEXT NOT NULL DEFAULT (datetime('now')),
                    FOREIGN KEY (user_id)   REFERENCES users(id)   ON DELETE CASCADE,
                    FOREIGN KEY (result_id) REFERENCES results(id) ON DELETE SET NULL
                );
                CREATE INDEX IF NOT EXISTS idx_chat_sessions_user
                    ON chat_sessions(user_id, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_chat_sessions_current
                    ON chat_sessions(user_id, result_id, idea_title, is_snapshot);

                CREATE TABLE IF NOT EXISTS idea_evolution (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id       INTEGER NOT NULL,
                    idea_title    TEXT    NOT NULL,
                    generation    INTEGER NOT NULL DEFAULT 0,
                    parent_id     INTEGER,
                    event_type    TEXT    NOT NULL,
                    score_before  REAL,
                    score_after   REAL,
                    diff_summary  TEXT,
                    snapshot_json TEXT    NOT NULL,
                    created_at    TEXT    NOT NULL DEFAULT (datetime('now')),
                    FOREIGN KEY (user_id)   REFERENCES users(id)   ON DELETE CASCADE,
                    FOREIGN KEY (parent_id) REFERENCES idea_evolution(id)
                );

                CREATE TABLE IF NOT EXISTS scientist_runs (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id         INTEGER NOT NULL,
                    topic           TEXT    NOT NULL,
                    status          TEXT    NOT NULL DEFAULT 'running',
                    iterations_json TEXT,
                    final_paper_md  TEXT,
                    final_paper_tex TEXT,
                    review_json     TEXT,
                    total_elapsed   REAL    DEFAULT 0,
                    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_scientist_runs_user
                    ON scientist_runs(user_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS bookmarks (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id     INTEGER NOT NULL,
                    idea_title  TEXT    NOT NULL,
                    idea_json   TEXT    NOT NULL,
                    note        TEXT    NOT NULL DEFAULT '',
                    tags        TEXT    NOT NULL DEFAULT '',
                    collection  TEXT    NOT NULL DEFAULT 'default',
                    rating      INTEGER NOT NULL DEFAULT 0,
                    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_bookmarks_user ON bookmarks(user_id);
                CREATE INDEX IF NOT EXISTS idx_bookmarks_user_created
                    ON bookmarks(user_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS share_tokens (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    token       TEXT    NOT NULL UNIQUE,
                    user_id     INTEGER NOT NULL,
                    idea_json   TEXT    NOT NULL,
                    topic       TEXT    NOT NULL DEFAULT '',
                    views       INTEGER NOT NULL DEFAULT 0,
                    likes       INTEGER NOT NULL DEFAULT 0,
                    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_share_token ON share_tokens(token);
                CREATE INDEX IF NOT EXISTS idx_share_tokens_popularity
                    ON share_tokens(views DESC, likes DESC);

                CREATE TABLE IF NOT EXISTS email_subscribers (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    email       TEXT    NOT NULL UNIQUE COLLATE NOCASE,
                    subscribed  INTEGER NOT NULL DEFAULT 1,
                    preferences TEXT    NOT NULL DEFAULT '',
                    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
                    last_sent   TEXT
                );

                CREATE TABLE IF NOT EXISTS subscriptions (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id             INTEGER NOT NULL UNIQUE,
                    tier                TEXT    NOT NULL DEFAULT 'free',
                    stripe_customer_id  TEXT,
                    stripe_subscription_id TEXT,
                    status              TEXT    NOT NULL DEFAULT 'active',
                    current_period_end  TEXT,
                    runs_this_month     INTEGER NOT NULL DEFAULT 0,
                    created_at          TEXT    NOT NULL DEFAULT (datetime('now')),
                    updated_at          TEXT    NOT NULL DEFAULT (datetime('now')),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS user_stats (
                    user_id       INTEGER PRIMARY KEY,
                    xp            INTEGER NOT NULL DEFAULT 0,
                    level         INTEGER NOT NULL DEFAULT 1,
                    current_streak INTEGER NOT NULL DEFAULT 0,
                    longest_streak INTEGER NOT NULL DEFAULT 0,
                    last_active_date TEXT,
                    total_ideas   INTEGER NOT NULL DEFAULT 0,
                    total_runs    INTEGER NOT NULL DEFAULT 0,
                    total_shares  INTEGER NOT NULL DEFAULT 0,
                    total_bookmarks INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS achievements (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id     INTEGER NOT NULL,
                    badge_key   TEXT    NOT NULL,
                    unlocked_at TEXT    NOT NULL DEFAULT (datetime('now')),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    UNIQUE(user_id, badge_key)
                );
                CREATE INDEX IF NOT EXISTS idx_achievements_user ON achievements(user_id);

                CREATE TABLE IF NOT EXISTS follows (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    follower_id INTEGER NOT NULL,
                    followed_id INTEGER NOT NULL,
                    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
                    FOREIGN KEY (follower_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (followed_id) REFERENCES users(id) ON DELETE CASCADE,
                    UNIQUE(follower_id, followed_id)
                );

                CREATE TABLE IF NOT EXISTS activity_feed (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id     INTEGER NOT NULL,
                    activity_type TEXT  NOT NULL,
                    content     TEXT    NOT NULL,
                    metadata    TEXT,
                    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_activity_user ON activity_feed(user_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS daily_picks (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id     INTEGER NOT NULL,
                    idea_json   TEXT    NOT NULL,
                    date        TEXT    NOT NULL,
                    seen        INTEGER NOT NULL DEFAULT 0,
                    liked       INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    UNIQUE(user_id, date)
                );

                CREATE TABLE IF NOT EXISTS onboarding_progress (
                    user_id     INTEGER PRIMARY KEY,
                    step        INTEGER NOT NULL DEFAULT 0,
                    completed   INTEGER NOT NULL DEFAULT 0,
                    tutorial_dismissed INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );
            """)
            conn.commit()
        finally:
            conn.close()
    # Always run migrations after init (safe to call repeatedly).
    migrate_db()


def migrate_db() -> None:
    """
    Apply pending schema migrations (idempotent — safe to call every startup).

    Each migration is wrapped in its own try/except so a single failure doesn't
    block the rest. Column-add failures with "duplicate column" are silently
    ignored (= migration already applied).
    """
    def _try_exec(conn, sql: str) -> None:
        try:
            conn.executescript(sql)
            conn.commit()
        except Exception as e:
            if "duplicate column" not in str(e).lower():
                pass  # non-blocking: log and continue

    with _lock:
        conn = _get_conn()
        try:
            # Migration 1: Add elapsed_seconds + estimated_cost_usd to results.
            _try_exec(conn, "ALTER TABLE results ADD COLUMN elapsed_seconds REAL DEFAULT 0;")
            _try_exec(conn, "ALTER TABLE results ADD COLUMN estimated_cost_usd REAL DEFAULT 0;")

            # Migration 2: Composite index for time-series queries on results.
            _try_exec(conn, """
                CREATE INDEX IF NOT EXISTS idx_results_user_created
                    ON results(user_id, created_at DESC);
            """)

            # Migration 3: Cost tracking audit log.
            _try_exec(conn, """
                CREATE TABLE IF NOT EXISTS cost_tracking_log (
                    id                INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id           INTEGER,
                    timestamp         TEXT    NOT NULL DEFAULT (datetime('now')),
                    provider          TEXT    NOT NULL,
                    model             TEXT    NOT NULL,
                    prompt_tokens     INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    cost_usd          REAL    NOT NULL,
                    stage             TEXT    NOT NULL DEFAULT 'unknown',
                    run_id            TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_cost_log_user
                    ON cost_tracking_log(user_id, timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_cost_log_run
                    ON cost_tracking_log(run_id);
            """)

            # Migration 4: Milestones table for engagement.
            _try_exec(conn, """
                CREATE TABLE IF NOT EXISTS milestones (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id         INTEGER NOT NULL,
                    milestone_type  TEXT    NOT NULL,
                    unlocked_at     TEXT    NOT NULL DEFAULT (datetime('now')),
                    notified        INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    UNIQUE(user_id, milestone_type)
                );
            """)

            # Migration 5: Add elapsed_seconds to debates.
            _try_exec(conn, "ALTER TABLE debates ADD COLUMN elapsed_seconds REAL DEFAULT 0;")

            # Migration 6: User feedback on ideas (useful/not useful).
            _try_exec(conn, """
                CREATE TABLE IF NOT EXISTS idea_feedback (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id     INTEGER NOT NULL,
                    idea_title  TEXT    NOT NULL,
                    feedback    TEXT    NOT NULL DEFAULT 'neutral',
                    comment     TEXT    NOT NULL DEFAULT '',
                    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    UNIQUE(user_id, idea_title)
                );
            """)

            # Migration 7: Referrals table.
            _try_exec(conn, """
                CREATE TABLE IF NOT EXISTS referrals (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    referrer_id INTEGER NOT NULL,
                    referred_id INTEGER NOT NULL,
                    code        TEXT    NOT NULL,
                    xp_awarded  INTEGER NOT NULL DEFAULT 0,
                    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
                    FOREIGN KEY (referrer_id) REFERENCES users(id),
                    FOREIGN KEY (referred_id) REFERENCES users(id),
                    UNIQUE(referred_id)
                );
            """)

        finally:
            conn.close()


def save_idea_feedback(user_id: int, idea_title: str, feedback: str,
                       comment: str = "") -> None:
    """Save user feedback (useful/not_useful/neutral) on an idea."""
    with _lock:
        conn = _get_conn()
        try:
            conn.execute(
                """INSERT INTO idea_feedback (user_id, idea_title, feedback, comment)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(user_id, idea_title)
                   DO UPDATE SET feedback = excluded.feedback,
                                 comment = excluded.comment,
                                 created_at = datetime('now')""",
                (user_id, idea_title, feedback, comment),
            )
            conn.commit()
        finally:
            conn.close()


def get_idea_feedback(user_id: int) -> Dict[str, str]:
    """Get all feedback for a user as {idea_title: feedback}."""
    with _lock:
        conn = _get_conn()
        try:
            rows = conn.execute(
                "SELECT idea_title, feedback FROM idea_feedback WHERE user_id = ?",
                (user_id,),
            ).fetchall()
            return {r["idea_title"]: r["feedback"] for r in rows}
        finally:
            conn.close()


# ── Persistent cost logging (writes to cost_tracking_log table) ──────────────

def log_cost(
    provider: str, model: str, prompt_tokens: int, completion_tokens: int,
    cost_usd: float, stage: str = "unknown",
    user_id: Optional[int] = None, run_id: Optional[str] = None,
) -> None:
    """Append one LLM-call cost record to the audit trail."""
    with _lock:
        conn = _get_conn()
        try:
            conn.execute(
                """INSERT INTO cost_tracking_log
                   (user_id, provider, model, prompt_tokens, completion_tokens,
                    cost_usd, stage, run_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (user_id, provider, model, prompt_tokens, completion_tokens,
                 cost_usd, stage, run_id),
            )
            conn.commit()
        except Exception:
            pass
        finally:
            conn.close()


def get_user_cost_summary(user_id: int) -> Dict[str, Any]:
    """Get cost summary for a user (total, by provider, by stage)."""
    with _lock:
        conn = _get_conn()
        try:
            total = conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) FROM cost_tracking_log WHERE user_id = ?",
                (user_id,),
            ).fetchone()[0]
            by_provider = conn.execute(
                """SELECT provider, SUM(cost_usd) as total, COUNT(*) as calls
                   FROM cost_tracking_log WHERE user_id = ?
                   GROUP BY provider ORDER BY total DESC""",
                (user_id,),
            ).fetchall()
            return {
                "total_usd": round(total, 4),
                "by_provider": [dict(r) for r in by_provider],
            }
        finally:
            conn.close()


# ── Password hashing ─────────────────────────────────────────────────────────

def _hash_password(password: str, salt: bytes) -> str:
    """PBKDF2-HMAC-SHA256 with 260k iterations."""
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 260_000)
    return dk.hex()


# ── User operations ──────────────────────────────────────────────────────────

def register_user(username: str, password: str) -> Optional[int]:
    """
    Register a new user. Returns user_id on success, None if username taken.
    """
    username = username.strip()
    if not username or not password:
        return None

    salt = os.urandom(32)
    pw_hash = _hash_password(password, salt)

    with _lock:
        conn = _get_conn()
        try:
            conn.execute(
                "INSERT INTO users (username, password_hash, salt) VALUES (?, ?, ?)",
                (username, pw_hash, salt.hex()),
            )
            conn.commit()
            user_id = conn.execute(
                "SELECT id FROM users WHERE username = ?", (username,)
            ).fetchone()["id"]
            return user_id
        except sqlite3.IntegrityError:
            return None  # username already exists
        finally:
            conn.close()


def login_user(username: str, password: str) -> Optional[int]:
    """
    Validate credentials. Returns user_id on success, None on failure.
    """
    username = username.strip()
    with _lock:
        conn = _get_conn()
        try:
            row = conn.execute(
                "SELECT id, password_hash, salt FROM users WHERE username = ?",
                (username,),
            ).fetchone()
            if row is None:
                return None
            pw_hash = _hash_password(password, bytes.fromhex(row["salt"]))
            if pw_hash == row["password_hash"]:
                return row["id"]
            return None
        finally:
            conn.close()


def get_user_profile(user_id: int) -> Optional[Dict[str, Any]]:
    """Return profile fields for the Manage Account page. Returns None
    if the user doesn't exist."""
    with _lock:
        conn = _get_conn()
        try:
            row = conn.execute(
                "SELECT id, username, created_at FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()


def change_password(
    user_id: int, old_password: str, new_password: str,
) -> Dict[str, Any]:
    """Change a user's password after verifying the old one.

    Returns {"ok": True} on success or {"ok": False, "error": "..."} with
    a human-readable reason on failure (wrong old password, too short,
    user not found).
    """
    if not new_password or len(new_password) < 6:
        return {"ok": False, "error": "New password must be at least 6 characters."}
    with _lock:
        conn = _get_conn()
        try:
            row = conn.execute(
                "SELECT password_hash, salt FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
            if row is None:
                return {"ok": False, "error": "User not found."}
            old_hash = _hash_password(old_password, bytes.fromhex(row["salt"]))
            if old_hash != row["password_hash"]:
                return {"ok": False, "error": "Current password is incorrect."}
            new_salt = os.urandom(32)
            new_hash = _hash_password(new_password, new_salt)
            conn.execute(
                "UPDATE users SET password_hash = ?, salt = ? WHERE id = ?",
                (new_hash, new_salt.hex(), user_id),
            )
            conn.commit()
            return {"ok": True}
        finally:
            conn.close()


def delete_user(user_id: int, password_confirm: str) -> Dict[str, Any]:
    """Delete a user account after verifying their password.

    Cascades to results / subscriptions / etc. via FK ON DELETE CASCADE
    where wired; for tables without cascade, deletes rows by user_id.
    Returns {"ok": True, "deleted": <user_id>} or {"ok": False, "error": ...}.
    """
    with _lock:
        conn = _get_conn()
        try:
            row = conn.execute(
                "SELECT password_hash, salt FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
            if row is None:
                return {"ok": False, "error": "User not found."}
            pw_hash = _hash_password(password_confirm, bytes.fromhex(row["salt"]))
            if pw_hash != row["password_hash"]:
                return {"ok": False, "error": "Password is incorrect."}

            # Manual cleanup for tables that may not have ON DELETE CASCADE
            # in older schemas. Wrapped in try/except per-table so a missing
            # table doesn't abort the whole deletion.
            for tbl in (
                "results", "subscriptions", "user_stats",
                "achievements", "saved_presets", "referrals",
                "remember_tokens", "session_recovery_tokens",
            ):
                try:
                    conn.execute(f"DELETE FROM {tbl} WHERE user_id = ?", (user_id,))
                except Exception:
                    pass

            conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
            conn.commit()
            return {"ok": True, "deleted": user_id}
        finally:
            conn.close()


def export_user_data(user_id: int) -> Dict[str, Any]:
    """Return everything we have on this user as a JSON-serializable dict.

    Used by the "Export my data" button on the Manage Account page (GDPR-
    style takeout). Does NOT include password hashes or salts.
    """
    out: Dict[str, Any] = {
        "exported_at": _datetime_now_iso(),
        "user": None,
        "subscription": None,
        "results": [],
    }
    with _lock:
        conn = _get_conn()
        try:
            row = conn.execute(
                "SELECT id, username, created_at FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
            if not row:
                return out
            out["user"] = dict(row)

            try:
                sub = conn.execute(
                    "SELECT id, user_id, tier, status, current_period_end, "
                    "       runs_this_month, created_at, updated_at "
                    "  FROM subscriptions WHERE user_id = ?",
                    (user_id,),
                ).fetchone()
                if sub:
                    out["subscription"] = dict(sub)
            except Exception:
                pass

            try:
                res = conn.execute(
                    "SELECT id, topic, coverage, ideas_count, results_json, created_at "
                    "  FROM results WHERE user_id = ? ORDER BY created_at DESC",
                    (user_id,),
                ).fetchall()
                out["results"] = [dict(r) for r in res]
            except Exception:
                pass
        finally:
            conn.close()
    return out


def _datetime_now_iso() -> str:
    """ISO-8601 'now' for export timestamps. Local helper so we don't
    import datetime at module top just for this."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


# ── Results operations ───────────────────────────────────────────────────────

def save_result(
    user_id: int,
    topic: str,
    coverage: float,
    ideas_count: int,
    results_dict: Dict[str, Any],
) -> int:
    """Save a pipeline result for a user. Returns the result row id."""
    results_json = json.dumps(results_dict, ensure_ascii=False)
    # Extract timing + cost from results if available (set by pipeline).
    stats = results_dict.get("stats", {})
    elapsed = stats.get("elapsed_seconds", 0)
    cost = stats.get("estimated_cost_usd", 0)
    with _lock:
        conn = _get_conn()
        try:
            cursor = conn.execute(
                """INSERT INTO results
                   (user_id, topic, coverage, ideas_count, results_json,
                    elapsed_seconds, estimated_cost_usd)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (user_id, topic.strip(), coverage, ideas_count, results_json,
                 elapsed, cost),
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()


def get_user_results(user_id: int) -> List[Dict[str, Any]]:
    """Get summary list of all results for a user (newest first)."""
    with _lock:
        conn = _get_conn()
        try:
            rows = conn.execute(
                """SELECT id, topic, coverage, ideas_count, created_at
                   FROM results WHERE user_id = ? ORDER BY created_at DESC""",
                (user_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


def search_user_results(
    user_id: int, query: str, limit: int = 50,
) -> List[Dict[str, Any]]:
    """Search a user's saved results by topic AND idea content.

    Performs a case-insensitive substring search over:
      - results.topic
      - results.results_json (the full pipeline output blob — so idea
        titles, motivations, methods, etc. are all searchable)

    Empty / whitespace-only query returns the full list (same shape as
    get_user_results). Token splitting: any whitespace splits the query
    into AND-ed terms — every term must appear somewhere in the row.
    """
    q = (query or "").strip()
    if not q:
        return get_user_results(user_id)[:limit]

    terms = [t for t in q.split() if t]
    if not terms:
        return get_user_results(user_id)[:limit]

    # Build SQL: WHERE (topic LIKE ? OR results_json LIKE ?) AND (..) ...
    where = ["user_id = ?"]
    params: List[Any] = [user_id]
    for t in terms:
        like = f"%{t}%"
        where.append("(LOWER(topic) LIKE LOWER(?) OR LOWER(results_json) LIKE LOWER(?))")
        params.extend([like, like])
    params.append(limit)

    sql = (
        "SELECT id, topic, coverage, ideas_count, created_at "
        "  FROM results "
        " WHERE " + " AND ".join(where) +
        " ORDER BY created_at DESC LIMIT ?"
    )
    with _lock:
        conn = _get_conn()
        try:
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


def load_result(result_id: int, user_id: int) -> Optional[Dict[str, Any]]:
    """Load the full results JSON for a specific result owned by user."""
    with _lock:
        conn = _get_conn()
        try:
            row = conn.execute(
                "SELECT results_json FROM results WHERE id = ? AND user_id = ?",
                (result_id, user_id),
            ).fetchone()
            if row is None:
                return None
            return json.loads(row["results_json"])
        finally:
            conn.close()


def get_user_results_full(user_id: int) -> List[Dict[str, Any]]:
    """
    Batched variant of get_user_results + load_result(per row): returns every
    saved run for the user with its full parsed `results_json` payload merged
    in, in a SINGLE SQL query. Eliminates the N+1 round-trip pattern used by
    `idea_evolution.get_idea_history` and `compute_quality_trajectory`.

    Returned dicts contain the same summary keys as `get_user_results`
    (id, topic, coverage, ideas_count, created_at) plus all keys from the
    parsed JSON (ideas, stats, etc.). Rows whose JSON fails to parse are
    skipped.
    """
    with _lock:
        conn = _get_conn()
        try:
            rows = conn.execute(
                """SELECT id, topic, coverage, ideas_count, created_at, results_json
                   FROM results WHERE user_id = ? ORDER BY created_at DESC""",
                (user_id,),
            ).fetchall()
        finally:
            conn.close()

    out: List[Dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        raw = d.pop("results_json", None)
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except (ValueError, TypeError):
            continue
        # Merge: summary fields take precedence over duplicates in JSON
        for k, v in payload.items():
            d.setdefault(k, v)
        out.append(d)
    return out


def delete_result(result_id: int, user_id: int) -> bool:
    """Delete a result owned by user. Returns True if a row was deleted."""
    with _lock:
        conn = _get_conn()
        try:
            cursor = conn.execute(
                "DELETE FROM results WHERE id = ? AND user_id = ?",
                (result_id, user_id),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()


# ── Chat session operations (💬 Chat tab persistence) ───────────────────────

def save_chat_session(
    user_id: int,
    history: List[Dict[str, Any]],
    result_id: Optional[int] = None,
    idea_title: Optional[str] = None,
    title: str = "",
    is_snapshot: bool = False,
    session_id: Optional[int] = None,
) -> int:
    """Save a chat transcript.

    Two flavors keyed by `is_snapshot`:

    - **Current session** (`is_snapshot=False`): upsert keyed on
      (user_id, result_id, idea_title). One row per chat context,
      overwritten as new messages arrive. Pass `session_id` only if
      you already know it — otherwise the helper finds-or-creates.

    - **Named snapshot** (`is_snapshot=True`): always inserts a fresh
      row, leaving prior snapshots intact so users can keep multiple
      "saved versions" of an evolving conversation.

    Returns the row id of the saved session.
    """
    history_json = json.dumps(history or [], ensure_ascii=False)
    msg_count = len(history or [])
    with _lock:
        conn = _get_conn()
        try:
            # ── Snapshot path: always INSERT ────────────────────────────
            if is_snapshot:
                cur = conn.execute(
                    "INSERT INTO chat_sessions "
                    "(user_id, result_id, idea_title, title, "
                    " history_json, message_count, is_snapshot) "
                    "VALUES (?, ?, ?, ?, ?, ?, 1)",
                    (user_id, result_id, idea_title, title or "Snapshot",
                     history_json, msg_count),
                )
                conn.commit()
                return cur.lastrowid

            # ── Current-session path: find-or-create then UPDATE ───────
            if session_id is None:
                row = conn.execute(
                    "SELECT id FROM chat_sessions "
                    " WHERE user_id = ? AND is_snapshot = 0 "
                    "   AND COALESCE(result_id, -1) = COALESCE(?, -1) "
                    "   AND COALESCE(idea_title, '') = COALESCE(?, '') "
                    " LIMIT 1",
                    (user_id, result_id, idea_title),
                ).fetchone()
                if row is None:
                    cur = conn.execute(
                        "INSERT INTO chat_sessions "
                        "(user_id, result_id, idea_title, title, "
                        " history_json, message_count, is_snapshot) "
                        "VALUES (?, ?, ?, ?, ?, ?, 0)",
                        (user_id, result_id, idea_title, title or "",
                         history_json, msg_count),
                    )
                    conn.commit()
                    return cur.lastrowid
                session_id = row["id"]

            conn.execute(
                "UPDATE chat_sessions SET history_json = ?, "
                "       message_count = ?, "
                "       title = COALESCE(NULLIF(?, ''), title), "
                "       updated_at = datetime('now') "
                " WHERE id = ? AND user_id = ?",
                (history_json, msg_count, title, session_id, user_id),
            )
            conn.commit()
            return session_id
        finally:
            conn.close()


def load_chat_session(session_id: int, user_id: int) -> Optional[Dict[str, Any]]:
    """Load one chat session by id. Returns None if not found / not owned."""
    with _lock:
        conn = _get_conn()
        try:
            row = conn.execute(
                "SELECT id, user_id, result_id, idea_title, title, "
                "       history_json, message_count, is_snapshot, "
                "       created_at, updated_at "
                "  FROM chat_sessions "
                " WHERE id = ? AND user_id = ?",
                (session_id, user_id),
            ).fetchone()
            if row is None:
                return None
            out = dict(row)
            try:
                out["history"] = json.loads(out.pop("history_json") or "[]")
            except Exception:
                out["history"] = []
            return out
        finally:
            conn.close()


def load_current_chat_session(
    user_id: int,
    result_id: Optional[int] = None,
    idea_title: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Convenience: load the auto-saved current session for a chat
    context (the panel calls this on open to hydrate session_state).
    Returns None when nothing has been saved yet."""
    with _lock:
        conn = _get_conn()
        try:
            row = conn.execute(
                "SELECT id FROM chat_sessions "
                " WHERE user_id = ? AND is_snapshot = 0 "
                "   AND COALESCE(result_id, -1) = COALESCE(?, -1) "
                "   AND COALESCE(idea_title, '') = COALESCE(?, '') "
                " LIMIT 1",
                (user_id, result_id, idea_title),
            ).fetchone()
        finally:
            conn.close()
    if row is None:
        return None
    return load_chat_session(row["id"], user_id)


def list_chat_sessions(
    user_id: int,
    result_id: Optional[int] = None,
    snapshots_only: bool = False,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """List chat sessions for a user, newest first. Filterable by
    result_id and by snapshot-vs-current."""
    where = ["user_id = ?"]
    params: List[Any] = [user_id]
    if result_id is not None:
        where.append("result_id = ?")
        params.append(result_id)
    if snapshots_only:
        where.append("is_snapshot = 1")
    params.append(limit)
    sql = (
        "SELECT id, result_id, idea_title, title, message_count, "
        "       is_snapshot, created_at, updated_at "
        "  FROM chat_sessions "
        " WHERE " + " AND ".join(where) +
        " ORDER BY updated_at DESC LIMIT ?"
    )
    with _lock:
        conn = _get_conn()
        try:
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


def delete_chat_session(session_id: int, user_id: int) -> bool:
    """Delete a chat session owned by user. Returns True if a row was deleted."""
    with _lock:
        conn = _get_conn()
        try:
            cur = conn.execute(
                "DELETE FROM chat_sessions WHERE id = ? AND user_id = ?",
                (session_id, user_id),
            )
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()


# ── Agent memory operations ──────────────────────────────────────────────────

def save_agent_memory(
    user_id: int, agent_role: str, memory_type: str,
    domain: str, content_dict: Dict[str, Any],
    relevance_score: float = 0.5,
) -> int:
    """Store an agent memory entry. Returns the row id."""
    content_json = json.dumps(content_dict, ensure_ascii=False)
    with _lock:
        conn = _get_conn()
        try:
            cur = conn.execute(
                """INSERT INTO agent_memory
                   (user_id, agent_role, memory_type, domain, content, relevance_score)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (user_id, agent_role, memory_type, domain, content_json, relevance_score),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()


def query_agent_memory(
    user_id: int, agent_role: str, domain: str, limit: int = 5,
) -> List[Dict[str, Any]]:
    """Retrieve relevant memories sorted by relevance × recency. Increments use_count."""
    with _lock:
        conn = _get_conn()
        try:
            rows = conn.execute(
                """SELECT id, agent_role, memory_type, domain, content,
                          relevance_score, created_at, use_count
                   FROM agent_memory
                   WHERE user_id = ? AND agent_role = ? AND domain = ?
                   ORDER BY relevance_score DESC, created_at DESC
                   LIMIT ?""",
                (user_id, agent_role, domain, limit),
            ).fetchall()
            if not rows:
                return []
            # Batch the use_count bump into a single UPDATE. Previously this
            # fired one UPDATE per row inside the read loop — an N+1 that
            # turned a 5-row read into 6 SQL statements. Using `id IN (...)`
            # collapses it to two statements total for any `limit`.
            ids = [r["id"] for r in rows]
            placeholders = ",".join("?" * len(ids))
            conn.execute(
                f"UPDATE agent_memory SET use_count = use_count + 1, "
                f"last_used = datetime('now') WHERE id IN ({placeholders})",
                ids,
            )
            conn.commit()
            results = []
            for r in rows:
                d = dict(r)
                d["content"] = json.loads(d["content"])
                results.append(d)
            return results
        finally:
            conn.close()


# ── Debate operations ────────────────────────────────────────────────────────

def save_debate(
    user_id: int, result_id: Optional[int], tournament_dict: Dict[str, Any],
    winner_title: str, rounds_count: int,
) -> int:
    """Save a tournament debate. Returns the debate row id."""
    tj = json.dumps(tournament_dict, ensure_ascii=False)
    with _lock:
        conn = _get_conn()
        try:
            cur = conn.execute(
                """INSERT INTO debates
                   (user_id, result_id, tournament_json, winner_idea_title, rounds_count)
                   VALUES (?, ?, ?, ?, ?)""",
                (user_id, result_id, tj, winner_title, rounds_count),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()


def get_user_debates(user_id: int) -> List[Dict[str, Any]]:
    """List all debates for a user (newest first)."""
    with _lock:
        conn = _get_conn()
        try:
            rows = conn.execute(
                """SELECT id, result_id, winner_idea_title, rounds_count, created_at
                   FROM debates WHERE user_id = ? ORDER BY created_at DESC""",
                (user_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


def load_debate(debate_id: int, user_id: int) -> Optional[Dict[str, Any]]:
    """Load full tournament JSON for a debate."""
    with _lock:
        conn = _get_conn()
        try:
            row = conn.execute(
                "SELECT tournament_json FROM debates WHERE id = ? AND user_id = ?",
                (debate_id, user_id),
            ).fetchone()
            if row is None:
                return None
            return json.loads(row["tournament_json"])
        finally:
            conn.close()


# ── Paper operations ─────────────────────────────────────────────────────────

def save_paper(
    user_id: int, idea_title: str, paper_markdown: str,
    debate_id: Optional[int] = None,
) -> int:
    """Save a generated paper. Returns the paper row id."""
    with _lock:
        conn = _get_conn()
        try:
            cur = conn.execute(
                """INSERT INTO papers_generated
                   (user_id, idea_title, debate_id, paper_markdown)
                   VALUES (?, ?, ?, ?)""",
                (user_id, idea_title, debate_id, paper_markdown),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()


def get_user_papers(user_id: int) -> List[Dict[str, Any]]:
    """List all papers for a user (newest first)."""
    with _lock:
        conn = _get_conn()
        try:
            rows = conn.execute(
                """SELECT id, idea_title, debate_id, created_at
                   FROM papers_generated WHERE user_id = ? ORDER BY created_at DESC""",
                (user_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


def load_paper(paper_id: int, user_id: int) -> Optional[str]:
    """Load a paper's markdown content."""
    with _lock:
        conn = _get_conn()
        try:
            row = conn.execute(
                "SELECT paper_markdown FROM papers_generated WHERE id = ? AND user_id = ?",
                (paper_id, user_id),
            ).fetchone()
            return row["paper_markdown"] if row else None
        finally:
            conn.close()


# ── Idea evolution operations ────────────────────────────────────────────────

def save_idea_event(
    user_id: int, idea_title: str, generation: int, event_type: str,
    score_before: float, score_after: float, diff_summary: str,
    snapshot_dict: Dict[str, Any], parent_id: Optional[int] = None,
) -> int:
    """Record an idea evolution event. Returns the row id."""
    sj = json.dumps(snapshot_dict, ensure_ascii=False)
    with _lock:
        conn = _get_conn()
        try:
            cur = conn.execute(
                """INSERT INTO idea_evolution
                   (user_id, idea_title, generation, parent_id, event_type,
                    score_before, score_after, diff_summary, snapshot_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (user_id, idea_title, generation, parent_id, event_type,
                 score_before, score_after, diff_summary, sj),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()


def get_idea_history(user_id: int, idea_title: str) -> List[Dict[str, Any]]:
    """Get evolution history for a specific idea."""
    with _lock:
        conn = _get_conn()
        try:
            rows = conn.execute(
                """SELECT id, idea_title, generation, event_type,
                          score_before, score_after, diff_summary, created_at
                   FROM idea_evolution
                   WHERE user_id = ? AND idea_title = ?
                   ORDER BY created_at ASC""",
                (user_id, idea_title),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


def get_all_user_ideas(user_id: int) -> List[Dict[str, Any]]:
    """Extract all ideas across all saved results for analytics. Flattens them."""
    with _lock:
        conn = _get_conn()
        try:
            rows = conn.execute(
                """SELECT id, topic, coverage, results_json, created_at
                   FROM results WHERE user_id = ? ORDER BY created_at DESC""",
                (user_id,),
            ).fetchall()
            all_ideas = []
            for r in rows:
                try:
                    data = json.loads(r["results_json"])
                    for idea in data.get("ideas", []):
                        idea["_run_id"] = r["id"]
                        idea["_topic"] = r["topic"]
                        idea["_run_date"] = r["created_at"]
                        all_ideas.append(idea)
                except (json.JSONDecodeError, KeyError):
                    pass
            return all_ideas
        finally:
            conn.close()


# ── Scientist run operations ─────────────────────────────────────────────

def save_scientist_run(
    user_id: int, topic: str, results_dict: Dict[str, Any],
) -> int:
    """Save a full automated scientist run. Returns the row id."""
    iterations_json = json.dumps(results_dict.get("iterations", []), ensure_ascii=False)
    paper = results_dict.get("final_paper", {}) or {}
    review = results_dict.get("final_review", {}) or {}
    with _lock:
        conn = _get_conn()
        try:
            cur = conn.execute(
                """INSERT INTO scientist_runs
                   (user_id, topic, status, iterations_json, final_paper_md,
                    final_paper_tex, review_json, total_elapsed)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (user_id, topic, results_dict.get("status", "completed"),
                 iterations_json, paper.get("markdown", ""),
                 paper.get("latex", ""), json.dumps(review, ensure_ascii=False),
                 results_dict.get("total_elapsed", 0)),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()


def get_user_scientist_runs(user_id: int) -> List[Dict[str, Any]]:
    """List all scientist runs for a user (newest first)."""
    with _lock:
        conn = _get_conn()
        try:
            rows = conn.execute(
                """SELECT id, topic, status, total_elapsed, created_at
                   FROM scientist_runs WHERE user_id = ? ORDER BY created_at DESC""",
                (user_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


def load_scientist_run(
    run_id: int,
    user_id: int,
    include_latex: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Load a full scientist run.

    `include_latex=False` (default) skips the `final_paper_tex` blob, which is
    only needed for the LaTeX export path. Skipping it avoids reading and
    transferring potentially hundreds of KB per call.
    """
    cols = (
        "id, user_id, topic, status, iterations_json, final_paper_md, "
        "review_json, total_elapsed, created_at"
    )
    if include_latex:
        cols += ", final_paper_tex"
    with _lock:
        conn = _get_conn()
        try:
            row = conn.execute(
                f"""SELECT {cols} FROM scientist_runs
                    WHERE id = ? AND user_id = ?""",
                (run_id, user_id),
            ).fetchone()
            if row is None:
                return None
            d = dict(row)
            if d.get("iterations_json"):
                d["iterations"] = json.loads(d["iterations_json"])
            if d.get("review_json"):
                d["review"] = json.loads(d["review_json"])
            return d
        finally:
            conn.close()


def load_scientist_run_summary(
    run_id: int, user_id: int,
) -> Optional[Dict[str, Any]]:
    """
    Lightweight variant: returns only the small metadata columns
    (no iterations_json / final_paper_md / final_paper_tex / review_json).
    Use this for status checks, listing details, or progress headers.
    """
    with _lock:
        conn = _get_conn()
        try:
            row = conn.execute(
                """SELECT id, user_id, topic, status, total_elapsed, created_at
                   FROM scientist_runs WHERE id = ? AND user_id = ?""",
                (run_id, user_id),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()


# ── Bookmark operations ──────────────────────────────────────────────────────

def bookmark_idea(
    user_id: int, idea_title: str, idea_dict: Dict[str, Any],
    note: str = "", tags: str = "", collection: str = "default", rating: int = 0,
) -> int:
    """Bookmark an idea with optional note, tags, and collection. Returns bookmark id."""
    with _lock:
        conn = _get_conn()
        try:
            cur = conn.execute(
                """INSERT INTO bookmarks (user_id, idea_title, idea_json, note, tags, collection, rating)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (user_id, idea_title, json.dumps(idea_dict, ensure_ascii=False, default=str),
                 note, tags, collection, rating),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()


def get_bookmarks(user_id: int, collection: str = None) -> List[Dict[str, Any]]:
    """Get all bookmarks for a user, optionally filtered by collection."""
    with _lock:
        conn = _get_conn()
        try:
            if collection:
                rows = conn.execute(
                    """SELECT id, user_id, idea_title, idea_json, note, tags,
                              collection, rating, created_at
                       FROM bookmarks WHERE user_id = ? AND collection = ?
                       ORDER BY created_at DESC""",
                    (user_id, collection),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT id, user_id, idea_title, idea_json, note, tags,
                              collection, rating, created_at
                       FROM bookmarks WHERE user_id = ?
                       ORDER BY created_at DESC""",
                    (user_id,),
                ).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                try:
                    d["idea"] = json.loads(d.get("idea_json", "{}"))
                except Exception:
                    d["idea"] = {}
                result.append(d)
            return result
        finally:
            conn.close()


def update_bookmark(bookmark_id: int, user_id: int,
                    note: str = None, tags: str = None,
                    collection: str = None, rating: int = None) -> bool:
    """Update a bookmark's note, tags, collection, or rating."""
    updates = []
    params = []
    if note is not None:
        updates.append("note = ?")
        params.append(note)
    if tags is not None:
        updates.append("tags = ?")
        params.append(tags)
    if collection is not None:
        updates.append("collection = ?")
        params.append(collection)
    if rating is not None:
        updates.append("rating = ?")
        params.append(rating)
    if not updates:
        return False
    params.extend([bookmark_id, user_id])
    with _lock:
        conn = _get_conn()
        try:
            conn.execute(
                f"UPDATE bookmarks SET {', '.join(updates)} WHERE id = ? AND user_id = ?",
                params,
            )
            conn.commit()
            return True
        finally:
            conn.close()


def delete_bookmark(bookmark_id: int, user_id: int) -> bool:
    """Delete a bookmark."""
    with _lock:
        conn = _get_conn()
        try:
            conn.execute("DELETE FROM bookmarks WHERE id = ? AND user_id = ?", (bookmark_id, user_id))
            conn.commit()
            return True
        finally:
            conn.close()


def get_bookmark_collections(user_id: int) -> List[str]:
    """Get all unique collection names for a user."""
    with _lock:
        conn = _get_conn()
        try:
            rows = conn.execute(
                "SELECT DISTINCT collection FROM bookmarks WHERE user_id = ? ORDER BY collection",
                (user_id,),
            ).fetchall()
            return [r["collection"] for r in rows]
        finally:
            conn.close()


# ── Share token operations ───────────────────────────────────────────────────

def create_share_token(user_id: int, idea_dict: Dict[str, Any], topic: str = "") -> str:
    """Create a public share token for an idea. Returns the token string."""
    import secrets
    token = secrets.token_urlsafe(12)
    with _lock:
        conn = _get_conn()
        try:
            conn.execute(
                """INSERT INTO share_tokens (token, user_id, idea_json, topic)
                   VALUES (?, ?, ?, ?)""",
                (token, user_id, json.dumps(idea_dict, default=str), topic),
            )
            conn.commit()
            return token
        finally:
            conn.close()


def get_shared_idea(token: str) -> Optional[Dict[str, Any]]:
    """Get a shared idea by its public token. Increments view count."""
    with _lock:
        conn = _get_conn()
        try:
            row = conn.execute(
                """SELECT id, token, user_id, idea_json, topic, views, likes, created_at
                   FROM share_tokens WHERE token = ?""",
                (token,),
            ).fetchone()
            if not row:
                return None
            conn.execute(
                "UPDATE share_tokens SET views = views + 1 WHERE token = ?", (token,),
            )
            conn.commit()
            d = dict(row)
            d["idea"] = json.loads(d.get("idea_json", "{}"))
            return d
        finally:
            conn.close()


def like_shared_idea(token: str) -> int:
    """Like a shared idea. Returns new like count."""
    with _lock:
        conn = _get_conn()
        try:
            conn.execute(
                "UPDATE share_tokens SET likes = likes + 1 WHERE token = ?", (token,),
            )
            conn.commit()
            row = conn.execute(
                "SELECT likes FROM share_tokens WHERE token = ?", (token,),
            ).fetchone()
            return row["likes"] if row else 0
        finally:
            conn.close()


def get_top_shared_ideas(limit: int = 20) -> List[Dict[str, Any]]:
    """Get top shared ideas by views (for public leaderboard)."""
    with _lock:
        conn = _get_conn()
        try:
            rows = conn.execute(
                """SELECT token, idea_json, topic, views, likes, created_at
                   FROM share_tokens ORDER BY (views + likes * 3) DESC LIMIT ?""",
                (limit,),
            ).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                try:
                    d["idea"] = json.loads(d.get("idea_json", "{}"))
                except Exception:
                    d["idea"] = {}
                result.append(d)
            return result
        finally:
            conn.close()


def get_top_shared_metadata(limit: int = 20) -> List[Dict[str, Any]]:
    """
    Lightweight variant of get_top_shared_ideas that skips the idea_json blob
    and its JSON parse step. Returns just token, topic, views, likes, created_at.
    Use this for leaderboard counts / stats where the full idea is not needed.
    """
    with _lock:
        conn = _get_conn()
        try:
            rows = conn.execute(
                """SELECT token, topic, views, likes, created_at
                   FROM share_tokens ORDER BY (views + likes * 3) DESC LIMIT ?""",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


# ── Email subscriber operations ──────────────────────────────────────────────

def add_email_subscriber(email: str, preferences: str = "") -> bool:
    """Add an email to the newsletter. Returns True if new, False if already exists."""
    try:
        with _lock:
            conn = _get_conn()
            try:
                conn.execute(
                    "INSERT INTO email_subscribers (email, preferences) VALUES (?, ?)",
                    (email.strip().lower(), preferences),
                )
                conn.commit()
                return True
            finally:
                conn.close()
    except sqlite3.IntegrityError:
        return False


def unsubscribe_email(email: str) -> bool:
    """Unsubscribe an email address."""
    with _lock:
        conn = _get_conn()
        try:
            conn.execute(
                "UPDATE email_subscribers SET subscribed = 0 WHERE email = ?",
                (email.strip().lower(),),
            )
            conn.commit()
            return True
        finally:
            conn.close()


def get_active_subscribers() -> List[Dict[str, Any]]:
    with _lock:
        conn = _get_conn()
        try:
            rows = conn.execute(
                """SELECT id, email, subscribed, preferences, created_at, last_sent
                   FROM email_subscribers WHERE subscribed = 1""",
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


# ── Subscription operations ──────────────────────────────────────────────────

def get_user_subscription(user_id: int) -> Dict[str, Any]:
    """Get user's subscription info. Creates free tier if none exists."""
    with _lock:
        conn = _get_conn()
        try:
            row = conn.execute(
                """SELECT id, user_id, tier, stripe_customer_id, stripe_subscription_id,
                          status, current_period_end, runs_this_month, created_at, updated_at
                   FROM subscriptions WHERE user_id = ?""", (user_id,),
            ).fetchone()
            if row:
                return dict(row)
            # Create free tier
            conn.execute(
                "INSERT INTO subscriptions (user_id, tier) VALUES (?, 'free')",
                (user_id,),
            )
            conn.commit()
            row = conn.execute(
                """SELECT id, user_id, tier, stripe_customer_id, stripe_subscription_id,
                          status, current_period_end, runs_this_month, created_at, updated_at
                   FROM subscriptions WHERE user_id = ?""", (user_id,),
            ).fetchone()
            return dict(row) if row else {"tier": "free", "runs_this_month": 0}
        finally:
            conn.close()


def update_subscription(user_id: int, tier: str = None,
                        stripe_customer_id: str = None,
                        stripe_subscription_id: str = None,
                        status: str = None,
                        current_period_end: str = None) -> None:
    updates = []
    params = []
    if tier is not None:
        updates.append("tier = ?"); params.append(tier)
    if stripe_customer_id is not None:
        updates.append("stripe_customer_id = ?"); params.append(stripe_customer_id)
    if stripe_subscription_id is not None:
        updates.append("stripe_subscription_id = ?"); params.append(stripe_subscription_id)
    if status is not None:
        updates.append("status = ?"); params.append(status)
    if current_period_end is not None:
        updates.append("current_period_end = ?"); params.append(current_period_end)
    if not updates:
        return
    updates.append("updated_at = datetime('now')")
    params.append(user_id)
    with _lock:
        conn = _get_conn()
        try:
            conn.execute(
                f"UPDATE subscriptions SET {', '.join(updates)} WHERE user_id = ?",
                params,
            )
            conn.commit()
        finally:
            conn.close()


def increment_run_count(user_id: int) -> int:
    """Increment user's monthly run count. Returns new count."""
    with _lock:
        conn = _get_conn()
        try:
            # Ensure subscription exists
            conn.execute(
                "INSERT OR IGNORE INTO subscriptions (user_id, tier) VALUES (?, 'free')",
                (user_id,),
            )
            conn.execute(
                "UPDATE subscriptions SET runs_this_month = runs_this_month + 1 WHERE user_id = ?",
                (user_id,),
            )
            conn.commit()
            row = conn.execute(
                "SELECT runs_this_month FROM subscriptions WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            return row["runs_this_month"] if row else 0
        finally:
            conn.close()


def search_bookmarks(user_id: int, query: str, limit: int = 500) -> List[Dict[str, Any]]:
    """Search bookmarks by title, note, or tags. Capped at `limit` rows for safety."""
    with _lock:
        conn = _get_conn()
        try:
            # Pre-compute the LIKE pattern once instead of three f-strings
            pattern = f"%{query}%"
            rows = conn.execute(
                """SELECT id, user_id, idea_title, idea_json, note, tags,
                          collection, rating, created_at
                   FROM bookmarks WHERE user_id = ?
                   AND (idea_title LIKE ? OR note LIKE ? OR tags LIKE ?)
                   ORDER BY created_at DESC LIMIT ?""",
                (user_id, pattern, pattern, pattern, limit),
            ).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                try:
                    d["idea"] = json.loads(d.get("idea_json", "{}"))
                except Exception:
                    d["idea"] = {}
                result.append(d)
            return result
        finally:
            conn.close()
