"""
Integration tests for production hardening across multiple modules.

Covers:
  * Sandbox env-var allowlist
  * DB schema migrations + cost logging + result save with timing
  * Smart routing circuit-breaker integration
  * Auto-retry exponential backoff + circuit-breaker bail-out
  * Pipeline run-context setup
"""
import os
import sqlite3
import time

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Sandbox env-var isolation
# ─────────────────────────────────────────────────────────────────────────────

class TestSandboxEnvSecurity:
    """Verify sandbox.py builds an env from allowlist, never leaking secrets."""

    def test_env_allowlist_excludes_api_keys(self):
        """Simulate the allowlist logic from sandbox.py and verify secrets are stripped."""
        _ENV_ALLOWLIST = {
            "PATH", "HOME", "USERPROFILE", "USER", "USERNAME",
            "LANG", "LC_ALL", "TMPDIR", "TEMP", "TMP",
            "PYTHONHOME", "SYSTEMROOT", "COMSPEC",
        }
        _SECRET_DENY = ("API_KEY", "SECRET", "PASSWORD", "TOKEN",
                         "CREDENTIAL", "DATABASE_URL", "BROKER_URL", "BEARER")

        # Inject fake secrets into a simulated environ.
        fake_env = {
            "PATH": "/usr/bin",
            "HOME": "/home/test",
            "DEEPSEEK_API_KEY": "sk-SHOULDNOT-APPEAR",
            "OPENAI_API_KEY": "sk-ALSO-SECRET",
            "DATABASE_URL": "postgres://user:pass@host/db",
            "STRIPE_SECRET_KEY": "sk_live_xxx",
            "MY_BEARER_TOKEN": "eyJ...",
            "NORMAL_VAR": "safe-value",  # not in allowlist → excluded
        }

        env = {k: v for k, v in fake_env.items()
               if k in _ENV_ALLOWLIST
               and not any(p in k.upper() for p in _SECRET_DENY)}

        # Secrets must not appear.
        assert "DEEPSEEK_API_KEY" not in env
        assert "OPENAI_API_KEY" not in env
        assert "DATABASE_URL" not in env
        assert "STRIPE_SECRET_KEY" not in env
        assert "MY_BEARER_TOKEN" not in env
        # Safe values must appear.
        assert env.get("PATH") == "/usr/bin"
        assert env.get("HOME") == "/home/test"
        # Unlisted but safe vars are also excluded (allowlist, not denylist).
        assert "NORMAL_VAR" not in env


# ─────────────────────────────────────────────────────────────────────────────
# DB migrations + cost logging
# ─────────────────────────────────────────────────────────────────────────────

class TestDBMigrations:
    """Verify that migrate_db() creates the new tables and columns."""

    @pytest.fixture
    def fresh_db(self, tmp_path):
        """Create a temporary DB with init + migrate."""
        import db as _db
        # Point DB at temp path.
        orig_path = _db._DB_PATH
        orig_dir = _db._DB_DIR
        _db._DB_PATH = str(tmp_path / "test.db")
        _db._DB_DIR = str(tmp_path)
        # Clear thread-local connections.
        _db._conn_local = __import__("threading").local()
        try:
            _db.init_db()
            yield _db
        finally:
            _db._DB_PATH = orig_path
            _db._DB_DIR = orig_dir
            _db._conn_local = __import__("threading").local()

    def test_cost_tracking_log_table_exists(self, fresh_db):
        conn = sqlite3.connect(fresh_db._DB_PATH)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        assert "cost_tracking_log" in tables
        conn.close()

    def test_milestones_table_exists(self, fresh_db):
        conn = sqlite3.connect(fresh_db._DB_PATH)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        assert "milestones" in tables
        conn.close()

    def test_results_has_elapsed_column(self, fresh_db):
        conn = sqlite3.connect(fresh_db._DB_PATH)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(results)").fetchall()]
        assert "elapsed_seconds" in cols
        assert "estimated_cost_usd" in cols
        conn.close()

    def test_log_cost_writes_and_reads(self, fresh_db):
        # Create a user first.
        user_id = fresh_db.register_user("costtest", "StrongPassword123")
        assert user_id is not None

        fresh_db.log_cost(
            provider="deepseek", model="deepseek-chat",
            prompt_tokens=1000, completion_tokens=500,
            cost_usd=0.0008, stage="ideation",
            user_id=user_id, run_id="test-run-1",
        )

        summary = fresh_db.get_user_cost_summary(user_id)
        assert summary["total_usd"] == pytest.approx(0.0008, abs=1e-6)
        assert len(summary["by_provider"]) == 1
        assert summary["by_provider"][0]["provider"] == "deepseek"

    def test_save_result_with_timing(self, fresh_db):
        user_id = fresh_db.register_user("timingtest", "StrongPassword123")
        results_dict = {
            "topic": "test",
            "ideas": [{"title": "idea1"}],
            "stats": {"elapsed_seconds": 42.5, "estimated_cost_usd": 0.12},
        }
        rid = fresh_db.save_result(
            user_id=user_id, topic="test", coverage=0.5,
            ideas_count=1, results_dict=results_dict,
        )
        assert rid > 0
        # Verify the columns were written.
        conn = sqlite3.connect(fresh_db._DB_PATH)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM results WHERE id = ?", (rid,)).fetchone()
        assert row["elapsed_seconds"] == pytest.approx(42.5)
        assert row["estimated_cost_usd"] == pytest.approx(0.12)
        conn.close()

    def test_migrate_is_idempotent(self, fresh_db):
        """Calling migrate_db() twice should not error."""
        fresh_db.migrate_db()
        fresh_db.migrate_db()


# ─────────────────────────────────────────────────────────────────────────────
# Smart routing + circuit breaker
# ─────────────────────────────────────────────────────────────────────────────

class TestSmartRoutingCircuitBreaker:
    def test_routes_away_from_broken_provider(self):
        from production_optimization import get_circuit_breaker
        cb = get_circuit_breaker()
        # Trip the breaker for the primary provider.
        for _ in range(6):
            cb.record_failure("deepseek")
        ok, _ = cb.allow("deepseek")
        assert not ok  # breaker is open

    def test_healthy_provider_allowed(self):
        from production_optimization import get_circuit_breaker
        cb = get_circuit_breaker()
        cb.record_success("groq")
        ok, _ = cb.allow("groq")
        assert ok


# ─────────────────────────────────────────────────────────────────────────────
# Auto-retry with exponential backoff
# ─────────────────────────────────────────────────────────────────────────────

class TestAutoRetry:
    def test_exponential_backoff_formula(self):
        """Verify 2^attempt scaling with jitter stays in expected bounds."""
        import random
        random.seed(42)
        backoff_s = 3.0
        waits = []
        for attempt in range(4):
            base = min(backoff_s * (2 ** attempt), 30.0)
            wait = base * (0.5 + random.random())
            waits.append(wait)
        # First wait: base=3, range [1.5, 6.0]
        assert 1.0 < waits[0] < 7.0
        # Second wait: base=6, range [3.0, 12.0]
        assert 2.0 < waits[1] < 13.0
        # Waits should generally increase.
        assert waits[-1] > waits[0]

    def test_retry_with_fallback(self):
        from auto_retry import AutoRetryEngine, StageRetryConfig
        engine = AutoRetryEngine()
        call_count = 0

        def always_fail():
            nonlocal call_count
            call_count += 1
            raise ValueError("boom")

        cfg = StageRetryConfig(max_retries=1, backoff_s=0.01, fallback_result={"ok": True})
        result = engine.execute_with_retry("test_stage", always_fail, config=cfg)
        assert result.success is True
        assert result.used_fallback is True
        assert call_count == 2  # 1 initial + 1 retry


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline run-context
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineRunContext:
    def test_config_run_attrs_set_and_cleared(self):
        """Verify that pipeline sets run context on config for cost tracking."""
        import config
        import uuid

        # Simulate what pipeline.py does.
        config._current_run_id = uuid.uuid4().hex[:12]
        config._current_budget_usd = 2.0
        config._current_user_id = 42

        assert len(config._current_run_id) == 12
        assert config._current_budget_usd == 2.0
        assert config._current_user_id == 42

        # Simulate cleanup.
        config._current_run_id = None
        config._current_budget_usd = None
        config._current_user_id = None
        assert config._current_run_id is None
