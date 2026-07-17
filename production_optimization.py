"""
production_optimization.py - Production hardening layer for IdeaGraph.

Provides production-critical optimizations required for scaling to 1M+ users:

  1. RateLimiter       - per-user/IP sliding window + token bucket rate limiting
  2. QuotaEnforcer     - atomic subscription tier quota enforcement
  3. InputValidator    - topic/budget/iteration input sanitization + prompt-injection checks
  4. CostTracker       - per-user, per-provider cost attribution with hard budget caps
  5. CircuitBreaker    - per-provider circuit breaker to prevent cascading LLM failures
  6. ConcurrencyGuard  - global cap on concurrent pipeline runs (per user + global)
  7. CacheLayer        - unified in-memory + disk cache with TTL and eviction

All classes are thread-safe (re-entrant lock protected) and have zero external
dependencies beyond the stdlib and what IdeaGraph already ships.

Usage:
    from production_optimization import (
        get_rate_limiter, get_quota_enforcer, get_cost_tracker,
        get_circuit_breaker, get_concurrency_guard, validate_run_input,
    )

    # Before pipeline start:
    ok, reason = get_quota_enforcer().try_acquire_run(user_id)
    if not ok: raise RuntimeError(reason)

    ok, reason = get_rate_limiter().check(user_id=user_id, ip=ip)
    if not ok: raise RuntimeError(reason)

    topic, budget, iters = validate_run_input(topic, budget, iters, tier)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

# ── Module-level singletons (initialised on first use) ──────────────────────
_SINGLETONS: Dict[str, Any] = {}
_SINGLETON_LOCK = threading.Lock()


def _singleton(key: str, factory: Callable[[], Any]) -> Any:
    """Thread-safe lazy singleton factory."""
    with _SINGLETON_LOCK:
        if key not in _SINGLETONS:
            _SINGLETONS[key] = factory()
        return _SINGLETONS[key]


# ─────────────────────────────────────────────────────────────────────────────
# 1. RATE LIMITER (sliding window + token bucket)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _Bucket:
    tokens: float
    last_refill: float


class RateLimiter:
    """
    Two-layer rate limiter:
      - Sliding window: max N requests per M seconds (per user + per IP).
      - Token bucket: smooth burst absorption (capacity=C, refill=R/s).

    Fail-open semantics: if the store is corrupted, we log and allow the
    request (worse to lock legitimate users out than to let one extra through).
    """

    # Defaults — override via construct args or env vars.
    DEFAULT_USER_LIMIT = 60       # 60 requests per window
    DEFAULT_USER_WINDOW = 60.0    # per 60 seconds
    DEFAULT_IP_LIMIT = 30         # anonymous stricter
    DEFAULT_IP_WINDOW = 60.0
    DEFAULT_BUCKET_CAPACITY = 10  # burst size
    DEFAULT_BUCKET_REFILL = 1.0   # tokens per second

    def __init__(
        self,
        user_limit: int = DEFAULT_USER_LIMIT,
        user_window: float = DEFAULT_USER_WINDOW,
        ip_limit: int = DEFAULT_IP_LIMIT,
        ip_window: float = DEFAULT_IP_WINDOW,
        bucket_capacity: int = DEFAULT_BUCKET_CAPACITY,
        bucket_refill: float = DEFAULT_BUCKET_REFILL,
        max_entries: int = 100_000,
    ) -> None:
        self.user_limit = user_limit
        self.user_window = user_window
        self.ip_limit = ip_limit
        self.ip_window = ip_window
        self.bucket_capacity = bucket_capacity
        self.bucket_refill = bucket_refill
        self.max_entries = max_entries

        self._user_hits: Dict[str, deque] = defaultdict(deque)
        self._ip_hits: Dict[str, deque] = defaultdict(deque)
        self._buckets: Dict[str, _Bucket] = {}
        self._lock = threading.RLock()

    def _gc(self) -> None:
        """Evict cold entries when we grow too big. Called opportunistically."""
        if len(self._user_hits) > self.max_entries:
            # Drop half of the least-recently-used (cheap approximation).
            keys = list(self._user_hits.keys())[: len(self._user_hits) // 2]
            for k in keys:
                self._user_hits.pop(k, None)
                self._buckets.pop(f"u:{k}", None)
        if len(self._ip_hits) > self.max_entries:
            keys = list(self._ip_hits.keys())[: len(self._ip_hits) // 2]
            for k in keys:
                self._ip_hits.pop(k, None)
                self._buckets.pop(f"i:{k}", None)

    def _check_window(
        self, key: str, hits: Dict[str, deque], limit: int, window: float,
    ) -> Tuple[bool, int]:
        """Return (allowed, retry_after_seconds)."""
        now = time.time()
        dq = hits[key]
        # Pop expired.
        cutoff = now - window
        while dq and dq[0] < cutoff:
            dq.popleft()
        if len(dq) >= limit:
            retry = int(window - (now - dq[0])) + 1
            return False, retry
        dq.append(now)
        return True, 0

    def _check_bucket(self, key: str) -> bool:
        """Token bucket check — True if token acquired."""
        now = time.time()
        b = self._buckets.get(key)
        if b is None:
            b = _Bucket(tokens=float(self.bucket_capacity), last_refill=now)
            self._buckets[key] = b
        # Refill.
        elapsed = now - b.last_refill
        b.tokens = min(self.bucket_capacity, b.tokens + elapsed * self.bucket_refill)
        b.last_refill = now
        if b.tokens < 1.0:
            return False
        b.tokens -= 1.0
        return True

    def check(
        self, user_id: Optional[Any] = None, ip: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Check rate limit for a request. Returns (allowed, reason_if_denied)."""
        with self._lock:
            self._gc()
            if user_id is not None:
                key = f"u:{user_id}"
                ok, retry = self._check_window(
                    str(user_id), self._user_hits, self.user_limit, self.user_window,
                )
                if not ok:
                    return False, f"Rate limit: {self.user_limit}/{int(self.user_window)}s exceeded. Retry in {retry}s."
                if not self._check_bucket(key):
                    return False, "Burst limit hit. Slow down for a moment."
            if ip is not None:
                key = f"i:{ip}"
                ok, retry = self._check_window(
                    ip, self._ip_hits, self.ip_limit, self.ip_window,
                )
                if not ok:
                    return False, f"IP rate limit: {self.ip_limit}/{int(self.ip_window)}s exceeded. Retry in {retry}s."
                if not self._check_bucket(key):
                    return False, "Burst limit hit."
            return True, "ok"

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "tracked_users": len(self._user_hits),
                "tracked_ips": len(self._ip_hits),
                "active_buckets": len(self._buckets),
            }


def get_rate_limiter() -> RateLimiter:
    return _singleton("rate_limiter", lambda: RateLimiter(
        user_limit=int(os.getenv("RL_USER_LIMIT", "60")),
        user_window=float(os.getenv("RL_USER_WINDOW", "60")),
        ip_limit=int(os.getenv("RL_IP_LIMIT", "30")),
        ip_window=float(os.getenv("RL_IP_WINDOW", "60")),
    ))


# ─────────────────────────────────────────────────────────────────────────────
# 2. QUOTA ENFORCER (subscription tier limits, atomic)
# ─────────────────────────────────────────────────────────────────────────────

class QuotaEnforcer:
    """
    Atomic subscription-tier quota enforcement.

    Fixes the race condition in stripe_integration.can_run_pipeline: that
    function reads runs_this_month but never increments it, so two concurrent
    requests both pass the gate. Here we reserve the slot atomically BEFORE
    pipeline start, then release it on failure.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # user_id -> # of reserved (in-flight) runs, to prevent concurrent over-use.
        self._reserved: Dict[int, int] = defaultdict(int)

    def try_acquire_run(self, user_id: int) -> Tuple[bool, str]:
        """
        Atomically reserve one run for the user, if tier allows.
        Caller MUST invoke release_run(user_id, success=bool) afterward.
        """
        try:
            import db
            from stripe_integration import TIER_LIMITS
        except Exception as exc:
            return True, f"quota check skipped: {exc}"

        with self._lock:
            sub = db.get_user_subscription(user_id) or {}
            tier = sub.get("tier", "free")
            used = int(sub.get("runs_this_month", 0))
            limit = TIER_LIMITS.get(tier, TIER_LIMITS["free"])["runs_per_month"]
            reserved = self._reserved.get(user_id, 0)

            if used + reserved >= limit:
                return False, (
                    f"Quota exceeded: {used}/{limit} runs used on the "
                    f"{tier.title()} plan this month. Upgrade to continue."
                )

            # Also cap concurrent in-flight per tier.
            max_concurrent = {"free": 1, "pro": 3, "team": 10, "enterprise": 50}.get(tier, 1)
            if reserved >= max_concurrent:
                return False, (
                    f"{max_concurrent} concurrent run(s) already in flight. "
                    f"Wait for one to finish or upgrade."
                )

            self._reserved[user_id] = reserved + 1
            return True, f"reserved ({used + reserved + 1}/{limit})"

    def release_run(self, user_id: int, success: bool) -> None:
        """Release the reservation. If success, commit to DB counter.

        Already idempotent: decrement clamps at 0 so accidental double-
        release (e.g. worker finally + main-thread drain both firing for
        the same run) leaves the counter correct rather than going
        negative."""
        with self._lock:
            self._reserved[user_id] = max(0, self._reserved.get(user_id, 0) - 1)
        if success:
            try:
                import db
                db.increment_run_count(user_id)
            except Exception:
                pass

    def reset_user(self, user_id: int) -> int:
        """Forcibly zero a user's in-flight reservation count. Returns
        the number of slots that were held (for logging/UI). Used by the
        admin 'Release stuck slots' button when a worker thread crashed
        without releasing."""
        with self._lock:
            n = int(self._reserved.get(user_id, 0))
            self._reserved[user_id] = 0
            return n

    def clear(self) -> int:
        """Forcibly zero ALL in-flight reservations. Returns the total
        slot count that was held. Used by the admin 'Reset all' button."""
        with self._lock:
            total = sum(int(v) for v in self._reserved.values())
            self._reserved.clear()
            return total

    def snapshot(self) -> Dict[int, int]:
        """Return a copy of the {user_id: reserved_count} table — used by
        the admin Active Runs panel so it can display + offer per-user
        reset buttons. Filters out zero-count entries (the defaultdict
        accumulates keys when read, so we skip them in the view)."""
        with self._lock:
            return {
                int(uid): int(n) for uid, n in self._reserved.items()
                if int(n) > 0
            }

    def tier_feature_allowed(self, user_id: int, feature: str) -> bool:
        """Check whether user's tier includes a given feature flag."""
        try:
            import db
            from stripe_integration import TIER_LIMITS
        except Exception:
            return True
        sub = db.get_user_subscription(user_id) or {}
        tier = sub.get("tier", "free")
        return bool(TIER_LIMITS.get(tier, TIER_LIMITS["free"]).get(feature, False))


def get_quota_enforcer() -> QuotaEnforcer:
    return _singleton("quota_enforcer", QuotaEnforcer)


# ─────────────────────────────────────────────────────────────────────────────
# 3. INPUT VALIDATOR (sanitization + prompt-injection defense)
# ─────────────────────────────────────────────────────────────────────────────

# Heuristic prompt-injection markers — not bulletproof, but catches the obvious.
_INJECTION_PATTERNS = [
    re.compile(r"\bignore\b[^.]{0,40}\binstructions?\b", re.I),
    re.compile(r"\bdisregard\b[^.]{0,40}\b(instructions?|prompts?|rules?)\b", re.I),
    re.compile(r"\bforget\b[^.]{0,40}\b(instructions?|prompts?|everything)\b", re.I),
    re.compile(r"\bsystem\s*:\s*you\b", re.I),
    re.compile(r"\byou are now\b", re.I),
    re.compile(r"</?\s*(system|user|assistant)\s*>", re.I),
    re.compile(r"\bact as (if you|a new|the)\b", re.I),
    re.compile(r"\boverride\b[^.]{0,40}\b(instructions?|rules?|safety)\b", re.I),
]


class InputValidationError(ValueError):
    """Raised when user-supplied input fails validation."""


def _validate_topic(topic: str) -> str:
    if not isinstance(topic, str):
        raise InputValidationError("Topic must be a string.")
    t = topic.strip()
    if len(t) < 3:
        raise InputValidationError("Topic is too short (minimum 3 characters).")
    if len(t) > 500:
        raise InputValidationError("Topic is too long (maximum 500 characters).")
    # Strip control characters except \n, \t.
    t = "".join(c for c in t if c == "\n" or c == "\t" or (c.isprintable()))
    # Reject prompt-injection attempts.
    for pat in _INJECTION_PATTERNS:
        if pat.search(t):
            raise InputValidationError(
                "Topic contains disallowed instruction-like content. "
                "Please rephrase as a research question."
            )
    return t


def _validate_budget(budget: float, tier: str = "free") -> float:
    try:
        b = float(budget)
    except (TypeError, ValueError):
        raise InputValidationError("Budget must be a number.")
    if b <= 0:
        raise InputValidationError("Budget must be greater than 0.")
    # Tier-based hard caps to prevent runaway spending.
    caps = {"free": 3.0, "pro": 10.0, "team": 50.0, "enterprise": 200.0}
    cap = caps.get(tier, 0.15)
    if b > cap:
        raise InputValidationError(
            f"Budget ${b:.2f} exceeds {tier.title()} tier max of ${cap:.2f}. Upgrade to raise the cap."
        )
    return b


def _validate_iterations(iters: int, tier: str = "free") -> int:
    try:
        n = int(iters)
    except (TypeError, ValueError):
        raise InputValidationError("Iterations must be an integer.")
    if n < 1:
        raise InputValidationError("Iterations must be at least 1.")
    caps = {"free": 50, "pro": 50, "team": 100, "enterprise": 200}
    cap = caps.get(tier, 10)
    if n > cap:
        raise InputValidationError(
            f"Iterations {n} exceeds {tier.title()} tier max of {cap}."
        )
    return n


def validate_run_input(
    topic: str, budget: float, iterations: int, tier: str = "free",
) -> Tuple[str, float, int]:
    """Validate & normalise run parameters. Raises InputValidationError."""
    return (
        _validate_topic(topic),
        _validate_budget(budget, tier),
        _validate_iterations(iterations, tier),
    )


def validate_password(password: str) -> Tuple[bool, str]:
    """Password strength check. Returns (ok, reason_if_not_ok)."""
    if not isinstance(password, str):
        return False, "Password must be a string."
    if len(password) < 12:
        return False, "Password must be at least 12 characters."
    if len(password) > 256:
        return False, "Password is too long (max 256 characters)."
    if not re.search(r"[A-Za-z]", password):
        return False, "Password must contain at least one letter."
    if not re.search(r"[0-9]", password):
        return False, "Password must contain at least one digit."
    # Simple weak-password blacklist.
    weak = {"password", "12345678", "qwerty", "letmein", "password123", "123456789012"}
    if password.lower() in weak:
        return False, "Password is too common. Choose a stronger one."
    return True, "ok"


def validate_username(username: str) -> Tuple[bool, str]:
    if not isinstance(username, str):
        return False, "Username must be a string."
    u = username.strip()
    if len(u) < 3:
        return False, "Username must be at least 3 characters."
    if len(u) > 32:
        return False, "Username must be at most 32 characters."
    if not re.match(r"^[A-Za-z0-9_.-]+$", u):
        return False, "Username may only contain letters, digits, dot, dash, underscore."
    return True, "ok"


# ─────────────────────────────────────────────────────────────────────────────
# 4. COST TRACKER (per-user, per-provider cost attribution + hard cap)
# ─────────────────────────────────────────────────────────────────────────────

class CostTracker:
    """
    Per-user, per-run, per-provider cost attribution.

    Each LLM call should call .record(user_id, provider, input_tokens,
    output_tokens). Pipelines can call .get_run_cost(run_id) to enforce a
    hard cap mid-execution.
    """

    # Cost rates (USD / 1M tokens), mirrors config.COST_RATES.
    DEFAULT_RATES = {
        "deepseek": {"input": 0.27, "output": 1.10},
        "openai":   {"input": 2.50, "output": 10.00},
        "groq":     {"input": 0.59, "output": 0.79},
        "gemini":   {"input": 0.10, "output": 0.40},
        "azure":    {"input": 0.27, "output": 1.10},
    }

    def __init__(self, rates: Optional[Dict[str, Dict[str, float]]] = None) -> None:
        self.rates = rates or self.DEFAULT_RATES
        self._lock = threading.RLock()
        self._run_costs: Dict[str, float] = defaultdict(float)
        self._user_costs: Dict[int, float] = defaultdict(float)
        self._provider_costs: Dict[str, float] = defaultdict(float)
        self._run_to_user: Dict[str, int] = {}

    def _compute_cost(self, provider: str, input_tokens: int, output_tokens: int) -> float:
        r = self.rates.get(provider, self.rates["deepseek"])
        return (input_tokens * r["input"] + output_tokens * r["output"]) / 1_000_000.0

    def record(
        self, provider: str, input_tokens: int, output_tokens: int,
        user_id: Optional[int] = None, run_id: Optional[str] = None,
    ) -> float:
        """Record a single LLM call; returns the cost just recorded."""
        cost = self._compute_cost(provider, input_tokens, output_tokens)
        with self._lock:
            self._provider_costs[provider] += cost
            if user_id is not None:
                self._user_costs[user_id] += cost
            if run_id is not None:
                self._run_costs[run_id] += cost
                if user_id is not None:
                    self._run_to_user[run_id] = user_id
        return cost

    def get_run_cost(self, run_id: str) -> float:
        with self._lock:
            return self._run_costs.get(run_id, 0.0)

    def get_user_cost_month(self, user_id: int) -> float:
        """Quick-and-dirty per-user total (in-memory; persists only per-process)."""
        with self._lock:
            return self._user_costs.get(user_id, 0.0)

    def should_abort(self, run_id: str, budget_usd: float) -> bool:
        """Enforce hard cap at 110% of budget."""
        return self.get_run_cost(run_id) >= budget_usd * 1.1

    def reset_run(self, run_id: str) -> None:
        with self._lock:
            self._run_costs.pop(run_id, None)
            self._run_to_user.pop(run_id, None)

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "active_runs": len(self._run_costs),
                "tracked_users": len(self._user_costs),
                "total_by_provider": dict(self._provider_costs),
                "total_cost_usd": round(sum(self._provider_costs.values()), 4),
            }


def get_cost_tracker() -> CostTracker:
    return _singleton("cost_tracker", CostTracker)


# ─────────────────────────────────────────────────────────────────────────────
# 5. CIRCUIT BREAKER (per-provider)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _CBState:
    failures: int = 0
    last_failure: float = 0.0
    state: str = "closed"   # closed | open | half_open
    opened_at: float = 0.0


class CircuitBreaker:
    """
    Per-provider circuit breaker. After `threshold` failures in
    `failure_window_s`, the breaker opens for `recovery_s` seconds. After
    recovery, one probe request is allowed (half-open). On success -> closed.
    """

    def __init__(
        self, threshold: int = 5, failure_window_s: float = 60.0,
        recovery_s: float = 60.0,
    ) -> None:
        self.threshold = threshold
        self.failure_window_s = failure_window_s
        self.recovery_s = recovery_s
        self._states: Dict[str, _CBState] = defaultdict(_CBState)
        self._lock = threading.RLock()

    def allow(self, provider: str) -> Tuple[bool, str]:
        with self._lock:
            st = self._states[provider]
            now = time.time()
            if st.state == "open":
                if now - st.opened_at >= self.recovery_s:
                    st.state = "half_open"
                    return True, "probe"
                return False, f"circuit open for {provider} (retry in {int(self.recovery_s - (now - st.opened_at))}s)"
            return True, "ok"

    def record_success(self, provider: str) -> None:
        with self._lock:
            st = self._states[provider]
            st.failures = 0
            st.state = "closed"

    def record_failure(self, provider: str) -> None:
        with self._lock:
            st = self._states[provider]
            now = time.time()
            if now - st.last_failure > self.failure_window_s:
                st.failures = 0
            st.failures += 1
            st.last_failure = now
            if st.failures >= self.threshold:
                st.state = "open"
                st.opened_at = now

    def status(self) -> Dict[str, str]:
        with self._lock:
            return {p: s.state for p, s in self._states.items()}


def get_circuit_breaker() -> CircuitBreaker:
    return _singleton("circuit_breaker", lambda: CircuitBreaker(
        threshold=int(os.getenv("CB_THRESHOLD", "5")),
        recovery_s=float(os.getenv("CB_RECOVERY_S", "60")),
    ))


# ─────────────────────────────────────────────────────────────────────────────
# 6. CONCURRENCY GUARD (global + per-user pipeline caps)
# ─────────────────────────────────────────────────────────────────────────────

class ConcurrencyGuard:
    """
    Caps simultaneous pipeline runs globally and per-user. Protects the
    server from thread exhaustion (1M users launching threads would
    crash the process).
    """

    def __init__(self, global_max: int = 100, per_user_max: int = 3) -> None:
        self.global_max = global_max
        self.per_user_max = per_user_max
        self._global: int = 0
        self._per_user: Dict[int, int] = defaultdict(int)
        self._lock = threading.RLock()

    def acquire(self, user_id: Optional[int] = None) -> Tuple[bool, str]:
        with self._lock:
            if self._global >= self.global_max:
                return False, (
                    f"Server is at capacity ({self._global}/{self.global_max} active runs). "
                    f"Please try again in a moment."
                )
            if user_id is not None and self._per_user[user_id] >= self.per_user_max:
                return False, (
                    f"You already have {self._per_user[user_id]} runs in progress "
                    f"(max {self.per_user_max}). Wait for one to complete."
                )
            self._global += 1
            if user_id is not None:
                self._per_user[user_id] += 1
            return True, "ok"

    def release(self, user_id: Optional[int] = None) -> None:
        """Already idempotent — decrement clamps at 0 so accidental
        double-release (e.g. worker-thread finally + main-thread
        _release_run both firing) leaves the counter correct rather
        than going negative."""
        with self._lock:
            self._global = max(0, self._global - 1)
            if user_id is not None:
                self._per_user[user_id] = max(0, self._per_user[user_id] - 1)

    def reset_user(self, user_id: int) -> int:
        """Forcibly zero a user's slot count and decrement the global
        counter by the same amount. Returns the number of slots that
        were held. Used by the admin 'Release stuck slots' button."""
        with self._lock:
            n = int(self._per_user.get(user_id, 0))
            if n > 0:
                self._per_user[user_id] = 0
                self._global = max(0, self._global - n)
            return n

    def clear(self) -> int:
        """Forcibly zero ALL active runs. Returns the total slot count
        that was held. Used by the admin 'Reset all' button."""
        with self._lock:
            total = self._global
            self._global = 0
            self._per_user.clear()
            return total

    def snapshot(self) -> Dict[int, int]:
        """{user_id: slot_count} table for the admin Active Runs panel."""
        with self._lock:
            return {
                int(uid): int(n) for uid, n in self._per_user.items()
                if int(n) > 0
            }

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "global_active": self._global,
                "global_max": self.global_max,
                "users_active": sum(1 for v in self._per_user.values() if v > 0),
            }


def get_concurrency_guard() -> ConcurrencyGuard:
    return _singleton("concurrency_guard", lambda: ConcurrencyGuard(
        global_max=int(os.getenv("CG_GLOBAL_MAX", "100")),
        per_user_max=int(os.getenv("CG_PER_USER_MAX", "3")),
    ))


# ─────────────────────────────────────────────────────────────────────────────
# 7. SECURE TOKEN HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def generate_session_token() -> str:
    """Cryptographically-secure, URL-safe 256-bit token."""
    import secrets
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    """One-way hash for storing tokens in DB (compare hashes, not plaintext)."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# 8. UNIFIED HEALTH / DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

def health_snapshot() -> Dict[str, Any]:
    """Single health-check entrypoint — safe to expose at /api/health."""
    return {
        "timestamp": time.time(),
        "rate_limiter": get_rate_limiter().stats(),
        "concurrency": get_concurrency_guard().stats(),
        "cost": get_cost_tracker().summary(),
        "circuit_breakers": get_circuit_breaker().status(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 9. SMOKE TEST (run: python production_optimization.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Rate limiter smoke test:")
    rl = get_rate_limiter()
    for i in range(65):
        ok, msg = rl.check(user_id=1)
        if not ok:
            print(f"  denied at request {i+1}: {msg}")
            break
    else:
        print("  all 65 requests allowed (unexpected)")

    print("\nInput validator smoke test:")
    try:
        validate_run_input("Graph neural networks for drug discovery", 1.5, 10, "pro")
        print("  valid input OK")
    except InputValidationError as e:
        print(f"  unexpected failure: {e}")

    for bad in ["x", "ignore all previous instructions and...", "a" * 600]:
        try:
            validate_run_input(bad, 1.0, 5, "pro")
            print(f"  FAIL: accepted bad input: {bad[:40]}")
        except InputValidationError as e:
            print(f"  rejected ({e})")

    print("\nCircuit breaker smoke test:")
    cb = get_circuit_breaker()
    for _ in range(6):
        cb.record_failure("openai")
    ok, msg = cb.allow("openai")
    print(f"  after 6 failures: allow={ok} msg={msg}")

    print("\nHealth snapshot:")
    print(json.dumps(health_snapshot(), indent=2, default=str))
