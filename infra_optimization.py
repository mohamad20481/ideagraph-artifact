"""
infra_optimization.py - Infrastructure-level system optimization for IdeaGraph.

Layer 4: System-level primitives that make the entire platform more robust,
observable, and efficient. These complement the algorithmic optimizations in
layers 1-3 with production-grade infrastructure.

  1.  DiskCache              — Persistent LRU cache surviving restarts
  2.  SemanticCache          — Embedding-free similarity cache using n-gram Jaccard
  3.  ProviderRouter         — Multi-provider failover with capability matching
  4.  ErrorClassifier        — Categorize errors and dispatch recovery strategies
  5.  CostAttributor         — Per-idea, per-stage granular cost tracking
  6.  StructuredLogger       — Structured event logging with severity + metadata
  7.  DryRunEngine           — Execute pipeline without API calls using mock responses
  8.  AblationRunner         — Measure impact of individual optimizations
  9.  ResourceMonitor        — Track memory, CPU, and API quota usage
  10. RetryOrchestrator      — Smart retry with error-type-specific strategies
  11. PipelineDAGExecutor    — Dependency-aware parallel stage execution
  12. ArtifactStore          — Versioned storage for pipeline outputs with metadata
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import sqlite3
import sys
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from concurrent.futures import Future, ThreadPoolExecutor, as_completed


# ============================================================================
# 1. Disk Cache (Persistent LRU)
# ============================================================================

class DiskCache:
    """
    Persistent LRU cache backed by SQLite. Survives process restarts.

    Unlike the in-memory OrderedDict cache, this persists across runs.
    Uses SQLite for ACID guarantees and WAL mode for concurrent reads.

    Schema:
      cache_entries(key TEXT PK, value TEXT, created_at REAL, accessed_at REAL, size_bytes INT)

    Eviction: LRU by accessed_at when total size exceeds max_size_mb.
    """

    def __init__(self, db_path: str = None, max_size_mb: float = 50.0, max_entries: int = 2048):
        self.db_path = db_path or str(Path(__file__).parent / "output" / ".disk_cache.db")
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._init_db()

    def _init_db(self) -> None:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    accessed_at REAL NOT NULL,
                    size_bytes INTEGER NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_accessed ON cache_entries(accessed_at)")
            try:
                conn.execute("PRAGMA journal_mode=WAL")
            except Exception:
                pass

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=5)

    def get(self, key: str) -> Optional[str]:
        """Retrieve a cached value, updating access time."""
        with self._lock:
            try:
                with self._connect() as conn:
                    row = conn.execute("SELECT value FROM cache_entries WHERE key = ?", (key,)).fetchone()
                    if row:
                        conn.execute("UPDATE cache_entries SET accessed_at = ? WHERE key = ?", (time.time(), key))
                        self._hits += 1
                        return row[0]
                    self._misses += 1
                    return None
            except Exception:
                self._misses += 1
                return None

    def put(self, key: str, value: str) -> None:
        """Store a value, evicting LRU entries if needed."""
        with self._lock:
            try:
                size = len(value.encode("utf-8"))
                now = time.time()
                with self._connect() as conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO cache_entries (key, value, created_at, accessed_at, size_bytes) VALUES (?, ?, ?, ?, ?)",
                        (key, value, now, now, size),
                    )
                    self._evict_if_needed(conn)
            except Exception:
                pass

    def _evict_if_needed(self, conn: sqlite3.Connection) -> None:
        """Evict LRU entries if size or count exceeds limits."""
        # Check count
        count = conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()[0]
        if count > self.max_entries:
            excess = count - self.max_entries
            conn.execute(
                "DELETE FROM cache_entries WHERE key IN (SELECT key FROM cache_entries ORDER BY accessed_at ASC LIMIT ?)",
                (excess,),
            )

        # Check size
        total_size = conn.execute("SELECT COALESCE(SUM(size_bytes), 0) FROM cache_entries").fetchone()[0]
        while total_size > self.max_size_bytes:
            row = conn.execute("SELECT key, size_bytes FROM cache_entries ORDER BY accessed_at ASC LIMIT 1").fetchone()
            if not row:
                break
            conn.execute("DELETE FROM cache_entries WHERE key = ?", (row[0],))
            total_size -= row[1]

    def invalidate(self, key: str) -> None:
        with self._lock:
            try:
                with self._connect() as conn:
                    conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
            except Exception:
                pass

    def clear(self) -> None:
        with self._lock:
            try:
                with self._connect() as conn:
                    conn.execute("DELETE FROM cache_entries")
            except Exception:
                pass

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> Dict[str, Any]:
        try:
            with self._connect() as conn:
                count = conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()[0]
                size = conn.execute("SELECT COALESCE(SUM(size_bytes), 0) FROM cache_entries").fetchone()[0]
        except Exception:
            count, size = 0, 0
        return {
            "entries": count,
            "size_mb": round(size / (1024 * 1024), 2),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{self.hit_rate:.1%}",
        }


# ============================================================================
# 2. Semantic Cache (N-gram Jaccard Similarity)
# ============================================================================

class SemanticCache:
    """
    Approximate similarity cache using character n-gram Jaccard similarity.

    Unlike exact-match caching, this finds and reuses responses for
    SIMILAR (not identical) prompts. No embedding model needed.

    Algorithm:
      1. Convert prompt to set of character 4-grams
      2. Compare Jaccard similarity with cached prompts
      3. If similarity > threshold, return cached response
      4. Otherwise, miss — caller must generate fresh response

    O(n×k) per lookup where n = cache size, k = avg prompt length.
    Fast enough for cache sizes < 500 (typical for a pipeline run).
    """

    def __init__(self, similarity_threshold: float = 0.75, max_entries: int = 256, ngram_size: int = 4):
        self.threshold = similarity_threshold
        self.max_entries = max_entries
        self.ngram_size = ngram_size
        self._entries: List[Tuple[Set[str], str, str]] = []  # (ngrams, prompt_hash, response)
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _to_ngrams(self, text: str) -> Set[str]:
        """Convert text to character n-gram set."""
        text = text.lower().strip()
        if len(text) < self.ngram_size:
            return {text}
        return {text[i:i + self.ngram_size] for i in range(len(text) - self.ngram_size + 1)}

    def _jaccard(self, a: Set[str], b: Set[str]) -> float:
        if not a or not b:
            return 0.0
        intersection = len(a & b)
        union = len(a | b)
        return intersection / union if union > 0 else 0.0

    def get(self, prompt: str) -> Optional[str]:
        """Find a semantically similar cached response."""
        query_ngrams = self._to_ngrams(prompt)
        with self._lock:
            best_sim = 0.0
            best_response = None
            for ngrams, _, response in self._entries:
                sim = self._jaccard(query_ngrams, ngrams)
                if sim > best_sim:
                    best_sim = sim
                    best_response = response

            if best_sim >= self.threshold:
                self._hits += 1
                return best_response
            self._misses += 1
            return None

    def put(self, prompt: str, response: str) -> None:
        """Cache a prompt-response pair."""
        ngrams = self._to_ngrams(prompt)
        prompt_hash = hashlib.md5(prompt.encode(), usedforsecurity=False).hexdigest()[:12]
        with self._lock:
            # Check if similar entry already exists
            for i, (existing_ngrams, _, _) in enumerate(self._entries):
                if self._jaccard(ngrams, existing_ngrams) > 0.9:
                    self._entries[i] = (ngrams, prompt_hash, response)
                    return
            # Add new
            if len(self._entries) >= self.max_entries:
                self._entries.pop(0)  # FIFO eviction
            self._entries.append((ngrams, prompt_hash, response))

    def stats(self) -> Dict[str, Any]:
        return {
            "entries": len(self._entries),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{self._hits / max(self._hits + self._misses, 1):.1%}",
            "threshold": self.threshold,
        }


# ============================================================================
# 3. Provider Router (Multi-Provider Failover)
# ============================================================================

@dataclass
class ProviderCapability:
    """Capability profile for an LLM provider."""
    name: str
    model: str
    max_context: int = 128000
    supports_json_mode: bool = True
    cost_per_1k_input: float = 0.001
    cost_per_1k_output: float = 0.004
    avg_latency_s: float = 5.0
    reliability: float = 0.99  # uptime ratio
    priority: int = 0  # lower = preferred


class ProviderRouter:
    """
    Multi-provider failover with intelligent routing.

    Routes LLM calls to the best available provider based on:
      - Task requirements (JSON mode, context length, latency)
      - Provider health (recent error rate)
      - Cost optimization (cheapest provider that meets requirements)
      - Failover chain (if primary fails, try secondary, then tertiary)

    Maintains per-provider health scores updated in real-time.
    """

    def __init__(self, providers: List[ProviderCapability] = None):
        self.providers = providers or [
            ProviderCapability("azure", "DeepSeek-V3.2-Speciale", cost_per_1k_input=0.00027, cost_per_1k_output=0.0011, priority=0, avg_latency_s=5.0),
            ProviderCapability("deepseek", "deepseek-chat", cost_per_1k_input=0.00027, cost_per_1k_output=0.0011, priority=1),
            ProviderCapability("gemini", "gemini-2.0-flash", cost_per_1k_input=0.0001, cost_per_1k_output=0.0004, priority=2, avg_latency_s=3.0),
            ProviderCapability("groq", "llama-3.3-70b-versatile", cost_per_1k_input=0.00059, cost_per_1k_output=0.00079, priority=3, avg_latency_s=2.0),
            ProviderCapability("openai", "gpt-4o", cost_per_1k_input=0.0025, cost_per_1k_output=0.01, priority=3, avg_latency_s=8.0),
        ]
        self._health: Dict[str, deque] = {p.name: deque(maxlen=20) for p in self.providers}
        self._lock = threading.Lock()

    def select_provider(
        self,
        requires_json: bool = False,
        max_cost_per_1k: float = float("inf"),
        max_latency_s: float = float("inf"),
        min_context: int = 0,
    ) -> Optional[ProviderCapability]:
        """Select the best available provider matching requirements."""
        candidates = []
        for p in self.providers:
            if requires_json and not p.supports_json_mode:
                continue
            if p.cost_per_1k_input > max_cost_per_1k:
                continue
            if p.avg_latency_s > max_latency_s:
                continue
            if p.max_context < min_context:
                continue
            health = self._get_health(p.name)
            if health < 0.3:
                continue  # skip unhealthy providers
            candidates.append((p.priority, -health, p.cost_per_1k_input, p))

        if not candidates:
            return None
        candidates.sort()
        return candidates[0][3]

    def get_failover_chain(self, primary: str, n: int = 2) -> List[ProviderCapability]:
        """Get ordered failover providers after primary."""
        chain = []
        for p in sorted(self.providers, key=lambda x: x.priority):
            if p.name != primary and self._get_health(p.name) > 0.3:
                chain.append(p)
                if len(chain) >= n:
                    break
        return chain

    def record_result(self, provider_name: str, success: bool, latency_s: float = 0) -> None:
        """Record a provider call result for health tracking."""
        with self._lock:
            if provider_name not in self._health:
                self._health[provider_name] = deque(maxlen=20)
            self._health[provider_name].append(1.0 if success else 0.0)
            # Update latency estimate
            for p in self.providers:
                if p.name == provider_name and latency_s > 0:
                    p.avg_latency_s = 0.8 * p.avg_latency_s + 0.2 * latency_s

    def _get_health(self, provider_name: str) -> float:
        """Get health score (0-1) for a provider."""
        with self._lock:
            history = self._health.get(provider_name, deque())
            if not history:
                return 1.0
            return sum(history) / len(history)

    def stats(self) -> Dict[str, Any]:
        return {
            p.name: {
                "model": p.model,
                "health": round(self._get_health(p.name), 2),
                "cost_1k_in": p.cost_per_1k_input,
                "latency_s": round(p.avg_latency_s, 1),
                "priority": p.priority,
            }
            for p in self.providers
        }


# ============================================================================
# 4. Error Classifier
# ============================================================================

class ErrorType(Enum):
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    AUTH = "auth"
    INVALID_REQUEST = "invalid_request"
    SERVER_ERROR = "server_error"
    PARSE_ERROR = "parse_error"
    CONTENT_FILTER = "content_filter"
    CONTEXT_LENGTH = "context_length"
    NETWORK = "network"
    UNKNOWN = "unknown"


class ErrorClassifier:
    """
    Categorize API errors and dispatch type-specific recovery strategies.

    Instead of treating all errors the same (exponential backoff), this
    classifies errors and applies the right recovery:
      - rate_limit → backoff + switch provider
      - timeout → retry with shorter max_tokens
      - context_length → truncate prompt + retry
      - auth → alert user, don't retry
      - parse_error → retry with JSON mode
      - content_filter → rephrase + retry
    """

    RECOVERY_STRATEGIES = {
        ErrorType.RATE_LIMIT: {"action": "backoff_and_switch", "max_retries": 3, "backoff_base": 15},
        ErrorType.TIMEOUT: {"action": "reduce_tokens", "max_retries": 2, "token_reduction": 0.5},
        ErrorType.AUTH: {"action": "alert", "max_retries": 0},
        ErrorType.INVALID_REQUEST: {"action": "truncate", "max_retries": 1},
        ErrorType.SERVER_ERROR: {"action": "switch_provider", "max_retries": 2},
        ErrorType.PARSE_ERROR: {"action": "force_json_mode", "max_retries": 2},
        ErrorType.CONTENT_FILTER: {"action": "rephrase", "max_retries": 1},
        ErrorType.CONTEXT_LENGTH: {"action": "truncate", "max_retries": 1, "truncation": 0.6},
        ErrorType.NETWORK: {"action": "retry", "max_retries": 3, "backoff_base": 5},
        ErrorType.UNKNOWN: {"action": "retry", "max_retries": 1},
    }

    def __init__(self):
        self._error_counts: Dict[ErrorType, int] = defaultdict(int)
        self._lock = threading.Lock()

    def classify(self, error: Exception) -> ErrorType:
        """Classify an exception into an ErrorType."""
        err_str = str(error).lower()
        err_type = type(error).__name__.lower()

        if "rate" in err_str or "429" in err_str or "ratelimit" in err_type:
            return ErrorType.RATE_LIMIT
        if "timeout" in err_str or "timed out" in err_str:
            return ErrorType.TIMEOUT
        if "auth" in err_str or "401" in err_str or "403" in err_str or "api key" in err_str:
            return ErrorType.AUTH
        if "context" in err_str and ("length" in err_str or "too long" in err_str or "maximum" in err_str):
            return ErrorType.CONTEXT_LENGTH
        if "content" in err_str and "filter" in err_str:
            return ErrorType.CONTENT_FILTER
        if "json" in err_str or "parse" in err_str or "decode" in err_str:
            return ErrorType.PARSE_ERROR
        if "500" in err_str or "502" in err_str or "503" in err_str or "server" in err_str:
            return ErrorType.SERVER_ERROR
        if "connection" in err_str or "network" in err_str or "dns" in err_str:
            return ErrorType.NETWORK
        if "invalid" in err_str or "400" in err_str:
            return ErrorType.INVALID_REQUEST
        return ErrorType.UNKNOWN

    def get_recovery(self, error_type: ErrorType) -> Dict[str, Any]:
        """Get recovery strategy for an error type."""
        with self._lock:
            self._error_counts[error_type] += 1
        return self.RECOVERY_STRATEGIES.get(error_type, self.RECOVERY_STRATEGIES[ErrorType.UNKNOWN])

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "error_counts": {k.value: v for k, v in self._error_counts.items()},
                "total_errors": sum(self._error_counts.values()),
                "most_common": max(self._error_counts, key=self._error_counts.get).value if self._error_counts else "none",
            }


# ============================================================================
# 5. Cost Attributor
# ============================================================================

class CostAttributor:
    """
    Per-idea, per-stage granular cost tracking.

    Attributes every API call to a specific (idea, stage) pair.
    Enables analysis of which ideas are expensive vs cheap to develop,
    and which stages consume the most budget per idea.
    """

    @dataclass
    class CostEntry:
        idea_id: str
        stage: str
        tokens_in: int
        tokens_out: int
        cost_usd: float
        timestamp: float
        duration_s: float = 0

    def __init__(self):
        self._entries: List["CostAttributor.CostEntry"] = []
        self._current_idea: str = ""
        self._current_stage: str = ""
        self._lock = threading.Lock()

    def set_context(self, idea_id: str = "", stage: str = "") -> None:
        """Set current attribution context."""
        with self._lock:
            if idea_id:
                self._current_idea = idea_id
            if stage:
                self._current_stage = stage

    def record(self, tokens_in: int, tokens_out: int, cost_usd: float, duration_s: float = 0) -> None:
        """Record a cost entry attributed to current context."""
        with self._lock:
            self._entries.append(self.CostEntry(
                idea_id=self._current_idea,
                stage=self._current_stage,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=cost_usd,
                timestamp=time.time(),
                duration_s=duration_s,
            ))

    def cost_by_idea(self) -> Dict[str, float]:
        """Total cost per idea."""
        with self._lock:
            costs: Dict[str, float] = defaultdict(float)
            for e in self._entries:
                if e.idea_id:
                    costs[e.idea_id] += e.cost_usd
        return dict(costs)

    def cost_by_stage(self) -> Dict[str, float]:
        """Total cost per stage."""
        with self._lock:
            costs: Dict[str, float] = defaultdict(float)
            for e in self._entries:
                if e.stage:
                    costs[e.stage] += e.cost_usd
        return dict(costs)

    def most_expensive_idea(self) -> Tuple[str, float]:
        by_idea = self.cost_by_idea()
        if not by_idea:
            return ("", 0)
        return max(by_idea.items(), key=lambda x: x[1])

    def stats(self) -> Dict[str, Any]:
        total = sum(e.cost_usd for e in self._entries)
        return {
            "total_cost_usd": round(total, 4),
            "entries": len(self._entries),
            "by_stage": {k: round(v, 4) for k, v in self.cost_by_stage().items()},
            "by_idea_count": len(self.cost_by_idea()),
            "most_expensive_idea": self.most_expensive_idea(),
        }


# ============================================================================
# 6. Structured Logger
# ============================================================================

class LogLevel(Enum):
    DEBUG = 0
    INFO = 1
    WARN = 2
    ERROR = 3
    CRITICAL = 4


@dataclass
class LogEvent:
    timestamp: float
    level: LogLevel
    stage: str
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0


class StructuredLogger:
    """
    Structured event logging with severity levels and metadata.

    Provides queryable, structured logs instead of ad-hoc print statements.
    Supports filtering by stage, level, and time range.
    Can export to JSON lines for external analysis.
    """

    def __init__(self, min_level: LogLevel = LogLevel.INFO, max_events: int = 5000):
        self.min_level = min_level
        self.max_events = max_events
        self._events: deque = deque(maxlen=max_events)
        self._lock = threading.Lock()
        self._stage_timers: Dict[str, float] = {}

    def log(self, level: LogLevel, stage: str, message: str, **metadata) -> None:
        if level.value < self.min_level.value:
            return
        with self._lock:
            self._events.append(LogEvent(
                timestamp=time.time(), level=level,
                stage=stage, message=message, metadata=metadata,
            ))

    def debug(self, stage: str, msg: str, **kw): self.log(LogLevel.DEBUG, stage, msg, **kw)
    def info(self, stage: str, msg: str, **kw): self.log(LogLevel.INFO, stage, msg, **kw)
    def warn(self, stage: str, msg: str, **kw): self.log(LogLevel.WARN, stage, msg, **kw)
    def error(self, stage: str, msg: str, **kw): self.log(LogLevel.ERROR, stage, msg, **kw)
    def critical(self, stage: str, msg: str, **kw): self.log(LogLevel.CRITICAL, stage, msg, **kw)

    def start_timer(self, stage: str) -> None:
        self._stage_timers[stage] = time.time()

    def stop_timer(self, stage: str) -> float:
        start = self._stage_timers.pop(stage, time.time())
        duration = time.time() - start
        self.info(stage, f"Stage completed in {duration:.1f}s", duration_s=duration)
        return duration

    def query(self, stage: str = None, level: LogLevel = None, last_n: int = 50) -> List[Dict]:
        """Query log events with optional filters."""
        with self._lock:
            events = list(self._events)
        if stage:
            events = [e for e in events if e.stage == stage]
        if level:
            events = [e for e in events if e.level.value >= level.value]
        return [
            {"time": e.timestamp, "level": e.level.value, "stage": e.stage,
             "message": e.message, "metadata": e.metadata}
            for e in events[-last_n:]
        ]

    def export_jsonl(self, path: str) -> int:
        """Export all events to JSON Lines file."""
        with self._lock:
            events = list(self._events)
        with open(path, "w") as f:
            for e in events:
                line = json.dumps({
                    "timestamp": e.timestamp, "level": e.level.name,
                    "stage": e.stage, "message": e.message,
                    "metadata": e.metadata,
                })
                f.write(line + "\n")
        return len(events)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            events = list(self._events)
        by_level = defaultdict(int)
        by_stage = defaultdict(int)
        for e in events:
            by_level[e.level.name] += 1
            by_stage[e.stage] += 1
        return {
            "total_events": len(events),
            "by_level": dict(by_level),
            "by_stage": dict(by_stage),
            "errors": by_level.get("ERROR", 0) + by_level.get("CRITICAL", 0),
        }


# ============================================================================
# 7. Dry Run Engine
# ============================================================================

class DryRunEngine:
    """
    Execute pipeline without API calls using mock/cached responses.

    Essential for:
      - Testing pipeline logic without spending budget
      - Benchmarking optimization techniques
      - Rapid iteration on pipeline structure
      - CI/CD validation

    Modes:
      - MOCK: returns synthetic responses
      - REPLAY: replays cached responses from a previous real run
      - HYBRID: uses cache when available, mock when not
    """

    class Mode(Enum):
        MOCK = "mock"
        REPLAY = "replay"
        HYBRID = "hybrid"

    def __init__(self, mode: "DryRunEngine.Mode" = None, replay_path: str = None):
        self.mode = mode or self.Mode.MOCK
        self.replay_path = replay_path
        self._replay_data: Dict[str, str] = {}
        self._call_count = 0
        self._lock = threading.Lock()

        if self.mode in (self.Mode.REPLAY, self.Mode.HYBRID) and replay_path:
            self._load_replay(replay_path)

    def _load_replay(self, path: str) -> None:
        try:
            with open(path) as f:
                self._replay_data = json.load(f)
        except Exception:
            pass

    def generate_mock_response(self, system: str, user: str, json_mode: bool = False) -> str:
        """Generate a synthetic response for testing."""
        self._call_count += 1
        if json_mode:
            return json.dumps({
                "title": f"Mock Idea {self._call_count}",
                "motivation": "Test motivation for pipeline validation",
                "method": "Mock method using synthetic data generation",
                "hypothesis": "Mock hypothesis for testing",
                "resources": "Standard compute resources",
                "expected_outcome": "Validation that pipeline logic works correctly",
                "risk_assessment": "Low risk - this is a mock response",
                "source_strategy": "A",
                "methodology_type": "supervised_learning",
                "novelty_level": "moderate",
                "score": 0.7,
                "quality": 0.65,
                "issues": [],
                "confidence": 0.8,
            })
        return (
            f"Mock response #{self._call_count} for dry-run testing.\n"
            f"System prompt length: {len(system)} chars\n"
            f"User prompt length: {len(user)} chars\n"
            "This is a synthetic response for pipeline validation."
        )

    def get_response(self, key: str, system: str, user: str, json_mode: bool = False) -> Optional[str]:
        """Get response based on mode."""
        if self.mode == self.Mode.REPLAY:
            return self._replay_data.get(key)
        if self.mode == self.Mode.HYBRID:
            cached = self._replay_data.get(key)
            if cached:
                return cached
        return self.generate_mock_response(system, user, json_mode)

    def save_for_replay(self, path: str, responses: Dict[str, str]) -> None:
        """Save real responses for future replay."""
        with open(path, "w") as f:
            json.dump(responses, f, indent=2)

    def stats(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "calls": self._call_count,
            "replay_entries": len(self._replay_data),
        }


# ============================================================================
# 8. Ablation Runner
# ============================================================================

class AblationRunner:
    """
    Measure impact of individual optimizations via systematic ablation.

    For each optimization:
      1. Run pipeline with optimization ENABLED
      2. Run pipeline with optimization DISABLED
      3. Compare: quality delta, cost delta, time delta
      4. Compute net impact score

    Results guide which optimizations to keep/remove.
    """

    @dataclass
    class AblationResult:
        optimization: str
        enabled_quality: float
        disabled_quality: float
        enabled_cost: float
        disabled_cost: float
        enabled_time: float
        disabled_time: float

        @property
        def quality_delta(self) -> float:
            return self.enabled_quality - self.disabled_quality

        @property
        def cost_delta(self) -> float:
            return self.enabled_cost - self.disabled_cost

        @property
        def net_impact(self) -> float:
            """Positive = optimization helps. Weights: 60% quality, 30% cost savings, 10% speed."""
            q_impact = self.quality_delta
            c_impact = (self.disabled_cost - self.enabled_cost) / max(self.disabled_cost, 0.01)
            t_impact = (self.disabled_time - self.enabled_time) / max(self.disabled_time, 1)
            return 0.6 * q_impact + 0.3 * c_impact + 0.1 * t_impact

    def __init__(self):
        self._results: Dict[str, "AblationRunner.AblationResult"] = {}

    def record(self, optimization: str, enabled_quality: float, disabled_quality: float,
               enabled_cost: float = 0, disabled_cost: float = 0,
               enabled_time: float = 0, disabled_time: float = 0) -> None:
        self._results[optimization] = self.AblationResult(
            optimization=optimization,
            enabled_quality=enabled_quality, disabled_quality=disabled_quality,
            enabled_cost=enabled_cost, disabled_cost=disabled_cost,
            enabled_time=enabled_time, disabled_time=disabled_time,
        )

    def get_rankings(self) -> List[Tuple[str, float]]:
        """Rank optimizations by net impact (highest first)."""
        return sorted(
            [(name, r.net_impact) for name, r in self._results.items()],
            key=lambda x: x[1], reverse=True,
        )

    def recommend_disable(self, threshold: float = -0.05) -> List[str]:
        """Recommend disabling optimizations that hurt performance."""
        return [name for name, r in self._results.items() if r.net_impact < threshold]

    def stats(self) -> Dict[str, Any]:
        return {
            "tested": len(self._results),
            "rankings": self.get_rankings(),
            "recommend_disable": self.recommend_disable(),
        }


# ============================================================================
# 9. Resource Monitor
# ============================================================================

class ResourceMonitor:
    """
    Track memory, thread count, and API quota usage.

    Provides real-time system health signals to prevent OOM, thread
    exhaustion, or quota burnout.
    """

    def __init__(self, memory_limit_mb: float = 2048, max_threads: int = 50):
        self.memory_limit_mb = memory_limit_mb
        self.max_threads = max_threads
        self._snapshots: deque = deque(maxlen=100)
        self._lock = threading.Lock()

    def snapshot(self) -> Dict[str, Any]:
        """Take a resource snapshot."""
        import os
        try:
            import psutil
            process = psutil.Process(os.getpid())
            mem_mb = process.memory_info().rss / (1024 * 1024)
            cpu_pct = process.cpu_percent()
        except ImportError:
            mem_mb = 0
            cpu_pct = 0

        snap = {
            "timestamp": time.time(),
            "memory_mb": round(mem_mb, 1),
            "threads": threading.active_count(),
            "cpu_pct": cpu_pct,
        }
        with self._lock:
            self._snapshots.append(snap)
        return snap

    @property
    def memory_pressure(self) -> bool:
        """Is memory usage above 80% of limit?"""
        with self._lock:
            if not self._snapshots:
                return False
            latest = self._snapshots[-1]
            return latest.get("memory_mb", 0) > self.memory_limit_mb * 0.8

    @property
    def thread_pressure(self) -> bool:
        return threading.active_count() > self.max_threads * 0.8

    def should_gc(self) -> bool:
        """Recommend garbage collection if memory is high."""
        return self.memory_pressure

    def run_gc(self) -> int:
        """Force garbage collection and return objects collected."""
        return gc.collect()

    def stats(self) -> Dict[str, Any]:
        snap = self.snapshot()
        return {
            "current": snap,
            "memory_pressure": self.memory_pressure,
            "thread_pressure": self.thread_pressure,
            "snapshots_recorded": len(self._snapshots),
        }


# ============================================================================
# 10. Retry Orchestrator
# ============================================================================

class RetryOrchestrator:
    """
    Smart retry with error-type-specific strategies.

    Combines ErrorClassifier with provider failover and adaptive backoff.
    Goes beyond simple exponential backoff by choosing the right recovery
    action per error type.
    """

    def __init__(self, error_classifier: ErrorClassifier = None, provider_router: ProviderRouter = None):
        self.classifier = error_classifier or ErrorClassifier()
        self.router = provider_router or ProviderRouter()
        self._retry_history: List[Dict] = []

    def handle_error(self, error: Exception, current_provider: str = "",
                     prompt_length: int = 0) -> Dict[str, Any]:
        """
        Classify error and return recovery action.

        Returns: {
            "action": str,  # "retry", "switch_provider", "truncate", "alert", etc.
            "provider": str or None,  # new provider if switching
            "max_tokens_reduction": float,  # multiply max_tokens by this
            "backoff_seconds": float,
            "should_retry": bool,
        }
        """
        error_type = self.classifier.classify(error)
        strategy = self.classifier.get_recovery(error_type)

        result = {
            "error_type": error_type.value,
            "action": strategy["action"],
            "should_retry": strategy["max_retries"] > 0,
            "max_retries": strategy["max_retries"],
            "provider": None,
            "max_tokens_reduction": 1.0,
            "backoff_seconds": 0,
        }

        if strategy["action"] == "backoff_and_switch":
            result["backoff_seconds"] = strategy.get("backoff_base", 15)
            failover = self.router.get_failover_chain(current_provider, n=1)
            if failover:
                result["provider"] = failover[0].name
                result["action"] = "switch_provider"

        elif strategy["action"] == "reduce_tokens":
            result["max_tokens_reduction"] = strategy.get("token_reduction", 0.5)

        elif strategy["action"] == "truncate":
            result["max_tokens_reduction"] = strategy.get("truncation", 0.6)

        elif strategy["action"] == "switch_provider":
            failover = self.router.get_failover_chain(current_provider, n=1)
            if failover:
                result["provider"] = failover[0].name

        elif strategy["action"] == "alert":
            result["should_retry"] = False

        self._retry_history.append({"error_type": error_type.value, "action": result["action"]})
        return result

    def stats(self) -> Dict[str, Any]:
        action_counts = defaultdict(int)
        for h in self._retry_history:
            action_counts[h["action"]] += 1
        return {
            "total_retries": len(self._retry_history),
            "by_action": dict(action_counts),
            "classifier": self.classifier.stats(),
        }


# ============================================================================
# 11. Pipeline DAG Executor
# ============================================================================

class PipelineDAGExecutor:
    """
    Dependency-aware parallel stage execution.

    Models pipeline stages as a DAG. Stages with satisfied dependencies
    run in parallel. Maximizes throughput while respecting ordering.

    Example DAG:
      ideation → tree_search → experiment_design → code_generation
                                                  → self_reflection_exp
      code_generation → execution → analysis → paper_writing → review
      self_reflection_exp ─────────────────────↗
    """

    @dataclass
    class StageNode:
        name: str
        dependencies: List[str] = field(default_factory=list)
        status: str = "pending"  # pending, running, completed, failed, skipped
        result: Any = None
        duration_s: float = 0

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self._stages: Dict[str, "PipelineDAGExecutor.StageNode"] = {}
        self._executor: Optional[ThreadPoolExecutor] = None  # lazy-init
        self._lock = threading.Lock()

    def _get_executor(self) -> ThreadPoolExecutor:
        """Lazily create the thread pool on first use."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        return self._executor

    def add_stage(self, name: str, dependencies: List[str] = None) -> None:
        self._stages[name] = self.StageNode(name=name, dependencies=dependencies or [])

    def get_ready_stages(self) -> List[str]:
        """Get stages whose dependencies are all completed."""
        ready = []
        with self._lock:
            for name, stage in self._stages.items():
                if stage.status != "pending":
                    continue
                deps_met = all(
                    self._stages.get(d, self.StageNode(d)).status in ("completed", "skipped")
                    for d in stage.dependencies
                )
                if deps_met:
                    ready.append(name)
        return ready

    def mark_completed(self, name: str, result: Any = None, duration_s: float = 0) -> None:
        with self._lock:
            if name in self._stages:
                self._stages[name].status = "completed"
                self._stages[name].result = result
                self._stages[name].duration_s = duration_s

    def mark_failed(self, name: str) -> None:
        with self._lock:
            if name in self._stages:
                self._stages[name].status = "failed"

    def mark_skipped(self, name: str) -> None:
        with self._lock:
            if name in self._stages:
                self._stages[name].status = "skipped"

    def execute_parallel(self, stage_fns: Dict[str, Callable]) -> Dict[str, Any]:
        """Execute ready stages in parallel, return results."""
        ready = self.get_ready_stages()
        futures = {}
        for name in ready:
            if name in stage_fns:
                with self._lock:
                    self._stages[name].status = "running"
                futures[name] = self._get_executor().submit(stage_fns[name])

        results = {}
        for name, future in futures.items():
            try:
                start = time.time()
                result = future.result(timeout=300)
                self.mark_completed(name, result, time.time() - start)
                results[name] = result
            except Exception as e:
                self.mark_failed(name)
                results[name] = {"error": str(e)}

        return results

    @property
    def all_complete(self) -> bool:
        return all(s.status in ("completed", "failed", "skipped") for s in self._stages.values())

    def critical_path(self) -> List[str]:
        """Estimate the critical path (longest dependency chain)."""
        def _depth(name: str, visited: set) -> int:
            if name in visited:
                return 0
            visited.add(name)
            stage = self._stages.get(name)
            if not stage or not stage.dependencies:
                return 1
            return 1 + max(_depth(d, visited) for d in stage.dependencies)

        depths = {name: _depth(name, set()) for name in self._stages}
        sorted_stages = sorted(depths.items(), key=lambda x: x[1], reverse=True)
        return [name for name, _ in sorted_stages]

    def shutdown(self):
        if self._executor is not None:
            self._executor.shutdown(wait=False)

    def stats(self) -> Dict[str, Any]:
        return {
            "stages": {
                name: {"status": s.status, "duration_s": round(s.duration_s, 1)}
                for name, s in self._stages.items()
            },
            "critical_path": self.critical_path(),
            "all_complete": self.all_complete,
        }


# ============================================================================
# 12. Artifact Store
# ============================================================================

class ArtifactStore:
    """
    Versioned storage for pipeline outputs with metadata and search.

    Stores: papers, code, experiment results, ideas, reviews.
    Each artifact has: id, type, version, content, metadata, timestamp.
    Supports search by type, metadata, and content substring.
    """

    @dataclass
    class Artifact:
        id: str
        artifact_type: str  # paper, code, experiment, idea, review
        version: int
        content: str
        metadata: Dict[str, Any]
        timestamp: float
        tags: List[str] = field(default_factory=list)

    def __init__(self, store_dir: str = None):
        self.store_dir = store_dir or str(Path(__file__).parent / "output" / "artifacts")
        os.makedirs(self.store_dir, exist_ok=True)
        self._index: Dict[str, List["ArtifactStore.Artifact"]] = defaultdict(list)
        self._lock = threading.Lock()
        self._load_index()

    def _index_path(self) -> str:
        return os.path.join(self.store_dir, "_index.json")

    def _load_index(self) -> None:
        try:
            path = self._index_path()
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                for item in data.get("artifacts", []):
                    a = self.Artifact(
                        id=item["id"], artifact_type=item["type"],
                        version=item["version"], content="",  # don't load content
                        metadata=item.get("metadata", {}),
                        timestamp=item.get("timestamp", 0),
                        tags=item.get("tags", []),
                    )
                    self._index[a.id].append(a)
        except Exception:
            pass

    def _save_index(self) -> None:
        try:
            artifacts = []
            for versions in self._index.values():
                for a in versions:
                    artifacts.append({
                        "id": a.id, "type": a.artifact_type,
                        "version": a.version, "metadata": a.metadata,
                        "timestamp": a.timestamp, "tags": a.tags,
                    })
            with open(self._index_path(), "w") as f:
                json.dump({"artifacts": artifacts}, f, indent=2)
        except Exception:
            pass

    def store(self, artifact_id: str, artifact_type: str, content: str,
              metadata: Dict[str, Any] = None, tags: List[str] = None) -> int:
        """Store an artifact. Returns version number."""
        with self._lock:
            existing = self._index.get(artifact_id, [])
            version = len(existing) + 1

            artifact = self.Artifact(
                id=artifact_id, artifact_type=artifact_type,
                version=version, content=content,
                metadata=metadata or {}, timestamp=time.time(),
                tags=tags or [],
            )

            # Save content to file
            content_path = os.path.join(self.store_dir, f"{artifact_id}_v{version}.json")
            with open(content_path, "w") as f:
                json.dump({"content": content, "metadata": metadata or {}}, f)

            self._index[artifact_id].append(artifact)
            self._save_index()
            return version

    def get(self, artifact_id: str, version: int = None) -> Optional["ArtifactStore.Artifact"]:
        """Get artifact by ID and optional version (latest if not specified)."""
        with self._lock:
            versions = self._index.get(artifact_id, [])
            if not versions:
                return None
            if version:
                for v in versions:
                    if v.version == version:
                        return self._load_content(v)
            return self._load_content(versions[-1])

    def _load_content(self, artifact: "ArtifactStore.Artifact") -> "ArtifactStore.Artifact":
        """Load content from disk."""
        content_path = os.path.join(self.store_dir, f"{artifact.id}_v{artifact.version}.json")
        try:
            with open(content_path) as f:
                data = json.load(f)
            artifact.content = data.get("content", "")
        except Exception:
            pass
        return artifact

    def search(self, artifact_type: str = None, tags: List[str] = None) -> List["ArtifactStore.Artifact"]:
        """Search artifacts by type and/or tags."""
        results = []
        with self._lock:
            for versions in self._index.values():
                latest = versions[-1] if versions else None
                if not latest:
                    continue
                if artifact_type and latest.artifact_type != artifact_type:
                    continue
                if tags and not any(t in latest.tags for t in tags):
                    continue
                results.append(latest)
        return results

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            types = defaultdict(int)
            for versions in self._index.values():
                if versions:
                    types[versions[-1].artifact_type] += 1
        return {
            "total_artifacts": sum(len(v) for v in self._index.values()),
            "unique_ids": len(self._index),
            "by_type": dict(types),
        }


# ============================================================================
# Master Infrastructure Optimizer
# ============================================================================

class InfraOptimizer:
    """Aggregates all infrastructure optimization components."""

    def __init__(self, enable_all: bool = True):
        self.disk_cache = DiskCache() if enable_all else None
        self.semantic_cache = SemanticCache() if enable_all else None
        self.provider_router = ProviderRouter() if enable_all else None
        self.error_classifier = ErrorClassifier() if enable_all else None
        self.cost_attributor = CostAttributor() if enable_all else None
        self.logger = StructuredLogger() if enable_all else None
        self.dry_run = None  # created on demand
        self.ablation = AblationRunner() if enable_all else None
        self.resource_monitor = ResourceMonitor() if enable_all else None
        self.retry_orchestrator = RetryOrchestrator(
            self.error_classifier, self.provider_router
        ) if enable_all else None
        self.dag_executor = PipelineDAGExecutor() if enable_all else None
        self.artifact_store = ArtifactStore() if enable_all else None

    def summary(self) -> Dict[str, Any]:
        result = {}
        if self.disk_cache: result["disk_cache"] = self.disk_cache.stats()
        if self.semantic_cache: result["semantic_cache"] = self.semantic_cache.stats()
        if self.provider_router: result["provider_router"] = self.provider_router.stats()
        if self.error_classifier: result["error_classifier"] = self.error_classifier.stats()
        if self.cost_attributor: result["cost_attributor"] = self.cost_attributor.stats()
        if self.logger: result["structured_logger"] = self.logger.stats()
        if self.ablation: result["ablation_runner"] = self.ablation.stats()
        if self.resource_monitor: result["resource_monitor"] = self.resource_monitor.stats()
        if self.retry_orchestrator: result["retry_orchestrator"] = self.retry_orchestrator.stats()
        if self.dag_executor: result["dag_executor"] = self.dag_executor.stats()
        if self.artifact_store: result["artifact_store"] = self.artifact_store.stats()
        return result

    def shutdown(self):
        if self.dag_executor:
            self.dag_executor.shutdown()
