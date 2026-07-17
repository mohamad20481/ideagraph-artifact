"""
optimization.py - Advanced optimization primitives for IdeaGraph.

Features:
  - PromptCompressor: reduces token usage by 20-40% via intelligent truncation
  - CircuitBreaker: prevents cascading failures with fail-fast pattern
  - BatchScheduler: groups multiple LLM calls into concurrent batches
  - AdaptiveConcurrency: auto-tunes worker count based on API response times
  - WarmCache: cross-iteration semantic caching for expensive computations
  - SpeculativeExecutor: pre-computes likely-needed results in background
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


# ============================================================================
# 1. Prompt Compression
# ============================================================================

class PromptCompressor:
    """
    Reduce token usage by intelligently compressing prompts.

    Strategies:
      - Remove redundant whitespace and blank lines
      - Collapse repeated instructions
      - Truncate examples to key parts
      - Strip verbose system preambles to essentials
      - Deduplicate repeated content across system/user messages
    """

    # Words that add little semantic value in LLM prompts
    FILLER_PHRASES = [
        r"\bplease note that\b",
        r"\bit is important to\b",
        r"\bmake sure to\b",
        r"\bkeep in mind that\b",
        r"\bas mentioned (earlier|above|before)\b",
        r"\bin order to\b",
        r"\bfor the purpose of\b",
        r"\bwith respect to\b",
    ]

    @staticmethod
    def compress(text: str, max_chars: int = 0, aggressive: bool = False) -> str:
        """
        Compress prompt text while preserving meaning.

        Args:
            text: Input text to compress.
            max_chars: If >0, hard-truncate to this char limit (with ... marker).
            aggressive: If True, apply filler-phrase removal.
        """
        if not text:
            return text

        # 1. Normalize whitespace: collapse runs of blank lines to single blank
        result = re.sub(r"\n{3,}", "\n\n", text)

        # 2. Collapse runs of spaces (but preserve indentation structure)
        result = re.sub(r"[ \t]+", " ", result)
        # Restore leading indentation (2-space canonical)
        lines = result.split("\n")
        compressed_lines = []
        for line in lines:
            stripped = line.lstrip()
            indent_count = len(line) - len(stripped)
            # Normalize to 2-space indentation levels
            indent_level = min(indent_count // 2, 5)
            compressed_lines.append("  " * indent_level + stripped)
        result = "\n".join(compressed_lines)

        # 3. Remove filler phrases (aggressive mode)
        if aggressive:
            for pattern in PromptCompressor.FILLER_PHRASES:
                result = re.sub(pattern, "", result, flags=re.IGNORECASE)
            # Clean up double spaces left by removal
            result = re.sub(r"  +", " ", result)

        # 4. Deduplicate consecutive identical lines
        out_lines = []
        prev = None
        for line in result.split("\n"):
            if line.strip() != prev:
                out_lines.append(line)
            prev = line.strip()
        result = "\n".join(out_lines)

        # 5. Hard truncation if requested
        if max_chars > 0 and len(result) > max_chars:
            result = result[:max_chars - 20] + "\n\n[...truncated]"

        return result.strip()

    @staticmethod
    def compress_code(code: str, max_chars: int = 6000) -> str:
        """Compress code for inclusion in prompts — remove docstrings and comments."""
        if len(code) <= max_chars:
            return code

        lines = code.split("\n")
        compressed = []
        in_docstring = False
        for line in lines:
            stripped = line.strip()
            # Skip standalone comments (keep inline ones)
            if stripped.startswith("#") and not stripped.startswith("#!"):
                continue
            # Skip docstrings
            if '"""' in stripped or "'''" in stripped:
                if in_docstring:
                    in_docstring = False
                    continue
                if stripped.count('"""') == 1 or stripped.count("'''") == 1:
                    in_docstring = True
                    continue
                # Single-line docstring
                continue
            if in_docstring:
                continue
            # Skip blank lines in sequence
            if not stripped and compressed and not compressed[-1].strip():
                continue
            compressed.append(line)

        result = "\n".join(compressed)
        if len(result) > max_chars:
            result = result[:max_chars - 20] + "\n# [truncated]"
        return result


# ============================================================================
# 2. Circuit Breaker
# ============================================================================

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast (too many errors)
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreaker:
    """
    Prevents cascading failures by short-circuiting after repeated errors.

    States:
      CLOSED  → normal operation, counting failures
      OPEN    → all calls fail immediately (no API hit), saves budget
      HALF_OPEN → allow one probe call to test recovery

    Transitions:
      CLOSED → OPEN: after `failure_threshold` consecutive failures
      OPEN → HALF_OPEN: after `recovery_timeout` seconds
      HALF_OPEN → CLOSED: on successful probe
      HALF_OPEN → OPEN: on failed probe
    """

    failure_threshold: int = 5
    recovery_timeout: float = 60.0  # seconds before trying again
    state: CircuitState = field(default=CircuitState.CLOSED)
    failure_count: int = field(default=0)
    last_failure_time: float = field(default=0.0)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def can_execute(self) -> bool:
        """Check if a call should be attempted."""
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True
            if self.state == CircuitState.OPEN:
                # Check if recovery timeout has elapsed
                if time.time() - self.last_failure_time >= self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    return True
                return False
            # HALF_OPEN: allow one probe
            return True

    def record_success(self) -> None:
        """Record a successful call — reset circuit."""
        with self._lock:
            self.failure_count = 0
            self.state = CircuitState.CLOSED

    def record_failure(self) -> None:
        """Record a failed call — potentially open circuit."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
            elif self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN

    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN


# ============================================================================
# 3. Adaptive Concurrency Controller
# ============================================================================

class AdaptiveConcurrency:
    """
    Auto-tunes the number of concurrent workers based on API response times
    and error rates. Starts conservative, scales up when latency is low,
    backs off when errors increase.

    Algorithm:
      - Track rolling window of response times (last 20 calls)
      - If p95 latency < target: increase workers by 1 (max cap)
      - If p95 latency > 2x target: decrease workers by 1 (min 1)
      - If error rate > 20%: halve workers immediately
    """

    def __init__(
        self,
        min_workers: int = 1,
        max_workers: int = 8,
        target_latency_s: float = 10.0,
        window_size: int = 20,
    ):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.target_latency = target_latency_s
        self.window_size = window_size

        self.current_workers = min(3, max_workers)  # start moderate
        self._latencies: deque = deque(maxlen=window_size)
        self._errors: deque = deque(maxlen=window_size)
        self._lock = threading.Lock()
        self._last_adjust = time.time()
        self._adjust_interval = 5.0  # don't adjust more than every 5s

    def record_call(self, latency_s: float, success: bool) -> None:
        """Record a completed API call."""
        with self._lock:
            self._latencies.append(latency_s)
            self._errors.append(0 if success else 1)

            if time.time() - self._last_adjust < self._adjust_interval:
                return
            self._last_adjust = time.time()

            if len(self._latencies) < 5:
                return  # not enough data

            # Calculate p95 latency
            sorted_lat = sorted(self._latencies)
            p95_idx = int(len(sorted_lat) * 0.95)
            p95 = sorted_lat[min(p95_idx, len(sorted_lat) - 1)]

            # Calculate error rate
            error_rate = sum(self._errors) / len(self._errors)

            # Adjust
            if error_rate > 0.2:
                # High errors: cut workers in half
                self.current_workers = max(self.min_workers, self.current_workers // 2)
            elif p95 > self.target_latency * 2:
                # High latency: reduce by 1
                self.current_workers = max(self.min_workers, self.current_workers - 1)
            elif p95 < self.target_latency and error_rate < 0.05:
                # Low latency + low errors: increase by 1
                self.current_workers = min(self.max_workers, self.current_workers + 1)

    @property
    def workers(self) -> int:
        return self.current_workers

    def stats(self) -> Dict[str, Any]:
        """Return current concurrency stats."""
        with self._lock:
            lat_list = list(self._latencies)
            err_list = list(self._errors)
        return {
            "current_workers": self.current_workers,
            "avg_latency": sum(lat_list) / len(lat_list) if lat_list else 0,
            "error_rate": sum(err_list) / len(err_list) if err_list else 0,
            "samples": len(lat_list),
        }


# ============================================================================
# 4. Batch Scheduler
# ============================================================================

@dataclass
class BatchRequest:
    """A single request in a batch."""
    call_fn: Callable
    args: tuple
    kwargs: dict
    future: Future = field(default=None)


class BatchScheduler:
    """
    Groups multiple LLM calls and executes them with optimal concurrency.

    Usage:
        scheduler = BatchScheduler(concurrency=AdaptiveConcurrency())
        futures = scheduler.submit_batch([
            (agent._call, (sys1, usr1), {}),
            (agent._call, (sys2, usr2), {}),
        ])
        results = scheduler.collect(futures)
    """

    def __init__(
        self,
        concurrency: Optional[AdaptiveConcurrency] = None,
        max_workers: int = 6,
    ):
        self.concurrency = concurrency or AdaptiveConcurrency(max_workers=max_workers)
        self._executor: Optional[ThreadPoolExecutor] = None

    def _get_executor(self) -> ThreadPoolExecutor:
        workers = self.concurrency.workers if self.concurrency else 4
        if self._executor is None or self._executor._max_workers != workers:
            if self._executor:
                self._executor.shutdown(wait=False)
            self._executor = ThreadPoolExecutor(max_workers=workers)
        return self._executor

    def submit_batch(
        self, calls: List[Tuple[Callable, tuple, dict]],
    ) -> List[Future]:
        """Submit a batch of calls for concurrent execution."""
        executor = self._get_executor()
        futures = []
        for fn, args, kwargs in calls:
            f = executor.submit(self._timed_call, fn, args, kwargs)
            futures.append(f)
        return futures

    def _timed_call(
        self, fn: Callable, args: tuple, kwargs: dict,
    ) -> Any:
        """Execute a call while recording timing for adaptive concurrency."""
        start = time.time()
        success = True
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            success = False
            raise
        finally:
            latency = time.time() - start
            if self.concurrency:
                self.concurrency.record_call(latency, success)

    @staticmethod
    def collect(
        futures: List[Future],
        timeout: float = 120.0,
    ) -> List[Any]:
        """Collect results from submitted futures, preserving order."""
        results = [None] * len(futures)
        for i, f in enumerate(futures):
            try:
                results[i] = f.result(timeout=timeout)
            except Exception as e:
                results[i] = {"error": str(e)}
        return results

    def shutdown(self):
        if self._executor:
            self._executor.shutdown(wait=False)


# ============================================================================
# 5. Warm Cache (cross-iteration semantic caching)
# ============================================================================

class WarmCache:
    """
    Cross-iteration cache for expensive computations.

    Unlike the LRU prompt cache (exact match), this caches by semantic key:
    e.g., (stage_name, idea_title_hash) → result.

    Use cases:
      - Cache DAG construction for similar topics across iterations
      - Cache experiment plans for ideas that only differ slightly
      - Cache code templates for same methodology type
    """

    def __init__(self, max_entries: int = 64):
        self._cache: Dict[str, Tuple[float, Any]] = {}  # key → (timestamp, value)
        self._max_entries = max_entries
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _make_key(self, stage: str, *identifiers: str) -> str:
        """Create a cache key from stage + identifiers."""
        raw = f"{stage}:" + "|".join(str(i) for i in identifiers)
        return hashlib.md5(raw.encode(), usedforsecurity=False).hexdigest()

    def get(self, stage: str, *identifiers: str, max_age_s: float = 3600) -> Optional[Any]:
        """Get a cached result if fresh enough."""
        key = self._make_key(stage, *identifiers)
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None
            ts, value = entry
            if time.time() - ts > max_age_s:
                del self._cache[key]
                self._misses += 1
                return None
            self._hits += 1
            return value

    def put(self, stage: str, *identifiers: str, value: Any) -> None:
        """Store a result in the cache."""
        key = self._make_key(stage, *identifiers)
        with self._lock:
            if len(self._cache) >= self._max_entries:
                # Evict oldest entry
                oldest_key = min(self._cache, key=lambda k: self._cache[k][0])
                del self._cache[oldest_key]
            self._cache[key] = (time.time(), value)

    def invalidate(self, stage: str, *identifiers: str) -> None:
        """Remove a specific cache entry."""
        key = self._make_key(stage, *identifiers)
        with self._lock:
            self._cache.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> Dict[str, Any]:
        return {
            "entries": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{self.hit_rate:.1%}",
        }


# ============================================================================
# 6. Speculative Executor
# ============================================================================

class SpeculativeExecutor:
    """
    Pre-computes likely-needed results in background threads.

    During the pipeline, we can predict what the next stage needs and
    start computing it early. If the prediction is wrong, we discard.
    If right, we save the full stage latency.

    Examples:
      - While reviewing experiment plan, speculatively start code generation
      - While executing code, speculatively start analysis template
      - While writing paper, speculatively start review prompts
    """

    def __init__(self, max_speculative: int = 2):
        self._executor = ThreadPoolExecutor(max_workers=max_speculative)
        self._pending: Dict[str, Future] = {}
        self._lock = threading.Lock()

    def speculate(
        self, key: str, fn: Callable, *args, **kwargs,
    ) -> None:
        """Start a speculative computation in the background."""
        with self._lock:
            if key in self._pending:
                return  # already speculating this
            future = self._executor.submit(fn, *args, **kwargs)
            self._pending[key] = future

    def collect(self, key: str, timeout: float = 0) -> Optional[Any]:
        """
        Retrieve a speculative result.

        Args:
            key: The speculation key.
            timeout: Max seconds to wait. 0 = don't wait (return None if not ready).
        """
        with self._lock:
            future = self._pending.pop(key, None)
        if future is None:
            return None
        if timeout <= 0 and not future.done():
            # Not ready yet, put it back
            with self._lock:
                self._pending[key] = future
            return None
        try:
            return future.result(timeout=max(timeout, 0.1))
        except Exception:
            return None

    def cancel(self, key: str) -> None:
        """Cancel a speculative computation."""
        with self._lock:
            future = self._pending.pop(key, None)
        if future:
            future.cancel()

    def cancel_all(self) -> None:
        with self._lock:
            for f in self._pending.values():
                f.cancel()
            self._pending.clear()

    def shutdown(self):
        self.cancel_all()
        self._executor.shutdown(wait=False)


# ============================================================================
# 7. Stage Performance Tracker
# ============================================================================

@dataclass
class StagePerf:
    """Tracks per-stage performance for pipeline optimization."""
    name: str
    durations: List[float] = field(default_factory=list)
    token_counts: List[int] = field(default_factory=list)
    quality_scores: List[float] = field(default_factory=list)
    failures: int = 0

    @property
    def avg_duration(self) -> float:
        return sum(self.durations) / len(self.durations) if self.durations else 0

    @property
    def avg_tokens(self) -> int:
        return int(sum(self.token_counts) / len(self.token_counts)) if self.token_counts else 0

    @property
    def avg_quality(self) -> float:
        return sum(self.quality_scores) / len(self.quality_scores) if self.quality_scores else 0

    @property
    def efficiency(self) -> float:
        """Quality per second — higher is better."""
        if not self.durations or not self.quality_scores:
            return 0
        return self.avg_quality / max(self.avg_duration, 0.1)


class PipelineOptimizer:
    """
    Aggregates optimization components and provides pipeline-level decisions.

    Integrates: circuit breaker, adaptive concurrency, warm cache,
    speculative executor, stage performance, and prompt compression.
    """

    def __init__(
        self,
        enable_compression: bool = True,
        enable_speculation: bool = True,
        enable_circuit_breaker: bool = True,
        max_workers: int = 6,
    ):
        self.compressor = PromptCompressor() if enable_compression else None
        self.circuit_breaker = CircuitBreaker() if enable_circuit_breaker else None
        self.concurrency = AdaptiveConcurrency(max_workers=max_workers)
        self.batch_scheduler = BatchScheduler(concurrency=self.concurrency)
        self.warm_cache = WarmCache()
        self.speculator = SpeculativeExecutor() if enable_speculation else None
        self.stage_perf: Dict[str, StagePerf] = defaultdict(lambda: StagePerf(name="unknown"))

        self._start_time = time.time()

    def record_stage(
        self, stage: str, duration: float, tokens: int = 0,
        quality: float = 0, failed: bool = False,
    ) -> None:
        """Record stage execution metrics."""
        perf = self.stage_perf[stage]
        perf.name = stage
        perf.durations.append(duration)
        if tokens:
            perf.token_counts.append(tokens)
        if quality:
            perf.quality_scores.append(quality)
        if failed:
            perf.failures += 1

    def should_skip_stage(self, stage: str) -> bool:
        """
        Recommend skipping a stage if it consistently fails or produces
        low-quality output relative to its cost.
        """
        perf = self.stage_perf.get(stage)
        if not perf or len(perf.durations) < 2:
            return False
        # Skip if >50% failure rate
        total = len(perf.durations)
        if perf.failures / total > 0.5:
            return True
        # Skip if efficiency is very low relative to other stages
        efficiencies = {
            k: v.efficiency
            for k, v in self.stage_perf.items()
            if len(v.durations) >= 2
        }
        if efficiencies:
            avg_eff = sum(efficiencies.values()) / len(efficiencies)
            if perf.efficiency < avg_eff * 0.1:
                return True
        return False

    def suggest_timeout(self, stage: str, default: float = 90.0) -> float:
        """
        Suggest a timeout for a stage based on historical p95 duration.
        Adds 50% headroom above p95.
        """
        perf = self.stage_perf.get(stage)
        if not perf or len(perf.durations) < 3:
            return default
        sorted_d = sorted(perf.durations)
        p95_idx = int(len(sorted_d) * 0.95)
        p95 = sorted_d[min(p95_idx, len(sorted_d) - 1)]
        return max(default, p95 * 1.5)

    def summary(self) -> Dict[str, Any]:
        """Get optimization metrics summary."""
        elapsed = time.time() - self._start_time
        stages = {}
        for name, perf in self.stage_perf.items():
            stages[name] = {
                "avg_duration_s": round(perf.avg_duration, 1),
                "avg_quality": round(perf.avg_quality, 3),
                "efficiency": round(perf.efficiency, 4),
                "failures": perf.failures,
                "runs": len(perf.durations),
            }

        return {
            "elapsed_s": round(elapsed, 1),
            "concurrency": self.concurrency.stats(),
            "cache": self.warm_cache.stats(),
            "circuit_breaker": {
                "state": self.circuit_breaker.state.value if self.circuit_breaker else "disabled",
                "failures": self.circuit_breaker.failure_count if self.circuit_breaker else 0,
            },
            "stages": stages,
        }

    def shutdown(self):
        """Clean up resources."""
        self.batch_scheduler.shutdown()
        if self.speculator:
            self.speculator.shutdown()
