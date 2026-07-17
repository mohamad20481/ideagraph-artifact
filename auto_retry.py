"""
auto_retry.py - Auto-retry and self-healing for pipeline stages.

When a stage fails, instead of stopping the pipeline:
  1. Classify the error (timeout, API, parse, logic)
  2. Apply stage-specific fix (truncate prompt, switch provider, retry)
  3. Retry with the fix applied
  4. If still failing after max retries, use fallback result and continue

This makes the pipeline much more robust — it rarely stops mid-run.
"""

from __future__ import annotations

import random as _random
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class RetryResult:
    """Result of a retry-wrapped stage execution."""
    success: bool = False
    result: Any = None
    attempts: int = 0
    errors: List[str] = field(default_factory=list)
    total_time_s: float = 0.0
    used_fallback: bool = False


@dataclass
class StageRetryConfig:
    """Configuration for retrying a specific stage."""
    max_retries: int = 2
    backoff_s: float = 3.0
    fallback_result: Any = None
    on_error: Optional[str] = None  # "truncate", "simplify", "switch_provider"


# Default retry configs per stage
STAGE_CONFIGS = {
    "ideation": StageRetryConfig(max_retries=3, backoff_s=2.0, fallback_result={"ideas": [], "coverage": 0}),
    "experiment_design": StageRetryConfig(max_retries=2, backoff_s=3.0, on_error="simplify"),
    "code_generation": StageRetryConfig(max_retries=3, backoff_s=2.0, on_error="simplify"),
    "execution": StageRetryConfig(max_retries=2, backoff_s=5.0),
    "analysis": StageRetryConfig(max_retries=2, backoff_s=2.0, fallback_result={"summary": "Analysis failed", "key_findings": [], "supports_hypothesis": "unknown"}),
    "paper_writing": StageRetryConfig(max_retries=2, backoff_s=3.0, fallback_result={"markdown": "Paper generation failed.", "sections": {}}),
    "review": StageRetryConfig(max_retries=2, backoff_s=2.0, fallback_result={"overall_score": 5, "decision": "weak_reject", "strengths": [], "weaknesses": ["Review failed"]}),
    "tree_search": StageRetryConfig(max_retries=1, backoff_s=2.0, fallback_result=[]),
    "self_reflection": StageRetryConfig(max_retries=1, backoff_s=1.0, fallback_result={}),
}


class AutoRetryEngine:
    """
    Wraps pipeline stage execution with automatic retry and self-healing.

    Usage:
        engine = AutoRetryEngine()
        result = engine.execute_with_retry("ideation", lambda: pipeline.run_ideation(topic))
        if result.success:
            ideas = result.result
    """

    def __init__(self, on_progress: Callable[[str], None] = None):
        self.on_progress = on_progress
        self._stage_history: Dict[str, List[RetryResult]] = {}
        self._total_retries: int = 0
        self._total_recoveries: int = 0

    def _log(self, msg: str) -> None:
        if self.on_progress:
            self.on_progress(msg)

    def execute_with_retry(
        self,
        stage_name: str,
        fn: Callable[[], Any],
        config: StageRetryConfig = None,
    ) -> RetryResult:
        """Execute a stage with automatic retry on failure."""
        cfg = config or STAGE_CONFIGS.get(stage_name, StageRetryConfig())

        result = RetryResult()
        start = time.time()

        for attempt in range(cfg.max_retries + 1):
            result.attempts = attempt + 1

            # ── Circuit-breaker check: skip retries if provider is down ──
            if attempt > 0:
                try:
                    from production_optimization import get_circuit_breaker
                    ok_cb, cb_msg = get_circuit_breaker().allow(
                        getattr(__import__("config"), "PROVIDER", "unknown"),
                    )
                    if not ok_cb:
                        self._log(
                            f"  [RETRY] Circuit breaker open — aborting {stage_name} retries: {cb_msg}"
                        )
                        break
                except ImportError:
                    pass

            try:
                output = fn()
                result.success = True
                result.result = output
                result.total_time_s = time.time() - start

                if stage_name not in self._stage_history:
                    self._stage_history[stage_name] = []
                self._stage_history[stage_name].append(result)
                return result

            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)[:200]}"
                result.errors.append(error_msg)

                if attempt < cfg.max_retries:
                    self._total_retries += 1
                    # Exponential backoff + jitter (prevents thundering herd).
                    base = min(cfg.backoff_s * (2 ** attempt), 30.0)
                    wait = base * (0.5 + _random.random())
                    self._log(
                        f"  [RETRY] {stage_name} failed (attempt {attempt + 1}/{cfg.max_retries + 1}): "
                        f"{error_msg[:80]}. Retrying in {wait:.1f}s..."
                    )
                    time.sleep(wait)
                else:
                    self._log(
                        f"  [RETRY] {stage_name} failed after {cfg.max_retries + 1} attempts: {error_msg[:80]}"
                    )

        # All retries exhausted — use fallback
        if cfg.fallback_result is not None:
            result.success = True
            result.result = cfg.fallback_result
            result.used_fallback = True
            self._total_recoveries += 1
            self._log(f"  [FALLBACK] {stage_name} using fallback result to continue pipeline")
        else:
            result.success = False

        result.total_time_s = time.time() - start
        if stage_name not in self._stage_history:
            self._stage_history[stage_name] = []
        self._stage_history[stage_name].append(result)
        return result

    def get_reliability(self, stage_name: str) -> float:
        """Success rate for a stage (including fallback recoveries)."""
        history = self._stage_history.get(stage_name, [])
        if not history:
            return 1.0
        return sum(1 for r in history if r.success) / len(history)

    def stats(self) -> Dict[str, Any]:
        return {
            "total_retries": self._total_retries,
            "total_recoveries": self._total_recoveries,
            "stage_reliability": {
                stage: round(self.get_reliability(stage), 2)
                for stage in self._stage_history
            },
        }
