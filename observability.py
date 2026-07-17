"""
observability.py - Structured logging + in-memory metrics + tracing.

Provides the three pillars of production observability without requiring any
external service (Datadog/Prometheus/Sentry). Designed to be *scrapeable*:

  * Logs    - JSON lines to stderr (ship via fluent-bit / Loki / CloudWatch)
  * Metrics - Prometheus-compatible text at /metrics
  * Traces  - context-managed spans with trace_id correlation

Drop-in usage:
    from observability import logger, metrics, trace_span, get_trace_id

    logger.info("pipeline_start", topic="LLM reasoning", user_id=42)

    metrics.inc("pipeline_runs_total", tags={"tier": "pro"})
    metrics.observe("pipeline_duration_seconds", 123.4, tags={"tier": "pro"})

    with trace_span("ideation_phase", topic=topic) as span:
        span["ideas_generated"] = 15
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import re
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Trace-id propagation (context-local, survives across async + threads)
# ─────────────────────────────────────────────────────────────────────────────

_trace_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "ideagraph_trace_id", default=None,
)


def get_trace_id() -> str:
    """Return the current trace id (or generate + stash one)."""
    tid = _trace_id_var.get()
    if tid is None:
        tid = uuid.uuid4().hex[:16]
        _trace_id_var.set(tid)
    return tid


def set_trace_id(trace_id: str) -> None:
    _trace_id_var.set(trace_id)


# ─────────────────────────────────────────────────────────────────────────────
# Structured JSON logger
# ─────────────────────────────────────────────────────────────────────────────

# Keys whose *values* we redact from log payloads — prevents accidental leaks
# of API keys, passwords, tokens into stdout/log-aggregator pipelines.
_SENSITIVE_KEY_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key)"),
    re.compile(r"(?i)(password|passwd)"),
    re.compile(r"(?i)(secret|token|bearer)"),
    re.compile(r"(?i)(authorization)"),
]
_REDACT = "***REDACTED***"


def _redact(obj: Any) -> Any:
    """Recursively redact sensitive fields in a dict/list/scalar for logging."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if any(p.search(str(k)) for p in _SENSITIVE_KEY_PATTERNS):
                out[k] = _REDACT
            else:
                out[k] = _redact(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_redact(x) for x in obj]
    return obj


class _JsonFormatter(logging.Formatter):
    """Format log records as JSON lines for structured log pipelines."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": round(time.time(), 6),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": _trace_id_var.get(),
        }
        # Per-call structured fields (attached via `extra={...}`).
        extras = getattr(record, "_extra", None)
        if isinstance(extras, dict):
            payload.update(_redact(extras))
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        try:
            return json.dumps(payload, default=str, ensure_ascii=False)
        except Exception:
            return json.dumps({"ts": payload["ts"], "level": "ERROR",
                               "message": "log_serialization_failed"})


class StructuredLogger:
    """
    Wrapper around stdlib logging that emits JSON lines with attached
    keyword context (event + fields).
    """

    def __init__(self, name: str = "ideagraph") -> None:
        self._log = logging.getLogger(name)
        if not self._log.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(_JsonFormatter())
            self._log.addHandler(handler)
            self._log.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
            self._log.propagate = False

    def _emit(self, level: int, event: str, **fields: Any) -> None:
        self._log.log(level, event, extra={"_extra": dict(fields)})

    def debug(self, event: str, **fields: Any) -> None:
        self._emit(logging.DEBUG, event, **fields)

    def info(self, event: str, **fields: Any) -> None:
        self._emit(logging.INFO, event, **fields)

    def warn(self, event: str, **fields: Any) -> None:
        self._emit(logging.WARNING, event, **fields)

    warning = warn

    def error(self, event: str, **fields: Any) -> None:
        self._emit(logging.ERROR, event, **fields)

    def exception(self, event: str, **fields: Any) -> None:
        self._log.exception(event, extra={"_extra": dict(fields)})


logger = StructuredLogger()


# ─────────────────────────────────────────────────────────────────────────────
# In-memory metrics registry (Prometheus-compatible)
# ─────────────────────────────────────────────────────────────────────────────

class _Metrics:
    """
    Lightweight counter + gauge + histogram registry.

    * counter  - monotonically increasing (request counts, errors).
    * gauge    - set to a current value (queue depth, active runs).
    * histogram - observe a value; exposes count/sum + bucketed counts.

    Output format: Prometheus text exposition v0.0.4. Scrapeable at /metrics.
    """

    _BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600)

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._counters: Dict[tuple, float] = {}
        self._gauges: Dict[tuple, float] = {}
        self._hist_buckets: Dict[tuple, Dict[float, int]] = {}
        self._hist_sum: Dict[tuple, float] = {}
        self._hist_count: Dict[tuple, int] = {}

    @staticmethod
    def _key(name: str, tags: Optional[Dict[str, str]]) -> tuple:
        if not tags:
            return (name, tuple())
        return (name, tuple(sorted(tags.items())))

    def inc(self, name: str, value: float = 1.0,
            tags: Optional[Dict[str, str]] = None) -> None:
        k = self._key(name, tags)
        with self._lock:
            self._counters[k] = self._counters.get(k, 0.0) + value

    def set(self, name: str, value: float,
            tags: Optional[Dict[str, str]] = None) -> None:
        k = self._key(name, tags)
        with self._lock:
            self._gauges[k] = value

    def observe(self, name: str, value: float,
                tags: Optional[Dict[str, str]] = None) -> None:
        k = self._key(name, tags)
        with self._lock:
            self._hist_sum[k] = self._hist_sum.get(k, 0.0) + value
            self._hist_count[k] = self._hist_count.get(k, 0) + 1
            bkt = self._hist_buckets.setdefault(
                k, {b: 0 for b in self._BUCKETS},
            )
            for b in self._BUCKETS:
                if value <= b:
                    bkt[b] += 1

    def render_prometheus(self) -> str:
        """Export metrics in Prometheus text format (v0.0.4)."""
        lines: list[str] = []
        with self._lock:
            for (name, tags), v in self._counters.items():
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{_format_metric(name, tags)} {v}")
            for (name, tags), v in self._gauges.items():
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{_format_metric(name, tags)} {v}")
            for (name, tags), buckets in self._hist_buckets.items():
                lines.append(f"# TYPE {name} histogram")
                for b, c in buckets.items():
                    tag_dict = dict(tags) if tags else {}
                    tag_dict["le"] = str(b)
                    lines.append(f"{_format_metric(name + '_bucket', tuple(sorted(tag_dict.items())))} {c}")
                tag_dict = dict(tags) if tags else {}
                tag_dict["le"] = "+Inf"
                lines.append(f"{_format_metric(name + '_bucket', tuple(sorted(tag_dict.items())))} {self._hist_count[(name, tags)]}")
                lines.append(f"{_format_metric(name + '_sum', tags)} {self._hist_sum[(name, tags)]}")
                lines.append(f"{_format_metric(name + '_count', tags)} {self._hist_count[(name, tags)]}")
        return "\n".join(lines) + "\n"

    def snapshot(self) -> Dict[str, Any]:
        """Structured snapshot (for health endpoints / debugging)."""
        with self._lock:
            return {
                "counters": {
                    f"{name}{'{' + ','.join(f'{k}={v}' for k, v in tags) + '}' if tags else ''}": val
                    for (name, tags), val in self._counters.items()
                },
                "gauges": {
                    f"{name}{'{' + ','.join(f'{k}={v}' for k, v in tags) + '}' if tags else ''}": val
                    for (name, tags), val in self._gauges.items()
                },
                "histograms": {
                    f"{name}{'{' + ','.join(f'{k}={v}' for k, v in tags) + '}' if tags else ''}": {
                        "count": self._hist_count[(name, tags)],
                        "sum": self._hist_sum[(name, tags)],
                        "mean": (self._hist_sum[(name, tags)] / self._hist_count[(name, tags)])
                                if self._hist_count[(name, tags)] else 0.0,
                    }
                    for (name, tags) in self._hist_count.keys()
                },
            }


def _format_metric(name: str, tags_tuple: tuple) -> str:
    """Render name + tags as Prometheus line: name{k=\"v\",k2=\"v2\"}."""
    if not tags_tuple:
        return name
    tag_str = ",".join(f'{k}="{_escape(str(v))}"' for k, v in tags_tuple)
    return f"{name}{{{tag_str}}}"


def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


metrics = _Metrics()


# ─────────────────────────────────────────────────────────────────────────────
# Span tracing (context-manager based)
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def trace_span(name: str, **fields: Any) -> Iterator[Dict[str, Any]]:
    """
    Context-managed span. On exit, emits a log line + histogram observation.

    Usage:
        with trace_span("ideation_phase", topic=topic) as span:
            ...
            span["ideas_generated"] = 15
    """
    trace_id = get_trace_id()
    span_id = uuid.uuid4().hex[:8]
    start = time.time()
    span: Dict[str, Any] = {
        "name": name, "span_id": span_id, "trace_id": trace_id,
        "start": start, **fields,
    }
    logger.debug("span_start", name=name, span_id=span_id, **fields)
    try:
        yield span
        duration = time.time() - start
        span["duration_s"] = round(duration, 6)
        span["status"] = "ok"
        logger.info("span_end", **span)
        metrics.observe(f"span_duration_seconds",
                        duration, tags={"span": name, "status": "ok"})
    except Exception as exc:
        duration = time.time() - start
        span["duration_s"] = round(duration, 6)
        span["status"] = "error"
        span["error"] = str(exc)
        logger.error("span_error", **span)
        metrics.observe("span_duration_seconds",
                        duration, tags={"span": name, "status": "error"})
        metrics.inc("span_errors_total", tags={"span": name})
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: LLM-call instrumentation
# ─────────────────────────────────────────────────────────────────────────────

def record_llm_call(
    provider: str, model: str, input_tokens: int, output_tokens: int,
    duration_s: float, status: str = "ok",
    user_id: Optional[int] = None, run_id: Optional[str] = None,
) -> None:
    """One-liner to record a full LLM call across logs + metrics + cost."""
    tags = {"provider": provider, "model": model, "status": status}
    metrics.inc("llm_calls_total", tags=tags)
    metrics.inc("llm_input_tokens_total", value=input_tokens, tags=tags)
    metrics.inc("llm_output_tokens_total", value=output_tokens, tags=tags)
    metrics.observe("llm_call_duration_seconds", duration_s, tags=tags)
    try:
        from production_optimization import get_cost_tracker
        cost = get_cost_tracker().record(
            provider, input_tokens, output_tokens,
            user_id=user_id, run_id=run_id,
        )
        metrics.inc("llm_cost_usd_total", value=cost, tags=tags)
    except Exception:
        cost = 0.0

    # Persist to DB audit trail (best-effort).
    if cost > 0:
        try:
            from db import log_cost as _log_cost
            _log_cost(
                provider=provider, model=model,
                prompt_tokens=input_tokens, completion_tokens=output_tokens,
                cost_usd=cost, stage="llm_call",
                user_id=user_id, run_id=run_id,
            )
        except Exception:
            pass

    logger.info(
        "llm_call",
        provider=provider, model=model,
        input_tokens=input_tokens, output_tokens=output_tokens,
        duration_s=round(duration_s, 3), cost_usd=round(cost, 6),
        status=status, user_id=user_id, run_id=run_id,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    set_trace_id("smoketest-trace")
    logger.info("startup", version="1.0.0", env="dev")

    with trace_span("demo_phase", topic="test"):
        metrics.inc("demo_counter")
        metrics.set("demo_gauge", 42)
        metrics.observe("demo_latency_seconds", 0.123)
        time.sleep(0.01)

    # Redaction check.
    logger.info("auth_attempt", username="alice", api_key="sk-secret-should-be-redacted")
    record_llm_call("deepseek", "deepseek-chat", 1000, 500, 0.8)

    print("\n--- Prometheus output ---", file=sys.stderr)
    print(metrics.render_prometheus(), file=sys.stderr)
