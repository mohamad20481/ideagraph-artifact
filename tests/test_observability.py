"""
Tests for observability.py — structured logs, metrics, tracing, redaction.
"""
import io
import json
import sys
import time

import pytest

import observability
from observability import (
    _Metrics,
    get_trace_id,
    logger,
    metrics,
    record_llm_call,
    set_trace_id,
    trace_span,
)


class TestStructuredLogger:
    """
    The logger's StreamHandler caches a stderr reference at module import,
    so stdio capture fixtures don't see its output. We redirect the handler's
    stream to a StringIO buffer for the duration of each test.
    """

    @pytest.fixture
    def buf(self):
        import logging as _logging
        handler = next(h for h in logger._log.handlers
                       if isinstance(h, _logging.StreamHandler))
        buf = io.StringIO()
        orig = handler.stream
        handler.stream = buf
        yield buf
        handler.stream = orig

    def test_emits_valid_json(self, buf):
        logger.info("test_event", user_id=1, ok=True)
        payload = json.loads(buf.getvalue().strip().splitlines()[-1])
        assert payload["message"] == "test_event"
        assert payload["user_id"] == 1
        assert payload["ok"] is True
        assert payload["level"] == "INFO"

    def test_redacts_api_keys(self, buf):
        logger.info("req",
                    api_key="sk-SECRET", password="pw",
                    bearer="Bearer xyz", username="alice")
        out = buf.getvalue()
        assert "sk-SECRET" not in out
        assert "Bearer xyz" not in out
        assert "alice" in out  # non-sensitive passes through
        assert "REDACTED" in out

    def test_propagates_trace_id(self, buf):
        set_trace_id("abc123")
        logger.info("event")
        payload = json.loads(buf.getvalue().strip().splitlines()[-1])
        assert payload["trace_id"] == "abc123"


class TestTracing:
    def test_trace_id_auto_generates(self):
        observability._trace_id_var.set(None)
        tid = get_trace_id()
        assert len(tid) >= 8
        assert get_trace_id() == tid  # stable within context

    def test_span_records_duration(self):
        m = _Metrics()
        # Patch global metrics for the span context manager.
        orig = observability.metrics
        observability.metrics = m
        try:
            with trace_span("test_phase"):
                time.sleep(0.01)
        finally:
            observability.metrics = orig
        snap = m.snapshot()
        hist_keys = list(snap["histograms"].keys())
        assert any("test_phase" in k for k in hist_keys)

    def test_span_error_status(self):
        m = _Metrics()
        orig = observability.metrics
        observability.metrics = m
        try:
            with pytest.raises(ValueError):
                with trace_span("failing"):
                    raise ValueError("boom")
        finally:
            observability.metrics = orig
        snap = m.snapshot()
        assert any("span_errors_total" in k for k in snap["counters"].keys())


class TestMetrics:
    def test_counter_increments(self):
        m = _Metrics()
        m.inc("requests_total")
        m.inc("requests_total", 5)
        s = m.snapshot()
        assert s["counters"]["requests_total"] == 6.0

    def test_counter_tags(self):
        m = _Metrics()
        m.inc("calls", tags={"status": "ok"})
        m.inc("calls", tags={"status": "err"})
        m.inc("calls", tags={"status": "ok"})
        s = m.snapshot()
        assert s["counters"]['calls{status=ok}'] == 2.0
        assert s["counters"]['calls{status=err}'] == 1.0

    def test_gauge_set(self):
        m = _Metrics()
        m.set("active_runs", 5)
        m.set("active_runs", 3)
        s = m.snapshot()
        assert s["gauges"]["active_runs"] == 3

    def test_histogram(self):
        m = _Metrics()
        for v in [0.1, 0.5, 1.0, 2.0]:
            m.observe("latency", v)
        s = m.snapshot()
        h = s["histograms"]["latency"]
        assert h["count"] == 4
        assert h["sum"] == pytest.approx(3.6)
        assert h["mean"] == pytest.approx(0.9)

    def test_prometheus_format(self):
        m = _Metrics()
        m.inc("requests_total", tags={"method": "GET"})
        m.set("queue_depth", 7)
        m.observe("duration_seconds", 0.5)
        out = m.render_prometheus()
        assert "# TYPE requests_total counter" in out
        assert 'requests_total{method="GET"}' in out
        assert "queue_depth 7" in out
        assert "duration_seconds_bucket" in out
        assert 'le="+Inf"' in out

    def test_prometheus_escapes_special_chars(self):
        m = _Metrics()
        m.inc("c", tags={"k": 'a"b\nc'})
        out = m.render_prometheus()
        assert '\\"' in out
        assert "\\n" in out


class TestLLMInstrumentation:
    def test_record_llm_call_all_metrics(self):
        m = _Metrics()
        orig = observability.metrics
        observability.metrics = m
        try:
            record_llm_call("deepseek", "deepseek-chat",
                            input_tokens=1000, output_tokens=500,
                            duration_s=1.2)
        finally:
            observability.metrics = orig
        s = m.snapshot()
        assert any("llm_calls_total" in k for k in s["counters"].keys())
        assert any("llm_input_tokens_total" in k for k in s["counters"].keys())
        assert any("llm_output_tokens_total" in k for k in s["counters"].keys())
        assert any("llm_call_duration_seconds" in k for k in s["histograms"].keys())
