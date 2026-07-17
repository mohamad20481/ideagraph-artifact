"""Tests for two bug fixes:

  1. agents/tree_search.py — crashed with IndexError when LLM returned
     zero root approaches (`nodes[0].score` on empty list).
  2. idea_regenerator.regenerate() — failed silently when LLM was
     unavailable / rate-limited / returned junk JSON. Now collects
     per-call diagnostics into an optional list parameter.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import idea_regenerator as ir
from agents.tree_search import ExperimentTreeSearch


# ─────────────────────────────────────────────────────────────────────────────
# Mock LLM client (for regenerate tests)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _Resp:
    success: bool
    text: str = ""
    error: str = ""


class _SeqClient:
    """Returns a queue of pre-baked responses; raises on missing."""

    def __init__(self, responses: List[Any]):
        self._q = list(responses)
        self.call_count = 0

    def call(self, **_kw) -> _Resp:
        self.call_count += 1
        if not self._q:
            return _Resp(False, error="queue exhausted")
        item = self._q.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, _Resp):
            return item
        if isinstance(item, dict):
            return _Resp(True, json.dumps(item))
        if isinstance(item, str):
            return _Resp(True, item)
        return _Resp(False, error=f"unknown item: {item}")


_PARENT_IDEA = {
    "title": "Original idea",
    "motivation": "why",
    "method": "do the thing",
    "hypothesis": "the thing works",
    "resources": "1 GPU-week",
    "expected_outcome": "good results",
    "risk_assessment": "the thing might break",
    "source_strategy": "A",
    "methodology_type": "empirical_study",
    "novelty_level": "moderate",
    "generation": 0,
}


def _valid_regen_response() -> dict:
    """A JSON payload that satisfies _dict_to_idea's required fields."""
    return {
        "title": "Refined idea",
        "motivation": "improved motivation",
        "method": "better method",
        "hypothesis": "sharper hypothesis",
        "resources": "1 GPU-day",
        "expected_outcome": "better results",
        "risk_assessment": "fewer risks",
        "methodology_type": "empirical_study",
        "novelty_level": "moderate",
        "lineage_note": "addressed feedback",
    }


# ─────────────────────────────────────────────────────────────────────────────
# tree_search.py: empty-approaches must not crash
# ─────────────────────────────────────────────────────────────────────────────

def test_tree_search_returns_empty_when_no_approaches():
    """Reproduces the original crash: when _generate_approaches returns
    [], the old code did `nodes[0].score` and raised IndexError."""
    search = ExperimentTreeSearch()
    progress_log: List[str] = []

    with patch.object(search, "_generate_approaches", return_value=[]):
        result = search.search(
            idea={"title": "x", "method": "m", "hypothesis": "h"},
            domain="ml", max_branches=3, max_depth=2,
            on_progress=progress_log.append,
        )
    assert result == []
    # User-visible diagnostic must mention the LLM being unavailable.
    joined = " ".join(progress_log).lower()
    assert "no approaches" in joined or "unavailable" in joined


def test_tree_search_does_not_call_evaluate_when_no_approaches():
    """When approaches is empty we must NOT waste LLM calls on evaluate
    or refine — the tree expansion should bail out immediately."""
    search = ExperimentTreeSearch()
    with patch.object(search, "_generate_approaches", return_value=[]), \
            patch.object(search, "_evaluate_approach") as _eval, \
            patch.object(search, "_refine_approach") as _refine:
        result = search.search(
            idea={"title": "x", "method": "m", "hypothesis": "h"},
            domain="ml", max_branches=3, max_depth=2,
        )
    assert result == []
    _eval.assert_not_called()
    _refine.assert_not_called()


def test_tree_search_happy_path_still_works():
    """Smoke check: the early-exit fix didn't break the success path."""
    search = ExperimentTreeSearch()
    fake_approaches = [
        {"approach": "A1", "method_variant": "m1", "dataset": "d", "key_difference": "k"},
        {"approach": "A2", "method_variant": "m2", "dataset": "d", "key_difference": "k"},
    ]
    with patch.object(search, "_generate_approaches", return_value=fake_approaches), \
            patch.object(search, "_evaluate_approach", side_effect=[0.7, 0.5]), \
            patch.object(search, "_refine_approach", return_value=[]):
        result = search.search(
            idea={"title": "x", "method": "m", "hypothesis": "h"},
            domain="ml", max_branches=2, max_depth=1,
        )
    assert len(result) == 2
    # Sorted descending by score.
    assert result[0].score >= result[1].score


def test_tree_search_handles_approach_missing_approach_key():
    """Defensive: if the LLM returns a dict without an 'approach' field
    (which the old code accessed directly with []), we should fall back
    to 'default approach' rather than KeyError."""
    search = ExperimentTreeSearch()
    fake_approaches = [
        {"method_variant": "m1"},  # no 'approach' key
    ]
    with patch.object(search, "_generate_approaches", return_value=fake_approaches), \
            patch.object(search, "_evaluate_approach", return_value=0.5):
        result = search.search(
            idea={"title": "x", "method": "m", "hypothesis": "h"},
            domain="ml", max_branches=1, max_depth=1,
        )
    assert len(result) == 1
    assert result[0].approach == "default approach"


# ─────────────────────────────────────────────────────────────────────────────
# idea_regenerator.regenerate(): diagnostics collection
# ─────────────────────────────────────────────────────────────────────────────

def test_regenerate_no_client_collects_diagnostic():
    diag: List[str] = []
    out = ir.regenerate(
        _PARENT_IDEA, "refine", n=2,
        claude_client=None, diagnostics=diag,
    )
    assert out == []
    assert len(diag) == 1
    # Diagnostic must point the user at the admin panel.
    assert "Admin" in diag[0] or "API key" in diag[0]


def test_regenerate_no_client_without_diagnostics_still_returns_empty():
    """Backward compatibility: old callers that don't pass diagnostics
    must keep getting [] silently."""
    out = ir.regenerate(_PARENT_IDEA, "refine", n=2, claude_client=None)
    assert out == []


def test_regenerate_collects_per_call_failure_messages():
    """All N calls failing must produce N diagnostic messages."""
    client = _SeqClient([
        _Resp(False, error="rate limited"),
        _Resp(False, error="timeout"),
        _Resp(False, error="server error 500"),
    ])
    diag: List[str] = []
    out = ir.regenerate(
        _PARENT_IDEA, "refine", n=3,
        claude_client=client, diagnostics=diag,
    )
    assert out == []
    assert len(diag) == 3
    assert all("API returned failure" in d for d in diag)
    assert any("rate limited" in d for d in diag)


def test_regenerate_collects_exception_messages():
    client = _SeqClient([RuntimeError("network blew up")])
    diag: List[str] = []
    ir.regenerate(
        _PARENT_IDEA, "refine", n=1,
        claude_client=client, diagnostics=diag,
    )
    assert len(diag) == 1
    assert "RuntimeError" in diag[0]
    assert "network blew up" in diag[0]


def test_regenerate_collects_unparseable_json_diagnostic():
    client = _SeqClient([_Resp(True, "this is not JSON at all{{{")])
    diag: List[str] = []
    out = ir.regenerate(
        _PARENT_IDEA, "refine", n=1,
        claude_client=client, diagnostics=diag,
    )
    assert out == []
    assert len(diag) == 1
    assert "not valid idea JSON" in diag[0]


def test_regenerate_mixed_success_and_failure_only_diagnoses_failures():
    """Successful calls produce ideas, failed calls produce diagnostics.
    No diagnostic should appear for a successful call."""
    client = _SeqClient([
        _valid_regen_response(),         # success
        _Resp(False, error="rate limit"),  # failure
        _valid_regen_response(),         # success
    ])
    diag: List[str] = []
    out = ir.regenerate(
        _PARENT_IDEA, "refine", n=3,
        claude_client=client, diagnostics=diag,
    )
    assert len(out) == 2
    assert len(diag) == 1
    assert "Call 2/3" in diag[0]


def test_regenerate_all_succeed_produces_empty_diagnostics():
    client = _SeqClient([
        _valid_regen_response(), _valid_regen_response(),
    ])
    diag: List[str] = []
    out = ir.regenerate(
        _PARENT_IDEA, "refine", n=2,
        claude_client=client, diagnostics=diag,
    )
    assert len(out) == 2
    assert diag == []


def test_regenerate_diagnostics_param_is_optional_and_defaults_none():
    """The new parameter must NOT be required — existing call sites
    that don't pass it should keep working."""
    client = _SeqClient([_Resp(False, error="x")])
    # No diagnostics= kwarg.
    out = ir.regenerate(_PARENT_IDEA, "refine", n=1, claude_client=client)
    assert out == []  # no crash, no missing-arg error


# ─────────────────────────────────────────────────────────────────────────────
# _classify_api_error: actionable hints for common provider errors
# ─────────────────────────────────────────────────────────────────────────────

def test_classifier_empty_input_returns_empty():
    assert ir._classify_api_error("") == ""
    assert ir._classify_api_error(None) == ""  # type: ignore[arg-type]


def test_classifier_unknown_error_returns_empty():
    """Novel error shapes should leave the diagnostic unchanged
    (empty hint), not produce a misleading one."""
    assert ir._classify_api_error("something totally novel") == ""
    assert ir._classify_api_error("XYZZY blew up") == ""


def test_classifier_recognizes_insufficient_balance():
    """The exact error string that surfaced in the user's report."""
    err = ('API 403: {"code":"INSUFFICIENT_BALANCE",'
           '"message":"Insufficient account balance"}')
    hint = ir._classify_api_error(err)
    assert hint
    assert "credit" in hint.lower() or "balance" in hint.lower()
    assert "🔌 LLM Provider" in hint


def test_classifier_recognizes_other_balance_forms():
    """Different providers phrase the same problem differently."""
    for err in (
        "out of credit",
        "Quota exceeded for project",
        "billing required: paid plan needed",
    ):
        hint = ir._classify_api_error(err)
        assert hint, f"missed: {err!r}"
        assert "credit" in hint.lower() or "free tier" in hint.lower()


def test_classifier_recognizes_auth_errors():
    for err in (
        "API 401: invalid_api_key",
        "Incorrect API key provided",
        "Authentication failed",
        "Unauthorized",
    ):
        hint = ir._classify_api_error(err)
        assert hint, f"missed: {err!r}"
        assert "API key" in hint or "🔑" in hint


def test_classifier_recognizes_rate_limits():
    for err in (
        "API 429: rate_limit_exceeded",
        "Too many requests in the last minute",
        "Throttled",
    ):
        hint = ir._classify_api_error(err)
        assert hint, f"missed: {err!r}"
        assert "rate-limit" in hint.lower() or "⏳" in hint


def test_classifier_recognizes_upstream_5xx():
    for err in (
        "API 502: Bad Gateway",
        "API 503: service unavailable",
        "Gateway timeout",
    ):
        hint = ir._classify_api_error(err)
        assert hint, f"missed: {err!r}"
        assert "transient" in hint.lower() or "🌐" in hint
        # Must point at the aiprimetech proxy as a likely culprit.
        assert "aiprimetech" in hint.lower() or "provider" in hint.lower()


def test_classifier_recognizes_context_length():
    for err in (
        "context_length_exceeded",
        "Maximum context length is 8192 tokens",
    ):
        hint = ir._classify_api_error(err)
        assert hint, f"missed: {err!r}"
        assert "context" in hint.lower() or "📏" in hint


def test_classifier_recognizes_model_not_found():
    for err in (
        "model_not_found: claude-99",
        "The model `gpt-99` does not exist",
        "No such model",
    ):
        hint = ir._classify_api_error(err)
        assert hint, f"missed: {err!r}"
        assert "model" in hint.lower() or "🎯" in hint


def test_diagnostic_includes_classifier_hint_for_balance_error():
    """End-to-end: when DeepSeek returns INSUFFICIENT_BALANCE, the
    diagnostic message should embed the classifier hint as a sub-line."""
    insufficient = _Resp(
        False,
        error=('API 403: {"code":"INSUFFICIENT_BALANCE",'
                '"message":"Insufficient account balance"}'),
    )
    client = _SeqClient([insufficient])
    diag: List[str] = []
    ir.regenerate(
        _PARENT_IDEA, "refine", n=1,
        claude_client=client, diagnostics=diag,
    )
    assert len(diag) == 1
    # The raw error must still be there for debugging.
    assert "INSUFFICIENT_BALANCE" in diag[0]
    # AND the actionable hint must be appended.
    assert "credit" in diag[0].lower() or "balance" in diag[0].lower()
    assert "🔌 LLM Provider" in diag[0] or "Admin" in diag[0]
    # The hint should be on a continuation line (separated by →).
    assert "→" in diag[0]


def test_diagnostic_omits_classifier_hint_for_unknown_error():
    """If the classifier doesn't recognize the error, the diagnostic
    should NOT include a misleading hint."""
    client = _SeqClient([_Resp(False, error="garbled nonsense from the proxy")])
    diag: List[str] = []
    ir.regenerate(
        _PARENT_IDEA, "refine", n=1,
        claude_client=client, diagnostics=diag,
    )
    assert len(diag) == 1
    # Raw error preserved.
    assert "garbled nonsense" in diag[0]
    # No arrow continuation line.
    assert "→" not in diag[0]
