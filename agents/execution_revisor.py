"""execution_revisor.py — closing the probe → archive feedback loop.

The IdeaGraph paper's central unsolved problem is the −0.34 feasibility gap:
probes only check surface properties (text-level signals), so an idea can
look great on paper and still be infeasible to actually execute.

This module addresses that by running a *deliberately tiny* simulated
proxy of the proposed experiment whenever an idea passes the probes — a
1k-sample, single-seed, smaller-model pretend-run that an LLM acts as
domain expert for. The resulting feasibility signal then updates the
idea's quality_score via Bayesian credit-assignment so that probe-passing
ideas which look infeasible at execution time get demoted in the QD
archive, and probe-borderline ideas that look unexpectedly feasible at
small scale get a boost.

Design principles:
  • Lazy: only invoked for ideas that pass the probe stage (saves cost).
  • Cached by idea content hash (safe to call repeatedly during a session).
  • Bayesian: combines probe-prior with execution-likelihood using inverse-
    variance weighting. A 1k-sample result is noisier than a 100k-sample
    result, and the prior dominates appropriately when the proxy is weak.
  • Transparent: every component of the update is recorded on the Idea
    itself so the UI can show "probe said 0.72, exec said 0.45, trust
    weight 0.38, posterior 0.62".

Public API:
    revise(idea, claude_client=None, **opts) -> RevisionResult
    blend_score(probe_q, exec_signal, trust) -> float
    bayesian_blend(probe_q, exec_signal, n_samples, n_seeds) -> tuple[float, float]
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# ── Module-level cache (thread-safe) ─────────────────────────────────────
# Keyed by (idea content hash, sample_size, n_seeds). LLM is deterministic
# enough at temperature 0 that the same idea+config should produce the same
# proxy result.
_CACHE: Dict[str, "RevisionResult"] = {}
_CACHE_LOCK = threading.Lock()
_CACHE_MAX_ENTRIES = 256


# ── Default proxy execution config ──────────────────────────────────────
# These knobs encode "deliberately tiny" — we want a result we can run
# in seconds, not the real experiment.
DEFAULT_SAMPLE_SIZE: int = 1_000
DEFAULT_N_SEEDS: int = 1
DEFAULT_TARGET_SAMPLE_SIZE: int = 100_000  # what a "real" run would use

# Sentinel distinguishing "caller did not pass anything" (we should try to
# auto-load a Claude client) from "caller passed None on purpose" (run the
# degraded path without touching the network).
_AUTOLOAD = object()


@dataclass
class RevisionResult:
    """Outcome of running a tiny-experiment proxy on an idea."""

    # Inputs / scoring
    probe_quality: float = 0.0           # original probe-based quality
    execution_signal: float = 0.0        # 0..1, "how feasible did this look at small scale?"
    blended_quality: float = 0.0         # posterior — what to overwrite quality_score with
    trust_weight: float = 0.0            # 0..1, how much we leaned on execution vs probe
    delta: float = 0.0                   # blended - probe (+ means exec helped, − hurt)

    # Provenance / explanation
    metric_name: str = ""                # e.g. "AUROC", "F1", "perplexity-relative-improvement"
    predicted_metric: Optional[float] = None   # raw number the LLM predicted
    confidence_interval: Optional[tuple] = None  # (low, high) for predicted_metric
    failure_modes: list = field(default_factory=list)
    sample_size: int = DEFAULT_SAMPLE_SIZE
    n_seeds: int = DEFAULT_N_SEEDS

    # LLM provenance
    raw_response: str = ""
    cost_usd: float = 0.0
    used_llm: bool = True                # False when degraded path was used

    # State
    success: bool = True
    error: Optional[str] = None

    def summary(self) -> str:
        """One-line summary for logs/UI."""
        arrow = "↑" if self.delta > 0 else ("↓" if self.delta < 0 else "→")
        return (
            f"probe={self.probe_quality:.2f} {arrow} blend={self.blended_quality:.2f} "
            f"(exec={self.execution_signal:.2f}, trust={self.trust_weight:.2f}, "
            f"Δ{self.delta:+.2f})"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "probe_quality": self.probe_quality,
            "execution_signal": self.execution_signal,
            "blended_quality": self.blended_quality,
            "trust_weight": self.trust_weight,
            "delta": self.delta,
            "metric_name": self.metric_name,
            "predicted_metric": self.predicted_metric,
            "confidence_interval": list(self.confidence_interval)
                                    if self.confidence_interval else None,
            "failure_modes": self.failure_modes,
            "sample_size": self.sample_size,
            "n_seeds": self.n_seeds,
            "cost_usd": self.cost_usd,
            "used_llm": self.used_llm,
            "success": self.success,
            "error": self.error,
        }


# ── Credit-assignment math ──────────────────────────────────────────────
# How much do you trust a 1k-sample result?
#
# Inverse-variance Bayesian update:
#   posterior_mean = (probe_q / σ²_probe + exec_q / σ²_exec) / (1/σ²_probe + 1/σ²_exec)
#   posterior_var  = 1 / (1/σ²_probe + 1/σ²_exec)
#
# σ²_probe is fixed (probe judgements are noisy ~ 0.10 std).
# σ²_exec depends on:
#   - sample_size: variance scales as 1/n
#   - n_seeds: averaging k seeds reduces variance by k
#   - representativeness: fixed factor for "this is a tiny proxy" pessimism
#
# This means: a 1-seed 1k run gets weight ~0.30, a 5-seed 50k run gets ~0.85.

_PROBE_STD: float = 0.18         # noise level of a probe judgement (text-based)
_EXEC_BASE_STD: float = 0.025    # noise of a perfect-scale single-seed exec
_EXEC_PROXY_PENALTY: float = 1.0  # additional pessimism for "this is a small proxy"


def _exec_std(n_samples: int, n_seeds: int) -> float:
    """Standard deviation of an execution-based feasibility estimate."""
    n_samples = max(1, int(n_samples))
    n_seeds = max(1, int(n_seeds))
    target = DEFAULT_TARGET_SAMPLE_SIZE
    # Variance shrinks with sample size and number of seeds:
    sample_factor = math.sqrt(target / n_samples)
    seed_factor = 1.0 / math.sqrt(n_seeds)
    return _EXEC_BASE_STD * sample_factor * seed_factor * _EXEC_PROXY_PENALTY


def bayesian_blend(probe_q: float, exec_signal: float,
                    n_samples: int = DEFAULT_SAMPLE_SIZE,
                    n_seeds: int = DEFAULT_N_SEEDS) -> tuple:
    """Inverse-variance Bayesian blend of probe quality with exec signal.

    Returns (posterior_mean, trust_weight) where trust_weight is the
    fraction of the final estimate attributable to the execution signal.
    """
    probe_q = max(0.0, min(1.0, float(probe_q)))
    exec_signal = max(0.0, min(1.0, float(exec_signal)))
    var_probe = _PROBE_STD ** 2
    var_exec = _exec_std(n_samples, n_seeds) ** 2
    inv_p = 1.0 / var_probe
    inv_e = 1.0 / var_exec
    posterior_mean = (probe_q * inv_p + exec_signal * inv_e) / (inv_p + inv_e)
    trust_weight = inv_e / (inv_p + inv_e)
    return posterior_mean, trust_weight


def blend_score(probe_q: float, exec_signal: float, trust: float) -> float:
    """Linear blend (used for direct UI experiments where trust is hand-set)."""
    trust = max(0.0, min(1.0, float(trust)))
    return (1.0 - trust) * probe_q + trust * exec_signal


# ── Tiny-experiment LLM call ────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a domain-expert reviewer who has run tens of thousands of "
    "small-scale ML and scientific experiments. The user describes a research "
    "idea. Your job is to mentally simulate running a DELIBERATELY TINY "
    "version of it — typically 1,000 samples and a single random seed, on a "
    "smaller-than-target model — and report what such a run would most likely "
    "show. Be calibrated and concrete: predict an actual metric value, give a "
    "95% confidence interval, and list specific likely failure modes. Do not "
    "be optimistic for politeness. Output JSON exactly matching the requested "
    "schema and nothing else."
)


def _user_prompt(idea_dict: Dict[str, Any], n_samples: int, n_seeds: int) -> str:
    title = idea_dict.get("title", "")
    method = idea_dict.get("method", "")
    hypothesis = idea_dict.get("hypothesis", "")
    expected = idea_dict.get("expected_outcome", "")
    resources = idea_dict.get("resources", "")
    return (
        f"Idea title: {title}\n"
        f"Method: {method}\n"
        f"Hypothesis: {hypothesis}\n"
        f"Expected outcome: {expected}\n"
        f"Resources: {resources}\n\n"
        f"You will simulate running a tiny proxy: {n_samples} samples, "
        f"{n_seeds} random seed(s), reduced model size. Be concrete.\n\n"
        "Return JSON with these exact keys:\n"
        "{\n"
        '  "metric_name":         <string, e.g. "AUROC", "F1", "MSE", "perplexity">,\n'
        '  "predicted_metric":    <number — what the tiny run would most likely produce>,\n'
        '  "ci_low":              <number — 95% CI low for predicted_metric>,\n'
        '  "ci_high":             <number — 95% CI high for predicted_metric>,\n'
        '  "execution_signal":    <number 0..1 — overall feasibility-at-small-scale, '
        'where 1 = very likely to produce a usable signal that scales, '
        '0 = very likely to fail or produce a misleading result>,\n'
        '  "failure_modes":       <list of 1-4 short strings, concrete things that '
        'would likely go wrong>,\n'
        '  "would_scale":         <boolean — would a positive small-scale signal '
        'plausibly hold at full scale?>,\n'
        '  "rationale":           <one short sentence>\n'
        "}\n"
        "Output only the JSON object."
    )


def _idea_hash(idea_dict: Dict[str, Any], n_samples: int, n_seeds: int) -> str:
    payload = {
        "title": idea_dict.get("title", ""),
        "method": idea_dict.get("method", ""),
        "hypothesis": idea_dict.get("hypothesis", ""),
        "n_samples": n_samples,
        "n_seeds": n_seeds,
    }
    s = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:24]


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def _parse_response(raw: str) -> Dict[str, Any]:
    """Parse LLM response, tolerating fenced code blocks."""
    s = (raw or "").strip()
    # Strip code fences if present
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
    # Find the first '{' and last '}' to be tolerant
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        s = s[start:end + 1]
    return json.loads(s)


# ── Degraded path (no LLM available) ────────────────────────────────────
# When the Claude provider is missing or the call fails, we still produce
# a sensible result so the pipeline continues. The degraded estimate is
# derived from probe scores: if probes were strong, exec_signal trends
# toward probe_q; if probes were borderline, we apply a small pessimism
# toward 0.5 (regression to the mean of "tiny experiments often surprise").

def _degraded_signal(probe_q: float, probe_scores: Optional[Dict[str, float]]) -> float:
    """Cheap fallback when LLM is unavailable. Returns exec_signal in 0..1."""
    base = float(probe_q)
    if probe_scores:
        # Penalize execution-relevant probes that are weak
        for k in ("code", "dataset", "constraint", "scalability", "specificity"):
            v = probe_scores.get(k)
            if isinstance(v, (int, float)) and v < 0.5:
                base *= 0.85
    # Regression toward 0.5 (small experiments often disappoint surprises)
    return 0.7 * base + 0.3 * 0.5


# ── Main entry point ────────────────────────────────────────────────────

def revise(
    idea: Any,
    claude_client: Any = _AUTOLOAD,
    n_samples: int = DEFAULT_SAMPLE_SIZE,
    n_seeds: int = DEFAULT_N_SEEDS,
    use_cache: bool = True,
    max_tokens: int = 512,
    temperature: float = 0.2,
) -> RevisionResult:
    """Run the tiny-experiment proxy on an idea and return a RevisionResult.

    `idea` may be an Idea dataclass instance or a plain dict — anything
    with a .to_dict() method or already a dict.

    `claude_client`: pass a client object to use it; pass ``None`` to force
    the degraded (no-LLM) path; omit the argument to auto-load the global
    client lazily.
    """
    if hasattr(idea, "to_dict"):
        idea_dict = idea.to_dict()
    elif isinstance(idea, dict):
        idea_dict = dict(idea)
    else:
        return RevisionResult(success=False, error="invalid idea type",
                                used_llm=False)

    probe_q = _coerce_float(idea_dict.get("quality_score", 0.0))
    cache_key = _idea_hash(idea_dict, n_samples, n_seeds)

    if use_cache:
        with _CACHE_LOCK:
            cached = _CACHE.get(cache_key)
            if cached is not None:
                return cached

    # Auto-load only if the sentinel was left in place. Explicit None means
    # "skip the LLM" — important for tests and offline runs.
    if claude_client is _AUTOLOAD:
        try:
            from claude_provider import get_claude_client
            claude_client = get_claude_client()
        except Exception:
            claude_client = None

    if claude_client is None:
        # Degraded path: produce a sensible result without an LLM
        exec_signal = _degraded_signal(probe_q, idea_dict.get("probe_scores"))
        blended, trust = bayesian_blend(probe_q, exec_signal, n_samples, n_seeds)
        result = RevisionResult(
            probe_quality=probe_q,
            execution_signal=exec_signal,
            blended_quality=blended,
            trust_weight=trust,
            delta=blended - probe_q,
            metric_name="(degraded estimate)",
            sample_size=n_samples,
            n_seeds=n_seeds,
            used_llm=False,
            success=True,
            error="no LLM client available — used degraded heuristic",
        )
    else:
        try:
            response = claude_client.call(
                system=_SYSTEM_PROMPT,
                user=_user_prompt(idea_dict, n_samples, n_seeds),
                max_tokens=max_tokens,
                temperature=temperature,
                json_mode=True,
            )
            if not getattr(response, "success", False):
                raise RuntimeError(getattr(response, "error", "LLM call failed"))
            parsed = _parse_response(getattr(response, "text", ""))
            exec_signal = max(0.0, min(1.0,
                                         _coerce_float(parsed.get("execution_signal"), 0.5)))
            predicted = parsed.get("predicted_metric")
            ci_low = parsed.get("ci_low")
            ci_high = parsed.get("ci_high")
            ci = None
            if ci_low is not None and ci_high is not None:
                ci = (_coerce_float(ci_low), _coerce_float(ci_high))
            blended, trust = bayesian_blend(probe_q, exec_signal, n_samples, n_seeds)
            failures = parsed.get("failure_modes") or []
            if not isinstance(failures, list):
                failures = [str(failures)]
            failures = [str(x)[:200] for x in failures][:6]
            result = RevisionResult(
                probe_quality=probe_q,
                execution_signal=exec_signal,
                blended_quality=blended,
                trust_weight=trust,
                delta=blended - probe_q,
                metric_name=str(parsed.get("metric_name", "") or "")[:60],
                predicted_metric=(_coerce_float(predicted)
                                   if predicted is not None else None),
                confidence_interval=ci,
                failure_modes=failures,
                sample_size=n_samples,
                n_seeds=n_seeds,
                raw_response=getattr(response, "text", "")[:2000],
                cost_usd=getattr(response, "cost_usd", 0.0) or 0.0,
                used_llm=True,
                success=True,
            )
        except Exception as e:
            # Failure-tolerant degrade
            exec_signal = _degraded_signal(probe_q, idea_dict.get("probe_scores"))
            blended, trust = bayesian_blend(probe_q, exec_signal, n_samples, n_seeds)
            result = RevisionResult(
                probe_quality=probe_q,
                execution_signal=exec_signal,
                blended_quality=blended,
                trust_weight=trust,
                delta=blended - probe_q,
                metric_name="(LLM failed → degraded)",
                sample_size=n_samples,
                n_seeds=n_seeds,
                used_llm=False,
                success=True,
                error=f"LLM error: {e}",
            )

    if use_cache:
        with _CACHE_LOCK:
            if len(_CACHE) >= _CACHE_MAX_ENTRIES:
                # Drop oldest 25% to keep cache bounded
                drop = list(_CACHE.keys())[: _CACHE_MAX_ENTRIES // 4]
                for k in drop:
                    _CACHE.pop(k, None)
            _CACHE[cache_key] = result

    return result


def clear_cache() -> None:
    """Clear the module-level cache (useful in tests)."""
    with _CACHE_LOCK:
        _CACHE.clear()


def cache_size() -> int:
    with _CACHE_LOCK:
        return len(_CACHE)
