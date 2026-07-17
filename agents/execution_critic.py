"""
agents/execution_critic.py - Four execution probes that evaluate research ideas
for feasibility before they enter the QD archive.

Probes (now combined into ONE LLM call instead of 4):
  1. Code skeleton probe   - can core method be implemented?
  2. Dataset probe         - are required datasets accessible?
  3. Compute constraint    - does it fit an academic compute budget?
  4. Novelty probe         - is it sufficiently differentiated?

Optimisations vs original:
  - Single combined LLM call returns all 4 scores → 75% fewer API calls
  - LRU probe result cache: frequently-accessed results stay cached longer
  - Early-exit: quality < 0.25 after combined probe → no revision worth trying
  - Novelty-weighted quality: novelty=0.35, code=0.25, dataset/constraint=0.20
    Novelty is the primary discriminator of research value; rewarding it more
    means the QD archive fills with genuinely interesting ideas, not just
    computationally easy ones.
Pass threshold: every probe >= 0.4
Quality = weighted average (novelty-weighted)
"""

from __future__ import annotations
import hashlib
import threading
from collections import OrderedDict
from typing import Any, Dict

from models.idea import Idea
from .base_agent import BaseAgent


_PASS_THRESHOLD = 0.4

# ── 10-Dimension Quality Rubric ──────────────────────────────────────────────
# Goes far beyond 4 basic probes — evaluates ideas like a real reviewer would.
_WEIGHTS = {
    # Core 4 (original probes, rebalanced)
    "code": 0.10,           # can it be implemented?
    "dataset": 0.10,        # is data available?
    "constraint": 0.08,     # fits academic budget?
    "novelty": 0.20,        # is it genuinely new?
    # New 6 dimensions
    "specificity": 0.12,    # concrete vs vague?
    "significance": 0.12,   # does it matter if solved?
    "clarity": 0.08,        # is it clearly stated?
    "testability": 0.08,    # can the hypothesis be verified?
    "scalability": 0.06,    # does it generalize?
    "risk_balance": 0.06,   # are risks identified and mitigated?
}
# Total = 1.00

# Minimum specificity score to accept (rejects vague ideas)
_SPECIFICITY_THRESHOLD = 0.35

# ── Probe result cache (LRU) ──────────────────────────────────────────────────
# Keyed by MD5 of idea content; avoids re-probing near-identical ideas.
# LRU eviction keeps the results we actually re-use (e.g. revision round-trips).
_PROBE_CACHE: OrderedDict[str, Dict[str, Any]] = OrderedDict()
_PROBE_CACHE_MAX = 256  # bumped from 128 — revision loops can generate ~240 unique probes
_PROBE_CACHE_LOCK = threading.Lock()


def _idea_cache_key(idea: Idea) -> str:
    raw = f"{idea.title}\x00{idea.method}\x00{idea.hypothesis}"
    return hashlib.md5(raw.encode(), usedforsecurity=False).hexdigest()


class ExecutionCritic(BaseAgent):

    def __init__(self) -> None:
        super().__init__(temperature=0.0)
        self._ctx_lock = threading.Lock()
        self._dag_papers: list = []  # injected by pipeline for grounded novelty
        self._archived_methods: list = []  # injected by pipeline for dedup

    def set_dag_context(self, paper_titles: list) -> None:
        """Inject DAG paper titles for grounded novelty evaluation (thread-safe)."""
        with self._ctx_lock:
            self._dag_papers = paper_titles[:10]

    def set_archived_methods(self, methods: list) -> None:
        """Inject archived idea methods for novelty comparison (thread-safe)."""
        with self._ctx_lock:
            self._archived_methods = methods[:15]

    def _snapshot_context(self) -> tuple:
        """Return a snapshot of (dag_papers, archived_methods) under the lock."""
        with self._ctx_lock:
            return (list(self._dag_papers), list(self._archived_methods))

    # ─────────────────────────────────────────────────────────────────────────
    # Master probe runner — single combined LLM call
    # ─────────────────────────────────────────────────────────────────────────
    def probe_all(self, idea: Idea) -> Dict[str, Any]:
        """
        Evaluate all 4 probes in ONE LLM call and aggregate results.

        Benefits over 4 parallel calls:
          - Sends idea text once instead of 4 times (~75% fewer input tokens)
          - 1 API round-trip instead of 4 concurrent ones (no rate-limit pressure)
          - Result is cached by idea content for free de-duplication

        Returns:
          {
            "all_pass": bool,
            "scores": {probe: float, ...},
            "details": {probe: dict, ...},
            "feedback": str,
            "quality": float,
          }
        """
        # ── Cache lookup (LRU) ───────────────────────────────────────────────
        ck = _idea_cache_key(idea)
        with _PROBE_CACHE_LOCK:
            if ck in _PROBE_CACHE:
                _PROBE_CACHE.move_to_end(ck)  # mark as recently used
                return _PROBE_CACHE[ck]

        # ── Heuristic shortcut: skip the LLM call for obviously bad ideas ─────
        try:
            import config as _cfg_sc
            if getattr(_cfg_sc, "ENABLE_PROBE_SHORTCUT", True):
                from speed_optimizer import quick_probe_shortcut
                _shortcut = quick_probe_shortcut({
                    "title": idea.title, "method": idea.method,
                    "hypothesis": idea.hypothesis,
                })
                if _shortcut is not None:
                    with _PROBE_CACHE_LOCK:
                        if len(_PROBE_CACHE) >= _PROBE_CACHE_MAX:
                            _PROBE_CACHE.popitem(last=False)
                        _PROBE_CACHE[ck] = _shortcut
                    return _shortcut
        except Exception:
            pass  # fall through to real probe on any error

        # ── Single combined LLM call ──────────────────────────────────────────
        # Build DAG-grounded novelty context (thread-safe snapshot)
        dag_papers, archived_methods = self._snapshot_context()
        novelty_grounding = ""
        if dag_papers:
            novelty_grounding = (
                "\n\nFor the NOVELTY assessment, compare against these known papers in the field:\n"
                + "\n".join(f"  - {t}" for t in dag_papers[:5])
            )
        if archived_methods:
            novelty_grounding += (
                "\n\nAlso compare against these already-generated idea methods (lower novelty if similar):\n"
                + "\n".join(f"  - {m[:80]}" for m in archived_methods[:5])
            )

        system = (
            "You are an expert research evaluator (NeurIPS/ICML reviewer level). "
            "Assess the research idea below across TEN dimensions.\n\n"
            "Return ONLY valid JSON with this structure:\n"
            '{\n'
            '  "code": {"score": 0.0-1.0, "issues": "...", "suggestions": "..."},\n'
            '  "dataset": {"score": 0.0-1.0, "datasets_found": [...], "issues": "...", "suggestions": "..."},\n'
            '  "constraint": {"score": 0.0-1.0, "estimated_compute": "...", "issues": "...", "suggestions": "..."},\n'
            '  "novelty": {"score": 0.0-1.0, "closest_work": "...", "issues": "...", "suggestions": "..."},\n'
            '  "specificity": {"score": 0.0-1.0, "issues": "..."},\n'
            '  "significance": {"score": 0.0-1.0, "issues": "..."},\n'
            '  "clarity": {"score": 0.0-1.0, "issues": "..."},\n'
            '  "testability": {"score": 0.0-1.0, "issues": "..."},\n'
            '  "scalability": {"score": 0.0-1.0, "issues": "..."},\n'
            '  "risk_balance": {"score": 0.0-1.0, "issues": "..."}\n'
            '}\n\n'
            "Scoring guide:\n"
            "  code:         1.0=clearly implementable in Python, 0.0=not implementable\n"
            "  dataset:      1.0=all data publicly available, 0.0=no public data exists\n"
            "  constraint:   1.0=fits 4xA100 1-month budget (~$5k), 0.0=industrial-scale only\n"
            "  novelty:      1.0=highly novel (NeurIPS-worthy), 0.0=already done\n"
            "  specificity:  1.0=concrete method with specific algorithms/datasets/metrics named, 0.0=vague hand-waving\n"
            "  significance: 1.0=solving this would advance the entire field, 0.0=trivial/incremental\n"
            "  clarity:      1.0=anyone in the field would understand it immediately, 0.0=confusing/ambiguous\n"
            "  testability:  1.0=hypothesis is clearly falsifiable with a specific experiment, 0.0=unfalsifiable\n"
            "  scalability:  1.0=method generalizes to many domains/scales, 0.0=only works on toy example\n"
            "  risk_balance: 1.0=risks identified with concrete mitigations, 0.0=no risk awareness"
            f"{novelty_grounding}"
        )
        user = (
            f"Research idea:\n{idea.to_prompt_str()}\n\n"
            "Evaluate all TEN dimensions. Be strict — only give high scores for genuinely excellent work. "
            "Score specificity LOW if the method is vague or lacks concrete technical details."
        )

        raw = self._call_json(system, user, max_tokens=768)

        # ── Parse combined result (10 dimensions) ─────────────────────────────
        all_dimensions = [
            "code", "dataset", "constraint", "novelty",
            "specificity", "significance", "clarity", "testability",
            "scalability", "risk_balance",
        ]

        parsed_results = {}
        for dim in all_dimensions:
            extra_keys = []
            if dim == "dataset":
                extra_keys = ["datasets_found"]
            elif dim == "constraint":
                extra_keys = ["estimated_compute"]
            elif dim == "novelty":
                extra_keys = ["closest_work"]
            parsed_results[dim] = _safe_probe_result(
                raw.get(dim, {}),
                keys=["score", "issues", "suggestions"] + extra_keys,
            )

        scores = {dim: parsed_results[dim].get("score", 0.0) for dim in all_dimensions}

        # Core 4 probes must pass threshold; new dimensions are quality boosters
        core_pass = all(scores.get(d, 0) >= _PASS_THRESHOLD for d in ["code", "dataset", "constraint", "novelty"])
        # Specificity gate: reject vague ideas
        specificity_pass = scores.get("specificity", 0.5) >= _SPECIFICITY_THRESHOLD
        all_pass = core_pass and specificity_pass

        quality = self.compute_quality(scores)

        feedback_parts = []
        dim_labels = {
            "code": "Code", "dataset": "Dataset", "constraint": "Compute",
            "novelty": "Novelty", "specificity": "Specificity",
            "significance": "Significance", "clarity": "Clarity",
            "testability": "Testability", "scalability": "Scalability",
            "risk_balance": "Risk Balance",
        }
        for dim in all_dimensions:
            result = parsed_results[dim]
            issues = result.get("issues", "")
            suggestions = result.get("suggestions", "")
            if issues:
                feedback_parts.append(f"{dim_labels.get(dim, dim)}: {issues}")
            if suggestions:
                feedback_parts.append(f"  -> {suggestions}")

        # Special feedback for specificity failure
        if not specificity_pass:
            feedback_parts.insert(0,
                f"REJECTED: Idea is too vague (specificity={scores.get('specificity', 0):.2f} < {_SPECIFICITY_THRESHOLD}). "
                "Add concrete algorithm names, specific datasets, exact metrics, and quantitative targets."
            )

        output = {
            "all_pass": all_pass,
            "scores": scores,
            "details": parsed_results,
            "feedback": "\n".join(feedback_parts) if feedback_parts else "All probes passed.",
            "quality": quality,
        }

        # ── Cache store (LRU evict if full) ──────────────────────────────────
        with _PROBE_CACHE_LOCK:
            if ck in _PROBE_CACHE:
                _PROBE_CACHE.move_to_end(ck)
            else:
                if len(_PROBE_CACHE) >= _PROBE_CACHE_MAX:
                    _PROBE_CACHE.popitem(last=False)  # evict least-recently-used
                _PROBE_CACHE[ck] = output

        return output

    # ── Keep individual probe methods for backward compatibility ──────────────
    def probe_code(self, idea: Idea) -> Dict[str, Any]:
        return self.probe_all(idea)["details"]["code"]

    def probe_dataset(self, idea: Idea) -> Dict[str, Any]:
        return self.probe_all(idea)["details"]["dataset"]

    def probe_constraint(self, idea: Idea) -> Dict[str, Any]:
        return self.probe_all(idea)["details"]["constraint"]

    def probe_novelty(self, idea: Idea) -> Dict[str, Any]:
        return self.probe_all(idea)["details"]["novelty"]

    # ─────────────────────────────────────────────────────────────────────────
    # Quality aggregation
    # ─────────────────────────────────────────────────────────────────────────
    def compute_quality(self, probe_results: Dict[str, float]) -> float:
        """Weighted average of probe scores (equal weights = 0.25 each)."""
        total = sum(
            _WEIGHTS.get(k, 0.25) * float(v)
            for k, v in probe_results.items()
        )
        return round(min(max(total, 0.0), 1.0), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────
def _safe_probe_result(raw: dict, keys: list) -> Dict[str, Any]:
    """Ensure all expected keys exist with safe defaults."""
    result: Dict[str, Any] = {}
    for k in keys:
        if k == "score":
            val = raw.get(k, 0.5)
            try:
                result[k] = float(val)
            except (TypeError, ValueError):
                result[k] = 0.5
        elif k in ("feasible", "novel"):
            result[k] = bool(raw.get(k, True))
        elif k == "datasets_found":
            val = raw.get(k, [])
            result[k] = val if isinstance(val, list) else []
        else:
            result[k] = str(raw.get(k, ""))
    return result
