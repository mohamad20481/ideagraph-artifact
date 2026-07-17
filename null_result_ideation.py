"""
null_result_ideation.py — design experiments whose deliverable is a clean negative.

Unlike `heretic` (attacks consensus claims rhetorically) and `failure_mode`
(designs robustness into a method), this mode treats *clean negative results*
as the primary research product. The output is a study whose success
condition is finding that something widely assumed *doesn't* work — on a
specific cohort, under a specific condition, beyond a specific scale.

Two phases:
  1. propose_null_target(topic, kind) — LLM names a specific claim/effect
     that should be tested for being null: "does X transfer to Y?",
     "does method M's reported gain hold on cohort C?", "does the effect
     scale linearly past N?".
  2. design_null_experiment(topic, target) — LLM designs an experiment
     whose primary deliverable is a well-powered negative — including
     the power analysis, the equivalence margin, and what registering
     this as a null result would change in the field.

Ideas have `source_strategy='Z'` (Zero/null). The null target lives on
`execution_meta.null_target`. A measured negative is a real contribution
only when the study is pre-registered, powered, and bounded — the prompt
enforces all three.

Public API:
    NullTarget                                       → dataclass
    NULL_KINDS                                       → catalog
    propose_null_target(topic, kind, ...)            → Optional[NullTarget]
    design_null_experiment(topic, target, ...)       → Optional[Idea]
    null_result_batch(topic, ..., n)                 → List[Idea]
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from models.idea import Idea, METHODOLOGY_TYPES, NOVELTY_LEVELS


_AUTOLOAD = object()


# Different categories of null target. The LLM picks ONE category and
# instantiates it concretely for the topic.
NULL_KINDS: List[str] = [
    "transfer_failure",         # method M, claimed for setting A, fails on setting B
    "scaling_plateau",          # the well-known scaling relation stops working past N
    "cohort_invalidity",        # the established effect doesn't hold on a specific cohort
    "ablation_unimportant",     # a component everyone keeps actually contributes ~0
    "feature_irrelevance",      # an assumed-load-bearing feature is statistically irrelevant
    "method_equivalence",       # an expensive method gives equivalent results to a cheap one
]


@dataclass
class NullTarget:
    """A specific claim/effect to test for being null."""
    kind: str = ""                  # one of NULL_KINDS
    claim_to_be_negated: str = ""   # the assertion the study would refute
    why_widely_assumed: str = ""    # how this assumption became default
    why_doubt_now: str = ""         # honest reason to suspect it's actually null
    population: str = ""            # the cohort/regime where the null would apply
    equivalence_margin: str = ""    # quantified margin under which "no effect" is declared
    stakes: float = 0.5             # 0..1 how much a confirmed null would change practice

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "claim_to_be_negated": self.claim_to_be_negated,
            "why_widely_assumed": self.why_widely_assumed,
            "why_doubt_now": self.why_doubt_now,
            "population": self.population,
            "equivalence_margin": self.equivalence_margin,
            "stakes": self.stakes,
        }


_TARGET_SYSTEM = (
    "You are an evaluation methodologist who specializes in null results. "
    "Given a research topic and a category of null target, name ONE "
    "specific, widely-assumed claim or effect in this topic that — if "
    "tested rigorously — might turn out to be null on a defined cohort or "
    "regime. State the claim concretely, the population where the null "
    "would apply, and the equivalence margin under which 'no effect' is "
    "declared. Avoid strawmen; the claim must be one the field actually "
    "treats as default. Return ONLY valid JSON."
)


def _target_user_prompt(topic: str, kind: str) -> str:
    return (
        f"Topic: {topic}\n"
        f"Null target category: {kind}\n\n"
        f"Name ONE concrete, currently-assumed claim or effect in this "
        f"topic that fits the category and might plausibly be null on a "
        f"defined cohort/regime. Quantify the equivalence margin — the "
        f"effect size below which we declare 'no meaningful effect'. Be "
        f"honest about why you doubt the canonical view.\n\n"
        f"Return JSON:\n"
        "{\n"
        '  "claim_to_be_negated": "<the assertion the study would refute, in one sentence>",\n'
        '  "why_widely_assumed": "<how this assumption became default>",\n'
        '  "why_doubt_now": "<honest reason to suspect it might be null>",\n'
        '  "population": "<cohort/regime where the null would apply>",\n'
        '  "equivalence_margin": "<concrete: e.g. \\"|Δaccuracy| < 1% on a 95% CI\\">",\n'
        '  "stakes": <0..1 — how much a confirmed null would shift field practice>\n'
        "}"
    )


def _parse_json(raw: str) -> Optional[Dict[str, Any]]:
    s = (raw or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
    start, end = s.find("{"), s.rfind("}")
    if start >= 0 and end > start:
        s = s[start:end + 1]
    try:
        return json.loads(s)
    except Exception:
        return None


def _coerce_unit(x: Any, default: float = 0.5) -> float:
    try:
        v = float(x)
        if v != v or v == float("inf") or v == float("-inf"):
            return default
        return max(0.0, min(1.0, v))
    except (TypeError, ValueError):
        return default


def propose_null_target(
    topic: str,
    kind: str,
    claude_client: Any = _AUTOLOAD,
    max_tokens: int = 700,
    temperature: float = 0.7,
) -> Optional[NullTarget]:
    if not topic or not topic.strip():
        raise ValueError("topic must be non-empty")
    if kind not in NULL_KINDS:
        raise ValueError(f"kind must be one of {NULL_KINDS}, got {kind!r}")
    if claude_client is _AUTOLOAD:
        try:
            from claude_provider import get_claude_client
            claude_client = get_claude_client()
        except Exception:
            claude_client = None
    if claude_client is None:
        return None
    try:
        resp = claude_client.call(
            system=_TARGET_SYSTEM,
            user=_target_user_prompt(topic.strip(), kind),
            max_tokens=max_tokens, temperature=temperature, json_mode=True,
        )
    except Exception:
        return None
    if not getattr(resp, "success", False):
        return None
    parsed = _parse_json(getattr(resp, "text", ""))
    if not parsed:
        return None
    claim = str(parsed.get("claim_to_be_negated") or "").strip()
    margin = str(parsed.get("equivalence_margin") or "").strip()
    if not (claim and margin):
        return None
    return NullTarget(
        kind=kind,
        claim_to_be_negated=claim[:400],
        why_widely_assumed=str(parsed.get("why_widely_assumed") or "")[:400],
        why_doubt_now=str(parsed.get("why_doubt_now") or "")[:400],
        population=str(parsed.get("population") or "")[:300],
        equivalence_margin=margin[:300],
        stakes=_coerce_unit(parsed.get("stakes"), 0.5),
    )


_DESIGN_SYSTEM = (
    "You are an evaluation methodologist designing a pre-registered null-"
    "result study. The user gives you a topic and a specific null target. "
    "Design ONE study whose PRIMARY deliverable is a well-powered "
    "negative result. The design must include (a) a pre-registered "
    "hypothesis and analysis, (b) a power analysis showing the study can "
    "detect the equivalence margin, and (c) explicit acceptance criteria "
    "for declaring 'null'. Name all three explicitly in the method. "
    "Output ONLY valid JSON. methodology_type must be one of: "
    f"{', '.join(METHODOLOGY_TYPES)}. novelty_level must be one of: "
    f"{', '.join(NOVELTY_LEVELS)}."
)


def _design_user_prompt(topic: str, target: NullTarget) -> str:
    return (
        f"Topic: {topic}\n\n"
        f"### Null target (deliverable: a clean negative)\n"
        f"  Category:            {target.kind}\n"
        f"  Claim to be negated: {target.claim_to_be_negated}\n"
        f"  Why widely assumed:  {target.why_widely_assumed}\n"
        f"  Why doubt now:       {target.why_doubt_now}\n"
        f"  Population:          {target.population}\n"
        f"  Equivalence margin:  {target.equivalence_margin}\n"
        f"  Stakes:              {target.stakes:.2f}\n\n"
        f"### Instructions\n"
        f"Design ONE study whose primary deliverable is a well-powered, "
        f"pre-registered NEGATIVE result. The success condition is "
        f"finding the effect is null within the equivalence margin. The "
        f"method MUST name: (a) the pre-registered hypothesis & analysis "
        f"plan, (b) the power analysis showing the study can detect the "
        f"margin, (c) the acceptance criteria for declaring null.\n\n"
        f"### Output\n"
        "Return JSON:\n"
        "{\n"
        '  "title": "<concise; signals the null as the deliverable>",\n'
        '  "motivation": "<why a clean null here would change the field>",\n'
        '  "method": "<the pre-registered design, power analysis, and acceptance criteria>",\n'
        '  "hypothesis": "<null hypothesis stated formally, with margin>",\n'
        '  "resources": "<resources for adequate power>",\n'
        '  "expected_outcome": "<the null result and what its CI must look like>",\n'
        '  "risk_assessment": "<risks if the study under-powers OR if a positive sneaks in>",\n'
        '  "source_strategy": "Z",\n'
        f'  "methodology_type": "<one of: {", ".join(METHODOLOGY_TYPES)}>",\n'
        f'  "novelty_level": "<one of: {", ".join(NOVELTY_LEVELS)}>",\n'
        '  "null_acceptance_criteria": "<one sentence: exactly when we declare null>",\n'
        '  "power_analysis_summary": "<one sentence: N required + effect size detectable>"\n'
        "}"
    )


def _build_idea(parsed: Dict[str, Any], topic: str,
                  target: NullTarget) -> Optional[Idea]:
    if not all(str(parsed.get(k, "")).strip() for k in ("title", "method", "hypothesis")):
        return None
    method_type = parsed.get("methodology_type") or ""
    if method_type not in METHODOLOGY_TYPES:
        method_type = None
    novelty = parsed.get("novelty_level") or ""
    if novelty not in NOVELTY_LEVELS:
        novelty = None
    idea = Idea(
        title=str(parsed.get("title", ""))[:200],
        motivation=str(parsed.get("motivation", ""))[:1000],
        method=str(parsed.get("method", ""))[:2000],
        hypothesis=str(parsed.get("hypothesis", ""))[:1000],
        resources=str(parsed.get("resources", ""))[:500],
        expected_outcome=str(parsed.get("expected_outcome", ""))[:500],
        risk_assessment=str(parsed.get("risk_assessment", ""))[:500],
        source_strategy="Z",
        methodology_type=method_type,
        novelty_level=novelty,
        generation=0,
        parent_title=None,
    )
    idea.execution_meta = {
        "null_target": target.to_dict(),
        "null_acceptance_criteria": str(
            parsed.get("null_acceptance_criteria", "")
        )[:400],
        "power_analysis_summary": str(
            parsed.get("power_analysis_summary", "")
        )[:400],
        "regen_mode": "null_result",
        "topic": topic,
    }
    return idea


def design_null_experiment(
    topic: str,
    target: NullTarget,
    claude_client: Any = _AUTOLOAD,
    max_tokens: int = 1000,
    temperature: float = 0.6,
) -> Optional[Idea]:
    if not topic or not topic.strip():
        raise ValueError("topic must be non-empty")
    if claude_client is _AUTOLOAD:
        try:
            from claude_provider import get_claude_client
            claude_client = get_claude_client()
        except Exception:
            claude_client = None
    if claude_client is None:
        return None
    try:
        resp = claude_client.call(
            system=_DESIGN_SYSTEM,
            user=_design_user_prompt(topic.strip(), target),
            max_tokens=max_tokens, temperature=temperature, json_mode=True,
        )
    except Exception:
        return None
    if not getattr(resp, "success", False):
        return None
    parsed = _parse_json(getattr(resp, "text", ""))
    if not parsed:
        return None
    return _build_idea(parsed, topic.strip(), target)


def null_result_batch(
    topic: str,
    claude_client: Any = _AUTOLOAD,
    n: int = 3,
    kinds: Optional[List[str]] = None,
    seed: Optional[int] = None,
) -> List[Idea]:
    """End-to-end: pick `n` distinct null kinds, propose a target for each,
    design the negative-result study. `kinds` overrides the random pick."""
    if not topic or not topic.strip():
        raise ValueError("topic must be non-empty")
    if n <= 0:
        return []
    if kinds is None:
        pool = list(NULL_KINDS)
        rng = random.Random(seed) if seed is not None else random
        rng.shuffle(pool)
        kinds = pool[:max(1, n)]
    for k in kinds:
        if k not in NULL_KINDS:
            raise ValueError(f"invalid null kind {k!r}")
    out: List[Idea] = []
    for k in kinds[:n]:
        target = propose_null_target(topic, k, claude_client=claude_client)
        if target is None:
            continue
        idea = design_null_experiment(topic, target, claude_client=claude_client)
        if idea is not None:
            out.append(idea)
    return out
