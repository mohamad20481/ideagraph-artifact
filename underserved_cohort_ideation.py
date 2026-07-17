"""
underserved_cohort_ideation.py — research aimed at populations the field has skipped.

Most novelty modes fix a *what* (a method, a constraint, a regime). This
mode fixes a *who*: identify a specific user-group or use-case that the
canonical literature systematically under-serves, and design research
that would explicitly serve them.

Two phases:
  1. identify_cohort(topic, dimension) — LLM names a specific cohort
     (low-resource language community, edge-device users, post-conflict
     researchers, screen-reader users, hospital nightshift, etc.) and
     why the canonical pipeline fails them.
  2. design_for_cohort(topic, cohort) — LLM designs research whose
     evaluation, dataset, and success metrics are anchored to that
     cohort, not to a generic benchmark.

Ideas have `source_strategy='W'` (Who-driven). The cohort lives on
`execution_meta.cohort`. The prompt requires the cohort to be specific
and the evaluation to be cohort-anchored — otherwise we're back to
generic benchmark optimization.

Public API:
    Cohort                                            → dataclass
    COHORT_DIMENSIONS                                 → catalog
    identify_cohort(topic, dimension, ...)            → Optional[Cohort]
    design_for_cohort(topic, cohort, ...)             → Optional[Idea]
    underserved_cohort_batch(topic, ..., n)           → List[Idea]
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from models.idea import Idea, METHODOLOGY_TYPES, NOVELTY_LEVELS


_AUTOLOAD = object()


# Different lenses for finding an underserved cohort. The LLM picks ONE
# dimension and instantiates a specific cohort under it.
COHORT_DIMENSIONS: List[str] = [
    "linguistic",            # low-resource languages, code-switching, dialects
    "infrastructural",       # intermittent network, edge devices, no GPU
    "economic",              # users in low-GDP regions, free-tier-only
    "physical_ability",      # screen-reader users, motor-impaired, visually impaired
    "geopolitical",          # post-conflict, sanctioned, displaced populations
    "professional",          # under-resourced clinics, public-defenders, smallholder farmers
    "temporal",              # users on shift schedules, time-pressured workflows
    "demographic",           # age cohorts the field tests least (children, elderly)
]


@dataclass
class Cohort:
    """A specific under-served user group / use case."""
    dimension: str = ""              # one of COHORT_DIMENSIONS
    name: str = ""                   # short label (e.g. "Levantine Arabic speakers")
    description: str = ""            # who they are, scale, and where they live
    why_underserved: str = ""        # what the canonical pipeline assumes that fails here
    canonical_failure_mode: str = "" # what concretely breaks when canonical methods meet them
    success_metric: str = ""         # cohort-anchored metric that matters to them
    overlooked_factor: float = 0.5   # 0..1 how badly the literature ignores them

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "name": self.name,
            "description": self.description,
            "why_underserved": self.why_underserved,
            "canonical_failure_mode": self.canonical_failure_mode,
            "success_metric": self.success_metric,
            "overlooked_factor": self.overlooked_factor,
        }


_COHORT_SYSTEM = (
    "You are a researcher who specializes in equitable evaluation. Given "
    "a topic and a cohort dimension, identify ONE specific real-world "
    "cohort the canonical literature systematically under-serves. Be "
    "concrete: name the cohort, describe scale and location, explain "
    "what assumption in the canonical pipeline fails for them, and "
    "propose a cohort-anchored success metric. Avoid vague gestures "
    "toward 'marginalized users' — name the group. Return ONLY valid JSON."
)


def _cohort_user_prompt(topic: str, dimension: str) -> str:
    return (
        f"Topic: {topic}\n"
        f"Cohort dimension: {dimension}\n\n"
        f"Identify ONE specific cohort along this dimension that the "
        f"current literature under-serves for this topic. Be concrete "
        f"about scale, geography, and what the canonical pipeline "
        f"assumes that breaks for them. Propose a success metric anchored "
        f"to *their* needs, not a generic benchmark.\n\n"
        f"Return JSON:\n"
        "{\n"
        '  "name": "<short, specific label, e.g. \\"Levantine Arabic speakers in NW Syria\\">",\n'
        '  "description": "<who they are, rough scale, where they live or work>",\n'
        '  "why_underserved": "<assumption in the canonical pipeline that fails them>",\n'
        '  "canonical_failure_mode": "<concretely what breaks when default methods meet them>",\n'
        '  "success_metric": "<cohort-anchored metric, e.g. \\"task success on dialectal voice queries\\">",\n'
        '  "overlooked_factor": <0..1 — how badly the literature ignores them>\n'
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


def identify_cohort(
    topic: str,
    dimension: str,
    claude_client: Any = _AUTOLOAD,
    max_tokens: int = 700,
    temperature: float = 0.75,
) -> Optional[Cohort]:
    if not topic or not topic.strip():
        raise ValueError("topic must be non-empty")
    if dimension not in COHORT_DIMENSIONS:
        raise ValueError(
            f"dimension must be one of {COHORT_DIMENSIONS}, got {dimension!r}"
        )
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
            system=_COHORT_SYSTEM,
            user=_cohort_user_prompt(topic.strip(), dimension),
            max_tokens=max_tokens, temperature=temperature, json_mode=True,
        )
    except Exception:
        return None
    if not getattr(resp, "success", False):
        return None
    parsed = _parse_json(getattr(resp, "text", ""))
    if not parsed:
        return None
    name = str(parsed.get("name") or "").strip()
    metric = str(parsed.get("success_metric") or "").strip()
    if not (name and metric):
        return None
    return Cohort(
        dimension=dimension,
        name=name[:200],
        description=str(parsed.get("description") or "")[:400],
        why_underserved=str(parsed.get("why_underserved") or "")[:400],
        canonical_failure_mode=str(
            parsed.get("canonical_failure_mode") or ""
        )[:400],
        success_metric=metric[:400],
        overlooked_factor=_coerce_unit(parsed.get("overlooked_factor"), 0.5),
    )


_DESIGN_SYSTEM = (
    "You are a researcher designing research for a specific underserved "
    "cohort. The user gives you a topic and a concrete cohort. Design "
    "ONE study whose evaluation, dataset choice, and success metric are "
    "all anchored to the cohort — not a generic benchmark. The method "
    "must include (a) how the cohort is concretely represented in the "
    "data, (b) the cohort-anchored success metric, and (c) the failure "
    "mode of canonical methods on this cohort that motivates the work. "
    "Output ONLY valid JSON. methodology_type must be one of: "
    f"{', '.join(METHODOLOGY_TYPES)}. novelty_level must be one of: "
    f"{', '.join(NOVELTY_LEVELS)}."
)


def _design_user_prompt(topic: str, cohort: Cohort) -> str:
    return (
        f"Topic: {topic}\n\n"
        f"### Under-served cohort\n"
        f"  Dimension:           {cohort.dimension}\n"
        f"  Name:                {cohort.name}\n"
        f"  Description:         {cohort.description}\n"
        f"  Why under-served:    {cohort.why_underserved}\n"
        f"  Canonical failure:   {cohort.canonical_failure_mode}\n"
        f"  Success metric:      {cohort.success_metric}\n"
        f"  Overlooked factor:   {cohort.overlooked_factor:.2f}\n\n"
        f"### Instructions\n"
        f"Design ONE study anchored to this cohort. Evaluation, dataset, "
        f"and success metric must all be cohort-anchored — not a generic "
        f"benchmark. State explicitly how the cohort is represented in "
        f"the data and what makes the success metric meaningful to them.\n\n"
        f"### Output\n"
        "Return JSON:\n"
        "{\n"
        '  "title": "<concise; names the cohort or their use case>",\n'
        '  "motivation": "<why this cohort matters and why canonical work skips them>",\n'
        '  "method": "<cohort-anchored design: data, evaluation, success metric>",\n'
        '  "hypothesis": "<falsifiable cohort-specific claim>",\n'
        '  "resources": "<resources including any community partnerships>",\n'
        '  "expected_outcome": "<cohort-anchored measurable outcome>",\n'
        '  "risk_assessment": "<technical risks AND ethical/representational risks>",\n'
        '  "source_strategy": "W",\n'
        f'  "methodology_type": "<one of: {", ".join(METHODOLOGY_TYPES)}>",\n'
        f'  "novelty_level": "<one of: {", ".join(NOVELTY_LEVELS)}>",\n'
        '  "cohort_representation": "<one sentence: how the cohort is concretely in the data>"\n'
        "}"
    )


def _build_idea(parsed: Dict[str, Any], topic: str,
                  cohort: Cohort) -> Optional[Idea]:
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
        source_strategy="W",
        methodology_type=method_type,
        novelty_level=novelty,
        generation=0,
        parent_title=None,
    )
    idea.execution_meta = {
        "cohort": cohort.to_dict(),
        "cohort_representation": str(
            parsed.get("cohort_representation", "")
        )[:400],
        "regen_mode": "underserved_cohort",
        "topic": topic,
    }
    return idea


def design_for_cohort(
    topic: str,
    cohort: Cohort,
    claude_client: Any = _AUTOLOAD,
    max_tokens: int = 1000,
    temperature: float = 0.7,
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
            user=_design_user_prompt(topic.strip(), cohort),
            max_tokens=max_tokens, temperature=temperature, json_mode=True,
        )
    except Exception:
        return None
    if not getattr(resp, "success", False):
        return None
    parsed = _parse_json(getattr(resp, "text", ""))
    if not parsed:
        return None
    return _build_idea(parsed, topic.strip(), cohort)


def underserved_cohort_batch(
    topic: str,
    claude_client: Any = _AUTOLOAD,
    n: int = 3,
    dimensions: Optional[List[str]] = None,
    seed: Optional[int] = None,
) -> List[Idea]:
    """End-to-end: pick `n` distinct cohort dimensions, identify one cohort
    under each, design cohort-anchored research. `dimensions` overrides the
    random pick."""
    if not topic or not topic.strip():
        raise ValueError("topic must be non-empty")
    if n <= 0:
        return []
    if dimensions is None:
        pool = list(COHORT_DIMENSIONS)
        rng = random.Random(seed) if seed is not None else random
        rng.shuffle(pool)
        dimensions = pool[:max(1, n)]
    for d in dimensions:
        if d not in COHORT_DIMENSIONS:
            raise ValueError(f"invalid cohort dimension {d!r}")
    out: List[Idea] = []
    for d in dimensions[:n]:
        cohort = identify_cohort(topic, d, claude_client=claude_client)
        if cohort is None:
            continue
        idea = design_for_cohort(topic, cohort, claude_client=claude_client)
        if idea is not None:
            out.append(idea)
    return out
