"""
failure_mode_ideation.py — antibody-by-design ideation.

Defense-driven, complements the offense-driven heretic and adversarial modes.
Two phases:

  1. enumerate_failure_modes(topic, n) — LLM names the most common ways
     research on this topic *fails*: data leakage, weak baseline,
     dataset-specific overfit, p-hacking, distribution shift,
     reproducibility gap, confounded evaluation, benchmark gaming…
  2. generate_immune_idea(topic, failure_mode) — LLM proposes ONE research
     idea designed to be structurally *immune* to that failure mode,
     naming the immunity mechanism explicitly in the method.

Result ideas have `source_strategy='Y'` (antibodY) with the targeted
failure mode stored on `execution_meta.failure_mode_targeted`. The point
isn't just to avoid the failure mode — it's to design research that
*would still be valid* if the failure mode were present in baseline work.

Public API:
    FailureMode                                       → dataclass
    DEFAULT_FAILURE_MODES                              → catalog
    enumerate_failure_modes(topic, ..., n)             → List[FailureMode]
    generate_immune_idea(topic, mode, ...)             → Optional[Idea]
    failure_mode_batch(topic, ..., n)                  → List[Idea]
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from models.idea import Idea, METHODOLOGY_TYPES, NOVELTY_LEVELS


_AUTOLOAD = object()


# Universal research-failure catalog. The LLM may also propose
# topic-specific failure modes; these are seeds the LLM can draw from
# or substitute. Order is rough frequency, most-common first.
DEFAULT_FAILURE_MODES: List[str] = [
    "data_leakage",
    "weak_baseline",
    "benchmark_overfit",
    "distribution_shift",
    "p_hacking_or_garden_of_forking_paths",
    "confounded_evaluation",
    "reproducibility_gap",
    "selection_bias",
    "label_noise_underestimated",
    "compute_or_data_unfair_comparison",
]


@dataclass
class FailureMode:
    """A specific way research on the topic typically fails."""
    name: str = ""                  # short label (e.g., "data_leakage")
    mechanism: str = ""             # how the failure actually happens
    common_signs: str = ""          # how to detect it in papers
    immunity_strategy: str = ""     # design principle that prevents it
    severity: float = 0.5           # 0..1, how often it invalidates results

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "mechanism": self.mechanism,
            "common_signs": self.common_signs,
            "immunity_strategy": self.immunity_strategy,
            "severity": self.severity,
        }


_ENUMERATOR_SYSTEM = (
    "You are a research methodologist who specializes in evaluation "
    "validity. The user gives you a research topic. Your job is to name "
    "the {n} most common ways research on this topic *fails* in the "
    "literature — not philosophical failures, but concrete methodological "
    "ones. For each, name the failure, explain the mechanism, list "
    "tell-tale signs reviewers look for, and propose a design principle "
    "that would make new research structurally immune. Return ONLY valid JSON."
)


def _enumerator_user_prompt(topic: str, n: int) -> str:
    return (
        f"Topic: {topic}\n\n"
        f"Name the {n} most common methodological failure modes in this "
        f"area. Include both general research-validity failures and any "
        f"topic-specific ones. For each, give a concrete immunity strategy "
        f"a researcher could *build into* a new study's design.\n\n"
        f"Reference catalog (you may use, refine, or substitute these): "
        f"{', '.join(DEFAULT_FAILURE_MODES)}.\n\n"
        f"Return JSON:\n"
        "{\n"
        '  "failure_modes": [\n'
        "    {\n"
        '      "name": "<short label, snake_case>",\n'
        '      "mechanism": "<how the failure actually happens>",\n'
        '      "common_signs": "<what reviewers look for to detect it>",\n'
        '      "immunity_strategy": "<design principle that prevents it>",\n'
        '      "severity": <0..1 — how often this invalidates published results>\n'
        "    },\n"
        "    ...\n"
        "  ]\n"
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


def enumerate_failure_modes(
    topic: str,
    claude_client: Any = _AUTOLOAD,
    n: int = 4,
    max_tokens: int = 900,
    temperature: float = 0.6,
) -> List[FailureMode]:
    if not topic or not topic.strip():
        raise ValueError("topic must be non-empty")
    if n <= 0:
        return []
    if claude_client is _AUTOLOAD:
        try:
            from claude_provider import get_claude_client
            claude_client = get_claude_client()
        except Exception:
            claude_client = None
    if claude_client is None:
        return []
    try:
        resp = claude_client.call(
            system=_ENUMERATOR_SYSTEM.format(n=n),
            user=_enumerator_user_prompt(topic.strip(), n),
            max_tokens=max_tokens, temperature=temperature, json_mode=True,
        )
    except Exception:
        return []
    if not getattr(resp, "success", False):
        return []
    parsed = _parse_json(getattr(resp, "text", ""))
    if not parsed or not isinstance(parsed.get("failure_modes"), list):
        return []
    out: List[FailureMode] = []
    for d in parsed["failure_modes"][:max(n, 10)]:
        if not isinstance(d, dict):
            continue
        name = str(d.get("name") or "").strip()
        immunity = str(d.get("immunity_strategy") or "").strip()
        if not (name and immunity):
            continue
        out.append(FailureMode(
            name=name[:120],
            mechanism=str(d.get("mechanism") or "")[:400],
            common_signs=str(d.get("common_signs") or "")[:400],
            immunity_strategy=immunity[:400],
            severity=_coerce_unit(d.get("severity"), 0.5),
        ))
    # Sort by severity descending — attack the worst offenders first.
    out.sort(key=lambda m: m.severity, reverse=True)
    return out


_IDEA_SYSTEM = (
    "You are a research methodologist. The user gives you a research topic "
    "and a specific failure mode common in this area. Your job is to "
    "propose ONE research idea whose design is *structurally immune* to "
    "that failure mode — not by avoidance, but by construction. Name the "
    "immunity mechanism explicitly in the method (e.g., pre-registered "
    "evaluation protocol, held-out distribution split, paired-comparison "
    "design, etc.). Output ONLY valid JSON. methodology_type must be one "
    f"of: {', '.join(METHODOLOGY_TYPES)}. novelty_level must be one of: "
    f"{', '.join(NOVELTY_LEVELS)}."
)


def _idea_user_prompt(topic: str, mode: FailureMode) -> str:
    return (
        f"Topic: {topic}\n\n"
        f"### Failure mode to be immune to\n"
        f"  Name:               {mode.name}\n"
        f"  Mechanism:          {mode.mechanism}\n"
        f"  Common signs:       {mode.common_signs}\n"
        f"  Immunity strategy:  {mode.immunity_strategy}\n\n"
        f"### Instructions\n"
        f"Propose ONE research idea whose design is *structurally immune* "
        f"to this failure mode. Bake the immunity into the method — not as "
        f"an addendum, but as a load-bearing part of the design. Name the "
        f"immunity mechanism explicitly in the method.\n\n"
        f"### Output\n"
        "Return JSON:\n"
        "{\n"
        '  "title": "<concise; signals the robust design>",\n'
        '  "motivation": "<why this failure mode matters to falsify cleanly>",\n'
        '  "method": "<concrete approach with immunity baked in>",\n'
        '  "hypothesis": "<falsifiable claim>",\n'
        '  "resources": "<resources needed>",\n'
        '  "expected_outcome": "<measurable outcome>",\n'
        '  "risk_assessment": "<technical risks AND remaining validity threats>",\n'
        '  "source_strategy": "Y",\n'
        f'  "methodology_type": "<one of: {", ".join(METHODOLOGY_TYPES)}>",\n'
        f'  "novelty_level": "<one of: {", ".join(NOVELTY_LEVELS)}>",\n'
        '  "immunity_mechanism": "<one sentence: how the design rules out the failure>"\n'
        "}"
    )


def _build_idea(parsed: Dict[str, Any], topic: str,
                  mode: FailureMode) -> Optional[Idea]:
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
        source_strategy="Y",
        methodology_type=method_type,
        novelty_level=novelty,
        generation=0,
        parent_title=None,
    )
    idea.execution_meta = {
        "failure_mode_targeted": mode.to_dict(),
        "immunity_mechanism": str(parsed.get("immunity_mechanism", ""))[:400],
        "regen_mode": "failure_mode",
        "topic": topic,
    }
    return idea


def generate_immune_idea(
    topic: str,
    mode: FailureMode,
    claude_client: Any = _AUTOLOAD,
    max_tokens: int = 900,
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
            system=_IDEA_SYSTEM,
            user=_idea_user_prompt(topic.strip(), mode),
            max_tokens=max_tokens, temperature=temperature, json_mode=True,
        )
    except Exception:
        return None
    if not getattr(resp, "success", False):
        return None
    parsed = _parse_json(getattr(resp, "text", ""))
    if not parsed:
        return None
    return _build_idea(parsed, topic.strip(), mode)


def failure_mode_batch(
    topic: str,
    claude_client: Any = _AUTOLOAD,
    n: int = 3,
) -> List[Idea]:
    """End-to-end: enumerate the top failure modes, then design an immune
    research idea for each. Returns ideas sorted by severity (worst first)."""
    modes = enumerate_failure_modes(topic, claude_client=claude_client, n=n)
    if not modes:
        return []
    out: List[Idea] = []
    for m in modes[:n]:
        idea = generate_immune_idea(topic, m, claude_client=claude_client)
        if idea is not None:
            out.append(idea)
    return out
