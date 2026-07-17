"""
contradiction_ideation.py — TRIZ-style contradiction-driven idea generation.

Instead of generating from frontier papers or empty cells, this module
asks: "what are the inherent contradictions in this topic?" — then
generates ideas that explicitly resolve them. Classic invention
methodology (TRIZ: Theory of Inventive Problem-Solving) translated to
research ideation.

Two phases:
  1. `extract_contradictions(topic)` — LLM names 3-6 genuine tensions
     in the topic, each framed as "we want X AND Y but they conflict".
  2. `generate_from_contradiction(topic, contradiction)` — LLM generates
     ONE idea that explicitly breaks the contradiction, with the
     resolution mechanism stated up-front.

Result ideas have `source_strategy='C'` (Contradiction-driven), with the
specific contradiction stored on `execution_meta.contradiction_resolved`.

Public API:
    Contradiction                                          → dataclass
    extract_contradictions(topic, claude_client=None, n=4) → List[Contradiction]
    generate_from_contradiction(topic, contradiction, ...) → Optional[Idea]
    generate_from_contradictions_batch(topic, n=4, ...)    → List[Idea]
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from models.idea import Idea, METHODOLOGY_TYPES, NOVELTY_LEVELS


_AUTOLOAD = object()


# ─────────────────────────────────────────────────────────────────────────────
# Dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Contradiction:
    """A genuine tension between two desirable properties of a research
    topic — the kind that historically drove paradigm shifts when broken."""

    statement: str = ""            # "We want X AND Y, but they conflict"
    forces_a: str = ""             # "X requires …"
    forces_b: str = ""             # "Y requires …"
    why_it_matters: str = ""       # 1-line significance
    resolution_hint: str = ""      # ideal direction for breaking it
    severity: float = 0.5          # 0..1, how genuinely contradictory (vs. easy)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "statement": self.statement,
            "forces_a": self.forces_a, "forces_b": self.forces_b,
            "why_it_matters": self.why_it_matters,
            "resolution_hint": self.resolution_hint,
            "severity": self.severity,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: extract contradictions
# ─────────────────────────────────────────────────────────────────────────────

_EXTRACTOR_SYSTEM = (
    "You are a research methodologist trained in TRIZ (Theory of Inventive "
    "Problem-Solving) and the history of scientific paradigm shifts. The "
    "user gives you a research topic. Your job is to identify the genuine "
    "*contradictions* inside it — places where two desirable properties "
    "are in fundamental tension, not just tradeoffs. Genuine contradictions "
    "are the kind that historically drove paradigm shifts when broken "
    "(e.g., wave-vs-particle, fast-vs-accurate, expressive-vs-compact). "
    "Avoid framing simple tradeoffs as contradictions. Return ONLY valid "
    "JSON matching the schema."
)


def _extractor_user_prompt(topic: str, n: int) -> str:
    return (
        f"Topic: {topic}\n\n"
        f"Identify the {n} most genuinely contradictory tensions inside this "
        f"topic — places where two desirable properties are in fundamental "
        f"conflict. For each contradiction, state both forces clearly, why "
        f"breaking it would matter for the field, and an ideal direction "
        f"for the break.\n\n"
        f"Return JSON with this exact structure:\n"
        "{\n"
        '  "contradictions": [\n'
        "    {\n"
        '      "statement":       "<one sentence: We want X AND Y but they conflict>",\n'
        '      "forces_a":        "<what X requires / drives toward>",\n'
        '      "forces_b":        "<what Y requires / drives toward>",\n'
        '      "why_it_matters":  "<one sentence on significance>",\n'
        '      "resolution_hint": "<ideal direction for breaking it>",\n'
        '      "severity":        <number 0..1; 1 = fundamental, 0 = simple tradeoff>\n'
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


def extract_contradictions(
    topic: str,
    claude_client: Any = _AUTOLOAD,
    n: int = 4,
    max_tokens: int = 800,
    temperature: float = 0.6,
) -> List[Contradiction]:
    """Identify N genuine contradictions in a research topic.

    Returns a list sorted by severity descending. Empty list if the LLM
    is unavailable or returns malformed output.
    """
    if not topic or not topic.strip():
        raise ValueError("extract_contradictions requires a non-empty topic")
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
            system=_EXTRACTOR_SYSTEM,
            user=_extractor_user_prompt(topic.strip(), n),
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=True,
        )
    except Exception:
        return []
    if not getattr(resp, "success", False):
        return []
    parsed = _parse_json(getattr(resp, "text", ""))
    if not parsed:
        return []

    items = parsed.get("contradictions") or []
    if not isinstance(items, list):
        return []

    out: List[Contradiction] = []
    for d in items[:max(n, 8)]:
        if not isinstance(d, dict):
            continue
        stmt = str(d.get("statement") or "").strip()
        if not stmt:
            continue
        out.append(Contradiction(
            statement=stmt[:400],
            forces_a=str(d.get("forces_a") or "")[:200],
            forces_b=str(d.get("forces_b") or "")[:200],
            why_it_matters=str(d.get("why_it_matters") or "")[:300],
            resolution_hint=str(d.get("resolution_hint") or "")[:300],
            severity=_coerce_unit(d.get("severity"), 0.5),
        ))
    out.sort(key=lambda c: c.severity, reverse=True)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: generate an idea that resolves a contradiction
# ─────────────────────────────────────────────────────────────────────────────

_GENERATOR_SYSTEM = (
    "You are a research scientist generating one idea that explicitly "
    "RESOLVES a stated contradiction in your field. The contradiction is "
    "a genuine tension between two desirable properties — your idea must "
    "name the resolution mechanism up-front (separation in time/space, "
    "phase transition, asymmetric routing, etc.), not just split the "
    "difference. Output ONLY valid JSON with the idea schema."
)


def _generator_user_prompt(topic: str, c: Contradiction) -> str:
    return (
        f"Topic: {topic}\n\n"
        f"### The contradiction to resolve\n"
        f"{c.statement}\n"
        f"  Force A: {c.forces_a}\n"
        f"  Force B: {c.forces_b}\n"
        f"  Why it matters: {c.why_it_matters}\n"
        f"  Resolution hint: {c.resolution_hint}\n\n"
        f"### Instructions\n"
        f"Propose ONE research idea that explicitly resolves this "
        f"contradiction. State the resolution mechanism in the method "
        f"section — do not just average the two forces; find a way to "
        f"make them both hold simultaneously. Be technically concrete.\n\n"
        f"### Output\n"
        "Return JSON:\n"
        "{\n"
        '  "title":              "<concise, mentions the resolution>",\n'
        '  "motivation":         "<why resolving this matters>",\n'
        '  "method":             "<technical approach, name the '
        'resolution mechanism explicitly>",\n'
        '  "hypothesis":         "<falsifiable claim about the resolution>",\n'
        '  "resources":          "<resources needed>",\n'
        '  "expected_outcome":   "<measurable outcome>",\n'
        '  "risk_assessment":    "<main risks>",\n'
        '  "source_strategy":    "C",\n'
        f'  "methodology_type":   "<one of: {", ".join(METHODOLOGY_TYPES)}>",\n'
        f'  "novelty_level":      "<one of: {", ".join(NOVELTY_LEVELS)}>",\n'
        '  "resolution_mechanism": "<one sentence: how the idea makes '
        'BOTH forces hold simultaneously>"\n'
        "}"
    )


def _build_idea_from_response(
    parsed: Dict[str, Any], topic: str, c: Contradiction,
) -> Optional[Idea]:
    required = ("title", "method", "hypothesis")
    if not all(str(parsed.get(k, "")).strip() for k in required):
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
        source_strategy="C",
        methodology_type=method_type,
        novelty_level=novelty,
        generation=0,
        parent_title=None,
    )
    idea.execution_meta = {
        "contradiction_resolved": c.to_dict(),
        "resolution_mechanism": str(parsed.get("resolution_mechanism", ""))[:400],
        "regen_mode": "contradiction_driven",
        "topic": topic,
    }
    return idea


def generate_from_contradiction(
    topic: str,
    contradiction: Contradiction,
    claude_client: Any = _AUTOLOAD,
    max_tokens: int = 900,
    temperature: float = 0.75,
) -> Optional[Idea]:
    """Generate ONE idea that explicitly resolves the given contradiction."""
    if not topic or not topic.strip():
        raise ValueError("generate_from_contradiction requires a topic")
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
            system=_GENERATOR_SYSTEM,
            user=_generator_user_prompt(topic.strip(), contradiction),
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=True,
        )
    except Exception:
        return None
    if not getattr(resp, "success", False):
        return None
    parsed = _parse_json(getattr(resp, "text", ""))
    if not parsed:
        return None
    return _build_idea_from_response(parsed, topic.strip(), contradiction)


def generate_from_contradictions_batch(
    topic: str,
    claude_client: Any = _AUTOLOAD,
    n: int = 4,
    max_tokens: int = 900,
) -> List[Idea]:
    """End-to-end: extract N contradictions, generate an idea for each.

    Returns ideas in severity order (most fundamental contradictions first).
    """
    contradictions = extract_contradictions(
        topic, claude_client=claude_client, n=n,
    )
    if not contradictions:
        return []

    out: List[Idea] = []
    for c in contradictions:
        idea = generate_from_contradiction(
            topic, c, claude_client=claude_client, max_tokens=max_tokens,
        )
        if idea is not None:
            out.append(idea)
    return out
