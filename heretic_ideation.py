"""
heretic_ideation.py — anti-consensus ideation.

Different from the Adversarial Critic (which attacks one specific idea),
the Heretic mode attacks the *field's* dominant consensus. Two phases:

  1. extract_dominant_beliefs(topic) — LLM names 3-5 beliefs most ML/research
     practitioners would accept without argument on this topic.
  2. generate_heretic_idea(topic, belief) — LLM proposes research
     explicitly designed to falsify that belief.

Result ideas have `source_strategy='H'` with the targeted belief stored
on `execution_meta.belief_targeted`. These ideas often score badly on
probes (orthodoxy is partly what probes check) — pair with the Exec Loop
to keep feasibility honest.

Public API:
    DominantBelief                                  → dataclass
    extract_dominant_beliefs(topic, ..., n)         → List[DominantBelief]
    generate_heretic_idea(topic, belief, ...)       → Optional[Idea]
    generate_heretic_batch(topic, ..., n)           → List[Idea]
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from models.idea import Idea, METHODOLOGY_TYPES, NOVELTY_LEVELS


_AUTOLOAD = object()


@dataclass
class DominantBelief:
    """A belief the field accepts as 'obvious' — therefore worth attacking."""
    statement: str = ""           # the belief in one sentence
    evidence_cited: str = ""      # what people typically point to for support
    why_canonical: str = ""       # why it became the default
    cracks: str = ""              # honest weaknesses in the case for it
    falsification_hint: str = ""  # ideal direction for attacking it
    confidence: float = 0.5       # how confident the LLM is that this is canonical

    def to_dict(self) -> Dict[str, Any]:
        return {
            "statement": self.statement,
            "evidence_cited": self.evidence_cited,
            "why_canonical": self.why_canonical,
            "cracks": self.cracks,
            "falsification_hint": self.falsification_hint,
            "confidence": self.confidence,
        }


_EXTRACTOR_SYSTEM = (
    "You are a contrarian intellectual historian. The user gives you a "
    "research topic. Your job is to name 3-5 beliefs that most "
    "practitioners in this field would currently accept without serious "
    "argument — even though those beliefs are not actually proven beyond "
    "doubt. These are the 'received wisdom' assumptions ripe for "
    "falsification. Be specific and honest; avoid strawmen. Return ONLY "
    "valid JSON."
)


def _extractor_user_prompt(topic: str, n: int) -> str:
    return (
        f"Topic: {topic}\n\n"
        f"Name {n} beliefs that most researchers in this field currently "
        f"treat as canonical — but which haven't actually been proven "
        f"beyond reasonable doubt. For each: what supporting evidence is "
        f"typically cited, why it became default, what cracks honest "
        f"observers would note, and an ideal direction for falsifying it.\n\n"
        f"Return JSON:\n"
        "{\n"
        '  "beliefs": [\n'
        "    {\n"
        '      "statement": "<one-sentence canonical belief>",\n'
        '      "evidence_cited": "<what people typically cite to support it>",\n'
        '      "why_canonical": "<why it became default>",\n'
        '      "cracks": "<honest weaknesses in the case for it>",\n'
        '      "falsification_hint": "<ideal direction to attack it>",\n'
        '      "confidence": <number 0..1 — how confident you are this '
        'really is canonical, not strawman>\n'
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


def extract_dominant_beliefs(
    topic: str,
    claude_client: Any = _AUTOLOAD,
    n: int = 4,
    max_tokens: int = 800,
    temperature: float = 0.7,
) -> List[DominantBelief]:
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
            system=_EXTRACTOR_SYSTEM,
            user=_extractor_user_prompt(topic.strip(), n),
            max_tokens=max_tokens, temperature=temperature, json_mode=True,
        )
    except Exception:
        return []
    if not getattr(resp, "success", False):
        return []
    parsed = _parse_json(getattr(resp, "text", ""))
    if not parsed or not isinstance(parsed.get("beliefs"), list):
        return []
    out: List[DominantBelief] = []
    for d in parsed["beliefs"][:max(n, 8)]:
        if not isinstance(d, dict):
            continue
        stmt = str(d.get("statement") or "").strip()
        if not stmt:
            continue
        out.append(DominantBelief(
            statement=stmt[:400],
            evidence_cited=str(d.get("evidence_cited") or "")[:400],
            why_canonical=str(d.get("why_canonical") or "")[:400],
            cracks=str(d.get("cracks") or "")[:400],
            falsification_hint=str(d.get("falsification_hint") or "")[:400],
            confidence=_coerce_unit(d.get("confidence"), 0.5),
        ))
    # Sort by confidence (most canonical first → most worth attacking)
    out.sort(key=lambda b: b.confidence, reverse=True)
    return out


_HERETIC_SYSTEM = (
    "You are a heretical research scientist. The user gives you a "
    "research topic and a belief most of the field currently accepts. "
    "Your job is to propose ONE research idea explicitly designed to "
    "*falsify* that belief — not just qualify it, falsify it. Be rigorous; "
    "design the experiment so a clean negative result would constitute a "
    "real refutation. Be honest about the field-political risk in "
    "risk_assessment. Output ONLY valid JSON. methodology_type must be "
    f"one of: {', '.join(METHODOLOGY_TYPES)}. novelty_level must be one "
    f"of: {', '.join(NOVELTY_LEVELS)}."
)


def _heretic_user_prompt(topic: str, belief: DominantBelief) -> str:
    return (
        f"Topic: {topic}\n\n"
        f"### Canonical belief to falsify\n"
        f"  Statement: {belief.statement}\n"
        f"  Why it's accepted: {belief.why_canonical}\n"
        f"  Evidence typically cited: {belief.evidence_cited}\n"
        f"  Honest cracks: {belief.cracks}\n"
        f"  Falsification hint: {belief.falsification_hint}\n\n"
        f"### Instructions\n"
        f"Propose ONE research idea that, if it succeeds, would falsify "
        f"the belief above. Design it so a clean negative result "
        f"genuinely refutes the canonical view (not just adds a "
        f"qualification). State the falsification mechanism explicitly "
        f"in the method.\n\n"
        f"### Output\n"
        "Return JSON:\n"
        "{\n"
        '  "title": "<concise, signals the heretical stance>",\n'
        '  "motivation": "<why falsifying this belief matters>",\n'
        '  "method": "<concrete approach, naming the falsification '
        'mechanism explicitly>",\n'
        '  "hypothesis": "<falsifiable claim opposing the canon>",\n'
        '  "resources": "<resources needed>",\n'
        '  "expected_outcome": "<measurable outcome that would constitute '
        'a refutation>",\n'
        '  "risk_assessment": "<technical risks AND field-political risks>",\n'
        '  "source_strategy": "H",\n'
        f'  "methodology_type": "<one of: {", ".join(METHODOLOGY_TYPES)}>",\n'
        f'  "novelty_level": "<one of: {", ".join(NOVELTY_LEVELS)}>",\n'
        '  "falsification_mechanism": "<one sentence: what kind of '
        'negative result would refute the canon>"\n'
        "}"
    )


def _build_idea(parsed: Dict[str, Any], topic: str,
                  belief: DominantBelief) -> Optional[Idea]:
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
        source_strategy="H",
        methodology_type=method_type,
        novelty_level=novelty,
        generation=0,
        parent_title=None,
    )
    idea.execution_meta = {
        "belief_targeted": belief.to_dict(),
        "falsification_mechanism": str(
            parsed.get("falsification_mechanism", "")
        )[:400],
        "regen_mode": "heretic",
        "topic": topic,
    }
    return idea


def generate_heretic_idea(
    topic: str,
    belief: DominantBelief,
    claude_client: Any = _AUTOLOAD,
    max_tokens: int = 900,
    temperature: float = 0.8,
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
            system=_HERETIC_SYSTEM,
            user=_heretic_user_prompt(topic.strip(), belief),
            max_tokens=max_tokens, temperature=temperature, json_mode=True,
        )
    except Exception:
        return None
    if not getattr(resp, "success", False):
        return None
    parsed = _parse_json(getattr(resp, "text", ""))
    if not parsed:
        return None
    return _build_idea(parsed, topic.strip(), belief)


def generate_heretic_batch(
    topic: str,
    claude_client: Any = _AUTOLOAD,
    n: int = 3,
) -> List[Idea]:
    """End-to-end: extract beliefs, generate a falsifying idea for each."""
    beliefs = extract_dominant_beliefs(topic, claude_client=claude_client, n=n)
    if not beliefs:
        return []
    out: List[Idea] = []
    for b in beliefs[:n]:
        idea = generate_heretic_idea(topic, b, claude_client=claude_client)
        if idea is not None:
            out.append(idea)
    return out
