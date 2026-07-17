"""
composable_primitive_ideation.py — design the primitive, not the paper.

Most novelty modes optimize a *finished* contribution. This one optimizes
for *downstream composability*: design research whose deliverable is a
small, reusable primitive that other people will pull into their own
pipelines as a load-bearing dependency. The headline is not the score
on a benchmark — it is the number of downstream uses.

Two phases:
  1. identify_primitive_slot(topic, kind) — LLM names a specific
     unfilled slot in the topic's tooling/method ecosystem: a missing
     dataset standard, a missing evaluation harness, a missing tiny
     interpretable building block. State why nothing fills it well.
  2. design_primitive(topic, slot) — LLM designs research whose
     deliverable IS the primitive: API surface, invariants it
     preserves, evaluation by adoption proxies (citations, forks,
     downstream replications), and an explicit non-goal list of what
     the primitive doesn't try to solve.

Ideas have `source_strategy='D'` (Downstream-composable). The primitive
slot lives on `execution_meta.primitive_slot`. The prompt enforces a
named API surface, an explicit non-goal list, and an adoption-proxy
metric so the contribution isn't just "another method that worked once".

Public API:
    PrimitiveSlot                                     → dataclass
    PRIMITIVE_KINDS                                   → catalog
    identify_primitive_slot(topic, kind, ...)         → Optional[PrimitiveSlot]
    design_primitive(topic, slot, ...)                → Optional[Idea]
    composable_primitive_batch(topic, ..., n)         → List[Idea]
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from models.idea import Idea, METHODOLOGY_TYPES, NOVELTY_LEVELS


_AUTOLOAD = object()


# Kinds of unfilled "slot" the LLM can target. The LLM picks ONE kind and
# names a concrete missing primitive within it for the topic.
PRIMITIVE_KINDS: List[str] = [
    "evaluation_harness",          # missing standardized test harness
    "dataset_standard",            # missing canonical small reference dataset
    "interface_protocol",          # missing inter-method protocol/contract
    "diagnostic_probe",            # missing tiny probe for a specific failure mode
    "compositional_block",         # missing reusable architectural building block
    "reference_implementation",    # missing minimal reference impl of a known idea
    "quality_metric",              # missing well-grounded metric for a known property
    "ablation_kit",                # missing standard set of ablations the field skips
]


@dataclass
class PrimitiveSlot:
    """A specific unfilled slot in the topic's tooling ecosystem."""
    kind: str = ""                  # one of PRIMITIVE_KINDS
    name: str = ""                  # short label (e.g. "DialectalASR-Probe")
    description: str = ""           # what the primitive would be, in one paragraph
    why_unfilled: str = ""          # why nothing fills this well today
    downstream_users: str = ""      # who would adopt it and for what
    adoption_proxy: str = ""        # how adoption would be measured (forks, citations…)
    blast_radius: float = 0.5       # 0..1, how many downstream uses it could unblock

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "description": self.description,
            "why_unfilled": self.why_unfilled,
            "downstream_users": self.downstream_users,
            "adoption_proxy": self.adoption_proxy,
            "blast_radius": self.blast_radius,
        }


_SLOT_SYSTEM = (
    "You are a tools-and-infrastructure researcher. Given a topic and a "
    "primitive-kind, identify ONE specific unfilled slot in the topic's "
    "tooling ecosystem — a missing primitive that, if it existed, would "
    "be pulled in as a load-bearing dependency by other people's work. "
    "Be specific about who would adopt it and how adoption would be "
    "measured. Return ONLY valid JSON."
)


def _slot_user_prompt(topic: str, kind: str) -> str:
    return (
        f"Topic: {topic}\n"
        f"Primitive kind: {kind}\n\n"
        f"Identify ONE specific unfilled slot of this kind for this topic. "
        f"Be concrete: what the primitive would be, why nothing fills it "
        f"today, who would adopt it, and how you would measure adoption "
        f"(forks, citations, downstream replications, etc.). Estimate "
        f"blast radius — how many downstream papers/projects could it "
        f"unblock.\n\n"
        f"Return JSON:\n"
        "{\n"
        '  "name": "<short, specific label, e.g. \\"DialectalASR-Probe v0\\">",\n'
        '  "description": "<what the primitive would be, in one paragraph>",\n'
        '  "why_unfilled": "<why nothing fills this slot today>",\n'
        '  "downstream_users": "<who would adopt it and for what>",\n'
        '  "adoption_proxy": "<how adoption would be measured>",\n'
        '  "blast_radius": <0..1 — how many downstream uses it could unblock>\n'
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


def identify_primitive_slot(
    topic: str,
    kind: str,
    claude_client: Any = _AUTOLOAD,
    max_tokens: int = 700,
    temperature: float = 0.7,
) -> Optional[PrimitiveSlot]:
    if not topic or not topic.strip():
        raise ValueError("topic must be non-empty")
    if kind not in PRIMITIVE_KINDS:
        raise ValueError(
            f"kind must be one of {PRIMITIVE_KINDS}, got {kind!r}"
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
            system=_SLOT_SYSTEM,
            user=_slot_user_prompt(topic.strip(), kind),
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
    proxy = str(parsed.get("adoption_proxy") or "").strip()
    if not (name and proxy):
        return None
    return PrimitiveSlot(
        kind=kind,
        name=name[:200],
        description=str(parsed.get("description") or "")[:600],
        why_unfilled=str(parsed.get("why_unfilled") or "")[:400],
        downstream_users=str(parsed.get("downstream_users") or "")[:400],
        adoption_proxy=proxy[:300],
        blast_radius=_coerce_unit(parsed.get("blast_radius"), 0.5),
    )


_DESIGN_SYSTEM = (
    "You are a researcher designing for downstream adoption. The user "
    "gives you a topic and a specific unfilled primitive slot. Design "
    "ONE study whose deliverable IS the primitive itself — not a paper "
    "that uses it. The method MUST include (a) a named API surface or "
    "interface contract, (b) an explicit list of non-goals (what the "
    "primitive deliberately doesn't try to do), and (c) the adoption "
    "proxy used as the success metric (citations, forks, downstream "
    "replications, etc.). Output ONLY valid JSON. methodology_type must "
    f"be one of: {', '.join(METHODOLOGY_TYPES)}. novelty_level must be "
    f"one of: {', '.join(NOVELTY_LEVELS)}."
)


def _design_user_prompt(topic: str, slot: PrimitiveSlot) -> str:
    return (
        f"Topic: {topic}\n\n"
        f"### Unfilled primitive slot\n"
        f"  Kind:               {slot.kind}\n"
        f"  Name:               {slot.name}\n"
        f"  Description:        {slot.description}\n"
        f"  Why unfilled:       {slot.why_unfilled}\n"
        f"  Downstream users:   {slot.downstream_users}\n"
        f"  Adoption proxy:     {slot.adoption_proxy}\n"
        f"  Blast radius:       {slot.blast_radius:.2f}\n\n"
        f"### Instructions\n"
        f"Design ONE study whose deliverable IS this primitive — not a "
        f"paper that uses it. The method must name (a) the API surface or "
        f"interface contract, (b) explicit non-goals, and (c) the "
        f"adoption proxy as the success metric.\n\n"
        f"### Output\n"
        "Return JSON:\n"
        "{\n"
        '  "title": "<concise; names the primitive>",\n'
        '  "motivation": "<why the slot matters and why nothing fills it>",\n'
        '  "method": "<the primitive design, including API and non-goals>",\n'
        '  "hypothesis": "<falsifiable adoption claim>",\n'
        '  "resources": "<resources needed to ship and maintain>",\n'
        '  "expected_outcome": "<adoption-anchored measurable outcome>",\n'
        '  "risk_assessment": "<technical risks AND adoption risks (\\"nobody uses it\\")>",\n'
        '  "source_strategy": "D",\n'
        f'  "methodology_type": "<one of: {", ".join(METHODOLOGY_TYPES)}>",\n'
        f'  "novelty_level": "<one of: {", ".join(NOVELTY_LEVELS)}>",\n'
        '  "api_surface": "<one sentence: the named interface contract>",\n'
        '  "non_goals": "<one sentence: what the primitive deliberately does not do>"\n'
        "}"
    )


def _build_idea(parsed: Dict[str, Any], topic: str,
                  slot: PrimitiveSlot) -> Optional[Idea]:
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
        source_strategy="D",
        methodology_type=method_type,
        novelty_level=novelty,
        generation=0,
        parent_title=None,
    )
    idea.execution_meta = {
        "primitive_slot": slot.to_dict(),
        "api_surface": str(parsed.get("api_surface", ""))[:400],
        "non_goals": str(parsed.get("non_goals", ""))[:400],
        "regen_mode": "composable_primitive",
        "topic": topic,
    }
    return idea


def design_primitive(
    topic: str,
    slot: PrimitiveSlot,
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
            user=_design_user_prompt(topic.strip(), slot),
            max_tokens=max_tokens, temperature=temperature, json_mode=True,
        )
    except Exception:
        return None
    if not getattr(resp, "success", False):
        return None
    parsed = _parse_json(getattr(resp, "text", ""))
    if not parsed:
        return None
    return _build_idea(parsed, topic.strip(), slot)


def composable_primitive_batch(
    topic: str,
    claude_client: Any = _AUTOLOAD,
    n: int = 3,
    kinds: Optional[List[str]] = None,
    seed: Optional[int] = None,
) -> List[Idea]:
    """End-to-end: pick `n` distinct primitive kinds, identify a slot
    under each, design the primitive. `kinds` overrides the random pick."""
    if not topic or not topic.strip():
        raise ValueError("topic must be non-empty")
    if n <= 0:
        return []
    if kinds is None:
        pool = list(PRIMITIVE_KINDS)
        rng = random.Random(seed) if seed is not None else random
        rng.shuffle(pool)
        kinds = pool[:max(1, n)]
    for k in kinds:
        if k not in PRIMITIVE_KINDS:
            raise ValueError(f"invalid primitive kind {k!r}")
    out: List[Idea] = []
    for k in kinds[:n]:
        slot = identify_primitive_slot(topic, k, claude_client=claude_client)
        if slot is None:
            continue
        idea = design_primitive(topic, slot, claude_client=claude_client)
        if idea is not None:
            out.append(idea)
    return out
