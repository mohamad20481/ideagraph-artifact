"""
analogy_ideation.py — cross-domain structural analogy.

Different from persona-swap (a viewpoint change) and counterfactual_literature
(an imagined missing-citation seed). Analogy mode does *structure-preserving
transfer*: pick a far domain, name an isomorphism between its structure and
the topic's structure, then port a method through that mapping.

Two phases:
  1. extract_analogy(topic, far_domain) — LLM picks a structural analogy:
     names the source domain, the structural element being mapped, the
     target-side counterpart, and *why the morphism preserves what matters*.
  2. generate_from_analogy(topic, bridge) — LLM transplants a concrete
     method from the source domain into the target topic via the bridge.

Ideas have `source_strategy='M'` (Morphism) with the bridge details on
`execution_meta.analogy_bridge`. Pair with the Exec Loop to keep the
imported method honest under the topic's actual constraints.

Public API:
    AnalogyBridge                                  → dataclass
    DEFAULT_DOMAINS                                → list of far-domain seeds
    extract_analogy(topic, far_domain, ...)        → Optional[AnalogyBridge]
    generate_from_analogy(topic, bridge, ...)      → Optional[Idea]
    analogy_batch(topic, n, domains, ...)          → List[Idea]
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from models.idea import Idea, METHODOLOGY_TYPES, NOVELTY_LEVELS


_AUTOLOAD = object()


DEFAULT_DOMAINS: List[str] = [
    "molecular biology",
    "ecology and population dynamics",
    "music theory and composition",
    "control theory and feedback systems",
    "statistical mechanics",
    "urban planning and city design",
    "immune system response",
    "evolutionary dynamics",
    "market design and auction theory",
    "linguistics and historical phonology",
    "topology and continuous deformation",
    "chemistry of reaction networks",
    "neuroscience of cortical circuits",
    "game theory and mechanism design",
    "cartography and map projections",
    "architecture and load-bearing structure",
]


@dataclass
class AnalogyBridge:
    """A structural mapping from a far domain into the target topic."""
    source_domain: str = ""           # e.g., "immune system response"
    source_structure: str = ""        # the structural element in the source
    target_counterpart: str = ""      # the matching structure on the topic side
    morphism: str = ""                # the mapping rule (what maps to what)
    invariant: str = ""               # what the mapping preserves (the why)
    risk_of_break: str = ""           # where the analogy is likely to break
    confidence: float = 0.5           # LLM confidence the bridge is non-trivial

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_domain": self.source_domain,
            "source_structure": self.source_structure,
            "target_counterpart": self.target_counterpart,
            "morphism": self.morphism,
            "invariant": self.invariant,
            "risk_of_break": self.risk_of_break,
            "confidence": self.confidence,
        }


_BRIDGE_SYSTEM = (
    "You are a researcher with deep training in two fields. Given a target "
    "research topic and a far source domain, identify a non-trivial *structural* "
    "analogy: a mapping (morphism) from a recognizable structure in the source "
    "domain to a structure in the target topic. The analogy must preserve a "
    "specific invariant (function, dynamic, or constraint) — otherwise it's "
    "just a metaphor. Be specific. Return ONLY valid JSON."
)


def _bridge_user_prompt(topic: str, far_domain: str) -> str:
    return (
        f"Target topic: {topic}\n"
        f"Source domain: {far_domain}\n\n"
        f"Find a structural analogy where some named structure in "
        f"{far_domain} maps cleanly to a structure in the target topic, and "
        f"name what the morphism preserves. Avoid surface-level metaphors. "
        f"Be honest about where the analogy breaks.\n\n"
        f"Return JSON:\n"
        "{\n"
        '  "source_structure": "<named element in the source domain>",\n'
        '  "target_counterpart": "<the matching structure on the topic side>",\n'
        '  "morphism": "<the mapping rule: what maps to what>",\n'
        '  "invariant": "<what the mapping preserves — function, dynamic, or constraint>",\n'
        '  "risk_of_break": "<where the analogy is likely to fail under stress>",\n'
        '  "confidence": <0..1 — how non-trivial / load-bearing this analogy is>\n'
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


def extract_analogy(
    topic: str,
    far_domain: str,
    claude_client: Any = _AUTOLOAD,
    max_tokens: int = 700,
    temperature: float = 0.8,
) -> Optional[AnalogyBridge]:
    if not topic or not topic.strip():
        raise ValueError("topic must be non-empty")
    if not far_domain or not far_domain.strip():
        raise ValueError("far_domain must be non-empty")
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
            system=_BRIDGE_SYSTEM,
            user=_bridge_user_prompt(topic.strip(), far_domain.strip()),
            max_tokens=max_tokens, temperature=temperature, json_mode=True,
        )
    except Exception:
        return None
    if not getattr(resp, "success", False):
        return None
    parsed = _parse_json(getattr(resp, "text", ""))
    if not parsed:
        return None
    structure = str(parsed.get("source_structure") or "").strip()
    counterpart = str(parsed.get("target_counterpart") or "").strip()
    morphism = str(parsed.get("morphism") or "").strip()
    if not (structure and counterpart and morphism):
        return None
    return AnalogyBridge(
        source_domain=far_domain.strip()[:200],
        source_structure=structure[:400],
        target_counterpart=counterpart[:400],
        morphism=morphism[:600],
        invariant=str(parsed.get("invariant") or "")[:400],
        risk_of_break=str(parsed.get("risk_of_break") or "")[:400],
        confidence=_coerce_unit(parsed.get("confidence"), 0.5),
    )


_IDEA_SYSTEM = (
    "You are a researcher porting a method across fields. Given a target "
    "research topic and a structural analogy bridge from a far domain, "
    "propose ONE research idea that transplants a concrete method or "
    "mechanism from the source side through the morphism. State the "
    "transplanted mechanism explicitly. Be honest about what the analogy "
    "buys you and where it might mislead. Output ONLY valid JSON. "
    f"methodology_type must be one of: {', '.join(METHODOLOGY_TYPES)}. "
    f"novelty_level must be one of: {', '.join(NOVELTY_LEVELS)}."
)


def _idea_user_prompt(topic: str, bridge: AnalogyBridge) -> str:
    return (
        f"Target topic: {topic}\n\n"
        f"### Analogy bridge\n"
        f"  Source domain:       {bridge.source_domain}\n"
        f"  Source structure:    {bridge.source_structure}\n"
        f"  Target counterpart:  {bridge.target_counterpart}\n"
        f"  Morphism:            {bridge.morphism}\n"
        f"  Invariant preserved: {bridge.invariant}\n"
        f"  Where it may break:  {bridge.risk_of_break}\n\n"
        f"### Instructions\n"
        f"Propose ONE research idea that transplants a concrete method from "
        f"the source domain into the target topic via this morphism. Name "
        f"the imported mechanism explicitly. Do NOT just rename existing "
        f"techniques — the analogy must do real work.\n\n"
        f"### Output\n"
        "Return JSON:\n"
        "{\n"
        '  "title": "<concise; signals the source→target transplant>",\n'
        '  "motivation": "<why the transplanted mechanism is worth porting>",\n'
        '  "method": "<concrete approach; name the imported mechanism>",\n'
        '  "hypothesis": "<falsifiable claim>",\n'
        '  "resources": "<resources needed>",\n'
        '  "expected_outcome": "<measurable outcome>",\n'
        '  "risk_assessment": "<technical risks + where the analogy may fail>",\n'
        '  "source_strategy": "M",\n'
        f'  "methodology_type": "<one of: {", ".join(METHODOLOGY_TYPES)}>",\n'
        f'  "novelty_level": "<one of: {", ".join(NOVELTY_LEVELS)}>",\n'
        '  "transplanted_mechanism": "<one sentence: what was ported and how>"\n'
        "}"
    )


def _build_idea(parsed: Dict[str, Any], topic: str,
                  bridge: AnalogyBridge) -> Optional[Idea]:
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
        source_strategy="M",
        methodology_type=method_type,
        novelty_level=novelty,
        generation=0,
        parent_title=None,
    )
    idea.execution_meta = {
        "analogy_bridge": bridge.to_dict(),
        "transplanted_mechanism": str(
            parsed.get("transplanted_mechanism", "")
        )[:400],
        "regen_mode": "analogy",
        "topic": topic,
    }
    return idea


def generate_from_analogy(
    topic: str,
    bridge: AnalogyBridge,
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
            system=_IDEA_SYSTEM,
            user=_idea_user_prompt(topic.strip(), bridge),
            max_tokens=max_tokens, temperature=temperature, json_mode=True,
        )
    except Exception:
        return None
    if not getattr(resp, "success", False):
        return None
    parsed = _parse_json(getattr(resp, "text", ""))
    if not parsed:
        return None
    return _build_idea(parsed, topic.strip(), bridge)


def analogy_batch(
    topic: str,
    claude_client: Any = _AUTOLOAD,
    n: int = 3,
    domains: Optional[List[str]] = None,
    seed: Optional[int] = None,
) -> List[Idea]:
    """End-to-end: pick `n` far domains, build a bridge for each, transplant a
    method through it. Domains are sampled without replacement from
    `domains` (default DEFAULT_DOMAINS).
    """
    if not topic or not topic.strip():
        raise ValueError("topic must be non-empty")
    if n <= 0:
        return []
    pool = list(domains) if domains else list(DEFAULT_DOMAINS)
    if not pool:
        return []
    rng = random.Random(seed) if seed is not None else random
    rng.shuffle(pool)
    chosen = pool[:max(1, n)]
    out: List[Idea] = []
    for domain in chosen:
        bridge = extract_analogy(topic, domain, claude_client=claude_client)
        if bridge is None:
            continue
        idea = generate_from_analogy(topic, bridge, claude_client=claude_client)
        if idea is not None:
            out.append(idea)
    return out
