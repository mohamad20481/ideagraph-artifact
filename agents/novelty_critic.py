"""
agents/novelty_critic.py — adversarial originality critic + revision agent.

The existing ExecutionCritic scores an idea's `novelty` probe but doesn't
actively *attack* it. This module adds a specialist whose only job is to
say "this isn't as original as you think it is" — citing specific prior
work it echoes, pointing at the unoriginal components, and proposing
concrete pivots that would make it more genuinely novel.

Optionally pair with `attack_and_revise()` to chain: critique → revise →
return a strengthened idea whose lineage explicitly references the
critique that drove the changes.

Public API:
    NoveltyCritique                                  → dataclass
    critique_novelty(idea, claude_client=None)       → NoveltyCritique
    attack_and_revise(idea, claude_client=None)      → (critique, revised_idea)
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from models.idea import Idea, METHODOLOGY_TYPES, NOVELTY_LEVELS


_AUTOLOAD = object()


# ─────────────────────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NoveltyCritique:
    """Adversarial assessment of an idea's originality."""

    originality_score: float = 0.0          # 0..1, post-critique calibration
    overall_verdict: str = ""               # human label
    critiques: List[str] = field(default_factory=list)
    similar_prior_work: List[str] = field(default_factory=list)
    pivots: List[str] = field(default_factory=list)
    confidence: float = 0.0
    raw_response: str = ""
    used_llm: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "originality_score": self.originality_score,
            "overall_verdict": self.overall_verdict,
            "critiques": list(self.critiques),
            "similar_prior_work": list(self.similar_prior_work),
            "pivots": list(self.pivots),
            "confidence": self.confidence,
            "used_llm": self.used_llm,
            "error": self.error,
        }

    def summary(self) -> str:
        return (
            f"{self.overall_verdict or '?'} · "
            f"originality={self.originality_score:.2f} · "
            f"{len(self.critiques)} critiques · "
            f"{len(self.pivots)} pivots"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Prompt construction
# ─────────────────────────────────────────────────────────────────────────────

_CRITIC_SYSTEM = (
    "You are an adversarial reviewer whose ONE job is to attack the "
    "originality of a research idea. You do NOT evaluate feasibility, "
    "significance, or clarity — only how genuinely novel the idea is. "
    "You are tough but precise: cite specific prior work (real papers, "
    "methods, frameworks) the idea echoes, name the unoriginal components, "
    "and propose concrete pivots that would make it more novel. Do not be "
    "polite for politeness's sake — research moves forward through honest "
    "critique. Return ONLY valid JSON matching the schema."
)


def _coerce_unit(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return max(0.0, min(1.0, v))
    except (TypeError, ValueError):
        return default


def _user_prompt(idea: Any) -> str:
    if hasattr(idea, "to_dict"):
        d = idea.to_dict()
    elif isinstance(idea, dict):
        d = idea
    else:
        d = {}
    return (
        f"Attack the originality of this research idea:\n\n"
        f"Title: {d.get('title', '')}\n"
        f"Motivation: {str(d.get('motivation', ''))[:400]}\n"
        f"Method: {str(d.get('method', ''))[:600]}\n"
        f"Hypothesis: {str(d.get('hypothesis', ''))[:300]}\n"
        f"Expected outcome: {str(d.get('expected_outcome', ''))[:200]}\n"
        f"Methodology type: {d.get('methodology_type', '?')}\n"
        f"Novelty level (self-reported): {d.get('novelty_level', '?')}\n\n"
        "Be specific. Return JSON with these keys:\n"
        "{\n"
        '  "originality_score":   <number 0..1; 1 = genuinely novel, '
        '0 = derivative of existing work>,\n'
        '  "overall_verdict":     <one of: "highly original", '
        '"moderately original", "incremental", "derivative">,\n'
        '  "critiques":           <list of 2-5 strings, each a specific '
        'unoriginal component of the idea>,\n'
        '  "similar_prior_work":  <list of 1-4 strings naming specific '
        'real papers, methods, or frameworks this echoes>,\n'
        '  "pivots":              <list of 2-4 strings, each a concrete '
        'way to make this more novel>,\n'
        '  "confidence":          <number 0..1; how confident in this '
        'critique you are>\n'
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


# ─────────────────────────────────────────────────────────────────────────────
# Main entry points
# ─────────────────────────────────────────────────────────────────────────────

def critique_novelty(
    idea: Any,
    claude_client: Any = _AUTOLOAD,
    max_tokens: int = 600,
    temperature: float = 0.3,
) -> NoveltyCritique:
    """Run the adversarial novelty critic on an idea.

    Returns a `NoveltyCritique` with originality_score, specific critiques,
    similar prior work named, and suggested pivots. When the LLM is
    unavailable, returns an error result rather than raising.
    """
    if claude_client is _AUTOLOAD:
        try:
            from claude_provider import get_claude_client
            claude_client = get_claude_client()
        except Exception:
            claude_client = None
    if claude_client is None:
        return NoveltyCritique(
            originality_score=0.5,
            overall_verdict="(no LLM available)",
            used_llm=False,
            error="LLM client not configured",
        )

    try:
        resp = claude_client.call(
            system=_CRITIC_SYSTEM,
            user=_user_prompt(idea),
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=True,
        )
    except Exception as e:
        return NoveltyCritique(
            originality_score=0.5, overall_verdict="(critic failed)",
            used_llm=False, error=f"LLM exception: {e}",
        )

    if not getattr(resp, "success", False):
        return NoveltyCritique(
            originality_score=0.5, overall_verdict="(critic failed)",
            used_llm=False,
            error=f"LLM call failed: {getattr(resp, 'error', '?')}",
        )

    parsed = _parse_json(getattr(resp, "text", ""))
    if not parsed:
        return NoveltyCritique(
            originality_score=0.5, overall_verdict="(unparseable response)",
            used_llm=False, error="LLM response was not valid JSON",
            raw_response=getattr(resp, "text", "")[:500],
        )

    # Coerce + bound everything
    verdicts = ("highly original", "moderately original",
                "incremental", "derivative")
    verdict = str(parsed.get("overall_verdict") or "").strip().lower()
    if verdict not in verdicts:
        score = _coerce_unit(parsed.get("originality_score"), 0.5)
        verdict = (
            "highly original" if score >= 0.75 else
            "moderately original" if score >= 0.50 else
            "incremental" if score >= 0.30 else "derivative"
        )

    critiques = parsed.get("critiques") or []
    pivots = parsed.get("pivots") or []
    similar = parsed.get("similar_prior_work") or []
    for lst, src in ((critiques, "critiques"),
                      (pivots, "pivots"),
                      (similar, "similar_prior_work")):
        if not isinstance(lst, list):
            parsed[src] = [str(lst)]

    return NoveltyCritique(
        originality_score=_coerce_unit(parsed.get("originality_score"), 0.5),
        overall_verdict=verdict,
        critiques=[str(c)[:300] for c in (parsed["critiques"]
                                            if "critiques" in parsed
                                            else critiques)[:6]],
        similar_prior_work=[str(s)[:200] for s in (parsed["similar_prior_work"]
                                                      if "similar_prior_work" in parsed
                                                      else similar)[:6]],
        pivots=[str(p)[:300] for p in (parsed["pivots"]
                                          if "pivots" in parsed
                                          else pivots)[:6]],
        confidence=_coerce_unit(parsed.get("confidence"), 0.5),
        raw_response=getattr(resp, "text", "")[:1500],
        used_llm=True,
    )


_REVISER_SYSTEM = (
    "You are a research scientist revising your own idea after an "
    "adversarial originality critique. Your job is to produce ONE revised "
    "version of the idea that directly addresses each of the critic's "
    "complaints — making it genuinely more novel without sacrificing "
    "feasibility. Output ONLY valid JSON with the same idea schema, plus "
    "a 'novelty_pivots_applied' list naming which critic pivots you "
    "incorporated."
)


def _reviser_user_prompt(idea: Any, critique: NoveltyCritique) -> str:
    if hasattr(idea, "to_dict"):
        d = idea.to_dict()
    elif isinstance(idea, dict):
        d = idea
    else:
        d = {}
    critiques_txt = "\n".join(f"  • {c}" for c in critique.critiques)
    pivots_txt = "\n".join(f"  • {p}" for p in critique.pivots)
    similar_txt = "\n".join(f"  • {s}" for s in critique.similar_prior_work)
    return (
        f"### Original idea\n"
        f"Title: {d.get('title','')}\n"
        f"Motivation: {str(d.get('motivation',''))[:400]}\n"
        f"Method: {str(d.get('method',''))[:600]}\n"
        f"Hypothesis: {str(d.get('hypothesis',''))[:300]}\n"
        f"Resources: {str(d.get('resources',''))[:200]}\n"
        f"Expected outcome: {str(d.get('expected_outcome',''))[:200]}\n"
        f"Risk assessment: {str(d.get('risk_assessment',''))[:200]}\n"
        f"methodology_type: {d.get('methodology_type','')}\n"
        f"novelty_level: {d.get('novelty_level','')}\n\n"
        f"### Critic's verdict: {critique.overall_verdict}\n"
        f"Originality score: {critique.originality_score:.2f}\n\n"
        f"### Critic's complaints (each MUST be addressed)\n"
        f"{critiques_txt}\n\n"
        f"### Critic's suggested pivots\n"
        f"{pivots_txt}\n\n"
        f"### Prior work the critic says you echo\n"
        f"{similar_txt}\n\n"
        f"### Output\n"
        "Produce ONE revised idea that incorporates at least 2 of the "
        "critic's pivots, addresses ALL of the critiques, and is genuinely "
        "different from the cited prior work. Output JSON:\n"
        "{\n"
        '  "title": "<concise revised title>",\n'
        '  "motivation": "<motivation>",\n'
        '  "method": "<revised method>",\n'
        '  "hypothesis": "<falsifiable hypothesis>",\n'
        '  "resources": "<resources needed>",\n'
        '  "expected_outcome": "<measurable outcome>",\n'
        '  "risk_assessment": "<main risks>",\n'
        '  "source_strategy": "N",\n'
        f'  "methodology_type": "<one of: {", ".join(METHODOLOGY_TYPES)}>",\n'
        f'  "novelty_level": "<one of: {", ".join(NOVELTY_LEVELS)}>",\n'
        '  "novelty_pivots_applied": <list of 2-4 strings naming which '
        'pivots you applied>\n'
        "}"
    )


def attack_and_revise(
    idea: Any,
    claude_client: Any = _AUTOLOAD,
    max_critic_tokens: int = 600,
    max_reviser_tokens: int = 900,
) -> Tuple[NoveltyCritique, Optional[Idea]]:
    """Chain: critique → revise. Returns (critique, revised_idea).

    The revised idea has `source_strategy='N'` (Novelty-revision),
    `parent_title` set to the original title, `generation` incremented,
    and `execution_meta` populated with the pivots applied.
    """
    critique = critique_novelty(idea, claude_client=claude_client,
                                  max_tokens=max_critic_tokens)
    if not critique.used_llm or not critique.critiques:
        return critique, None

    # We need a real client now for the revision step
    if claude_client is _AUTOLOAD:
        try:
            from claude_provider import get_claude_client
            claude_client = get_claude_client()
        except Exception:
            claude_client = None
    if claude_client is None:
        return critique, None

    try:
        resp = claude_client.call(
            system=_REVISER_SYSTEM,
            user=_reviser_user_prompt(idea, critique),
            max_tokens=max_reviser_tokens,
            temperature=0.7,
            json_mode=True,
        )
    except Exception:
        return critique, None
    if not getattr(resp, "success", False):
        return critique, None

    parsed = _parse_json(getattr(resp, "text", ""))
    if not parsed:
        return critique, None
    required = ("title", "method", "hypothesis")
    if not all(str(parsed.get(k, "")).strip() for k in required):
        return critique, None

    method_type = parsed.get("methodology_type") or ""
    if method_type not in METHODOLOGY_TYPES:
        method_type = None
    novelty = parsed.get("novelty_level") or ""
    if novelty not in NOVELTY_LEVELS:
        novelty = None

    parent_dict = (idea.to_dict() if hasattr(idea, "to_dict")
                    else (idea if isinstance(idea, dict) else {}))

    revised = Idea(
        title=str(parsed.get("title", ""))[:200],
        motivation=str(parsed.get("motivation", ""))[:1000],
        method=str(parsed.get("method", ""))[:2000],
        hypothesis=str(parsed.get("hypothesis", ""))[:1000],
        resources=str(parsed.get("resources", ""))[:500],
        expected_outcome=str(parsed.get("expected_outcome", ""))[:500],
        risk_assessment=str(parsed.get("risk_assessment", ""))[:500],
        source_strategy="N",
        methodology_type=method_type,
        novelty_level=novelty,
        generation=int(parent_dict.get("generation") or 0) + 1,
        parent_title=str(parent_dict.get("title", ""))[:200],
    )
    pivots_applied = parsed.get("novelty_pivots_applied") or []
    if not isinstance(pivots_applied, list):
        pivots_applied = [str(pivots_applied)]
    revised.execution_meta = {
        "novelty_critique": critique.to_dict(),
        "pivots_applied": [str(p)[:200] for p in pivots_applied[:6]],
        "regen_mode": "novelty_revision",
        "lineage_note": (
            f"Adversarial revision: addressed "
            f"{len(critique.critiques)} originality critiques."
        ),
    }
    return critique, revised
