"""
stakeholder_pareto_ideation.py — design with a measurable win for each stakeholder.

Most novelty modes optimize for a single objective. This one forces
multi-party negotiation into the design: pick 3+ stakeholders relevant
to the topic (researcher, end-user, funder, regulator, domain expert, …)
and design research with a *measurable* win for each — no party gets a
hand-wave. If you cannot name a metric per stakeholder, the design fails.

Two phases:
  1. cast_stakeholders(topic, roles, n) — LLM expands a chosen set of
     stakeholder roles into concrete parties for this topic, each with a
     concrete win condition and a measurable metric.
  2. design_pareto_idea(topic, stakeholders) — LLM designs ONE research
     idea whose method explicitly produces a measurable win for every
     stakeholder, and names the tradeoffs being made.

Ideas have `source_strategy='S'` (Stakeholder-Pareto). The stakeholder
cast lives on `execution_meta.stakeholders`. The prompt enforces a
per-stakeholder metric — vague platitudes about "society benefits" fail
the validator.

Public API:
    Stakeholder                                          → dataclass
    DEFAULT_ROLES                                        → catalog
    cast_stakeholders(topic, roles, ...)                 → List[Stakeholder]
    design_pareto_idea(topic, stakeholders, ...)         → Optional[Idea]
    stakeholder_pareto_batch(topic, ..., n)              → List[Idea]
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from models.idea import Idea, METHODOLOGY_TYPES, NOVELTY_LEVELS


_AUTOLOAD = object()


# Role lenses the LLM picks from. Each role is instantiated as a specific
# Stakeholder for the topic. Researcher/end-user are almost always in the
# cast; the others rotate to force breadth.
DEFAULT_ROLES: List[str] = [
    "researcher",            # the one running future studies on the contribution
    "end_user",              # the human who consumes the result of the system
    "funder",                # whoever pays for the work (agency, foundation, company)
    "regulator",             # the body that polices the deployment context
    "domain_expert",         # the practitioner who knows the field deeply
    "operator",              # the person who runs the deployed system day-to-day
    "auditor",               # the person tasked with verifying claims after the fact
    "future_researcher",     # the person extending the work in 3-5 years
]


@dataclass
class Stakeholder:
    """A concrete party with a concrete measurable win condition."""
    role: str = ""                 # one of DEFAULT_ROLES
    name: str = ""                 # the concrete party (e.g., "NIH study section")
    win_condition: str = ""        # what success means for them
    metric: str = ""               # a *measurable* quantity for the win
    risk_if_ignored: str = ""      # what goes wrong if this party is unhappy

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "name": self.name,
            "win_condition": self.win_condition,
            "metric": self.metric,
            "risk_if_ignored": self.risk_if_ignored,
        }


_CAST_SYSTEM = (
    "You are a research strategist designing multi-stakeholder studies. "
    "Given a topic and a list of stakeholder roles, instantiate each "
    "role as a CONCRETE party for this topic. For each party, name a "
    "win condition AND a measurable metric for that win — not a "
    "platitude. Be specific about the party (not 'society' but a named "
    "type of organization or user). Return ONLY valid JSON."
)


def _cast_user_prompt(topic: str, roles: List[str]) -> str:
    return (
        f"Topic: {topic}\n"
        f"Roles to instantiate: {', '.join(roles)}\n\n"
        f"For each role, name the concrete party and give them a "
        f"measurable win. 'Society benefits' is not a metric. 'Reduces "
        f"clinician note-writing time by ≥ 20 minutes per shift' IS a "
        f"metric. State what goes wrong if this party is ignored.\n\n"
        f"Return JSON:\n"
        "{\n"
        '  "stakeholders": [\n'
        "    {\n"
        '      "role": "<one of the roles above>",\n'
        '      "name": "<concrete party, e.g. \\"emergency-department clinicians\\">",\n'
        '      "win_condition": "<what success means for them>",\n'
        '      "metric": "<measurable quantity for the win>",\n'
        '      "risk_if_ignored": "<what breaks if this party is unhappy>"\n'
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


def cast_stakeholders(
    topic: str,
    roles: List[str],
    claude_client: Any = _AUTOLOAD,
    max_tokens: int = 900,
    temperature: float = 0.7,
) -> List[Stakeholder]:
    if not topic or not topic.strip():
        raise ValueError("topic must be non-empty")
    if not roles:
        raise ValueError("roles must be non-empty")
    for r in roles:
        if r not in DEFAULT_ROLES:
            raise ValueError(
                f"invalid role {r!r}; must be one of {DEFAULT_ROLES}"
            )
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
            system=_CAST_SYSTEM,
            user=_cast_user_prompt(topic.strip(), list(roles)),
            max_tokens=max_tokens, temperature=temperature, json_mode=True,
        )
    except Exception:
        return []
    if not getattr(resp, "success", False):
        return []
    parsed = _parse_json(getattr(resp, "text", ""))
    if not parsed or not isinstance(parsed.get("stakeholders"), list):
        return []
    out: List[Stakeholder] = []
    for d in parsed["stakeholders"][:max(len(roles), 10)]:
        if not isinstance(d, dict):
            continue
        role = str(d.get("role") or "").strip()
        name = str(d.get("name") or "").strip()
        metric = str(d.get("metric") or "").strip()
        if not (role in DEFAULT_ROLES and name and metric):
            continue
        out.append(Stakeholder(
            role=role,
            name=name[:200],
            win_condition=str(d.get("win_condition") or "")[:400],
            metric=metric[:400],
            risk_if_ignored=str(d.get("risk_if_ignored") or "")[:400],
        ))
    return out


_DESIGN_SYSTEM = (
    "You are a researcher designing a multi-stakeholder Pareto study. "
    "The user gives you a topic and a cast of stakeholders, each with a "
    "named measurable win. Design ONE study that produces a MEASURABLE "
    "win for EVERY stakeholder in the cast. Name the tradeoffs explicitly "
    "— if one party's win comes at another's cost, say where the "
    "compromise lives. Do NOT hand-wave one stakeholder; if you can't "
    "measure their win, the design fails. Output ONLY valid JSON. "
    f"methodology_type must be one of: {', '.join(METHODOLOGY_TYPES)}. "
    f"novelty_level must be one of: {', '.join(NOVELTY_LEVELS)}."
)


def _design_user_prompt(topic: str, cast: List[Stakeholder]) -> str:
    cast_block = "\n".join(
        f"  - [{s.role}] {s.name} — metric: {s.metric}; "
        f"win: {s.win_condition}"
        for s in cast
    )
    return (
        f"Topic: {topic}\n\n"
        f"### Stakeholder cast (every party needs a measurable win)\n"
        f"{cast_block}\n\n"
        f"### Instructions\n"
        f"Design ONE study that produces a measurable win for EVERY "
        f"stakeholder above. Name the tradeoffs explicitly — if one "
        f"win costs another, say where. If you cannot measure any "
        f"party's win, mark the design as invalid in risk_assessment.\n\n"
        f"### Output\n"
        "Return JSON:\n"
        "{\n"
        '  "title": "<concise; signals multi-party design>",\n'
        '  "motivation": "<why aligning all these parties matters>",\n'
        '  "method": "<the design that yields a measurable win per party>",\n'
        '  "hypothesis": "<falsifiable conjunction across all wins>",\n'
        '  "resources": "<resources including coordination cost>",\n'
        '  "expected_outcome": "<per-stakeholder measurable outcome bundle>",\n'
        '  "risk_assessment": "<technical risks AND coordination failure modes>",\n'
        '  "source_strategy": "S",\n'
        f'  "methodology_type": "<one of: {", ".join(METHODOLOGY_TYPES)}>",\n'
        f'  "novelty_level": "<one of: {", ".join(NOVELTY_LEVELS)}>",\n'
        '  "tradeoffs_named": "<one sentence: where the design forces compromise>",\n'
        '  "per_stakeholder_metric": "<JSON object: role -> measurable metric>"\n'
        "}"
    )


def _build_idea(parsed: Dict[str, Any], topic: str,
                  cast: List[Stakeholder]) -> Optional[Idea]:
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
        source_strategy="S",
        methodology_type=method_type,
        novelty_level=novelty,
        generation=0,
        parent_title=None,
    )
    # per_stakeholder_metric may arrive as a dict or a string; keep whatever
    # the LLM gave us but clip length defensively.
    psm = parsed.get("per_stakeholder_metric")
    if not isinstance(psm, (dict, str)):
        psm = ""
    if isinstance(psm, str):
        psm = psm[:800]
    idea.execution_meta = {
        "stakeholders": [s.to_dict() for s in cast],
        "tradeoffs_named": str(parsed.get("tradeoffs_named", ""))[:400],
        "per_stakeholder_metric": psm,
        "regen_mode": "stakeholder_pareto",
        "topic": topic,
    }
    return idea


def design_pareto_idea(
    topic: str,
    stakeholders: List[Stakeholder],
    claude_client: Any = _AUTOLOAD,
    max_tokens: int = 1100,
    temperature: float = 0.7,
) -> Optional[Idea]:
    if not topic or not topic.strip():
        raise ValueError("topic must be non-empty")
    if len(stakeholders) < 2:
        raise ValueError("at least two stakeholders required for a Pareto design")
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
            user=_design_user_prompt(topic.strip(), stakeholders),
            max_tokens=max_tokens, temperature=temperature, json_mode=True,
        )
    except Exception:
        return None
    if not getattr(resp, "success", False):
        return None
    parsed = _parse_json(getattr(resp, "text", ""))
    if not parsed:
        return None
    return _build_idea(parsed, topic.strip(), stakeholders)


def stakeholder_pareto_batch(
    topic: str,
    claude_client: Any = _AUTOLOAD,
    n: int = 3,
    cast_size: int = 3,
    roles: Optional[List[str]] = None,
    seed: Optional[int] = None,
) -> List[Idea]:
    """End-to-end: for each of `n` runs, cast `cast_size` stakeholders
    (random subset of `roles` or DEFAULT_ROLES) and design one Pareto
    idea. Different random subsets across runs produce structurally
    different designs.
    """
    if not topic or not topic.strip():
        raise ValueError("topic must be non-empty")
    if n <= 0:
        return []
    if cast_size < 2:
        raise ValueError("cast_size must be >= 2")
    pool_all = list(roles) if roles else list(DEFAULT_ROLES)
    for r in pool_all:
        if r not in DEFAULT_ROLES:
            raise ValueError(f"invalid role {r!r}")
    if len(pool_all) < cast_size:
        raise ValueError(
            f"cast_size={cast_size} exceeds available roles={len(pool_all)}"
        )
    rng = random.Random(seed) if seed is not None else random.Random()
    out: List[Idea] = []
    for _ in range(n):
        pool = list(pool_all)
        rng.shuffle(pool)
        chosen = pool[:cast_size]
        cast = cast_stakeholders(topic, chosen, claude_client=claude_client)
        if len(cast) < 2:
            continue
        idea = design_pareto_idea(topic, cast, claude_client=claude_client)
        if idea is not None:
            out.append(idea)
    return out
