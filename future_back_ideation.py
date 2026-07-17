"""
future_back_ideation.py — imagine the field at a future date, then
back-propagate research today that would have enabled that vision.

Two-phase: imagine_future() asks the LLM to describe a coherent scenario
at year Y (optionally tagged with a flavor — best-case, worst-case,
surprising); back_propagate() then asks what research RIGHT NOW would
most directly enable that scenario.

Public API:
    FutureVision                                       → dataclass
    SCENARIOS                                          → List[str]
    imagine_future(topic, year, scenario, ...)         → Optional[FutureVision]
    back_propagate(topic, vision, ...)                 → Optional[Idea]
    future_back_batch(topic, n=3, scenarios=None, ...) → List[Idea]
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from models.idea import Idea, METHODOLOGY_TYPES, NOVELTY_LEVELS


_AUTOLOAD = object()


SCENARIOS: List[str] = [
    "best_case",     # field made the right calls
    "worst_case",    # field went sideways; what to learn
    "surprising",   # unexpected paradigm shift
    "neutral",       # plausible default trajectory
]


@dataclass
class FutureVision:
    topic: str = ""
    year: int = 2040
    scenario: str = "neutral"
    description: str = ""        # 2-3 paragraph picture
    what_is_solved: List[str] = field(default_factory=list)
    what_is_commonplace: List[str] = field(default_factory=list)
    what_is_still_hard: List[str] = field(default_factory=list)
    surprising_capability: str = ""   # the "wait, that's possible?" thing

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic, "year": self.year, "scenario": self.scenario,
            "description": self.description,
            "what_is_solved": list(self.what_is_solved),
            "what_is_commonplace": list(self.what_is_commonplace),
            "what_is_still_hard": list(self.what_is_still_hard),
            "surprising_capability": self.surprising_capability,
        }


_FUTURE_SYSTEM = (
    "You are a research forecaster. The user gives you a topic, a target "
    "year, and a scenario flavor. Your job is to paint a concrete picture "
    "of the field at that year under that scenario — what's solved, what's "
    "commonplace, what's still hard. Be specific and concrete; avoid "
    "vague futurism. Return ONLY valid JSON matching the schema."
)


def _future_user_prompt(topic: str, year: int, scenario: str) -> str:
    flavor_hints = {
        "best_case": "The field made the right strategic calls. What does that look like?",
        "worst_case": "The field went sideways or stalled. What's missing that we should be working on now?",
        "surprising": "A paradigm shift happened that nobody saw coming. Pick a non-obvious one.",
        "neutral": "Default plausible trajectory — extrapolate today's trends honestly.",
    }
    flavor = flavor_hints.get(scenario, flavor_hints["neutral"])
    return (
        f"Topic: {topic}\n"
        f"Year: {year}\n"
        f"Scenario: {scenario} — {flavor}\n\n"
        "Paint a concrete picture of this field in the given year and "
        "scenario. Be specific.\n\n"
        "Return JSON:\n"
        "{\n"
        '  "description": "<2-3 paragraph picture>",\n'
        '  "what_is_solved":      <list of 3-5 specific things that '
        'are routinely solved by then>,\n'
        '  "what_is_commonplace": <list of 3-5 capabilities that are '
        'now everyday>,\n'
        '  "what_is_still_hard":  <list of 3-5 things that remain '
        'unsolved or get harder>,\n'
        '  "surprising_capability": "<one specific capability that '
        'would surprise a researcher from today>"\n'
        "}"
    )


_BACK_SYSTEM = (
    "You are a research strategist. Given a vision of a field in a future "
    "year, your job is to identify ONE research direction that, if pursued "
    "today, would most directly enable that vision. Work backward from the "
    "future capabilities to the foundational research that would have to "
    "happen first. Be specific and technically concrete. Output ONLY valid "
    "JSON with the idea schema. methodology_type must be one of: "
    f"{', '.join(METHODOLOGY_TYPES)}. novelty_level must be one of: "
    f"{', '.join(NOVELTY_LEVELS)}."
)


def _back_user_prompt(topic: str, vision: FutureVision) -> str:
    solved = "\n".join(f"  • {x}" for x in vision.what_is_solved)
    common = "\n".join(f"  • {x}" for x in vision.what_is_commonplace)
    hard = "\n".join(f"  • {x}" for x in vision.what_is_still_hard)
    return (
        f"Topic: {topic}\n"
        f"Year of vision: {vision.year} ({vision.scenario})\n\n"
        f"### Vision\n{vision.description}\n\n"
        f"### What's solved by {vision.year}\n{solved}\n\n"
        f"### What's commonplace by {vision.year}\n{common}\n\n"
        f"### What's still hard\n{hard}\n\n"
        f"### Surprising capability\n{vision.surprising_capability}\n\n"
        "### Instructions\n"
        "Identify ONE research direction worth pursuing TODAY that, if "
        "successful, would most directly enable this future. Not the "
        "incremental next paper — the foundational missing piece. State "
        "explicitly in the motivation how this idea connects backward "
        "from the future vision.\n\n"
        "Return JSON:\n"
        "{\n"
        '  "title": "<concise>",\n'
        '  "motivation": "<why this enables the future vision; cite '
        'specific elements from the vision>",\n'
        '  "method": "<concrete technical approach>",\n'
        '  "hypothesis": "<testable prediction>",\n'
        '  "resources": "<resources needed>",\n'
        '  "expected_outcome": "<measurable result>",\n'
        '  "risk_assessment": "<main risks>",\n'
        '  "source_strategy": "U",\n'
        f'  "methodology_type": "<one of: {", ".join(METHODOLOGY_TYPES)}>",\n'
        f'  "novelty_level": "<one of: {", ".join(NOVELTY_LEVELS)}>",\n'
        '  "future_link": "<one sentence: how this idea connects to '
        'the vision>"\n'
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


def imagine_future(
    topic: str,
    year: int = 2040,
    scenario: str = "neutral",
    claude_client: Any = _AUTOLOAD,
    max_tokens: int = 700,
    temperature: float = 0.85,
) -> Optional[FutureVision]:
    if not topic or not topic.strip():
        raise ValueError("topic must be non-empty")
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}, got {scenario!r}")
    if not (2026 <= int(year) <= 2100):
        raise ValueError("year must be between 2026 and 2100")

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
            system=_FUTURE_SYSTEM,
            user=_future_user_prompt(topic.strip(), int(year), scenario),
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
    if not str(parsed.get("description", "")).strip():
        return None

    def _str_list(v):
        if isinstance(v, list):
            return [str(x)[:300] for x in v[:8]]
        return [str(v)[:300]] if v else []

    return FutureVision(
        topic=topic.strip(),
        year=int(year),
        scenario=scenario,
        description=str(parsed.get("description", ""))[:2000],
        what_is_solved=_str_list(parsed.get("what_is_solved")),
        what_is_commonplace=_str_list(parsed.get("what_is_commonplace")),
        what_is_still_hard=_str_list(parsed.get("what_is_still_hard")),
        surprising_capability=str(parsed.get("surprising_capability", ""))[:400],
    )


def back_propagate(
    topic: str,
    vision: FutureVision,
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
            system=_BACK_SYSTEM,
            user=_back_user_prompt(topic.strip(), vision),
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
        source_strategy="U",  # U = Futures (back-propagated)
        methodology_type=method_type,
        novelty_level=novelty,
        generation=0,
        parent_title=None,
    )
    idea.execution_meta = {
        "vision": vision.to_dict(),
        "future_link": str(parsed.get("future_link", ""))[:400],
        "regen_mode": "future_back",
        "topic": topic.strip(),
    }
    return idea


def future_back_batch(
    topic: str,
    claude_client: Any = _AUTOLOAD,
    n: int = 3,
    scenarios: Optional[List[str]] = None,
    year: int = 2040,
) -> List[Idea]:
    """Generate N ideas by imagining N different scenarios + back-propagating."""
    if not topic or not topic.strip():
        raise ValueError("topic must be non-empty")
    if scenarios is None:
        scenarios = ["best_case", "surprising", "worst_case", "neutral"][:max(1, n)]
    out: List[Idea] = []
    for sc in scenarios[:n]:
        vision = imagine_future(topic, year=year, scenario=sc,
                                  claude_client=claude_client)
        if vision is None:
            continue
        idea = back_propagate(topic, vision, claude_client=claude_client)
        if idea is not None:
            out.append(idea)
    return out
