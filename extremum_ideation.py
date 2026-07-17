"""
extremum_ideation.py — push one axis to its limit.

Most novelty modes operate in the same regime as ordinary research. This one
forces non-incremental thinking by pinning one design axis to an extreme
(10× data, 1/100× compute, 1-shot, real-time, 1B params, 1K params) and
asking: what ideas only make sense at *that* regime?

Two phases:
  1. propose_extreme(topic, axis, direction) — LLM names a specific
     extreme operating point on the chosen axis, with the magnitude and
     why the regime is hard.
  2. generate_at_extreme(topic, regime) — LLM proposes ONE idea that only
     makes sense in that regime — must exploit the regime, not just
     tolerate it.

Result ideas have `source_strategy='T'` (exTremum) with the regime stored
on `execution_meta.regime`. Useful for finding ideas that look obvious in
hindsight but require a hard commitment to one constraint.

Public API:
    AXES                                           → catalog of axes & directions
    Regime                                         → dataclass
    propose_extreme(topic, axis, direction, ...)   → Optional[Regime]
    generate_at_extreme(topic, regime, ...)        → Optional[Idea]
    extremum_batch(topic, ..., n)                  → List[Idea]
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from models.idea import Idea, METHODOLOGY_TYPES, NOVELTY_LEVELS


_AUTOLOAD = object()


# axis_name → (direction_name → human description)
AXES: Dict[str, Dict[str, str]] = {
    "compute": {
        "minimal":  "fits on a single laptop CPU, no GPU",
        "maximal":  "uses 10,000+ GPU-hours / a frontier cluster",
    },
    "data": {
        "minimal":  "one or zero training examples (zero/one-shot)",
        "maximal":  "billions of examples / full web-scale corpus",
    },
    "parameters": {
        "minimal":  "fewer than 1,000 trainable parameters",
        "maximal":  "more than 100 billion parameters",
    },
    "latency": {
        "minimal":  "real-time, sub-10ms end-to-end",
        "maximal":  "offline / batch only, can run for days",
    },
    "precision": {
        "minimal":  "1-bit / ternary quantized end-to-end",
        "maximal":  "double-precision fp64 throughout",
    },
    "supervision": {
        "minimal":  "no labels at all (fully unsupervised)",
        "maximal":  "exhaustive expert-annotated supervision per example",
    },
    "deployment": {
        "minimal":  "runs on edge device with no network",
        "maximal":  "spans a global multi-region cluster",
    },
    "interpretability": {
        "minimal":  "completely opaque, no inspection possible",
        "maximal":  "every prediction has a human-auditable proof",
    },
}


def _all_axis_direction_pairs() -> List[Tuple[str, str]]:
    return [(a, d) for a, dirs in AXES.items() for d in dirs.keys()]


@dataclass
class Regime:
    """A specific extreme operating point on one design axis."""
    axis: str = ""              # e.g., "compute"
    direction: str = ""         # "minimal" or "maximal"
    magnitude: str = ""         # concrete quantity (e.g., "1 CPU core, 4GB RAM")
    why_hard: str = ""          # what fundamentally breaks at this extreme
    what_changes: str = ""      # what tradeoff or assumption is forced to flip
    only_here: str = ""         # what only makes sense at this regime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "axis": self.axis,
            "direction": self.direction,
            "magnitude": self.magnitude,
            "why_hard": self.why_hard,
            "what_changes": self.what_changes,
            "only_here": self.only_here,
        }

    @property
    def label(self) -> str:
        return f"{self.axis}:{self.direction}"


_REGIME_SYSTEM = (
    "You are a research strategist who specializes in non-incremental "
    "ideas. The user gives a topic, a design axis, and a direction (minimal "
    "or maximal). Your job is to name a *specific* extreme operating point "
    "on that axis for this topic, with a concrete magnitude, and explain "
    "what fundamentally breaks at that regime — and what new opportunities "
    "only exist there. Be specific about quantities. Return ONLY valid JSON."
)


def _regime_user_prompt(topic: str, axis: str, direction: str) -> str:
    description = AXES.get(axis, {}).get(direction, "")
    return (
        f"Topic: {topic}\n"
        f"Axis: {axis}\n"
        f"Direction: {direction}\n"
        f"Sketch of the regime: {description}\n\n"
        f"Name a concrete extreme operating point on this axis for this "
        f"topic. State the magnitude in concrete units. Explain what "
        f"fundamentally breaks at this regime, what assumption is forced "
        f"to flip, and what becomes possible *only* here.\n\n"
        f"Return JSON:\n"
        "{\n"
        '  "magnitude": "<concrete quantity, e.g. \\"≤ 1k labelled examples\\">",\n'
        '  "why_hard": "<what fundamentally breaks at this regime>",\n'
        '  "what_changes": "<which tradeoff or assumption is forced to flip>",\n'
        '  "only_here": "<what becomes possible only at this regime>"\n'
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


def propose_extreme(
    topic: str,
    axis: str,
    direction: str,
    claude_client: Any = _AUTOLOAD,
    max_tokens: int = 600,
    temperature: float = 0.7,
) -> Optional[Regime]:
    if not topic or not topic.strip():
        raise ValueError("topic must be non-empty")
    if axis not in AXES:
        raise ValueError(f"axis must be one of {list(AXES.keys())}, got {axis!r}")
    if direction not in AXES[axis]:
        raise ValueError(
            f"direction must be one of {list(AXES[axis].keys())} "
            f"for axis {axis!r}, got {direction!r}"
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
            system=_REGIME_SYSTEM,
            user=_regime_user_prompt(topic.strip(), axis, direction),
            max_tokens=max_tokens, temperature=temperature, json_mode=True,
        )
    except Exception:
        return None
    if not getattr(resp, "success", False):
        return None
    parsed = _parse_json(getattr(resp, "text", ""))
    if not parsed:
        return None
    magnitude = str(parsed.get("magnitude") or "").strip()
    if not magnitude:
        return None
    return Regime(
        axis=axis, direction=direction,
        magnitude=magnitude[:400],
        why_hard=str(parsed.get("why_hard") or "")[:400],
        what_changes=str(parsed.get("what_changes") or "")[:400],
        only_here=str(parsed.get("only_here") or "")[:400],
    )


_IDEA_SYSTEM = (
    "You are a research scientist who excels at ideas that only make "
    "sense at one specific operating regime. The user gives you a topic "
    "and an extreme regime. Propose ONE research idea that *exploits* "
    "the regime — an idea that would be uninteresting or impossible in "
    "the normal operating range. The idea must actively use the "
    "regime's extreme, not just survive it. Output ONLY valid JSON. "
    f"methodology_type must be one of: {', '.join(METHODOLOGY_TYPES)}. "
    f"novelty_level must be one of: {', '.join(NOVELTY_LEVELS)}."
)


def _idea_user_prompt(topic: str, regime: Regime) -> str:
    return (
        f"Topic: {topic}\n\n"
        f"### Operating regime (pinned to extreme)\n"
        f"  Axis:           {regime.axis}\n"
        f"  Direction:      {regime.direction}\n"
        f"  Magnitude:      {regime.magnitude}\n"
        f"  Why hard:       {regime.why_hard}\n"
        f"  What flips:     {regime.what_changes}\n"
        f"  Only possible:  {regime.only_here}\n\n"
        f"### Instructions\n"
        f"Propose ONE research idea that *only* makes sense at this regime "
        f"— would be uninteresting in the normal range. The idea must "
        f"actively exploit the regime, not just tolerate it.\n\n"
        f"### Output\n"
        "Return JSON:\n"
        "{\n"
        '  "title": "<concise; signals the regime>",\n'
        '  "motivation": "<why this regime is worth committing to>",\n'
        '  "method": "<concrete approach; name the exploit explicitly>",\n'
        '  "hypothesis": "<falsifiable claim>",\n'
        '  "resources": "<resources needed at this regime>",\n'
        '  "expected_outcome": "<measurable outcome>",\n'
        '  "risk_assessment": "<what breaks if the regime relaxes>",\n'
        '  "source_strategy": "T",\n'
        f'  "methodology_type": "<one of: {", ".join(METHODOLOGY_TYPES)}>",\n'
        f'  "novelty_level": "<one of: {", ".join(NOVELTY_LEVELS)}>",\n'
        '  "regime_exploit": "<one sentence: how the idea uses the extreme>"\n'
        "}"
    )


def _build_idea(parsed: Dict[str, Any], topic: str,
                  regime: Regime) -> Optional[Idea]:
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
        source_strategy="T",
        methodology_type=method_type,
        novelty_level=novelty,
        generation=0,
        parent_title=None,
    )
    idea.execution_meta = {
        "regime": regime.to_dict(),
        "regime_exploit": str(parsed.get("regime_exploit", ""))[:400],
        "regen_mode": "extremum",
        "topic": topic,
    }
    return idea


def generate_at_extreme(
    topic: str,
    regime: Regime,
    claude_client: Any = _AUTOLOAD,
    max_tokens: int = 900,
    temperature: float = 0.85,
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
            user=_idea_user_prompt(topic.strip(), regime),
            max_tokens=max_tokens, temperature=temperature, json_mode=True,
        )
    except Exception:
        return None
    if not getattr(resp, "success", False):
        return None
    parsed = _parse_json(getattr(resp, "text", ""))
    if not parsed:
        return None
    return _build_idea(parsed, topic.strip(), regime)


def extremum_batch(
    topic: str,
    claude_client: Any = _AUTOLOAD,
    n: int = 3,
    pairs: Optional[List[Tuple[str, str]]] = None,
    seed: Optional[int] = None,
) -> List[Idea]:
    """End-to-end: pick `n` distinct (axis, direction) pairs without
    replacement, propose a regime for each, generate one idea that exploits
    each regime. `pairs` overrides the random pick.
    """
    if not topic or not topic.strip():
        raise ValueError("topic must be non-empty")
    if n <= 0:
        return []
    if pairs is None:
        pool = _all_axis_direction_pairs()
        rng = random.Random(seed) if seed is not None else random
        rng.shuffle(pool)
        pairs = pool[:max(1, n)]
    # Validate caller-supplied pairs early.
    for ax, dr in pairs:
        if ax not in AXES or dr not in AXES[ax]:
            raise ValueError(f"invalid (axis, direction) pair: ({ax!r}, {dr!r})")
    out: List[Idea] = []
    for ax, dr in pairs[:n]:
        regime = propose_extreme(topic, ax, dr, claude_client=claude_client)
        if regime is None:
            continue
        idea = generate_at_extreme(topic, regime, claude_client=claude_client)
        if idea is not None:
            out.append(idea)
    return out
