"""
constraint_stacking.py — multi-constraint hard-satisfaction ideation.

Forces ideas to satisfy ALL of a stacked set of constraints simultaneously
(e.g., "public data only" + "<1 hour on a single GPU" + "no neural nets").
Decades of design research show that hard constraints unlock creativity
by closing off the obvious solution paths.

Public API:
    CONSTRAINT_LIBRARY                              → Dict[category, List[str]]
    suggest_constraints(topic, claude_client, n)    → List[str]
    generate_with_constraints(topic, constraints, ...) → Optional[Idea]
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from models.idea import Idea, METHODOLOGY_TYPES, NOVELTY_LEVELS


_AUTOLOAD = object()


# Constraint library — curated catalog of hard constraints that historically
# unlock creative solutions when stacked. Grouped by category for the UI.
CONSTRAINT_LIBRARY: Dict[str, List[str]] = {
    "data": [
        "Uses only publicly available datasets",
        "No labeled data — fully self-supervised",
        "Synthetic data only — no real-world samples",
        "Single small dataset (≤10k examples)",
        "Multi-domain data with no domain labels",
        "Streaming / online data only (no replay buffer)",
    ],
    "compute": [
        "Runs on a single GPU in under 1 hour",
        "Runs on CPU only — no GPU available",
        "Mobile-device inference (<200MB, <500ms latency)",
        "Training cost ≤$100 in cloud compute",
        "Fits in 8GB of GPU memory",
        "Single-pass through data — no multi-epoch training",
    ],
    "method": [
        "No deep neural networks",
        "Fully interpretable end-to-end (no black boxes)",
        "Zero learned parameters — only analytic / closed-form",
        "Provable theoretical guarantees required",
        "No gradient-based optimization",
        "Must work without backpropagation",
    ],
    "output": [
        "Result must be a reusable open-source tool",
        "Must produce a new benchmark, not just numbers",
        "Findings explainable to a non-technical reader",
        "Must include a falsifiable counter-prediction",
        "Reproducible with a single command on a fresh laptop",
    ],
    "audience": [
        "Domain expert (not ML researcher) is the primary reader",
        "Useful in clinical / regulated environments",
        "Resource-constrained (developing-world) deployment",
        "Edge device with no internet connectivity",
    ],
}


_SYSTEM = (
    "You are a research scientist generating a single research idea that "
    "MUST simultaneously satisfy a stacked set of hard constraints. Treat "
    "each constraint as a non-negotiable design requirement, not a "
    "suggestion. If the constraint stack genuinely cannot be satisfied, "
    "say so in lineage_note and propose the closest feasible neighbor. "
    "Output ONLY valid JSON with the schema. methodology_type must be one "
    f"of: {', '.join(METHODOLOGY_TYPES)}. novelty_level must be one of: "
    f"{', '.join(NOVELTY_LEVELS)}."
)


def _user_prompt(topic: str, constraints: List[str]) -> str:
    constraint_lines = "\n".join(f"  {i + 1}. {c}" for i, c in enumerate(constraints))
    return (
        f"Topic: {topic}\n\n"
        f"### CONSTRAINTS (all must hold simultaneously)\n"
        f"{constraint_lines}\n\n"
        f"### Instructions\n"
        f"Propose ONE research idea on this topic that satisfies EVERY "
        f"constraint above at the same time. Be specific about *how* each "
        f"constraint is satisfied. Hard constraints are not a tradeoff — "
        f"do not say 'we relax X' or 'we approximate Y'.\n\n"
        f"### Output\n"
        "Return JSON:\n"
        "{\n"
        '  "title": "<concise>",\n'
        '  "motivation": "<why this matters under these constraints>",\n'
        '  "method": "<technical approach, naming how each constraint is met>",\n'
        '  "hypothesis": "<testable prediction>",\n'
        '  "resources": "<minimal resources>",\n'
        '  "expected_outcome": "<measurable result>",\n'
        '  "risk_assessment": "<main risks under the constraint stack>",\n'
        '  "source_strategy": "K",\n'
        f'  "methodology_type": "<one of {", ".join(METHODOLOGY_TYPES)}>",\n'
        f'  "novelty_level": "<one of {", ".join(NOVELTY_LEVELS)}>",\n'
        '  "constraints_satisfied": <list of strings — for each constraint, '
        'one line explaining how the idea satisfies it>,\n'
        '  "feasibility_note": "<one sentence — is the constraint stack '
        'genuinely satisfiable, or did we approximate?>"\n'
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


def suggest_constraints(
    topic: str,
    claude_client: Any = _AUTOLOAD,
    n: int = 3,
) -> List[str]:
    """Ask the LLM to propose N constraints likely to unlock novelty on
    `topic`. Falls back to a random sample from CONSTRAINT_LIBRARY when
    the LLM is unavailable."""
    if not topic or not topic.strip():
        raise ValueError("topic must be non-empty")

    if claude_client is _AUTOLOAD:
        try:
            from claude_provider import get_claude_client
            claude_client = get_claude_client()
        except Exception:
            claude_client = None

    if claude_client is None:
        # Deterministic fallback: pick from across categories
        import random
        rng = random.Random(hash(topic) & 0xFFFFFFFF)
        pool = []
        for cat in CONSTRAINT_LIBRARY.values():
            pool.extend(cat)
        rng.shuffle(pool)
        return pool[: max(1, int(n))]

    sys = (
        "You suggest hard research constraints that would unlock genuine "
        "novelty on a given topic. Return ONLY valid JSON with a "
        '"constraints" list of N concise constraint strings.'
    )
    user = (
        f"Topic: {topic}\n\n"
        f"Propose {n} hard constraints that, when stacked, would force a "
        f"researcher off the well-trodden path on this topic. Each "
        f"constraint should be specific and verifiable, not a vague "
        f"preference. Output JSON: "
        '{"constraints": ["constraint 1", ...]}'
    )
    try:
        resp = claude_client.call(
            system=sys, user=user, max_tokens=400, temperature=0.7,
            json_mode=True,
        )
    except Exception:
        return []
    if not getattr(resp, "success", False):
        return []
    parsed = _parse_json(getattr(resp, "text", ""))
    if not parsed or not isinstance(parsed.get("constraints"), list):
        return []
    return [str(c)[:200] for c in parsed["constraints"][:n]]


def generate_with_constraints(
    topic: str,
    constraints: List[str],
    claude_client: Any = _AUTOLOAD,
    max_tokens: int = 900,
    temperature: float = 0.75,
) -> Optional[Idea]:
    """Generate an idea satisfying ALL constraints. Returns None on failure."""
    if not topic or not topic.strip():
        raise ValueError("topic must be non-empty")
    constraints = [c.strip() for c in (constraints or []) if c and c.strip()]
    if not constraints:
        raise ValueError("at least one constraint is required")

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
            system=_SYSTEM,
            user=_user_prompt(topic.strip(), constraints),
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
        source_strategy="K",  # K = Constraint-stacked
        methodology_type=method_type,
        novelty_level=novelty,
        generation=0,
        parent_title=None,
    )
    sat = parsed.get("constraints_satisfied") or []
    if not isinstance(sat, list):
        sat = [str(sat)]
    idea.execution_meta = {
        "constraints": list(constraints),
        "constraints_satisfied": [str(s)[:300] for s in sat[:10]],
        "feasibility_note": str(parsed.get("feasibility_note", ""))[:400],
        "regen_mode": "constraint_stacking",
        "topic": topic.strip(),
    }
    return idea
