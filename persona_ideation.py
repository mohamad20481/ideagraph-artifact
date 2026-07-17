"""
persona_ideation.py — parallel persona-swap ideation with diversity filter.

Cheaper and more controllable than the Multi-LLM Ensemble: instead of
exploiting *model-family* priors, exploit *persona* priors. Each persona
is a tight role-prompt that frames the same topic from a fundamentally
different vantage point — and you're forcing the same model to inhabit
each one in turn, so any diversity comes from the role, not the family.

Personas are intentionally diverse in stance (skeptic vs. enthusiast,
generalist vs. specialist, methodologist vs. visionary). After generation,
the existing diversity filter from multi_llm_ensemble keeps only ideas
that aren't near-duplicates of each other.

Public API:
    PERSONAS                                       → Dict[id, dict]
    generate_under_persona(topic, persona_id, ...) → Optional[Idea]
    persona_swap(topic, persona_ids, ...)          → PersonaResult
"""
from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from models.idea import Idea, METHODOLOGY_TYPES, NOVELTY_LEVELS


_AUTOLOAD = object()


# ─────────────────────────────────────────────────────────────────────────────
# Persona catalog
# ─────────────────────────────────────────────────────────────────────────────

PERSONAS: Dict[str, Dict[str, Any]] = {
    "skeptic": {
        "label": "🤨 Skeptic",
        "tagline": "Defaults to disbelieving; designs falsification experiments",
        "instruction": (
            "You are a deeply skeptical researcher whose default stance is "
            "that most popular methods don't work as advertised. Propose an "
            "idea that takes a popular method seriously enough to set up a "
            "real falsification test — not just a benchmark, a fair "
            "experiment where a clear negative result would matter."
        ),
        "default_temp": 0.65,
    },
    "industry_practitioner": {
        "label": "🏭 Industry Practitioner",
        "tagline": "Deployment-first; cares about real-world cost / latency / risk",
        "instruction": (
            "You are a senior engineer at a company that ships ML systems to "
            "real users. You don't care about leaderboards — you care about "
            "deployment cost, latency, on-call burden, and what breaks in "
            "production. Propose research that targets a real deployment "
            "constraint nobody in academia is taking seriously."
        ),
        "default_temp": 0.6,
    },
    "philosopher": {
        "label": "🧠 Philosopher",
        "tagline": "Asks about meaning, definitions, and category errors",
        "instruction": (
            "You are a philosopher of science. You notice when researchers "
            "are confused about what they're actually measuring, when a "
            "definition has quietly shifted, when a category error is "
            "lurking in the framing. Propose research that clarifies a "
            "definitional or conceptual confusion in the field."
        ),
        "default_temp": 0.75,
    },
    "historian": {
        "label": "📜 Historian",
        "tagline": "Sees today's hype through the lens of past hype cycles",
        "instruction": (
            "You are a historian of science. You've seen this kind of hype "
            "cycle before — pick a specific past example that today's "
            "researchers are unconsciously repeating, and propose research "
            "that pre-empts the next phase of the cycle by learning from "
            "the previous one."
        ),
        "default_temp": 0.70,
    },
    "naive_outsider": {
        "label": "🌱 Naive Outsider",
        "tagline": "Hasn't read the canon; asks 'why isn't anyone trying X?'",
        "instruction": (
            "You are a smart non-specialist who hasn't read the standard "
            "literature in this field. Propose the most obvious thing the "
            "field would do if it weren't constrained by what's currently "
            "considered respectable. Ask the question the experts are too "
            "embarrassed to ask out loud."
        ),
        "default_temp": 0.85,
    },
    "methodologist": {
        "label": "📐 Methodologist",
        "tagline": "Cares about experiment design, reproducibility, statistical power",
        "instruction": (
            "You are a methodologist obsessed with experiment design. You "
            "notice when studies are underpowered, when confounds aren't "
            "controlled, when 'state-of-the-art' rests on a single seed. "
            "Propose research that tackles a methodological weakness most "
            "of the field has normalized."
        ),
        "default_temp": 0.55,
    },
    "futurist": {
        "label": "🚀 Futurist",
        "tagline": "Frames every problem from 10 years out",
        "instruction": (
            "You are a futurist who instinctively reasons backward from "
            "what's possible in 10 years. Propose research that's only "
            "interesting if you assume capabilities will scale dramatically "
            "— and would lay the foundation that the field will need before "
            "the scale-up arrives."
        ),
        "default_temp": 0.85,
    },
    "contrarian": {
        "label": "⚡ Contrarian",
        "tagline": "Goes against whatever this year's trend is",
        "instruction": (
            "You are a contrarian. Identify the trendy direction this year "
            "in this topic, then propose research that goes the *opposite* "
            "direction — but make the case rigorously, not just for the "
            "sake of being different."
        ),
        "default_temp": 0.80,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Prompt + parser
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_TEMPLATE = (
    "{persona_instruction}\n\n"
    "You will be given a research topic. Propose ONE research idea that "
    "reflects your persona's unique vantage point — do not produce a "
    "generic idea and tag it with the persona name. Output ONLY valid "
    "JSON. methodology_type must be one of: "
    f"{', '.join(METHODOLOGY_TYPES)}. novelty_level must be one of: "
    f"{', '.join(NOVELTY_LEVELS)}."
)


def _user_prompt(topic: str, hint: str = "") -> str:
    hint_section = f"\n\nAdditional framing: {hint}\n" if hint else "\n"
    return (
        f"Topic: {topic}{hint_section}\n"
        "Return JSON:\n"
        "{\n"
        '  "title": "<concise>",\n'
        '  "motivation": "<from your persona\'s vantage point>",\n'
        '  "method": "<concrete technical approach>",\n'
        '  "hypothesis": "<testable prediction>",\n'
        '  "resources": "<resources needed>",\n'
        '  "expected_outcome": "<measurable results>",\n'
        '  "risk_assessment": "<main risks>",\n'
        '  "source_strategy": "P",\n'
        f'  "methodology_type": "<one of {", ".join(METHODOLOGY_TYPES)}>",\n'
        f'  "novelty_level": "<one of {", ".join(NOVELTY_LEVELS)}>",\n'
        '  "persona_signature": "<one sentence: what makes this idea '
        'recognizably from your persona, not a generic one>"\n'
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


def _build_idea(parsed: Dict[str, Any], topic: str,
                  persona_id: str) -> Optional[Idea]:
    if not all(str(parsed.get(k, "")).strip()
               for k in ("title", "method", "hypothesis")):
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
        source_strategy="P",
        methodology_type=method_type,
        novelty_level=novelty,
        generation=0, parent_title=None,
    )
    idea.execution_meta = {
        "persona_id": persona_id,
        "persona_label": PERSONAS.get(persona_id, {}).get("label", persona_id),
        "persona_signature": str(parsed.get("persona_signature", ""))[:400],
        "regen_mode": "persona_swap",
        "topic": topic,
    }
    return idea


# ─────────────────────────────────────────────────────────────────────────────
# Single-persona generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_under_persona(
    topic: str,
    persona_id: str,
    claude_client: Any = _AUTOLOAD,
    max_tokens: int = 900,
    temperature: Optional[float] = None,
    hint: str = "",
) -> Optional[Idea]:
    """Generate ONE idea under the chosen persona."""
    if not topic or not topic.strip():
        raise ValueError("topic must be non-empty")
    if persona_id not in PERSONAS:
        raise ValueError(
            f"persona_id must be one of {sorted(PERSONAS.keys())}, "
            f"got {persona_id!r}"
        )
    if claude_client is _AUTOLOAD:
        try:
            from claude_provider import get_claude_client
            claude_client = get_claude_client()
        except Exception:
            claude_client = None
    if claude_client is None:
        return None

    persona = PERSONAS[persona_id]
    sys = _SYSTEM_TEMPLATE.format(persona_instruction=persona["instruction"])
    temp = temperature if temperature is not None else persona.get(
        "default_temp", 0.7,
    )

    try:
        resp = claude_client.call(
            system=sys, user=_user_prompt(topic.strip(), hint=hint),
            max_tokens=max_tokens, temperature=temp, json_mode=True,
        )
    except Exception:
        return None
    if not getattr(resp, "success", False):
        return None
    parsed = _parse_json(getattr(resp, "text", ""))
    if not parsed:
        return None
    return _build_idea(parsed, topic.strip(), persona_id)


# ─────────────────────────────────────────────────────────────────────────────
# Batch with diversity filter
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PersonaResult:
    topic: str = ""
    personas_used: List[str] = field(default_factory=list)
    kept_ideas: List[Idea] = field(default_factory=list)
    all_ideas: List[Idea] = field(default_factory=list)
    rejected_pairs: List[Dict[str, Any]] = field(default_factory=list)
    persona_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)
    elapsed_s: float = 0.0

    def summary(self) -> str:
        return (
            f"Persona swap: {len(self.kept_ideas)} kept / "
            f"{len(self.all_ideas)} generated across "
            f"{len(self.personas_used)} personas · {self.elapsed_s:.1f}s"
        )


def persona_swap(
    topic: str,
    persona_ids: Optional[List[str]] = None,
    claude_client: Any = _AUTOLOAD,
    n_per_persona: int = 1,
    similarity_threshold: float = 0.55,
    hint: str = "",
    timeout_s: float = 60.0,
    max_workers: int = 6,
) -> PersonaResult:
    """Generate ideas under multiple personas in parallel, then keep only
    those that are pairwise diverse.

    Reuses the Jaccard-based diversity filter from multi_llm_ensemble so
    near-duplicates from different personas don't pollute the result.
    """
    if not topic or not topic.strip():
        raise ValueError("topic must be non-empty")
    if persona_ids is None:
        persona_ids = list(PERSONAS.keys())
    persona_ids = [p for p in persona_ids if p in PERSONAS]
    if not persona_ids:
        return PersonaResult(topic=topic.strip())
    if n_per_persona <= 0:
        return PersonaResult(topic=topic.strip(),
                               personas_used=list(persona_ids))

    if claude_client is _AUTOLOAD:
        try:
            from claude_provider import get_claude_client
            claude_client = get_claude_client()
        except Exception:
            claude_client = None

    started = time.time()
    persona_stats: Dict[str, Dict[str, int]] = {
        p: {"ok": 0, "fail": 0} for p in persona_ids
    }

    if claude_client is None:
        return PersonaResult(
            topic=topic.strip(),
            personas_used=list(persona_ids),
            persona_stats=persona_stats,
            elapsed_s=time.time() - started,
        )

    tasks: List[Tuple[str, int]] = []
    for p in persona_ids:
        for k in range(int(n_per_persona)):
            tasks.append((p, k))

    lock = threading.Lock()
    results: List[Tuple[Idea, str]] = []

    def _do(persona_id: str, k_idx: int):
        # Slight temperature jitter per attempt
        base_temp = PERSONAS[persona_id].get("default_temp", 0.7)
        temp = min(0.95, base_temp + 0.05 * k_idx)
        idea = generate_under_persona(
            topic.strip(), persona_id,
            claude_client=claude_client,
            temperature=temp, hint=hint,
        )
        with lock:
            if idea is not None:
                persona_stats[persona_id]["ok"] += 1
                results.append((idea, persona_id))
            else:
                persona_stats[persona_id]["fail"] += 1

    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as pool:
        futures = [pool.submit(_do, p, k) for p, k in tasks]
        try:
            for fut in as_completed(futures, timeout=timeout_s):
                try:
                    fut.result(timeout=2.0)
                except Exception:
                    pass
        except FuturesTimeout:
            pass

    elapsed = time.time() - started

    # Apply diversity filter — reuse multi_llm_ensemble's filter
    try:
        from multi_llm_ensemble import _diversity_filter
        kept, rejected = _diversity_filter(results, similarity_threshold)
    except ImportError:
        kept, rejected = results, []

    return PersonaResult(
        topic=topic.strip(),
        personas_used=list(persona_ids),
        kept_ideas=[i for i, _ in kept],
        all_ideas=[i for i, _ in results],
        rejected_pairs=rejected,
        persona_stats=persona_stats,
        elapsed_s=elapsed,
    )
