"""
inversion_ideation.py — Jeopardy-style backward derivation.

Different from every other mode in the Novelty Lab. The others start from a
*question*, a *constraint*, a *belief*, or a *regime* and search forward
for a method. Inversion does the opposite: start from an unusual *answer*
(e.g., "a 1-page method beats GPT-4 on task X", "a frozen 50M-parameter
model outperforms fine-tuning on benchmark Y"), and work *backward* to
derive the research question whose answer that would be.

Two phases:
  1. propose_answer(topic, tone) — LLM names a specific, surprising candidate
     answer/result that, if true, would matter — concrete enough to be
     testable, surprising enough to be worth publishing.
  2. derive_question(topic, answer) — LLM derives the research question
     and concrete method that would make the answer true.

Ideas have `source_strategy='I'` (Inversion / Jeopardy). The candidate
answer lives on `execution_meta.candidate_answer`. Best paired with the
Exec Loop — many candidate answers turn out infeasible, which the Loop
will catch.

Public API:
    CandidateAnswer                                    → dataclass
    ANSWER_TONES                                       → catalog
    propose_answer(topic, tone, ...)                   → Optional[CandidateAnswer]
    derive_question(topic, answer, ...)                → Optional[Idea]
    inversion_batch(topic, ..., n)                     → List[Idea]
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from models.idea import Idea, METHODOLOGY_TYPES, NOVELTY_LEVELS


_AUTOLOAD = object()


# Different framings for the "surprising answer". Mix of upward (something
# small beats something big), downward (something big fails at something
# easy), null-effect, and regime-shift answers. The LLM picks ONE and
# anchors a research question to it.
ANSWER_TONES: List[str] = [
    "underdog_wins",        # a small/cheap method beats a big/expensive one
    "favourite_fails",      # a celebrated method fails on a humble target
    "tradeoff_inversion",   # the usual cost/benefit ranking flips
    "null_effect",          # something widely assumed to matter, doesn't
    "phase_transition",     # behavior changes qualitatively at a threshold
    "transfer_surprise",    # ability transfers between unrelated tasks
]


@dataclass
class CandidateAnswer:
    """A surprising-but-specific candidate result the research must derive."""
    tone: str = ""              # one of ANSWER_TONES
    headline: str = ""          # the result, in one falsifiable sentence
    why_surprising: str = ""    # what canonical assumption it would violate
    why_plausible: str = ""     # honest reason to believe it could be true
    measurable_claim: str = ""  # concrete quantitative form (e.g. "F1 ≥ 0.8")
    plausibility: float = 0.5   # 0..1 LLM's honest estimate it could be true

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tone": self.tone,
            "headline": self.headline,
            "why_surprising": self.why_surprising,
            "why_plausible": self.why_plausible,
            "measurable_claim": self.measurable_claim,
            "plausibility": self.plausibility,
        }


_ANSWER_SYSTEM = (
    "You are a research strategist who plays Jeopardy with science. The "
    "user gives you a research topic and a tone (e.g., 'an underdog wins' "
    "or 'the favourite fails'). Your job is NOT to propose research — it "
    "is to name a SPECIFIC, SURPRISING candidate result/answer that, if "
    "true, would be worth publishing. The candidate must be concrete "
    "enough to be testable and quantitative. Be honest about whether you "
    "actually believe it. Return ONLY valid JSON."
)


def _answer_user_prompt(topic: str, tone: str) -> str:
    return (
        f"Topic: {topic}\n"
        f"Tone: {tone}\n\n"
        f"Name ONE specific, surprising candidate answer/result that, if "
        f"true, would matter in this topic. State it as a measurable claim. "
        f"Be honest about plausibility — we'd rather a 30% plausibility "
        f"answer that's specific than a 90% plausibility platitude.\n\n"
        f"Return JSON:\n"
        "{\n"
        '  "headline": "<the result in one falsifiable sentence>",\n'
        '  "why_surprising": "<what canonical assumption it would violate>",\n'
        '  "why_plausible": "<honest reason to think it could actually be true>",\n'
        '  "measurable_claim": "<concrete quantitative form, e.g. \\"F1 >= 0.8 with <= 100M params\\">",\n'
        '  "plausibility": <0..1 honest estimate it could be true>\n'
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


def propose_answer(
    topic: str,
    tone: str,
    claude_client: Any = _AUTOLOAD,
    max_tokens: int = 600,
    temperature: float = 0.9,
) -> Optional[CandidateAnswer]:
    if not topic or not topic.strip():
        raise ValueError("topic must be non-empty")
    if tone not in ANSWER_TONES:
        raise ValueError(f"tone must be one of {ANSWER_TONES}, got {tone!r}")
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
            system=_ANSWER_SYSTEM,
            user=_answer_user_prompt(topic.strip(), tone),
            max_tokens=max_tokens, temperature=temperature, json_mode=True,
        )
    except Exception:
        return None
    if not getattr(resp, "success", False):
        return None
    parsed = _parse_json(getattr(resp, "text", ""))
    if not parsed:
        return None
    headline = str(parsed.get("headline") or "").strip()
    claim = str(parsed.get("measurable_claim") or "").strip()
    if not (headline and claim):
        return None
    return CandidateAnswer(
        tone=tone,
        headline=headline[:400],
        why_surprising=str(parsed.get("why_surprising") or "")[:400],
        why_plausible=str(parsed.get("why_plausible") or "")[:400],
        measurable_claim=claim[:400],
        plausibility=_coerce_unit(parsed.get("plausibility"), 0.5),
    )


_QUESTION_SYSTEM = (
    "You are a research scientist doing reverse engineering. The user "
    "gives you a topic and a SPECIFIC candidate result that someone "
    "claims (the 'answer'). Your job is to derive the research question "
    "and concrete method whose execution would make this answer true. "
    "Treat the answer as a target — design backwards from it. Be honest "
    "about whether the answer is achievable. Output ONLY valid JSON. "
    f"methodology_type must be one of: {', '.join(METHODOLOGY_TYPES)}. "
    f"novelty_level must be one of: {', '.join(NOVELTY_LEVELS)}."
)


def _question_user_prompt(topic: str, answer: CandidateAnswer) -> str:
    return (
        f"Topic: {topic}\n\n"
        f"### Target answer (work backward from this)\n"
        f"  Tone:             {answer.tone}\n"
        f"  Headline:         {answer.headline}\n"
        f"  Why surprising:   {answer.why_surprising}\n"
        f"  Why plausible:    {answer.why_plausible}\n"
        f"  Measurable claim: {answer.measurable_claim}\n"
        f"  Plausibility:     {answer.plausibility:.2f}\n\n"
        f"### Instructions\n"
        f"Derive the research question + concrete method that would make "
        f"this answer true. Work backward: what setup, comparisons, "
        f"datasets, ablations would produce this result? State the "
        f"derived question explicitly in the motivation.\n\n"
        f"### Output\n"
        "Return JSON:\n"
        "{\n"
        '  "title": "<concise, signals the inversion>",\n'
        '  "motivation": "<derived research question + why the answer matters>",\n'
        '  "method": "<concrete approach reverse-engineered from the target>",\n'
        '  "hypothesis": "<falsifiable form of the measurable claim>",\n'
        '  "resources": "<resources needed>",\n'
        '  "expected_outcome": "<measurable outcome — the target answer>",\n'
        '  "risk_assessment": "<honest risks if the answer turns out false>",\n'
        '  "source_strategy": "I",\n'
        f'  "methodology_type": "<one of: {", ".join(METHODOLOGY_TYPES)}>",\n'
        f'  "novelty_level": "<one of: {", ".join(NOVELTY_LEVELS)}>",\n'
        '  "derived_question": "<one sentence: the research question this answer demands>"\n'
        "}"
    )


def _build_idea(parsed: Dict[str, Any], topic: str,
                  answer: CandidateAnswer) -> Optional[Idea]:
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
        source_strategy="I",
        methodology_type=method_type,
        novelty_level=novelty,
        generation=0,
        parent_title=None,
    )
    idea.execution_meta = {
        "candidate_answer": answer.to_dict(),
        "derived_question": str(parsed.get("derived_question", ""))[:400],
        "regen_mode": "inversion",
        "topic": topic,
    }
    return idea


def derive_question(
    topic: str,
    answer: CandidateAnswer,
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
            system=_QUESTION_SYSTEM,
            user=_question_user_prompt(topic.strip(), answer),
            max_tokens=max_tokens, temperature=temperature, json_mode=True,
        )
    except Exception:
        return None
    if not getattr(resp, "success", False):
        return None
    parsed = _parse_json(getattr(resp, "text", ""))
    if not parsed:
        return None
    return _build_idea(parsed, topic.strip(), answer)


def inversion_batch(
    topic: str,
    claude_client: Any = _AUTOLOAD,
    n: int = 3,
    tones: Optional[List[str]] = None,
    seed: Optional[int] = None,
) -> List[Idea]:
    """End-to-end: pick `n` distinct tones, propose a surprising answer
    for each, derive the question. `tones` overrides the random pick."""
    if not topic or not topic.strip():
        raise ValueError("topic must be non-empty")
    if n <= 0:
        return []
    if tones is None:
        pool = list(ANSWER_TONES)
        rng = random.Random(seed) if seed is not None else random
        rng.shuffle(pool)
        tones = pool[:max(1, n)]
    for t in tones:
        if t not in ANSWER_TONES:
            raise ValueError(f"invalid tone {t!r}")
    out: List[Idea] = []
    for t in tones[:n]:
        ans = propose_answer(topic, t, claude_client=claude_client)
        if ans is None:
            continue
        idea = derive_question(topic, ans, claude_client=claude_client)
        if idea is not None:
            out.append(idea)
    return out
