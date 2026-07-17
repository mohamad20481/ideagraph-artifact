"""
counterfactual_literature.py — generate ideas from research the canon
actively avoided rather than from top-cited papers.

Frontier extension uses popular papers as seeds (which biases toward
incremental extension). This module inverts that: it asks the LLM to
imagine plausible *counterfactual* literature — research lines that were
attempted but abandoned, findings that were retracted, niche papers
nobody followed up on, results that contradict the consensus and got
buried. Each such entry then becomes the seed for a new idea.

Output ideas have `source_strategy='L'` (Literature counterfactual) with
the imagined entry stored on `execution_meta.literature_entry`.

⚠️ Framing note: the counterfactual entries are LLM-generated, not real
papers. The point is to *force the LLM out of canon*, not to claim that
specific abandoned research actually exists. This is a creativity tool,
not a literature-review tool.

Public API:
    LITERATURE_KINDS                                   → List[str]
    CounterfactualEntry                                → dataclass
    imagine_counterfactual_literature(topic, kind, ...) → Optional[CounterfactualEntry]
    generate_from_counterfactual(topic, entry, ...)    → Optional[Idea]
    counterfactual_batch(topic, n, kinds, ...)         → List[Idea]
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from models.idea import Idea, METHODOLOGY_TYPES, NOVELTY_LEVELS


_AUTOLOAD = object()


LITERATURE_KINDS: List[str] = [
    "abandoned_direction",   # Researchers tried this, gave up, never published the failure
    "retracted_finding",     # Result was retracted but the question is still open
    "niche_unfollowed",      # Published, ignored, never cited — but the idea has legs
    "contrarian_buried",     # Contradicts consensus; got buried as a result
    "early_pioneer",         # 1970s/80s precursor whose ideas don't survive in modern form
    "cross_field_orphan",    # Solved in another field; never made the jump to ours
]


@dataclass
class CounterfactualEntry:
    """One imagined literature entry — the inspiration, not a real citation."""
    kind: str = ""
    title: str = ""              # plausible-sounding paper title
    authors: str = ""            # plausible author hint (e.g., "a 1998 Stanford group")
    year: int = 2010
    summary: str = ""            # what the entry would have claimed
    why_neglected: str = ""      # why the canon ignored / buried / retracted it
    what_to_revive: str = ""     # which thread modern research should pick back up

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind, "title": self.title, "authors": self.authors,
            "year": self.year, "summary": self.summary,
            "why_neglected": self.why_neglected,
            "what_to_revive": self.what_to_revive,
        }


_IMAGINE_SYSTEM = (
    "You are an intellectual archeologist of research. The user gives you "
    "a topic and a kind of *counterfactual* literature — research that "
    "was attempted but abandoned, retracted, ignored, or buried. Your "
    "job is to *imagine* one plausible such entry on the topic. The "
    "entry doesn't have to be a real paper — it should be a plausible "
    "story about what such a paper *would have* claimed and why the "
    "field would have neglected it. Be specific, technically grounded, "
    "and intellectually honest about why the canon walked away. Output "
    "ONLY valid JSON."
)


_KIND_DESCRIPTIONS = {
    "abandoned_direction": (
        "An entire research direction the field pursued briefly, then gave "
        "up on — usually because results didn't reproduce or the early "
        "implementations were too primitive. The question is still open."
    ),
    "retracted_finding": (
        "A specific finding that was retracted (for methodology errors, "
        "fraud, or replication failure) — but where the underlying "
        "scientific question never got revisited cleanly."
    ),
    "niche_unfollowed": (
        "A published paper that nobody cited, but whose central insight "
        "has aged surprisingly well. The technique was too odd or too "
        "early for its moment."
    ),
    "contrarian_buried": (
        "A paper that contradicted the prevailing consensus when it was "
        "published, got dismissed or ignored as a result, and never got a "
        "fair hearing."
    ),
    "early_pioneer": (
        "A 1970s–1990s precursor whose framing the field has lost. The "
        "ideas don't survive in modern form, but they pointed at something "
        "real."
    ),
    "cross_field_orphan": (
        "A method that was solved in an adjacent field (economics, "
        "ecology, control theory, …) but never made the conceptual jump "
        "into the user's topic."
    ),
}


def _imagine_user_prompt(topic: str, kind: str) -> str:
    desc = _KIND_DESCRIPTIONS.get(kind, "")
    return (
        f"Topic: {topic}\n\n"
        f"### Kind of counterfactual literature\n"
        f"  Type: {kind}\n"
        f"  Description: {desc}\n\n"
        f"### Instructions\n"
        f"Imagine ONE plausible entry of this kind on this topic. Give it "
        f"a plausible-sounding title, a hint about authors (group/decade "
        f"is enough), a year, what the entry would have claimed, why the "
        f"canon neglected it, and what's worth reviving from it today.\n\n"
        f"Return JSON:\n"
        "{\n"
        '  "title":          "<plausible paper title>",\n'
        '  "authors":        "<author hint, e.g. \'1998 Stanford robotics group\'>",\n'
        '  "year":           <plausible year>,\n'
        '  "summary":        "<2-3 sentences on what this entry would have claimed>",\n'
        '  "why_neglected":  "<honest reason the field walked away>",\n'
        '  "what_to_revive": "<one sentence: which thread modern research '
        'should pick back up>"\n'
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


def imagine_counterfactual_literature(
    topic: str,
    kind: str = "abandoned_direction",
    claude_client: Any = _AUTOLOAD,
    max_tokens: int = 500,
    temperature: float = 0.85,
) -> Optional[CounterfactualEntry]:
    if not topic or not topic.strip():
        raise ValueError("topic must be non-empty")
    if kind not in LITERATURE_KINDS:
        raise ValueError(
            f"kind must be one of {LITERATURE_KINDS}, got {kind!r}"
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
            system=_IMAGINE_SYSTEM,
            user=_imagine_user_prompt(topic.strip(), kind),
            max_tokens=max_tokens, temperature=temperature, json_mode=True,
        )
    except Exception:
        return None
    if not getattr(resp, "success", False):
        return None
    parsed = _parse_json(getattr(resp, "text", ""))
    if not parsed:
        return None
    if not str(parsed.get("title", "")).strip():
        return None
    try:
        year = int(parsed.get("year", 2010))
    except (TypeError, ValueError):
        year = 2010
    return CounterfactualEntry(
        kind=kind,
        title=str(parsed.get("title", ""))[:200],
        authors=str(parsed.get("authors", ""))[:200],
        year=max(1970, min(2025, year)),
        summary=str(parsed.get("summary", ""))[:800],
        why_neglected=str(parsed.get("why_neglected", ""))[:400],
        what_to_revive=str(parsed.get("what_to_revive", ""))[:400],
    )


_GENERATE_SYSTEM = (
    "You are a research scientist drawing inspiration from a "
    "counterfactual literature entry — research that was abandoned, "
    "retracted, ignored, or solved in an adjacent field. Your job is to "
    "propose ONE modern research idea that picks up the abandoned thread "
    "and addresses why the canon walked away from it. Be honest about "
    "why this time would be different (better tools, better data, better "
    "understanding). Output ONLY valid JSON. methodology_type must be "
    f"one of: {', '.join(METHODOLOGY_TYPES)}. novelty_level must be one "
    f"of: {', '.join(NOVELTY_LEVELS)}."
)


def _generate_user_prompt(topic: str, e: CounterfactualEntry) -> str:
    return (
        f"Topic: {topic}\n\n"
        f"### Counterfactual literature inspiration\n"
        f"  Kind: {e.kind}\n"
        f"  Imagined title: {e.title}\n"
        f"  Imagined authors: {e.authors} ({e.year})\n"
        f"  Summary: {e.summary}\n"
        f"  Why neglected: {e.why_neglected}\n"
        f"  Thread to revive: {e.what_to_revive}\n\n"
        f"### Instructions\n"
        f"Propose ONE modern research idea that picks up the abandoned "
        f"thread. State explicitly in motivation what's different now "
        f"(tools / data / understanding) that gives this another shot. "
        f"The idea should be defensibly novel today — not just a rerun "
        f"of the original.\n\n"
        f"### Output\n"
        "Return JSON:\n"
        "{\n"
        '  "title":             "<concise modern title>",\n'
        '  "motivation":        "<why this matters NOW; cite the counterfactual entry>",\n'
        '  "method":            "<concrete technical approach>",\n'
        '  "hypothesis":        "<testable prediction>",\n'
        '  "resources":         "<resources needed>",\n'
        '  "expected_outcome":  "<measurable result>",\n'
        '  "risk_assessment":   "<main risks, incl. why this could fail like the original>",\n'
        '  "source_strategy":   "L",\n'
        f'  "methodology_type":  "<one of: {", ".join(METHODOLOGY_TYPES)}>",\n'
        f'  "novelty_level":     "<one of: {", ".join(NOVELTY_LEVELS)}>",\n'
        '  "what_changed":      "<one sentence: what\'s different now '
        'that makes this worth trying again>"\n'
        "}"
    )


def _build_idea(parsed: Dict[str, Any], topic: str,
                  entry: CounterfactualEntry) -> Optional[Idea]:
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
        source_strategy="L",
        methodology_type=method_type,
        novelty_level=novelty,
        generation=0, parent_title=None,
    )
    idea.execution_meta = {
        "literature_entry": entry.to_dict(),
        "what_changed": str(parsed.get("what_changed", ""))[:400],
        "regen_mode": "counterfactual_literature",
        "topic": topic,
    }
    return idea


def generate_from_counterfactual(
    topic: str,
    entry: CounterfactualEntry,
    claude_client: Any = _AUTOLOAD,
    max_tokens: int = 900,
    temperature: float = 0.75,
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
            system=_GENERATE_SYSTEM,
            user=_generate_user_prompt(topic.strip(), entry),
            max_tokens=max_tokens, temperature=temperature, json_mode=True,
        )
    except Exception:
        return None
    if not getattr(resp, "success", False):
        return None
    parsed = _parse_json(getattr(resp, "text", ""))
    if not parsed:
        return None
    return _build_idea(parsed, topic.strip(), entry)


def counterfactual_batch(
    topic: str,
    claude_client: Any = _AUTOLOAD,
    n: int = 3,
    kinds: Optional[List[str]] = None,
) -> List[Idea]:
    """Imagine N counterfactual entries (one per kind), generate an idea
    from each. Default rotates through the kind catalog."""
    if not topic or not topic.strip():
        raise ValueError("topic must be non-empty")
    if kinds is None:
        kinds = LITERATURE_KINDS[:max(1, n)]
    kinds = [k for k in kinds if k in LITERATURE_KINDS]
    out: List[Idea] = []
    for k in kinds[:n]:
        entry = imagine_counterfactual_literature(
            topic, k, claude_client=claude_client,
        )
        if entry is None:
            continue
        idea = generate_from_counterfactual(
            topic, entry, claude_client=claude_client,
        )
        if idea is not None:
            out.append(idea)
    return out
