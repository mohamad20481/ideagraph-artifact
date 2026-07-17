"""
embedding_exploration.py — frontier-concept generation.

Where embedding-based novelty checkers REJECT ideas that are too similar
to existing ones, this module flips the lens: it finds the under-explored
*concept directions* in the existing archive and explicitly generates new
ideas TOWARD those directions.

Two-step pipeline:
  1. compute_frontier_concepts(archive, topic) — extracts a per-token
     "coverage" vector across the archive's titles+methods, identifies
     concepts that are highly relevant to `topic` but rarely covered,
     and asks the LLM to expand them into a frontier description.
  2. generate_at_frontier(topic, concepts) — generates an idea that
     explicitly targets the frontier.

No heavy ML dependencies — uses token-frequency analysis on the existing
ideas plus the LLM as the concept-expansion oracle.

Public API:
    FrontierAnalysis                                         → dataclass
    compute_frontier_concepts(archived_ideas, topic, ...)    → FrontierAnalysis
    generate_at_frontier(topic, frontier, ...)                → Optional[Idea]
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from models.idea import Idea, METHODOLOGY_TYPES, NOVELTY_LEVELS


_AUTOLOAD = object()


# ─────────────────────────────────────────────────────────────────────────────
# Frontier analysis
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FrontierAnalysis:
    topic: str = ""
    covered_concepts: List[Tuple[str, int]] = field(default_factory=list)
    underexplored_concepts: List[str] = field(default_factory=list)
    frontier_description: str = ""
    frontier_seeds: List[str] = field(default_factory=list)
    n_archive_ideas: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "covered_concepts": [list(t) for t in self.covered_concepts],
            "underexplored_concepts": list(self.underexplored_concepts),
            "frontier_description": self.frontier_description,
            "frontier_seeds": list(self.frontier_seeds),
            "n_archive_ideas": self.n_archive_ideas,
        }


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-]{3,}")
_STOPWORDS = frozenset([
    "the", "and", "for", "with", "this", "that", "from", "are", "was",
    "have", "has", "will", "use", "uses", "using", "their", "they", "our",
    "ours", "but", "not", "all", "can", "could", "may", "into", "via",
    "more", "less", "than", "such", "also", "however", "thus", "while",
    "which", "where", "when", "what", "how", "why", "who", "whom",
    "would", "should", "shall", "must", "might", "been", "being",
    "very", "really", "much", "many", "some", "most", "least", "few",
    "research", "study", "approach", "method", "methods", "paper",
    "model", "models", "task", "tasks", "data", "datasets", "results",
    "based", "using", "show", "shows", "shown", "propose", "proposed",
    "new", "novel", "existing", "current", "prior", "previous", "future",
    "first", "second", "third", "well", "good", "high", "low",
    "make", "makes", "made", "give", "gives", "given", "take", "takes",
])


def _tokens_from(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "")
            if t.lower() not in _STOPWORDS]


def _idea_text(idea: Any) -> str:
    d = idea.to_dict() if hasattr(idea, "to_dict") else (
        idea if isinstance(idea, dict) else {}
    )
    return " ".join([
        str(d.get("title", "")),
        str(d.get("motivation", "")),
        str(d.get("method", "")),
        str(d.get("hypothesis", "")),
    ])


def _coverage_distribution(ideas: List[Any]) -> Counter:
    """Token frequency across the archive."""
    counts: Counter = Counter()
    for idea in ideas:
        toks = set(_tokens_from(_idea_text(idea)))
        for t in toks:
            counts[t] += 1
    return counts


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


_FRONTIER_SYSTEM = (
    "You are a research-mapping specialist. The user gives you a topic and "
    "a list of concepts that are *over-represented* in their current "
    "research archive. Your job is to name 5-8 concepts that are HIGHLY "
    "relevant to the topic but UNDER-EXPLORED in the archive — and then "
    "describe the resulting research frontier. Be specific and avoid "
    "generic buzzwords. Output ONLY valid JSON."
)


def _frontier_user_prompt(topic: str, top_covered: List[str]) -> str:
    covered_str = ", ".join(top_covered[:25]) or "(none)"
    return (
        f"Topic: {topic}\n\n"
        f"### Over-represented concepts in the existing archive\n"
        f"{covered_str}\n\n"
        f"### Instructions\n"
        f"Name 5-8 concepts that are highly relevant to the topic but "
        f"missing or under-explored in the archive above. Then describe "
        f"the research frontier these concepts open up — what kinds of "
        f"ideas would live in that region, and why nobody's been there.\n\n"
        f"Return JSON:\n"
        "{\n"
        '  "underexplored_concepts": <list of 5-8 specific concepts>,\n'
        '  "frontier_description":   "<2-3 sentence picture of the '
        'frontier these concepts open up>",\n'
        '  "frontier_seeds":         <list of 3-5 concrete "what-if" '
        'questions that target the frontier>\n'
        "}"
    )


def compute_frontier_concepts(
    archived_ideas: List[Any],
    topic: str,
    claude_client: Any = _AUTOLOAD,
    top_n_covered: int = 25,
) -> FrontierAnalysis:
    """Identify the under-explored concept frontier in the archive."""
    if not topic or not topic.strip():
        raise ValueError("topic must be non-empty")

    counts = _coverage_distribution(archived_ideas)
    n = len(archived_ideas)
    # Most-covered concepts
    top_covered = counts.most_common(top_n_covered)
    analysis = FrontierAnalysis(
        topic=topic.strip(),
        covered_concepts=[(t, c) for t, c in top_covered],
        n_archive_ideas=n,
    )

    if claude_client is _AUTOLOAD:
        try:
            from claude_provider import get_claude_client
            claude_client = get_claude_client()
        except Exception:
            claude_client = None
    if claude_client is None:
        return analysis

    try:
        resp = claude_client.call(
            system=_FRONTIER_SYSTEM,
            user=_frontier_user_prompt(
                topic.strip(), [t for t, _ in top_covered],
            ),
            max_tokens=600,
            temperature=0.75,
            json_mode=True,
        )
    except Exception:
        return analysis
    if not getattr(resp, "success", False):
        return analysis
    parsed = _parse_json(getattr(resp, "text", ""))
    if not parsed:
        return analysis

    underex = parsed.get("underexplored_concepts") or []
    if isinstance(underex, list):
        analysis.underexplored_concepts = [str(x)[:200] for x in underex[:10]]
    analysis.frontier_description = str(parsed.get("frontier_description", ""))[:1000]
    seeds = parsed.get("frontier_seeds") or []
    if isinstance(seeds, list):
        analysis.frontier_seeds = [str(x)[:300] for x in seeds[:8]]
    return analysis


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: generate an idea targeting the frontier
# ─────────────────────────────────────────────────────────────────────────────

_GEN_SYSTEM = (
    "You are a research scientist generating an idea that targets a "
    "specific under-explored frontier in your field. The user gives you "
    "the topic, a list of under-explored concepts, and 'what-if' frontier "
    "seeds. Your idea MUST be anchored in the under-explored concepts — "
    "explicitly name which ones in the motivation. Output ONLY valid JSON "
    "matching the schema. methodology_type must be one of: "
    f"{', '.join(METHODOLOGY_TYPES)}. novelty_level must be one of: "
    f"{', '.join(NOVELTY_LEVELS)}."
)


def _gen_user_prompt(topic: str, frontier: FrontierAnalysis) -> str:
    underex_str = "\n".join(f"  • {c}" for c in frontier.underexplored_concepts)
    seeds_str = "\n".join(f"  • {s}" for s in frontier.frontier_seeds)
    return (
        f"Topic: {topic}\n\n"
        f"### Under-explored frontier concepts\n{underex_str}\n\n"
        f"### Frontier picture\n{frontier.frontier_description}\n\n"
        f"### Frontier seeds (what-if questions)\n{seeds_str}\n\n"
        f"### Instructions\n"
        f"Generate ONE research idea that explicitly targets the frontier "
        f"above. Anchor the idea in at least two of the under-explored "
        f"concepts — name them in the motivation. The novelty must come "
        f"from being in the under-explored region, not from generic "
        f"cleverness.\n\n"
        f"Return JSON:\n"
        "{\n"
        '  "title": "<concise>",\n'
        '  "motivation": "<why this matters; names the under-explored '
        'concepts being targeted>",\n'
        '  "method": "<concrete technical approach>",\n'
        '  "hypothesis": "<testable prediction>",\n'
        '  "resources": "<resources needed>",\n'
        '  "expected_outcome": "<measurable result>",\n'
        '  "risk_assessment": "<main risks>",\n'
        '  "source_strategy": "X",\n'
        f'  "methodology_type": "<one of: {", ".join(METHODOLOGY_TYPES)}>",\n'
        f'  "novelty_level": "<one of: {", ".join(NOVELTY_LEVELS)}>",\n'
        '  "frontier_concepts_used": <list of 2-5 strings: which '
        'under-explored concepts this idea targets>\n'
        "}"
    )


def generate_at_frontier(
    topic: str,
    frontier: FrontierAnalysis,
    claude_client: Any = _AUTOLOAD,
    max_tokens: int = 900,
    temperature: float = 0.8,
) -> Optional[Idea]:
    """Generate one idea targeting the under-explored frontier."""
    if not topic or not topic.strip():
        raise ValueError("topic must be non-empty")
    if not frontier.underexplored_concepts and not frontier.frontier_seeds:
        # Nothing to target — fail explicitly rather than generate generic
        return None
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
            system=_GEN_SYSTEM,
            user=_gen_user_prompt(topic.strip(), frontier),
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
        source_strategy="X",  # X = eXploration (embedding gradient)
        methodology_type=method_type,
        novelty_level=novelty,
        generation=0,
        parent_title=None,
    )
    used = parsed.get("frontier_concepts_used") or []
    if not isinstance(used, list):
        used = [str(used)]
    idea.execution_meta = {
        "frontier_concepts_used": [str(c)[:200] for c in used[:6]],
        "frontier_description": frontier.frontier_description,
        "regen_mode": "embedding_exploration",
        "topic": topic.strip(),
    }
    return idea
