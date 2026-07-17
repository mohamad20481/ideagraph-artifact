"""
corpus_novelty.py — operationalized novelty against an external reference corpus.

The other 18 Novelty Lab modes generate ideas with various heuristics, but
none of them measures novelty against a *reference corpus of work that
already exists*. This module fills that gap.

The operational claim is honest: "novel" means "far from anything in our
corpus", NOT "no human ever suggested this". Pick the corpus carefully —
the strongest claim comes from a large external corpus (Semantic Scholar,
arXiv, USPTO). This module ships with two offline-safe corpus loaders
(seed-text list, internal-archive ideas) and a stub for adding network
loaders later.

The 7-stage pipeline:
  1. Build/refresh a ReferenceCorpus
  2. Run multiple generators (persona, analogy, heretic by default —
     'varied by persona, temperature, forced cross-domain analogies' from
     the design manifesto)
  3. Embed each candidate (TF over filtered tokens, sparse dict vectors;
     pluggable embedder for future neural backends)
  4. Compute novelty = 1 - max(cosine similarity to corpus)
  5. Optional MAP-Elites deduplication on (methodology, novelty_level)
  6. Sort by novelty descending, attach NoveltyAssessment to each
  7. Return top K with corpus_limits disclaimer in each idea's execution_meta

Public API:
    CORPUS_LIMITS_DISCLAIMER                              → str
    CorpusEntry                                           → dataclass
    NoveltyAssessment                                     → dataclass
    ReferenceCorpus                                       → class
    DEFAULT_GENERATORS                                    → catalog
    tokenize(text)                                        → List[str]
    text_vector(text)                                     → Dict[str, float]
    cosine_similarity(v1, v2)                             → float
    assess_idea_novelty(idea, corpus)                     → NoveltyAssessment
    corpus_anchored_batch(topic, corpus, ...)             → List[Idea]
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from models.idea import Idea, METHODOLOGY_TYPES, NOVELTY_LEVELS


_AUTOLOAD = object()


# ── Honest-limits disclaimer (shown in UI + saved to every idea) ────────────

CORPUS_LIMITS_DISCLAIMER: str = (
    "Novelty here means *far from anything in the chosen reference "
    "corpus* — it does NOT prove no human ever suggested the idea. "
    "Strength of the claim is bounded by the corpus: a small or "
    "narrow corpus yields a weak claim. For the strongest claim, "
    "use a broad external corpus (Semantic Scholar ~200M papers, "
    "arXiv, USPTO/EPO patents, recent web crawl)."
)


# ── Tokenization / vectorization (offline-safe default) ─────────────────────
# A small English stop-word set keeps token vectors focused on content words
# rather than glue. Kept inline (no nltk dependency) so the module imports
# cleanly in any environment.
_STOPWORDS: set = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
    "while", "of", "to", "in", "on", "at", "by", "for", "with", "about",
    "from", "as", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "this", "that",
    "these", "those", "i", "you", "he", "she", "it", "we", "they",
    "what", "which", "who", "whom", "how", "where", "why", "any", "all",
    "no", "not", "so", "than", "very", "more", "most", "such", "some",
    "into", "out", "up", "down", "over", "under", "also", "however",
    "thus", "hence", "therefore", "use", "using", "used", "based",
    "new", "novel", "study", "studies", "research", "researchers",
    "paper", "papers", "method", "methods", "approach", "approaches",
    "show", "shows", "shown", "demonstrate", "demonstrates", "propose",
    "proposed", "proposes", "result", "results", "find", "finds",
    "found", "via", "between", "among",
}

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-]{2,}")


def tokenize(text: str) -> List[str]:
    """Lowercase alphabetic tokens, ≥3 chars, stopwords filtered.

    Deliberately simple — good enough for short titles/abstracts where
    word-overlap is a meaningful signal. Swap for a neural embedder via
    `corpus.embedder = ...` for stronger semantic matching.
    """
    if not text:
        return []
    out: List[str] = []
    for tok in _TOKEN_RE.findall(text):
        t = tok.lower()
        if t in _STOPWORDS:
            continue
        out.append(t)
    return out


def text_vector(text: str) -> Dict[str, float]:
    """Sparse unit-length TF vector over filtered tokens."""
    toks = tokenize(text)
    if not toks:
        return {}
    counts: Dict[str, float] = {}
    for t in toks:
        counts[t] = counts.get(t, 0.0) + 1.0
    # L2 normalize so cosine == dot product.
    norm = math.sqrt(sum(v * v for v in counts.values()))
    if norm <= 0.0:
        return {}
    return {k: v / norm for k, v in counts.items()}


def cosine_similarity(v1: Dict[str, float], v2: Dict[str, float]) -> float:
    """Dot product of two unit vectors. Returns 0.0 if either is empty."""
    if not v1 or not v2:
        return 0.0
    # Iterate over the smaller side for speed.
    small, large = (v1, v2) if len(v1) <= len(v2) else (v2, v1)
    return sum(val * large.get(key, 0.0) for key, val in small.items())


# ── Dataclasses ─────────────────────────────────────────────────────────────

@dataclass
class CorpusEntry:
    """One reference item in the corpus."""
    title: str
    abstract: str = ""
    source: str = "seed"          # "seed", "archive", "semantic_scholar", "arxiv", "patent"
    year: Optional[int] = None
    excerpt_max: int = 240
    _vec: Optional[Dict[str, float]] = field(default=None, repr=False)

    def text(self) -> str:
        if self.abstract:
            return f"{self.title}. {self.abstract}"
        return self.title

    def excerpt(self) -> str:
        body = self.abstract or self.title
        return (body[: self.excerpt_max].rstrip()
                  + ("…" if len(body) > self.excerpt_max else ""))


@dataclass
class NoveltyAssessment:
    """Result of scoring one candidate against a corpus."""
    score: float = 0.0              # 0..1, higher = more novel
    nearest_similarity: float = 0.0  # raw cosine to closest entry
    nearest_title: str = ""
    nearest_source: str = ""
    nearest_year: Optional[int] = None
    nearest_excerpt: str = ""
    corpus_size: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "nearest_similarity": round(self.nearest_similarity, 4),
            "nearest_title": self.nearest_title,
            "nearest_source": self.nearest_source,
            "nearest_year": self.nearest_year,
            "nearest_excerpt": self.nearest_excerpt,
            "corpus_size": self.corpus_size,
        }


# ── ReferenceCorpus ─────────────────────────────────────────────────────────

class ReferenceCorpus:
    """A bag of CorpusEntries with cached text vectors.

    Pluggable embedder: set `corpus.embedder = my_fn` where `my_fn(str) ->
    Dict[str, float]` to swap in a neural backend later. Default is the
    offline TF tokenizer above.
    """

    def __init__(
        self,
        entries: Optional[List[CorpusEntry]] = None,
        embedder: Optional[Callable[[str], Dict[str, float]]] = None,
    ) -> None:
        self._entries: List[CorpusEntry] = list(entries or [])
        self.embedder: Callable[[str], Dict[str, float]] = (
            embedder or text_vector
        )
        # Pre-cache vectors for any entries passed in at construction.
        for e in self._entries:
            if e._vec is None:
                e._vec = self.embedder(e.text())

    @property
    def entries(self) -> List[CorpusEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def add(
        self,
        title: str,
        abstract: str = "",
        source: str = "seed",
        year: Optional[int] = None,
    ) -> None:
        if not title or not title.strip():
            raise ValueError("title must be non-empty")
        entry = CorpusEntry(
            title=title.strip(),
            abstract=(abstract or "").strip(),
            source=(source or "seed").strip() or "seed",
            year=year,
        )
        entry._vec = self.embedder(entry.text())
        self._entries.append(entry)

    def nearest_neighbor(
        self, text: str,
    ) -> Tuple[Optional[CorpusEntry], float]:
        """Return (entry, similarity) for the corpus entry most similar
        to `text`. Returns `(None, 0.0)` if the corpus is empty or the
        text yields an empty vector."""
        if not self._entries:
            return None, 0.0
        q = self.embedder(text)
        if not q:
            return None, 0.0
        best_e: Optional[CorpusEntry] = None
        best_sim = -1.0
        for e in self._entries:
            if e._vec is None:
                e._vec = self.embedder(e.text())
            sim = cosine_similarity(q, e._vec)
            if sim > best_sim:
                best_sim = sim
                best_e = e
        return best_e, max(0.0, best_sim)

    def novelty_score(self, text: str) -> float:
        """Operational novelty = 1 - max cosine similarity to corpus."""
        _, sim = self.nearest_neighbor(text)
        return max(0.0, 1.0 - sim)

    # ── Loaders ────────────────────────────────────────────────────────────

    @classmethod
    def from_seed_texts(
        cls, seed_texts: List[str], source: str = "seed",
    ) -> "ReferenceCorpus":
        """Build from a list of 'Title. Abstract' strings or plain titles."""
        entries: List[CorpusEntry] = []
        for raw in seed_texts:
            s = (raw or "").strip()
            if not s:
                continue
            if "." in s and len(s.split(".", 1)[1].strip()) > 0:
                title, abstract = s.split(".", 1)
                entries.append(CorpusEntry(
                    title=title.strip(),
                    abstract=abstract.strip(),
                    source=source,
                ))
            else:
                entries.append(CorpusEntry(title=s, source=source))
        return cls(entries)

    @classmethod
    def from_archive(
        cls, ideas: List[Dict[str, Any]],
    ) -> "ReferenceCorpus":
        """Use the user's own archived ideas as the reference corpus.

        Useful when you want 'novel relative to what I've generated so
        far' — the corpus claim is correspondingly weak but transparent.
        """
        entries: List[CorpusEntry] = []
        for it in ideas or []:
            title = str(it.get("title", "") or "").strip()
            if not title:
                continue
            method = str(it.get("method", "") or "")
            motivation = str(it.get("motivation", "") or "")
            abstract = " ".join(t for t in (motivation, method) if t).strip()
            entries.append(CorpusEntry(
                title=title, abstract=abstract, source="archive",
            ))
        return cls(entries)


# ── Single-candidate scoring ────────────────────────────────────────────────

def _idea_text(idea: Dict[str, Any]) -> str:
    """Concatenate the fields of an idea that matter for semantic novelty."""
    parts = [
        str(idea.get("title", "") or ""),
        str(idea.get("motivation", "") or ""),
        str(idea.get("method", "") or ""),
        str(idea.get("hypothesis", "") or ""),
    ]
    return ". ".join(p for p in parts if p.strip())


def assess_idea_novelty(
    idea: Dict[str, Any],
    corpus: ReferenceCorpus,
) -> NoveltyAssessment:
    """Score one candidate idea against a corpus.

    The score is `1 - max(cosine similarity)`, capped to [0, 1]. The
    nearest entry's title/excerpt is attached so the UI can show *what*
    the candidate is being pushed away from.
    """
    if not isinstance(idea, dict) or not idea:
        raise ValueError("idea must be a non-empty dict")
    if not isinstance(corpus, ReferenceCorpus):
        raise ValueError("corpus must be a ReferenceCorpus")

    text = _idea_text(idea)
    nearest, sim = corpus.nearest_neighbor(text) if text else (None, 0.0)
    if nearest is None:
        # Empty corpus → novelty is maximally honest at 1.0 (the user
        # has supplied nothing to be near to). UI must surface this so
        # the user doesn't mistake it for a strong claim.
        return NoveltyAssessment(
            score=1.0, nearest_similarity=0.0,
            nearest_title="(corpus empty)", nearest_source="",
            nearest_excerpt="",
            corpus_size=len(corpus),
        )
    return NoveltyAssessment(
        score=max(0.0, 1.0 - sim),
        nearest_similarity=sim,
        nearest_title=nearest.title,
        nearest_source=nearest.source,
        nearest_year=nearest.year,
        nearest_excerpt=nearest.excerpt(),
        corpus_size=len(corpus),
    )


# ── End-to-end corpus-anchored batch ────────────────────────────────────────

# Default generator stages, in the order the design manifesto names:
# persona variation, cross-domain analogy, heretic falsification.
DEFAULT_GENERATORS: List[str] = ["persona", "analogy", "heretic"]
_VALID_GENERATORS: set = {
    "persona", "analogy", "heretic", "contradiction", "constraint",
    "future_back", "frontier", "genetic", "counterfactual",
    "failure_mode", "extremum", "inversion", "null_result",
    "underserved_cohort", "composable_primitive", "stakeholder_pareto",
}


def _gather_candidates(
    topic: str,
    generators: List[str],
    n_per_generator: int,
    claude_client: Any,
) -> List[Idea]:
    """Call each named generator's batch function and aggregate results.

    Imports are lazy so missing optional modules don't kill the whole
    pipeline — a missing generator is silently skipped.
    """
    out: List[Idea] = []
    for gen in generators:
        try:
            if gen == "persona":
                from persona_ideation import persona_swap
                result = persona_swap(
                    topic, n_per_persona=1,
                    claude_client=claude_client,
                )
                # persona_swap returns a PersonaResult dataclass with .ideas
                ideas_list = getattr(result, "ideas", result)
                out.extend(ideas_list or [])
            elif gen == "analogy":
                from analogy_ideation import analogy_batch
                out.extend(analogy_batch(
                    topic, n=n_per_generator,
                    claude_client=claude_client,
                ))
            elif gen == "heretic":
                from heretic_ideation import generate_heretic_batch
                out.extend(generate_heretic_batch(
                    topic, n=n_per_generator,
                    claude_client=claude_client,
                ))
            elif gen == "contradiction":
                from contradiction_ideation import (
                    generate_from_contradictions_batch,
                )
                out.extend(generate_from_contradictions_batch(
                    topic, n=n_per_generator,
                    claude_client=claude_client,
                ))
            elif gen == "future_back":
                from future_back_ideation import future_back_batch
                out.extend(future_back_batch(
                    topic, n=n_per_generator,
                    claude_client=claude_client,
                ))
            elif gen == "counterfactual":
                from counterfactual_literature import counterfactual_batch
                out.extend(counterfactual_batch(
                    topic, n=n_per_generator,
                    claude_client=claude_client,
                ))
            elif gen == "extremum":
                from extremum_ideation import extremum_batch
                out.extend(extremum_batch(
                    topic, n=n_per_generator,
                    claude_client=claude_client,
                ))
            elif gen == "inversion":
                from inversion_ideation import inversion_batch
                out.extend(inversion_batch(
                    topic, n=n_per_generator,
                    claude_client=claude_client,
                ))
            elif gen == "underserved_cohort":
                from underserved_cohort_ideation import underserved_cohort_batch
                out.extend(underserved_cohort_batch(
                    topic, n=n_per_generator,
                    claude_client=claude_client,
                ))
            # Other generators can be added similarly; silently skip if
            # not wired in here.
        except Exception:
            continue
    return out


def _map_elites_dedup(ideas_with_scores: List[Tuple[Idea, NoveltyAssessment]]
                          ) -> List[Tuple[Idea, NoveltyAssessment]]:
    """Keep at most one idea per (methodology_type, novelty_level) cell
    — the most novel one. Ideas without cell coordinates pass through
    untouched."""
    best: Dict[Tuple[str, str], Tuple[Idea, NoveltyAssessment]] = {}
    passthrough: List[Tuple[Idea, NoveltyAssessment]] = []
    for idea, ass in ideas_with_scores:
        meth = idea.methodology_type
        nov = idea.novelty_level
        if not (meth and nov):
            passthrough.append((idea, ass))
            continue
        key = (meth, nov)
        cur = best.get(key)
        if cur is None or ass.score > cur[1].score:
            best[key] = (idea, ass)
    return list(best.values()) + passthrough


def corpus_anchored_batch(
    topic: str,
    corpus: ReferenceCorpus,
    claude_client: Any = _AUTOLOAD,
    n_per_generator: int = 2,
    generators: Optional[List[str]] = None,
    keep_top: int = 5,
    min_novelty: float = 0.0,
    map_elites: bool = True,
) -> List[Idea]:
    """End-to-end corpus-anchored novelty pipeline.

    Stages:
      1. For each generator in `generators`, run its batch
      2. For each candidate, compute `assess_idea_novelty(...)`
      3. Drop anything below `min_novelty`
      4. If `map_elites`, keep only the most-novel idea per cell
      5. Sort by novelty descending, take top `keep_top`
      6. Attach NoveltyAssessment to `idea.execution_meta`
      7. Re-stamp `source_strategy='Q'` so the corpus-anchored origin is
         visible even when an underlying generator stamped its own code

    Each returned idea carries the corpus_limits disclaimer on
    `execution_meta.corpus_limits` so downstream UI cannot omit it.
    """
    if not topic or not topic.strip():
        raise ValueError("topic must be non-empty")
    if not isinstance(corpus, ReferenceCorpus):
        raise ValueError("corpus must be a ReferenceCorpus")
    if keep_top <= 0:
        return []
    if not 0.0 <= min_novelty <= 1.0:
        raise ValueError("min_novelty must be in [0, 1]")
    if generators is None:
        generators = list(DEFAULT_GENERATORS)
    bad = [g for g in generators if g not in _VALID_GENERATORS]
    if bad:
        raise ValueError(
            f"invalid generator(s): {bad}. Valid: {sorted(_VALID_GENERATORS)}"
        )
    if claude_client is _AUTOLOAD:
        try:
            from claude_provider import get_claude_client
            claude_client = get_claude_client()
        except Exception:
            claude_client = None

    # ── Stage 1: gather candidates ─────────────────────────────────────────
    candidates = _gather_candidates(
        topic.strip(), generators, n_per_generator, claude_client,
    )
    if not candidates:
        return []

    # ── Stage 2: score each candidate ──────────────────────────────────────
    scored: List[Tuple[Idea, NoveltyAssessment]] = []
    for cand in candidates:
        try:
            cand_dict = cand.to_dict()
        except Exception:
            continue
        ass = assess_idea_novelty(cand_dict, corpus)
        scored.append((cand, ass))

    # ── Stage 3: novelty floor ─────────────────────────────────────────────
    if min_novelty > 0.0:
        scored = [(i, a) for (i, a) in scored if a.score >= min_novelty]
    if not scored:
        return []

    # ── Stage 4: optional MAP-Elites cell dedup ───────────────────────────
    if map_elites:
        scored = _map_elites_dedup(scored)

    # ── Stage 5: sort + truncate ──────────────────────────────────────────
    scored.sort(key=lambda x: x[1].score, reverse=True)
    scored = scored[:keep_top]

    # ── Stages 6+7: stamp metadata ─────────────────────────────────────────
    out: List[Idea] = []
    for idea, ass in scored:
        meta = idea.execution_meta or {}
        if not isinstance(meta, dict):
            meta = {}
        prior_strategy = idea.source_strategy or ""
        meta["corpus_novelty"] = ass.to_dict()
        meta["corpus_limits"] = CORPUS_LIMITS_DISCLAIMER
        meta["upstream_strategy"] = prior_strategy
        meta["regen_mode"] = "corpus_anchored"
        meta["topic"] = topic.strip()
        idea.execution_meta = meta
        idea.source_strategy = "Q"
        out.append(idea)
    return out
