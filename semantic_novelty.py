"""
semantic_novelty.py — Dense, embedding-based novelty (the "advanced" tier).

Every other novelty scorer in IdeaGraph is LEXICAL: corpus_novelty,
arxiv_novelty, multi_corpus_novelty, density_novelty (LOF) all vectorize
text as TF / TF-IDF bag-of-words and measure cosine overlap. That makes
them blind to *semantic* novelty — an idea phrased with entirely fresh
vocabulary scores "novel" even if it restates a known result, and an
idea that paraphrases prior work scores "novel" because the words differ.

This module fixes that by scoring novelty in DENSE sentence-embedding
space (sentence-transformers, local, no API key):

    novelty = 1 − max_j cos( embed(idea), embed(paper_j) )

Two ideas that *mean* the same thing land close together even with no
shared words; two that share jargon but differ in meaning land far apart.
This is the standard "advanced" upgrade over lexical novelty.

Cost / deps:
    - Uses sentence-transformers + torch (already installed). The model
      (default all-MiniLM-L6-v2, ~80MB) downloads once from HuggingFace
      on first use, then is cached on disk and reused process-wide.
    - One arXiv API call per batch for the corpus (disk-cached), same as
      arxiv_novelty. Embedding compute is CPU-fast for typical sizes.
    - Degrades gracefully: if the model can't load (e.g. offline first
      run) every call returns score 0.0 with an `error`, never raising.

Public API (mirrors arxiv_novelty.py):
    compute_semantic_novelty_for_idea(idea, topic, ...)  → dict
    compute_semantic_novelty_for_batch(ideas, topic, ...) → list[dict]
    semantic_novelty_key(idea) → float                    # sort reader
    get_embedder() → SentenceTransformer | None           # lazy singleton
"""
from __future__ import annotations

import os as _os

# transformers (a sentence-transformers dependency) eagerly imports
# TensorFlow / tf_keras whenever TF is installed — and this environment's
# TF/protobuf stack is broken (tf_keras protobuf gencode/runtime
# mismatch). We only ever use the torch backend, so disable the TF + Flax
# import paths up front. MUST run before sentence_transformers/transformers
# are first imported (which happens lazily in get_embedder below).
_os.environ.setdefault("USE_TF", "0")
_os.environ.setdefault("USE_FLAX", "0")
_os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import threading
from typing import Any, Dict, List, Optional


# ── Lazy embedding model singleton ──────────────────────────────────────────
# Loading a SentenceTransformer is the expensive part (~1-3s + a one-time
# model download). Load once, guarded by a lock, and reuse for the life
# of the process. _EMBEDDER_FAILED memoizes a hard failure so we don't
# retry a doomed import on every idea.

_EMBEDDER = None
_EMBEDDER_LOCK = threading.Lock()
_EMBEDDER_FAILED: Optional[str] = None


def _model_name() -> str:
    try:
        import config as _cfg
        return getattr(_cfg, "SEMANTIC_NOVELTY_MODEL", "") \
            or "sentence-transformers/all-MiniLM-L6-v2"
    except Exception:
        return "sentence-transformers/all-MiniLM-L6-v2"


def get_embedder():
    """Return the cached SentenceTransformer, loading it on first call.
    Returns None (and memoizes why) if sentence-transformers or the model
    is unavailable — callers must handle None gracefully."""
    global _EMBEDDER, _EMBEDDER_FAILED
    if _EMBEDDER is not None:
        return _EMBEDDER
    if _EMBEDDER_FAILED is not None:
        return None
    with _EMBEDDER_LOCK:
        if _EMBEDDER is not None:
            return _EMBEDDER
        if _EMBEDDER_FAILED is not None:
            return None
        try:
            from sentence_transformers import SentenceTransformer
            _EMBEDDER = SentenceTransformer(_model_name())
            return _EMBEDDER
        except Exception as e:
            _EMBEDDER_FAILED = f"{type(e).__name__}: {str(e)[:200]}"
            return None


def embedder_status() -> Dict[str, Any]:
    """Lightweight introspection for the UI: is the model loaded / failed?
    Does NOT trigger a load."""
    return {
        "loaded": _EMBEDDER is not None,
        "model": _model_name(),
        "error": _EMBEDDER_FAILED,
    }


# ── Text extraction ──────────────────────────────────────────────────────────

def _idea_text(idea: Dict[str, Any]) -> str:
    """Concatenate the semantically meaningful fields of an idea."""
    parts = [
        idea.get("title"), idea.get("motivation"), idea.get("hypothesis"),
        idea.get("method"), idea.get("description"),
    ]
    return ". ".join(str(p).strip() for p in parts if p and str(p).strip())


# ── Public entry points ──────────────────────────────────────────────────────

def _failed(topic, since, until, msg) -> Dict[str, Any]:
    return {
        "score": 0.0, "nearest_title": "", "nearest_similarity": 0.0,
        "corpus_size": 0, "query": topic, "since": since, "until": until,
        "model": _model_name(), "error": msg,
    }


def compute_semantic_novelty_for_batch(
    ideas: List[Dict[str, Any]],
    topic: str,
    max_papers: int = 30,
    since: Optional[str] = None,
    until: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Score a batch of ideas' SEMANTIC novelty against the live arXiv
    corpus for `topic`. One arXiv call (disk-cached), one model load, all
    embeddings computed in two batched encode() calls. Stamps each idea's
    execution_meta["semantic_novelty"].
    """
    if not ideas:
        return []

    # Shared, memoized bi-encoder pass — corpus + ideas are encoded ONCE per
    # run and reused across all three dense-embedding tiers. Cosine tiers use
    # the L2-normalized view (dot product == cosine).
    from embedding_context import get_shared_context
    ctx = get_shared_context(
        ideas, topic, max_papers=max_papers, since=since, until=until,
    )
    if ctx.error:
        return [_failed(topic, since, until, ctx.error) for _ in ideas]

    corpus_titles = ctx.corpus_titles
    corpus_texts = ctx.corpus_texts
    corpus_mat = ctx.corpus_unit()
    idea_mat = ctx.idea_unit()

    # Score: novelty = 1 − max cosine to corpus.
    out: List[Dict[str, Any]] = []
    # (N_ideas × dim) @ (dim × N_corpus) → (N_ideas × N_corpus)
    sims = idea_mat @ corpus_mat.T
    for row_idx, idea in enumerate(ideas):
        row = sims[row_idx]
        j = int(row.argmax()) if row.size else -1
        nearest_sim = float(row[j]) if j >= 0 else 0.0
        score = max(0.0, min(1.0, 1.0 - nearest_sim))
        payload = {
            "score": round(score, 4),
            "nearest_title": corpus_titles[j] if j >= 0 else "",
            "nearest_similarity": round(nearest_sim, 4),
            "corpus_size": len(corpus_texts),
            "query": topic, "since": since, "until": until,
            "model": _model_name(), "error": None,
        }
        if isinstance(idea, dict):
            meta = idea.get("execution_meta")
            if not isinstance(meta, dict):
                meta = {}
            meta["semantic_novelty"] = payload
            idea["execution_meta"] = meta
        out.append(payload)
    return out


def compute_semantic_novelty_for_idea(
    idea: Dict[str, Any],
    topic: str,
    max_papers: int = 30,
    since: Optional[str] = None,
    until: Optional[str] = None,
) -> Dict[str, Any]:
    """Single-idea convenience wrapper around the batch scorer."""
    res = compute_semantic_novelty_for_batch(
        [idea], topic, max_papers=max_papers, since=since, until=until,
    )
    return res[0] if res else _failed(topic, since, until, "no result")


def semantic_novelty_key(idea: Dict[str, Any]) -> float:
    """Sort-mode reader. Returns the cached semantic-novelty score for an
    idea, or 0.0 if not yet computed."""
    if not isinstance(idea, dict):
        return 0.0
    meta = idea.get("execution_meta") or {}
    sn = meta.get("semantic_novelty") or {}
    try:
        return float(sn.get("score") or 0.0)
    except (TypeError, ValueError):
        return 0.0
