"""
embedding_context.py — one shared bi-encoder pass for all dense novelty tiers.

semantic_novelty, crossencoder_novelty, and mahalanobis_novelty all score the
SAME ideas against the SAME arXiv corpus with the SAME bi-encoder. The model
was already a shared singleton (semantic_novelty.get_embedder), BUT each tier
independently re-encoded the corpus AND the ideas — so within a single pipeline
run the bi-encoder ran encode() six times over identical text (2 per tier × 3
tiers). On a memory-tight machine that also tripled the peak encode buffers,
which is part of why the tiers were failing to complete.

This module memoizes ONE pass:
  • pull the arXiv corpus once (already disk-cached at the adapter level),
  • encode the corpus and the ideas once each, RAW (un-normalized),
  • hand every tier the same matrices.

Cosine tiers (semantic, reranked) take an L2-normalized *view* (cheap numpy,
computed lazily once); the Mahalanobis tier uses the raw embeddings for its
covariance. Net effect: 2 encodes per run instead of 6 (~3× less CPU), lower
peak memory, and the tiers actually finish. Strictly additive + defensive:
any failure yields a context with `.error` set, never raises.

Public API:
    get_shared_context(ideas, topic, max_papers, since, until) → EmbeddingContext
    idea_text(idea) → str
"""
from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional


def idea_text(idea: Dict[str, Any]) -> str:
    """Concatenate the semantically meaningful fields of an idea. Kept
    identical to the per-tier extractors so the memo key is stable."""
    parts = [
        idea.get("title"), idea.get("motivation"), idea.get("hypothesis"),
        idea.get("method"), idea.get("description"),
    ]
    return ". ".join(str(p).strip() for p in parts if p and str(p).strip())


def _l2norm(mat):
    """Row-wise L2 normalize (so dot product == cosine). Zero rows stay zero."""
    import numpy as np
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return mat / n


class EmbeddingContext:
    """Shared, per-run bundle. `error` set → tiers should surface it and skip.
    corpus_raw / idea_raw are un-normalized (for Mahalanobis); corpus_unit() /
    idea_unit() lazily return the normalized views (for cosine tiers)."""

    __slots__ = (
        "embedder", "papers", "corpus_texts", "corpus_titles", "corpus_raw",
        "idea_texts", "idea_raw", "error", "_corpus_unit", "_idea_unit",
    )

    def __init__(
        self, embedder=None, papers=None, corpus_texts=None, corpus_titles=None,
        corpus_raw=None, idea_texts=None, idea_raw=None, error=None,
    ):
        self.embedder = embedder
        self.papers = papers or []
        self.corpus_texts = corpus_texts or []
        self.corpus_titles = corpus_titles or []
        self.corpus_raw = corpus_raw
        self.idea_texts = idea_texts or []
        self.idea_raw = idea_raw
        self.error = error
        self._corpus_unit = None
        self._idea_unit = None

    def corpus_unit(self):
        if self._corpus_unit is None and self.corpus_raw is not None:
            self._corpus_unit = _l2norm(self.corpus_raw)
        return self._corpus_unit

    def idea_unit(self):
        if self._idea_unit is None and self.idea_raw is not None:
            self._idea_unit = _l2norm(self.idea_raw)
        return self._idea_unit


# ── Memo cache (tiny LRU — one run's three tiers share one entry) ────────────
_CACHE: "OrderedDict[str, EmbeddingContext]" = OrderedDict()
_CACHE_MAX = 4
_LOCK = threading.RLock()


def _key(topic, since, until, max_papers, idea_texts: List[str]) -> str:
    h = hashlib.md5(
        "\x00".join(idea_texts).encode("utf-8", "ignore"),
        usedforsecurity=False,
    ).hexdigest()
    return f"{(topic or '').strip()}|{since or ''}|{until or ''}|{int(max_papers)}|{h}"


def clear_cache() -> None:
    with _LOCK:
        _CACHE.clear()


def get_shared_context(
    ideas: List[Dict[str, Any]],
    topic: str,
    max_papers: int = 30,
    since: Optional[str] = None,
    until: Optional[str] = None,
) -> EmbeddingContext:
    """Memoized EmbeddingContext for (topic, ideas, date window). The first
    novelty tier in a run computes it; the other two hit the cache and reuse
    the same corpus + idea embeddings. Never raises."""
    if not ideas:
        return EmbeddingContext(error="no ideas")
    if not topic or not topic.strip():
        return EmbeddingContext(error="empty topic")

    idea_texts = [idea_text(i) or (i.get("title") or "") for i in ideas]
    key = _key(topic, since, until, max_papers, idea_texts)
    with _LOCK:
        ctx = _CACHE.get(key)
        if ctx is not None:
            _CACHE.move_to_end(key)
            return ctx

    ctx = _build_context(idea_texts, topic, max_papers, since, until)
    with _LOCK:
        _CACHE[key] = ctx
        _CACHE.move_to_end(key)
        while len(_CACHE) > _CACHE_MAX:
            _CACHE.popitem(last=False)
    return ctx


def _build_context(idea_texts, topic, max_papers, since, until) -> EmbeddingContext:
    # Shared bi-encoder singleton (loads at most once process-wide).
    try:
        from semantic_novelty import get_embedder
        embedder = get_embedder()
    except Exception as e:
        return EmbeddingContext(error=f"bi-encoder import failed: {e}")
    if embedder is None:
        return EmbeddingContext(error="embedding model unavailable")

    try:
        import numpy as np  # noqa: F401  (ensures numpy present before encode)
    except Exception as e:
        return EmbeddingContext(error=f"numpy missing: {e}")

    # Corpus — cached, no-key arXiv adapter.
    try:
        from tools import arxiv as _arxiv
        papers = _arxiv.search(
            topic, max_results=max_papers, since=since, until=until,
        )
    except Exception as e:
        return EmbeddingContext(error=f"arXiv search failed: {e}")
    if not papers:
        err = "no arXiv results for topic"
        try:
            le = _arxiv.last_error()
            if le and le.get("message"):
                err = le["message"]
        except Exception:
            pass
        return EmbeddingContext(error=err)

    corpus_texts: List[str] = []
    corpus_titles: List[str] = []
    for p in papers:
        title = getattr(p, "title", "") or ""
        abstract = getattr(p, "abstract", "") or ""
        text = f"{title}. {abstract}".strip()
        if text:
            corpus_texts.append(text)
            corpus_titles.append(title)
    if not corpus_texts:
        return EmbeddingContext(error="corpus had no usable text")

    # THE shared pass: encode corpus + ideas once each, raw (un-normalized).
    try:
        corpus_raw = embedder.encode(
            corpus_texts, normalize_embeddings=False,
            convert_to_numpy=True, show_progress_bar=False,
        )
        idea_raw = embedder.encode(
            idea_texts, normalize_embeddings=False,
            convert_to_numpy=True, show_progress_bar=False,
        )
    except Exception as e:
        return EmbeddingContext(error=f"embedding failed: {e}")

    return EmbeddingContext(
        embedder=embedder, papers=papers,
        corpus_texts=corpus_texts, corpus_titles=corpus_titles,
        corpus_raw=corpus_raw, idea_texts=idea_texts, idea_raw=idea_raw,
    )
