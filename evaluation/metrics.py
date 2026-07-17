"""
evaluation/metrics.py — canonical definitions of RAS, DI / Div-Pair, and NFT.

These are THE definitions. If a paper reports these metrics, it must state
the formulas below and cite this file as the reference implementation.

Definitions
-----------
Embedding. e(x) is the L2-normalized sentence embedding of the idea's
canonical text (see `idea_to_text`), from the model named in `embedder_name`
(default: the project bi-encoder, sentence-transformers/all-MiniLM-L6-v2).

Pairwise dissimilarity. For ideas i, j:
    d(i, j) = 1 - max(0, cos(e_i, e_j))            in [0, 1]
(The floor at 0 keeps d in [0,1]; sentence embeddings of topical text almost
never produce negative cosine, and when they do we treat them as "maximally
dissimilar" rather than letting d exceed 1.)

Diversity Index (per topic t with idea set I_t, |I_t| >= 2):
    DI_t  = (2 / (|I_t| (|I_t|-1))) * sum_{i<j} d(i, j)
    DI    = mean_t DI_t
Div-Pair (over a flat pool P, |P| >= 2) applies the SAME formula to P.

Reference Alignment Score (per topic t with reference text r_t):
    RAS_t = agg_{i in I_t} cos(e_i, e(r_t))     agg in {max (default), mean}
    RAS   = mean_t max(0, RAS_t)
`max` asks "did the BEST idea recover the reference contribution?" (the
retrieval-style reading); `mean` asks "how aligned is the portfolio overall?".
Whichever is used must be stated.

Novelty-Feasibility Tradeoff (per idea i with judge scores nov_i, feas_i in
[0, 1]; see evaluation/judges.py for how those are produced):
    NFT_i = 2 * nov_i * feas_i / (nov_i + feas_i)    (harmonic mean; 0 if both 0)
    NFT   = mean_i NFT_i
Harmonic mean is chosen deliberately: an idea that is very novel but
infeasible (or vice versa) scores LOW — that is what "tradeoff" means here.
The per-idea-then-average order is also deliberate (a portfolio cannot score
well by containing some purely-novel and some purely-feasible ideas).

All functions raise EvaluationError instead of degrading silently.
"""
from __future__ import annotations

import os as _os
_os.environ.setdefault("USE_TF", "0")
_os.environ.setdefault("USE_FLAX", "0")

from typing import Any, Dict, List, Optional, Sequence

from evaluation import EvaluationError

# Canonical field order for embedding an idea / benchmark tuple.
_TEXT_FIELDS = (
    "title", "motivation", "hypothesis", "method", "method_sketch",
    "dataset", "metrics", "baselines", "expected_outcome",
)


def idea_to_text(idea: Dict[str, Any]) -> str:
    """Canonical text of an idea or CrossVal tuple: the non-empty fields of
    `_TEXT_FIELDS`, in that order, joined by '. '. Raises if nothing usable."""
    if not isinstance(idea, dict):
        raise EvaluationError(f"idea must be a dict, got {type(idea).__name__}")
    parts = [
        str(idea[k]).strip() for k in _TEXT_FIELDS
        if idea.get(k) and str(idea[k]).strip()
    ]
    if not parts:
        raise EvaluationError(
            f"idea has no usable text fields ({', '.join(_TEXT_FIELDS)}): "
            f"keys present = {sorted(idea.keys())}"
        )
    return ". ".join(parts)


def embedder_name() -> str:
    try:
        import config as _cfg
        return getattr(_cfg, "SEMANTIC_NOVELTY_MODEL", "") \
            or "sentence-transformers/all-MiniLM-L6-v2"
    except Exception:
        return "sentence-transformers/all-MiniLM-L6-v2"


def _get_embedder(embedder: Any = None):
    """Return a .encode()-capable embedder. Injectable for tests; otherwise
    the shared project bi-encoder. Raises EvaluationError if unavailable —
    evaluation NEVER silently falls back."""
    if embedder is not None:
        return embedder
    try:
        from semantic_novelty import get_embedder
        emb = get_embedder()
    except Exception as e:
        raise EvaluationError(f"could not import shared embedder: {e}") from e
    if emb is None:
        raise EvaluationError(
            "embedding model unavailable (semantic_novelty.get_embedder "
            "returned None) — evaluation requires a working embedder"
        )
    return emb


def embed_texts(texts: Sequence[str], embedder: Any = None):
    """L2-normalized embeddings (n × dim numpy array) for `texts`."""
    import numpy as np
    if not texts:
        raise EvaluationError("embed_texts: empty text list")
    for i, t in enumerate(texts):
        if not t or not str(t).strip():
            raise EvaluationError(f"embed_texts: text #{i} is empty")
    emb = _get_embedder(embedder)
    mat = emb.encode(
        list(texts), normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    )
    mat = np.asarray(mat, dtype="float64")
    # Re-normalize defensively (stub embedders in tests may skip it).
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    if (norms == 0).any():
        raise EvaluationError("embed_texts: got a zero-norm embedding")
    return mat / norms


def pairwise_dissimilarity(mat) -> float:
    """Mean over unordered pairs of d(i,j) = 1 - max(0, cos). mat must be
    row-normalized (n >= 2)."""
    import numpy as np
    n = mat.shape[0]
    if n < 2:
        raise EvaluationError(f"pairwise_dissimilarity needs >= 2 items, got {n}")
    sims = mat @ mat.T
    iu = np.triu_indices(n, k=1)
    d = 1.0 - np.maximum(0.0, sims[iu])
    return float(d.mean())


# ── Public metrics ───────────────────────────────────────────────────────────

def diversity_index(ideas: Sequence[Dict[str, Any]], embedder: Any = None) -> float:
    """DI over one idea set (>= 2 ideas). Range [0, 1], higher = more diverse."""
    if len(ideas) < 2:
        raise EvaluationError(f"diversity_index needs >= 2 ideas, got {len(ideas)}")
    mat = embed_texts([idea_to_text(i) for i in ideas], embedder=embedder)
    return pairwise_dissimilarity(mat)


def div_pair(pool: Sequence[Dict[str, Any]], embedder: Any = None) -> float:
    """Div-Pair: identical formula to DI, applied to a flat pool of ideas
    (e.g. all generated ideas in a benchmark)."""
    return diversity_index(pool, embedder=embedder)


def ras_topic(
    ideas: Sequence[Dict[str, Any]],
    reference_text: str,
    agg: str = "max",
    embedder: Any = None,
) -> float:
    """RAS for ONE topic: agg over ideas of cos(e_idea, e_reference),
    floored at 0. `reference_text` is the reference paper's contribution
    statement (title + abstract is acceptable and must be stated)."""
    if not ideas:
        raise EvaluationError("ras_topic: empty idea set")
    if not reference_text or not str(reference_text).strip():
        raise EvaluationError("ras_topic: empty reference_text")
    if agg not in ("max", "mean"):
        raise EvaluationError(f"ras_topic: agg must be 'max' or 'mean', got {agg!r}")
    import numpy as np
    texts = [idea_to_text(i) for i in ideas] + [str(reference_text).strip()]
    mat = embed_texts(texts, embedder=embedder)
    idea_mat, ref_vec = mat[:-1], mat[-1]
    sims = idea_mat @ ref_vec
    val = float(sims.max()) if agg == "max" else float(sims.mean())
    return max(0.0, val)


def ras(
    per_topic_ideas: Dict[str, Sequence[Dict[str, Any]]],
    per_topic_reference: Dict[str, str],
    agg: str = "max",
    embedder: Any = None,
) -> float:
    """RAS over a benchmark: mean over topics of ras_topic. Every topic must
    have a reference; a missing one is an error, not a skip."""
    if not per_topic_ideas:
        raise EvaluationError("ras: no topics")
    missing = sorted(set(per_topic_ideas) - set(per_topic_reference))
    if missing:
        raise EvaluationError(f"ras: topics missing references: {missing[:5]}")
    vals = [
        ras_topic(per_topic_ideas[t], per_topic_reference[t],
                  agg=agg, embedder=embedder)
        for t in sorted(per_topic_ideas)
    ]
    return float(sum(vals) / len(vals))


def nft_idea(nov: float, feas: float) -> float:
    """Per-idea NFT: harmonic mean of novelty and feasibility (both in [0,1])."""
    for name, v in (("nov", nov), ("feas", feas)):
        try:
            fv = float(v)
        except (TypeError, ValueError) as e:
            raise EvaluationError(f"nft_idea: {name} not numeric: {v!r}") from e
        if not (0.0 <= fv <= 1.0):
            raise EvaluationError(f"nft_idea: {name}={fv} outside [0, 1]")
    nov_f, feas_f = float(nov), float(feas)
    if nov_f + feas_f == 0.0:
        return 0.0
    return 2.0 * nov_f * feas_f / (nov_f + feas_f)


def nft(scored_ideas: Sequence[Dict[str, float]]) -> float:
    """NFT over a set: mean over ideas of nft_idea(nov, feas). Each element
    must carry 'nov' and 'feas' in [0, 1] (see judges.to_unit_scores)."""
    if not scored_ideas:
        raise EvaluationError("nft: empty idea list")
    vals = []
    for k, s in enumerate(scored_ideas):
        if "nov" not in s or "feas" not in s:
            raise EvaluationError(f"nft: idea #{k} missing 'nov'/'feas' keys")
        vals.append(nft_idea(s["nov"], s["feas"]))
    return float(sum(vals) / len(vals))
