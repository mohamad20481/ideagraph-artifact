"""
bayesian_surprise.py — measure how much an idea epistemically shifts the
model's belief about a research field.

Background (Extension 3 from the paper roadmap):
    Standard LLM-as-judge novelty scores reward exotic-sounding ideas
    that may be physically impossible while penalizing rigorous
    incremental work. Bayesian Surprise grounds novelty in information
    theory instead: it asks how much the model's distribution over
    "what the field looks like" must update to accommodate the new
    hypothesis. Formally:

        S(H) = D_KL( P(X | H) ∥ P(X) )

    where P(X) is the prior over field-describing text (without H) and
    P(X | H) is the posterior (with H prepended).

Because no current LLM API exposes uniform token-level logprobs, we
approximate the KL divergence with a **portable surrogate**:

    1. Sample N short "describe the field" responses from the LLM
       WITHOUT the hypothesis  → aggregate token-frequency vector P
    2. Sample N short "describe the field" responses WITH the
       hypothesis prepended    → aggregate token-frequency vector P|H
    3. Surprise ≈ 1 − cos(P, P|H)     (range [0, 1])

    On token-frequency distributions this is a Bhattacharyya-style
    proxy for KL — bounded, cheap, and provider-agnostic. For
    practitioners who prefer a divergence proper, we also expose a
    Jensen-Shannon variant that is symmetric and finite.

We then multiply Surprise by a one-shot **Plausibility** score (LLM
rates the hypothesis 0-1 on physical/empirical believability) to
penalize exotic-but-impossible ideas:

    BayesianScore(H) = Surprise(H) × Plausibility(H)

This is the "high information gain AND low perplexity under
verification" criterion. High BayesianScore = idea that genuinely
moves the model's worldview *and* is verifiable.

Cost note: 2N + 1 LLM calls per idea by default (N=3 → 7 calls/idea).
Cache the result via `compute_surprise_for_idea(...)`'s `cache` arg or
the disk-cache helper at the bottom.

Public API:
    compute_bayesian_surprise(hypothesis, topic, llm_client, ...) → dict
    compute_surprise_for_idea(idea_dict, topic, llm_client, ...) → dict
    bayesian_score_key(idea) → float       (for sort_modes integration)
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

# Reuse the codebase's tokenizer + cosine. They're tuned for the same
# scientific-text vocabulary the rest of the pipeline reads.
try:
    from corpus_novelty import text_vector, cosine_similarity, tokenize
    _HAS_CORPUS_NOVELTY = True
except Exception:
    _HAS_CORPUS_NOVELTY = False


_AUTOLOAD = object()


# ── System prompts ──────────────────────────────────────────────────────────

_PRIOR_SYSTEM = (
    "You are a research-trends analyst. Given a topic, you describe "
    "the state of the art, the main open problems, and the active "
    "directions in 2-4 sentences. Use precise technical vocabulary. "
    "Do NOT speculate about specific unpublished ideas — describe "
    "the field as it stands."
)

_POST_SYSTEM = (
    "You are a research-trends analyst. You are given a topic AND a "
    "newly-proposed hypothesis H. Describe how the field changes if "
    "H turns out to be true: which directions get reinforced, which "
    "get invalidated, and what new problems open up. 2-4 sentences."
)

_PLAUSIBILITY_SYSTEM = (
    "You are a senior reviewer. Rate the physical and empirical "
    "plausibility of the following hypothesis on a 0.0-1.0 scale "
    "where 0 = violates known physics or basic statistics, 0.5 = "
    "speculative but conceivable, 1.0 = consistent with the standard "
    "model of the field. Reply with JUST a number, no prose."
)


# ── Core math ──────────────────────────────────────────────────────────────

def _aggregate_token_vector(texts: List[str]) -> Dict[str, float]:
    """Average the unit-length token-frequency vectors of N texts to
    approximate an *expected* token distribution under the policy. The
    result is also unit-normalized so it stays cosine-compatible."""
    if not texts or not _HAS_CORPUS_NOVELTY:
        return {}
    acc: Dict[str, float] = {}
    n_valid = 0
    for t in texts:
        v = text_vector(t or "")
        if not v:
            continue
        n_valid += 1
        for k, val in v.items():
            acc[k] = acc.get(k, 0.0) + val
    if not acc or n_valid == 0:
        return {}
    # Average then L2-renormalize so cos(P, P|H) = dot product.
    avg = {k: v / n_valid for k, v in acc.items()}
    norm = math.sqrt(sum(v * v for v in avg.values()))
    if norm <= 0.0:
        return {}
    return {k: v / norm for k, v in avg.items()}


def surprise_from_vectors(
    prior_vec: Dict[str, float], posterior_vec: Dict[str, float],
) -> float:
    """1 − cos(P, P|H). Range [0, 1] (vectors are non-negative L2-unit).

    Interpretation:
      0.00 — the field-description didn't budge; H is a non-event.
      0.30 — moderate shift; H reshapes some directions.
      0.70 — major shift; H redirects much of the discourse.
      1.00 — orthogonal distributions; H is unrelated or paradigm-breaking.
    """
    if not prior_vec or not posterior_vec or not _HAS_CORPUS_NOVELTY:
        return 0.0
    sim = cosine_similarity(prior_vec, posterior_vec)
    # Numerical noise can drift slightly above 1; clamp.
    return max(0.0, min(1.0, 1.0 - sim))


def jensen_shannon_divergence(
    prior_vec: Dict[str, float], posterior_vec: Dict[str, float],
) -> float:
    """Symmetric, bounded KL alternative: JSD(P, Q) ∈ [0, log 2 ≈ 0.693].

    Normalized to [0, 1] before returning so it composes with the
    cosine-distance surprise on the same scale.
    """
    if not prior_vec or not posterior_vec:
        return 0.0
    # Re-normalize to sum-to-1 so they're proper probability vectors.
    def _to_prob(v: Dict[str, float]) -> Dict[str, float]:
        s = sum(v.values())
        if s <= 0.0:
            return {}
        return {k: val / s for k, val in v.items()}

    p = _to_prob(prior_vec)
    q = _to_prob(posterior_vec)
    if not p or not q:
        return 0.0
    keys = set(p) | set(q)
    m = {k: 0.5 * (p.get(k, 0.0) + q.get(k, 0.0)) for k in keys}

    def _kl(a: Dict[str, float], b: Dict[str, float]) -> float:
        out = 0.0
        for k, av in a.items():
            bv = b.get(k, 0.0)
            if av > 0.0 and bv > 0.0:
                out += av * math.log(av / bv)
        return out

    jsd = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    # Normalize by log 2 so [0, 1].
    return max(0.0, min(1.0, jsd / math.log(2.0)))


# ── LLM-driven sampling ─────────────────────────────────────────────────────

def _resolve_client(llm_client: Any) -> Any:
    """Resolve the autoload sentinel to the project's default Claude client."""
    if llm_client is not _AUTOLOAD:
        return llm_client
    try:
        from claude_provider import get_claude_client
        return get_claude_client()
    except Exception:
        return None


def _safe_call_text(
    client: Any, system: str, user: str,
    max_tokens: int = 220, temperature: float = 0.8,
) -> str:
    """Wrap the project's LLM call returning bare text. Returns "" on
    any failure so the caller's aggregation step can skip cleanly."""
    if client is None:
        return ""
    try:
        resp = client.call(
            system=system, user=user,
            max_tokens=max_tokens, temperature=temperature,
        )
    except Exception:
        return ""
    if not getattr(resp, "success", False):
        return ""
    return (getattr(resp, "text", "") or "").strip()


def _parse_plausibility(s: str) -> float:
    """Extract the first float in [0,1] from an LLM rating reply."""
    if not s:
        return 0.5
    import re
    m = re.search(r"([0-9]*\.?[0-9]+)", s)
    if not m:
        return 0.5
    try:
        v = float(m.group(1))
    except Exception:
        return 0.5
    return max(0.0, min(1.0, v))


# ── Public entry points ────────────────────────────────────────────────────

@dataclass
class SurpriseResult:
    """One Bayesian-Surprise measurement for a hypothesis."""
    surprise: float          # 1 − cos(P, P|H) ∈ [0, 1]
    jsd: float               # Jensen-Shannon variant ∈ [0, 1]
    plausibility: float      # LLM-rated 0-1
    bayesian_score: float    # surprise × plausibility
    n_samples: int           # samples per direction
    cache_key: str           # for disk cache lookups

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_bayesian_surprise(
    hypothesis: str,
    topic: str,
    llm_client: Any = _AUTOLOAD,
    n_samples: int = 3,
    max_tokens: int = 220,
    rate_plausibility: bool = True,
) -> SurpriseResult:
    """Run the full prior/posterior/plausibility pipeline for one hypothesis.

    Cost: 2N + (1 if rate_plausibility else 0) LLM calls.

    Returns a zero-filled SurpriseResult if the LLM client is unavailable
    or the prompts come back empty — never raises, so the QD loop keeps
    running even on transient API failures.
    """
    cache_key = _hash_key(hypothesis, topic, n_samples)
    if not hypothesis or not hypothesis.strip():
        return SurpriseResult(0.0, 0.0, 0.5, 0.0, 0, cache_key)

    client = _resolve_client(llm_client)
    if client is None:
        return SurpriseResult(0.0, 0.0, 0.5, 0.0, 0, cache_key)

    prior_user = (
        f"Topic: {topic}\n\n"
        f"Describe the current state of this field — main directions, "
        f"open problems, dominant methods. 2-4 sentences."
    )
    posterior_user = (
        f"Topic: {topic}\n"
        f"Proposed hypothesis (H): {hypothesis.strip()}\n\n"
        f"If H is true, describe how the field changes: what gets "
        f"reinforced, what gets invalidated, what new problems open. "
        f"2-4 sentences."
    )

    prior_texts: List[str] = []
    posterior_texts: List[str] = []
    for _ in range(max(1, int(n_samples))):
        prior_texts.append(_safe_call_text(
            client, _PRIOR_SYSTEM, prior_user,
            max_tokens=max_tokens, temperature=0.8,
        ))
        posterior_texts.append(_safe_call_text(
            client, _POST_SYSTEM, posterior_user,
            max_tokens=max_tokens, temperature=0.8,
        ))

    prior_vec = _aggregate_token_vector(prior_texts)
    post_vec = _aggregate_token_vector(posterior_texts)
    surprise = surprise_from_vectors(prior_vec, post_vec)
    jsd = jensen_shannon_divergence(prior_vec, post_vec)

    plausibility = 0.5
    if rate_plausibility:
        raw = _safe_call_text(
            client, _PLAUSIBILITY_SYSTEM,
            f"Topic: {topic}\nHypothesis: {hypothesis.strip()}\n"
            f"Rate plausibility 0.0-1.0. Reply with just a number.",
            max_tokens=10, temperature=0.0,
        )
        plausibility = _parse_plausibility(raw)

    return SurpriseResult(
        surprise=surprise,
        jsd=jsd,
        plausibility=plausibility,
        bayesian_score=surprise * plausibility,
        n_samples=int(n_samples),
        cache_key=cache_key,
    )


def compute_surprise_for_idea(
    idea: Dict[str, Any],
    topic: str,
    llm_client: Any = _AUTOLOAD,
    n_samples: int = 3,
    rate_plausibility: bool = True,
    persist_to_meta: bool = True,
) -> Dict[str, Any]:
    """Convenience: run on an Idea dict and (optionally) stamp the
    result onto `idea["execution_meta"]["bayesian_surprise"]` so the
    sort_modes registry can read it later without recomputing.

    The hypothesis used for the prior/posterior comparison is the
    idea's `hypothesis` field if present, otherwise `motivation`,
    otherwise `title`. Falls back to empty (zero score) if all three
    are missing.
    """
    hypothesis = (
        (idea.get("hypothesis") or "").strip()
        or (idea.get("motivation") or "").strip()
        or (idea.get("title") or "").strip()
    )
    result = compute_bayesian_surprise(
        hypothesis=hypothesis,
        topic=topic,
        llm_client=llm_client,
        n_samples=n_samples,
        rate_plausibility=rate_plausibility,
    )
    payload = result.to_dict()
    if persist_to_meta and isinstance(idea, dict):
        meta = idea.get("execution_meta")
        if not isinstance(meta, dict):
            meta = {}
        meta["bayesian_surprise"] = payload
        idea["execution_meta"] = meta
    return payload


def bayesian_score_key(idea: Dict[str, Any]) -> float:
    """Read the cached Bayesian score from execution_meta for sort_modes.

    Returns 0.0 if surprise hasn't been computed for this idea. The sort
    handler should call `compute_surprise_for_idea` first on the visible
    set to populate the meta before sorting.
    """
    if not isinstance(idea, dict):
        return 0.0
    meta = idea.get("execution_meta") or {}
    bs = meta.get("bayesian_surprise") or {}
    try:
        return float(bs.get("bayesian_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def surprise_key(idea: Dict[str, Any]) -> float:
    """Read cached raw surprise (without plausibility weighting). Used
    by the directional 'epistemic_shift' sort mode."""
    if not isinstance(idea, dict):
        return 0.0
    meta = idea.get("execution_meta") or {}
    bs = meta.get("bayesian_surprise") or {}
    try:
        return float(bs.get("surprise", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


# ── Caching ─────────────────────────────────────────────────────────────────

def _hash_key(hypothesis: str, topic: str, n_samples: int) -> str:
    h = hashlib.sha256(
        f"{topic}||{hypothesis}||N={n_samples}".encode("utf-8")
    ).hexdigest()
    return h[:16]


_DEFAULT_CACHE_DIR = ".ideagraph_surprise_cache"


def cached_compute_surprise_for_idea(
    idea: Dict[str, Any],
    topic: str,
    llm_client: Any = _AUTOLOAD,
    n_samples: int = 3,
    rate_plausibility: bool = True,
    cache_dir: str = _DEFAULT_CACHE_DIR,
) -> Dict[str, Any]:
    """Same as compute_surprise_for_idea but persists each result to a
    JSON file keyed by SHA-256 of (topic, hypothesis, N). Subsequent
    calls with the same args read from disk — useful for re-runs that
    don't want to pay 6 LLM calls per idea every time."""
    hypothesis = (
        (idea.get("hypothesis") or "").strip()
        or (idea.get("motivation") or "").strip()
        or (idea.get("title") or "").strip()
    )
    key = _hash_key(hypothesis, topic, n_samples)
    try:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"{key}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            # Stamp to meta and return without re-LLM-calling.
            meta = idea.get("execution_meta") or {}
            meta["bayesian_surprise"] = payload
            idea["execution_meta"] = meta
            return payload
    except Exception:
        pass

    payload = compute_surprise_for_idea(
        idea, topic, llm_client=llm_client,
        n_samples=n_samples,
        rate_plausibility=rate_plausibility,
        persist_to_meta=True,
    )

    try:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"{payload.get('cache_key', key)}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return payload
