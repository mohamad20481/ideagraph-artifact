"""
multi_llm_ensemble.py — parallel cross-provider ideation + diversity filter.

Generates research ideas on a topic across multiple LLM providers
(DeepSeek / Kimi / Anthropic / OpenAI / Gemini / Groq) in parallel, then
keeps only those whose pairwise method-tokens are sufficiently distant.
Different model families have different priors, so this exploits that
mechanically rather than relying on prompt-luck to produce diversity.

Public API:
    EnsembleResult                                  → dataclass
    available_providers()                            → List[str]
    ensemble_generate(topic, providers=None, ...)    → EnsembleResult
"""
from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from models.idea import Idea, METHODOLOGY_TYPES, NOVELTY_LEVELS


# ─────────────────────────────────────────────────────────────────────────────
# Provider client adapters
# ─────────────────────────────────────────────────────────────────────────────
#
# We need a uniform .call(system, user, ...) -> Response interface across
# different providers. base_agent's _build_client_for_provider returns an
# openai.OpenAI client (or a stub for anthropic); we wrap each to look like
# claude_provider.ClaudeResponse for downstream parsing.


@dataclass
class _ProviderResponse:
    success: bool
    text: str = ""
    model: str = ""
    error: Optional[str] = None


class _ProviderClient:
    """Thin uniform wrapper around any configured LLM provider."""

    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model

    def call(self, system: str, user: str,
              max_tokens: int = 1024, temperature: float = 0.7,
              json_mode: bool = True) -> _ProviderResponse:
        """Make a single provider call. Returns _ProviderResponse."""
        import config
        if self.provider == "anthropic":
            try:
                from claude_provider import get_claude_client
                client = get_claude_client()
                r = client.call(
                    system=system, user=user,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    json_mode=json_mode,
                )
                return _ProviderResponse(
                    success=bool(getattr(r, "success", False)),
                    text=getattr(r, "text", "") or "",
                    model=getattr(r, "model", "") or self.model,
                    error=getattr(r, "error", None),
                )
            except Exception as e:
                return _ProviderResponse(success=False, error=f"{type(e).__name__}: {e}")

        # OpenAI-compatible providers (deepseek, kimi, openai, gemini, groq, azure)
        try:
            from agents.base_agent import BaseAgent
            client = BaseAgent._build_client_for_provider(self.provider)
        except Exception as e:
            return _ProviderResponse(success=False, error=f"client_build: {e}")

        try:
            # Kimi temperature coercion (mirrors base_agent logic)
            temp = float(temperature)
            if self.provider == "kimi":
                if (self.model or "").lower().startswith("kimi-k2"):
                    temp = 1.0
                else:
                    temp = max(0.3, temp)

            resp_fmt = {}
            if json_mode and self.provider in ("deepseek", "openai", "gemini",
                                                 "azure", "kimi", "xai"):
                resp_fmt = {"response_format": {"type": "json_object"}}

            r = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=temp,
                timeout=60,
                **resp_fmt,
            )
            text = (r.choices[0].message.content or "") if r.choices else ""
            return _ProviderResponse(
                success=True, text=text, model=self.model,
            )
        except Exception as e:
            return _ProviderResponse(
                success=False, error=f"{type(e).__name__}: {str(e)[:200]}",
            )


# ─────────────────────────────────────────────────────────────────────────────
# Provider availability
# ─────────────────────────────────────────────────────────────────────────────

def available_providers() -> List[str]:
    """Return the list of providers with a configured API key."""
    import config
    candidates = [
        ("deepseek", "DEEPSEEK_API_KEY"),
        ("kimi", "KIMI_API_KEY"),
        ("anthropic", "ANTHROPIC_API_KEY"),
        ("openai", "OPENAI_API_KEY"),
        ("gemini", "GEMINI_API_KEY"),
        ("groq", "GROQ_API_KEY"),
        ("azure", "AZURE_API_KEY"),
    ]
    out = []
    for name, attr in candidates:
        v = getattr(config, attr, "") or ""
        if v and not v.startswith(("sk-xxx", "your-")):
            out.append(name)
    return out


def _default_model(provider: str) -> str:
    """Get the default model for a provider."""
    import config
    return config._DEFAULT_MODELS.get(provider, "")


# ─────────────────────────────────────────────────────────────────────────────
# Prompt + parser
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM = (
    "You are a research scientist generating one novel research idea on "
    "the given topic. Take an opinionated, specific stance — your model "
    "family has its own perspective, lean into it. Output ONLY valid JSON "
    "with the schema. methodology_type must be one of: "
    f"{', '.join(METHODOLOGY_TYPES)}. novelty_level must be one of: "
    f"{', '.join(NOVELTY_LEVELS)}."
)


def _user_prompt(topic: str, hint: str = "") -> str:
    hint_section = f"\n\nAdditional framing: {hint}\n" if hint else "\n"
    return (
        f"Topic: {topic}"
        f"{hint_section}\n"
        "Return JSON:\n"
        "{\n"
        '  "title": "<concise>",\n'
        '  "motivation": "<why this matters>",\n'
        '  "method": "<concrete technical approach>",\n'
        '  "hypothesis": "<testable prediction>",\n'
        '  "resources": "<datasets, compute, software>",\n'
        '  "expected_outcome": "<measurable results>",\n'
        '  "risk_assessment": "<main risks>",\n'
        '  "source_strategy": "E",\n'
        f'  "methodology_type": "<one of {", ".join(METHODOLOGY_TYPES)}>",\n'
        f'  "novelty_level": "<one of {", ".join(NOVELTY_LEVELS)}>"\n'
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


def _dict_to_idea(d: Dict[str, Any], provider: str, model: str,
                    topic: str) -> Optional[Idea]:
    if not d:
        return None
    if not all(str(d.get(k, "")).strip() for k in ("title", "method", "hypothesis")):
        return None
    method_type = d.get("methodology_type") or ""
    if method_type not in METHODOLOGY_TYPES:
        method_type = None
    novelty = d.get("novelty_level") or ""
    if novelty not in NOVELTY_LEVELS:
        novelty = None
    idea = Idea(
        title=str(d.get("title", ""))[:200],
        motivation=str(d.get("motivation", ""))[:1000],
        method=str(d.get("method", ""))[:2000],
        hypothesis=str(d.get("hypothesis", ""))[:1000],
        resources=str(d.get("resources", ""))[:500],
        expected_outcome=str(d.get("expected_outcome", ""))[:500],
        risk_assessment=str(d.get("risk_assessment", ""))[:500],
        source_strategy="E",  # E = Ensemble
        methodology_type=method_type,
        novelty_level=novelty,
        generation=0,
        parent_title=None,
    )
    idea.execution_meta = {
        "ensemble_provider": provider,
        "ensemble_model": model,
        "regen_mode": "multi_llm_ensemble",
        "topic": topic,
    }
    return idea


# ─────────────────────────────────────────────────────────────────────────────
# Diversity filter
# ─────────────────────────────────────────────────────────────────────────────

_WORD_RE = re.compile(r"[A-Za-z]{3,}")
_STOP = frozenset([
    "the", "and", "for", "with", "this", "that", "from", "are", "was",
    "have", "has", "will", "use", "uses", "using", "their", "they", "our",
    "ours", "but", "not", "all", "can", "could", "may", "into", "via",
    "more", "less", "than", "such", "also", "however", "thus", "while",
])


def _idea_tokens(idea: Any) -> set:
    """Bag of content words from title + method + hypothesis for Jaccard."""
    d = idea.to_dict() if hasattr(idea, "to_dict") else (
        idea if isinstance(idea, dict) else {}
    )
    text = " ".join([
        str(d.get("title", "")),
        str(d.get("method", "")),
        str(d.get("hypothesis", "")),
    ]).lower()
    toks = _WORD_RE.findall(text)
    return {t for t in toks if t not in _STOP}


def _jaccard(a: set, b: set) -> float:
    """Standard Jaccard similarity. Empty sets → 0.0 similarity."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _diversity_filter(
    ideas: List[Tuple[Idea, str]],
    similarity_threshold: float,
) -> Tuple[List[Tuple[Idea, str]], List[Dict[str, Any]]]:
    """Greedy keep-first filter: drop any idea whose Jaccard similarity to
    a kept idea exceeds the threshold. Order matters (earlier ideas have
    priority); the ensemble shuffles provider order so no provider is
    systematically advantaged.

    Returns (kept_list, rejected_pairs) where rejected_pairs contains
    {kept_title, rejected_title, similarity} entries.
    """
    kept: List[Tuple[Idea, str]] = []
    kept_tokens: List[set] = []
    rejected: List[Dict[str, Any]] = []
    for idea, provider in ideas:
        toks = _idea_tokens(idea)
        max_sim = 0.0
        max_against = None
        for k, k_toks in zip(kept, kept_tokens):
            s = _jaccard(toks, k_toks)
            if s > max_sim:
                max_sim = s
                max_against = k[0]
        if max_sim > similarity_threshold and max_against is not None:
            rejected.append({
                "kept_title": max_against.title,
                "rejected_title": idea.title,
                "rejected_provider": provider,
                "similarity": round(max_sim, 3),
            })
        else:
            kept.append((idea, provider))
            kept_tokens.append(toks)
    return kept, rejected


# ─────────────────────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EnsembleResult:
    topic: str = ""
    providers_used: List[str] = field(default_factory=list)
    kept_ideas: List[Idea] = field(default_factory=list)
    all_ideas: List[Idea] = field(default_factory=list)
    rejected_pairs: List[Dict[str, Any]] = field(default_factory=list)
    provider_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)
    elapsed_s: float = 0.0

    def summary(self) -> str:
        n_kept = len(self.kept_ideas)
        n_total = len(self.all_ideas)
        return (
            f"Ensemble: {n_kept} kept / {n_total} generated across "
            f"{len(self.providers_used)} providers · {self.elapsed_s:.1f}s"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def ensemble_generate(
    topic: str,
    providers: Optional[List[str]] = None,
    n_per_provider: int = 1,
    similarity_threshold: float = 0.60,
    hint: str = "",
    timeout_s: float = 60.0,
    max_workers: int = 6,
) -> EnsembleResult:
    """Generate ideas in parallel across multiple LLM providers, then keep
    only those that are pairwise diverse.

    Args:
        topic: research topic to generate on.
        providers: which providers to use. Defaults to all configured.
        n_per_provider: how many idea calls per provider (1 is fastest;
            2-3 increases coverage but multiplies cost).
        similarity_threshold: 0..1. Pairs with Jaccard above this are
            considered duplicates; only the first one is kept.
        hint: optional framing text appended to every prompt (e.g. "focus
            on clinical applications").
        timeout_s: wall-clock per-call timeout.
        max_workers: thread-pool size for parallel calls.

    Returns an `EnsembleResult` with kept_ideas, the full set, the
    rejected pairs, and per-provider stats.
    """
    if not topic or not topic.strip():
        raise ValueError("ensemble_generate requires a non-empty topic")
    if providers is None:
        providers = available_providers()
    if not providers:
        return EnsembleResult(topic=topic.strip())
    if n_per_provider <= 0:
        return EnsembleResult(topic=topic.strip(), providers_used=list(providers))

    # Build clients once per provider
    clients: Dict[str, _ProviderClient] = {}
    for p in providers:
        clients[p] = _ProviderClient(p, _default_model(p))

    # Schedule (provider, attempt_index) calls in parallel. Vary temperature
    # slightly per attempt so n>1 doesn't produce duplicates.
    tasks: List[Tuple[str, int]] = []
    for p in providers:
        for k in range(int(n_per_provider)):
            tasks.append((p, k))

    user_prompt = _user_prompt(topic.strip(), hint=hint)
    started = time.time()
    raw_outputs: List[Tuple[str, _ProviderResponse]] = []
    lock = threading.Lock()
    provider_stats: Dict[str, Dict[str, int]] = {
        p: {"ok": 0, "fail": 0, "parsed": 0} for p in providers
    }

    def _do(provider: str, k_idx: int) -> Tuple[str, _ProviderResponse]:
        cli = clients[provider]
        temp = 0.70 + 0.07 * k_idx  # bumps with each attempt
        return provider, cli.call(
            system=_SYSTEM,
            user=user_prompt,
            max_tokens=900,
            temperature=temp,
            json_mode=True,
        )

    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as pool:
        futures = [pool.submit(_do, p, k) for p, k in tasks]
        try:
            for fut in as_completed(futures, timeout=timeout_s):
                try:
                    p, r = fut.result(timeout=2.0)
                    with lock:
                        if r.success:
                            provider_stats[p]["ok"] += 1
                            raw_outputs.append((p, r))
                        else:
                            provider_stats[p]["fail"] += 1
                except Exception:
                    pass
        except FuturesTimeout:
            # Salvage whatever's done; let the rest die with the pool
            pass

    elapsed = time.time() - started

    # Parse responses into Idea objects
    all_ideas_with_prov: List[Tuple[Idea, str]] = []
    for provider, r in raw_outputs:
        parsed = _parse_json(r.text)
        if not parsed:
            continue
        idea = _dict_to_idea(parsed, provider, r.model or _default_model(provider),
                                topic.strip())
        if idea is not None:
            all_ideas_with_prov.append((idea, provider))
            provider_stats[provider]["parsed"] += 1

    # Diversity filter
    kept, rejected = _diversity_filter(all_ideas_with_prov, similarity_threshold)

    return EnsembleResult(
        topic=topic.strip(),
        providers_used=list(providers),
        kept_ideas=[i for i, _ in kept],
        all_ideas=[i for i, _ in all_ideas_with_prov],
        rejected_pairs=rejected,
        provider_stats=provider_stats,
        elapsed_s=elapsed,
    )
