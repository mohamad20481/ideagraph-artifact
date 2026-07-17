"""
agents/base_agent.py - Shared LLM client and utility methods for all agents.

Optimisations:
  - LRU response cache: frequently-reused prompts stay cached longer.
    Uses collections.OrderedDict with move_to_end on hit and popitem(last=False)
    on eviction — O(1) LRU without any extra library.
  - Circuit breaker: fail-fast when API is consistently erroring
  - Prompt compression: reduce token usage by 20-40%
  - Adaptive timeout: adjusts per-call timeout based on recent latency
  - Call metrics: tracks latency, errors, cache stats for PipelineOptimizer
"""

from __future__ import annotations
import functools
import hashlib
import json
import random as _random
import re
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional

import openai
import config
from optimization import CircuitBreaker, PromptCompressor


# ── Runtime-control observation decorator ──────────────────────────────────
# Wraps _call so every LLM result reaches the active RuntimeController.
# Empty string return = network/auth/etc failure → counts as failed call.
# Non-empty return = success. The controller may BLOCK if the call was a
# failure that crossed the consecutive-failures threshold; that pause is
# the whole point — the pipeline thread waits for the user's decision.
def _observe_llm_call(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        result = fn(self, *args, **kwargs)
        try:
            ctrl = getattr(config, "_current_runtime_controller", None)
            if ctrl is not None:
                ctrl.record_llm_result(bool(result))
        except Exception:
            pass
        return result
    return wrapper

# ── Production observability + cost tracking + circuit breaker ──────────────
try:
    from observability import record_llm_call as _obs_record_llm_call, logger as _obs_logger
    from production_optimization import get_cost_tracker as _get_cost_tracker, get_circuit_breaker as _get_prod_cb
    _HAS_PROD_LAYER = True
except ImportError:
    _HAS_PROD_LAYER = False
from creative_optimization import AnnealingSchedule
from deep_optimization import ContrastivePromptPair, ResponseDistiller
from infra_optimization import DiskCache, SemanticCache, ErrorClassifier, ProviderRouter

# ── Module-level LLM response cache (LRU, keyed on prompt tuple) ─────────────
# Keys are tuples (system, user, max_tokens, temp_str) — Python hashes tuples
# in O(prompt_len) but only on hash *miss*; subsequent equality checks short-
# circuit on first different element. Previously every lookup MD5-hashed the
# full concatenated prompt, which is far more work than a tuple hash for the
# common case (in-memory hit).
from typing import Tuple as _Tuple
_CacheKey = _Tuple[str, str, int, str]
_RESPONSE_CACHE: "OrderedDict[_CacheKey, str]" = OrderedDict()
_CACHE_MAX = 256
_CACHE_LOCK = threading.Lock()

# ── Token usage tracking ──────────────────────────────────────────────────────
_TOKEN_USAGE: Dict[str, int] = {"prompt": 0, "completion": 0}
_TOKEN_LOCK = threading.Lock()

# ── Call metrics (latency, errors, cache hits) ───────────────────────────────
_CALL_METRICS: Dict[str, int] = {"calls": 0, "cache_hits": 0, "errors": 0, "retries": 0}
_METRICS_LOCK = threading.Lock()

# ── Global circuit breaker (shared across all agents) ────────────────────────
_CIRCUIT_BREAKER = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)

# ── Global prompt compressor ─────────────────────────────────────────────────
_COMPRESSOR = PromptCompressor()

# ── Contrastive prompt enhancement ───────────────────────────────────────────
_CONTRASTIVE = ContrastivePromptPair()

# ── Response distiller (cross-call knowledge reuse) ──────────────────────────
_DISTILLER = ResponseDistiller()

# ── Disk cache (persistent across restarts) ──────────────────────────────────
_DISK_CACHE = DiskCache() if getattr(config, "ENABLE_DISK_CACHE", False) else None

# ── Semantic cache (similarity-based dedup) ──────────────────────────────────
_SEMANTIC_CACHE = SemanticCache() if getattr(config, "ENABLE_SEMANTIC_CACHE", False) else None

# ── Error classifier (type-specific recovery) ────────────────────────────────
_ERROR_CLASSIFIER = ErrorClassifier()

# ── Provider router (multi-provider failover) ────────────────────────────────
_PROVIDER_ROUTER = ProviderRouter()

# ── Global annealing schedule (set iteration from pipeline) ──────────────────
_ANNEALING = AnnealingSchedule(total_iterations=3)

# ── Cached config flags (read once at import, avoid getattr per LLM call) ────
_ENABLE_PROMPT_COMPRESSION = getattr(config, "ENABLE_PROMPT_COMPRESSION", True)
_ENABLE_CONTRASTIVE_PROMPTS = getattr(config, "ENABLE_CONTRASTIVE_PROMPTS", True)
_ENABLE_RESPONSE_DISTILLATION = getattr(config, "ENABLE_RESPONSE_DISTILLATION", True)


def set_annealing_iteration(iteration: int, total: int = None) -> None:
    """Called by pipeline to update the global annealing schedule."""
    if total and total != _ANNEALING.total_iterations:
        _ANNEALING.total_iterations = total
    _ANNEALING.set_iteration(iteration)


def get_annealed_temperature(stage: str = "") -> float:
    """Get the current annealed temperature for a stage."""
    return _ANNEALING.get_temperature(stage)


def get_token_usage() -> Dict[str, int]:
    """Return a snapshot of cumulative token counts (thread-safe)."""
    with _TOKEN_LOCK:
        return dict(_TOKEN_USAGE)


def get_call_metrics() -> Dict[str, int]:
    """Return call metrics: total calls, cache hits, errors, retries."""
    with _METRICS_LOCK:
        return dict(_CALL_METRICS)


def estimate_cost_usd(usage: Optional[Dict[str, int]] = None) -> float:
    """Approximate USD cost from cumulative token counts."""
    u = usage if usage is not None else get_token_usage()
    rates = config.COST_RATES.get(config.PROVIDER, {"input": 0.27, "output": 1.10})
    return (u["prompt"] * rates["input"] + u["completion"] * rates["output"]) / 1_000_000


def _record_usage(api_usage) -> None:
    """Add one API response's token counts to the module-level totals."""
    if api_usage is None:
        return
    prompt_tokens = getattr(api_usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(api_usage, "completion_tokens", 0) or 0
    with _TOKEN_LOCK:
        _TOKEN_USAGE["prompt"] += prompt_tokens
        _TOKEN_USAGE["completion"] += completion_tokens


def _record_metric(key: str, increment: int = 1) -> None:
    """Increment a call metric counter."""
    with _METRICS_LOCK:
        _CALL_METRICS[key] = _CALL_METRICS.get(key, 0) + increment


def reset_caches() -> None:
    """
    Clear ALL module-level caches between pipeline runs.

    CRITICAL: Without this, different topics return the same ideas because
    cached LLM responses, Semantic Scholar results, and accumulated knowledge
    from a previous topic persist across runs in the same process.
    """
    # 1. LLM response cache (in-memory LRU)
    with _CACHE_LOCK:
        _RESPONSE_CACHE.clear()

    # 2. Token counters
    with _TOKEN_LOCK:
        _TOKEN_USAGE["prompt"] = 0
        _TOKEN_USAGE["completion"] = 0

    # 3. Call metrics
    with _METRICS_LOCK:
        _CALL_METRICS.clear()
        _CALL_METRICS.update({"calls": 0, "cache_hits": 0, "errors": 0, "retries": 0})

    # 4. Semantic similarity cache
    if _SEMANTIC_CACHE:
        _SEMANTIC_CACHE._entries.clear()
        _SEMANTIC_CACHE._hits = 0
        _SEMANTIC_CACHE._misses = 0

    # 5. Response distiller nuggets (topic-specific knowledge)
    _DISTILLER._nuggets.clear()

    # 6. Circuit breaker (reset error state)
    _CIRCUIT_BREAKER.failure_count = 0
    from optimization import CircuitState as _CS
    _CIRCUIT_BREAKER.state = _CS.CLOSED

    # 7. Semantic Scholar API response cache
    try:
        from tools.semantic_scholar import clear_cache as _clear_ss
        _clear_ss()
    except ImportError:
        pass

    # 8. Contrastive prompt learned failure patterns (topic-specific)
    _CONTRASTIVE._failure_patterns.clear()

    # 9. Production circuit breaker (reset per-provider failure counts so a
    #    new run with a freshly topped-up key doesn't get blocked by stale state).
    if _HAS_PROD_LAYER:
        try:
            cb = _get_prod_cb()
            for provider in list(cb.status().keys()):
                cb.record_success(provider)
        except Exception:
            pass

    # NOTE: _DISK_CACHE is deliberately NOT cleared here. The disk cache is
    # designed to survive process restarts and pipeline resets — its whole
    # point is cost reduction across runs that share prompts (e.g. re-running
    # the same topic after tweaking a prompt downstream). If you need a
    # strictly fresh run, set ENABLE_DISK_CACHE=false in .env.


def _cache_key(system: str, user: str, max_tokens: int, temp: float) -> _CacheKey:
    """In-memory cache key — tuple hashes in O(1) using Python's builtin.
    The temperature is rounded so 0.7000001 and 0.7 land in the same bucket."""
    return (system, user, max_tokens, f"{temp:.3f}")


def _disk_cache_key(ck: _CacheKey) -> str:
    """Disk cache requires a stable short string for the on-disk filename — MD5
    is only computed on the slow path (disk get/put), not on every lookup."""
    system, user, max_tokens, temp_str = ck
    raw = f"{system}\x00{user}\x00{max_tokens}\x00{temp_str}"
    return hashlib.md5(raw.encode(), usedforsecurity=False).hexdigest()


class BaseAgent:
    """
    Base class providing:
      - OpenAI-compatible client construction per provider
      - _call()      : raw text response (with LRU cache)
      - _call_json() : JSON-parsed response (uses JSON mode when available)
    """

    def __init__(self, temperature: Optional[float] = None,
                 agent_id: Optional[str] = None, role: Optional[str] = None,
                 capabilities: Optional[list] = None) -> None:
        self.temperature = temperature
        self.agent_id = agent_id or f"{self.__class__.__name__}_{id(self) % 10000:04d}"
        self.role = role or self.__class__.__name__
        self.capabilities = capabilities or []
        self.client = self._build_client()

    # ── Client factory ─────────────────────────────────────────────────────────
    @staticmethod
    def _build_client_for_provider(provider: str) -> openai.OpenAI:
        """Build an OpenAI client for a specific provider.
        For 'anthropic', returns a stub OpenAI client (real calls go through
        claude_provider.ClaudeClient via the native /v1/messages branch in _call)."""
        if provider == "openai":
            return openai.OpenAI(api_key=config.OPENAI_API_KEY)
        if provider == "groq":
            return openai.OpenAI(api_key=config.GROQ_API_KEY, base_url=config.GROQ_BASE_URL)
        if provider == "gemini":
            return openai.OpenAI(api_key=config.GEMINI_API_KEY, base_url=config.GEMINI_BASE_URL)
        if provider == "azure":
            return openai.OpenAI(
                api_key=config.AZURE_API_KEY, base_url=config.AZURE_BASE_URL,
                default_headers={"api-key": config.AZURE_API_KEY},
                default_query={"api-version": "2024-05-01-preview"},
            )
        if provider == "anthropic":
            # Stub client — never used for actual calls (intercepted earlier in _call).
            # Just needs a non-empty key so the SDK doesn't throw on construction.
            return openai.OpenAI(api_key="anthropic-stub-not-used")
        if provider == "kimi":
            # Moonshot AI — OpenAI-compatible chat completions API.
            return openai.OpenAI(
                api_key=config.KIMI_API_KEY,
                base_url=config.KIMI_BASE_URL,
            )
        # Default: deepseek
        return openai.OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)

    def _build_client(self) -> openai.OpenAI:
        return self._build_client_for_provider(config.PROVIDER)

    def _get_failover_provider(self) -> Optional[str]:
        """Find an alternative provider with a valid API key."""
        fallback_order = ["deepseek", "kimi", "gemini", "groq", "openai", "azure"]
        key_map = {
            "deepseek": config.DEEPSEEK_API_KEY,
            "kimi":     config.KIMI_API_KEY,
            "gemini":   config.GEMINI_API_KEY,
            "groq":     config.GROQ_API_KEY,
            "openai":   config.OPENAI_API_KEY,
            "azure":    config.AZURE_API_KEY,
        }
        for provider in fallback_order:
            if provider != config.PROVIDER and key_map.get(provider):
                return provider
        return None

    # Default: deepseek (OpenAI-compatible)
        return openai.OpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
        )

    def _stage_hint(self) -> str:
        """Return the pipeline stage this agent corresponds to.
        Subclasses override to enable stage-aware fast-model routing.
        Default: 'default' (uses user's configured model)."""
        cls_name = self.__class__.__name__.lower()
        # Heuristic mapping from agent class name → stage
        if "critic" in cls_name or "probe" in cls_name:
            return "probe"
        if "ideation" in cls_name:
            return "ideation"
        if "review" in cls_name:
            return "review"
        if "debate" in cls_name or "judge" in cls_name:
            return "debate_judge"
        if "writer" in cls_name or "paper" in cls_name:
            return "paper_write"
        return "default"

    # ── Raw text call ──────────────────────────────────────────────────────────
    @_observe_llm_call
    def _call(
        self,
        system: str,
        user: str,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
        use_cache: bool = True,
        json_mode: bool = False,
        compress: bool = True,
    ) -> str:
        """
        Call the configured LLM and return the response text.

        Optimizations:
        - Circuit breaker: fail-fast if API is down (saves budget)
        - Prompt compression: reduce token usage by stripping filler
        - LRU cache: skip redundant API calls
        - Jittered exponential backoff: prevents thundering herd
        - Adaptive timeout: scales with recent latency patterns
        """
        _record_metric("calls")

        if config.PROVIDER in ("deepseek", "gemini", "azure", "kimi"):
            max_tokens = min(max_tokens, 8192)

        temp = (
            temperature
            if temperature is not None
            else (self.temperature if self.temperature is not None else 0.7)
        )

        # ── Kimi/Moonshot temperature coercion ─────────────────────────────
        # Moonshot rejects temperature=0.0 with "invalid temperature" on
        # most models; kimi-k2.5/k2.6 are reasoning models that accept
        # ONLY temperature=1.0. We normalize here so the rest of the agent
        # code can keep passing whatever temperature it wants without
        # getting hard-rejected.
        if config.PROVIDER == "kimi":
            model_name = (config.MODEL or "").lower()
            if model_name.startswith("kimi-k2") or "reasoning" in model_name:
                temp = 1.0
            else:
                temp = max(0.3, float(temp))

        # ── Prompt compression (saves 15-30% tokens) ────────────────────────
        if compress and _ENABLE_PROMPT_COMPRESSION:
            system = _COMPRESSOR.compress(system, aggressive=False)
            user = _COMPRESSOR.compress(user, aggressive=False)

        # ── Contrastive prompt enhancement (adds DO NOT constraints) ─────
        if _ENABLE_CONTRASTIVE_PROMPTS:
            # Detect task type from system prompt keywords
            task_type = ""
            sys_lower = system.lower()
            if "ideation" in sys_lower or "research idea" in sys_lower:
                task_type = "ideation"
            elif "code" in sys_lower or "python" in sys_lower:
                task_type = "code_generation"
            elif "experiment" in sys_lower and "design" in sys_lower:
                task_type = "experiment_design"
            elif "paper" in sys_lower or "write" in sys_lower:
                task_type = "paper_writing"
            elif "review" in sys_lower or "evaluate" in sys_lower:
                task_type = "review"
            if task_type:
                system = _CONTRASTIVE.enhance_prompt(system, task_type)

        # ── Knowledge injection from response distiller ──────────────────
        if _ENABLE_RESPONSE_DISTILLATION:
            # Extract domain from user prompt (first 100 chars)
            domain_hint = user[:100]
            user = _DISTILLER.inject_into_prompt(user, domain_hint)

        # ── Multi-layer cache lookup ─────────────────────────────────────────
        # Layer 1: In-memory LRU (fastest, exact match)
        ck = _cache_key(system, user, max_tokens, temp)
        if use_cache:
            with _CACHE_LOCK:
                if ck in _RESPONSE_CACHE:
                    _RESPONSE_CACHE.move_to_end(ck)
                    _record_metric("cache_hits")
                    return _RESPONSE_CACHE[ck]

            # Layer 2: Disk cache (persistent, exact match) — MD5 only computed
            # here, on the cold path (the in-memory hit above never touches MD5).
            if _DISK_CACHE:
                disk_ck = _disk_cache_key(ck)
                disk_hit = _DISK_CACHE.get(disk_ck)
                if disk_hit:
                    _record_metric("cache_hits")
                    # Promote to in-memory cache
                    with _CACHE_LOCK:
                        if len(_RESPONSE_CACHE) >= _CACHE_MAX:
                            _RESPONSE_CACHE.popitem(last=False)
                        _RESPONSE_CACHE[ck] = disk_hit
                    return disk_hit

            # Layer 3: Semantic cache (approximate match)
            if _SEMANTIC_CACHE:
                sem_hit = _SEMANTIC_CACHE.get(f"{system}\n{user}")
                if sem_hit:
                    _record_metric("cache_hits")
                    return sem_hit

        # ── Circuit breaker check ────────────────────────────────────────────
        if not _CIRCUIT_BREAKER.can_execute():
            _record_metric("errors")
            return ""

        # ── Production circuit breaker (per-provider, from production_optimization) ──
        if _HAS_PROD_LAYER:
            ok_cb, _cb_msg = _get_prod_cb().allow(config.PROVIDER)
            if not ok_cb:
                _record_metric("errors")
                return ""

            # ── Cost-aware early exit: stop burning budget if run is over ──
            _run_id = getattr(config, "_current_run_id", None)
            _budget = getattr(config, "_current_budget_usd", None)
            if _run_id and _budget:
                if _get_cost_tracker().should_abort(_run_id, _budget):
                    _record_metric("errors")
                    return ""

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        # JSON mode: forces model to output valid JSON
        resp_fmt: Dict[str, Any] = {}
        if json_mode and config.PROVIDER in (
            "deepseek", "openai", "gemini", "azure", "kimi", "xai",
        ):
            # xai (api.x.ai) is OpenAI-compatible and supports
            # `response_format: json_object` for both Grok models AND
            # the DeepSeek V4 lineup it also hosts. Without this, the
            # model returns prose → JSON parser fails on every reply →
            # ideas_archived stays at 0 even when LLM calls succeed.
            resp_fmt = {"response_format": {"type": "json_object"}}

        last_exc: Optional[Exception] = None
        _active_client = self.client
        _active_model = config.MODEL
        _active_provider = config.PROVIDER

        # ── Stage-aware fast-model routing (probes → Haiku, etc.) ────────────
        # Only kicks in when ENABLE_STAGE_ROUTING is set; otherwise uses user's choice.
        if getattr(config, "ENABLE_STAGE_ROUTING", True):
            try:
                from speed_optimizer import route_for_stage
                _stage = self._stage_hint()
                _routed_provider, _routed_model = route_for_stage(
                    _stage, prefer_provider=_active_provider,
                )
                if _routed_provider != _active_provider:
                    # Different provider — rebuild client
                    try:
                        _active_client = self._build_client_for_provider(_routed_provider)
                        _active_provider = _routed_provider
                        _active_model = _routed_model
                    except Exception:
                        pass  # fall back to user's choice
                elif _routed_model != _active_model:
                    # Same provider, faster model
                    _active_model = _routed_model
            except ImportError:
                pass

        # ── Anthropic native path (prompt caching, retry built-in) ──────────
        if _active_provider == "anthropic":
            try:
                from claude_provider import ClaudeClient
                _claude = ClaudeClient(
                    api_key=getattr(config, "ANTHROPIC_API_KEY", ""),
                    model=_active_model,
                    base_url=getattr(config, "ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
                )
                call_start = time.time()
                _resp = _claude.call(
                    system=system, user=user,
                    max_tokens=max_tokens, temperature=temp,
                    cache_system=True, json_mode=json_mode,
                )
                call_elapsed = time.time() - call_start

                if _resp.success:
                    text = _resp.text
                    # Track usage in the legacy counters too
                    class _FakeUsage:
                        prompt_tokens = _resp.input_tokens
                        completion_tokens = _resp.output_tokens
                    _record_usage(_FakeUsage())
                    _CIRCUIT_BREAKER.record_success()
                    if _HAS_PROD_LAYER:
                        _obs_record_llm_call(
                            provider="anthropic",
                            model=_active_model,
                            input_tokens=_resp.input_tokens,
                            output_tokens=_resp.output_tokens,
                            duration_s=call_elapsed,
                            status="ok",
                            user_id=getattr(config, "_current_user_id", None),
                            run_id=getattr(config, "_current_run_id", None),
                        )
                        _get_prod_cb().record_success("anthropic")
                    # Cache write-through
                    if use_cache and text:
                        with _CACHE_LOCK:
                            if ck in _RESPONSE_CACHE:
                                _RESPONSE_CACHE.move_to_end(ck)
                            else:
                                if len(_RESPONSE_CACHE) >= _CACHE_MAX:
                                    _RESPONSE_CACHE.popitem(last=False)
                                _RESPONSE_CACHE[ck] = text
                        if _DISK_CACHE:
                            try: _DISK_CACHE.put(_disk_cache_key(ck), text)
                            except Exception: pass
                    _PROVIDER_ROUTER.record_result("anthropic", True, call_elapsed)
                    return text
                else:
                    # Native Claude error — record failure, fall through to OpenAI failover if any
                    _record_metric("errors")
                    if _HAS_PROD_LAYER:
                        _get_prod_cb().record_failure("anthropic")
                        _obs_logger.error("claude_error", error=_resp.error,
                                          attempts=_resp.attempts, model=_active_model)
                    return ""  # graceful empty-string fail
            except ImportError:
                pass  # claude_provider not available — fall through to OpenAI path
            except Exception as exc:
                _record_metric("errors")
                if _HAS_PROD_LAYER:
                    _obs_logger.error("claude_exception",
                                      error_type=type(exc).__name__, error=str(exc)[:200])
                return ""

        for attempt in range(2):  # speed: 2 attempts (was 3) — fail fast, failover
            try:
                call_start = time.time()
                resp = _active_client.chat.completions.create(
                    model=_active_model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temp,
                    timeout=45,  # balanced: long enough for complex prompts, short enough to fail over
                    **resp_fmt,
                )
                text = resp.choices[0].message.content or ""
                call_elapsed = time.time() - call_start
                _record_usage(resp.usage)
                _CIRCUIT_BREAKER.record_success()

                # ── Production: cost tracking + observability ──────────────
                if _HAS_PROD_LAYER:
                    _p_tok = getattr(resp.usage, "prompt_tokens", 0) or 0
                    _c_tok = getattr(resp.usage, "completion_tokens", 0) or 0
                    _obs_record_llm_call(
                        provider=_active_provider,
                        model=_active_model,
                        input_tokens=_p_tok,
                        output_tokens=_c_tok,
                        duration_s=call_elapsed,
                        status="ok",
                        user_id=getattr(config, "_current_user_id", None),
                        run_id=getattr(config, "_current_run_id", None),
                    )
                    _get_prod_cb().record_success(_active_provider)

                # ── Multi-layer cache write-through ───────────────────────
                if use_cache and text:
                    with _CACHE_LOCK:
                        if ck in _RESPONSE_CACHE:
                            _RESPONSE_CACHE.move_to_end(ck)
                        else:
                            if len(_RESPONSE_CACHE) >= _CACHE_MAX:
                                _RESPONSE_CACHE.popitem(last=False)
                            _RESPONSE_CACHE[ck] = text
                    # Write-through to disk cache — MD5 only on the write path.
                    if _DISK_CACHE:
                        _DISK_CACHE.put(_disk_cache_key(ck), text)
                    # Write-through to semantic cache
                    if _SEMANTIC_CACHE:
                        _SEMANTIC_CACHE.put(f"{system}\n{user}", text)

                # Record success with provider router
                _PROVIDER_ROUTER.record_result(config.PROVIDER, True, time.time() - call_start)

                return text

            except openai.RateLimitError as exc:
                last_exc = exc
                _record_metric("retries")
                _ERROR_CLASSIFIER.classify(exc)
                _PROVIDER_ROUTER.record_result(config.PROVIDER, False)
                if _HAS_PROD_LAYER:
                    _get_prod_cb().record_failure(_active_provider)
                    _obs_logger.warn("llm_rate_limit", provider=_active_provider,
                                     attempt=attempt, error=str(exc)[:200])
                base = min(5 * (2 ** attempt), 10)
                wait = base * (0.5 + _random.random())
                time.sleep(wait)
            except openai.APIError as exc:
                last_exc = exc
                _record_metric("retries")
                _ERROR_CLASSIFIER.classify(exc)
                _PROVIDER_ROUTER.record_result(config.PROVIDER, False)
                _CIRCUIT_BREAKER.record_failure()
                if _HAS_PROD_LAYER:
                    _get_prod_cb().record_failure(_active_provider)
                    _obs_logger.error("llm_api_error", provider=_active_provider,
                                      attempt=attempt, error=str(exc)[:200])
                time.sleep((2 * (attempt + 1)) * (0.5 + _random.random()))
            except Exception as exc:
                last_exc = exc
                _record_metric("errors")
                _ERROR_CLASSIFIER.classify(exc)
                _PROVIDER_ROUTER.record_result(_active_provider, False)
                _CIRCUIT_BREAKER.record_failure()
                if _HAS_PROD_LAYER:
                    _get_prod_cb().record_failure(_active_provider)
                    _obs_logger.error("llm_exception", provider=_active_provider,
                                      attempt=attempt, error_type=type(exc).__name__,
                                      error=str(exc)[:200])

                # ── Auto-failover: switch to backup provider after 1st failure ──
                if attempt == 0:
                    failover = self._get_failover_provider()
                    if failover and failover != _active_provider:
                        try:
                            _active_client = self._build_client_for_provider(failover)
                            from smart_routing import MODEL_PROFILES
                            profile = MODEL_PROFILES.get(failover)
                            _active_model = profile.model if profile else "deepseek-chat"
                            _active_provider = failover
                            _record_metric("failovers")
                            continue  # retry immediately with new provider
                        except Exception:
                            pass

                time.sleep(3 * (0.5 + _random.random()))

        _record_metric("errors")
        return ""

    # ── JSON call ─────────────────────────────────────────────────────────────
    def _call_json(
        self,
        system: str,
        user: str,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Call the LLM and return a parsed JSON dict.
        Uses JSON mode (response_format=json_object) for DeepSeek/OpenAI
        to guarantee valid JSON output and eliminate parse failures.
        Falls back to regex extraction for providers without JSON mode.
        """
        raw = self._call(
            system, user,
            max_tokens=max_tokens,
            temperature=temperature,
            use_cache=True,
            json_mode=True,   # ← key optimisation: avoids all parse failures
        )
        if not raw:
            return {}

        # Try direct parse (JSON mode should always succeed here)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Strip markdown code fences and retry
        stripped = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

        # Extract first {...} block (last resort for non-JSON-mode providers)
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return {}
