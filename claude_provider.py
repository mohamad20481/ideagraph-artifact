"""
claude_provider.py - Native Anthropic Claude API integration for IdeaGraph.

Patterns ported from ChatApp's ClaudeArchitectService:

  * Native /v1/messages endpoint (NOT OpenAI-compatible proxy)
  * Bearer auth + anthropic-version header
  * Prompt caching via cache_control:ephemeral (~90% input-token discount)
  * Retry policy: 3 attempts, exponential backoff (1s/2s) on transient errors
  * Atomic budget reservation with refund on failure
  * 180s timeout (Opus can take >60s)
  * Token usage tracking (input, output, cache_read, cache_write)

Models supported:
  - claude-opus-4-7        (premium, $15/$75 per M tokens)
  - claude-sonnet-4-6      (balanced, $3/$15)
  - claude-haiku-4-5       (fast/cheap, $1/$5)

Why native instead of OpenAI proxy:
  - Prompt caching gives huge cost savings on repeated system prompts
  - Tool use / extended thinking only available via native API
  - Better error messages and rate-limit headers
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests


ANTHROPIC_VERSION = "2023-06-01"
# aiprimetech.io is the proxy used by ChatApp (PhD thesis project).
# It accepts the same Anthropic API contract — Bearer auth + /v1/messages.
DEFAULT_BASE_URL = "https://aiprimetech.io"
PROMPT_CACHING_BETA = "prompt-caching-2024-07-31"

# Statuses we retry on (matches ChatApp's TransientStatuses set).
TRANSIENT_STATUSES = {408, 425, 429, 500, 502, 503, 504}

# Per-million-token pricing (USD) — kept in sync with config.COST_RATES.
CLAUDE_PRICING = {
    "claude-opus-4-7":   {"input": 15.0, "output": 75.0, "cache_read": 1.50, "cache_write": 18.75},
    "claude-sonnet-4-6": {"input":  3.0, "output": 15.0, "cache_read": 0.30, "cache_write":  3.75},
    "claude-haiku-4-5":  {"input":  1.0, "output":  5.0, "cache_read": 0.10, "cache_write":  1.25},
}

AVAILABLE_MODELS = list(CLAUDE_PRICING.keys())


@dataclass
class ClaudeResponse:
    success: bool
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""
    error: Optional[str] = None
    attempts: int = 0


class ClaudeClient:
    """
    Thread-safe Claude API client with budget tracking + retry + prompt caching.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-6",
        base_url: str = DEFAULT_BASE_URL,
        timeout_s: float = 180.0,
        call_budget: int = 100,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.call_budget = call_budget
        self._calls_made = 0
        self._lock = threading.Lock()

        # Reuse one Session for connection pooling
        self._session = requests.Session()

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key) and not self.api_key.startswith("sk-xxx")

    @property
    def remaining_budget(self) -> int:
        with self._lock:
            return max(0, self.call_budget - self._calls_made)

    def reset_budget(self) -> None:
        with self._lock:
            self._calls_made = 0

    def _reserve_slot(self) -> bool:
        """Atomically claim a budget slot. Returns False if exhausted."""
        with self._lock:
            if self._calls_made >= self.call_budget:
                return False
            self._calls_made += 1
            return True

    def _refund_slot(self) -> None:
        with self._lock:
            self._calls_made = max(0, self._calls_made - 1)

    def call(
        self,
        system: str,
        user: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        model_override: Optional[str] = None,
        cache_system: bool = True,
        json_mode: bool = False,
    ) -> ClaudeResponse:
        """
        Make a single Claude API call with retry + caching.

        Args:
            system: System prompt (cached if cache_system=True).
            user: User message content.
            max_tokens: Max output tokens.
            temperature: Sampling temperature [0.0, 1.0].
            model_override: Use a specific model for this call (must be in AVAILABLE_MODELS).
            cache_system: If True, mark system prompt for ephemeral caching.
            json_mode: If True, append JSON-only instruction to user prompt.
        """
        if not self.is_configured:
            return ClaudeResponse(
                success=False, text="",
                error="ANTHROPIC_API_KEY not configured", model=self.model,
            )

        # Validate model
        effective_model = self.model
        if model_override and model_override in AVAILABLE_MODELS:
            effective_model = model_override

        # Reserve budget slot
        if not self._reserve_slot():
            return ClaudeResponse(
                success=False, text="",
                error=f"Budget exhausted ({self.call_budget} calls). Reset to continue.",
                model=effective_model,
            )

        # Build request body
        if json_mode:
            user = f"{user}\n\nReturn ONLY valid JSON. No prose, no markdown fences."

        # System prompt as content-block array with cache_control for ~90% discount
        if cache_system:
            system_blocks = [{
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }]
        else:
            system_blocks = [{"type": "text", "text": system}]

        body = {
            "model": effective_model,
            "max_tokens": max_tokens,
            "temperature": max(0.0, min(1.0, temperature)),
            "system": system_blocks,
            "messages": [{"role": "user", "content": user}],
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "anthropic-version": ANTHROPIC_VERSION,
            "anthropic-beta": PROMPT_CACHING_BETA,
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/v1/messages"

        # Retry with exponential backoff
        max_attempts = 3
        last_error: Optional[str] = None
        last_status = 0

        for attempt in range(1, max_attempts + 1):
            try:
                resp = self._session.post(
                    url, headers=headers, json=body, timeout=self.timeout_s,
                )
                last_status = resp.status_code

                if resp.status_code == 200:
                    data = resp.json()
                    usage = data.get("usage", {})
                    in_tok = int(usage.get("input_tokens", 0))
                    out_tok = int(usage.get("output_tokens", 0))
                    cache_r = int(usage.get("cache_read_input_tokens", 0))
                    cache_w = int(usage.get("cache_creation_input_tokens", 0))

                    # Extract text from content array
                    content = data.get("content", [])
                    text = ""
                    if content and isinstance(content, list):
                        for block in content:
                            if block.get("type") == "text":
                                text += block.get("text", "")

                    cost = self._compute_cost(
                        effective_model, in_tok, out_tok, cache_r, cache_w,
                    )

                    return ClaudeResponse(
                        success=True, text=text,
                        input_tokens=in_tok, output_tokens=out_tok,
                        cache_read_tokens=cache_r, cache_write_tokens=cache_w,
                        cost_usd=cost, model=effective_model,
                        attempts=attempt,
                    )

                # Non-2xx response
                last_error = (resp.text or "")[:200]

                # Don't retry 4xx (auth/validation errors)
                if last_status not in TRANSIENT_STATUSES:
                    self._refund_slot()
                    return ClaudeResponse(
                        success=False, text="",
                        error=f"API {last_status}: {last_error}",
                        model=effective_model, attempts=attempt,
                    )

            except requests.Timeout:
                last_error = "Request timeout"
                last_status = 504
            except requests.RequestException as e:
                last_error = str(e)[:200]
                last_status = 0

            # Backoff before next attempt (skip after last)
            if attempt < max_attempts:
                delay = 2 ** (attempt - 1)  # 1s, 2s
                time.sleep(delay)

        # All attempts exhausted on transient errors
        self._refund_slot()
        return ClaudeResponse(
            success=False, text="",
            error=f"All {max_attempts} attempts failed. Last: {last_error}",
            model=effective_model, attempts=max_attempts,
        )

    @staticmethod
    def _compute_cost(
        model: str, input_tok: int, output_tok: int,
        cache_read_tok: int = 0, cache_write_tok: int = 0,
    ) -> float:
        """Compute USD cost based on per-token Claude pricing."""
        rates = CLAUDE_PRICING.get(model, CLAUDE_PRICING["claude-sonnet-4-6"])
        # Note: input_tokens excludes cache reads/writes per Anthropic API contract
        cost = (
            input_tok * rates["input"]
            + output_tok * rates["output"]
            + cache_read_tok * rates["cache_read"]
            + cache_write_tok * rates["cache_write"]
        ) / 1_000_000.0
        return round(cost, 6)


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI-compatible client (DeepSeek / Kimi / OpenAI / Groq / Gemini / Azure)
# ─────────────────────────────────────────────────────────────────────────────
#
# Every non-Anthropic provider supported by IdeaGraph speaks the OpenAI
# chat-completions wire format: POST {base_url}/chat/completions with
# {"model","messages","max_tokens","temperature",...}. This class wraps
# that format and presents the SAME interface as ClaudeClient.call() —
# returning a ClaudeResponse — so the rest of the codebase (regenerator,
# Novelty Lab modules, etc.) doesn't have to know which backend it got.

class OpenAICompatClient:
    """Provider-agnostic OpenAI-compatible chat client.

    Same .call(system, user, max_tokens, temperature, json_mode) signature
    as ClaudeClient — returns a ClaudeResponse so callers are unchanged.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        provider_name: str = "openai",
        timeout_s: float = 180.0,
        call_budget: int = 100,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.provider_name = provider_name
        self.timeout_s = timeout_s
        self.call_budget = call_budget
        self._calls_made = 0
        self._lock = threading.Lock()
        self._session = requests.Session()

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key) and not self.api_key.startswith("sk-xxx")

    @property
    def remaining_budget(self) -> int:
        with self._lock:
            return max(0, self.call_budget - self._calls_made)

    def reset_budget(self) -> None:
        with self._lock:
            self._calls_made = 0

    def _reserve_slot(self) -> bool:
        with self._lock:
            if self._calls_made >= self.call_budget:
                return False
            self._calls_made += 1
            return True

    def _refund_slot(self) -> None:
        with self._lock:
            self._calls_made = max(0, self._calls_made - 1)

    def _coerce_temperature(self, t: float) -> float:
        """Provider-specific temperature quirks.

        Kimi's kimi-k2 reasoning models REQUIRE temperature=1.0 — they
        reject other values with a 400. This mirrors base_agent's coercion.
        """
        t = max(0.0, min(2.0, float(t)))
        if (self.provider_name == "kimi"
                and (self.model or "").lower().startswith("kimi-k2")):
            return 1.0
        return t

    def call(
        self,
        system: str,
        user: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        model_override: Optional[str] = None,
        cache_system: bool = True,  # unused — OpenAI-compat has no prompt caching
        json_mode: bool = False,
    ) -> ClaudeResponse:
        if not self.is_configured:
            return ClaudeResponse(
                success=False, text="",
                error=f"{self.provider_name.upper()}_API_KEY not configured",
                model=self.model,
            )

        effective_model = (model_override or self.model).strip()
        if not effective_model:
            return ClaudeResponse(
                success=False, text="",
                error=f"No model configured for provider {self.provider_name}",
                model="",
            )

        if not self._reserve_slot():
            return ClaudeResponse(
                success=False, text="",
                error=f"Budget exhausted ({self.call_budget} calls). Reset to continue.",
                model=effective_model,
            )

        body: Dict[str, Any] = {
            "model": effective_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": self._coerce_temperature(temperature),
        }
        if json_mode:
            # Both DeepSeek + Kimi + OpenAI + Groq support this; Gemini's
            # OpenAI-compat shim does too. Safe to send unconditionally.
            body["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/chat/completions"

        max_attempts = 3
        last_error: Optional[str] = None
        last_status = 0

        for attempt in range(1, max_attempts + 1):
            try:
                resp = self._session.post(
                    url, headers=headers, json=body, timeout=self.timeout_s,
                )
                last_status = resp.status_code

                if resp.status_code == 200:
                    data = resp.json()
                    choices = data.get("choices") or []
                    text = ""
                    if choices:
                        msg = (choices[0] or {}).get("message") or {}
                        text = msg.get("content") or ""
                    usage = data.get("usage") or {}
                    in_tok = int(usage.get("prompt_tokens", 0))
                    out_tok = int(usage.get("completion_tokens", 0))
                    cost = self._compute_cost(in_tok, out_tok)
                    return ClaudeResponse(
                        success=True, text=text,
                        input_tokens=in_tok, output_tokens=out_tok,
                        cost_usd=cost, model=effective_model,
                        attempts=attempt,
                    )

                # Surface the provider's error verbatim so the regenerator's
                # _classify_api_error can recognise INSUFFICIENT_BALANCE etc.
                last_error = (resp.text or "")[:300]

                if last_status not in TRANSIENT_STATUSES:
                    self._refund_slot()
                    return ClaudeResponse(
                        success=False, text="",
                        error=f"API {last_status}: {last_error}",
                        model=effective_model, attempts=attempt,
                    )

            except requests.Timeout:
                last_error = "Request timeout"
                last_status = 504
            except requests.RequestException as e:
                last_error = str(e)[:200]
                last_status = 0

            if attempt < max_attempts:
                time.sleep(2 ** (attempt - 1))

        self._refund_slot()
        return ClaudeResponse(
            success=False, text="",
            error=f"All {max_attempts} attempts failed. Last: {last_error}",
            model=effective_model, attempts=max_attempts,
        )

    def _compute_cost(self, input_tok: int, output_tok: int) -> float:
        """Cost via config.COST_RATES — same rates the rest of the app uses."""
        try:
            import config
            rates = (config.COST_RATES.get(self.provider_name)
                       or {"input": 0.0, "output": 0.0})
        except Exception:
            rates = {"input": 0.0, "output": 0.0}
        return round(
            (input_tok * rates["input"] + output_tok * rates["output"])
            / 1_000_000.0,
            6,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Per-provider config table (api_key, base_url, default_model)
# ─────────────────────────────────────────────────────────────────────────────

def _provider_credentials(provider: str) -> Tuple[str, str, str]:
    """Return (api_key, base_url, default_model) for one of the supported
    OpenAI-compatible providers. Reads from os.environ first, then config."""
    import os
    try:
        import config as cfg
    except Exception:
        cfg = None  # type: ignore[assignment]

    def _ck(name: str) -> str:
        v = os.getenv(name, "")
        if v:
            return v
        return (getattr(cfg, name, "") or "") if cfg else ""

    if provider == "deepseek":
        base = (cfg and getattr(cfg, "DEEPSEEK_BASE_URL", "")) or "https://api.deepseek.com"
        return _ck("DEEPSEEK_API_KEY"), base, "deepseek-chat"
    if provider == "kimi":
        base = (cfg and getattr(cfg, "KIMI_BASE_URL", "")) or "https://api.moonshot.ai/v1"
        return (_ck("KIMI_API_KEY") or _ck("MOONSHOT_API_KEY")), base, "moonshot-v1-32k"
    if provider == "openai":
        return _ck("OPENAI_API_KEY"), "https://api.openai.com/v1", "gpt-4o"
    if provider == "groq":
        base = (cfg and getattr(cfg, "GROQ_BASE_URL", "")) or "https://api.groq.com/openai/v1"
        return _ck("GROQ_API_KEY"), base, "llama-3.3-70b-versatile"
    if provider == "xai":
        # xAI Grok (Elon's Grok) — OpenAI-compatible at api.x.ai. Auth
        # key starts with `xai-…`. Distinct from `groq` above (Groq is
        # fast Llama inference at api.groq.com). Key resolution prefers
        # GROK_API_KEY then XAI_API_KEY for backward compat.
        base = (cfg and getattr(cfg, "XAI_BASE_URL", "")) or "https://api.x.ai/v1"
        key = _ck("GROK_API_KEY") or _ck("XAI_API_KEY")
        return key, base, "grok-2-latest"
    if provider == "gemini":
        base = ((cfg and getattr(cfg, "GEMINI_BASE_URL", ""))
                or "https://generativelanguage.googleapis.com/v1beta/openai/")
        return _ck("GEMINI_API_KEY"), base, "gemini-2.0-flash"
    if provider == "azure":
        base = _ck("AZURE_BASE_URL")
        return _ck("AZURE_API_KEY"), base, "DeepSeek-V3.2-Speciale"
    return "", "", ""


# ─────────────────────────────────────────────────────────────────────────────
# Singleton accessor — now provider-aware
# ─────────────────────────────────────────────────────────────────────────────

_GLOBAL_CLIENT: Optional[Any] = None  # ClaudeClient OR OpenAICompatClient
_GLOBAL_LOCK = threading.Lock()


def get_claude_client(reload: bool = False) -> Optional[Any]:
    """Get the configured global LLM client (singleton).

    DESPITE THE HISTORICAL NAME, this is the provider-aware client
    factory. It dispatches based on `config.PROVIDER`:

      • provider == "anthropic" → ClaudeClient (Anthropic /v1/messages)
      • provider in {deepseek, kimi, openai, groq, gemini, azure, xai}
        → OpenAICompatClient pointed at that provider's base_url + key
        (xai = xAI Grok at api.x.ai; distinct from groq's Llama inference)

    Existing callers (regenerator, every Novelty Lab module, etc.) need
    no change — both client classes expose the SAME `.call()` interface
    returning a `ClaudeResponse`.
    """
    global _GLOBAL_CLIENT
    with _GLOBAL_LOCK:
        if _GLOBAL_CLIENT is not None and not reload:
            return _GLOBAL_CLIENT

        try:
            import os
            try:
                import config as cfg
            except Exception:
                cfg = None  # type: ignore[assignment]

            provider = ((getattr(cfg, "PROVIDER", "") or "anthropic")
                          .lower().strip()) if cfg else "anthropic"

            if provider in ("anthropic", "claude"):
                api_key = (
                    os.getenv("ANTHROPIC_API_KEY")
                    or os.getenv("ANTHROPIC_AUTH_TOKEN")
                    or ""
                )
                base_url = os.getenv("ANTHROPIC_BASE_URL", DEFAULT_BASE_URL).strip()
                if cfg and not api_key:
                    api_key = getattr(cfg, "ANTHROPIC_API_KEY", "") or ""
                model = (getattr(cfg, "MODEL", "claude-sonnet-4-6")
                          if cfg else "claude-sonnet-4-6")
                if model not in AVAILABLE_MODELS:
                    model = "claude-sonnet-4-6"
                _GLOBAL_CLIENT = ClaudeClient(
                    api_key=api_key, model=model, base_url=base_url,
                )
                return _GLOBAL_CLIENT

            # OpenAI-compatible providers — DeepSeek, Kimi, OpenAI, Groq,
            # Gemini, Azure. Each has a different base_url + env var key.
            api_key, base_url, default_model = _provider_credentials(provider)
            if not api_key or not base_url:
                # Mis-configured — return None so the caller can degrade
                # to its no-LLM path with a sensible diagnostic.
                _GLOBAL_CLIENT = None
                return None
            model = (getattr(cfg, "MODEL", "") or default_model) if cfg else default_model
            _GLOBAL_CLIENT = OpenAICompatClient(
                api_key=api_key, model=model, base_url=base_url,
                provider_name=provider,
            )
            return _GLOBAL_CLIENT
        except Exception:
            return None


# Alias for clearer future code — same singleton accessor.
get_llm_client = get_claude_client


def is_anthropic_provider(provider: str) -> bool:
    """True if the given provider name uses the native Anthropic API."""
    return provider.lower() in ("anthropic", "claude")
