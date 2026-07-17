"""Tests for the provider-aware claude_provider.get_claude_client().

Reproduces the bug the user hit: switching `config.PROVIDER` in the admin
dashboard had no effect because the regenerator + Novelty Lab modules
called `get_claude_client()` which ALWAYS returned a ClaudeClient pointed
at aiprimetech.io with the Anthropic key — billing an unrelated account
instead of the one the user had topped up on Kimi.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import claude_provider as cp
import config as cfg


@pytest.fixture(autouse=True)
def _isolate_globals(monkeypatch):
    """Reset the module-level singleton + provider config between tests
    so they don't leak state."""
    # Wipe the singleton
    monkeypatch.setattr(cp, "_GLOBAL_CLIENT", None)
    # Snapshot + restore the relevant config attrs
    saved = {
        k: getattr(cfg, k) for k in (
            "PROVIDER", "MODEL",
            "DEEPSEEK_API_KEY", "KIMI_API_KEY", "OPENAI_API_KEY",
            "GROQ_API_KEY", "GEMINI_API_KEY", "AZURE_API_KEY",
            "AZURE_BASE_URL", "ANTHROPIC_API_KEY",
        )
    }
    yield
    for k, v in saved.items():
        setattr(cfg, k, v)
    monkeypatch.setattr(cp, "_GLOBAL_CLIENT", None)


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch — provider switching returns the right client class
# ─────────────────────────────────────────────────────────────────────────────

def test_anthropic_provider_returns_claude_client(monkeypatch):
    cfg.PROVIDER = "anthropic"
    cfg.MODEL = "claude-sonnet-4-6"
    monkeypatch.setattr(cfg, "ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")

    client = cp.get_claude_client(reload=True)
    assert isinstance(client, cp.ClaudeClient)
    assert client.model == "claude-sonnet-4-6"


def test_kimi_provider_returns_openai_compat_client(monkeypatch):
    """The user's exact failing case: PROVIDER=kimi must produce a Kimi-
    targeted client, NOT a ClaudeClient pointed at aiprimetech."""
    cfg.PROVIDER = "kimi"
    cfg.MODEL = "moonshot-v1-32k"
    monkeypatch.setattr(cfg, "KIMI_API_KEY", "test-kimi-key")
    monkeypatch.setenv("KIMI_API_KEY", "test-kimi-key")

    client = cp.get_claude_client(reload=True)
    assert isinstance(client, cp.OpenAICompatClient)
    assert client.provider_name == "kimi"
    assert client.model == "moonshot-v1-32k"
    assert "moonshot" in client.base_url
    assert client.api_key == "test-kimi-key"


def test_deepseek_provider_returns_openai_compat_client(monkeypatch):
    cfg.PROVIDER = "deepseek"
    cfg.MODEL = "deepseek-chat"
    monkeypatch.setattr(cfg, "DEEPSEEK_API_KEY", "test-ds-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-ds-key")

    client = cp.get_claude_client(reload=True)
    assert isinstance(client, cp.OpenAICompatClient)
    assert client.provider_name == "deepseek"
    assert "deepseek.com" in client.base_url


def test_openai_provider(monkeypatch):
    cfg.PROVIDER = "openai"
    cfg.MODEL = "gpt-4o"
    monkeypatch.setattr(cfg, "OPENAI_API_KEY", "sk-openai-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    client = cp.get_claude_client(reload=True)
    assert isinstance(client, cp.OpenAICompatClient)
    assert client.provider_name == "openai"
    assert "openai.com" in client.base_url


def test_groq_provider(monkeypatch):
    cfg.PROVIDER = "groq"
    cfg.MODEL = "llama-3.3-70b-versatile"
    monkeypatch.setattr(cfg, "GROQ_API_KEY", "gsk-test")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    client = cp.get_claude_client(reload=True)
    assert isinstance(client, cp.OpenAICompatClient)
    assert "groq" in client.base_url


def test_gemini_provider(monkeypatch):
    cfg.PROVIDER = "gemini"
    cfg.MODEL = "gemini-2.0-flash"
    monkeypatch.setattr(cfg, "GEMINI_API_KEY", "AIza-test")
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test")
    client = cp.get_claude_client(reload=True)
    assert isinstance(client, cp.OpenAICompatClient)
    assert "googleapis" in client.base_url


def test_azure_provider(monkeypatch):
    cfg.PROVIDER = "azure"
    cfg.MODEL = "DeepSeek-V3.2-Speciale"
    monkeypatch.setattr(cfg, "AZURE_API_KEY", "az-test")
    monkeypatch.setattr(cfg, "AZURE_BASE_URL", "https://test.azure-api.net/")
    monkeypatch.setenv("AZURE_API_KEY", "az-test")
    monkeypatch.setenv("AZURE_BASE_URL", "https://test.azure-api.net/")
    client = cp.get_claude_client(reload=True)
    assert isinstance(client, cp.OpenAICompatClient)
    assert client.base_url.startswith("https://test.azure-api.net")


def test_unknown_provider_returns_none(monkeypatch):
    cfg.PROVIDER = "totally_fake_provider"
    cfg.MODEL = "whatever"
    client = cp.get_claude_client(reload=True)
    assert client is None


def test_missing_key_for_provider_returns_none(monkeypatch):
    cfg.PROVIDER = "kimi"
    monkeypatch.setattr(cfg, "KIMI_API_KEY", "")
    monkeypatch.delenv("KIMI_API_KEY", raising=False)
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    client = cp.get_claude_client(reload=True)
    assert client is None


def test_singleton_caches_across_calls(monkeypatch):
    cfg.PROVIDER = "kimi"
    cfg.MODEL = "moonshot-v1-32k"
    monkeypatch.setattr(cfg, "KIMI_API_KEY", "test-kimi-key")
    monkeypatch.setenv("KIMI_API_KEY", "test-kimi-key")
    c1 = cp.get_claude_client(reload=True)
    c2 = cp.get_claude_client()  # no reload — should hit cache
    assert c1 is c2


def test_reload_after_provider_switch_returns_new_client(monkeypatch):
    """The admin dashboard's 'Apply now' must invalidate the singleton
    so the next call returns a client for the NEW provider."""
    cfg.PROVIDER = "kimi"
    monkeypatch.setattr(cfg, "KIMI_API_KEY", "test-kimi-key")
    monkeypatch.setenv("KIMI_API_KEY", "test-kimi-key")
    c1 = cp.get_claude_client(reload=True)
    assert c1.provider_name == "kimi"

    # Now flip to deepseek and reload — must get a different client.
    cfg.PROVIDER = "deepseek"
    monkeypatch.setattr(cfg, "DEEPSEEK_API_KEY", "test-ds-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-ds-key")
    c2 = cp.get_claude_client(reload=True)
    assert c2.provider_name == "deepseek"
    assert c1 is not c2


# ─────────────────────────────────────────────────────────────────────────────
# OpenAICompatClient — request/response shape + error surfacing
# ─────────────────────────────────────────────────────────────────────────────

def _ok_response(text: str = "hello") -> MagicMock:
    """Build a mock OpenAI-format 200 response."""
    m = MagicMock()
    m.status_code = 200
    m.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 42, "completion_tokens": 7},
    }
    return m


def _err_response(status: int, body: str) -> MagicMock:
    m = MagicMock()
    m.status_code = status
    m.text = body
    return m


def test_openai_compat_call_happy_path():
    c = cp.OpenAICompatClient(
        api_key="k", model="moonshot-v1-32k",
        base_url="https://api.moonshot.ai/v1", provider_name="kimi",
    )
    with patch.object(c._session, "post", return_value=_ok_response("yo")):
        resp = c.call(system="sys", user="usr", max_tokens=50)
    assert resp.success is True
    assert resp.text == "yo"
    assert resp.input_tokens == 42
    assert resp.output_tokens == 7
    assert resp.model == "moonshot-v1-32k"


def test_openai_compat_call_posts_to_chat_completions():
    c = cp.OpenAICompatClient(
        api_key="k", model="m", base_url="https://api.deepseek.com",
        provider_name="deepseek",
    )
    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["body"] = json
        captured["headers"] = headers
        return _ok_response("ok")

    with patch.object(c._session, "post", side_effect=_fake_post):
        c.call(system="sys", user="usr", max_tokens=10, json_mode=True)

    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer k"
    assert captured["body"]["model"] == "m"
    assert captured["body"]["messages"][0]["role"] == "system"
    assert captured["body"]["messages"][1]["role"] == "user"
    # json_mode adds response_format
    assert captured["body"]["response_format"] == {"type": "json_object"}


def test_openai_compat_kimi_k2_temperature_coercion():
    """Kimi's kimi-k2 reasoning models require temperature=1.0."""
    c = cp.OpenAICompatClient(
        api_key="k", model="kimi-k2.6", base_url="https://api.moonshot.ai/v1",
        provider_name="kimi",
    )
    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["temp"] = json["temperature"]
        return _ok_response()

    with patch.object(c._session, "post", side_effect=_fake_post):
        c.call(system="s", user="u", temperature=0.3)
    assert captured["temp"] == 1.0


def test_openai_compat_non_kimi_temperature_passthrough():
    c = cp.OpenAICompatClient(
        api_key="k", model="deepseek-chat", base_url="https://api.deepseek.com",
        provider_name="deepseek",
    )
    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["temp"] = json["temperature"]
        return _ok_response()

    with patch.object(c._session, "post", side_effect=_fake_post):
        c.call(system="s", user="u", temperature=0.3)
    assert captured["temp"] == 0.3


def test_openai_compat_non_retryable_4xx_returns_error_verbatim():
    """The INSUFFICIENT_BALANCE error body must be surfaced verbatim so
    the regenerator's _classify_api_error can recognize it."""
    c = cp.OpenAICompatClient(
        api_key="k", model="moonshot-v1-32k",
        base_url="https://api.moonshot.ai/v1", provider_name="kimi",
    )
    insufficient = _err_response(
        403, '{"code":"INSUFFICIENT_BALANCE","message":"Insufficient account balance"}',
    )
    with patch.object(c._session, "post", return_value=insufficient):
        resp = c.call(system="s", user="u")
    assert resp.success is False
    assert "INSUFFICIENT_BALANCE" in resp.error
    assert resp.attempts == 1  # 4xx is NOT retried


def test_openai_compat_5xx_is_retried():
    c = cp.OpenAICompatClient(
        api_key="k", model="m", base_url="https://api.moonshot.ai/v1",
        provider_name="kimi",
    )
    call_log = []

    def _fake_post(url, headers=None, json=None, timeout=None):
        call_log.append(1)
        return _err_response(503, "service unavailable")

    with patch.object(c._session, "post", side_effect=_fake_post), \
            patch("claude_provider.time.sleep"):  # don't actually sleep in tests
        resp = c.call(system="s", user="u")
    assert resp.success is False
    assert len(call_log) == 3  # 3 attempts
    assert "All 3 attempts failed" in resp.error


def test_openai_compat_unconfigured_key_short_circuits():
    c = cp.OpenAICompatClient(
        api_key="", model="m", base_url="https://example.com",
        provider_name="openai",
    )
    with patch.object(c._session, "post") as p:
        resp = c.call(system="s", user="u")
    assert resp.success is False
    assert "OPENAI_API_KEY" in resp.error
    p.assert_not_called()


def test_openai_compat_budget_exhaustion():
    c = cp.OpenAICompatClient(
        api_key="k", model="m", base_url="https://example.com",
        provider_name="openai", call_budget=2,
    )
    with patch.object(c._session, "post", return_value=_ok_response()):
        assert c.call(system="s", user="u").success is True
        assert c.call(system="s", user="u").success is True
        # 3rd call: budget exhausted, no HTTP call made.
        resp = c.call(system="s", user="u")
    assert resp.success is False
    assert "Budget exhausted" in resp.error


def test_openai_compat_cost_uses_config_rates(monkeypatch):
    c = cp.OpenAICompatClient(
        api_key="k", model="moonshot-v1-32k",
        base_url="https://api.moonshot.ai/v1", provider_name="kimi",
    )
    # config.COST_RATES['kimi'] = {'input': 0.60, 'output': 2.50} per M tokens.
    with patch.object(c._session, "post", return_value=_ok_response()):
        resp = c.call(system="s", user="u")
    # 42 prompt + 7 completion tokens
    expected = (42 * 0.60 + 7 * 2.50) / 1_000_000
    assert abs(resp.cost_usd - round(expected, 6)) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# Same interface as ClaudeClient — drop-in replacement
# ─────────────────────────────────────────────────────────────────────────────

def test_openai_compat_has_same_public_api_as_claude_client():
    """Both clients must expose .call, .is_configured, .model, .base_url,
    .api_key, .remaining_budget, .reset_budget — so the regenerator and
    Novelty Lab modules don't have to know which class they got."""
    required = {"call", "is_configured", "model", "base_url", "api_key",
                "remaining_budget", "reset_budget"}
    assert required <= set(dir(cp.ClaudeClient)) | {
        a for a in dir(cp.ClaudeClient(api_key="x"))
    }
    assert required <= set(dir(cp.OpenAICompatClient)) | {
        a for a in dir(cp.OpenAICompatClient(
            api_key="x", model="m", base_url="https://example.com",
        ))
    }


def test_get_llm_client_alias_exists():
    """The clearer name `get_llm_client` should be exported alongside the
    legacy `get_claude_client`."""
    assert hasattr(cp, "get_llm_client")
    assert cp.get_llm_client is cp.get_claude_client
