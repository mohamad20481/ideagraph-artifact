"""Tests for the xAI Grok LLM provider wiring.

Distinct from the existing `grok` IMAGE provider in
ideagraph_image_renderer.py — the LLM provider is registered under the
name `xai` to avoid collision with both `groq` (Llama inference) and
the image-only `grok`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── config.py registration ──────────────────────────────────────────────────

def test_xai_in_supported_providers():
    import importlib, config
    importlib.reload(config)
    assert "xai" in config.SUPPORTED_PROVIDERS


def test_xai_has_default_model():
    import importlib, config
    importlib.reload(config)
    assert config._DEFAULT_MODELS.get("xai") == "grok-2-latest"


def test_xai_has_cost_rates():
    import importlib, config
    importlib.reload(config)
    rates = config.COST_RATES.get("xai")
    assert rates is not None
    assert rates["input"] > 0 and rates["output"] > 0


def test_xai_known_models_catalog_exists():
    import importlib, config
    importlib.reload(config)
    catalog = config.XAI_KNOWN_MODELS
    assert isinstance(catalog, list)
    assert "grok-2-latest" in catalog
    assert "grok-3" in catalog


def test_xai_base_url_set():
    import importlib, config
    importlib.reload(config)
    assert config.XAI_BASE_URL == "https://api.x.ai/v1"


def test_xai_default_in_known_models():
    """Default model must be in catalog so the dropdown opens on it,
    not on Custom…"""
    import importlib, config
    importlib.reload(config)
    assert config._DEFAULT_MODELS["xai"] in config.XAI_KNOWN_MODELS


# ── claude_provider.py dispatcher ──────────────────────────────────────────

def test_provider_credentials_xai_resolves_to_xai_endpoint(monkeypatch):
    """The dispatcher must route `xai` to api.x.ai with the GROK_API_KEY."""
    monkeypatch.setenv("GROK_API_KEY", "xai-test-grok-key")
    import importlib, claude_provider, config
    importlib.reload(config)
    importlib.reload(claude_provider)
    key, base, default = claude_provider._provider_credentials("xai")
    assert key == "xai-test-grok-key"
    assert base == "https://api.x.ai/v1"
    assert default == "grok-2-latest"


def test_xai_dispatcher_has_xai_api_key_fallback():
    """The xai branch should fall back to XAI_API_KEY if GROK_API_KEY
    is empty. Asserted via source grep to avoid env-state races with
    dotenv reload picking up the real .env value."""
    src = (ROOT / "claude_provider.py").read_text(encoding="utf-8")
    # The dispatcher's xai branch must reference both names.
    # Find the section around `provider == "xai"`.
    idx = src.find('provider == "xai"')
    assert idx >= 0, "xai branch not found in dispatcher"
    section = src[idx:idx + 600]
    assert "GROK_API_KEY" in section
    assert "XAI_API_KEY" in section


def test_provider_credentials_xai_does_not_route_to_groq():
    """Critical: 'xai' MUST NOT dispatch to api.groq.com — they're
    different services with different keys."""
    import claude_provider
    key, base, default = claude_provider._provider_credentials("xai")
    assert "groq.com" not in base, (
        f"xai dispatched to Groq endpoint {base!r} — would 401."
    )
    assert "x.ai" in base


def test_provider_credentials_groq_does_not_route_to_xai():
    """The reverse: 'groq' must stay on api.groq.com."""
    import claude_provider
    key, base, default = claude_provider._provider_credentials("groq")
    assert "x.ai" not in base
    assert "groq.com" in base


# ── admin_dashboard.py wiring ──────────────────────────────────────────────

def test_admin_provider_meta_includes_xai():
    import admin_dashboard
    assert "xai" in admin_dashboard._PROVIDER_META
    name, emoji = admin_dashboard._PROVIDER_META["xai"]
    # Label must make Groq/Grok distinction clear.
    assert "xAI" in name or "Grok" in name


def test_admin_api_key_for_xai_returns_grok_api_key(monkeypatch):
    """The admin _api_key_for(cfg, 'xai') must return GROK_API_KEY."""
    monkeypatch.setenv("GROK_API_KEY", "xai-test-key-for-admin")
    import importlib, config, admin_dashboard
    importlib.reload(config)
    importlib.reload(admin_dashboard)
    assert admin_dashboard._api_key_for(config, "xai") == "xai-test-key-for-admin"


def test_admin_provider_key_names_includes_xai():
    import admin_dashboard
    assert "xai" in admin_dashboard._PROVIDER_KEY_NAMES
    env_var, cfg_attr = admin_dashboard._PROVIDER_KEY_NAMES["xai"]
    assert env_var == "GROK_API_KEY"
    assert cfg_attr == "GROK_API_KEY"


# ── _test_provider_api_key covers xai ──────────────────────────────────────

def test_test_provider_api_key_xai_pings_xai_endpoint():
    """The 🧪 Test connection button must probe api.x.ai for the xai
    provider — NOT api.groq.com."""
    import admin_dashboard
    with patch("requests.get") as mock_get:
        fake_resp = MagicMock()
        fake_resp.status_code = 200
        mock_get.return_value = fake_resp
        admin_dashboard._test_provider_api_key("xai", "xai-test-key")
    args, kwargs = mock_get.call_args
    url = args[0] if args else kwargs.get("url")
    # The probe should hit xAI, not Groq.
    assert "x.ai" in url, f"xai probe hit wrong endpoint: {url}"
    assert "groq.com" not in url
    # Auth header must be Bearer xai-test-key.
    assert kwargs["headers"]["Authorization"] == "Bearer xai-test-key"


# ── No clash with image-renderer 'grok' provider ───────────────────────────

def test_xai_llm_distinct_from_grok_image_provider():
    """The image-renderer has a 'grok' provider for grok-2-image. The
    LLM dispatcher must not confuse the two — they live in different
    namespaces (config.SUPPORTED_PROVIDERS vs PROVIDER_REGISTRY)."""
    import config
    import ideagraph_image_renderer as ir
    # LLM side uses 'xai'.
    assert "xai" in config.SUPPORTED_PROVIDERS
    # Image side uses 'grok'.
    assert "grok" in ir.PROVIDER_REGISTRY
    # The names don't clash.
    assert "grok" not in config.SUPPORTED_PROVIDERS
    assert "xai" not in ir.PROVIDER_REGISTRY


# ── Sidebar wiring (app.py) ────────────────────────────────────────────────

def test_app_sidebar_xai_in_source():
    """Verify the sidebar provider metadata file includes xai."""
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    assert '"xai":' in src, "xai not added to sidebar _provider_meta / _api_key_map"
    # The label should clarify Groq vs Grok.
    assert "xAI Grok" in src
