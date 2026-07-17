"""Tests for GrokImageProvider (xAI) + account-page logout flow.

All HTTP calls are mocked — no network, no real xAI account required.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ideagraph_image_renderer as ir


# ── Provider registration ───────────────────────────────────────────────────

def test_grok_registered_in_provider_registry():
    """The dropdown auto-populates from PROVIDER_REGISTRY — make sure
    grok is in there."""
    assert "grok" in ir.PROVIDER_REGISTRY
    assert ir.PROVIDER_REGISTRY["grok"] is ir.GrokImageProvider


def test_grok_has_provider_defaults():
    """When the admin UI switches to grok, it reads model + endpoint
    from PROVIDER_DEFAULTS — make sure they exist and look reasonable."""
    d = ir.PROVIDER_DEFAULTS.get("grok") or {}
    assert d.get("endpoint", "").startswith("https://api.x.ai")
    assert d.get("model", "").startswith("grok-2-image")
    # Catalog should include a couple of known model aliases.
    assert "grok-2-image" in d.get("known_models", [])


# ── GrokImageProvider.is_configured ─────────────────────────────────────────

def test_grok_is_configured_true_for_real_key():
    p = ir.GrokImageProvider(api_key="xai-real_key_xyz")
    assert p.is_configured is True


def test_grok_is_configured_false_for_empty_key():
    assert ir.GrokImageProvider(api_key="").is_configured is False


def test_grok_is_configured_false_for_placeholder():
    """Placeholder keys (sk-xxx... pattern) shouldn't be treated as real."""
    assert ir.GrokImageProvider(api_key="sk-xxxxxxxxx").is_configured is False


# ── GrokImageProvider._generate_raw — happy paths ──────────────────────────

def _fake_response(status_code=200, json_payload=None, text=""):
    r = MagicMock()
    r.status_code = status_code
    r.text = text or ""
    r.json.return_value = json_payload or {}
    return r


def test_grok_generate_returns_image_url():
    """Stripe-style happy path: HTTP 200, single data item with url."""
    p = ir.GrokImageProvider(api_key="xai-test_xyz")
    fake = _fake_response(json_payload={
        "data": [{
            "url": "https://imgen.x.ai/test/abc.png",
            "revised_prompt": "a soft editorial figure of a neural net",
        }],
    })
    with patch.object(ir, "requests") as mock_req, \
         patch.object(ir, "_HAS_REQUESTS", True):
        mock_req.post.return_value = fake
        result = p.generate("a neural network diagram")

    assert result.get("image_url") == "https://imgen.x.ai/test/abc.png"
    assert result.get("revised_prompt", "").startswith("a soft")
    # Verify the request shape.
    args, kwargs = mock_req.post.call_args
    assert args[0].endswith("/v1/images/generations")
    assert kwargs["headers"]["Authorization"] == "Bearer xai-test_xyz"
    body = kwargs["json"]
    assert body["model"] == "grok-2-image"
    assert body["prompt"] == "a neural network diagram"
    assert body["n"] == 1


def test_grok_generate_handles_b64_response():
    """xAI normally returns urls but some setups return b64_json — handle it."""
    import base64
    sample = base64.b64encode(b"PNG_FAKE_BYTES").decode("ascii")
    p = ir.GrokImageProvider(api_key="xai-test_xyz")
    fake = _fake_response(json_payload={
        "data": [{"b64_json": sample}],
    })
    with patch.object(ir, "requests") as mock_req, \
         patch.object(ir, "_HAS_REQUESTS", True):
        mock_req.post.return_value = fake
        result = p.generate("test")

    assert result.get("image_bytes") == b"PNG_FAKE_BYTES"


# ── Failure modes ───────────────────────────────────────────────────────────

def test_grok_generate_http_error_surfaces_status_code():
    p = ir.GrokImageProvider(api_key="xai-test_xyz")
    fake = _fake_response(
        status_code=401,
        text='{"error":"invalid api key"}',
    )
    with patch.object(ir, "requests") as mock_req, \
         patch.object(ir, "_HAS_REQUESTS", True):
        mock_req.post.return_value = fake
        result = p.generate("test")
    assert "HTTP 401" in result.get("error", "")


def test_grok_generate_network_error_handled():
    p = ir.GrokImageProvider(api_key="xai-test_xyz")
    with patch.object(ir, "requests") as mock_req, \
         patch.object(ir, "_HAS_REQUESTS", True):
        mock_req.post.side_effect = RuntimeError("connection refused")
        result = p.generate("test")
    assert "network" in result.get("error", "").lower()


def test_grok_generate_unconfigured_short_circuits():
    """No key → don't even hit the network."""
    p = ir.GrokImageProvider(api_key="")
    with patch.object(ir, "requests") as mock_req:
        result = p.generate("test")
        mock_req.post.assert_not_called()
    assert "not configured" in result.get("error", "").lower()


def test_grok_generate_empty_data_array_handled():
    """xAI returned 200 but data was empty — surface a useful error."""
    p = ir.GrokImageProvider(api_key="xai-test_xyz")
    fake = _fake_response(json_payload={"data": []})
    with patch.object(ir, "requests") as mock_req, \
         patch.object(ir, "_HAS_REQUESTS", True):
        mock_req.post.return_value = fake
        result = p.generate("test")
    assert "no images" in result.get("error", "").lower()


# ── Provider-specific key resolution in the renderer ───────────────────────

def test_grok_key_resolution_from_env(monkeypatch):
    """Setting GROK_API_KEY should give Grok-provider instances the key
    even if NANO_BANANA_API_KEY is empty/different."""
    monkeypatch.setenv("GROK_API_KEY", "xai-from_env_xyz")
    monkeypatch.setenv("NANO_BANANA_PROVIDER", "grok")
    monkeypatch.delenv("NANO_BANANA_API_KEY", raising=False)
    monkeypatch.delenv("BFL_API_KEY", raising=False)
    r = ir.NanoBananaImageRenderer()
    assert r.provider.name == "grok"
    assert r.provider.api_key == "xai-from_env_xyz"


def test_grok_key_xai_alias_also_works(monkeypatch):
    """XAI_API_KEY should be honored as a fallback alias."""
    monkeypatch.delenv("GROK_API_KEY", raising=False)
    monkeypatch.setenv("XAI_API_KEY", "xai-from_alias_xyz")
    monkeypatch.setenv("NANO_BANANA_PROVIDER", "grok")
    monkeypatch.delenv("NANO_BANANA_API_KEY", raising=False)
    monkeypatch.delenv("BFL_API_KEY", raising=False)
    r = ir.NanoBananaImageRenderer()
    assert r.provider.api_key == "xai-from_alias_xyz"


def test_grok_key_takes_precedence_over_generic_nano_banana_key(monkeypatch):
    """Per-provider key beats the generic chain — so a user can have
    both GROK_API_KEY and NANO_BANANA_API_KEY=AIza… in .env and switch
    providers without manually re-pasting."""
    monkeypatch.setenv("GROK_API_KEY", "xai-specific_xyz")
    monkeypatch.setenv("NANO_BANANA_API_KEY", "AIza-generic_xyz")
    monkeypatch.setenv("NANO_BANANA_PROVIDER", "grok")
    r = ir.NanoBananaImageRenderer()
    assert r.provider.api_key == "xai-specific_xyz"


def test_non_grok_provider_ignores_grok_key(monkeypatch):
    """When provider is flux_bfl, GROK_API_KEY should NOT leak into it."""
    monkeypatch.setenv("GROK_API_KEY", "xai-should_not_leak")
    monkeypatch.setenv("NANO_BANANA_API_KEY", "bfl-real_key")
    monkeypatch.setenv("NANO_BANANA_PROVIDER", "flux_bfl")
    r = ir.NanoBananaImageRenderer()
    assert r.provider.api_key == "bfl-real_key"
    assert "xai" not in r.provider.api_key


# ── Logout flow on the Manage Account page ─────────────────────────────────

def _make_st_stub():
    stub = MagicMock()
    stub.columns.side_effect = lambda spec, **kw: (
        [MagicMock() for _ in range(spec)]
        if isinstance(spec, int)
        else [MagicMock() for _ in spec]
    )
    for attr in ("expander", "container", "form"):
        cm = getattr(stub, attr)
        cm.return_value.__enter__ = MagicMock(return_value=stub)
        cm.return_value.__exit__ = MagicMock(return_value=None)
    def _tabs(labels):
        out = []
        for _ in labels:
            t = MagicMock()
            t.__enter__ = MagicMock(return_value=stub)
            t.__exit__ = MagicMock(return_value=None)
            out.append(t)
        return out
    stub.tabs.side_effect = _tabs
    stub.button.return_value = False
    stub.form_submit_button.return_value = False
    stub.checkbox.return_value = False
    stub.text_input.return_value = ""
    stub.session_state = {}
    return stub


def test_account_page_logout_button_clears_session():
    """Clicking 🚪 Logout on the Manage Account page wipes user_id /
    username / logged_in, sets logged_in=False, and reruns."""
    import account_ui
    st = _make_st_stub()
    # First button (← Back) → False, second button (🚪 Logout) → True.
    # Note: the actual order depends on render flow; we want only the
    # logout button to fire, so the back button (called first) is False
    # and logout (called next in the same render pass) is True.
    st.button.side_effect = [False, True] + [False] * 50
    st.session_state.update({
        "user_id": 42, "username": "alice", "logged_in": True,
        "running": True, "done": False,
    })
    with patch("auth_ui._clear_remember_token", create=True), \
         patch("auth_ui._clear_session_recovery_token", create=True):
        account_ui.render_account_page(st, 42)

    assert st.session_state.get("logged_in") is False
    assert "user_id" not in st.session_state
    assert "username" not in st.session_state
    assert st.rerun.called
