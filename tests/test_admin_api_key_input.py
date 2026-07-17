"""Tests for the admin LLM-Provider API-key input + deepseek-v4 catalog.

Covers:
  - deepseek-v4 model in catalog + label
  - _persist_env_updates round-trip (preserves comments, updates in place, appends)
  - _set_provider_api_key: runtime cfg + os.environ + persist=True/False
  - Provider→env-var mapping for all 7 supported providers
  - Backward-compat: _persist_to_env still works
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


# ── Catalog: deepseek-v4 ────────────────────────────────────────────────────

def test_deepseek_v4_in_catalog():
    import importlib, config
    importlib.reload(config)
    assert "deepseek-v4" in config.DEEPSEEK_KNOWN_MODELS


def test_deepseek_v4_label_includes_pre_ga_marker():
    """Confirm the label dict has a friendly label for deepseek-v4."""
    import importlib, admin_dashboard
    importlib.reload(admin_dashboard)
    # _ds_labels is local to _render_llm_provider_panel; we check the
    # source for the entry directly.
    src = (ROOT / "admin_dashboard.py").read_text(encoding="utf-8")
    assert '"deepseek-v4"' in src
    # The label should mention "V4" so the user sees it in the dropdown.
    assert "V4" in src


# ── _persist_env_updates round-trip ────────────────────────────────────────

@pytest.fixture()
def tmp_env(tmp_path, monkeypatch):
    """Point _persist_env_updates at a temp .env file by monkeypatching
    its __file__-derived path. Returns the path."""
    import admin_dashboard
    env_path = tmp_path / ".env"
    # The function joins os.path.dirname(__file__) — monkeypatch __file__.
    fake_dir = str(tmp_path)
    monkeypatch.setattr(
        admin_dashboard, "__file__",
        os.path.join(fake_dir, "admin_dashboard.py"),
    )
    return env_path


def test_persist_env_updates_creates_file_when_missing(tmp_env):
    import admin_dashboard
    err = admin_dashboard._persist_env_updates({"FOO": "bar"})
    assert err is None
    assert tmp_env.read_text() == "FOO=bar\n"


def test_persist_env_updates_updates_existing_key_in_place(tmp_env):
    import admin_dashboard
    tmp_env.write_text("FOO=old\nBAR=keep\n")
    err = admin_dashboard._persist_env_updates({"FOO": "new"})
    assert err is None
    text = tmp_env.read_text()
    assert "FOO=new" in text
    assert "FOO=old" not in text
    # Other keys preserved.
    assert "BAR=keep" in text


def test_persist_env_updates_preserves_comments_and_blank_lines(tmp_env):
    import admin_dashboard
    tmp_env.write_text(
        "# section header\n"
        "\n"
        "FOO=old\n"
        "# trailing comment\n"
    )
    admin_dashboard._persist_env_updates({"FOO": "new"})
    text = tmp_env.read_text()
    assert "# section header" in text
    assert "# trailing comment" in text


def test_persist_env_updates_appends_missing_keys_at_end(tmp_env):
    import admin_dashboard
    tmp_env.write_text("FOO=foo\n")
    admin_dashboard._persist_env_updates({"NEW": "newval"})
    text = tmp_env.read_text()
    assert "FOO=foo" in text  # untouched
    assert text.endswith("NEW=newval\n")  # appended


def test_persist_env_updates_multiple_keys_at_once(tmp_env):
    import admin_dashboard
    tmp_env.write_text("A=old_a\n")
    admin_dashboard._persist_env_updates({"A": "new_a", "B": "new_b"})
    text = tmp_env.read_text()
    assert "A=new_a" in text
    assert "A=old_a" not in text
    assert "B=new_b" in text


# ── _persist_to_env backward-compat ────────────────────────────────────────

def test_persist_to_env_still_works(tmp_env):
    """The existing wrapper must keep working for the original switch path."""
    import admin_dashboard
    err = admin_dashboard._persist_to_env(
        provider="deepseek", model="deepseek-chat",
    )
    assert err is None
    text = tmp_env.read_text()
    assert "IDEAGRAPH_PROVIDER=deepseek" in text
    assert "IDEAGRAPH_MODEL=deepseek-chat" in text


# ── _PROVIDER_KEY_NAMES mapping ────────────────────────────────────────────

def test_provider_key_names_covers_all_supported_providers():
    """Every entry in SUPPORTED_PROVIDERS must have a key-name mapping
    so the API-key widget works for every provider, not just deepseek."""
    import importlib, config, admin_dashboard
    importlib.reload(config); importlib.reload(admin_dashboard)
    for p in config.SUPPORTED_PROVIDERS:
        assert p in admin_dashboard._PROVIDER_KEY_NAMES, (
            f"Provider {p!r} is in SUPPORTED_PROVIDERS but has no "
            f"entry in _PROVIDER_KEY_NAMES — the API-key widget would "
            f"fail when this provider is selected."
        )


def test_provider_key_names_uses_canonical_env_vars():
    """Each mapping should point at the env-var name the rest of the
    codebase reads (DEEPSEEK_API_KEY etc.)."""
    import admin_dashboard
    assert admin_dashboard._PROVIDER_KEY_NAMES["deepseek"] == (
        "DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY",
    )
    assert admin_dashboard._PROVIDER_KEY_NAMES["openai"] == (
        "OPENAI_API_KEY", "OPENAI_API_KEY",
    )


# ── _set_provider_api_key behavior ─────────────────────────────────────────

def test_set_provider_api_key_unknown_provider_returns_error():
    import admin_dashboard
    result = admin_dashboard._set_provider_api_key(
        provider="not_a_real_provider", new_key="x",
    )
    assert result["ok"] is False
    assert "Unknown provider" in result["env_error"]


def test_set_provider_api_key_runtime_sets_os_environ_and_cfg(monkeypatch):
    """Without persist, the new key should appear on cfg.* and os.environ
    but the .env file should NOT be touched."""
    import importlib, config, admin_dashboard
    importlib.reload(config)
    importlib.reload(admin_dashboard)
    # Save baseline.
    original_key = config.DEEPSEEK_API_KEY
    original_env = os.environ.get("DEEPSEEK_API_KEY", "")
    try:
        result = admin_dashboard._set_provider_api_key(
            provider="deepseek",
            new_key="sk-runtime-only-test-key",
            persist_env=False,
        )
        assert result["ok"] is True
        assert result["runtime_set"] is True
        assert result["persisted"] is False
        assert config.DEEPSEEK_API_KEY == "sk-runtime-only-test-key"
        assert os.environ["DEEPSEEK_API_KEY"] == "sk-runtime-only-test-key"
    finally:
        # Restore.
        config.DEEPSEEK_API_KEY = original_key
        if original_env:
            os.environ["DEEPSEEK_API_KEY"] = original_env
        else:
            os.environ.pop("DEEPSEEK_API_KEY", None)


def test_set_provider_api_key_persist_writes_env(tmp_env, monkeypatch):
    """With persist=True, the .env file should be updated."""
    import importlib, config, admin_dashboard
    importlib.reload(config)
    importlib.reload(admin_dashboard)
    # Re-apply the tmp_env monkeypatch since reload reset __file__.
    monkeypatch.setattr(
        admin_dashboard, "__file__",
        os.path.join(str(tmp_env.parent), "admin_dashboard.py"),
    )
    original_key = config.DEEPSEEK_API_KEY
    try:
        result = admin_dashboard._set_provider_api_key(
            provider="deepseek",
            new_key="sk-persist-test-key",
            persist_env=True,
        )
        assert result["ok"] is True
        assert result["persisted"] is True
        assert result["env_error"] is None
        text = tmp_env.read_text()
        assert "DEEPSEEK_API_KEY=sk-persist-test-key" in text
    finally:
        config.DEEPSEEK_API_KEY = original_key


def test_set_provider_api_key_empty_key_refuses_by_default():
    """Safety: empty string is REJECTED to prevent accidental wipe of a
    working key. Caller must pass force_clear=True for revocation."""
    import importlib, config, admin_dashboard
    importlib.reload(config)
    importlib.reload(admin_dashboard)
    original_key = config.DEEPSEEK_API_KEY
    try:
        result = admin_dashboard._set_provider_api_key(
            provider="deepseek", new_key="", persist_env=False,
        )
        assert result["ok"] is False
        assert "Empty API key" in (result.get("env_error") or "")
        # Existing key untouched.
        assert config.DEEPSEEK_API_KEY == original_key
    finally:
        config.DEEPSEEK_API_KEY = original_key


def test_set_provider_api_key_force_clear_does_wipe():
    """When the caller passes force_clear=True, empty-key wipe IS allowed.
    This is the revocation path."""
    import importlib, config, admin_dashboard
    importlib.reload(config)
    importlib.reload(admin_dashboard)
    original_key = config.DEEPSEEK_API_KEY
    try:
        # Set a non-empty key first.
        admin_dashboard._set_provider_api_key(
            provider="deepseek", new_key="some-key-to-clear",
            persist_env=False,
        )
        # Now clear with force_clear=True.
        result = admin_dashboard._set_provider_api_key(
            provider="deepseek", new_key="",
            persist_env=False, force_clear=True,
        )
        assert result["ok"] is True
        assert config.DEEPSEEK_API_KEY == ""
    finally:
        config.DEEPSEEK_API_KEY = original_key


def test_set_provider_api_key_returns_client_reload_status():
    """The result dict should expose client_reloaded so the UI can show
    a warning if the cached client refused to rebuild."""
    import importlib, config, admin_dashboard
    importlib.reload(config)
    importlib.reload(admin_dashboard)
    original_key = config.DEEPSEEK_API_KEY
    try:
        result = admin_dashboard._set_provider_api_key(
            provider="deepseek", new_key="sk-x", persist_env=False,
        )
        # Either reload succeeded (True) or it failed and the error is reported.
        assert "client_reloaded" in result
        if not result["client_reloaded"]:
            assert result.get("client_reload_error")
    finally:
        config.DEEPSEEK_API_KEY = original_key


def test_no_double_claude_client_reload_in_apply_handler():
    """v6 reviewer caught: get_claude_client(reload=True) was called
    unconditionally in the outer Apply handler AND inside
    _set_provider_api_key — double-reload waste when key changed.

    v7 fix: the outer reload is now gated by
    `_provider_or_model_changed and not _key_changed`, so the two paths
    never both fire. Verify via source grep that the gate is present."""
    import re
    src = (ROOT / "admin_dashboard.py").read_text(encoding="utf-8")
    # The outer reload should be guarded by the conditional.
    assert "_provider_or_model_changed and not _key_changed" in src, (
        "Outer get_claude_client reload is not gated by "
        "(_provider_or_model_changed and not _key_changed) — risks "
        "the v6 double-invalidation regression."
    )


# ── _test_provider_api_key (recovery UX) ───────────────────────────────────

def test_test_provider_api_key_empty_key_returns_no_test():
    import admin_dashboard
    result = admin_dashboard._test_provider_api_key("deepseek", "")
    assert result["ok"] is False
    assert "Empty" in result["message"]


def test_test_provider_api_key_unknown_provider_returns_error():
    import admin_dashboard
    result = admin_dashboard._test_provider_api_key("not_a_provider", "x")
    assert result["ok"] is False
    assert "No probe configured" in result["message"]


def test_test_provider_api_key_200_means_ok():
    """A 200 status from /v1/models means the key works."""
    import admin_dashboard
    with patch("requests.get") as mock_get:
        fake_resp = MagicMock()
        fake_resp.status_code = 200
        mock_get.return_value = fake_resp
        result = admin_dashboard._test_provider_api_key("deepseek", "sk-test")
    assert result["ok"] is True
    assert result["status_code"] == 200


def test_test_provider_api_key_401_means_invalid():
    """The actual user-trap scenario: typed key returns 401."""
    import admin_dashboard
    with patch("requests.get") as mock_get:
        fake_resp = MagicMock()
        fake_resp.status_code = 401
        fake_resp.text = '{"error": "invalid"}'
        mock_get.return_value = fake_resp
        result = admin_dashboard._test_provider_api_key("deepseek", "sk-bad")
    assert result["ok"] is False
    assert result["status_code"] == 401
    assert "invalid" in result["message"].lower() or "expired" in result["message"].lower()


def test_test_provider_api_key_429_is_warning_not_failure():
    """Rate-limit means the key WORKS — just throttled. Don't scare the user."""
    import admin_dashboard
    with patch("requests.get") as mock_get:
        fake_resp = MagicMock()
        fake_resp.status_code = 429
        mock_get.return_value = fake_resp
        result = admin_dashboard._test_provider_api_key("deepseek", "sk-test")
    # Returns ok=False because the request didn't succeed, but the
    # message tells the user the key likely works.
    assert result["status_code"] == 429
    assert "rate-limit" in result["message"].lower() or "likely works" in result["message"]


def test_test_provider_api_key_network_error_handled():
    """Network errors should not crash, just return ok=False."""
    import admin_dashboard
    with patch("requests.get") as mock_get:
        mock_get.side_effect = RuntimeError("connection refused")
        result = admin_dashboard._test_provider_api_key("deepseek", "sk-test")
    assert result["ok"] is False
    assert "network" in result["message"].lower()


def test_test_provider_api_key_anthropic_uses_xapikey_header():
    """Anthropic uses x-api-key header, not Bearer."""
    import admin_dashboard
    with patch("requests.get") as mock_get:
        fake_resp = MagicMock()
        fake_resp.status_code = 200
        mock_get.return_value = fake_resp
        admin_dashboard._test_provider_api_key("anthropic", "sk-ant-test")
    args, kwargs = mock_get.call_args
    assert kwargs["headers"]["x-api-key"] == "sk-ant-test"
    assert "anthropic-version" in kwargs["headers"]


# ── _reset_provider_key_to_env (recovery UX) ───────────────────────────────

def test_reset_provider_key_to_env_unknown_provider_errors():
    import admin_dashboard
    result = admin_dashboard._reset_provider_key_to_env("not_a_provider")
    assert result["ok"] is False
    assert "Unknown provider" in result["error"]


def test_reset_provider_key_to_env_restores_from_env(tmp_env, monkeypatch):
    """Reset reads the .env file directly, restoring the key the
    operator typed in the admin panel got overridden."""
    import importlib, config, admin_dashboard
    importlib.reload(config)
    importlib.reload(admin_dashboard)
    monkeypatch.setattr(
        admin_dashboard, "__file__",
        os.path.join(str(tmp_env.parent), "admin_dashboard.py"),
    )
    tmp_env.write_text(
        "# DeepSeek\n"
        "DEEPSEEK_API_KEY=sk-restored-from-env-364b8\n"
    )
    # Simulate the bad runtime state.
    config.DEEPSEEK_API_KEY = "sk-bad-runtime-key-f985"
    try:
        result = admin_dashboard._reset_provider_key_to_env("deepseek")
        assert result["ok"] is True
        assert result["had_value"] is True
        assert config.DEEPSEEK_API_KEY == "sk-restored-from-env-364b8"
        assert os.environ["DEEPSEEK_API_KEY"] == "sk-restored-from-env-364b8"
    finally:
        config.DEEPSEEK_API_KEY = ""


def test_reset_provider_key_to_env_missing_env_clears_runtime(tmp_env, monkeypatch):
    """If .env doesn't have the key, runtime is cleared (had_value=False)."""
    import importlib, config, admin_dashboard
    importlib.reload(config)
    importlib.reload(admin_dashboard)
    monkeypatch.setattr(
        admin_dashboard, "__file__",
        os.path.join(str(tmp_env.parent), "admin_dashboard.py"),
    )
    tmp_env.write_text("OTHER_KEY=foo\n")  # no DEEPSEEK_API_KEY
    config.DEEPSEEK_API_KEY = "sk-runtime-key"
    try:
        result = admin_dashboard._reset_provider_key_to_env("deepseek")
        assert result["ok"] is True
        assert result["had_value"] is False
        assert config.DEEPSEEK_API_KEY == ""
    finally:
        config.DEEPSEEK_API_KEY = ""


def test_set_provider_api_key_works_for_every_supported_provider():
    """Smoke test all 7 providers go through the runtime-set path
    without raising."""
    import importlib, config, admin_dashboard
    importlib.reload(config)
    importlib.reload(admin_dashboard)
    for p in config.SUPPORTED_PROVIDERS:
        env_var, cfg_attr = admin_dashboard._PROVIDER_KEY_NAMES[p]
        original_cfg = getattr(config, cfg_attr, "")
        original_env = os.environ.get(env_var, "")
        try:
            result = admin_dashboard._set_provider_api_key(
                provider=p, new_key=f"test-{p}-key", persist_env=False,
            )
            assert result["ok"] is True, f"provider {p} failed: {result}"
            assert getattr(config, cfg_attr) == f"test-{p}-key"
            assert os.environ[env_var] == f"test-{p}-key"
        finally:
            setattr(config, cfg_attr, original_cfg)
            if original_env:
                os.environ[env_var] = original_env
            else:
                os.environ.pop(env_var, None)
