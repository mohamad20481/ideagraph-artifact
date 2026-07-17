"""Tests for the admin-dashboard LLM provider panel."""
from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import admin_dashboard
import config as cfg


# ── _api_key_for ────────────────────────────────────────────────────────────

def test_api_key_for_known_providers():
    """Known providers route to the right cfg attribute."""
    class _Stub:
        DEEPSEEK_API_KEY = "ds"
        OPENAI_API_KEY = "oa"
        GROQ_API_KEY = "gq"
        GEMINI_API_KEY = "gm"
        AZURE_API_KEY = "az"
        ANTHROPIC_API_KEY = "an"
        KIMI_API_KEY = "km"

    s = _Stub()
    assert admin_dashboard._api_key_for(s, "deepseek") == "ds"
    assert admin_dashboard._api_key_for(s, "openai") == "oa"
    assert admin_dashboard._api_key_for(s, "groq") == "gq"
    assert admin_dashboard._api_key_for(s, "gemini") == "gm"
    assert admin_dashboard._api_key_for(s, "azure") == "az"
    assert admin_dashboard._api_key_for(s, "anthropic") == "an"
    assert admin_dashboard._api_key_for(s, "kimi") == "km"


def test_api_key_for_unknown_provider_returns_empty():
    class _Stub:
        pass
    assert admin_dashboard._api_key_for(_Stub(), "totally_unknown") == ""


def test_api_key_for_missing_attr_returns_empty():
    class _Stub:
        # has none of the expected attrs
        pass
    assert admin_dashboard._api_key_for(_Stub(), "deepseek") == ""


# ── _persist_to_env ─────────────────────────────────────────────────────────

def test_persist_to_env_updates_existing_keys(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(textwrap.dedent("""\
        # comment line
        IDEAGRAPH_PROVIDER=deepseek
        IDEAGRAPH_MODEL=deepseek-chat
        OTHER_VAR=keepme
    """), encoding="utf-8")

    monkeypatch.setattr(admin_dashboard, "__file__",
                          str(tmp_path / "admin_dashboard.py"))
    err = admin_dashboard._persist_to_env("anthropic", "claude-opus-4-7")
    assert err is None
    text = env_file.read_text(encoding="utf-8")
    assert "IDEAGRAPH_PROVIDER=anthropic" in text
    assert "IDEAGRAPH_MODEL=claude-opus-4-7" in text
    assert "OTHER_VAR=keepme" in text
    # Comment preserved
    assert "# comment line" in text
    # Old values are gone (substring check still finds the new ones; assert
    # the old MODEL line specifically is not present)
    assert "IDEAGRAPH_MODEL=deepseek-chat" not in text


def test_persist_to_env_appends_missing_keys(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("OTHER=x\n", encoding="utf-8")
    monkeypatch.setattr(admin_dashboard, "__file__",
                          str(tmp_path / "admin_dashboard.py"))
    err = admin_dashboard._persist_to_env("kimi", "moonshot-v1-32k")
    assert err is None
    text = env_file.read_text(encoding="utf-8")
    assert "OTHER=x" in text
    assert "IDEAGRAPH_PROVIDER=kimi" in text
    assert "IDEAGRAPH_MODEL=moonshot-v1-32k" in text


def test_persist_to_env_creates_file_when_missing(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    assert not env_file.exists()
    monkeypatch.setattr(admin_dashboard, "__file__",
                          str(tmp_path / "admin_dashboard.py"))
    err = admin_dashboard._persist_to_env("openai", "gpt-4o")
    assert err is None
    text = env_file.read_text(encoding="utf-8")
    assert "IDEAGRAPH_PROVIDER=openai" in text
    assert "IDEAGRAPH_MODEL=gpt-4o" in text


def test_persist_to_env_leaves_unrelated_lines_intact(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(textwrap.dedent("""\
        DEEPSEEK_API_KEY=sk-abc
        ANTHROPIC_API_KEY=sk-xyz
        # provider config
        IDEAGRAPH_PROVIDER=deepseek
    """), encoding="utf-8")
    monkeypatch.setattr(admin_dashboard, "__file__",
                          str(tmp_path / "admin_dashboard.py"))
    admin_dashboard._persist_to_env("groq", "llama-3.3-70b-versatile")
    text = env_file.read_text(encoding="utf-8")
    assert "DEEPSEEK_API_KEY=sk-abc" in text
    assert "ANTHROPIC_API_KEY=sk-xyz" in text
    assert "IDEAGRAPH_PROVIDER=groq" in text
    assert "IDEAGRAPH_MODEL=llama-3.3-70b-versatile" in text


def test_persist_to_env_returns_error_on_write_failure(monkeypatch):
    # Point to a non-writable directory using a directory path that doesn't
    # exist — open() will fail and the function must return the error string.
    bogus = "Z:/__nonexistent_admin_dir__/admin_dashboard.py"
    monkeypatch.setattr(admin_dashboard, "__file__", bogus)
    err = admin_dashboard._persist_to_env("openai", "gpt-4o")
    assert err is not None
    assert isinstance(err, str) and err  # non-empty


# ── _PROVIDER_META covers all supported providers ──────────────────────────

def test_provider_meta_covers_every_supported_provider():
    for p in cfg.SUPPORTED_PROVIDERS:
        assert p in admin_dashboard._PROVIDER_META, (
            f"_PROVIDER_META missing entry for supported provider {p!r}"
        )


# ── render_admin_dashboard wires the new tab ───────────────────────────────

def test_render_admin_dashboard_creates_llm_provider_tab():
    """Smoke check: admin dashboard renders without exception and creates 4 tabs."""

    class _FakeTab:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    class _FakeSt:
        def __init__(self):
            self.tab_labels = []

        def title(self, *a, **kw): pass
        def caption(self, *a, **kw): pass
        def error(self, *a, **kw): pass

        def tabs(self, labels):
            self.tab_labels = list(labels)
            return tuple(_FakeTab() for _ in labels)

    fake = _FakeSt()
    # Will throw inside _render_stats / _render_population / _render_llm
    # because we don't fully mock streamlit — but the tab labels are
    # recorded before any sub-renderer runs, which is what we're checking.
    try:
        admin_dashboard.render_admin_dashboard(fake)
    except Exception:
        pass
    assert any("LLM Provider" in t for t in fake.tab_labels), (
        f"LLM Provider tab missing from {fake.tab_labels}"
    )
    # 6 admin tabs after Visual Rendering was added: Platform Stats,
    # Pipeline Simulator, Population (Federation), LLM Provider,
    # Feature Toggles, Visual Rendering.
    assert len(fake.tab_labels) == 6


# ── module exports / public surface ─────────────────────────────────────────

def test_panel_function_exists():
    assert hasattr(admin_dashboard, "_render_llm_provider_panel")
    assert callable(admin_dashboard._render_llm_provider_panel)


def test_provider_meta_keys_are_lowercase():
    for k in admin_dashboard._PROVIDER_META:
        assert k == k.lower()
