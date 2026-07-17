"""Tests for the operator feature-toggles system (admin dashboard tab)."""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import admin_dashboard
import config as cfg


# ── Config flag ─────────────────────────────────────────────────────────────

def test_config_has_corpus_anchored_flag():
    """The flag must exist on the config module with a sane default."""
    assert hasattr(cfg, "ENABLE_CORPUS_ANCHORED_NOVELTY")
    assert isinstance(cfg.ENABLE_CORPUS_ANCHORED_NOVELTY, bool)


def test_corpus_anchored_default_is_true():
    """When the env var is unset, the default ships as True so the
    feature is discoverable in fresh installs."""
    # We can't test the actual default without mucking with env at import
    # time, but we can assert the var is a string-coerced bool.
    src = (ROOT / "config.py").read_text(encoding="utf-8")
    assert "ENABLE_CORPUS_ANCHORED_NOVELTY" in src
    assert "IDEAGRAPH_CORPUS_ANCHORED_NOVELTY" in src
    # The default literal in the source should be "true".
    assert '"IDEAGRAPH_CORPUS_ANCHORED_NOVELTY", "true"' in src


# ── FEATURE_TOGGLES registry ────────────────────────────────────────────────

def test_feature_toggles_registry_exists():
    assert hasattr(admin_dashboard, "FEATURE_TOGGLES")
    assert isinstance(admin_dashboard.FEATURE_TOGGLES, list)
    assert len(admin_dashboard.FEATURE_TOGGLES) >= 1


def test_feature_toggles_entries_well_formed():
    """Every registry entry must have the required keys."""
    required = {"cfg_attr", "env_key", "default", "label", "description"}
    for t in admin_dashboard.FEATURE_TOGGLES:
        assert required <= set(t.keys()), \
            f"toggle missing keys: {required - set(t.keys())}"
        assert isinstance(t["cfg_attr"], str) and t["cfg_attr"]
        assert isinstance(t["env_key"], str) and t["env_key"].startswith("IDEAGRAPH_")
        assert isinstance(t["default"], bool)
        assert isinstance(t["label"], str) and t["label"]
        assert isinstance(t["description"], str) and len(t["description"]) > 20


def test_feature_toggles_includes_corpus_anchored():
    attrs = [t["cfg_attr"] for t in admin_dashboard.FEATURE_TOGGLES]
    assert "ENABLE_CORPUS_ANCHORED_NOVELTY" in attrs


def test_every_toggle_attr_exists_on_config():
    """Each toggle's `cfg_attr` must be a real attribute on the config
    module — otherwise the toggle silently does nothing."""
    for t in admin_dashboard.FEATURE_TOGGLES:
        assert hasattr(cfg, t["cfg_attr"]), (
            f"config has no attribute {t['cfg_attr']!r} "
            f"referenced by toggle {t['label']!r}"
        )


def test_no_duplicate_toggle_env_keys():
    keys = [t["env_key"] for t in admin_dashboard.FEATURE_TOGGLES]
    assert len(keys) == len(set(keys))


def test_no_duplicate_toggle_cfg_attrs():
    attrs = [t["cfg_attr"] for t in admin_dashboard.FEATURE_TOGGLES]
    assert len(attrs) == len(set(attrs))


# ── _persist_toggles_to_env ─────────────────────────────────────────────────

def test_persist_toggles_to_env_creates_file(tmp_path, monkeypatch):
    monkeypatch.setattr(admin_dashboard, "__file__",
                          str(tmp_path / "admin_dashboard.py"))
    env_file = tmp_path / ".env"
    assert not env_file.exists()
    err = admin_dashboard._persist_toggles_to_env(
        {"IDEAGRAPH_CORPUS_ANCHORED_NOVELTY": True},
    )
    assert err is None
    text = env_file.read_text(encoding="utf-8")
    assert "IDEAGRAPH_CORPUS_ANCHORED_NOVELTY=true" in text


def test_persist_toggles_to_env_appends_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(admin_dashboard, "__file__",
                          str(tmp_path / "admin_dashboard.py"))
    env_file = tmp_path / ".env"
    env_file.write_text("OTHER=keepme\n", encoding="utf-8")
    err = admin_dashboard._persist_toggles_to_env(
        {"IDEAGRAPH_CORPUS_ANCHORED_NOVELTY": False},
    )
    assert err is None
    text = env_file.read_text(encoding="utf-8")
    assert "OTHER=keepme" in text
    assert "IDEAGRAPH_CORPUS_ANCHORED_NOVELTY=false" in text


def test_persist_toggles_to_env_updates_existing(tmp_path, monkeypatch):
    monkeypatch.setattr(admin_dashboard, "__file__",
                          str(tmp_path / "admin_dashboard.py"))
    env_file = tmp_path / ".env"
    env_file.write_text(textwrap.dedent("""\
        # comment line
        IDEAGRAPH_CORPUS_ANCHORED_NOVELTY=true
        IDEAGRAPH_PROVIDER=deepseek
    """), encoding="utf-8")
    err = admin_dashboard._persist_toggles_to_env(
        {"IDEAGRAPH_CORPUS_ANCHORED_NOVELTY": False},
    )
    assert err is None
    text = env_file.read_text(encoding="utf-8")
    assert "IDEAGRAPH_CORPUS_ANCHORED_NOVELTY=false" in text
    assert "IDEAGRAPH_CORPUS_ANCHORED_NOVELTY=true" not in text
    # Other keys + comments preserved.
    assert "IDEAGRAPH_PROVIDER=deepseek" in text
    assert "# comment line" in text


def test_persist_toggles_to_env_multiple_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(admin_dashboard, "__file__",
                          str(tmp_path / "admin_dashboard.py"))
    err = admin_dashboard._persist_toggles_to_env({
        "IDEAGRAPH_CORPUS_ANCHORED_NOVELTY": False,
        "IDEAGRAPH_FAKE_FUTURE_TOGGLE": True,
    })
    assert err is None
    text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "IDEAGRAPH_CORPUS_ANCHORED_NOVELTY=false" in text
    assert "IDEAGRAPH_FAKE_FUTURE_TOGGLE=true" in text


def test_persist_toggles_to_env_returns_error_on_write_fail(monkeypatch):
    bogus = "Z:/__nonexistent_dir__/admin_dashboard.py"
    monkeypatch.setattr(admin_dashboard, "__file__", bogus)
    err = admin_dashboard._persist_toggles_to_env(
        {"IDEAGRAPH_CORPUS_ANCHORED_NOVELTY": True},
    )
    assert err is not None
    assert isinstance(err, str) and err


# ── Admin dashboard renders the new tab ────────────────────────────────────

def test_admin_dashboard_has_feature_toggles_tab():
    """Smoke check: the admin dashboard creates a Feature Toggles tab."""
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
    try:
        admin_dashboard.render_admin_dashboard(fake)
    except Exception:
        pass  # sub-renderers will throw on the fake st; we only check tabs
    assert any("Feature Toggles" in t for t in fake.tab_labels)
    # 6 admin tabs after Visual Rendering was added (was 5 before).
    assert len(fake.tab_labels) == 6
    assert any("Visual Rendering" in t for t in fake.tab_labels)


def test_feature_toggles_panel_function_exists():
    assert hasattr(admin_dashboard, "_render_feature_toggles_panel")
    assert callable(admin_dashboard._render_feature_toggles_panel)


# ── App.py wiring: gated radio option ──────────────────────────────────────

def test_app_radio_gated_on_corpus_anchored_flag():
    """The Novelty Lab radio must conditionally include
    `corpus_anchored` based on the config flag."""
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    # The append-when-enabled pattern must be present.
    assert "ENABLE_CORPUS_ANCHORED_NOVELTY" in src
    assert '_novelty_options.append("corpus_anchored")' in src
    # The session-state reset when the saved mode disappears must exist.
    assert (
        'st.session_state["novelty_mode"] = _novelty_options[0]'
        in src
    ) or (
        "_novelty_options[0]" in src
    )


def test_app_radio_default_options_count_18():
    """When corpus_anchored is disabled, the radio has 18 options.
    When enabled, 19. Smoke-check via source inspection."""
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    # Find the _novelty_options literal.
    idx = src.find("_novelty_options = [")
    assert idx >= 0
    block = src[idx:idx + 1200]
    # Count quoted mode names within the literal (close at next `]`).
    end = block.find("]")
    inside = block[:end]
    # Each quoted string corresponds to one mode.
    import re
    modes = re.findall(r'"([a-z_]+)"', inside)
    # The literal omits corpus_anchored — it's appended conditionally.
    assert "corpus_anchored" not in modes
    assert len(modes) == 18
