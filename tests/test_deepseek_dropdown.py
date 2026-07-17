"""Tests for the DeepSeek model dropdown in the admin LLM Provider panel.

DeepSeek was already a registered backend (config.DEEPSEEK_API_KEY,
COST_RATES, _DEFAULT_MODELS, _PROVIDER_META) but the model picker fell
back to a plain text input — users had to memorize names like
'deepseek-chat' vs 'deepseek-reasoner'. The dropdown change makes
DeepSeek's model UX match Anthropic + Gemini.

These tests verify the catalog + the dropdown wiring without spinning
up a real Streamlit session.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── Catalog ────────────────────────────────────────────────────────────────

def test_config_exposes_deepseek_known_models():
    import importlib, config
    importlib.reload(config)
    catalog = getattr(config, "DEEPSEEK_KNOWN_MODELS", None)
    assert isinstance(catalog, list), "DEEPSEEK_KNOWN_MODELS missing or wrong type"
    assert len(catalog) >= 4, "catalog too small — need at least chat / reasoner / versions"


def test_deepseek_catalog_contains_canonical_aliases():
    """deepseek-chat (non-reasoning, default) and deepseek-reasoner
    (R1) are the two production aliases users will pick most often."""
    import importlib, config
    importlib.reload(config)
    catalog = config.DEEPSEEK_KNOWN_MODELS
    assert "deepseek-chat" in catalog
    assert "deepseek-reasoner" in catalog


def test_deepseek_default_model_is_in_catalog():
    """The default DeepSeek model (`config._DEFAULT_MODELS['deepseek']`)
    must appear in the catalog — otherwise the dropdown falls through to
    'Custom…' on a fresh install."""
    import importlib, config
    importlib.reload(config)
    default = config._DEFAULT_MODELS.get("deepseek")
    assert default in config.DEEPSEEK_KNOWN_MODELS, (
        f"default model {default!r} not in catalog — UI would "
        f"open on Custom… instead of the curated entry"
    )


def test_deepseek_remains_in_supported_providers():
    """Regression: DeepSeek must stay in SUPPORTED_PROVIDERS so the
    provider dropdown still lists it."""
    import importlib, config
    importlib.reload(config)
    assert "deepseek" in config.SUPPORTED_PROVIDERS


def test_deepseek_has_cost_rates():
    """Cost-rate row is required for the rate metric in the panel."""
    import importlib, config
    importlib.reload(config)
    rates = config.COST_RATES.get("deepseek")
    assert rates is not None
    assert "input" in rates and "output" in rates
    assert rates["input"] > 0.0 and rates["output"] > 0.0


# ── Admin panel renders a DeepSeek-specific dropdown branch ────────────────

def _make_st_stub():
    """Stub st module sufficient for _render_llm_provider_panel to run."""
    stub = MagicMock()

    def _make_col():
        c = MagicMock()
        c.button = MagicMock(return_value=False)
        c.metric = MagicMock()
        c.markdown = MagicMock()
        return c

    stub.columns.side_effect = lambda spec, **kw: (
        [_make_col() for _ in range(spec)]
        if isinstance(spec, int)
        else [_make_col() for _ in spec]
    )
    for attr in ("expander", "container", "form"):
        getattr(stub, attr).return_value.__enter__ = MagicMock(return_value=stub)
        getattr(stub, attr).return_value.__exit__ = MagicMock(return_value=None)
    stub.button.return_value = False
    stub.checkbox.return_value = False
    stub.text_input.return_value = ""
    stub.dataframe.return_value = None
    # The provider selectbox in the panel is keyed `admin_llm_provider`
    # and the deepseek model selectbox is keyed `admin_llm_deepseek_model`.
    # We return values via side_effect based on call kwargs/key.

    def _selectbox(label, options, index=0, format_func=None, key=None, **kw):
        return options[index] if 0 <= index < len(options) else (
            options[0] if options else None
        )

    stub.selectbox.side_effect = _selectbox
    stub.session_state = {}
    return stub


def test_admin_panel_renders_deepseek_model_dropdown_when_selected():
    """When the user picks deepseek as the provider, the panel should
    render the curated dropdown (selectbox keyed
    `admin_llm_deepseek_model`), NOT just a freeform text input."""
    import config
    import admin_dashboard
    # Force deepseek to be the active provider so the dropdown renders.
    config.PROVIDER = "deepseek"
    config.MODEL = "deepseek-chat"
    st = _make_st_stub()
    admin_dashboard._render_llm_provider_panel(st)
    selectbox_keys = [
        c.kwargs.get("key", "") for c in st.selectbox.call_args_list
    ]
    assert "admin_llm_deepseek_model" in selectbox_keys, (
        f"DeepSeek curated dropdown not rendered. Selectbox keys seen: "
        f"{selectbox_keys}"
    )


def test_admin_panel_offers_custom_option_for_deepseek():
    """The Custom… escape hatch should be in the options list so users
    can type a brand-new alias not yet in the catalog."""
    import config
    import admin_dashboard
    config.PROVIDER = "deepseek"
    config.MODEL = "deepseek-chat"
    st = _make_st_stub()
    admin_dashboard._render_llm_provider_panel(st)
    deepseek_calls = [
        c for c in st.selectbox.call_args_list
        if c.kwargs.get("key") == "admin_llm_deepseek_model"
    ]
    assert deepseek_calls, "DeepSeek selectbox not called"
    options = deepseek_calls[0].kwargs.get("options") or deepseek_calls[0].args[1]
    assert any("Custom" in opt for opt in options), (
        f"Custom… option missing from DeepSeek dropdown: {options}"
    )


def test_admin_panel_does_not_render_deepseek_dropdown_for_other_providers():
    """When provider is gemini / anthropic / etc., the DeepSeek
    selectbox should NOT render — verifying the elif branch isolates
    correctly."""
    import config
    import admin_dashboard
    config.PROVIDER = "anthropic"
    config.MODEL = "claude-sonnet-4-6"
    st = _make_st_stub()
    admin_dashboard._render_llm_provider_panel(st)
    selectbox_keys = [
        c.kwargs.get("key", "") for c in st.selectbox.call_args_list
    ]
    assert "admin_llm_deepseek_model" not in selectbox_keys
