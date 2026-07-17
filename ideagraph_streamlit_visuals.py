"""
ideagraph_streamlit_visuals.py — standalone Streamlit demo for the
visual-abstract renderer.

Run on its own (separate from the full IdeaGraph app):

    streamlit run ideagraph_streamlit_visuals.py --server.port 8511

Useful for sanity-checking the FLUX/Nano-Banana API key + provider before
wiring it into a larger pipeline. Not required when running the main app —
the same panel is already embedded in every idea card there.
"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from ideagraph_image_renderer import (
    NanoBananaImageRenderer,
    IdeaVisual,
    build_prompt,
    cache_key_for_prompt,
    display_idea_with_visual,
)


st.set_page_config(
    page_title="IdeaGraph — Visual Abstract Demo",
    page_icon="🎨",
    layout="wide",
)

st.title("🎨 IdeaGraph Visual Abstract Demo")
st.caption(
    "Sanity-check the FLUX / Nano Banana / Runway API key + provider "
    "before integrating into the full IdeaGraph pipeline. The same panel "
    "appears on every idea card in the main app."
)

# ── Sidebar: configuration ─────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")
    st.markdown(
        "Set `NANO_BANANA_API_KEY` (or `BFL_API_KEY`) in your environment "
        "or `.env`. You can also paste a key here for this session only."
    )
    session_key = st.text_input(
        "API key (session-only — not persisted)",
        type="password",
        value="",
        help="Overrides the environment key for this Streamlit session.",
    )
    model = st.text_input("Model", value="flux-pro-1.0")
    endpoint = st.text_input(
        "Endpoint",
        value="https://api.bfl.ml/v1",
        help="Default = BlackForest Labs. Override for Runway/Replicate/etc.",
    )
    cache_dir = st.text_input("Cache dir", value=".ideagraph_visual_cache")

    st.divider()
    st.markdown("**Effective key source**")
    if session_key:
        st.success("Session input (this tab only)")
    elif os.getenv("NANO_BANANA_API_KEY"):
        st.success("`NANO_BANANA_API_KEY` env var")
    elif os.getenv("BFL_API_KEY"):
        st.success("`BFL_API_KEY` env var")
    elif Path(".nanobang_config").exists():
        st.success("`.nanobang_config` file")
    else:
        st.warning("⚠️ No key found. Set one in the sidebar or your env.")


# ── Main: idea form ────────────────────────────────────────────────────────
st.subheader("Idea to visualize")

c1, c2 = st.columns([2, 1])
with c1:
    title = st.text_input(
        "Title",
        value="Linear attention via random feature maps",
    )
    method = st.text_area(
        "Method",
        value=(
            "Approximate softmax attention by projecting queries and keys "
            "through random feature maps, yielding O(n) per-token cost "
            "and a closed-form unbiased estimator."
        ),
        height=120,
    )
with c2:
    methodology_type = st.selectbox(
        "Methodology",
        options=[
            "empirical_study", "theoretical_analysis", "system_design",
            "dataset_creation", "survey_meta_analysis", "tool_library",
            "interdisciplinary_bridge",
        ],
        index=0,
    )
    novelty_level = st.selectbox(
        "Novelty level",
        options=["incremental", "moderate", "substantial"],
        index=1,
    )

idea = {
    "title": title.strip() or "(untitled idea)",
    "method": method.strip(),
    "methodology_type": methodology_type,
    "novelty_level": novelty_level,
}

# ── Prompt preview ─────────────────────────────────────────────────────────
with st.expander("📝 Prompt that will be sent", expanded=False):
    _prompt = build_prompt(idea)
    st.code(_prompt, language=None)
    st.caption(
        f"Cache key (sha256): `{cache_key_for_prompt(_prompt)[:16]}…`"
    )

# ── Generate ────────────────────────────────────────────────────────────────
go_col, force_col = st.columns([3, 1])
go = go_col.button(
    "🎨 Generate visual abstract", type="primary",
    use_container_width=True,
)
force = force_col.button(
    "↻ Force re-roll", use_container_width=True,
    help="Bypass cache and call the API again.",
)

if go or force:
    if not title.strip():
        st.error("Title is required.")
    else:
        renderer = NanoBananaImageRenderer(
            api_key=session_key,
            model=model,
            endpoint=endpoint,
            cache_dir=cache_dir,
        )
        if not renderer.is_configured:
            st.error(
                "No API key configured. Paste one in the sidebar, or "
                "set `NANO_BANANA_API_KEY` / `BFL_API_KEY` in your env."
            )
        else:
            with st.spinner(
                f"Calling {renderer.provider.name} (model={model})… "
                "this takes 10–30s for FLUX-pro."
            ):
                visual = renderer.render(idea, force=force)
            st.session_state["_demo_visual"] = visual


# ── Result ─────────────────────────────────────────────────────────────────
v: IdeaVisual = st.session_state.get("_demo_visual")
if v is not None:
    st.divider()
    st.subheader("Result")
    display_idea_with_visual(idea, st, visual=v, show_prompt=True)

    with st.expander("Debug — full IdeaVisual"):
        st.json(v.to_dict())
