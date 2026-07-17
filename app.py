"""
app.py - Streamlit UI for IdeaGraph.

Run with:
    streamlit run app.py

Architecture:
  - Pipeline runs in a background thread to keep the UI responsive.
  - Progress messages are communicated via a thread-safe queue stored in st.session_state.
  - Results are stored in st.session_state once the pipeline completes.
  - User accounts and saved results stored in SQLite via db.py.
"""

from __future__ import annotations
import json
import os
import queue
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="IdeaGraph — AI-Powered Research Ideation Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "**IdeaGraph** generates diverse, publication-grade research "
            "ideas via Quality-Diversity optimization, multi-LLM "
            "ensembles, and 19 novelty-augmentation modes (TRIZ "
            "contradictions, persona swap, corpus-anchored novelty, "
            "and more). Built for PhD students and research groups."
        ),
        "Report a bug": "https://github.com/anthropics/claude-code/issues",
    },
)

# ── SEO meta + viewport for mobile / iPad / laptop ────────────────────────────
# IMPORTANT: viewport allows pinch-zoom (no `maximum-scale=1.0, user-scalable=no`)
# — accessibility best practice. Streamlit's default viewport already covers
# width=device-width; we add SEO meta + responsive theme-color + OG/Twitter
# cards via a single HTML block so any future hosted deployment gets crawled
# cleanly and previews nicely on Twitter / LinkedIn / Slack / Discord.
_SEO_DESCRIPTION = (
    "IdeaGraph is an AI-powered research ideation platform that generates "
    "diverse, publication-grade research ideas using Quality-Diversity "
    "optimization, multi-LLM ensembles, and 19 novelty-augmentation modes — "
    "for PhD students and research groups."
)
st.markdown(
    f"""
    <meta name="description" content="{_SEO_DESCRIPTION}">
    <meta name="keywords" content="research ideation, AI, LLM, quality-diversity, MAP-Elites, novelty search, PhD, research assistant, idea generation, scientific writing">
    <meta name="author" content="IdeaGraph">
    <meta name="robots" content="index, follow">
    <meta name="theme-color" content="#0ea5e9">
    <meta name="color-scheme" content="light">

    <!-- Open Graph (Facebook, LinkedIn, Slack, Discord) -->
    <meta property="og:type" content="website">
    <meta property="og:title" content="IdeaGraph — AI-Powered Research Ideation Platform">
    <meta property="og:description" content="{_SEO_DESCRIPTION}">
    <meta property="og:site_name" content="IdeaGraph">
    <meta property="og:locale" content="en_US">

    <!-- Twitter Card -->
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:title" content="IdeaGraph — AI-Powered Research Ideation Platform">
    <meta name="twitter:description" content="{_SEO_DESCRIPTION}">

    <!-- Apple touch icons for iPad / iPhone home-screen install -->
    <link rel="apple-touch-icon" sizes="180x180" href="data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🧠</text></svg>">
    <meta name="apple-mobile-web-app-title" content="IdeaGraph">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">

    <!-- Format detection: don't auto-link random numbers in body text as phones -->
    <meta name="format-detection" content="telephone=no">

    <!-- JSON-LD structured data (schema.org WebApplication) — gives
         Google rich snippets, sitelinks, and Knowledge Graph entries. -->
    <script type="application/ld+json">
    {{
      "@context": "https://schema.org",
      "@type": "WebApplication",
      "name": "IdeaGraph",
      "applicationCategory": "ResearchApplication",
      "operatingSystem": "Web (Chrome, Firefox, Safari, Edge)",
      "description": "{_SEO_DESCRIPTION}",
      "url": "https://ideagraph.app/",
      "inLanguage": "en",
      "audience": {{
        "@type": "Audience",
        "audienceType": "PhD students, research groups, academic researchers"
      }},
      "featureList": [
        "Quality-Diversity research ideation (MAP-Elites grid)",
        "Multi-LLM ensemble across 7 providers",
        "25 publication-grade sort + group modes",
        "19 Novelty Lab modes including corpus-anchored novelty",
        "Adversarial novelty critic",
        "Chat-to-optimize per-idea refinement",
        "Pareto-front + lineage tree analysis",
        "Mobile / iPad / laptop responsive UI"
      ],
      "offers": {{
        "@type": "Offer",
        "price": "0",
        "priceCurrency": "USD",
        "availability": "https://schema.org/InStock"
      }}
    }}
    </script>

    <!-- Skip-to-content link for keyboard / screen-reader users.
         Hidden visually until focused, then jumps to the main content. -->
    <a href="#main-content" class="skip-to-content">Skip to main content</a>

    <!-- Plain-text fallback for users with JavaScript disabled.
         Streamlit can't render without JS, so be explicit about it. -->
    <noscript>
      <div style="padding:24px;font-family:sans-serif;text-align:center">
        <h1>IdeaGraph requires JavaScript</h1>
        <p>IdeaGraph is a Streamlit application that uses JavaScript to
        render its interface. Please enable JavaScript in your browser
        settings, or visit our documentation for command-line usage.</p>
      </div>
    </noscript>
    """,
    unsafe_allow_html=True,
)

# ── Custom CSS: white + sky blue theme ───────────────────────────────────────
st.markdown("""<style>
/* ── Skip-to-content link (a11y): visually hidden until focused ─────────── */
.skip-to-content {
    position: absolute;
    top: -100px;
    left: 0;
    background: #0c4a6e;
    color: #fff;
    padding: 12px 20px;
    text-decoration: none;
    font-weight: 600;
    border-bottom-right-radius: 8px;
    z-index: 10000;
    transition: top 0.15s ease;
}
.skip-to-content:focus,
.skip-to-content:focus-visible {
    top: 0;
    outline: 3px solid #0ea5e9;
    outline-offset: 2px;
}

/* ── Focus indicators (keyboard users) — visible everywhere ─────────────── */
button:focus-visible,
[role="tab"]:focus-visible,
summary:focus-visible,
a:focus-visible,
input:focus-visible,
textarea:focus-visible,
select:focus-visible {
    outline: 3px solid #0ea5e9 !important;
    outline-offset: 2px !important;
    border-radius: 4px;
}

/* ── Main landmark anchor target (for skip-to-content) ───────────────────── */
#main-content {
    scroll-margin-top: 16px;
}

/* ── Sidebar: sky blue gradient ──────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #e0f2fe 0%, #f0f9ff 100%) !important;
    border-right: 1px solid #bae6fd;
}

/* ── Buttons: sky blue ───────────────────────────────────────────────────── */
.stButton > button[kind="primary"] {
    background: #0ea5e9 !important;
    border: none !important;
    border-radius: 8px !important;
    color: white !important;
    font-weight: 600 !important;
    transition: all 0.2s ease;
}
.stButton > button[kind="primary"]:hover {
    background: #0284c7 !important;
    box-shadow: 0 4px 12px rgba(14,165,233,0.3);
}
.stButton > button[kind="secondary"] {
    border-radius: 8px !important;
    border: 1px solid #bae6fd !important;
    color: #0369a1 !important;
    background: white !important;
}
.stButton > button[kind="secondary"]:hover {
    background: #f0f9ff !important;
    border-color: #0ea5e9 !important;
}

/* ── Expanders (idea cards): soft sky border ──────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid #e0f2fe !important;
    border-radius: 10px !important;
    margin-bottom: 8px !important;
    background: white;
    transition: box-shadow 0.2s ease;
}
[data-testid="stExpander"]:hover {
    box-shadow: 0 2px 12px rgba(14,165,233,0.12);
    border-color: #7dd3fc !important;
}

/* ── Metrics: sky card style ─────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #f0f9ff;
    border-radius: 10px;
    padding: 10px 14px;
    border: 1px solid #e0f2fe;
}
[data-testid="stMetric"] label {
    color: #0369a1 !important;
    font-size: 0.72rem !important;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #0c4a6e !important;
    font-weight: 700;
}

/* ── Tabs: sky underline ─────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    border-bottom: 2px solid #e0f2fe;
}
.stTabs [data-baseweb="tab"] {
    padding: 8px 14px !important;
    font-size: 0.84rem !important;
    color: #64748b;
}
.stTabs [aria-selected="true"] {
    color: #0284c7 !important;
    border-bottom: 2px solid #0ea5e9 !important;
    font-weight: 600;
}

/* ── Inputs: clean white with sky focus ───────────────────────────────────── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    border-radius: 8px !important;
    border: 1px solid #cbd5e1 !important;
    background: white !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 3px rgba(56,189,248,0.15) !important;
}

/* ── Progress bar: sky gradient ──────────────────────────────────────────── */
.stProgress > div > div {
    background: linear-gradient(90deg, #38bdf8, #0ea5e9) !important;
    border-radius: 4px;
}

/* ── Alerts ──────────────────────────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 8px !important;
}

/* ── Dividers ────────────────────────────────────────────────────────────── */
hr {
    border-color: #e0f2fe !important;
}

/* ── Scrollbar: subtle ───────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #bae6fd; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #7dd3fc; }

/* ── Container borders (st.container(border=True)) ───────────────────────── */
[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid #e0f2fe !important;
    border-radius: 10px !important;
    background: #fafeff;
}

/* ── Selectbox & multiselect ─────────────────────────────────────────────── */
[data-baseweb="select"] > div {
    border-radius: 8px !important;
    border-color: #bae6fd !important;
}

/* ── Checkbox ────────────────────────────────────────────────────────────── */
[data-testid="stCheckbox"] label span[data-testid="stMarkdownContainer"] {
    font-size: 0.9rem;
}

/* ── Download button ─────────────────────────────────────────────────────── */
[data-testid="stDownloadButton"] button {
    border-radius: 8px !important;
    border: 1px solid #bae6fd !important;
    color: #0369a1 !important;
    font-weight: 500;
    background: #f0f9ff !important;
}
[data-testid="stDownloadButton"] button:hover {
    background: #e0f2fe !important;
}

/* ── Slider ──────────────────────────────────────────────────────────────── */
[data-testid="stSlider"] [role="slider"] {
    background: #0ea5e9 !important;
}

/* ── Expander header font ────────────────────────────────────────────────── */
[data-testid="stExpander"] summary span {
    font-weight: 500;
}

/* ── Toast/success/info messages ─────────────────────────────────────────── */
[data-testid="stAlert"][data-baseweb="notification"] {
    border-radius: 8px !important;
}

/* ── Hide clutter ────────────────────────────────────────────────────────── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* ── Pulse animation ─────────────────────────────────────────────────────── */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}
.running-indicator {
    animation: pulse 1.5s ease-in-out infinite;
    color: #0ea5e9;
    font-weight: 600;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* MOBILE RESPONSIVE (< 768px)                                                */
/* ═══════════════════════════════════════════════════════════════════════════ */
@media (max-width: 768px) {
    /* ── Shrink padding on main container ─────────────────────────────────── */
    .main .block-container {
        padding: 1rem 0.8rem !important;
    }

    /* ── Sidebar collapses by default — make toggle area bigger ───────────── */
    [data-testid="stSidebar"] {
        min-width: 260px !important;
        max-width: 280px !important;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding: 0.8rem !important;
    }

    /* ── Stack columns vertically on mobile ──────────────────────────────── */
    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        flex: 1 1 100% !important;
        min-width: 100% !important;
        width: 100% !important;
    }

    /* ── Metrics: smaller on mobile ──────────────────────────────────────── */
    [data-testid="stMetric"] {
        padding: 8px 10px;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.1rem !important;
    }
    [data-testid="stMetric"] label {
        font-size: 0.65rem !important;
    }

    /* ── Tabs: scroll horizontally, smaller text ─────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        overflow-x: auto !important;
        -webkit-overflow-scrolling: touch;
        scrollbar-width: none;
        flex-wrap: nowrap !important;
    }
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
        display: none;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.75rem !important;
        padding: 6px 10px !important;
        white-space: nowrap;
        flex-shrink: 0;
    }

    /* ── Expanders: full width, smaller text ──────────────────────────────── */
    [data-testid="stExpander"] {
        border-radius: 8px !important;
    }
    [data-testid="stExpander"] summary {
        font-size: 0.82rem !important;
        padding: 10px 12px !important;
    }

    /* ── Buttons: larger tap targets ─────────────────────────────────────── */
    .stButton > button {
        min-height: 44px !important;
        font-size: 0.85rem !important;
    }

    /* ── Text inputs: larger for thumb typing ─────────────────────────────── */
    [data-testid="stTextInput"] input {
        min-height: 44px !important;
        font-size: 16px !important;  /* prevents iOS zoom on focus */
    }
    [data-testid="stTextArea"] textarea {
        font-size: 16px !important;
    }

    /* ── Plotly charts: shrink height on mobile ───────────────────────────── */
    .js-plotly-plot {
        max-height: 280px !important;
    }

    /* ── Download buttons: stack vertically ───────────────────────────────── */
    [data-testid="stDownloadButton"] {
        width: 100% !important;
    }
    [data-testid="stDownloadButton"] button {
        width: 100% !important;
        min-height: 44px !important;
    }

    /* ── Probe score bars: 2 columns instead of long row ─────────────────── */
    /* Handled by flex-wrap in the inline HTML */

    /* ── Hide less-important elements on mobile ──────────────────────────── */
    .desktop-only {
        display: none !important;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* SMALL MOBILE (< 480px)                                                     */
/* ═══════════════════════════════════════════════════════════════════════════ */
@media (max-width: 480px) {
    .main .block-container {
        padding: 0.5rem 0.5rem !important;
    }

    /* ── Even smaller metrics ─────────────────────────────────────────────── */
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 0.95rem !important;
    }

    /* ── Tabs: even smaller ──────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab"] {
        font-size: 0.7rem !important;
        padding: 5px 8px !important;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* TABLET (768px - 1024px)                                                    */
/* ═══════════════════════════════════════════════════════════════════════════ */
@media (min-width: 769px) and (max-width: 1024px) {
    .main .block-container {
        padding: 1.5rem 1.2rem !important;
    }

    /* ── 2-column layout instead of 4-5 ──────────────────────────────────── */
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        min-width: 48% !important;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* TOUCH-FRIENDLY EVERYWHERE                                                  */
/* ═══════════════════════════════════════════════════════════════════════════ */
/* Larger checkbox/radio tap areas */
[data-testid="stCheckbox"] {
    padding: 4px 0;
}
[data-testid="stCheckbox"] label {
    min-height: 36px;
    display: flex;
    align-items: center;
}

/* Smooth scroll on all containers */
[data-testid="stAppViewContainer"],
[data-testid="stSidebar"] {
    scroll-behavior: smooth;
    -webkit-overflow-scrolling: touch;
}

/* Prevent text selection flash on rapid taps */
button, [role="tab"], summary {
    -webkit-tap-highlight-color: transparent;
    user-select: none;
}

/* ─────────────────────────────────────────────────────────────────────── */
/* ENHANCED POLISH                                                         */
/* ─────────────────────────────────────────────────────────────────────── */

/* ── Headings: tight, deep navy ─────────────────────────────────────────── */
h1, h2, h3 {
    color: #0c4a6e !important;
    letter-spacing: -0.02em;
    font-weight: 700 !important;
}
h1 { font-size: 1.8rem !important; }
h2 { font-size: 1.4rem !important; }
h3 { font-size: 1.15rem !important; }

/* ── Subtle entrance animation for cards ─────────────────────────────────── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}
[data-testid="stExpander"] {
    animation: fadeInUp 0.3s ease-out;
}

/* ── Sidebar slider track styling ────────────────────────────────────────── */
[data-testid="stSidebar"] [data-testid="stSlider"] {
    padding: 4px 0;
}
[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
    background: #0ea5e9 !important;
    border: 2px solid white !important;
    box-shadow: 0 1px 3px rgba(14,165,233,0.4);
}

/* ── Sidebar select-slider track ────────────────────────────────────────── */
[data-testid="stSidebar"] [data-baseweb="slider"] > div > div {
    background: #bae6fd !important;
}
[data-testid="stSidebar"] [data-baseweb="slider"] > div > div > div {
    background: #0ea5e9 !important;
}

/* ── Sidebar selectbox dropdown ──────────────────────────────────────────── */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: white !important;
    border: 1px solid #bae6fd !important;
    border-radius: 8px !important;
}

/* ── Sidebar checkbox styling ────────────────────────────────────────────── */
[data-testid="stSidebar"] [data-testid="stCheckbox"] {
    padding: 4px 6px;
    border-radius: 6px;
    transition: background 0.15s;
}
[data-testid="stSidebar"] [data-testid="stCheckbox"]:hover {
    background: rgba(14,165,233,0.08);
}
[data-testid="stSidebar"] [data-testid="stCheckbox"] label {
    font-size: 0.82rem !important;
}

/* ── Sidebar text labels ────────────────────────────────────────────────── */
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
    font-size: 0.78rem !important;
    font-weight: 600;
    color: #075985 !important;
}

/* ── Buttons: smoother shadow + active state ─────────────────────────────── */
.stButton > button {
    transition: all 0.18s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
.stButton > button[kind="primary"]:active {
    transform: translateY(1px);
    box-shadow: 0 1px 4px rgba(14,165,233,0.4);
}

/* ── Primary button: subtle gradient ─────────────────────────────────────── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%) !important;
}

/* ── Caption text: warmer color ──────────────────────────────────────────── */
[data-testid="stCaptionContainer"], .stCaption, small {
    color: #64748b !important;
}

/* ── Code blocks: sky tinted ─────────────────────────────────────────────── */
code, [data-testid="stCode"] {
    background: #f0f9ff !important;
    color: #0c4a6e !important;
    border-radius: 4px;
    padding: 1px 6px;
    font-size: 0.85em !important;
}
pre code {
    border: 1px solid #e0f2fe;
    padding: 8px 12px;
}

/* ── Markdown bold: navy weight ──────────────────────────────────────────── */
strong, b {
    color: #0c4a6e;
    font-weight: 700;
}

/* ── Info/success/warning/error: rounded with accent bars ────────────────── */
.stAlert > div {
    border-radius: 10px !important;
    border-left-width: 4px !important;
}

/* ── Plotly charts: clean container ──────────────────────────────────────── */
.js-plotly-plot {
    border-radius: 10px;
    overflow: hidden;
}

/* ── Tooltip / help icon ─────────────────────────────────────────────────── */
[data-testid="stTooltipIcon"] svg {
    fill: #0ea5e9 !important;
}

/* ── Form submit button: full sky ────────────────────────────────────────── */
[data-testid="stFormSubmitButton"] button {
    border-radius: 8px !important;
    font-weight: 600;
}

/* ── Data table styling ──────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #e0f2fe;
}

/* ── Code copy button ────────────────────────────────────────────────────── */
[data-testid="stCodeCopyButton"] {
    border-radius: 6px;
    background: rgba(14,165,233,0.1) !important;
}

/* ── Number input ────────────────────────────────────────────────────────── */
[data-testid="stNumberInput"] input {
    border-radius: 8px !important;
    border: 1px solid #cbd5e1 !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 3px rgba(56,189,248,0.15) !important;
}

/* ── Color picker, date input ────────────────────────────────────────────── */
[data-testid="stColorPicker"] > div,
[data-testid="stDateInput"] > div > div {
    border-radius: 8px !important;
}

/* ── Subtle hover on stCaption text ──────────────────────────────────────── */
.stCaption:hover { color: #475569 !important; }

/* ── Spinner sky color ──────────────────────────────────────────────────── */
[data-testid="stSpinner"] > div > i {
    border-top-color: #0ea5e9 !important;
}

/* ── Radio buttons: pill style ──────────────────────────────────────────── */
[data-testid="stRadio"] label {
    background: white;
    border: 1px solid #bae6fd;
    border-radius: 20px;
    padding: 4px 12px;
    margin: 2px 4px 2px 0;
    transition: all 0.15s;
}
[data-testid="stRadio"] label:hover {
    background: #f0f9ff;
}

/* ── Empty containers: subtle dotted background ──────────────────────────── */
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"]:empty {
    background-image: radial-gradient(#bae6fd 1px, transparent 1px);
    background-size: 12px 12px;
    min-height: 40px;
}

/* ── Tab content padding ─────────────────────────────────────────────────── */
[data-baseweb="tab-panel"] {
    padding-top: 16px !important;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* MODERN RESULTS DISPLAY                                                      */
/* ═══════════════════════════════════════════════════════════════════════════ */

/* ── Idea card: glassmorphism vibe ──────────────────────────────────────── */
[data-testid="stExpander"] {
    background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(240,249,255,0.9)) !important;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
}

/* ── Idea card content: tighter spacing ──────────────────────────────────── */
[data-testid="stExpander"] [data-testid="stVerticalBlock"] {
    gap: 0.5rem !important;
}

/* ── Polished metric values ──────────────────────────────────────────────── */
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-feature-settings: "tnum"; /* tabular numbers for alignment */
    font-variant-numeric: tabular-nums;
}

/* ── Badges (colored chips used everywhere) ──────────────────────────────── */
.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.02em;
    margin-right: 4px;
}

/* ── Stats blocks: subtle hover lift ─────────────────────────────────────── */
div[style*="border-radius:10px"]:not([data-testid]) {
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* DEVICE-SPECIFIC OPTIMIZATIONS                                               */
/* ═══════════════════════════════════════════════════════════════════════════ */

/* ── Large desktop (1440px+): use generous space ─────────────────────────── */
@media (min-width: 1441px) {
    .main .block-container {
        max-width: 1280px !important;
        padding: 2rem 2.5rem !important;
    }
    h1 { font-size: 2.2rem !important; }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
    }
}

/* ── Standard desktop (1025-1440px) ──────────────────────────────────────── */
@media (min-width: 1025px) and (max-width: 1440px) {
    .main .block-container {
        max-width: 1100px !important;
        padding: 1.5rem 2rem !important;
    }
}

/* ── Tablet portrait (601-768px) ─────────────────────────────────────────── */
@media (min-width: 601px) and (max-width: 768px) {
    /* 2-column metrics row */
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        flex: 1 1 48% !important;
        min-width: 48% !important;
    }
    /* Compact tabs */
    .stTabs [data-baseweb="tab"] {
        font-size: 0.8rem !important;
        padding: 8px 10px !important;
    }
    /* Idea cards full-width */
    [data-testid="stExpander"] {
        margin-left: -4px;
        margin-right: -4px;
    }
}

/* ── Mobile landscape (481-600px) ────────────────────────────────────────── */
@media (min-width: 481px) and (max-width: 600px) {
    .main .block-container {
        padding: 0.75rem 0.6rem !important;
    }
    h1 { font-size: 1.4rem !important; }
    h2 { font-size: 1.15rem !important; }
    h3 { font-size: 1rem !important; }
    [data-testid="stMetric"] {
        padding: 6px 10px;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1rem !important;
    }
}

/* ── Phone portrait (320-480px) ──────────────────────────────────────────── */
@media (max-width: 480px) {
    /* Stack everything */
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        flex: 1 1 100% !important;
        min-width: 100% !important;
        margin-bottom: 4px;
    }
    /* Mini titles */
    h1 { font-size: 1.3rem !important; }
    h2 { font-size: 1.1rem !important; }
    h3 { font-size: 0.95rem !important; }
    /* Mini metrics */
    [data-testid="stMetric"] {
        padding: 6px 8px;
    }
    [data-testid="stMetric"] label {
        font-size: 0.6rem !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 0.9rem !important;
    }
    /* Tabs scrollable horizontal */
    .stTabs [data-baseweb="tab-list"] {
        overflow-x: auto !important;
        white-space: nowrap;
        flex-wrap: nowrap !important;
    }
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
        display: none;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.7rem !important;
        padding: 6px 8px !important;
        flex-shrink: 0;
    }
    /* Sidebar narrower */
    [data-testid="stSidebar"] {
        min-width: 240px !important;
        max-width: 260px !important;
    }
    /* Reduce expander padding */
    [data-testid="stExpander"] summary {
        padding: 8px 10px !important;
        font-size: 0.78rem !important;
    }
    /* Idea card metric row stacks */
    [data-testid="stExpander"] [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
        gap: 4px !important;
    }
    /* Hide help icons on mobile (clutter) */
    [data-testid="stTooltipIcon"] {
        display: none !important;
    }
    /* Buttons shrink text but keep tap target */
    .stButton > button {
        font-size: 0.78rem !important;
        padding: 8px 10px !important;
    }
    /* Code blocks: smaller font, scroll if long */
    pre, code {
        font-size: 0.75em !important;
        overflow-x: auto !important;
    }
}

/* ── Tiny phones (under 360px) ───────────────────────────────────────────── */
@media (max-width: 360px) {
    .main .block-container {
        padding: 0.5rem 0.4rem !important;
    }
    h1 { font-size: 1.1rem !important; }
    h2 { font-size: 1rem !important; }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 0.85rem !important;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.65rem !important;
        padding: 4px 6px !important;
    }
}

/* ── Landscape phone (height < 500px): minimize vertical padding ─────────── */
@media (max-height: 500px) and (orientation: landscape) {
    .main .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
    }
    h1 { font-size: 1.2rem !important; }
}

/* ── Print styles ────────────────────────────────────────────────────────── */
@media print {
    [data-testid="stSidebar"], .stButton, [data-testid="stDownloadButton"] {
        display: none !important;
    }
    .main .block-container {
        max-width: 100% !important;
        padding: 0 !important;
    }
    [data-testid="stExpander"] {
        page-break-inside: avoid;
        background: white !important;
    }
}

/* ── Reduce motion accessibility ─────────────────────────────────────────── */
@media (prefers-reduced-motion: reduce) {
    * {
        animation-duration: 0.01ms !important;
        transition-duration: 0.01ms !important;
    }
}

/* ── High contrast mode ──────────────────────────────────────────────────── */
@media (prefers-contrast: high) {
    [data-testid="stExpander"] {
        border-width: 2px !important;
    }
    .stButton > button {
        border-width: 2px !important;
    }
}

/* ── Mobile-friendly inline HTML widgets ─────────────────────────────────── */
@media (max-width: 768px) {
    /* Results banner: stack text */
    div[style*="display:flex"][style*="justify-content:space-between"] {
        flex-direction: column !important;
        gap: 4px;
    }

    /* Phase badges: 2x2 grid instead of 4 columns */
    /* Handled by the column stacking above */

    /* Code blocks (referral links): wrap */
    pre, code {
        word-break: break-all !important;
        font-size: 12px !important;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* iPad portrait (768-820 wide) — bridges the 768 / 1024 gap                  */
/* ═══════════════════════════════════════════════════════════════════════════ */
@media (min-width: 768px) and (max-width: 834px) and (orientation: portrait) {
    .main .block-container {
        padding: 1.2rem 1rem !important;
        max-width: 100% !important;
    }
    /* iPad portrait: 2-column where 3+ requested */
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(n+3) {
        flex: 1 1 100% !important;
        min-width: 100% !important;
        margin-top: 8px;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* Accessibility — respect prefers-reduced-motion                              */
/* ═══════════════════════════════════════════════════════════════════════════ */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.001ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.001ms !important;
        scroll-behavior: auto !important;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* Accessibility — high contrast mode                                          */
/* ═══════════════════════════════════════════════════════════════════════════ */
@media (prefers-contrast: more) {
    [data-testid="stExpander"] { border-width: 2px !important; }
    .stButton > button { border-width: 2px !important; }
    /* Stronger text colors */
    .stMarkdown, p { color: #000 !important; }
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* Print stylesheet — clean export of idea archive                             */
/* ═══════════════════════════════════════════════════════════════════════════ */
@media print {
    /* Hide chrome */
    [data-testid="stSidebar"],
    [data-testid="stHeader"],
    .stTabs [data-baseweb="tab-list"],
    .stButton,
    [data-testid="stDownloadButton"],
    [data-testid="stSpinner"] {
        display: none !important;
    }
    /* Flatten layout */
    .main .block-container {
        max-width: 100% !important;
        padding: 0 !important;
    }
    /* Expand all collapsed content for the print */
    [data-testid="stExpander"] details {
        display: block !important;
    }
    [data-testid="stExpander"] details > div {
        display: block !important;
    }
    /* Black text, white background */
    body, .stMarkdown, p, h1, h2, h3, h4 {
        color: #000 !important;
        background: #fff !important;
    }
    /* Avoid page breaks inside idea cards */
    [data-testid="stExpander"] {
        break-inside: avoid;
        page-break-inside: avoid;
        border: 1px solid #999 !important;
        margin-bottom: 8px;
    }
    /* Page setup */
    @page {
        size: A4;
        margin: 1.5cm 1.2cm;
    }
}
</style>
""", unsafe_allow_html=True)

# ── Database init ─────────────────────────────────────────────────────────────
import db
import db_cache  # @st.cache_data wrappers for read-only db calls
import auth_ui

db.init_db()

# ── Session state initialisation ──────────────────────────────────────────────
def _init_state() -> None:
    defaults = {
        "running": False,
        "done": False,
        "results": None,
        "progress_queue": queue.Queue(),
        "progress_log": deque(maxlen=500),      # bounded to cap memory
        "progress_events": deque(maxlen=200),    # pre-parsed __EVENT__ dicts
        "progress_text": deque(maxlen=500),      # plain-text messages
        "error": None,
        "result_saved": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ── Handle public URL params (share links, landing page) ──────────────────────
query_params = st.query_params

# Public share page (no login required)
if "share" in query_params:
    from sharing import render_shared_idea_page
    token = query_params["share"]
    render_shared_idea_page(token)
    st.stop()

# Referral code — stash for after registration
if "ref" in query_params:
    st.session_state["_referral_code"] = query_params["ref"]

# ── Stripe Checkout return flow ──────────────────────────────────────────────
# When the user returns from Stripe with ?checkout=success&session_id=...,
# poll Stripe once to verify the payment, then persist the new tier via
# db.update_subscription(). Clear the param so a refresh doesn't re-fire.
# (We don't gate this on logged_in — the user is still logged in via the
# session-recovery token; Stripe just bounced their browser back here.)
_checkout_flag = query_params.get("checkout", "")
if _checkout_flag == "success":
    _session_id = query_params.get("session_id", "")
    try:
        import stripe_integration as _si
        if _session_id:
            _verify = _si.verify_and_apply_checkout(_session_id)
            if _verify.get("ok"):
                st.session_state["_stripe_just_paid"] = {
                    "tier": _verify.get("tier"),
                    "session_id": _session_id,
                }
            else:
                st.session_state["_stripe_error"] = _verify.get(
                    "error", "Checkout verification failed."
                )
        else:
            st.session_state["_stripe_error"] = (
                "Stripe returned no session_id — couldn't verify payment."
            )
    except Exception as _e:
        st.session_state["_stripe_error"] = f"Stripe verification crashed: {_e}"
    # Strip the params so a refresh doesn't loop.
    try:
        st.query_params.clear()
    except Exception:
        pass
elif _checkout_flag == "cancel":
    st.session_state["_stripe_error"] = (
        "Checkout canceled — your plan was not changed."
    )
    try:
        st.query_params.clear()
    except Exception:
        pass

# Landing page (no login required) — shown if ?landing=1 or user not logged in + no auth form
if query_params.get("landing") == "1":
    from landing import render_landing_page
    render_landing_page()
    st.stop()

# ── Auth gate ─────────────────────────────────────────────────────────────────
if not auth_ui.is_logged_in():
    # Show landing page by default, with auth form toggle
    if st.session_state.get("_show_auth"):
        auth_ui.show_auth_page()
    else:
        from landing import render_landing_page
        render_landing_page()
    st.stop()


# ── Background pipeline thread ────────────────────────────────────────────────
def _run_pipeline_thread(
    topic: str, budget: float, iterations: int,
    provider: str, model: str,
    progress_queue: queue.Queue,
    debate_enabled: bool = False,
    user_id: Optional[int] = None,
    runtime_controller: Optional[Any] = None,
    release_once: Optional[Any] = None,
) -> None:
    """Runs in a daemon thread. Auto-saves results on completion.

    `release_once` is a run-scoped callback built by `_make_release_once`
    in the main thread and passed in. The worker MUST call it in its
    finally block to guarantee the slot is freed even when the main
    thread's `_drain_queue` never runs (tab close / browser refresh
    mid-run). It's idempotent — if the main thread already fired it
    via `_release_run`, this call no-ops.
    """
    _uid = user_id
    results = None
    _success = False

    def _safe_queue_put(item):
        try:
            progress_queue.put(item, block=False)
        except Exception:
            pass

    try:
        import config as _cfg
        if provider:
            _cfg.PROVIDER = provider
        if model:
            _cfg.MODEL = model

        from pipeline import IdeaGraphPipeline

        def on_progress(msg: str) -> None:
            _safe_queue_put(("progress", msg))

        pipeline = IdeaGraphPipeline()
        results = pipeline.run(
            topic=topic,
            budget_usd=budget,
            max_iterations=iterations,
            on_progress=on_progress,
            debate_enabled=debate_enabled,
            user_id=user_id,
            runtime_controller=runtime_controller,
        )

        # Auto-save to DB (survives session expiry)
        if _uid and results and results.get("ideas"):
            try:
                db.save_result(
                    user_id=_uid, topic=topic,
                    coverage=results.get("coverage", 0.0),
                    ideas_count=len(results.get("ideas", [])),
                    results_dict=results,
                )
                results["_auto_saved"] = True
                try:
                    db_cache.invalidate_user_results()
                except Exception:
                    pass
            except Exception:
                pass

        # Match legacy main-thread drain billing semantic — see
        # _run_produced_ideas. Single source of truth keeps worker,
        # drain, and tests from drifting apart.
        _success = _run_produced_ideas(results)
        _safe_queue_put(("done", results))
    except Exception as exc:
        _safe_queue_put(("error", str(exc)))
    finally:
        # ── Run-scoped release: closes the tab-close leak WITHOUT
        # double-decrementing per-user / global counters across
        # concurrent runs. The callback's internal lock+flag ensures
        # only the first caller (worker finally vs main-thread
        # _release_run) does the actual release. See _make_release_once.
        if release_once is not None:
            try:
                release_once(success=_success)
            except Exception:
                pass


def _preflight_run(
    topic: str, budget: float, iterations: int,
) -> Optional[tuple]:
    """
    Run all production-gates before launching a pipeline:
      1. Input validation (rejects prompt-injection + DoS payloads)
      2. Rate limiter (per-user + per-login-ratelimiter)
      3. Quota enforcement (atomic tier limit check + slot reservation)
      4. Concurrency guard (global + per-user pipeline caps)

    Returns sanitised (topic, budget, iterations) on success, or None
    (after showing the error via st.error) on failure.

    The caller MUST call _release_run(user_id, success=bool) on completion
    to unwind the quota reservation + concurrency slot.
    """
    from production_optimization import (
        InputValidationError,
        get_concurrency_guard,
        get_quota_enforcer,
        get_rate_limiter,
        validate_run_input,
    )
    user_id = st.session_state.get("user_id")

    # Determine tier (affects input caps).
    tier = "free"
    try:
        from stripe_integration import get_user_tier
        if user_id:
            tier = get_user_tier(user_id)
    except Exception:
        pass

    # 1. Input validation.
    try:
        topic, budget, iterations = validate_run_input(
            topic, budget, iterations, tier=tier,
        )
    except InputValidationError as exc:
        st.error(f"Invalid input: {exc}")
        return None

    # 2. Rate limit (per user, fallback to session id).
    rl_key = user_id or st.session_state.get("_rl_fallback")
    if rl_key is None:
        import uuid as _uuid
        rl_key = _uuid.uuid4().hex[:12]
        st.session_state["_rl_fallback"] = rl_key
    ok_rl, msg_rl = get_rate_limiter().check(user_id=rl_key)
    if not ok_rl:
        st.error(msg_rl)
        return None

    # 3. Quota enforcement (only if logged in with a tier).
    if user_id:
        ok_q, msg_q = get_quota_enforcer().try_acquire_run(user_id)
        if not ok_q:
            st.error(msg_q)
            return None
        st.session_state["_quota_acquired"] = True

    # 4. Concurrency guard.
    ok_c, msg_c = get_concurrency_guard().acquire(user_id=user_id)
    if not ok_c:
        # Roll back quota reservation if we got one.
        if user_id and st.session_state.get("_quota_acquired"):
            get_quota_enforcer().release_run(user_id, success=False)
            st.session_state.pop("_quota_acquired", None)
        st.error(msg_c)
        return None
    st.session_state["_concurrency_acquired"] = True
    return (topic, budget, iterations)


def _run_produced_ideas(results: Optional[Dict[str, Any]]) -> bool:
    """Single source of truth for "did this run produce billable output?"

    Used by _run_pipeline_thread, _run_scientist_thread, and _drain_queue
    so they cannot drift apart. Matches the legacy main-thread drain
    semantic: a run is only billed against the monthly quota when it
    returned at least one idea. Empty / None results are free.

    Specifically: returns True iff results is a dict containing a
    non-empty list under the 'ideas' key. A non-list under 'ideas'
    (e.g. an upstream bug producing an int) is treated as not-billable.
    """
    if not results or not isinstance(results, dict):
        return False
    ideas = results.get("ideas")
    if not isinstance(ideas, list):
        return False
    return len(ideas) > 0


def _release_run(success: bool) -> None:
    """Release quota + concurrency after pipeline completes.

    Run-scoped: each launched run owns a single callback closure (made
    by _make_release_once below) that holds a lock+flag. The callback
    is registered in session_state by _start_pipeline/_start_scientist
    AND captured by the worker thread. Whichever side fires first does
    the release; the other is a no-op. This prevents the double-release
    bleed that happens when both the worker-thread finally AND the
    main-thread drain release the same per-USER (not per-RUN) counter.

    Legacy `_quota_acquired` / `_concurrency_acquired` flags are popped
    here for backward-compat with any earlier session_state set before
    this code path existed — popping a missing key is a no-op.
    """
    cb = st.session_state.pop("_release_callback", None)
    if cb is not None:
        try:
            cb(success=success)
        except Exception:
            pass
    # Legacy cleanup — no-op if already-popped, and the callback above
    # is the actual release path for this build.
    st.session_state.pop("_quota_acquired", None)
    st.session_state.pop("_concurrency_acquired", None)


def _make_release_once(user_id: Optional[int]):
    """Build a run-scoped release callback.

    Returns a callable `release_once(success: bool)` that releases the
    concurrency slot + quota reservation for `user_id` EXACTLY ONCE.
    Subsequent calls are no-ops (lock-guarded `_done` flag).

    Both the worker thread (in its `finally`) and the main thread (in
    `_release_run`) call this callable. Whichever fires first does the
    real work; the other returns immediately. The result is run-scoped
    accounting — counters decrement once per actual run instead of
    once per release-call-site.
    """
    state = {"done": False}
    lock = threading.Lock()

    def _release_once(success: bool = False) -> None:
        with lock:
            if state["done"]:
                return
            state["done"] = True
        # Lock released — these calls are themselves thread-safe (each
        # guard has its own RLock). Don't hold our lock while calling
        # other modules to avoid creating a lock-order hazard.
        try:
            from production_optimization import (
                get_concurrency_guard, get_quota_enforcer,
            )
            if user_id is not None:
                try:
                    get_quota_enforcer().release_run(user_id, success=bool(success))
                except Exception:
                    pass
            try:
                get_concurrency_guard().release(user_id=user_id)
            except Exception:
                pass
        except Exception:
            pass

    return _release_once


def _start_pipeline(
    topic: str, budget: float, iterations: int,
    provider: str = "", model: str = "",
    debate_enabled: bool = False,
) -> None:
    preflight = _preflight_run(topic, budget, iterations)
    if preflight is None:
        return
    topic, budget, iterations = preflight

    st.session_state.running = True
    st.session_state.done = False
    st.session_state.results = None
    st.session_state.error = None
    st.session_state.progress_log = []
    st.session_state.progress_queue = queue.Queue()

    # Build a RuntimeController if the user opted into interactive control
    # (sidebar toggle). The controller is shared between the worker thread
    # and the main UI thread so the banner can show pause state.
    _ctrl = None
    if st.session_state.get("_runtime_control_enabled", True):
        try:
            from runtime_control import RuntimeController
            _ctrl = RuntimeController(
                budget_limit_usd=float(budget),
                budget_pause_threshold=float(
                    st.session_state.get("_runtime_budget_pause_threshold", 0.85)
                ),
                max_network_failures=int(
                    st.session_state.get("_runtime_max_network_failures", 4)
                ),
                decision_timeout_s=int(
                    st.session_state.get("_runtime_decision_timeout_s", 600)
                ),
                run_id=(f"{st.session_state.get('user_id', '')}:"
                          f"{int(time.time())}"),
            )
        except Exception:
            _ctrl = None
    st.session_state["_runtime_controller"] = _ctrl

    # Build a run-scoped release callback and stash a reference for the
    # main thread to call via _release_run on done/error. The worker
    # thread also receives it directly. First caller wins; second is a
    # no-op. See _make_release_once for the lock+flag mechanics.
    _release_once = _make_release_once(st.session_state.get("user_id"))
    st.session_state["_release_callback"] = _release_once

    thread = threading.Thread(
        target=_run_pipeline_thread,
        args=(topic, budget, iterations, provider, model,
              st.session_state.progress_queue,
              debate_enabled, st.session_state.get("user_id"),
              _ctrl, _release_once),
        daemon=True,
    )
    thread.start()


def _run_scientist_thread(
    topic: str, budget: float, iterations: int,
    provider: str, model: str,
    progress_queue: queue.Queue,
    debate_enabled: bool, exec_timeout: int, max_sci_iters: int,
    user_id: Optional[int] = None,
    release_once: Optional[Any] = None,
) -> None:
    """Runs v2 automated scientist in a daemon thread.

    Auto-saves results to DB when complete (and on partial progress) so
    they survive session expiry, page reload, or browser close.
    """
    # Capture user_id locally so it survives session destruction
    _uid = user_id
    _topic = topic
    results = None
    _scientist_success = False

    def _safe_queue_put(item):
        """Put to queue without crashing if session is dead."""
        try:
            progress_queue.put(item, block=False)
        except Exception:
            pass

    def _safe_save(results_dict, tag=""):
        """Save results to DB — bulletproof against any errors."""
        if not _uid or not results_dict:
            return False
        try:
            ideas_count = len(results_dict.get("ideas", []))
            if ideas_count == 0:
                return False
            db.save_result(
                user_id=_uid,
                topic=_topic + (f" [{tag}]" if tag else ""),
                coverage=results_dict.get("coverage", 0.0),
                ideas_count=ideas_count,
                results_dict=results_dict,
            )
            try:
                db_cache.invalidate_user_results()
            except Exception:
                pass
            return True
        except Exception:
            return False

    try:
        import config as _cfg
        if provider:
            _cfg.PROVIDER = provider
        if model:
            _cfg.MODEL = model

        # Pass ideation knobs to config so the pipeline can read them
        _knobs = st.session_state.get("_ideation_knobs")
        if _knobs is not None:
            _cfg._ideation_knobs = _knobs

        from pipeline_v2 import AutomatedScientist

        def on_progress(msg: str) -> None:
            _safe_queue_put(("progress", msg))

        scientist = AutomatedScientist()
        results = scientist.run(
            topic=topic,
            budget_usd=budget,
            max_ideation_iterations=iterations,
            max_scientist_iterations=max_sci_iters,
            execution_timeout=exec_timeout,
            on_progress=on_progress,
            debate_enabled=debate_enabled,
            user_id=user_id,
        )

        # ── Auto-save to DB (session-independent) ─────────────────────────
        if _safe_save(results):
            if results is not None:
                results["_auto_saved"] = True

        # Match legacy billing semantic — single source of truth lives
        # in _run_produced_ideas so worker, drain, and tests cannot drift.
        _scientist_success = _run_produced_ideas(results)
        _safe_queue_put(("done", results))

    except Exception as exc:
        # Save whatever partial results we have
        if results:
            _safe_save(results, tag="partial")
        _safe_queue_put(("error", str(exc)))
    finally:
        # ── Run-scoped release (matches _run_pipeline_thread). The
        # callback is lock+flag-guarded so this and the main-thread
        # _release_run can't both decrement the per-user counter.
        if release_once is not None:
            try:
                release_once(success=_scientist_success)
            except Exception:
                pass


def _start_scientist(
    topic: str, budget: float, iterations: int,
    provider: str = "", model: str = "",
    debate_enabled: bool = False,
    exec_timeout: int = 600, max_sci_iters: int = 2,
) -> None:
    preflight = _preflight_run(topic, budget, iterations)
    if preflight is None:
        return
    topic, budget, iterations = preflight

    # Debate is a premium feature — strip it for free tier.
    if debate_enabled:
        try:
            from production_optimization import get_quota_enforcer
            uid = st.session_state.get("user_id")
            if uid and not get_quota_enforcer().tier_feature_allowed(uid, "priority_support"):
                debate_enabled = False
                st.info("Debate arena is a Pro feature — disabled for this run.")
        except Exception:
            pass

    st.session_state.running = True
    st.session_state.done = False
    st.session_state.results = None
    st.session_state.error = None
    st.session_state.progress_log = []
    st.session_state.progress_queue = queue.Queue()

    # Run-scoped release callback (matches _start_pipeline). Whichever
    # side fires release first (worker's finally vs main-thread
    # _release_run) wins; the other is a lock-guarded no-op. Prevents
    # the per-user counter bleed that a naive double-release would
    # introduce when multiple runs are in flight for the same user.
    _release_once_sci = _make_release_once(st.session_state.get("user_id"))
    st.session_state["_release_callback"] = _release_once_sci

    thread = threading.Thread(
        target=_run_scientist_thread,
        args=(topic, budget, iterations, provider, model,
              st.session_state.progress_queue,
              debate_enabled, exec_timeout, max_sci_iters,
              st.session_state.get("user_id"),
              _release_once_sci),
        daemon=True,
    )
    thread.start()


def _drain_queue() -> bool:
    """
    Drain all pending messages from the progress queue.
    Returns True if the pipeline just completed (done or error).
    """
    finished = False
    q: queue.Queue = st.session_state.progress_queue
    while True:
        try:
            kind, payload = q.get_nowait()
        except queue.Empty:
            break

        if kind == "progress":
            # Dedup consecutive plain-text progress messages (events keep all)
            if isinstance(payload, str) and payload.startswith("__EVENT__"):
                st.session_state.progress_log.append(payload)
                try:
                    st.session_state.progress_events.append(json.loads(payload[9:]))
                except Exception:
                    pass
            else:
                # Skip if identical to the most recent plain-text message
                _ptext = st.session_state.progress_text
                if _ptext and _ptext[-1] == payload:
                    continue
                st.session_state.progress_log.append(payload)
                _ptext.append(payload)
        elif kind == "done":
            st.session_state.results = payload
            st.session_state.running = False
            st.session_state.done = True
            finished = True
            # Release production-gate slots (quota + concurrency).
            # Uses _run_produced_ideas so this stays bit-identical with
            # the worker's _success computation.
            try:
                _release_run(success=_run_produced_ideas(payload))
            except Exception:
                pass

            # ── Check for new achievements ─────────────────────────────────
            try:
                import engagement
                _uid_ach = st.session_state.get("user_id")
                if _uid_ach and payload:
                    new_achievements = engagement.check_achievements_after_run(_uid_ach, payload)
                    if new_achievements:
                        st.session_state["_new_achievements"] = new_achievements
                    # Award XP for quality ideas
                    ideas = payload.get("ideas", [])
                    for idea in ideas:
                        if idea.get("quality_score", 0) >= 0.6:
                            engagement.award_xp(_uid_ach, "generate_quality_idea")
                    # Post to activity feed
                    if ideas:
                        topic = payload.get("topic", "research")
                        best = max(ideas, key=lambda x: x.get("quality_score", 0))
                        engagement.post_activity(
                            _uid_ach, "pipeline_run",
                            f"Ran a pipeline on '{topic[:50]}' and generated {len(ideas)} ideas",
                            {"topic": topic, "best_quality": best.get("quality_score", 0)},
                        )
            except Exception:
                pass
        elif kind == "error":
            st.session_state.error = payload
            st.session_state.running = False
            st.session_state.done = True
            finished = True
            try:
                _release_run(success=False)
            except Exception:
                pass

    return finished


# ── Daily check-in on first load of each session ─────────────────────────────
_uid_checkin = st.session_state.get("user_id")
if _uid_checkin and not st.session_state.get("_checked_in_today"):
    try:
        import engagement
        result = engagement.check_in_user(_uid_checkin)
        st.session_state["_checked_in_today"] = True
        st.session_state["_checkin_result"] = result
    except Exception:
        pass

# ── Apply referral code (once, after login) ──────────────────────────────────
if _uid_checkin and st.session_state.get("_referral_code") and not st.session_state.get("_referral_applied"):
    try:
        from growth import apply_referral
        _ref_result = apply_referral(_uid_checkin, st.session_state["_referral_code"])
        if _ref_result.get("success"):
            st.session_state["_referral_bonus"] = _ref_result["referred_xp"]
        st.session_state["_referral_applied"] = True
    except Exception:
        pass

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="padding:4px 0 8px 0">'
        '<span style="font-size:20px;font-weight:700;letter-spacing:-0.3px">'
        '🧠 IdeaGraph</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    auth_ui.show_user_menu()

    # ── Runtime control (pause-on-budget / pause-on-network) ─────────────
    with st.expander("⏯️ Runtime control", expanded=False):
        st.session_state["_runtime_control_enabled"] = st.checkbox(
            "Enable interactive pause/stop",
            value=st.session_state.get("_runtime_control_enabled", True),
            help="When on, the pipeline pauses if budget approaches its "
                 "limit or the network seems down — and asks whether to "
                 "continue (with optional top-up) or stop early.",
            key="_rc_toggle",
        )
        st.session_state["_runtime_budget_pause_threshold"] = st.slider(
            "Budget pause threshold",
            min_value=0.50, max_value=0.99,
            value=float(st.session_state.get(
                "_runtime_budget_pause_threshold", 0.85)),
            step=0.05,
            help="Pause when this fraction of the budget is consumed.",
            key="_rc_budget_thresh",
        )
        st.session_state["_runtime_max_network_failures"] = st.slider(
            "Pause after N consecutive LLM failures",
            min_value=2, max_value=10,
            value=int(st.session_state.get(
                "_runtime_max_network_failures", 4)),
            step=1,
            help="If this many LLM calls fail in a row, pause and ask "
                 "the user whether to retry or stop.",
            key="_rc_net_thresh",
        )
        st.session_state["_runtime_decision_timeout_s"] = st.slider(
            "Decision timeout (s)",
            min_value=60, max_value=1800,
            value=int(st.session_state.get(
                "_runtime_decision_timeout_s", 600)),
            step=30,
            help="How long the pipeline waits for your decision before "
                 "defaulting to STOP (saves the partial archive).",
            key="_rc_timeout",
        )

    # ── User stats widget (streak, XP, level) ─────────────────────────────
    try:
        import engagement
        _uid_side = st.session_state.get("user_id")
        if _uid_side:
            stats = engagement.get_user_stats(_uid_side)
            level = stats.get("level", 1)
            xp = stats.get("xp", 0)
            streak = stats.get("current_streak", 0)
            xp_in = stats.get("xp_in_level", 0)
            xp_needed = stats.get("xp_needed", 100)
            progress = stats.get("level_progress", 0)

            # Check-in notification
            checkin = st.session_state.get("_checkin_result")
            if checkin and checkin.get("xp_earned", 0) > 0:
                st.success(f"+{checkin['xp_earned']} XP daily login!")
                if checkin.get("bonus_msg"):
                    st.info(checkin["bonus_msg"])
                if checkin.get("leveled_up"):
                    st.balloons()
                    st.success(f"🎉 LEVEL UP! Now level {checkin['new_level']}")
                st.session_state["_checkin_result"] = None  # show only once

            # Stats row — styled cards
            _xp_pct = int(progress * 100)
            st.markdown(
                f'<div style="background:linear-gradient(135deg,#f0f9ff,#e0f2fe);'
                f'border:1px solid #bae6fd;border-radius:12px;padding:14px 16px;margin:4px 0 10px 0">'
                # Level + Streak row
                f'<div style="display:flex;justify-content:space-between;margin-bottom:10px">'
                f'<div style="text-align:center;flex:1">'
                f'<div style="font-size:11px;color:#0369a1;font-weight:600;text-transform:uppercase;letter-spacing:0.05em">Level</div>'
                f'<div style="font-size:24px;font-weight:800;color:#0c4a6e">{level}</div>'
                f'</div>'
                f'<div style="width:1px;background:#bae6fd"></div>'
                f'<div style="text-align:center;flex:1">'
                f'<div style="font-size:11px;color:#0369a1;font-weight:600;text-transform:uppercase;letter-spacing:0.05em">Streak</div>'
                f'<div style="font-size:24px;font-weight:800;color:#0c4a6e">🔥 {streak}d</div>'
                f'</div>'
                f'</div>'
                # XP progress bar
                f'<div style="margin-bottom:4px;display:flex;justify-content:space-between;align-items:center">'
                f'<span style="font-size:11px;font-weight:600;color:#0369a1">{xp_in}/{xp_needed} XP</span>'
                f'<span style="font-size:11px;color:#64748b">{_xp_pct}%</span>'
                f'</div>'
                f'<div style="background:#bae6fd;border-radius:6px;height:10px;overflow:hidden">'
                f'<div style="background:linear-gradient(90deg,#0ea5e9,#38bdf8);'
                f'height:100%;width:{_xp_pct}%;border-radius:6px;'
                f'transition:width 0.5s ease"></div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # ── Weekly Challenge widget ──────────────────────────────────
            try:
                from growth import get_challenge_progress, claim_challenge_reward
                _cp = get_challenge_progress(_uid_side)
                _ch = _cp["challenge"]
                _prog = _cp["progress"]
                _goal = _ch["goal_count"]
                _done = _cp["completed"]
                _claimed = _cp["claimed"]

                with st.container():
                    _ch_pct = int(min(_prog / max(_goal, 1), 1.0) * 100)
                    _ch_bar_color = "linear-gradient(90deg,#10b981,#34d399)" if _done else "linear-gradient(90deg,#f59e0b,#fbbf24)"
                    st.markdown(
                        f'<div style="background:#fffbeb;border:1px solid #fde68a;border-radius:10px;'
                        f'padding:12px 14px;margin-bottom:8px">'
                        f'<div style="font-size:13px;font-weight:700;color:#92400e">🏅 {_ch["title"]}</div>'
                        f'<div style="font-size:11px;color:#a16207;margin:4px 0 8px 0">'
                        f'{_ch["description"]} — <b>+{_ch["xp_reward"]} XP</b></div>'
                        f'<div style="display:flex;justify-content:space-between;margin-bottom:3px">'
                        f'<span style="font-size:10px;font-weight:600;color:#92400e">{_prog}/{_goal}</span>'
                        f'<span style="font-size:10px;color:#a16207">{_ch_pct}%</span>'
                        f'</div>'
                        f'<div style="background:#fde68a;border-radius:5px;height:8px;overflow:hidden">'
                        f'<div style="background:{_ch_bar_color};height:100%;width:{_ch_pct}%;border-radius:5px"></div>'
                        f'</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    if _done and not _claimed:
                        if st.button(f"Claim {_ch['xp_reward']} XP!", key="claim_challenge",
                                     type="primary", use_container_width=True):
                            _result = claim_challenge_reward(_uid_side)
                            if _result.get("success"):
                                st.success(f"+{_result['xp_earned']} XP!")
                                st.rerun()
                            else:
                                st.error(_result.get("reason", "Error"))
                    elif _claimed:
                        st.caption("Claimed this week!")
                    else:
                        st.caption(f"{_ch['days_remaining']}d remaining")
            except Exception:
                pass

            # ── Referral link widget ─────────────────────────────────────
            try:
                from growth import get_or_create_referral, get_referral_link
                _ref = get_or_create_referral(_uid_side)
                _ref_link = get_referral_link(_uid_side)
                st.markdown(
                    f'<div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;'
                    f'padding:10px 14px;margin:6px 0">'
                    f'<div style="font-size:12px;font-weight:700;color:#166534">🎁 Invite Friends</div>'
                    f'<div style="font-size:10px;color:#15803d;margin:3px 0">'
                    f'You both get XP! {_ref["referrals"]} referred so far.</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.code(_ref_link, language=None)
            except Exception:
                pass

    except Exception:
        pass

    st.divider()

    st.markdown(
        '<div style="font-size:12px;font-weight:700;color:#0369a1;'
        'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px">'
        '⚙️ Configuration</div>',
        unsafe_allow_html=True,
    )

    import config as _cfg

    # ── Provider / Model selection ─────────────────────────────────────────
    _api_key_map = {
        "deepseek":  _cfg.DEEPSEEK_API_KEY,
        "openai":    _cfg.OPENAI_API_KEY,
        "groq":      _cfg.GROQ_API_KEY,
        "gemini":    _cfg.GEMINI_API_KEY,
        "azure":     _cfg.AZURE_API_KEY,
        "anthropic": getattr(_cfg, "ANTHROPIC_API_KEY", ""),
        # xAI Grok shares the GROK_API_KEY env var with the image-gen
        # provider — both auth against the same xAI account.
        "xai":       getattr(_cfg, "GROK_API_KEY", ""),
    }

    # Add kimi to the api-key map so its key-presence shows correctly.
    _api_key_map["kimi"] = getattr(_cfg, "KIMI_API_KEY", "")

    # Friendly display names + brand emoji per provider. Emoji is just
    # decoration; the canonical text label drives identification because
    # some browsers/OS don't render every emoji glyph (DeepSeek 🐋 on
    # older Windows fonts renders as a blank box, hiding the entry).
    _provider_meta = {
        "deepseek":  ("DeepSeek",  "🐋"),
        "openai":    ("OpenAI",    "🟢"),
        "groq":      ("Groq (Llama fast inference)", "⚡"),
        "gemini":    ("Gemini",    "✨"),
        "azure":     ("Azure",     "🔷"),
        "anthropic": ("Anthropic Claude", "🧠"),
        "kimi":      ("Kimi (Moonshot)", "🌙"),
        # xAI Grok — DIFFERENT from Groq above (Groq = fast Llama
        # inference; xAI Grok = Elon's Grok chat). Labels make the
        # distinction visible in the dropdown.
        "xai":       ("xAI Grok (Elon's Grok)", "🚀"),
    }

    def _provider_label(p: str) -> str:
        has_key = bool(_api_key_map.get(p))
        name, emoji = _provider_meta.get(p, (p.capitalize(), "•"))
        status = "✓" if has_key else "(no key)"
        # Put the name FIRST so unrenderable emojis can't hide the entry.
        return f"{name} {emoji} {status}"

    provider_select = st.selectbox(
        "LLM Provider",
        options=_cfg.SUPPORTED_PROVIDERS,
        index=_cfg.SUPPORTED_PROVIDERS.index(_cfg.PROVIDER)
        if _cfg.PROVIDER in _cfg.SUPPORTED_PROVIDERS else 0,
        format_func=_provider_label,
        help="Select the LLM provider. Set API keys in .env file.",
        disabled=st.session_state.running,
    )

    default_model = _cfg._DEFAULT_MODELS.get(provider_select, "")

    # For Anthropic: show curated Claude model dropdown instead of free-text
    if provider_select == "anthropic":
        try:
            from claude_provider import AVAILABLE_MODELS as _CLAUDE_MODELS, CLAUDE_PRICING
            _model_labels = {
                "claude-opus-4-7":   "Opus 4.7 — Premium ($15/$75 per M)",
                "claude-sonnet-4-6": "Sonnet 4.6 — Balanced ($3/$15)",
                "claude-haiku-4-5":  "Haiku 4.5 — Fast & Cheap ($1/$5)",
            }
            _idx = _CLAUDE_MODELS.index(default_model) if default_model in _CLAUDE_MODELS else 1
            model_input = st.selectbox(
                "Claude Model",
                options=_CLAUDE_MODELS,
                index=_idx,
                format_func=lambda m: _model_labels.get(m, m),
                help="Pick the Claude tier. Sonnet is recommended for most ideation tasks.",
                disabled=st.session_state.running,
                key=f"claude_model_{provider_select}",
            )
        except ImportError:
            model_input = st.text_input(
                "Model", value=default_model,
                key=f"model_input_{provider_select}",
                disabled=st.session_state.running, autocomplete="off",
            )
    else:
        model_input = st.text_input(
            "Model",
            value=default_model,
            key=f"model_input_{provider_select}",
            help="Model name (auto-filled from provider default, editable).",
            disabled=st.session_state.running,
            autocomplete="off",
        )

    st.divider()

    # ── Saved Presets (one-click re-run) ────────────────────────────────────
    try:
        from speed_optimizer import (
            BUILTIN_PRESETS, get_user_presets, save_preset, delete_preset, Preset,
        )
        _uid_preset = st.session_state.get("user_id")
        _user_presets = get_user_presets(_uid_preset) if _uid_preset else []
        _all_presets = list(BUILTIN_PRESETS) + list(_user_presets)
        _preset_names = ["— No preset —"] + [p.name for p in _all_presets]

        st.markdown(
            '<div style="font-size:12px;font-weight:700;color:#0369a1;'
            'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px">'
            '⭐ Presets</div>',
            unsafe_allow_html=True,
        )
        _preset_choice = st.selectbox(
            "Preset", _preset_names, index=0,
            help="Quick-start with a curated configuration",
            disabled=st.session_state.running,
            label_visibility="collapsed",
            key="preset_choice",
        )
        if _preset_choice != "— No preset —":
            _selected = next((p for p in _all_presets if p.name == _preset_choice), None)
            if _selected and st.session_state.get("_applied_preset") != _preset_choice:
                # Apply preset values to session state
                st.session_state["_preset_topic"] = _selected.topic
                st.session_state["_preset_provider"] = _selected.provider
                st.session_state["_preset_model"] = _selected.model
                st.session_state["_preset_budget"] = _selected.budget_usd
                st.session_state["_preset_iters"] = _selected.iterations
                st.session_state["_preset_creativity"] = _selected.creativity
                st.session_state["_preset_time"] = _selected.time_weeks
                st.session_state["_preset_risk"] = _selected.risk
                st.session_state["_preset_domain"] = _selected.domain
                st.session_state["_preset_repro"] = _selected.enable_repro
                st.session_state["_preset_fmea"] = _selected.enable_fmea
                st.session_state["_preset_adv"] = _selected.enable_adversarial
                st.session_state["_applied_preset"] = _preset_choice
                st.toast(f"Applied: {_preset_choice}", icon="⭐")
    except Exception:
        pass

    st.markdown(
        '<div style="font-size:12px;font-weight:700;color:#0369a1;'
        'text-transform:uppercase;letter-spacing:0.08em;margin:8px 0 8px 0">'
        '🎯 Research Topic</div>',
        unsafe_allow_html=True,
    )

    _default_topic = st.session_state.pop("_preset_topic", None) or "large language model reasoning and planning"
    # Note: st.text_area doesn't accept `autocomplete` (only st.text_input
    # does). Browsers rarely flag textareas for autocomplete anyway, so we
    # leave it off here.
    topic_input = st.text_area(
        "Research topic",
        value=_default_topic,
        height=100,
        help="Describe the research area you want to explore.",
    )
    budget_slider = st.slider(
        "API budget (USD)",
        min_value=0.5,
        max_value=10.0,
        value=2.0,
        step=0.5,
        help="Approximate maximum spend on LLM API calls.",
    )
    iterations_slider = st.slider(
        "Max iterations",
        min_value=3,
        max_value=50,
        value=20,
        step=1,
        help="Maximum number of ideation loop iterations.",
    )

    debate_checkbox = st.checkbox(
        "Enable Debate Arena",
        value=False,
        help="After ideation, top ideas compete in a tournament-style debate.",
        disabled=st.session_state.running,
    )

    # ── Ideation Knobs (creativity, time, risk) ───────────────────────────
    st.markdown(
        '<div style="font-size:12px;font-weight:700;color:#0369a1;'
        'text-transform:uppercase;letter-spacing:0.08em;margin:8px 0">'
        '🎛️ Ideation Knobs</div>',
        unsafe_allow_html=True,
    )
    knob_creativity = st.slider(
        "Creativity Level", 0.0, 1.0, 0.7, 0.05,
        help="0.0 = safe/incremental, 1.0 = radical/moonshot",
        disabled=st.session_state.running,
        key="knob_creativity",
    )
    knob_time = st.select_slider(
        "Time Budget",
        options=[2, 4, 8, 12, 24, 52],
        value=12,
        format_func=lambda x: f"{x} weeks",
        help="How long should generated ideas take to execute?",
        disabled=st.session_state.running,
        key="knob_time",
    )
    knob_risk = st.select_slider(
        "Risk Tolerance",
        options=["low", "medium", "high"],
        value="medium",
        format_func=str.title,
        help="Low = proven approaches, High = high-risk/high-reward",
        disabled=st.session_state.running,
        key="knob_risk",
    )
    knob_domain = st.selectbox(
        "Domain Persona",
        options=["auto", "ml", "nlp", "vision", "rl", "bio", "graph", "drug", "robotics"],
        index=0,
        format_func=lambda x: x.upper() if x != "auto" else "Auto-detect",
        help="Tune the expert persona to your research domain",
        disabled=st.session_state.running,
        key="knob_domain",
    )
    # ── Enhancement toggle cards (Repro / FMEA / Adversarial) ────────────
    st.markdown(
        '<div style="font-size:11px;font-weight:600;color:#0369a1;margin:6px 0 4px 0;'
        'text-transform:uppercase;letter-spacing:0.05em">✨ Idea Enhancements</div>',
        unsafe_allow_html=True,
    )

    # Read current state from session (defaults if first render)
    _cur_repro = st.session_state.get("knob_repro", True)
    _cur_fmea = st.session_state.get("knob_fmea", True)
    _cur_adv = st.session_state.get("knob_adv", False)

    _kc1, _kc2, _kc3 = st.columns(3)

    def _toggle_card(col, key, default, icon, label, color_on, color_off, help_text):
        cur = st.session_state.get(key, default)
        is_on = bool(cur)
        bg = f"linear-gradient(135deg,{color_on}20,{color_on}10)" if is_on else "rgba(255,255,255,0.5)"
        border = f"{color_on}80" if is_on else "#cbd5e1"
        text_color = color_on if is_on else "#94a3b8"
        status = "ON" if is_on else "OFF"
        status_bg = color_on if is_on else "#94a3b8"
        col.markdown(
            f'<div style="background:{bg};border:1.5px solid {border};border-radius:10px;'
            f'padding:8px 6px;text-align:center;margin-bottom:4px;'
            f'transition:all 0.2s ease;min-height:62px;'
            f'box-shadow:{"0 2px 6px " + color_on + "30" if is_on else "none"}">'
            f'<div style="font-size:18px;line-height:1;margin-bottom:2px;'
            f'opacity:{1.0 if is_on else 0.4}">{icon}</div>'
            f'<div style="font-size:10px;font-weight:700;color:{text_color};'
            f'text-transform:uppercase;letter-spacing:0.04em;line-height:1.1">{label}</div>'
            f'<div style="display:inline-block;background:{status_bg};color:white;'
            f'font-size:8px;font-weight:700;padding:1px 6px;border-radius:6px;'
            f'margin-top:3px;letter-spacing:0.05em">{status}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        # Hidden actual checkbox below (label_visibility="collapsed" trick)
        return col.checkbox(
            label, value=default, key=key,
            help=help_text,
            disabled=st.session_state.running,
            label_visibility="collapsed",
        )

    knob_repro = _toggle_card(
        _kc1, "knob_repro", True, "🔬", "Repro Specs",
        "#0ea5e9", "#cbd5e1",
        "Force concrete reproducibility specs (versions, GPU-hours, seeds)",
    )
    knob_fmea = _toggle_card(
        _kc2, "knob_fmea", True, "⚠️", "FMEA",
        "#f59e0b", "#cbd5e1",
        "Generate failure-mode & effects analysis with mitigations",
    )
    knob_adv = _toggle_card(
        _kc3, "knob_adv", False, "🔄", "Adversarial",
        "#a855f7", "#cbd5e1",
        "Generate contrary twin of each idea (flips core assumption)",
    )

    # Stash knobs in session state for pipeline thread
    try:
        from idea_enhancer import IdeationKnobs as _IK
        st.session_state["_ideation_knobs"] = _IK(
            creativity_level=knob_creativity,
            time_budget_weeks=knob_time,
            risk_tolerance=knob_risk,
            domain_persona=knob_domain,
            enable_reproducibility=knob_repro,
            enable_fmea=knob_fmea,
            enable_adversarial=knob_adv,
        )
    except Exception:
        pass

    st.divider()
    st.markdown("##### Advanced Settings")
    exec_timeout = st.slider("Execution timeout (sec)", 60, 3600, 600, 60,
                              help="Max time for experiment code execution.",
                              disabled=st.session_state.running)
    max_sci_iters = st.slider("Scientist iterations", 1, 5, 2,
                               help="How many idea→experiment→paper cycles to run.",
                               disabled=st.session_state.running)

    # ── Cost & time prediction (shown before run) ─────────────────────────
    try:
        from intelligence import predict_run_cost
        prediction = predict_run_cost(
            budget_usd=budget_slider, iterations=iterations_slider,
            debate_enabled=debate_checkbox, provider=provider_select,
        )
        _est_cost = prediction['estimated_cost_usd']
        _est_min = prediction['estimated_minutes']
        st.markdown(
            f'<div style="background:#f0f9ff;border:1px solid #bae6fd;border-radius:8px;'
            f'padding:8px 12px;margin:4px 0 8px 0;display:flex;justify-content:space-around">'
            f'<div style="text-align:center">'
            f'<div style="font-size:10px;color:#0369a1;font-weight:600">EST. COST</div>'
            f'<div style="font-size:16px;font-weight:700;color:#0c4a6e">${_est_cost:.3f}</div>'
            f'</div>'
            f'<div style="width:1px;background:#bae6fd"></div>'
            f'<div style="text-align:center">'
            f'<div style="font-size:10px;color:#0369a1;font-weight:600">EST. TIME</div>'
            f'<div style="font-size:16px;font-weight:700;color:#0c4a6e">~{_est_min:.0f} min</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    except Exception:
        pass

    start_disabled = st.session_state.running

    if st.button(
        "Run Automated Scientist" if not st.session_state.running else "Running …",
        disabled=start_disabled,
        type="primary",
        use_container_width=True,
    ):
        if topic_input.strip():
            st.session_state.result_saved = False
            _start_scientist(
                topic_input.strip(), budget_slider, iterations_slider,
                provider_select, model_input.strip() or default_model,
                debate_checkbox, exec_timeout, max_sci_iters,
            )
            # Only rerun if the pipeline actually started (preflight passed).
            if st.session_state.running:
                st.rerun()
        else:
            st.warning("Please enter a research topic.")

    # ── Save current as preset ───────────────────────────────────────────
    _uid_save_preset = st.session_state.get("user_id")
    if _uid_save_preset:
        with st.expander("💾 Save as preset"):
            _new_preset_name = st.text_input(
                "Preset name", placeholder="e.g. My GNN Research",
                key="new_preset_name", autocomplete="off",
            )
            if st.button("Save", key="save_preset_btn", use_container_width=True,
                         disabled=not _new_preset_name.strip()):
                try:
                    from speed_optimizer import Preset, save_preset
                    _p = Preset(
                        name=_new_preset_name.strip(),
                        topic=topic_input.strip()[:300],
                        provider=provider_select,
                        model=model_input.strip() or default_model,
                        budget_usd=float(budget_slider),
                        iterations=int(iterations_slider),
                        debate_enabled=bool(debate_checkbox),
                        creativity=float(knob_creativity),
                        time_weeks=int(knob_time),
                        risk=str(knob_risk),
                        domain=str(knob_domain),
                        enable_repro=bool(knob_repro),
                        enable_fmea=bool(knob_fmea),
                        enable_adversarial=bool(knob_adv),
                    )
                    if save_preset(_uid_save_preset, _p):
                        st.success(f"Saved: {_p.name}")
                        st.rerun()
                    else:
                        st.error("Failed to save")
                except Exception as e:
                    st.error(f"Error: {e}")

    # ── Stripe return-flow banner ─────────────────────────────────────────
    # Surface the result of the Checkout verification done up top. Show
    # once, then clear so a rerun doesn't repeat the banner.
    _just_paid = st.session_state.pop("_stripe_just_paid", None)
    if _just_paid:
        st.success(
            f"🎉 Payment confirmed — you're now on the "
            f"**{_just_paid.get('tier', '?').title()}** plan!"
        )
    _stripe_err = st.session_state.pop("_stripe_error", None)
    if _stripe_err:
        st.warning(_stripe_err)

    # ── Subscription / Plan widget ────────────────────────────────────────
    # Compact summary in the sidebar (tier + usage bar); full feature
    # checklist, comparison view, and upgrade CTAs live in the "💳 Plan
    # & billing" expander below so the sidebar stays scannable.
    st.divider()
    try:
        import billing as _billing_mod
        uid_sub = st.session_state.get("user_id")
        if uid_sub:
            _info = _billing_mod.get_plan(uid_sub)
            _plan = _info["_plan"]
            _used = int(_info.get("runs_this_month") or 0)
            _limit = _plan.monthly_run_limit
            st.markdown(f"**{_plan.label} Plan** · `${_plan.price_usd_monthly:.0f}/mo`")
            if _limit < 0:
                st.caption(f"{_used} runs this month · unlimited")
            else:
                st.progress(min(_used / max(_limit, 1), 1.0))
                st.caption(f"{_used}/{_limit} runs this month")
            with st.expander("💳 Plan & billing", expanded=False):
                _billing_mod.render_plan_card(st, uid_sub)
    except Exception:
        pass

    # ── Saved results list ────────────────────────────────────────────────
    auth_ui.show_saved_results_sidebar()

    # ── Smart Recommender ─────────────────────────────────────────────────
    st.divider()
    with st.expander("Smart Topic Recommender"):
        if st.button("Analyze Portfolio Gaps", key="recommend_btn"):
            uid_rec = st.session_state.get("user_id")
            if uid_rec:
                with st.spinner("Analyzing..."):
                    try:
                        from agents.topic_recommender import TopicRecommender
                        all_ideas_rec = db_cache.get_all_user_ideas(uid_rec)
                        domains = sorted(set(i.get("_topic", "")[:50] for i in all_ideas_rec))
                        rec = TopicRecommender()
                        suggestions = rec.recommend(all_ideas_rec, domains)
                        for s in suggestions:
                            st.markdown(f"**{s.get('topic', '')}**")
                            st.caption(f"{s.get('gap_type', '')} — {s.get('rationale', '')}")
                    except Exception as e:
                        st.error(f"Recommender failed: {e}")


# ── Main area ─────────────────────────────────────────────────────────────────
# If the sidebar's "Manage account" button was clicked, swap the main
# area for the dedicated account page and short-circuit the rest of the
# app (tabs, pipeline runner, etc.). The page renders a "← Back" button
# that clears the flag.
if st.session_state.get("_show_account_page") and st.session_state.get("user_id"):
    try:
        import account_ui
        account_ui.render_account_page(st, st.session_state.get("user_id"))
    except Exception as _acct_err:
        st.error(f"Account page failed to render: {_acct_err}")
    st.stop()

# Skip-to-content target + ARIA main landmark for screen readers.
st.markdown(
    '<main id="main-content" role="main" aria-label="IdeaGraph main content">'
    '<div style="margin-bottom:4px">'
    '<span style="font-size:26px;font-weight:700;color:#0c4a6e">🧠 IdeaGraph</span>'
    '<span style="font-size:14px;color:#0369a1;margin-left:10px">Automated Research Scientist</span>'
    '</div>'
    '</main>',
    unsafe_allow_html=True,
)

# Phase indicator (7 pipeline stages)
def _phase_badge(label: str, active: bool, done: bool) -> str:
    if done:
        return (
            f'<div style="text-align:center;padding:6px 4px">'
            f'<div style="font-size:18px">✅</div>'
            f'<div style="font-size:11px;color:#059669;font-weight:600">{label}</div>'
            f'</div>'
        )
    if active:
        return (
            f'<div style="text-align:center;padding:6px 4px">'
            f'<div style="font-size:18px" class="running-indicator">⚙️</div>'
            f'<div style="font-size:11px;color:#0ea5e9;font-weight:600">{label}</div>'
            f'</div>'
        )
    return (
        f'<div style="text-align:center;padding:6px 4px">'
        f'<div style="font-size:18px;opacity:0.3">⏳</div>'
        f'<div style="font-size:11px;color:#94a3b8">{label}</div>'
        f'</div>'
    )

log = st.session_state.progress_log
log_text = " ".join(log)
stage1_done = any("Stage 2" in m or "Stage 1/7" in m for m in log) and any("Best idea" in m for m in log)
stage2_done = any("Stage 3" in m for m in log)
stage3_done = any("Stage 4" in m for m in log)
stage4_done = any("Stage 5" in m for m in log)
stage5_done = any("Stage 6" in m for m in log)
stage6_done = any("Stage 7" in m for m in log)
stage7_done = any("AUTOMATED SCIENTIST COMPLETE" in m or "Paper ACCEPTED" in m for m in log)
results_ready = st.session_state.done and st.session_state.results is not None

# Also support v1-only progress messages for backward compat
if not any("Stage" in m for m in log):
    stage1_done = any("Phase 2" in m or "Ideation loop" in m for m in log)
    stage7_done = any("Pipeline complete" in m or "Coverage target" in m for m in log)

row1 = st.columns(4)
with row1[0]:
    st.markdown(_phase_badge("Ideation", st.session_state.running and not stage1_done, stage1_done), unsafe_allow_html=True)
with row1[1]:
    st.markdown(_phase_badge("Experiment", stage1_done and not stage3_done, stage3_done), unsafe_allow_html=True)
with row1[2]:
    st.markdown(_phase_badge("Code & Run", stage3_done and not stage4_done, stage4_done), unsafe_allow_html=True)
with row1[3]:
    st.markdown(_phase_badge("Paper & Review", stage5_done and not stage7_done, stage7_done or results_ready), unsafe_allow_html=True)

st.divider()

# ── Live Dashboard (while running) ────────────────────────────────────────────
if st.session_state.running or (st.session_state.done and not results_ready):
    finished = _drain_queue()

    # Use pre-parsed event/text lists (populated in _drain_queue) instead
    # of re-parsing the full progress_log on every Streamlit rerun.
    _live_events = list(st.session_state.progress_events)
    _text_messages = list(st.session_state.progress_text)

    # ── Runtime-control banner ────────────────────────────────────────────
    # Surfaces pause state from the RuntimeController (if any) and lets
    # the user decide Continue / Stop while the pipeline thread blocks.
    _rc = st.session_state.get("_runtime_controller")
    if _rc is not None:
        _rc_status = _rc.status()
        _rc_state = _rc_status["state"]
        if _rc_status["is_paused"]:
            _is_budget = _rc_state == "paused_budget"
            _is_network = _rc_state == "paused_network"
            _is_user = _rc_state == "paused_user"
            _hdr_color = ("#dc2626" if _is_network
                           else "#f59e0b" if _is_budget else "#0ea5e9")
            _hdr_icon = ("🔌" if _is_network
                          else "💰" if _is_budget else "⏸️")
            _hdr_text = ("Network unreachable" if _is_network
                          else "Budget threshold reached" if _is_budget
                          else "Pipeline paused by you")

            st.markdown(
                f'<div style="background:#fef2f2;border:2px solid {_hdr_color};'
                f'border-radius:12px;padding:14px 18px;margin:8px 0">'
                f'<div style="font-size:16px;font-weight:800;color:{_hdr_color}">'
                f'{_hdr_icon} {_hdr_text} — pipeline waiting for your decision</div>'
                f'<div style="font-size:13px;color:#7f1d1d;margin-top:4px">'
                f'{_rc_status["pause_reason"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            _rc_c1, _rc_c2, _rc_c3, _rc_c4 = st.columns(4)
            _rc_c1.metric("State", _rc_state.replace("paused_", "").title())
            _rc_c2.metric("Cost so far",
                            f"${_rc_status['current_cost_usd']:.3f}")
            _rc_c3.metric("Budget cap",
                            f"${_rc_status['budget_limit_usd']:.2f}",
                            f"{_rc_status['budget_used_frac']*100:.0f}% used")
            _rc_c4.metric("Net failures",
                            _rc_status["consecutive_network_failures"],
                            f"of {_rc_status['max_network_failures']} threshold")

            _btn_a, _btn_b, _btn_c = st.columns([2, 1, 2])
            with _btn_a:
                _topup = st.number_input(
                    "Add to budget (optional, $)",
                    min_value=0.0, max_value=20.0,
                    value=(0.50 if _is_budget else 0.0),
                    step=0.10, format="%.2f",
                    key="_rc_topup_input",
                    help=("Adds dollars to the per-run budget cap so the "
                            "pipeline can keep going."),
                )
            with _btn_b:
                if st.button(
                    "▶ Continue",
                    type="primary", use_container_width=True,
                    key="_rc_continue_btn",
                ):
                    _rc.decide("continue", budget_topup=float(_topup))
                    st.session_state.progress_log.append(
                        f"[runtime] resumed (topup=${_topup:.2f})"
                    )
                    st.rerun()
            with _btn_c:
                if st.button(
                    "⏹ Stop & save partial",
                    use_container_width=True,
                    key="_rc_stop_btn",
                ):
                    _rc.decide("stop")
                    st.session_state.progress_log.append(
                        "[runtime] stopped by user — partial archive saved"
                    )
                    st.rerun()

            with st.expander("Recent runtime events", expanded=False):
                for e in _rc.event_log(max_entries=20):
                    st.text(f"  {e['event']:<22}  {e['detail']}")

        else:
            # Compact running summary (only when NOT paused)
            _budget_used = _rc_status["budget_used_frac"]
            _bar_color = ("#dc2626" if _budget_used > 0.85
                            else "#f59e0b" if _budget_used > 0.65
                            else "#10b981")
            _running_btn1, _running_btn2 = st.columns([5, 1])
            with _running_btn1:
                st.markdown(
                    f'<div style="display:flex;gap:14px;align-items:center;'
                    f'margin:4px 0;font-size:13px;color:#475569">'
                    f'<span><b>Runtime:</b> {_rc_state}</span>'
                    f'<span>· cost <b>${_rc_status["current_cost_usd"]:.3f}</b> '
                    f'/ ${_rc_status["budget_limit_usd"]:.2f}</span>'
                    f'<span>· LLM <b>{_rc_status["llm_calls_total"]}</b> '
                    f'({_rc_status["llm_calls_failed"]} failed)</span>'
                    f'</div>'
                    f'<div style="height:6px;background:#e2e8f0;border-radius:3px;'
                    f'overflow:hidden;margin:4px 0">'
                    f'<div style="height:100%;width:{min(100, _budget_used*100):.0f}%;'
                    f'background:{_bar_color}"></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with _running_btn2:
                if (_rc_state == "running" and
                        st.button("⏸ Pause",
                                    use_container_width=True,
                                    key="_rc_pause_btn",
                                    help="Pause the pipeline at the next "
                                         "safe checkpoint.")):
                    _rc.request_pause()
                    st.rerun()

    # ── Live Metrics Row ──────────────────────────────────────────────────
    if _live_events:
        latest = _live_events[-1]
        lm1, lm2, lm3, lm4, lm5 = st.columns(5)
        lm1.metric("Coverage", f"{latest.get('coverage', 0):.1%}")
        lm2.metric("Ideas", f"{latest.get('ideas_archived', 0)}/{latest.get('ideas_attempted', 0)}")
        lm3.metric("Quality", f"{latest.get('quality_mean', 0):.3f}")
        lm4.metric("Cost", f"${latest.get('cost_usd', 0):.3f}")
        budget = latest.get('budget_usd', 1)
        spent = latest.get('cost_usd', 0)
        lm5.metric("Budget", f"{(1 - spent / max(budget, 0.01)) * 100:.0f}% left")

        # ── Coverage Trend Chart ──────────────────────────────────────────
        if len(_live_events) >= 2:
            try:
                import plotly.graph_objects as go
                fig = go.Figure()
                iters = [e.get("iteration", i) for i, e in enumerate(_live_events)]
                coverages = [e.get("coverage", 0) * 100 for e in _live_events]
                qualities = [e.get("quality_mean", 0) * 100 for e in _live_events]
                fig.add_trace(go.Scatter(x=iters, y=coverages, mode="lines+markers", name="Coverage %", line=dict(color="#2ecc71", width=3)))
                fig.add_trace(go.Scatter(x=iters, y=qualities, mode="lines+markers", name="Quality %", line=dict(color="#3498db", width=2)))
                fig.update_layout(
                    height=250, margin=dict(l=40, r=20, t=30, b=30),
                    title="Live: Coverage & Quality Trend",
                    xaxis_title="Iteration", yaxis_title="%",
                    legend=dict(orientation="h", y=1.1),
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                pass

    # ── Text Progress Log ─────────────────────────────────────────────────
    if _text_messages:
        with st.expander("Progress log", expanded=not bool(_live_events)):
            for msg in _text_messages[-30:]:
                st.text(msg)

    if st.session_state.running:
        _n_logs = len(st.session_state.progress_log)
        _elapsed_est = _n_logs * 2
        st.markdown(
            f'<div style="background:#f0f9ff;border:1px solid #bae6fd;border-radius:10px;'
            f'padding:14px 18px;margin:8px 0">'
            f'<span class="running-indicator">⚡ Pipeline running...</span>'
            f'<span style="color:#0369a1;font-size:0.85rem;margin-left:12px">'
            f'{_n_logs} steps completed | ~{_elapsed_est}s elapsed</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        time.sleep(2)
        st.rerun()
    elif finished:
        st.rerun()

elif st.session_state.running:
    _drain_queue()
    time.sleep(2)
    st.rerun()

# ── Error display + retry button ──────────────────────────────────────────────
if st.session_state.error:
    st.error(f"Pipeline error: {st.session_state.error}")
    _retry_col1, _retry_col2 = st.columns([1, 3])
    with _retry_col1:
        if st.button("Retry", type="primary", key="retry_btn"):
            st.session_state.error = None
            st.session_state.done = False
            st.rerun()
    with _retry_col2:
        st.caption("Click Retry to go back to the run form, or check the log below.")
    with st.expander("Progress log"):
        for msg in st.session_state.progress_log:
            st.text(msg)

# ── Results ───────────────────────────────────────────────────────────────────
if results_ready:
    results: Dict[str, Any] = st.session_state.results

    # Also drain any remaining messages
    _drain_queue()

    coverage = results.get("coverage", 0.0)
    ideas: List[Dict] = results.get("ideas", [])
    archive_data: Dict = results.get("archive", {})
    dag_summary: Dict = results.get("dag_summary", {})
    stats: Dict = results.get("stats", {})

    # Determine if v2 results are available (needed early for banner)
    is_v2 = results.get("mode") == "v2_scientist"

    # ── Hero Summary Banner ─────────────────────────────────────────────────
    topic = results.get("topic", "Research")
    elapsed_s = stats.get("elapsed_seconds", 0)
    cost = stats.get("estimated_cost_usd", results.get("estimated_cost_usd", 0))
    q_mean = stats.get("quality_mean", 0)

    if is_v2:
        final_rev = results.get("final_review") or {}
        decision = final_rev.get("decision", "").lower()
    else:
        decision = ""

    # Status color
    if not ideas:
        _banner_bg = "#fef2f2"; _banner_border = "#fecaca"; _banner_icon = "❌"; _banner_text = "No ideas generated"
    elif decision in ("strong_accept", "accept"):
        _banner_bg = "#f0fdf4"; _banner_border = "#bbf7d0"; _banner_icon = "🎉"; _banner_text = "Paper Accepted!"
    elif coverage >= 0.4:
        _banner_bg = "#f0f9ff"; _banner_border = "#bae6fd"; _banner_icon = "✅"; _banner_text = "Pipeline Complete"
    else:
        _banner_bg = "#fffbeb"; _banner_border = "#fde68a"; _banner_icon = "⚠️"; _banner_text = "Partial Results"

    import html as _html_mod
    _safe_topic = _html_mod.escape(topic)
    st.markdown(
        f'<div style="background:{_banner_bg};border:1px solid {_banner_border};'
        f'border-radius:12px;padding:16px 20px;margin-bottom:12px">'
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px">'
        f'<div style="flex:1;min-width:0">'
        f'<span style="font-size:20px;margin-right:6px">{_banner_icon}</span>'
        f'<span style="font-size:17px;font-weight:700;color:#0c4a6e">{_banner_text}</span>'
        f'<div style="color:#64748b;font-size:13px;margin-top:4px;'
        f'word-break:break-word">{_safe_topic}</div>'
        f'</div>'
        f'<div style="color:#64748b;font-size:12px;white-space:nowrap">'
        f'{elapsed_s:.0f}s | ${cost:.3f}'
        f'</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Key metrics row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Coverage", f"{coverage:.1%}")
    m2.metric("Ideas", len(ideas))
    m3.metric("DAG Papers", dag_summary.get("node_count", 0))
    m4.metric("Avg Quality", f"{q_mean:.2f}" if q_mean else "—")
    m5.metric("Cost", f"${cost:.3f}" if cost else "—")

    # ── Top Ideas Preview (visible immediately after pipeline finishes) ────
    # Surfaces full titles + key metrics for the top ideas without requiring
    # the user to click into the Ideas tab. Each title is HTML-escaped and
    # rendered without truncation.
    if ideas:
        _ranked_ideas = sorted(ideas, key=lambda x: x.get("quality_score", 0),
                                reverse=True)
        _top_n = min(5, len(_ranked_ideas))
        st.markdown(
            f'<div style="font-size:13px;font-weight:700;color:#0c4a6e;'
            f'margin:14px 0 6px 0;text-transform:uppercase;letter-spacing:0.06em">'
            f'🏆 Top {_top_n} ideas</div>',
            unsafe_allow_html=True,
        )
        for _rank, _idea in enumerate(_ranked_ideas[:_top_n], 1):
            _q = _idea.get("quality_score", 0)
            _q_color = ("#10b981" if _q >= 0.7
                          else "#f59e0b" if _q >= 0.4 else "#ef4444")
            _q_dot = ("🟢" if _q >= 0.7 else "🟡" if _q >= 0.4 else "🔴")
            _meth = (_idea.get("methodology_type") or "").replace("_", " ").title()
            _nov = (_idea.get("novelty_level") or "").capitalize()
            _strat = _idea.get("source_strategy") or "?"
            _full_title = _html_mod.escape(_idea.get("title", "Untitled"))
            st.markdown(
                f'<div style="background:#fafafa;border:1px solid #e2e8f0;'
                f'border-left:4px solid {_q_color};border-radius:8px;'
                f'padding:10px 14px;margin:5px 0;display:flex;'
                f'justify-content:space-between;align-items:flex-start;gap:12px">'
                f'<div style="flex:1;min-width:0">'
                f'<div style="font-size:14px;font-weight:700;color:#0c4a6e;'
                f'word-break:break-word">'
                f'#{_rank}. {_full_title}</div>'
                f'<div style="font-size:11px;color:#64748b;margin-top:2px">'
                f'{_meth or "?"} · {_nov or "?"} · strategy {_strat}</div>'
                f'</div>'
                f'<div style="text-align:right;white-space:nowrap;'
                f'font-size:13px;font-weight:700;color:{_q_color}">'
                f'{_q_dot} q={_q:.2f}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        if len(_ranked_ideas) > _top_n:
            st.caption(
                f"…and {len(_ranked_ideas) - _top_n} more in the **Ideas** tab below."
            )

        # Quick copy widget — Streamlit's st.code block has a native
        # one-click copy button in its top-right corner.
        with st.expander(f"📋 Copy these {_top_n} titles", expanded=False):
            _top_blob = "\n".join(
                f"{_r}. {(_i.get('title') or 'Untitled')}"
                for _r, _i in enumerate(_ranked_ideas[:_top_n], 1)
            )
            st.code(_top_blob, language=None)

    st.divider()

    # Load tournament data: first check results dict, then DB
    tournament_data = results.get("tournament")
    if not tournament_data:
        user_id = st.session_state.get("user_id")
        if user_id:
            user_debates = db_cache.get_user_debates(user_id)
            if user_debates:
                # Load the most recent debate
                latest_debate = user_debates[0]
                tournament_data = db_cache.load_debate(latest_debate["id"], user_id)

    if is_v2:
        tab_names = ["Scientist", "Analytics", "Archive", "Ideas", "💬 Chat", "Regenerate", "Novelty Lab", "Simulate", "Exec Loop", "Reviewer Lens", "Provenance", "Compare", "Mashup", "Trends", "Collab", "Debate", "Papers", "History", "Recommend", "Cross-Domain", "DAG", "Proposal", "Evolution", "Community", "Log"]
    else:
        tab_names = ["Analytics", "Archive", "Ideas", "💬 Chat", "Regenerate", "Novelty Lab", "Simulate", "Exec Loop", "Reviewer Lens", "Provenance", "Compare", "Mashup", "Trends", "Collab", "Debate", "Papers", "History", "Recommend", "Cross-Domain", "DAG", "Proposal", "Evolution", "Community", "Log"]

    tabs = st.tabs(tab_names)
    tab_offset = 1 if is_v2 else 0

    if is_v2:
        tab_scientist = tabs[0]
    tab_analytics = tabs[0 + tab_offset]
    tab_archive = tabs[1 + tab_offset]
    tab_ideas = tabs[2 + tab_offset]
    tab_result_chat = tabs[3 + tab_offset]  # NEW: chat with the loaded result
    tab_regenerate = tabs[4 + tab_offset]   # regenerate from existing idea
    tab_novelty = tabs[5 + tab_offset]      # novelty lab (critic / contradictions / ensemble)
    tab_simulate = tabs[6 + tab_offset]     # visual simulation
    tab_exec_loop = tabs[7 + tab_offset]    # execution-aware revision loop
    tab_reviewer = tabs[8 + tab_offset]     # reviewer-aware acceptance lens
    tab_provenance = tabs[9 + tab_offset]   # provenance tracing + behavioral study
    tab_compare = tabs[10 + tab_offset]
    tab_mashup = tabs[11 + tab_offset]
    tab_trends = tabs[12 + tab_offset]
    tab_collab = tabs[13 + tab_offset]
    tab_debate = tabs[14 + tab_offset]
    tab_papers = tabs[15 + tab_offset]
    tab_history = tabs[16 + tab_offset]
    tab_recommend = tabs[17 + tab_offset]
    tab_cross = tabs[18 + tab_offset]
    tab_dag = tabs[19 + tab_offset]
    tab_proposal = tabs[20 + tab_offset]
    tab_evolution = tabs[21 + tab_offset]
    tab_community = tabs[22 + tab_offset]
    tab_log = tabs[23 + tab_offset]


    # ── Tab: Automated Scientist (v2 only) ────────────────────────────────
    if is_v2:
        with tab_scientist:
            st.subheader("Automated Scientist Results")

            final_review = results.get("final_review") or {}

            # ── Key metrics row ───────────────────────────────────────────
            s1, s2, s3, s4, s5 = st.columns(5)
            s1.metric("Iterations", results.get("total_iterations", 0))
            elapsed_s = results.get("total_elapsed", 0)
            s2.metric("Time", f"{elapsed_s // 60:.0f}m {elapsed_s % 60:.0f}s" if elapsed_s > 60 else f"{elapsed_s:.0f}s")
            review_score = final_review.get("overall_score", 0)
            s3.metric("Review", f"{review_score:.1f}/10")
            decision = final_review.get("decision", "N/A")
            s4.metric("Decision", decision.replace("_", " ").title() if decision != "N/A" else "—")
            s5.metric("Cost", f"${results.get('estimated_cost_usd', 0):.3f}")

            # ── Pipeline stage summary (visual) ──────────────────────────
            st.markdown("##### Pipeline Stages")
            stage_icons = {
                "ideation": "💡", "experiment_design": "🔬", "tree_search": "🌳",
                "code_generation": "💻", "execution": "⚡", "analysis": "📊",
                "paper": "📝", "review": "👩‍🔬",
            }
            iterations = results.get("iterations", [])
            if iterations:
                latest_iter = iterations[-1]
                stages = latest_iter.get("stages", {})
                if not stages:
                    st.caption("No stage data yet.")
                else:
                    stage_cols = st.columns(min(len(stages), 8))
                    for col_idx, (stage_name, stage_data) in enumerate(stages.items()):
                        if col_idx >= len(stage_cols):
                            break
                        with stage_cols[col_idx]:
                            icon = stage_icons.get(stage_name, "⚙️")
                            display_name = stage_name.replace("_", " ").title()[:12]
                            # Extract quality indicator
                            sq = stage_data.get("quality_score",
                                 stage_data.get("best_quality",
                                 stage_data.get("reflection_confidence",
                                 stage_data.get("overall_score", 0))))
                            if isinstance(sq, (int, float)) and sq > 0:
                                color = "🟢" if sq >= 0.6 else "🟡" if sq >= 0.3 else "🔴"
                                st.markdown(f"{icon} **{display_name}**\n\n{color} {sq:.2f}")
                            else:
                                st.markdown(f"{icon} **{display_name}**\n\n✅")

            st.divider()

            # ── Two-column: Best idea + Cost breakdown ────────────────────
            left_col, right_col = st.columns(2)
            with left_col:
                st.markdown("##### Best Idea")
                best = results.get("best_idea") or {}
                if best:
                    bq = best.get("quality_score", 0)
                    badge = "🟢 Excellent" if bq >= 0.7 else "🟡 Good" if bq >= 0.5 else "🔴 Needs Work"
                    st.markdown(f"**{best.get('title', 'N/A')}**")
                    st.caption(f"Quality: {bq:.3f} ({badge})")
                    st.caption(f"Type: {(best.get('methodology_type') or '?').replace('_', ' ').title()}")
                    st.caption(f"Novelty: {(best.get('novelty_level') or '?').capitalize()}")
                    with st.expander("Method details"):
                        st.write(best.get("method", "N/A")[:500])
                else:
                    st.info("No ideas generated")

            with right_col:
                st.markdown("##### Performance")
                call_metrics = results.get("call_metrics", {})
                perf_data = {
                    "API Calls": call_metrics.get("calls", 0),
                    "Cache Hits": call_metrics.get("cache_hits", 0),
                    "Errors": call_metrics.get("errors", 0),
                    "Retries": call_metrics.get("retries", 0),
                }
                for label, value in perf_data.items():
                    st.caption(f"**{label}:** {value}")

                retry_stats = results.get("retry_stats", {})
                if retry_stats.get("total_retries", 0) > 0:
                    st.caption(f"**Auto-retries:** {retry_stats['total_retries']} | Recoveries: {retry_stats.get('total_recoveries', 0)}")

            st.divider()

            # ── Iteration details (cleaner) ───────────────────────────────
            if iterations:
                st.markdown("##### Iteration Details")
                for iter_data in iterations:
                    iter_num = iter_data.get("iteration", "?")
                    iter_status = iter_data.get("status", "?")
                    elapsed = iter_data.get("elapsed_seconds", 0)
                    status_icon = "✅" if iter_status == "completed" else "❌" if iter_status == "error" else "⏳"

                    with st.expander(f"{status_icon} Iteration {iter_num} — {elapsed:.0f}s"):
                        stages = iter_data.get("stages", {})
                        for stage_name, stage_data in stages.items():
                            if not isinstance(stage_data, dict):
                                continue
                            st.markdown(f"**{stage_icons.get(stage_name, '⚙️')} {stage_name.replace('_', ' ').title()}**")
                            # Show key metrics inline, not raw JSON
                            key_fields = {k: v for k, v in stage_data.items()
                                          if v is not None and k not in ("reflection_issues", "files")}
                            if key_fields:
                                cols_stage = st.columns(min(len(key_fields), 4))
                                for ci, (k, v) in enumerate(list(key_fields.items())[:4]):
                                    with cols_stage[ci]:
                                        display_v = f"{v:.2f}" if isinstance(v, float) else str(v)[:40]
                                        st.caption(f"**{k.replace('_', ' ').title()}:** {display_v}")

            st.divider()

            # ── Paper + Review side by side ───────────────────────────────
            paper_col, review_col = st.columns(2)

            with paper_col:
                final_paper = results.get("final_paper")
                if final_paper:
                    st.markdown("##### Generated Paper")
                    paper_md = final_paper.get("markdown", "")
                    if paper_md:
                        st.caption(f"{len(paper_md)} characters | {len(paper_md.split())} words")
                        with st.expander("Read Paper", expanded=False):
                            st.markdown(paper_md[:5000])
                else:
                    st.caption("No paper generated")

            with review_col:
                if final_review and final_review.get("decision"):
                    st.markdown("##### Peer Review")
                    score = final_review.get("overall_score", 0)
                    score_bar = "█" * int(score) + "░" * (10 - int(score))
                    st.markdown(f"**Score:** {score_bar} {score:.1f}/10")
                    st.markdown(f"**Decision:** {final_review.get('decision', '?').replace('_', ' ').title()}")

                    strengths = final_review.get("strengths", [])
                    if strengths:
                        st.markdown("**Strengths:**")
                        for s in strengths[:3]:
                            st.markdown(f"- ✅ {s[:120]}")

                    weaknesses = final_review.get("weaknesses", [])
                    if weaknesses:
                        st.markdown("**Weaknesses:**")
                        for w in weaknesses[:3]:
                            st.markdown(f"- ⚠️ {w[:120]}")
                else:
                    st.caption("No review available")

    # ── Tab 0: Analytics Dashboard ────────────────────────────────────────
    with tab_analytics:
        st.subheader("Analytics Dashboard")
        user_id_analytics = st.session_state.get("user_id")
        if user_id_analytics:
            try:
                from analytics import compute_analytics, HAS_PLOTLY
                from analytics import (build_quality_histogram, build_domain_comparison,
                                       build_methodology_heatmap, build_strategy_pie,
                                       build_probe_failure_chart, build_novelty_distribution,
                                       build_strategy_success_heatmap)

                a_data = compute_analytics(user_id_analytics)

                if a_data["total_ideas"] == 0:
                    st.info("No ideas saved yet. Run pipelines and save results to see analytics.")
                else:
                    # Summary metrics
                    ac1, ac2, ac3, ac4 = st.columns(4)
                    ac1.metric("Total Ideas", a_data["total_ideas"])
                    ac2.metric("Total Runs", a_data["total_runs"])
                    ac3.metric("Avg Quality", f"{a_data['quality_mean']:.3f}")
                    ac4.metric("Best Quality", f"{a_data['quality_max']:.3f}")

                    if HAS_PLOTLY:
                        # Charts
                        ch1, ch2 = st.columns(2)
                        with ch1:
                            st.plotly_chart(build_quality_histogram(a_data["quality_scores"]),
                                            use_container_width=True)
                        with ch2:
                            st.plotly_chart(build_strategy_pie(a_data["strategy_counts"]),
                                            use_container_width=True)

                        st.plotly_chart(build_domain_comparison(a_data["domain_stats"]),
                                        use_container_width=True)

                        all_ideas_for_heatmap = db_cache.get_all_user_ideas(user_id_analytics)
                        st.plotly_chart(build_methodology_heatmap(all_ideas_for_heatmap),
                                        use_container_width=True)

                        # New v2 analytics charts
                        st.markdown("### Advanced Analytics")
                        adv1, adv2 = st.columns(2)
                        with adv1:
                            probe_fig = build_probe_failure_chart(all_ideas_for_heatmap)
                            if probe_fig:
                                st.plotly_chart(probe_fig, use_container_width=True)
                        with adv2:
                            novelty_fig = build_novelty_distribution(all_ideas_for_heatmap)
                            if novelty_fig:
                                st.plotly_chart(novelty_fig, use_container_width=True)

                        strat_heatmap = build_strategy_success_heatmap(all_ideas_for_heatmap)
                        if strat_heatmap:
                            st.plotly_chart(strat_heatmap, use_container_width=True)
                    else:
                        st.warning("Install plotly for interactive charts: `pip install plotly`")

                    # Leaderboard
                    st.markdown("### Top 15 Ideas")
                    for rank, ti in enumerate(a_data["top_ideas"][:15], 1):
                        q = ti.get("quality_score", 0)
                        color = "🟢" if q >= 0.6 else "🟡" if q >= 0.4 else "🔴"
                        st.markdown(
                            f"{color} **#{rank}** — {ti.get('title', '')} "
                            f"(q={q:.3f}, {ti.get('_topic', '')[:30]})"
                        )
            except Exception as e:
                st.error(f"Analytics error: {e}")

    # ── Tab 1: Archive heatmap ─────────────────────────────────────────────
    with tab_archive:
        st.subheader("Quality-Diversity Archive")
        st.caption(
            "7 methodology types × 3 novelty levels. "
            "Each cell shows quality score (0–1). Hover for idea title."
        )

        grid = archive_data.get("grid", [])
        novelty_labels = archive_data.get("novelty_labels", ["incremental", "moderate", "substantial"])

        if grid:
            # ── Interactive Plotly heatmap ─────────────────────────────────
            try:
                import plotly.graph_objects as go

                method_labels = []
                z_values = []
                hover_texts = []

                for row_data in grid:
                    method_name = row_data.get("methodology", "").replace("_", " ").title()
                    method_labels.append(method_name)
                    z_row = []
                    hover_row = []
                    for cell in row_data.get("cells", []):
                        q = cell.get("quality")
                        title = cell.get("title", "")
                        if q is None:
                            z_row.append(0)
                            hover_row.append("Empty")
                        else:
                            z_row.append(q)
                            hover_row.append(f"<b>{title}</b><br>Quality: {q:.3f}")
                    z_values.append(z_row)
                    hover_texts.append(hover_row)

                fig = go.Figure(data=go.Heatmap(
                    z=z_values,
                    x=[nl.capitalize() for nl in novelty_labels],
                    y=method_labels,
                    text=[[f"{v:.2f}" if v > 0 else "—" for v in row] for row in z_values],
                    texttemplate="%{text}",
                    textfont={"size": 14, "color": "white"},
                    hovertext=hover_texts,
                    hovertemplate="%{y}<br>%{x}<br>%{hovertext}<extra></extra>",
                    colorscale=[
                        [0, "#ecf0f1"],      # empty / zero
                        [0.01, "#e74c3c"],   # low quality (red)
                        [0.4, "#f39c12"],    # medium (orange)
                        [0.6, "#f1c40f"],    # good (yellow)
                        [0.8, "#2ecc71"],    # high (green)
                        [1.0, "#27ae60"],    # excellent (dark green)
                    ],
                    zmin=0, zmax=1,
                ))
                fig.update_layout(
                    height=350,
                    margin=dict(l=160, r=20, t=30, b=40),
                    template="plotly_dark",
                    xaxis=dict(side="top"),
                )
                st.plotly_chart(fig, use_container_width=True)

            except ImportError:
                # Fallback: HTML table
                header = (
                    "<tr><th style='text-align:left'>Methodology</th>"
                    + "".join(f"<th>{nl.capitalize()}</th>" for nl in novelty_labels)
                    + "</tr>"
                )
                rows = []
                for row_data in grid:
                    method = row_data.get("methodology", "").replace("_", " ").title()
                    cells_html = [f"<td><b>{method}</b></td>"]
                    for cell in row_data.get("cells", []):
                        q = cell.get("quality")
                        if q is None:
                            cells_html.append(
                            "<td style='color:#94a3b8;text-align:center;padding:8px;font-size:12px'>—</td>"
                        )
                        else:
                            bg = "#ecfdf5" if q >= 0.6 else "#fefce8" if q >= 0.4 else "#fef2f2"
                            tc = "#059669" if q >= 0.6 else "#ca8a04" if q >= 0.4 else "#dc2626"
                            cells_html.append(
                                f"<td style='background:{bg};text-align:center;padding:8px;"
                                f"color:{tc};font-weight:600;border-radius:4px'>{q:.3f}</td>"
                            )
                    rows.append("<tr>" + "".join(cells_html) + "</tr>")
                table_html = (
                    "<table style='width:100%;border-collapse:separate;border-spacing:3px;"
                    "font-size:13px'>"
                    f"<thead><tr style='background:#f0f9ff;color:#0369a1'>{header}</tr></thead>"
                    f"<tbody>{''.join(rows)}</tbody></table>"
                )
                st.markdown(table_html, unsafe_allow_html=True)

            # ── Archive summary stats ─────────────────────────────────────
            filled = sum(1 for row in grid for cell in row.get("cells", []) if cell.get("quality") is not None)
            total = sum(len(row.get("cells", [])) for row in grid)
            st.caption(f"Filled: {filled}/{total} cells ({filled/max(total,1):.0%}) | Coverage: {coverage:.1%}")

        else:
            st.info("No archive data available.")

        # ── Interactive Quality-Diversity (Extension 5) ───────────────────
        # Sculpt the archive: freeze valuable cells, prune low-feasibility
        # ones, and force directed crossovers between two parents under
        # a user-supplied constraint. State persists in session_state
        # per result.
        st.markdown("---")
        try:
            import iqd_controls as _iqd_mod
            _iqd_archive_id = (
                (st.session_state.get("results") or {}).get("_result_id")
                or "current"
            )
            with st.expander(
                "🎛️ Interactive iQD — sculpt the archive "
                "(Extension 5)",
                expanded=False,
            ):
                _iqd_mod.render_iqd_panel(
                    st, ideas, archive_id=_iqd_archive_id,
                )
        except ImportError:
            pass
        except Exception as _iqd_err:
            st.caption(f"iQD panel unavailable: {_iqd_err}")

    # ── Tab 2: Ideas ───────────────────────────────────────────────────────
    with tab_ideas:
        # ── Jump-from-chat banner ─────────────────────────────────────────
        # When the user clicks an idea title in the 💬 Chat tab's idea
        # index, that handler stamps _jump_to_idea_title in session_state.
        # Read it here so the matching expander below opens automatically.
        _jump_title = st.session_state.get("_jump_to_idea_title")
        if _jump_title:
            jc1, jc2 = st.columns([5, 1])
            jc1.success(
                f"📍 **Jumped from chat:** {_jump_title[:80]} — the "
                f"matching idea is auto-expanded below."
            )
            if jc2.button("✕ Clear", key="ideas_clear_jump",
                          use_container_width=True):
                st.session_state.pop("_jump_to_idea_title", None)
                st.session_state.pop("_jump_to_idea_idx", None)
                st.rerun()

        st.subheader(f"Generated Ideas ({len(ideas)})")

        if not ideas:
            st.markdown(
                "<div role='status' aria-live='polite' "
                "style='background:linear-gradient(135deg,#f0f9ff,#e0f2fe);"
                "border:1px solid #7dd3fc;border-radius:12px;"
                "padding:32px 24px;text-align:center;margin:16px 0'>"
                "<div style='font-size:56px;line-height:1'>💡</div>"
                "<div style='font-size:20px;font-weight:700;"
                "color:#0c4a6e;margin-top:10px'>"
                "No ideas yet — let's generate some</div>"
                "<div style='font-size:14px;color:#0369a1;margin-top:8px;"
                "max-width:520px;margin-left:auto;margin-right:auto'>"
                "Open the sidebar (top-left ←), enter a research topic, "
                "and click <b>Run pipeline</b>. The Quality-Diversity "
                "engine will explore your topic across a methodology × "
                "novelty grid and surface the best ideas.</div>"
                "<div style='font-size:12px;color:#475569;margin-top:14px'>"
                "Tip: start with a focused topic like <i>\"efficient "
                "attention for long contexts\"</i> — broader topics "
                "give noisier ideas.</div>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            # ── Friendly glossary expander (collapsed by default) ────────
            with st.expander("📖 Glossary — what do these terms mean?",
                              expanded=False):
                st.markdown(
                    """
- **Quality score** (0–1) — composite probe score from the LLM judge
  across feasibility, novelty, significance, clarity, and specificity.
  ≥0.7 = strong; 0.4–0.7 = decent; <0.4 = weak.
- **Methodology type** — one of 7 paper-shape categories: empirical
  study, theoretical analysis, system design, dataset creation,
  survey/meta-analysis, tool/library, interdisciplinary bridge.
- **Novelty level** — `substantial` (genuinely new direction),
  `moderate` (new combination of existing ideas), `incremental`
  (modest improvement on existing work).
- **Source strategy code** — single-letter origin tag. `A`/`B`/`C` =
  baseline strategies; `R` = Regenerated; `N` = Novelty-revision;
  `E`/`K`/`U`/`X`/`G`/`H`/`P`/`L`/`M`/`Y`/`T`/`I`/`Z`/`W`/`D`/`S`/`Q` =
  the 19 Novelty Lab modes; `V` = chat-revised.
- **QD grid** — the methodology (rows) × novelty (cols) Quality-
  Diversity archive that keeps the best idea per cell.
- **Pareto front** — non-dominated subset on Quality × Novelty —
  every idea where no other idea is *strictly better on both axes*.
- **Generation** — refinement depth. Generation 0 = original; +1 for
  each chat-revise / regenerate / novelty-critic pass.
                    """
                )

            # ── Controls: Sort, Filter, Group ─────────────────────────────
            # The sort/group menu pulls from idea_sorting.SORT_MODES /
            # GROUP_MODES — see that module for the catalog (Pareto front,
            # diversity-interleaved, top-per-methodology, lineage-grouped,
            # corpus-novelty, originality, composite, etc.).
            try:
                from idea_sorting import (
                    SORT_MODES as _SORT_MODES,
                    GROUP_MODES as _GROUP_MODES,
                    DIRECTIONAL_MODES as _DIRECTIONAL_MODES,
                    sort_ideas as _sort_ideas,
                    group_ideas as _group_ideas,
                )
                _sorting_ok = True
            except ImportError:
                _sorting_ok = False

            if _sorting_ok:
                ctrl1, ctrl2, ctrl3 = st.columns([2, 1, 1])
                with ctrl1:
                    _sort_keys = list(_SORT_MODES.keys())
                    _default_idx = (_sort_keys.index("quality")
                                       if "quality" in _sort_keys else 0)
                    sort_mode = st.selectbox(
                        "Sort by",
                        options=_sort_keys,
                        index=_default_idx,
                        format_func=lambda k: _SORT_MODES[k]["label"],
                        key="idea_sort_mode",
                        help="Pick a sort mode. Some modes (Pareto, "
                             "diversity, lineage, …) have a fixed "
                             "natural order and ignore the direction "
                             "toggle.",
                    )
                    st.caption(_SORT_MODES[sort_mode]["description"])
                with ctrl2:
                    _is_directional = sort_mode in _DIRECTIONAL_MODES
                    sort_desc = st.toggle(
                        "↓ descending",
                        value=_SORT_MODES[sort_mode].get(
                            "default_desc", True,
                        ),
                        disabled=(not _is_directional),
                        key=f"idea_sort_desc_{sort_mode}",
                        help=("Flip the order." if _is_directional else
                                "This sort has a fixed natural order."),
                    )
                with ctrl3:
                    group_by = st.selectbox(
                        "Group by",
                        options=list(_GROUP_MODES.keys()),
                        index=0,
                        format_func=lambda k: _GROUP_MODES[k],
                        key="idea_group_by",
                        help="Break the list into sections.",
                    )

                # ── Search box (full-text on title + method) ────────────
                # A clicked quick-filter chip writes its query into a
                # pending session_state key; we read + clear it here so
                # the chip click "fills" the search box on the next run.
                _pending_q = st.session_state.pop(
                    "_idea_search_pending", None,
                )
                search_q = st.text_input(
                    "🔎 Search ideas (matches title + method)",
                    value=_pending_q if _pending_q is not None
                          else st.session_state.get("idea_search", ""),
                    key="idea_search",
                    placeholder="e.g. \"attention\", \"protein\", \"dialect\" — case-insensitive",
                    help="Substring match against title + method text. "
                         "Combines with the filters below (AND).",
                )

                fctrl1, fctrl2, fctrl3 = st.columns([1, 1, 1])
                with fctrl1:
                    filter_method = st.selectbox(
                        "Methodology filter",
                        ["All"] + sorted(set(
                            (i.get("methodology_type") or "?")
                            .replace("_", " ").title() for i in ideas
                        )),
                        key="idea_filter_method",
                    )
                with fctrl2:
                    filter_novelty = st.selectbox(
                        "Novelty filter",
                        ["All", "Incremental", "Moderate", "Substantial"],
                        key="idea_filter_novelty",
                    )
                with fctrl3:
                    filter_quality = st.slider(
                        "Min quality", 0.0, 1.0, 0.0, 0.05,
                        key="idea_quality_filter",
                    )

                # ── Quick filter chips ──────────────────────────────────
                # One-click presets that snap the user to a useful view.
                # Each writes pending state (search + filter selectbox
                # values) and triggers a rerun.
                st.markdown(
                    "<div style='font-size:11px;color:#64748b;"
                    "text-transform:uppercase;letter-spacing:0.06em;"
                    "margin:4px 0 -2px 0;font-weight:700'>"
                    "Quick filters</div>",
                    unsafe_allow_html=True,
                )
                _chip_cols = st.columns(6)
                if _chip_cols[0].button(
                    "🟢 High quality",
                    key="chip_high_q",
                    use_container_width=True,
                    help="Filter to ideas with quality ≥ 0.7.",
                ):
                    st.session_state["idea_quality_filter"] = 0.7
                    st.rerun()
                if _chip_cols[1].button(
                    "🌟 Substantial",
                    key="chip_substantial",
                    use_container_width=True,
                    help="Filter to ideas marked substantial novelty.",
                ):
                    st.session_state["idea_filter_novelty"] = "Substantial"
                    st.rerun()
                if _chip_cols[2].button(
                    "♻️ Regenerated",
                    key="chip_regen",
                    use_container_width=True,
                    help="Show only ideas derived via Regenerate (any strategy).",
                ):
                    st.session_state["_idea_chip_regen"] = True
                    st.rerun()
                if _chip_cols[3].button(
                    "🧪 Lab-novel",
                    key="chip_lab_novel",
                    use_container_width=True,
                    help="Ideas from Novelty Lab modes (M, H, P, etc.).",
                ):
                    st.session_state["_idea_chip_lab_novel"] = True
                    st.rerun()
                if _chip_cols[4].button(
                    "💎 Pareto",
                    key="chip_pareto",
                    use_container_width=True,
                    help="Switch sort to Pareto front (Q × N).",
                ):
                    st.session_state["idea_sort_mode"] = "pareto"
                    st.rerun()
                if _chip_cols[5].button(
                    "🧹 Clear all",
                    key="chip_clear",
                    use_container_width=True,
                    help="Reset all filters + search + chips.",
                ):
                    st.session_state["idea_search"] = ""
                    st.session_state["idea_filter_method"] = "All"
                    st.session_state["idea_filter_novelty"] = "All"
                    st.session_state["idea_quality_filter"] = 0.0
                    st.session_state.pop("_idea_chip_regen", None)
                    st.session_state.pop("_idea_chip_lab_novel", None)
                    st.session_state["idea_sort_mode"] = "quality"
                    st.rerun()
            else:
                # Fallback when idea_sorting isn't importable — keep the
                # original 5-option UI so the app still renders.
                ctrl1, ctrl2, ctrl3, ctrl4 = st.columns(4)
                with ctrl1:
                    sort_by = st.selectbox(
                        "Sort by",
                        ["Quality ↓", "Quality ↑", "Novelty",
                          "Methodology", "Strategy"],
                        key="idea_sort",
                    )
                with ctrl2:
                    filter_method = st.selectbox(
                        "Methodology", ["All"] + sorted(set(
                            (i.get("methodology_type") or "?")
                            .replace("_", " ").title() for i in ideas
                        )),
                        key="idea_filter_method",
                    )
                with ctrl3:
                    filter_novelty = st.selectbox(
                        "Novelty",
                        ["All", "Incremental", "Moderate", "Substantial"],
                        key="idea_filter_novelty",
                    )
                with ctrl4:
                    filter_quality = st.slider(
                        "Min quality", 0.0, 1.0, 0.0, 0.05,
                        key="idea_quality_filter",
                    )
                sort_mode = None
                sort_desc = True
                group_by = "none"

            # Apply filters first (independent of sort mode).
            filtered_ideas = list(ideas)
            if filter_method != "All":
                filtered_ideas = [i for i in filtered_ideas if (i.get("methodology_type") or "?").replace("_", " ").title() == filter_method]
            if filter_novelty != "All":
                filtered_ideas = [i for i in filtered_ideas if (i.get("novelty_level") or "?").capitalize() == filter_novelty]
            if filter_quality > 0:
                filtered_ideas = [i for i in filtered_ideas if i.get("quality_score", 0) >= filter_quality]
            # Text-search: substring match on title + method (case-insensitive).
            _search_text = (search_q or "").strip().lower() if _sorting_ok else ""
            if _search_text:
                filtered_ideas = [
                    i for i in filtered_ideas
                    if _search_text in (i.get("title") or "").lower()
                    or _search_text in (i.get("method") or "").lower()
                ]
            # Chip-driven filters.
            if st.session_state.get("_idea_chip_regen"):
                # "R" = regeneration strategy; only show regenerated ideas.
                filtered_ideas = [
                    i for i in filtered_ideas
                    if (i.get("source_strategy") or "") == "R"
                ]
            if st.session_state.get("_idea_chip_lab_novel"):
                # Novelty Lab strategies (non-baseline single-letter codes).
                _lab_codes = {"N", "C", "E", "K", "U", "X", "G", "H", "P",
                              "L", "M", "Y", "T", "I", "Z", "W", "D", "S",
                              "Q", "V", "F"}
                filtered_ideas = [
                    i for i in filtered_ideas
                    if (i.get("source_strategy") or "") in _lab_codes
                ]

            # Apply sorting.
            if _sorting_ok and sort_mode is not None:
                filtered_ideas = _sort_ideas(
                    filtered_ideas, sort_mode,
                    descending=bool(sort_desc),
                )
            else:
                # Legacy fallback path (only reached if the import failed).
                if sort_by == "Quality ↓":
                    filtered_ideas.sort(key=lambda x: x.get("quality_score", 0), reverse=True)
                elif sort_by == "Quality ↑":
                    filtered_ideas.sort(key=lambda x: x.get("quality_score", 0))
                elif sort_by == "Novelty":
                    novelty_order = {"substantial": 3, "moderate": 2, "incremental": 1}
                    filtered_ideas.sort(key=lambda x: novelty_order.get(x.get("novelty_level", ""), 0), reverse=True)
                elif sort_by == "Methodology":
                    filtered_ideas.sort(key=lambda x: x.get("methodology_type", ""))
                elif sort_by == "Strategy":
                    filtered_ideas.sort(key=lambda x: x.get("source_strategy", ""))

            st.caption(f"Showing {len(filtered_ideas)} of {len(ideas)} ideas")

            # ── Friendly empty state when filters return 0 results ──────
            if not filtered_ideas and ideas:
                st.markdown(
                    "<div role='status' aria-live='polite' "
                    "style='background:linear-gradient(135deg,#fef9c3,#fef08a);"
                    "border:1px solid #facc15;border-radius:12px;"
                    "padding:24px;text-align:center;margin:16px 0'>"
                    "<div style='font-size:48px;line-height:1'>🔍</div>"
                    "<div style='font-size:18px;font-weight:700;"
                    "color:#713f12;margin-top:8px'>"
                    "No ideas match your filters</div>"
                    "<div style='font-size:13px;color:#78350f;margin-top:6px'>"
                    f"You have <b>{len(ideas)}</b> idea(s) in this session, "
                    "but none clear the current search + filters.</div>"
                    "<div style='font-size:13px;color:#78350f;margin-top:6px'>"
                    "Try the <b>🧹 Clear all</b> chip above, or relax the "
                    "Min-quality slider.</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )

            # ── Quick stats bar ───────────────────────────────────────────
            if filtered_ideas:
                qs1, qs2, qs3, qs4 = st.columns(4)
                avg_q = sum(i.get("quality_score", 0) for i in filtered_ideas) / len(filtered_ideas)
                max_q = max(i.get("quality_score", 0) for i in filtered_ideas)
                n_high = sum(1 for i in filtered_ideas if i.get("quality_score", 0) >= 0.6)
                strategies = set(i.get("source_strategy", "?") for i in filtered_ideas)
                qs1.metric("Avg Quality", f"{avg_q:.3f}")
                qs2.metric("Max Quality", f"{max_q:.3f}")
                qs3.metric("High Quality (≥0.6)", n_high)
                qs4.metric("Strategies Used", ", ".join(sorted(strategies)))

                # ── Inline quality distribution + methodology breakdown ──
                try:
                    import plotly.graph_objects as go
                    from plotly.subplots import make_subplots

                    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Quality Distribution", "By Methodology"),
                        specs=[[{"type": "histogram"}, {"type": "pie"}]],
                    )

                    # Quality histogram
                    q_values = [i.get("quality_score", 0) for i in filtered_ideas]
                    colors = ["#e74c3c" if q < 0.4 else "#f39c12" if q < 0.6 else "#2ecc71" for q in q_values]
                    fig.add_trace(go.Histogram(
                        x=q_values, nbinsx=10, name="Quality",
                        marker_color="#3498db", opacity=0.8,
                    ), row=1, col=1)

                    # Methodology pie
                    from collections import Counter
                    method_counts = Counter(
                        (i.get("methodology_type") or "?").replace("_", " ").title()
                        for i in filtered_ideas
                    )
                    fig.add_trace(go.Pie(
                        labels=list(method_counts.keys()),
                        values=list(method_counts.values()),
                        hole=0.4, textinfo="label+value",
                        textposition="inside",
                    ), row=1, col=2)

                    fig.update_layout(
                        height=250, template="plotly_dark",
                        margin=dict(l=40, r=20, t=40, b=20),
                        showlegend=False,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    pass

            st.divider()

            # ── Copy-all-titles widget ────────────────────────────────────
            # One st.code block with every filtered idea's title, ready to
            # copy in a single click via Streamlit's native code-block copy
            # button. Collapsed by default to keep the list clean.
            if filtered_ideas:
                with st.expander(
                    f"📋 Copy titles ({len(filtered_ideas)} ideas)",
                    expanded=False,
                ):
                    _all_titles_blob = "\n".join(
                        f"{_i_t}. {(_it.get('title') or 'Untitled')}"
                        for _i_t, _it in enumerate(filtered_ideas, 1)
                    )
                    st.code(_all_titles_blob, language=None)
                    st.caption(
                        "Click the **📋 icon** at the top-right of the box "
                        "above to copy all titles. Or copy a single title "
                        "from the same icon inside each idea below."
                    )

            # ── 🔮 Bayesian Surprise computation panel ────────────────────
            # On-demand: clicking the button runs 2N+1 LLM calls per idea
            # (default N=3 → 7 calls × ideas count), so we make this opt-in.
            # Once computed, ideas get scored on the surprise/plausibility/
            # bayesian_score axes and can be sorted via the new sort modes.
            if filtered_ideas:
                with st.expander(
                    "🔮 Bayesian Surprise (Extension 3 — epistemic novelty)",
                    expanded=False,
                ):
                    st.caption(
                        "Measures how much the LLM's belief about the "
                        "field shifts when conditioned on each idea — a "
                        "principled alternative to LLM-as-judge novelty "
                        "scores that overvalue exotic-sounding ideas. "
                        "**Cost: ~7 LLM calls per idea**, results cached "
                        "to disk by hypothesis hash."
                    )
                    _bs_cols = st.columns([1, 1, 2])
                    _bs_n = _bs_cols[0].number_input(
                        "Samples per direction (N)",
                        min_value=1, max_value=8,
                        value=int(st.session_state.get(
                            "_bs_n_samples", 3,
                        )),
                        step=1,
                        key="_bs_n_samples_input",
                        help="Higher N = more reliable distribution "
                              "estimate but more LLM calls.",
                    )
                    _bs_rate_pl = _bs_cols[1].checkbox(
                        "Rate plausibility",
                        value=True,
                        key="_bs_rate_pl",
                        help="+1 LLM call per idea. Off = pure epistemic "
                              "shift, no feasibility weighting.",
                    )
                    _bs_max_ideas = _bs_cols[2].slider(
                        "Limit to top N ideas (by quality)",
                        min_value=1,
                        max_value=min(50, len(filtered_ideas)),
                        value=min(10, len(filtered_ideas)),
                        key="_bs_max_ideas",
                        help="Restrict compute cost. Use 50 to score all.",
                    )
                    if st.button(
                        f"🚀 Compute Bayesian Surprise "
                        f"({_bs_max_ideas} ideas · "
                        f"~{(2*int(_bs_n) + int(_bs_rate_pl)) * int(_bs_max_ideas)} "
                        f"LLM calls)",
                        key="_bs_compute_btn",
                        type="primary",
                    ):
                        try:
                            from bayesian_surprise import (
                                cached_compute_surprise_for_idea,
                            )
                            # Rank by quality and take top-N for compute.
                            _by_q = sorted(
                                filtered_ideas,
                                key=lambda i: float(
                                    i.get("quality_score") or 0.0
                                ),
                                reverse=True,
                            )[:int(_bs_max_ideas)]
                            _topic_for_bs = (
                                st.session_state.get("topic_input")
                                or (st.session_state.get("results", {}) or {})
                                    .get("topic", "")
                                or "research"
                            )
                            _progress = st.progress(0.0)
                            _log = st.empty()
                            _ok, _fail = 0, 0
                            for _bi, _idea in enumerate(_by_q):
                                try:
                                    cached_compute_surprise_for_idea(
                                        _idea,
                                        topic=_topic_for_bs,
                                        n_samples=int(_bs_n),
                                        rate_plausibility=bool(_bs_rate_pl),
                                    )
                                    _ok += 1
                                except Exception as _bse:
                                    _fail += 1
                                    _log.warning(
                                        f"#{_bi+1} {_idea.get('title','?')[:40]}: {_bse}"
                                    )
                                _progress.progress((_bi + 1) / len(_by_q))
                            _log.success(
                                f"✅ Computed for {_ok} idea(s); "
                                f"{_fail} failed. Sort by **🔮 "
                                f"Bayesian Surprise** above to rank."
                            )
                        except ImportError as _ie:
                            st.error(f"bayesian_surprise module missing: {_ie}")
                        except Exception as _e:
                            st.error(f"Compute failed: {_e}")

                    # ── Show histogram of computed scores ──────────────
                    _scored = [
                        i for i in filtered_ideas
                        if (i.get("execution_meta") or {})
                            .get("bayesian_surprise")
                    ]
                    if _scored:
                        _surps = [
                            float((i["execution_meta"]
                                   ["bayesian_surprise"]
                                   .get("surprise") or 0.0))
                            for i in _scored
                        ]
                        _plaus = [
                            float((i["execution_meta"]
                                   ["bayesian_surprise"]
                                   .get("plausibility") or 0.0))
                            for i in _scored
                        ]
                        _hist_cols = st.columns(3)
                        _hist_cols[0].metric(
                            "Scored ideas", len(_scored),
                        )
                        _hist_cols[1].metric(
                            "Mean epistemic shift",
                            f"{sum(_surps)/max(1,len(_surps)):.2f}",
                        )
                        _hist_cols[2].metric(
                            "Mean plausibility",
                            f"{sum(_plaus)/max(1,len(_plaus)):.2f}",
                        )

            # ── ⚔️ Debate-Fitness computation panel (Extension 1 lite) ───
            # T rounds of Proposer-vs-Critic dialogue per idea; fitness =
            # σ(τ · (Σ proposer − Σ critic)). Cost: 3T LLM calls / idea.
            if filtered_ideas:
                with st.expander(
                    "⚔️ Debate Fitness (Extension 1 — adversarial survival)",
                    expanded=False,
                ):
                    st.caption(
                        "Run T rounds of Proposer-vs-Critic dialogue per "
                        "idea; fitness = σ(τ · (Σ proposer − Σ critic)). "
                        "Surfaces ideas that survive direct attack on "
                        "assumptions, baselines, and scaling. "
                        "**Cost: ~3T LLM calls per idea.** Disk-cached."
                    )
                    _df_cols = st.columns([1, 1, 1, 1])
                    _df_T = _df_cols[0].number_input(
                        "Rounds (T)",
                        min_value=1, max_value=6,
                        value=2, step=1,
                        key="_df_T",
                    )
                    _df_tau = _df_cols[1].number_input(
                        "Sigmoid temp (τ)",
                        min_value=0.5, max_value=5.0,
                        value=2.0, step=0.5,
                        key="_df_tau",
                        help="Higher τ = sharper fitness; lower = softer.",
                    )
                    _df_max = _df_cols[2].slider(
                        "Top N by quality",
                        min_value=1,
                        max_value=min(50, len(filtered_ideas)),
                        value=min(10, len(filtered_ideas)),
                        key="_df_max",
                    )
                    _df_est = 3 * int(_df_T) * int(_df_max)
                    _df_cols[3].metric("Est. LLM calls", _df_est)

                    if st.button(
                        f"⚔️ Run Debate Fitness on top {int(_df_max)} ideas",
                        key="_df_compute_btn",
                        type="primary",
                    ):
                        try:
                            from debate_fitness import (
                                cached_compute_fitness_for_idea,
                            )
                            _by_q = sorted(
                                filtered_ideas,
                                key=lambda i: float(
                                    i.get("quality_score") or 0.0
                                ),
                                reverse=True,
                            )[:int(_df_max)]
                            _progress = st.progress(0.0)
                            _log = st.empty()
                            _ok, _fail = 0, 0
                            for _bi, _idea in enumerate(_by_q):
                                try:
                                    cached_compute_fitness_for_idea(
                                        _idea,
                                        n_rounds=int(_df_T),
                                        tau=float(_df_tau),
                                    )
                                    _ok += 1
                                except Exception as _dfe:
                                    _fail += 1
                                    _log.warning(
                                        f"#{_bi+1} "
                                        f"{_idea.get('title', '?')[:40]}: {_dfe}"
                                    )
                                _progress.progress((_bi + 1) / len(_by_q))
                            _log.success(
                                f"✅ Computed for {_ok} idea(s); "
                                f"{_fail} failed. Sort by **⚔️ Debate "
                                f"Fitness** above to rank by adversarial "
                                f"survival."
                            )
                        except ImportError as _ie:
                            st.error(f"debate_fitness module missing: {_ie}")
                        except Exception as _e:
                            st.error(f"Compute failed: {_e}")

                    _df_scored = [
                        i for i in filtered_ideas
                        if (i.get("execution_meta") or {})
                            .get("debate_fitness")
                    ]
                    if _df_scored:
                        _fits = [
                            float((i["execution_meta"]
                                   ["debate_fitness"]
                                   .get("fitness") or 0.5))
                            for i in _df_scored
                        ]
                        _margins = [
                            float((i["execution_meta"]
                                   ["debate_fitness"]
                                   .get("margin") or 0.0))
                            for i in _df_scored
                        ]
                        _dfh_cols = st.columns(3)
                        _dfh_cols[0].metric("Scored ideas", len(_df_scored))
                        _dfh_cols[1].metric(
                            "Mean fitness",
                            f"{sum(_fits)/max(1,len(_fits)):.2f}",
                        )
                        _dfh_cols[2].metric(
                            "Mean margin",
                            f"{sum(_margins)/max(1,len(_margins)):+.2f}",
                        )

            # ── Idea Cards ────────────────────────────────────────────────
            # If a Group-by mode is active, render section headers between
            # buckets. The sort order is preserved within each section.
            sorted_ideas = filtered_ideas
            _sectioned: list = []
            if _sorting_ok and group_by and group_by != "none":
                _sectioned = _group_ideas(sorted_ideas, group_by)
            else:
                _sectioned = [("", sorted_ideas)]

            _global_idx = 0
            for _section_label, _section_ideas in _sectioned:
                if _section_label:
                    st.markdown(
                        f"<div style='margin-top:18px;margin-bottom:6px;"
                        f"padding:8px 14px;background:#f0f9ff;"
                        f"border-left:4px solid #0ea5e9;border-radius:6px;"
                        f"font-size:13px;font-weight:700;color:#0c4a6e;"
                        f"text-transform:uppercase;letter-spacing:0.05em'>"
                        f"{_section_label} "
                        f"<span style='color:#64748b;font-weight:400;"
                        f"text-transform:none;letter-spacing:normal'>"
                        f"· {len(_section_ideas)} idea(s)</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                for idea in _section_ideas:
                    _global_idx += 1
                    i = _global_idx
                    q = idea.get("quality_score", 0.0)
                    title = idea.get("title", "Untitled")
                    method = idea.get("methodology_type", "?").replace("_", " ").title()
                    novelty = idea.get("novelty_level", "?").capitalize()
                    strategy = idea.get("source_strategy", "?")

                    color = "🟢" if q >= 0.6 else "🟡" if q >= 0.4 else "🔴"
                    _grade = "A+" if q >= 0.8 else "A" if q >= 0.7 else "B+" if q >= 0.6 else "B" if q >= 0.5 else "C" if q >= 0.4 else "D"
                    # Show the FULL idea title in the expander label so users
                    # never lose the end of a long title. Prepend a 📍 marker
                    # when this is the jump target so it's easy to spot.
                    _is_jump_target = bool(
                        _jump_title and _jump_title == title
                    )
                    _marker = "📍 " if _is_jump_target else ""
                    label = f"{_marker}{color} #{i} {_grade} — {title} ({method} | {novelty})"

                    # Auto-expand: the top-ranked idea (i == 1) by default,
                    # OR the idea the user just jumped to from the chat tab.
                    with st.expander(
                        label,
                        expanded=(i == 1 or _is_jump_target),
                    ):
                        # ── Copy-title widget (st.code has a native copy button) ──
                        st.code(title, language=None)

                        # ── Maturity badge + header metrics ────────────────────
                        try:
                            from intelligence import compute_maturity, MATURITY_EMOJI, MATURITY_COLOR
                            _mat = compute_maturity(idea)
                            _mat_emoji = MATURITY_EMOJI.get(_mat["level"], "📝")
                            _mat_color = MATURITY_COLOR.get(_mat["level"], "#94a3b8")
                            _mat_pct = _mat["progress_pct"]
                            _mat_bg = f"{_mat_color}15"  # 15 = ~8% opacity hex
                            st.markdown(
                                f'<div style="display:flex;align-items:center;gap:10px;'
                                f'background:{_mat_bg};border:1px solid {_mat_color}40;'
                                f'border-radius:8px;padding:8px 12px;margin-bottom:8px">'
                                f'<span style="font-size:18px">{_mat_emoji}</span>'
                                f'<div style="flex:1">'
                                f'<div style="display:flex;justify-content:space-between;align-items:center">'
                                f'<span style="color:{_mat_color};font-weight:700;font-size:12px;'
                                f'text-transform:uppercase;letter-spacing:0.04em">{_mat["label"]}</span>'
                                f'<span style="color:#64748b;font-size:11px">{_mat_pct:.0f}%</span>'
                                f'</div>'
                                f'<div style="background:#e2e8f0;border-radius:3px;height:4px;margin-top:4px">'
                                f'<div style="background:{_mat_color};height:100%;width:{_mat_pct}%;'
                                f'border-radius:3px"></div></div>'
                                f'<div style="color:#64748b;font-size:10px;margin-top:3px">'
                                f'Next: {_mat["next_step"][:55]}</div>'
                                f'</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                        except Exception:
                            pass

                        # Header row with key metrics
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Quality", f"{q:.3f}")
                        c2.metric("Type", method[:15])
                        c3.metric("Novelty", novelty)
                        c4.metric("Strategy", {"A": "Frontier", "B": "Bridge", "C": "Gap-Fill"}.get(strategy, strategy))

                        # Probe scores as mini progress bars
                        scores = idea.get("probe_scores") or {}
                        if scores:
                            probe_items = [(pk, pv) for pk, pv in scores.items() if isinstance(pv, (int, float))]
                            if probe_items:
                                probe_html = "<div style='display:flex;flex-wrap:wrap;gap:8px;margin:8px 0 12px 0'>"
                                for pk, pv in probe_items:
                                    pct = int(pv * 100)
                                    bar_color = "#10b981" if pv >= 0.6 else "#f59e0b" if pv >= 0.4 else "#ef4444"
                                    probe_html += (
                                        f"<div style='flex:1;min-width:70px'>"
                                        f"<div style='font-size:10px;color:#64748b;margin-bottom:2px'>"
                                        f"{pk.replace('_',' ').title()}</div>"
                                        f"<div style='background:#e0f2fe;border-radius:4px;height:6px;overflow:hidden'>"
                                        f"<div style='background:{bar_color};height:100%;width:{pct}%;border-radius:4px'></div>"
                                        f"</div>"
                                        f"<div style='font-size:9px;color:#94a3b8;text-align:right'>{pv:.2f}</div>"
                                        f"</div>"
                                    )
                                probe_html += "</div>"
                                st.markdown(probe_html, unsafe_allow_html=True)

                        # ── Radar chart + quality explanation ─────────────
                        _radar_col, _explain_col = st.columns([1, 1])
                        with _radar_col:
                            try:
                                from analytics import build_idea_radar
                                _radar_fig = build_idea_radar(idea, title=f"Strength Radar")
                                if _radar_fig:
                                    st.plotly_chart(_radar_fig, use_container_width=True, key=f"radar_{i}")
                            except Exception:
                                pass
                        with _explain_col:
                            # Quality explanation
                            _expl = idea.get("quality_explanation", "")
                            if not _expl and scores:
                                try:
                                    from intelligence import explain_quality
                                    _expl = explain_quality(scores, idea.get("title", ""))
                                except Exception:
                                    pass
                            if _expl:
                                st.markdown("**Quality Analysis**")
                                st.markdown(_expl)
                            # Claim verification
                            try:
                                from intelligence import ClaimVerifier
                                _dag_papers = []
                                if results:
                                    _dag_summary = results.get("dag_summary", {})
                                    _dag_papers = _dag_summary.get("papers", [])
                                if _dag_papers:
                                    _verifier = ClaimVerifier()
                                    _claims = _verifier.verify_against_dag(idea, _dag_papers)
                                    if _claims:
                                        st.markdown("**Claim Verification**")
                                        for _cl in _claims[:3]:
                                            _icon = "✅" if _cl["status"] == "supported" else "❓"
                                            st.caption(f"{_icon} {_cl['claim'][:80]}... — {_cl['evidence'][:60]}")
                            except Exception:
                                pass

                        # ── Idea details in two-column layout ──────────────
                        detail_left, detail_right = st.columns(2)
                        with detail_left:
                            st.markdown("**Motivation**")
                            st.write(idea.get("motivation", "") or "—")
                            st.markdown("**Method**")
                            st.write(idea.get("method", "") or "—")
                            st.markdown("**Hypothesis**")
                            st.write(idea.get("hypothesis", "") or "—")
                        with detail_right:
                            st.markdown("**Resources**")
                            st.write(idea.get("resources", "") or "—")
                            st.markdown("**Expected Outcome**")
                            st.write(idea.get("expected_outcome", "") or "—")
                            st.markdown("**Risk Assessment**")
                            st.write(idea.get("risk_assessment", "") or "—")

                        # Action buttons row: Bookmark + Share + Feedback
                        bk_col1, bk_col2, bk_fb_up, bk_fb_down, bk_col3 = st.columns([1, 1, 0.5, 0.5, 2])
                        with bk_col1:
                            if st.button("🔖 Bookmark", key=f"bk_{i}", type="secondary", use_container_width=True):
                                uid = st.session_state.get("user_id")
                                if uid:
                                    db.bookmark_idea(uid, title, idea)
                                    db_cache.invalidate_user_bookmarks()
                                    st.success("Bookmarked!")
                                else:
                                    st.warning("Log in to bookmark")
                        with bk_col2:
                            if st.button("🔗 Share", key=f"sh_{i}", type="primary", use_container_width=True):
                                uid = st.session_state.get("user_id")
                                if uid:
                                    try:
                                        from sharing import create_share_url
                                        share_url = create_share_url(uid, idea, results.get("topic", ""))
                                        st.session_state[f"_share_url_{i}"] = share_url
                                    except Exception as e:
                                        st.error(f"Share error: {e}")
                        with bk_fb_up:
                            if st.button("👍", key=f"fb_up_{i}", use_container_width=True,
                                         help="Mark as useful"):
                                _fb_uid = st.session_state.get("user_id")
                                if _fb_uid:
                                    db.save_idea_feedback(_fb_uid, title, "useful")
                                    st.success("Thanks!")
                        with bk_fb_down:
                            if st.button("👎", key=f"fb_down_{i}", use_container_width=True,
                                         help="Mark as not useful"):
                                _fb_uid = st.session_state.get("user_id")
                                if _fb_uid:
                                    db.save_idea_feedback(_fb_uid, title, "not_useful")
                                    st.caption("Noted")
                        with bk_col3:
                            if st.session_state.get(f"_share_url_{i}"):
                                st.code(st.session_state[f"_share_url_{i}"], language=None)
                            else:
                                bk_note = st.text_input("Note", key=f"bk_note_{i}", placeholder="Add a note...", label_visibility="collapsed", autocomplete="off")
                                if bk_note and st.button("Save note", key=f"bk_save_{i}"):
                                    uid = st.session_state.get("user_id")
                                    if uid:
                                        db.bookmark_idea(uid, title, idea, note=bk_note)
                                        db_cache.invalidate_user_bookmarks()
                                        st.success("Saved!")

                        # ── Idea Enhancements (Repro / FMEA / Domain / Adversarial) ──
                        _repro = idea.get("_reproducibility")
                        _fmea = idea.get("_fmea")
                        _domain = idea.get("_domain")
                        _adv = idea.get("_adversarial_twin")

                        if _repro or _fmea or _domain:
                            _enh_cols = st.columns(3)
                            with _enh_cols[0]:
                                if _repro:
                                    _rs = _repro.get("score", 0)
                                    _rc = "#10b981" if _rs >= 0.7 else "#f59e0b" if _rs >= 0.4 else "#ef4444"
                                    _rgrad_l = "#f0f9ff" if _rs >= 0.7 else "#fef3c7" if _rs >= 0.4 else "#fef2f2"
                                    _rgrad_r = "#e0f2fe" if _rs >= 0.7 else "#fde68a" if _rs >= 0.4 else "#fecaca"
                                    st.markdown(
                                        f'<div style="background:linear-gradient(135deg,{_rgrad_l},{_rgrad_r});'
                                        f'border:1px solid {_rc}40;'
                                        f'border-radius:10px;padding:10px 12px;text-align:center;'
                                        f'box-shadow:0 1px 3px rgba(0,0,0,0.04)">'
                                        f'<div style="font-size:18px;margin-bottom:2px">🔬</div>'
                                        f'<div style="font-size:9px;color:#0369a1;font-weight:700;'
                                        f'text-transform:uppercase;letter-spacing:0.06em">Reproducibility</div>'
                                        f'<div style="font-size:22px;font-weight:800;color:{_rc};line-height:1.2">'
                                        f'{_rs:.0%}</div>'
                                        f'<div style="font-size:9px;color:#64748b">'
                                        f'{len(_repro.get("missing", []))} specs missing</div>'
                                        f'</div>',
                                        unsafe_allow_html=True,
                                    )
                            with _enh_cols[1]:
                                if _fmea:
                                    _fs = _fmea.get("summary", {})
                                    _hrc = _fs.get("high_risk_count", 0)
                                    _fcolor = "#dc2626" if _hrc > 0 else "#f59e0b"
                                    st.markdown(
                                        f'<div style="background:linear-gradient(135deg,#fef2f2,#fee2e2);'
                                        f'border:1px solid {_fcolor}40;'
                                        f'border-radius:10px;padding:10px 12px;text-align:center;'
                                        f'box-shadow:0 1px 3px rgba(0,0,0,0.04)">'
                                        f'<div style="font-size:18px;margin-bottom:2px">⚠️</div>'
                                        f'<div style="font-size:9px;color:#991b1b;font-weight:700;'
                                        f'text-transform:uppercase;letter-spacing:0.06em">Failure Modes</div>'
                                        f'<div style="font-size:22px;font-weight:800;color:{_fcolor};line-height:1.2">'
                                        f'{_fs.get("total", 0)}</div>'
                                        f'<div style="font-size:9px;color:#64748b">'
                                        f'{_hrc} high-risk</div>'
                                        f'</div>',
                                        unsafe_allow_html=True,
                                    )
                            with _enh_cols[2]:
                                if _domain:
                                    st.markdown(
                                        f'<div style="background:linear-gradient(135deg,#f0fdf4,#dcfce7);'
                                        f'border:1px solid #10b98140;'
                                        f'border-radius:10px;padding:10px 12px;text-align:center;'
                                        f'box-shadow:0 1px 3px rgba(0,0,0,0.04)">'
                                        f'<div style="font-size:18px;margin-bottom:2px">🌍</div>'
                                        f'<div style="font-size:9px;color:#166534;font-weight:700;'
                                        f'text-transform:uppercase;letter-spacing:0.06em">Domain</div>'
                                        f'<div style="font-size:18px;font-weight:800;color:#15803d;line-height:1.2;'
                                        f'margin:2px 0">'
                                        f'{_domain.upper()}</div>'
                                        f'<div style="font-size:9px;color:#64748b">'
                                        f'expert persona</div>'
                                        f'</div>',
                                        unsafe_allow_html=True,
                                    )

                            # FMEA expandable details
                            if _fmea and _fmea.get("failure_modes"):
                                with st.expander(f"⚠️ {len(_fmea['failure_modes'])} Failure Modes & Mitigations"):
                                    for fm in _fmea["failure_modes"]:
                                        _rpn = fm.get("risk_priority", 0)
                                        _color = "#dc2626" if _rpn >= 12 else "#f59e0b" if _rpn >= 8 else "#10b981"
                                        st.markdown(
                                            f'<div style="border-left:3px solid {_color};padding:6px 12px;margin:4px 0">'
                                            f'<b>{fm.get("mode","?")}</b> '
                                            f'(severity {fm.get("severity",0)}/5, '
                                            f'detectability {fm.get("detectability",0)}/5)<br>'
                                            f'<span style="color:#64748b;font-size:12px">'
                                            f'Cause: {fm.get("cause","?")} → Effect: {fm.get("effect","?")}</span><br>'
                                            f'<span style="color:#10b981;font-size:12px">'
                                            f'✅ {fm.get("mitigation","?")}</span>'
                                            f'</div>',
                                            unsafe_allow_html=True,
                                        )

                            # Reproducibility missing items
                            if _repro and _repro.get("missing"):
                                with st.expander(f"🔧 Missing Reproducibility Specs ({len(_repro['missing'])})"):
                                    for _m in _repro["missing"]:
                                        st.markdown(f"- ❌ {_m.title()}")

                            # Adversarial twin
                            if _adv:
                                with st.expander("🔄 Adversarial Twin (contrary version)"):
                                    st.markdown(f"**{_adv.get('title', '')}**")
                                    st.caption(_adv.get("hypothesis", ""))

                        # ── "Ideas Like This" recommendations ─────────────────
                        try:
                            from intelligence import IdeaRecommender
                            _recommender = IdeaRecommender()
                            _similar = _recommender.recommend(idea, sorted_ideas, n=3)
                            if _similar:
                                st.caption("**Similar ideas you might like:**")
                                for _sim in _similar:
                                    _sim_q = _sim.get("quality_score", 0)
                                    _sim_color = "🟢" if _sim_q >= 0.6 else "🟡" if _sim_q >= 0.4 else "🔴"
                                    st.caption(
                                        f"  {_sim_color} {_sim.get('title','?')} "
                                        f"(q={_sim_q:.2f}, match={_sim['_similarity']:.0%}) — "
                                        f"{_sim['_recommendation_reason']}"
                                    )
                        except Exception:
                            pass

                        # ── 🎨 Visual abstract panel ────────────────────────────
                        # Generates a FLUX/Nano-Banana visual abstract for THIS
                        # idea on demand. Cached on disk (sha256 of prompt), so
                        # the same idea never re-bills the image API.
                        # Gated by config.ENABLE_VISUAL_RENDERING — hidden when
                        # the admin toggle is OFF.
                        try:
                            import config as _cfg_visual
                            _visual_on = bool(getattr(
                                _cfg_visual, "ENABLE_VISUAL_RENDERING", True,
                            ))
                        except Exception:
                            _visual_on = True
                        if _visual_on:
                            try:
                                from ideagraph_image_renderer import (
                                    NanoBananaImageRenderer,
                                    display_idea_with_visual,
                                )
                                _visual_ok = True
                            except ImportError:
                                _visual_ok = False
                            if _visual_ok:
                                _visual_key = (
                                    f"_idea_visual_{i}_"
                                    f"{hash(title) & 0xffff:04x}"
                                )
                                with st.expander(
                                    "🎨 Visual abstract", expanded=False,
                                ):
                                    st.caption(
                                        "Generate a paper-figure-style "
                                        "illustration for this idea. "
                                        "Cached on disk per (idea, style)."
                                    )

                                    # Style + N-sample selectors.
                                    # Pull the catalog so adding a new
                                    # style in the renderer surfaces here
                                    # automatically.
                                    try:
                                        from ideagraph_image_renderer import (
                                            IMAGE_STYLE_PRESETS as _STYLES,
                                            DEFAULT_STYLE as _DEFAULT_STYLE,
                                        )
                                    except ImportError:
                                        _STYLES = {}
                                        _DEFAULT_STYLE = "editorial"

                                    _vc1, _vc2 = st.columns([2, 1])
                                    _style_choice = _vc1.selectbox(
                                        "Style",
                                        options=list(_STYLES.keys()) or [_DEFAULT_STYLE],
                                        index=(
                                            list(_STYLES.keys()).index(_DEFAULT_STYLE)
                                            if _DEFAULT_STYLE in _STYLES else 0
                                        ),
                                        format_func=lambda k: (
                                            _STYLES[k].label if k in _STYLES else k
                                        ),
                                        key=f"{_visual_key}_style",
                                        help=(
                                            "Pick a visual style. Each "
                                            "produces a different look. "
                                            "Cache key includes style, "
                                            "so styles are independently "
                                            "cached."
                                        ),
                                    )
                                    _n_samples = _vc2.number_input(
                                        "N samples",
                                        min_value=1, max_value=8,
                                        value=1, step=1,
                                        key=f"{_visual_key}_n",
                                        help=(
                                            "Generate N variants in a "
                                            "row — pick the best. Each "
                                            "sample is a separate API "
                                            "call (cost scales with N)."
                                        ),
                                    )

                                    _cached_visual = (
                                        st.session_state.get(_visual_key)
                                    )
                                    _btn_cols = st.columns([2, 1, 1])
                                    if _btn_cols[0].button(
                                        f"🎨 Generate "
                                        f"({int(_n_samples)} sample"
                                        f"{'s' if int(_n_samples) != 1 else ''})",
                                        type="primary",
                                        use_container_width=True,
                                        key=f"{_visual_key}_go",
                                    ):
                                        try:
                                            _renderer = NanoBananaImageRenderer()
                                            if int(_n_samples) > 1:
                                                with st.spinner(
                                                    f"Rendering "
                                                    f"{int(_n_samples)} "
                                                    f"variants via "
                                                    f"{_renderer.provider.name}…"
                                                ):
                                                    _samples = _renderer.render_n_samples(
                                                        idea,
                                                        n=int(_n_samples),
                                                        style=_style_choice,
                                                    )
                                                st.session_state[
                                                    f"{_visual_key}_samples"
                                                ] = _samples
                                                # Show the first one as the
                                                # "primary" cached visual.
                                                if _samples:
                                                    _cached_visual = _samples[0]
                                                    st.session_state[
                                                        _visual_key
                                                    ] = _cached_visual
                                            else:
                                                with st.spinner(
                                                    "Rendering via "
                                                    f"{_renderer.provider.name}… "
                                                    "(~10–30s)"
                                                ):
                                                    _from_renderer = (
                                                        __import__(
                                                            'ideagraph_image_renderer',
                                                            fromlist=['build_prompt'],
                                                        )
                                                    )
                                                    _styled_prompt = (
                                                        _from_renderer.build_prompt(
                                                            idea, style=_style_choice,
                                                        )
                                                    )
                                                    _v = _renderer.render(
                                                        idea,
                                                        prompt_override=_styled_prompt,
                                                    )
                                                st.session_state[_visual_key] = _v
                                                st.session_state.pop(
                                                    f"{_visual_key}_samples", None,
                                                )
                                                _cached_visual = _v
                                        except Exception as _ve:
                                            st.error(
                                                f"Renderer error: {_ve}"
                                            )

                                    # If we have multi-sample results,
                                    # render them as a grid so the user
                                    # can compare and pick the best.
                                    _multi_samples = st.session_state.get(
                                        f"{_visual_key}_samples"
                                    )
                                    if _multi_samples and len(_multi_samples) > 1:
                                        st.markdown(
                                            f"**{len(_multi_samples)} "
                                            f"variants — click a thumbnail "
                                            f"to set it as the primary**"
                                        )
                                        _sample_cols = st.columns(
                                            min(len(_multi_samples), 4),
                                        )
                                        for _si, _sv in enumerate(_multi_samples):
                                            _col = _sample_cols[
                                                _si % len(_sample_cols)
                                            ]
                                            with _col:
                                                if _sv.success:
                                                    _src = (
                                                        _sv.cached_path
                                                        or _sv.image_url
                                                    )
                                                    try:
                                                        st.image(
                                                            _src,
                                                            use_container_width=True,
                                                        )
                                                    except TypeError:
                                                        st.image(
                                                            _src,
                                                            use_column_width=True,
                                                        )
                                                    if st.button(
                                                        f"✓ Use #{_si + 1}",
                                                        key=f"{_visual_key}_pick_{_si}",
                                                        use_container_width=True,
                                                    ):
                                                        st.session_state[
                                                            _visual_key
                                                        ] = _sv
                                                        _cached_visual = _sv
                                                        st.rerun()
                                                else:
                                                    st.caption(
                                                        f"❌ {_sv.error or 'failed'}"
                                                    )
                                    if _cached_visual is not None and _btn_cols[1].button(
                                        "↻ Re-roll",
                                        use_container_width=True,
                                        key=f"{_visual_key}_force",
                                        help="Bypass the cache and "
                                              "call the API again.",
                                    ):
                                        try:
                                            _renderer = NanoBananaImageRenderer()
                                            with st.spinner("Re-rendering…"):
                                                _v = _renderer.render(
                                                    idea, force=True,
                                                )
                                            st.session_state[_visual_key] = _v
                                            _cached_visual = _v
                                        except Exception as _ve:
                                            st.error(
                                                f"Re-render failed: {_ve}"
                                            )
                                    if _cached_visual is not None and _btn_cols[2].button(
                                        "🗑️ Clear",
                                        use_container_width=True,
                                        key=f"{_visual_key}_clear",
                                    ):
                                        st.session_state.pop(
                                            _visual_key, None,
                                        )
                                        _cached_visual = None
                                        st.rerun()
                                    if _cached_visual is not None:
                                        display_idea_with_visual(
                                            idea, st,
                                            visual=_cached_visual,
                                            show_prompt=True,
                                        )
                                        # 📥 Download the generated asset
                                        # with a meaningful filename.
                                        try:
                                            from ideagraph_image_renderer import (
                                                read_visual_bytes as _read_bytes,
                                                safe_filename as _safe_fn,
                                            )
                                            _dl_bytes = _read_bytes(_cached_visual)
                                            if _dl_bytes is not None:
                                                _dl_name = _safe_fn(
                                                    title,
                                                    style=_style_choice,
                                                    media_type=(
                                                        _cached_visual.media_type
                                                    ),
                                                )
                                                _dl_mime = (
                                                    "video/mp4"
                                                    if _cached_visual.is_video
                                                    else "image/png"
                                                )
                                                st.download_button(
                                                    "📥 Download this visual",
                                                    data=_dl_bytes,
                                                    file_name=_dl_name,
                                                    mime=_dl_mime,
                                                    use_container_width=True,
                                                    key=f"{_visual_key}_dl",
                                                )
                                            elif _cached_visual.image_url:
                                                # No local bytes & couldn't
                                                # fetch URL — show the URL
                                                # so the user can save it
                                                # via their browser.
                                                st.caption(
                                                    f"Direct URL: "
                                                    f"[{_cached_visual.image_url[:60]}…]"
                                                    f"({_cached_visual.image_url})"
                                                )
                                        except ImportError:
                                            pass

                                    # ── 📊 Multi-panel paper figure ──
                                    # Generates a coordinated set of
                                    # panel-specific images (concept +
                                    # method + experiment + results) —
                                    # the way a real paper has Fig 1-4.
                                    # Inspired by what Kimi does for
                                    # PPT generation with Nano Banana.
                                    try:
                                        from ideagraph_image_renderer import (
                                            FIGURE_TEMPLATES,
                                            DEFAULT_FIGURE_SET,
                                            FigurePanel as _FigurePanel,
                                        )
                                        _figset_ok = True
                                    except ImportError:
                                        _figset_ok = False

                                    if _figset_ok:
                                        st.markdown(
                                            "<div style='margin-top:18px;"
                                            "font-size:11px;color:#64748b;"
                                            "text-transform:uppercase;"
                                            "letter-spacing:0.06em;"
                                            "font-weight:700'>"
                                            "📊 Multi-panel paper figure"
                                            "</div>",
                                            unsafe_allow_html=True,
                                        )
                                        st.caption(
                                            "Generate a coordinated set "
                                            "of paper-figure panels "
                                            "(like Kimi's PPT mode) — "
                                            "Fig 1 (concept), Fig 2 "
                                            "(method), Fig 3 "
                                            "(experiment), Fig 4 "
                                            "(results). Pick which "
                                            "panels you want."
                                        )
                                        _figset_key = (
                                            f"_idea_figset_{i}_"
                                            f"{hash(title) & 0xffff:04x}"
                                        )
                                        _panel_options = list(
                                            FIGURE_TEMPLATES.keys()
                                        )
                                        _figset_panels = st.multiselect(
                                            "Panels to generate",
                                            options=_panel_options,
                                            default=list(DEFAULT_FIGURE_SET),
                                            format_func=lambda k: (
                                                FIGURE_TEMPLATES[k]["label"]
                                            ),
                                            key=f"{_figset_key}_panels",
                                            help="Each panel uses its "
                                                  "own focused prompt. "
                                                  "Independently cached.",
                                        )
                                        if st.button(
                                            f"📊 Generate {len(_figset_panels)} "
                                            f"panel(s)",
                                            type="primary",
                                            use_container_width=True,
                                            key=f"{_figset_key}_go",
                                            disabled=(
                                                not _figset_panels
                                            ),
                                        ):
                                            try:
                                                _renderer_fs = (
                                                    NanoBananaImageRenderer()
                                                )
                                                with st.spinner(
                                                    f"Rendering "
                                                    f"{len(_figset_panels)} "
                                                    "panels…"
                                                ):
                                                    _panels_out = (
                                                        _renderer_fs.render_figure_set(
                                                            idea,
                                                            panels=list(
                                                                _figset_panels
                                                            ),
                                                        )
                                                    )
                                                st.session_state[
                                                    _figset_key
                                                ] = _panels_out
                                            except Exception as _fe:
                                                st.error(
                                                    f"Figure set error: "
                                                    f"{_fe}"
                                                )

                                        _panels_cached = (
                                            st.session_state.get(_figset_key)
                                        )
                                        if _panels_cached:
                                            _ok = [
                                                p for p in _panels_cached
                                                if p.visual.success
                                            ]
                                            _fail = [
                                                p for p in _panels_cached
                                                if not p.visual.success
                                            ]
                                            st.caption(
                                                f"✅ {len(_ok)} ok · "
                                                f"❌ {len(_fail)} failed"
                                            )
                                            # 2-column grid for the
                                            # panels (Streamlit auto-
                                            # stacks on mobile via the
                                            # existing responsive CSS).
                                            _cols_pair = st.columns(2)
                                            for _pi, _panel in enumerate(
                                                _panels_cached
                                            ):
                                                _col = _cols_pair[_pi % 2]
                                                with _col:
                                                    st.markdown(
                                                        f"**{_panel.label}**"
                                                    )
                                                    if _panel.visual.success:
                                                        _src = (
                                                            _panel.visual.cached_path
                                                            or _panel.visual.image_url
                                                        )
                                                        try:
                                                            st.image(
                                                                _src,
                                                                use_container_width=True,
                                                            )
                                                        except TypeError:
                                                            st.image(
                                                                _src,
                                                                use_column_width=True,
                                                            )
                                                    else:
                                                        st.warning(
                                                            f"❌ {_panel.visual.error or 'failed'}"
                                                        )
                                            # 📦 Bundle the entire
                                            # figure set as a single ZIP
                                            # with meaningful filenames.
                                            try:
                                                from ideagraph_image_renderer import (
                                                    bundle_visuals_as_zip as _bundle_zip,
                                                    safe_filename as _safe_fn_zip,
                                                )
                                                _ok_panels = [
                                                    p for p in _panels_cached
                                                    if p.visual.success
                                                ]
                                                if _ok_panels:
                                                    _zip_bytes = _bundle_zip(
                                                        [p.visual for p in _ok_panels],
                                                        idea_title=title,
                                                        panel_labels=[
                                                            p.panel_id
                                                            for p in _ok_panels
                                                        ],
                                                    )
                                                    if _zip_bytes:
                                                        _zip_name = (
                                                            _safe_fn_zip(
                                                                title,
                                                                style="figureset",
                                                                media_type="image",
                                                            ).replace(".png", ".zip")
                                                        )
                                                        st.download_button(
                                                            f"📦 Download all "
                                                            f"{len(_ok_panels)} "
                                                            f"panels as ZIP",
                                                            data=_zip_bytes,
                                                            file_name=_zip_name,
                                                            mime="application/zip",
                                                            use_container_width=True,
                                                            key=f"{_figset_key}_zipdl",
                                                        )
                                            except ImportError:
                                                pass

                        # ── Chat-to-optimize panel ──────────────────────────────
                        # Lets the user converse with the LLM about THIS idea,
                        # then crystallize the dialogue into a refined version
                        # (source_strategy='V', generation += 1). Six optimizations
                        # live here: 4 chat modes, quick-action prompts, regenerate,
                        # auto-truncate, diff view, markdown export.
                        try:
                            from idea_chat import (
                                ChatMessage as _ChatMessage,
                                CHAT_MODES as _CHAT_MODES,
                                DEFAULT_MODE as _DEFAULT_MODE,
                                SUGGESTED_PROMPTS as _SUGGESTED_PROMPTS,
                                chat_turn as _chat_turn,
                                crystallize as _crystallize,
                                diff_ideas as _diff_ideas,
                                estimate_turn_tokens as _estimate_tokens,
                                export_markdown as _export_md,
                                regenerate_last as _regenerate_last,
                                truncate_history as _truncate_history,
                            )
                            _chat_ok = True
                        except ImportError:
                            _chat_ok = False

                        if _chat_ok:
                            _chat_key = f"_idea_chat_{i}_{hash(title) & 0xffff:04x}"
                            _mode_key = f"{_chat_key}_mode"
                            _last_diff_key = f"{_chat_key}_last_diff"
                            if _chat_key not in st.session_state:
                                st.session_state[_chat_key] = []
                            if _mode_key not in st.session_state:
                                st.session_state[_mode_key] = _DEFAULT_MODE
                            _hist_dicts = st.session_state[_chat_key]
                            _n_turns = len(_hist_dicts)
                            _current_mode = st.session_state[_mode_key]

                            # Pending quick-prompt selection (set by a chip click).
                            _pending_key = f"{_chat_key}_pending"
                            _pending_prompt = st.session_state.pop(
                                _pending_key, None,
                            )

                            with st.expander(
                                f"💬 Chat to optimize this idea"
                                + (f" ({_n_turns} turn{'s' if _n_turns != 1 else ''})"
                                    if _n_turns else ""),
                                expanded=False,
                            ):
                                st.caption(
                                    "Talk to the LLM about THIS specific idea. "
                                    "Switch **mode** to change the assistant's "
                                    "stance. Click **Crystallize** to save a "
                                    "refined version (strategy `V`)."
                                )

                                # ── Mode selector ───────────────────────────────
                                _mode_options = list(_CHAT_MODES.keys())
                                _mode_idx = (_mode_options.index(_current_mode)
                                                if _current_mode in _mode_options
                                                else 0)
                                _selected_mode = st.radio(
                                    "Mode",
                                    options=_mode_options,
                                    index=_mode_idx,
                                    format_func=lambda k: (
                                        f"{_CHAT_MODES[k]['label']} — "
                                        f"{_CHAT_MODES[k]['description']}"
                                    ),
                                    horizontal=False,
                                    key=f"{_chat_key}_mode_radio",
                                    label_visibility="collapsed",
                                )
                                if _selected_mode != _current_mode:
                                    st.session_state[_mode_key] = _selected_mode
                                    _current_mode = _selected_mode

                                # ── Auto-truncate banner ───────────────────────
                                _was_truncated = False
                                if _n_turns > 20:
                                    _hist_objs_for_check = [
                                        _ChatMessage(role=m["role"],
                                                        content=m["content"])
                                        for m in _hist_dicts
                                    ]
                                    _truncated_objs, _was_truncated = \
                                        _truncate_history(
                                            _hist_objs_for_check,
                                            max_turns=20, keep_recent=10,
                                        )
                                    if _was_truncated:
                                        st.info(
                                            f"ℹ️ Chat is {_n_turns} turns long. "
                                            f"Next call will use a truncated "
                                            f"history (most recent 10 turns + "
                                            f"context placeholder) to stay in "
                                            f"the token budget."
                                        )

                                # ── Token / cost estimate ──────────────────────
                                _tok = _estimate_tokens(
                                    idea, [
                                        _ChatMessage(role=m["role"],
                                                        content=m["content"])
                                        for m in _hist_dicts
                                    ],
                                    user_message="",
                                    mode=_current_mode,
                                )
                                st.caption(
                                    f"~{_tok['total_input']} input tokens "
                                    f"queued · system {_tok['system']} · "
                                    f"context {_tok['context']} · "
                                    f"history {_tok['history']}"
                                )

                                # ── Render the conversation so far ─────────────
                                for _msg in _hist_dicts:
                                    _role = _msg.get("role", "user")
                                    _content = _msg.get("content", "")
                                    if _role == "user":
                                        st.markdown(
                                            f"<div style='background:#eff6ff;"
                                            f"border-left:3px solid #3b82f6;"
                                            f"padding:8px 12px;border-radius:6px;"
                                            f"margin:4px 0;font-size:13px;"
                                            f"color:#1e3a8a'>"
                                            f"<b>You:</b> {_html_mod.escape(_content)}"
                                            f"</div>",
                                            unsafe_allow_html=True,
                                        )
                                    else:
                                        st.markdown(
                                            f"<div style='background:#f1f5f9;"
                                            f"border-left:3px solid #64748b;"
                                            f"padding:8px 12px;border-radius:6px;"
                                            f"margin:4px 0;font-size:13px;"
                                            f"color:#0f172a'>"
                                            f"<b>Assistant:</b> "
                                            f"{_html_mod.escape(_content)}"
                                            f"</div>",
                                            unsafe_allow_html=True,
                                        )

                                # ── Quick-action prompt chips ──────────────────
                                st.markdown(
                                    "<div style='font-size:11px;color:#64748b;"
                                    "text-transform:uppercase;letter-spacing:"
                                    "0.06em;margin:6px 0 4px 0;font-weight:700'>"
                                    "Quick prompts</div>",
                                    unsafe_allow_html=True,
                                )
                                _chip_cols = st.columns(4)
                                for _ci, _chip in enumerate(_SUGGESTED_PROMPTS):
                                    if _chip_cols[_ci % 4].button(
                                        _chip["label"],
                                        key=f"{_chat_key}_chip_{_ci}",
                                        help=_chip["prompt"],
                                        use_container_width=True,
                                    ):
                                        st.session_state[_pending_key] = \
                                            _chip["prompt"]
                                        st.rerun()

                                # ── Compose form ───────────────────────────────
                                with st.form(
                                    key=f"{_chat_key}_form",
                                    clear_on_submit=True,
                                ):
                                    _initial_value = _pending_prompt or ""
                                    _user_msg = st.text_area(
                                        "Your message",
                                        value=_initial_value,
                                        key=f"{_chat_key}_input",
                                        placeholder="e.g. The method is too "
                                                      "expensive — suggest a "
                                                      "cheaper variant that "
                                                      "still tests the hypothesis.",
                                        height=80,
                                        label_visibility="collapsed",
                                    )
                                    _form_cols = st.columns([1, 1, 1, 1])
                                    _send = _form_cols[0].form_submit_button(
                                        "Send", type="primary",
                                        use_container_width=True,
                                    )
                                    _can_regen = (
                                        _n_turns >= 2
                                        and _hist_dicts[-1].get("role") == "assistant"
                                        and _hist_dicts[-2].get("role") == "user"
                                    )
                                    _regen = _form_cols[1].form_submit_button(
                                        "↻ Re-roll",
                                        use_container_width=True,
                                        disabled=(not _can_regen),
                                        help="Re-roll the assistant's last "
                                              "reply with the same user "
                                              "message + current mode.",
                                    )
                                    _clear = _form_cols[2].form_submit_button(
                                        "Clear",
                                        use_container_width=True,
                                    )
                                    _crystallize_btn = _form_cols[3].form_submit_button(
                                        "💾 Crystallize",
                                        use_container_width=True,
                                        disabled=(_n_turns == 0),
                                        help="Save the discussion as a "
                                              "refined idea with strategy V.",
                                    )

                                if _send and _user_msg and _user_msg.strip():
                                    # Append user turn first so it's visible if
                                    # the LLM call fails.
                                    st.session_state[_chat_key].append(
                                        {"role": "user",
                                          "content": _user_msg.strip()}
                                    )
                                    _history_objs = [
                                        _ChatMessage(role=m["role"],
                                                        content=m["content"])
                                        for m in st.session_state[_chat_key][:-1]
                                    ]
                                    # Apply auto-truncation if history is long.
                                    _history_objs, _ = _truncate_history(
                                        _history_objs,
                                        max_turns=20, keep_recent=10,
                                    )
                                    with st.spinner(
                                        f"Thinking ({_CHAT_MODES[_current_mode]['label']})…"
                                    ):
                                        _reply = _chat_turn(
                                            idea, _history_objs,
                                            _user_msg.strip(),
                                            mode=_current_mode,
                                        )
                                    if _reply:
                                        st.session_state[_chat_key].append(
                                            {"role": "assistant",
                                              "content": _reply}
                                        )
                                    else:
                                        st.session_state[_chat_key].append({
                                            "role": "assistant",
                                            "content": "⚠️ LLM call failed. "
                                                       "Check that an API key "
                                                       "is configured.",
                                        })
                                    st.rerun()

                                if _regen and _can_regen:
                                    _history_objs = [
                                        _ChatMessage(role=m["role"],
                                                        content=m["content"])
                                        for m in st.session_state[_chat_key]
                                    ]
                                    with st.spinner("Re-rolling last reply…"):
                                        _new_reply = _regenerate_last(
                                            idea, _history_objs,
                                            mode=_current_mode,
                                        )
                                    if _new_reply:
                                        st.session_state[_chat_key][-1] = {
                                            "role": "assistant",
                                            "content": _new_reply,
                                        }
                                        st.rerun()
                                    else:
                                        st.error("Re-roll failed.")

                                if _clear:
                                    st.session_state[_chat_key] = []
                                    st.session_state.pop(_last_diff_key, None)
                                    st.rerun()

                                if _crystallize_btn and _n_turns > 0:
                                    _history_objs = [
                                        _ChatMessage(role=m["role"],
                                                        content=m["content"])
                                        for m in st.session_state[_chat_key]
                                    ]
                                    with st.spinner("Crystallizing chat into refined idea…"):
                                        _new_idea = _crystallize(
                                            idea, _history_objs,
                                        )
                                    if _new_idea is None:
                                        st.error("Crystallization failed — try "
                                                  "more turns or check the LLM "
                                                  "client.")
                                    else:
                                        _diff = _diff_ideas(idea, _new_idea)
                                        ideas.append(_new_idea.to_dict())
                                        st.session_state[_last_diff_key] = _diff
                                        st.session_state[_chat_key] = []
                                        st.success(
                                            f"Saved refined idea (strategy V, "
                                            f"generation {_new_idea.generation}). "
                                            f"{len(_diff)} field(s) changed. "
                                            f"Find it at the bottom of the list."
                                        )
                                        st.rerun()

                                # ── Diff view (after last crystallize) ─────────
                                _diff_state = st.session_state.get(_last_diff_key)
                                if _diff_state:
                                    with st.expander(
                                        f"🔍 What changed in the last crystallize "
                                        f"({len(_diff_state)} field(s))",
                                        expanded=False,
                                    ):
                                        for _f, _ba in _diff_state.items():
                                            _label = _f.replace("_", " ").title()
                                            st.markdown(f"**{_label}**")
                                            _c1, _c2 = st.columns(2)
                                            _c1.markdown(
                                                f"<div style='background:#fef2f2;"
                                                f"border-left:3px solid #ef4444;"
                                                f"padding:6px 10px;border-radius:"
                                                f"4px;font-size:12px;color:#7f1d1d'>"
                                                f"<b>Before</b><br>"
                                                f"{_html_mod.escape(_ba['before'] or '—')}"
                                                f"</div>",
                                                unsafe_allow_html=True,
                                            )
                                            _c2.markdown(
                                                f"<div style='background:#f0fdf4;"
                                                f"border-left:3px solid #10b981;"
                                                f"padding:6px 10px;border-radius:"
                                                f"4px;font-size:12px;color:#14532d'>"
                                                f"<b>After</b><br>"
                                                f"{_html_mod.escape(_ba['after'] or '—')}"
                                                f"</div>",
                                                unsafe_allow_html=True,
                                            )
                                        if st.button(
                                            "Dismiss diff",
                                            key=f"{_chat_key}_dismiss_diff",
                                        ):
                                            st.session_state.pop(_last_diff_key, None)
                                            st.rerun()

                                # ── Export to markdown ─────────────────────────
                                if _n_turns > 0:
                                    _md = _export_md(
                                        idea,
                                        [_ChatMessage(role=m["role"],
                                                          content=m["content"])
                                            for m in _hist_dicts],
                                        mode=_current_mode,
                                    )
                                    _safe_title = "".join(
                                        c for c in title[:40]
                                        if c.isalnum() or c in " -_"
                                    ).strip().replace(" ", "_") or "chat"
                                    st.download_button(
                                        "📥 Download chat as Markdown",
                                        data=_md,
                                        file_name=f"chat_{_safe_title}.md",
                                        mime="text/markdown",
                                        key=f"{_chat_key}_export",
                                        use_container_width=True,
                                    )

            # ── Bookmarks Section ─────────────────────────────────────────
            st.divider()
            st.markdown("### My Bookmarks")
            uid_bk = st.session_state.get("user_id")
            if uid_bk:
                collections = db_cache.get_bookmark_collections(uid_bk)
                if collections:
                    col_filter = st.selectbox("Collection", ["All"] + collections, key="bk_collection_filter")
                    bookmarks = db_cache.get_bookmarks(uid_bk, collection=None if col_filter == "All" else col_filter)
                else:
                    bookmarks = db_cache.get_bookmarks(uid_bk)

                # Search
                bk_search = st.text_input("Search bookmarks", key="bk_search", placeholder="Search by title, note, or tag...", autocomplete="off")
                if bk_search:
                    bookmarks = db.search_bookmarks(uid_bk, bk_search)

                if bookmarks:
                    st.caption(f"{len(bookmarks)} bookmarks")
                    for bk in bookmarks:
                        idea_data = bk.get("idea", {})
                        bk_q = idea_data.get("quality_score", 0)
                        bk_color = "🟢" if bk_q >= 0.6 else "🟡" if bk_q >= 0.4 else "🔴"
                        with st.expander(f"{bk_color} {bk.get('idea_title', '?')} — {bk.get('created_at', '')[:10]}"):
                            st.metric("Quality", f"{bk_q:.3f}")
                            if bk.get("note"):
                                st.info(f"Note: {bk['note']}")
                            if bk.get("tags"):
                                st.caption(f"Tags: {bk['tags']}")
                            st.caption(f"Collection: {bk.get('collection', 'default')}")
                            st.write(idea_data.get("method", ""))
                            if st.button("Remove", key=f"rm_bk_{bk['id']}"):
                                db.delete_bookmark(bk["id"], uid_bk)
                                db_cache.invalidate_user_bookmarks()
                                st.rerun()
                else:
                    st.caption("No bookmarks yet. Click 'Bookmark' on any idea above.")
            else:
                st.info("Log in to use bookmarks.")

    # ── Tab 3: DAG ─────────────────────────────────────────────────────────
    with tab_dag:
        st.subheader("Knowledge DAG & Idea Explorer")

        dag_col1, dag_col2, dag_col3 = st.columns(3)
        with dag_col1:
            st.metric("Papers", dag_summary.get("node_count", 0))
        with dag_col2:
            st.metric("Edges", dag_summary.get("edge_count", 0))
        with dag_col3:
            st.metric("Clusters", dag_summary.get("cluster_count", 0))

        # ── Plotly Graph Explorer (new) ───────────────────────────────────
        try:
            from graph_explorer import build_dag_figure, build_idea_connection_graph
            dag_fig = build_dag_figure(dag_summary, ideas)
            if dag_fig:
                st.plotly_chart(dag_fig, use_container_width=True)

            # Idea connection graph
            if ideas and len(ideas) >= 3:
                st.markdown("##### Idea Connection Graph")
                idea_fig = build_idea_connection_graph(ideas)
                if idea_fig:
                    st.plotly_chart(idea_fig, use_container_width=True)
        except Exception as e:
            st.caption(f"Graph explorer: {e}")

        # ── Evolution Timeline ────────────────────────────────────────────
        uid_evo = st.session_state.get("user_id")
        if uid_evo:
            try:
                from idea_evolution import build_quality_trajectory_chart, get_idea_history, diff_idea_versions

                st.markdown("##### Quality Evolution Over Time")
                evo_fig = build_quality_trajectory_chart(uid_evo)
                if evo_fig:
                    st.plotly_chart(evo_fig, use_container_width=True)
                else:
                    st.caption("Run 2+ pipelines and save results to see quality evolution.")

                # Idea version history
                history = get_idea_history(uid_evo)
                if history:
                    st.markdown("##### Idea Version History")
                    multi_version = {k: v for k, v in history.items() if len(v) >= 2}
                    if multi_version:
                        for key, versions in list(multi_version.items())[:5]:
                            title = versions[-1].get("title", key)
                            with st.expander(f"📜 {title} ({len(versions)} versions)"):
                                for vi, v in enumerate(versions):
                                    q = v.get("quality_score", 0)
                                    color = "🟢" if q >= 0.6 else "🟡" if q >= 0.4 else "🔴"
                                    st.caption(
                                        f"v{vi+1} — {color} q={q:.3f} — "
                                        f"{v.get('_run_date', '?')} — {v.get('_run_topic', '')[:30]}"
                                    )
                                # Diff between first and last
                                if len(versions) >= 2:
                                    diff = diff_idea_versions(versions[0], versions[-1])
                                    if diff:
                                        st.markdown("**Changes (v1 → latest):**")
                                        for field, changes in diff.items():
                                            if isinstance(changes, dict) and "old" in changes and "new" in changes:
                                                st.markdown(f"- **{field}**: {changes.get('old', '')[:80]} → {changes.get('new', '')[:80]}")
                    else:
                        st.caption("Run the same topic multiple times to track idea evolution.")
            except Exception as e:
                st.caption(f"Evolution timeline: {e}")

        st.divider()

        # Interactive network graph
        papers = dag_summary.get("papers", [])
        edges = dag_summary.get("edges", [])
        if papers and edges:
            try:
                from pyvis.network import Network
                import streamlit.components.v1 as components
                import tempfile, os

                net = Network(height="500px", width="100%", bgcolor="#1a1a2e",
                              font_color="white", directed=True)
                net.barnes_hut(gravity=-3000, spring_length=150)

                # Color palette for clusters
                colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
                          "#1abc9c", "#e67e22", "#34495e", "#e91e63", "#00bcd4"]

                for p in papers:
                    pid = str(p.get("id", ""))
                    cluster = p.get("cluster", 0)
                    is_frontier = p.get("frontier", False)
                    color = colors[cluster % len(colors)]
                    shape = "star" if is_frontier else "dot"
                    net.add_node(pid,
                                 label=p.get("title", ""),
                                 title=f"{p.get('title', '')} ({p.get('year', '?')})\nCitations: {p.get('citations', 0)}",
                                 color=color, shape=shape,
                                 size=15 + min(p.get("citations", 0) // 5, 20))

                edge_colors = {"extends": "#3498db", "applies": "#2ecc71", "combines": "#f39c12",
                               "contrasts": "#e74c3c", "generalizes": "#9b59b6", "enables": "#1abc9c"}
                for e in edges:
                    etype = e.get("type", "extends")
                    net.add_edge(str(e.get("from", "")), str(e.get("to", "")),
                                 color=edge_colors.get(etype, "#888"), title=etype)

                with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
                    net.save_graph(f.name)
                    with open(f.name, 'r') as hf:
                        html_content = hf.read()
                    os.unlink(f.name)

                st.markdown("### Interactive Knowledge Graph")
                st.caption("Stars = frontier papers, size = citation count, colors = clusters")
                components.html(html_content, height=520)
            except ImportError:
                st.caption("Install pyvis for interactive graph: `pip install pyvis`")
            except Exception as e:
                st.caption(f"Graph visualization error: {e}")

        clusters = dag_summary.get("clusters", [])
        if clusters:
            st.markdown("### Research Clusters")
            for cluster in clusters:
                theme = cluster.get("theme", f"Cluster {cluster.get('cluster_id')}")
                maturity = cluster.get("maturity", "")
                n_papers = cluster.get("paper_count", 0)
                oqs = cluster.get("open_questions", [])

                with st.expander(f"Cluster {cluster.get('cluster_id')}: {theme} ({n_papers} papers, {maturity})"):
                    if oqs:
                        st.markdown("**Open Questions**")
                        for oq in oqs:
                            st.markdown(f"- {oq}")
                    else:
                        st.caption("No open questions annotated.")
        else:
            st.info("No cluster data available.")

        # Frontier papers
        frontier_papers = [
            p for p in dag_summary.get("papers", []) if p.get("frontier")
        ]
        if frontier_papers:
            st.markdown("### Frontier Papers")
            st.caption("These papers have no known successors — ideal starting points for new ideas.")
            for fp in frontier_papers[:15]:
                st.markdown(
                    f"- **{fp.get('title', 'Untitled')}** ({fp.get('year', '?')}) "
                    f"— Cluster {fp.get('cluster', '?')}"
                )

    # ── Chat-with-result Tab ───────────────────────────────────────────────
    # Conversational interface anchored to the loaded result. Different
    # from idea_chat (single-idea dialogue): this scopes the model's
    # context to ALL ideas in the result + the topic, so users can ask
    # cross-idea questions like "rank these by 2-week feasibility" or
    # "find the two that most contradict each other".
    with tab_result_chat:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">'
            '<span style="font-size:22px">💬</span>'
            '<span style="font-size:18px;font-weight:700;color:#0c4a6e">'
            'Chat with this result</span></div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Ask anything about the ideas in this result: compare, "
            "rank, summarize, find contradictions, group by methodology. "
            "Per-result transcript is preserved across reruns."
        )
        try:
            import result_chat as _rc_mod
            _rc_result_id = None
            try:
                _rc_result_id = (
                    st.session_state.get("results", {}) or {}
                ).get("_result_id")
            except Exception:
                _rc_result_id = None
            _rc_mod.render_chat_panel(
                st, st.session_state.get("results"), _rc_result_id,
            )
        except Exception as _rc_err:
            st.error(f"Result chat unavailable: {_rc_err}")

    # ── Regenerate Tab ─────────────────────────────────────────────────────
    # Pick any existing idea, choose a regeneration mode, and produce N
    # derivative ideas via LLM. Lineage (parent_title, generation) is set
    # automatically so the new ideas can flow back into the rest of the app.
    with tab_regenerate:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">'
            '<span style="font-size:22px">🔄</span>'
            '<span style="font-size:18px;font-weight:700;color:#0c4a6e">'
            'Regenerate Ideas</span></div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Two modes: derive from a single parent idea (refine/extend/pivot/"
            "contrast/cross-domain/mutate/topic-transplant), or take a "
            "**fresh take** on the same topic with new sources/angles that "
            "don't duplicate the existing ideas."
        )

        if not ideas:
            st.info("Run a pipeline first so there's something to regenerate from.")
        else:
            try:
                from idea_regenerator import (
                    regenerate as _regen,
                    regenerate_fresh as _regen_fresh,
                    REGEN_MODES,
                )
            except ImportError as _e:
                st.error(f"idea_regenerator unavailable: {_e}")
                _regen = None
                _regen_fresh = None

            _regen_op_mode = st.radio(
                "Operation",
                options=["from_parent", "fresh_take"],
                format_func=lambda k: (
                    "🌳 From a parent — derive new ideas from one existing idea"
                    if k == "from_parent"
                    else "🔀 Fresh take — same topic, different sources"
                ),
                horizontal=False,
                key="regen_op_mode",
            )

        # Branch on operation mode. The original parent-driven UI runs only
        # when from_parent is selected; fresh_take has its own block below.
        if ideas and _regen_op_mode == "fresh_take" and _regen_fresh is not None:
            _topic = results.get("topic", "")
            st.markdown(
                f"<div style='background:#f0fdf4;border-left:4px solid #10b981;"
                f"padding:10px 14px;margin:8px 0;border-radius:6px;font-size:13px'>"
                f"<b>🔀 Fresh Take.</b> Generate new ideas on the SAME topic but "
                f"explicitly avoid duplicating any of your existing "
                f"{len(ideas)} ideas. The LLM is told to prefer "
                f"under-represented methodologies, different theoretical "
                f"angles, and different prior-work lineages.</div>",
                unsafe_allow_html=True,
            )

            _ftc1, _ftc2 = st.columns([2, 1])
            with _ftc1:
                _ft_topic = st.text_input(
                    "Topic",
                    value=_topic,
                    help="Defaults to the topic of the current run. Edit to "
                         "broaden or narrow the framing.",
                    key="_regen_fresh_topic",
                    autocomplete="off",
                )
            with _ftc2:
                _ft_n = st.number_input(
                    "How many fresh ideas",
                    min_value=1, max_value=8, value=3, step=1,
                    key="_regen_fresh_n",
                )

            # Show methodology distribution of existing ideas so user sees
            # what the LLM will be told to AVOID
            from collections import Counter as _Counter
            _meth_dist = _Counter(
                (i.get("methodology_type") or "?").replace("_", " ").title()
                for i in ideas
            )
            _meth_chart = " · ".join(
                f"{m} ({c})" for m, c in _meth_dist.most_common()
            )
            st.caption(
                f"**Existing methodology distribution:** {_meth_chart}. "
                "The LLM will be steered toward unused / under-represented "
                "methodology types for genuine diversity."
            )

            with st.expander(
                f"📋 The {len(ideas)} existing ideas the LLM will avoid",
                expanded=False,
            ):
                for _i, _it in enumerate(
                        sorted(ideas,
                                key=lambda x: x.get("quality_score", 0),
                                reverse=True),
                        1):
                    _q = _it.get("quality_score", 0)
                    _t = _it.get("title", "Untitled")
                    _m = (_it.get("methodology_type") or "?").replace("_", " ")
                    _n = _it.get("novelty_level") or "?"
                    st.markdown(
                        f"  {_i}. **[{_m} × {_n}]** {_t}  ·  q={_q:.2f}"
                    )

            _ft_disabled = not _ft_topic.strip()
            if st.button(
                f"🔀 Generate {_ft_n} fresh idea{'s' if _ft_n != 1 else ''}",
                type="primary", use_container_width=True,
                key="regen_fresh_go_btn",
                disabled=_ft_disabled,
            ):
                with st.spinner(
                    f"Generating {_ft_n} fresh idea(s) on '{_ft_topic[:40]}'…"
                ):
                    try:
                        _fresh_new = _regen_fresh(
                            _ft_topic.strip(),
                            ideas,
                            n=int(_ft_n),
                        )
                    except ValueError as _e:
                        st.error(str(_e))
                        _fresh_new = []
                st.session_state["_regen_fresh_results"] = [
                    i.to_dict() for i in _fresh_new
                ]
                st.session_state["_regen_fresh_topic_used"] = _ft_topic.strip()

            _fresh_results = st.session_state.get("_regen_fresh_results", [])
            if (_fresh_results and
                    st.session_state.get("_regen_fresh_topic_used") ==
                    _ft_topic.strip()):
                st.markdown("---")
                st.markdown(
                    f"### {len(_fresh_results)} fresh idea(s) on the same topic"
                )
                import html as _html_mod
                for _i, _d in enumerate(_fresh_results, 1):
                    _meta = _d.get("execution_meta") or {}
                    _diverge = _meta.get("divergence_note", "")
                    _ttl = _d.get("title", "Untitled")
                    with st.expander(
                        f"🔀 #{_i}. {_ttl}",
                        expanded=(_i == 1),
                    ):
                        st.code(_ttl, language=None)
                        if _diverge:
                            st.markdown(
                                f"<div style='background:#dbeafe;"
                                f"border-left:4px solid #0ea5e9;"
                                f"padding:8px 12px;border-radius:6px;"
                                f"font-size:13px;color:#1e3a8a;"
                                f"margin-bottom:10px'>"
                                f"<b>What's different:</b> "
                                f"{_html_mod.escape(_diverge)}</div>",
                                unsafe_allow_html=True,
                            )
                        st.markdown(f"**Motivation.** {_d.get('motivation','')}")
                        st.markdown(f"**Method.** {_d.get('method','')}")
                        st.markdown(f"**Hypothesis.** {_d.get('hypothesis','')}")
                        if _d.get("expected_outcome"):
                            st.markdown(
                                f"**Expected outcome.** {_d['expected_outcome']}"
                            )
                        if _d.get("resources"):
                            st.markdown(f"**Resources.** {_d['resources']}")
                        _c1, _c2, _c3 = st.columns(3)
                        _c1.metric("methodology",
                                    (_d.get("methodology_type") or "?")
                                    .replace("_", " "))
                        _c2.metric("novelty",
                                    _d.get("novelty_level") or "?")
                        _c3.metric("strategy", "F (Fresh)")

                # ── Save options: add-to-session / DB / JSON / Markdown ──
                _save_c1, _save_c2 = st.columns([1, 1])
                with _save_c1:
                    if st.button("➕ Add to current session",
                                  use_container_width=True,
                                  key="regen_fresh_add_btn",
                                  help="Inject these ideas into the active "
                                       "session so the rest of the app "
                                       "(Compare / Simulate / Reviewer Lens / "
                                       "Exec Loop / Provenance) sees them."):
                        ideas.extend(_fresh_results)
                        st.session_state["_regen_fresh_results"] = []
                        st.success(
                            f"Added {len(_fresh_results)} fresh idea(s) to "
                            "your current results."
                        )
                        st.rerun()
                with _save_c2:
                    _uid_save = st.session_state.get("user_id")
                    _save_disabled = not _uid_save
                    if st.button(
                        "💾 Save as new run",
                        use_container_width=True,
                        key="regen_fresh_save_db_btn",
                        disabled=_save_disabled,
                        help=("Persist these ideas as a new entry in your "
                              "History tab — survives page reload + browser "
                              "close.") if not _save_disabled else
                              "Log in to save runs.",
                    ):
                        try:
                            _save_topic = (
                                f"[fresh take] {_ft_topic.strip()}"
                            )
                            _save_payload = {
                                "topic": _save_topic,
                                "ideas": _fresh_results,
                                "coverage": 0.0,
                                "mode": "regenerate_fresh",
                                "source_run_topic":
                                    results.get("topic", ""),
                                "stats": {
                                    "elapsed_seconds": 0,
                                    "estimated_cost_usd": 0,
                                },
                            }
                            _new_id = db.save_result(
                                user_id=_uid_save,
                                topic=_save_topic,
                                coverage=0.0,
                                ideas_count=len(_fresh_results),
                                results_dict=_save_payload,
                            )
                            try:
                                db_cache.invalidate_user_results()
                            except Exception:
                                pass
                            st.success(
                                f"✅ Saved as run #{_new_id} — visible in your "
                                "**History** tab."
                            )
                        except Exception as _e:
                            st.error(f"Save failed: {_e}")

                _exp_c1, _exp_c2 = st.columns(2)
                with _exp_c1:
                    import json as _json_mod
                    _fresh_json = _json_mod.dumps(
                        {"topic": _ft_topic, "mode": "fresh_take",
                          "ideas": _fresh_results},
                        ensure_ascii=False, indent=2,
                    )
                    st.download_button(
                        "📥 Download JSON",
                        data=_fresh_json,
                        file_name=f"regenerated_fresh_{int(time.time())}.json",
                        mime="application/json",
                        use_container_width=True,
                        key="regen_fresh_dl_json",
                    )
                with _exp_c2:
                    _md_lines = [f"# Fresh-take ideas: {_ft_topic}", ""]
                    for _i_md, _idea_md in enumerate(_fresh_results, 1):
                        _md_lines.append(
                            f"## {_i_md}. {_idea_md.get('title','Untitled')}"
                        )
                        _meta_md = _idea_md.get("execution_meta") or {}
                        if _meta_md.get("divergence_note"):
                            _md_lines.append(
                                f"> **What's different:** "
                                f"{_meta_md['divergence_note']}"
                            )
                        for _fld in ("motivation", "method", "hypothesis",
                                       "expected_outcome", "resources",
                                       "risk_assessment"):
                            _val = _idea_md.get(_fld, "").strip()
                            if _val:
                                _md_lines.append(
                                    f"**{_fld.replace('_',' ').title()}.** "
                                    f"{_val}"
                                )
                        _md_lines.append("")
                    st.download_button(
                        "📄 Download Markdown",
                        data="\n".join(_md_lines),
                        file_name=f"regenerated_fresh_{int(time.time())}.md",
                        mime="text/markdown",
                        use_container_width=True,
                        key="regen_fresh_dl_md",
                    )

        # Existing parent-driven UI runs only in from_parent mode
        elif ideas and _regen_op_mode == "from_parent" and _regen is not None:
            if True:
                _ideas_sorted = sorted(
                    ideas, key=lambda x: x.get("quality_score", 0), reverse=True,
                )
                _idea_labels = [
                    f"{i.get('title','Untitled')} (q={i.get('quality_score',0):.2f}, "
                    f"gen={i.get('generation',0)})"
                    for i in _ideas_sorted
                ]
                _parent_idx = st.selectbox(
                    "Parent idea",
                    options=range(len(_idea_labels)),
                    format_func=lambda i: _idea_labels[i],
                    key="regen_parent_idx",
                )
                _parent = _ideas_sorted[_parent_idx]

                _r1, _r2 = st.columns([2, 1])
                with _r1:
                    _mode_keys = list(REGEN_MODES.keys())
                    _mode = st.selectbox(
                        "Mode",
                        options=_mode_keys,
                        format_func=lambda k: (
                            f"{REGEN_MODES[k]['label']}  —  "
                            f"{REGEN_MODES[k]['tagline']}"
                        ),
                        key="regen_mode",
                    )
                with _r2:
                    _n_variants = st.number_input(
                        "Variants", min_value=1, max_value=5, value=2, step=1,
                        key="regen_n",
                    )

                _cfg = REGEN_MODES[_mode]
                st.markdown(
                    f"<div style='background:#f0f9ff;border-left:4px solid #0ea5e9;"
                    f"padding:10px 14px;margin:8px 0;border-radius:6px;'>"
                    f"<b>{_cfg['label']}</b> — {_cfg['description']}</div>",
                    unsafe_allow_html=True,
                )

                # ── Target topic input (required for transplant; optional for others) ──
                # Lets the user redirect any regeneration to a different domain.
                _requires_topic = bool(_cfg.get("requires_target_topic"))
                _topic_default = (results.get("topic", "")
                                    if _requires_topic else "")
                _topic_label = (
                    "🎯 Target topic (required)" if _requires_topic
                    else "🎯 Target topic — override the parent's domain (optional)"
                )
                _topic_help = (
                    "The new domain to transplant the idea into. The "
                    "regenerated ideas will be specifically about this topic, "
                    "not the parent's."
                    if _requires_topic
                    else "Leave blank to keep the parent's domain. Set this "
                            "to e.g. 'protein folding' to apply the chosen "
                            "regeneration mode to a different topic."
                )
                _target_topic = st.text_input(
                    _topic_label,
                    value=st.session_state.get("_regen_target_topic",
                                                  _topic_default),
                    placeholder=("e.g., materials discovery, protein folding, "
                                  "robotics control"),
                    help=_topic_help,
                    key="_regen_target_topic_input",
                    autocomplete="off",
                )
                st.session_state["_regen_target_topic"] = _target_topic
                if _requires_topic and not _target_topic.strip():
                    st.warning(
                        "✋ Topic Transplant requires a target topic. Enter "
                        "the new domain above to enable Generate."
                    )

                # Show parent summary card (full title; HTML-escaped)
                import html as _html_mod
                _q = _parent.get("quality_score", 0.0)
                _meth = _parent.get("methodology_type", "?") or "?"
                _nov = _parent.get("novelty_level", "?") or "?"
                _parent_title_safe = _html_mod.escape(_parent.get("title", "Untitled"))
                st.markdown(
                    f"<div style='background:#fafafa;border:1px solid #e2e8f0;"
                    f"border-radius:8px;padding:12px 16px;margin:8px 0;'>"
                    f"<div style='font-size:11px;color:#64748b;text-transform:uppercase;"
                    f"letter-spacing:0.06em;font-weight:700'>Parent</div>"
                    f"<div style='font-size:15px;font-weight:700;color:#0c4a6e;"
                    f"margin-top:2px;word-break:break-word'>{_parent_title_safe}</div>"
                    f"<div style='font-size:12px;color:#64748b;margin-top:4px'>"
                    f"q={_q:.2f} · {_meth.replace('_',' ')} × {_nov} · "
                    f"gen={_parent.get('generation',0)}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                _btn_disabled = _requires_topic and not _target_topic.strip()
                _btn_label = (
                    f"🔄 Generate {_n_variants} variant"
                    f"{'s' if _n_variants != 1 else ''}"
                    + (f" → {_target_topic.strip()[:40]}"
                        if _target_topic.strip() else "")
                )
                _go_btn = st.button(
                    _btn_label,
                    type="primary", use_container_width=True,
                    key="regen_go_btn",
                    disabled=_btn_disabled,
                )
                if _go_btn:
                    with st.spinner(f"Generating {_n_variants} {_cfg['label']} variant(s)…"):
                        # Collect per-call diagnostics so the user sees
                        # WHY zero variants came back instead of just
                        # "No variants returned".
                        _regen_diag: list = []
                        try:
                            _new_ideas = _regen(
                                _parent, _mode, n=int(_n_variants),
                                weak_probes=_parent.get("probe_scores"),
                                target_topic=_target_topic.strip(),
                                diagnostics=_regen_diag,
                            )
                        except ValueError as _e:
                            st.error(str(_e))
                            _new_ideas = []
                    st.session_state["_regen_results"] = [
                        i.to_dict() for i in _new_ideas
                    ]
                    st.session_state["_regen_parent_title"] = (
                        _parent.get("title", "")
                    )
                    st.session_state["_regen_run_topic"] = _target_topic.strip()
                    st.session_state["_regen_diagnostics"] = list(_regen_diag)

                _new = st.session_state.get("_regen_results", [])
                if _new and st.session_state.get("_regen_parent_title") == \
                        _parent.get("title", ""):
                    st.markdown("---")
                    st.markdown(f"### {len(_new)} new idea(s) derived from this parent")
                    for _i, _d in enumerate(_new, 1):
                        _meta = _d.get("execution_meta") or {}
                        _ln = _meta.get("lineage_note", "")
                        _badge = REGEN_MODES.get(_meta.get("regen_mode", ""),
                                                   {}).get("label", "🔄")
                        with st.expander(
                            f"{_badge} #{_i}. {_d.get('title','Untitled')}",
                            expanded=(_i == 1),
                        ):
                            st.code(_d.get("title", "Untitled"), language=None)
                            _c1, _c2 = st.columns([3, 1])
                            with _c1:
                                if _ln:
                                    st.markdown(
                                        f"<div style='background:#fef9c3;"
                                        f"border-left:4px solid #facc15;"
                                        f"padding:8px 12px;border-radius:6px;"
                                        f"font-size:13px;color:#713f12;"
                                        f"margin-bottom:10px'>"
                                        f"<b>Lineage:</b> {_ln}</div>",
                                        unsafe_allow_html=True,
                                    )
                                st.markdown(f"**Motivation.** {_d.get('motivation','')}")
                                st.markdown(f"**Method.** {_d.get('method','')}")
                                st.markdown(f"**Hypothesis.** {_d.get('hypothesis','')}")
                                if _d.get("expected_outcome"):
                                    st.markdown(
                                        f"**Expected outcome.** {_d['expected_outcome']}"
                                    )
                                if _d.get("resources"):
                                    st.markdown(
                                        f"**Resources.** {_d['resources']}"
                                    )
                                if _d.get("risk_assessment"):
                                    st.markdown(
                                        f"**Risks.** {_d['risk_assessment']}"
                                    )
                            with _c2:
                                st.metric("methodology",
                                          (_d.get("methodology_type") or "?")
                                          .replace("_", " "))
                                st.metric("novelty",
                                          _d.get("novelty_level") or "?")
                                st.metric("generation",
                                          _d.get("generation", 1),
                                          help="Generations from the original "
                                               "(parent generation + 1).")

                    # ── Save options: session / DB / JSON / Markdown ──────
                    _save_pc1, _save_pc2 = st.columns([1, 1])
                    with _save_pc1:
                        if st.button(
                            "➕ Add to current session",
                            use_container_width=True,
                            key="regen_add_btn",
                            help="Inject these ideas into the active session "
                                 "so the rest of the app (Compare / Simulate "
                                 "/ Reviewer Lens / Exec Loop / Provenance) "
                                 "sees them.",
                        ):
                            ideas.extend(_new)
                            st.session_state["_regen_results"] = []
                            st.success(
                                f"Added {len(_new)} new idea(s) to your "
                                "current results."
                            )
                            st.rerun()
                    with _save_pc2:
                        _uid_save_p = st.session_state.get("user_id")
                        _save_disabled_p = not _uid_save_p
                        if st.button(
                            "💾 Save as new run",
                            use_container_width=True,
                            key="regen_save_db_btn",
                            disabled=_save_disabled_p,
                            help=("Persist these ideas as a new entry in your "
                                  "History tab — survives page reload + "
                                  "browser close.")
                                if not _save_disabled_p else
                                "Log in to save runs.",
                        ):
                            try:
                                _src_topic = results.get("topic", "")
                                _save_topic_p = (
                                    f"[regen·{_mode}] "
                                    f"{_parent.get('title','')}"
                                )
                                _save_payload_p = {
                                    "topic": _save_topic_p,
                                    "ideas": _new,
                                    "coverage": 0.0,
                                    "mode": "regenerate_from_parent",
                                    "regen_mode": _mode,
                                    "parent_title": _parent.get("title", ""),
                                    "source_run_topic": _src_topic,
                                    "stats": {
                                        "elapsed_seconds": 0,
                                        "estimated_cost_usd": 0,
                                    },
                                }
                                _new_id_p = db.save_result(
                                    user_id=_uid_save_p,
                                    topic=_save_topic_p,
                                    coverage=0.0,
                                    ideas_count=len(_new),
                                    results_dict=_save_payload_p,
                                )
                                try:
                                    db_cache.invalidate_user_results()
                                except Exception:
                                    pass
                                st.success(
                                    f"✅ Saved as run #{_new_id_p} — visible "
                                    "in your **History** tab."
                                )
                            except Exception as _e:
                                st.error(f"Save failed: {_e}")

                    _exp_pc1, _exp_pc2 = st.columns(2)
                    with _exp_pc1:
                        import json as _json_mod
                        _regen_json = _json_mod.dumps(
                            {"mode": _mode,
                              "parent_title": _parent.get("title", ""),
                              "ideas": _new},
                            ensure_ascii=False, indent=2,
                        )
                        st.download_button(
                            "📥 Download JSON",
                            data=_regen_json,
                            file_name=(f"regen_{_mode}_"
                                        f"{int(time.time())}.json"),
                            mime="application/json",
                            use_container_width=True,
                            key="regen_dl_json",
                        )
                    with _exp_pc2:
                        _md_lines_p = [
                            f"# Regenerated ideas — mode: {_mode}",
                            f"_Parent:_ **{_parent.get('title','')}**",
                            "",
                        ]
                        for _i_mdp, _idea_mdp in enumerate(_new, 1):
                            _md_lines_p.append(
                                f"## {_i_mdp}. "
                                f"{_idea_mdp.get('title','Untitled')}"
                            )
                            _meta_mdp = _idea_mdp.get("execution_meta") or {}
                            if _meta_mdp.get("lineage_note"):
                                _md_lines_p.append(
                                    f"> **Lineage:** "
                                    f"{_meta_mdp['lineage_note']}"
                                )
                            if _meta_mdp.get("target_topic"):
                                _md_lines_p.append(
                                    f"> **Target topic:** "
                                    f"{_meta_mdp['target_topic']}"
                                )
                            for _fld in ("motivation", "method", "hypothesis",
                                          "expected_outcome", "resources",
                                          "risk_assessment"):
                                _val = _idea_mdp.get(_fld, "").strip()
                                if _val:
                                    _md_lines_p.append(
                                        f"**{_fld.replace('_',' ').title()}.** "
                                        f"{_val}"
                                    )
                            _md_lines_p.append("")
                        st.download_button(
                            "📄 Download Markdown",
                            data="\n".join(_md_lines_p),
                            file_name=(f"regen_{_mode}_"
                                        f"{int(time.time())}.md"),
                            mime="text/markdown",
                            use_container_width=True,
                            key="regen_dl_md",
                        )
                elif _go_btn:
                    _diag = st.session_state.get("_regen_diagnostics") or []
                    if _diag:
                        # Surface the actual per-call failure reasons.
                        _bullets = "\n".join(f"- {d}" for d in _diag)
                        st.warning(
                            f"**No variants returned.** "
                            f"{len(_diag)} diagnostic message(s) from the "
                            f"regenerator:\n\n{_bullets}\n\n"
                            f"Common fixes: open **Admin Dashboard → "
                            f"🔌 LLM Provider** and (a) switch to a faster/"
                            f"more reliable provider (DeepSeek and Kimi are "
                            f"typically the most reliable), (b) verify the "
                            f"API key is set, or (c) try a smaller `n` and "
                            f"re-run."
                        )
                    else:
                        st.warning(
                            "No variants returned. The LLM may be "
                            "unavailable or rate-limited. Check the Log "
                            "tab for details."
                        )

    # ── Novelty Lab Tab ────────────────────────────────────────────────────
    # Three modes for pushing genuine novelty (vs. the Regenerate tab's
    # variation-of-existing-ideas focus):
    #   🎯 Adversarial Critic  — attack & revise an idea's originality
    #   ⚔️ Contradiction-driven — TRIZ-style: extract tensions, resolve them
    #   🧪 Multi-LLM Ensemble  — parallel cross-provider + diversity filter
    with tab_novelty:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">'
            '<span style="font-size:22px">🧪</span>'
            '<span style="font-size:18px;font-weight:700;color:#0c4a6e">'
            'Novelty Lab</span></div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Three mechanisms for pushing genuine novelty: an adversarial "
            "critic that attacks originality and a revision agent that "
            "addresses the critique; TRIZ-style contradiction extraction "
            "→ resolution; and parallel multi-LLM ensemble with a pairwise "
            "diversity filter. Pair any of these with the **Exec Loop** "
            "tab afterward to keep feasibility honest."
        )

        # Build the radio options list — corpus_anchored is gated by an
        # admin feature toggle (config.ENABLE_CORPUS_ANCHORED_NOVELTY).
        # If it's disabled and the user previously had it selected, fall
        # back to the first option so the radio doesn't raise.
        _novelty_options = [
            "adversarial", "contradiction", "ensemble",
            "constraint", "future_back", "frontier", "genetic",
            "heretic", "persona", "counterfactual",
            "analogy", "failure_mode", "extremum",
            "inversion", "null_result", "underserved_cohort",
            "composable_primitive", "stakeholder_pareto",
        ]
        try:
            import config as _cfg_nl
            if getattr(_cfg_nl, "ENABLE_CORPUS_ANCHORED_NOVELTY", True):
                _novelty_options.append("corpus_anchored")
        except Exception:
            _novelty_options.append("corpus_anchored")

        # If session state holds a mode that's no longer available
        # (because the admin just turned it off), reset it.
        if (
            "novelty_mode" in st.session_state
            and st.session_state["novelty_mode"] not in _novelty_options
        ):
            st.session_state["novelty_mode"] = _novelty_options[0]

        _novelty_mode = st.radio(
            "Mode",
            options=_novelty_options,
            format_func=lambda k: {
                "adversarial": "🎯 Adversarial Novelty Critic — attack & revise",
                "contradiction": "⚔️ Contradiction-driven — TRIZ-style",
                "ensemble": "🧪 Multi-LLM Ensemble — cross-provider diversity",
                "constraint": "🔒 Constraint Stacking — multi-constraint satisfaction",
                "future_back": "🔮 Future-back — imagine 2040 → back-propagate",
                "frontier": "📐 Embedding-gradient — target the under-explored frontier",
                "genetic": "🧬 Genetic Algorithm — crossover + mutate over a population",
                "heretic": "🪖 Heretic Mode — falsify dominant beliefs in the field",
                "persona": "🎭 Persona-swap — same topic through 8 expert lenses",
                "counterfactual": "🔍 Counterfactual Literature — pick up abandoned threads",
                "analogy": "🌉 Analogy-bridge — transplant a method from a far domain",
                "failure_mode": "🛡️ Failure-mode antibody — design immunity into the method",
                "extremum": "🚀 Extremum — pin one axis to its limit, design for that regime",
                "inversion": "🔄 Inversion / Jeopardy — start from a surprising answer, derive the question",
                "null_result": "🅾️ Null-result lab — design a pre-registered clean negative",
                "underserved_cohort": "🤝 Underserved cohort — research anchored to a specific who",
                "composable_primitive": "🧩 Composable primitive — design the building block, not the paper",
                "stakeholder_pareto": "🤹 Stakeholder Pareto — a measurable win for every party",
                "corpus_anchored": "🛰️ Corpus-anchored — operationalized novelty vs a reference corpus",
            }[k],
            horizontal=False,
            key="novelty_mode",
        )

        # ── Mode A: Adversarial novelty critic ──────────────────────────
        if _novelty_mode == "adversarial":
            try:
                from agents.novelty_critic import (
                    critique_novelty, attack_and_revise,
                )
            except ImportError as _e:
                st.error(f"novelty_critic unavailable: {_e}")
                critique_novelty = None

            if not ideas:
                st.info("Run a pipeline first so there's something to critique.")
            elif critique_novelty is not None:
                _adv_ideas = sorted(
                    ideas, key=lambda x: x.get("quality_score", 0),
                    reverse=True,
                )
                _adv_idx = st.selectbox(
                    "Idea to attack",
                    options=range(len(_adv_ideas)),
                    format_func=lambda i: (
                        f"{_adv_ideas[i].get('title','Untitled')} "
                        f"(q={_adv_ideas[i].get('quality_score',0):.2f}, "
                        f"novelty={_adv_ideas[i].get('novelty_level','?')})"
                    ),
                    key="adv_idx",
                )
                # Default OFF so the first click is a single LLM call
                # (the critique). Users who want the revision can opt in
                # — it ~doubles the wall-clock wait.
                _adv_revise = st.checkbox(
                    "🔁 Also revise the idea after critique",
                    value=False,
                    help="When on, runs a SECOND LLM call to produce a "
                         "revised idea that addresses each critique. "
                         "Adds ~10–15s to the wait.",
                    key="adv_revise",
                )
                st.caption(
                    "⏱️ Estimated wait: **~5–10s** for the critique alone, "
                    "**~15–25s** with revise enabled. Results are cached "
                    "per-idea, so re-clicking the same idea is instant."
                )

                # ── Per-idea cache (avoids re-running the LLM on the
                # same idea + same revise flag).
                _adv_cache = st.session_state.setdefault("_adv_cache", {})
                _target = _adv_ideas[_adv_idx]
                _cache_key = (
                    str(_target.get("title", ""))[:200],
                    str(_target.get("method", ""))[:200],
                    bool(_adv_revise),
                )

                _btn_cols = st.columns([3, 1])
                _go = _btn_cols[0].button(
                    "🎯 Attack originality", type="primary",
                    use_container_width=True, key="adv_go_btn",
                )
                _force = _btn_cols[1].button(
                    "↻ Re-run", use_container_width=True, key="adv_force_btn",
                    help="Bypass the cache and call the LLM again.",
                    disabled=(_cache_key not in _adv_cache),
                )

                if _go or _force:
                    import time as _time
                    # Serve from cache unless the user forced a re-run.
                    if (not _force) and _cache_key in _adv_cache:
                        _cached = _adv_cache[_cache_key]
                        st.session_state["_adv_critique"] = _cached["critique"]
                        st.session_state["_adv_revised"] = _cached["revised"]
                        st.session_state["_adv_target_title"] = (
                            _target.get("title", "")
                        )
                        st.success(
                            f"⚡ Served from cache (originally took "
                            f"{_cached['elapsed_s']:.1f}s)."
                        )
                    else:
                        _t0 = _time.perf_counter()
                        if _adv_revise:
                            try:
                                _stage = st.status(
                                    "Step 1/2: adversarial critic at work… "
                                    "(this is the slow part)",
                                    expanded=False,
                                )
                            except Exception:
                                _stage = None
                            if _stage is not None:
                                with _stage:
                                    _critique = critique_novelty(_target)
                                    _t_crit = _time.perf_counter() - _t0
                                    _stage.update(
                                        label=(
                                            f"Step 2/2: revising idea to "
                                            f"address critique… "
                                            f"(critique took {_t_crit:.1f}s)"
                                        )
                                    )
                                    if (_critique.used_llm
                                            and _critique.critiques):
                                        _, _revised = attack_and_revise(
                                            _target,
                                        )
                                    else:
                                        _revised = None
                                    _stage.update(
                                        label=(
                                            f"Done in "
                                            f"{_time.perf_counter()-_t0:.1f}s"
                                        ),
                                        state="complete",
                                    )
                            else:
                                # st.status not available — fall back.
                                with st.spinner("Adversarial critic + revise…"):
                                    _critique, _revised = attack_and_revise(
                                        _target,
                                    )
                        else:
                            with st.spinner(
                                "Adversarial critic at work… (~5–10s)"
                            ):
                                _critique = critique_novelty(_target)
                                _revised = None

                        _elapsed = _time.perf_counter() - _t0
                        st.session_state["_adv_critique"] = _critique.to_dict()
                        st.session_state["_adv_revised"] = (
                            _revised.to_dict() if _revised else None
                        )
                        st.session_state["_adv_target_title"] = (
                            _target.get("title", "")
                        )
                        # Cache the result for instant re-display.
                        _adv_cache[_cache_key] = {
                            "critique": st.session_state["_adv_critique"],
                            "revised":  st.session_state["_adv_revised"],
                            "elapsed_s": _elapsed,
                        }
                        st.caption(
                            f"⏱️ Took **{_elapsed:.1f}s**. Result is cached "
                            f"— re-clicking this idea is instant."
                        )

                _crit = st.session_state.get("_adv_critique")
                if _crit and st.session_state.get(
                        "_adv_target_title") == _adv_ideas[_adv_idx].get("title", ""):
                    _verdict = _crit.get("overall_verdict", "")
                    _score = _crit.get("originality_score", 0.0)
                    _color = ("#ef4444" if _score < 0.30
                                else "#f59e0b" if _score < 0.55
                                else "#0ea5e9" if _score < 0.75
                                else "#10b981")
                    st.markdown(
                        f"<div style='background:#fef2f2;border:2px solid {_color};"
                        f"border-radius:10px;padding:12px 16px;margin:8px 0'>"
                        f"<div style='font-size:11px;color:{_color};"
                        f"font-weight:700;text-transform:uppercase;"
                        f"letter-spacing:0.06em'>Critic's verdict</div>"
                        f"<div style='font-size:18px;font-weight:800;color:{_color};"
                        f"margin-top:2px'>{_verdict}</div>"
                        f"<div style='font-size:13px;color:#475569;margin-top:4px'>"
                        f"Originality score: <b>{_score:.2f}</b> · "
                        f"Confidence: <b>{_crit.get('confidence',0)*100:.0f}%</b>"
                        f"</div></div>",
                        unsafe_allow_html=True,
                    )
                    if _crit.get("error"):
                        st.caption(f"⚠️ {_crit['error']}")
                    if _crit.get("critiques"):
                        st.markdown("**🎯 Specific critiques**")
                        for _cc in _crit["critiques"]:
                            st.markdown(f"- {_cc}")
                    if _crit.get("similar_prior_work"):
                        st.markdown("**📚 Similar prior work**")
                        for _pp in _crit["similar_prior_work"]:
                            st.markdown(f"- {_pp}")
                    if _crit.get("pivots"):
                        st.markdown("**💡 Suggested pivots toward novelty**")
                        for _pv in _crit["pivots"]:
                            st.markdown(f"- {_pv}")

                    _revised_d = st.session_state.get("_adv_revised")
                    if _revised_d:
                        st.markdown("---")
                        st.markdown(
                            f"### 🔁 Revised idea — incorporates "
                            f"{len(_revised_d.get('execution_meta',{}).get('pivots_applied',[]))} "
                            "pivot(s)"
                        )
                        import html as _html_mod
                        st.markdown(
                            f"<div style='background:#f0fdf4;border:1px solid #10b981;"
                            f"border-radius:8px;padding:12px 16px'>"
                            f"<div style='font-size:15px;font-weight:700;"
                            f"color:#065f46'>"
                            f"{_html_mod.escape(_revised_d.get('title',''))}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(f"**Method.** {_revised_d.get('method','')}")
                        st.markdown(
                            f"**Hypothesis.** {_revised_d.get('hypothesis','')}"
                        )
                        _ms = _revised_d.get("execution_meta", {})
                        if _ms.get("pivots_applied"):
                            st.markdown("**Pivots applied:**")
                            for _pa in _ms["pivots_applied"]:
                                st.markdown(f"- ✅ {_pa}")

                        if st.button("➕ Add revised idea to session",
                                      use_container_width=True,
                                      key="adv_add_btn"):
                            ideas.append(_revised_d)
                            st.session_state["_adv_revised"] = None
                            st.success("Added.")
                            st.rerun()

        # ── Mode B: Contradiction-driven ────────────────────────────────
        elif _novelty_mode == "contradiction":
            try:
                from contradiction_ideation import (
                    extract_contradictions,
                    generate_from_contradiction,
                    generate_from_contradictions_batch,
                )
            except ImportError as _e:
                st.error(f"contradiction_ideation unavailable: {_e}")
                extract_contradictions = None

            if extract_contradictions is not None:
                _con_topic = st.text_input(
                    "Topic",
                    value=results.get("topic", ""),
                    help="Topic to mine for fundamental contradictions.",
                    key="_con_topic",
                    autocomplete="off",
                )
                _con_n = st.slider(
                    "Number of contradictions to extract",
                    min_value=2, max_value=6, value=4, step=1,
                    key="_con_n",
                )

                _cn_c1, _cn_c2 = st.columns(2)
                with _cn_c1:
                    if st.button(
                        "🔍 Extract contradictions only",
                        use_container_width=True,
                        disabled=not _con_topic.strip(),
                        key="con_extract_btn",
                    ):
                        with st.spinner("Mining contradictions…"):
                            _cons = extract_contradictions(
                                _con_topic.strip(),
                                n=int(_con_n),
                            )
                        st.session_state["_con_contradictions"] = [
                            c.to_dict() for c in _cons
                        ]
                        st.session_state["_con_ideas"] = []
                with _cn_c2:
                    if st.button(
                        "⚔️ Extract + generate ideas",
                        type="primary",
                        use_container_width=True,
                        disabled=not _con_topic.strip(),
                        key="con_batch_btn",
                    ):
                        with st.spinner(
                            f"Mining contradictions and generating "
                            f"{_con_n} ideas…"
                        ):
                            _ideas_new = generate_from_contradictions_batch(
                                _con_topic.strip(),
                                n=int(_con_n),
                            )
                        st.session_state["_con_ideas"] = [
                            i.to_dict() for i in _ideas_new
                        ]
                        st.session_state["_con_contradictions"] = [
                            (i.execution_meta or {}).get("contradiction_resolved", {})
                            for i in _ideas_new
                        ]

                _cons_state = st.session_state.get("_con_contradictions", [])
                _ideas_state = st.session_state.get("_con_ideas", [])

                if _cons_state:
                    st.markdown("---")
                    st.markdown(
                        f"### {len(_cons_state)} contradiction(s) identified"
                    )
                    for _ix, _c in enumerate(_cons_state, 1):
                        _sev = _c.get("severity", 0)
                        _bar_color = ("#dc2626" if _sev >= 0.75
                                        else "#f59e0b" if _sev >= 0.50 else "#0ea5e9")
                        st.markdown(
                            f"<div style='background:#fafafa;border:1px solid #e2e8f0;"
                            f"border-left:5px solid {_bar_color};border-radius:8px;"
                            f"padding:10px 14px;margin:6px 0'>"
                            f"<div style='font-size:11px;color:{_bar_color};"
                            f"font-weight:700;text-transform:uppercase'>"
                            f"Contradiction #{_ix} · severity {_sev:.2f}</div>"
                            f"<div style='font-size:14px;font-weight:700;"
                            f"color:#0c4a6e;margin-top:2px'>"
                            f"{_c.get('statement','')}</div>"
                            f"<div style='font-size:12px;color:#64748b;margin-top:4px'>"
                            f"<b>↗</b> {_c.get('forces_a','')}<br>"
                            f"<b>↘</b> {_c.get('forces_b','')}<br>"
                            f"<b>Why it matters:</b> {_c.get('why_it_matters','')}<br>"
                            f"<b>Resolution hint:</b> {_c.get('resolution_hint','')}"
                            f"</div></div>",
                            unsafe_allow_html=True,
                        )

                if _ideas_state:
                    st.markdown("---")
                    st.markdown(
                        f"### {len(_ideas_state)} idea(s) generated from contradictions"
                    )
                    for _i, _d in enumerate(_ideas_state, 1):
                        with st.expander(
                            f"⚔️ #{_i}. {_d.get('title','Untitled')}",
                            expanded=(_i == 1),
                        ):
                            st.code(_d.get("title", "Untitled"), language=None)
                            _meta_c = _d.get("execution_meta") or {}
                            if _meta_c.get("resolution_mechanism"):
                                st.markdown(
                                    f"<div style='background:#dbeafe;"
                                    f"border-left:4px solid #0ea5e9;"
                                    f"padding:8px 12px;border-radius:6px;"
                                    f"font-size:13px;color:#1e3a8a;"
                                    f"margin-bottom:10px'>"
                                    f"<b>Resolution mechanism:</b> "
                                    f"{_meta_c['resolution_mechanism']}"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )
                            st.markdown(f"**Motivation.** {_d.get('motivation','')}")
                            st.markdown(f"**Method.** {_d.get('method','')}")
                            st.markdown(f"**Hypothesis.** {_d.get('hypothesis','')}")

                    if st.button("➕ Add all contradiction-driven ideas to session",
                                  use_container_width=True,
                                  key="con_add_btn"):
                        ideas.extend(_ideas_state)
                        st.session_state["_con_ideas"] = []
                        st.success(f"Added {len(_ideas_state)} idea(s).")
                        st.rerun()

        # ── Mode C: Multi-LLM ensemble ──────────────────────────────────
        elif _novelty_mode == "ensemble":
            try:
                from multi_llm_ensemble import (
                    ensemble_generate, available_providers,
                )
            except ImportError as _e:
                st.error(f"multi_llm_ensemble unavailable: {_e}")
                ensemble_generate = None

            if ensemble_generate is not None:
                _all_provs = available_providers()
                if not _all_provs:
                    st.warning(
                        "No LLM providers are configured — add API keys to "
                        ".env to use the ensemble."
                    )
                else:
                    st.caption(
                        f"**Configured providers:** {', '.join(_all_provs)}. "
                        "Each provider generates ideas in parallel; ideas with "
                        "pairwise Jaccard similarity above the threshold are "
                        "dropped (the kept one is reported)."
                    )
                    _en_topic = st.text_input(
                        "Topic",
                        value=results.get("topic", ""),
                        key="_en_topic",
                        autocomplete="off",
                    )
                    _en_c1, _en_c2 = st.columns(2)
                    with _en_c1:
                        _en_picked = st.multiselect(
                            "Providers to use",
                            options=_all_provs,
                            default=_all_provs,
                            key="_en_picked",
                        )
                        _en_n = st.slider(
                            "Ideas per provider", min_value=1, max_value=3,
                            value=1, step=1, key="_en_n",
                        )
                    with _en_c2:
                        _en_thresh = st.slider(
                            "Similarity threshold",
                            min_value=0.20, max_value=0.90, value=0.55,
                            step=0.05, key="_en_thresh",
                            help="Pairs above this Jaccard similarity are "
                                 "considered duplicates and the later one "
                                 "is dropped. Lower = more aggressive filtering.",
                        )
                        _en_hint = st.text_input(
                            "Optional framing hint",
                            placeholder="e.g., focus on clinical applications",
                            key="_en_hint",
                            autocomplete="off",
                        )

                    if st.button(
                        f"🧪 Generate via {len(_en_picked) or 0} provider(s)",
                        type="primary", use_container_width=True,
                        disabled=(not _en_topic.strip() or not _en_picked),
                        key="en_go_btn",
                    ):
                        with st.spinner(
                            f"Calling {len(_en_picked)} provider(s) in parallel "
                            f"({_en_n} idea(s) each)…"
                        ):
                            _en_result = ensemble_generate(
                                _en_topic.strip(),
                                providers=list(_en_picked),
                                n_per_provider=int(_en_n),
                                similarity_threshold=float(_en_thresh),
                                hint=_en_hint.strip(),
                            )
                        st.session_state["_en_result"] = {
                            "topic": _en_result.topic,
                            "providers_used": _en_result.providers_used,
                            "kept_ideas": [i.to_dict() for i in _en_result.kept_ideas],
                            "all_ideas": [i.to_dict() for i in _en_result.all_ideas],
                            "rejected_pairs": _en_result.rejected_pairs,
                            "provider_stats": _en_result.provider_stats,
                            "elapsed_s": _en_result.elapsed_s,
                        }

                    _en_state = st.session_state.get("_en_result")
                    if _en_state and _en_state.get("topic") == _en_topic.strip():
                        st.markdown("---")
                        _kept = _en_state.get("kept_ideas") or []
                        _all = _en_state.get("all_ideas") or []
                        _rej = _en_state.get("rejected_pairs") or []
                        _stats = _en_state.get("provider_stats") or {}

                        _m1, _m2, _m3, _m4 = st.columns(4)
                        _m1.metric("Kept", len(_kept))
                        _m2.metric("Generated", len(_all))
                        _m3.metric("Filtered out", len(_rej))
                        _m4.metric("Time", f"{_en_state.get('elapsed_s',0):.1f}s")

                        with st.expander("📊 Per-provider stats", expanded=False):
                            for _prov, _ps in _stats.items():
                                st.markdown(
                                    f"- **{_prov}**: ok={_ps.get('ok',0)} · "
                                    f"fail={_ps.get('fail',0)} · "
                                    f"parsed={_ps.get('parsed',0)}"
                                )
                        if _rej:
                            with st.expander(
                                f"🚫 Filtered duplicates ({len(_rej)})",
                                expanded=False,
                            ):
                                for _r in _rej:
                                    st.markdown(
                                        f"- *{_r['rejected_title']}* "
                                        f"({_r['rejected_provider']}) "
                                        f"≈ *{_r['kept_title']}* "
                                        f"(sim={_r['similarity']})"
                                    )

                        if _kept:
                            st.markdown(f"### {len(_kept)} kept idea(s)")
                            for _i, _d in enumerate(_kept, 1):
                                _meta_e = _d.get("execution_meta") or {}
                                _provtag = _meta_e.get("ensemble_provider", "?")
                                with st.expander(
                                    f"🧪 #{_i}. [{_provtag}] {_d.get('title','Untitled')}",
                                    expanded=(_i == 1),
                                ):
                                    st.code(_d.get("title", "Untitled"),
                                            language=None)
                                    st.markdown(
                                        f"**Provider model.** "
                                        f"`{_meta_e.get('ensemble_model','?')}`"
                                    )
                                    st.markdown(
                                        f"**Motivation.** {_d.get('motivation','')}"
                                    )
                                    st.markdown(
                                        f"**Method.** {_d.get('method','')}"
                                    )
                                    st.markdown(
                                        f"**Hypothesis.** {_d.get('hypothesis','')}"
                                    )

                            if st.button(
                                "➕ Add kept ideas to session",
                                use_container_width=True,
                                key="en_add_btn",
                            ):
                                ideas.extend(_kept)
                                st.session_state["_en_result"] = None
                                st.success(f"Added {len(_kept)} idea(s).")
                                st.rerun()
                        else:
                            st.warning(
                                "No ideas survived. Check provider stats — "
                                "some calls may have failed (network / rate "
                                "limit / model unavailable)."
                            )

        # ── Mode D: Constraint stacking ─────────────────────────────────
        elif _novelty_mode == "constraint":
            try:
                from constraint_stacking import (
                    generate_with_constraints, suggest_constraints,
                    CONSTRAINT_LIBRARY,
                )
            except ImportError as _e:
                st.error(f"constraint_stacking unavailable: {_e}")
                generate_with_constraints = None

            if generate_with_constraints is not None:
                _con_topic_cs = st.text_input(
                    "Topic",
                    value=results.get("topic", ""),
                    key="_cs_topic",
                    autocomplete="off",
                )
                st.caption(
                    "Pick constraints from the library + add your own. "
                    "The LLM must satisfy ALL of them simultaneously."
                )
                _cs_picks: list = []
                for _cat, _lst in CONSTRAINT_LIBRARY.items():
                    _picked = st.multiselect(
                        f"{_cat.title()} constraints",
                        options=_lst,
                        key=f"_cs_picks_{_cat}",
                    )
                    _cs_picks.extend(_picked)
                _custom = st.text_area(
                    "Custom constraints (one per line)",
                    placeholder="e.g., Must be runnable inside a Jupyter notebook",
                    key="_cs_custom", height=80,
                )
                _custom_lines = [
                    line.strip() for line in (_custom or "").splitlines()
                    if line.strip()
                ]
                _all_constraints = list(_cs_picks) + _custom_lines

                if _all_constraints:
                    st.markdown(
                        f"**{len(_all_constraints)} active constraint(s)** "
                        "— the idea must satisfy all of them:"
                    )
                    for _c in _all_constraints:
                        st.markdown(f"- 🔒 {_c}")

                _cs_btn1, _cs_btn2 = st.columns([1, 1])
                with _cs_btn1:
                    if st.button(
                        "💡 Suggest constraints for me",
                        use_container_width=True,
                        disabled=not _con_topic_cs.strip(),
                        key="cs_suggest_btn",
                    ):
                        with st.spinner("Asking LLM for constraints…"):
                            _sugg = suggest_constraints(
                                _con_topic_cs.strip(), n=3,
                            )
                        st.session_state["_cs_suggested"] = _sugg
                with _cs_btn2:
                    if st.button(
                        f"🔒 Generate under {len(_all_constraints)} constraint(s)",
                        type="primary", use_container_width=True,
                        disabled=(not _con_topic_cs.strip()
                                    or not _all_constraints),
                        key="cs_go_btn",
                    ):
                        with st.spinner("Generating constrained idea…"):
                            _cs_idea = generate_with_constraints(
                                _con_topic_cs.strip(), _all_constraints,
                            )
                        st.session_state["_cs_idea"] = (
                            _cs_idea.to_dict() if _cs_idea else None
                        )

                _sugg = st.session_state.get("_cs_suggested") or []
                if _sugg:
                    st.markdown("**Suggested constraints** (copy into custom):")
                    for _s in _sugg:
                        st.markdown(f"- 💡 {_s}")

                _cs_idea_state = st.session_state.get("_cs_idea")
                if _cs_idea_state:
                    st.markdown("---")
                    _d = _cs_idea_state
                    st.markdown(
                        f"### 🔒 {_d.get('title','Untitled')}"
                    )
                    _meta = _d.get("execution_meta") or {}
                    if _meta.get("feasibility_note"):
                        st.markdown(
                            f"<div style='background:#fef3c7;"
                            f"border-left:4px solid #f59e0b;padding:8px 12px;"
                            f"border-radius:6px;font-size:13px;color:#7c2d12'>"
                            f"<b>Feasibility note:</b> {_meta['feasibility_note']}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    st.markdown(f"**Motivation.** {_d.get('motivation','')}")
                    st.markdown(f"**Method.** {_d.get('method','')}")
                    st.markdown(f"**Hypothesis.** {_d.get('hypothesis','')}")
                    if _meta.get("constraints_satisfied"):
                        st.markdown("**How each constraint is satisfied:**")
                        for _cs in _meta["constraints_satisfied"]:
                            st.markdown(f"- ✅ {_cs}")
                    if st.button("➕ Add to session",
                                  use_container_width=True,
                                  key="cs_add_btn"):
                        ideas.append(_cs_idea_state)
                        st.session_state["_cs_idea"] = None
                        st.success("Added.")
                        st.rerun()

        # ── Mode E: Future-back ─────────────────────────────────────────
        elif _novelty_mode == "future_back":
            try:
                from future_back_ideation import (
                    imagine_future, back_propagate, future_back_batch,
                    SCENARIOS,
                )
            except ImportError as _e:
                st.error(f"future_back_ideation unavailable: {_e}")
                imagine_future = None

            if imagine_future is not None:
                _fb_topic = st.text_input(
                    "Topic",
                    value=results.get("topic", ""),
                    key="_fb_topic",
                    autocomplete="off",
                )
                _fb_c1, _fb_c2 = st.columns(2)
                with _fb_c1:
                    _fb_year = st.slider(
                        "Year to imagine",
                        min_value=2030, max_value=2060,
                        value=2040, step=5, key="_fb_year",
                    )
                    _fb_scenarios = st.multiselect(
                        "Scenarios",
                        options=SCENARIOS,
                        default=["best_case", "surprising"],
                        format_func=lambda s: s.replace("_", " ").title(),
                        key="_fb_scenarios",
                    )
                with _fb_c2:
                    _fb_n = st.slider(
                        "Visions / ideas to generate",
                        min_value=1, max_value=4, value=2, step=1,
                        key="_fb_n",
                    )

                if st.button(
                    f"🔮 Imagine {_fb_year} → generate {_fb_n} idea(s)",
                    type="primary", use_container_width=True,
                    disabled=(not _fb_topic.strip() or not _fb_scenarios),
                    key="fb_go_btn",
                ):
                    with st.spinner(
                        f"Imagining {_fb_year} and back-propagating…"
                    ):
                        _fb_ideas = future_back_batch(
                            _fb_topic.strip(),
                            n=int(_fb_n),
                            scenarios=list(_fb_scenarios)[:int(_fb_n)],
                            year=int(_fb_year),
                        )
                    st.session_state["_fb_ideas"] = [
                        i.to_dict() for i in _fb_ideas
                    ]

                _fb_state = st.session_state.get("_fb_ideas") or []
                if _fb_state:
                    st.markdown("---")
                    st.markdown(f"### {len(_fb_state)} future-back idea(s)")
                    for _i, _d in enumerate(_fb_state, 1):
                        _meta = _d.get("execution_meta") or {}
                        _vis = _meta.get("vision", {})
                        _sc = _vis.get("scenario", "?")
                        with st.expander(
                            f"🔮 #{_i}. [{_sc}] {_d.get('title','Untitled')}",
                            expanded=(_i == 1),
                        ):
                            st.code(_d.get("title", "Untitled"), language=None)
                            if _vis.get("description"):
                                st.markdown(
                                    f"<div style='background:#ede9fe;"
                                    f"border-left:4px solid #a855f7;"
                                    f"padding:10px 14px;border-radius:6px;"
                                    f"font-size:13px;color:#3b0764;"
                                    f"margin-bottom:10px'>"
                                    f"<b>Vision ({_vis.get('year','?')}, "
                                    f"{_sc}):</b><br>"
                                    f"{_vis['description'][:600]}"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )
                            if _meta.get("future_link"):
                                st.markdown(
                                    f"**Future link:** _{_meta['future_link']}_"
                                )
                            st.markdown(f"**Motivation.** {_d.get('motivation','')}")
                            st.markdown(f"**Method.** {_d.get('method','')}")
                            st.markdown(f"**Hypothesis.** {_d.get('hypothesis','')}")

                    if st.button("➕ Add all to session",
                                  use_container_width=True,
                                  key="fb_add_btn"):
                        ideas.extend(_fb_state)
                        st.session_state["_fb_ideas"] = []
                        st.success(f"Added {len(_fb_state)} idea(s).")
                        st.rerun()

        # ── Mode F: Embedding-gradient frontier ─────────────────────────
        elif _novelty_mode == "frontier":
            try:
                from embedding_exploration import (
                    compute_frontier_concepts, generate_at_frontier,
                )
            except ImportError as _e:
                st.error(f"embedding_exploration unavailable: {_e}")
                compute_frontier_concepts = None

            if not ideas:
                st.info(
                    "Run a pipeline first — frontier analysis needs an "
                    "archive of existing ideas to find what's under-explored."
                )
            elif compute_frontier_concepts is not None:
                _fr_topic = st.text_input(
                    "Topic",
                    value=results.get("topic", ""),
                    key="_fr_topic",
                    autocomplete="off",
                )
                _fr_n_gen = st.slider(
                    "Frontier ideas to generate",
                    min_value=1, max_value=4, value=2, step=1, key="_fr_n",
                )

                _fr_b1, _fr_b2 = st.columns([1, 1])
                with _fr_b1:
                    if st.button(
                        "🔍 Analyze frontier only",
                        use_container_width=True,
                        disabled=not _fr_topic.strip(),
                        key="fr_analyze_btn",
                    ):
                        with st.spinner("Mining the archive for blind spots…"):
                            _fr_analysis = compute_frontier_concepts(
                                ideas, _fr_topic.strip(),
                            )
                        st.session_state["_fr_analysis"] = _fr_analysis.to_dict()
                        st.session_state["_fr_ideas"] = []
                with _fr_b2:
                    if st.button(
                        f"📐 Analyze + generate {_fr_n_gen} idea(s)",
                        type="primary", use_container_width=True,
                        disabled=not _fr_topic.strip(),
                        key="fr_go_btn",
                    ):
                        with st.spinner(
                            "Mining frontier + generating ideas…"
                        ):
                            _fr_analysis = compute_frontier_concepts(
                                ideas, _fr_topic.strip(),
                            )
                            _fr_out: list = []
                            for _k in range(int(_fr_n_gen)):
                                _i_new = generate_at_frontier(
                                    _fr_topic.strip(), _fr_analysis,
                                    temperature=min(0.95, 0.75 + 0.05 * _k),
                                )
                                if _i_new is not None:
                                    _fr_out.append(_i_new)
                        st.session_state["_fr_analysis"] = _fr_analysis.to_dict()
                        st.session_state["_fr_ideas"] = [
                            i.to_dict() for i in _fr_out
                        ]

                _fr_a = st.session_state.get("_fr_analysis")
                if _fr_a:
                    st.markdown("---")
                    st.markdown(
                        f"### Frontier analysis ({_fr_a.get('n_archive_ideas',0)} archived ideas)"
                    )
                    _covered = _fr_a.get("covered_concepts") or []
                    if _covered:
                        st.markdown(
                            "**Over-represented in archive (top 10):** "
                            + " · ".join(
                                f"`{t}` ({c})" for t, c in _covered[:10]
                            )
                        )
                    _under = _fr_a.get("underexplored_concepts") or []
                    if _under:
                        st.markdown("**🆕 Under-explored concepts** "
                                      "(LLM-identified blind spots):")
                        for _u in _under:
                            st.markdown(f"- {_u}")
                    if _fr_a.get("frontier_description"):
                        st.markdown(
                            f"<div style='background:#dbeafe;"
                            f"border-left:4px solid #0ea5e9;padding:10px 14px;"
                            f"border-radius:6px;font-size:13px;color:#1e3a8a'>"
                            f"<b>Frontier:</b> "
                            f"{_fr_a['frontier_description']}</div>",
                            unsafe_allow_html=True,
                        )
                    if _fr_a.get("frontier_seeds"):
                        st.markdown("**Frontier seeds (what-if):**")
                        for _s in _fr_a["frontier_seeds"]:
                            st.markdown(f"- ❓ {_s}")

                _fr_ideas_state = st.session_state.get("_fr_ideas") or []
                if _fr_ideas_state:
                    st.markdown("---")
                    st.markdown(
                        f"### {len(_fr_ideas_state)} idea(s) at the frontier"
                    )
                    for _i, _d in enumerate(_fr_ideas_state, 1):
                        _meta = _d.get("execution_meta") or {}
                        with st.expander(
                            f"📐 #{_i}. {_d.get('title','Untitled')}",
                            expanded=(_i == 1),
                        ):
                            st.code(_d.get("title", "Untitled"), language=None)
                            if _meta.get("frontier_concepts_used"):
                                st.markdown(
                                    "**Frontier concepts targeted:** "
                                    + " · ".join(
                                        f"`{c}`"
                                        for c in _meta["frontier_concepts_used"]
                                    )
                                )
                            st.markdown(f"**Motivation.** {_d.get('motivation','')}")
                            st.markdown(f"**Method.** {_d.get('method','')}")
                            st.markdown(f"**Hypothesis.** {_d.get('hypothesis','')}")

                    if st.button("➕ Add all to session",
                                  use_container_width=True,
                                  key="fr_add_btn"):
                        ideas.extend(_fr_ideas_state)
                        st.session_state["_fr_ideas"] = []
                        st.success(f"Added {len(_fr_ideas_state)} idea(s).")
                        st.rerun()

        # ── Mode G: Genetic algorithm ───────────────────────────────────
        elif _novelty_mode == "genetic":
            try:
                from genetic_ideation import evolve, EvolutionResult
            except ImportError as _e:
                st.error(f"genetic_ideation unavailable: {_e}")
                evolve = None

            if not ideas:
                st.info(
                    "Run a pipeline first — the genetic algorithm needs "
                    "an initial population of ideas to evolve."
                )
            elif evolve is not None:
                _ga_c1, _ga_c2, _ga_c3 = st.columns(3)
                with _ga_c1:
                    _ga_n_gen = st.slider(
                        "Generations", min_value=1, max_value=5, value=2,
                        step=1, key="_ga_n_gen",
                        help="More generations = more diverse offspring, "
                             "more LLM calls.",
                    )
                    _ga_seed = st.number_input(
                        "Random seed", min_value=0, max_value=9999,
                        value=42, step=1, key="_ga_seed",
                    )
                with _ga_c2:
                    _ga_crossover = st.slider(
                        "Crossover rate", min_value=0.0, max_value=1.0,
                        value=0.6, step=0.05, key="_ga_crossover",
                    )
                    _ga_elite = st.slider(
                        "Elite to keep", min_value=1, max_value=4,
                        value=2, step=1, key="_ga_elite",
                    )
                with _ga_c3:
                    _ga_mutation = st.slider(
                        "Mutation rate", min_value=0.0, max_value=1.0,
                        value=0.3, step=0.05, key="_ga_mutation",
                    )
                    _ga_pool_size = st.slider(
                        "Initial pool (top-N ideas)",
                        min_value=4, max_value=min(20, max(4, len(ideas))),
                        value=min(8, len(ideas)), step=1, key="_ga_pool",
                    )

                _est = int(_ga_n_gen) * int(_ga_pool_size)
                st.caption(
                    f"⚠️ Estimated LLM calls per generation: ~{_ga_pool_size - _ga_elite}. "
                    f"Total over {_ga_n_gen} generations: ~{_est}. "
                    f"Cost scales with this; start small."
                )

                if st.button(
                    f"🧬 Evolve {_ga_pool_size} ideas × {_ga_n_gen} generations",
                    type="primary", use_container_width=True,
                    key="ga_go_btn",
                ):
                    _ga_pool = sorted(
                        ideas, key=lambda x: x.get("quality_score", 0),
                        reverse=True,
                    )[: int(_ga_pool_size)]
                    with st.spinner(
                        f"Running GA: {_ga_n_gen} generations on "
                        f"{len(_ga_pool)} ideas…"
                    ):
                        _ga_result = evolve(
                            _ga_pool,
                            n_generations=int(_ga_n_gen),
                            crossover_rate=float(_ga_crossover),
                            mutation_rate=float(_ga_mutation),
                            elite_keep=int(_ga_elite),
                            seed=int(_ga_seed),
                        )
                    st.session_state["_ga_result"] = {
                        "n_generations": _ga_result.n_generations,
                        "final_population": [
                            i.to_dict() for i in _ga_result.final_population
                        ],
                        "fitness_history": _ga_result.fitness_history,
                        "crossover_count": _ga_result.crossover_count,
                        "mutation_count": _ga_result.mutation_count,
                        "elapsed_s": _ga_result.elapsed_s,
                        "initial_size": _ga_result.initial_size,
                    }

                _ga_state = st.session_state.get("_ga_result")
                if _ga_state:
                    st.markdown("---")
                    _h = _ga_state.get("fitness_history") or []
                    _m1, _m2, _m3, _m4 = st.columns(4)
                    _m1.metric("Generations", _ga_state.get("n_generations", 0))
                    _m2.metric("Crossovers", _ga_state.get("crossover_count", 0))
                    _m3.metric("Mutations", _ga_state.get("mutation_count", 0))
                    _m4.metric("Time", f"{_ga_state.get('elapsed_s',0):.1f}s")

                    if len(_h) >= 2:
                        try:
                            import plotly.graph_objects as go
                            _fig = go.Figure()
                            for _g_idx, _gen_scores in enumerate(_h):
                                _fig.add_trace(go.Box(
                                    y=_gen_scores,
                                    name=f"Gen {_g_idx}",
                                    boxmean=True,
                                ))
                            _fig.update_layout(
                                height=260,
                                margin=dict(l=40, r=20, t=20, b=30),
                                yaxis_title="Fitness",
                                showlegend=False,
                                plot_bgcolor="rgba(0,0,0,0)",
                            )
                            st.plotly_chart(_fig, use_container_width=True,
                                              key="ga_fitness_chart")
                        except Exception:
                            pass

                    _final = _ga_state.get("final_population") or []
                    if _final:
                        st.markdown(
                            f"### Top {min(5, len(_final))} survivors"
                        )
                        for _i, _d in enumerate(_final[:5], 1):
                            _meta = _d.get("execution_meta") or {}
                            _mode_str = _meta.get("regen_mode", "")
                            _badge = ("🧬 crossover"
                                       if "crossover" in _mode_str
                                       else "🧪 mutation"
                                       if "mutation" in _mode_str
                                       else "👴 parent")
                            with st.expander(
                                f"#{_i}. {_badge} {_d.get('title','Untitled')}",
                                expanded=(_i == 1),
                            ):
                                st.code(_d.get("title", "Untitled"),
                                        language=None)
                                if _meta.get("lineage_note"):
                                    st.markdown(
                                        f"> _{_meta['lineage_note']}_"
                                    )
                                if _meta.get("mutation_kind"):
                                    st.markdown(
                                        f"**Mutation:** {_meta['mutation_kind']}"
                                    )
                                _pa, _pb = (
                                    _meta.get("parent_a_title", ""),
                                    _meta.get("parent_b_title", ""),
                                )
                                if _pa or _pb:
                                    _parents_str = " × ".join(
                                        x for x in [_pa, _pb] if x
                                    )
                                    st.markdown(
                                        f"**Parents:** {_parents_str}"
                                    )
                                st.markdown(f"**Method.** {_d.get('method','')}")
                                st.markdown(
                                    f"**Hypothesis.** {_d.get('hypothesis','')}"
                                )

                        if st.button(
                            "➕ Add top survivors to session",
                            use_container_width=True,
                            key="ga_add_btn",
                        ):
                            ideas.extend(_final[:5])
                            st.session_state["_ga_result"] = None
                            st.success(f"Added top {min(5, len(_final))} idea(s).")
                            st.rerun()

        # ── Mode H: Heretic / anti-consensus ──────────────────────────────
        elif _novelty_mode == "heretic":
            try:
                from heretic_ideation import (
                    extract_dominant_beliefs, generate_heretic_idea,
                    generate_heretic_batch,
                )
            except ImportError as _e:
                st.error(f"heretic_ideation unavailable: {_e}")
                extract_dominant_beliefs = None

            if extract_dominant_beliefs is not None:
                st.markdown(
                    '<div style="background:#fef2f2;border:2px solid #dc2626;'
                    'border-radius:10px;padding:12px 16px;margin:8px 0;'
                    'color:#7c2d12;font-size:13px">'
                    '<b>⚠️ Heretic mode.</b> Names beliefs the field currently '
                    'accepts as canonical, then proposes research designed to '
                    "*falsify* them. These ideas often score worse on probes "
                    "than orthodox proposals — that's the point. Pair with the "
                    "<b>Exec Loop</b> to keep feasibility honest."
                    '</div>',
                    unsafe_allow_html=True,
                )
                _h_topic = st.text_input(
                    "Topic", value=results.get("topic", ""),
                    key="_h_topic", autocomplete="off",
                )
                _h_n = st.slider(
                    "Number of canonical beliefs to attack",
                    min_value=2, max_value=5, value=3, step=1, key="_h_n",
                )

                _hb1, _hb2 = st.columns(2)
                with _hb1:
                    if st.button(
                        "🔍 Extract dominant beliefs only",
                        use_container_width=True,
                        disabled=not _h_topic.strip(),
                        key="h_extract_btn",
                    ):
                        with st.spinner("Identifying the field's canon…"):
                            _h_beliefs = extract_dominant_beliefs(
                                _h_topic.strip(), n=int(_h_n),
                            )
                        st.session_state["_h_beliefs"] = [
                            b.to_dict() for b in _h_beliefs
                        ]
                        st.session_state["_h_ideas"] = []
                with _hb2:
                    if st.button(
                        f"🪖 Extract + falsify {_h_n} belief(s)",
                        type="primary", use_container_width=True,
                        disabled=not _h_topic.strip(),
                        key="h_batch_btn",
                    ):
                        with st.spinner(
                            f"Naming canon + generating {_h_n} heretical idea(s)…"
                        ):
                            _h_ideas_new = generate_heretic_batch(
                                _h_topic.strip(), n=int(_h_n),
                            )
                        st.session_state["_h_ideas"] = [
                            i.to_dict() for i in _h_ideas_new
                        ]
                        st.session_state["_h_beliefs"] = [
                            (i.execution_meta or {}).get("belief_targeted", {})
                            for i in _h_ideas_new
                        ]

                _h_beliefs_state = st.session_state.get("_h_beliefs", [])
                _h_ideas_state = st.session_state.get("_h_ideas", [])

                if _h_beliefs_state:
                    st.markdown("---")
                    st.markdown(f"### {len(_h_beliefs_state)} canonical belief(s) identified")
                    for _ix, _b in enumerate(_h_beliefs_state, 1):
                        _conf = _b.get("confidence", 0)
                        _color = ("#dc2626" if _conf >= 0.75
                                    else "#f59e0b" if _conf >= 0.50 else "#0ea5e9")
                        st.markdown(
                            f"<div style='background:#fafafa;border:1px solid #e2e8f0;"
                            f"border-left:5px solid {_color};border-radius:8px;"
                            f"padding:10px 14px;margin:6px 0'>"
                            f"<div style='font-size:11px;color:{_color};"
                            f"font-weight:700;text-transform:uppercase'>"
                            f"Belief #{_ix} · canonical-confidence {_conf:.2f}</div>"
                            f"<div style='font-size:14px;font-weight:700;color:#0c4a6e;margin-top:2px'>"
                            f"{_b.get('statement','')}</div>"
                            f"<div style='font-size:12px;color:#64748b;margin-top:4px'>"
                            f"<b>Evidence cited:</b> {_b.get('evidence_cited','')}<br>"
                            f"<b>Why canonical:</b> {_b.get('why_canonical','')}<br>"
                            f"<b>Cracks:</b> {_b.get('cracks','')}<br>"
                            f"<b>Falsification hint:</b> {_b.get('falsification_hint','')}"
                            f"</div></div>",
                            unsafe_allow_html=True,
                        )

                if _h_ideas_state:
                    st.markdown("---")
                    st.markdown(
                        f"### {len(_h_ideas_state)} heretical idea(s)"
                    )
                    for _i, _d in enumerate(_h_ideas_state, 1):
                        _meta = _d.get("execution_meta") or {}
                        with st.expander(
                            f"🪖 #{_i}. {_d.get('title','Untitled')}",
                            expanded=(_i == 1),
                        ):
                            st.code(_d.get("title", "Untitled"), language=None)
                            if _meta.get("falsification_mechanism"):
                                st.markdown(
                                    f"<div style='background:#fee2e2;"
                                    f"border-left:4px solid #dc2626;"
                                    f"padding:8px 12px;border-radius:6px;"
                                    f"font-size:13px;color:#7f1d1d;"
                                    f"margin-bottom:10px'>"
                                    f"<b>Falsification mechanism:</b> "
                                    f"{_meta['falsification_mechanism']}</div>",
                                    unsafe_allow_html=True,
                                )
                            st.markdown(f"**Motivation.** {_d.get('motivation','')}")
                            st.markdown(f"**Method.** {_d.get('method','')}")
                            st.markdown(f"**Hypothesis.** {_d.get('hypothesis','')}")
                            st.markdown(
                                f"**Risk (incl. field-political).** "
                                f"{_d.get('risk_assessment','')}"
                            )

                    if st.button("➕ Add all heretic ideas to session",
                                  use_container_width=True,
                                  key="h_add_btn"):
                        ideas.extend(_h_ideas_state)
                        st.session_state["_h_ideas"] = []
                        st.success(f"Added {len(_h_ideas_state)} idea(s).")
                        st.rerun()

        # ── Mode I: Persona-swap ──────────────────────────────────────────
        elif _novelty_mode == "persona":
            try:
                from persona_ideation import (
                    PERSONAS, persona_swap, generate_under_persona,
                )
            except ImportError as _e:
                st.error(f"persona_ideation unavailable: {_e}")
                PERSONAS = None

            if PERSONAS is not None:
                _p_topic = st.text_input(
                    "Topic", value=results.get("topic", ""),
                    key="_p_topic", autocomplete="off",
                )
                _persona_keys = list(PERSONAS.keys())
                _picked_personas = st.multiselect(
                    "Personas to generate under",
                    options=_persona_keys,
                    default=["skeptic", "industry_practitioner",
                              "philosopher", "naive_outsider"],
                    format_func=lambda k: (
                        f"{PERSONAS[k]['label']}  —  "
                        f"{PERSONAS[k]['tagline']}"
                    ),
                    key="_p_picked",
                )
                _pc1, _pc2 = st.columns(2)
                with _pc1:
                    _p_n_per = st.slider(
                        "Ideas per persona", min_value=1, max_value=3,
                        value=1, step=1, key="_p_n_per",
                    )
                with _pc2:
                    _p_thresh = st.slider(
                        "Similarity threshold (diversity filter)",
                        min_value=0.20, max_value=0.90, value=0.55,
                        step=0.05, key="_p_thresh",
                        help="Pairs above this Jaccard similarity are "
                             "considered duplicates; later one dropped.",
                    )

                if st.button(
                    f"🎭 Generate via {len(_picked_personas)} persona(s)",
                    type="primary", use_container_width=True,
                    disabled=(not _p_topic.strip() or not _picked_personas),
                    key="p_go_btn",
                ):
                    with st.spinner(
                        f"Inhabiting {len(_picked_personas)} personas in parallel…"
                    ):
                        _p_result = persona_swap(
                            _p_topic.strip(),
                            persona_ids=list(_picked_personas),
                            n_per_persona=int(_p_n_per),
                            similarity_threshold=float(_p_thresh),
                        )
                    st.session_state["_p_result"] = {
                        "topic": _p_result.topic,
                        "personas_used": _p_result.personas_used,
                        "kept_ideas": [i.to_dict() for i in _p_result.kept_ideas],
                        "all_ideas": [i.to_dict() for i in _p_result.all_ideas],
                        "rejected_pairs": _p_result.rejected_pairs,
                        "persona_stats": _p_result.persona_stats,
                        "elapsed_s": _p_result.elapsed_s,
                    }

                _p_state = st.session_state.get("_p_result")
                if _p_state and _p_state.get("topic") == _p_topic.strip():
                    st.markdown("---")
                    _kept = _p_state.get("kept_ideas") or []
                    _all = _p_state.get("all_ideas") or []
                    _rej = _p_state.get("rejected_pairs") or []
                    _pstats = _p_state.get("persona_stats") or {}

                    _m1, _m2, _m3, _m4 = st.columns(4)
                    _m1.metric("Kept", len(_kept))
                    _m2.metric("Generated", len(_all))
                    _m3.metric("Filtered out", len(_rej))
                    _m4.metric("Time",
                                 f"{_p_state.get('elapsed_s',0):.1f}s")

                    with st.expander("📊 Per-persona stats", expanded=False):
                        for _pid, _ps in _pstats.items():
                            st.markdown(
                                f"- {PERSONAS.get(_pid,{}).get('label',_pid)}: "
                                f"ok={_ps.get('ok',0)} fail={_ps.get('fail',0)}"
                            )
                    if _rej:
                        with st.expander(
                            f"🚫 Filtered duplicates ({len(_rej)})",
                            expanded=False,
                        ):
                            for _r in _rej:
                                st.markdown(
                                    f"- *{_r['rejected_title']}* "
                                    f"({_r.get('rejected_provider','?')}) "
                                    f"≈ *{_r['kept_title']}* "
                                    f"(sim={_r['similarity']})"
                                )

                    if _kept:
                        st.markdown(f"### {len(_kept)} idea(s) survived diversity filter")
                        for _i, _d in enumerate(_kept, 1):
                            _meta = _d.get("execution_meta") or {}
                            _ptag = _meta.get("persona_label", "?")
                            with st.expander(
                                f"🎭 #{_i}. {_ptag} — {_d.get('title','Untitled')}",
                                expanded=(_i == 1),
                            ):
                                st.code(_d.get("title", "Untitled"),
                                        language=None)
                                if _meta.get("persona_signature"):
                                    st.markdown(
                                        f"<div style='background:#ede9fe;"
                                        f"border-left:4px solid #a855f7;"
                                        f"padding:8px 12px;border-radius:6px;"
                                        f"font-size:13px;color:#3b0764;"
                                        f"margin-bottom:10px'>"
                                        f"<b>Persona signature:</b> "
                                        f"{_meta['persona_signature']}</div>",
                                        unsafe_allow_html=True,
                                    )
                                st.markdown(
                                    f"**Motivation.** {_d.get('motivation','')}"
                                )
                                st.markdown(f"**Method.** {_d.get('method','')}")
                                st.markdown(
                                    f"**Hypothesis.** {_d.get('hypothesis','')}"
                                )
                        if st.button("➕ Add kept ideas to session",
                                      use_container_width=True,
                                      key="p_add_btn"):
                            ideas.extend(_kept)
                            st.session_state["_p_result"] = None
                            st.success(f"Added {len(_kept)} idea(s).")
                            st.rerun()
                    else:
                        st.warning("No ideas survived. Check per-persona stats.")

        # ── Mode J: Counterfactual literature ─────────────────────────────
        elif _novelty_mode == "counterfactual":
            try:
                from counterfactual_literature import (
                    LITERATURE_KINDS, imagine_counterfactual_literature,
                    generate_from_counterfactual, counterfactual_batch,
                )
            except ImportError as _e:
                st.error(f"counterfactual_literature unavailable: {_e}")
                LITERATURE_KINDS = None

            if LITERATURE_KINDS is not None:
                st.markdown(
                    '<div style="background:#f0f9ff;border:1px solid #0ea5e9;'
                    'border-radius:10px;padding:12px 16px;margin:8px 0;'
                    'color:#0c4a6e;font-size:13px">'
                    '<b>ℹ️ Note.</b> Counterfactual literature entries are '
                    '<i>imagined by the LLM</i> as creative seeds — they are '
                    'not real papers. The point is to push the LLM out of '
                    'the canon, not to claim specific abandoned research '
                    'actually exists. Treat any author/year as a hint, '
                    'not a citation.'
                    '</div>',
                    unsafe_allow_html=True,
                )
                _cf_topic = st.text_input(
                    "Topic", value=results.get("topic", ""),
                    key="_cf_topic", autocomplete="off",
                )
                _cf_kinds = st.multiselect(
                    "Counterfactual literature kinds",
                    options=LITERATURE_KINDS,
                    default=["abandoned_direction", "contrarian_buried",
                              "cross_field_orphan"],
                    format_func=lambda k: k.replace("_", " ").title(),
                    key="_cf_kinds",
                )

                if st.button(
                    f"🔍 Imagine {len(_cf_kinds)} entries → generate ideas",
                    type="primary", use_container_width=True,
                    disabled=(not _cf_topic.strip() or not _cf_kinds),
                    key="cf_go_btn",
                ):
                    with st.spinner(
                        f"Imagining counterfactual literature + generating ideas…"
                    ):
                        _cf_ideas = counterfactual_batch(
                            _cf_topic.strip(),
                            n=len(_cf_kinds),
                            kinds=list(_cf_kinds),
                        )
                    st.session_state["_cf_ideas"] = [
                        i.to_dict() for i in _cf_ideas
                    ]

                _cf_state = st.session_state.get("_cf_ideas") or []
                if _cf_state:
                    st.markdown("---")
                    st.markdown(f"### {len(_cf_state)} counterfactual idea(s)")
                    for _i, _d in enumerate(_cf_state, 1):
                        _meta = _d.get("execution_meta") or {}
                        _e = _meta.get("literature_entry", {})
                        _kind = _e.get("kind", "?")
                        with st.expander(
                            f"🔍 #{_i}. [{_kind.replace('_',' ')}] "
                            f"{_d.get('title','Untitled')}",
                            expanded=(_i == 1),
                        ):
                            st.code(_d.get("title", "Untitled"),
                                    language=None)
                            if _e:
                                st.markdown(
                                    f"<div style='background:#fef3c7;"
                                    f"border-left:4px solid #f59e0b;"
                                    f"padding:10px 14px;border-radius:6px;"
                                    f"font-size:13px;color:#713f12;"
                                    f"margin-bottom:10px'>"
                                    f"<b>Counterfactual seed (imagined):</b><br>"
                                    f"<i>{_e.get('title','')}</i><br>"
                                    f"<span style='font-size:11px'>"
                                    f"{_e.get('authors','')} · {_e.get('year','?')}</span><br>"
                                    f"<b>What it would claim:</b> {_e.get('summary','')}<br>"
                                    f"<b>Why neglected:</b> {_e.get('why_neglected','')}<br>"
                                    f"<b>Thread to revive:</b> {_e.get('what_to_revive','')}"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )
                            if _meta.get("what_changed"):
                                st.markdown(
                                    f"**What's different now:** "
                                    f"_{_meta['what_changed']}_"
                                )
                            st.markdown(f"**Motivation.** {_d.get('motivation','')}")
                            st.markdown(f"**Method.** {_d.get('method','')}")
                            st.markdown(f"**Hypothesis.** {_d.get('hypothesis','')}")

                    if st.button("➕ Add all counterfactual ideas to session",
                                  use_container_width=True,
                                  key="cf_add_btn"):
                        ideas.extend(_cf_state)
                        st.session_state["_cf_ideas"] = []
                        st.success(f"Added {len(_cf_state)} idea(s).")
                        st.rerun()

        # ── Mode K: Analogy-bridge (cross-domain structural transplant) ───
        elif _novelty_mode == "analogy":
            try:
                from analogy_ideation import (
                    DEFAULT_DOMAINS, analogy_batch,
                )
            except ImportError as _e:
                st.error(f"analogy_ideation unavailable: {_e}")
                analogy_batch = None

            if analogy_batch is not None:
                st.caption(
                    "Pick a far source domain — the LLM names a structural "
                    "isomorphism with your topic, then transplants a "
                    "concrete method through that mapping. Strategy code "
                    "**M** (Morphism)."
                )
                _an_topic = st.text_input(
                    "Topic", value=results.get("topic", ""),
                    key="_an_topic", autocomplete="off",
                )
                _an_domains = st.multiselect(
                    "Source domains",
                    options=DEFAULT_DOMAINS,
                    default=DEFAULT_DOMAINS[:3],
                    key="_an_domains",
                    help="Each chosen domain → one structural-analogy idea.",
                )

                if st.button(
                    f"🌉 Bridge {len(_an_domains)} domain(s) → ideas",
                    type="primary", use_container_width=True,
                    disabled=(not _an_topic.strip() or not _an_domains),
                    key="an_go_btn",
                ):
                    with st.spinner("Mapping morphisms + transplanting methods…"):
                        _an_ideas = analogy_batch(
                            _an_topic.strip(),
                            n=len(_an_domains),
                            domains=list(_an_domains),
                        )
                    st.session_state["_an_ideas"] = [
                        i.to_dict() for i in _an_ideas
                    ]

                _an_state = st.session_state.get("_an_ideas") or []
                if _an_state:
                    st.markdown("---")
                    st.markdown(f"### {len(_an_state)} analogy idea(s)")
                    for _i, _d in enumerate(_an_state, 1):
                        _meta = _d.get("execution_meta") or {}
                        _bridge = _meta.get("analogy_bridge", {})
                        _src = _bridge.get("source_domain", "?")
                        with st.expander(
                            f"🌉 #{_i}. [{_src}] "
                            f"{_d.get('title','Untitled')}",
                            expanded=(_i == 1),
                        ):
                            st.code(_d.get("title", "Untitled"),
                                    language=None)
                            if _bridge:
                                st.markdown(
                                    f"<div style='background:#ecfeff;"
                                    f"border-left:4px solid #06b6d4;"
                                    f"padding:10px 14px;border-radius:6px;"
                                    f"font-size:13px;color:#164e63;"
                                    f"margin-bottom:10px'>"
                                    f"<b>Source domain:</b> {_bridge.get('source_domain','')}<br>"
                                    f"<b>Source structure:</b> {_bridge.get('source_structure','')}<br>"
                                    f"<b>Target counterpart:</b> {_bridge.get('target_counterpart','')}<br>"
                                    f"<b>Morphism:</b> {_bridge.get('morphism','')}<br>"
                                    f"<b>Invariant preserved:</b> {_bridge.get('invariant','')}<br>"
                                    f"<b>Where it may break:</b> {_bridge.get('risk_of_break','')}"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )
                            if _meta.get("transplanted_mechanism"):
                                st.markdown(
                                    f"**Transplanted mechanism:** "
                                    f"_{_meta['transplanted_mechanism']}_"
                                )
                            st.markdown(f"**Motivation.** {_d.get('motivation','')}")
                            st.markdown(f"**Method.** {_d.get('method','')}")
                            st.markdown(f"**Hypothesis.** {_d.get('hypothesis','')}")

                    if st.button("➕ Add all analogy ideas to session",
                                  use_container_width=True,
                                  key="an_add_btn"):
                        ideas.extend(_an_state)
                        st.session_state["_an_ideas"] = []
                        st.success(f"Added {len(_an_state)} idea(s).")
                        st.rerun()

        # ── Mode L: Failure-mode antibody (immune-by-design) ──────────────
        elif _novelty_mode == "failure_mode":
            try:
                from failure_mode_ideation import (
                    DEFAULT_FAILURE_MODES, failure_mode_batch,
                )
            except ImportError as _e:
                st.error(f"failure_mode_ideation unavailable: {_e}")
                failure_mode_batch = None

            if failure_mode_batch is not None:
                st.caption(
                    "Enumerate the most common ways research in this topic "
                    "*fails* (data leakage, weak baseline, p-hacking, …), "
                    "then design ideas whose method is structurally immune "
                    "to each. Strategy code **Y** (antibodY)."
                )
                _fm_topic = st.text_input(
                    "Topic", value=results.get("topic", ""),
                    key="_fm_topic", autocomplete="off",
                )
                _fm_n = st.slider(
                    "How many failure modes to attack?",
                    min_value=1, max_value=6, value=3, step=1,
                    key="_fm_n",
                )
                with st.expander("Reference failure-mode catalog",
                                  expanded=False):
                    st.markdown(
                        "The LLM may use, refine, or replace these:\n\n"
                        + "\n".join(
                            f"- `{m}`" for m in DEFAULT_FAILURE_MODES
                        )
                    )

                if st.button(
                    f"🛡️ Enumerate {_fm_n} failure mode(s) → immune ideas",
                    type="primary", use_container_width=True,
                    disabled=(not _fm_topic.strip()),
                    key="fm_go_btn",
                ):
                    with st.spinner("Enumerating failures + designing antibodies…"):
                        _fm_ideas = failure_mode_batch(
                            _fm_topic.strip(), n=int(_fm_n),
                        )
                    st.session_state["_fm_ideas"] = [
                        i.to_dict() for i in _fm_ideas
                    ]

                _fm_state = st.session_state.get("_fm_ideas") or []
                if _fm_state:
                    st.markdown("---")
                    st.markdown(f"### {len(_fm_state)} immune idea(s)")
                    for _i, _d in enumerate(_fm_state, 1):
                        _meta = _d.get("execution_meta") or {}
                        _mode = _meta.get("failure_mode_targeted", {})
                        _name = _mode.get("name", "?")
                        with st.expander(
                            f"🛡️ #{_i}. [{_name}] "
                            f"{_d.get('title','Untitled')}",
                            expanded=(_i == 1),
                        ):
                            st.code(_d.get("title", "Untitled"),
                                    language=None)
                            if _mode:
                                _sev = _mode.get("severity", 0.5)
                                st.markdown(
                                    f"<div style='background:#fef2f2;"
                                    f"border-left:4px solid #ef4444;"
                                    f"padding:10px 14px;border-radius:6px;"
                                    f"font-size:13px;color:#7f1d1d;"
                                    f"margin-bottom:10px'>"
                                    f"<b>Failure mode:</b> {_mode.get('name','')} "
                                    f"(severity {_sev:.2f})<br>"
                                    f"<b>Mechanism:</b> {_mode.get('mechanism','')}<br>"
                                    f"<b>Common signs:</b> {_mode.get('common_signs','')}<br>"
                                    f"<b>Immunity strategy:</b> {_mode.get('immunity_strategy','')}"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )
                            if _meta.get("immunity_mechanism"):
                                st.markdown(
                                    f"**Immunity baked in:** "
                                    f"_{_meta['immunity_mechanism']}_"
                                )
                            st.markdown(f"**Motivation.** {_d.get('motivation','')}")
                            st.markdown(f"**Method.** {_d.get('method','')}")
                            st.markdown(f"**Hypothesis.** {_d.get('hypothesis','')}")

                    if st.button("➕ Add all immune ideas to session",
                                  use_container_width=True,
                                  key="fm_add_btn"):
                        ideas.extend(_fm_state)
                        st.session_state["_fm_ideas"] = []
                        st.success(f"Added {len(_fm_state)} idea(s).")
                        st.rerun()

        # ── Mode M: Extremum (pin one axis to its limit) ──────────────────
        elif _novelty_mode == "extremum":
            try:
                from extremum_ideation import AXES, extremum_batch
            except ImportError as _e:
                st.error(f"extremum_ideation unavailable: {_e}")
                AXES = None

            if AXES is not None:
                st.caption(
                    "Pin one design axis (compute, data, params, latency, "
                    "…) to its **minimal** or **maximal** extreme. The LLM "
                    "proposes ideas that only make sense at that regime. "
                    "Strategy code **T** (exTremum)."
                )
                _ex_topic = st.text_input(
                    "Topic", value=results.get("topic", ""),
                    key="_ex_topic", autocomplete="off",
                )
                _ex_pair_labels = [
                    f"{ax} → {direction}" for ax in AXES
                    for direction in AXES[ax]
                ]
                _ex_pair_map = {
                    f"{ax} → {direction}": (ax, direction)
                    for ax in AXES for direction in AXES[ax]
                }
                _ex_selection = st.multiselect(
                    "Axis / direction pairs",
                    options=_ex_pair_labels,
                    default=[
                        "compute → minimal",
                        "data → minimal",
                        "interpretability → maximal",
                    ],
                    key="_ex_pairs",
                    help="Each pair → one regime-exploiting idea.",
                )

                if st.button(
                    f"🚀 Push {len(_ex_selection)} axis/regime(s) → ideas",
                    type="primary", use_container_width=True,
                    disabled=(not _ex_topic.strip() or not _ex_selection),
                    key="ex_go_btn",
                ):
                    _pairs = [_ex_pair_map[s] for s in _ex_selection]
                    with st.spinner("Proposing regimes + exploiting extremes…"):
                        _ex_ideas = extremum_batch(
                            _ex_topic.strip(),
                            n=len(_pairs),
                            pairs=_pairs,
                        )
                    st.session_state["_ex_ideas"] = [
                        i.to_dict() for i in _ex_ideas
                    ]

                _ex_state = st.session_state.get("_ex_ideas") or []
                if _ex_state:
                    st.markdown("---")
                    st.markdown(f"### {len(_ex_state)} extremum idea(s)")
                    for _i, _d in enumerate(_ex_state, 1):
                        _meta = _d.get("execution_meta") or {}
                        _reg = _meta.get("regime", {})
                        _lbl = (
                            f"{_reg.get('axis','?')}:{_reg.get('direction','?')}"
                        )
                        with st.expander(
                            f"🚀 #{_i}. [{_lbl}] "
                            f"{_d.get('title','Untitled')}",
                            expanded=(_i == 1),
                        ):
                            st.code(_d.get("title", "Untitled"),
                                    language=None)
                            if _reg:
                                st.markdown(
                                    f"<div style='background:#f5f3ff;"
                                    f"border-left:4px solid #8b5cf6;"
                                    f"padding:10px 14px;border-radius:6px;"
                                    f"font-size:13px;color:#4c1d95;"
                                    f"margin-bottom:10px'>"
                                    f"<b>Regime:</b> {_reg.get('axis','')} "
                                    f"→ {_reg.get('direction','')}<br>"
                                    f"<b>Magnitude:</b> {_reg.get('magnitude','')}<br>"
                                    f"<b>Why hard:</b> {_reg.get('why_hard','')}<br>"
                                    f"<b>What flips:</b> {_reg.get('what_changes','')}<br>"
                                    f"<b>Only possible here:</b> {_reg.get('only_here','')}"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )
                            if _meta.get("regime_exploit"):
                                st.markdown(
                                    f"**Regime exploit:** "
                                    f"_{_meta['regime_exploit']}_"
                                )
                            st.markdown(f"**Motivation.** {_d.get('motivation','')}")
                            st.markdown(f"**Method.** {_d.get('method','')}")
                            st.markdown(f"**Hypothesis.** {_d.get('hypothesis','')}")

                    if st.button("➕ Add all extremum ideas to session",
                                  use_container_width=True,
                                  key="ex_add_btn"):
                        ideas.extend(_ex_state)
                        st.session_state["_ex_ideas"] = []
                        st.success(f"Added {len(_ex_state)} idea(s).")
                        st.rerun()

        # ── Mode N: Inversion / Jeopardy ───────────────────────────────────
        elif _novelty_mode == "inversion":
            try:
                from inversion_ideation import ANSWER_TONES, inversion_batch
            except ImportError as _e:
                st.error(f"inversion_ideation unavailable: {_e}")
                ANSWER_TONES = None

            if ANSWER_TONES is not None:
                st.caption(
                    "Jeopardy mode: the LLM names a specific surprising "
                    "*answer* (the underdog wins, the favourite fails, the "
                    "effect vanishes…) and works backward to the research "
                    "question. Strategy code **I** (Inversion)."
                )
                _iv_topic = st.text_input(
                    "Topic", value=results.get("topic", ""),
                    key="_iv_topic", autocomplete="off",
                )
                _iv_tones = st.multiselect(
                    "Answer tones",
                    options=ANSWER_TONES,
                    default=["underdog_wins", "favourite_fails",
                              "tradeoff_inversion"],
                    format_func=lambda k: k.replace("_", " ").title(),
                    key="_iv_tones",
                )

                if st.button(
                    f"🔄 Imagine {len(_iv_tones)} answer(s) → derive questions",
                    type="primary", use_container_width=True,
                    disabled=(not _iv_topic.strip() or not _iv_tones),
                    key="iv_go_btn",
                ):
                    with st.spinner("Imagining answers + deriving questions…"):
                        _iv_ideas = inversion_batch(
                            _iv_topic.strip(),
                            n=len(_iv_tones),
                            tones=list(_iv_tones),
                        )
                    st.session_state["_iv_ideas"] = [
                        i.to_dict() for i in _iv_ideas
                    ]

                _iv_state = st.session_state.get("_iv_ideas") or []
                if _iv_state:
                    st.markdown("---")
                    st.markdown(f"### {len(_iv_state)} inversion idea(s)")
                    for _i, _d in enumerate(_iv_state, 1):
                        _meta = _d.get("execution_meta") or {}
                        _ans = _meta.get("candidate_answer", {})
                        _tone = _ans.get("tone", "?")
                        with st.expander(
                            f"🔄 #{_i}. [{_tone.replace('_',' ')}] "
                            f"{_d.get('title','Untitled')}",
                            expanded=(_i == 1),
                        ):
                            st.code(_d.get("title", "Untitled"),
                                    language=None)
                            if _ans:
                                _pl = _ans.get("plausibility", 0.5)
                                st.markdown(
                                    f"<div style='background:#fffbeb;"
                                    f"border-left:4px solid #f59e0b;"
                                    f"padding:10px 14px;border-radius:6px;"
                                    f"font-size:13px;color:#78350f;"
                                    f"margin-bottom:10px'>"
                                    f"<b>Candidate answer ({_tone}):</b><br>"
                                    f"<i>{_ans.get('headline','')}</i><br>"
                                    f"<b>Why surprising:</b> {_ans.get('why_surprising','')}<br>"
                                    f"<b>Why plausible:</b> {_ans.get('why_plausible','')}<br>"
                                    f"<b>Measurable claim:</b> {_ans.get('measurable_claim','')}<br>"
                                    f"<b>Plausibility:</b> {_pl:.2f}"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )
                            if _meta.get("derived_question"):
                                st.markdown(
                                    f"**Derived question:** "
                                    f"_{_meta['derived_question']}_"
                                )
                            st.markdown(f"**Motivation.** {_d.get('motivation','')}")
                            st.markdown(f"**Method.** {_d.get('method','')}")
                            st.markdown(f"**Hypothesis.** {_d.get('hypothesis','')}")

                    if st.button("➕ Add all inversion ideas to session",
                                  use_container_width=True,
                                  key="iv_add_btn"):
                        ideas.extend(_iv_state)
                        st.session_state["_iv_ideas"] = []
                        st.success(f"Added {len(_iv_state)} idea(s).")
                        st.rerun()

        # ── Mode O: Null-result lab ────────────────────────────────────────
        elif _novelty_mode == "null_result":
            try:
                from null_result_ideation import NULL_KINDS, null_result_batch
            except ImportError as _e:
                st.error(f"null_result_ideation unavailable: {_e}")
                NULL_KINDS = None

            if NULL_KINDS is not None:
                st.caption(
                    "Pre-registered negative-result lab: design a study "
                    "whose primary deliverable is a clean null — with "
                    "power analysis, equivalence margin, and acceptance "
                    "criteria. Strategy code **Z** (Zero / null)."
                )
                _nr_topic = st.text_input(
                    "Topic", value=results.get("topic", ""),
                    key="_nr_topic", autocomplete="off",
                )
                _nr_kinds = st.multiselect(
                    "Null target categories",
                    options=NULL_KINDS,
                    default=["transfer_failure", "cohort_invalidity",
                              "ablation_unimportant"],
                    format_func=lambda k: k.replace("_", " ").title(),
                    key="_nr_kinds",
                )

                if st.button(
                    f"🅾️ Test {len(_nr_kinds)} null target(s) → designs",
                    type="primary", use_container_width=True,
                    disabled=(not _nr_topic.strip() or not _nr_kinds),
                    key="nr_go_btn",
                ):
                    with st.spinner("Selecting targets + designing power analyses…"):
                        _nr_ideas = null_result_batch(
                            _nr_topic.strip(),
                            n=len(_nr_kinds),
                            kinds=list(_nr_kinds),
                        )
                    st.session_state["_nr_ideas"] = [
                        i.to_dict() for i in _nr_ideas
                    ]

                _nr_state = st.session_state.get("_nr_ideas") or []
                if _nr_state:
                    st.markdown("---")
                    st.markdown(f"### {len(_nr_state)} null-result design(s)")
                    for _i, _d in enumerate(_nr_state, 1):
                        _meta = _d.get("execution_meta") or {}
                        _tgt = _meta.get("null_target", {})
                        _kind = _tgt.get("kind", "?")
                        with st.expander(
                            f"🅾️ #{_i}. [{_kind.replace('_',' ')}] "
                            f"{_d.get('title','Untitled')}",
                            expanded=(_i == 1),
                        ):
                            st.code(_d.get("title", "Untitled"),
                                    language=None)
                            if _tgt:
                                _stk = _tgt.get("stakes", 0.5)
                                st.markdown(
                                    f"<div style='background:#f1f5f9;"
                                    f"border-left:4px solid #475569;"
                                    f"padding:10px 14px;border-radius:6px;"
                                    f"font-size:13px;color:#0f172a;"
                                    f"margin-bottom:10px'>"
                                    f"<b>Null target ({_kind}):</b><br>"
                                    f"<i>{_tgt.get('claim_to_be_negated','')}</i><br>"
                                    f"<b>Population:</b> {_tgt.get('population','')}<br>"
                                    f"<b>Equivalence margin:</b> {_tgt.get('equivalence_margin','')}<br>"
                                    f"<b>Why doubt now:</b> {_tgt.get('why_doubt_now','')}<br>"
                                    f"<b>Stakes if null confirmed:</b> {_stk:.2f}"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )
                            if _meta.get("null_acceptance_criteria"):
                                st.markdown(
                                    f"**Acceptance criteria for declaring null:** "
                                    f"_{_meta['null_acceptance_criteria']}_"
                                )
                            if _meta.get("power_analysis_summary"):
                                st.markdown(
                                    f"**Power analysis:** "
                                    f"_{_meta['power_analysis_summary']}_"
                                )
                            st.markdown(f"**Motivation.** {_d.get('motivation','')}")
                            st.markdown(f"**Method.** {_d.get('method','')}")
                            st.markdown(f"**Null hypothesis.** {_d.get('hypothesis','')}")

                    if st.button("➕ Add all null-result designs to session",
                                  use_container_width=True,
                                  key="nr_add_btn"):
                        ideas.extend(_nr_state)
                        st.session_state["_nr_ideas"] = []
                        st.success(f"Added {len(_nr_state)} idea(s).")
                        st.rerun()

        # ── Mode P: Underserved cohort ─────────────────────────────────────
        elif _novelty_mode == "underserved_cohort":
            try:
                from underserved_cohort_ideation import (
                    COHORT_DIMENSIONS, underserved_cohort_batch,
                )
            except ImportError as _e:
                st.error(f"underserved_cohort_ideation unavailable: {_e}")
                COHORT_DIMENSIONS = None

            if COHORT_DIMENSIONS is not None:
                st.caption(
                    "Fix a *who*, not a *what*: identify a specific cohort "
                    "the canonical literature under-serves and anchor the "
                    "research design — evaluation, dataset, success metric "
                    "— to them. Strategy code **W** (Who-driven)."
                )
                _uc_topic = st.text_input(
                    "Topic", value=results.get("topic", ""),
                    key="_uc_topic", autocomplete="off",
                )
                _uc_dims = st.multiselect(
                    "Cohort dimensions",
                    options=COHORT_DIMENSIONS,
                    default=["linguistic", "infrastructural", "geopolitical"],
                    format_func=lambda k: k.replace("_", " ").title(),
                    key="_uc_dims",
                )

                if st.button(
                    f"🤝 Identify {len(_uc_dims)} cohort(s) → cohort-anchored ideas",
                    type="primary", use_container_width=True,
                    disabled=(not _uc_topic.strip() or not _uc_dims),
                    key="uc_go_btn",
                ):
                    with st.spinner("Identifying cohorts + anchoring designs…"):
                        _uc_ideas = underserved_cohort_batch(
                            _uc_topic.strip(),
                            n=len(_uc_dims),
                            dimensions=list(_uc_dims),
                        )
                    st.session_state["_uc_ideas"] = [
                        i.to_dict() for i in _uc_ideas
                    ]

                _uc_state = st.session_state.get("_uc_ideas") or []
                if _uc_state:
                    st.markdown("---")
                    st.markdown(f"### {len(_uc_state)} cohort-anchored idea(s)")
                    for _i, _d in enumerate(_uc_state, 1):
                        _meta = _d.get("execution_meta") or {}
                        _coh = _meta.get("cohort", {})
                        _dim = _coh.get("dimension", "?")
                        _name = _coh.get("name", "?")
                        with st.expander(
                            f"🤝 #{_i}. [{_dim}] {_name} — "
                            f"{_d.get('title','Untitled')}",
                            expanded=(_i == 1),
                        ):
                            st.code(_d.get("title", "Untitled"),
                                    language=None)
                            if _coh:
                                _ov = _coh.get("overlooked_factor", 0.5)
                                st.markdown(
                                    f"<div style='background:#f0fdf4;"
                                    f"border-left:4px solid #16a34a;"
                                    f"padding:10px 14px;border-radius:6px;"
                                    f"font-size:13px;color:#14532d;"
                                    f"margin-bottom:10px'>"
                                    f"<b>Cohort ({_dim}):</b> {_coh.get('name','')}<br>"
                                    f"<b>Who they are:</b> {_coh.get('description','')}<br>"
                                    f"<b>Why under-served:</b> {_coh.get('why_underserved','')}<br>"
                                    f"<b>Canonical failure for them:</b> {_coh.get('canonical_failure_mode','')}<br>"
                                    f"<b>Cohort-anchored success metric:</b> {_coh.get('success_metric','')}<br>"
                                    f"<b>How badly overlooked:</b> {_ov:.2f}"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )
                            if _meta.get("cohort_representation"):
                                st.markdown(
                                    f"**Cohort in the data:** "
                                    f"_{_meta['cohort_representation']}_"
                                )
                            st.markdown(f"**Motivation.** {_d.get('motivation','')}")
                            st.markdown(f"**Method.** {_d.get('method','')}")
                            st.markdown(f"**Hypothesis.** {_d.get('hypothesis','')}")

                    if st.button("➕ Add all cohort-anchored ideas to session",
                                  use_container_width=True,
                                  key="uc_add_btn"):
                        ideas.extend(_uc_state)
                        st.session_state["_uc_ideas"] = []
                        st.success(f"Added {len(_uc_state)} idea(s).")
                        st.rerun()

        # ── Mode Q: Composable primitive ───────────────────────────────────
        elif _novelty_mode == "composable_primitive":
            try:
                from composable_primitive_ideation import (
                    PRIMITIVE_KINDS, composable_primitive_batch,
                )
            except ImportError as _e:
                st.error(f"composable_primitive_ideation unavailable: {_e}")
                PRIMITIVE_KINDS = None

            if PRIMITIVE_KINDS is not None:
                st.caption(
                    "Design the building block, not the paper. The "
                    "deliverable IS a small reusable primitive that "
                    "downstream papers will import as a load-bearing "
                    "dependency. Strategy code **D** (Downstream-composable)."
                )
                _cp_topic = st.text_input(
                    "Topic", value=results.get("topic", ""),
                    key="_cp_topic", autocomplete="off",
                )
                _cp_kinds = st.multiselect(
                    "Primitive kinds",
                    options=PRIMITIVE_KINDS,
                    default=["evaluation_harness", "diagnostic_probe",
                              "compositional_block"],
                    format_func=lambda k: k.replace("_", " ").title(),
                    key="_cp_kinds",
                )

                if st.button(
                    f"🧩 Spec {len(_cp_kinds)} primitive(s) → designs",
                    type="primary", use_container_width=True,
                    disabled=(not _cp_topic.strip() or not _cp_kinds),
                    key="cp_go_btn",
                ):
                    with st.spinner("Identifying slots + designing primitives…"):
                        _cp_ideas = composable_primitive_batch(
                            _cp_topic.strip(),
                            n=len(_cp_kinds),
                            kinds=list(_cp_kinds),
                        )
                    st.session_state["_cp_ideas"] = [
                        i.to_dict() for i in _cp_ideas
                    ]

                _cp_state = st.session_state.get("_cp_ideas") or []
                if _cp_state:
                    st.markdown("---")
                    st.markdown(f"### {len(_cp_state)} primitive design(s)")
                    for _i, _d in enumerate(_cp_state, 1):
                        _meta = _d.get("execution_meta") or {}
                        _slot = _meta.get("primitive_slot", {})
                        _kind = _slot.get("kind", "?")
                        _name = _slot.get("name", "?")
                        with st.expander(
                            f"🧩 #{_i}. [{_kind}] {_name} — "
                            f"{_d.get('title','Untitled')}",
                            expanded=(_i == 1),
                        ):
                            st.code(_d.get("title", "Untitled"),
                                    language=None)
                            if _slot:
                                _br = _slot.get("blast_radius", 0.5)
                                st.markdown(
                                    f"<div style='background:#eff6ff;"
                                    f"border-left:4px solid #3b82f6;"
                                    f"padding:10px 14px;border-radius:6px;"
                                    f"font-size:13px;color:#1e3a8a;"
                                    f"margin-bottom:10px'>"
                                    f"<b>Primitive slot ({_kind}):</b> {_slot.get('name','')}<br>"
                                    f"<b>Description:</b> {_slot.get('description','')}<br>"
                                    f"<b>Why unfilled today:</b> {_slot.get('why_unfilled','')}<br>"
                                    f"<b>Downstream users:</b> {_slot.get('downstream_users','')}<br>"
                                    f"<b>Adoption proxy:</b> {_slot.get('adoption_proxy','')}<br>"
                                    f"<b>Blast radius:</b> {_br:.2f}"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )
                            if _meta.get("api_surface"):
                                st.markdown(
                                    f"**API surface / interface:** "
                                    f"_{_meta['api_surface']}_"
                                )
                            if _meta.get("non_goals"):
                                st.markdown(
                                    f"**Explicit non-goals:** "
                                    f"_{_meta['non_goals']}_"
                                )
                            st.markdown(f"**Motivation.** {_d.get('motivation','')}")
                            st.markdown(f"**Method.** {_d.get('method','')}")
                            st.markdown(f"**Hypothesis.** {_d.get('hypothesis','')}")

                    if st.button("➕ Add all primitive designs to session",
                                  use_container_width=True,
                                  key="cp_add_btn"):
                        ideas.extend(_cp_state)
                        st.session_state["_cp_ideas"] = []
                        st.success(f"Added {len(_cp_state)} idea(s).")
                        st.rerun()

        # ── Mode R: Stakeholder Pareto ─────────────────────────────────────
        elif _novelty_mode == "stakeholder_pareto":
            try:
                from stakeholder_pareto_ideation import (
                    DEFAULT_ROLES, stakeholder_pareto_batch,
                )
            except ImportError as _e:
                st.error(f"stakeholder_pareto_ideation unavailable: {_e}")
                DEFAULT_ROLES = None

            if DEFAULT_ROLES is not None:
                st.caption(
                    "Pick 3+ stakeholders (researcher / end-user / funder / "
                    "regulator / domain expert / …); the LLM designs a "
                    "study with a *measurable* win for each. No party gets "
                    "a hand-wave. Strategy code **S**."
                )
                _sp_topic = st.text_input(
                    "Topic", value=results.get("topic", ""),
                    key="_sp_topic", autocomplete="off",
                )
                _sp_n = st.slider(
                    "How many Pareto designs?",
                    min_value=1, max_value=5, value=2, step=1,
                    key="_sp_n",
                )
                _sp_cast_size = st.slider(
                    "Stakeholders per design",
                    min_value=2, max_value=min(6, len(DEFAULT_ROLES)),
                    value=3, step=1, key="_sp_cast_size",
                )
                _sp_roles = st.multiselect(
                    "Roles pool to sample from",
                    options=DEFAULT_ROLES,
                    default=DEFAULT_ROLES[:6],
                    format_func=lambda k: k.replace("_", " ").title(),
                    key="_sp_roles",
                    help="Each design samples cast_size roles from this pool.",
                )

                if st.button(
                    f"🤹 Generate {_sp_n} Pareto design(s)",
                    type="primary", use_container_width=True,
                    disabled=(
                        not _sp_topic.strip()
                        or len(_sp_roles) < _sp_cast_size
                    ),
                    key="sp_go_btn",
                ):
                    with st.spinner("Casting stakeholders + reconciling wins…"):
                        _sp_ideas = stakeholder_pareto_batch(
                            _sp_topic.strip(),
                            n=int(_sp_n),
                            cast_size=int(_sp_cast_size),
                            roles=list(_sp_roles),
                        )
                    st.session_state["_sp_ideas"] = [
                        i.to_dict() for i in _sp_ideas
                    ]

                _sp_state = st.session_state.get("_sp_ideas") or []
                if _sp_state:
                    st.markdown("---")
                    st.markdown(f"### {len(_sp_state)} Pareto design(s)")
                    for _i, _d in enumerate(_sp_state, 1):
                        _meta = _d.get("execution_meta") or {}
                        _cast = _meta.get("stakeholders") or []
                        with st.expander(
                            f"🤹 #{_i}. {_d.get('title','Untitled')} "
                            f"— {len(_cast)} parties",
                            expanded=(_i == 1),
                        ):
                            st.code(_d.get("title", "Untitled"),
                                    language=None)
                            if _cast:
                                _rows = "".join(
                                    f"<li><b>[{s.get('role','?')}]</b> "
                                    f"{s.get('name','')} — "
                                    f"<i>metric:</i> {s.get('metric','')}; "
                                    f"<i>win:</i> {s.get('win_condition','')}"
                                    f"</li>"
                                    for s in _cast
                                )
                                st.markdown(
                                    f"<div style='background:#fdf2f8;"
                                    f"border-left:4px solid #ec4899;"
                                    f"padding:10px 14px;border-radius:6px;"
                                    f"font-size:13px;color:#831843;"
                                    f"margin-bottom:10px'>"
                                    f"<b>Stakeholder cast:</b>"
                                    f"<ul style='margin:6px 0 0 14px;padding:0'>{_rows}</ul>"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )
                            if _meta.get("tradeoffs_named"):
                                st.markdown(
                                    f"**Tradeoffs named:** "
                                    f"_{_meta['tradeoffs_named']}_"
                                )
                            _psm = _meta.get("per_stakeholder_metric")
                            if isinstance(_psm, dict) and _psm:
                                _bullets = "\n".join(
                                    f"- **{k}** → {v}"
                                    for k, v in _psm.items()
                                )
                                st.markdown(
                                    f"**Per-stakeholder metric bundle:**\n\n{_bullets}"
                                )
                            elif isinstance(_psm, str) and _psm.strip():
                                st.markdown(
                                    f"**Per-stakeholder metric bundle:** "
                                    f"{_psm}"
                                )
                            st.markdown(f"**Motivation.** {_d.get('motivation','')}")
                            st.markdown(f"**Method.** {_d.get('method','')}")
                            st.markdown(f"**Hypothesis.** {_d.get('hypothesis','')}")

                    if st.button("➕ Add all Pareto designs to session",
                                  use_container_width=True,
                                  key="sp_add_btn"):
                        ideas.extend(_sp_state)
                        st.session_state["_sp_ideas"] = []
                        st.success(f"Added {len(_sp_state)} idea(s).")
                        st.rerun()

        # ── Mode S: Corpus-anchored novelty ────────────────────────────────
        else:  # _novelty_mode == "corpus_anchored"
            try:
                from corpus_novelty import (
                    CORPUS_LIMITS_DISCLAIMER,
                    DEFAULT_GENERATORS,
                    ReferenceCorpus,
                    corpus_anchored_batch,
                )
            except ImportError as _e:
                st.error(f"corpus_novelty unavailable: {_e}")
                ReferenceCorpus = None

            if ReferenceCorpus is not None:
                st.caption(
                    "Run multiple generators, then **score each candidate "
                    "against a reference corpus** (embedding-distance to "
                    "nearest neighbor). Surfaces nearest matches so you "
                    "see *what* the score is pushing away from. "
                    "Strategy code **Q**."
                )

                # Honest-limits disclaimer (per the design manifesto).
                st.markdown(
                    f"<div style='background:#fef3c7;border-left:4px solid "
                    f"#f59e0b;padding:10px 14px;border-radius:6px;"
                    f"font-size:12px;color:#713f12;margin:8px 0'>"
                    f"<b>⚠️ Honest limits.</b> {CORPUS_LIMITS_DISCLAIMER}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                _ca_topic = st.text_input(
                    "Topic", value=results.get("topic", ""),
                    key="_ca_topic", autocomplete="off",
                )

                # ── Corpus source ───────────────────────────────────────
                _ca_source = st.radio(
                    "Reference corpus source",
                    options=["archive", "seed_paste"],
                    format_func=lambda k: {
                        "archive": (
                            f"📚 This session's archived ideas "
                            f"({len(ideas)} entries — weak claim, fully "
                            f"transparent)"
                        ),
                        "seed_paste": (
                            "📋 Paste seed texts (one per line — "
                            "'Title. Abstract')"
                        ),
                    }[k],
                    key="_ca_corpus_source",
                    horizontal=False,
                )

                _ca_corpus = None
                if _ca_source == "archive":
                    _ca_corpus = ReferenceCorpus.from_archive(ideas)
                    st.caption(
                        f"Corpus loaded from this session's archive: "
                        f"**{len(_ca_corpus)} entries**."
                    )
                else:
                    _seed_input = st.text_area(
                        "Seed entries (one per line: 'Title. Abstract' "
                        "or just 'Title')",
                        height=140,
                        key="_ca_seed_input",
                        placeholder=(
                            "Self-attention is all you need. Transformer "
                            "models using only attention.\n"
                            "BERT pretraining. Masked language modeling on "
                            "large text corpora.\n"
                            "..."
                        ),
                    )
                    _seed_lines = [
                        ln for ln in (_seed_input or "").splitlines()
                        if ln.strip()
                    ]
                    if _seed_lines:
                        _ca_corpus = ReferenceCorpus.from_seed_texts(
                            _seed_lines, source="seed_paste",
                        )
                        st.caption(
                            f"Corpus loaded from pasted text: "
                            f"**{len(_ca_corpus)} entries**."
                        )

                # ── Generator picker ────────────────────────────────────
                _ca_gens = st.multiselect(
                    "Generators (each contributes candidates to score)",
                    options=[
                        "persona", "analogy", "heretic",
                        "contradiction", "future_back", "counterfactual",
                        "extremum", "inversion", "underserved_cohort",
                    ],
                    default=list(DEFAULT_GENERATORS),
                    format_func=lambda k: k.replace("_", " ").title(),
                    key="_ca_generators",
                    help="The design manifesto names persona, analogy, "
                          "and heretic as the canonical varied generators.",
                )

                _gc1, _gc2 = st.columns(2)
                _ca_n_per = _gc1.slider(
                    "Candidates per generator",
                    min_value=1, max_value=5, value=2, step=1,
                    key="_ca_n_per",
                )
                _ca_keep = _gc2.slider(
                    "Keep top N (after scoring)",
                    min_value=1, max_value=10, value=5, step=1,
                    key="_ca_keep",
                )

                _tc1, _tc2 = st.columns(2)
                _ca_min_nov = _tc1.slider(
                    "Min novelty floor (0=keep all)",
                    min_value=0.0, max_value=0.95, value=0.0, step=0.05,
                    key="_ca_min_nov",
                )
                _ca_map_elites = _tc2.toggle(
                    "MAP-Elites cell dedup",
                    value=True,
                    key="_ca_map_elites",
                    help="Keep at most one idea per "
                          "methodology × novelty cell (most-novel wins).",
                )

                _ca_disabled = (
                    not _ca_topic.strip()
                    or not _ca_gens
                    or _ca_corpus is None
                    or len(_ca_corpus) == 0
                )
                if st.button(
                    "🛰️ Generate + score against corpus",
                    type="primary", use_container_width=True,
                    disabled=_ca_disabled,
                    key="ca_go_btn",
                ):
                    with st.spinner(
                        f"Running {len(_ca_gens)} generator(s) × "
                        f"{_ca_n_per} candidates + scoring against "
                        f"{len(_ca_corpus)}-entry corpus…"
                    ):
                        _ca_ideas = corpus_anchored_batch(
                            _ca_topic.strip(),
                            _ca_corpus,
                            n_per_generator=int(_ca_n_per),
                            generators=list(_ca_gens),
                            keep_top=int(_ca_keep),
                            min_novelty=float(_ca_min_nov),
                            map_elites=bool(_ca_map_elites),
                        )
                    st.session_state["_ca_ideas"] = [
                        i.to_dict() for i in _ca_ideas
                    ]

                _ca_state = st.session_state.get("_ca_ideas") or []
                if _ca_state:
                    st.markdown("---")
                    st.markdown(
                        f"### {len(_ca_state)} corpus-anchored idea(s) "
                        f"— sorted by novelty"
                    )
                    for _i, _d in enumerate(_ca_state, 1):
                        _meta = _d.get("execution_meta") or {}
                        _nov = _meta.get("corpus_novelty") or {}
                        _score = float(_nov.get("score", 0.0))
                        _sim = float(_nov.get("nearest_similarity", 0.0))
                        _upstream = _meta.get("upstream_strategy", "?")
                        # Color band by novelty score.
                        if _score >= 0.7:
                            _band = "#10b981"
                        elif _score >= 0.4:
                            _band = "#f59e0b"
                        else:
                            _band = "#ef4444"
                        with st.expander(
                            f"🛰️ #{_i}. [novelty {_score:.2f}] "
                            f"{_d.get('title','Untitled')}",
                            expanded=(_i == 1),
                        ):
                            st.code(_d.get("title", "Untitled"),
                                    language=None)
                            st.markdown(
                                f"<div style='background:#f8fafc;"
                                f"border-left:4px solid {_band};"
                                f"padding:10px 14px;border-radius:6px;"
                                f"font-size:13px;color:#0f172a;"
                                f"margin-bottom:10px'>"
                                f"<b>Novelty score:</b> {_score:.3f} "
                                f"(1 − cosine to nearest)<br>"
                                f"<b>Nearest similarity:</b> {_sim:.3f}<br>"
                                f"<b>Nearest in corpus:</b> "
                                f"<i>{_nov.get('nearest_title','—')}</i> "
                                f"<span style='color:#64748b;font-size:11px'>"
                                f"[{_nov.get('nearest_source','?')}"
                                f"{', ' + str(_nov.get('nearest_year')) if _nov.get('nearest_year') else ''}]"
                                f"</span><br>"
                                f"<b>Excerpt:</b> "
                                f"<span style='color:#475569'>"
                                f"{_nov.get('nearest_excerpt','—')}</span><br>"
                                f"<b>Corpus size:</b> "
                                f"{_nov.get('corpus_size',0)}<br>"
                                f"<b>Upstream generator:</b> "
                                f"strategy <code>{_upstream}</code>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                            st.markdown(f"**Motivation.** {_d.get('motivation','')}")
                            st.markdown(f"**Method.** {_d.get('method','')}")
                            st.markdown(f"**Hypothesis.** {_d.get('hypothesis','')}")
                            st.caption(
                                f"⚠️ {_meta.get('corpus_limits', '')[:200]}…"
                            )

                    if st.button(
                        "➕ Add all corpus-anchored ideas to session",
                        use_container_width=True,
                        key="ca_add_btn",
                    ):
                        ideas.extend(_ca_state)
                        st.session_state["_ca_ideas"] = []
                        st.success(f"Added {len(_ca_state)} idea(s).")
                        st.rerun()

    # ── Visual Simulation Tab ──────────────────────────────────────────────
    with tab_simulate:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">'
            '<span style="font-size:22px">🎬</span>'
            '<span style="font-size:18px;font-weight:700;color:#0c4a6e">'
            'Visual Simulation</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.caption("See what executing this idea would look like — method flow, expected outcomes, timeline, and resource cost.")

        if not ideas:
            st.info("Run a pipeline first to simulate ideas.")
        else:
            _sim_sorted = sorted(ideas, key=lambda x: x.get("quality_score", 0), reverse=True)
            _sim_labels = [
                f"{i.get('title','?')} (q={i.get('quality_score',0):.2f})"
                for i in _sim_sorted
            ]
            _sim_idx = st.selectbox(
                "Pick an idea to simulate:",
                range(len(_sim_labels)),
                format_func=lambda i: _sim_labels[i],
                key="simulate_idea_select",
            )
            _sim_idea = _sim_sorted[_sim_idx]

            if st.button("🎬 Run Visual Simulation",
                          type="primary", use_container_width=True,
                          key="run_sim_btn"):
                st.session_state["_run_sim"] = True
                st.session_state["_sim_target_title"] = _sim_idea.get("title", "")

            # ── 🎥 Veo animated explainer ────────────────────────────
            # Generate a short MP4 clip showing the method or expected
            # results in motion. Uses Veo 3 via the long-running
            # operation protocol. Requires paid AI Studio credit.
            try:
                from ideagraph_image_renderer import (
                    VeoVideoProvider, build_video_prompt,
                    VEO_VIDEO_MODELS,
                )
                _veo_ok = True
            except ImportError:
                _veo_ok = False

            if _veo_ok:
                # Pull the catalog from the renderer module so adding
                # a new VEO_ANIMATION_STYLES entry surfaces here
                # automatically.
                try:
                    from ideagraph_image_renderer import (
                        VEO_ANIMATION_STYLES as _VEO_STYLES,
                    )
                except ImportError:
                    _VEO_STYLES = {}

                with st.expander(
                    "🎥 Animated explainer (Veo) — generate a short "
                    "MP4 clip",
                    expanded=False,
                ):
                    st.caption(
                        "Render a 5-8s animated video for this idea. "
                        "Uses Google Veo (paid AI Studio credit "
                        "required). Veo can take 30-120s end-to-end."
                    )
                    _veo_key = (
                        f"_sim_veo_"
                        f"{hash(_sim_idea.get('title','')) & 0xffff:04x}"
                    )
                    _style_keys = list(_VEO_STYLES.keys()) or [
                        "method_animation", "result_reveal",
                    ]
                    _veo_style = st.radio(
                        "Animation style",
                        options=_style_keys,
                        format_func=lambda k: (
                            f"{_VEO_STYLES[k]['label']} — "
                            f"{_VEO_STYLES[k]['description']}"
                            if k in _VEO_STYLES else k
                        ),
                        key=f"{_veo_key}_style",
                        horizontal=False,
                    )
                    _veo_model = st.selectbox(
                        "Veo model",
                        options=VEO_VIDEO_MODELS,
                        index=0,
                        key=f"{_veo_key}_model",
                        help="Veo 3 (default) is highest quality. "
                              "Veo 2 is older and may be slightly "
                              "cheaper. Custom names not yet "
                              "supported here — use the API directly.",
                    )
                    _veo_duration = st.slider(
                        "Duration (seconds)", 4, 8, 6, 1,
                        key=f"{_veo_key}_duration",
                    )
                    _veo_aspect = st.selectbox(
                        "Aspect ratio",
                        options=["16:9", "9:16", "1:1"],
                        index=0,
                        key=f"{_veo_key}_aspect",
                    )

                    if st.button(
                        "🎥 Generate animated explainer (Veo)",
                        type="primary",
                        use_container_width=True,
                        key=f"{_veo_key}_go",
                    ):
                        try:
                            import config as _cfg_veo
                            _veo_provider = VeoVideoProvider(
                                api_key=getattr(_cfg_veo,
                                                  "NANO_BANANA_API_KEY",
                                                  ""),
                                model=_veo_model,
                                endpoint=getattr(
                                    _cfg_veo, "NANO_BANANA_ENDPOINT",
                                    "https://generativelanguage.googleapis.com/v1beta",
                                ) or "https://generativelanguage.googleapis.com/v1beta",
                                duration_s=int(_veo_duration),
                                aspect_ratio=_veo_aspect,
                            )
                            _veo_prompt = build_video_prompt(
                                _sim_idea, style=_veo_style,
                            )
                            with st.expander(
                                "Prompt being sent", expanded=False,
                            ):
                                st.code(_veo_prompt, language=None)
                            with st.spinner(
                                f"Calling Veo ({_veo_model})… this "
                                "takes 30-120s. Don't refresh."
                            ):
                                _veo_result = (
                                    _veo_provider.generate_video(_veo_prompt)
                                )
                            st.session_state[_veo_key] = _veo_result
                        except Exception as _vee:
                            st.session_state[_veo_key] = {
                                "error": f"{type(_vee).__name__}: {_vee}",
                            }

                    _veo_state = st.session_state.get(_veo_key)
                    if _veo_state:
                        if "error" in _veo_state:
                            st.error(
                                f"❌ Veo generation failed: "
                                f"{_veo_state['error']}\n\n"
                                "Common fixes:\n"
                                "1. Veo requires **paid Google AI "
                                "Studio credit** (https://aistudio"
                                ".google.com/usage).\n"
                                "2. If you get 404, try a different "
                                "Veo model from the dropdown above.\n"
                                "3. Veo may be region-gated — check "
                                "https://ai.google.dev/gemini-api/docs/"
                                "video for availability."
                            )
                        elif _veo_state.get("video_url"):
                            st.success("✅ Video generated.")
                            try:
                                st.video(_veo_state["video_url"])
                            except Exception as _ve:
                                st.warning(
                                    f"Couldn't display video: {_ve}. "
                                    f"URL: {_veo_state['video_url']}"
                                )
                            # 📥 Download — fetches the .mp4 bytes from
                            # the URL on click. Veo videos can be 10-50
                            # MB; cap timeout for slow connections.
                            try:
                                from ideagraph_image_renderer import (
                                    safe_filename as _safe_fn_v,
                                )
                                import requests as _req_v
                                if st.button(
                                    "📥 Fetch & save this video",
                                    use_container_width=True,
                                    key=f"{_veo_key}_fetch_btn",
                                    help="Downloads the .mp4 from "
                                          "Google's CDN and offers it "
                                          "with a meaningful filename.",
                                ):
                                    try:
                                        with st.spinner(
                                            "Fetching video bytes from "
                                            "Google's CDN…"
                                        ):
                                            _r = _req_v.get(
                                                _veo_state["video_url"],
                                                timeout=120,
                                            )
                                        if _r.status_code == 200:
                                            st.session_state[
                                                f"{_veo_key}_dlbytes"
                                            ] = _r.content
                                        else:
                                            st.error(
                                                f"Fetch failed: HTTP "
                                                f"{_r.status_code}"
                                            )
                                    except Exception as _fe:
                                        st.error(f"Fetch error: {_fe}")
                                _veo_dl_bytes = st.session_state.get(
                                    f"{_veo_key}_dlbytes"
                                )
                                if _veo_dl_bytes:
                                    _veo_dl_name = _safe_fn_v(
                                        _sim_idea.get("title", "video"),
                                        style=_veo_style,
                                        media_type="video",
                                    )
                                    st.download_button(
                                        "💾 Save .mp4 to your computer",
                                        data=_veo_dl_bytes,
                                        file_name=_veo_dl_name,
                                        mime="video/mp4",
                                        use_container_width=True,
                                        key=f"{_veo_key}_dlbtn",
                                    )
                            except ImportError:
                                pass
                        elif _veo_state.get("video_bytes"):
                            st.success("✅ Video generated (inline).")
                            try:
                                st.video(
                                    _veo_state["video_bytes"],
                                    format=_veo_state.get(
                                        "mime_type", "video/mp4",
                                    ),
                                )
                            except Exception as _ve:
                                st.warning(
                                    f"Couldn't display video: {_ve}"
                                )
                            # 📥 Inline bytes path — direct download.
                            try:
                                from ideagraph_image_renderer import (
                                    safe_filename as _safe_fn_vb,
                                )
                                _veo_dl_name = _safe_fn_vb(
                                    _sim_idea.get("title", "video"),
                                    style=_veo_style,
                                    media_type="video",
                                )
                                st.download_button(
                                    "💾 Save .mp4 to your computer",
                                    data=_veo_state["video_bytes"],
                                    file_name=_veo_dl_name,
                                    mime=_veo_state.get(
                                        "mime_type", "video/mp4",
                                    ),
                                    use_container_width=True,
                                    key=f"{_veo_key}_dlbtn_inline",
                                )
                            except ImportError:
                                pass

            if st.session_state.get("_run_sim") and \
               st.session_state.get("_sim_target_title") == _sim_idea.get("title", ""):
                try:
                    from idea_simulator import run_simulation
                    with st.spinner("🎬 Simulating..."):
                        sim = run_simulation(_sim_idea)

                    # Header summary
                    _stats = sim["outcome_stats"]
                    _res = sim["resource_stats"]
                    _hc1, _hc2, _hc3, _hc4 = st.columns(4)
                    _hc1.metric("📈 Median Outcome", f"{_stats['p50']:.1f}%")
                    _hc2.metric("✅ Success Rate", f"{_stats['success_pct']:.0f}%")
                    _hc3.metric("💰 Est. Cost", f"${_res['cost_usd']:.0f}")
                    _hc4.metric("⏱️ Duration", f"{_res['time_weeks']}w")

                    st.markdown("---")

                    # ═══════════════════════════════════════════════════════
                    # 🎥 IDEA-TO-VIDEO  (NotebookLM-style narrated pitch)
                    # ═══════════════════════════════════════════════════════
                    with st.expander("🎥 Convert idea to video (narrated pitch)", expanded=False):
                        try:
                            from idea_video import (
                                VIDEO_STYLES,
                                generate_video_script,
                                build_video_embed,
                                build_video_html,
                                estimate_duration_s,
                            )
                            import streamlit.components.v1 as _components

                            _style_keys = list(VIDEO_STYLES.keys())
                            _style = st.selectbox(
                                "🎨 Video style",
                                options=_style_keys,
                                format_func=lambda k: (
                                    f"{VIDEO_STYLES[k]['label']}  —  "
                                    f"{VIDEO_STYLES[k]['description']}"
                                ),
                                index=0,
                                key=f"vid_style_{_sim_idx}",
                                help="Pick the tone and visual treatment. Each style "
                                     "rewrites the script and changes the look.",
                            )

                            _slides = generate_video_script(_sim_idea, style=_style)
                            _vid_total = estimate_duration_s(_slides)
                            _style_cfg = VIDEO_STYLES[_style]

                            _vc1, _vc2, _vc3, _vc4 = st.columns(4)
                            _vc1.metric("🎞️ Slides", len(_slides))
                            _vc2.metric("⏱️ Duration",
                                        f"{_vid_total // 60}:{_vid_total % 60:02d}")
                            _vc3.metric("⚡ Default Rate",
                                        f"{_style_cfg['default_rate']:.2f}×")
                            _vc4.metric("🎙️ Voice",
                                        "Web Speech")

                            _vid_embed = build_video_embed(
                                _slides, _sim_idea, autoplay=False, style=_style,
                            )
                            _components.html(_vid_embed, height=820, scrolling=False)

                            st.caption(
                                f"**{_style_cfg['label']}** · "
                                f"{_style_cfg['description']}. "
                                "Click **▶** to play. **🧑 Human** mode (default ON) "
                                "applies the full naturalization stack: conversational "
                                "rewrites, slow→crisp→slow rate arc, **emotion prosody** "
                                "(questions rise, exclamations peak, concerns drop, "
                                "curiosity lifts), **drawn-out fillers** (um/uh/so/well "
                                "spoken slower at lower pitch), Web-Audio **breath** + "
                                "**lip-smack** transients between thoughts, and **music "
                                "ducking** so the synthesized pad sits politely under "
                                "the voice. Toggle off for the original robot read-aloud. "
                                "**👥 Duo** adds Female + Male voice pairing with "
                                "**Mm-hmm./Right./Yeah.** backchannels from the listening "
                                "host. **🎵 Music** turns on per-style ambient. Plus "
                                "reaction emojis, chapter markers, particle backdrops, "
                                "and confetti finale. 🎉"
                            )

                            _vid_html = build_video_html(
                                _slides, _sim_idea, autoplay=True, style=_style,
                            )
                            _safe_name = (
                                "".join(c if c.isalnum() else "_"
                                        for c in _sim_idea.get("title", "idea"))[:60]
                                or "idea"
                            )
                            st.download_button(
                                f"📥 Download {_style_cfg['label']} as standalone HTML",
                                data=_vid_html,
                                file_name=f"{_safe_name}_{_style}_video.html",
                                mime="text/html",
                                key=f"vid_dl_{_sim_idx}_{_style}",
                                use_container_width=True,
                            )

                            with st.expander("📜 View narration script", expanded=False):
                                for _i, _s in enumerate(_slides, 1):
                                    st.markdown(
                                        f"**{_i}. {_s['icon']} {_s['title']}** "
                                        f"_({_s['duration_s']}s)_  \n"
                                        f"{_s['narration']}"
                                    )
                        except ImportError as e:
                            st.caption(f"Video module not available: {e}")
                        except Exception as e:
                            st.caption(f"Video render error: {e}")

                    st.markdown("---")

                    # Method flow
                    if sim.get("method_flow"):
                        st.plotly_chart(sim["method_flow"], use_container_width=True,
                                        key=f"sim_flow_{_sim_idx}")
                        st.caption(f"Detected **{len(sim['stages'])} stages** in the method.")

                    # Outcome distribution + Timeline side-by-side
                    _row1c1, _row1c2 = st.columns(2)
                    with _row1c1:
                        if sim.get("outcome_dist"):
                            st.plotly_chart(sim["outcome_dist"],
                                            use_container_width=True,
                                            key=f"sim_outcome_{_sim_idx}")
                            st.caption(
                                f"Mean **{_stats['mean']:.1f}±{_stats['std']:.1f}%** | "
                                f"P10 **{_stats['p10']:.1f}%**, P90 **{_stats['p90']:.1f}%**"
                            )
                    with _row1c2:
                        if sim.get("timeline"):
                            st.plotly_chart(sim["timeline"],
                                            use_container_width=True,
                                            key=f"sim_timeline_{_sim_idx}")

                    # Resource gauges
                    if sim.get("resources"):
                        st.plotly_chart(sim["resources"], use_container_width=True,
                                        key=f"sim_resources_{_sim_idx}")

                    # Plain English narrative
                    _success_label = (
                        "very likely succeed" if _stats["success_pct"] > 80 else
                        "probably succeed" if _stats["success_pct"] > 60 else
                        "have mixed results" if _stats["success_pct"] > 40 else
                        "be high-risk"
                    )
                    st.markdown(
                        f'<div style="background:#f0f9ff;border:1px solid #bae6fd;'
                        f'border-radius:10px;padding:14px 18px;margin-top:12px">'
                        f'<div style="font-size:13px;font-weight:700;color:#0369a1;'
                        f'margin-bottom:6px">📋 Simulation Summary</div>'
                        f'<div style="font-size:13px;color:#0c4a6e;line-height:1.6">'
                        f'Based on {_stats["n_trials"]} Monte Carlo trials, this idea would '
                        f'<b>{_success_label}</b> with a median performance of '
                        f'<b>{_stats["p50"]:.1f}%</b> ({_stats["p10"]:.1f}%–{_stats["p90"]:.1f}% range). '
                        f'Estimated cost <b>${_res["cost_usd"]:.0f}</b> over '
                        f'<b>{_res["time_weeks"]} weeks</b> using '
                        f'<b>{_res["gpu_hours"]} GPU-hours</b> and '
                        f'<b>{_res["data_gb"]:.1f}GB</b> of data.'
                        f'</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    # ═══════════════════════════════════════════════════════
                    # ADVANCED ANALYTICS (5 new visualizations)
                    # ═══════════════════════════════════════════════════════
                    st.markdown("---")
                    st.markdown(
                        '<div style="font-size:14px;font-weight:700;color:#0c4a6e;'
                        'margin:6px 0">🔬 Advanced Analytics</div>',
                        unsafe_allow_html=True,
                    )

                    # ── 1. What-If Sliders ──────────────────────────────
                    with st.expander("🎚️ What-If Analysis (interactive sliders)"):
                        from idea_simulator import build_what_if_chart
                        _wi_c1, _wi_c2, _wi_c3 = st.columns(3)
                        with _wi_c1:
                            _whatif_compute = st.select_slider(
                                "Compute",
                                options=[0.25, 0.5, 1.0, 2.0, 4.0],
                                value=1.0,
                                format_func=lambda x: f"{x}× baseline",
                                key=f"wi_compute_{_sim_idx}",
                            )
                        with _wi_c2:
                            _whatif_data = st.slider(
                                "Data Quality", -0.3, 0.3, 0.0, 0.05,
                                help="Negative = worse data; positive = better data",
                                key=f"wi_data_{_sim_idx}",
                            )
                        with _wi_c3:
                            _whatif_novelty = st.slider(
                                "Novelty Bet", -0.3, 0.3, 0.0, 0.05,
                                help="Higher = riskier, wider variance",
                                key=f"wi_novelty_{_sim_idx}",
                            )

                        try:
                            _wi_result = build_what_if_chart(
                                _sim_idea,
                                {
                                    "compute_multiplier": _whatif_compute,
                                    "data_quality_boost": _whatif_data,
                                    "novelty_bet": _whatif_novelty,
                                },
                            )
                            if _wi_result:
                                _wi_fig, _wi_data_obj = _wi_result
                                st.plotly_chart(_wi_fig, use_container_width=True,
                                                key=f"sim_whatif_{_sim_idx}")
                                _delta = _wi_data_obj["delta_p50"]
                                _delta_color = "#10b981" if _delta > 0 else "#ef4444" if _delta < 0 else "#94a3b8"
                                st.markdown(
                                    f'<div style="background:#f0f9ff;border-left:4px solid {_delta_color};'
                                    f'padding:8px 14px;margin-top:8px;border-radius:6px">'
                                    f'<b style="color:{_delta_color}">Net effect: {_delta:+.1f}% on median</b> '
                                    f'<span style="color:#64748b;font-size:12px">'
                                    f'(P10 {_wi_data_obj["adjusted"]["p10"]:.0f}% / '
                                    f'P90 {_wi_data_obj["adjusted"]["p90"]:.0f}%)</span>'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )
                        except Exception as e:
                            st.caption(f"What-If error: {e}")

                    # ── 2. Sensitivity Tornado ──────────────────────────
                    with st.expander("🌀 Sensitivity Analysis (which factor matters most)"):
                        try:
                            from idea_simulator import build_sensitivity_tornado, compute_sensitivity
                            _tornado = build_sensitivity_tornado(_sim_idea)
                            if _tornado:
                                st.plotly_chart(_tornado, use_container_width=True,
                                                key=f"sim_tornado_{_sim_idx}")
                            _sens = compute_sensitivity(_sim_idea)
                            if _sens:
                                _top = _sens[0]
                                st.caption(
                                    f"**Most impactful factor:** {_top['factor']} "
                                    f"(swings outcome by {_top['range']:.1f}%)"
                                )
                        except Exception as e:
                            st.caption(f"Sensitivity error: {e}")

                    # ── 3. Risk Waterfall ───────────────────────────────
                    with st.expander("⛓️ Risk Waterfall (FMEA → success rate)"):
                        try:
                            from idea_simulator import build_risk_waterfall
                            _waterfall = build_risk_waterfall(_sim_idea)
                            if _waterfall:
                                st.plotly_chart(_waterfall, use_container_width=True,
                                                key=f"sim_waterfall_{_sim_idx}")
                                st.caption(
                                    "Each red bar shows how much one failure mode "
                                    "(from FMEA analysis) reduces the projected success rate."
                                )
                        except Exception as e:
                            st.caption(f"Waterfall error: {e}")

                    # ── 4. Multi-Idea Comparison ────────────────────────
                    with st.expander("📊 Compare With Other Ideas (overlay 2-5)"):
                        _multi_options = [
                            f"{i.get('title','?')} (q={i.get('quality_score',0):.2f})"
                            for i in _sim_sorted
                        ]
                        _multi_default = [_sim_idx]
                        if len(_sim_sorted) >= 2:
                            _multi_default.append(0 if _sim_idx != 0 else 1)
                        _multi_picked = st.multiselect(
                            "Pick up to 5 ideas to overlay",
                            options=list(range(len(_sim_sorted))),
                            default=_multi_default,
                            format_func=lambda i: _multi_options[i],
                            max_selections=5,
                            key=f"sim_multi_{_sim_idx}",
                        )
                        if len(_multi_picked) >= 2:
                            try:
                                from idea_simulator import build_multi_outcome_overlay
                                _multi_ideas = [_sim_sorted[i] for i in _multi_picked]
                                _multi = build_multi_outcome_overlay(_multi_ideas)
                                if _multi:
                                    _multi_fig, _multi_summaries = _multi
                                    st.plotly_chart(_multi_fig, use_container_width=True,
                                                    key=f"sim_multi_chart_{_sim_idx}")
                                    # Mini stats table
                                    _stats_html = (
                                        '<table style="width:100%;border-collapse:collapse;'
                                        'font-size:12px;margin-top:8px">'
                                        '<tr style="background:#f0f9ff;color:#0369a1">'
                                        '<th style="padding:6px;text-align:left">Idea</th>'
                                        '<th style="padding:6px">P10</th>'
                                        '<th style="padding:6px">P50</th>'
                                        '<th style="padding:6px">P90</th>'
                                        '<th style="padding:6px">Success%</th></tr>'
                                    )
                                    for s in _multi_summaries:
                                        _stats_html += (
                                            f'<tr><td style="padding:6px;border-top:1px solid #e0f2fe">'
                                            f'<span style="display:inline-block;width:10px;height:10px;'
                                            f'border-radius:50%;background:{s["color"]};'
                                            f'margin-right:6px"></span>{s["title"]}</td>'
                                            f'<td style="padding:6px;text-align:center;'
                                            f'border-top:1px solid #e0f2fe">{s["p10"]:.0f}%</td>'
                                            f'<td style="padding:6px;text-align:center;'
                                            f'border-top:1px solid #e0f2fe;font-weight:700">'
                                            f'{s["p50"]:.0f}%</td>'
                                            f'<td style="padding:6px;text-align:center;'
                                            f'border-top:1px solid #e0f2fe">{s["p90"]:.0f}%</td>'
                                            f'<td style="padding:6px;text-align:center;'
                                            f'border-top:1px solid #e0f2fe">{s["success_pct"]:.0f}%</td></tr>'
                                        )
                                    _stats_html += '</table>'
                                    st.markdown(_stats_html, unsafe_allow_html=True)
                            except Exception as e:
                                st.caption(f"Multi-overlay error: {e}")

                    # ── 5. Pareto Frontier ──────────────────────────────
                    with st.expander("⭐ Pareto Frontier (cost vs quality across all ideas)"):
                        try:
                            from idea_simulator import (
                                build_pareto_scatter, compute_pareto_frontier,
                            )
                            _pareto = build_pareto_scatter(ideas)
                            if _pareto:
                                st.plotly_chart(_pareto, use_container_width=True,
                                                key=f"sim_pareto_{_sim_idx}")
                                _opt = compute_pareto_frontier(ideas)
                                _opt_titles = [
                                    ideas[i].get("title", "?")
                                    for i, o in enumerate(_opt) if o
                                ]
                                if _opt_titles:
                                    st.caption(
                                        f"⭐ **Pareto-optimal ideas:** "
                                        f"{', '.join(_opt_titles[:5])}"
                                    )
                        except Exception as e:
                            st.caption(f"Pareto error: {e}")

                    # ═══════════════════════════════════════════════════════
                    # PRO ANALYTICS (5 more visualizations)
                    # ═══════════════════════════════════════════════════════
                    st.markdown("---")
                    st.markdown(
                        '<div style="font-size:14px;font-weight:700;color:#0c4a6e;'
                        'margin:6px 0">🎬 Pro Analytics</div>',
                        unsafe_allow_html=True,
                    )

                    # ── 1. Animated Execution Playback ──────────────────
                    with st.expander("▶️ Execution Replay (CI/CD-style stage progress)"):
                        try:
                            from idea_simulator import build_execution_playback
                            _playback = build_execution_playback(_sim_idea)
                            if _playback:
                                st.plotly_chart(_playback, use_container_width=True,
                                                key=f"sim_playback_{_sim_idx}")
                                st.caption("Click ▶ Play to watch each stage 'complete' over time.")
                        except Exception as e:
                            st.caption(f"Playback error: {e}")

                    # ── 2. Confidence Cone ──────────────────────────────
                    with st.expander("🌡️ Confidence Cone (uncertainty narrows over time)"):
                        try:
                            from idea_simulator import build_confidence_cone
                            _cone = build_confidence_cone(_sim_idea)
                            if _cone:
                                st.plotly_chart(_cone, use_container_width=True,
                                                key=f"sim_cone_{_sim_idx}")
                                st.caption(
                                    "At week 0, predictions have wide spread (we know little). "
                                    "As the project progresses, the P10–P90 band narrows toward the final estimate."
                                )
                        except Exception as e:
                            st.caption(f"Cone error: {e}")

                    # ── 3. Budget Burn-Down ─────────────────────────────
                    with st.expander("💰 Budget Burn-Down (cumulative spend over weeks)"):
                        try:
                            from idea_simulator import build_budget_burndown
                            _burn = build_budget_burndown(_sim_idea)
                            if _burn:
                                st.plotly_chart(_burn, use_container_width=True,
                                                key=f"sim_burn_{_sim_idx}")
                                st.caption(
                                    "GPU-hours and USD spend accumulate week-by-week. "
                                    "Heaviest usage during method implementation + experiments."
                                )
                        except Exception as e:
                            st.caption(f"Burn-down error: {e}")

                    # ── 4. 3D Idea Space ────────────────────────────────
                    with st.expander("🌐 3D Idea Space (cost × quality × novelty)"):
                        try:
                            from idea_simulator import build_3d_idea_space
                            # Find this idea's index in the unsorted ideas list
                            _hl_idx = -1
                            for _i, _it in enumerate(ideas):
                                if _it.get("title") == _sim_idea.get("title"):
                                    _hl_idx = _i
                                    break
                            _scatter3d = build_3d_idea_space(ideas, highlight_idx=_hl_idx)
                            if _scatter3d:
                                st.plotly_chart(_scatter3d, use_container_width=True,
                                                key=f"sim_3d_{_sim_idx}")
                                st.caption(
                                    "Drag to rotate. The selected idea is highlighted "
                                    "as a large amber diamond."
                                )
                        except Exception as e:
                            st.caption(f"3D scatter error: {e}")

                    # ── 5. Probe Score Sunburst ─────────────────────────
                    with st.expander("🎯 Probe Score Sunburst (10-D radial breakdown)"):
                        try:
                            from idea_simulator import build_probe_sunburst
                            _sun = build_probe_sunburst(_sim_idea)
                            if _sun:
                                st.plotly_chart(_sun, use_container_width=True,
                                                key=f"sim_sun_{_sim_idx}")
                                st.caption(
                                    "Inner ring: 3 probe categories. Outer ring: individual scores. "
                                    "Green = strong, amber = mid, red = weak."
                                )
                            else:
                                st.info("This idea has no probe scores.")
                        except Exception as e:
                            st.caption(f"Sunburst error: {e}")

                    # ═══════════════════════════════════════════════════════
                    # ELITE ANALYTICS (5 final visualizations)
                    # ═══════════════════════════════════════════════════════
                    st.markdown("---")
                    st.markdown(
                        '<div style="font-size:14px;font-weight:700;color:#0c4a6e;'
                        'margin:6px 0">💎 Elite Analytics</div>',
                        unsafe_allow_html=True,
                    )

                    # ── 1. Carbon Footprint ─────────────────────────────
                    with st.expander("🌍 Carbon Footprint (CO₂, energy, equivalents)"):
                        try:
                            from idea_simulator import build_carbon_footprint, estimate_carbon
                            _carbon_fig = build_carbon_footprint(_sim_idea)
                            if _carbon_fig:
                                st.plotly_chart(_carbon_fig, use_container_width=True,
                                                key=f"sim_carbon_{_sim_idx}")
                            _c = estimate_carbon(_sim_idea)
                            st.caption(
                                f"Estimated **{_c['kg_co2']:.1f} kg CO₂** from "
                                f"**{_c['gpu_hours']} GPU-hours** "
                                f"(equivalent to **{_c['miles_driven_eq']:.0f} miles** driven, "
                                f"or what **{_c['trees_year_eq']:.1f} trees** absorb in 1 year)."
                            )
                        except Exception as e:
                            st.caption(f"Carbon error: {e}")

                    # ── 2. Citation Forecast ────────────────────────────
                    with st.expander("📈 Citation Forecast (5-year projection)"):
                        try:
                            from idea_simulator import build_citation_forecast, forecast_citations
                            _cite_fig = build_citation_forecast(_sim_idea)
                            if _cite_fig:
                                st.plotly_chart(_cite_fig, use_container_width=True,
                                                key=f"sim_cite_{_sim_idx}")
                            _f = forecast_citations(_sim_idea)
                            st.caption(
                                f"Projected **{_f['asymptote']:.0f} citations** total over 5 years "
                                f"(**{_f['quality_tier']}** tier). "
                                f"Estimated h-index contribution: **+{_f['h_index_contrib']}**."
                            )
                        except Exception as e:
                            st.caption(f"Citation error: {e}")

                    # ── 3. Idea Similarity Network ──────────────────────
                    with st.expander("🕸️ Idea Similarity Network (graph of all ideas)"):
                        try:
                            from idea_simulator import build_similarity_network
                            _net_hl = -1
                            for _i, _it in enumerate(ideas):
                                if _it.get("title") == _sim_idea.get("title"):
                                    _net_hl = _i
                                    break
                            _net = build_similarity_network(ideas, highlight_idx=_net_hl)
                            if _net:
                                st.plotly_chart(_net, use_container_width=True,
                                                key=f"sim_net_{_sim_idx}")
                                st.caption(
                                    "Each circle is an idea (sized by quality). "
                                    "Connected pairs share method-level vocabulary. "
                                    "Selected idea highlighted in amber."
                                )
                            else:
                                st.info("Need at least 2 ideas for network view.")
                        except Exception as e:
                            st.caption(f"Network error: {e}")

                    # ── 4. Success Funnel ───────────────────────────────
                    with st.expander("🎯 Success Funnel (P-success at each stage)"):
                        try:
                            from idea_simulator import build_success_funnel
                            _funnel = build_success_funnel(_sim_idea)
                            if _funnel:
                                st.plotly_chart(_funnel, use_container_width=True,
                                                key=f"sim_funnel_{_sim_idx}")
                                st.caption(
                                    "Each stage compounds: data → method → results → "
                                    "significance → publication. The bottom number is "
                                    "P(paper accepted) given all stages succeed."
                                )
                        except Exception as e:
                            st.caption(f"Funnel error: {e}")

                    # ── 5. Stage Criticality ────────────────────────────
                    with st.expander("🎚️ Stage Criticality (which failure hurts most?)"):
                        try:
                            from idea_simulator import build_stage_criticality
                            _crit = build_stage_criticality(_sim_idea)
                            if _crit:
                                st.plotly_chart(_crit, use_container_width=True,
                                                key=f"sim_crit_{_sim_idx}")
                                st.caption(
                                    "If you had to pick one stage to focus de-risking effort on, "
                                    "pick the top bar — that's the one whose failure would hurt "
                                    "your final success probability the most."
                                )
                        except Exception as e:
                            st.caption(f"Criticality error: {e}")

                    # ═══════════════════════════════════════════════════════
                    # NOVELTY ANALYTICS (5 truly novel visualizations)
                    # ═══════════════════════════════════════════════════════
                    st.markdown("---")
                    st.markdown(
                        '<div style="font-size:14px;font-weight:700;'
                        'background:linear-gradient(90deg,#a855f7,#0ea5e9);'
                        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;'
                        'margin:6px 0">✨ Novelty Analytics</div>',
                        unsafe_allow_html=True,
                    )

                    # ── 1. Novelty Constellation ────────────────────────
                    with st.expander("🌟 Novelty Constellation (where your idea lives in concept space)"):
                        try:
                            from idea_simulator import build_novelty_constellation
                            _other_ideas = [i for i in ideas if i.get("title") != _sim_idea.get("title")]
                            _dag_papers = (results.get("dag_summary") or {}).get("papers", [])
                            _const = build_novelty_constellation(
                                _sim_idea, _other_ideas, _dag_papers,
                            )
                            if _const:
                                st.plotly_chart(_const, use_container_width=True,
                                                key=f"sim_const_{_sim_idx}")
                                st.caption(
                                    "Each blue dot is a related paper. Other ideas are circles. "
                                    "Your idea is the glowing **amber star** — the further from "
                                    "the crowd, the more conceptually novel."
                                )
                        except Exception as e:
                            st.caption(f"Constellation error: {e}")

                    # ── 2. Idea DNA Fingerprint ─────────────────────────
                    with st.expander("🧬 Idea DNA (unique fingerprint)"):
                        try:
                            from idea_simulator import build_dna_fingerprint
                            _dna = build_dna_fingerprint(_sim_idea)
                            if _dna:
                                st.plotly_chart(_dna, use_container_width=True,
                                                key=f"sim_dna_{_sim_idx}")
                                st.caption(
                                    "Each idea has a **unique 16-band fingerprint** generated from "
                                    "its methodology, novelty, and probe scores. Same idea → same "
                                    "fingerprint. Share the ID with collaborators."
                                )
                        except Exception as e:
                            st.caption(f"DNA error: {e}")

                    # ── 3. Time Machine ─────────────────────────────────
                    with st.expander("⏳ Time Machine (recency of method components)"):
                        try:
                            from idea_simulator import build_time_machine, detect_techniques
                            _tm = build_time_machine(_sim_idea)
                            if _tm:
                                st.plotly_chart(_tm, use_container_width=True,
                                                key=f"sim_tm_{_sim_idx}")
                                _techs = detect_techniques(_sim_idea)
                                if _techs:
                                    _years = [t["year"] for t in _techs]
                                    _avg = sum(_years) / len(_years)
                                    st.caption(
                                        f"Detected **{len(_techs)} techniques**, "
                                        f"oldest **{min(_years)}**, newest **{max(_years)}**, "
                                        f"average year **{_avg:.0f}**. Green = cutting-edge "
                                        f"(<3y), blue = mainstream, amber = mature, gray = vintage."
                                    )
                            else:
                                st.info(
                                    "No known ML techniques detected in the method. "
                                    "Add specific terms (e.g., 'transformer', 'GNN') to see the timeline."
                                )
                        except Exception as e:
                            st.caption(f"Time Machine error: {e}")

                    # ── 4. Reviewer Chat Simulator ──────────────────────
                    with st.expander("💬 Reviewer Chat (predicted feedback from 3 reviewers)"):
                        try:
                            from idea_simulator import simulate_reviewer_chat
                            _chat = simulate_reviewer_chat(_sim_idea)
                            for _msg in _chat:
                                _color = {
                                    "verdict": _msg.get("color", "#0ea5e9"),
                                    "positive": "#10b981",
                                    "negative": "#ef4444",
                                    "neutral": "#f59e0b",
                                }.get(_msg["sentiment"], "#0ea5e9")
                                _bg = {
                                    "verdict": "#0c4a6e",
                                    "positive": "#f0fdf4",
                                    "negative": "#fef2f2",
                                    "neutral": "#fffbeb",
                                }.get(_msg["sentiment"], "#f0f9ff")
                                _text_color = "white" if _msg["sentiment"] == "verdict" else "#0c4a6e"
                                st.markdown(
                                    f'<div style="background:{_bg};border-left:4px solid {_color};'
                                    f'border-radius:8px;padding:10px 14px;margin:6px 0">'
                                    f'<div style="font-weight:700;color:{_text_color};font-size:13px;'
                                    f'margin-bottom:4px">{_msg["role"]}</div>'
                                    f'<div style="color:{_text_color};font-size:13px;line-height:1.5">'
                                    f'{_msg["msg"]}</div>'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )
                            st.caption(
                                "These reviews are derived purely from probe scores — no LLM call. "
                                "Reviewer reactions are deterministic for the same idea."
                            )
                        except Exception as e:
                            st.caption(f"Reviewer chat error: {e}")

                    # ── 5. Tarot Reading ────────────────────────────────
                    with st.expander("🔮 Idea Tarot (Past · Present · Future)"):
                        try:
                            from idea_simulator import generate_tarot, tarot_to_html
                            _cards = generate_tarot(_sim_idea)
                            st.markdown(tarot_to_html(_cards), unsafe_allow_html=True)
                            st.caption(
                                "Three narrative cards based on the idea's novelty, quality, "
                                "and significance. Past = origin context, Present = current state, "
                                "Future = projected outcome."
                            )
                        except Exception as e:
                            st.caption(f"Tarot error: {e}")

                    # ═══════════════════════════════════════════════════════
                    # MYTHIC ANALYTICS (5 wild-card visualizations)
                    # ═══════════════════════════════════════════════════════
                    st.markdown("---")
                    st.markdown(
                        '<div style="font-size:14px;font-weight:700;'
                        'background:linear-gradient(90deg,#ec4899,#f59e0b,#10b981);'
                        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;'
                        'margin:6px 0">🃏 Mythic Analytics</div>',
                        unsafe_allow_html=True,
                    )

                    # ── 1. Idea Pokémon Card ────────────────────────────
                    with st.expander("🃏 Idea Card (collectible stat block)"):
                        try:
                            from idea_simulator import build_pokemon_card
                            _card_html = build_pokemon_card(_sim_idea)
                            st.markdown(_card_html, unsafe_allow_html=True)
                            st.caption(
                                "A collectible stat card for your idea. Same idea always "
                                "produces the same card — share the ID with collaborators "
                                "or screenshot for social media."
                            )
                        except Exception as e:
                            st.caption(f"Card error: {e}")

                    # ── 2. Idea Weather Forecast ────────────────────────
                    with st.expander("☀️ Project Weather Forecast (7-week outlook)"):
                        try:
                            from idea_simulator import build_weather_forecast, weather_to_html
                            _forecast = build_weather_forecast(_sim_idea)
                            _html = weather_to_html(_forecast)
                            st.markdown(_html, unsafe_allow_html=True)
                            _stormy = sum(1 for f in _forecast if f["risk"] >= 0.6)
                            _sunny = sum(1 for f in _forecast if f["risk"] < 0.4)
                            st.caption(
                                f"**{_sunny}** sunny weeks, **{_stormy}** stormy weeks. "
                                "Plan checkpoints + buffer time around the rainy patches."
                            )
                        except Exception as e:
                            st.caption(f"Weather error: {e}")

                    # ── 3. Twin Universe ────────────────────────────────
                    with st.expander("🌌 Twin Universe (parallel paths)"):
                        try:
                            from idea_simulator import (
                                build_twin_universe, twin_universe_summary,
                            )
                            _twin = build_twin_universe(_sim_idea)
                            if _twin:
                                st.plotly_chart(_twin, use_container_width=True,
                                                key=f"sim_twin_{_sim_idx}")
                            _summary = twin_universe_summary(_sim_idea)
                            for _u in _summary:
                                _delta_color = ("#10b981" if "Better" in _u["verdict"]
                                                 else "#ef4444" if "Worse" in _u["verdict"]
                                                 else "#64748b")
                                st.markdown(
                                    f'<div style="border-left:3px solid {_u["color"]};'
                                    f'padding:6px 12px;margin:4px 0">'
                                    f'<b>{_u["name"]}</b>'
                                    f'<span style="color:#64748b;font-size:12px;'
                                    f'margin-left:8px">{_u["description"]}</span><br>'
                                    f'<span style="color:#0c4a6e">P50: <b>{_u["p50"]}%</b></span> '
                                    f'<span style="color:{_delta_color};font-weight:600;'
                                    f'margin-left:8px">{_u["verdict"]} ({_u["delta"]:+.1f}%)</span>'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )
                        except Exception as e:
                            st.caption(f"Twin universe error: {e}")

                    # ── 4. Origin Story ─────────────────────────────────
                    with st.expander("📖 Origin Story (3-paragraph myth)"):
                        try:
                            from idea_simulator import generate_origin_story
                            _story = generate_origin_story(_sim_idea)
                            st.markdown(
                                f'<div style="background:linear-gradient(135deg,#fef3c7,#fef9c3);'
                                f'border-left:5px solid #f59e0b;border-radius:10px;'
                                f'padding:18px 22px;margin:8px 0;color:#451a03;line-height:1.7;'
                                f'font-family:Georgia,serif">'
                                f'{_story}</div>',
                                unsafe_allow_html=True,
                            )
                            st.caption(
                                "An auto-generated narrative interpreting your idea as a "
                                "three-act story: Origin · Journey · Destiny."
                            )
                        except Exception as e:
                            st.caption(f"Origin story error: {e}")

                    # ── 5. Probability Cloud ────────────────────────────
                    with st.expander("🌫️ Probability Cloud (compute × data → success)"):
                        try:
                            from idea_simulator import build_probability_cloud
                            _cloud = build_probability_cloud(_sim_idea)
                            if _cloud:
                                st.plotly_chart(_cloud, use_container_width=True,
                                                key=f"sim_cloud_{_sim_idx}")
                                st.caption(
                                    "Each cell shows the projected median outcome (P50) "
                                    "for that combination of compute + data quality. "
                                    "Run **80 Monte Carlo trials** per cell. "
                                    "Use this to identify the cheapest path to a target outcome."
                                )
                        except Exception as e:
                            st.caption(f"Probability cloud error: {e}")

                    # ═══════════════════════════════════════════════════════
                    # CINEMATIC ANALYTICS (5 storytelling visualizations)
                    # ═══════════════════════════════════════════════════════
                    st.markdown("---")
                    st.markdown(
                        '<div style="font-size:14px;font-weight:700;'
                        'background:linear-gradient(90deg,#a855f7,#ec4899,#f59e0b);'
                        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;'
                        'margin:6px 0">🎬 Cinematic Analytics</div>',
                        unsafe_allow_html=True,
                    )

                    # ── 1. Movie Trailer ────────────────────────────────
                    with st.expander("🎬 Movie Trailer (Hollywood pitch deck)"):
                        try:
                            from idea_simulator import (
                                generate_movie_trailer, trailer_to_html,
                            )
                            _trailer = generate_movie_trailer(_sim_idea)
                            st.markdown(trailer_to_html(_trailer),
                                        unsafe_allow_html=True)
                            st.caption(
                                f"**{_trailer['rating']}** · {_trailer['stars']}/5 stars. "
                                "An auto-generated cinematic pitch with tagline, "
                                "synopsis, and imaginary cast — share for fun, or use "
                                "the framing as a memorable elevator pitch."
                            )
                        except Exception as e:
                            st.caption(f"Trailer error: {e}")

                    # ── 2. Quest Log ────────────────────────────────────
                    with st.expander("🏆 Quest Log (RPG-style next steps)"):
                        try:
                            from idea_simulator import (
                                generate_quest_log, quest_log_to_html,
                            )
                            _quests = generate_quest_log(_sim_idea)
                            st.markdown(quest_log_to_html(_quests),
                                        unsafe_allow_html=True)
                            _total_xp = sum(q.get("xp", 0) for q in _quests)
                            _active = sum(1 for q in _quests if q["status"] == "active")
                            st.caption(
                                f"**{len(_quests)} quests** · {_active} active · "
                                f"{_total_xp} XP total reward. Side quests are derived "
                                "from your three weakest probe dimensions — completing "
                                "them is the fastest path to higher quality."
                            )
                        except Exception as e:
                            st.caption(f"Quest log error: {e}")

                    # ── 3. Conference Match ─────────────────────────────
                    with st.expander("📅 Conference Match (target venue ranker)"):
                        try:
                            from idea_simulator import (
                                match_conferences, conference_match_to_html,
                            )
                            _matches = match_conferences(_sim_idea)
                            st.markdown(conference_match_to_html(_matches),
                                        unsafe_allow_html=True)
                            _strong = sum(1 for m in _matches
                                          if "Strong" in m["verdict"])
                            st.caption(
                                f"**{_strong}/{len(_matches)} venues** look like a "
                                "strong fit. Match scores combine probe-weighted fit "
                                "(60%) and overall quality (40%) — submit to the "
                                "highest-ranked venue whose deadline you can hit."
                            )
                        except Exception as e:
                            st.caption(f"Conference match error: {e}")

                    # ── 4. Idea Mosaic ──────────────────────────────────
                    with st.expander("🎨 Idea Mosaic (stained-glass signature art)"):
                        try:
                            from idea_simulator import build_idea_mosaic
                            _mosaic = build_idea_mosaic(_sim_idea)
                            if _mosaic:
                                st.plotly_chart(_mosaic, use_container_width=True,
                                                key=f"sim_mosaic_{_sim_idx}")
                                st.caption(
                                    "A unique stained-glass signature for your idea. "
                                    "Tile colors are deterministically derived from the "
                                    "title hash, and the palette shifts with novelty + "
                                    "quality. Same idea always produces the same mosaic."
                                )
                        except Exception as e:
                            st.caption(f"Mosaic error: {e}")

                    # ── 5. Acceptance Speech ────────────────────────────
                    with st.expander("🎤 Acceptance Speech (best-paper preview)"):
                        try:
                            from idea_simulator import generate_acceptance_speech
                            _speech = generate_acceptance_speech(_sim_idea)
                            st.markdown(
                                f'<div style="background:linear-gradient(135deg,'
                                f'#fef3c7,#fffbeb);border:2px solid #f59e0b;'
                                f'border-radius:12px;padding:20px 26px;margin:8px 0;'
                                f'color:#451a03;line-height:1.7;font-style:italic;'
                                f'font-family:Georgia,serif">{_speech}</div>',
                                unsafe_allow_html=True,
                            )
                            st.caption(
                                "A 30-second best-paper award acceptance speech. "
                                "The tone shifts with your novelty + quality scores. "
                                "Read it out loud — if it doesn't make you smile, "
                                "the idea may need more conviction."
                            )
                        except Exception as e:
                            st.caption(f"Speech error: {e}")
                except ImportError:
                    st.error("idea_simulator module not available")
                except Exception as e:
                    st.error(f"Simulation error: {e}")
            else:
                st.caption("Click **Run Visual Simulation** to generate the visualizations.")

    # ── Execution-Aware Revision Tab ───────────────────────────────────────
    # Closes the probe → archive feedback loop the IdeaGraph paper notes as
    # the unsolved feasibility gap. Run a deliberately tiny LLM-simulated
    # proxy of each idea and Bayesian-blend the result into quality_score.
    with tab_exec_loop:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">'
            '<span style="font-size:22px">🔁</span>'
            '<span style="font-size:18px;font-weight:700;color:#0c4a6e">'
            'Execution-Aware Revision Loop</span></div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Closes the probe → archive feedback gap by running a deliberately "
            "tiny proxy experiment for each probe-passing idea, then "
            "Bayesian-blending the resulting feasibility signal into "
            "`quality_score`. Trust weight scales with sample size and seed count."
        )

        if not ideas:
            st.info("Run a pipeline first.")
        else:
            try:
                from agents.execution_revisor import (
                    revise as _exec_revise,
                    bayesian_blend as _exec_blend,
                    DEFAULT_SAMPLE_SIZE, DEFAULT_N_SEEDS,
                    DEFAULT_TARGET_SAMPLE_SIZE,
                )
            except ImportError as _imp_err:
                st.error(f"execution_revisor unavailable: {_imp_err}")
                _exec_revise = None

            if _exec_revise is not None:
                _ec1, _ec2, _ec3 = st.columns([2, 1, 1])
                with _ec1:
                    _exec_n_samples = st.select_slider(
                        "Proxy sample size",
                        options=[100, 500, 1000, 2500, 10000, 25000, 100000],
                        value=DEFAULT_SAMPLE_SIZE,
                        help=f"Smaller = noisier, lower trust. Target run is "
                             f"~{DEFAULT_TARGET_SAMPLE_SIZE:,}.",
                        key="exec_n_samples",
                    )
                with _ec2:
                    _exec_n_seeds = st.number_input(
                        "Seeds", min_value=1, max_value=20,
                        value=DEFAULT_N_SEEDS, step=1,
                        help="More seeds = lower variance, higher trust.",
                        key="exec_n_seeds",
                    )
                with _ec3:
                    _, _trust_preview = _exec_blend(
                        0.7, 0.4, n_samples=_exec_n_samples,
                        n_seeds=_exec_n_seeds,
                    )
                    st.metric("Trust weight",
                                f"{_trust_preview*100:.0f}%",
                                help="How much the blended quality leans on the "
                                     "execution signal at this configuration.")

                _passing = [i for i in ideas
                             if i.get("probe_passed") or i.get("quality_score", 0) >= 0.4]
                _passing = sorted(_passing,
                                    key=lambda x: x.get("quality_score", 0),
                                    reverse=True)
                if not _passing:
                    st.info("No probe-passing ideas to revise.")
                else:
                    st.markdown(f"**{len(_passing)} probe-passing ideas** ready for revision.")
                    if st.button("🔁 Run execution-aware revision",
                                  type="primary", use_container_width=True,
                                  key="exec_revise_btn"):
                        st.session_state["_exec_revise_run"] = True
                        st.session_state["_exec_revise_results"] = []
                        with st.spinner(
                            f"Running tiny-experiment proxy ({_exec_n_samples:,} "
                            f"samples × {_exec_n_seeds} seed) for {len(_passing)} ideas…"
                        ):
                            _rrows = []
                            for _i, _it in enumerate(_passing):
                                _rev = _exec_revise(
                                    _it, n_samples=_exec_n_samples,
                                    n_seeds=_exec_n_seeds,
                                )
                                _rrows.append({"idea": _it, "rev": _rev})
                            st.session_state["_exec_revise_results"] = _rrows

                    _results = st.session_state.get("_exec_revise_results", [])
                    if _results:
                        # ── Summary metrics ─────────────────────────────────
                        _deltas = [r["rev"].delta for r in _results
                                    if r["rev"].success]
                        _improved = sum(1 for d in _deltas if d > 0.02)
                        _hurt = sum(1 for d in _deltas if d < -0.02)
                        _neutral = len(_deltas) - _improved - _hurt
                        _used_llm = sum(1 for r in _results if r["rev"].used_llm)
                        _total_cost = sum(r["rev"].cost_usd for r in _results)

                        _m1, _m2, _m3, _m4 = st.columns(4)
                        _m1.metric("📈 Boosted", _improved,
                                    help="Execution signal raised quality_score by ≥0.02")
                        _m2.metric("📉 Demoted", _hurt,
                                    help="Execution signal lowered quality_score by ≥0.02")
                        _m3.metric("➖ Unchanged", _neutral,
                                    help="Δ within ±0.02")
                        _m4.metric("💰 Cost",
                                    f"${_total_cost:.3f}",
                                    f"{_used_llm}/{len(_results)} via LLM")

                        st.markdown("---")
                        st.markdown("### Probe vs. Execution Signal")
                        try:
                            import plotly.graph_objects as _go
                            _xs = list(range(1, len(_results) + 1))
                            _probe_qs = [r["rev"].probe_quality for r in _results]
                            _blend_qs = [r["rev"].blended_quality for r in _results]
                            _exec_qs = [r["rev"].execution_signal for r in _results]
                            _titles = [r["idea"].get("title", "?") for r in _results]
                            _fig = _go.Figure()
                            _fig.add_trace(_go.Bar(
                                x=_xs, y=_probe_qs, name="Probe-only",
                                marker_color="#94a3b8", text=_titles,
                                hovertemplate="<b>%{text}</b><br>Probe: %{y:.2f}<extra></extra>",
                            ))
                            _fig.add_trace(_go.Bar(
                                x=_xs, y=_exec_qs, name="Execution signal",
                                marker_color="#f59e0b",
                                hovertemplate="Exec: %{y:.2f}<extra></extra>",
                            ))
                            _fig.add_trace(_go.Scatter(
                                x=_xs, y=_blend_qs, name="Bayesian blend",
                                mode="lines+markers",
                                marker=dict(size=10, color="#0ea5e9",
                                              line=dict(width=2, color="white")),
                                line=dict(width=3, color="#0ea5e9"),
                                hovertemplate="<b>Blend: %{y:.2f}</b><extra></extra>",
                            ))
                            _fig.update_layout(
                                height=380, barmode="group",
                                xaxis_title="Idea (rank by probe quality)",
                                yaxis_title="Score",
                                yaxis=dict(range=[0, 1]),
                                margin=dict(l=40, r=20, t=20, b=40),
                                legend=dict(orientation="h", yanchor="bottom",
                                              y=1.02, xanchor="right", x=1),
                            )
                            st.plotly_chart(_fig, use_container_width=True,
                                              key="exec_loop_chart")
                        except Exception as _e:
                            st.caption(f"Chart render error: {_e}")

                        st.markdown("---")
                        st.markdown("### Per-idea revision details")
                        # Sort by absolute delta to surface the biggest moves
                        _sorted_results = sorted(
                            _results,
                            key=lambda r: abs(r["rev"].delta),
                            reverse=True,
                        )
                        for _row in _sorted_results:
                            _it = _row["idea"]
                            _rv = _row["rev"]
                            _arrow = ("📈" if _rv.delta > 0.02 else
                                       "📉" if _rv.delta < -0.02 else "➖")
                            _color = ("#10b981" if _rv.delta > 0.02 else
                                       "#ef4444" if _rv.delta < -0.02 else "#64748b")
                            _ttl = _it.get("title", "Untitled")
                            with st.expander(
                                f"{_arrow} {_ttl}  ·  Δ {_rv.delta:+.3f}",
                                expanded=False,
                            ):
                                _c1, _c2, _c3, _c4 = st.columns(4)
                                _c1.metric("Probe quality",
                                            f"{_rv.probe_quality:.2f}")
                                _c2.metric("Execution signal",
                                            f"{_rv.execution_signal:.2f}")
                                _c3.metric("Trust weight",
                                            f"{_rv.trust_weight*100:.0f}%")
                                _c4.metric("Blended (posterior)",
                                            f"{_rv.blended_quality:.2f}",
                                            f"{_rv.delta:+.3f}")
                                if _rv.metric_name:
                                    st.markdown(f"**Metric:** {_rv.metric_name}")
                                if _rv.predicted_metric is not None:
                                    _ci_str = ""
                                    if _rv.confidence_interval:
                                        _ci_str = (f"  · 95% CI ["
                                                    f"{_rv.confidence_interval[0]:.2f}, "
                                                    f"{_rv.confidence_interval[1]:.2f}]")
                                    st.markdown(
                                        f"**Predicted value:** "
                                        f"`{_rv.predicted_metric:.3f}`{_ci_str}"
                                    )
                                if _rv.failure_modes:
                                    st.markdown("**Likely failure modes:**")
                                    for _fm in _rv.failure_modes:
                                        st.markdown(f"- {_fm}")
                                if _rv.error:
                                    st.caption(f"⚠️ {_rv.error}")
                                _src_strat = _it.get("source_strategy", "")
                                _meth = _it.get("methodology_type", "?")
                                _nov = _it.get("novelty_level", "?")
                                st.caption(
                                    f"Strategy {_src_strat} · {_meth} · {_nov} "
                                    f"· {_rv.sample_size:,} samples × {_rv.n_seeds} "
                                    f"seed{'s' if _rv.n_seeds != 1 else ''}"
                                )
                    elif not st.session_state.get("_exec_revise_run"):
                        st.caption(
                            "**How it works.** Each probe-passing idea is sent to "
                            "an LLM acting as a domain-expert reviewer. The reviewer "
                            "mentally simulates running a tiny version of the proposed "
                            "experiment and reports a predicted metric, 95% confidence "
                            "interval, and a 0–1 *execution_signal*. We combine that "
                            "with the probe quality using inverse-variance Bayesian "
                            "weighting — small experiments get small trust weights "
                            "(more noise), bigger ones get larger weights. The "
                            "posterior overwrites `quality_score` so the QD archive "
                            "ranks ideas by feasibility-aware quality, not just "
                            "surface-level probe agreement."
                        )

    # ── Reviewer Lens Tab ──────────────────────────────────────────────────
    # ⚠️  Intellectually controversial: this scores ideas by predicted
    # acceptance at a target venue. It's a tool, not a recommendation.
    # See acceptance_predictor.py docstring for full framing.
    with tab_reviewer:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">'
            '<span style="font-size:22px">📜</span>'
            '<span style="font-size:18px;font-weight:700;color:#0c4a6e">'
            'Reviewer Lens — venue-aware acceptance prediction</span></div>',
            unsafe_allow_html=True,
        )

        # Disclaimer banner — load-bearing for the framing
        st.markdown(
            '<div style="background:#fff7ed;border:2px solid #f59e0b;'
            'border-radius:10px;padding:12px 16px;margin:8px 0;'
            'color:#7c2d12;font-size:13px;line-height:1.55">'
            '<b>⚠️ A note on what this is — and isn\'t.</b><br>'
            "This panel predicts the probability that each of your ideas would "
            "be accepted at a chosen peer-review venue, based on patterns "
            "common to that venue's acceptance bar (rigor + venue-specific "
            "methodology and novelty preferences). "
            "<b>This is descriptive, not prescriptive.</b> "
            "The point of research is to find what's true, not what reviewers "
            "will accept; using this to <i>select</i> ideas instead of "
            "<i>understand them</i> would be a misuse. We surface this tool "
            "because some researchers find it useful for matching a finished "
            "idea to the right venue — not for choosing which ideas to pursue."
            '</div>',
            unsafe_allow_html=True,
        )

        if not ideas:
            st.info("Run a pipeline first so there's something to score.")
        else:
            try:
                from acceptance_predictor import (
                    VENUE_PROFILES, score_idea, compare_venues, rank_ideas,
                )
            except ImportError as _e:
                st.error(f"acceptance_predictor unavailable: {_e}")
                _venue_profiles = None
            else:
                _venue_profiles = VENUE_PROFILES

            if _venue_profiles:
                _all_venues = list(_venue_profiles.keys())
                _r1, _r2, _r3 = st.columns([2, 1, 1])
                with _r1:
                    _venue = st.selectbox(
                        "Target venue",
                        options=_all_venues,
                        index=0,
                        format_func=lambda v: (
                            f"{v}  —  {_venue_profiles[v]['tier']}  "
                            f"({int(_venue_profiles[v]['acceptance_rate']*100)}%)"
                        ),
                        key="rev_venue",
                    )
                with _r2:
                    _mode = st.radio(
                        "Scoring mode",
                        options=["heuristic", "llm"],
                        format_func=lambda m: (
                            "🧮 Heuristic (instant)" if m == "heuristic"
                            else "🤖 LLM (1 call/idea)"
                        ),
                        horizontal=False,
                        key="rev_mode",
                    )
                with _r3:
                    _max_ideas = st.number_input(
                        "Max ideas", min_value=1, max_value=50,
                        value=min(15, len(ideas)), step=1,
                        key="rev_max_ideas",
                    )

                _profile = _venue_profiles[_venue]
                st.markdown(
                    f"<div style='background:#f0f9ff;border-left:4px solid #0ea5e9;"
                    f"padding:10px 14px;margin:6px 0;border-radius:6px;font-size:13px'>"
                    f"<b>{_venue}</b> — {_profile['description']}</div>",
                    unsafe_allow_html=True,
                )

                _llm_client = None
                if _mode == "llm":
                    try:
                        from claude_provider import get_claude_client
                        _llm_client = get_claude_client()
                    except Exception as _e:
                        st.warning(
                            f"LLM client unavailable ({_e}); falling back "
                            "to heuristic mode."
                        )

                if st.button(
                    f"📜 Score {min(int(_max_ideas), len(ideas))} ideas at {_venue}",
                    type="primary", use_container_width=True,
                    key="rev_score_btn",
                ):
                    _to_score = sorted(
                        ideas, key=lambda x: x.get("quality_score", 0),
                        reverse=True,
                    )[:int(_max_ideas)]
                    with st.spinner(
                        f"Scoring {len(_to_score)} idea(s) at {_venue}…"
                    ):
                        _ranked = rank_ideas(
                            _to_score, _venue, mode=_mode,
                            claude_client=_llm_client,
                        )
                    st.session_state["_rev_results"] = [
                        {"idea": i, "result": r.to_dict()}
                        for i, r in _ranked
                    ]
                    st.session_state["_rev_venue"] = _venue
                    st.session_state["_rev_mode_used"] = _mode

                _rows = st.session_state.get("_rev_results", [])
                if _rows and st.session_state.get("_rev_venue") == _venue:
                    n = len(_rows)
                    n_accept = sum(1 for r in _rows
                                    if r["result"]["decision"] == "accept")
                    n_borderline = sum(1 for r in _rows
                                        if r["result"]["decision"] == "borderline")
                    n_reject = sum(1 for r in _rows
                                    if r["result"]["decision"] == "reject")
                    avg_p = sum(r["result"]["accept_prob"] for r in _rows) / max(1, n)
                    used_llm = sum(1 for r in _rows if r["result"]["used_llm"])

                    st.markdown("---")
                    _h1, _h2, _h3, _h4, _h5 = st.columns(5)
                    _h1.metric("✅ Predicted accept", n_accept)
                    _h2.metric("⚖️ Borderline", n_borderline)
                    _h3.metric("❌ Predicted reject", n_reject)
                    _h4.metric("Mean p(accept)", f"{avg_p:.2f}")
                    _h5.metric("LLM-scored",
                                f"{used_llm}/{n}",
                                f"mode={st.session_state.get('_rev_mode_used','heuristic')}")

                    # ── Per-idea ranked list ───────────────────────────────
                    st.markdown(f"### Ideas ranked by predicted accept @ {_venue}")
                    for _i, _row in enumerate(_rows, 1):
                        _it = _row["idea"]
                        _r = _row["result"]
                        _dec = _r["decision"]
                        _color = ("#10b981" if _dec == "accept"
                                   else "#f59e0b" if _dec == "borderline"
                                   else "#ef4444")
                        _icon = ("✅" if _dec == "accept"
                                  else "⚖️" if _dec == "borderline"
                                  else "❌")
                        _ttl = _it.get("title", "Untitled")
                        with st.expander(
                            f"{_icon} #{_i}. {_ttl}  ·  p={_r['accept_prob']:.2f} "
                            f"({_dec})",
                            expanded=(_i == 1),
                        ):
                            _c1, _c2, _c3 = st.columns(3)
                            _c1.metric("Accept prob",
                                        f"{_r['accept_prob']:.2f}",
                                        help="Predicted probability of acceptance "
                                             "at this venue.")
                            _c2.metric("Confidence",
                                        f"{_r['confidence']*100:.0f}%",
                                        help="How far from the borderline (0.5) "
                                             "this prediction is.")
                            _c3.metric("Decision", _dec.title())

                            if _r.get("top_strengths"):
                                st.markdown("**What reviewers will likely praise**")
                                for _s in _r["top_strengths"]:
                                    st.markdown(f"- ✅ {_s}")
                            if _r.get("top_weaknesses"):
                                st.markdown("**What reviewers will likely push back on**")
                                for _w in _r["top_weaknesses"]:
                                    st.markdown(f"- ⚠️ {_w}")
                            if _r.get("error"):
                                st.caption(f"⚠️ {_r['error']}")

                            # Idea fields condensed for context
                            st.markdown(
                                f"<div style='background:#fafafa;border-radius:6px;"
                                f"padding:8px 12px;margin-top:8px;font-size:12px;"
                                f"color:#475569'>"
                                f"<b>Method.</b> {(_it.get('method','') or '')[:240]}"
                                f"…<br>"
                                f"<b>Hypothesis.</b> {(_it.get('hypothesis','') or '')[:200]}"
                                f"</div>",
                                unsafe_allow_html=True,
                            )

                    # ── Multi-venue comparison heatmap ─────────────────────
                    st.markdown("---")
                    st.markdown("### Multi-venue comparison")
                    if st.button(
                        "🔬 Compare top 5 ideas across all venues",
                        use_container_width=True,
                        key="rev_compare_venues_btn",
                    ):
                        _top5 = _rows[:5]
                        _all_v = list(_venue_profiles.keys())
                        _matrix = []
                        for _row in _top5:
                            _it = _row["idea"]
                            _scores = compare_venues(
                                _it, venues=_all_v,
                                mode="heuristic",  # always heuristic for the matrix
                            )
                            _by_v = {r.venue: r.accept_prob for r in _scores}
                            _matrix.append([_by_v.get(v, 0.0) for v in _all_v])

                        try:
                            import plotly.graph_objects as _go
                            _fig = _go.Figure(data=_go.Heatmap(
                                z=_matrix,
                                x=_all_v,
                                y=[r["idea"].get("title", "?")[:36] + "…"
                                    for r in _top5],
                                colorscale=[[0, "#fef2f2"], [0.5, "#fde68a"],
                                              [1, "#10b981"]],
                                zmin=0, zmax=1,
                                text=[[f"{v:.2f}" for v in row]
                                       for row in _matrix],
                                texttemplate="%{text}",
                                hovertemplate="<b>%{y}</b> @ <b>%{x}</b><br>"
                                              "p(accept) = %{z:.3f}<extra></extra>",
                                colorbar=dict(title="p(accept)", thickness=12),
                            ))
                            _fig.update_layout(
                                height=360,
                                margin=dict(l=180, r=20, t=20, b=20),
                                plot_bgcolor="rgba(0,0,0,0)",
                                xaxis=dict(side="top"),
                                yaxis=dict(autorange="reversed"),
                            )
                            st.plotly_chart(_fig, use_container_width=True,
                                              key="rev_venue_heatmap")
                            st.caption(
                                "Each cell shows the heuristic accept probability "
                                "for that idea at that venue. Use this to spot "
                                "the **best venue match** for an idea — not to "
                                "decide whether the idea is good."
                            )
                        except Exception as _e:
                            st.caption(f"Heatmap unavailable: {_e}")

                with st.expander("📋 How the score is computed", expanded=False):
                    st.markdown(
                        "**Heuristic mode** combines the 10 existing probe scores "
                        "with two venue-fit terms:\n"
                        "- `methodology_match`: how well the idea's `methodology_type` "
                        "matches what this venue typically values (e.g. NeurIPS "
                        "favors empirical / theoretical; KDD favors empirical / system).\n"
                        "- `novelty_match`: how well `novelty_level` aligns with "
                        "the venue's expected bar (top venues demand "
                        "*substantial* novelty; workshops accept *incremental*).\n"
                        "\nFeatures are passed through a venue-specific "
                        "linear-then-sigmoid model. The bias is calibrated so "
                        "that an 'average' idea (all features 0.5) maps to that "
                        "venue's published acceptance rate.\n"
                        "\n**LLM mode** sends one call per idea; the model acts as "
                        "a senior reviewer at the venue and returns a calibrated "
                        "probability plus 3 strengths and 3 weaknesses.\n"
                        "\nIn the proposed paper, the heuristic weights would be "
                        "replaced by fits on real (idea, venue, accept/reject) "
                        "tuples scraped from OpenReview. The public API is "
                        "identical, so swapping in trained weights is a "
                        "one-file change."
                    )

    # ── Provenance Tab ─────────────────────────────────────────────────────
    # Per-idea provenance + within-subjects behavioral study (the CHI/FAccT
    # experiment: do reviewers trust ideas more when they can see where the
    # idea came from?). Provenance is backfilled from existing Idea fields,
    # so this works even on legacy ideas.
    with tab_provenance:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">'
            '<span style="font-size:22px">🧬</span>'
            '<span style="font-size:18px;font-weight:700;color:#0c4a6e">'
            'Idea Provenance & Trust-Calibration Study</span></div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Trace every accepted idea back to its constituent inputs — "
            "source DAG nodes, generation strategy, parent idea (if any), "
            "probe critique, execution-aware revision. Then, optionally, "
            "run the within-subjects behavioral study from the proposed "
            "CHI/FAccT paper: does showing provenance change how much "
            "reviewers trust an idea?"
        )

        if not ideas:
            st.info("Run a pipeline first so there's something to trace.")
        else:
            try:
                from idea_provenance import (
                    extract_provenance,
                    render_provenance_card_html,
                    build_provenance_figure,
                    behavioral_assignment,
                    summarize_behavioral_study,
                    STRATEGY_LABELS,
                )
            except ImportError as _e:
                st.error(f"idea_provenance unavailable: {_e}")
                extract_provenance = None

            if extract_provenance is not None:
                _prov_mode = st.radio(
                    "Mode",
                    options=["viewer", "study"],
                    format_func=lambda m: (
                        "🔍 Provenance viewer (single idea)"
                        if m == "viewer"
                        else "🧪 Behavioral study (within-subjects A/B)"
                    ),
                    horizontal=True,
                    key="prov_mode",
                )

                # ── VIEWER MODE ─────────────────────────────────────────────
                if _prov_mode == "viewer":
                    _sorted_ideas = sorted(
                        ideas, key=lambda x: x.get("quality_score", 0),
                        reverse=True,
                    )
                    _idx = st.selectbox(
                        "Pick an idea to trace",
                        options=range(len(_sorted_ideas)),
                        format_func=lambda i: (
                            f"{_sorted_ideas[i].get('title','Untitled')} "
                            f"(q={_sorted_ideas[i].get('quality_score',0):.2f}, "
                            f"strat={_sorted_ideas[i].get('source_strategy','?')})"
                        ),
                        key="prov_viewer_idx",
                    )
                    _it = _sorted_ideas[_idx]

                    _record = extract_provenance(
                        _it,
                        dag_summary=results.get("dag_summary"),
                    )
                    _card_html = render_provenance_card_html(_record)
                    st.markdown(_card_html, unsafe_allow_html=True)

                    _fig = build_provenance_figure(_record)
                    if _fig is not None:
                        st.plotly_chart(
                            _fig, use_container_width=True,
                            key=f"prov_fig_{_idx}",
                        )

                    # Strategy distribution (group statistics)
                    st.markdown("---")
                    st.markdown("**Strategy distribution across all ideas**")
                    _by_strat: Dict[str, int] = {}
                    for _i_other in ideas:
                        _s = (_i_other.get("source_strategy") or "?").upper()
                        _by_strat[_s] = _by_strat.get(_s, 0) + 1
                    _strat_cols = st.columns(len(STRATEGY_LABELS))
                    for _ci, (_code, _meta) in enumerate(STRATEGY_LABELS.items()):
                        _strat_cols[_ci].metric(
                            f"{_meta['icon']} {_code}",
                            _by_strat.get(_code, 0),
                            help=_meta["label"],
                        )

                    with st.expander("📋 Sources used to construct this trace",
                                       expanded=False):
                        for _src in _record.sources_used:
                            st.markdown(f"- `{_src}`")
                        if not _record.sources_used:
                            st.caption("No sources recorded.")

                # ── BEHAVIORAL STUDY MODE ──────────────────────────────────
                else:
                    st.markdown(
                        '<div style="background:#fef3c7;border:1px solid #f59e0b;'
                        'border-radius:8px;padding:10px 14px;margin:6px 0;'
                        'color:#7c2d12;font-size:13px">'
                        '<b>About this study.</b> You will see a series of ideas '
                        'in random order. Half of them include their provenance; '
                        'half do not. Rate how much you trust each idea on a 1–5 '
                        'scale. After you rate them all, we report whether '
                        'showing provenance changes your trust ratings — and '
                        'whether it improves <i>calibration</i> (alignment '
                        "between trust and the system's quality_score). "
                        '</div>',
                        unsafe_allow_html=True,
                    )

                    _r1, _r2 = st.columns([1, 1])
                    _study_n = _r1.slider(
                        "Number of ideas to rate",
                        min_value=4, max_value=min(20, len(ideas)),
                        value=min(8, len(ideas)), step=2,
                        key="prov_study_n",
                    )
                    _study_seed = _r2.number_input(
                        "Randomization seed",
                        min_value=0, max_value=9999, value=42, step=1,
                        key="prov_study_seed",
                        help="Same seed = same idea selection and same A/B "
                             "assignment. Use a fresh seed for a fresh study.",
                    )

                    if st.button("▶ Start study", type="primary",
                                  use_container_width=True,
                                  key="prov_study_start"):
                        # Pick top-2N ideas (so the rating sample is interesting),
                        # then sub-sample N at random.
                        import random as _rnd
                        rng = _rnd.Random(int(_study_seed))
                        _pool = sorted(
                            ideas, key=lambda x: x.get("quality_score", 0),
                            reverse=True,
                        )[:max(int(_study_n) * 2, int(_study_n))]
                        rng.shuffle(_pool)
                        _picked = _pool[: int(_study_n)]
                        _assign = behavioral_assignment(
                            len(_picked), seed=int(_study_seed),
                        )
                        st.session_state["_prov_study"] = {
                            "ideas": _picked,
                            "conditions": _assign,
                            "ratings": [None] * len(_picked),
                            "seed": int(_study_seed),
                        }

                    _study = st.session_state.get("_prov_study")
                    if _study:
                        _study_ideas = _study["ideas"]
                        _conditions = _study["conditions"]
                        _ratings = _study["ratings"]

                        _completed = sum(1 for r in _ratings if r is not None)
                        st.progress(
                            _completed / len(_study_ideas),
                            text=f"Rated {_completed} / {len(_study_ideas)}",
                        )

                        # Render each idea with its assigned condition
                        for _ix, (_idea, _cond) in enumerate(
                                zip(_study_ideas, _conditions)):
                            with st.expander(
                                f"#{_ix + 1}. "
                                f"{_idea.get('title','Untitled')}"
                                + (" ✅" if _ratings[_ix] is not None else ""),
                                expanded=(_ratings[_ix] is None
                                           and _ix == _completed),
                            ):
                                # Idea content (always shown)
                                st.markdown(
                                    f"**Title.** {_idea.get('title','')}\n\n"
                                    f"**Method.** {_idea.get('method','')}\n\n"
                                    f"**Hypothesis.** {_idea.get('hypothesis','')}"
                                )

                                # Provenance — only when assigned
                                if _cond == "with":
                                    _rec = extract_provenance(
                                        _idea,
                                        dag_summary=results.get("dag_summary"),
                                    )
                                    st.markdown(
                                        render_provenance_card_html(_rec),
                                        unsafe_allow_html=True,
                                    )
                                else:
                                    st.caption(
                                        "_(provenance hidden in this trial)_"
                                    )

                                _rating = st.radio(
                                    "How much do you trust this idea?",
                                    options=[1, 2, 3, 4, 5],
                                    horizontal=True,
                                    format_func=lambda v: ["1 — None",
                                                              "2 — Low",
                                                              "3 — Mixed",
                                                              "4 — High",
                                                              "5 — Very high"][v-1],
                                    key=f"prov_rate_{_ix}",
                                    index=(_ratings[_ix] - 1
                                            if _ratings[_ix] else 2),
                                )
                                if st.button("Save rating",
                                              key=f"prov_save_{_ix}"):
                                    _ratings[_ix] = int(_rating)
                                    st.session_state["_prov_study"]["ratings"] = _ratings
                                    st.success("Saved.")
                                    st.rerun()

                        if _completed >= len(_study_ideas):
                            st.markdown("---")
                            st.markdown("### 📊 Study results")

                            _records = [
                                {
                                    "condition": c,
                                    "trust_rating": r,
                                    "quality_score": float(
                                        _idea.get("quality_score", 0.0)
                                    ),
                                }
                                for c, r, _idea in zip(
                                    _conditions, _ratings, _study_ideas)
                                if r is not None
                            ]
                            _summary = summarize_behavioral_study(_records)

                            _h1, _h2, _h3, _h4 = st.columns(4)
                            _h1.metric(
                                "Trust (with provenance)",
                                f"{_summary['mean_trust_with']:.2f}",
                                help=f"N = {_summary['n_with']}",
                            )
                            _h2.metric(
                                "Trust (without provenance)",
                                f"{_summary['mean_trust_without']:.2f}",
                                help=f"N = {_summary['n_without']}",
                            )
                            _h3.metric(
                                "Δ trust",
                                f"{_summary['trust_delta']:+.2f}",
                                help="Positive = provenance increased trust.",
                            )
                            _h4.metric(
                                "Δ calibration",
                                f"{_summary['calibration_delta']:+.2f}",
                                help="Pearson correlation between your trust "
                                     "ratings and the system's quality_score, "
                                     "with provenance vs without. Positive = "
                                     "provenance helped you tell good ideas "
                                     "from bad.",
                            )

                            if _summary["trust_delta"] > 0.4:
                                _verdict = (
                                    "✅ **Provenance increased trust**. "
                                    "Whether that's *good* depends on whether "
                                    "the more-trusted ideas were also better "
                                    "(see calibration)."
                                )
                                st.success(_verdict)
                            elif _summary["trust_delta"] < -0.4:
                                _verdict = (
                                    "⚠️ **Provenance decreased trust**. "
                                    "This sometimes happens when the seeds or "
                                    "strategy don't inspire confidence."
                                )
                                st.warning(_verdict)
                            else:
                                st.info(
                                    "↔️ Provenance had a small effect on trust "
                                    "in this small sample."
                                )

                            with st.expander("Raw data", expanded=False):
                                st.json(_records)

                            if st.button("🔄 Restart study",
                                          key="prov_study_reset"):
                                del st.session_state["_prov_study"]
                                st.rerun()

    # ── Compare Tab ────────────────────────────────────────────────────────
    with tab_compare:
        st.subheader("Idea Comparison")
        st.caption("Compare 2-5 ideas side-by-side. The system picks a winner per dimension.")

        if not ideas or len(ideas) < 2:
            st.info("Need at least 2 ideas to compare. Run a pipeline first.")
        else:
            sorted_for_compare = sorted(ideas, key=lambda x: x.get("quality_score", 0), reverse=True)
            compare_options = [
                f"{i.get('title', 'Untitled')} (q={i.get('quality_score', 0):.2f})"
                for i in sorted_for_compare
            ]

            selected = st.multiselect(
                "Select 2-5 ideas to compare:",
                options=list(range(len(compare_options))),
                format_func=lambda x: compare_options[x],
                max_selections=5,
                key="compare_tab_select",
            )

            # ── Comparison Matrix ──────────────────────────────────────────
            if len(selected) >= 2:
                _matrix_ideas = [sorted_for_compare[i] for i in selected]
                try:
                    from speed_optimizer import build_comparison_matrix
                    _mtx = build_comparison_matrix(_matrix_ideas)
                    _summary = _mtx["summary"]

                    # Winner banner
                    _winner_idx = _summary["overall_winner"]
                    _winner_title = _summary["titles"][_winner_idx][:50]
                    _wins = _summary["wins_per_idea"][_winner_idx]
                    st.markdown(
                        f'<div style="background:linear-gradient(135deg,#fef3c7,#fffbeb);'
                        f'border:2px solid #f59e0b;border-radius:12px;padding:14px 18px;'
                        f'margin-bottom:12px;text-align:center">'
                        f'<div style="font-size:24px;margin-bottom:2px">🏆</div>'
                        f'<div style="font-size:11px;color:#a16207;font-weight:700;'
                        f'text-transform:uppercase;letter-spacing:0.06em">Overall Winner</div>'
                        f'<div style="font-size:16px;font-weight:700;color:#92400e">'
                        f'{_winner_title}</div>'
                        f'<div style="font-size:11px;color:#a16207;margin-top:3px">'
                        f'Wins on {_wins}/{len(_mtx["headers"])} dimensions</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    # Side-by-side matrix as HTML table
                    _idea_titles = [t[:35] for t in _summary["titles"]]
                    _table_rows = ['<tr style="background:#f0f9ff">'
                                   '<th style="text-align:left;padding:8px 10px;font-size:11px;'
                                   'color:#0369a1;text-transform:uppercase;letter-spacing:0.04em">'
                                   'Dimension</th>']
                    for n, t in enumerate(_idea_titles):
                        _is_overall = (n == _winner_idx)
                        _bg = "#fef3c7" if _is_overall else "transparent"
                        _table_rows.append(
                            f'<th style="text-align:center;padding:8px 10px;font-size:11px;'
                            f'color:#0369a1;text-transform:uppercase;letter-spacing:0.04em;'
                            f'background:{_bg}">'
                            f'{"🏆 " if _is_overall else ""}Idea {chr(65+n)}<br>'
                            f'<span style="font-weight:400;font-size:10px;color:#64748b">{t}</span>'
                            f'</th>'
                        )
                    _table_rows.append('</tr>')

                    for r_idx, (label, row_vals) in enumerate(zip(_mtx["headers"], _mtx["rows"])):
                        _winner_for_row = _mtx["winners"].get(label)
                        _row_html = (
                            '<tr>'
                            f'<td style="padding:6px 10px;font-size:12px;font-weight:600;color:#0c4a6e;'
                            'border-top:1px solid #e0f2fe">{label}</td>'
                        ).format(label=label)
                        for c_idx, val in enumerate(row_vals):
                            _is_winner_cell = (_winner_for_row == c_idx)
                            _color = "#10b981" if _is_winner_cell else "#334155"
                            _wt = "700" if _is_winner_cell else "500"
                            _bg = "#f0fdf4" if _is_winner_cell else "white"
                            _row_html += (
                                f'<td style="padding:6px 10px;text-align:center;font-size:12px;'
                                f'color:{_color};font-weight:{_wt};background:{_bg};'
                                f'border-top:1px solid #e0f2fe">'
                                f'{"✓ " if _is_winner_cell else ""}{val}</td>'
                            )
                        _row_html += '</tr>'
                        _table_rows.append(_row_html)

                    _table_html = (
                        '<table style="width:100%;border-collapse:collapse;border-radius:8px;'
                        'overflow:hidden;border:1px solid #bae6fd;background:white">'
                        + ''.join(_table_rows) + '</table>'
                    )
                    st.markdown(_table_html, unsafe_allow_html=True)

                    # Wins bar chart
                    st.markdown("**Wins per idea:**")
                    for n, (title, wins) in enumerate(zip(_summary["titles"],
                                                           _summary["wins_per_idea"])):
                        _pct = int(wins / max(len(_mtx["headers"]), 1) * 100)
                        _bar_color = "#f59e0b" if n == _winner_idx else "#0ea5e9"
                        st.markdown(
                            f'<div style="margin:4px 0">'
                            f'<div style="font-size:12px;color:#0c4a6e;margin-bottom:2px">'
                            f'<b>Idea {chr(65+n)}:</b> {title} — {wins} wins ({_pct}%)</div>'
                            f'<div style="background:#e0f2fe;border-radius:4px;height:8px;overflow:hidden">'
                            f'<div style="background:{_bar_color};width:{_pct}%;height:100%"></div>'
                            f'</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                except Exception as e:
                    st.error(f"Matrix build error: {e}")

            # Keep the legacy radar/details path below for additional context
            if False and len(selected) >= 2:
                compare_ideas = [sorted_for_compare[i] for i in selected]

                # ── Radar Chart Comparison ────────────────────────────────
                try:
                    from analytics import build_ideas_comparison_radar, build_idea_radar
                    radar = build_ideas_comparison_radar(compare_ideas)
                    if radar:
                        st.plotly_chart(radar, use_container_width=True)
                except Exception:
                    pass

                # ── Side-by-side cards ────────────────────────────────────
                cols = st.columns(len(compare_ideas))
                for col, idea in zip(cols, compare_ideas):
                    with col:
                        q = idea.get("quality_score", 0)
                        color = "🟢" if q >= 0.7 else "🟡" if q >= 0.5 else "🔴"
                        st.markdown(f"### {color} {idea.get('title', '')}")
                        st.metric("Quality", f"{q:.3f}")
                        st.caption(f"**Type:** {(idea.get('methodology_type') or '?').replace('_', ' ').title()}")
                        st.caption(f"**Novelty:** {(idea.get('novelty_level') or '?').capitalize()}")
                        st.caption(f"**Strategy:** {idea.get('source_strategy', '?')}")

                        scores = idea.get("probe_scores") or {}
                        if scores:
                            st.markdown("**Probe Scores:**")
                            for k, v in scores.items():
                                if isinstance(v, (int, float)):
                                    bar = "█" * int(v * 10) + "░" * (10 - int(v * 10))
                                    st.text(f"  {k:10s} {bar} {v:.2f}")

                        # Individual radar
                        try:
                            from analytics import build_idea_radar
                            ind_radar = build_idea_radar(idea)
                            if ind_radar:
                                st.plotly_chart(ind_radar, use_container_width=True)
                        except Exception:
                            pass

                st.divider()

                # ── Quality Ranking Bar Chart ─────────────────────────────
                try:
                    from analytics import build_quality_ranking_bar
                    rank_bar = build_quality_ranking_bar(ideas, top_n=10)
                    if rank_bar:
                        st.markdown("### Quality Rankings")
                        st.plotly_chart(rank_bar, use_container_width=True)
                except Exception:
                    pass

                # ── Detailed comparison table ─────────────────────────────
                st.markdown("### Detailed Comparison")
                fields = ["motivation", "method", "hypothesis", "resources", "expected_outcome", "risk_assessment"]
                for field_name in fields:
                    st.markdown(f"**{field_name.replace('_', ' ').title()}**")
                    field_cols = st.columns(len(compare_ideas))
                    for fc, idea in zip(field_cols, compare_ideas):
                        with fc:
                            st.write(idea.get(field_name, "N/A")[:300])
                    st.divider()
            else:
                st.info("Select at least 2 ideas above to compare.")

    # ── Mashup Tab ─────────────────────────────────────────────────────────
    with tab_mashup:
        st.subheader("Idea Mashup Generator")
        st.caption("Select 2 ideas → AI generates a novel hybrid combining the best of both.")

        if not ideas or len(ideas) < 2:
            st.info("Need at least 2 ideas to mashup.")
        else:
            sorted_for_mashup = sorted(ideas, key=lambda x: x.get("quality_score", 0), reverse=True)
            mashup_labels = [f"{i.get('title', '?')} (q={i.get('quality_score',0):.2f})" for i in sorted_for_mashup]

            mash_col1, mash_col2 = st.columns(2)
            with mash_col1:
                idx_a = st.selectbox("Idea A", range(len(mashup_labels)), format_func=lambda i: mashup_labels[i], key="mash_a")
            with mash_col2:
                idx_b = st.selectbox("Idea B", range(len(mashup_labels)), index=min(1, len(mashup_labels)-1), format_func=lambda i: mashup_labels[i], key="mash_b")

            if idx_a != idx_b:
                if st.button("🧬 Generate Mashup", type="primary", use_container_width=True):
                    with st.spinner("Creating hybrid idea..."):
                        try:
                            from creative_features import IdeaMashupGenerator
                            gen = IdeaMashupGenerator()
                            hybrid = gen.mashup(sorted_for_mashup[idx_a], sorted_for_mashup[idx_b])
                            if hybrid and hybrid.get("title"):
                                st.success(f"Mashup created: **{hybrid.get('title', '?')}**")
                                with st.container(border=True):
                                    st.markdown(f"### 🧬 {hybrid.get('title', 'Hybrid Idea')}")
                                    st.caption(f"Parents: {hybrid.get('_parent_a', '?')} + {hybrid.get('_parent_b', '?')}")
                                    st.markdown(f"**Synergy:** {hybrid.get('synergy_explanation', 'N/A')}")
                                    st.divider()
                                    mc1, mc2 = st.columns(2)
                                    with mc1:
                                        st.markdown(f"**Method:** {hybrid.get('method', 'N/A')[:300]}")
                                    with mc2:
                                        st.markdown(f"**Hypothesis:** {hybrid.get('hypothesis', 'N/A')[:300]}")

                                    # Visual abstract for the mashup
                                    try:
                                        from creative_features import generate_visual_abstract
                                        poster = generate_visual_abstract(hybrid, results.get("topic", ""))
                                        st.download_button("📊 Download Visual Abstract", data=poster,
                                            file_name="mashup_poster.html", mime="text/html")
                                    except Exception:
                                        pass
                            else:
                                st.warning("Mashup generation failed. Try different ideas.")
                        except Exception as e:
                            st.error(f"Mashup error: {e}")
            else:
                st.warning("Select two different ideas to mashup.")

    # ── Trends Tab ────────────────────────────────────────────────────────
    with tab_trends:
        st.subheader("Research Trend Predictor")
        st.caption("AI predicts which topics will be hot based on quality, engagement, and novelty patterns.")

        try:
            from creative_features import TrendPredictor
            predictor = TrendPredictor()

            uid_trend = st.session_state.get("user_id")
            user_results = db_cache.get_user_results(uid_trend) if uid_trend else []
            shared = db_cache.get_top_shared_ideas(limit=30)

            trends = predictor.predict_trends(user_results, shared)

            if trends:
                for i, t in enumerate(trends[:10], 1):
                    direction_emoji = {"rising": "🔥", "stable": "📊", "emerging": "🌱"}.get(t["direction"], "📊")
                    score = t["trend_score"]
                    bar_width = int(score * 100)

                    with st.container(border=True):
                        tc1, tc2 = st.columns([4, 1])
                        with tc1:
                            st.markdown(f"**{direction_emoji} #{i}. {t['topic'][:60]}**")
                            st.caption(t["prediction"])
                            st.markdown(
                                f"<div style='background:#333;border-radius:4px;height:8px;'>"
                                f"<div style='background:#3498db;height:100%;width:{bar_width}%;border-radius:4px;'></div>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                        with tc2:
                            st.metric("Score", f"{score:.2f}")
            else:
                st.info("Run more pipelines and share ideas to generate trend predictions.")
        except Exception as e:
            st.error(f"Trends error: {e}")

    # ── Collab Tab ────────────────────────────────────────────────────────
    with tab_collab:
        st.subheader("Collaborative Workspace")
        st.caption("Create shared rooms where your team works on the same research topic together.")

        uid_collab = st.session_state.get("user_id")
        if not uid_collab:
            st.info("Log in to use collaborative workspaces.")
        else:
            try:
                from creative_features import (
                    create_workspace, join_workspace, get_user_workspaces,
                    add_idea_to_workspace, get_workspace_ideas, vote_workspace_idea,
                    post_workspace_chat, get_workspace_chat, get_workspace_members,
                )

                # Create or join
                cw_col1, cw_col2 = st.columns(2)
                with cw_col1:
                    with st.expander("Create New Workspace"):
                        ws_name = st.text_input("Workspace name", key="ws_name", placeholder="My Research Team", autocomplete="off")
                        ws_topic = st.text_input("Research topic", key="ws_topic", placeholder="Optional focus area", autocomplete="off")
                        if st.button("Create", key="ws_create", type="primary"):
                            if ws_name:
                                ws = create_workspace(uid_collab, ws_name, ws_topic)
                                st.success(f"Created! Invite code: **{ws['invite_code']}**")
                                st.code(ws["invite_code"])
                with cw_col2:
                    with st.expander("Join Workspace"):
                        invite = st.text_input("Invite code", key="ws_invite", autocomplete="off")
                        if st.button("Join", key="ws_join", type="primary"):
                            if invite:
                                ws = join_workspace(uid_collab, invite.strip())
                                if ws:
                                    st.success(f"Joined: {ws['name']}")
                                else:
                                    st.error("Invalid invite code.")

                # List workspaces
                workspaces = get_user_workspaces(uid_collab)
                if workspaces:
                    st.markdown("### Your Workspaces")
                    for ws in workspaces:
                        with st.expander(f"📁 {ws['name']} ({ws.get('role', 'member')})"):
                            st.caption(f"Topic: {ws.get('topic', '—')} | Invite: `{ws.get('invite_code', '')}`")

                            # Members
                            members = get_workspace_members(ws["id"])
                            st.caption(f"Members: {', '.join(m['username'] for m in members)}")

                            # Ideas in workspace
                            ws_ideas = get_workspace_ideas(ws["id"])
                            if ws_ideas:
                                st.markdown("**Shared Ideas:**")
                                for wi in ws_ideas[:10]:
                                    idea_data = wi.get("idea", {})
                                    q = idea_data.get("quality_score", 0)
                                    wc1, wc2, wc3 = st.columns([3, 1, 1])
                                    with wc1:
                                        st.markdown(f"**{idea_data.get('title', '?')}** by {wi.get('username', '?')}")
                                        if wi.get("comment"):
                                            st.caption(f"💬 {wi['comment'][:100]}")
                                    with wc2:
                                        st.caption(f"q={q:.2f}")
                                    with wc3:
                                        if st.button(f"👍 {wi.get('votes', 0)}", key=f"vote_{wi['id']}"):
                                            vote_workspace_idea(wi["id"])
                                            st.rerun()

                            # Add idea from current results
                            if ideas:
                                add_opts = [f"{i.get('title', '?')}" for i in ideas[:10]]
                                sel_idea = st.selectbox("Add idea to workspace", range(len(add_opts)),
                                    format_func=lambda i: add_opts[i], key=f"ws_add_{ws['id']}")
                                add_comment = st.text_input("Comment", key=f"ws_cmt_{ws['id']}", placeholder="Why this idea...", autocomplete="off")
                                if st.button("Share to Workspace", key=f"ws_share_{ws['id']}"):
                                    add_idea_to_workspace(ws["id"], uid_collab, ideas[sel_idea], add_comment)
                                    st.success("Idea shared!")
                                    st.rerun()

                            # Chat
                            st.markdown("**Team Chat:**")
                            chat_msgs = get_workspace_chat(ws["id"], limit=10)
                            for msg in chat_msgs:
                                st.caption(f"**{msg['username']}**: {msg['message']} ({msg['created_at'][:16]})")
                            chat_input = st.text_input("Message", key=f"ws_chat_{ws['id']}", placeholder="Say something...", autocomplete="off")
                            if chat_input and st.button("Send", key=f"ws_send_{ws['id']}"):
                                post_workspace_chat(ws["id"], uid_collab, chat_input)
                                st.rerun()
                else:
                    st.info("Create a workspace or join one using an invite code.")
            except Exception as e:
                st.error(f"Workspace error: {e}")

    # ── Visual Abstract download on each idea card ────────────────────────
    # (handled via the export section and individual idea download)

    # ── Debate Tab ─────────────────────────────────────────────────────────
    with tab_debate:
        if not tournament_data:
            st.info("No debate data yet. Enable **Debate Arena** in the sidebar and run a pipeline, or run a tournament on your saved ideas.")
        else:
            st.subheader("Tournament Debate Arena")
            champion = tournament_data.get("champion_title", "")
            if champion:
                st.success(f"Champion: **{champion}**")
            st.caption(
                f"{tournament_data.get('entrant_count', 0)} ideas entered | "
                f"{tournament_data.get('total_matches', 0)} matches played"
            )

            rounds = tournament_data.get("rounds", [])
            for round_idx, round_matches in enumerate(rounds):
                st.markdown(f"### Round {round_idx + 1}")
                for match_idx, match in enumerate(round_matches):
                    idea_a = match.get("idea_a") or {}
                    idea_b = match.get("idea_b") or {}
                    title_a = (idea_a.get("title") or "Idea A")
                    title_b = (idea_b.get("title") or "Idea B")
                    winner = match.get("winner_side", "a")
                    verdict = match.get("judge_verdict") or {}
                    winner_label = title_a if winner == "a" else title_b

                    with st.expander(
                        f"Match {match_idx+1}: {title_a} vs {title_b} — Winner: {winner_label}"
                    ):
                        mc1, mc2 = st.columns(2)
                        with mc1:
                            st.markdown(f"**Idea A:** {title_a}")
                            st.caption(f"Advocate: {(match.get('advocate_a_role') or '?').capitalize()}")
                        with mc2:
                            st.markdown(f"**Idea B:** {title_b}")
                            st.caption(f"Advocate: {(match.get('advocate_b_role') or '?').capitalize()}")

                        st.markdown("**Debate Exchanges:**")
                        for ex in match.get("exchanges", []):
                            side_label = "A" if ex.get("side") == "a" else "B"
                            role = ex.get("role", "").capitalize()
                            st.markdown(
                                f"**[Round {ex.get('round', '?')}, {role} for Idea {side_label}]**"
                            )
                            st.write(ex.get("argument", ""))
                            st.divider()

                        if verdict:
                            st.markdown("**Judge Verdict:**")
                            st.write(verdict.get("reasoning", ""))
                            scores_col1, scores_col2 = st.columns(2)
                            scores_col1.metric("Score A", f"{verdict.get('score_a', 0):.2f}")
                            scores_col2.metric("Score B", f"{verdict.get('score_b', 0):.2f}")

                            # Consensus synthesis button
                            if st.button(f"Synthesize Consensus", key=f"consensus_{round_idx}_{match_idx}"):
                                with st.spinner("Synthesizing..."):
                                    try:
                                        from agents.debate_arena import DebateArena
                                        arena = DebateArena()
                                        hybrid = arena.synthesize_consensus(match, "research")
                                        if hybrid:
                                            st.success(f"Hybrid: **{hybrid.get('title', '')}**")
                                            st.write(f"**Method:** {hybrid.get('method', '')[:200]}")
                                    except Exception as e:
                                        st.error(f"Synthesis failed: {e}")

            # Refine Losers button
            st.divider()
            if st.button("Refine Losing Ideas (Gen 1)", type="secondary", key="refine_losers_btn"):
                with st.spinner("Refining losing ideas with debate feedback..."):
                    try:
                        from agents.debate_arena import DebateArena
                        arena = DebateArena()
                        refined = arena.refine_losers(tournament_data, "research")
                        if refined:
                            st.success(f"Refined {len(refined)} ideas!")
                            for ri, r in enumerate(refined):
                                with st.expander(f"Gen 1: {r.get('title', '')}"):
                                    st.write(f"**Method:** {r.get('method', '')[:300]}")
                                    st.write(f"**Parent:** {r.get('parent_title', '')}")
                        else:
                            st.info("No ideas could be refined.")
                    except Exception as e:
                        st.error(f"Refinement failed: {e}")

    # ── Papers Tab ────────────────────────────────────────────────────────
    with tab_papers:
        st.subheader("Paper Generator")
        st.caption("Generate a full academic paper draft from any idea.")

        if not ideas:
            st.info("No ideas available. Run a pipeline first.")
        else:
            sorted_for_paper = sorted(ideas, key=lambda x: x.get("quality_score", 0), reverse=True)
            paper_idea_titles = [f"{i.get('title', 'Untitled')} (q={i.get('quality_score', 0):.2f})" for i in sorted_for_paper]

            # Single paper generation
            selected_paper_idx = st.selectbox(
                "Select an idea to generate a paper:", range(len(paper_idea_titles)),
                format_func=lambda i: paper_idea_titles[i],
                key="paper_idea_select",
            )

            # Batch paper generation
            st.divider()
            st.markdown("### Batch Generation")
            batch_n = st.slider("Generate papers for top N ideas:", 1, min(10, len(ideas)), 3, key="batch_n")
            if st.button(f"Batch Generate Top {batch_n} Papers", key="batch_papers_btn"):
                progress = st.progress(0)
                status = st.empty()
                for bi in range(batch_n):
                    idea_dict = sorted_for_paper[bi]
                    status.text(f"Generating paper {bi+1}/{batch_n}: {idea_dict.get('title', '')}...")
                    try:
                        import config as _cfg
                        from agents.paper_generator import PaperGenerator
                        from models.idea import Idea as IdeaModel
                        idea_obj = IdeaModel(
                            title=idea_dict.get("title", ""), motivation=idea_dict.get("motivation", ""),
                            method=idea_dict.get("method", ""), hypothesis=idea_dict.get("hypothesis", ""),
                            resources=idea_dict.get("resources", ""), expected_outcome=idea_dict.get("expected_outcome", ""),
                            risk_assessment=idea_dict.get("risk_assessment", ""),
                            source_strategy=idea_dict.get("source_strategy", ""),
                            methodology_type=idea_dict.get("methodology_type"),
                            novelty_level=idea_dict.get("novelty_level"),
                            quality_score=idea_dict.get("quality_score", 0),
                        )
                        gen = PaperGenerator()
                        paper_md = gen.generate_paper(idea_obj, dag_summary, None)
                        uid = st.session_state.get("user_id")
                        if uid:
                            db.save_paper(uid, idea_obj.title, paper_md)
                            db_cache.invalidate_user_papers()
                    except Exception as e:
                        st.warning(f"Paper {bi+1} failed: {e}")
                    progress.progress((bi + 1) / batch_n)
                status.text(f"Done! Generated {batch_n} papers.")
                st.rerun()

            st.divider()

            if st.button("Generate Paper Draft", type="primary", key="gen_paper_btn"):
                with st.spinner("Generating paper (6 sections)..."):
                    try:
                        import config as _cfg
                        from agents.paper_generator import PaperGenerator
                        from models.idea import Idea as IdeaModel

                        idea_dict = sorted_for_paper[selected_paper_idx]
                        idea_obj = IdeaModel(
                            title=idea_dict.get("title", ""),
                            motivation=idea_dict.get("motivation", ""),
                            method=idea_dict.get("method", ""),
                            hypothesis=idea_dict.get("hypothesis", ""),
                            resources=idea_dict.get("resources", ""),
                            expected_outcome=idea_dict.get("expected_outcome", ""),
                            risk_assessment=idea_dict.get("risk_assessment", ""),
                            source_strategy=idea_dict.get("source_strategy", ""),
                            methodology_type=idea_dict.get("methodology_type"),
                            novelty_level=idea_dict.get("novelty_level"),
                            quality_score=idea_dict.get("quality_score", 0),
                            debate_rank=idea_dict.get("debate_rank"),
                        )
                        gen = PaperGenerator()
                        paper_md = gen.generate_paper(idea_obj, dag_summary, idea_dict.get("debate_history"))

                        st.markdown("---")
                        st.markdown(paper_md)

                        # Save to DB
                        user_id = st.session_state.get("user_id")
                        if user_id:
                            db.save_paper(user_id, idea_obj.title, paper_md)
                            db_cache.invalidate_user_papers()

                        st.download_button(
                            "Download Paper (.md)", paper_md,
                            file_name=f"paper_{idea_obj.title[:30].replace(' ', '_')}.md",
                            mime="text/markdown",
                        )
                    except Exception as e:
                        st.error(f"Paper generation failed: {e}")

        # Show previously generated papers
        user_id = st.session_state.get("user_id")
        if user_id:
            saved_papers = db_cache.get_user_papers(user_id)
            if saved_papers:
                st.divider()
                st.markdown("### Saved Papers")
                for p in saved_papers:
                    with st.expander(f"{p['idea_title']} — {p['created_at'][:16]}"):
                        paper_content = db_cache.load_paper(p["id"], user_id)
                        if paper_content:
                            st.markdown(paper_content[:2000] + ("..." if len(paper_content) > 2000 else ""))
                            st.download_button(
                                "Download", paper_content,
                                file_name=f"paper_{p['id']}.md",
                                mime="text/markdown", key=f"dl_paper_{p['id']}",
                            )

    # ── Recommend Tab ─────────────────────────────────────────────────────
    with tab_recommend:
        st.subheader("Smart Topic Recommender")
        st.caption("AI-powered suggestions for your next research direction based on your portfolio.")

        user_id = st.session_state.get("user_id")
        if user_id:
            all_user_ideas = db_cache.get_all_user_ideas(user_id)
            past_topics = sorted(set(i.get("_topic", "") for i in all_user_ideas if i.get("_topic")))

            if not all_user_ideas:
                st.info("Run some pipelines first to get personalized recommendations.")
            else:
                st.markdown(f"**Your portfolio:** {len(all_user_ideas)} ideas across {len(past_topics)} topics")

                if st.button("Get Recommendations", type="primary", key="recommend_tab_btn"):
                    with st.spinner("Analyzing your portfolio and finding opportunities..."):
                        try:
                            from agents.topic_recommender import TopicRecommender
                            recommender = TopicRecommender()
                            recs = recommender.recommend(all_user_ideas, past_topics)
                            st.session_state["recommendations"] = recs
                        except Exception as e:
                            st.error(f"Recommendation failed: {e}")

                recs = st.session_state.get("recommendations", [])
                if recs:
                    for i, rec in enumerate(recs):
                        impact = rec.get("gap_type") or rec.get("expected_impact", "medium")
                        impact_icon = "🔴" if impact == "high" else "🟡" if impact == "medium" else "🟢"
                        topic = rec.get("topic", "Unknown")
                        rationale = rec.get("rationale", "")

                        with st.expander(f"{impact_icon} {topic}"):
                            st.write(rationale)
                            gap = rec.get("gap_type", "")
                            if gap:
                                st.caption(f"**Gap type:** {gap.replace('_', ' ')}")
                            builds_on = rec.get("builds_on", "")
                            if builds_on:
                                st.caption(f"**Builds on:** {builds_on}")

                            if st.button(f"Use this topic", key=f"use_topic_{i}"):
                                st.session_state["_prefill_topic"] = topic
                                st.success(f"Topic set! Go to sidebar and click Run Automated Scientist.")
        else:
            st.info("Log in to get personalized recommendations.")

    # ── Cross-Domain Tab ──────────────────────────────────────────────────
    with tab_cross:
        st.subheader("Cross-Domain Synthesis")
        st.caption("Combine ideas from different research domains to generate novel hybrid ideas.")

        user_id = st.session_state.get("user_id")
        if user_id:
            saved_results = db_cache.get_user_results(user_id)
            if len(saved_results) < 2:
                st.info("You need at least 2 saved results from different topics. Run more pipelines and save them!")
            else:
                result_labels = [
                    f"{r['topic'][:50]} ({r['ideas_count']} ideas, {r['created_at'][:10]})"
                    for r in saved_results
                ]
                selected_indices = st.multiselect(
                    "Select 2-3 saved runs to cross-pollinate:",
                    range(len(result_labels)),
                    format_func=lambda i: result_labels[i],
                    max_selections=3,
                    key="cross_domain_select",
                )

                if len(selected_indices) >= 2 and st.button(
                    "Run Cross-Domain Synthesis", type="primary", key="cross_domain_btn"
                ):
                    with st.spinner("Synthesizing across domains..."):
                        try:
                            from agents.cross_domain import CrossDomainSynthesizer

                            runs_data = []
                            for idx in selected_indices:
                                r = saved_results[idx]
                                full_result = db_cache.load_result(r["id"], user_id)
                                if full_result:
                                    runs_data.append(full_result)

                            synth = CrossDomainSynthesizer()
                            cd_result = synth.synthesize(runs_data)

                            # Display results
                            if cd_result.hybrid_ideas:
                                st.markdown("### Hybrid Ideas")
                                for hi in cd_result.hybrid_ideas:
                                    with st.expander(f"{hi.get('title', 'Hybrid Idea')}"):
                                        st.write(f"**Method:** {hi.get('method', '')}")
                                        st.write(f"**Hypothesis:** {hi.get('hypothesis', '')}")
                                        st.write(f"**Motivation:** {hi.get('motivation', '')}")

                            if cd_result.challenge_exchanges:
                                st.markdown("### Cross-Domain Challenges")
                                for ce in cd_result.challenge_exchanges:
                                    with st.expander(
                                        f"{ce.get('challenger_domain', '')[:25]} challenges "
                                        f"\"{ce.get('idea_title', '')}\""
                                    ):
                                        st.markdown("**Challenge:**")
                                        st.write(ce.get("challenge", ""))
                                        st.markdown("**Defense:**")
                                        st.write(ce.get("defense", ""))
                        except Exception as e:
                            st.error(f"Cross-domain synthesis failed: {e}")
        else:
            st.info("Log in to use cross-domain synthesis.")

    # ── History Tab ────────────────────────────────────────────────────────
    with tab_history:
        st.subheader("Pipeline Run History")
        st.caption("Compare quality across runs and track improvement over time.")

        user_id_hist = st.session_state.get("user_id")
        if user_id_hist:
            past_runs = db_cache.get_user_results(user_id_hist)
            if not past_runs:
                st.info("No saved runs yet. Save your results to build a history.")
            else:
                st.markdown(f"**{len(past_runs)} saved runs**")

                # History table
                for i, run in enumerate(past_runs[:20]):
                    cols_h = st.columns([3, 1, 1, 1])
                    cols_h[0].write(f"**{run['topic'][:50]}**")
                    cols_h[1].write(f"Coverage: {run.get('coverage', 0):.1%}")
                    cols_h[2].write(f"Ideas: {run.get('ideas_count', 0)}")
                    cols_h[3].write(f"{run.get('created_at', '')[:10]}")

                # Quality trend across runs
                if len(past_runs) >= 2:
                    try:
                        import plotly.graph_objects as go
                        run_dates = [r.get("created_at", "")[:10] for r in reversed(past_runs[:20])]
                        run_coverages = [r.get("coverage", 0) * 100 for r in reversed(past_runs[:20])]
                        run_ideas = [r.get("ideas_count", 0) for r in reversed(past_runs[:20])]

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=run_dates, y=run_coverages,
                            mode="lines+markers", name="Coverage %",
                            line=dict(color="#2ecc71", width=3),
                        ))
                        fig.add_trace(go.Bar(
                            x=run_dates, y=run_ideas,
                            name="Ideas Count", marker_color="rgba(52, 152, 219, 0.5)",
                            yaxis="y2",
                        ))
                        fig.update_layout(
                            title="Run History: Coverage & Ideas Over Time",
                            template="plotly_dark", height=350,
                            yaxis=dict(title="Coverage %"),
                            yaxis2=dict(title="Ideas", overlaying="y", side="right"),
                            legend=dict(orientation="h", y=1.1),
                            margin=dict(t=50, b=40),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except ImportError:
                        pass

                # Load & compare past runs
                st.markdown("### Compare Past Runs")
                compare_runs = st.multiselect(
                    "Select 2 runs to compare:",
                    options=list(range(len(past_runs[:10]))),
                    format_func=lambda i: f"{past_runs[i]['topic'][:40]} ({past_runs[i].get('created_at', '')[:10]})",
                    max_selections=2,
                    key="history_compare",
                )
                if len(compare_runs) == 2:
                    run_a = db_cache.load_result(past_runs[compare_runs[0]]["id"], user_id_hist)
                    run_b = db_cache.load_result(past_runs[compare_runs[1]]["id"], user_id_hist)
                    if run_a and run_b:
                        hc1, hc2 = st.columns(2)
                        for col, run_data, run_meta in [(hc1, run_a, past_runs[compare_runs[0]]), (hc2, run_b, past_runs[compare_runs[1]])]:
                            with col:
                                st.markdown(f"**{run_meta['topic'][:40]}**")
                                st.metric("Coverage", f"{run_data.get('coverage', 0):.1%}")
                                st.metric("Ideas", len(run_data.get("ideas", [])))
                                run_stats = run_data.get("stats", {})
                                st.metric("Quality", f"{run_stats.get('quality_mean', 0):.3f}")
                                st.metric("Cost", f"${run_stats.get('estimated_cost_usd', 0):.4f}")
        else:
            st.info("Log in to see your pipeline history.")

    # ── Log Tab ───────────────────────────────────────────────────────────
    # ── Tab: Research Proposal ──────────────────────────────────────────────
    with tab_proposal:
        st.subheader("Research Proposal Export")
        if ideas:
            # Let user pick which idea to generate a proposal for
            idea_titles = [f"{i+1}. {idea.get('title', '?')}" for i, idea in enumerate(ideas)]
            selected_idx = st.selectbox("Select idea for proposal", range(len(idea_titles)),
                                        format_func=lambda i: idea_titles[i], key="proposal_idea_select")
            selected_idea = ideas[selected_idx]

            if st.button("Generate Proposal", type="primary", key="gen_proposal_btn"):
                try:
                    from growth import generate_proposal_markdown, export_proposal_docx
                    _dag_papers = []
                    _dag_s = results.get("dag_summary", {})
                    if isinstance(_dag_s, dict):
                        _dag_papers = _dag_s.get("papers", [])
                    _topic = results.get("topic", "")
                    _md = generate_proposal_markdown(selected_idea, _dag_papers, _topic)
                    st.session_state["_proposal_md"] = _md
                    st.session_state["_proposal_idea"] = selected_idea
                    st.success("Proposal generated!")
                except Exception as e:
                    st.error(f"Error: {e}")

            if st.session_state.get("_proposal_md"):
                st.markdown(st.session_state["_proposal_md"])

                # Download buttons
                dl_col1, dl_col2 = st.columns(2)
                with dl_col1:
                    st.download_button(
                        "Download Markdown",
                        st.session_state["_proposal_md"],
                        file_name="research_proposal.md",
                        mime="text/markdown",
                        key="dl_proposal_md",
                    )
                with dl_col2:
                    try:
                        from growth import export_proposal_docx
                        _docx_bytes = export_proposal_docx(
                            st.session_state.get("_proposal_idea", selected_idea),
                            dag_papers=[], topic=results.get("topic", ""),
                        )
                        if _docx_bytes:
                            st.download_button(
                                "Download DOCX",
                                _docx_bytes,
                                file_name="research_proposal.docx",
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                key="dl_proposal_docx",
                            )
                    except Exception:
                        pass
        else:
            st.info("Run the pipeline first to generate ideas for a proposal.")

    # ── Tab: Idea Evolution Tree ─────────────────────────────────────────────
    with tab_evolution:
        st.subheader("Idea Evolution Tree")
        if ideas:
            try:
                from growth import build_evolution_tree, render_evolution_tree_plotly
                _tree = build_evolution_tree(ideas)
                st.metric("Total Ideas", _tree["total_ideas"])
                st.metric("Root Ideas", _tree["total_roots"])

                _tree_fig = render_evolution_tree_plotly(ideas)
                if _tree_fig:
                    st.plotly_chart(_tree_fig, use_container_width=True, key="evolution_tree")
                else:
                    st.info("Install plotly for interactive evolution tree.")

                # Show lineage for ideas with parents
                _children = [i for i in ideas if i.get("parent_title")]
                if _children:
                    st.markdown("**Idea Lineage:**")
                    for c in _children:
                        _parent = c.get("parent_title", "?")
                        _child = c.get("title", "?")
                        _q_before = c.get("generation", 0) - 1
                        _q_after = c.get("quality_score", 0)
                        st.caption(f"  {_parent} → **{_child}** (gen {c.get('generation', '?')}, q={_q_after:.2f})")
                else:
                    st.caption("No idea revisions detected. Ideas with parent_title will show lineage here.")
            except Exception as e:
                st.error(f"Evolution tree error: {e}")
        else:
            st.info("Run the pipeline first to see idea evolution.")

    # ── Tab: Community (Trending Feed) ───────────────────────────────────────
    with tab_community:
        st.subheader("Community Trending Feed")
        try:
            from growth import render_trending_feed
            render_trending_feed(st)
        except Exception as e:
            st.error(f"Trending feed error: {e}")

    with tab_log:
        st.subheader("Progress Log")
        log_text = "\n".join(st.session_state.progress_log)
        # `label_visibility="collapsed"` keeps the visual layout but
        # gives screen readers + Chrome a non-empty label to work with.
        # (st.text_area doesn't support `autocomplete`; that's a text_input
        # parameter only — see Streamlit's API.)
        st.text_area(
            "Pipeline progress log",
            value=log_text,
            height=400,
            disabled=True,
            label_visibility="collapsed",
            key="progress_log_view",
        )

    # ── Auto-save results on first display (skip if thread already saved) ─
    if not st.session_state.get("result_saved") and ideas and not results.get("_auto_saved"):
        uid_autosave = st.session_state.get("user_id")
        if uid_autosave:
            try:
                db.save_result(
                    user_id=uid_autosave,
                    topic=results.get("topic", "Untitled"),
                    coverage=results.get("coverage", 0.0),
                    ideas_count=len(ideas),
                    results_dict=results,
                )
                db_cache.invalidate_user_results()
            except Exception:
                pass
    st.session_state.result_saved = True

    # ── Download & Save & Export ─────────────────────────────────────────
    st.divider()
    st.markdown(
        '<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">'
        '<span style="font-size:20px">📦</span>'
        '<span style="font-size:18px;font-weight:700;color:#0c4a6e">Export & Save</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    _topic_slug = results.get("topic", "results")[:30].replace(" ", "_")

    row1_c1, row1_c2, row1_c3 = st.columns(3)
    with row1_c1:
        if st.session_state.get("result_saved"):
            st.success("Saved!")
        else:
            if st.button("Save to My Results", type="primary", use_container_width=True):
                user_id = st.session_state.get("user_id")
                if user_id:
                    db.save_result(
                        user_id=user_id,
                        topic=results.get("topic", "Untitled"),
                        coverage=results.get("coverage", 0.0),
                        ideas_count=len(results.get("ideas", [])),
                        results_dict=results,
                    )
                    db_cache.invalidate_user_results()
                    st.session_state.result_saved = True
                    st.rerun()
    with row1_c2:
        json_str = json.dumps(results, indent=2, ensure_ascii=False, default=str)
        st.download_button(
            label="Download JSON",
            data=json_str,
            file_name=f"ideagraph_{_topic_slug}.json",
            mime="application/json",
            type="secondary", use_container_width=True,
        )
    with row1_c3:
        try:
            from report_exporter import generate_report, HAS_DOCX
            if HAS_DOCX:
                docx_bytes = generate_report(
                    results, tournament_data,
                    username=st.session_state.get("username", "Researcher"),
                )
                st.download_button(
                    "Export Report (.docx)", docx_bytes,
                    file_name=f"ideagraph_report_{_topic_slug}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    type="secondary", use_container_width=True,
                )
            else:
                st.caption("Install python-docx for DOCX export")
        except Exception as e:
            st.caption(f"Report: {e}")

    # ── Export options (PDF + HTML + MD + ZIP) ───────────────────────────
    try:
        from export import (
            export_ideas_html, export_ideas_markdown, export_full_zip,
            export_paper_html, HAS_REPORTLAB,
        )
        _topic_name = results.get("topic", "research")

        # Row 1: PDF exports (primary)
        if HAS_REPORTLAB and ideas:
            st.markdown('<div style="font-size:13px;font-weight:700;color:#0369a1;text-transform:uppercase;letter-spacing:0.05em;margin:4px 0 8px 0">📄 PDF Downloads</div>', unsafe_allow_html=True)
            pdf1, pdf2 = st.columns(2)
            from export import generate_ideas_pdf, generate_paper_pdf

            with pdf1:
                try:
                    ideas_pdf = generate_ideas_pdf(ideas, _topic_name, stats)
                    st.download_button(
                        "Ideas Report (.pdf)",
                        data=ideas_pdf,
                        file_name=f"ideas_report_{_topic_slug}.pdf",
                        mime="application/pdf",
                        type="primary", use_container_width=True,
                    )
                except Exception as e:
                    st.caption(f"PDF error: {e}")

            with pdf2:
                final_paper_exp = results.get("final_paper")
                if final_paper_exp and final_paper_exp.get("markdown"):
                    try:
                        paper_pdf = generate_paper_pdf(final_paper_exp, _topic_name)
                        st.download_button(
                            "Paper (.pdf)",
                            data=paper_pdf,
                            file_name=f"paper_{_topic_slug}.pdf",
                            mime="application/pdf",
                            type="primary", use_container_width=True,
                        )
                    except Exception as e:
                        st.caption(f"Paper PDF error: {e}")
                else:
                    st.caption("No paper to export")

        # Row 2: Other formats
        st.markdown('<div style="font-size:13px;font-weight:700;color:#0369a1;text-transform:uppercase;letter-spacing:0.05em;margin:4px 0 8px 0">📋 Other Formats</div>', unsafe_allow_html=True)
        row2_c1, row2_c2, row2_c3, row2_c4 = st.columns(4)

        with row2_c1:
            html_report = export_ideas_html(ideas, _topic_name, stats)
            st.download_button(
                "Ideas (.html)",
                data=html_report,
                file_name=f"ideas_report_{_topic_slug}.html",
                mime="text/html",
                type="secondary", use_container_width=True,
            )

        with row2_c2:
            md_report = export_ideas_markdown(ideas, _topic_name, stats)
            st.download_button(
                "Ideas (.md)",
                data=md_report,
                file_name=f"ideas_report_{_topic_slug}.md",
                mime="text/markdown",
                type="secondary", use_container_width=True,
            )

        with row2_c3:
            zip_data = export_full_zip(results, _topic_name)
            st.download_button(
                "Full Export (.zip)",
                data=zip_data,
                file_name=f"ideagraph_full_{_topic_slug}.zip",
                mime="application/zip",
                type="secondary", use_container_width=True,
            )

        with row2_c4:
            final_paper_html = results.get("final_paper")
            if final_paper_html and final_paper_html.get("markdown"):
                paper_html = export_paper_html(final_paper_html, _topic_name)
                st.download_button(
                    "Paper (.html)",
                    data=paper_html,
                    file_name=f"paper_{_topic_slug}.html",
                    mime="text/html",
                    type="secondary", use_container_width=True,
                )
            else:
                st.download_button(
                    "JSON (raw)",
                    data=json.dumps(results.get("stats", {}), indent=2, default=str),
                    file_name=f"stats_{_topic_slug}.json",
                    mime="application/json",
                    type="secondary", use_container_width=True,
                )

        # ── Row 3: Research tool exports (Zotero + Notion) ───────────────
        if ideas:
            st.markdown('<div style="font-size:13px;font-weight:700;color:#0369a1;text-transform:uppercase;letter-spacing:0.05em;margin:4px 0 8px 0">🔬 Research Tools</div>', unsafe_allow_html=True)
            row3_c1, row3_c2 = st.columns(2)
            try:
                from export import export_ideas_bibtex, export_ideas_notion

                with row3_c1:
                    bibtex = export_ideas_bibtex(ideas, _topic_name)
                    st.download_button(
                        "📚 Zotero (.bib)",
                        data=bibtex,
                        file_name=f"ideagraph_{_topic_slug}.bib",
                        mime="application/x-bibtex",
                        type="secondary", use_container_width=True,
                        help="Import into Zotero or Mendeley as BibTeX",
                    )

                with row3_c2:
                    notion_md = export_ideas_notion(ideas, _topic_name, stats)
                    st.download_button(
                        "📝 Notion (.md)",
                        data=notion_md,
                        file_name=f"ideagraph_{_topic_slug}_notion.md",
                        mime="text/markdown",
                        type="secondary", use_container_width=True,
                        help="Import into Notion as a formatted page",
                    )
            except Exception as e:
                st.caption(f"Tool export: {e}")

        # ── Row 4: LaTeX export ──────────────────────────────────────────
        st.markdown('<div style="font-size:13px;font-weight:700;color:#0369a1;text-transform:uppercase;letter-spacing:0.05em;margin:4px 0 8px 0">🎓 Academic Exports</div>', unsafe_allow_html=True)
        _latex_col1, _latex_col2 = st.columns(2)
        with _latex_col1:
            try:
                from export import export_ideas_latex
                _dag_papers_export = []
                _ds = results.get("dag_summary", {})
                if isinstance(_ds, dict):
                    _dag_papers_export = _ds.get("papers", [])
                latex_src = export_ideas_latex(
                    ideas, _topic_name, stats, dag_papers=_dag_papers_export,
                )
                st.download_button(
                    "LaTeX (.tex)",
                    latex_src,
                    file_name=f"ideagraph_{_topic_slug}.tex",
                    mime="text/x-tex",
                    type="secondary", use_container_width=True,
                    help="Compilable LaTeX — import directly into Overleaf",
                )
            except Exception as e:
                st.caption(f"LaTeX: {e}")
        with _latex_col2:
            try:
                from growth import generate_proposal_markdown
                _best_idea = max(ideas, key=lambda x: x.get("quality_score", 0))
                _prop_md = generate_proposal_markdown(_best_idea, topic=_topic_name)
                st.download_button(
                    "Proposal (.md)",
                    _prop_md,
                    file_name=f"proposal_{_topic_slug}.md",
                    mime="text/markdown",
                    type="secondary", use_container_width=True,
                    help="2-page research proposal for your best idea",
                )
            except Exception:
                pass

    except Exception as e:
        st.caption(f"Export: {e}")

    # ── "Run Again" button ──────────────────────────────────────────────────
    st.divider()
    st.markdown(
        '<div style="background:#f0f9ff;border:1px solid #bae6fd;border-radius:10px;'
        'padding:16px 20px;margin:4px 0 12px 0">'
        '<div style="color:#0369a1;font-size:14px;margin-bottom:8px">'
        '💡 Want to explore further? Adjust settings in the sidebar and run again.</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    _ra_col1, _ra_col2, _ra_col3 = st.columns([1, 1, 2])
    with _ra_col1:
        if st.button("🔄 Run Again", type="primary", use_container_width=True, key="run_again_btn"):
            st.session_state.done = False
            st.session_state.results = None
            st.session_state.result_saved = False
            st.session_state.error = None
            st.session_state.progress_log = []
            st.rerun()
    with _ra_col2:
        if st.button("🏠 Home", type="secondary", use_container_width=True, key="home_btn"):
            st.session_state.done = False
            st.session_state.results = None
            st.rerun()

    # ── Admin Dashboard (admin users only) ───────────────────────────────────
    _admin_uid = st.session_state.get("user_id")
    if _admin_uid:
        try:
            from admin_dashboard import is_admin, render_admin_dashboard
            if is_admin(_admin_uid):
                with st.expander("Admin Dashboard"):
                    render_admin_dashboard(st)
        except Exception:
            pass

elif not st.session_state.running and not st.session_state.done:
    # ── Engagement Hub (home screen with daily hook) ──────────────────────
    import engagement

    _uid_hub = st.session_state.get("user_id")
    _username = st.session_state.get("username", "Researcher")

    # ── Welcome banner ──────────────────────────────────────────────────────
    st.markdown(
        f'<div style="background:linear-gradient(135deg,#f0f9ff,#e0f2fe);'
        f'border:1px solid #bae6fd;border-radius:14px;padding:24px 22px;margin-bottom:16px">'
        f'<div style="font-size:24px;font-weight:700;color:#0c4a6e;margin-bottom:6px">'
        f'👋 Welcome back, {_username}</div>'
        f'<div style="font-size:14px;color:#0369a1;line-height:1.5">'
        f'Enter a research topic in the sidebar and click '
        f'<b>Run Automated Scientist</b> to generate ideas.</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Onboarding tutorial (new users) ───────────────────────────────────
    if _uid_hub:
        onboard = engagement.get_onboarding_state(_uid_hub)
        if not onboard.get("tutorial_dismissed") and not onboard.get("completed"):
            step_idx = min(onboard.get("step", 0), len(engagement.ONBOARDING_STEPS) - 1)
            step = engagement.ONBOARDING_STEPS[step_idx]
            _total_steps = len(engagement.ONBOARDING_STEPS)
            _pct = int((step_idx / max(_total_steps, 1)) * 100)
            st.markdown(
                f'<div style="background:linear-gradient(135deg,#f0f9ff,#e0f2fe);'
                f'border:1px solid #7dd3fc;border-radius:12px;padding:16px 18px;'
                f'margin-bottom:10px">'
                # Header
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'margin-bottom:8px">'
                f'<div>'
                f'<span style="font-size:22px;margin-right:6px">{step["emoji"]}</span>'
                f'<span style="font-size:15px;font-weight:700;color:#0c4a6e">'
                f'Step {step_idx + 1}/{_total_steps}: {step["title"]}</span>'
                f'</div>'
                f'<span style="font-size:13px;font-weight:700;color:#0284c7">{_pct}%</span>'
                f'</div>'
                # Description
                f'<div style="font-size:13px;color:#334155;margin-bottom:10px;line-height:1.5">'
                f'{step["desc"]}</div>'
                # Progress bar
                f'<div style="background:#bae6fd;border-radius:6px;height:10px;overflow:hidden">'
                f'<div style="background:linear-gradient(90deg,#0ea5e9,#38bdf8);'
                f'height:100%;width:{_pct}%;border-radius:6px;'
                f'transition:width 0.5s ease"></div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            _ob_col1, _ob_col2 = st.columns([1, 1])
            with _ob_col1:
                if st.button("Next →", key="ob_next", type="primary", use_container_width=True):
                    engagement.advance_onboarding(_uid_hub)
                    st.rerun()
            with _ob_col2:
                if st.button("Skip tutorial", key="ob_skip", use_container_width=True):
                    engagement.dismiss_onboarding(_uid_hub)
                    st.rerun()

    # ── Stats banner ──────────────────────────────────────────────────────
    if _uid_hub:
        stats = engagement.get_user_stats(_uid_hub)
        _s_level = stats.get("level", 1)
        _s_xp = stats.get("xp", 0)
        _s_streak = stats.get("current_streak", 0)
        _s_ideas = stats.get("total_ideas", 0)
        _s_runs = stats.get("total_runs", 0)
        _stats_items = [
            ("🏆", "Level", str(_s_level)),
            ("⭐", "XP", f"{_s_xp:,}"),
            ("🔥", "Streak", f"{_s_streak}d"),
            ("💡", "Ideas", str(_s_ideas)),
            ("🎯", "Runs", str(_s_runs)),
        ]
        _stats_html = '<div style="display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap">'
        for _icon, _label, _val in _stats_items:
            _stats_html += (
                f'<div style="flex:1;min-width:80px;background:#f0f9ff;border:1px solid #e0f2fe;'
                f'border-radius:10px;padding:10px 8px;text-align:center">'
                f'<div style="font-size:16px">{_icon}</div>'
                f'<div style="font-size:16px;font-weight:700;color:#0c4a6e">{_val}</div>'
                f'<div style="font-size:10px;color:#0369a1;font-weight:600;'
                f'text-transform:uppercase;letter-spacing:0.04em">{_label}</div>'
                f'</div>'
            )
        _stats_html += '</div>'
        st.markdown(_stats_html, unsafe_allow_html=True)

    st.divider()

    # ── Home tabs: Daily Pick, Achievements, Feed, How it Works ───────────
    hub_tabs = st.tabs(["🌟 Daily Pick", "🧬 Mutation Lab", "🎯 Blind Review", "📜 Manifesto", "🎰 Roulette", "🔮 Prophecy", "🏅 Olympics", "🤝 Collaborators", "🏆 Achievements", "🧠 Mentor", "📣 Feed", "ℹ️ How it Works"])

    # ── Daily Pick Tab ────────────────────────────────────────────────────
    with hub_tabs[0]:
        st.markdown(
            '<div style="margin-bottom:12px">'
            '<span style="font-size:18px;font-weight:700;color:#0c4a6e">🌟 Your Daily Research Idea</span>'
            '<span style="font-size:12px;color:#64748b;margin-left:8px">Refreshed every 24 hours</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        if _uid_hub:
            pick = engagement.generate_daily_pick(_uid_hub)
            if pick:
                q = pick.get("quality_score", 0)
                _dq_color = "#10b981" if q >= 0.7 else "#f59e0b" if q >= 0.4 else "#ef4444"
                _dq_grade = "A+" if q >= 0.8 else "A" if q >= 0.7 else "B+" if q >= 0.6 else "B" if q >= 0.5 else "C" if q >= 0.4 else "D"
                _dq_method = (pick.get("methodology_type") or "?").replace("_", " ").title()
                _dq_novelty = (pick.get("novelty_level") or "?").capitalize()

                st.markdown(
                    f'<div style="background:white;border:1px solid #e0f2fe;border-radius:12px;'
                    f'padding:18px 20px;box-shadow:0 2px 8px rgba(14,165,233,0.08)">'
                    # Title + grade
                    f'<div style="display:flex;justify-content:space-between;align-items:start;margin-bottom:10px">'
                    f'<div style="font-size:17px;font-weight:700;color:#0c4a6e;flex:1">'
                    f'{pick.get("title", "Untitled")}</div>'
                    f'<div style="background:{_dq_color};color:white;font-weight:700;font-size:14px;'
                    f'padding:4px 10px;border-radius:6px;margin-left:10px">{_dq_grade}</div>'
                    f'</div>'
                    # Tags row
                    f'<div style="display:flex;gap:8px;margin-bottom:12px">'
                    f'<span style="background:#f0f9ff;color:#0369a1;font-size:11px;font-weight:600;'
                    f'padding:3px 8px;border-radius:4px">{_dq_method}</span>'
                    f'<span style="background:#f0f9ff;color:#0369a1;font-size:11px;font-weight:600;'
                    f'padding:3px 8px;border-radius:4px">{_dq_novelty}</span>'
                    f'<span style="background:#f0f9ff;color:#0369a1;font-size:11px;font-weight:600;'
                    f'padding:3px 8px;border-radius:4px">q={q:.2f}</span>'
                    f'</div>'
                    # Content
                    f'<div style="font-size:13px;color:#334155;margin-bottom:6px">'
                    f'<b>Motivation:</b> {(pick.get("motivation","") or "N/A")[:200]}</div>'
                    f'<div style="font-size:13px;color:#334155">'
                    f'<b>Method:</b> {(pick.get("method","") or "N/A")[:200]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                dp_col1, dp_col2 = st.columns(2)
                with dp_col1:
                    if st.button("❤️ Love it", key="daily_love", use_container_width=True):
                        engagement.award_xp(_uid_hub, "bookmark_idea")
                        st.success("+5 XP earned!")
                with dp_col2:
                    if st.button("🔄 Generate Similar", key="daily_similar", use_container_width=True):
                        st.info("Tip: Use the sidebar to run a new pipeline with a related topic!")
            else:
                st.info("Run your first pipeline to unlock personalized daily picks!")
        else:
            st.info("Log in to see your daily pick.")

    # ── Mutation Lab Tab ─────────────────────────────────────────────────
    with hub_tabs[1]:
        st.markdown('<div style="font-size:18px;font-weight:700;color:#0c4a6e;margin-bottom:8px">🧬 Idea Mutation Lab</div>', unsafe_allow_html=True)
        st.caption("Click any mutation to see an instant 'what-if' variant of your idea.")

        if _uid_hub:
            try:
                from creative_lab import MUTATION_TYPES, _fallback_mutation
                _all_ideas = db_cache.get_all_user_ideas(_uid_hub)
                if _all_ideas:
                    _idea_titles = [f"{i.get('title','?')}" for i in _all_ideas[:20]]
                    _sel_idx = st.selectbox("Select idea to mutate", range(len(_idea_titles)),
                                           format_func=lambda i: _idea_titles[i], key="mutation_idea")
                    _sel_idea = _all_ideas[_sel_idx]

                    _mut_cols = st.columns(len(MUTATION_TYPES))
                    for _mc, _mut in zip(_mut_cols, MUTATION_TYPES):
                        with _mc:
                            if st.button(f"{_mut['icon']}\n{_mut['label']}", key=f"mut_{_mut['id']}",
                                         use_container_width=True, help=_mut["description"]):
                                _mutated = _fallback_mutation(_sel_idea, _mut["id"])
                                st.session_state["_last_mutation"] = _mutated

                    if st.session_state.get("_last_mutation"):
                        _m = st.session_state["_last_mutation"]
                        st.markdown(
                            f'<div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;'
                            f'padding:14px 16px;margin-top:10px">'
                            f'<div style="font-size:14px;font-weight:700;color:#166534">'
                            f'{_m.get("_mutation_label","Mutation")}: {_m.get("title","")}</div>'
                            f'<div style="font-size:12px;color:#15803d;margin-top:6px">'
                            f'<b>Hypothesis:</b> {(_m.get("hypothesis","") or "")[:150]}</div>'
                            f'<div style="font-size:12px;color:#15803d;margin-top:4px">'
                            f'<b>Method:</b> {(_m.get("method","") or "")[:150]}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("Run a pipeline first to get ideas to mutate.")
            except Exception as e:
                st.error(f"Mutation Lab error: {e}")
        else:
            st.info("Log in to use the Mutation Lab.")

    # ── Blind Peer Review Tab ────────────────────────────────────────────
    with hub_tabs[2]:
        st.markdown('<div style="font-size:18px;font-weight:700;color:#0c4a6e;margin-bottom:8px">🎯 Blind Peer Review</div>', unsafe_allow_html=True)
        st.caption("The mentor takes the opposite stance and tries to destroy your idea. You'll emerge stronger.")

        if _uid_hub:
            try:
                from creative_lab import generate_blind_review
                _all_ideas_br = db_cache.get_all_user_ideas(_uid_hub)
                if _all_ideas_br:
                    _br_titles = [f"{i.get('title','?')}" for i in _all_ideas_br[:20]]
                    _br_idx = st.selectbox("Select idea to review", range(len(_br_titles)),
                                          format_func=lambda i: _br_titles[i], key="blind_review_idea")
                    if st.button("🔥 Destroy My Idea", type="primary", key="run_blind_review"):
                        with st.spinner("Adversarial review in progress..."):
                            _review = generate_blind_review(_all_ideas_br[_br_idx])
                            st.session_state["_blind_review"] = _review

                    if st.session_state.get("_blind_review"):
                        _br = st.session_state["_blind_review"]
                        st.markdown(
                            f'<div style="background:#fef2f2;border:1px solid #fecaca;border-radius:10px;'
                            f'padding:14px 16px;margin-top:10px">'
                            f'<div style="font-size:14px;font-weight:700;color:#991b1b;margin-bottom:8px">'
                            f'⚔️ Verdict: {_br.get("verdict","")}</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown("**Attacks:**")
                        for _atk in _br.get("attacks", []):
                            st.markdown(f"- ❌ {_atk}")
                        st.markdown("**Adversarial Questions:**")
                        for _q in _br.get("questions", []):
                            st.markdown(f"- ❓ {_q}")
                        st.markdown(
                            f'<div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;'
                            f'padding:10px 14px;margin-top:8px">'
                            f'<span style="font-weight:700;color:#166534">✅ Strength found:</span> '
                            f'{_br.get("strength_found","")}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("Run a pipeline first.")
            except Exception as e:
                st.error(f"Blind Review error: {e}")
        else:
            st.info("Log in to use Blind Peer Review.")

    # ── Manifesto Tab ────────────────────────────────────────────────────
    with hub_tabs[3]:
        st.markdown('<div style="font-size:18px;font-weight:700;color:#0c4a6e;margin-bottom:8px">📜 Research Manifesto</div>', unsafe_allow_html=True)
        st.caption("Your auto-generated research identity — shareable as your 'Research DNA.'")

        if _uid_hub:
            try:
                from creative_lab import generate_manifesto, compute_personality
                _all_ideas_man = db_cache.get_all_user_ideas(_uid_hub)
                _manifesto = generate_manifesto(_all_ideas_man, _username)

                st.markdown(
                    f'<div style="background:linear-gradient(135deg,#f0f9ff,#e0f2fe);'
                    f'border:1px solid #7dd3fc;border-radius:14px;padding:20px 22px;'
                    f'margin-bottom:12px">'
                    f'<div style="font-size:13px;color:#0369a1;font-weight:600;'
                    f'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px">'
                    f'{_manifesto.get("style","Researcher")} | {_username}</div>'
                    f'<div style="font-size:15px;color:#0c4a6e;line-height:1.6;font-style:italic">'
                    f'"{_manifesto.get("manifesto","")}"</div>'
                    f'<div style="margin-top:10px;display:flex;gap:6px;flex-wrap:wrap">',
                    unsafe_allow_html=True,
                )
                for _theme in _manifesto.get("themes", []):
                    st.markdown(
                        f'<span style="background:#e0f2fe;color:#0369a1;font-size:11px;'
                        f'padding:2px 8px;border-radius:4px;font-weight:500">{_theme}</span>',
                        unsafe_allow_html=True,
                    )

                # Personality radar
                _pers = compute_personality(_all_ideas_man)
                try:
                    import plotly.graph_objects as go
                    _dims = _pers["dimensions"]
                    _cats = list(_dims.keys()) + [list(_dims.keys())[0]]
                    _vals = list(_dims.values()) + [list(_dims.values())[0]]
                    _fig = go.Figure(go.Scatterpolar(
                        r=_vals, theta=_cats, fill="toself",
                        line=dict(color="#0ea5e9", width=2),
                        fillcolor="rgba(14,165,233,0.15)",
                    ))
                    _fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                        title="Research Personality Profile",
                        height=350, margin=dict(t=40, b=20),
                    )
                    st.plotly_chart(_fig, use_container_width=True, key="personality_radar")
                except ImportError:
                    pass

                st.caption(f"Dominant trait: **{_pers['dominant']}** | Blind spot: **{_pers['blind_spot']}**")
            except Exception as e:
                st.error(f"Manifesto error: {e}")
        else:
            st.info("Log in to generate your manifesto.")

    # ── Roulette Tab ─────────────────────────────────────────────────────
    with hub_tabs[4]:
        st.markdown('<div style="font-size:18px;font-weight:700;color:#0c4a6e;margin-bottom:8px">🎰 Idea Roulette</div>', unsafe_allow_html=True)
        st.caption("Spin to discover a random high-quality idea from the community.")

        if st.button("🎰 SPIN!", type="primary", use_container_width=True, key="spin_roulette"):
            try:
                from creative_lab import spin_roulette
                from growth import get_trending_ideas
                _community = get_trending_ideas(limit=50)
                _comm_as_ideas = [
                    {"title": c["title"], "method_preview": c.get("method_preview",""),
                     "quality_score": c.get("quality_score", 0), "username": c.get("username",""),
                     "methodology_type": c.get("methodology_type","")}
                    for c in _community
                ]
                _spin = spin_roulette(_comm_as_ideas)
                if _spin:
                    st.session_state["_roulette_result"] = _spin
                else:
                    st.info("No community ideas yet. Share some ideas first!")
            except Exception:
                st.info("Share ideas to build the community pool.")

        if st.session_state.get("_roulette_result"):
            _r = st.session_state["_roulette_result"]
            _rq = _r.get("quality_score", 0)
            st.markdown(
                f'<div style="background:linear-gradient(135deg,#fffbeb,#fef3c7);'
                f'border:2px solid #f59e0b;border-radius:12px;padding:16px 18px;'
                f'text-align:center;margin-top:10px">'
                f'<div style="font-size:24px;margin-bottom:4px">🎰</div>'
                f'<div style="font-size:16px;font-weight:700;color:#92400e">'
                f'{_r.get("title","Untitled")}</div>'
                f'<div style="font-size:12px;color:#a16207;margin-top:6px">'
                f'by @{_r.get("username","?")} | q={_rq:.2f} | '
                f'{(_r.get("methodology_type","") or "").replace("_"," ").title()}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Prophecy Tab ─────────────────────────────────────────────────────
    with hub_tabs[5]:
        st.markdown('<div style="font-size:18px;font-weight:700;color:#0c4a6e;margin-bottom:8px">🔮 Idea Prophecy</div>', unsafe_allow_html=True)
        st.caption("Which of your ideas will trend next week?")

        if _uid_hub:
            try:
                from creative_lab import predict_trending
                from growth import get_trending_ideas
                _user_ideas_pr = db_cache.get_all_user_ideas(_uid_hub)
                _comm_ideas_pr = get_trending_ideas(limit=100)
                _comm_as_dicts = [{"title": c["title"], "method": c.get("method_preview","")} for c in _comm_ideas_pr]
                _predictions = predict_trending(_user_ideas_pr, _comm_as_dicts)

                if _predictions:
                    for _p in _predictions:
                        _conf_color = "#10b981" if _p["confidence"] == "high" else "#f59e0b" if _p["confidence"] == "medium" else "#94a3b8"
                        st.markdown(
                            f'<div style="background:white;border:1px solid #e0f2fe;border-radius:8px;'
                            f'padding:10px 14px;margin-bottom:6px;display:flex;justify-content:space-between;align-items:center">'
                            f'<div style="flex:1">'
                            f'<div style="font-size:13px;font-weight:600;color:#0c4a6e">{_p["title"]}</div>'
                            f'<div style="font-size:11px;color:#64748b">{_p["reason"]}</div>'
                            f'</div>'
                            f'<div style="background:{_conf_color};color:white;font-size:12px;font-weight:700;'
                            f'padding:4px 10px;border-radius:6px">{int(_p["prophecy_score"]*100)}%</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("Generate more ideas to unlock predictions.")
            except Exception as e:
                st.caption(f"Prophecy: {e}")
        else:
            st.info("Log in to see predictions.")

    # ── Olympics Tab ─────────────────────────────────────────────────────
    with hub_tabs[6]:
        st.markdown('<div style="font-size:18px;font-weight:700;color:#0c4a6e;margin-bottom:8px">🏅 Idea Olympics</div>', unsafe_allow_html=True)

        try:
            from creative_lab import get_current_olympic_dimension, compute_olympic_score
            _olym = get_current_olympic_dimension()
            st.markdown(
                f'<div style="background:linear-gradient(135deg,#fef3c7,#fffbeb);'
                f'border:2px solid #f59e0b;border-radius:12px;padding:14px 16px;margin-bottom:10px">'
                f'<div style="font-size:20px;text-align:center">{_olym["icon"]}</div>'
                f'<div style="font-size:15px;font-weight:700;color:#92400e;text-align:center">'
                f'This Week: {_olym["title"]}</div>'
                f'<div style="font-size:12px;color:#a16207;text-align:center">{_olym["description"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if _uid_hub:
                _user_ideas_ol = db_cache.get_all_user_ideas(_uid_hub)
                if _user_ideas_ol:
                    _my_score = compute_olympic_score(_user_ideas_ol)
                    st.metric("Your Score", f"{_my_score:.3f}")
                else:
                    st.info("Generate ideas to compete!")
        except Exception as e:
            st.caption(f"Olympics: {e}")

    # ── Collaborators Tab ────────────────────────────────────────────────
    with hub_tabs[7]:
        st.markdown('<div style="font-size:18px;font-weight:700;color:#0c4a6e;margin-bottom:8px">🤝 Find Collaborators</div>', unsafe_allow_html=True)
        st.caption("Researchers with complementary strengths on similar topics.")

        if _uid_hub:
            try:
                from creative_lab import find_collaborators, compute_personality
                _my_ideas = db_cache.get_all_user_ideas(_uid_hub)
                _my_pers = compute_personality(_my_ideas)
                st.caption(f"Your style: **{_my_pers['dominant']}** | Looking for: **{_my_pers['blind_spot']}**")

                # For now show placeholder (real implementation needs multi-user data)
                st.info(
                    "As more researchers join IdeaGraph, we'll match you with "
                    "complementary collaborators based on your research personality. "
                    f"You're a **{_my_pers['dominant']}** — we'll find you a **{_my_pers['blind_spot']}**."
                )
            except Exception:
                st.info("Generate ideas to unlock collaborator matching.")
        else:
            st.info("Log in to find collaborators.")

    # ── Achievements Tab ──────────────────────────────────────────────────
    with hub_tabs[8]:
        st.markdown(
            '<div style="margin-bottom:12px">'
            '<span style="font-size:18px;font-weight:700;color:#0c4a6e">🏆 Your Achievements</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        if _uid_hub:
            unlocked = engagement.get_user_achievements(_uid_hub)
            unlocked_keys = {a["key"] for a in unlocked}

            st.caption(f"Unlocked: **{len(unlocked)}/{len(engagement.ACHIEVEMENTS)}** badges")
            st.progress(len(unlocked) / len(engagement.ACHIEVEMENTS))

            # New achievements popup
            new_ach = st.session_state.get("_new_achievements") or []
            if new_ach:
                st.balloons()
                for a in new_ach:
                    st.success(f"🎉 NEW: **{a['emoji']} {a['name']}** — {a['desc']} (+{a['xp']} XP)")
                st.session_state["_new_achievements"] = None

            # Grid of all achievements
            ach_list = list(engagement.ACHIEVEMENTS.items())
            for row_start in range(0, len(ach_list), 4):
                row_cols = st.columns(4)
                for i, (key, ach) in enumerate(ach_list[row_start:row_start + 4]):
                    with row_cols[i]:
                        is_unlocked = key in unlocked_keys
                        opacity = "1.0" if is_unlocked else "0.3"
                        bg = "#2ecc71" if is_unlocked else "#7f8c8d"
                        st.markdown(
                            f"""
                            <div style="background: rgba(46, 204, 113, 0.1) if {is_unlocked} else rgba(127, 140, 141, 0.1);
                                        border: 2px solid {bg}; border-radius: 10px;
                                        padding: 12px; text-align: center; opacity: {opacity};
                                        margin-bottom: 8px; min-height: 110px;">
                                <div style="font-size: 32px;">{ach['emoji']}</div>
                                <div style="font-weight: bold; font-size: 12px;">{ach['name']}</div>
                                <div style="font-size: 10px; color: #888; margin-top: 4px;">{ach['desc'][:60]}</div>
                                <div style="font-size: 10px; color: #3498db; margin-top: 4px;">+{ach['xp']} XP</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
        else:
            st.info("Log in to see your achievements.")

    # ── Research Mentor Chat Tab ─────────────────────────────────────────
    with hub_tabs[9]:
        st.markdown("#### 🧠 AI Research Mentor")
        st.caption("Ask questions about your research, get idea improvements, explore literature gaps.")

        try:
            from mentor_chat import ResearchMentor

            # Initialize mentor in session
            if "_mentor" not in st.session_state:
                st.session_state["_mentor"] = ResearchMentor()
                st.session_state["_mentor_messages"] = []

            mentor = st.session_state["_mentor"]

            # Load context from results if available
            if st.session_state.get("results") and not st.session_state.get("_mentor_ctx_loaded"):
                uid_m = st.session_state.get("user_id")
                bks = db_cache.get_bookmarks(uid_m) if uid_m else []
                mentor.set_context(
                    results=st.session_state["results"],
                    bookmarks=bks,
                    topic=st.session_state["results"].get("topic", ""),
                )
                st.session_state["_mentor_ctx_loaded"] = True

            # Quick suggestion buttons
            suggestions = mentor.get_quick_suggestions()
            st.markdown("**Quick questions:**")
            q_cols = st.columns(4)
            for qi, suggestion in enumerate(suggestions[:8]):
                col = q_cols[qi % 4]
                if col.button(suggestion[:35], key=f"mentor_q_{qi}", use_container_width=True):
                    st.session_state["_mentor_input"] = suggestion

            st.divider()

            # Chat history
            for msg in st.session_state.get("_mentor_messages", []):
                with st.chat_message("user"):
                    st.write(msg["user"])
                with st.chat_message("assistant", avatar="🧠"):
                    st.write(msg["assistant"])

            # Input
            prefill = st.session_state.pop("_mentor_input", "")
            user_input = st.chat_input("Ask your research mentor...", key="mentor_chat_input")
            if prefill and not user_input:
                user_input = prefill

            if user_input:
                with st.chat_message("user"):
                    st.write(user_input)
                with st.chat_message("assistant", avatar="🧠"):
                    with st.spinner("Thinking..."):
                        response = mentor.chat(user_input)
                    st.write(response)
                st.session_state["_mentor_messages"].append({
                    "user": user_input, "assistant": response,
                })

        except Exception as e:
            st.error(f"Mentor error: {e}")
            st.caption("The mentor requires a working LLM provider. Check your API key in settings.")

    # ── Activity Feed Tab ─────────────────────────────────────────────────
    with hub_tabs[10]:
        st.markdown("#### Community Activity")

        feed_col1, feed_col2 = st.columns([4, 1])
        with feed_col2:
            following_only = st.checkbox("Following only", value=False, key="feed_following")

        if _uid_hub:
            activities = engagement.get_activity_feed(_uid_hub, following_only=following_only, limit=15)
            if activities:
                for act in activities:
                    username = act.get("username", "Someone")
                    content = act.get("content", "")
                    created = act.get("created_at", "")[:16]
                    atype = act.get("activity_type", "")
                    icon = {"pipeline_run": "⚡", "share_idea": "🔗", "bookmark": "🔖", "achievement": "🏆"}.get(atype, "📢")

                    with st.container(border=True):
                        fc1, fc2 = st.columns([4, 1])
                        with fc1:
                            st.markdown(f"{icon} **{username}** — {content}")
                            st.caption(created)
                        with fc2:
                            if username != _username:
                                if st.button("Follow", key=f"follow_{act['id']}", use_container_width=True):
                                    engagement.follow_user(_uid_hub, act["user_id"])
                                    st.success("Following!")
            else:
                st.info("No activity yet. Be the first to run a pipeline!")
        else:
            st.info("Log in to see the activity feed.")

    # ── Agent Lab Tab ──────────────────────────────────────────────────────
    with hub_tabs[11]:  # How it Works (moved)
        st.markdown("#### 🤖 Smart Agent Simulator")
        st.caption(
            "Spawn synthetic users (PhD students, researchers, professors) that autonomously "
            "use the app, provide feedback, and generate optimization suggestions. All data "
            "is saved for offline analysis."
        )

        try:
            import simulated_users as su

            # Config row
            al_col1, al_col2, al_col3, al_col4 = st.columns(4)
            with al_col1:
                n_agents = st.slider("Agents", 1, 20, 5, 1, key="al_n_agents")
            with al_col2:
                topics_per = st.slider("Topics/agent", 1, 5, 2, 1, key="al_topics")
            with al_col3:
                use_real = st.checkbox("Real pipeline", value=False, key="al_real",
                                        help="If unchecked, uses fast mock data (no API calls)")
            with al_col4:
                st.metric("Total runs", n_agents * topics_per)

            # Persona selector
            persona_keys = list(su.PERSONAS.keys())
            persona_labels = [su.PERSONAS[k]["name"] for k in persona_keys]
            selected_personas = st.multiselect(
                "Personas to simulate",
                options=persona_keys,
                default=persona_keys,
                format_func=lambda k: su.PERSONAS[k]["name"],
                key="al_personas",
            )

            # Run button
            if st.button("🚀 Run Agent Simulation", type="primary", use_container_width=True, key="al_run"):
                if not selected_personas:
                    st.error("Select at least one persona.")
                else:
                    sim = su.AgentSimulator()
                    progress_bar = st.progress(0.0, text="Starting agents...")
                    log_area = st.empty()
                    log_messages = []

                    def on_prog(msg):
                        log_messages.append(msg)

                    # Run synchronously (mock is fast; real is slow but we inform user)
                    if use_real:
                        st.warning("Using real pipeline — this may take several minutes per agent.")

                    total = n_agents * topics_per
                    done_count = [0]

                    def step_progress(msg):
                        log_messages.append(msg)
                        done_count[0] += 1
                        progress_bar.progress(
                            done_count[0] / total,
                            text=f"[{done_count[0]}/{total}] Running agents..."
                        )
                        log_area.code("\n".join(log_messages[-8:]))

                    with st.spinner("Agents running..."):
                        sim.run_batch(
                            n_agents=n_agents,
                            topics_per_agent=topics_per,
                            personas=selected_personas,
                            use_real_pipeline=use_real,
                            on_progress=step_progress,
                        )

                    progress_bar.progress(1.0, text="Complete!")

                    # Store in session for display
                    st.session_state["_agent_sim_data"] = {
                        "stats": sim.aggregate_stats(),
                        "pains": sim.pain_point_analysis(),
                        "likes": sim.liked_features_analysis(),
                        "suggestions": sim.generate_optimization_suggestions(),
                        "runs": sim.all_runs,
                    }

                    # Auto-export
                    paths = sim.export_all()
                    st.session_state["_agent_export_paths"] = paths
                    st.success(f"✅ Completed {len(sim.all_runs)} runs! Data saved.")

            # Display results if available
            sim_data = st.session_state.get("_agent_sim_data")
            if sim_data:
                st.divider()

                # Aggregate stats
                stats = sim_data["stats"]
                st.markdown("### 📊 Simulation Results")
                r1, r2, r3, r4, r5 = st.columns(5)
                r1.metric("Total Runs", stats.get("total_runs", 0))
                r2.metric("Success Rate", f"{(stats.get('successful_runs', 0) / max(stats.get('total_runs', 1), 1)):.0%}")
                r3.metric("Avg Quality", f"{stats.get('avg_quality_mean', 0):.3f}")
                r4.metric("Avg Satisfaction", f"{stats.get('avg_satisfaction', 0):.0%}")
                r5.metric("Recommend", f"{stats.get('recommend_rate', 0):.0%}")

                # Pain points & liked features side by side
                pain_col, like_col = st.columns(2)
                with pain_col:
                    st.markdown("##### 😤 Top Pain Points")
                    pains = sim_data["pains"]
                    if pains:
                        for pain, count in list(pains.items())[:5]:
                            st.markdown(f"- **{count}x** — {pain}")
                    else:
                        st.caption("No pain points reported!")

                with like_col:
                    st.markdown("##### ❤️ Top Liked Features")
                    likes = sim_data["likes"]
                    if likes:
                        for like, count in list(likes.items())[:5]:
                            st.markdown(f"- **{count}x** — {like}")
                    else:
                        st.caption("No features highlighted yet.")

                # Optimization suggestions
                suggestions = sim_data["suggestions"]
                if suggestions:
                    st.markdown("##### 🔧 Auto-Optimization Suggestions")
                    for s in suggestions:
                        with st.container(border=True):
                            opt_c1, opt_c2 = st.columns([3, 1])
                            with opt_c1:
                                st.markdown(f"**{s.parameter}**: `{s.old_value}` → `{s.new_value}`")
                                st.caption(s.reason)
                            with opt_c2:
                                st.metric("Expected", f"+{s.expected_improvement:.0%}")

                # Per-persona chart
                runs = sim_data["runs"]
                if runs:
                    st.markdown("##### 📈 Per-Persona Performance")
                    try:
                        import plotly.graph_objects as go
                        from collections import defaultdict

                        by_persona = defaultdict(list)
                        for r in runs:
                            if not r.error:
                                by_persona[r.persona].append(r)

                        persona_names = []
                        avg_qualities = []
                        avg_satisfactions = []
                        for p, runs_list in by_persona.items():
                            persona_names.append(su.PERSONAS.get(p, {}).get("name", p)[:20])
                            avg_qualities.append(sum(r.quality_mean for r in runs_list) / len(runs_list))
                            avg_satisfactions.append(sum(r.satisfaction for r in runs_list) / len(runs_list))

                        fig = go.Figure()
                        fig.add_trace(go.Bar(name="Quality", x=persona_names, y=avg_qualities, marker_color="#3498db"))
                        fig.add_trace(go.Bar(name="Satisfaction", x=persona_names, y=avg_satisfactions, marker_color="#2ecc71"))
                        fig.update_layout(
                            barmode="group", height=350,
                            template="plotly_dark",
                            title="Quality vs Satisfaction per Persona",
                            xaxis=dict(tickangle=-30),
                            margin=dict(l=40, r=20, t=50, b=80),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except ImportError:
                        pass

                # Download buttons
                paths = st.session_state.get("_agent_export_paths", {})
                if paths:
                    st.markdown("##### 📥 Download Simulation Data")
                    dl_col1, dl_col2, dl_col3 = st.columns(3)
                    with dl_col1:
                        try:
                            with open(paths["csv"], "r", encoding="utf-8") as f:
                                st.download_button(
                                    "📊 CSV (Excel)", data=f.read(),
                                    file_name=os.path.basename(paths["csv"]),
                                    mime="text/csv", use_container_width=True,
                                )
                        except Exception:
                            pass
                    with dl_col2:
                        try:
                            with open(paths["json"], "r", encoding="utf-8") as f:
                                st.download_button(
                                    "🧾 JSON", data=f.read(),
                                    file_name=os.path.basename(paths["json"]),
                                    mime="application/json", use_container_width=True,
                                )
                        except Exception:
                            pass
                    with dl_col3:
                        try:
                            with open(paths["markdown"], "r", encoding="utf-8") as f:
                                st.download_button(
                                    "📝 Markdown Report", data=f.read(),
                                    file_name=os.path.basename(paths["markdown"]),
                                    mime="text/markdown", use_container_width=True,
                                )
                        except Exception:
                            pass

        except Exception as e:
            st.error(f"Agent Lab error: {e}")

    # ── How it Works Tab ──────────────────────────────────────────────────
    if False:  # How it Works content folded into tab 11 above
        st.markdown("#### How IdeaGraph Works")
        st.info(
            "Enter a research topic in the sidebar → click **Run Automated Scientist** → "
            "get 10-50 novel research ideas in minutes."
        )
        how_col_a, how_col_b, how_col_c = st.columns(3)
        with how_col_a:
            st.markdown("**🎯 Strategy A — Frontier Extension**")
            st.caption("Proposes next steps beyond the current research frontier.")
        with how_col_b:
            st.markdown("**🌉 Strategy B — Cross-Cluster Bridging**")
            st.caption("Combines insights from two distinct research communities.")
        with how_col_c:
            st.markdown("**🔍 Strategy C — Gap-Filling**")
            st.caption("Fills structural holes between disconnected research streams.")

        st.markdown("---")
        st.markdown("**Earn XP by:**")
        xp_cols = st.columns(3)
        xp_items = [
            ("⚡ Running pipeline", "+50 XP"),
            ("💡 Quality idea (>0.6)", "+10 XP each"),
            ("🔗 Sharing an idea", "+20 XP"),
            ("🔖 Bookmarking", "+5 XP"),
            ("🔥 Daily login streak", "+10-15 XP/day"),
            ("🏆 Unlocking achievement", "+25-2000 XP"),
        ]
        for idx, (label, xp) in enumerate(xp_items):
            with xp_cols[idx % 3]:
                st.markdown(f"{label} — **{xp}**")
