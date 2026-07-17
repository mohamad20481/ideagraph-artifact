"""SEO + responsive-CSS audit for app.py.

These tests treat app.py as a source artifact and assert that the SEO
meta tags + media queries are present and well-formed. They DO NOT
actually render the app — they check the source text. That's enough to
catch accidental regressions (someone removing a meta tag during a
refactor)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

APP = (ROOT / "app.py").read_text(encoding="utf-8")


# ── Page config ─────────────────────────────────────────────────────────────

def test_page_config_present():
    """The single st.set_page_config call must be the first Streamlit
    call (Streamlit raises otherwise)."""
    m = re.search(r"st\.set_page_config\(", APP)
    assert m is not None
    # Must contain page_title, page_icon, layout.
    block = APP[m.start():m.start() + 1500]
    assert "page_title=" in block
    assert "page_icon=" in block
    assert 'layout="wide"' in block or "layout='wide'" in block


def test_page_title_descriptive_for_seo():
    """A bare 'IdeaGraph' page_title is too short for SEO. Should include
    a tagline (≥ 20 chars)."""
    m = re.search(r'page_title="([^"]+)"', APP)
    assert m is not None
    title = m.group(1)
    assert len(title) >= 20, f"page_title too short for SEO: {title!r}"
    assert "IdeaGraph" in title


def test_menu_items_about_set():
    """The About menu should explain what the app does."""
    assert '"About":' in APP
    # Should mention the core concepts so the About dialog is useful.
    block = re.search(r'"About":\s*\(([^)]+)\)', APP, re.DOTALL)
    assert block is not None
    text = block.group(1)
    assert "PhD" in text or "research" in text.lower()


# ── SEO meta tags ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("meta_name,required", [
    ("description",     True),
    ("keywords",        True),
    ("author",          True),
    ("robots",          True),
    ("theme-color",     True),
    ("color-scheme",    True),
    ("format-detection", True),
])
def test_seo_meta_name_tags_present(meta_name, required):
    pattern = rf'<meta\s+name="{re.escape(meta_name)}"[^>]*>'
    assert re.search(pattern, APP), \
        f"Missing <meta name=\"{meta_name}\"> — required for SEO/UX"


def test_seo_robots_indexable():
    """Don't ship a `noindex` by accident on a marketing surface."""
    m = re.search(r'<meta\s+name="robots"\s+content="([^"]+)"', APP)
    assert m is not None
    content = m.group(1).lower()
    assert "index" in content and "noindex" not in content


def test_seo_description_length():
    """Best-practice meta description length is ~120–160 chars, but
    anything in [80, 320] is acceptable for a research-tool landing."""
    # Match only the `_SEO_DESCRIPTION = (...)` Python literal's interior,
    # stopping at the FIRST closing paren on its own line.
    src = APP
    start = src.find("_SEO_DESCRIPTION")
    assert start >= 0, "_SEO_DESCRIPTION constant not found"
    open_paren = src.find("(", start)
    close_paren = src.find(")", open_paren)
    inner = src[open_paren + 1:close_paren]
    # Concatenate every "..." string in the parenthesized block.
    text = "".join(re.findall(r'"([^"]+)"', inner))
    assert text, "could not parse _SEO_DESCRIPTION literal"
    assert 80 <= len(text) <= 320, \
        f"description length {len(text)} chars — outside [80,320]: {text!r}"


# ── Open Graph + Twitter Card ──────────────────────────────────────────────

@pytest.mark.parametrize("og_prop", [
    "og:type", "og:title", "og:description", "og:site_name", "og:locale",
])
def test_open_graph_tags_present(og_prop):
    pattern = rf'<meta\s+property="{re.escape(og_prop)}"[^>]*>'
    assert re.search(pattern, APP), \
        f"Missing <meta property=\"{og_prop}\"> — needed for FB/LinkedIn/Slack previews"


@pytest.mark.parametrize("tw_name", [
    "twitter:card", "twitter:title", "twitter:description",
])
def test_twitter_card_tags_present(tw_name):
    pattern = rf'<meta\s+name="{re.escape(tw_name)}"[^>]*>'
    assert re.search(pattern, APP), \
        f"Missing <meta name=\"{tw_name}\"> — needed for Twitter previews"


def test_twitter_card_type_valid():
    m = re.search(r'<meta\s+name="twitter:card"\s+content="([^"]+)"', APP)
    assert m is not None
    valid = {"summary", "summary_large_image", "app", "player"}
    assert m.group(1) in valid, \
        f"twitter:card content {m.group(1)!r} not in {valid}"


# ── Apple / mobile-app tags ────────────────────────────────────────────────

@pytest.mark.parametrize("apple_tag", [
    "apple-touch-icon",
    "apple-mobile-web-app-title",
    "apple-mobile-web-app-capable",
    "apple-mobile-web-app-status-bar-style",
])
def test_apple_mobile_web_app_tags_present(apple_tag):
    """iOS Safari uses these when the user adds the app to home screen."""
    if apple_tag == "apple-touch-icon":
        pattern = rf'<link\s+rel="{re.escape(apple_tag)}"[^>]*>'
    else:
        pattern = rf'<meta\s+name="{re.escape(apple_tag)}"[^>]*>'
    assert re.search(pattern, APP), f"Missing {apple_tag!r}"


# ── Viewport / accessibility ───────────────────────────────────────────────

def test_viewport_does_not_disable_pinch_zoom():
    """`maximum-scale=1.0, user-scalable=no` is an accessibility
    anti-pattern. Make sure we never re-introduce it."""
    bad = re.search(
        r'<meta[^>]*name="viewport"[^>]*'
        r'(maximum-scale=1\.0|user-scalable=no)',
        APP,
    )
    assert bad is None, (
        "Viewport meta disables pinch-zoom — accessibility violation. "
        "Remove `maximum-scale=1.0` and `user-scalable=no`."
    )


# ── Responsive CSS — media-query coverage ─────────────────────────────────

def test_has_mobile_media_query():
    assert "@media (max-width: 768px)" in APP, \
        "Missing tablet/mobile breakpoint"


def test_has_small_mobile_media_query():
    assert "@media (max-width: 480px)" in APP, \
        "Missing small-phone breakpoint"


def test_has_laptop_breakpoint():
    assert "min-width: 769px" in APP or "max-width: 1024px" in APP, \
        "Missing laptop/iPad-landscape breakpoint"


def test_has_ipad_portrait_breakpoint():
    """iPad portrait (768–820) needs its own rule because most layouts
    are designed for either phone (<768) or laptop (>1024)."""
    assert "(orientation: portrait)" in APP, \
        "Missing iPad-portrait orientation rule"


def test_columns_stack_on_mobile():
    """The key responsive contract: st.columns() must stack vertically
    on phones. Check that the rule exists."""
    # The selector is data-testid="stColumn".
    assert "stColumn" in APP and "flex: 1 1 100%" in APP, (
        "Columns aren't being forced to 100% width on mobile — they "
        "stay side-by-side and overflow."
    )


def test_mobile_inputs_prevent_ios_zoom():
    """iOS Safari zooms when input font-size < 16px. The CSS must set
    16px on text inputs at mobile widths."""
    # Look for `font-size: 16px` near an input selector.
    block = re.search(
        r"@media \(max-width: 768px\).*?(?=@media|$)",
        APP, re.DOTALL,
    )
    assert block is not None
    text = block.group(0)
    assert "16px" in text, \
        "Mobile inputs must be ≥16px font-size to prevent iOS zoom"


def test_touch_targets_meet_44px_minimum():
    """Apple HIG + WCAG: tap targets should be ≥44×44 px."""
    assert "min-height: 44px" in APP, \
        "Missing 44px min-height on tap targets (buttons / inputs)"


# ── Accessibility / motion / contrast / print ─────────────────────────────

def test_respects_prefers_reduced_motion():
    assert "prefers-reduced-motion" in APP, \
        "Missing @media (prefers-reduced-motion) rule — animations " \
        "will run for users who turned them off."


def test_respects_prefers_contrast():
    assert "prefers-contrast" in APP, \
        "Missing @media (prefers-contrast: more) rule for users who " \
        "need stronger contrast."


def test_print_stylesheet_present():
    assert "@media print" in APP, \
        "Missing @media print rule — printed pages will include the " \
        "sidebar, spinners, and download buttons."


def test_print_hides_sidebar_and_chrome():
    """The print stylesheet must hide non-content chrome."""
    m = re.search(r"@media print\s*\{(.+?)^\}", APP, re.DOTALL | re.MULTILINE)
    assert m is not None
    text = m.group(1)
    assert "stSidebar" in text and "display: none" in text


# ── Heading hierarchy ─────────────────────────────────────────────────────

def test_heading_sizes_defined():
    """h1/h2/h3 sizes should be explicit so the document outline reads
    consistently across browsers."""
    assert "h1 {" in APP or "h1 ," in APP
    assert re.search(r"h1\s*\{[^}]*font-size", APP)


# ── JSON-LD structured data (schema.org WebApplication) ────────────────────

def test_json_ld_block_present():
    """Rich snippets / Google Knowledge Graph need a JSON-LD block."""
    assert '<script type="application/ld+json">' in APP


def test_json_ld_is_webapplication_type():
    """The JSON-LD must declare a real schema.org type."""
    m = re.search(
        r'<script type="application/ld\+json">(.+?)</script>',
        APP, re.DOTALL,
    )
    assert m is not None
    body = m.group(1)
    assert "schema.org" in body
    assert '"@type": "WebApplication"' in body


def test_json_ld_has_required_fields():
    """Google's WebApplication rich result wants name, description,
    applicationCategory, operatingSystem at minimum."""
    m = re.search(
        r'<script type="application/ld\+json">(.+?)</script>',
        APP, re.DOTALL,
    )
    body = m.group(1)
    for required in ("name", "description", "applicationCategory",
                       "operatingSystem", "audience", "featureList"):
        assert f'"{required}"' in body, f"JSON-LD missing {required!r}"


# ── Accessibility — skip link, noscript, ARIA main landmark ────────────────

def test_skip_to_content_link_present():
    """Keyboard / screen-reader users need a way to jump past the
    sidebar to the main content."""
    assert "skip-to-content" in APP, \
        "Missing skip-to-content link"
    assert 'href="#main-content"' in APP, \
        "Skip link must target #main-content"


def test_skip_to_content_visible_on_focus():
    """The link must become visible on focus (not stay off-screen)."""
    css = re.search(
        r"\.skip-to-content\s*\{(.+?)\}", APP, re.DOTALL,
    )
    css_focus = re.search(
        r"\.skip-to-content:focus[^{]*\{(.+?)\}", APP, re.DOTALL,
    )
    assert css is not None
    assert css_focus is not None
    # Default hides it; focus must reposition it on-screen.
    assert "top:" in css_focus.group(1) or "top:" in css_focus.group(0)


def test_main_content_landmark_exists():
    """Screen-reader users navigate by ARIA landmarks (main, nav, etc.).
    The `<main>` element + id=main-content provides both."""
    assert 'id="main-content"' in APP
    assert 'role="main"' in APP


def test_noscript_fallback_present():
    """Streamlit apps require JS — be explicit about it for users who
    have JS disabled (privacy-conscious users, some accessibility tools)."""
    assert "<noscript>" in APP
    assert "JavaScript" in APP


def test_focus_visible_styles_defined():
    """Power users / keyboard users need visible focus rings."""
    assert ":focus-visible" in APP


# ── Search + quick-filter chips ────────────────────────────────────────────

def test_search_input_present():
    """The Ideas tab must have a text-search input."""
    assert "Search ideas (matches title + method)" in APP


def test_quick_filter_chips_present():
    """Six one-click filter chips: high quality, substantial, "
    "regenerated, lab-novel, pareto, clear-all."""
    for chip in (
        "🟢 High quality", "🌟 Substantial", "♻️ Regenerated",
        "🧪 Lab-novel", "💎 Pareto", "🧹 Clear all",
    ):
        assert chip in APP, f"Missing quick-filter chip: {chip!r}"


def test_search_filter_chip_keys_consistent():
    """Chips write to specific session_state keys that the filter step
    must read. Defending against divergence."""
    assert '"idea_quality_filter"' in APP
    assert '"idea_filter_novelty"' in APP
    assert '"_idea_chip_regen"' in APP
    assert '"_idea_chip_lab_novel"' in APP
    assert '"idea_sort_mode"' in APP


# ── Friendly empty states ──────────────────────────────────────────────────

def test_empty_state_no_ideas_yet():
    """First-run experience: when ideas==[] the user sees a helpful
    onboarding message, not just `st.info('No ideas were archived yet.')`."""
    assert "No ideas yet — let's generate some" in APP
    assert "Run pipeline" in APP


def test_empty_state_zero_results_after_filter():
    """When filter returns 0 but ideas != []."""
    assert "No ideas match your filters" in APP
    assert "Clear all" in APP


def test_empty_states_have_aria_live():
    """Empty-state banners must announce themselves to screen readers."""
    assert "aria-live='polite'" in APP or 'aria-live="polite"' in APP


# ── Glossary ───────────────────────────────────────────────────────────────

def test_glossary_expander_present():
    """A collapsed-by-default glossary explains the jargon
    (QD grid, methodology, novelty level, source strategy code, etc.)."""
    assert "Glossary — what do these terms mean?" in APP
    # Spot-check the key terms are explained.
    for term in ("Quality score", "Methodology type", "Novelty level",
                  "Source strategy code", "QD grid", "Pareto front",
                  "Generation"):
        assert term in APP, f"Glossary missing entry: {term!r}"
