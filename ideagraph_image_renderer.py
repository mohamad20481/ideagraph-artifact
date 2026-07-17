"""
ideagraph_image_renderer.py — generate visual abstracts for research ideas.

Architecture (provider-pluggable):

    Idea (dict) ──► build_prompt(idea) ──► ImageProvider.generate(prompt)
                                                      │
                                                      ▼
                                            IdeaVisual (url, path, error)
                                                      │
                                                      ▼
                                      DiskCache  ◄────┘  keyed by sha256(prompt)

The default `FluxBFLProvider` targets BlackForest Labs' FLUX endpoint at
api.bfl.ml because that's the simplest path to a real FLUX image with the
fewest hops. To use Runway, Nano Banana, or another provider, write a
subclass implementing `_generate_raw(prompt) -> Dict[image_url, ...]`
and pass it to `NanoBananaImageRenderer(provider=YourProvider(...))`.

Configuration sources (in priority order):
  1. Constructor arguments
  2. NANO_BANANA_API_KEY env var (or BFL_API_KEY for the BFL provider)
  3. config.NANO_BANANA_API_KEY (if config.py defines it)
  4. .nanobang_config file in CWD

No-network mode: when `is_configured()` is False, every `render()` call
returns an IdeaVisual with `error="API key not configured"` rather than
crashing — the UI uses that to show a friendly "add a key" hint.

Public API:
    IdeaVisual                                       → dataclass
    ImageProvider                                    → abstract base
    FluxBFLProvider                                  → default provider (api.bfl.ml)
    NanoBananaImageRenderer                          → main entry point
    build_prompt(idea)                               → str
    cache_key_for_prompt(prompt)                     → str (sha256)
    render_idea_visuals(ideas, api_key=None)         → List[IdeaVisual] (sync batch)
    display_idea_with_visual(idea, st, visual=None)  → renders an idea card with image
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


log = logging.getLogger("ideagraph.image_renderer")


# ── Dependencies (optional — module degrades gracefully) ────────────────────
# aiohttp + tenacity are listed in requirements.txt but the module must
# still import even when missing (e.g. in a fresh checkout that hasn't
# run pip install yet). The sync `requests` path is the always-available
# fallback.

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    requests = None  # type: ignore
    _HAS_REQUESTS = False

try:
    import aiohttp
    _HAS_AIOHTTP = True
except ImportError:
    aiohttp = None  # type: ignore
    _HAS_AIOHTTP = False

try:
    from tenacity import (
        retry, stop_after_attempt, wait_exponential, retry_if_exception_type,
    )
    _HAS_TENACITY = True
except ImportError:
    _HAS_TENACITY = False


# ── Result type ─────────────────────────────────────────────────────────────

@dataclass
class IdeaVisual:
    """Result of rendering one idea into a visual abstract.

    `media_type` distinguishes image vs video:
      - "image"  — image_url / cached_path is a PNG/JPEG/WebP
      - "video"  — image_url / cached_path is an MP4 (for Veo provider)

    The field names stay `image_url` / `cached_path` for backward
    compatibility with the existing UI code; `media_type` is the
    discriminator for `display_idea_with_visual()`.
    """
    idea_title: str
    prompt: str = ""
    image_url: Optional[str] = None       # URL of the media (image or video)
    cached_path: Optional[str] = None     # local file path if downloaded
    cache_key: str = ""                   # sha256(prompt) — also the filename stem
    error: Optional[str] = None
    generated_at: str = ""
    provider: str = ""
    model: str = ""
    attempts: int = 0
    media_type: str = "image"             # "image" or "video"

    @property
    def success(self) -> bool:
        return bool(self.image_url or self.cached_path) and not self.error

    @property
    def is_video(self) -> bool:
        return self.media_type == "video"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Prompt building + cache keys ────────────────────────────────────────────

_DEFAULT_PROMPT_TEMPLATE = (
    "A clean, modern scientific visual abstract for a research idea. "
    "Composition: central concept icon with 2-3 supporting elements "
    "arranged in a balanced layout. "
    "Research topic: {title}. "
    "Core method: {method_excerpt}. "
    "Field: {methodology}. "
    "Render as a single illustrative panel suitable for the abstract "
    "page of a research paper. No watermarks, no signatures, no text."
)


# ── Image style presets ───────────────────────────────────────────────────
#
# Each preset is a named "look" — a suffix appended to the base prompt
# that steers the image-gen model toward a specific visual style. Lets
# users go from "boring editorial illustration" → "isometric 3D render"
# / "Nature journal figure" / "hand-drawn whiteboard sketch" / etc.
# with one dropdown click.
#
# Prompt suffix is the canonical-style modifier; some models (FLUX
# especially) respond strongly to the specific style tags. Negative
# prompt is what to suppress (only used by providers that support it —
# FLUX does, Imagen does not).
#
# Adding a style: drop another entry into the dict. The admin UI picks
# them up automatically.

@dataclass
class StylePreset:
    name: str
    label: str
    description: str
    prompt_suffix: str
    negative_prompt: str = ""


IMAGE_STYLE_PRESETS: Dict[str, StylePreset] = {
    "editorial": StylePreset(
        name="editorial",
        label="📰 Editorial illustration (default)",
        description="Clean flat-design icons, sky-blue + navy palette.",
        prompt_suffix=(
            " Style: minimal flat-design editorial illustration, soft "
            "sky-blue and deep navy on a light background, no text "
            "labels, no people, no logos. Crisp vector look."
        ),
        negative_prompt=(
            "text, words, labels, signature, watermark, people, faces, "
            "logos, blurry, low quality"
        ),
    ),
    "scientific_paper": StylePreset(
        name="scientific_paper",
        label="🔬 Scientific paper (Nature/Science style)",
        description="Two-tone grayscale + single accent color, journal style.",
        prompt_suffix=(
            " Style: Nature/Science journal figure, two-tone grayscale "
            "with a single accent color, technical illustration, no "
            "text labels. White background, clean typographic grid."
        ),
        negative_prompt=(
            "text, words, labels, photorealistic, gradient, cartoon, "
            "people, signature, watermark"
        ),
    ),
    "isometric_3d": StylePreset(
        name="isometric_3d",
        label="🎮 Isometric 3D render",
        description="Blender-style isometric scene with soft shadows.",
        prompt_suffix=(
            " Style: isometric 3D render, blender-style, soft shadows, "
            "muted pastel palette with sky-blue accents, clean studio "
            "lighting. No text labels."
        ),
        negative_prompt=(
            "text, words, labels, flat, 2D, hand-drawn, signature, "
            "watermark, people"
        ),
    ),
    "sketch": StylePreset(
        name="sketch",
        label="✏️ Hand-drawn whiteboard sketch",
        description="Loose ink lines on white, lab-notebook style.",
        prompt_suffix=(
            " Style: hand-drawn whiteboard sketch, dark ink on white, "
            "loose imperfect lines, scientific lab-notebook aesthetic, "
            "no text labels. Light watercolor accents in sky-blue."
        ),
        negative_prompt=(
            "text, words, labels, photorealistic, 3D, polished, "
            "signature, watermark"
        ),
    ),
    "blueprint": StylePreset(
        name="blueprint",
        label="📐 Technical blueprint",
        description="White lines on deep blue — schematic style.",
        prompt_suffix=(
            " Style: technical engineering blueprint, white and amber "
            "lines on a deep navy background, precise schematic with "
            "measurement-style annotations (no actual numbers or text). "
            "Drafting grid background."
        ),
        negative_prompt=(
            "text, words, numbers, photorealistic, color, cartoon, "
            "signature, watermark"
        ),
    ),
    "infographic": StylePreset(
        name="infographic",
        label="📊 Modern infographic",
        description="Bold colors, multiple sections, data-viz icons.",
        prompt_suffix=(
            " Style: clean modern infographic, bold flat colors, "
            "multiple visual sections in a grid layout, data-viz icons "
            "(charts, arrows, gears), white background. No actual text."
        ),
        negative_prompt=(
            "text, words, labels, photorealistic, dark background, "
            "messy, signature, watermark"
        ),
    ),
    "dark_mode": StylePreset(
        name="dark_mode",
        label="🌙 Dark mode futuristic",
        description="Deep navy background, glowing sky-blue + amber accents.",
        prompt_suffix=(
            " Style: dark mode, deep navy / near-black background with "
            "glowing sky-blue and amber accents, futuristic editorial "
            "illustration. Subtle particle effects. No text labels."
        ),
        negative_prompt=(
            "text, words, labels, light background, photorealistic, "
            "people, signature, watermark"
        ),
    ),
    "photorealistic": StylePreset(
        name="photorealistic",
        label="📸 Photorealistic (lab/studio)",
        description="Photo-style render of a scientific scene.",
        prompt_suffix=(
            " Style: photorealistic photograph, shot on a Hasselblad "
            "camera with 85mm lens, soft natural lighting, "
            "scientific laboratory or studio setting, shallow depth of "
            "field, no text. Editorial photography style."
        ),
        negative_prompt=(
            "text, words, labels, cartoon, illustration, sketch, "
            "people, signature, watermark"
        ),
    ),
}


DEFAULT_STYLE: str = "editorial"


def apply_style(prompt: str, style: str = DEFAULT_STYLE) -> str:
    """Append the style preset's suffix to a base prompt.

    Unknown styles fall back to DEFAULT_STYLE rather than raising — the
    UI lets users type arbitrary preset names via "Custom" paths.
    """
    if not prompt:
        return prompt
    preset = IMAGE_STYLE_PRESETS.get(style) or IMAGE_STYLE_PRESETS[DEFAULT_STYLE]
    return prompt.rstrip() + preset.prompt_suffix


def build_prompt(
    idea: Dict[str, Any],
    template: str = _DEFAULT_PROMPT_TEMPLATE,
    method_excerpt_chars: int = 180,
    style: str = DEFAULT_STYLE,
) -> str:
    """Construct a stable image-generation prompt from an idea dict.

    Stability matters because the cache key is sha256(prompt) — if the
    prompt drifts on every call, the cache misses every time. We pull
    only deterministic fields (title, method excerpt, methodology) and
    apply a deterministic excerpt length.

    `style` selects a named visual preset from IMAGE_STYLE_PRESETS;
    its prompt suffix is appended to the base template. Different
    styles → different cache keys → all are cached independently.
    """
    if not isinstance(idea, dict):
        raise TypeError("idea must be a dict")
    title = str(idea.get("title", "") or "").strip() or "(untitled idea)"
    method = str(idea.get("method", "") or "").strip()
    methodology = (
        str(idea.get("methodology_type", "") or "")
        .replace("_", " ").strip() or "research"
    )
    method_excerpt = method[:method_excerpt_chars].rstrip()
    if len(method) > method_excerpt_chars:
        method_excerpt += "…"
    base = template.format(
        title=title,
        method_excerpt=method_excerpt or "an empirical study",
        methodology=methodology,
    )
    return apply_style(base, style=style)


def cache_key_for_prompt(prompt: str) -> str:
    """SHA-256 of the prompt — stable hex digest used as the cache filename."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


# ── Multi-panel paper-figure templates ─────────────────────────────────────
#
# Inspired by what Kimi's PPT generator does with Nano Banana: instead of
# ONE generic visual abstract per idea, generate a coordinated SET of
# panel-specific images that together tell the story of the research
# idea — the way a real paper has Figure 1 (concept), Figure 2 (method),
# Figure 3 (experimental setup), Figure 4 (results).
#
# Each template is a focused prompt tailored to ONE role. The renderer
# composes the final prompt from the template + the idea's fields. The
# cache key (sha256 of the final prompt) is naturally unique per panel,
# so panels are independently cached.
#
# The user picks which panels to generate via the UI — most ideas want
# all four; some only want the concept + method pair.

@dataclass
class FigurePanel:
    """One panel in a multi-panel paper-figure set."""
    panel_id: str            # e.g. "concept", "method"
    label: str               # human-readable name
    visual: IdeaVisual
    order: int = 0           # rendering order, 0-indexed


FIGURE_TEMPLATES: Dict[str, Dict[str, str]] = {
    "concept": {
        "label": "🧠 Concept",
        "description": "High-level visual metaphor of the core insight.",
        "prompt_template": (
            "A clean editorial illustration for a research paper. "
            "Concept-level visual metaphor that captures the CORE "
            "INSIGHT of the idea. Style: minimal flat-design icons in "
            "soft sky-blue and deep navy on a light background. "
            "Composition: central concept with 2-3 orbiting supporting "
            "elements. No text labels, no people, no logos. "
            "Topic: {title}. "
            "Core idea: {motivation_excerpt}."
        ),
    },
    "method": {
        "label": "⚙️ Method / Architecture",
        "description": "Box-and-arrow technical method diagram.",
        "prompt_template": (
            "A clean technical method diagram for a research paper, "
            "Figure 2 style. Box-and-arrow architecture showing data "
            "flow, components, and processing stages from left to "
            "right. Each box represents one component. Editorial style, "
            "soft sky-blue and deep navy, no text labels (the figure "
            "will be captioned externally). "
            "Topic: {title}. "
            "Method: {method_excerpt}."
        ),
    },
    "experiment": {
        "label": "🧪 Experimental design",
        "description": "Dataset + evaluation pipeline + metrics.",
        "prompt_template": (
            "An experimental setup diagram for a research paper, "
            "Figure 3 style. Shows dataset on the left, evaluation "
            "pipeline in the middle, metrics on the right. Clean "
            "editorial style with abstracted icons (no text labels). "
            "Soft sky-blue accents. "
            "Topic: {title}. "
            "Setup: {method_excerpt}. "
            "Expected outcome: {outcome_excerpt}."
        ),
    },
    "results": {
        "label": "📊 Expected results",
        "description": "Hypothesized chart / bar graph illustration.",
        "prompt_template": (
            "A scientific bar chart or scatter plot illustration showing "
            "the HYPOTHESIZED results of a research study. Two-axis "
            "chart with soft sky-blue and amber accents, clean axes, "
            "abstracted bars/dots (no actual numbers, no text labels). "
            "Y-axis represents the key metric. "
            "Topic: {title}. "
            "Hypothesis being illustrated: {hypothesis_excerpt}."
        ),
    },
    "comparison": {
        "label": "⚖️ Comparison / Baseline",
        "description": "Side-by-side comparison with prior work.",
        "prompt_template": (
            "A side-by-side comparison illustration for a research "
            "paper. Left panel shows the baseline / prior approach, "
            "right panel shows the proposed method. Both panels use "
            "the same visual style for fair comparison. Editorial "
            "style, soft sky-blue with amber accents on the proposed "
            "side. No text labels. "
            "Topic: {title}. "
            "Proposed method: {method_excerpt}."
        ),
    },
    "limitations": {
        "label": "🚧 Limitations / Risk",
        "description": "Visual showing what could go wrong.",
        "prompt_template": (
            "A research-paper figure showing the LIMITATIONS or risk "
            "factors of a proposed method. A diagram with three or "
            "four warning-flagged elements showing where the method "
            "might fail. Editorial style with red-amber accents on "
            "the risk regions, soft sky-blue for the base method. "
            "No text labels. "
            "Topic: {title}. "
            "Risks: {risk_excerpt}."
        ),
    },
}


# Default panel set when the user just says "make me a figure set".
# Concept + method + experiment + results is the canonical 4-panel layout
# every paper has.
DEFAULT_FIGURE_SET: List[str] = ["concept", "method", "experiment", "results"]


def build_panel_prompt(
    idea: Dict[str, Any],
    panel_id: str,
    excerpt_chars: int = 180,
    style: str = DEFAULT_STYLE,
) -> str:
    """Build a panel-specific prompt for one idea + one template ID.

    Raises ValueError if `panel_id` isn't in FIGURE_TEMPLATES — fail
    early rather than silently render with a missing template.

    `style` applies the same named visual preset as `build_prompt()`
    to the panel — so a 4-panel set with style='isometric_3d' will
    produce four coordinated 3D-style panels.
    """
    if not isinstance(idea, dict):
        raise TypeError("idea must be a dict")
    if panel_id not in FIGURE_TEMPLATES:
        raise ValueError(
            f"unknown panel_id {panel_id!r}; "
            f"must be one of {sorted(FIGURE_TEMPLATES)}"
        )
    template = FIGURE_TEMPLATES[panel_id]["prompt_template"]
    title = str(idea.get("title", "") or "").strip() or "(untitled idea)"
    motivation = str(idea.get("motivation", "") or "").strip()
    method = str(idea.get("method", "") or "").strip()
    hypothesis = str(idea.get("hypothesis", "") or "").strip()
    outcome = str(idea.get("expected_outcome", "") or "").strip()
    risk = str(idea.get("risk_assessment", "") or "").strip()

    def _excerpt(text: str) -> str:
        t = text[:excerpt_chars].rstrip()
        return (t + "…") if len(text) > excerpt_chars else t

    base = template.format(
        title=title,
        motivation_excerpt=_excerpt(motivation)
            or "the proposed research idea",
        method_excerpt=_excerpt(method)
            or "the proposed approach",
        hypothesis_excerpt=_excerpt(hypothesis)
            or "the falsifiable claim",
        outcome_excerpt=_excerpt(outcome)
            or "the measurable outcome",
        risk_excerpt=_excerpt(risk) or "key risks",
    )
    return apply_style(base, style=style)


# ── Disk cache ──────────────────────────────────────────────────────────────

class DiskCache:
    """Simple cache of (prompt → IdeaVisual JSON + image bytes).

    Each entry is two files:
      <cache_key>.json  ── IdeaVisual metadata
      <cache_key>.bin   ── raw image bytes (PNG/JPEG/WebP — provider-decided)

    The bin file is optional; if the provider returns only a URL and we
    haven't downloaded it, the .bin won't exist yet — that's fine, the
    URL is the source of truth.
    """

    def __init__(self, cache_dir: str = ".ideagraph_visual_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _meta_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _bin_path(self, key: str, media_type: str = "image") -> Path:
        """Path of the cached binary asset.

        Image media uses `.bin` (backwards-compatible). Video media uses
        `.mp4` so Streamlit's st.video() and the OS can recognize it."""
        ext = ".mp4" if media_type == "video" else ".bin"
        return self.cache_dir / f"{key}{ext}"

    def get(self, key: str) -> Optional[IdeaVisual]:
        meta = self._meta_path(key)
        if not meta.exists():
            return None
        try:
            with open(meta, encoding="utf-8") as f:
                d = json.load(f)
            visual = IdeaVisual(**d)
            # If the bin exists, prefer it over a possibly-expired URL.
            # Check both extensions (image .bin and video .mp4).
            for ext_path in (
                self._bin_path(key, media_type=visual.media_type),
                self._bin_path(key, media_type="image"),
                self._bin_path(key, media_type="video"),
            ):
                if ext_path.exists():
                    visual.cached_path = str(ext_path)
                    break
            return visual
        except Exception as e:
            log.warning(f"DiskCache.get({key[:12]}…) failed: {e}")
            return None

    def put(self, key: str, visual: IdeaVisual,
            image_bytes: Optional[bytes] = None) -> None:
        try:
            if image_bytes is not None:
                bin_path = self._bin_path(key, media_type=visual.media_type)
                bin_path.write_bytes(image_bytes)
                visual.cached_path = str(bin_path)
            with open(self._meta_path(key), "w", encoding="utf-8") as f:
                json.dump(visual.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            log.warning(f"DiskCache.put({key[:12]}…) failed: {e}")

    def has(self, key: str) -> bool:
        return self._meta_path(key).exists()

    def clear(self) -> int:
        """Remove all cached entries. Returns count of files deleted."""
        n = 0
        for f in self.cache_dir.glob("*"):
            try:
                f.unlink()
                n += 1
            except Exception:
                pass
        return n


# ── Provider abstraction ────────────────────────────────────────────────────

class ImageProvider:
    """Base class for image-generation backends. Subclass and implement
    `_generate_raw(prompt) -> Dict` returning at minimum `{"image_url": str}`
    (or `{"image_bytes": bytes}` for inline payloads).

    The renderer wraps `_generate_raw` with retry, caching, and timeout
    logic, so subclasses focus only on the HTTP shape of their API.
    """

    name: str = "abstract"

    def __init__(
        self,
        api_key: str = "",
        model: str = "",
        endpoint: str = "",
        timeout_s: float = 120.0,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.model = model
        self.endpoint = endpoint
        self.timeout_s = timeout_s

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key) and not self.api_key.startswith("sk-xxx")

    def _generate_raw(self, prompt: str) -> Dict[str, Any]:
        """Subclass hook. Must return a dict containing either
        `image_url` (str) or `image_bytes` (bytes), plus any extras."""
        raise NotImplementedError

    def generate(self, prompt: str) -> Dict[str, Any]:
        """Public synchronous entry. Adds retry + error wrapping."""
        if not self.is_configured:
            return {"error": "API key not configured for provider "
                              f"{self.name!r}"}
        try:
            return self._generate_raw(prompt)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {str(e)[:200]}"}


# ── Default provider: BlackForest Labs FLUX (api.bfl.ml) ───────────────────

class FluxBFLProvider(ImageProvider):
    """FLUX via BlackForest Labs' official API.

    Async polling protocol:
      1. POST {endpoint}/{model} with x-key header → returns {"id": "..."}
      2. GET {endpoint}/get_result?id=... → polls until status=Ready
      3. Final response has {"result": {"sample": "https://..."}}

    Fail-soft: any non-200 or network error returns an error dict rather
    than raising.
    """

    name = "flux_bfl"

    def __init__(
        self,
        api_key: str = "",
        model: str = "flux-pro-1.0",
        endpoint: str = "https://api.bfl.ml/v1",
        timeout_s: float = 120.0,
        poll_interval_s: float = 1.5,
    ) -> None:
        super().__init__(api_key, model, endpoint, timeout_s)
        self.poll_interval_s = poll_interval_s

    def _generate_raw(self, prompt: str) -> Dict[str, Any]:
        if not _HAS_REQUESTS:
            return {"error": "requests library not installed; "
                              "pip install requests"}

        headers = {
            "Content-Type": "application/json",
            "x-key": self.api_key,
            "Accept": "application/json",
        }
        body = {
            "prompt": prompt,
            "width": 1024,
            "height": 1024,
            "prompt_upsampling": False,
            "safety_tolerance": 2,
            "output_format": "png",
        }
        submit_url = f"{self.endpoint.rstrip('/')}/{self.model}"

        # 1. Submit
        try:
            r = requests.post(submit_url, headers=headers,
                                 json=body, timeout=30)
        except requests.RequestException as e:
            return {"error": f"network error on submit: {e}"}
        if r.status_code != 200:
            return {"error": f"submit HTTP {r.status_code}: {r.text[:200]}"}
        try:
            sub = r.json()
        except ValueError:
            return {"error": "submit response was not JSON"}
        task_id = sub.get("id")
        if not task_id:
            return {"error": f"submit response had no task id: {sub}"}

        # 2. Poll
        poll_url = f"{self.endpoint.rstrip('/')}/get_result"
        deadline = time.time() + self.timeout_s
        last_status = "unknown"
        while time.time() < deadline:
            try:
                pr = requests.get(poll_url, headers=headers,
                                       params={"id": task_id}, timeout=15)
            except requests.RequestException as e:
                # transient — keep polling
                last_status = f"poll-error: {e}"
                time.sleep(self.poll_interval_s)
                continue
            if pr.status_code != 200:
                last_status = f"poll HTTP {pr.status_code}"
                time.sleep(self.poll_interval_s)
                continue
            try:
                p = pr.json()
            except ValueError:
                last_status = "poll non-JSON"
                time.sleep(self.poll_interval_s)
                continue
            status = (p.get("status") or "").lower()
            last_status = status
            if status in ("ready", "complete", "completed", "success"):
                result = p.get("result") or {}
                img_url = (
                    result.get("sample") or result.get("image_url")
                    or result.get("url")
                )
                if not img_url:
                    return {"error": f"ready response missing image url: {p}"}
                return {"image_url": img_url, "raw_response": p}
            if status in ("failed", "error"):
                return {"error": f"provider reported failure: {p}"}
            time.sleep(self.poll_interval_s)
        return {"error": f"polling timed out after {self.timeout_s}s "
                          f"(last status: {last_status})"}


# ── Provider: Google AI Studio (Imagen 3 + Gemini Image) ────────────────────

class GeminiImagenProvider(ImageProvider):
    """Google AI Studio / Gemini API image generation.

    Targets `imagen-3.0-generate-002` (Google's dedicated text-to-image
    model) by default. Single-shot synchronous protocol — no polling,
    image returned inline as base64.

    Endpoint pattern:
        POST {endpoint}/models/{model}:predict
        Auth: `x-goog-api-key: AIza...` (also accepts ?key= query param)
        Body: {"instances": [{"prompt": ...}],
               "parameters": {"sampleCount": 1, "aspectRatio": "1:1",
                              "personGeneration": "ALLOW_ADULT"}}
        Response: {"predictions": [{"bytesBase64Encoded": "...",
                                     "mimeType": "image/png"}]}

    Get an API key from https://aistudio.google.com/apikey — format
    `AIzaSy...` (39 chars). The `gen-lang-client-*` value you sometimes
    see is a Google CLIENT IDENTIFIER, not an API key — auth will fail
    if you paste that here.

    To use Gemini-2.5-Flash-Image ("Nano Banana") instead of Imagen,
    set model="gemini-2.5-flash-image-preview" and use the
    `:generateContent` endpoint — that's a different request/response
    shape and would need a separate subclass.
    """

    name = "gemini_imagen"

    def __init__(
        self,
        api_key: str = "",
        model: str = "imagen-3.0-generate-002",
        endpoint: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout_s: float = 120.0,
        aspect_ratio: str = "1:1",
        person_generation: str = "ALLOW_ADULT",
    ) -> None:
        super().__init__(api_key, model, endpoint, timeout_s)
        self.aspect_ratio = aspect_ratio
        self.person_generation = person_generation

    @property
    def is_configured(self) -> bool:
        # Google AI Studio keys start with "AIza". Reject the
        # `gen-lang-client-...` client identifier value early — pasting
        # it would auth-fail with a confusing 401.
        if not self.api_key:
            return False
        if self.api_key.startswith("sk-xxx"):
            return False
        if self.api_key.startswith("gen-lang-client-"):
            return False
        return True

    def generate(self, prompt: str) -> Dict[str, Any]:
        """Override the base `generate()` to catch the most common Google
        misconfiguration — pasting a `gen-lang-client-…` client identifier
        as the key — and surface a SPECIFIC error before the base class's
        generic "not configured" message kicks in.
        """
        if self.api_key and self.api_key.startswith("gen-lang-client-"):
            return {"error": (
                "The value provided is a Google client identifier "
                "(gen-lang-client-…), not an API key. Get the real "
                "API key from https://aistudio.google.com/apikey — "
                "it starts with `AIza`."
            )}
        return super().generate(prompt)

    def _generate_raw(self, prompt: str) -> Dict[str, Any]:
        if not _HAS_REQUESTS:
            return {"error": "requests library not installed; "
                              "pip install requests"}
        if self.api_key.startswith("gen-lang-client-"):
            return {"error": (
                "The value provided is a Google client identifier "
                "(gen-lang-client-…), not an API key. Get the real "
                "API key from https://aistudio.google.com/apikey — "
                "it starts with `AIza`."
            )}
        url = f"{self.endpoint.rstrip('/')}/models/{self.model}:predict"
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        body = {
            "instances": [{"prompt": prompt}],
            "parameters": {
                "sampleCount": 1,
                "aspectRatio": self.aspect_ratio,
                "personGeneration": self.person_generation,
            },
        }
        try:
            r = requests.post(
                url, headers=headers, json=body, timeout=self.timeout_s,
            )
        except requests.RequestException as e:
            return {"error": f"network error: {e}"}
        if r.status_code != 200:
            # Google returns helpful error JSON; surface it verbatim.
            return {"error": f"HTTP {r.status_code}: {r.text[:400]}"}
        try:
            data = r.json()
        except ValueError:
            return {"error": "response was not JSON"}
        preds = data.get("predictions") or []
        if not preds:
            return {"error": (
                f"response had no predictions: "
                f"{json.dumps(data)[:300]}"
            )}
        pred = preds[0] if isinstance(preds[0], dict) else {}
        b64 = pred.get("bytesBase64Encoded")
        if not b64:
            return {"error": (
                "prediction missing bytesBase64Encoded "
                f"(keys: {list(pred.keys()) if pred else 'empty'})"
            )}
        try:
            img_bytes = base64.b64decode(b64)
        except Exception as e:
            return {"error": f"base64 decode failed: {e}"}
        return {
            "image_bytes": img_bytes,
            "mime_type": pred.get("mimeType", "image/png"),
        }


# ── Provider: Gemini Flash Image ("Nano Banana") ───────────────────────────

class GeminiFlashImageProvider(ImageProvider):
    """Google Gemini 2.x Flash Image via the `:generateContent` endpoint.

    This is the model commonly referred to as "Nano Banana" in Google's
    docs. UNLIKE `GeminiImagenProvider`, it:
      - Uses `:generateContent` (not `:predict`)
      - Wraps the prompt in `contents.parts.text`
      - Asks for image output via `generationConfig.responseModalities`
      - Returns the image inline in `candidates[0].content.parts[*].inlineData`
        (the parts list may also contain TEXT entries — walk it to find
        the inlineData entry)

    Critically, this endpoint is accessible with a standard
    `AIzaSy...` AI Studio key — no Vertex AI / paid GCP project
    required. So this is usually the right adapter for users hitting a
    404 on Imagen.

    Get a key from https://aistudio.google.com/apikey.
    """

    name = "gemini_flash_image"

    def __init__(
        self,
        api_key: str = "",
        model: str = "gemini-2.5-flash-image-preview",
        endpoint: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout_s: float = 120.0,
    ) -> None:
        super().__init__(api_key, model, endpoint, timeout_s)

    @property
    def is_configured(self) -> bool:
        if not self.api_key:
            return False
        if self.api_key.startswith("sk-xxx"):
            return False
        if self.api_key.startswith("gen-lang-client-"):
            return False
        return True

    def generate(self, prompt: str) -> Dict[str, Any]:
        if self.api_key and self.api_key.startswith("gen-lang-client-"):
            return {"error": (
                "The value provided is a Google client identifier "
                "(gen-lang-client-…), not an API key. Get the real "
                "API key from https://aistudio.google.com/apikey — "
                "it starts with `AIza`."
            )}
        return super().generate(prompt)

    def _generate_raw(self, prompt: str) -> Dict[str, Any]:
        if not _HAS_REQUESTS:
            return {"error": "requests library not installed; "
                              "pip install requests"}
        url = (
            f"{self.endpoint.rstrip('/')}/models/{self.model}:generateContent"
        )
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        body = {
            "contents": [{
                "role": "user",
                "parts": [{"text": prompt}],
            }],
            "generationConfig": {
                # Tell the model we want an image back. Some variants
                # only respect this when both TEXT and IMAGE are listed.
                "responseModalities": ["IMAGE", "TEXT"],
            },
        }
        try:
            r = requests.post(
                url, headers=headers, json=body, timeout=self.timeout_s,
            )
        except requests.RequestException as e:
            return {"error": f"network error: {e}"}
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}: {r.text[:400]}"}
        try:
            data = r.json()
        except ValueError:
            return {"error": "response was not JSON"}

        # Walk candidates[0].content.parts looking for inlineData.
        candidates = data.get("candidates") or []
        if not candidates:
            # If the model refused (safety filter, etc.) we get a
            # promptFeedback block instead of candidates.
            pf = data.get("promptFeedback") or {}
            if pf:
                return {"error": f"promptFeedback (likely safety): {pf}"}
            return {"error": (
                f"response had no candidates: {json.dumps(data)[:300]}"
            )}
        content = (candidates[0] or {}).get("content") or {}
        parts = content.get("parts") or []
        img_bytes: Optional[bytes] = None
        mime = "image/png"
        for part in parts:
            if not isinstance(part, dict):
                continue
            inline = part.get("inlineData") or part.get("inline_data") or {}
            if isinstance(inline, dict) and inline.get("data"):
                try:
                    img_bytes = base64.b64decode(inline["data"])
                    mime = inline.get("mimeType") or inline.get(
                        "mime_type", "image/png",
                    )
                    break
                except Exception as e:
                    return {"error": f"base64 decode failed: {e}"}
        if img_bytes is None:
            return {"error": (
                "no inlineData image part in response. The model may "
                "have returned only text (it does that sometimes for "
                "very abstract prompts). Try a more visual prompt, or "
                "switch to FLUX which always returns an image."
            )}
        return {"image_bytes": img_bytes, "mime_type": mime}


# ── Diagnostic helper: list models the user's key can actually access ─────
#
# Google gates image-generation models per account/region/tier. A user
# can get HTTP 404 on `imagen-3.0-generate-002` AND on
# `gemini-2.5-flash-image-preview` even with a valid AIza... key — those
# models simply aren't enabled for them. This helper calls Google's
# ListModels endpoint to enumerate what the key CAN see, with a
# heuristic flag for "looks like image-gen".

def list_gemini_models(
    api_key: str,
    endpoint: str = "https://generativelanguage.googleapis.com/v1beta",
    timeout_s: float = 30.0,
) -> Dict[str, Any]:
    """Call Google's ListModels endpoint.

    Returns `{"error": "..."}` on failure, or
    `{"models": [{"name": ..., "supports_image_gen": bool, ...}, ...]}`
    on success.

    Each model dict carries:
      - name             (the model name, e.g. "gemini-1.5-flash")
      - display_name     (human label)
      - generation_methods (list — e.g. ["generateContent", "countTokens"])
      - supports_image_gen (bool — heuristic: name contains "imagen" /
                                "image" AND supports generateContent or
                                predict)
    """
    if not _HAS_REQUESTS:
        return {"error": "requests library not installed"}
    if not api_key:
        return {"error": "API key is required to list models"}
    if api_key.startswith("gen-lang-client-"):
        return {"error": (
            "The value provided is a Google client identifier "
            "(gen-lang-client-…), not an API key. Get the real key "
            "from https://aistudio.google.com/apikey — it starts with `AIza`."
        )}

    url = f"{endpoint.rstrip('/')}/models"
    try:
        # ListModels supports either x-goog-api-key header OR ?key= param;
        # use the header for consistency with our generate calls.
        r = requests.get(
            url,
            headers={"x-goog-api-key": api_key},
            timeout=timeout_s,
        )
    except requests.RequestException as e:
        return {"error": f"network error: {e}"}
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}: {r.text[:400]}"}
    try:
        data = r.json()
    except ValueError:
        return {"error": "response was not JSON"}

    raw_models = data.get("models") or []
    out: List[Dict[str, Any]] = []
    for m in raw_models:
        if not isinstance(m, dict):
            continue
        # Names come prefixed with "models/" — strip for display.
        raw_name = m.get("name", "")
        name = raw_name.replace("models/", "", 1) if raw_name else ""
        methods = m.get("supportedGenerationMethods") or []
        name_lc = name.lower()
        looks_image = (
            ("image" in name_lc or "imagen" in name_lc)
            and ("generateContent" in methods or "predict" in methods)
        )
        out.append({
            "name": name,
            "display_name": m.get("displayName", name),
            "description": m.get("description", "")[:200],
            "generation_methods": list(methods),
            "supports_image_gen": bool(looks_image),
        })

    # Image-gen models first, then everything else, both alphabetical.
    out.sort(key=lambda m: (not m["supports_image_gen"], m["name"]))
    return {"models": out, "count": len(out)}


# ── xAI Grok image provider (grok-2-image) ─────────────────────────────────
#
# xAI's image API is OpenAI-compatible:
#   POST https://api.x.ai/v1/images/generations
#   Authorization: Bearer xai-…
#   {"model": "grok-2-image", "prompt": "…", "n": 1, "response_format": "url"}
#
# Response:
#   {"data": [{"url": "https://…/img.png", "revised_prompt": "…"}], ...}
#
# Notes:
#   - Only `prompt` and `model` (and optional `n` 1-10) are accepted. xAI
#     does NOT support `size`, `style`, `quality`, or `response_format=b64_json`
#     at time of writing — silently ignored if you send them.
#   - xAI does NOT have a public video generation API as of late 2025.
#     For video, keep using the `veo` provider.

class GrokImageProvider(ImageProvider):
    """xAI Grok image generation via OpenAI-compatible /v1/images/generations.

    xAI's Image API surface is a strict subset of OpenAI's: just `model`,
    `prompt`, `n` (1-10), and `response_format`. Other knobs (size, style,
    quality, hd) are silently dropped server-side — don't bother sending
    them. Returns a URL pointing at an xAI-hosted PNG that's valid for
    ~24h; we cache the bytes locally on first render via the renderer's
    DiskCache layer.
    """

    name = "grok"

    def __init__(
        self,
        api_key: str = "",
        model: str = "grok-2-image",
        endpoint: str = "https://api.x.ai/v1",
        timeout_s: float = 90.0,
    ) -> None:
        super().__init__(api_key, model, endpoint, timeout_s)

    @property
    def is_configured(self) -> bool:
        # xAI keys are prefixed `xai-…`. Don't accept obvious placeholders.
        if not self.api_key or self.api_key.startswith("sk-xxx"):
            return False
        return True

    def _generate_raw(self, prompt: str) -> Dict[str, Any]:
        if not _HAS_REQUESTS:
            return {"error": "requests library not installed; "
                              "pip install requests"}
        url = f"{self.endpoint.rstrip('/')}/images/generations"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "prompt": prompt,
            "n": 1,
            "response_format": "url",
        }
        try:
            r = requests.post(
                url, headers=headers, json=body, timeout=self.timeout_s,
            )
        except Exception as e:
            return {"error": f"network: {type(e).__name__}: {str(e)[:200]}"}
        if r.status_code != 200:
            return {"error": (
                f"HTTP {r.status_code}: {r.text[:400]}"
            )}
        try:
            payload = r.json()
        except Exception as e:
            return {"error": f"non-JSON response: {str(e)[:200]}"}
        data = payload.get("data") or []
        if not data:
            return {"error": (
                f"xAI returned no images: {json.dumps(payload)[:300]}"
            )}
        first = data[0]
        if first.get("url"):
            return {
                "image_url": first["url"],
                "revised_prompt": first.get("revised_prompt", ""),
            }
        if first.get("b64_json"):
            try:
                return {
                    "image_bytes": base64.b64decode(first["b64_json"]),
                    "revised_prompt": first.get("revised_prompt", ""),
                }
            except Exception as e:
                return {"error": f"b64 decode failed: {e}"}
        return {"error": (
            f"xAI response had no url or b64_json: {json.dumps(first)[:300]}"
        )}


# ── Provider registry — string name → class ───────────────────────────────
#
# When constructing a renderer by name (from config or admin UI), look
# up here. New providers register themselves by adding an entry.

PROVIDER_REGISTRY: Dict[str, type] = {
    "flux_bfl":           FluxBFLProvider,
    "gemini_imagen":      GeminiImagenProvider,
    "gemini_flash_image": GeminiFlashImageProvider,
    "grok":               GrokImageProvider,
}


# Sensible defaults per provider — used when the user hasn't overridden
# the model or endpoint. Lets the admin "Provider" dropdown snap the
# other fields to sane values when switched. `known_models` is a curated
# list of model names the admin UI offers in a dropdown (with a custom-text
# option for variants not yet in the list).
PROVIDER_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "flux_bfl": {
        "model":    "flux-pro-1.0",
        "endpoint": "https://api.bfl.ml/v1",
        "known_models": [
            "flux-pro-1.0",
            "flux-pro-1.1",
            "flux-pro-1.1-ultra",
            "flux-dev",
            "flux-schnell",
        ],
    },
    "gemini_imagen": {
        # Default to the newest GA Imagen 4 variant — works with paid
        # Google AI Studio credit. Falls back through older Imagen 3
        # variants if your tier doesn't have Imagen 4 yet.
        "model":    "imagen-4.0-generate-001",
        "endpoint": "https://generativelanguage.googleapis.com/v1beta",
        "known_models": [
            # Imagen 4 Ultra (newest, highest quality — paid tier)
            "imagen-4.0-ultra-generate-001",
            "imagen-4.0-ultra-generate-preview-06-06",
            # Imagen 4 (newest GA)
            "imagen-4.0-generate-001",
            "imagen-4.0-fast-generate-001",
            "imagen-4.0-generate-preview-06-06",
            "imagen-4.0-fast-generate-preview-06-06",
            # Imagen 3 (widely available, cheapest)
            "imagen-3.0-generate-002",
            "imagen-3.0-generate-001",
            "imagen-3.0-fast-generate-001",
        ],
    },
    "gemini_flash_image": {
        # Default to the GA name; `gemini-2.5-flash-image-preview` is
        # the preview alias for the same model. Both should work on
        # paid AI Studio tier.
        "model":    "gemini-2.5-flash-image",
        "endpoint": "https://generativelanguage.googleapis.com/v1beta",
        "known_models": [
            # Gemini 3 Pro Image — "Nano Banana Pro" (paid; exact API
            # name varies by tier/region — try each in turn or use the
            # 📋 List models button below to see what your key actually
            # has access to. Removed `nano-banana-pro` from this list
            # because it's the colloquial name and never resolves on the
            # generativelanguage.googleapis.com endpoint.)
            "gemini-3-pro-image-preview",
            "gemini-3-pro-image",
            "gemini-3.0-pro-image-preview",
            # Gemini 2.5 image generators ("Nano Banana")
            "gemini-2.5-flash-image",
            "gemini-2.5-flash-image-preview",
            # Gemini 2.0 image generators (older but often most reliable)
            "gemini-2.0-flash-preview-image-generation",
            "gemini-2.0-flash-exp",
            "gemini-2.0-flash-001",
            "gemini-2.0-flash",
            # Gemini 2.5 chat models (text-only — listed for completeness;
            # they don't produce images but ListModels surfaces them too)
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            # Gemini 1.5 (legacy text-only — rarely useful for image gen
            # but kept so the dropdown matches what ListModels returns)
            "gemini-1.5-pro",
            "gemini-1.5-pro-002",
            "gemini-1.5-flash",
            "gemini-1.5-flash-002",
            "gemini-1.5-flash-8b",
        ],
    },
}


# ── Video generation (Veo) ────────────────────────────────────────────────
#
# Veo 3 produces VIDEO, not still images. Different protocol from image:
# POST → returns a long-running operation name → poll operation status
# → final response carries an MP4 URL (or inline bytes for short clips).

VEO_VIDEO_MODELS: List[str] = [
    "veo-3.0-generate-001",
    "veo-3.0-generate-preview",
    "veo-2.0-generate-001",
]

VEO_INFO_MESSAGE: str = (
    "🎬 **Veo 3** is Google's video-generation model — it produces "
    "short MP4 clips (5-8s), not still images. Uses a long-running "
    "operation protocol: submit → poll operation → download MP4. "
    "Wired into the **Visual Simulation tab** as an optional animated "
    "explainer for any idea."
)


@dataclass
class IdeaVideo:
    """Result of rendering one idea into a short video clip."""
    idea_title: str
    prompt: str = ""
    video_url: Optional[str] = None       # remote URL from Veo
    cached_path: Optional[str] = None     # local .mp4 if downloaded
    cache_key: str = ""
    error: Optional[str] = None
    generated_at: str = ""
    provider: str = ""
    model: str = ""
    duration_s: float = 0.0
    attempts: int = 0

    @property
    def success(self) -> bool:
        return bool(self.video_url or self.cached_path) and not self.error

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VeoVideoProvider(ImageProvider):
    """Google Veo via the Gemini API's long-running operation protocol.

    Inherits from ImageProvider so the renderer can dispatch it the
    same way as image providers. `_generate_raw()` returns:

        {"video_url":   "https://…/clip.mp4", "is_video": True}
            OR
        {"video_bytes": b"…",  "mime_type": "video/mp4", "is_video": True}

    The renderer detects `is_video=True` and stamps `media_type="video"`
    on the resulting IdeaVisual + saves the cached file as `.mp4`.

    Flow:
      1. POST {endpoint}/models/{model}:predictLongRunning → operation name
      2. Poll GET {endpoint}/{operation_name} until done=True
      3. Extract video URL or inline base64 from response

    Veo can take 30-120s end-to-end. Default `timeout_s` is 300s.
    Veo is paid-only on Google AI Studio.
    """

    name = "veo"

    def __init__(
        self,
        api_key: str = "",
        model: str = "veo-3.0-generate-001",
        endpoint: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout_s: float = 300.0,
        poll_interval_s: float = 5.0,
        duration_s: int = 6,
        aspect_ratio: str = "16:9",
        person_generation: Optional[str] = None,
    ) -> None:
        super().__init__(api_key, model, endpoint, timeout_s)
        # endpoint normalization: super stored it with rstrip("/")
        self.poll_interval_s = poll_interval_s
        self.duration_s = duration_s
        self.aspect_ratio = aspect_ratio
        # `personGeneration` valid values vary per Veo model + region.
        # Veo 2 accepts "dont_allow" / "allow_adult" / "allow_all";
        # Veo 3 (as of writing) rejects "allow_adult" on most tiers.
        # Default to None → we omit the parameter from the request, so
        # Google uses its server-side default. Override only if you know
        # what your specific model accepts.
        self.person_generation = person_generation

    @property
    def is_configured(self) -> bool:
        if not self.api_key:
            return False
        if self.api_key.startswith("sk-xxx"):
            return False
        if self.api_key.startswith("gen-lang-client-"):
            return False
        return True

    def generate(self, prompt: str) -> Dict[str, Any]:
        """Override base `generate()` to catch the gen-lang-client-… case
        with a specific error before is_configured short-circuits."""
        if self.api_key and self.api_key.startswith("gen-lang-client-"):
            return {"error": (
                "The value provided is a Google client identifier "
                "(gen-lang-client-…), not an API key. Get the real "
                "key from https://aistudio.google.com/apikey."
            )}
        return super().generate(prompt)

    def _generate_raw(self, prompt: str) -> Dict[str, Any]:
        """ImageProvider interface — returns a dict with the same
        keys image providers use, plus `is_video=True` so the renderer
        knows to save the file as .mp4 and stamp media_type."""
        result = self.generate_video(prompt)
        if "error" in result:
            return result
        result["is_video"] = True
        return result

    def generate_video(self, prompt: str) -> Dict[str, Any]:
        """Synchronous: submit → poll → return result dict.

        Returns one of:
          {"video_url": "..."}                — happy path with URL
          {"video_bytes": b"...", "mime_type": "video/mp4"}  — inline base64
          {"error": "..."}                    — anything went wrong
        """
        if not _HAS_REQUESTS:
            return {"error": "requests library not installed"}
        # Client-identifier check stays here too for direct callers of
        # generate_video() that bypass generate().
        if self.api_key.startswith("gen-lang-client-"):
            return {"error": (
                "The value provided is a Google client identifier "
                "(gen-lang-client-…), not an API key. Get the real "
                "key from https://aistudio.google.com/apikey."
            )}
        if not self.is_configured:
            return {"error": (
                "Veo API key not configured. Veo requires paid Google "
                "AI Studio credit at https://aistudio.google.com/usage."
            )}

        # 1. Submit
        submit_url = (
            f"{self.endpoint}/models/{self.model}:predictLongRunning"
        )
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        parameters: Dict[str, Any] = {
            "aspectRatio": self.aspect_ratio,
            "durationSeconds": int(self.duration_s),
            "sampleCount": 1,
        }
        # Only include personGeneration when explicitly set — different
        # Veo models accept different values and the wrong one returns
        # `HTTP 400: allow_adult for personGeneration is currently not
        # supported`. When omitted, Google uses its safer server-side
        # default.
        if self.person_generation:
            parameters["personGeneration"] = self.person_generation
        body = {
            "instances": [{"prompt": prompt}],
            "parameters": parameters,
        }
        try:
            r = requests.post(submit_url, headers=headers,
                                 json=body, timeout=60)
        except requests.RequestException as e:
            return {"error": f"network error on submit: {e}"}
        if r.status_code != 200:
            return {"error": (
                f"submit HTTP {r.status_code}: {r.text[:400]}"
            )}
        try:
            sub = r.json()
        except ValueError:
            return {"error": "submit response was not JSON"}
        op_name = sub.get("name")
        if not op_name:
            return {"error": f"submit response had no operation name: {sub}"}

        # 2. Poll the operation until done.
        poll_url = f"{self.endpoint}/{op_name}"
        deadline = time.time() + self.timeout_s
        while time.time() < deadline:
            try:
                pr = requests.get(poll_url, headers=headers, timeout=30)
            except requests.RequestException:
                time.sleep(self.poll_interval_s)
                continue
            if pr.status_code != 200:
                time.sleep(self.poll_interval_s)
                continue
            try:
                p = pr.json()
            except ValueError:
                time.sleep(self.poll_interval_s)
                continue
            if not p.get("done"):
                time.sleep(self.poll_interval_s)
                continue
            # 3. Done — extract video URL or inline bytes.
            if "error" in p:
                err = p["error"]
                msg = err.get("message", str(err)) if isinstance(
                    err, dict
                ) else str(err)
                return {"error": f"operation failed: {msg}"}
            resp = p.get("response") or {}
            gen = (resp.get("generateVideoResponse") or {})
            samples = gen.get("generatedSamples") or []
            if not samples:
                # Some Veo variants nest the video under a different key.
                vids = resp.get("videos") or []
                if vids and isinstance(vids[0], dict):
                    samples = [{"video": vids[0]}]
            if not samples:
                return {"error": (
                    f"operation done but no samples in response: "
                    f"{json.dumps(p)[:300]}"
                )}
            video = (samples[0] or {}).get("video") or {}
            uri = video.get("uri") or video.get("url")
            if uri:
                return {"video_url": uri}
            inline = video.get("bytesBase64Encoded") or video.get("data")
            if inline:
                try:
                    return {
                        "video_bytes": base64.b64decode(inline),
                        "mime_type": video.get("mimeType", "video/mp4"),
                    }
                except Exception as e:
                    return {"error": f"base64 decode failed: {e}"}
            return {"error": (
                f"sample had no video uri or bytes: "
                f"{json.dumps(samples[0])[:300]}"
            )}
        return {"error": (
            f"polling timed out after {self.timeout_s}s "
            f"(Veo can take 30-120s; consider raising timeout)"
        )}


# ── Register Veo into PROVIDER_REGISTRY (after class definition) ──────────
# VeoVideoProvider is defined here, BELOW PROVIDER_REGISTRY, because Veo's
# class itself depends on ImageProvider being defined first AND has a lot
# of detail we wanted physically separated from image providers. Patch
# it into the registry + defaults now that the class exists.

PROVIDER_REGISTRY["veo"] = VeoVideoProvider
PROVIDER_DEFAULTS["veo"] = {
    "model":    "veo-3.0-generate-001",
    "endpoint": "https://generativelanguage.googleapis.com/v1beta",
    "known_models": list(VEO_VIDEO_MODELS),
}

# Grok defaults — image-only. xAI has no public video model yet, so the
# Visual Simulation tab keeps using `veo` for animations regardless of
# what the image provider is set to.
PROVIDER_DEFAULTS["grok"] = {
    "model":    "grok-2-image",
    "endpoint": "https://api.x.ai/v1",
    "known_models": [
        "grok-2-image",
        "grok-2-image-1212",
        "grok-2-image-latest",
    ],
}


# ── Veo animation styles catalog ──────────────────────────────────────────
#
# Each style is a named animation pattern with its own prompt template.
# Multiple styles let users go from "boring method diagram animation"
# to "particle flow / network growth / zoom reveal / before-after split"
# with one click.
#
# Add a new style: drop another entry into the dict. The Visual
# Simulation tab picks them up automatically.

VEO_ANIMATION_STYLES: Dict[str, Dict[str, str]] = {
    "method_animation": {
        "label": "⚙️ Method animation",
        "description": (
            "Icons + arrows appear in sequence, showing data flow."
        ),
        "prompt_template": (
            "A short, calm animation (5-8 seconds) showing a method "
            "diagram coming alive: three or four flat-design icons "
            "appear in sequence, connected by animated arrows showing "
            "data flow from left to right. Editorial style, soft sky-"
            "blue and deep navy on a light background, no text labels. "
            "Smooth camera movement, no people. "
            "Topic: {title}. Method: {method_excerpt}."
        ),
    },
    "result_reveal": {
        "label": "📊 Result reveal",
        "description": (
            "A scientific bar chart draws itself axis-by-axis."
        ),
        "prompt_template": (
            "A short, calm animation (5-8 seconds) showing a "
            "scientific bar chart drawing itself axis-by-axis to "
            "reveal hypothesized results. Editorial style, soft "
            "sky-blue + amber accents on a light background, no text "
            "labels. Smooth camera movement. "
            "Topic: {title}. Hypothesis: {hypothesis_excerpt}."
        ),
    },
    "zoom_reveal": {
        "label": "🔍 Zoom reveal",
        "description": (
            "Wide aerial view zooming smoothly into the core concept."
        ),
        "prompt_template": (
            "A short, calm animation (5-8 seconds) starting from a "
            "wide aerial / top-down view that smoothly zooms into a "
            "central concept icon over the course of the clip. "
            "Editorial style, soft sky-blue and deep navy on a light "
            "background, no text labels. Continuous camera zoom, no "
            "cuts, no people. "
            "Topic: {title}. Concept: {motivation_excerpt}."
        ),
    },
    "before_after": {
        "label": "🔄 Before / after split",
        "description": (
            "Side-by-side: baseline (left, slower) vs proposed (right, faster)."
        ),
        "prompt_template": (
            "A short, calm animation (5-8 seconds) showing a side-by-"
            "side split screen: left panel = baseline approach "
            "(animated slower / less effective), right panel = "
            "proposed approach (animated faster / more effective). "
            "Editorial style, soft sky-blue on the left, amber accents "
            "on the right, no text labels. Continuous synchronized "
            "animation. "
            "Topic: {title}. Proposed method: {method_excerpt}."
        ),
    },
    "network_growth": {
        "label": "🕸️ Network growth",
        "description": (
            "Nodes appear one by one; edges connect into a network."
        ),
        "prompt_template": (
            "A short, calm animation (5-8 seconds) showing a network "
            "graph growing: nodes appear one by one in a spatial "
            "layout, then edges connect them into a coherent structure "
            "by the end of the clip. Editorial style, soft sky-blue "
            "nodes and deep navy edges on a light background, no text "
            "labels. Smooth camera, no people. "
            "Topic: {title}."
        ),
    },
    "particle_flow": {
        "label": "✨ Particle flow",
        "description": (
            "Particles flow through a system from input to output."
        ),
        "prompt_template": (
            "A short, calm animation (5-8 seconds) showing particles "
            "of light flowing from a source on the left, through a "
            "processing system in the middle, into a result on the "
            "right. Editorial style, soft sky-blue and amber on a "
            "dark navy background, no text labels. Smooth camera, no "
            "people. "
            "Topic: {title}. Process: {method_excerpt}."
        ),
    },
}


def build_video_prompt(
    idea: Dict[str, Any],
    style: str = "method_animation",
    method_excerpt_chars: int = 200,
) -> str:
    """Build a Veo prompt for a short animated explainer of one idea.

    `style` selects from VEO_ANIMATION_STYLES (6 named patterns).
    Unknown styles fall back to method_animation rather than raising.
    """
    if not isinstance(idea, dict):
        raise TypeError("idea must be a dict")
    title = str(idea.get("title", "") or "").strip() or "(untitled idea)"
    method = str(idea.get("method", "") or "").strip()
    motivation = str(idea.get("motivation", "") or "").strip()
    hypothesis = str(idea.get("hypothesis", "") or "").strip()

    def _excerpt(text: str, limit: int = method_excerpt_chars) -> str:
        t = text[:limit].rstrip()
        return (t + "…") if len(text) > limit else t

    method_excerpt = _excerpt(method) or "the proposed approach"
    motivation_excerpt = _excerpt(motivation) or "the core insight"
    hypothesis_excerpt = _excerpt(hypothesis, 200) or "the falsifiable claim"

    template = (
        VEO_ANIMATION_STYLES.get(style)
        or VEO_ANIMATION_STYLES["method_animation"]
    )["prompt_template"]
    return template.format(
        title=title,
        method_excerpt=method_excerpt,
        motivation_excerpt=motivation_excerpt,
        hypothesis_excerpt=hypothesis_excerpt,
    )


# ── Main renderer ───────────────────────────────────────────────────────────

class NanoBananaImageRenderer:
    """Render one or many ideas into visual abstracts.

    Lazily downloads images to the disk cache so subsequent calls for
    the same idea are instant. Thread-safe for concurrent reads;
    concurrent writes go through atomic-write-then-rename in DiskCache.
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "",
        provider: Optional[ImageProvider] = None,
        provider_name: str = "",
        cache_dir: str = ".ideagraph_visual_cache",
        max_concurrent: int = 3,
        timeout_s: float = 120.0,
        endpoint: str = "",
    ) -> None:
        # Provider resolution:
        #   1. Explicit `provider=` instance → use it as-is
        #   2. Explicit `provider_name=` → look up class in PROVIDER_REGISTRY
        #   3. NANO_BANANA_PROVIDER env var
        #   4. config.NANO_BANANA_PROVIDER
        #   5. Default = flux_bfl
        if isinstance(provider, ImageProvider):
            self.provider = provider
        else:
            resolved_name = (
                (provider_name or "").strip().lower()
                or os.getenv("NANO_BANANA_PROVIDER", "").strip().lower()
                or self._provider_name_from_config()
                or "flux_bfl"
            )
            ProviderCls = PROVIDER_REGISTRY.get(
                resolved_name, FluxBFLProvider,
            )
            # Resolve API key. Provider-specific env vars (e.g.
            # GROK_API_KEY for the `grok` provider) take precedence over
            # the generic NANO_BANANA_API_KEY chain so users can keep
            # multiple keys in .env and switch providers without re-pasting.
            api_key = (
                api_key
                or self._provider_specific_key(resolved_name)
                or os.getenv("NANO_BANANA_API_KEY", "")
                or os.getenv("BFL_API_KEY", "")
                or self._key_from_config()
                or self._key_from_nanobang_config()
            )
            # Snap model/endpoint to provider-specific defaults if the
            # caller didn't override them. Lets users switch providers
            # in the admin UI without manually updating model + endpoint.
            defaults = PROVIDER_DEFAULTS.get(resolved_name, {})
            effective_model = model or defaults.get("model", "")
            effective_endpoint = endpoint or defaults.get("endpoint", "")
            self.provider = ProviderCls(
                api_key=api_key, model=effective_model,
                endpoint=effective_endpoint, timeout_s=timeout_s,
            )
        self.cache = DiskCache(cache_dir)
        self.max_concurrent = max(1, int(max_concurrent))

    @staticmethod
    def _provider_name_from_config() -> str:
        try:
            import config  # type: ignore
            return (getattr(config, "NANO_BANANA_PROVIDER", "") or "").lower()
        except Exception:
            return ""

    @staticmethod
    def _provider_specific_key(provider_name: str) -> str:
        """Return the per-provider API key (env var + config attr) if the
        provider has its own canonical key name. Empty string otherwise —
        the caller then falls back to the generic NANO_BANANA_API_KEY chain.
        """
        # Map provider name → (env_var, config_attr).
        per_provider = {
            "grok": ("GROK_API_KEY", "GROK_API_KEY"),
        }
        entry = per_provider.get(provider_name)
        if not entry:
            return ""
        env_var, cfg_attr = entry
        v = os.getenv(env_var, "") or os.getenv("XAI_API_KEY", "")  # XAI alias
        if v:
            return v
        try:
            import config  # type: ignore
            return getattr(config, cfg_attr, "") or ""
        except Exception:
            return ""

    @staticmethod
    def _key_from_config() -> str:
        try:
            import config  # type: ignore
            return getattr(config, "NANO_BANANA_API_KEY", "") or ""
        except Exception:
            return ""

    @staticmethod
    def _key_from_nanobang_config() -> str:
        p = Path(".nanobang_config")
        if not p.exists():
            return ""
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == "NANO_BANANA_API_KEY":
                    return v.strip().strip('"').strip("'")
        except Exception:
            return ""
        return ""

    @property
    def is_configured(self) -> bool:
        return self.provider.is_configured

    def _download_image(self, url: str) -> Optional[bytes]:
        """Best-effort fetch of the image bytes. Returns None on failure
        so the URL-only path still works."""
        if not _HAS_REQUESTS or not url:
            return None
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200:
                return r.content
        except Exception:
            pass
        return None

    def render_n_samples(
        self,
        idea: Dict[str, Any],
        n: int = 4,
        style: str = DEFAULT_STYLE,
        force: bool = True,
    ) -> List[IdeaVisual]:
        """Generate N variants of the same idea/style — useful when
        you want to pick the best from several candidates.

        Each variant gets a sample-index marker injected into the
        prompt so the cache treats them as distinct entries (sha256 of
        the prompt is the cache key — same prompt = same key = same
        cached result, which is the opposite of what we want here).

        `force=True` by default since the *point* of N-samples is to
        explore variety; cache hits would defeat that. Set False to
        re-use cached samples after the first run.

        Returns a list of `n` IdeaVisuals — successes and failures
        intermixed; check `.success` per item.
        """
        if not isinstance(idea, dict) or not idea:
            raise ValueError("idea must be a non-empty dict")
        if n <= 0:
            return []
        if n > 16:
            raise ValueError(
                "n samples capped at 16 — generating more is costly "
                "and rarely useful. Pick from these first."
            )
        out: List[IdeaVisual] = []
        base_prompt = build_prompt(idea, style=style)
        for i in range(n):
            # Inject a sample-index marker — invisible to the model
            # (it's a trailing comment-style cue) but creates a unique
            # cache key per sample, AND the model picks up subtle
            # variation from the changing prompt.
            variant_prompt = (
                f"{base_prompt} [variant {i + 1} of {n} — "
                f"emphasize different visual angle]"
            )
            v = self.render(
                idea, force=force, prompt_override=variant_prompt,
            )
            out.append(v)
        return out

    def render(
        self,
        idea: Dict[str, Any],
        force: bool = False,
        prompt_override: Optional[str] = None,
    ) -> IdeaVisual:
        """Render a single idea synchronously.

        Cache flow:
          1. Build prompt → cache_key.
          2. If `force=False` AND cache has it, return cached.
          3. Otherwise, call provider, download bytes, persist, return.
        """
        title = str(idea.get("title", "") or "").strip() or "(untitled)"
        prompt = prompt_override or build_prompt(idea)
        key = cache_key_for_prompt(prompt)

        if not force:
            cached = self.cache.get(key)
            if cached and cached.success:
                cached.cache_key = key
                return cached

        if not self.is_configured:
            return IdeaVisual(
                idea_title=title, prompt=prompt, cache_key=key,
                error=(
                    "NANO_BANANA_API_KEY not configured. "
                    "Set it via Admin Dashboard → 🎨 Visual Rendering "
                    "(persists to .env). For Google, you also need "
                    "paid AI Studio credit for image-gen models — "
                    "https://aistudio.google.com/usage."
                ),
                generated_at=_utcnow_iso(),
                provider=self.provider.name, model=self.provider.model,
            )

        # Retry around the provider call.
        attempts, raw = 0, {}
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            attempts = attempt
            raw = self.provider.generate(prompt)
            if "error" not in raw:
                break
            err = raw.get("error", "")
            # Don't retry on user-facing config errors.
            if "API key" in err or "not installed" in err:
                break
            if attempt < max_attempts:
                time.sleep(2 ** (attempt - 1))  # 1s, 2s

        visual = IdeaVisual(
            idea_title=title, prompt=prompt, cache_key=key,
            generated_at=_utcnow_iso(),
            provider=self.provider.name, model=self.provider.model,
            attempts=attempts,
        )
        if "error" in raw:
            visual.error = str(raw["error"])[:400]
        else:
            # Video providers return `is_video=True` + `video_url` /
            # `video_bytes`; map them onto the same image_url /
            # cached_path fields with media_type="video" so the rest of
            # the renderer + cache work uniformly.
            if raw.get("is_video"):
                visual.media_type = "video"
                visual.image_url = raw.get("video_url")
                media_bytes = raw.get("video_bytes")
            else:
                visual.media_type = "image"
                visual.image_url = raw.get("image_url")
                media_bytes = raw.get("image_bytes")
            if not media_bytes and visual.image_url:
                # Don't try to download videos — they can be hundreds of MB.
                # Keep the URL only; the UI displays via st.video(url).
                if visual.media_type == "image":
                    media_bytes = self._download_image(visual.image_url)
            self.cache.put(key, visual, image_bytes=media_bytes)

        return visual

    # ── Batch async ─────────────────────────────────────────────────────

    async def render_async(self, idea: Dict[str, Any],
                              force: bool = False) -> IdeaVisual:
        """Async render — runs the sync render() in a thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.render, idea, force)

    async def render_batch_async(
        self,
        ideas: List[Dict[str, Any]],
        force: bool = False,
    ) -> List[IdeaVisual]:
        """Render a batch with bounded concurrency. Returns results in
        the same order as the input list."""
        if not ideas:
            return []
        sem = asyncio.Semaphore(self.max_concurrent)

        async def _one(idea):
            async with sem:
                return await self.render_async(idea, force=force)

        return await asyncio.gather(*(_one(i) for i in ideas))

    def render_batch(
        self,
        ideas: List[Dict[str, Any]],
        force: bool = False,
    ) -> List[IdeaVisual]:
        """Sync batch wrapper around render_batch_async."""
        if not ideas:
            return []
        try:
            return asyncio.run(self.render_batch_async(ideas, force=force))
        except RuntimeError:
            # Already in an event loop (e.g. inside Streamlit on some
            # builds). Fall back to sequential sync.
            return [self.render(i, force=force) for i in ideas]

    # ── Multi-panel paper-figure set (Kimi-PPT-style) ───────────────────

    def render_figure_set(
        self,
        idea: Dict[str, Any],
        panels: Optional[List[str]] = None,
        force: bool = False,
    ) -> List[FigurePanel]:
        """Generate a SET of panel-specific images for one idea.

        Each panel uses its own prompt template (concept, method,
        experiment, results, comparison, limitations — see
        FIGURE_TEMPLATES). Panels are independently cached: re-rendering
        the same idea with the same panel list is free.

        `panels` defaults to DEFAULT_FIGURE_SET (concept + method +
        experiment + results — the canonical 4-figure paper layout).
        Returns a list of FigurePanel in the order requested. Failed
        panels are included with `visual.error` set.
        """
        if not isinstance(idea, dict) or not idea:
            raise ValueError("idea must be a non-empty dict")
        panels_list = list(panels) if panels else list(DEFAULT_FIGURE_SET)
        if not panels_list:
            return []
        unknown = [p for p in panels_list if p not in FIGURE_TEMPLATES]
        if unknown:
            raise ValueError(
                f"unknown panel(s): {unknown}. "
                f"Valid: {sorted(FIGURE_TEMPLATES)}"
            )
        out: List[FigurePanel] = []
        for i, panel_id in enumerate(panels_list):
            prompt = build_panel_prompt(idea, panel_id)
            visual = self.render(
                idea, force=force, prompt_override=prompt,
            )
            out.append(FigurePanel(
                panel_id=panel_id,
                label=FIGURE_TEMPLATES[panel_id]["label"],
                visual=visual,
                order=i,
            ))
        return out

    async def render_figure_set_async(
        self,
        idea: Dict[str, Any],
        panels: Optional[List[str]] = None,
        force: bool = False,
    ) -> List[FigurePanel]:
        """Async version — runs panels concurrently up to max_concurrent."""
        if not isinstance(idea, dict) or not idea:
            raise ValueError("idea must be a non-empty dict")
        panels_list = list(panels) if panels else list(DEFAULT_FIGURE_SET)
        if not panels_list:
            return []
        unknown = [p for p in panels_list if p not in FIGURE_TEMPLATES]
        if unknown:
            raise ValueError(
                f"unknown panel(s): {unknown}. "
                f"Valid: {sorted(FIGURE_TEMPLATES)}"
            )
        sem = asyncio.Semaphore(self.max_concurrent)

        async def _one(panel_id: str, order: int) -> FigurePanel:
            async with sem:
                prompt = build_panel_prompt(idea, panel_id)
                loop = asyncio.get_event_loop()
                visual = await loop.run_in_executor(
                    None,
                    lambda p=prompt: self.render(idea, force=force,
                                                       prompt_override=p),
                )
                return FigurePanel(
                    panel_id=panel_id,
                    label=FIGURE_TEMPLATES[panel_id]["label"],
                    visual=visual,
                    order=order,
                )

        coros = [_one(p, i) for i, p in enumerate(panels_list)]
        return await asyncio.gather(*coros)


# ── Convenience helpers (top-level — match the script's API) ──────────────

def render_idea_visuals(
    ideas: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    model: str = "flux-pro-1.0",
    cache_dir: str = ".ideagraph_visual_cache",
    max_concurrent: int = 3,
    force: bool = False,
) -> List[IdeaVisual]:
    """Sync batch helper. Matches the top-level API documented in the
    setup script (`from ideagraph_image_renderer import render_idea_visuals`)."""
    renderer = NanoBananaImageRenderer(
        api_key=api_key or "",
        model=model,
        cache_dir=cache_dir,
        max_concurrent=max_concurrent,
    )
    return renderer.render_batch(ideas, force=force)


def display_idea_with_visual(
    idea: Dict[str, Any],
    st_module,
    visual: Optional[IdeaVisual] = None,
    show_prompt: bool = False,
) -> None:
    """Render an idea + its visual abstract into the given Streamlit
    container/module. Safe to call without `visual` — shows a placeholder.

    Usage (from inside a Streamlit script):
        from ideagraph_image_renderer import display_idea_with_visual
        display_idea_with_visual(idea_dict, st, visual=v)
    """
    if visual is None:
        st_module.info(
            "No visual generated yet. Click **🎨 Generate visual abstract** "
            "to create one (~10–30s)."
        )
        return
    title = visual.idea_title or idea.get("title", "Untitled")
    if visual.success:
        src = visual.cached_path or visual.image_url
        if visual.is_video:
            # Video render — st.video() takes a URL, path, or bytes.
            try:
                st_module.video(src)
                st_module.caption(title)
            except Exception as e:
                st_module.warning(
                    f"Couldn't display video: {e}. URL: {src}"
                )
        else:
            # Streamlit's st.image accepts a path OR a remote URL.
            try:
                st_module.image(
                    src, caption=title, use_container_width=True,
                )
            except TypeError:
                # Older Streamlit API used use_column_width.
                st_module.image(
                    src, caption=title, use_column_width=True,
                )
        if show_prompt and visual.prompt:
            with st_module.expander("View prompt used"):
                st_module.code(visual.prompt, language=None)
    else:
        st_module.warning(
            f"⚠️ Visual generation failed for **{title}**.\n\n"
            f"Provider: `{visual.provider}` · Model: `{visual.model}` · "
            f"Attempts: {visual.attempts}\n\n"
            f"Error: {visual.error or 'unknown'}"
        )


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# ── Save / export helpers ─────────────────────────────────────────────────
#
# Generated visuals live in `.ideagraph_visual_cache/<sha256>.{bin,mp4}` —
# stable cache keys but unfriendly filenames. These helpers produce
# meaningful names (idea title + style + timestamp) and bytes for
# Streamlit's st.download_button. They also bundle multiple visuals
# into a zip for batch-download.

import re as _save_re  # local alias to avoid colliding with re at module top


def safe_filename(
    title: str,
    *,
    style: str = "",
    panel_id: str = "",
    media_type: str = "image",
    timestamp: bool = True,
) -> str:
    """Build a user-friendly download filename for a generated visual.

    Examples:
      safe_filename("Transformer attention", style="isometric_3d")
        → "Transformer_attention__isometric_3d__20260527_142315.png"
      safe_filename("X", panel_id="method", media_type="video")
        → "X__method__20260527_142315.mp4"

    Filesystem-safe across Windows / macOS / Linux. Strips Unicode
    punctuation, collapses runs of underscores, caps length so very
    long titles don't break the download.
    """
    slug = _save_re.sub(r"[^A-Za-z0-9]+", "_", (title or "untitled")).strip("_")
    slug = (slug[:60] or "untitled")
    parts = [slug]
    if panel_id:
        parts.append(_save_re.sub(r"[^A-Za-z0-9]+", "_", panel_id).strip("_"))
    if style:
        parts.append(_save_re.sub(r"[^A-Za-z0-9]+", "_", style).strip("_"))
    if timestamp:
        parts.append(
            datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        )
    name = "__".join(p for p in parts if p)
    ext = ".mp4" if media_type == "video" else ".png"
    return f"{name}{ext}"


def read_visual_bytes(
    visual: "IdeaVisual",
    download_url_if_missing: bool = True,
    timeout_s: float = 60.0,
) -> Optional[bytes]:
    """Return the raw bytes of a generated visual, downloading from
    the remote URL if we only have a URL (no cached file).

    Returns None if both paths fail — caller should fall back to
    showing the URL as a link instead of a download button.

    Video URLs can be large; we still fetch them when requested
    (only when the user explicitly clicks save), but cap via timeout.
    """
    # 1. Prefer the local cached file when we have one.
    if visual and visual.cached_path:
        try:
            return Path(visual.cached_path).read_bytes()
        except Exception as e:
            log.warning(
                f"read_visual_bytes: cached_path "
                f"{visual.cached_path!r} failed: {e}"
            )
    # 2. Fall back to fetching the URL.
    if (
        download_url_if_missing
        and visual and visual.image_url
        and _HAS_REQUESTS
    ):
        try:
            r = requests.get(visual.image_url, timeout=timeout_s)
            if r.status_code == 200:
                return r.content
            log.warning(
                f"read_visual_bytes: URL fetch returned "
                f"HTTP {r.status_code}"
            )
        except Exception as e:
            log.warning(f"read_visual_bytes: URL fetch failed: {e}")
    return None


def bundle_visuals_as_zip(
    visuals: List["IdeaVisual"],
    idea_title: str,
    *,
    panel_labels: Optional[List[str]] = None,
) -> Optional[bytes]:
    """Bundle a list of visuals into a single in-memory ZIP archive.

    Each entry is named by `safe_filename()` so the ZIP unpacks to
    human-readable files. Skips visuals that aren't successful.
    Returns the ZIP bytes, or None if every visual was a failure.

    `panel_labels` (optional, parallel to `visuals`): if provided,
    each filename includes the panel label as a `panel_id` — useful
    when bundling a multi-panel figure set.
    """
    import io
    import zipfile
    if not visuals:
        return None
    buf = io.BytesIO()
    n_added = 0
    with zipfile.ZipFile(buf, mode="w",
                            compression=zipfile.ZIP_STORED) as zf:
        for i, v in enumerate(visuals):
            if not (v and v.success):
                continue
            data = read_visual_bytes(v)
            if data is None:
                continue
            panel_id = ""
            if panel_labels and i < len(panel_labels):
                # The label may contain emoji + words; pull the
                # word portion via the slug normalization.
                panel_id = panel_labels[i] or ""
            name = safe_filename(
                idea_title or v.idea_title or "untitled",
                panel_id=panel_id,
                media_type=v.media_type,
                timestamp=False,  # one shared timestamp on the zip
            )
            # If multiple visuals end up with the same name (e.g.
            # same style + same panel_id), append an index.
            try:
                existing = {info.filename for info in zf.infolist()}
                if name in existing:
                    stem, dot, ext = name.rpartition(".")
                    name = f"{stem}__{i + 1}.{ext}" if dot else f"{name}__{i + 1}"
            except Exception:
                pass
            zf.writestr(name, data)
            n_added += 1
    if n_added == 0:
        return None
    return buf.getvalue()
