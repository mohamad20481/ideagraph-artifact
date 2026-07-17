# IdeaGraph Visual Rendering — Integration Guide

Add paper-figure-style visual abstracts to IdeaGraph ideas via the FLUX
image API (BlackForest Labs, Nano Banana, Runway, or any FLUX-compatible
endpoint).

## What you get

- A `🎨 Visual abstract` expander on every idea card in the Ideas tab —
  one-click generation, on-disk cache (≈ free re-clicks).
- `NanoBananaImageRenderer` — provider-pluggable Python class for use in
  your own scripts.
- `render_idea_visuals(ideas, api_key=...)` — sync batch helper.
- `display_idea_with_visual(idea, st, visual=v)` — Streamlit helper.
- Admin toggle `🎨 Visual abstract rendering` in the Admin Dashboard →
  🎚️ Feature Toggles tab — switch the feature OFF without removing code.

## Quick start

### 1. Install dependencies

Already in `requirements.txt` if you ran the setup script; otherwise:

```bash
pip install aiohttp tenacity requests
```

### 2. Set your API key

Three ways, in priority order:

```bash
# (a) Environment variable (recommended)
export NANO_BANANA_API_KEY='your_flux_key_here'

# (b) .env file in the project root
echo "NANO_BANANA_API_KEY=your_flux_key_here" >> .env

# (c) .nanobang_config file in CWD (template auto-created)
cat > .nanobang_config <<'EOF'
NANO_BANANA_API_KEY=your_flux_key_here
MODEL=flux-pro-1.0
EOF
```

The renderer also recognizes `BFL_API_KEY` (BlackForest Labs convention).

Get a key from:
- **BlackForest Labs (FLUX official)**: https://api.bfl.ml — recommended
- **Runway ML**: https://www.runwayml.com
- **Nano Banana**: https://nanobang.com

### 3. Use it from inside IdeaGraph

Just refresh the app — the 🎨 panel appears on every idea card. Click
`🎨 Generate visual abstract` and wait ~10–30s.

### 4. Use it from your own scripts

```python
from ideagraph_image_renderer import (
    NanoBananaImageRenderer, render_idea_visuals, IdeaVisual,
)

# Single idea
renderer = NanoBananaImageRenderer()  # picks up env var automatically
visual = renderer.render({
    "title": "Linear attention via random feature maps",
    "method": "Approximate softmax attention with random features…",
    "methodology_type": "empirical_study",
})

if visual.success:
    print("Image URL:", visual.image_url)
    print("Cached at:", visual.cached_path)
else:
    print("Failed:", visual.error)

# Batch
ideas = [{...}, {...}, {...}]
visuals = render_idea_visuals(ideas, api_key="...")
```

## API reference

### `IdeaVisual` (dataclass)

| Field | Type | Description |
|---|---|---|
| `idea_title` | str | Required. The idea's title. |
| `prompt` | str | The prompt that was sent to the API. |
| `image_url` | str \| None | Remote URL of the generated image. |
| `cached_path` | str \| None | Local path to downloaded image bytes. |
| `cache_key` | str | sha256(prompt) — also the filename stem in the cache. |
| `error` | str \| None | Error message if generation failed. |
| `generated_at` | str | ISO 8601 UTC timestamp. |
| `provider` | str | The provider name (e.g. `flux_bfl`). |
| `model` | str | The model used (e.g. `flux-pro-1.0`). |
| `attempts` | int | Number of retry attempts. |

`visual.success` is True when an image was produced and no error.

### `NanoBananaImageRenderer(api_key, model, provider, cache_dir, max_concurrent, timeout_s, endpoint)`

- `api_key` (str): Defaults to `$NANO_BANANA_API_KEY` then `$BFL_API_KEY`
  then `config.NANO_BANANA_API_KEY` then `.nanobang_config`.
- `model` (str): Default `flux-pro-1.0`.
- `provider` (ImageProvider): Override the entire backend. See
  *Custom providers* below.
- `cache_dir` (str): Default `.ideagraph_visual_cache/`.
- `max_concurrent` (int): Concurrency cap for `render_batch_async`. Default 3.
- `timeout_s` (float): Total provider-call timeout. Default 120s.
- `endpoint` (str): Default `https://api.bfl.ml/v1`.

Methods:

- `render(idea, force=False, prompt_override=None) → IdeaVisual` (sync)
- `render_async(idea, force=False) → IdeaVisual`
- `render_batch(ideas, force=False) → List[IdeaVisual]` (sync wrapper)
- `render_batch_async(ideas, force=False) → List[IdeaVisual]`
- `is_configured` (property) — True when an API key is set.

### `render_idea_visuals(ideas, api_key=None, ...)` (top-level)

Sync batch helper. Matches the API documented in the setup script.

### `display_idea_with_visual(idea, st_module, visual=None, show_prompt=False)`

Renders an idea + its visual in a Streamlit container. Safe to call
with `visual=None` — shows a placeholder.

## Cache behavior

- Cache key = `sha256(prompt)`. Stable across runs.
- Two files per cached entry:
  - `.ideagraph_visual_cache/<key>.json` — metadata (URL, prompt, etc.)
  - `.ideagraph_visual_cache/<key>.bin` — raw image bytes
- The cache is read on every `render()` call. Pass `force=True` to bypass.
- Clear all cached visuals:

  ```python
  from ideagraph_image_renderer import DiskCache
  DiskCache(".ideagraph_visual_cache").clear()
  ```

## Custom providers

To swap in Runway, Replicate, or self-hosted FLUX, subclass
`ImageProvider`:

```python
from ideagraph_image_renderer import ImageProvider, NanoBananaImageRenderer

class RunwayProvider(ImageProvider):
    name = "runway"

    def _generate_raw(self, prompt: str) -> dict:
        import requests
        r = requests.post(
            "https://api.runwayml.com/v1/image/generate",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"prompt": prompt, "model": self.model},
            timeout=self.timeout_s,
        )
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}: {r.text[:200]}"}
        return {"image_url": r.json()["url"]}

renderer = NanoBananaImageRenderer(
    provider=RunwayProvider(api_key="...", model="..."),
)
```

The wrapper handles retry, caching, and IdeaVisual construction.

## Admin controls

In the Admin Dashboard → 🎚️ Feature Toggles tab:

- **🎨 Visual abstract rendering** — flip OFF to hide the panel from the
  Ideas tab and stop all API requests, without removing any code.

In the Admin Dashboard → 🔌 LLM Provider tab:

- The `NANO_BANANA_API_KEY` is reported alongside the other provider
  keys. You can edit `.env` and click **Reload Anthropic client** /
  hit refresh to pick up the new key.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| "NANO_BANANA_API_KEY not configured" | No env var, no .env entry, no .nanobang_config | Set the key in any of the 3 places (see Quick Start) |
| Provider error: "submit HTTP 401" | Bad / expired API key | Generate a new key at api.bfl.ml |
| Provider error: "submit HTTP 403 INSUFFICIENT_BALANCE" | Account out of credit | Top up the provider account |
| "polling timed out after 120s" | Server overloaded / large image | Retry — increase `timeout_s` constructor arg |
| Cache hit on a clearly-stale image | Same prompt → same key | Click **↻ Re-roll** or pass `force=True` |

## Cost notes

FLUX-pro-1.0 is ~$0.05/image at BlackForest Labs as of writing. With
the on-disk cache, you pay once per unique idea + only when you click
**↻ Re-roll**. A typical 20-idea archive costs ~$1 to fully visualize.

## Standalone demo

`ideagraph_streamlit_visuals.py` is a minimal demo script you can run
without the full IdeaGraph app:

```bash
streamlit run ideagraph_streamlit_visuals.py --server.port 8511
```

Useful for sanity-checking your API key + provider before integrating.
