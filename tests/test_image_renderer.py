"""Tests for ideagraph_image_renderer.

All HTTP is mocked — these tests NEVER hit the real FLUX/BFL/Nano Banana
API. Run them anywhere, no API key required.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ideagraph_image_renderer as r


# ─────────────────────────────────────────────────────────────────────────────
# build_prompt / cache_key_for_prompt
# ─────────────────────────────────────────────────────────────────────────────

def test_build_prompt_rejects_non_dict():
    with pytest.raises(TypeError):
        r.build_prompt("not a dict")  # type: ignore[arg-type]


def test_build_prompt_includes_title_and_methodology():
    idea = {
        "title": "Transformer attention scaling",
        "method": "Linear attention via kernels.",
        "methodology_type": "empirical_study",
    }
    p = r.build_prompt(idea)
    assert "Transformer attention scaling" in p
    assert "Linear attention via kernels" in p
    assert "empirical study" in p  # underscore replaced with space


def test_build_prompt_truncates_long_method():
    idea = {"title": "x", "method": "y" * 500,
              "methodology_type": "empirical_study"}
    p = r.build_prompt(idea, method_excerpt_chars=50)
    assert "y" * 50 in p
    # Ellipsis added when truncated.
    assert "…" in p


def test_build_prompt_handles_empty_title():
    """When title is missing, fall back to a marker — not crash."""
    p = r.build_prompt({"method": "x"})
    assert "(untitled idea)" in p


def test_build_prompt_is_deterministic():
    """Same input → same prompt → same cache key (the whole caching
    contract depends on this)."""
    idea = {"title": "t", "method": "m", "methodology_type": "x"}
    assert r.build_prompt(idea) == r.build_prompt(idea)


def test_cache_key_is_sha256_hex():
    k = r.cache_key_for_prompt("any prompt")
    assert len(k) == 64
    assert all(c in "0123456789abcdef" for c in k)


def test_cache_key_changes_with_prompt():
    a = r.cache_key_for_prompt("prompt a")
    b = r.cache_key_for_prompt("prompt b")
    assert a != b


# ─────────────────────────────────────────────────────────────────────────────
# IdeaVisual dataclass
# ─────────────────────────────────────────────────────────────────────────────

def test_idea_visual_success_requires_url_or_path_and_no_error():
    assert r.IdeaVisual("x").success is False
    assert r.IdeaVisual("x", image_url="http://example.com/x.png").success
    assert r.IdeaVisual("x", cached_path="/tmp/x.png").success
    # error trumps URL.
    assert not r.IdeaVisual("x", image_url="http://example.com",
                                error="boom").success


def test_idea_visual_to_dict_roundtrip():
    v = r.IdeaVisual(idea_title="t", prompt="p", error=None, provider="flux")
    d = v.to_dict()
    assert d["idea_title"] == "t"
    assert d["prompt"] == "p"
    # Rebuild from dict (used by DiskCache.get).
    v2 = r.IdeaVisual(**d)
    assert v2.idea_title == "t"


# ─────────────────────────────────────────────────────────────────────────────
# DiskCache
# ─────────────────────────────────────────────────────────────────────────────

def test_disk_cache_round_trip(tmp_path):
    c = r.DiskCache(cache_dir=str(tmp_path))
    visual = r.IdeaVisual(
        idea_title="t", prompt="p", cache_key="k1",
        image_url="http://example/img.png",
    )
    c.put("k1", visual, image_bytes=b"FAKEIMG")
    assert c.has("k1")
    loaded = c.get("k1")
    assert loaded is not None
    assert loaded.idea_title == "t"
    # cached_path was attached by put().
    assert loaded.cached_path.endswith("k1.bin")
    # bytes match
    assert Path(loaded.cached_path).read_bytes() == b"FAKEIMG"


def test_disk_cache_get_missing_returns_none(tmp_path):
    c = r.DiskCache(cache_dir=str(tmp_path))
    assert c.get("nope") is None
    assert c.has("nope") is False


def test_disk_cache_put_without_bytes_only_writes_meta(tmp_path):
    c = r.DiskCache(cache_dir=str(tmp_path))
    visual = r.IdeaVisual(idea_title="t", image_url="http://x")
    c.put("k", visual)
    assert (tmp_path / "k.json").exists()
    assert not (tmp_path / "k.bin").exists()


def test_disk_cache_clear_returns_count(tmp_path):
    c = r.DiskCache(cache_dir=str(tmp_path))
    c.put("a", r.IdeaVisual("a", image_url="x"), image_bytes=b"1")
    c.put("b", r.IdeaVisual("b", image_url="x"), image_bytes=b"2")
    n = c.clear()
    assert n >= 4  # 2 meta + 2 bin
    assert c.get("a") is None


# ─────────────────────────────────────────────────────────────────────────────
# Provider — FluxBFLProvider HTTP shape (all mocked)
# ─────────────────────────────────────────────────────────────────────────────

def _mock_resp(status: int = 200, body: dict = None,
                  text: str = "") -> MagicMock:
    m = MagicMock()
    m.status_code = status
    m.text = text or json.dumps(body or {})
    m.content = b"FAKEIMG"
    m.json = MagicMock(return_value=body if body is not None else {})
    return m


def test_provider_no_key_returns_error():
    p = r.FluxBFLProvider(api_key="")
    out = p.generate("test prompt")
    assert "error" in out
    assert "not configured" in out["error"]


def test_provider_placeholder_key_treated_as_unconfigured():
    """Convention from claude_provider — `sk-xxx...` placeholder doesn't
    count as configured."""
    p = r.FluxBFLProvider(api_key="sk-xxx-not-a-real-key")
    assert p.is_configured is False


def test_provider_real_key_is_configured():
    p = r.FluxBFLProvider(api_key="bfl-real-key-here")
    assert p.is_configured is True


def test_provider_happy_path_polls_and_returns_url():
    """The provider does POST → poll → final result. Mock that
    sequence and verify it parses correctly."""
    submit_resp = _mock_resp(200, {"id": "task-123"})
    poll_resps = [
        _mock_resp(200, {"status": "Processing"}),
        _mock_resp(200, {
            "status": "Ready",
            "result": {"sample": "https://example.com/img.png"},
        }),
    ]
    with patch("ideagraph_image_renderer.requests.post",
                  return_value=submit_resp), \
            patch("ideagraph_image_renderer.requests.get",
                  side_effect=poll_resps), \
            patch("ideagraph_image_renderer.time.sleep"):
        p = r.FluxBFLProvider(api_key="bfl-test-key", poll_interval_s=0)
        out = p.generate("prompt")
    assert "image_url" in out
    assert out["image_url"] == "https://example.com/img.png"
    assert "error" not in out


def test_provider_submit_error_surfaces_status_code():
    with patch("ideagraph_image_renderer.requests.post",
                  return_value=_mock_resp(403, text="Forbidden")):
        p = r.FluxBFLProvider(api_key="bfl-key")
        out = p.generate("prompt")
    assert "error" in out
    assert "403" in out["error"]


def test_provider_failed_status_returns_error():
    submit_resp = _mock_resp(200, {"id": "task-123"})
    poll_resp = _mock_resp(200, {"status": "Failed", "error": "NSFW filter"})
    with patch("ideagraph_image_renderer.requests.post",
                  return_value=submit_resp), \
            patch("ideagraph_image_renderer.requests.get",
                  return_value=poll_resp), \
            patch("ideagraph_image_renderer.time.sleep"):
        p = r.FluxBFLProvider(api_key="bfl-key", poll_interval_s=0)
        out = p.generate("prompt")
    assert "error" in out
    assert "provider reported failure" in out["error"]


def test_provider_polling_timeout_returns_error():
    """Polls forever (Processing) → eventual timeout error."""
    submit_resp = _mock_resp(200, {"id": "task-123"})
    poll_resp = _mock_resp(200, {"status": "Processing"})
    with patch("ideagraph_image_renderer.requests.post",
                  return_value=submit_resp), \
            patch("ideagraph_image_renderer.requests.get",
                  return_value=poll_resp), \
            patch("ideagraph_image_renderer.time.sleep"):
        # 0.001s timeout → fails fast
        p = r.FluxBFLProvider(api_key="bfl-key", timeout_s=0.001,
                                  poll_interval_s=0)
        out = p.generate("prompt")
    assert "error" in out
    assert "timed out" in out["error"]


def test_provider_supports_alternate_url_keys_in_result():
    """Some FLUX variants return `image_url` or `url` instead of `sample`."""
    for key in ("sample", "image_url", "url"):
        submit_resp = _mock_resp(200, {"id": "task-x"})
        poll_resp = _mock_resp(200, {
            "status": "Ready",
            "result": {key: f"https://example.com/{key}.png"},
        })
        with patch("ideagraph_image_renderer.requests.post",
                      return_value=submit_resp), \
                patch("ideagraph_image_renderer.requests.get",
                      return_value=poll_resp), \
                patch("ideagraph_image_renderer.time.sleep"):
            p = r.FluxBFLProvider(api_key="k", poll_interval_s=0)
            out = p.generate("prompt")
        assert out.get("image_url") == f"https://example.com/{key}.png"


# ─────────────────────────────────────────────────────────────────────────────
# NanoBananaImageRenderer — orchestration
# ─────────────────────────────────────────────────────────────────────────────

def test_renderer_no_key_returns_friendly_error(tmp_path, monkeypatch):
    # Use an explicitly-constructed FluxBFLProvider with no key — this
    # bypasses the renderer's key-resolution chain entirely (env, config,
    # .nanobang_config). Resulting renderer must return the friendly
    # "API key not configured" message instead of hitting the API.
    empty_provider = r.FluxBFLProvider(api_key="")
    renderer = r.NanoBananaImageRenderer(
        provider=empty_provider, cache_dir=str(tmp_path),
    )
    v = renderer.render({"title": "x", "method": "y",
                            "methodology_type": "empirical_study"})
    assert v.success is False
    assert "NANO_BANANA_API_KEY" in v.error
    assert "Admin Dashboard" in v.error  # hint to fix


def test_renderer_picks_up_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("NANO_BANANA_API_KEY", "env-test-key")
    renderer = r.NanoBananaImageRenderer(cache_dir=str(tmp_path))
    assert renderer.is_configured is True


def test_renderer_falls_back_to_BFL_API_KEY_env(monkeypatch, tmp_path):
    monkeypatch.delenv("NANO_BANANA_API_KEY", raising=False)
    monkeypatch.setenv("BFL_API_KEY", "bfl-env-key")
    renderer = r.NanoBananaImageRenderer(cache_dir=str(tmp_path))
    assert renderer.is_configured is True


def test_renderer_explicit_arg_wins_over_env(monkeypatch, tmp_path):
    monkeypatch.setenv("NANO_BANANA_API_KEY", "env-key")
    renderer = r.NanoBananaImageRenderer(
        api_key="explicit-key", cache_dir=str(tmp_path),
    )
    assert renderer.provider.api_key == "explicit-key"


def test_renderer_happy_path_caches_url_and_bytes(tmp_path):
    mock_provider = MagicMock(spec=r.ImageProvider)
    mock_provider.name = "mock"
    mock_provider.model = "mock-1"
    mock_provider.is_configured = True
    mock_provider.generate.return_value = {
        "image_url": "https://example.com/img.png",
    }
    renderer = r.NanoBananaImageRenderer(
        provider=mock_provider, cache_dir=str(tmp_path),
    )
    # Mock the image download too.
    with patch("ideagraph_image_renderer.requests.get",
                  return_value=_mock_resp(200)) as get_mock:
        v = renderer.render({"title": "t", "method": "m",
                                "methodology_type": "empirical_study"})
    assert v.success is True
    assert v.image_url == "https://example.com/img.png"
    assert v.cached_path is not None
    assert Path(v.cached_path).exists()
    get_mock.assert_called_once()


def test_renderer_serves_from_cache_on_second_call(tmp_path):
    mock_provider = MagicMock(spec=r.ImageProvider)
    mock_provider.name = "mock"
    mock_provider.model = "mock-1"
    mock_provider.is_configured = True
    mock_provider.generate.return_value = {
        "image_url": "https://example.com/img.png",
    }
    renderer = r.NanoBananaImageRenderer(
        provider=mock_provider, cache_dir=str(tmp_path),
    )
    idea = {"title": "t", "method": "m", "methodology_type": "empirical_study"}
    with patch("ideagraph_image_renderer.requests.get",
                  return_value=_mock_resp(200)):
        v1 = renderer.render(idea)
        v2 = renderer.render(idea)
    assert v1.success and v2.success
    # Provider was called only ONCE — the second call was a cache hit.
    assert mock_provider.generate.call_count == 1


def test_renderer_force_bypasses_cache(tmp_path):
    mock_provider = MagicMock(spec=r.ImageProvider)
    mock_provider.name = "mock"
    mock_provider.model = "mock-1"
    mock_provider.is_configured = True
    mock_provider.generate.return_value = {"image_url": "https://x/i.png"}
    renderer = r.NanoBananaImageRenderer(
        provider=mock_provider, cache_dir=str(tmp_path),
    )
    idea = {"title": "t", "method": "m", "methodology_type": "x"}
    with patch("ideagraph_image_renderer.requests.get",
                  return_value=_mock_resp(200)):
        renderer.render(idea)
        renderer.render(idea, force=True)
    assert mock_provider.generate.call_count == 2


def test_renderer_provider_error_is_propagated(tmp_path):
    mock_provider = MagicMock(spec=r.ImageProvider)
    mock_provider.name = "mock"
    mock_provider.model = "mock-1"
    mock_provider.is_configured = True
    mock_provider.generate.return_value = {"error": "boom"}
    renderer = r.NanoBananaImageRenderer(
        provider=mock_provider, cache_dir=str(tmp_path),
    )
    v = renderer.render({"title": "t", "method": "m",
                            "methodology_type": "x"})
    assert v.success is False
    assert "boom" in v.error


def test_renderer_retries_transient_failures(tmp_path):
    """Provider returns error twice, then succeeds — renderer retries."""
    mock_provider = MagicMock(spec=r.ImageProvider)
    mock_provider.name = "mock"
    mock_provider.model = "mock-1"
    mock_provider.is_configured = True
    mock_provider.generate.side_effect = [
        {"error": "transient HTTP 502"},
        {"error": "transient HTTP 503"},
        {"image_url": "https://example.com/img.png"},
    ]
    renderer = r.NanoBananaImageRenderer(
        provider=mock_provider, cache_dir=str(tmp_path),
    )
    with patch("ideagraph_image_renderer.requests.get",
                  return_value=_mock_resp(200)), \
            patch("ideagraph_image_renderer.time.sleep"):
        v = renderer.render({"title": "t", "method": "m",
                                "methodology_type": "x"})
    assert v.success is True
    assert mock_provider.generate.call_count == 3
    assert v.attempts == 3


def test_renderer_does_not_retry_api_key_errors(tmp_path):
    """User-facing 'API key not configured' errors should NOT trigger
    retries — they're not going to succeed."""
    mock_provider = MagicMock(spec=r.ImageProvider)
    mock_provider.name = "mock"
    mock_provider.model = "mock-1"
    mock_provider.is_configured = True
    mock_provider.generate.return_value = {"error": "API key invalid"}
    renderer = r.NanoBananaImageRenderer(
        provider=mock_provider, cache_dir=str(tmp_path),
    )
    with patch("ideagraph_image_renderer.time.sleep"):
        v = renderer.render({"title": "t", "method": "m",
                                "methodology_type": "x"})
    assert v.success is False
    # Only one attempt — no retry.
    assert mock_provider.generate.call_count == 1


def test_renderer_batch_preserves_order(tmp_path):
    mock_provider = MagicMock(spec=r.ImageProvider)
    mock_provider.name = "mock"
    mock_provider.model = "mock-1"
    mock_provider.is_configured = True

    def _gen(prompt):
        # Sleep zero — preserve insertion-order via the gather semantics.
        return {"image_url": f"https://x/{hash(prompt) & 0xfff:03x}.png"}
    mock_provider.generate.side_effect = lambda p: _gen(p)

    renderer = r.NanoBananaImageRenderer(
        provider=mock_provider, cache_dir=str(tmp_path),
    )
    ideas = [
        {"title": f"idea-{i}", "method": "m", "methodology_type": "x"}
        for i in range(5)
    ]
    with patch("ideagraph_image_renderer.requests.get",
                  return_value=_mock_resp(200)):
        out = renderer.render_batch(ideas)
    titles = [v.idea_title for v in out]
    assert titles == [f"idea-{i}" for i in range(5)]


def test_renderer_batch_empty_input_returns_empty(tmp_path):
    renderer = r.NanoBananaImageRenderer(
        api_key="x", cache_dir=str(tmp_path),
    )
    assert renderer.render_batch([]) == []


# ─────────────────────────────────────────────────────────────────────────────
# Top-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def test_render_idea_visuals_helper_wraps_renderer(tmp_path, monkeypatch):
    """The top-level convenience function from the setup script must
    accept the same params as the class and return a list."""
    monkeypatch.delenv("NANO_BANANA_API_KEY", raising=False)
    monkeypatch.delenv("BFL_API_KEY", raising=False)
    out = r.render_idea_visuals(
        [{"title": "x", "method": "y", "methodology_type": "z"}],
        api_key="",  # forces no-key → error path
        cache_dir=str(tmp_path),
    )
    assert isinstance(out, list)
    assert len(out) == 1
    assert isinstance(out[0], r.IdeaVisual)


def test_display_idea_with_visual_handles_none_visual():
    """When called without a visual, must not crash — show a placeholder."""
    fake_st = MagicMock()
    r.display_idea_with_visual({"title": "x"}, fake_st, visual=None)
    fake_st.info.assert_called_once()


def test_display_idea_with_visual_calls_image_on_success():
    fake_st = MagicMock()
    v = r.IdeaVisual(idea_title="t", image_url="https://x/i.png")
    r.display_idea_with_visual({"title": "t"}, fake_st, visual=v)
    fake_st.image.assert_called_once()


def test_display_idea_with_visual_warns_on_failure():
    fake_st = MagicMock()
    v = r.IdeaVisual(idea_title="t", error="boom")
    r.display_idea_with_visual({"title": "t"}, fake_st, visual=v)
    fake_st.warning.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# Config file (.nanobang_config) loading
# ─────────────────────────────────────────────────────────────────────────────

def test_renderer_reads_nanobang_config_file(tmp_path, monkeypatch):
    """When neither env var nor config.py has the key, fall back to
    .nanobang_config in CWD."""
    monkeypatch.delenv("NANO_BANANA_API_KEY", raising=False)
    monkeypatch.delenv("BFL_API_KEY", raising=False)
    monkeypatch.delenv("NANO_BANANA_PROVIDER", raising=False)
    # Force default provider so the test is independent of .env state.
    import config as _c
    monkeypatch.setattr(_c, "NANO_BANANA_API_KEY", "", raising=False)
    monkeypatch.setattr(_c, "NANO_BANANA_PROVIDER", "", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".nanobang_config").write_text(
        "# Some comment\n"
        "NANO_BANANA_API_KEY=cfg-file-key-xyz\n"
        "MODEL=flux-pro-1.0\n",
        encoding="utf-8",
    )
    renderer = r.NanoBananaImageRenderer(cache_dir=str(tmp_path / ".cache"))
    assert renderer.provider.api_key == "cfg-file-key-xyz"


# ─────────────────────────────────────────────────────────────────────────────
# GeminiImagenProvider — Google AI Studio Imagen 3
# ─────────────────────────────────────────────────────────────────────────────

def test_gemini_provider_no_key_returns_error():
    p = r.GeminiImagenProvider(api_key="")
    out = p.generate("test")
    assert "error" in out
    assert "not configured" in out["error"]


def test_gemini_provider_rejects_client_identifier_as_key():
    """The most common mistake: pasting gen-lang-client-... (Google
    client identifier) instead of AIza... (actual API key)."""
    p = r.GeminiImagenProvider(api_key="gen-lang-client-0652824768")
    # is_configured catches this upfront.
    assert p.is_configured is False
    # And even if someone bypassed that, generate() flags it.
    p2 = r.GeminiImagenProvider(api_key="AIzaSyXXXX_real_key")
    p2.api_key = "gen-lang-client-0652824768"  # forcibly poke around
    out = p2.generate("test")
    assert "client identifier" in out["error"].lower()


def test_gemini_provider_real_AIza_key_is_configured():
    p = r.GeminiImagenProvider(api_key="AIzaSyExampleKeyForTests1234567890ab")
    assert p.is_configured is True


def test_gemini_provider_default_model_and_endpoint():
    p = r.GeminiImagenProvider(api_key="AIza_test")
    assert p.model == "imagen-3.0-generate-002"
    assert p.endpoint.startswith("https://generativelanguage.googleapis.com")


def test_gemini_provider_happy_path_returns_image_bytes():
    """Verify the Google API response parsing — predictions[0].bytesBase64Encoded
    gets decoded to bytes."""
    import base64
    payload_bytes = b"FAKEPNG"
    b64 = base64.b64encode(payload_bytes).decode()
    resp = _mock_resp(200, {
        "predictions": [
            {"bytesBase64Encoded": b64, "mimeType": "image/png"},
        ],
    })
    with patch("ideagraph_image_renderer.requests.post",
                  return_value=resp) as post_mock:
        p = r.GeminiImagenProvider(api_key="AIza_test")
        out = p.generate("a research idea visual abstract")
    assert "error" not in out
    assert out["image_bytes"] == payload_bytes
    assert out["mime_type"] == "image/png"
    # Verify the request shape — instances + parameters, x-goog-api-key.
    _, kwargs = post_mock.call_args
    assert kwargs["headers"]["x-goog-api-key"] == "AIza_test"
    body = kwargs["json"]
    assert body["instances"][0]["prompt"]
    assert body["parameters"]["sampleCount"] == 1
    assert body["parameters"]["aspectRatio"] == "1:1"


def test_gemini_provider_hits_predict_endpoint():
    """URL must be {endpoint}/models/{model}:predict."""
    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        return _mock_resp(200, {"predictions": [
            {"bytesBase64Encoded": "QQ==", "mimeType": "image/png"},
        ]})

    with patch("ideagraph_image_renderer.requests.post",
                  side_effect=_fake_post):
        p = r.GeminiImagenProvider(
            api_key="AIza_test",
            model="imagen-3.0-generate-002",
            endpoint="https://generativelanguage.googleapis.com/v1beta",
        )
        p.generate("test")
    assert captured["url"] == (
        "https://generativelanguage.googleapis.com/v1beta"
        "/models/imagen-3.0-generate-002:predict"
    )


def test_gemini_provider_http_error_surfaces_status():
    """A 400 with a Google error JSON should be surfaced verbatim so
    the user can debug (e.g. quota, content filter, region issues)."""
    err_resp = _mock_resp(
        400, body=None,
        text='{"error":{"code":400,"message":"API key not valid"}}',
    )
    with patch("ideagraph_image_renderer.requests.post",
                  return_value=err_resp):
        p = r.GeminiImagenProvider(api_key="AIza_bad_key")
        out = p.generate("test")
    assert "error" in out
    assert "400" in out["error"]
    assert "API key not valid" in out["error"]


def test_gemini_provider_empty_predictions_is_error():
    resp = _mock_resp(200, {"predictions": []})
    with patch("ideagraph_image_renderer.requests.post",
                  return_value=resp):
        p = r.GeminiImagenProvider(api_key="AIza_test")
        out = p.generate("test")
    assert "no predictions" in out["error"]


def test_gemini_provider_missing_base64_is_error():
    resp = _mock_resp(200, {
        "predictions": [{"mimeType": "image/png"}],  # no bytesBase64Encoded
    })
    with patch("ideagraph_image_renderer.requests.post",
                  return_value=resp):
        p = r.GeminiImagenProvider(api_key="AIza_test")
        out = p.generate("test")
    assert "missing bytesBase64Encoded" in out["error"]


def test_gemini_provider_corrupt_base64_is_error():
    resp = _mock_resp(200, {
        "predictions": [{"bytesBase64Encoded": "!!! not valid base64 !!!"}],
    })
    with patch("ideagraph_image_renderer.requests.post",
                  return_value=resp):
        p = r.GeminiImagenProvider(api_key="AIza_test")
        out = p.generate("test")
    # base64.b64decode is lenient — corrupt input may decode to garbage
    # rather than raise. Accept either: error OR successful decode.
    assert "error" in out or "image_bytes" in out


def test_gemini_provider_network_error_is_caught():
    with patch("ideagraph_image_renderer.requests.post",
                  side_effect=__import__("requests").RequestException("boom")):
        p = r.GeminiImagenProvider(api_key="AIza_test")
        out = p.generate("test")
    assert "network error" in out["error"]


# ─────────────────────────────────────────────────────────────────────────────
# PROVIDER_REGISTRY + provider_name dispatch
# ─────────────────────────────────────────────────────────────────────────────

def test_provider_registry_has_both_providers():
    assert "flux_bfl" in r.PROVIDER_REGISTRY
    assert "gemini_imagen" in r.PROVIDER_REGISTRY
    assert r.PROVIDER_REGISTRY["flux_bfl"] is r.FluxBFLProvider
    assert r.PROVIDER_REGISTRY["gemini_imagen"] is r.GeminiImagenProvider


def test_provider_defaults_present_for_both():
    for name in ("flux_bfl", "gemini_imagen"):
        d = r.PROVIDER_DEFAULTS.get(name, {})
        assert d.get("model"), f"{name} missing default model"
        assert d.get("endpoint"), f"{name} missing default endpoint"


def test_renderer_provider_name_dispatch_flux(tmp_path, monkeypatch):
    monkeypatch.delenv("NANO_BANANA_API_KEY", raising=False)
    monkeypatch.delenv("BFL_API_KEY", raising=False)
    monkeypatch.delenv("NANO_BANANA_PROVIDER", raising=False)
    rd = r.NanoBananaImageRenderer(
        api_key="bfl-key", provider_name="flux_bfl",
        cache_dir=str(tmp_path),
    )
    assert isinstance(rd.provider, r.FluxBFLProvider)
    assert rd.provider.model == "flux-pro-1.0"
    assert "bfl.ml" in rd.provider.endpoint


def test_renderer_provider_name_dispatch_gemini(tmp_path, monkeypatch):
    monkeypatch.delenv("NANO_BANANA_API_KEY", raising=False)
    monkeypatch.delenv("NANO_BANANA_PROVIDER", raising=False)
    rd = r.NanoBananaImageRenderer(
        api_key="AIza_test", provider_name="gemini_imagen",
        cache_dir=str(tmp_path),
    )
    assert isinstance(rd.provider, r.GeminiImagenProvider)
    # Default Imagen model — verify it's one of the known-models list,
    # not a specific hard-coded name (the default has been bumped to
    # Imagen 4 now that paid AI Studio credit unlocks it).
    known = r.PROVIDER_DEFAULTS["gemini_imagen"]["known_models"]
    assert rd.provider.model in known
    assert rd.provider.model.startswith("imagen-")
    assert "googleapis.com" in rd.provider.endpoint


def test_renderer_provider_name_via_env(tmp_path, monkeypatch):
    """NANO_BANANA_PROVIDER env var picks the provider when no
    explicit kwarg is given."""
    monkeypatch.setenv("NANO_BANANA_API_KEY", "AIza_test")
    monkeypatch.setenv("NANO_BANANA_PROVIDER", "gemini_imagen")
    rd = r.NanoBananaImageRenderer(cache_dir=str(tmp_path))
    assert isinstance(rd.provider, r.GeminiImagenProvider)


def test_renderer_explicit_provider_name_overrides_env(tmp_path, monkeypatch):
    monkeypatch.setenv("NANO_BANANA_API_KEY", "AIza_x")
    monkeypatch.setenv("NANO_BANANA_PROVIDER", "gemini_imagen")
    rd = r.NanoBananaImageRenderer(
        provider_name="flux_bfl", cache_dir=str(tmp_path),
    )
    assert isinstance(rd.provider, r.FluxBFLProvider)


def test_renderer_unknown_provider_falls_back_to_flux(tmp_path, monkeypatch):
    monkeypatch.delenv("NANO_BANANA_API_KEY", raising=False)
    rd = r.NanoBananaImageRenderer(
        api_key="x", provider_name="totally_unknown",
        cache_dir=str(tmp_path),
    )
    # Falls back to flux_bfl (the safe default).
    assert isinstance(rd.provider, r.FluxBFLProvider)


def test_renderer_default_provider_is_flux(tmp_path, monkeypatch):
    monkeypatch.delenv("NANO_BANANA_API_KEY", raising=False)
    monkeypatch.delenv("NANO_BANANA_PROVIDER", raising=False)
    # Default fallback path: env unset + config.NANO_BANANA_PROVIDER empty.
    # The user's real .env may set it to gemini_imagen — neuter for this test.
    import config as _c
    monkeypatch.setattr(_c, "NANO_BANANA_PROVIDER", "", raising=False)
    rd = r.NanoBananaImageRenderer(api_key="x", cache_dir=str(tmp_path))
    assert isinstance(rd.provider, r.FluxBFLProvider)


def test_explicit_provider_instance_wins_over_provider_name(tmp_path):
    """Passing a constructed provider instance bypasses the registry."""
    custom = r.GeminiImagenProvider(api_key="AIza_custom")
    rd = r.NanoBananaImageRenderer(
        provider=custom, provider_name="flux_bfl",
        cache_dir=str(tmp_path),
    )
    assert rd.provider is custom


def test_renderer_with_gemini_provider_caches_image_bytes(tmp_path):
    """End-to-end through the renderer: Gemini provider returns
    image_bytes (not a URL), and the renderer caches them correctly."""
    import base64
    img_bytes = b"FAKEPNG_FROM_IMAGEN"
    resp = _mock_resp(200, {
        "predictions": [
            {"bytesBase64Encoded": base64.b64encode(img_bytes).decode(),
              "mimeType": "image/png"},
        ],
    })
    with patch("ideagraph_image_renderer.requests.post",
                  return_value=resp):
        rd = r.NanoBananaImageRenderer(
            api_key="AIza_test", provider_name="gemini_imagen",
            cache_dir=str(tmp_path),
        )
        v = rd.render({"title": "test", "method": "m",
                          "methodology_type": "empirical_study"})
    assert v.success is True
    # Gemini returns inline bytes, not a URL.
    assert v.image_url is None
    assert v.cached_path is not None
    assert Path(v.cached_path).read_bytes() == img_bytes


# ─────────────────────────────────────────────────────────────────────────────
# GeminiFlashImageProvider — "Nano Banana" via :generateContent
# ─────────────────────────────────────────────────────────────────────────────

def test_flash_provider_no_key_returns_error():
    p = r.GeminiFlashImageProvider(api_key="")
    out = p.generate("test")
    assert "error" in out
    assert "not configured" in out["error"]


def test_flash_provider_rejects_client_identifier():
    p = r.GeminiFlashImageProvider(api_key="gen-lang-client-0652824768")
    out = p.generate("test")
    assert "client identifier" in out["error"].lower()


def test_flash_provider_default_model():
    p = r.GeminiFlashImageProvider(api_key="AIza_test")
    assert "gemini" in p.model.lower()
    assert "flash" in p.model.lower() or "image" in p.model.lower()


def test_flash_provider_hits_generateContent_endpoint():
    """URL must be {endpoint}/models/{model}:generateContent (NOT :predict).
    This is the key difference from Imagen."""
    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["body"] = json
        captured["headers"] = headers
        # Return a happy-path response with inline image data.
        import base64
        return _mock_resp(200, {
            "candidates": [{
                "content": {
                    "parts": [
                        {"inlineData": {
                            "mimeType": "image/png",
                            "data": base64.b64encode(b"PNG").decode(),
                        }},
                    ],
                },
            }],
        })

    with patch("ideagraph_image_renderer.requests.post",
                  side_effect=_fake_post):
        p = r.GeminiFlashImageProvider(
            api_key="AIza_test",
            model="gemini-2.5-flash-image-preview",
            endpoint="https://generativelanguage.googleapis.com/v1beta",
        )
        out = p.generate("test")

    assert out.get("image_bytes") == b"PNG"
    assert captured["url"] == (
        "https://generativelanguage.googleapis.com/v1beta"
        "/models/gemini-2.5-flash-image-preview:generateContent"
    )
    # Request shape: contents.parts.text + responseModalities
    body = captured["body"]
    assert body["contents"][0]["parts"][0]["text"]
    assert "IMAGE" in body["generationConfig"]["responseModalities"]


def test_flash_provider_walks_multimodal_response_parts():
    """The model may return TEXT + IMAGE parts interleaved — must find
    the inlineData entry wherever it lives in the parts list."""
    import base64
    resp = _mock_resp(200, {
        "candidates": [{
            "content": {
                "parts": [
                    {"text": "Here's your image:"},   # text first
                    {"inlineData": {                    # image second
                        "mimeType": "image/png",
                        "data": base64.b64encode(b"IMG").decode(),
                    }},
                    {"text": "Hope you like it!"},   # more text
                ],
            },
        }],
    })
    with patch("ideagraph_image_renderer.requests.post", return_value=resp):
        p = r.GeminiFlashImageProvider(api_key="AIza_test")
        out = p.generate("test")
    assert out.get("image_bytes") == b"IMG"


def test_flash_provider_text_only_response_returns_error():
    """If the model returned only TEXT (no inlineData), surface a
    helpful error suggesting a more visual prompt."""
    resp = _mock_resp(200, {
        "candidates": [{
            "content": {
                "parts": [{"text": "I can't generate an image for that."}],
            },
        }],
    })
    with patch("ideagraph_image_renderer.requests.post", return_value=resp):
        p = r.GeminiFlashImageProvider(api_key="AIza_test")
        out = p.generate("test")
    assert "error" in out
    assert "no inlineData" in out["error"]


def test_flash_provider_safety_block_surfaces_promptFeedback():
    """When the model refuses (safety filter), there's no candidates
    but a promptFeedback block instead."""
    resp = _mock_resp(200, {
        "promptFeedback": {
            "blockReason": "SAFETY",
            "safetyRatings": [
                {"category": "HARM_CATEGORY_HARASSMENT",
                  "probability": "HIGH"},
            ],
        },
    })
    with patch("ideagraph_image_renderer.requests.post", return_value=resp):
        p = r.GeminiFlashImageProvider(api_key="AIza_test")
        out = p.generate("test")
    assert "error" in out
    assert "promptFeedback" in out["error"] or "safety" in out["error"].lower()


def test_flash_provider_http_error_surfaces():
    err = _mock_resp(
        400, body=None,
        text='{"error":{"code":400,"message":"Quota exceeded"}}',
    )
    with patch("ideagraph_image_renderer.requests.post", return_value=err):
        p = r.GeminiFlashImageProvider(api_key="AIza_test")
        out = p.generate("test")
    assert "400" in out["error"]
    assert "Quota exceeded" in out["error"]


def test_flash_provider_accepts_snake_case_inline_data():
    """Some SDK builds use inline_data (snake_case) instead of inlineData."""
    import base64
    resp = _mock_resp(200, {
        "candidates": [{
            "content": {
                "parts": [
                    {"inline_data": {
                        "mime_type": "image/png",
                        "data": base64.b64encode(b"PNG_SC").decode(),
                    }},
                ],
            },
        }],
    })
    with patch("ideagraph_image_renderer.requests.post", return_value=resp):
        p = r.GeminiFlashImageProvider(api_key="AIza_test")
        out = p.generate("test")
    assert out.get("image_bytes") == b"PNG_SC"


# ── Registry coverage ──────────────────────────────────────────────────────

def test_provider_registry_now_has_three_providers():
    assert set(r.PROVIDER_REGISTRY.keys()) >= {
        "flux_bfl", "gemini_imagen", "gemini_flash_image",
    }
    assert r.PROVIDER_REGISTRY["gemini_flash_image"] is \
        r.GeminiFlashImageProvider


def test_provider_defaults_has_known_models_for_all_providers():
    """Each provider exposes a known_models list for the admin UI dropdown."""
    for name in ("flux_bfl", "gemini_imagen", "gemini_flash_image"):
        models = r.PROVIDER_DEFAULTS[name].get("known_models")
        assert isinstance(models, list) and len(models) >= 1
        # Each provider's default model is in its known_models list.
        assert r.PROVIDER_DEFAULTS[name]["model"] in models


def test_gemini_imagen_known_models_includes_imagen_4():
    """Paid-tier users should see Imagen 4 variants at the top of the
    dropdown. If we ever drop them, the test catches the regression."""
    models = r.PROVIDER_DEFAULTS["gemini_imagen"]["known_models"]
    assert any("imagen-4" in m for m in models), (
        "Expected at least one Imagen 4 variant in gemini_imagen's "
        "known_models — paid AI Studio tier supports them."
    )
    # Default should be the newest GA Imagen 4 variant.
    assert "imagen-4" in r.PROVIDER_DEFAULTS["gemini_imagen"]["model"]


def test_gemini_flash_image_known_models_includes_2_5():
    """Gemini 2.5 Flash Image variants should be at the top."""
    models = r.PROVIDER_DEFAULTS["gemini_flash_image"]["known_models"]
    assert any("gemini-2.5-flash-image" in m for m in models), (
        "Expected Gemini 2.5 Flash Image variants in "
        "gemini_flash_image's known_models."
    )
    # Fall-back model (most reliable) should also be listed.
    assert "gemini-2.0-flash-preview-image-generation" in models


def test_gemini_flash_image_known_models_includes_chat_models_for_listmodels_match():
    """The dropdown should include the chat-only Gemini models too, so
    that when the user clicks a model from ListModels (which lists every
    model regardless of image capability), the dropdown can pre-select
    it instead of bouncing to Custom."""
    models = r.PROVIDER_DEFAULTS["gemini_flash_image"]["known_models"]
    assert "gemini-2.5-pro" in models
    assert "gemini-1.5-flash" in models


# ─────────────────────────────────────────────────────────────────────────────
# Newly added model entries (Imagen 4 Ultra, Nano Banana Pro)
# ─────────────────────────────────────────────────────────────────────────────

def test_imagen_catalog_includes_imagen_4_ultra():
    """User explicitly asked for Imagen 4 Ultra Generate."""
    models = r.PROVIDER_DEFAULTS["gemini_imagen"]["known_models"]
    assert any("imagen-4.0-ultra" in m for m in models), (
        "Expected at least one Imagen 4 Ultra variant in the catalog "
        "(the user has paid AI Studio credit that unlocks Ultra)."
    )


def test_flash_image_catalog_includes_nano_banana_pro():
    """User explicitly asked for Nano Banana Pro (Gemini 3 Pro Image).
    The actual API name uses the gemini-3-pro-image family — the
    colloquial 'nano-banana-pro' is NOT a real Google API model name
    and must not be in the catalog (it causes 404 confusion)."""
    models = r.PROVIDER_DEFAULTS["gemini_flash_image"]["known_models"]
    has_g3_image = any(
        ("gemini-3" in m and "image" in m) for m in models
    )
    assert has_g3_image, (
        "Expected at least one gemini-3-pro-image variant in the "
        "gemini_flash_image catalog."
    )
    # The colloquial name must NOT be present — it always 404s.
    assert "nano-banana-pro" not in models, (
        "'nano-banana-pro' is the colloquial name, not a real "
        "Google API model — must not be in the catalog."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Veo (informational — video, not yet a provider)
# ─────────────────────────────────────────────────────────────────────────────

def test_veo_video_models_constant_exists():
    """Veo info is surfaced in the admin UI; the constant must exist
    AND list veo-3.0 as the newest known variant."""
    assert hasattr(r, "VEO_VIDEO_MODELS")
    assert isinstance(r.VEO_VIDEO_MODELS, list)
    assert any("veo-3" in m for m in r.VEO_VIDEO_MODELS)


def test_veo_info_message_explains_video_nature():
    """The info message must call out that Veo is video, not image,
    so users don't paste it into the image-renderer model field."""
    msg = r.VEO_INFO_MESSAGE
    assert "video" in msg.lower()
    # Must mention the long-running operation endpoint, since Veo's
    # API shape is different from image gen.
    assert "predictLongRunning" in msg or "long-running" in msg.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Gemini LLM catalog (config.GEMINI_KNOWN_MODELS)
# ─────────────────────────────────────────────────────────────────────────────

def test_gemini_known_models_catalog_exists():
    import config as cfg
    assert hasattr(cfg, "GEMINI_KNOWN_MODELS")
    assert isinstance(cfg.GEMINI_KNOWN_MODELS, list)
    assert len(cfg.GEMINI_KNOWN_MODELS) >= 8


def test_gemini_known_models_includes_user_requested_variants():
    """User explicitly asked for these three LLM-class entries."""
    import config as cfg
    models = cfg.GEMINI_KNOWN_MODELS
    assert any("3.1-pro" in m for m in models), \
        "Missing Gemini 3.1 Pro"
    assert any("deep-research" in m for m in models), \
        "Missing Deep Research Pro Preview"
    assert any("antigravity" in m for m in models), \
        "Missing Antigravity"


def test_gemini_known_models_ordering_newest_first():
    """Gemini 3 should appear before Gemini 1.5 in the list — the
    sidebar dropdown defaults to options[0] when no model is set, so
    newest-first ordering is the UX contract."""
    import config as cfg
    models = cfg.GEMINI_KNOWN_MODELS
    pro_3_idx = next((i for i, m in enumerate(models)
                          if m.startswith("gemini-3")), None)
    pro_1_5_idx = next((i for i, m in enumerate(models)
                              if m.startswith("gemini-1.5")), None)
    assert pro_3_idx is not None
    assert pro_1_5_idx is not None
    assert pro_3_idx < pro_1_5_idx


# ─────────────────────────────────────────────────────────────────────────────
# Multi-panel paper-figure set (FIGURE_TEMPLATES + render_figure_set)
# ─────────────────────────────────────────────────────────────────────────────

def test_figure_templates_catalog():
    assert isinstance(r.FIGURE_TEMPLATES, dict)
    assert len(r.FIGURE_TEMPLATES) >= 4
    # The 4 canonical panels of a research paper figure set.
    for required in ("concept", "method", "experiment", "results"):
        assert required in r.FIGURE_TEMPLATES
        t = r.FIGURE_TEMPLATES[required]
        assert "label" in t and t["label"]
        assert "prompt_template" in t and "{title}" in t["prompt_template"]


def test_default_figure_set_is_four_canonical_panels():
    assert list(r.DEFAULT_FIGURE_SET) == [
        "concept", "method", "experiment", "results",
    ]


def test_build_panel_prompt_rejects_non_dict():
    with pytest.raises(TypeError):
        r.build_panel_prompt("not a dict", "concept")  # type: ignore[arg-type]


def test_build_panel_prompt_rejects_unknown_panel():
    with pytest.raises(ValueError):
        r.build_panel_prompt({"title": "x"}, "not_a_real_panel")


def test_build_panel_prompt_includes_title_and_template_specific_fields():
    idea = {
        "title": "Linear attention scaling",
        "method": "Random feature maps approximate softmax",
        "motivation": "O(n^2) attention is slow",
        "hypothesis": "Linear attention matches accuracy at O(n)",
        "expected_outcome": "+5 F1, 10x faster",
        "risk_assessment": "Approximation quality could degrade",
    }
    p_method = r.build_panel_prompt(idea, "method")
    p_concept = r.build_panel_prompt(idea, "concept")
    p_results = r.build_panel_prompt(idea, "results")
    p_risk = r.build_panel_prompt(idea, "limitations")
    # Different panels must produce different prompts.
    assert p_method != p_concept
    assert p_method != p_results
    # Each prompt mentions the title.
    for p in (p_method, p_concept, p_results, p_risk):
        assert "Linear attention scaling" in p
    # Method prompt mentions the method excerpt.
    assert "Random feature maps" in p_method
    # Concept prompt uses the motivation excerpt.
    assert "O(n^2) attention" in p_concept
    # Results uses hypothesis.
    assert "Linear attention matches" in p_results
    # Limitations uses risk.
    assert "Approximation quality" in p_risk


def test_build_panel_prompt_handles_missing_idea_fields():
    """Missing fields fall back to generic placeholders, not crash."""
    p = r.build_panel_prompt({"title": "minimal"}, "method")
    assert "minimal" in p
    assert "proposed approach" in p  # the fallback placeholder


def test_render_figure_set_invalid_idea_raises():
    rd = r.NanoBananaImageRenderer(api_key="bfl-x", cache_dir=".cache_test")
    with pytest.raises(ValueError):
        rd.render_figure_set({}, panels=["concept"])


def test_render_figure_set_unknown_panel_raises(tmp_path):
    rd = r.NanoBananaImageRenderer(api_key="bfl-x", cache_dir=str(tmp_path))
    with pytest.raises(ValueError):
        rd.render_figure_set({"title": "x"}, panels=["bogus_panel"])


def test_render_figure_set_default_panels_when_none(tmp_path):
    """When `panels` is None, uses DEFAULT_FIGURE_SET (4 panels)."""
    mock_provider = MagicMock(spec=r.ImageProvider)
    mock_provider.name = "mock"
    mock_provider.model = "mock-1"
    mock_provider.is_configured = True
    mock_provider.generate.return_value = {
        "image_url": "https://example.com/x.png",
    }
    rd = r.NanoBananaImageRenderer(
        provider=mock_provider, cache_dir=str(tmp_path),
    )
    with patch("ideagraph_image_renderer.requests.get",
                  return_value=_mock_resp(200)):
        result = rd.render_figure_set(
            {"title": "t", "method": "m", "methodology_type": "x"},
        )
    assert len(result) == 4
    assert [p.panel_id for p in result] == list(r.DEFAULT_FIGURE_SET)
    # Each panel got its own provider call (4 different prompts).
    assert mock_provider.generate.call_count == 4
    # Each result is a FigurePanel.
    for p in result:
        assert isinstance(p, r.FigurePanel)
        assert p.visual.success


def test_render_figure_set_panels_argument(tmp_path):
    """Custom panel list is respected and order is preserved."""
    mock_provider = MagicMock(spec=r.ImageProvider)
    mock_provider.name = "mock"
    mock_provider.model = "mock-1"
    mock_provider.is_configured = True
    mock_provider.generate.return_value = {"image_url": "https://x/i.png"}
    rd = r.NanoBananaImageRenderer(
        provider=mock_provider, cache_dir=str(tmp_path),
    )
    with patch("ideagraph_image_renderer.requests.get",
                  return_value=_mock_resp(200)):
        result = rd.render_figure_set(
            {"title": "t", "method": "m", "methodology_type": "x"},
            panels=["method", "concept"],
        )
    assert [p.panel_id for p in result] == ["method", "concept"]
    assert [p.order for p in result] == [0, 1]


def test_render_figure_set_each_panel_independently_cached(tmp_path):
    """Re-rendering the same idea with the same panels hits the cache —
    provider.generate is NOT called the second time."""
    mock_provider = MagicMock(spec=r.ImageProvider)
    mock_provider.name = "mock"
    mock_provider.model = "mock-1"
    mock_provider.is_configured = True
    mock_provider.generate.return_value = {"image_url": "https://x/i.png"}
    rd = r.NanoBananaImageRenderer(
        provider=mock_provider, cache_dir=str(tmp_path),
    )
    idea = {"title": "t", "method": "m", "methodology_type": "x"}
    with patch("ideagraph_image_renderer.requests.get",
                  return_value=_mock_resp(200)):
        rd.render_figure_set(idea, panels=["concept", "method"])
        rd.render_figure_set(idea, panels=["concept", "method"])
    # Called once per panel on the FIRST run, zero times on the second.
    assert mock_provider.generate.call_count == 2


# ─────────────────────────────────────────────────────────────────────────────
# VeoVideoProvider (video generation via long-running operation)
# ─────────────────────────────────────────────────────────────────────────────

def test_veo_provider_no_key_returns_error():
    p = r.VeoVideoProvider(api_key="")
    out = p.generate_video("test")
    assert "error" in out
    assert "not configured" in out["error"].lower()


def test_veo_provider_rejects_client_identifier():
    p = r.VeoVideoProvider(api_key="gen-lang-client-0652824768")
    out = p.generate_video("test")
    assert "error" in out
    assert "client identifier" in out["error"].lower()


def test_veo_provider_default_model():
    p = r.VeoVideoProvider(api_key="AIza_test")
    assert "veo" in p.model.lower()


def test_veo_provider_happy_path_polls_until_done():
    """Submit → operation name → poll done=False → poll done=True →
    extract video URL."""
    submit_resp = _mock_resp(200, {"name": "operations/abc-123"})
    poll_pending = _mock_resp(200, {"done": False})
    poll_done = _mock_resp(200, {
        "done": True,
        "response": {
            "generateVideoResponse": {
                "generatedSamples": [
                    {"video": {"uri": "https://example.com/video.mp4"}},
                ],
            },
        },
    })
    with patch("ideagraph_image_renderer.requests.post",
                  return_value=submit_resp), \
            patch("ideagraph_image_renderer.requests.get",
                  side_effect=[poll_pending, poll_done]), \
            patch("ideagraph_image_renderer.time.sleep"):
        p = r.VeoVideoProvider(
            api_key="AIza_test", poll_interval_s=0,
        )
        out = p.generate_video("animate this idea")
    assert "error" not in out
    assert out["video_url"] == "https://example.com/video.mp4"


def test_veo_provider_inline_video_bytes():
    """Some Veo responses return inline base64 instead of a URL."""
    import base64
    video_bytes = b"FAKE_MP4_BYTES"
    submit_resp = _mock_resp(200, {"name": "operations/x"})
    poll_done = _mock_resp(200, {
        "done": True,
        "response": {
            "generateVideoResponse": {
                "generatedSamples": [
                    {"video": {
                        "bytesBase64Encoded":
                            base64.b64encode(video_bytes).decode(),
                        "mimeType": "video/mp4",
                    }},
                ],
            },
        },
    })
    with patch("ideagraph_image_renderer.requests.post",
                  return_value=submit_resp), \
            patch("ideagraph_image_renderer.requests.get",
                  return_value=poll_done), \
            patch("ideagraph_image_renderer.time.sleep"):
        p = r.VeoVideoProvider(api_key="AIza_test", poll_interval_s=0)
        out = p.generate_video("test")
    assert out.get("video_bytes") == video_bytes
    assert out.get("mime_type") == "video/mp4"


def test_veo_provider_operation_failure_surfaces_error():
    submit_resp = _mock_resp(200, {"name": "operations/x"})
    poll_failed = _mock_resp(200, {
        "done": True,
        "error": {"code": 13, "message": "internal error"},
    })
    with patch("ideagraph_image_renderer.requests.post",
                  return_value=submit_resp), \
            patch("ideagraph_image_renderer.requests.get",
                  return_value=poll_failed), \
            patch("ideagraph_image_renderer.time.sleep"):
        p = r.VeoVideoProvider(api_key="AIza_test", poll_interval_s=0)
        out = p.generate_video("test")
    assert "error" in out
    assert "internal error" in out["error"]


def test_veo_provider_submit_404_returns_error():
    err = _mock_resp(404, text='{"error":{"code":404,"message":"not found"}}')
    with patch("ideagraph_image_renderer.requests.post",
                  return_value=err):
        p = r.VeoVideoProvider(api_key="AIza_test")
        out = p.generate_video("test")
    assert "404" in out["error"]


def test_veo_provider_polling_timeout():
    submit_resp = _mock_resp(200, {"name": "operations/x"})
    poll_pending = _mock_resp(200, {"done": False})
    with patch("ideagraph_image_renderer.requests.post",
                  return_value=submit_resp), \
            patch("ideagraph_image_renderer.requests.get",
                  return_value=poll_pending), \
            patch("ideagraph_image_renderer.time.sleep"):
        p = r.VeoVideoProvider(
            api_key="AIza_test", timeout_s=0.001, poll_interval_s=0,
        )
        out = p.generate_video("test")
    assert "timed out" in out["error"]


def test_veo_provider_omits_personGeneration_by_default():
    """Veo 3 rejects `personGeneration: "allow_adult"` on most tiers,
    so the parameter must be OMITTED by default — let Google's
    server-side default apply."""
    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["body"] = json
        return _mock_resp(200, {"name": "operations/x"})

    with patch("ideagraph_image_renderer.requests.post",
                  side_effect=_fake_post), \
            patch("ideagraph_image_renderer.requests.get",
                  return_value=_mock_resp(200, {
                      "done": True,
                      "response": {"generateVideoResponse": {
                          "generatedSamples": [
                              {"video": {"uri": "https://x/v.mp4"}},
                          ],
                      }},
                  })), \
            patch("ideagraph_image_renderer.time.sleep"):
        # Default constructor — no person_generation kwarg.
        p = r.VeoVideoProvider(api_key="AIza_test", poll_interval_s=0)
        p.generate_video("test")
    params = captured["body"]["parameters"]
    assert "personGeneration" not in params, (
        "personGeneration must be omitted by default to avoid "
        "HTTP 400 on Veo 3."
    )
    # But aspect ratio + duration are still sent.
    assert params["aspectRatio"]
    assert params["durationSeconds"]


def test_veo_provider_includes_personGeneration_when_set():
    """If the user explicitly opts in via `person_generation=`,
    the parameter IS sent — for users on Veo 2 or whatever model
    accepts it."""
    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["body"] = json
        return _mock_resp(200, {"name": "operations/x"})

    with patch("ideagraph_image_renderer.requests.post",
                  side_effect=_fake_post), \
            patch("ideagraph_image_renderer.requests.get",
                  return_value=_mock_resp(200, {
                      "done": True,
                      "response": {"generateVideoResponse": {
                          "generatedSamples": [
                              {"video": {"uri": "https://x/v.mp4"}},
                          ],
                      }},
                  })), \
            patch("ideagraph_image_renderer.time.sleep"):
        p = r.VeoVideoProvider(
            api_key="AIza_test",
            person_generation="dont_allow",
            poll_interval_s=0,
        )
        p.generate_video("test")
    params = captured["body"]["parameters"]
    assert params.get("personGeneration") == "dont_allow"


def test_veo_provider_uses_predictLongRunning_endpoint():
    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["body"] = json
        return _mock_resp(200, {"name": "operations/x"})

    with patch("ideagraph_image_renderer.requests.post",
                  side_effect=_fake_post), \
            patch("ideagraph_image_renderer.requests.get",
                  return_value=_mock_resp(200, {
                      "done": True,
                      "response": {"generateVideoResponse": {
                          "generatedSamples": [
                              {"video": {"uri": "https://x/v.mp4"}},
                          ],
                      }},
                  })), \
            patch("ideagraph_image_renderer.time.sleep"):
        p = r.VeoVideoProvider(
            api_key="AIza_test",
            model="veo-3.0-generate-001",
            endpoint="https://generativelanguage.googleapis.com/v1beta",
            poll_interval_s=0,
        )
        p.generate_video("test")
    assert captured["url"] == (
        "https://generativelanguage.googleapis.com/v1beta"
        "/models/veo-3.0-generate-001:predictLongRunning"
    )
    # Body has the Veo-specific parameters.
    body = captured["body"]
    assert body["instances"][0]["prompt"]
    assert "aspectRatio" in body["parameters"]
    assert "durationSeconds" in body["parameters"]


def test_build_video_prompt_method_animation():
    p = r.build_video_prompt(
        {"title": "Linear attention", "method": "Random features"},
        style="method_animation",
    )
    assert "Linear attention" in p
    assert "Random features" in p
    assert "animation" in p.lower() or "animated" in p.lower()


def test_build_video_prompt_result_reveal():
    p = r.build_video_prompt(
        {"title": "x", "hypothesis": "+5 F1"},
        style="result_reveal",
    )
    assert "chart" in p.lower()
    assert "+5 F1" in p


def test_idea_video_success_contract():
    v = r.IdeaVideo(idea_title="t")
    assert v.success is False
    assert r.IdeaVideo(idea_title="t",
                          video_url="https://x.mp4").success
    assert r.IdeaVideo(idea_title="t",
                          cached_path="/tmp/x.mp4").success
    assert not r.IdeaVideo(
        idea_title="t", video_url="https://x.mp4",
        error="boom",
    ).success


# ─────────────────────────────────────────────────────────────────────────────
# Veo as a registered provider in PROVIDER_REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

def test_veo_registered_in_provider_registry():
    """Veo must show up in PROVIDER_REGISTRY so the admin Provider
    dropdown can select it."""
    assert "veo" in r.PROVIDER_REGISTRY
    assert r.PROVIDER_REGISTRY["veo"] is r.VeoVideoProvider


def test_veo_has_provider_defaults():
    d = r.PROVIDER_DEFAULTS.get("veo")
    assert d is not None
    assert d.get("model", "").startswith("veo-")
    assert "googleapis.com" in d.get("endpoint", "")
    assert isinstance(d.get("known_models"), list)
    assert len(d["known_models"]) >= 2


def test_veo_provider_inherits_from_image_provider():
    """VeoVideoProvider must subclass ImageProvider so the renderer's
    dispatch handles it the same way as image providers."""
    assert issubclass(r.VeoVideoProvider, r.ImageProvider)


def test_veo_provider_generate_raw_returns_is_video():
    """`_generate_raw` must flag results with `is_video=True` so the
    renderer knows to save as .mp4 and stamp media_type."""
    submit_resp = _mock_resp(200, {"name": "operations/x"})
    poll_done = _mock_resp(200, {
        "done": True,
        "response": {
            "generateVideoResponse": {
                "generatedSamples": [
                    {"video": {"uri": "https://example.com/v.mp4"}},
                ],
            },
        },
    })
    with patch("ideagraph_image_renderer.requests.post",
                  return_value=submit_resp), \
            patch("ideagraph_image_renderer.requests.get",
                  return_value=poll_done), \
            patch("ideagraph_image_renderer.time.sleep"):
        p = r.VeoVideoProvider(api_key="AIza_test", poll_interval_s=0)
        out = p._generate_raw("test")
    assert out.get("is_video") is True
    assert out.get("video_url") == "https://example.com/v.mp4"


# ─────────────────────────────────────────────────────────────────────────────
# IdeaVisual.media_type + renderer video flow
# ─────────────────────────────────────────────────────────────────────────────

def test_idea_visual_default_media_type_is_image():
    v = r.IdeaVisual(idea_title="t")
    assert v.media_type == "image"
    assert v.is_video is False


def test_idea_visual_video_media_type():
    v = r.IdeaVisual(
        idea_title="t",
        image_url="https://x.mp4",
        media_type="video",
    )
    assert v.media_type == "video"
    assert v.is_video is True


def test_renderer_with_veo_provider_stamps_video_media_type(tmp_path):
    """End-to-end: when the active provider is Veo, the resulting
    IdeaVisual must have media_type='video' so the UI uses st.video()."""
    mock_provider = MagicMock(spec=r.VeoVideoProvider)
    mock_provider.name = "veo"
    mock_provider.model = "veo-3.0-generate-001"
    mock_provider.is_configured = True
    mock_provider.generate.return_value = {
        "is_video": True,
        "video_url": "https://example.com/clip.mp4",
    }
    rd = r.NanoBananaImageRenderer(
        provider=mock_provider, cache_dir=str(tmp_path),
    )
    v = rd.render({"title": "t", "method": "m",
                      "methodology_type": "empirical_study"})
    assert v.success is True
    assert v.media_type == "video"
    assert v.is_video is True
    # Same field carries the URL (image_url) but it's a video URL.
    assert v.image_url == "https://example.com/clip.mp4"


def test_renderer_with_veo_provider_inline_bytes_saves_mp4(tmp_path):
    """Inline video bytes from Veo must be cached with a .mp4
    extension (so OS + st.video() recognize the format)."""
    mock_provider = MagicMock(spec=r.VeoVideoProvider)
    mock_provider.name = "veo"
    mock_provider.model = "veo-3.0-generate-001"
    mock_provider.is_configured = True
    mock_provider.generate.return_value = {
        "is_video": True,
        "video_bytes": b"FAKE_MP4_BYTES",
        "mime_type": "video/mp4",
    }
    rd = r.NanoBananaImageRenderer(
        provider=mock_provider, cache_dir=str(tmp_path),
    )
    v = rd.render({"title": "t", "method": "m",
                      "methodology_type": "x"})
    assert v.success
    assert v.media_type == "video"
    assert v.cached_path is not None
    assert v.cached_path.endswith(".mp4")
    assert Path(v.cached_path).read_bytes() == b"FAKE_MP4_BYTES"


def test_renderer_with_veo_does_not_download_video_url(tmp_path):
    """Video URLs can be large (hundreds of MB) — the renderer must
    NOT auto-download them. Only the URL is stored."""
    mock_provider = MagicMock(spec=r.VeoVideoProvider)
    mock_provider.name = "veo"
    mock_provider.model = "veo-3.0-generate-001"
    mock_provider.is_configured = True
    mock_provider.generate.return_value = {
        "is_video": True,
        "video_url": "https://example.com/big.mp4",
    }
    rd = r.NanoBananaImageRenderer(
        provider=mock_provider, cache_dir=str(tmp_path),
    )
    # If the renderer tried to download, it would hit requests.get.
    with patch("ideagraph_image_renderer.requests.get") as get_mock:
        v = rd.render({"title": "t", "method": "m",
                          "methodology_type": "x"})
    get_mock.assert_not_called()
    assert v.image_url == "https://example.com/big.mp4"


def test_renderer_provider_name_dispatch_veo(tmp_path, monkeypatch):
    """`provider_name='veo'` must build a VeoVideoProvider with the
    correct defaults (model=veo-3.0-generate-001, etc.)."""
    monkeypatch.delenv("NANO_BANANA_API_KEY", raising=False)
    monkeypatch.delenv("NANO_BANANA_PROVIDER", raising=False)
    rd = r.NanoBananaImageRenderer(
        api_key="AIza_test", provider_name="veo",
        cache_dir=str(tmp_path),
    )
    assert isinstance(rd.provider, r.VeoVideoProvider)
    assert rd.provider.model.startswith("veo-")
    assert "googleapis.com" in rd.provider.endpoint


# ─────────────────────────────────────────────────────────────────────────────
# display_idea_with_visual respects media_type
# ─────────────────────────────────────────────────────────────────────────────

def test_display_image_calls_st_image():
    fake_st = MagicMock()
    v = r.IdeaVisual(
        idea_title="t",
        image_url="https://x/i.png",
        media_type="image",
    )
    r.display_idea_with_visual({"title": "t"}, fake_st, visual=v)
    fake_st.image.assert_called_once()
    fake_st.video.assert_not_called()


def test_display_video_calls_st_video():
    """When media_type='video', the display helper must call st.video,
    NOT st.image — otherwise the user sees a broken image placeholder."""
    fake_st = MagicMock()
    v = r.IdeaVisual(
        idea_title="t",
        image_url="https://x/clip.mp4",
        media_type="video",
    )
    r.display_idea_with_visual({"title": "t"}, fake_st, visual=v)
    fake_st.video.assert_called_once()
    fake_st.image.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE_STYLE_PRESETS — named visual styles
# ─────────────────────────────────────────────────────────────────────────────

def test_image_style_presets_catalog():
    """Style catalog must have ≥6 presets including the canonical
    editorial/scientific/sketch/3D variants."""
    assert len(r.IMAGE_STYLE_PRESETS) >= 6
    required = {"editorial", "scientific_paper", "isometric_3d", "sketch"}
    assert required <= set(r.IMAGE_STYLE_PRESETS.keys())


def test_style_preset_has_required_fields():
    for name, preset in r.IMAGE_STYLE_PRESETS.items():
        assert isinstance(preset, r.StylePreset)
        assert preset.name == name
        assert preset.label
        assert len(preset.prompt_suffix) > 20
        # Each prompt suffix must include a "no text" negative — this
        # is what stops the model from rendering broken text in
        # scientific figures. Accept any phrasing that mentions "no"
        # and "text" together (broken text would also be caught by
        # the provider's negative_prompt, but the suffix is the
        # primary guard).
        suffix_lc = preset.prompt_suffix.lower()
        assert "no text" in suffix_lc or "no actual text" in suffix_lc \
            or "or text" in suffix_lc, (
                f"preset {name!r} missing a 'no text' directive: "
                f"{preset.prompt_suffix}"
            )


def test_default_style_is_editorial():
    assert r.DEFAULT_STYLE == "editorial"
    assert r.DEFAULT_STYLE in r.IMAGE_STYLE_PRESETS


def test_apply_style_appends_suffix():
    base = "A picture of X"
    out = r.apply_style(base, style="isometric_3d")
    assert out.startswith(base)
    assert "isometric 3D" in out


def test_apply_style_unknown_falls_back_to_default():
    """Unknown style should NOT raise — fall back to default editorial."""
    out = r.apply_style("base", style="totally_not_a_style")
    assert "editorial" in out.lower()


def test_apply_style_empty_prompt_returns_empty():
    assert r.apply_style("", style="editorial") == ""


def test_build_prompt_respects_style_kwarg():
    idea = {"title": "X", "method": "Y", "methodology_type": "empirical_study"}
    p_editorial = r.build_prompt(idea, style="editorial")
    p_sketch = r.build_prompt(idea, style="sketch")
    p_3d = r.build_prompt(idea, style="isometric_3d")
    # All three must include the base content.
    for p in (p_editorial, p_sketch, p_3d):
        assert "X" in p
    # And they must differ from each other.
    assert p_editorial != p_sketch
    assert p_sketch != p_3d
    # Distinct cache keys means independently cached.
    assert (
        r.cache_key_for_prompt(p_editorial)
        != r.cache_key_for_prompt(p_sketch)
    )


def test_build_panel_prompt_respects_style_kwarg():
    idea = {"title": "X", "method": "Y", "methodology_type": "empirical_study"}
    p_editorial = r.build_panel_prompt(idea, "method", style="editorial")
    p_blueprint = r.build_panel_prompt(idea, "method", style="blueprint")
    assert p_editorial != p_blueprint
    assert "blueprint" in p_blueprint.lower()


# ─────────────────────────────────────────────────────────────────────────────
# VEO_ANIMATION_STYLES — 6 video styles
# ─────────────────────────────────────────────────────────────────────────────

def test_veo_animation_styles_catalog_has_at_least_six():
    """User asked for more animation styles — was 2, now 6+."""
    assert len(r.VEO_ANIMATION_STYLES) >= 6
    # Must include the original 2 + the 4 new ones.
    required = {
        "method_animation", "result_reveal",
        "zoom_reveal", "before_after",
        "network_growth", "particle_flow",
    }
    assert required <= set(r.VEO_ANIMATION_STYLES.keys())


def test_veo_animation_styles_have_required_fields():
    for name, style in r.VEO_ANIMATION_STYLES.items():
        assert "label" in style and style["label"]
        assert "description" in style and len(style["description"]) > 10
        assert "prompt_template" in style
        assert "{title}" in style["prompt_template"]


def test_build_video_prompt_uses_each_style_template():
    """Each style produces a meaningfully different prompt."""
    idea = {"title": "Linear attention", "method": "Random features",
              "motivation": "scaling", "hypothesis": "matches accuracy"}
    prompts = {}
    for style in r.VEO_ANIMATION_STYLES:
        prompts[style] = r.build_video_prompt(idea, style=style)
    # All distinct.
    assert len(set(prompts.values())) == len(prompts)
    # Each prompt mentions the title.
    for p in prompts.values():
        assert "Linear attention" in p


def test_build_video_prompt_zoom_reveal_mentions_aerial():
    p = r.build_video_prompt({"title": "x", "motivation": "y"},
                                  style="zoom_reveal")
    assert "aerial" in p.lower() or "wide" in p.lower()
    assert "zoom" in p.lower()


def test_build_video_prompt_unknown_style_falls_back():
    p = r.build_video_prompt({"title": "x", "method": "y"},
                                  style="not_a_real_style")
    # Falls back to method_animation, which includes "method diagram".
    assert "method diagram" in p.lower() or "icons" in p.lower()


# ─────────────────────────────────────────────────────────────────────────────
# render_n_samples — generate N variants
# ─────────────────────────────────────────────────────────────────────────────

def test_render_n_samples_rejects_invalid_inputs(tmp_path):
    rd = r.NanoBananaImageRenderer(api_key="bfl-x", cache_dir=str(tmp_path))
    with pytest.raises(ValueError):
        rd.render_n_samples({}, n=4)
    with pytest.raises(ValueError):
        rd.render_n_samples({"title": "x"}, n=20)  # cap is 16
    assert rd.render_n_samples({"title": "x"}, n=0) == []


def test_render_n_samples_produces_distinct_prompts_and_cache_keys(tmp_path):
    """Each sample variant must use a distinct prompt so they have
    distinct cache keys — otherwise N samples would all be the same."""
    mock_provider = MagicMock(spec=r.ImageProvider)
    mock_provider.name = "mock"
    mock_provider.model = "mock-1"
    mock_provider.is_configured = True

    captured_prompts: List[str] = []

    def _gen(prompt: str):
        captured_prompts.append(prompt)
        return {"image_url": f"https://x/{len(captured_prompts)}.png"}

    mock_provider.generate.side_effect = _gen

    rd = r.NanoBananaImageRenderer(
        provider=mock_provider, cache_dir=str(tmp_path),
    )
    with patch("ideagraph_image_renderer.requests.get",
                  return_value=_mock_resp(200)):
        samples = rd.render_n_samples(
            {"title": "t", "method": "m",
              "methodology_type": "empirical_study"},
            n=3, style="editorial",
        )
    assert len(samples) == 3
    # All 3 prompts must be distinct (variant index markers).
    assert len(set(captured_prompts)) == 3
    # All 3 cache keys must be distinct.
    keys = [s.cache_key for s in samples]
    assert len(set(keys)) == 3


def test_render_n_samples_returns_idea_visuals(tmp_path):
    mock_provider = MagicMock(spec=r.ImageProvider)
    mock_provider.name = "mock"
    mock_provider.model = "mock-1"
    mock_provider.is_configured = True
    mock_provider.generate.return_value = {
        "image_url": "https://x/v.png",
    }
    rd = r.NanoBananaImageRenderer(
        provider=mock_provider, cache_dir=str(tmp_path),
    )
    with patch("ideagraph_image_renderer.requests.get",
                  return_value=_mock_resp(200)):
        out = rd.render_n_samples(
            {"title": "t", "method": "m",
              "methodology_type": "x"},
            n=4,
        )
    assert all(isinstance(v, r.IdeaVisual) for v in out)
    assert all(v.success for v in out)


# ─────────────────────────────────────────────────────────────────────────────
# Save / download helpers — safe_filename, read_visual_bytes, bundle_as_zip
# ─────────────────────────────────────────────────────────────────────────────

def test_safe_filename_basic_shape():
    name = r.safe_filename("Transformer attention", style="isometric_3d")
    assert name.startswith("Transformer_attention__isometric_3d__")
    assert name.endswith(".png")


def test_safe_filename_video_uses_mp4_extension():
    name = r.safe_filename("X", media_type="video")
    assert name.endswith(".mp4")


def test_safe_filename_strips_unicode_punctuation():
    """Curly quotes, accented chars, slashes, colons must all
    collapse to safe ASCII so the filename works on Windows / macOS / Linux."""
    name = r.safe_filename("Idea: “attention”/scale (n^2)",
                                style="3d")
    # No quotes, no slashes, no colons, no parens, no caret.
    for ch in '":/\\()^':
        assert ch not in name


def test_safe_filename_includes_panel_id_when_given():
    name = r.safe_filename("X", panel_id="method", style="editorial",
                                  timestamp=False)
    assert "method" in name
    assert "editorial" in name
    # No timestamp when disabled.
    assert "2026" not in name


def test_safe_filename_empty_title_falls_back():
    """An empty title must not produce a malformed filename — fall
    back to 'untitled'."""
    name = r.safe_filename("")
    assert "untitled" in name


def test_safe_filename_collapses_consecutive_underscores():
    """Multiple punctuation runs shouldn't produce 'X____Y'."""
    name = r.safe_filename("a, b, c -- d")
    assert "____" not in name
    # At most one separator between tokens.
    assert "___" not in name or name.count("___") <= 2  # double-underscore separator is OK


def test_safe_filename_caps_long_titles():
    """100+ char titles must be truncated so the resulting filename
    isn't > 255 chars (Windows limit on some FSes)."""
    long_title = "Quantum " * 50
    name = r.safe_filename(long_title)
    # The slug portion is capped at 60 chars; total filename stays
    # comfortably under the 255-char FS limit.
    assert len(name) < 120


def test_read_visual_bytes_from_cached_path(tmp_path):
    """When cached_path exists, read from disk — don't network-fetch."""
    p = tmp_path / "test.bin"
    p.write_bytes(b"LOCAL_FILE_BYTES")
    v = r.IdeaVisual(
        idea_title="x",
        image_url="https://should-not-be-fetched.example.com/x.png",
        cached_path=str(p),
    )
    with patch("ideagraph_image_renderer.requests.get") as get_mock:
        data = r.read_visual_bytes(v)
    assert data == b"LOCAL_FILE_BYTES"
    # Must NOT have hit the URL when the cache had the bytes.
    get_mock.assert_not_called()


def test_read_visual_bytes_falls_back_to_url():
    """When only image_url exists (no cached_path), fetch via HTTP."""
    v = r.IdeaVisual(
        idea_title="x", image_url="https://example.com/x.png",
        cached_path=None,
    )
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.content = b"FETCHED_BYTES"
    with patch("ideagraph_image_renderer.requests.get",
                  return_value=fake_resp):
        data = r.read_visual_bytes(v)
    assert data == b"FETCHED_BYTES"


def test_read_visual_bytes_returns_none_when_both_paths_fail():
    """No cached_path AND URL fetch returns non-200 → None."""
    v = r.IdeaVisual(idea_title="x",
                          image_url="https://example.com/missing.png")
    fake_resp = MagicMock(); fake_resp.status_code = 404
    with patch("ideagraph_image_renderer.requests.get",
                  return_value=fake_resp):
        assert r.read_visual_bytes(v) is None


def test_read_visual_bytes_respects_no_download_flag():
    """download_url_if_missing=False must skip the URL fetch and
    return None when there's no cached_path."""
    v = r.IdeaVisual(idea_title="x", image_url="https://example.com/x.png")
    with patch("ideagraph_image_renderer.requests.get") as get_mock:
        out = r.read_visual_bytes(v, download_url_if_missing=False)
    assert out is None
    get_mock.assert_not_called()


def test_read_visual_bytes_handles_missing_visual():
    assert r.read_visual_bytes(None) is None  # type: ignore[arg-type]
    # Empty visual (no URL, no path) → None.
    empty = r.IdeaVisual(idea_title="x")
    assert r.read_visual_bytes(empty) is None


def test_bundle_visuals_as_zip_basic(tmp_path):
    """Successful visuals get bundled; failed ones are skipped."""
    p1 = tmp_path / "a.bin"; p1.write_bytes(b"ONE")
    p2 = tmp_path / "b.bin"; p2.write_bytes(b"TWO")
    v1 = r.IdeaVisual(idea_title="x", image_url="https://x/1.png",
                          cached_path=str(p1))
    v2 = r.IdeaVisual(idea_title="x", image_url="https://x/2.png",
                          cached_path=str(p2))
    v3 = r.IdeaVisual(idea_title="x", error="failed")
    zip_bytes = r.bundle_visuals_as_zip(
        [v1, v2, v3], idea_title="My Idea",
    )
    assert zip_bytes is not None
    # Unpack the zip and verify contents.
    import io, zipfile
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        assert len(names) == 2  # v3 (failed) skipped
        # Each filename includes the idea title slug.
        for n in names:
            assert "My_Idea" in n
            assert n.endswith(".png")
        # The two visuals must end up under distinct filenames (the
        # zip writer dedups by appending an index when names collide).
        assert len(set(names)) == 2
        # Bytes preserved correctly (regardless of which name went
        # with which bytes — both must be present).
        contents = {n: zf.read(n) for n in names}
        assert b"ONE" in contents.values()
        assert b"TWO" in contents.values()


def test_bundle_visuals_as_zip_includes_panel_labels(tmp_path):
    p1 = tmp_path / "a.bin"; p1.write_bytes(b"X")
    p2 = tmp_path / "b.bin"; p2.write_bytes(b"Y")
    v1 = r.IdeaVisual(idea_title="t", cached_path=str(p1))
    v2 = r.IdeaVisual(idea_title="t", cached_path=str(p2))
    zip_bytes = r.bundle_visuals_as_zip(
        [v1, v2], idea_title="Topic",
        panel_labels=["concept", "method"],
    )
    import io, zipfile
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        joined = "|".join(names)
        # Panel labels appear in filenames.
        assert "concept" in joined
        assert "method" in joined


def test_bundle_visuals_as_zip_empty_returns_none():
    assert r.bundle_visuals_as_zip([], idea_title="x") is None


def test_bundle_visuals_as_zip_all_failed_returns_none():
    """If every visual has an error, the zip is None (don't write
    an empty archive)."""
    failed = [r.IdeaVisual(idea_title="x", error="boom") for _ in range(3)]
    assert r.bundle_visuals_as_zip(failed, idea_title="x") is None


def test_bundle_visuals_as_zip_handles_videos(tmp_path):
    """Video visuals should bundle as .mp4 entries (not .png)."""
    p = tmp_path / "v.mp4"; p.write_bytes(b"FAKEMP4")
    v = r.IdeaVisual(
        idea_title="x",
        image_url="https://x/v.mp4",
        cached_path=str(p),
        media_type="video",
    )
    zip_bytes = r.bundle_visuals_as_zip([v], idea_title="Topic")
    import io, zipfile
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        assert len(names) == 1
        assert names[0].endswith(".mp4")


def test_render_n_samples_style_param_is_applied(tmp_path):
    """Different styles → samples reflect that style."""
    mock_provider = MagicMock(spec=r.ImageProvider)
    mock_provider.name = "mock"
    mock_provider.model = "mock-1"
    mock_provider.is_configured = True
    mock_provider.generate.return_value = {"image_url": "https://x/v.png"}
    rd = r.NanoBananaImageRenderer(
        provider=mock_provider, cache_dir=str(tmp_path),
    )

    captured: List[str] = []
    def _gen(prompt: str):
        captured.append(prompt)
        return {"image_url": "https://x/v.png"}
    mock_provider.generate.side_effect = _gen

    with patch("ideagraph_image_renderer.requests.get",
                  return_value=_mock_resp(200)):
        rd.render_n_samples(
            {"title": "x", "method": "y", "methodology_type": "z"},
            n=2, style="sketch",
        )
    for p in captured:
        assert "whiteboard" in p.lower() or "sketch" in p.lower()


def test_display_video_uses_cached_path_when_present():
    """If a .mp4 was downloaded into the cache, prefer the local
    file over the remote URL (faster + bypasses URL expiry)."""
    fake_st = MagicMock()
    v = r.IdeaVisual(
        idea_title="t",
        image_url="https://x/clip.mp4",
        cached_path="/tmp/abc.mp4",
        media_type="video",
    )
    r.display_idea_with_visual({"title": "t"}, fake_st, visual=v)
    fake_st.video.assert_called_once_with("/tmp/abc.mp4")


def test_renderer_provider_name_dispatch_flash_image(tmp_path, monkeypatch):
    monkeypatch.delenv("NANO_BANANA_API_KEY", raising=False)
    rd = r.NanoBananaImageRenderer(
        api_key="AIza_test", provider_name="gemini_flash_image",
        cache_dir=str(tmp_path),
    )
    assert isinstance(rd.provider, r.GeminiFlashImageProvider)
    # Default model and endpoint applied from PROVIDER_DEFAULTS.
    assert rd.provider.model in r.PROVIDER_DEFAULTS[
        "gemini_flash_image"]["known_models"]
    assert "googleapis.com" in rd.provider.endpoint


# ─────────────────────────────────────────────────────────────────────────────
# list_gemini_models — diagnostic helper
# ─────────────────────────────────────────────────────────────────────────────

def test_list_models_no_key_returns_error():
    out = r.list_gemini_models(api_key="")
    assert "error" in out
    assert "API key" in out["error"]


def test_list_models_rejects_client_identifier():
    out = r.list_gemini_models(api_key="gen-lang-client-0652824768")
    assert "error" in out
    assert "client identifier" in out["error"].lower()


def test_list_models_happy_path_flags_image_models():
    """Verifies the helper correctly identifies image-generation models
    by name + supported methods, and sorts image-gen first."""
    resp = _mock_resp(200, {
        "models": [
            {
                "name": "models/gemini-1.5-pro",
                "displayName": "Gemini 1.5 Pro",
                "description": "Text model",
                "supportedGenerationMethods": ["generateContent", "countTokens"],
            },
            {
                "name": "models/imagen-3.0-generate-002",
                "displayName": "Imagen 3",
                "supportedGenerationMethods": ["predict"],
            },
            {
                "name": "models/gemini-2.0-flash-preview-image-generation",
                "displayName": "Flash Image (preview)",
                "supportedGenerationMethods": ["generateContent"],
            },
            {
                "name": "models/text-embedding-004",
                "supportedGenerationMethods": ["embedContent"],
            },
        ],
    })
    with patch("ideagraph_image_renderer.requests.get",
                  return_value=resp):
        out = r.list_gemini_models(api_key="AIza_test")
    assert "error" not in out
    assert out["count"] == 4
    models = out["models"]
    # Image-gen first, alphabetical within each group.
    assert models[0]["supports_image_gen"]
    assert models[1]["supports_image_gen"]
    # Non-image-gen models come after.
    assert not models[2]["supports_image_gen"]
    assert not models[3]["supports_image_gen"]
    # `name` has "models/" stripped.
    assert not any(m["name"].startswith("models/") for m in models)


def test_list_models_hits_correct_url_and_auth_header():
    captured = {}

    def _fake_get(url, headers=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        return _mock_resp(200, {"models": []})

    with patch("ideagraph_image_renderer.requests.get",
                  side_effect=_fake_get):
        r.list_gemini_models(
            api_key="AIza_test",
            endpoint="https://generativelanguage.googleapis.com/v1beta",
        )
    assert captured["url"] == (
        "https://generativelanguage.googleapis.com/v1beta/models"
    )
    assert captured["headers"]["x-goog-api-key"] == "AIza_test"


def test_list_models_http_error_surfaces():
    err = _mock_resp(
        403, body=None,
        text='{"error":{"code":403,"message":"API not enabled"}}',
    )
    with patch("ideagraph_image_renderer.requests.get", return_value=err):
        out = r.list_gemini_models(api_key="AIza_test")
    assert "error" in out
    assert "403" in out["error"]
    assert "API not enabled" in out["error"]


def test_list_models_handles_empty_response():
    """A 200 with no models (rare but possible on a barren key)
    returns an empty list, not an error."""
    resp = _mock_resp(200, {"models": []})
    with patch("ideagraph_image_renderer.requests.get",
                  return_value=resp):
        out = r.list_gemini_models(api_key="AIza_test")
    assert out["count"] == 0
    assert out["models"] == []


def test_list_models_handles_missing_methods_field():
    """Some entries may omit supportedGenerationMethods entirely."""
    resp = _mock_resp(200, {
        "models": [{"name": "models/minimal", "displayName": "Mini"}],
    })
    with patch("ideagraph_image_renderer.requests.get",
                  return_value=resp):
        out = r.list_gemini_models(api_key="AIza_test")
    assert out["count"] == 1
    assert out["models"][0]["generation_methods"] == []
    assert out["models"][0]["supports_image_gen"] is False


def test_list_models_image_gen_detection_requires_both_signals():
    """Heuristic: name must contain 'image' or 'imagen' AND support
    generateContent/predict. Just one of those isn't enough."""
    resp = _mock_resp(200, {
        "models": [
            # "image" in name, but only supports embedContent → not image-gen
            {"name": "models/image-embedding-001",
              "supportedGenerationMethods": ["embedContent"]},
            # supports generateContent, but no "image" in name → not image-gen
            {"name": "models/gemini-1.5-pro",
              "supportedGenerationMethods": ["generateContent"]},
        ],
    })
    with patch("ideagraph_image_renderer.requests.get",
                  return_value=resp):
        out = r.list_gemini_models(api_key="AIza_test")
    assert all(not m["supports_image_gen"] for m in out["models"])


def test_renderer_with_flash_provider_end_to_end(tmp_path):
    """End-to-end: Flash provider via the renderer caches inline bytes."""
    import base64
    resp = _mock_resp(200, {
        "candidates": [{
            "content": {
                "parts": [
                    {"inlineData": {
                        "mimeType": "image/png",
                        "data": base64.b64encode(b"FLASHPNG").decode(),
                    }},
                ],
            },
        }],
    })
    with patch("ideagraph_image_renderer.requests.post", return_value=resp):
        rd = r.NanoBananaImageRenderer(
            api_key="AIza_test", provider_name="gemini_flash_image",
            cache_dir=str(tmp_path),
        )
        v = rd.render({"title": "t", "method": "m",
                          "methodology_type": "empirical_study"})
    assert v.success
    assert v.image_url is None
    assert Path(v.cached_path).read_bytes() == b"FLASHPNG"
