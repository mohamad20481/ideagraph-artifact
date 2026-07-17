"""Idea-to-Video — NotebookLM-style narrated slideshow generator.

Auto-generates a multi-slide research-pitch video from an idea dict, packaged
as a self-contained HTML player that narrates with the browser's Web Speech
API (no server-side TTS dependency, no API cost).

Supports 5 distinct styles (documentary / trailer / TED talk / news / pitch deck)
with per-style scripts, voice tone, gradients, and visual flourishes. Live
word-by-word captions, voice + rate controls, and a confetti finale.

Public API:
    VIDEO_STYLES                                          -> Dict[str, dict]
    generate_video_script(idea, style="documentary")      -> List[Slide]
    build_video_embed(slides, idea, style=..., ...)       -> str  (fragment)
    build_video_html(slides, idea, style=..., ...)        -> str  (full doc)
    estimate_duration_s(slides)                            -> int
"""
from __future__ import annotations
from typing import Any, Dict, List
import html as _html
import json


# ─────────────────────────────────────────────────────────────────────────────
# Style configuration
# ─────────────────────────────────────────────────────────────────────────────

VIDEO_STYLES: Dict[str, Dict[str, Any]] = {
    "documentary": {
        "label": "📹 Documentary",
        "description": "Neutral, informative — museum-narrator tone",
        "default_rate": 1.0,
        "default_pitch": 1.0,
        "show_lower_third": False,
        "title_size": "38px",
        "captions": True,
    },
    "trailer": {
        "label": "🎞️ Movie Trailer",
        "description": "Dramatic hooks, bold visuals, gravelly voice",
        "default_rate": 0.88,
        "default_pitch": 0.65,
        "show_lower_third": False,
        "title_size": "56px",
        "captions": True,
    },
    "ted_talk": {
        "label": "🎤 TED Talk",
        "description": "Inspirational, slow, single-speaker storytelling",
        "default_rate": 0.95,
        "default_pitch": 1.05,
        "show_lower_third": False,
        "title_size": "44px",
        "captions": True,
    },
    "news": {
        "label": "📰 News Report",
        "description": "Anchor-style with breaking-news lower-third",
        "default_rate": 1.15,
        "default_pitch": 1.0,
        "show_lower_third": True,
        "title_size": "36px",
        "captions": True,
    },
    "pitch_deck": {
        "label": "💼 Pitch Deck",
        "description": "Fast YC-style — bullets, numbers, no fluff",
        "default_rate": 1.1,
        "default_pitch": 1.0,
        "show_lower_third": False,
        "title_size": "34px",
        "captions": True,
    },
}


_GRADIENTS = {
    "documentary": [
        "linear-gradient(135deg,#0c4a6e 0%,#0ea5e9 100%)",
        "linear-gradient(135deg,#581c87 0%,#a855f7 100%)",
        "linear-gradient(135deg,#7c2d12 0%,#f59e0b 100%)",
        "linear-gradient(135deg,#064e3b 0%,#10b981 100%)",
        "linear-gradient(135deg,#1e3a8a 0%,#3b82f6 100%)",
        "linear-gradient(135deg,#831843 0%,#ec4899 100%)",
        "linear-gradient(135deg,#0c4a6e 0%,#0369a1 100%)",
        "linear-gradient(135deg,#1e293b 0%,#475569 100%)",
    ],
    "trailer": [
        "radial-gradient(circle at top right,#7f1d1d 0%,#000 80%)",
        "radial-gradient(circle at bottom left,#1e1b4b 0%,#000 80%)",
        "radial-gradient(circle at center,#7c2d12 0%,#000 80%)",
        "linear-gradient(135deg,#000 0%,#3b0764 100%)",
        "radial-gradient(circle at top,#dc2626 0%,#000 75%)",
        "linear-gradient(135deg,#000 0%,#831843 100%)",
    ],
    "ted_talk": [
        "linear-gradient(135deg,#7c0a02 0%,#dc2626 100%)",
        "linear-gradient(135deg,#7c0a02 0%,#dc2626 100%)",
        "linear-gradient(135deg,#991b1b 0%,#ef4444 100%)",
        "linear-gradient(135deg,#7c0a02 0%,#dc2626 100%)",
        "linear-gradient(135deg,#7c0a02 0%,#dc2626 100%)",
    ],
    "news": [
        "linear-gradient(135deg,#1e3a8a 0%,#1e40af 100%)",
        "linear-gradient(135deg,#1e3a8a 0%,#1e40af 100%)",
        "linear-gradient(135deg,#0f172a 0%,#1e40af 100%)",
        "linear-gradient(135deg,#1e3a8a 0%,#1e40af 100%)",
        "linear-gradient(135deg,#1e3a8a 0%,#1e40af 100%)",
        "linear-gradient(135deg,#7f1d1d 0%,#991b1b 100%)",
        "linear-gradient(135deg,#1e3a8a 0%,#1e40af 100%)",
        "linear-gradient(135deg,#0f172a 0%,#1e40af 100%)",
    ],
    "pitch_deck": [
        "linear-gradient(135deg,#0f172a 0%,#1e293b 100%)",
        "linear-gradient(135deg,#0f172a 0%,#0c4a6e 100%)",
        "linear-gradient(135deg,#0f172a 0%,#064e3b 100%)",
        "linear-gradient(135deg,#0f172a 0%,#581c87 100%)",
        "linear-gradient(135deg,#0f172a 0%,#7c2d12 100%)",
        "linear-gradient(135deg,#0f172a 0%,#831843 100%)",
        "linear-gradient(135deg,#0f172a 0%,#1e293b 100%)",
        "linear-gradient(135deg,#0f172a 0%,#0c4a6e 100%)",
        "linear-gradient(135deg,#0f172a 0%,#064e3b 100%)",
        "linear-gradient(135deg,#10b981 0%,#064e3b 100%)",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _trim(text: str, max_chars: int) -> str:
    if not text:
        return ""
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return cut + "…"


def _novelty_label(score: float) -> str:
    if score >= 0.8: return "revolutionary"
    if score >= 0.6: return "fresh"
    if score >= 0.4: return "thoughtful"
    return "incremental"


def _quality_label(score: float) -> str:
    if score >= 0.7: return "ready to execute"
    if score >= 0.4: return "promising but needs refinement"
    return "early-stage — major work ahead"


def _significance_label(score: float) -> str:
    if score >= 0.7: return "highly significant"
    if score >= 0.4: return "moderately significant"
    return "narrow but useful"


def _split_method(method: str, max_bullets: int = 3) -> List[str]:
    if not method:
        return []
    sentences = [s.strip() for s in method.replace(";", ".").split(".") if s.strip()]
    return [_trim(s, 90) for s in sentences[:max_bullets]]


# ─────────────────────────────────────────────────────────────────────────────
# Visual primitives — animated SVG/HTML embedded in slides
# ─────────────────────────────────────────────────────────────────────────────

def _viz_bars(items: List[Dict[str, Any]], label: str = "") -> Dict[str, Any]:
    """Animated horizontal bars. items=[{label, value(0-100), color?}]"""
    return {"type": "bars", "items": items, "label": label}


def _viz_gauge(value: float, label: str = "", suffix: str = "%") -> Dict[str, Any]:
    """Radial arc gauge that sweeps from 0 to value (0-100)."""
    return {"type": "gauge", "value": float(value), "label": label, "suffix": suffix}


def _viz_counter(value: float, label: str = "",
                  suffix: str = "", decimals: int = 0) -> Dict[str, Any]:
    """Animated number that ticks up from 0 to value."""
    return {"type": "counter", "value": float(value), "label": label,
            "suffix": suffix, "decimals": int(decimals)}


def _viz_histogram(values: List[float], label: str = "") -> Dict[str, Any]:
    """Vertical bars (for distributions). Each value 0-100."""
    return {"type": "histogram", "values": [float(v) for v in values], "label": label}


def _viz_timeline(milestones: List[Dict[str, Any]], label: str = "") -> Dict[str, Any]:
    """Horizontal timeline. milestones=[{week, label}]"""
    return {"type": "timeline", "milestones": milestones, "label": label}


def _outcome_histogram_from_quality(quality: float) -> List[float]:
    """Generate a deterministic outcome curve shifted by quality."""
    # 8 buckets; mean shifts from ~3 (low q) to ~5.5 (high q)
    mean = 2.5 + quality * 3.5
    return [
        max(5, 100 * (1.0 / (1 + abs(i - mean)))) for i in range(8)
    ]


def _idea_fields(idea: Dict[str, Any]):
    return {
        "title": idea.get("title", "Untitled"),
        "method": idea.get("method", "") or "",
        "hypothesis": idea.get("hypothesis", "") or "",
        "expected": idea.get("expected_outcome", "") or "",
        "method_type": (idea.get("methodology_type") or "research").replace("_", " "),
        "quality": idea.get("quality_score", 0.5),
        "probe": idea.get("probe_scores") or {},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-style script generators
# ─────────────────────────────────────────────────────────────────────────────

def _gen_documentary(idea: Dict[str, Any]) -> List[Dict[str, Any]]:
    f = _idea_fields(idea)
    g = _GRADIENTS["documentary"]
    novelty = f["probe"].get("novelty", 0.5)
    significance = f["probe"].get("significance", 0.5)
    slides = []
    slides.append({
        "title": _trim(f["title"], 80),
        "subtitle": f"A {_novelty_label(novelty)} {f['method_type']}",
        "body": "", "bullets": [],
        "narration": (f"Today, we present a research idea: {_trim(f['title'], 100)}. "
                      f"A {_novelty_label(novelty)} approach to {f['method_type']}."),
        "duration_s": 6, "gradient": g[0], "icon": "🎬",
    })
    problem = _trim(f["hypothesis"] or f["method"] or f["expected"]
                     or "An open question that has resisted easy answers.", 240)
    slides.append({
        "title": "The Problem", "subtitle": "What gap are we tackling",
        "body": problem, "bullets": [],
        "narration": f"Here's the question we're investigating: {_trim(problem, 200)}",
        "duration_s": 9, "gradient": g[1], "icon": "❓",
    })
    if f["hypothesis"]:
        slides.append({
            "title": "Our Hypothesis", "subtitle": "What we believe to be true",
            "body": _trim(f["hypothesis"], 220), "bullets": [],
            "narration": f"Our hypothesis: {_trim(f['hypothesis'], 200)}",
            "duration_s": 8, "gradient": g[2], "icon": "💡",
        })
    method_bullets = _split_method(f["method"])
    slides.append({
        "title": "How We'll Do It", "subtitle": f["method_type"].title(),
        "body": "" if method_bullets else _trim(f["method"] or "Methodology to be detailed.", 220),
        "bullets": method_bullets,
        "narration": f"Our method, in short: {_trim(f['method'] or 'a structured investigation', 220)}",
        "duration_s": 10, "gradient": g[3], "icon": "⚙️",
    })
    if f["expected"]:
        slides.append({
            "title": "What We Expect", "subtitle": "If we succeed",
            "body": _trim(f["expected"], 220), "bullets": [],
            "narration": f"If this works, here's what we expect: {_trim(f['expected'], 200)}",
            "duration_s": 9, "gradient": g[4], "icon": "📈",
            "data_visual": _viz_histogram(
                _outcome_histogram_from_quality(f["quality"]),
                label="Projected outcome distribution",
            ),
        })
    weak_all = sorted(
        [(k, v) for k, v in f["probe"].items() if isinstance(v, (int, float))],
        key=lambda kv: kv[1])
    weak = weak_all[:3]
    risks = [f"{k.replace('_', ' ').title()} — {int(v * 100)}% confidence"
             for k, v in weak if v < 0.7]
    if not risks:
        risks = ["No major risks identified — strong probe scores across dimensions."]
    risk_bars = [
        {"label": k.replace("_", " ").title(), "value": int(v * 100),
         "color": "#fbbf24" if v >= 0.5 else "#f87171"}
        for k, v in weak_all[:5]
    ]
    slides.append({
        "title": "Risks & Mitigations", "subtitle": "Where the project could break",
        "body": "", "bullets": [],
        "narration": "The biggest risks we've identified: " + "; ".join(risks[:3]) + ".",
        "duration_s": 10, "gradient": g[5], "icon": "⚠️",
        "data_visual": _viz_bars(risk_bars, label="Probe confidence"),
    })
    sig = _significance_label(significance)
    slides.append({
        "title": "Why It Matters", "subtitle": sig.title(),
        "body": (f"This work is {sig}. Success would advance {f['method_type']} "
                 f"and could enable follow-on research."),
        "bullets": [],
        "narration": (f"Why does this matter? This work is {sig}. "
                      f"If it succeeds, it advances {f['method_type']}."),
        "duration_s": 9, "gradient": g[6], "icon": "🌍",
        "data_visual": _viz_gauge(int(significance * 100),
                                    label="Significance score"),
    })
    qlabel = _quality_label(f["quality"])
    slides.append({
        "title": "Next Steps", "subtitle": qlabel.title(), "body": "",
        "bullets": [
            "Validate the core hypothesis with a minimal experiment",
            "Strengthen the weakest probe dimensions",
            "Draft a submission for the best-fit venue",
        ],
        "narration": (f"This idea is {qlabel}. Validate the hypothesis, "
                      "strengthen the weak dimensions, and target the best-fit venue. "
                      "Thanks for watching."),
        "duration_s": 9, "gradient": g[7], "icon": "🚀",
        "data_visual": _viz_counter(int(f["quality"] * 100),
                                      label="Overall quality", suffix="%"),
    })
    return slides


def _gen_trailer(idea: Dict[str, Any]) -> List[Dict[str, Any]]:
    f = _idea_fields(idea)
    g = _GRADIENTS["trailer"]
    novelty = f["probe"].get("novelty", 0.5)
    nov_label = _novelty_label(novelty)
    return [
        {"title": "In a world…", "subtitle": "where progress has stalled",
         "body": "", "bullets": [],
         "narration": "In a world where progress has stalled, where the obvious paths have all been tried…",
         "duration_s": 7, "gradient": g[0], "icon": "🌒"},
        {"title": "One idea dares to ask:", "subtitle": "what if?",
         "body": _trim(f["hypothesis"] or f["method"] or "What if we approached it differently?", 180),
         "bullets": [],
         "narration": f"One idea dares to ask the question. {_trim(f['hypothesis'] or 'What if we did this differently', 180)}",
         "duration_s": 9, "gradient": g[1], "icon": "❓"},
        {"title": _trim(f["title"], 60), "subtitle": f"a {nov_label} {f['method_type']}",
         "body": "", "bullets": [],
         "narration": f"This summer… {_trim(f['title'], 80)}. A {nov_label} approach.",
         "duration_s": 6, "gradient": g[2], "icon": "🎬"},
        {"title": "The Stakes", "subtitle": "What could go wrong",
         "body": "", "bullets": [],
         "narration": "But the stakes are high. Every dimension is tested. Every assumption challenged.",
         "duration_s": 9, "gradient": g[3], "icon": "⚠️",
         "data_visual": _viz_bars(
             [{"label": k.replace("_"," ").title(), "value": int(v*100),
               "color": "#dc2626"}
              for k, v in sorted(
                  [(k,v) for k,v in f["probe"].items() if isinstance(v,(int,float))],
                  key=lambda kv: kv[1])[:4]] or
             [{"label": "Risk", "value": 50, "color": "#dc2626"}],
             label="Risk profile"),
         },
        {"title": "And if it works…", "subtitle": "Everything changes",
         "body": _trim(f["expected"] or "Everything changes.", 180), "bullets": [],
         "narration": f"And if it works… {_trim(f['expected'] or 'everything changes', 160)}.",
         "duration_s": 9, "gradient": g[4], "icon": "🚀",
         "data_visual": _viz_counter(int(max(f["quality"], 0.5) * 100),
                                       label="Potential impact", suffix="%")},
        {"title": "Coming Soon", "subtitle": "To a research lab near you",
         "body": "", "bullets": [],
         "narration": "Coming soon. To a research lab near you. Submission window 2026.",
         "duration_s": 6, "gradient": g[5], "icon": "🎟️"},
    ]


def _gen_ted_talk(idea: Dict[str, Any]) -> List[Dict[str, Any]]:
    f = _idea_fields(idea)
    g = _GRADIENTS["ted_talk"]
    sig = _significance_label(f["probe"].get("significance", 0.5))
    return [
        {"title": "When I first encountered this problem…", "subtitle": "Let me tell you a story",
         "body": _trim(f["hypothesis"] or f["method"] or "an open question I couldn't shake", 200),
         "bullets": [],
         "narration": ("When I first encountered this problem, I'll be honest — I didn't know what to do. "
                       f"It came down to one question: {_trim(f['hypothesis'] or f['method'] or 'why does this matter', 180)}"),
         "duration_s": 14, "gradient": g[0], "icon": "🎤"},
        {"title": "The Idea", "subtitle": _trim(f["title"], 60),
         "body": _trim(f["hypothesis"] or f["title"], 180), "bullets": [],
         "narration": (f"The idea is simple. {_trim(f['hypothesis'] or f['title'], 180)}. "
                       "Simple to state. Hard to prove."),
         "duration_s": 12, "gradient": g[1], "icon": "💡"},
        {"title": "How We Got Here", "subtitle": "The path",
         "body": "", "bullets": _split_method(f["method"]) or ["Build it.", "Test it.", "Measure it."],
         "narration": (f"Here's how we plan to find out. {_trim(f['method'] or 'a structured investigation', 200)}. "
                       "Three steps. No shortcuts."),
         "duration_s": 13, "gradient": g[2], "icon": "🛤️"},
        {"title": "Why You Should Care", "subtitle": sig.title(),
         "body": (f"This work is {sig}. And here's the thing — if we get this right, "
                  "it changes the questions everyone else gets to ask."),
         "bullets": [],
         "narration": (f"Why should you care? Because this work is {sig}. "
                       "If we get this right, it changes what questions become possible. "
                       "That's the whole game."),
         "duration_s": 13, "gradient": g[3], "icon": "❤️",
         "data_visual": _viz_gauge(
             int(f["probe"].get("significance", 0.5) * 100),
             label="Significance")},
        {"title": "Thank You.", "subtitle": "What I want you to take away",
         "body": "", "bullets": [
             "Find the question you can't stop thinking about",
             "Build the simplest experiment that could prove it wrong",
             "Then tell someone",
         ],
         "narration": ("Find the question you can't stop thinking about. "
                       "Build the simplest experiment that could prove it wrong. "
                       "Then tell someone. Thank you."),
         "duration_s": 11, "gradient": g[4], "icon": "🙏"},
    ]


def _gen_news(idea: Dict[str, Any]) -> List[Dict[str, Any]]:
    f = _idea_fields(idea)
    g = _GRADIENTS["news"]
    novelty = f["probe"].get("novelty", 0.5)
    nov_label = _novelty_label(novelty)
    weak = sorted(
        [(k, v) for k, v in f["probe"].items() if isinstance(v, (int, float))],
        key=lambda kv: kv[1])[:2]
    risks = [f"{k.replace('_', ' ').title()}" for k, v in weak if v < 0.7]
    if not risks:
        risks = ["execution"]
    sig = _significance_label(f["probe"].get("significance", 0.5))
    qlabel = _quality_label(f["quality"])
    slides = [
        {"title": "BREAKING", "subtitle": "Live from the IdeaGraph newsdesk",
         "body": _trim(f["title"], 100), "bullets": [],
         "narration": (f"Good evening. We have breaking news from the research desk. "
                       f"A new idea has just been filed: {_trim(f['title'], 100)}. "
                       f"Our team is live with the details."),
         "duration_s": 10, "gradient": g[0], "icon": "📰",
         "lower_third": "BREAKING NEWS"},
        {"title": "The Story", "subtitle": "What we know so far",
         "body": _trim(f["hypothesis"] or f["method"] or "Details still emerging.", 220),
         "bullets": [],
         "narration": (f"Here's what we know. The team behind this idea claims: "
                       f"{_trim(f['hypothesis'] or f['method'] or 'a fresh angle on a long-standing question', 200)}."),
         "duration_s": 10, "gradient": g[1], "icon": "📡",
         "lower_third": "DEVELOPING STORY"},
        {"title": "The Approach", "subtitle": f["method_type"].title(),
         "body": "", "bullets": _split_method(f["method"]) or ["Methodology under review."],
         "narration": (f"On the methodology — a {nov_label} {f['method_type']} approach. "
                       f"{_trim(f['method'] or 'Specifics will follow', 180)}."),
         "duration_s": 10, "gradient": g[2], "icon": "🔬",
         "lower_third": "EXCLUSIVE"},
        {"title": "Expected Impact", "subtitle": sig.title(),
         "body": _trim(f["expected"] or f"Analysts call the work {sig}.", 220),
         "bullets": [],
         "narration": (f"Analysts are calling this work {sig}. "
                       f"If verified: {_trim(f['expected'] or 'meaningful advances are likely', 180)}."),
         "duration_s": 10, "gradient": g[3], "icon": "📊",
         "lower_third": "MARKET REACTION",
         "data_visual": _viz_gauge(
             int(f["probe"].get("significance", 0.5) * 100),
             label="Impact rating")},
        {"title": "Concerns", "subtitle": "Where reviewers might push back",
         "body": "", "bullets": [],
         "narration": ("Not everyone is convinced. Critics point to questions around "
                       + ", ".join(risks[:3]) + ". The story is still developing."),
         "duration_s": 10, "gradient": g[5], "icon": "⚖️",
         "lower_third": "ANALYSIS",
         "data_visual": _viz_bars(
             [{"label": k.replace("_"," ").title(), "value": int(v*100),
               "color": "#fbbf24"}
              for k, v in weak[:4]] or
             [{"label": "Execution", "value": 50, "color": "#fbbf24"}],
             label="Reviewer concern level (lower = more concern)")},
        {"title": "What's Next", "subtitle": qlabel.title(), "body": "",
         "bullets": [
             "Validation experiment imminent",
             "Conference submission planned",
             "Follow-up coverage to come",
         ],
         "narration": (f"What's next? The idea is {qlabel}. We'll bring you updates as they unfold. "
                       "Back to you in the studio."),
         "duration_s": 9, "gradient": g[7], "icon": "📺",
         "lower_third": "WHAT'S NEXT"},
    ]
    # Pad gradient assignment indices
    return slides


def _gen_pitch_deck(idea: Dict[str, Any]) -> List[Dict[str, Any]]:
    f = _idea_fields(idea)
    g = _GRADIENTS["pitch_deck"]
    novelty = f["probe"].get("novelty", 0.5)
    weak = sorted(
        [(k, v) for k, v in f["probe"].items() if isinstance(v, (int, float))],
        key=lambda kv: kv[1])[:3]
    risks = [k.replace("_", " ").title() for k, v in weak if v < 0.7]
    return [
        {"title": "PROBLEM", "subtitle": "What's broken",
         "body": _trim(f["hypothesis"] or f["method"] or "A gap in the literature.", 180),
         "bullets": [],
         "narration": f"Slide 1. The problem. {_trim(f['hypothesis'] or 'a gap that current methods cannot close', 160)}.",
         "duration_s": 6, "gradient": g[0], "icon": "🎯"},
        {"title": "WHY NOW", "subtitle": "Timing",
         "body": "", "bullets": [
             "Recent advances unlock new approaches",
             "Compute and data both accessible",
             f"The field is ready for {_novelty_label(novelty)} ideas",
         ],
         "narration": "Slide 2. Why now. Three things: recent advances open new approaches; compute and data are both accessible; the field is ready.",
         "duration_s": 7, "gradient": g[1], "icon": "⏰"},
        {"title": "INSIGHT", "subtitle": "What others miss",
         "body": _trim(f["hypothesis"] or f["title"], 180), "bullets": [],
         "narration": f"Slide 3. The insight. {_trim(f['hypothesis'] or f['title'], 180)}.",
         "duration_s": 7, "gradient": g[2], "icon": "💡"},
        {"title": "SOLUTION", "subtitle": _trim(f["title"], 60),
         "body": "", "bullets": _split_method(f["method"], 4) or ["Approach to be detailed"],
         "narration": f"Slide 4. The solution. {_trim(f['method'] or 'a structured investigation', 200)}.",
         "duration_s": 8, "gradient": g[3], "icon": "🛠️"},
        {"title": "MARKET", "subtitle": "Who needs this",
         "body": (f"Researchers in {f['method_type']}. "
                  "Practitioners who currently work around this gap. Reviewers and funders watching the space."),
         "bullets": [],
         "narration": "Slide 5. The market. Researchers in this area, practitioners working around the gap, and reviewers tracking the space.",
         "duration_s": 7, "gradient": g[4], "icon": "🌍"},
        {"title": "RISKS", "subtitle": "What could kill us",
         "body": "", "bullets": [],
         "narration": "Slide 6. Risks. Top three: " + ", ".join((risks or ["execution"])[:3]) + ".",
         "duration_s": 7, "gradient": g[5], "icon": "⚠️",
         "data_visual": _viz_bars(
             [{"label": k.replace("_", " ").title(), "value": int(v*100),
               "color": "#f87171"}
              for k, v in weak[:3]] or
             [{"label": "Execution", "value": 50, "color": "#f87171"}],
             label="Risk severity")},
        {"title": "MITIGATION", "subtitle": "How we de-risk",
         "body": "", "bullets": [
             "Minimal experiment first to test the core claim",
             "Pre-registered analysis plan",
             "Public code + data from day one",
         ],
         "narration": "Slide 7. Mitigation. Minimal experiment first. Pre-registered plan. Open code from day one.",
         "duration_s": 7, "gradient": g[6], "icon": "🛡️"},
        {"title": "TIMELINE", "subtitle": "12 weeks to first result",
         "body": "", "bullets": [],
         "narration": "Slide 8. Timeline. Twelve weeks. Data and baselines first. Then the pipeline. Then ablations and writeup.",
         "duration_s": 8, "gradient": g[7], "icon": "📅",
         "data_visual": _viz_timeline(
             [{"week": "W1-4", "label": "Data + baselines"},
              {"week": "W5-8", "label": "Full pipeline"},
              {"week": "W9-12", "label": "Ablations + writeup"}],
             label="Twelve-week plan")},
        {"title": "OUTCOME", "subtitle": "What success looks like",
         "body": _trim(f["expected"] or "Quantitative gains over current SOTA.", 180),
         "bullets": [],
         "narration": f"Slide 9. The outcome. {_trim(f['expected'] or 'measurable improvement over the strongest available baseline', 180)}.",
         "duration_s": 8, "gradient": g[8], "icon": "📈",
         "data_visual": _viz_counter(int(f["quality"] * 100),
                                       label="Projected quality", suffix="%")},
        {"title": "THE ASK", "subtitle": _quality_label(f["quality"]).title(),
         "body": "", "bullets": [
             "GPU-time + dataset access",
             "12 weeks of focused work",
             "One reviewer who'll read drafts",
         ],
         "narration": "Slide 10. The ask. Compute. Twelve weeks of focused work. One critical reviewer. That's it. Thank you.",
         "duration_s": 8, "gradient": g[9], "icon": "🤝"},
    ]


_GENERATORS = {
    "documentary": _gen_documentary,
    "trailer": _gen_trailer,
    "ted_talk": _gen_ted_talk,
    "news": _gen_news,
    "pitch_deck": _gen_pitch_deck,
}


def generate_video_script(idea: Dict[str, Any],
                            style: str = "documentary") -> List[Dict[str, Any]]:
    """Generate the slide list for a given style.

    Each slide dict contains: title, subtitle, body, bullets, narration,
    duration_s, gradient, icon. News-style slides also include 'lower_third'.
    """
    gen = _GENERATORS.get(style, _gen_documentary)
    return gen(idea)


def estimate_duration_s(slides: List[Dict[str, Any]]) -> int:
    """Total video duration in seconds."""
    return sum(int(s.get("duration_s", 8)) for s in slides)


# ─────────────────────────────────────────────────────────────────────────────
# Slide rendering
# ─────────────────────────────────────────────────────────────────────────────

def _visual_to_html(viz: Dict[str, Any]) -> str:
    """Render a data_visual dict as animated HTML/SVG."""
    if not viz:
        return ""
    vtype = viz.get("type")
    label = _html.escape(viz.get("label", "") or "")
    label_html = (f'<div class="ig-viz-label">{label}</div>' if label else "")

    if vtype == "bars":
        rows = []
        for it in viz.get("items", [])[:6]:
            lbl = _html.escape(str(it.get("label", "")))
            val = max(0, min(100, float(it.get("value", 0))))
            color = _html.escape(str(it.get("color", "#0ea5e9")))
            rows.append(
                f'<div class="ig-bar-row">'
                f'<span class="ig-bar-label">{lbl}</span>'
                f'<div class="ig-bar-track">'
                f'<div class="ig-bar-fill" data-target="{val}" '
                f'style="background:{color}"></div></div>'
                f'<span class="ig-bar-pct">{int(val)}%</span>'
                f'</div>'
            )
        return (
            f'<div class="ig-visual ig-bars-viz" data-type="bars">'
            f'{label_html}{"".join(rows)}</div>'
        )

    if vtype == "gauge":
        val = max(0, min(100, float(viz.get("value", 0))))
        suffix = _html.escape(str(viz.get("suffix", "%")))
        # Half-circle arc: radius 80, circumference of half = pi*80 ≈ 251.3
        arc_len = 251.3
        return (
            f'<div class="ig-visual ig-gauge-viz" data-type="gauge">'
            f'{label_html}'
            f'<svg viewBox="0 0 200 120" class="ig-gauge-svg">'
            f'<path d="M 20 100 A 80 80 0 0 1 180 100" '
            f'stroke="rgba(255,255,255,0.15)" stroke-width="14" fill="none" '
            f'stroke-linecap="round"/>'
            f'<path class="ig-gauge-arc" d="M 20 100 A 80 80 0 0 1 180 100" '
            f'stroke="url(#ig-grad)" stroke-width="14" fill="none" '
            f'stroke-linecap="round" '
            f'data-target="{val}" data-arclen="{arc_len}" '
            f'style="stroke-dasharray:{arc_len};stroke-dashoffset:{arc_len}"/>'
            f'<defs><linearGradient id="ig-grad" x1="0" x2="1">'
            f'<stop offset="0%" stop-color="#0ea5e9"/>'
            f'<stop offset="100%" stop-color="#10b981"/>'
            f'</linearGradient></defs>'
            f'</svg>'
            f'<div class="ig-gauge-num">'
            f'<span class="ig-gauge-val" data-target="{val}">0</span>'
            f'<span class="ig-gauge-suffix">{suffix}</span></div>'
            f'</div>'
        )

    if vtype == "counter":
        val = float(viz.get("value", 0))
        suffix = _html.escape(str(viz.get("suffix", "")))
        decimals = int(viz.get("decimals", 0))
        return (
            f'<div class="ig-visual ig-counter-viz" data-type="counter">'
            f'{label_html}'
            f'<div class="ig-counter-num">'
            f'<span class="ig-counter-val" data-target="{val}" '
            f'data-decimals="{decimals}">0</span>'
            f'<span class="ig-counter-suffix">{suffix}</span>'
            f'</div></div>'
        )

    if vtype == "histogram":
        bars = []
        vals = viz.get("values", [])
        for i, v in enumerate(vals[:12]):
            h = max(2, min(100, float(v)))
            bars.append(
                f'<div class="ig-histo-bar" data-target="{h}" '
                f'style="--i:{i}"></div>'
            )
        return (
            f'<div class="ig-visual ig-histogram-viz" data-type="histogram">'
            f'{label_html}'
            f'<div class="ig-histo-grid">{"".join(bars)}</div></div>'
        )

    if vtype == "timeline":
        items = []
        ms = viz.get("milestones", [])
        for i, m in enumerate(ms[:5]):
            week = _html.escape(str(m.get("week", "")))
            lbl = _html.escape(str(m.get("label", "")))
            items.append(
                f'<div class="ig-tl-item" style="--d:{i*200}ms">'
                f'<div class="ig-tl-dot"></div>'
                f'<div class="ig-tl-week">{week}</div>'
                f'<div class="ig-tl-label">{lbl}</div></div>'
            )
        return (
            f'<div class="ig-visual ig-timeline-viz" data-type="timeline">'
            f'{label_html}'
            f'<div class="ig-tl-line"></div>'
            f'<div class="ig-tl-items">{"".join(items)}</div></div>'
        )

    return ""


def _slide_to_html(slide: Dict[str, Any], idx: int, total: int, style: str) -> str:
    body = _html.escape(slide.get("body", "") or "")
    bullets_html = ""
    if slide.get("bullets"):
        items = "".join(
            f'<li>{_html.escape(b)}</li>' for b in slide["bullets"]
        )
        bullets_html = f'<ul class="ig-bullets">{items}</ul>'
    body_html = f'<p class="ig-body">{body}</p>' if body else ""

    icon = slide.get("icon", "🎬")
    title = _html.escape(slide.get("title", ""))
    subtitle = _html.escape(slide.get("subtitle", ""))
    gradient = slide.get("gradient", _GRADIENTS["documentary"][0])
    visual_html = _visual_to_html(slide.get("data_visual"))

    lower_third = ""
    if VIDEO_STYLES.get(style, {}).get("show_lower_third"):
        lt_text = _html.escape(slide.get("lower_third", "BREAKING NEWS"))
        lower_third = (
            f'<div class="ig-lower-third">'
            f'<span class="ig-lt-tag">LIVE</span>'
            f'<span class="ig-lt-text">{lt_text}</span>'
            f'</div>'
        )

    return (
        f'<div class="ig-slide" data-slide-index="{idx}" '
        f'style="background:{gradient}">'
        f'<div class="ig-slide-counter">{idx + 1} / {total}</div>'
        f'<div class="ig-slide-icon">{icon}</div>'
        f'<div class="ig-slide-subtitle">{subtitle}</div>'
        f'<h1 class="ig-slide-title">{title}</h1>'
        f'{body_html}'
        f'{bullets_html}'
        f'{visual_html}'
        f'{lower_third}'
        f'</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Player CSS
# ─────────────────────────────────────────────────────────────────────────────

_PLAYER_CSS = """
.ig-video-wrap{max-width:960px;margin:0 auto;border-radius:16px;
  overflow:hidden;box-shadow:0 12px 32px rgba(15,23,42,0.18);
  background:#0f172a;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
  position:relative}
.ig-stage{position:relative;aspect-ratio:16/9;width:100%;background:#000;overflow:hidden}
.ig-slide{position:absolute;inset:0;color:white;padding:48px 56px;
  display:flex;flex-direction:column;justify-content:center;
  opacity:0;transform:scale(1.04);
  transition:opacity 0.6s ease,transform 0.8s ease;pointer-events:none}
.ig-slide.active{opacity:1;transform:scale(1);pointer-events:auto;
  animation:ig-kenburns 14s ease-out forwards}
@keyframes ig-kenburns{
  0%{transform:scale(1)}
  100%{transform:scale(1.06)}
}
.ig-slide-icon{font-size:42px;margin-bottom:8px}
.ig-slide-counter{position:absolute;top:18px;right:24px;
  font-size:11px;color:rgba(255,255,255,0.7);letter-spacing:0.08em;
  background:rgba(0,0,0,0.25);padding:4px 10px;border-radius:10px;
  font-weight:600;z-index:3}
.ig-slide-subtitle{font-size:13px;text-transform:uppercase;
  letter-spacing:0.12em;color:rgba(255,255,255,0.85);font-weight:600;
  margin-bottom:6px}
.ig-slide-title{font-size:38px;font-weight:800;line-height:1.18;
  margin:0 0 6px 0;color:white;letter-spacing:-0.01em}
.ig-body{font-size:21px;line-height:1.55;color:rgba(255,255,255,0.92);margin:18px 0}
.ig-bullets{font-size:20px;color:rgba(255,255,255,0.95);
  list-style:none;padding:0;margin:18px 0}
.ig-bullets li{margin:8px 0;line-height:1.5;
  padding-left:24px;position:relative}
.ig-bullets li:before{content:"▸";position:absolute;left:0;
  color:rgba(255,255,255,0.6);font-weight:700}

/* Style: trailer — bigger titles, dark dramatic look */
.ig-style-trailer .ig-slide-title{font-size:56px;letter-spacing:-0.02em;
  text-shadow:0 4px 24px rgba(0,0,0,0.5)}
.ig-style-trailer .ig-slide-subtitle{color:#fca5a5}
.ig-style-trailer .ig-slide-icon{font-size:54px}

/* Style: TED — single accent color, large typography */
.ig-style-ted_talk .ig-slide-title{font-size:44px;letter-spacing:-0.01em}
.ig-style-ted_talk .ig-slide-subtitle{color:#fecaca}
.ig-style-ted_talk .ig-slide{padding:60px 80px}

/* Style: news — anchor look + lower third */
.ig-style-news .ig-slide-title{font-size:36px;text-transform:uppercase;
  letter-spacing:0.02em}
.ig-style-news .ig-slide{padding:36px 48px 96px 48px}
.ig-lower-third{position:absolute;bottom:0;left:0;right:0;
  background:linear-gradient(180deg,rgba(0,0,0,0) 0%,#7f1d1d 50%,#991b1b 100%);
  padding:24px 32px 16px 32px;display:flex;align-items:center;gap:12px;
  border-top:3px solid #fbbf24}
.ig-lt-tag{background:#fbbf24;color:#7f1d1d;padding:4px 10px;
  font-weight:800;font-size:11px;letter-spacing:0.08em;border-radius:4px}
.ig-lt-text{color:white;font-weight:700;font-size:14px;
  text-transform:uppercase;letter-spacing:0.04em}

/* Style: pitch deck — minimal, all caps, monospaced flavor */
.ig-style-pitch_deck .ig-slide-title{font-size:34px;letter-spacing:0.02em;
  font-weight:900}
.ig-style-pitch_deck .ig-slide-subtitle{color:#7dd3fc;
  font-family:ui-monospace,Menlo,monospace}

/* Captions */
.ig-captions{position:absolute;left:24px;right:24px;bottom:24px;
  background:rgba(0,0,0,0.65);color:white;padding:12px 18px;
  border-radius:10px;font-size:16px;line-height:1.5;
  box-shadow:0 4px 12px rgba(0,0,0,0.3);z-index:4;
  backdrop-filter:blur(6px);max-height:30%;overflow:hidden;
  display:none}
.ig-captions.visible{display:block}
.ig-captions .word{transition:color 0.15s,background 0.15s;padding:0 1px;
  border-radius:3px}
.ig-captions .word.active{background:rgba(14,165,233,0.45);color:#fff;
  box-shadow:0 0 0 2px rgba(14,165,233,0.25)}
.ig-captions .word.spoken{color:rgba(255,255,255,0.55)}

/* Lower-third doesn't conflict with captions: shift them up */
.ig-style-news .ig-captions{bottom:90px}

.ig-controls{display:flex;align-items:center;gap:12px;flex-wrap:wrap;
  padding:14px 20px;background:#0f172a;color:white}
.ig-btn{background:#0ea5e9;border:none;color:white;width:44px;height:44px;
  border-radius:50%;font-size:18px;cursor:pointer;display:flex;
  align-items:center;justify-content:center;transition:transform 0.1s}
.ig-btn:hover{transform:scale(1.06);background:#0284c7}
.ig-btn-secondary{background:rgba(255,255,255,0.08);width:36px;height:36px;
  font-size:14px}
.ig-btn-secondary:hover{background:rgba(255,255,255,0.18)}
.ig-progress-track{flex:1;min-width:120px;height:6px;
  background:rgba(255,255,255,0.12);border-radius:3px;overflow:hidden;cursor:pointer}
.ig-progress-fill{height:100%;background:linear-gradient(90deg,#0ea5e9,#38bdf8);
  width:0%;transition:width 0.2s ease}
.ig-time{font-size:12px;color:rgba(255,255,255,0.7);
  font-variant-numeric:tabular-nums;min-width:74px;text-align:right}
.ig-status{padding:10px 20px;background:#1e293b;color:rgba(255,255,255,0.75);
  font-size:12px;border-top:1px solid rgba(255,255,255,0.05)}
.ig-status b{color:#7dd3fc}

.ig-extras{display:flex;flex-wrap:wrap;align-items:center;gap:10px;
  padding:10px 20px;background:#0b1220;color:rgba(255,255,255,0.85);
  font-size:12px;border-top:1px solid rgba(255,255,255,0.06)}
.ig-extras label{display:flex;align-items:center;gap:6px;
  color:rgba(255,255,255,0.7);font-weight:600;letter-spacing:0.02em}
.ig-extras select,.ig-extras input[type="range"]{
  background:rgba(255,255,255,0.08);color:white;border:1px solid rgba(255,255,255,0.12);
  border-radius:6px;padding:4px 8px;font-size:12px;outline:none}
.ig-extras select{min-width:140px}
.ig-extras input[type="range"]{width:120px;padding:0}
.ig-rate-val{font-variant-numeric:tabular-nums;color:#7dd3fc;
  min-width:38px;text-align:right}
.ig-cap-toggle{cursor:pointer;user-select:none}

/* Data visuals on slides */
.ig-visual{margin:14px 0 8px 0;color:white;width:100%;max-width:560px}
.ig-viz-label{font-size:12px;color:rgba(255,255,255,0.7);
  text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;
  font-weight:600}

/* Bars */
.ig-bars-viz .ig-bar-row{display:grid;grid-template-columns:130px 1fr 50px;
  gap:10px;align-items:center;margin:6px 0;font-size:13px}
.ig-bar-label{color:rgba(255,255,255,0.92);font-weight:600;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ig-bar-track{height:14px;background:rgba(255,255,255,0.12);
  border-radius:7px;overflow:hidden;position:relative}
.ig-bar-fill{height:100%;width:0%;border-radius:7px;
  background:#0ea5e9;transition:width 1.1s cubic-bezier(.2,.7,.3,1);
  box-shadow:0 0 12px rgba(14,165,233,0.4)}
.ig-bar-pct{font-variant-numeric:tabular-nums;color:rgba(255,255,255,0.9);
  font-weight:700;text-align:right;font-size:13px}

/* Gauge */
.ig-gauge-viz{display:flex;align-items:center;gap:18px;margin:10px 0}
.ig-gauge-svg{width:160px;height:96px;flex-shrink:0}
.ig-gauge-arc{transition:stroke-dashoffset 1.4s cubic-bezier(.2,.7,.3,1);
  filter:drop-shadow(0 0 8px rgba(14,165,233,0.5))}
.ig-gauge-num{display:flex;align-items:baseline;gap:2px}
.ig-gauge-val{font-size:42px;font-weight:800;color:white;
  font-variant-numeric:tabular-nums;letter-spacing:-0.02em}
.ig-gauge-suffix{font-size:18px;color:rgba(255,255,255,0.7);font-weight:600}

/* Counter */
.ig-counter-viz{margin:10px 0}
.ig-counter-num{display:flex;align-items:baseline;gap:4px}
.ig-counter-val{font-size:64px;font-weight:900;color:white;
  font-variant-numeric:tabular-nums;letter-spacing:-0.03em;
  text-shadow:0 4px 16px rgba(0,0,0,0.3)}
.ig-counter-suffix{font-size:24px;color:rgba(255,255,255,0.75);
  font-weight:700;margin-left:4px}

/* Histogram */
.ig-histogram-viz{margin:10px 0}
.ig-histo-grid{display:flex;align-items:flex-end;gap:5px;
  height:90px;padding:0 2px}
.ig-histo-bar{flex:1;height:0%;
  background:linear-gradient(180deg,#7dd3fc 0%,#0284c7 100%);
  border-radius:4px 4px 0 0;
  transition:height 1.0s cubic-bezier(.2,.7,.3,1) calc(var(--i,0)*60ms);
  box-shadow:0 0 8px rgba(14,165,233,0.3)}

/* Timeline */
.ig-timeline-viz{position:relative;padding:24px 0 8px 0;margin:10px 0}
.ig-tl-line{position:absolute;left:5%;right:5%;top:36px;height:3px;
  background:linear-gradient(90deg,rgba(255,255,255,0.1) 0%,
    rgba(255,255,255,0.45) 50%,rgba(255,255,255,0.1) 100%);
  border-radius:2px}
.ig-tl-items{display:flex;justify-content:space-between;
  position:relative;padding:0 4%}
.ig-tl-item{flex:1;text-align:center;opacity:0;
  transform:translateY(8px);
  transition:opacity 0.6s ease var(--d,0ms),
    transform 0.6s ease var(--d,0ms)}
.ig-tl-item.shown{opacity:1;transform:translateY(0)}
.ig-tl-dot{width:14px;height:14px;border-radius:50%;
  background:#0ea5e9;margin:0 auto 8px;
  box-shadow:0 0 0 4px rgba(14,165,233,0.25);
  border:2px solid white}
.ig-tl-week{font-size:11px;color:rgba(255,255,255,0.7);font-weight:700;
  text-transform:uppercase;letter-spacing:0.06em}
.ig-tl-label{font-size:13px;color:white;font-weight:600;margin-top:3px}

/* Speaker indicator (pulses while narrating) */
.ig-speaker{position:absolute;top:18px;left:24px;display:flex;
  align-items:center;gap:6px;z-index:3;
  background:rgba(0,0,0,0.3);padding:5px 10px;border-radius:14px;
  font-size:11px;color:rgba(255,255,255,0.85);font-weight:600}
.ig-speaker-dot{width:8px;height:8px;border-radius:50%;
  background:#10b981;box-shadow:0 0 0 0 rgba(16,185,129,0.6)}
.ig-speaker.live .ig-speaker-dot{
  animation:ig-speaker-pulse 1.4s ease-in-out infinite}
@keyframes ig-speaker-pulse{
  0%{box-shadow:0 0 0 0 rgba(16,185,129,0.7)}
  70%{box-shadow:0 0 0 10px rgba(16,185,129,0)}
  100%{box-shadow:0 0 0 0 rgba(16,185,129,0)}
}

/* Voice quality badges in dropdown */
.ig-extras select option[data-quality="high"]::before{content:"✨ "}

/* Confetti canvas */
.ig-confetti{position:absolute;inset:0;pointer-events:none;z-index:5}

/* ────────────────────────────────────────────────────────────────────────
   Animated particle backgrounds per style — pure CSS, GPU-accelerated.
   ─────────────────────────────────────────────────────────────────────── */

.ig-stage::before, .ig-stage::after{
  content:"";position:absolute;inset:0;pointer-events:none;z-index:1}

/* Documentary — soft drifting dots */
.ig-style-documentary .ig-stage::before{
  background-image:
    radial-gradient(circle at 20% 30%, rgba(255,255,255,0.07) 1.5px, transparent 2px),
    radial-gradient(circle at 70% 60%, rgba(255,255,255,0.05) 1.5px, transparent 2px),
    radial-gradient(circle at 40% 80%, rgba(255,255,255,0.04) 1px, transparent 2px),
    radial-gradient(circle at 90% 20%, rgba(255,255,255,0.06) 1.5px, transparent 2px);
  background-size:140px 140px;
  animation:ig-drift 28s linear infinite}
@keyframes ig-drift{
  0%{background-position:0 0,30px 60px,60px 30px,90px 100px}
  100%{background-position:140px 140px,170px 200px,200px 170px,230px 240px}
}

/* Trailer — rising embers */
.ig-style-trailer .ig-stage::before{
  background-image:
    radial-gradient(circle at 15% 100%, rgba(245,158,11,0.55) 1.5px, transparent 4px),
    radial-gradient(circle at 35% 100%, rgba(220,38,38,0.5) 2px, transparent 5px),
    radial-gradient(circle at 60% 100%, rgba(245,158,11,0.45) 1.5px, transparent 4px),
    radial-gradient(circle at 85% 100%, rgba(239,68,68,0.55) 2px, transparent 5px);
  background-size:240px 240px;
  animation:ig-embers 9s linear infinite;
  filter:blur(0.5px)}
@keyframes ig-embers{
  0%{background-position:0 0,40px 0,80px 0,120px 0;opacity:0.9}
  100%{background-position:0 -240px,40px -240px,80px -240px,120px -240px;opacity:0.6}
}
.ig-style-trailer .ig-stage::after{
  background:radial-gradient(ellipse at center,transparent 40%,rgba(0,0,0,0.6) 100%)}

/* TED — warm spotlight from top */
.ig-style-ted_talk .ig-stage::before{
  background:linear-gradient(180deg,rgba(255,255,255,0.10) 0%,
    rgba(255,255,255,0.04) 30%,transparent 70%);
  animation:ig-glow 6s ease-in-out infinite alternate}
@keyframes ig-glow{
  0%{opacity:0.7}
  100%{opacity:1}
}
.ig-style-ted_talk .ig-stage::after{
  background:radial-gradient(ellipse at 50% -10%,rgba(255,200,150,0.18) 0%,transparent 60%)}

/* News — scan lines + corner static */
.ig-style-news .ig-stage::after{
  background-image:linear-gradient(180deg,transparent 50%,rgba(0,0,0,0.06) 50%);
  background-size:100% 4px;z-index:2;
  animation:ig-scan 14s linear infinite}
@keyframes ig-scan{
  0%{background-position:0 0}
  100%{background-position:0 100px}
}
.ig-style-news .ig-stage::before{
  background:radial-gradient(circle at 5% 5%, rgba(251,191,36,0.10) 0%,transparent 30%),
    radial-gradient(circle at 95% 95%, rgba(220,38,38,0.10) 0%,transparent 30%)}

/* Pitch deck — pulsing grid */
.ig-style-pitch_deck .ig-stage::before{
  background-image:
    linear-gradient(rgba(125,211,252,0.05) 1px,transparent 1px),
    linear-gradient(90deg,rgba(125,211,252,0.05) 1px,transparent 1px);
  background-size:40px 40px;
  animation:ig-grid-pulse 6s ease-in-out infinite}
@keyframes ig-grid-pulse{
  0%,100%{opacity:0.6}
  50%{opacity:1}
}

/* Slide content sits above particles */
.ig-slide{z-index:2}
.ig-lower-third{z-index:3}

/* ────────────────────────────────────────────────────────────────────────
   Chapter markers on the progress bar
   ─────────────────────────────────────────────────────────────────────── */

.ig-progress-track{position:relative}
.ig-chapters{position:absolute;inset:0;display:flex;pointer-events:none;
  z-index:2}
.ig-chapter{height:100%;border-right:1px solid rgba(255,255,255,0.32);
  pointer-events:auto;cursor:pointer;transition:background 0.2s;
  position:relative}
.ig-chapter:hover{background:rgba(255,255,255,0.12)}
.ig-chapter:last-child{border-right:none}
.ig-chapter-tip{position:absolute;bottom:120%;left:50%;transform:translateX(-50%);
  background:#0f172a;color:white;padding:6px 10px;border-radius:6px;
  font-size:11px;font-weight:600;white-space:nowrap;
  opacity:0;pointer-events:none;transition:opacity 0.15s;
  border:1px solid rgba(255,255,255,0.15);z-index:10}
.ig-chapter:hover .ig-chapter-tip{opacity:1}

/* ────────────────────────────────────────────────────────────────────────
   Dialogue (two-speaker) styling
   ─────────────────────────────────────────────────────────────────────── */

.ig-cap-line{margin:6px 0;padding-left:2px}
.ig-cap-tag{display:inline-block;font-weight:800;font-size:10px;
  padding:2px 8px;border-radius:6px;margin-right:8px;letter-spacing:0.06em;
  vertical-align:middle}
.ig-cap-a .ig-cap-tag{background:#ec4899;color:white}
.ig-cap-b .ig-cap-tag{background:#0ea5e9;color:white}
.ig-cap-a .word.active{background:rgba(236,72,153,0.45)}
.ig-cap-b .word.active{background:rgba(14,165,233,0.45)}

.ig-speaker[data-host="A"] .ig-speaker-dot{background:#ec4899;
  box-shadow:0 0 0 0 rgba(236,72,153,0.6)}
.ig-speaker[data-host="A"].live .ig-speaker-dot{
  animation:ig-speaker-pulse-a 1.4s ease-in-out infinite}
.ig-speaker[data-host="B"] .ig-speaker-dot{background:#0ea5e9}
.ig-speaker[data-host="B"].live .ig-speaker-dot{
  animation:ig-speaker-pulse-b 1.4s ease-in-out infinite}
@keyframes ig-speaker-pulse-a{
  0%{box-shadow:0 0 0 0 rgba(236,72,153,0.7)}
  70%{box-shadow:0 0 0 10px rgba(236,72,153,0)}
  100%{box-shadow:0 0 0 0 rgba(236,72,153,0)}
}
@keyframes ig-speaker-pulse-b{
  0%{box-shadow:0 0 0 0 rgba(14,165,233,0.7)}
  70%{box-shadow:0 0 0 10px rgba(14,165,233,0)}
  100%{box-shadow:0 0 0 0 rgba(14,165,233,0)}
}
.ig-speaker[data-host="A"] .ig-speaker-name::after{content:" — Host A"}
.ig-speaker[data-host="B"] .ig-speaker-name::after{content:" — Host B"}

/* ────────────────────────────────────────────────────────────────────────
   Floating emoji reactions
   ─────────────────────────────────────────────────────────────────────── */

.ig-reaction{position:absolute;font-size:48px;pointer-events:none;z-index:6;
  animation:ig-reaction-float 3.2s ease-out forwards;
  filter:drop-shadow(0 4px 10px rgba(0,0,0,0.45));
  user-select:none}
@keyframes ig-reaction-float{
  0%{opacity:0;transform:translateY(30px) scale(0.4) rotate(-8deg)}
  18%{opacity:1;transform:translateY(0) scale(1.25) rotate(2deg)}
  30%{transform:translateY(-20px) scale(1.05) rotate(-4deg)}
  100%{opacity:0;transform:translateY(-220px) scale(1.3) rotate(12deg)}
}

/* ────────────────────────────────────────────────────────────────────────
   Sequential content reveal — typewriter body + staggered bullets
   ─────────────────────────────────────────────────────────────────────── */

.ig-body.ig-typing::after{content:"▋";color:rgba(255,255,255,0.5);
  margin-left:2px;animation:ig-caret 0.8s steps(2) infinite}
@keyframes ig-caret{
  0%,50%{opacity:1}
  51%,100%{opacity:0}
}
.ig-bullets li.ig-hidden{opacity:0;transform:translateX(-18px)}
.ig-bullets li{transition:opacity 0.45s ease, transform 0.45s ease}

/* ────────────────────────────────────────────────────────────────────────
   Music toggle button state
   ─────────────────────────────────────────────────────────────────────── */

.ig-music-toggle.ig-music-on{background:#10b981;color:white}
.ig-music-toggle.ig-music-on:hover{background:#059669}
.ig-human-toggle.ig-human-on{background:#a855f7;color:white}
.ig-human-toggle.ig-human-on:hover{background:#9333ea}

@media (max-width:640px){
  .ig-slide{padding:24px 28px}
  .ig-slide-title{font-size:24px !important}
  .ig-style-trailer .ig-slide-title{font-size:32px !important}
  .ig-style-ted_talk .ig-slide-title{font-size:28px !important}
  .ig-slide .ig-slide-icon{font-size:32px}
  .ig-body{font-size:15px}
  .ig-bullets{font-size:14px}
  .ig-captions{font-size:13px;padding:8px 12px}
}
"""


# ─────────────────────────────────────────────────────────────────────────────
# JSON-safe encoding for inline <script>
# ─────────────────────────────────────────────────────────────────────────────

def _json_for_script(obj) -> str:
    """JSON-encode safely for embedding inside <script>...</script>."""
    return (
        json.dumps(obj)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace(" ", "\\u2028")
        .replace(" ", "\\u2029")
    )


# ─────────────────────────────────────────────────────────────────────────────
# Player JS (captions + voice + rate + confetti)
# ─────────────────────────────────────────────────────────────────────────────

_PLAYER_JS_TEMPLATE = r"""
(function(){
  const slides = __SLIDES__;
  const config = __CONFIG__;
  const totalDur = slides.reduce((s,x)=>s+(x.duration_s||8),0);
  let idx = 0, playing = false, paused = false, muted = false;
  let timer = null, progressTimer = null, slideStart = 0, elapsedBefore = 0;
  let captionsOn = true, sentenceCancelled = false;
  let currentRate = config.rate, currentPitch = config.pitch;
  let chosenVoiceURI = '';
  let cachedVoice = null;
  let dialogueMode = false;
  let cachedPair = null;
  let musicEnabled = false;
  let audioCtx = null, musicNodes = null;
  let humanizeEnabled = true;

  const root = document.getElementById('ig-video-root');
  if(!root) return;
  const slideEls = root.querySelectorAll('.ig-slide');
  const fill = root.querySelector('.ig-progress-fill');
  const timeEl = root.querySelector('.ig-time');
  const playBtn = root.querySelector('.ig-play');
  const muteBtn = root.querySelector('.ig-mute');
  const status = root.querySelector('.ig-status');
  const track = root.querySelector('.ig-progress-track');
  const captionEl = root.querySelector('.ig-captions');
  const voiceSel = root.querySelector('.ig-voice');
  const rateSlider = root.querySelector('.ig-rate');
  const rateVal = root.querySelector('.ig-rate-val');
  const capToggle = root.querySelector('.ig-cap-toggle');
  const modeBtn = root.querySelector('.ig-mode-toggle');
  const musicBtn = root.querySelector('.ig-music-toggle');
  const humanBtn = root.querySelector('.ig-human-toggle');
  const confettiCanvas = root.querySelector('.ig-confetti');
  const speakerEl = root.querySelector('.ig-speaker');
  const chaptersEl = root.querySelector('.ig-chapters');
  const stageEl = root.querySelector('.ig-stage');

  function fmt(s){ s=Math.max(0,Math.round(s)); return Math.floor(s/60)+':'+String(s%60).padStart(2,'0'); }
  function escapeHtml(s){
    return String(s).replace(/[&<>"']/g, c => ({
      '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
    }[c]));
  }

  // ── Voice quality scoring ───────────────────────────────────────────────
  // Higher score = more human-sounding. We bias hard toward neural/natural
  // voices and away from older robotic ones (eSpeak, Microsoft Sam, etc.).
  function scoreVoice(v){
    const name = (v.name||'');
    const lang = (v.lang||'').toLowerCase();
    let score = 0;
    if(lang.startsWith('en')) score += 30;
    if(lang === 'en-us' || lang === 'en-gb') score += 8;
    if(/natural|neural|premium|enhanced|wavenet|studio|polyglot/i.test(name)) score += 70;
    if(/online/i.test(name)) score += 22;
    if(/google.*natural|microsoft.*natural/i.test(name)) score += 12;
    // Modern human-named voices (Apple/MS/Google high-quality)
    const human = /\b(aria|guy|jenny|samantha|daniel|alex|karen|tessa|allison|moira|fiona|kate|ava|emma|brian|tom|ryan|amy|olivia|ethan|sara|libby|liam|noah|jane|lucy|maxwell|davis)\b/i;
    if(human.test(name)) score += 18;
    // Penalize known robotic engines and old voices
    if(/espeak|festival|robot|compact|skinny/i.test(name)) score -= 80;
    if(/microsoft (sam|david|mark|hazel)\b/i.test(name)) score -= 25;
    if(/^(zira|david)$/i.test(name)) score -= 15;
    if(v.default) score += 4;
    return score;
  }
  function pickBestVoice(voices){
    if(!voices || !voices.length) return null;
    let best = null, bestS = -Infinity;
    for(const v of voices){
      const s = scoreVoice(v);
      if(s > bestS){ bestS = s; best = v; }
    }
    return best;
  }
  function isHighQuality(v){ return scoreVoice(v) >= 60; }

  // ── Two-speaker pairing (NotebookLM-style podcast) ─────────────────────
  // Gender-classifies voices by name to pick a Female + Male pair so the
  // dialogue actually sounds like two distinct people.
  function classifyGender(v){
    const n = (v.name||'').toLowerCase();
    const fem = /\b(aria|jenny|samantha|karen|tessa|allison|moira|fiona|kate|ava|emma|olivia|sara|amy|libby|jane|lucy|zira|hazel|susan|alice|catherine|woman|female|f4)\b/;
    const mas = /\b(guy|daniel|alex|brian|tom|ryan|ethan|liam|noah|maxwell|davis|david|mark|sam|james|george|man|male|m4)\b/;
    if(fem.test(n)) return 'F';
    if(mas.test(n)) return 'M';
    return 'U';
  }
  function pickPair(voices){
    if(!voices || voices.length === 0) return null;
    const ranked = voices.map(v=>({v, s:scoreVoice(v), g:classifyGender(v)}))
                          .sort((a,b)=>b.s-a.s);
    const female = ranked.find(x=>x.g==='F');
    const male = ranked.find(x=>x.g==='M' && (!female || x.v !== female.v));
    if(female && male) return [female.v, male.v];
    // Fallback: top 2 different voices
    if(ranked.length >= 2) return [ranked[0].v, ranked[1].v];
    // Single voice — modulate via pitch difference at speak time
    return [ranked[0].v, ranked[0].v];
  }
  function refreshPair(){
    if(!('speechSynthesis' in window)) return;
    cachedPair = pickPair(window.speechSynthesis.getVoices());
  }

  // ── Captions ───────────────────────────────────────────────────────────
  function renderCaptions(text){
    if(!captionEl) return;
    if(dialogueMode){
      const sentences = splitSentences(text);
      let html = '', wordIdx = 0;
      sentences.forEach((sent, sIdx)=>{
        const tag = sIdx % 2 === 0 ? 'A' : 'B';
        html += '<div class="ig-cap-line ig-cap-' + tag.toLowerCase() + '">';
        html += '<span class="ig-cap-tag">' + tag + '</span>';
        const words = sent.split(/\s+/);
        words.forEach(w=>{
          if(!w) return;
          html += '<span class="word" data-i="' + wordIdx + '">' + escapeHtml(w) + '</span> ';
          wordIdx++;
        });
        html += '</div>';
      });
      captionEl.innerHTML = html;
    } else {
      const words = text.split(/\s+/);
      captionEl.innerHTML = words.map((w,i)=>
        '<span class="word" data-i="'+i+'">'+escapeHtml(w)+'</span>'
      ).join(' ');
    }
    captionEl.classList.toggle('visible', captionsOn);
  }
  function highlightWordIndex(globalIdx){
    if(!captionEl) return;
    const spans = captionEl.querySelectorAll('.word');
    spans.forEach((sp, i)=>{
      sp.classList.remove('active');
      sp.classList.toggle('spoken', i < globalIdx);
      if(i === globalIdx) sp.classList.add('active');
    });
  }

  // ── Cadence engine ─────────────────────────────────────────────────────
  // Splits narration into sentences and queues each utterance with subtle
  // pitch/rate variation + natural pauses. This is the difference between
  // monotone TTS and something that sounds *almost* human.
  function splitSentences(text){
    return text.split(/(?<=[.!?…])\s+/).map(s=>s.trim()).filter(Boolean);
  }
  function speakSentences(rawText){
    if(muted || !('speechSynthesis' in window)) return;
    try{ window.speechSynthesis.cancel(); }catch(e){}
    sentenceCancelled = false;
    if(speakerEl) speakerEl.classList.add('live');
    // Humanize FIRST so the words being spoken sound like a person, not paper
    const text = humanize(rawText);
    const phrases = splitPhrases(text);
    if(!phrases.length) return;
    // Word offsets in the humanized text (captions render the same)
    const wordOffsets = [];
    let acc = 0;
    for(const ph of phrases){
      wordOffsets.push(acc);
      acc += ph.split(/\s+/).filter(Boolean).length;
    }
    let i = 0;
    function next(){
      if(sentenceCancelled || muted){
        if(speakerEl) speakerEl.classList.remove('live');
        return;
      }
      if(i >= phrases.length){
        if(speakerEl) speakerEl.classList.remove('live');
        return;
      }
      const phrase = phrases[i];
      const wordOffset = wordOffsets[i];
      // 1. Speech-rate arc: slow open, crisp middle, slow tag-line
      const arcRate = rateArc(i, phrases.length, currentRate);
      // 2. Emotion-based prosody (the big one — questions rise, etc.)
      const emo = humanizeEnabled ? classifyEmotion(phrase) : 'neutral';
      const prosody = EMOTION_PROSODY[emo];
      // 3. Drawn-out filler delivery
      const filler = humanizeEnabled && isFillerPhrase(phrase);
      const r = filler
        ? arcRate * 0.72
        : arcRate * prosody.rate * (0.94 + Math.random()*0.12);
      const p = filler
        ? currentPitch * 0.86
        : currentPitch * prosody.pitch * (0.96 + Math.random()*0.08);
      const u = new SpeechSynthesisUtterance(phrase);
      u.rate = r; u.pitch = p; u.volume = filler ? 0.85 : 1.0;
      if(cachedVoice) u.voice = cachedVoice;
      // 4. Lip-smack before key sentences (start of slide + at random moments)
      if(humanizeEnabled && !filler && (i === 0 || Math.random() < 0.18)){
        try{ playLipSmack(); }catch(e){}
      }
      u.onstart = function(){ duckMusic(); };
      u.onboundary = function(ev){
        if(ev.name === 'word' || ev.charIndex !== undefined){
          const before = phrase.substring(0, ev.charIndex || 0);
          const localWords = before.trim().split(/\s+/).filter(Boolean).length;
          highlightWordIndex(wordOffset + localWords);
        }
      };
      u.onend = function(){
        i++;
        const last = phrase.slice(-1);
        let pause;
        if(last === '?' || last === '!') pause = 320;
        else if(last === '.' || last === '…') pause = 200;
        else if(last === ';' || last === '—' || last === '–') pause = 160;
        else if(last === ',') pause = 120;
        else pause = 100;
        // 5. Music swells back when speaker pauses
        if(i >= phrases.length) unduckMusic();
        // 6. Breath cue (~30% chance) at sentence ends
        if(humanizeEnabled && (last === '.' || last === '?' || last === '!')
           && Math.random() < 0.30 && i < phrases.length){
          playBreath();
        }
        setTimeout(next, pause);
      };
      u.onerror = function(){
        i++; unduckMusic(); setTimeout(next, 100);
      };
      detectReactions(phrase);
      window.speechSynthesis.speak(u);
    }
    next();
  }
  function stopSpeaking(){
    sentenceCancelled = true;
    try{ window.speechSynthesis && window.speechSynthesis.cancel(); }catch(e){}
    if(speakerEl) speakerEl.classList.remove('live');
  }

  // ── Dialogue cadence (two speakers alternating) ────────────────────────
  function speakSentencesDialogue(rawText){
    if(muted || !('speechSynthesis' in window)) return;
    try{ window.speechSynthesis.cancel(); }catch(e){}
    sentenceCancelled = false;
    if(speakerEl){ speakerEl.classList.add('live'); }
    const text = humanize(rawText);
    // For dialogue, prefer SENTENCES (full thoughts per speaker) over phrases
    // — backchannels fill the gap, no need to over-fragment a turn.
    const sentences = splitSentences(text);
    if(!sentences.length) return;
    if(!cachedPair) refreshPair();
    const pair = cachedPair || [cachedVoice, cachedVoice];
    const sameVoice = pair[0] === pair[1];
    const wordOffsets = [];
    let acc = 0;
    for(const s of sentences){
      wordOffsets.push(acc);
      acc += s.split(/\s+/).filter(Boolean).length;
    }
    let i = 0;
    function next(){
      if(sentenceCancelled || muted){
        if(speakerEl) speakerEl.classList.remove('live');
        return;
      }
      if(i >= sentences.length){
        if(speakerEl){ speakerEl.classList.remove('live'); speakerEl.removeAttribute('data-host'); }
        return;
      }
      const sent = sentences[i];
      const speakerIdx = i % 2;
      const voice = pair[speakerIdx];
      const otherVoice = pair[1 - speakerIdx];
      const wordOffset = wordOffsets[i];
      const pitchModifier = sameVoice
        ? (speakerIdx === 0 ? 1.18 : 0.82)
        : (speakerIdx === 0 ? 1.06 : 0.94);
      const arcRate = rateArc(i, sentences.length, currentRate);
      // Emotion prosody applies in dialogue too
      const emo = humanizeEnabled ? classifyEmotion(sent) : 'neutral';
      const prosody = EMOTION_PROSODY[emo];
      const filler = humanizeEnabled && isFillerPhrase(sent);
      const r = filler
        ? arcRate * 0.72
        : arcRate * prosody.rate * (0.94 + Math.random()*0.12);
      const p = filler
        ? currentPitch * pitchModifier * 0.88
        : currentPitch * pitchModifier * prosody.pitch * (0.97 + Math.random()*0.06);
      const u = new SpeechSynthesisUtterance(sent);
      u.rate = r; u.pitch = p; u.volume = filler ? 0.85 : 1.0;
      if(voice) u.voice = voice;
      // Lip-smack on speaker turn-change
      if(humanizeEnabled && !filler && Math.random() < 0.30){
        try{ playLipSmack(); }catch(e){}
      }
      u.onstart = function(){ duckMusic(); };
      u.onboundary = function(ev){
        if(ev.name === 'word' || ev.charIndex !== undefined){
          const before = sent.substring(0, ev.charIndex || 0);
          const localWords = before.trim().split(/\s+/).filter(Boolean).length;
          highlightWordIndex(wordOffset + localWords);
        }
      };
      u.onend = function(){
        i++;
        const last = sent.slice(-1);
        const pause = (last === '?' || last === '!') ? 360
                     : (last === '.' || last === '…') ? 240
                     : 180;
        if(i >= sentences.length) unduckMusic();
        // 25% chance the listener throws in a backchannel ("Mm-hmm.", "Right.")
        if(humanizeEnabled && i < sentences.length && Math.random() < 0.25){
          setTimeout(function(){ playBackchannel(otherVoice); }, 60);
          setTimeout(next, pause + 600);
        } else if(humanizeEnabled && (last === '.' || last === '?')
                  && Math.random() < 0.20){
          playBreath();
          setTimeout(next, pause);
        } else {
          setTimeout(next, pause);
        }
      };
      u.onerror = function(){ i++; unduckMusic(); setTimeout(next, 100); };
      if(speakerEl){ speakerEl.dataset.host = speakerIdx === 0 ? 'A' : 'B'; }
      detectReactions(sent);
      window.speechSynthesis.speak(u);
    }
    next();
  }
  function speak(text){
    if(dialogueMode) speakSentencesDialogue(text);
    else speakSentences(text);
  }

  // ── Visual animation ───────────────────────────────────────────────────
  function animateSlideVisual(slideEl){
    const v = slideEl.querySelector('.ig-visual');
    if(!v) return;
    const t = v.dataset.type;
    if(t === 'bars'){
      v.querySelectorAll('.ig-bar-fill').forEach((b,i)=>{
        const tgt = b.dataset.target;
        b.style.width = '0%';
        setTimeout(()=>{ b.style.width = tgt + '%'; }, 30 + i*80);
      });
    } else if(t === 'gauge'){
      const arc = v.querySelector('.ig-gauge-arc');
      const num = v.querySelector('.ig-gauge-val');
      const tgt = parseFloat(arc.dataset.target);
      const arclen = parseFloat(arc.dataset.arclen);
      arc.style.strokeDashoffset = arclen;
      setTimeout(()=>{
        arc.style.strokeDashoffset = arclen * (1 - tgt/100);
      }, 60);
      // Number ticker
      const dur = 1400;
      const start = performance.now();
      function tick(ts){
        const p = Math.min(1, (ts-start)/dur);
        const e = 1 - Math.pow(1-p, 3);
        if(num) num.textContent = Math.round(tgt * e);
        if(p < 1) requestAnimationFrame(tick);
      }
      requestAnimationFrame(tick);
    } else if(t === 'counter'){
      const el = v.querySelector('.ig-counter-val');
      const tgt = parseFloat(el.dataset.target);
      const dec = parseInt(el.dataset.decimals||'0', 10);
      const dur = 1500;
      const start = performance.now();
      function tick(ts){
        const p = Math.min(1, (ts-start)/dur);
        const e = 1 - Math.pow(1-p, 3);
        const val = tgt * e;
        el.textContent = dec ? val.toFixed(dec) : Math.round(val);
        if(p < 1) requestAnimationFrame(tick);
      }
      requestAnimationFrame(tick);
    } else if(t === 'histogram'){
      v.querySelectorAll('.ig-histo-bar').forEach((b)=>{
        const tgt = b.dataset.target;
        b.style.height = '0%';
        setTimeout(()=>{ b.style.height = tgt + '%'; }, 50);
      });
    } else if(t === 'timeline'){
      v.querySelectorAll('.ig-tl-item').forEach((it)=>{
        it.classList.remove('shown');
        setTimeout(()=>it.classList.add('shown'), 30);
      });
    }
  }

  function show(i){
    idx = Math.max(0, Math.min(slides.length-1, i));
    slideEls.forEach((el,n)=>el.classList.toggle('active', n===idx));
    elapsedBefore = slides.slice(0,idx).reduce((s,x)=>s+(x.duration_s||8),0);
    slideStart = Date.now();
    if(status){ status.innerHTML = 'Slide <b>'+(idx+1)+'/'+slides.length+'</b> &middot; '+escapeHtml(slides[idx].title||''); }
    renderCaptions(slides[idx].narration || '');
    animateSlideVisual(slideEls[idx]);
    revealSlideContent(slideEls[idx]);
    if(playing){ speak(slides[idx].narration || ''); scheduleNext(); }
    updateProgress();
  }

  function scheduleNext(){
    if(timer) clearTimeout(timer);
    const dur = (slides[idx].duration_s||8)*1000 / Math.max(0.5, currentRate);
    timer = setTimeout(()=>{
      if(idx < slides.length-1){ show(idx+1); }
      else { stop(); fireConfetti(); }
    }, dur);
  }

  function play(){
    playing = true; paused = false; playBtn.textContent='⏸';
    if(idx >= slides.length-1 && elapsedBefore + (Date.now()-slideStart)/1000 >= totalDur - 1){
      idx = 0;
    }
    if(musicEnabled) startMusic();
    show(idx);
    if(progressTimer) clearInterval(progressTimer);
    progressTimer = setInterval(updateProgress, 200);
  }
  function pause(){
    playing = false; paused = true; playBtn.textContent='▶';
    if(timer) clearTimeout(timer);
    stopSpeaking();
    stopMusic();
    if(progressTimer) clearInterval(progressTimer);
  }
  function stop(){
    playing = false; paused = false; playBtn.textContent='▶';
    if(timer) clearTimeout(timer);
    stopSpeaking();
    stopMusic();
    if(progressTimer) clearInterval(progressTimer);
    fill.style.width = '100%';
    timeEl.textContent = fmt(totalDur)+' / '+fmt(totalDur);
    if(status){ status.innerHTML = '🎉 <b>The end.</b> Click ▶ to replay.'; }
  }
  function updateProgress(){
    const slideElapsed = playing ? (Date.now()-slideStart)/1000 : 0;
    const total = elapsedBefore + Math.min(slideElapsed, slides[idx].duration_s||8);
    fill.style.width = Math.min(100, (total/totalDur)*100)+'%';
    timeEl.textContent = fmt(total)+' / '+fmt(totalDur);
  }

  function refreshChosenVoice(){
    if(!('speechSynthesis' in window)) return;
    const voices = window.speechSynthesis.getVoices();
    if(!voices || !voices.length) return;
    if(chosenVoiceURI){
      cachedVoice = voices.find(v=>v.voiceURI===chosenVoiceURI) || cachedVoice;
    } else {
      cachedVoice = pickBestVoice(voices);
    }
    refreshPair();
  }

  // ── Chapter markers ────────────────────────────────────────────────────
  function renderChapters(){
    if(!chaptersEl) return;
    let cum = 0;
    const html = slides.map(function(s, i){
      const left = (cum / totalDur) * 100;
      cum += s.duration_s || 8;
      const right = (cum / totalDur) * 100;
      const width = right - left;
      return '<div class="ig-chapter" data-idx="'+i+'" '
        + 'style="left:'+left+'%;width:'+width+'%">'
        + '<div class="ig-chapter-tip">'
        + (i+1)+'. '+escapeHtml(s.title||'')
        + ' &middot; '+(s.duration_s||8)+'s</div>'
        + '</div>';
    }).join('');
    chaptersEl.innerHTML = html;
    chaptersEl.querySelectorAll('.ig-chapter').forEach(function(c){
      c.addEventListener('click', function(e){
        e.stopPropagation();
        const i = parseInt(c.dataset.idx, 10) || 0;
        show(i);
      });
    });
  }

  // ── Synthesized background music ───────────────────────────────────────
  // Web Audio API generates an ambient chord pad whose voicing/timbre/filter
  // matches the video style. No samples, no API calls — pure oscillator math.
  const MUSIC_CONFIGS = {
    documentary: {root:196, intervals:[0,4,7,11], type:'sine',     filter:1400, lfoHz:0.06, gain:0.05},
    trailer:     {root:110, intervals:[0,3,7,10],type:'sawtooth', filter:480,  lfoHz:0.10, gain:0.04},
    ted_talk:    {root:174, intervals:[0,4,7],   type:'triangle', filter:1700, lfoHz:0.05, gain:0.05},
    news:        {root:220, intervals:[0,3,7],   type:'square',   filter:760,  lfoHz:0.12, gain:0.025},
    pitch_deck:  {root:130, intervals:[0,7,14],  type:'triangle', filter:900,  lfoHz:0.04, gain:0.04}
  };

  function ensureAudioCtx(){
    if(audioCtx) return audioCtx;
    try{
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }catch(e){ audioCtx = null; }
    return audioCtx;
  }

  function startMusic(){
    if(musicNodes) return;
    const ctx = ensureAudioCtx();
    if(!ctx) return;
    if(ctx.state === 'suspended'){ try{ ctx.resume(); }catch(e){} }
    const cfg = MUSIC_CONFIGS[config.style] || MUSIC_CONFIGS.documentary;
    const master = ctx.createGain();
    master.gain.setValueAtTime(0, ctx.currentTime);
    master.gain.linearRampToValueAtTime(cfg.gain, ctx.currentTime + 3.0);
    master.connect(ctx.destination);

    const filter = ctx.createBiquadFilter();
    filter.type = 'lowpass';
    filter.frequency.value = cfg.filter;
    filter.Q.value = 1.4;
    filter.connect(master);

    // Slow LFO modulating the filter cutoff for a breathing pad
    const lfo = ctx.createOscillator();
    lfo.frequency.value = cfg.lfoHz;
    const lfoGain = ctx.createGain();
    lfoGain.gain.value = cfg.filter * 0.35;
    lfo.connect(lfoGain);
    lfoGain.connect(filter.frequency);
    lfo.start();

    const oscs = cfg.intervals.map(function(semi, i){
      const o = ctx.createOscillator();
      o.type = cfg.type;
      o.frequency.value = cfg.root * Math.pow(2, semi/12);
      // Tiny detune per voice for chorus-y warmth
      o.detune.value = (i - cfg.intervals.length/2) * 5;
      const oG = ctx.createGain();
      oG.gain.value = 1.0 / cfg.intervals.length;
      o.connect(oG);
      oG.connect(filter);
      o.start();
      return o;
    });

    musicNodes = {ctx:ctx, master:master, filter:filter, oscs:oscs, lfo:lfo};
  }

  function stopMusic(){
    if(!musicNodes) return;
    const m = musicNodes;
    const t = m.ctx.currentTime;
    try{
      m.master.gain.cancelScheduledValues(t);
      m.master.gain.setValueAtTime(m.master.gain.value, t);
      m.master.gain.linearRampToValueAtTime(0, t + 1.0);
    }catch(e){}
    setTimeout(function(){
      try{ m.oscs.forEach(function(o){ try{o.stop();}catch(e){} }); }catch(e){}
      try{ m.lfo.stop(); }catch(e){}
      try{ m.master.disconnect(); }catch(e){}
    }, 1200);
    musicNodes = null;
  }

  // ── Emoji reactions ────────────────────────────────────────────────────
  // Detected from the spoken sentence. Keep the keyword list tight and the
  // pool moderate so reactions feel earned, not spammy.
  const REACTION_RULES = [
    {re:/\b(amazing|incredible|breakthrough|brilliant|extraordinary|astonish)\w*/i, e:'🤩'},
    {re:/\b(novel|innovative|original|unprecedented|first|never before)\w*/i,        e:'✨'},
    {re:/\b(insight|idea|hypothesis|epiphany)\w*/i,                                  e:'💡'},
    {re:/\b(success|achievement|win|triumph|wins?)\b/i,                              e:'🏆'},
    {re:/\b(grow|scale|expand|bigger|increase|improve)\w*/i,                         e:'📈'},
    {re:/\b(launch|deploy|ship|release|coming soon|ready)\b/i,                       e:'🚀'},
    {re:/\b(love|favorite|best|outstanding)\b/i,                                     e:'❤️'},
    {re:/\b(fast|quick|rapid|swift|instant)\w*/i,                                    e:'⚡'},
    {re:/\b(risk|danger|threat|warning|caution)\w*/i,                                e:'⚠️'},
    {re:/\b(problem|issue|broken|fail|trouble)\w*/i,                                 e:'😬'},
    {re:/\b(question|ask|wonder|curious|why)\b/i,                                    e:'🤔'},
    {re:/\b(important|critical|key|crucial|essential)\w*/i,                          e:'🔑'},
    {re:/\b(world|global|universal|everyone|everywhere)\w*/i,                        e:'🌍'},
    {re:/\b(money|cost|fund|budget|invest|expensive)\w*/i,                           e:'💰'},
    {re:/\b(team|together|collaborate|partner)\w*/i,                                 e:'👥'},
    {re:/\b(time|deadline|schedule|fast|slow)\w*/i,                                  e:'⏰'},
    {re:/\b(data|dataset|database|measure|statistic)\w*/i,                           e:'📊'},
    {re:/\b(experiment|test|trial|prove)\w*/i,                                       e:'🧪'},
    {re:/\b(thanks|thank you|grateful)\b/i,                                          e:'🙏'},
    {re:/\b(audience|reviewer|critic|listen)\w*/i,                                   e:'👀'}
  ];

  function emitReaction(emoji){
    if(!stageEl) return;
    const el = document.createElement('div');
    el.className = 'ig-reaction';
    el.textContent = emoji;
    el.style.left = (12 + Math.random() * 76) + '%';
    el.style.bottom = (15 + Math.random() * 25) + '%';
    stageEl.appendChild(el);
    setTimeout(function(){ try{ el.remove(); }catch(e){} }, 3400);
  }

  function detectReactions(sentence){
    const seen = {};
    for(let j = 0; j < REACTION_RULES.length; j++){
      const r = REACTION_RULES[j];
      if(r.re.test(sentence) && !seen[r.e]){
        seen[r.e] = true;
        // Stagger so multiple emojis don't pile up
        setTimeout(emitReaction.bind(null, r.e), 200 + Math.random() * 700);
      }
    }
  }

  // ── Humanization layer ─────────────────────────────────────────────────
  // Rewrites narration to sound less like a paper abstract being read aloud
  // and more like a person actually speaking. Style-aware (TED gets warm
  // storytelling hooks, news gets anchor cadence, etc.). Filler rate is
  // intentionally low so it doesn't feel performative.
  const HUMAN_HOOKS = {
    documentary: ['', '', 'Now, ', 'So, ', 'Right — '],
    trailer:     ['', '', 'Picture this — ', 'Listen — ', 'Imagine — '],
    ted_talk:    ['So, ', 'OK so, ', "Here's the thing — ", 'Now, ', 'Look, '],
    news:        ['', '', 'Now to this — ', 'And — ', 'Coming up: '],
    pitch_deck:  ['Look, ', 'OK so, ', 'Bottom line — ', "Here's the play — "]
  };
  const MID_FILLERS = [' Right? ', ' I mean, ', ' You know? ', ' Actually, ',
                        ' OK so ', ' And ', ' But — ', ' Honestly, '];
  const FORMAL_TO_CASUAL = [
    [/^Today, we present a research idea: /i, "OK so what we're looking at is "],
    [/^Today, we present /i, "So today we're talking about "],
    [/^Our hypothesis: /i, "Here's our hypothesis — "],
    [/^Our method, in short: /i, "And here's how we'll do it — "],
    [/^If this works, here's what we expect: /i, "If this works? "],
    [/^Why does this matter\?/i, "OK so, why does this matter?"],
    [/^Slide \d+\. /i, ''],
    [/^Here's the question we're investigating: /i, "Here's the question — "],
    [/\bnonetheless\b/gi, 'still'],
    [/\bfurthermore\b/gi, 'and also'],
    [/\bin conclusion\b/gi, 'so to wrap up'],
    [/\butilize\b/gi, 'use'],
    [/\bsubsequently\b/gi, 'then'],
    [/\bdemonstrate\b/gi, 'show'],
    [/\butilizes\b/gi, 'uses'],
    [/\bcommence\b/gi, 'start'],
    [/\bterminate\b/gi, 'end'],
    [/\bregarding\b/gi, 'about']
  ];

  function humanize(text){
    if(!humanizeEnabled || !text) return text;
    let out = text;
    // Casual replacements first
    FORMAL_TO_CASUAL.forEach(function(rule){
      out = out.replace(rule[0], rule[1]);
    });
    // Style-specific opening hook (40% chance)
    const hooks = HUMAN_HOOKS[config.style] || HUMAN_HOOKS.documentary;
    if(Math.random() < 0.40){
      const h = hooks[Math.floor(Math.random()*hooks.length)];
      if(h){
        out = h + out.charAt(0).toLowerCase() + out.slice(1);
      }
    }
    // Mid-sentence filler injection (~18% per sentence boundary)
    out = out.replace(/\.\s+([A-Z])/g, function(m, c){
      if(Math.random() < 0.18){
        const f = MID_FILLERS[Math.floor(Math.random()*MID_FILLERS.length)];
        return '.' + f + c.toLowerCase();
      }
      return m;
    });
    return out;
  }

  // ── Phrase-level cadence ───────────────────────────────────────────────
  // Real speech doesn't break only at periods. People pause at commas,
  // semicolons, and em-dashes too. Splitting on those gives finer prosody
  // control — and keeps each utterance short enough that pitch/rate
  // variation reads as "natural" not "buggy".
  function splitPhrases(text){
    // First split on hard sentence boundaries
    const sentences = text.split(/(?<=[.!?…])\s+/).map(s=>s.trim()).filter(Boolean);
    const phrases = [];
    sentences.forEach(function(s){
      // Within a sentence, split on commas/em-dashes/semicolons IF the
      // resulting fragment is at least 4 words (else don't fragment short ones).
      const subs = s.split(/(?<=[,;—–])\s+/);
      let buf = '';
      subs.forEach(function(sub){
        const wc = sub.split(/\s+/).filter(Boolean).length;
        if(buf && (buf + ' ' + sub).split(/\s+/).filter(Boolean).length < 5){
          buf = buf + ' ' + sub;
        } else if(wc < 3 && buf){
          buf = buf + ' ' + sub;
        } else {
          if(buf) phrases.push(buf.trim());
          buf = sub;
        }
      });
      if(buf) phrases.push(buf.trim());
    });
    return phrases;
  }

  // ── Speech-rate arc per slide ──────────────────────────────────────────
  // Mimics how a presenter naturally paces a thought: slower opening,
  // crisper middle, slow-down at the tag-line for emphasis.
  function rateArc(idx, total, baseRate){
    if(total <= 1) return baseRate;
    const pos = idx / (total - 1);  // 0 .. 1
    // Sine-shaped: starts at 0.94, peaks at ~1.04, ends at 0.88
    const arc = 0.94 + 0.10 * Math.sin(pos * Math.PI) - (pos > 0.85 ? 0.06 : 0);
    return baseRate * arc;
  }

  // ── Emotion-based prosody ──────────────────────────────────────────────
  // The single biggest reason TTS sounds "robotic": every phrase is
  // delivered with the same baseline. Real speakers raise pitch on
  // questions, peak on exclamations, drop on concerns, lift slightly on
  // curiosity. This per-phrase classifier sets the prosody before
  // sentence-level random jitter is applied on top.
  const EMOTION_PROSODY = {
    question:   {pitch: 1.12, rate: 0.96},  // rising, slightly slower
    excitement: {pitch: 1.10, rate: 1.08},  // higher, brisker
    concern:    {pitch: 0.88, rate: 0.93},  // lower, slower (gravitas)
    curious:    {pitch: 1.05, rate: 0.94},  // mild lift, contemplative
    soft:       {pitch: 0.94, rate: 0.92},  // hushed, intimate
    neutral:    {pitch: 1.00, rate: 1.00}
  };
  function classifyEmotion(phrase){
    const t = phrase.trim();
    if(/[?]\s*$/.test(t)) return 'question';
    if(/[!]\s*$/.test(t)) return 'excitement';
    if(/\b(amazing|incredible|breakthrough|brilliant|wow|extraordinary|astonish|launch|win)\w*/i.test(t)) return 'excitement';
    if(/\b(risk|danger|threat|problem|issue|concern|warning|broken|fail)\w*/i.test(t)) return 'concern';
    if(/\b(maybe|perhaps|could|might|wondering|curious|imagine|picture)\b/i.test(t)) return 'curious';
    if(/[…]\s*$/.test(t) || /\b(quietly|softly|whisper|gently|honestly)\b/i.test(t)) return 'soft';
    return 'neutral';
  }

  // ── Drawn-out filler detection ─────────────────────────────────────────
  // If a phrase IS a filler ("um", "you know", "so,"), it should be
  // delivered slower and with a lower pitch — that's what makes it sound
  // like a thinking pause rather than another spoken sentence.
  function isFillerPhrase(phrase){
    const t = phrase.trim().toLowerCase().replace(/[.,!?\s]+$/, '');
    return /^(um+|uh+|hmm+|so|well|right|i mean|you know|actually|honestly|listen|look|ok so|now)$/i.test(t);
  }

  // ── Audible breath synthesis ───────────────────────────────────────────
  // Brief filtered noise burst to simulate the soft breath between thoughts
  // — barely audible, but the brain registers it as "human".
  function playBreath(){
    const ctx = ensureAudioCtx();
    if(!ctx) return;
    if(ctx.state === 'suspended'){ try{ ctx.resume(); }catch(e){} }
    const dur = 0.18 + Math.random() * 0.10;
    const buf = ctx.createBuffer(1, ctx.sampleRate * dur, ctx.sampleRate);
    const data = buf.getChannelData(0);
    for(let i = 0; i < data.length; i++){
      // Pink-ish noise with envelope
      const env = Math.sin((i / data.length) * Math.PI);
      data[i] = (Math.random() * 2 - 1) * env * 0.6;
    }
    const src = ctx.createBufferSource();
    src.buffer = buf;
    const filt = ctx.createBiquadFilter();
    filt.type = 'bandpass';
    filt.frequency.value = 450 + Math.random() * 200;
    filt.Q.value = 0.6;
    const g = ctx.createGain();
    g.gain.value = 0.025;
    src.connect(filt); filt.connect(g); g.connect(ctx.destination);
    src.start();
  }

  // ── Lip-smack / breath-in transient ────────────────────────────────────
  // The barely-audible click humans make right before they start speaking
  // — when the lips part and the tongue clicks off the roof of the mouth.
  // ~80ms transient, high-passed, very low volume. This is the most
  // unconscious "human" cue.
  function playLipSmack(){
    const ctx = ensureAudioCtx();
    if(!ctx) return;
    if(ctx.state === 'suspended'){ try{ ctx.resume(); }catch(e){} }
    const dur = 0.06 + Math.random() * 0.03;
    const buf = ctx.createBuffer(1, ctx.sampleRate * dur, ctx.sampleRate);
    const data = buf.getChannelData(0);
    for(let i = 0; i < data.length; i++){
      // Very fast attack/decay envelope on noise — a click, not a hiss
      const t = i / data.length;
      const env = Math.exp(-t * 18) * Math.sin(t * Math.PI * 2);
      data[i] = (Math.random() * 2 - 1) * env * 0.7;
    }
    const src = ctx.createBufferSource();
    src.buffer = buf;
    const filt = ctx.createBiquadFilter();
    filt.type = 'highpass';
    filt.frequency.value = 1100 + Math.random() * 400;
    filt.Q.value = 0.9;
    const g = ctx.createGain();
    g.gain.value = 0.022;
    src.connect(filt); filt.connect(g); g.connect(ctx.destination);
    src.start();
  }

  // ── Music ducking ──────────────────────────────────────────────────────
  // Standard podcast / film mix trick: when the voice is talking, the
  // music drops to ~30% so it doesn't fight the narration. When the voice
  // pauses, the music gently swells back. This single change adds more
  // "produced" feel than any other audio touch.
  let musicDucked = false;
  function duckMusic(){
    if(!musicNodes || musicDucked) return;
    musicDucked = true;
    try{
      const t = musicNodes.ctx.currentTime;
      musicNodes.master.gain.cancelScheduledValues(t);
      const cur = musicNodes.master.gain.value;
      musicNodes.master.gain.setValueAtTime(cur, t);
      musicNodes.master.gain.linearRampToValueAtTime(cur * 0.30, t + 0.30);
    }catch(e){}
  }
  function unduckMusic(){
    if(!musicNodes || !musicDucked) return;
    musicDucked = false;
    try{
      const cfg = MUSIC_CONFIGS[config.style] || MUSIC_CONFIGS.documentary;
      const t = musicNodes.ctx.currentTime;
      musicNodes.master.gain.cancelScheduledValues(t);
      const cur = musicNodes.master.gain.value;
      musicNodes.master.gain.setValueAtTime(cur, t);
      musicNodes.master.gain.linearRampToValueAtTime(cfg.gain, t + 0.50);
    }catch(e){}
  }

  // ── Backchannel utterances (dialogue listening cues) ───────────────────
  // The non-active speaker occasionally says "Mm-hmm" / "Right" / "Yeah"
  // between the active speaker's sentences. This is what makes podcasts
  // feel like a conversation instead of two monologues.
  const BACKCHANNELS = ['Mm-hmm.', 'Right.', 'Yeah.', 'OK.', 'Got it.',
                         'Hmm.', 'Sure.', 'Mm.'];
  function playBackchannel(voice){
    if(!('speechSynthesis' in window) || muted) return;
    const text = BACKCHANNELS[Math.floor(Math.random()*BACKCHANNELS.length)];
    const u = new SpeechSynthesisUtterance(text);
    u.rate = currentRate * (0.85 + Math.random()*0.10);
    u.pitch = currentPitch * (0.82 + Math.random()*0.16);
    u.volume = 0.55;  // quieter than primary voice
    if(voice) u.voice = voice;
    try{ window.speechSynthesis.speak(u); }catch(e){}
  }

  // ── Sequential content reveal ──────────────────────────────────────────
  // Typewriter for body, staggered fade-in for bullets. Synced to slide
  // entry rather than to narration boundaries so it reliably looks alive.
  function revealSlideContent(slideEl){
    if(!slideEl) return;
    const body = slideEl.querySelector('.ig-body');
    if(body && !body.dataset.igRevealed){
      const original = body.textContent || '';
      body.dataset.igRevealed = '1';
      body.dataset.igFull = original;
      body.textContent = '';
      body.classList.add('ig-typing');
      let i = 0;
      const speed = Math.max(15, Math.min(38, 1200 / Math.max(20, original.length)));
      function tick(){
        if(i > original.length){
          body.classList.remove('ig-typing');
          return;
        }
        body.textContent = original.slice(0, i++);
        setTimeout(tick, speed + Math.random()*15);
      }
      setTimeout(tick, 350);
    } else if(body && body.dataset.igFull){
      // Replay reveal on revisit
      const full = body.dataset.igFull;
      body.textContent = '';
      body.classList.add('ig-typing');
      let i = 0;
      const speed = Math.max(15, Math.min(38, 1200 / Math.max(20, full.length)));
      function tick2(){
        if(i > full.length){
          body.classList.remove('ig-typing');
          return;
        }
        body.textContent = full.slice(0, i++);
        setTimeout(tick2, speed + Math.random()*15);
      }
      setTimeout(tick2, 350);
    }
    const bullets = slideEl.querySelectorAll('.ig-bullets li');
    bullets.forEach(function(b, i){
      b.classList.add('ig-hidden');
      setTimeout(function(){ b.classList.remove('ig-hidden'); },
                 600 + i * 320);
    });
  }

  function populateVoices(){
    if(!voiceSel || !('speechSynthesis' in window)) return;
    const voices = window.speechSynthesis.getVoices();
    if(!voices || !voices.length) return;
    // Sort by quality score so the best voices are at the top
    const ranked = voices.map(v=>({v, s:scoreVoice(v)})).sort((a,b)=>b.s-a.s);
    const best = ranked.length ? ranked[0].v : null;
    const cur = voiceSel.value;
    voiceSel.innerHTML = '';
    const optAuto = document.createElement('option');
    optAuto.value = '';
    optAuto.textContent = best
      ? '✨ Auto — '+best.name+' (best match)'
      : '🤖 Auto (best match)';
    voiceSel.appendChild(optAuto);
    ranked.slice(0, 40).forEach(({v, s})=>{
      const o = document.createElement('option');
      o.value = v.voiceURI;
      const badge = s >= 80 ? '✨ '
                  : s >= 50 ? '⭐ '
                  : '';
      o.textContent = badge + v.name + ' — ' + v.lang;
      if(s >= 50) o.dataset.quality = 'high';
      voiceSel.appendChild(o);
    });
    if(cur) voiceSel.value = cur;
    refreshChosenVoice();
  }

  function fireConfetti(){
    if(!confettiCanvas) return;
    const ctx = confettiCanvas.getContext('2d');
    const stage = root.querySelector('.ig-stage');
    confettiCanvas.width = stage.clientWidth;
    confettiCanvas.height = stage.clientHeight;
    const colors = ['#0ea5e9','#a855f7','#f59e0b','#10b981','#ec4899','#fbbf24'];
    const N = 120;
    const parts = [];
    for(let i=0;i<N;i++){
      parts.push({
        x: Math.random()*confettiCanvas.width,
        y: -10 - Math.random()*60,
        vx: (Math.random()-0.5)*4,
        vy: 2 + Math.random()*4,
        rot: Math.random()*Math.PI*2,
        vrot: (Math.random()-0.5)*0.2,
        size: 6 + Math.random()*8,
        color: colors[i % colors.length],
        life: 0,
      });
    }
    let t = 0;
    function frame(){
      ctx.clearRect(0,0,confettiCanvas.width,confettiCanvas.height);
      let alive = 0;
      parts.forEach(p=>{
        p.x += p.vx; p.y += p.vy; p.vy += 0.08; p.rot += p.vrot;
        p.life++;
        if(p.y < confettiCanvas.height + 30) alive++;
        ctx.save();
        ctx.translate(p.x, p.y); ctx.rotate(p.rot);
        ctx.fillStyle = p.color;
        ctx.fillRect(-p.size/2, -p.size/4, p.size, p.size/2);
        ctx.restore();
      });
      t++;
      if(alive > 0 && t < 240) requestAnimationFrame(frame);
      else ctx.clearRect(0,0,confettiCanvas.width,confettiCanvas.height);
    }
    frame();
  }

  // Wire controls
  playBtn.addEventListener('click', ()=>{ if(playing) pause(); else play(); });
  muteBtn.addEventListener('click', ()=>{
    muted = !muted; muteBtn.textContent = muted?'🔇':'🔊';
    if(muted){ stopSpeaking(); }
    else if(playing){ speak(slides[idx].narration||''); }
  });
  root.querySelector('.ig-prev').addEventListener('click', ()=>show(idx-1));
  root.querySelector('.ig-next').addEventListener('click', ()=>show(idx+1));
  track.addEventListener('click', (e)=>{
    const r = track.getBoundingClientRect();
    const t = ((e.clientX-r.left)/r.width)*totalDur;
    let acc=0, target=0;
    for(let i=0;i<slides.length;i++){
      acc += slides[i].duration_s||8;
      if(acc >= t){ target=i; break; }
    }
    show(target);
  });
  if(voiceSel){
    voiceSel.addEventListener('change', ()=>{
      chosenVoiceURI = voiceSel.value;
      refreshChosenVoice();
      if(playing) speak(slides[idx].narration||'');
    });
  }
  if(rateSlider){
    rateSlider.addEventListener('input', ()=>{
      currentRate = parseFloat(rateSlider.value);
      if(rateVal) rateVal.textContent = currentRate.toFixed(2)+'×';
      if(playing) speak(slides[idx].narration||'');
    });
  }
  if(capToggle){
    capToggle.addEventListener('click', ()=>{
      captionsOn = !captionsOn;
      if(captionEl) captionEl.classList.toggle('visible', captionsOn);
      capToggle.textContent = captionsOn ? '💬 CC on' : '💬 CC off';
    });
  }
  if(modeBtn){
    modeBtn.addEventListener('click', ()=>{
      dialogueMode = !dialogueMode;
      modeBtn.textContent = dialogueMode ? '👥 Duo' : '🎙️ Solo';
      modeBtn.title = dialogueMode
        ? 'Two-speaker podcast mode (alternating voices)'
        : 'Single-narrator mode';
      renderCaptions(slides[idx].narration || '');
      if(playing) speak(slides[idx].narration || '');
    });
  }
  if(musicBtn){
    musicBtn.addEventListener('click', ()=>{
      musicEnabled = !musicEnabled;
      musicBtn.classList.toggle('ig-music-on', musicEnabled);
      musicBtn.textContent = musicEnabled ? '🎵 Music on' : '🎵 Music off';
      if(musicEnabled && playing) startMusic();
      if(!musicEnabled) stopMusic();
    });
  }
  if(humanBtn){
    humanBtn.addEventListener('click', ()=>{
      humanizeEnabled = !humanizeEnabled;
      humanBtn.classList.toggle('ig-human-on', humanizeEnabled);
      humanBtn.textContent = humanizeEnabled ? '🧑 Human' : '🤖 Robot';
      humanBtn.title = humanizeEnabled
        ? 'Conversational mode (hooks, fillers, breaths, rate arc)'
        : 'Robotic mode (read narration verbatim)';
      if(playing) speak(slides[idx].narration||'');
    });
  }

  renderChapters();
  show(0);
  if(!('speechSynthesis' in window) && status){
    status.innerHTML = '<b>Note:</b> your browser does not support voice narration. Slides will still auto-advance.';
  }
  if('speechSynthesis' in window){
    populateVoices();
    refreshChosenVoice();
    window.speechSynthesis.onvoiceschanged = function(){
      populateVoices();
      refreshChosenVoice();
    };
  }
  if(config.autoplay) setTimeout(play, 400);
})();
"""


def _player_js(slides, autoplay: bool, style: str) -> str:
    cfg = VIDEO_STYLES.get(style, VIDEO_STYLES["documentary"])
    config = {
        "autoplay": bool(autoplay),
        "rate": float(cfg["default_rate"]),
        "pitch": float(cfg["default_pitch"]),
        "captions": bool(cfg.get("captions", True)),
        "style": style,
    }
    return (
        _PLAYER_JS_TEMPLATE
        .replace("__SLIDES__", _json_for_script(slides))
        .replace("__CONFIG__", _json_for_script(config))
    )


# ─────────────────────────────────────────────────────────────────────────────
# Embed builders
# ─────────────────────────────────────────────────────────────────────────────

def build_video_embed(slides: List[Dict[str, Any]], idea: Dict[str, Any],
                       autoplay: bool = False,
                       style: str = "documentary") -> str:
    """Return an HTML fragment (no <html>/<body>) safe to embed in Streamlit."""
    if style not in VIDEO_STYLES:
        style = "documentary"
    style_cfg = VIDEO_STYLES[style]
    total_s = estimate_duration_s(slides)
    slide_html = "".join(_slide_to_html(s, i, len(slides), style)
                          for i, s in enumerate(slides))
    js = _player_js(slides, autoplay, style)

    voice_panel = (
        '<div class="ig-extras">'
        '<label>🎙️ Voice <select class="ig-voice">'
        '<option value="">🤖 Auto (best match)</option></select></label>'
        f'<label>⚡ Rate <input type="range" class="ig-rate" '
        f'min="0.6" max="1.6" step="0.05" value="{style_cfg["default_rate"]}">'
        f'<span class="ig-rate-val">{style_cfg["default_rate"]:.2f}×</span></label>'
        '<button class="ig-btn ig-btn-secondary ig-cap-toggle" '
        'title="Toggle captions" style="width:auto;padding:0 12px;border-radius:6px;'
        'font-size:12px;height:30px">💬 CC on</button>'
        '<button class="ig-btn ig-btn-secondary ig-mode-toggle" '
        'title="Single-narrator mode" '
        'style="width:auto;padding:0 12px;border-radius:6px;'
        'font-size:12px;height:30px">🎙️ Solo</button>'
        '<button class="ig-btn ig-btn-secondary ig-music-toggle" '
        'title="Toggle synthesized background music" '
        'style="width:auto;padding:0 12px;border-radius:6px;'
        'font-size:12px;height:30px">🎵 Music off</button>'
        '<button class="ig-btn ig-btn-secondary ig-human-toggle ig-human-on" '
        'title="Conversational mode (hooks, fillers, breaths, rate arc)" '
        'style="width:auto;padding:0 12px;border-radius:6px;'
        'font-size:12px;height:30px">🧑 Human</button>'
        f'<span style="margin-left:auto;color:#7dd3fc;font-weight:600">'
        f'{_html.escape(style_cfg["label"])}</span>'
        '</div>'
    )

    speaker_html = (
        '<div class="ig-speaker">'
        '<span class="ig-speaker-dot"></span>'
        '<span class="ig-speaker-name">NARRATING</span>'
        '</div>'
    )

    return (
        f'<style>{_PLAYER_CSS}</style>'
        f'<div class="ig-video-wrap ig-style-{style}" id="ig-video-root">'
        f'<div class="ig-stage">{slide_html}'
        f'{speaker_html}'
        f'<div class="ig-captions"></div>'
        f'<canvas class="ig-confetti"></canvas>'
        f'</div>'
        f'<div class="ig-controls">'
        f'<button class="ig-btn ig-btn-secondary ig-prev" title="Previous">⏮</button>'
        f'<button class="ig-btn ig-play" title="Play/Pause">▶</button>'
        f'<button class="ig-btn ig-btn-secondary ig-next" title="Next">⏭</button>'
        f'<div class="ig-progress-track" title="Click to jump">'
        f'<div class="ig-progress-fill"></div>'
        f'<div class="ig-chapters"></div></div>'
        f'<div class="ig-time">0:00 / {total_s//60}:{total_s%60:02d}</div>'
        f'<button class="ig-btn ig-btn-secondary ig-mute" title="Mute narration">🔊</button>'
        f'</div>'
        f'{voice_panel}'
        f'<div class="ig-status">Ready. Click <b>▶</b> to begin narrated playback.</div>'
        f'</div>'
        f'<script>{js}</script>'
    )


def build_video_html(slides: List[Dict[str, Any]], idea: Dict[str, Any],
                      autoplay: bool = False,
                      style: str = "documentary") -> str:
    """Return a full standalone HTML document for download/sharing."""
    title = _html.escape(idea.get("title", "Idea Video"))
    embed = build_video_embed(slides, idea, autoplay=autoplay, style=style)
    return (
        f'<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>{title} — IdeaGraph Video</title>'
        f'<style>body{{margin:0;padding:24px 12px;background:#f1f5f9;'
        f'min-height:100vh;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif}}'
        f'.ig-footer{{text-align:center;color:#64748b;font-size:12px;margin-top:18px}}</style>'
        f'</head><body>{embed}'
        f'<div class="ig-footer">Generated by IdeaGraph · '
        f'Voice: Web Speech API · Open in Chrome/Edge/Safari for best results</div>'
        f'</body></html>'
    )
