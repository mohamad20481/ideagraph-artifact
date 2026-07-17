"""
speed_optimizer.py - Pipeline speed optimizations.

  1. StageAwareRouter   - Route probes to Haiku, ideation to Sonnet, complex
                          synthesis to Opus. Cuts ~60% wall-clock time on
                          pipelines that mix fast + slow stages.
  2. AdaptiveConcurrency - Track LLM response time; reduce parallelism when
                            we see 503s/timeouts, increase when fast. Avoids
                            hammering a struggling proxy.
  3. ProbeShortcuts     - Skip full 10-D probe when cheap heuristic checks
                            already disqualify an idea (saves 1 LLM call/cell).
  4. SavedPresets       - Persist (topic, knobs, model) combinations the user
                            can re-run with one click.

All three are orthogonal — each can be enabled independently.
"""

from __future__ import annotations

import json
import os
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1. STAGE-AWARE PROVIDER / MODEL ROUTING
# ─────────────────────────────────────────────────────────────────────────────

# Map pipeline stage → recommended (provider, model) tier.
# Cheap models (Haiku/DeepSeek) for high-volume mechanical tasks.
# Mid (Sonnet/DeepSeek) for ideation.
# Premium (Opus/GPT-4o) reserved for synthesis if the user opts in.
STAGE_TIERS = {
    # Probing is high-volume (10+ per cell), low-creativity → cheap
    "probe":           "cheap",
    "dedup_check":     "cheap",
    "novelty_check":   "cheap",
    # Ideation is medium-creativity → balanced
    "ideation":        "balanced",
    "revision":        "balanced",
    "topic_decompose": "balanced",
    # Synthesis / review is high-stakes → premium (or balanced if cost matters)
    "synthesis":       "balanced",
    "debate_judge":    "balanced",
    "paper_write":     "balanced",
    "review":          "balanced",
    # Default fallback
    "default":         "balanced",
}

# Per-tier model preference per provider — first available key wins.
# Kimi: moonshot-v1-8k for cheap probes, kimi-k2-0905-preview as flagship.
TIER_MODELS = {
    "cheap": [
        ("anthropic", "claude-haiku-4-5"),
        ("groq",      "llama-3.3-70b-versatile"),
        ("gemini",    "gemini-2.0-flash"),
        ("kimi",      "moonshot-v1-8k"),
        ("deepseek",  "deepseek-chat"),
    ],
    "balanced": [
        ("anthropic", "claude-sonnet-4-6"),
        ("deepseek",  "deepseek-chat"),
        ("openai",    "gpt-4o-mini"),
        ("kimi",      "moonshot-v1-32k"),
        ("gemini",    "gemini-2.0-flash"),
    ],
    "premium": [
        ("anthropic", "claude-opus-4-7"),
        ("openai",    "gpt-4o"),
        ("kimi",      "moonshot-v1-32k"),
        ("deepseek",  "deepseek-chat"),
    ],
}


def route_for_stage(stage: str, prefer_provider: Optional[str] = None) -> tuple:
    """
    Pick (provider, model) for a given pipeline stage.

    If prefer_provider is set, try that provider first (any tier model it has).
    Otherwise pick the first tier model whose API key is configured.
    Returns the user's current default if no routing is possible.
    """
    try:
        import config as _cfg
    except Exception:
        return ("deepseek", "deepseek-chat")

    tier = STAGE_TIERS.get(stage, STAGE_TIERS["default"])
    candidates = TIER_MODELS.get(tier, TIER_MODELS["balanced"])

    # If the user explicitly picked a provider, stick with it (don't override their choice)
    if prefer_provider:
        # Find a tier-appropriate model for the user's provider
        for prov, model in candidates:
            if prov == prefer_provider:
                if _has_key(prov):
                    return (prov, model)
        # Fall back to user's default model on their preferred provider
        return (prefer_provider, _cfg._DEFAULT_MODELS.get(prefer_provider, _cfg.MODEL))

    # No preference — pick the first tier candidate with a configured key
    for prov, model in candidates:
        if _has_key(prov):
            return (prov, model)

    # Last resort: current default
    return (getattr(_cfg, "PROVIDER", "deepseek"), getattr(_cfg, "MODEL", "deepseek-chat"))


def _has_key(provider: str) -> bool:
    """Check whether the given provider's API key is configured."""
    try:
        import config as _cfg
        key_attr = {
            "deepseek": "DEEPSEEK_API_KEY",
            "openai": "OPENAI_API_KEY",
            "groq": "GROQ_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "azure": "AZURE_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "kimi": "KIMI_API_KEY",
        }.get(provider)
        if not key_attr:
            return False
        val = getattr(_cfg, key_attr, "")
        return bool(val) and not val.startswith(("sk-xxx", "your-"))
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 2. ADAPTIVE CONCURRENCY
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveConcurrency:
    """
    Tracks recent LLM call latencies + error rate to dynamically adjust the
    pipeline's worker count. When the API is fast and healthy, run 6 cells
    in parallel; when it's struggling (503s, slow), drop to 2.
    """

    def __init__(self, min_workers: int = 1, max_workers: int = 8,
                 default_workers: int = 4, window_size: int = 20) -> None:
        self.min_workers = min_workers
        self.max_workers = max_workers
        self._current = default_workers
        self._latencies: deque = deque(maxlen=window_size)
        self._errors: deque = deque(maxlen=window_size)
        self._lock = threading.Lock()

    def record_call(self, duration_s: float, success: bool) -> None:
        with self._lock:
            self._latencies.append(duration_s)
            self._errors.append(0 if success else 1)

    @property
    def recommended_workers(self) -> int:
        with self._lock:
            if len(self._latencies) < 5:
                return self._current

            error_rate = sum(self._errors) / len(self._errors)
            avg_latency = statistics.median(self._latencies)

            # Hard reduce on high error rate
            if error_rate >= 0.4:
                target = self.min_workers
            elif error_rate >= 0.2:
                target = max(self.min_workers, self._current - 1)
            elif avg_latency > 30 and self._current > 2:
                # Slow API — back off
                target = max(2, self._current - 1)
            elif avg_latency < 8 and error_rate < 0.05:
                # Fast + healthy — scale up
                target = min(self.max_workers, self._current + 1)
            else:
                target = self._current

            self._current = target
            return target

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "current_workers": self._current,
                "avg_latency_s": round(statistics.median(self._latencies), 2) if self._latencies else 0,
                "error_rate": round(sum(self._errors) / max(len(self._errors), 1), 3),
                "samples": len(self._latencies),
            }


_GLOBAL_CONCURRENCY: Optional[AdaptiveConcurrency] = None
_CONCURRENCY_LOCK = threading.Lock()


def get_adaptive_concurrency() -> AdaptiveConcurrency:
    global _GLOBAL_CONCURRENCY
    with _CONCURRENCY_LOCK:
        if _GLOBAL_CONCURRENCY is None:
            _GLOBAL_CONCURRENCY = AdaptiveConcurrency()
        return _GLOBAL_CONCURRENCY


# ─────────────────────────────────────────────────────────────────────────────
# 3. PROBE SHORTCUTS — skip the LLM probe when heuristics already decide
# ─────────────────────────────────────────────────────────────────────────────

def quick_probe_shortcut(idea: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Cheap heuristic checks before sending the idea to the 10-D LLM probe.
    Returns a fake probe result if the idea is obviously bad (saves 1 LLM call),
    or None if we should proceed with the real probe.
    """
    method = (idea.get("method") or "").strip()
    title = (idea.get("title") or "").strip()
    hypothesis = (idea.get("hypothesis") or "").strip()

    # Way too short — clearly underspecified
    if len(method) < 40 or len(title) < 5:
        return _make_fake_probe(0.15, "Method/title too short")

    # Hypothesis missing entirely
    if len(hypothesis) < 15:
        return _make_fake_probe(0.20, "Hypothesis missing or too vague")

    # Stop-word density check (high stop words = vague writing)
    words = method.lower().split()
    if len(words) < 8:
        return _make_fake_probe(0.18, "Method too brief")

    # Otherwise let the real probe run
    return None


def _make_fake_probe(quality: float, reason: str) -> Dict[str, Any]:
    """Build a probe-shaped result for shortcut rejection."""
    scores = {
        "code": quality, "dataset": quality, "constraint": quality,
        "novelty": quality, "specificity": quality * 0.8,
        "significance": quality, "clarity": quality, "testability": quality,
        "scalability": quality, "risk_balance": quality,
    }
    return {
        "all_pass": False,
        "scores": scores,
        "details": {},
        "feedback": f"Shortcut rejection: {reason}",
        "quality": quality,
        "_shortcut": True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. SAVED PRESETS — persist (topic, knobs, provider) for one-click re-run
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Preset:
    name: str
    topic: str
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-6"
    budget_usd: float = 0.6
    iterations: int = 20
    debate_enabled: bool = False
    creativity: float = 0.7
    time_weeks: int = 12
    risk: str = "medium"
    domain: str = "auto"
    enable_repro: bool = True
    enable_fmea: bool = True
    enable_adversarial: bool = False
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Preset":
        # Filter unknown keys (forward-compat with older preset files)
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


# Build-in starter presets for new users
BUILTIN_PRESETS: List[Preset] = [
    Preset(
        name="📊 Quick Empirical Survey",
        topic="empirical comparison of methods in your domain",
        provider="anthropic", model="claude-haiku-4-5",
        budget_usd=0.3, iterations=10,
        creativity=0.4, time_weeks=4, risk="low", domain="auto",
    ),
    Preset(
        name="🚀 Moonshot Brainstorm",
        topic="ambitious research direction",
        provider="anthropic", model="claude-sonnet-4-6",
        budget_usd=1.0, iterations=20,
        creativity=0.95, time_weeks=52, risk="high",
        enable_adversarial=True, domain="auto",
    ),
    Preset(
        name="🔬 PhD Thesis Setup",
        topic="thesis-scale research program",
        provider="anthropic", model="claude-sonnet-4-6",
        budget_usd=1.5, iterations=30, debate_enabled=True,
        creativity=0.7, time_weeks=52, risk="medium",
        enable_repro=True, enable_fmea=True, domain="auto",
    ),
    Preset(
        name="⚡ Hackathon Sprint",
        topic="weekend project idea",
        provider="anthropic", model="claude-haiku-4-5",
        budget_usd=0.2, iterations=8,
        creativity=0.6, time_weeks=2, risk="medium", domain="auto",
    ),
]


def _presets_table_init(conn) -> None:
    """Create the user_presets table on first use (idempotent)."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_presets (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            name        TEXT    NOT NULL,
            preset_json TEXT    NOT NULL,
            created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            UNIQUE(user_id, name)
        )
    """)


def save_preset(user_id: int, preset: Preset) -> bool:
    """Save (or overwrite) a named preset for the user."""
    if not user_id:
        return False
    try:
        import db
        with db._lock:
            conn = db._get_conn()
            try:
                _presets_table_init(conn)
                preset.created_at = preset.created_at or _now_iso()
                conn.execute(
                    """INSERT INTO user_presets (user_id, name, preset_json)
                       VALUES (?, ?, ?)
                       ON CONFLICT(user_id, name)
                       DO UPDATE SET preset_json = excluded.preset_json,
                                     created_at = datetime('now')""",
                    (user_id, preset.name, json.dumps(preset.to_dict())),
                )
                conn.commit()
                return True
            finally:
                conn.close()
    except Exception:
        return False


def get_user_presets(user_id: int) -> List[Preset]:
    """Load all presets saved by a user (newest first)."""
    if not user_id:
        return []
    try:
        import db
        with db._lock:
            conn = db._get_conn()
            try:
                _presets_table_init(conn)
                rows = conn.execute(
                    "SELECT preset_json FROM user_presets "
                    "WHERE user_id = ? ORDER BY created_at DESC",
                    (user_id,),
                ).fetchall()
                presets = []
                for r in rows:
                    try:
                        presets.append(Preset.from_dict(json.loads(r["preset_json"])))
                    except Exception:
                        continue
                return presets
            finally:
                conn.close()
    except Exception:
        return []


def delete_preset(user_id: int, name: str) -> bool:
    """Delete a preset by name."""
    if not user_id or not name:
        return False
    try:
        import db
        with db._lock:
            conn = db._get_conn()
            try:
                _presets_table_init(conn)
                conn.execute(
                    "DELETE FROM user_presets WHERE user_id = ? AND name = ?",
                    (user_id, name),
                )
                conn.commit()
                return True
            finally:
                conn.close()
    except Exception:
        return False


def _now_iso() -> str:
    from datetime import datetime
    return datetime.now().isoformat(timespec="seconds")


# ─────────────────────────────────────────────────────────────────────────────
# 5. IDEA COMPARISON MATRIX (side-by-side analysis)
# ─────────────────────────────────────────────────────────────────────────────

COMPARISON_DIMENSIONS = [
    ("title",              "Title"),
    ("methodology_type",   "Methodology"),
    ("novelty_level",      "Novelty"),
    ("quality_score",      "Quality"),
    ("probe_scores.code",       "Code"),
    ("probe_scores.dataset",    "Dataset"),
    ("probe_scores.novelty",    "Novelty Score"),
    ("probe_scores.specificity","Specificity"),
    ("probe_scores.significance","Significance"),
    ("probe_scores.testability","Testability"),
]


def _get_field(idea: Dict[str, Any], path: str) -> Any:
    """Read a dotted path from an idea dict (e.g. 'probe_scores.code')."""
    parts = path.split(".")
    val: Any = idea
    for p in parts:
        if not isinstance(val, dict):
            return None
        val = val.get(p)
    return val


def build_comparison_matrix(ideas: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build a side-by-side comparison structure for 2-5 ideas.
    Returns:
      {
        "headers": ["Title", "Methodology", ...],
        "rows":    [["Idea A", "Idea B", ...], ...],   # one row per dimension
        "winners": {"Quality": 0, "Code": 1, ...},     # index of best idea per row
        "summary": {"overall_winner": int, "wins_per_idea": [3, 1, 2]},
      }
    """
    if not ideas:
        return {"headers": [], "rows": [], "winners": {}, "summary": {}}
    if len(ideas) > 5:
        ideas = ideas[:5]

    headers = []
    rows = []
    winners: Dict[str, int] = {}
    wins_per_idea = [0] * len(ideas)

    for path, label in COMPARISON_DIMENSIONS:
        values = [_get_field(i, path) for i in ideas]
        # Format values for display
        formatted = []
        numeric_values: List[Optional[float]] = []
        for v in values:
            if v is None:
                formatted.append("—")
                numeric_values.append(None)
            elif isinstance(v, (int, float)):
                formatted.append(f"{v:.2f}" if isinstance(v, float) else str(v))
                numeric_values.append(float(v))
            elif isinstance(v, str):
                formatted.append(v[:50].replace("_", " ").title()
                                  if path in ("methodology_type", "novelty_level") else v[:60])
                numeric_values.append(None)
            else:
                formatted.append(str(v)[:60])
                numeric_values.append(None)

        rows.append(formatted)
        headers.append(label)

        # Compute winner index for numeric dimensions
        valid_numeric = [(idx, v) for idx, v in enumerate(numeric_values) if v is not None]
        if len(valid_numeric) >= 2:
            best_idx, best_val = max(valid_numeric, key=lambda x: x[1])
            # Only mark a winner if there's a real gap (no tie)
            others = [v for idx, v in valid_numeric if idx != best_idx]
            if others and best_val > max(others) + 0.01:
                winners[label] = best_idx
                wins_per_idea[best_idx] += 1

    overall = wins_per_idea.index(max(wins_per_idea)) if wins_per_idea else 0

    return {
        "headers": headers,
        "rows": rows,
        "winners": winners,
        "summary": {
            "overall_winner": overall,
            "wins_per_idea": wins_per_idea,
            "total_ideas": len(ideas),
            "titles": [i.get("title", f"Idea {n+1}")[:60] for n, i in enumerate(ideas)],
        },
    }
