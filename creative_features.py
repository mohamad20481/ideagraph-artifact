"""
creative_features.py - Advanced creative features for IdeaGraph.

  1. Visual Abstract Generator — beautiful HTML poster for each idea
  2. Research Trend Predictor — predict hot topics from citation patterns
  3. Idea Mashup Generator — combine 2 ideas into a novel hybrid
  4. Collaborative Workspace — shared research rooms
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import sqlite3
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from agents.base_agent import BaseAgent

try:
    import db
except ImportError:
    db = None


# ═══════════════════════════════════════════════════════════════════════════
# 1. VISUAL ABSTRACT GENERATOR
# ═══════════════════════════════════════════════════════════════════════════

# Gradient color palettes for visual abstracts
POSTER_PALETTES = [
    {"bg": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)", "accent": "#f1c40f", "text": "white"},
    {"bg": "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)", "accent": "#ffecd2", "text": "white"},
    {"bg": "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)", "accent": "#fddb92", "text": "#1a1a2e"},
    {"bg": "linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)", "accent": "#fa709a", "text": "#1a1a2e"},
    {"bg": "linear-gradient(135deg, #fa709a 0%, #fee140 100%)", "accent": "#30cfd0", "text": "#1a1a2e"},
    {"bg": "linear-gradient(135deg, #a18cd1 0%, #fbc2eb 100%)", "accent": "#fad0c4", "text": "#2d3436"},
    {"bg": "linear-gradient(135deg, #fccb90 0%, #d57eeb 100%)", "accent": "#a1ffce", "text": "#2d3436"},
    {"bg": "linear-gradient(135deg, #0c3547 0%, #2c7744 100%)", "accent": "#f1c40f", "text": "white"},
]


def generate_visual_abstract(idea: Dict[str, Any], topic: str = "") -> str:
    """
    Generate a beautiful HTML visual abstract (poster) for an idea.
    Returns HTML string that can be displayed or downloaded.
    """
    title = idea.get("title", "Untitled Idea")[:100]
    q = idea.get("quality_score", 0)
    method_type = (idea.get("methodology_type") or "research").replace("_", " ").title()
    novelty = (idea.get("novelty_level") or "moderate").capitalize()
    strategy = {"A": "Frontier Extension", "B": "Cross-Cluster Bridge", "C": "Gap-Filling"}.get(
        idea.get("source_strategy", ""), "Research"
    )

    motivation = idea.get("motivation", "")[:200]
    method = idea.get("method", "")[:250]
    hypothesis = idea.get("hypothesis", "")[:200]
    outcome = idea.get("expected_outcome", "")[:200]

    # Pick palette based on title hash for consistency
    palette_idx = int(hashlib.md5(title.encode()).hexdigest()[:2], 16) % len(POSTER_PALETTES)
    p = POSTER_PALETTES[palette_idx]

    # Quality badge
    if q >= 0.8:
        badge = "A+"
        badge_color = "#27ae60"
    elif q >= 0.7:
        badge = "A"
        badge_color = "#2ecc71"
    elif q >= 0.5:
        badge = "B"
        badge_color = "#f39c12"
    else:
        badge = "C"
        badge_color = "#e74c3c"

    # Probe scores bar
    scores = idea.get("probe_scores", {})
    probe_bars = ""
    for pk in ["code", "dataset", "constraint", "novelty"]:
        pv = scores.get(pk, 0)
        if isinstance(pv, (int, float)):
            pct = int(pv * 100)
            bar_color = "#2ecc71" if pv >= 0.6 else "#f39c12" if pv >= 0.4 else "#e74c3c"
            probe_bars += f"""
            <div style="margin-bottom:6px;">
                <div style="font-size:10px;opacity:0.8;">{pk.title()}</div>
                <div style="background:rgba(255,255,255,0.2);border-radius:4px;height:6px;">
                    <div style="background:{bar_color};height:100%;width:{pct}%;border-radius:4px;"></div>
                </div>
            </div>
            """

    return f"""
    <!DOCTYPE html>
    <html><head><meta charset="utf-8">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', -apple-system, sans-serif; }}
    </style>
    </head><body>
    <div style="width:800px; background:{p['bg']}; color:{p['text']};
                padding:40px; border-radius:16px; position:relative;">

        <!-- Badge -->
        <div style="position:absolute; top:20px; right:20px;
                    background:{badge_color}; color:white; width:50px; height:50px;
                    border-radius:50%; display:flex; align-items:center; justify-content:center;
                    font-size:20px; font-weight:bold; box-shadow:0 4px 15px rgba(0,0,0,0.3);">
            {badge}
        </div>

        <!-- Header -->
        <div style="font-size:10px; text-transform:uppercase; letter-spacing:3px; opacity:0.8;">
            {method_type} &bull; {novelty} &bull; {strategy}
        </div>
        <h1 style="font-size:28px; margin:12px 0 8px 0; line-height:1.3; max-width:680px; color:{p['text']};">
            {title}
        </h1>
        <div style="font-size:12px; opacity:0.7; margin-bottom:24px;">
            Quality Score: {q:.2f} | Generated by IdeaGraph
            {f' | Topic: {topic[:50]}' if topic else ''}
        </div>

        <!-- Content grid -->
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:20px;">

            <!-- Left column -->
            <div>
                <div style="background:rgba(255,255,255,0.15); border-radius:10px; padding:16px; margin-bottom:16px;">
                    <div style="font-size:11px; text-transform:uppercase; letter-spacing:1px; opacity:0.7; margin-bottom:6px;">
                        Motivation
                    </div>
                    <div style="font-size:13px; line-height:1.6;">{motivation}</div>
                </div>

                <div style="background:rgba(255,255,255,0.15); border-radius:10px; padding:16px;">
                    <div style="font-size:11px; text-transform:uppercase; letter-spacing:1px; opacity:0.7; margin-bottom:6px;">
                        Hypothesis
                    </div>
                    <div style="font-size:13px; line-height:1.6;">{hypothesis}</div>
                </div>
            </div>

            <!-- Right column -->
            <div>
                <div style="background:rgba(255,255,255,0.15); border-radius:10px; padding:16px; margin-bottom:16px;">
                    <div style="font-size:11px; text-transform:uppercase; letter-spacing:1px; opacity:0.7; margin-bottom:6px;">
                        Method
                    </div>
                    <div style="font-size:13px; line-height:1.6;">{method}</div>
                </div>

                <div style="background:rgba(255,255,255,0.15); border-radius:10px; padding:16px;">
                    <div style="font-size:11px; text-transform:uppercase; letter-spacing:1px; opacity:0.7; margin-bottom:6px;">
                        Expected Outcome
                    </div>
                    <div style="font-size:13px; line-height:1.6;">{outcome}</div>
                </div>
            </div>
        </div>

        <!-- Probe scores -->
        <div style="margin-top:20px; background:rgba(0,0,0,0.15); border-radius:10px; padding:14px;">
            <div style="font-size:10px; text-transform:uppercase; letter-spacing:1px; opacity:0.7; margin-bottom:8px;">
                Quality Probes
            </div>
            <div style="display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:10px;">
                {probe_bars}
            </div>
        </div>

        <!-- Footer -->
        <div style="margin-top:20px; text-align:center; font-size:10px; opacity:0.5;">
            Generated by IdeaGraph &mdash; AI-Powered Research Ideation Platform &mdash; {datetime.now().strftime('%Y-%m-%d')}
        </div>
    </div>
    </body></html>
    """


# ═══════════════════════════════════════════════════════════════════════════
# 2. RESEARCH TREND PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════

class TrendPredictor:
    """
    Predict hot research topics based on idea generation patterns,
    quality trends, and community engagement.

    Signals:
      - Topic frequency: how often a topic is explored
      - Quality velocity: is quality improving over time for this topic?
      - Community engagement: views + likes on shared ideas
      - Novelty index: ratio of "substantial" novelty ideas
    """

    def predict_trends(self, user_results: List[Dict] = None,
                       shared_ideas: List[Dict] = None) -> List[Dict[str, Any]]:
        """Predict top trending topics with confidence scores."""
        topic_signals: Dict[str, Dict[str, float]] = {}

        # Signal 1: Topic frequency from user results
        if user_results:
            for r in user_results:
                topic = r.get("topic", "")[:50]
                if not topic:
                    continue
                if topic not in topic_signals:
                    topic_signals[topic] = {"frequency": 0, "quality": 0, "engagement": 0, "novelty": 0, "count": 0}
                topic_signals[topic]["frequency"] += 1
                topic_signals[topic]["count"] += 1
                topic_signals[topic]["quality"] += r.get("coverage", 0)

        # Signal 2: Community engagement from shared ideas
        if shared_ideas:
            for item in shared_ideas:
                idea = item.get("idea", {})
                topic = item.get("topic", "")[:50]
                if not topic:
                    continue
                if topic not in topic_signals:
                    topic_signals[topic] = {"frequency": 0, "quality": 0, "engagement": 0, "novelty": 0, "count": 0}
                views = item.get("views", 0)
                likes = item.get("likes", 0)
                topic_signals[topic]["engagement"] += views + likes * 3
                q = idea.get("quality_score", 0)
                topic_signals[topic]["quality"] += q
                topic_signals[topic]["count"] += 1
                if idea.get("novelty_level") == "substantial":
                    topic_signals[topic]["novelty"] += 1

        # Compute trend score
        trends = []
        for topic, signals in topic_signals.items():
            count = max(signals["count"], 1)
            avg_quality = signals["quality"] / count
            trend_score = (
                signals["frequency"] * 0.25 +
                avg_quality * 0.30 +
                min(signals["engagement"] / 100, 1.0) * 0.25 +
                min(signals["novelty"] / 3, 1.0) * 0.20
            )

            # Predict direction
            if trend_score > 0.6:
                direction = "rising"
                prediction = "Hot in 6 months"
            elif trend_score > 0.4:
                direction = "stable"
                prediction = "Steady interest"
            else:
                direction = "emerging"
                prediction = "Early stage, watch closely"

            trends.append({
                "topic": topic,
                "trend_score": round(trend_score, 3),
                "direction": direction,
                "prediction": prediction,
                "signals": {
                    "frequency": signals["frequency"],
                    "avg_quality": round(avg_quality, 3),
                    "engagement": signals["engagement"],
                    "novelty_count": signals["novelty"],
                },
            })

        trends.sort(key=lambda x: x["trend_score"], reverse=True)
        return trends


# ═══════════════════════════════════════════════════════════════════════════
# 3. IDEA MASHUP GENERATOR
# ═══════════════════════════════════════════════════════════════════════════

class IdeaMashupGenerator(BaseAgent):
    """
    Combine 2 ideas into a novel hybrid using LLM-powered synthesis.
    The mashup preserves the best elements of each parent idea while
    creating something genuinely new.
    """

    def __init__(self):
        super().__init__(temperature=0.8)

    def mashup(self, idea_a: Dict, idea_b: Dict) -> Dict[str, Any]:
        """Generate a mashup of two ideas. Returns a new hybrid idea dict."""
        system = (
            "You are a creative research synthesizer. Given two research ideas, "
            "create a NOVEL HYBRID that combines the best elements of both into "
            "something genuinely new and more powerful than either alone.\n\n"
            "The hybrid should:\n"
            "- Take the strongest method from one idea and the strongest problem from the other\n"
            "- Create a new hypothesis that neither idea alone would suggest\n"
            "- Be more than just 'do both' — find a creative synergy\n\n"
            "Return ONLY valid JSON with these keys: "
            "title, motivation, method, hypothesis, resources, expected_outcome, "
            "risk_assessment, synergy_explanation, parent_ideas"
        )

        user = (
            f"IDEA A:\n"
            f"  Title: {idea_a.get('title', '?')}\n"
            f"  Method: {idea_a.get('method', '?')[:200]}\n"
            f"  Hypothesis: {idea_a.get('hypothesis', '?')[:150]}\n"
            f"  Quality: {idea_a.get('quality_score', 0):.2f}\n\n"
            f"IDEA B:\n"
            f"  Title: {idea_b.get('title', '?')}\n"
            f"  Method: {idea_b.get('method', '?')[:200]}\n"
            f"  Hypothesis: {idea_b.get('hypothesis', '?')[:150]}\n"
            f"  Quality: {idea_b.get('quality_score', 0):.2f}\n\n"
            f"Create a novel HYBRID that creatively combines both."
        )

        result = self._call_json(system, user, max_tokens=1024)
        if result:
            result["_parent_a"] = idea_a.get("title", "?")[:50]
            result["_parent_b"] = idea_b.get("title", "?")[:50]
            result["_mashup"] = True
            # Estimate quality as average of parents + novelty bonus
            qa = idea_a.get("quality_score", 0.5)
            qb = idea_b.get("quality_score", 0.5)
            result["quality_score"] = min(1.0, (qa + qb) / 2 + 0.1)
        return result or {}


# ═══════════════════════════════════════════════════════════════════════════
# 4. COLLABORATIVE WORKSPACE
# ═══════════════════════════════════════════════════════════════════════════

def _ensure_collab_tables():
    """Create collaboration tables if they don't exist."""
    if not db:
        return
    try:
        with db._lock:
            conn = db._get_conn()
            try:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS workspaces (
                        id          INTEGER PRIMARY KEY AUTOINCREMENT,
                        name        TEXT    NOT NULL,
                        topic       TEXT    NOT NULL DEFAULT '',
                        owner_id    INTEGER NOT NULL,
                        invite_code TEXT    NOT NULL UNIQUE,
                        created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
                        FOREIGN KEY (owner_id) REFERENCES users(id) ON DELETE CASCADE
                    );
                    CREATE TABLE IF NOT EXISTS workspace_members (
                        id           INTEGER PRIMARY KEY AUTOINCREMENT,
                        workspace_id INTEGER NOT NULL,
                        user_id      INTEGER NOT NULL,
                        role         TEXT    NOT NULL DEFAULT 'member',
                        joined_at    TEXT    NOT NULL DEFAULT (datetime('now')),
                        FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE,
                        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                        UNIQUE(workspace_id, user_id)
                    );
                    CREATE TABLE IF NOT EXISTS workspace_ideas (
                        id           INTEGER PRIMARY KEY AUTOINCREMENT,
                        workspace_id INTEGER NOT NULL,
                        user_id      INTEGER NOT NULL,
                        idea_json    TEXT    NOT NULL,
                        comment      TEXT    NOT NULL DEFAULT '',
                        votes        INTEGER NOT NULL DEFAULT 0,
                        created_at   TEXT    NOT NULL DEFAULT (datetime('now')),
                        FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE,
                        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                    );
                    CREATE TABLE IF NOT EXISTS workspace_chat (
                        id           INTEGER PRIMARY KEY AUTOINCREMENT,
                        workspace_id INTEGER NOT NULL,
                        user_id      INTEGER NOT NULL,
                        message      TEXT    NOT NULL,
                        created_at   TEXT    NOT NULL DEFAULT (datetime('now')),
                        FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE,
                        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                    );
                """)
                conn.commit()
            finally:
                conn.close()
    except Exception:
        pass


def create_workspace(owner_id: int, name: str, topic: str = "") -> Dict[str, Any]:
    """Create a new workspace. Returns workspace info with invite code."""
    import secrets
    _ensure_collab_tables()
    invite_code = secrets.token_urlsafe(8)

    with db._lock:
        conn = db._get_conn()
        try:
            cur = conn.execute(
                "INSERT INTO workspaces (name, topic, owner_id, invite_code) VALUES (?, ?, ?, ?)",
                (name, topic, owner_id, invite_code),
            )
            ws_id = cur.lastrowid
            # Add owner as admin member
            conn.execute(
                "INSERT INTO workspace_members (workspace_id, user_id, role) VALUES (?, ?, 'admin')",
                (ws_id, owner_id),
            )
            conn.commit()
            return {"id": ws_id, "name": name, "topic": topic, "invite_code": invite_code}
        finally:
            conn.close()


def join_workspace(user_id: int, invite_code: str) -> Optional[Dict]:
    """Join a workspace via invite code."""
    _ensure_collab_tables()
    with db._lock:
        conn = db._get_conn()
        try:
            ws = conn.execute(
                "SELECT * FROM workspaces WHERE invite_code = ?", (invite_code,),
            ).fetchone()
            if not ws:
                return None
            try:
                conn.execute(
                    "INSERT INTO workspace_members (workspace_id, user_id) VALUES (?, ?)",
                    (ws["id"], user_id),
                )
                conn.commit()
            except sqlite3.IntegrityError:
                pass  # already a member
            return dict(ws)
        finally:
            conn.close()


def get_user_workspaces(user_id: int) -> List[Dict]:
    """Get all workspaces a user is a member of."""
    _ensure_collab_tables()
    with db._lock:
        conn = db._get_conn()
        try:
            rows = conn.execute(
                """SELECT w.*, wm.role FROM workspaces w
                   JOIN workspace_members wm ON wm.workspace_id = w.id
                   WHERE wm.user_id = ? ORDER BY w.created_at DESC""",
                (user_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


def add_idea_to_workspace(workspace_id: int, user_id: int, idea: Dict, comment: str = "") -> int:
    """Add an idea to a workspace. Returns idea ID."""
    _ensure_collab_tables()
    with db._lock:
        conn = db._get_conn()
        try:
            cur = conn.execute(
                "INSERT INTO workspace_ideas (workspace_id, user_id, idea_json, comment) VALUES (?, ?, ?, ?)",
                (workspace_id, user_id, json.dumps(idea, default=str), comment),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()


def get_workspace_ideas(workspace_id: int) -> List[Dict]:
    _ensure_collab_tables()
    with db._lock:
        conn = db._get_conn()
        try:
            rows = conn.execute(
                """SELECT wi.*, u.username FROM workspace_ideas wi
                   JOIN users u ON u.id = wi.user_id
                   WHERE wi.workspace_id = ? ORDER BY wi.votes DESC, wi.created_at DESC""",
                (workspace_id,),
            ).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                try:
                    d["idea"] = json.loads(d.get("idea_json", "{}"))
                except Exception:
                    d["idea"] = {}
                result.append(d)
            return result
        finally:
            conn.close()


def vote_workspace_idea(idea_id: int) -> int:
    _ensure_collab_tables()
    with db._lock:
        conn = db._get_conn()
        try:
            conn.execute("UPDATE workspace_ideas SET votes = votes + 1 WHERE id = ?", (idea_id,))
            conn.commit()
            row = conn.execute("SELECT votes FROM workspace_ideas WHERE id = ?", (idea_id,)).fetchone()
            return row["votes"] if row else 0
        finally:
            conn.close()


def post_workspace_chat(workspace_id: int, user_id: int, message: str) -> None:
    _ensure_collab_tables()
    with db._lock:
        conn = db._get_conn()
        try:
            conn.execute(
                "INSERT INTO workspace_chat (workspace_id, user_id, message) VALUES (?, ?, ?)",
                (workspace_id, user_id, message),
            )
            conn.commit()
        finally:
            conn.close()


def get_workspace_chat(workspace_id: int, limit: int = 30) -> List[Dict]:
    _ensure_collab_tables()
    with db._lock:
        conn = db._get_conn()
        try:
            rows = conn.execute(
                """SELECT wc.*, u.username FROM workspace_chat wc
                   JOIN users u ON u.id = wc.user_id
                   WHERE wc.workspace_id = ? ORDER BY wc.created_at DESC LIMIT ?""",
                (workspace_id, limit),
            ).fetchall()
            return [dict(r) for r in reversed(rows)]
        finally:
            conn.close()


def get_workspace_members(workspace_id: int) -> List[Dict]:
    _ensure_collab_tables()
    with db._lock:
        conn = db._get_conn()
        try:
            rows = conn.execute(
                """SELECT u.id, u.username, wm.role, wm.joined_at FROM workspace_members wm
                   JOIN users u ON u.id = wm.user_id WHERE wm.workspace_id = ?""",
                (workspace_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()
