"""
engagement.py - Retention & engagement systems for IdeaGraph.

Implements the "Hooked" model psychology:
  - TRIGGER: Daily streaks, notifications
  - ACTION: One-click daily pick view
  - VARIABLE REWARD: Random achievements, surprise XP bonuses
  - INVESTMENT: Stored streaks, XP levels, bookmarks

Components:
  1. Streak system (daily login bonus, 7/30/100 day milestones)
  2. XP + Level system (gamification)
  3. Achievement badges (28 unlockable badges)
  4. Daily personalized idea pick (reason to return every day)
  5. Social activity feed (FOMO)
"""

from __future__ import annotations

import json
import random
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import db

# ═══════════════════════════════════════════════════════════════════════════
# XP & LEVEL SYSTEM
# ═══════════════════════════════════════════════════════════════════════════

XP_REWARDS = {
    "run_pipeline": 50,
    "generate_quality_idea": 10,
    "bookmark_idea": 5,
    "share_idea": 20,
    "receive_like": 3,
    "daily_login": 10,
    "streak_day": 15,
    "complete_onboarding": 100,
    "unlock_achievement": 25,
    "write_note": 5,
}


def xp_required_for_level(level: int) -> int:
    """XP curve: exponential growth, level 1 = 0, level 2 = 100, level 5 = ~800."""
    if level <= 1:
        return 0
    return int(100 * (level - 1) ** 1.5)


# Precompute the XP-threshold table once at import time. With the current
# curve, level 200 needs ~280 k XP, well above any realistic player; cap at
# 200 so `level_from_xp` is a simple bisect lookup instead of a Python loop
# that re-evaluates `(level - 1) ** 1.5` on every iteration.
_MAX_PRECOMPUTED_LEVEL = 200
_LEVEL_THRESHOLDS: List[int] = [
    xp_required_for_level(level) for level in range(1, _MAX_PRECOMPUTED_LEVEL + 1)
]


def level_from_xp(xp: int) -> int:
    """Compute current level from total XP. O(log N) bisect over precomputed thresholds."""
    if xp <= 0:
        return 1
    import bisect
    # bisect_right gives the count of thresholds <= xp; that's the level.
    idx = bisect.bisect_right(_LEVEL_THRESHOLDS, xp)
    if idx >= _MAX_PRECOMPUTED_LEVEL:
        # Fallback for the (essentially impossible) above-cap case.
        level = _MAX_PRECOMPUTED_LEVEL
        while xp_required_for_level(level + 1) <= xp:
            level += 1
        return level
    return idx


def xp_progress_in_level(xp: int) -> Tuple[int, int, float]:
    """Return (xp_in_level, xp_needed_for_next, progress_pct)."""
    level = level_from_xp(xp)
    # Use the precomputed table directly to avoid two more pow() calls.
    if 1 <= level <= _MAX_PRECOMPUTED_LEVEL:
        current_threshold = _LEVEL_THRESHOLDS[level - 1]
    else:
        current_threshold = xp_required_for_level(level)
    if 1 <= level < _MAX_PRECOMPUTED_LEVEL:
        next_threshold = _LEVEL_THRESHOLDS[level]
    else:
        next_threshold = xp_required_for_level(level + 1)
    xp_in = xp - current_threshold
    xp_needed = next_threshold - current_threshold
    pct = xp_in / max(xp_needed, 1)
    return xp_in, xp_needed, pct


# ═══════════════════════════════════════════════════════════════════════════
# STREAK SYSTEM
# ═══════════════════════════════════════════════════════════════════════════

def _ensure_stats(user_id: int) -> Dict[str, Any]:
    """Ensure user_stats row exists. Returns the row."""
    with db._lock:
        conn = db._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM user_stats WHERE user_id = ?", (user_id,),
            ).fetchone()
            if not row:
                conn.execute(
                    "INSERT INTO user_stats (user_id) VALUES (?)", (user_id,),
                )
                conn.commit()
                row = conn.execute(
                    "SELECT * FROM user_stats WHERE user_id = ?", (user_id,),
                ).fetchone()
            return dict(row) if row else {}
        finally:
            conn.close()


def check_in_user(user_id: int) -> Dict[str, Any]:
    """
    Call on every login/action. Updates streak, awards XP.
    Returns {'streak': N, 'xp_earned': M, 'leveled_up': bool, 'new_level': X}
    """
    stats = _ensure_stats(user_id)
    # Snapshot now() once and derive both date strings from the same instant.
    _now = datetime.now()
    today = _now.strftime("%Y-%m-%d")
    yesterday = (_now - timedelta(days=1)).strftime("%Y-%m-%d")
    last_active = stats.get("last_active_date", "")

    if last_active == today:
        # Already checked in today
        return {"streak": stats["current_streak"], "xp_earned": 0, "leveled_up": False}

    # Compute new streak
    if last_active == yesterday:
        new_streak = stats["current_streak"] + 1
    else:
        new_streak = 1

    longest = max(stats.get("longest_streak", 0), new_streak)
    xp_earned = XP_REWARDS["daily_login"] + (XP_REWARDS["streak_day"] if new_streak > 1 else 0)

    # Streak milestone bonuses
    streak_bonus_msg = ""
    if new_streak == 7:
        xp_earned += 100
        streak_bonus_msg = "🔥 7-day streak! +100 XP bonus"
    elif new_streak == 30:
        xp_earned += 500
        streak_bonus_msg = "🎯 30-day streak! +500 XP bonus"
    elif new_streak == 100:
        xp_earned += 2000
        streak_bonus_msg = "💎 100-day LEGEND! +2000 XP bonus"

    old_xp = stats.get("xp", 0)
    new_xp = old_xp + xp_earned
    old_level = level_from_xp(old_xp)
    new_level = level_from_xp(new_xp)
    leveled_up = new_level > old_level

    with db._lock:
        conn = db._get_conn()
        try:
            conn.execute(
                """UPDATE user_stats SET
                   current_streak = ?, longest_streak = ?,
                   last_active_date = ?, xp = ?, level = ?
                   WHERE user_id = ?""",
                (new_streak, longest, today, new_xp, new_level, user_id),
            )
            conn.commit()
        finally:
            conn.close()

    # Check for streak achievements
    if new_streak == 7:
        unlock_achievement(user_id, "streak_7")
    elif new_streak == 30:
        unlock_achievement(user_id, "streak_30")
    elif new_streak == 100:
        unlock_achievement(user_id, "streak_100")

    return {
        "streak": new_streak,
        "xp_earned": xp_earned,
        "leveled_up": leveled_up,
        "new_level": new_level,
        "bonus_msg": streak_bonus_msg,
    }


def award_xp(user_id: int, action: str, multiplier: int = 1) -> int:
    """Award XP for an action. Returns new total XP."""
    _ensure_stats(user_id)
    amount = XP_REWARDS.get(action, 0) * multiplier
    if amount == 0:
        return 0
    with db._lock:
        conn = db._get_conn()
        try:
            conn.execute(
                "UPDATE user_stats SET xp = xp + ?, level = ? WHERE user_id = ?",
                (amount, 0, user_id),  # level will be recalculated
            )
            # Recompute level
            row = conn.execute(
                "SELECT xp FROM user_stats WHERE user_id = ?", (user_id,),
            ).fetchone()
            if row:
                new_level = level_from_xp(row["xp"])
                conn.execute(
                    "UPDATE user_stats SET level = ? WHERE user_id = ?",
                    (new_level, user_id),
                )
            conn.commit()
            return row["xp"] if row else 0
        finally:
            conn.close()


def get_user_stats(user_id: int) -> Dict[str, Any]:
    """Get complete user stats for display."""
    stats = _ensure_stats(user_id)
    xp = stats.get("xp", 0)
    xp_in, xp_needed, pct = xp_progress_in_level(xp)
    return {
        **stats,
        "xp_in_level": xp_in,
        "xp_needed": xp_needed,
        "level_progress": pct,
    }


# ═══════════════════════════════════════════════════════════════════════════
# ACHIEVEMENT SYSTEM
# ═══════════════════════════════════════════════════════════════════════════

ACHIEVEMENTS = {
    # Onboarding
    "first_run": {"name": "First Steps", "emoji": "👶", "desc": "Complete your first pipeline run", "xp": 50},
    "first_idea": {"name": "Idea Born", "emoji": "💡", "desc": "Generate your first idea", "xp": 25},
    "first_bookmark": {"name": "Keeper", "emoji": "🔖", "desc": "Bookmark your first idea", "xp": 25},
    "first_share": {"name": "Spreader", "emoji": "📣", "desc": "Share your first idea publicly", "xp": 50},
    "first_export": {"name": "Exporter", "emoji": "📤", "desc": "Download your first PDF", "xp": 25},
    "complete_profile": {"name": "Identified", "emoji": "🪪", "desc": "Complete your profile", "xp": 25},

    # Volume
    "ten_ideas": {"name": "Prolific", "emoji": "✨", "desc": "Generate 10 ideas", "xp": 50},
    "hundred_ideas": {"name": "Idea Factory", "emoji": "🏭", "desc": "Generate 100 ideas", "xp": 200},
    "thousand_ideas": {"name": "Legend", "emoji": "👑", "desc": "Generate 1,000 ideas", "xp": 1000},

    # Quality
    "high_quality": {"name": "Quality Hunter", "emoji": "🎯", "desc": "Generate an idea with quality > 0.8", "xp": 100},
    "perfect_score": {"name": "Perfection", "emoji": "💯", "desc": "Generate an idea with quality > 0.95", "xp": 500},

    # Streaks
    "streak_7": {"name": "Dedicated", "emoji": "🔥", "desc": "7-day login streak", "xp": 100},
    "streak_30": {"name": "Committed", "emoji": "🎯", "desc": "30-day login streak", "xp": 500},
    "streak_100": {"name": "Obsessed", "emoji": "💎", "desc": "100-day login streak", "xp": 2000},

    # Social
    "first_follower": {"name": "Attractive", "emoji": "🌟", "desc": "Get your first follower", "xp": 50},
    "ten_followers": {"name": "Influencer", "emoji": "💫", "desc": "Get 10 followers", "xp": 200},
    "viral_idea": {"name": "Viral", "emoji": "🚀", "desc": "An idea with 100+ views", "xp": 300},

    # Diversity
    "all_methods": {"name": "Omnivore", "emoji": "🌈", "desc": "Use all 7 methodology types", "xp": 150},
    "cross_domain": {"name": "Bridge Builder", "emoji": "🌉", "desc": "Generate 5 cross-domain ideas", "xp": 150},

    # Exploration
    "five_topics": {"name": "Explorer", "emoji": "🗺️", "desc": "Explore 5 different topics", "xp": 100},
    "twenty_topics": {"name": "Polymath", "emoji": "🎓", "desc": "Explore 20 different topics", "xp": 500},

    # Power user
    "night_owl": {"name": "Night Owl", "emoji": "🦉", "desc": "Run pipeline between midnight and 5am", "xp": 50},
    "early_bird": {"name": "Early Bird", "emoji": "🐦", "desc": "Run pipeline before 6am", "xp": 50},
    "weekend_warrior": {"name": "Weekend Warrior", "emoji": "⚔️", "desc": "Run pipeline on Sunday", "xp": 50},
    "speed_runner": {"name": "Speed Runner", "emoji": "⚡", "desc": "Complete pipeline in under 2 minutes", "xp": 100},

    # Subscription
    "first_upgrade": {"name": "Believer", "emoji": "⭐", "desc": "Upgrade to Pro tier", "xp": 500},
    "team_player": {"name": "Team Player", "emoji": "👥", "desc": "Join or create a team", "xp": 300},
    "big_spender": {"name": "Big Spender", "emoji": "💰", "desc": "Upgrade to Enterprise", "xp": 1000},
}


def unlock_achievement(user_id: int, badge_key: str) -> Optional[Dict[str, Any]]:
    """
    Unlock an achievement. Returns badge info if newly unlocked, None if already had it.
    """
    if badge_key not in ACHIEVEMENTS:
        return None

    try:
        with db._lock:
            conn = db._get_conn()
            try:
                conn.execute(
                    "INSERT INTO achievements (user_id, badge_key) VALUES (?, ?)",
                    (user_id, badge_key),
                )
                conn.commit()
                # Award XP for the achievement
                xp = ACHIEVEMENTS[badge_key].get("xp", 25)
                conn.execute(
                    "UPDATE user_stats SET xp = xp + ? WHERE user_id = ?",
                    (xp, user_id),
                )
                conn.commit()
                return {**ACHIEVEMENTS[badge_key], "key": badge_key, "new": True}
            except sqlite3.IntegrityError:
                return None  # already unlocked
            finally:
                conn.close()
    except Exception:
        return None


def get_user_achievements(user_id: int) -> List[Dict[str, Any]]:
    """Get all unlocked achievements for a user."""
    with db._lock:
        conn = db._get_conn()
        try:
            rows = conn.execute(
                "SELECT badge_key, unlocked_at FROM achievements WHERE user_id = ? ORDER BY unlocked_at DESC",
                (user_id,),
            ).fetchall()
            result = []
            for r in rows:
                bk = r["badge_key"]
                if bk in ACHIEVEMENTS:
                    result.append({
                        **ACHIEVEMENTS[bk],
                        "key": bk,
                        "unlocked_at": r["unlocked_at"],
                    })
            return result
        finally:
            conn.close()


def check_achievements_after_run(user_id: int, results: Dict[str, Any]) -> List[Dict]:
    """Check for newly unlocked achievements after a pipeline run."""
    unlocked = []
    stats = _ensure_stats(user_id)
    ideas = results.get("ideas", [])
    total_ideas = stats.get("total_ideas", 0) + len(ideas)

    # Update totals
    with db._lock:
        conn = db._get_conn()
        try:
            conn.execute(
                "UPDATE user_stats SET total_runs = total_runs + 1, total_ideas = total_ideas + ? WHERE user_id = ?",
                (len(ideas), user_id),
            )
            conn.commit()
        finally:
            conn.close()

    # First run
    if stats.get("total_runs", 0) == 0 and ideas:
        a = unlock_achievement(user_id, "first_run")
        if a: unlocked.append(a)
        a = unlock_achievement(user_id, "first_idea")
        if a: unlocked.append(a)

    # Volume milestones
    if total_ideas >= 10 and stats.get("total_ideas", 0) < 10:
        a = unlock_achievement(user_id, "ten_ideas")
        if a: unlocked.append(a)
    if total_ideas >= 100 and stats.get("total_ideas", 0) < 100:
        a = unlock_achievement(user_id, "hundred_ideas")
        if a: unlocked.append(a)
    if total_ideas >= 1000 and stats.get("total_ideas", 0) < 1000:
        a = unlock_achievement(user_id, "thousand_ideas")
        if a: unlocked.append(a)

    # Quality achievements
    max_q = max((i.get("quality_score", 0) for i in ideas), default=0)
    if max_q >= 0.8:
        a = unlock_achievement(user_id, "high_quality")
        if a: unlocked.append(a)
    if max_q >= 0.95:
        a = unlock_achievement(user_id, "perfect_score")
        if a: unlocked.append(a)

    # Time-based achievements (snapshot now() once for both reads)
    _now = datetime.now()
    hour = _now.hour
    weekday = _now.weekday()
    if 0 <= hour < 5:
        a = unlock_achievement(user_id, "night_owl")
        if a: unlocked.append(a)
    elif hour < 6:
        a = unlock_achievement(user_id, "early_bird")
        if a: unlocked.append(a)
    if weekday == 6:
        a = unlock_achievement(user_id, "weekend_warrior")
        if a: unlocked.append(a)

    # Speed run
    elapsed = results.get("total_elapsed", results.get("stats", {}).get("elapsed_seconds", 999))
    if elapsed and elapsed < 120 and ideas:
        a = unlock_achievement(user_id, "speed_runner")
        if a: unlocked.append(a)

    # Diversity (all methodology types)
    method_types = set(i.get("methodology_type", "") for i in ideas)
    if len(method_types) >= 7:
        a = unlock_achievement(user_id, "all_methods")
        if a: unlocked.append(a)

    return unlocked


# ═══════════════════════════════════════════════════════════════════════════
# DAILY PERSONALIZED IDEA PICK
# ═══════════════════════════════════════════════════════════════════════════

def generate_daily_pick(user_id: int) -> Optional[Dict[str, Any]]:
    """
    Generate or retrieve today's personalized idea pick for a user.
    Picks a random top-quality idea from user's bookmarks or previous runs.
    This gives users a reason to return every day.
    """
    today = datetime.now().strftime("%Y-%m-%d")

    # Check if already generated today
    with db._lock:
        conn = db._get_conn()
        try:
            existing = conn.execute(
                "SELECT * FROM daily_picks WHERE user_id = ? AND date = ?",
                (user_id, today),
            ).fetchone()
            if existing:
                return json.loads(existing["idea_json"])
        finally:
            conn.close()

    # Build candidate pool from user's best ideas
    candidates = []

    # Option 1: From bookmarks
    bookmarks = db.get_bookmarks(user_id)
    for bk in bookmarks:
        idea = bk.get("idea", {})
        if idea.get("title"):
            candidates.append(idea)

    # Option 2: From all user's saved results
    results_list = db.get_user_results(user_id)
    for r in results_list[:5]:
        full = db.load_result(r["id"], user_id)
        if full:
            ideas = full.get("ideas", [])
            high_q = [i for i in ideas if i.get("quality_score", 0) >= 0.5]
            candidates.extend(high_q[:5])

    # Option 3: From community top ideas (fallback for new users)
    if not candidates:
        top_shared = db.get_top_shared_ideas(limit=10)
        candidates = [item.get("idea", {}) for item in top_shared if item.get("idea")]

    if not candidates:
        return None

    # Pick randomly from top candidates (sorted by quality)
    candidates.sort(key=lambda x: x.get("quality_score", 0), reverse=True)
    pool = candidates[:10]
    pick = random.choice(pool)

    # Store for today
    with db._lock:
        conn = db._get_conn()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO daily_picks (user_id, idea_json, date) VALUES (?, ?, ?)",
                (user_id, json.dumps(pick, default=str), today),
            )
            conn.commit()
        finally:
            conn.close()

    return pick


# ═══════════════════════════════════════════════════════════════════════════
# SOCIAL FOLLOW SYSTEM
# ═══════════════════════════════════════════════════════════════════════════

def follow_user(follower_id: int, followed_id: int) -> bool:
    """Follow another user. Returns True if newly followed."""
    if follower_id == followed_id:
        return False
    try:
        with db._lock:
            conn = db._get_conn()
            try:
                conn.execute(
                    "INSERT INTO follows (follower_id, followed_id) VALUES (?, ?)",
                    (follower_id, followed_id),
                )
                conn.commit()
                # Check for achievements
                count = conn.execute(
                    "SELECT COUNT(*) as c FROM follows WHERE followed_id = ?",
                    (followed_id,),
                ).fetchone()
                followers = count["c"] if count else 0
                if followers == 1:
                    unlock_achievement(followed_id, "first_follower")
                elif followers == 10:
                    unlock_achievement(followed_id, "ten_followers")
                return True
            except sqlite3.IntegrityError:
                return False
            finally:
                conn.close()
    except Exception:
        return False


def unfollow_user(follower_id: int, followed_id: int) -> None:
    with db._lock:
        conn = db._get_conn()
        try:
            conn.execute(
                "DELETE FROM follows WHERE follower_id = ? AND followed_id = ?",
                (follower_id, followed_id),
            )
            conn.commit()
        finally:
            conn.close()


def get_followers(user_id: int) -> List[Dict]:
    with db._lock:
        conn = db._get_conn()
        try:
            rows = conn.execute(
                """SELECT u.id, u.username FROM follows f
                   JOIN users u ON u.id = f.follower_id
                   WHERE f.followed_id = ?""",
                (user_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


def get_following(user_id: int) -> List[Dict]:
    with db._lock:
        conn = db._get_conn()
        try:
            rows = conn.execute(
                """SELECT u.id, u.username FROM follows f
                   JOIN users u ON u.id = f.followed_id
                   WHERE f.follower_id = ?""",
                (user_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


# ═══════════════════════════════════════════════════════════════════════════
# ACTIVITY FEED
# ═══════════════════════════════════════════════════════════════════════════

def post_activity(user_id: int, activity_type: str, content: str, metadata: Dict = None) -> None:
    """Post an activity to the feed."""
    with db._lock:
        conn = db._get_conn()
        try:
            conn.execute(
                "INSERT INTO activity_feed (user_id, activity_type, content, metadata) VALUES (?, ?, ?, ?)",
                (user_id, activity_type, content, json.dumps(metadata or {}, default=str)),
            )
            conn.commit()
        finally:
            conn.close()


def get_activity_feed(user_id: int, following_only: bool = False, limit: int = 20) -> List[Dict]:
    """Get activity feed (global or followed-only)."""
    with db._lock:
        conn = db._get_conn()
        try:
            if following_only:
                rows = conn.execute(
                    """SELECT a.*, u.username FROM activity_feed a
                       JOIN users u ON u.id = a.user_id
                       WHERE a.user_id IN (SELECT followed_id FROM follows WHERE follower_id = ?)
                       OR a.user_id = ?
                       ORDER BY a.created_at DESC LIMIT ?""",
                    (user_id, user_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT a.*, u.username FROM activity_feed a
                       JOIN users u ON u.id = a.user_id
                       ORDER BY a.created_at DESC LIMIT ?""",
                    (limit,),
                ).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                try:
                    d["metadata"] = json.loads(d.get("metadata") or "{}")
                except Exception:
                    d["metadata"] = {}
                result.append(d)
            return result
        finally:
            conn.close()


# ═══════════════════════════════════════════════════════════════════════════
# ONBOARDING
# ═══════════════════════════════════════════════════════════════════════════

ONBOARDING_STEPS = [
    {"title": "Welcome!", "desc": "IdeaGraph generates novel research ideas using AI + real literature.", "emoji": "👋"},
    {"title": "Pick a Topic", "desc": "Enter any research topic in the sidebar. The more specific, the better.", "emoji": "🎯"},
    {"title": "Run Pipeline", "desc": "Click 'Run Automated Scientist'. Takes 2-10 minutes depending on budget.", "emoji": "⚡"},
    {"title": "Explore Ideas", "desc": "Your ideas appear ranked by quality. Expand each to see the full details.", "emoji": "💡"},
    {"title": "Bookmark & Share", "desc": "Bookmark favorites, share public links, export to PDF/Zotero/Notion.", "emoji": "🔖"},
    {"title": "Level Up", "desc": "Earn XP for every action. Unlock achievements. Maintain daily streaks!", "emoji": "🏆"},
]


def get_onboarding_state(user_id: int) -> Dict[str, Any]:
    with db._lock:
        conn = db._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM onboarding_progress WHERE user_id = ?", (user_id,),
            ).fetchone()
            if not row:
                conn.execute(
                    "INSERT INTO onboarding_progress (user_id) VALUES (?)", (user_id,),
                )
                conn.commit()
                return {"step": 0, "completed": 0, "tutorial_dismissed": 0}
            return dict(row)
        finally:
            conn.close()


def advance_onboarding(user_id: int) -> int:
    state = get_onboarding_state(user_id)
    new_step = state.get("step", 0) + 1
    completed = 1 if new_step >= len(ONBOARDING_STEPS) else 0
    with db._lock:
        conn = db._get_conn()
        try:
            conn.execute(
                "UPDATE onboarding_progress SET step = ?, completed = ? WHERE user_id = ?",
                (new_step, completed, user_id),
            )
            conn.commit()
        finally:
            conn.close()
    if completed:
        unlock_achievement(user_id, "first_run")
        award_xp(user_id, "complete_onboarding")
    return new_step


def dismiss_onboarding(user_id: int) -> None:
    with db._lock:
        conn = db._get_conn()
        try:
            conn.execute(
                "UPDATE onboarding_progress SET tutorial_dismissed = 1 WHERE user_id = ?",
                (user_id,),
            )
            conn.commit()
        finally:
            conn.close()
