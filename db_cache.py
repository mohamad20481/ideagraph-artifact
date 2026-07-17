"""
db_cache.py - Streamlit @st.cache_data wrappers over read-only db functions.

Why this exists
---------------
app.py has 14+ tabs, and tab bodies make direct db.get_*/db.load_* calls at
render time. Every st.rerun() re-executes every tab body, so a single user
action (e.g. clicking a button in one tab) triggers dozens of SQLite reads
across all the other tabs — none of which changed. Layered on top of a
background pipeline thread that also touches the DB, this produced obvious
UI lag even with the thread-local connection fix in db.py.

What this does
--------------
Each wrapper is a thin `@st.cache_data(ttl=60)` around the matching function
in db.py. Streamlit keys cache entries on the argument tuple, so per-user
rows stay isolated automatically. TTL=60s bounds staleness for the case
where nobody calls an explicit invalidator (e.g. data changed in another
session or process).

Invalidation
------------
On every write in app.py, call the matching `invalidate_*` helper so the
reader sees fresh data immediately instead of waiting out the TTL. Each
invalidator uses `.clear()` to wipe the whole function cache — simpler and
safer than per-arg clearing, and writes are rare enough that the re-fetch
cost is negligible.

All wrappers forward unchanged arguments and return values. Drop-in use:

    import db_cache
    results = db_cache.get_user_results(uid)     # instead of db.get_user_results(uid)
    db.save_result(uid, ...)
    db_cache.invalidate_user_results()
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st

import db


# ── Per-user list reads ────────────────────────────────────────────────────

@st.cache_data(ttl=60, show_spinner=False)
def get_user_results(user_id: int) -> List[Dict[str, Any]]:
    return db.get_user_results(user_id)


@st.cache_data(ttl=60, show_spinner=False)
def get_user_debates(user_id: int) -> List[Dict[str, Any]]:
    return db.get_user_debates(user_id)


@st.cache_data(ttl=60, show_spinner=False)
def get_user_papers(user_id: int) -> List[Dict[str, Any]]:
    return db.get_user_papers(user_id)


@st.cache_data(ttl=60, show_spinner=False)
def get_user_scientist_runs(user_id: int) -> List[Dict[str, Any]]:
    return db.get_user_scientist_runs(user_id)


@st.cache_data(ttl=60, show_spinner=False)
def get_all_user_ideas(user_id: int) -> List[Dict[str, Any]]:
    return db.get_all_user_ideas(user_id)


@st.cache_data(ttl=60, show_spinner=False)
def get_user_results_full(user_id: int) -> List[Dict[str, Any]]:
    """Cached batched-load of all user results with full JSON payload merged in."""
    return db.get_user_results_full(user_id)


@st.cache_data(ttl=60, show_spinner=False)
def get_user_subscription(user_id: int) -> Dict[str, Any]:
    return db.get_user_subscription(user_id)


@st.cache_data(ttl=60, show_spinner=False)
def get_bookmarks(
    user_id: int, collection: Optional[str] = None
) -> List[Dict[str, Any]]:
    return db.get_bookmarks(user_id, collection=collection)


@st.cache_data(ttl=60, show_spinner=False)
def get_bookmark_collections(user_id: int) -> List[str]:
    return db.get_bookmark_collections(user_id)


# ── Row-level loaders (keyed by row id + owning user_id) ───────────────────

@st.cache_data(ttl=60, show_spinner=False)
def load_result(result_id: int, user_id: int) -> Optional[Dict[str, Any]]:
    return db.load_result(result_id, user_id)


@st.cache_data(ttl=60, show_spinner=False)
def load_debate(debate_id: int, user_id: int) -> Optional[Dict[str, Any]]:
    return db.load_debate(debate_id, user_id)


@st.cache_data(ttl=60, show_spinner=False)
def load_paper(paper_id: int, user_id: int) -> Optional[str]:
    return db.load_paper(paper_id, user_id)


@st.cache_data(ttl=60, show_spinner=False)
def load_scientist_run(run_id: int, user_id: int) -> Optional[Dict[str, Any]]:
    return db.load_scientist_run(run_id, user_id)


# ── Global (non-user) reads ────────────────────────────────────────────────

@st.cache_data(ttl=60, show_spinner=False)
def get_top_shared_ideas(limit: int = 20) -> List[Dict[str, Any]]:
    return db.get_top_shared_ideas(limit=limit)


@st.cache_data(ttl=60, show_spinner=False)
def get_top_shared_metadata(limit: int = 20) -> List[Dict[str, Any]]:
    """Cached lightweight variant — leaderboard counts without idea_json blob."""
    return db.get_top_shared_metadata(limit=limit)


# ── Invalidators ───────────────────────────────────────────────────────────
# Call the matching helper from app.py right after the corresponding db write.
# Each one clears every wrapper whose data could be affected by that write.

def invalidate_user_results() -> None:
    """Clear after db.save_result / db.delete_result."""
    get_user_results.clear()
    get_user_results_full.clear()
    get_all_user_ideas.clear()
    load_result.clear()


def invalidate_user_debates() -> None:
    """Clear after db.save_debate / db.delete_debate."""
    get_user_debates.clear()
    load_debate.clear()


def invalidate_user_papers() -> None:
    """Clear after db.save_paper / db.delete_paper."""
    get_user_papers.clear()
    load_paper.clear()


def invalidate_user_scientist_runs() -> None:
    """Clear after db.save_scientist_run."""
    get_user_scientist_runs.clear()
    load_scientist_run.clear()


def invalidate_user_bookmarks() -> None:
    """Clear after db.bookmark_idea / db.delete_bookmark."""
    get_bookmarks.clear()
    get_bookmark_collections.clear()


def invalidate_user_subscription() -> None:
    """Clear after subscription changes."""
    get_user_subscription.clear()


def invalidate_shared_ideas() -> None:
    """Clear after db.share_idea / db.unshare_idea."""
    get_top_shared_ideas.clear()
    get_top_shared_metadata.clear()
