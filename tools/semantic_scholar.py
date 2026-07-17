"""
tools/semantic_scholar.py - Thin wrapper around the Semantic Scholar API.

All public functions return lists of paper dicts with keys:
  paperId, title, abstract, year, citationCount

Rate-limit strategy (no API key = 1 req/s public limit):
  - Hard concurrency cap: 1 concurrent request (semaphore)
  - 1.2 s inter-request gap (rate-limit lock)
  - On 429: single short wait (5 s) then give up — never loop forever
  - On network error: 1 retry only
"""

from __future__ import annotations
import threading
import time
import hashlib
import json
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import os
import requests


_BASE = "https://api.semanticscholar.org/graph/v1"
_FIELDS = "paperId,title,abstract,year,citationCount"
_TIMEOUT = 15  # seconds per request

# Persistent HTTP session — reuses TCP connections (keep-alive) and
# pools connections to semanticscholar.org, avoiding per-request DNS
# lookups and TLS handshakes.  Previously each call went through
# requests.get() which creates a throwaway session.
_SESSION = requests.Session()

# Concurrency limit: 1 without API key (public ~1 req/s), 3 with key.
# Initialised properly at module load; no runtime _value mutation.
_HAS_API_KEY = bool(os.getenv("SEMANTIC_SCHOLAR_API_KEY", ""))
_API_SEMAPHORE = threading.Semaphore(3 if _HAS_API_KEY else 1)
# Enforces minimum gap between requests to avoid 429
_RATE_LOCK = threading.Lock()
_LAST_REQUEST_TIME: float = 0.0
_MIN_GAP = 0.5 if _HAS_API_KEY else 1.2  # seconds between requests

# ── Response cache (true LRU via OrderedDict) ────────────────────────────────
# Previously a plain dict — evicted FIFO (insertion order) instead of LRU,
# so frequently-accessed entries got evicted just like stale ones. Now uses
# OrderedDict.move_to_end() on cache hits and popitem(last=False) on eviction.
_CACHE: OrderedDict[str, Any] = OrderedDict()
_CACHE_MAX = 512
_CACHE_LOCK = threading.RLock()


def _cache_key(url: str, params: Dict[str, Any]) -> str:
    raw = url + "\x00" + json.dumps(params, sort_keys=True)
    return hashlib.md5(raw.encode(), usedforsecurity=False).hexdigest()


def clear_cache() -> None:
    """Clear the Semantic Scholar response cache between pipeline runs."""
    with _CACHE_LOCK:
        _CACHE.clear()


def _get(url: str, params: Dict[str, Any], api_key: str) -> Optional[Dict]:
    """
    GET helper with cache + rate-limit throttle.
    - Returns cached result instantly (no network).
    - Enforces 1.2 s between requests to stay under public rate limit.
    - On 429: waits 5 s and retries ONCE, then gives up.
    - On network error: retries ONCE after 3 s.
    - Never blocks for more than ~25 s total per call.
    """
    global _LAST_REQUEST_TIME

    ck = _cache_key(url, params)
    with _CACHE_LOCK:
        if ck in _CACHE:
            _CACHE.move_to_end(ck)  # mark as recently used (LRU)
            return _CACHE[ck]

    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    with _API_SEMAPHORE:
        # Enforce minimum inter-request gap
        with _RATE_LOCK:
            now = time.time()
            gap = _MIN_GAP - (now - _LAST_REQUEST_TIME)
            if gap > 0:
                time.sleep(gap)
            _LAST_REQUEST_TIME = time.time()

        for attempt in range(2):  # max 2 attempts
            try:
                resp = _SESSION.get(url, params=params, headers=headers, timeout=_TIMEOUT)
                if resp.status_code == 200:
                    data = resp.json()
                    with _CACHE_LOCK:
                        if len(_CACHE) >= _CACHE_MAX:
                            _CACHE.popitem(last=False)  # evict LRU (oldest access)
                        _CACHE[ck] = data
                    return data
                if resp.status_code == 429:
                    if attempt == 0:
                        time.sleep(5)  # wait 5 s then try once more
                        continue
                    return None  # give up on second 429
                return None  # other error (404, 500, etc.)
            except requests.RequestException:
                if attempt == 0:
                    time.sleep(3)
                    continue
                return None

    return None


def _normalise(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure all expected keys exist with safe default values."""
    return {
        "paperId": raw.get("paperId") or "",
        "title": raw.get("title") or "",
        "abstract": raw.get("abstract") or "",
        "year": raw.get("year") or 0,
        "citationCount": raw.get("citationCount") or 0,
    }


def search_papers(query: str, limit: int = 20, api_key: str = "") -> List[Dict[str, Any]]:
    """
    Full-text search for papers matching *query*.
    Returns up to *limit* normalised paper dicts.
    """
    url = f"{_BASE}/paper/search"
    params = {"query": query, "limit": min(limit, 100), "fields": _FIELDS}
    data = _get(url, params, api_key)
    if not data:
        return []
    papers = []
    for item in data.get("data", []):
        p = _normalise(item)
        if p["paperId"]:
            papers.append(p)
    return papers


def get_citations(paper_id: str, limit: int = 10, api_key: str = "") -> List[Dict[str, Any]]:
    """
    Papers that *cite* the given paper (forward references).
    Returns up to *limit* normalised paper dicts.
    """
    url = f"{_BASE}/paper/{paper_id}/citations"
    params = {"limit": min(limit, 100), "fields": _FIELDS}
    data = _get(url, params, api_key)
    if not data:
        return []
    papers = []
    for item in data.get("data", []):
        citing = item.get("citingPaper") or {}
        p = _normalise(citing)
        if p["paperId"] and p["paperId"] != paper_id:
            papers.append(p)
    return papers


def get_references(paper_id: str, limit: int = 10, api_key: str = "") -> List[Dict[str, Any]]:
    """
    Papers *referenced by* the given paper (backward references).
    Returns up to *limit* normalised paper dicts.
    """
    url = f"{_BASE}/paper/{paper_id}/references"
    params = {"limit": min(limit, 100), "fields": _FIELDS}
    data = _get(url, params, api_key)
    if not data:
        return []
    papers = []
    for item in data.get("data", []):
        ref = item.get("citedPaper") or {}
        p = _normalise(ref)
        if p["paperId"] and p["paperId"] != paper_id:
            papers.append(p)
    return papers
