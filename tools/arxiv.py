"""
tools/arxiv.py — Thin, polite wrapper around the arXiv API.

The arXiv API (export.arxiv.org/api/query) is free, requires no key, but
asks clients to:
  - Throttle to ≤ 1 request every 3 seconds (their published guideline)
  - Identify themselves via User-Agent

We honor both. All public functions return lists of `ArxivPaper` dicts
with keys: arxiv_id, title, abstract, authors, year, primary_category,
categories, pdf_url, published, updated.

Why we need this:
  - Bayesian Surprise / corpus_novelty currently has Semantic Scholar
    only. arXiv covers preprints S2 lags on, and the API is faster + free.
  - HindSight evaluation needs a corpus snapshot cut at a year boundary —
    arXiv's `submittedDate:[YYYY1 TO YYYY2]` filter makes this trivial.
  - Real-time novelty surveillance wants arXiv RSS for new-paper alerts.

Rate-limit strategy:
  - Hard concurrency cap: 1 in-flight request (semaphore)
  - 3 s inter-request gap (arXiv's published guideline)
  - On 503/429: single 10 s wait then give up — never loop forever
  - On network error: 1 retry only

Caching: LRU in-memory cache (512 entries) + optional disk cache
(`.ideagraph_arxiv_cache/`) keyed by query hash. Disk cache survives
restarts; in-memory cache survives only the Streamlit process.

Public API:
    search(query, max_results=20, sort_by="relevance",
           since=None, until=None) → List[ArxivPaper]
    fetch_by_id(arxiv_id) → ArxivPaper | None
    to_corpus_entries(papers) → List[CorpusEntry]
    clear_cache() → None
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from collections import OrderedDict
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import requests


_BASE = "https://export.arxiv.org/api/query"
# arXiv's export endpoint can be SLOW under load — empirically 15-30s
# for even a small query, occasionally up to 50s when their servers are
# stressed. Default to 60s and let operators override via env var if
# they need to be more or less forgiving.
_TIMEOUT = int(os.getenv("ARXIV_HTTP_TIMEOUT", "60"))
_USER_AGENT = "IdeaGraph/1.0 (research-ideation pipeline; +https://github.com/anonymous-authors/IdeaGraph)"

# Persistent HTTP session with keep-alive.
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": _USER_AGENT})

# Rate-limiting (arXiv asks for ≤ 1 req / 3 s; we'll honor that).
_API_SEMAPHORE = threading.Semaphore(1)
_RATE_LOCK = threading.Lock()
_LAST_REQUEST_TIME: float = 0.0
_MIN_GAP = 3.0

# Last error state — lets the UI surface "rate limited" vs "network down"
# vs "empty results for this query" with the right message instead of
# silently returning [] for everything. Cleared on each successful call.
_LAST_ERROR_LOCK = threading.Lock()
_LAST_ERROR: Optional[Dict[str, Any]] = None

# In-memory LRU cache.
_CACHE: "OrderedDict[str, Any]" = OrderedDict()
_CACHE_MAX = 512
_CACHE_LOCK = threading.RLock()

# Disk cache (lazy-created).
_DISK_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".ideagraph_arxiv_cache",
)

# Sidecar index file inside the cache dir. Each line is a JSON object:
# {query, params, hash, n_results, ts}. Lets `list_cached_queries()`
# show the user what's available offline so they can keep working
# when the live API is rate-limited.
_DISK_INDEX_PATH = os.path.join(_DISK_CACHE_DIR, "_index.jsonl")
_DISK_INDEX_LOCK = threading.Lock()


# ── XML namespaces in arXiv's Atom feed ────────────────────────────────────
_NS = {
    "atom":  "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}


# ── Result dataclass ───────────────────────────────────────────────────────

@dataclass
class ArxivPaper:
    """One arXiv paper. Returned by search/fetch_by_id."""
    arxiv_id: str
    title: str
    abstract: str
    authors: List[str]
    year: Optional[int]
    primary_category: str
    categories: List[str]
    pdf_url: str
    published: str          # ISO-8601 first-version timestamp
    updated: str            # ISO-8601 latest-version timestamp

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Caching helpers ────────────────────────────────────────────────────────

def _cache_key(query: str, **params: Any) -> str:
    raw = query + "\x00" + json.dumps(params, sort_keys=True)
    return hashlib.md5(raw.encode(), usedforsecurity=False).hexdigest()


def _cache_get(key: str) -> Optional[Any]:
    with _CACHE_LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]
    return None


def _cache_put(key: str, value: Any) -> None:
    with _CACHE_LOCK:
        _CACHE[key] = value
        _CACHE.move_to_end(key)
        while len(_CACHE) > _CACHE_MAX:
            _CACHE.popitem(last=False)


def _disk_cache_path(key: str) -> str:
    return os.path.join(_DISK_CACHE_DIR, f"{key}.json")


def _disk_cache_get(key: str) -> Optional[Any]:
    path = _disk_cache_path(key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _disk_cache_put(key: str, value: Any) -> None:
    try:
        os.makedirs(_DISK_CACHE_DIR, exist_ok=True)
        with open(_disk_cache_path(key), "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False)
    except Exception:
        pass


def _disk_index_append(
    query: str, params: Dict[str, Any], key: str, n_results: int,
) -> None:
    """Append one row to the disk cache index so `list_cached_queries`
    can show users what's available offline. Idempotent: if the index
    already has this hash, we skip — avoids growing the file forever."""
    try:
        os.makedirs(_DISK_CACHE_DIR, exist_ok=True)
        with _DISK_INDEX_LOCK:
            # Check whether this hash is already indexed.
            if os.path.exists(_DISK_INDEX_PATH):
                with open(_DISK_INDEX_PATH, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            row = json.loads(line)
                        except Exception:
                            continue
                        if row.get("hash") == key:
                            return
            row = {
                "query": query,
                "params": params,
                "hash": key,
                "n_results": int(n_results),
                "ts": time.time(),
            }
            with open(_DISK_INDEX_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass


def list_cached_queries(limit: int = 50) -> List[Dict[str, Any]]:
    """Return the queries previously fetched from arXiv that are now
    available offline in `.ideagraph_arxiv_cache/`. Newest first.

    Each entry: {query, params, hash, n_results, ts}.
    """
    if not os.path.exists(_DISK_INDEX_PATH):
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with _DISK_INDEX_LOCK:
            with open(_DISK_INDEX_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                        rows.append(row)
                    except Exception:
                        continue
    except Exception:
        return []
    rows.sort(key=lambda r: r.get("ts", 0), reverse=True)
    return rows[: max(1, int(limit))]


def clear_cache() -> None:
    """Clear both in-memory and disk caches. Useful between test runs."""
    with _CACHE_LOCK:
        _CACHE.clear()


def last_error() -> Optional[Dict[str, Any]]:
    """Return a snapshot of the most-recent error from `_http_get`, or
    None if the last call succeeded. Used by the UI to distinguish
    'rate-limited' from 'empty results for this specific query'.

    Shape:
        {"kind": "rate_limited" | "network" | "http" | "parse",
         "status_code": int | None,
         "message": str,
         "url_hint": str}
    """
    with _LAST_ERROR_LOCK:
        return dict(_LAST_ERROR) if _LAST_ERROR else None


def _set_last_error(
    kind: str, status_code: Optional[int], message: str, url_hint: str = "",
) -> None:
    global _LAST_ERROR
    with _LAST_ERROR_LOCK:
        _LAST_ERROR = {
            "kind": kind,
            "status_code": status_code,
            "message": message,
            "url_hint": url_hint,
        }


def _clear_last_error() -> None:
    global _LAST_ERROR
    with _LAST_ERROR_LOCK:
        _LAST_ERROR = None


# ── Rate-limit gate ────────────────────────────────────────────────────────

def _wait_for_slot() -> None:
    """Enforce the 3-second inter-request gap."""
    global _LAST_REQUEST_TIME
    with _RATE_LOCK:
        elapsed = time.time() - _LAST_REQUEST_TIME
        if elapsed < _MIN_GAP:
            time.sleep(_MIN_GAP - elapsed)
        _LAST_REQUEST_TIME = time.time()


# ── HTTP request + parse ───────────────────────────────────────────────────

def _http_get(query_str: str) -> Optional[str]:
    """Fetch raw Atom XML for an arXiv query. Returns None on failure
    and records details via `_set_last_error` so the UI can surface
    'rate-limited' vs 'network down' vs 'parse failure'."""
    url = f"{_BASE}?{query_str}"
    with _API_SEMAPHORE:
        _wait_for_slot()
        last_status: Optional[int] = None
        for attempt in range(2):
            try:
                r = _SESSION.get(url, timeout=_TIMEOUT)
            except requests.Timeout as e:
                # arXiv is consistently slow under load. Retry once
                # with a longer timeout before giving up; surface a
                # specific message so the UI says "increase timeout"
                # rather than the generic "network error".
                if attempt == 0:
                    time.sleep(2)
                    continue
                _set_last_error(
                    kind="timeout",
                    status_code=None,
                    message=(
                        f"arXiv took longer than {_TIMEOUT}s to respond. "
                        f"Try a smaller pool size or set "
                        f"ARXIV_HTTP_TIMEOUT=120 in .env."
                    ),
                    url_hint=url[:120],
                )
                return None
            except Exception as e:
                if attempt == 0:
                    time.sleep(2)
                    continue
                _set_last_error(
                    kind="network",
                    status_code=None,
                    message=(
                        f"network error: {type(e).__name__}: "
                        f"{str(e)[:200]}"
                    ),
                    url_hint=url[:120],
                )
                return None
            last_status = r.status_code
            if r.status_code == 200:
                _clear_last_error()
                return r.text
            if r.status_code in (429, 503):
                # arXiv's back-off signal. After our second attempt,
                # surface a clear error so the UI can say "wait X
                # minutes" instead of "no results".
                if attempt == 0:
                    time.sleep(10)
                    continue
                _set_last_error(
                    kind="rate_limited",
                    status_code=r.status_code,
                    message=(
                        "arXiv rate-limit hit. Your IP is being "
                        "throttled. Wait 5-60 minutes before retrying. "
                        "(The disk cache still works — re-running an "
                        "earlier query is fine.)"
                    ),
                    url_hint=url[:120],
                )
                return None
            _set_last_error(
                kind="http",
                status_code=r.status_code,
                message=f"arXiv returned HTTP {r.status_code}",
                url_hint=url[:120],
            )
            return None
        # Loop exited without success/failure record (shouldn't happen).
        _set_last_error(
            kind="http", status_code=last_status,
            message="arXiv unreachable after 2 attempts",
            url_hint=url[:120],
        )
    return None


def _parse_feed(xml_text: str) -> List[ArxivPaper]:
    """Parse arXiv's Atom feed into a list of ArxivPaper. Returns [] on
    malformed input rather than raising."""
    if not xml_text:
        return []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    out: List[ArxivPaper] = []
    for entry in root.findall("atom:entry", _NS):
        try:
            paper = _parse_entry(entry)
        except Exception:
            continue
        if paper is not None:
            out.append(paper)
    return out


def _parse_entry(entry: ET.Element) -> Optional[ArxivPaper]:
    """Convert one <entry> element to an ArxivPaper, or None if it lacks
    required fields."""
    def _text(tag: str) -> str:
        el = entry.find(tag, _NS)
        return (el.text or "").strip() if el is not None else ""

    title = re.sub(r"\s+", " ", _text("atom:title"))
    summary = re.sub(r"\s+", " ", _text("atom:summary"))
    raw_id = _text("atom:id")
    if not title or not raw_id:
        return None

    # raw_id looks like https://arxiv.org/abs/2403.01234v1
    arxiv_id = raw_id.split("/abs/")[-1] if "/abs/" in raw_id else raw_id
    # Strip version suffix for canonical id (e.g. 2403.01234v1 → 2403.01234)
    canonical_id = re.sub(r"v\d+$", "", arxiv_id)

    published = _text("atom:published")
    updated = _text("atom:updated")
    year: Optional[int] = None
    if published and len(published) >= 4 and published[:4].isdigit():
        year = int(published[:4])

    authors: List[str] = []
    for a in entry.findall("atom:author/atom:name", _NS):
        if a.text:
            authors.append(a.text.strip())

    primary_cat = ""
    primary_el = entry.find("arxiv:primary_category", _NS)
    if primary_el is not None:
        primary_cat = primary_el.get("term") or ""

    cats: List[str] = []
    for c in entry.findall("atom:category", _NS):
        term = c.get("term")
        if term:
            cats.append(term)

    pdf_url = ""
    for link in entry.findall("atom:link", _NS):
        if link.get("title") == "pdf" and link.get("href"):
            pdf_url = link.get("href")  # type: ignore[assignment]
            break

    return ArxivPaper(
        arxiv_id=canonical_id,
        title=title,
        abstract=summary,
        authors=authors,
        year=year,
        primary_category=primary_cat,
        categories=cats,
        pdf_url=pdf_url,
        published=published,
        updated=updated,
    )


# ── Public API ─────────────────────────────────────────────────────────────

_VALID_SORTS = {"relevance", "lastUpdatedDate", "submittedDate"}


def search(
    query: str,
    max_results: int = 20,
    sort_by: str = "relevance",
    since: Optional[str] = None,
    until: Optional[str] = None,
    use_disk_cache: bool = True,
) -> List[ArxivPaper]:
    """Search arXiv. Returns up to `max_results` papers.

    Args:
        query:       Free-text query. Will be sent as `all:<query>`.
        max_results: 1..200. arXiv enforces 2000 max; we cap at 200 to
                     stay polite.
        sort_by:     "relevance" | "lastUpdatedDate" | "submittedDate"
        since:       "YYYYMMDD" lower bound for submittedDate
        until:       "YYYYMMDD" upper bound for submittedDate
        use_disk_cache: read+write `.ideagraph_arxiv_cache/`

    Returns an empty list on any failure (network, parse, rate-limit).
    """
    if not query or not query.strip():
        return []
    max_results = max(1, min(200, int(max_results)))
    if sort_by not in _VALID_SORTS:
        sort_by = "relevance"

    search_q = f"all:{query.strip()}"
    if since or until:
        lo = since or "00000101"
        hi = until or "99991231"
        search_q = f"({search_q}) AND submittedDate:[{lo} TO {hi}]"

    params = (
        f"search_query={requests.utils.quote(search_q)}"
        f"&start=0"
        f"&max_results={max_results}"
        f"&sortBy={sort_by}"
        f"&sortOrder=descending"
    )
    key = _cache_key(query, max_results=max_results, sort_by=sort_by,
                     since=since or "", until=until or "")

    cached = _cache_get(key)
    if cached is not None:
        return [ArxivPaper(**p) for p in cached]
    if use_disk_cache:
        disk = _disk_cache_get(key)
        if disk is not None:
            _cache_put(key, disk)
            return [ArxivPaper(**p) for p in disk]

    xml = _http_get(params)
    if xml is None:
        return []
    papers = _parse_feed(xml)
    payload = [p.to_dict() for p in papers]
    _cache_put(key, payload)
    if use_disk_cache:
        _disk_cache_put(key, payload)
        # Index the query so the UI can list cached queries when the
        # live API is rate-limited. Only record successful + non-empty
        # responses — empty ones aren't useful to expose.
        if papers:
            _disk_index_append(
                query=query,
                params={
                    "max_results": max_results,
                    "sort_by": sort_by,
                    "since": since or "",
                    "until": until or "",
                },
                key=key,
                n_results=len(papers),
            )
    return papers


def fetch_by_id(arxiv_id: str) -> Optional[ArxivPaper]:
    """Fetch a single paper by its arXiv ID (e.g. "2403.01234" or
    "cs.CL/0501001" for legacy ids). Returns None if not found."""
    if not arxiv_id or not arxiv_id.strip():
        return None
    canonical = re.sub(r"v\d+$", "", arxiv_id.strip())
    key = _cache_key("id_lookup", arxiv_id=canonical)
    cached = _cache_get(key)
    if cached is not None and cached:
        return ArxivPaper(**cached[0]) if isinstance(cached, list) else None

    params = f"id_list={requests.utils.quote(canonical)}&max_results=1"
    xml = _http_get(params)
    if xml is None:
        return None
    papers = _parse_feed(xml)
    if not papers:
        return None
    payload = [papers[0].to_dict()]
    _cache_put(key, payload)
    return papers[0]


def to_corpus_entries(papers: List[ArxivPaper]) -> List[Any]:
    """Convert ArxivPapers to CorpusEntry instances for `corpus_novelty`.

    Returns a list of CorpusEntry; if `corpus_novelty` import fails
    (e.g. running outside the project), returns the raw papers as dicts."""
    try:
        from corpus_novelty import CorpusEntry
    except Exception:
        return [p.to_dict() for p in papers]

    out = []
    for p in papers:
        out.append(CorpusEntry(
            title=p.title,
            abstract=p.abstract,
            source="arxiv",
            year=p.year,
        ))
    return out
