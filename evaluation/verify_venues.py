"""
evaluation/verify_venues.py — gather DOCUMENTED venue evidence for a
published-half worklist, so the human verifier reviews evidence instead of
researching 50 papers from scratch.

Evidence sources (both are records, not model memory):
  1. arXiv metadata (single batched id_list query): the `journal_ref` field
     (author-supplied publication record) and the `comment` field (authors
     routinely write "Accepted at ACL 2025"). journal_ref is STRONG evidence;
     an accept-phrase comment is MODERATE evidence.
  2. DBLP (https://dblp.org/search/publ/api): a curated bibliography. A hit
     whose normalized title matches ours and whose venue is NOT "CoRR"
     (DBLP's name for arXiv preprints) is STRONG evidence of the venue.

Policy:
  * venue is auto-filled ONLY when evidence exists; the evidence (source +
    raw string) is stored alongside in `venue_evidence` for the human to
    check at promote time.
  * entries with no evidence keep venue="VERIFY" — under the benchmark
    protocol these are likely preprint-only and should be REPLACED, unless
    the human finds a venue the indexes missed.
  * nothing in this module ever invents a venue.

Usage:
    python -m evaluation.verify_venues --worklist data/crossval/published_worklist.jsonl
(rewrites the worklist in place; prints an evidence report)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

import requests

from evaluation import EvaluationError

_ARXIV_API = "https://export.arxiv.org/api/query"
_DBLP_API = "https://dblp.org/search/publ/api"
_UA = {"User-Agent": "IdeaGraph-eval/1.0 (venue verification; polite)"}
_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

# Accept-phrases that make an arXiv comment count as venue evidence.
_ACCEPT_RE = re.compile(
    r"\b(accepted|to appear|camera[- ]ready|published|appearing)\b", re.IGNORECASE
)


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


# ── arXiv metadata (one batched call) ────────────────────────────────────────

def fetch_arxiv_metadata(ids: List[str]) -> Dict[str, Dict[str, str]]:
    """{arxiv_id: {journal_ref, comment, title}} for all ids in ONE query."""
    if not ids:
        return {}
    r = requests.get(
        _ARXIV_API,
        params={"id_list": ",".join(ids), "max_results": len(ids)},
        headers=_UA, timeout=60,
    )
    if r.status_code != 200:
        raise EvaluationError(f"arXiv metadata query failed: HTTP {r.status_code}")
    out: Dict[str, Dict[str, str]] = {}
    root = ET.fromstring(r.text)
    for entry in root.findall("atom:entry", _NS):
        raw_id = (entry.findtext("atom:id", "", _NS) or "")
        aid = re.sub(r"v\d+$", "", raw_id.split("/abs/")[-1]) if "/abs/" in raw_id else ""
        if not aid:
            continue
        jr = entry.findtext("arxiv:journal_ref", "", _NS) or ""
        cm = entry.findtext("arxiv:comment", "", _NS) or ""
        ti = re.sub(r"\s+", " ", entry.findtext("atom:title", "", _NS) or "")
        out[aid] = {"journal_ref": jr.strip(), "comment": cm.strip(),
                    "title": ti.strip()}
    return out


# Lookup-failure counters, so "no evidence" is never silently conflated with
# "source unreachable" in the report.
_ERRS = {"dblp": 0, "s2": 0, "openalex": 0}


# ── OpenAlex by arXiv DOI (reachable where S2/DBLP are blocked) ──────────────
# OpenAlex merges the preprint and the published version into one work; when
# a published location is known, its source is a real venue rather than arXiv.

_OA_API = "https://api.openalex.org/works/doi:10.48550/arXiv.{aid}"
_OA_NON_VENUES = re.compile(r"arxiv|corr", re.IGNORECASE)


def openalex_venue(arxiv_id: str, min_gap_s: float = 0.4,
                   _state: Dict[str, float] = {"last": 0.0}) -> Optional[Dict[str, str]]:
    wait = _state["last"] + min_gap_s - time.time()
    if wait > 0:
        time.sleep(wait)
    _state["last"] = time.time()
    params = {"select": "title,publication_year,primary_location,locations"}
    try:
        import config as _cfg
        mt = (getattr(_cfg, "OPENALEX_MAILTO", "") or "").strip()
        if mt:
            params["mailto"] = mt
    except Exception:
        pass
    try:
        r = requests.get(_OA_API.format(aid=arxiv_id), params=params,
                         headers=_UA, timeout=30)
    except requests.RequestException:
        _ERRS["openalex"] += 1
        return None
    if r.status_code == 404:
        return None                      # unknown to OpenAlex — a real answer
    if r.status_code != 200:
        _ERRS["openalex"] += 1
        return None
    try:
        info = r.json()
    except Exception:
        _ERRS["openalex"] += 1
        return None
    locs = [info.get("primary_location") or {}] + list(info.get("locations") or [])
    for loc in locs:
        src = (loc or {}).get("source") or {}
        name = str(src.get("display_name") or "").strip()
        if name and not _OA_NON_VENUES.search(name):
            return {"venue": name,
                    "year": str(info.get("publication_year") or ""),
                    "oa_title": str(info.get("title") or "")}
    return None


# ── Semantic Scholar by arXiv id (throttled; works where DBLP is blocked) ────

_S2_API = "https://api.semanticscholar.org/graph/v1/paper/arXiv:{aid}"
_S2_NON_VENUES = {"", "arxiv.org", "corr", "arxiv"}


def s2_venue(arxiv_id: str, min_gap_s: float = 3.0, retries: int = 2,
             _state: Dict[str, float] = {"last": 0.0}) -> Optional[Dict[str, str]]:
    """Semantic Scholar's recorded venue for an arXiv id, or None. Direct id
    lookup — no title matching needed. Patient: the public pool rate-limits
    aggressively, so retry with backoff before counting a failure. Venues
    that just mean 'the preprint itself' are excluded."""
    info = None
    for attempt in range(1 + retries):
        wait = _state["last"] + min_gap_s - time.time()
        if wait > 0:
            time.sleep(wait)
        _state["last"] = time.time()
        try:
            r = requests.get(_S2_API.format(aid=arxiv_id),
                             params={"fields": "title,venue,year"},
                             headers=_UA, timeout=30)
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code == 404:
            return None                  # id unknown to S2 — a real answer
        if r.status_code != 200:
            time.sleep(3 * (attempt + 1))
            continue
        try:
            info = r.json()
            break
        except Exception:
            continue
    if info is None:
        _ERRS["s2"] += 1
        return None
    venue = str(info.get("venue") or "").strip()
    if venue.lower() in _S2_NON_VENUES:
        return None
    return {"venue": venue, "year": str(info.get("year") or ""),
            "s2_title": str(info.get("title") or "")}


# ── DBLP title lookup (throttled per query) ──────────────────────────────────

def dblp_venue(title: str, min_gap_s: float = 1.5,
               _state: Dict[str, float] = {"last": 0.0}) -> Optional[Dict[str, str]]:
    """Best DBLP hit for `title` whose venue is not CoRR. Returns
    {venue, year, dblp_title} or None. Throttled to stay polite."""
    wait = _state["last"] + min_gap_s - time.time()
    if wait > 0:
        time.sleep(wait)
    _state["last"] = time.time()
    try:
        r = requests.get(_DBLP_API, params={"q": title, "format": "json", "h": 5},
                         headers=_UA, timeout=30)
    except requests.RequestException:
        _ERRS["dblp"] += 1
        return None
    if r.status_code != 200:
        _ERRS["dblp"] += 1
        return None
    try:
        hits = (r.json().get("result", {}).get("hits", {}).get("hit")) or []
    except Exception:
        return None
    want = _norm_title(title)
    for h in hits:
        info = h.get("info") or {}
        got = _norm_title(info.get("title", ""))
        if not got:
            continue
        # exact-ish match: one contains the other and lengths comparable
        if not (want in got or got in want):
            continue
        if abs(len(got) - len(want)) > 20:
            continue
        venue = info.get("venue")
        if isinstance(venue, list):
            venue = ", ".join(str(v) for v in venue)
        venue = str(venue or "").strip()
        if not venue or venue.lower() == "corr":
            continue                      # CoRR == the arXiv preprint itself
        return {"venue": venue, "year": str(info.get("year") or ""),
                "dblp_title": info.get("title", "")}
    return None


# ── main pass ────────────────────────────────────────────────────────────────

def verify(worklist_path: str) -> Dict[str, int]:
    rows: List[Dict[str, Any]] = []
    with open(worklist_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise EvaluationError(f"{worklist_path}: empty")

    ids = [r["arxiv_id"] for r in rows if r.get("arxiv_id")]
    print(f"fetching arXiv metadata for {len(ids)} ids (one batched call)…")
    meta = fetch_arxiv_metadata(ids)

    stats = {"journal_ref": 0, "comment": 0, "openalex": 0, "s2": 0,
             "dblp": 0, "kept": 0, "unresolved": 0}
    for k, r in enumerate(rows, 1):
        aid = r.get("arxiv_id") or ""
        m = meta.get(aid, {})
        evidence: Optional[Dict[str, str]] = None

        # Keep evidence already gathered on a previous pass — re-lookups
        # waste rate-limited budget and can only lose information.
        if r.get("venue") not in ("", "VERIFY", None) and r.get("venue_evidence"):
            stats["kept"] += 1
            print(f"  [{k:02d}] {aid:>13}  (kept) {str(r['venue'])[:62]}")
            continue

        if m.get("journal_ref"):
            evidence = {"source": "arxiv:journal_ref", "raw": m["journal_ref"]}
            r["venue"] = m["journal_ref"][:120]
            stats["journal_ref"] += 1
        elif m.get("comment") and _ACCEPT_RE.search(m["comment"]):
            evidence = {"source": "arxiv:comment", "raw": m["comment"][:300]}
            r["venue"] = f"per arXiv comment: {m['comment'][:100]}"
            stats["comment"] += 1
        else:
            hit = openalex_venue(aid) if aid else None
            if hit:
                evidence = {"source": "openalex", "raw": json.dumps(hit)}
                yr = f" {hit['year']}" if hit.get("year") else ""
                r["venue"] = f"{hit['venue']}{yr}"
                stats["openalex"] += 1
            else:
                hit = s2_venue(aid) if aid else None
                if hit:
                    evidence = {"source": "semantic_scholar", "raw": json.dumps(hit)}
                    yr = f" {hit['year']}" if hit.get("year") else ""
                    r["venue"] = f"{hit['venue']}{yr}"
                    stats["s2"] += 1
                else:
                    hit = dblp_venue(r.get("title") or m.get("title") or "")
                    if hit:
                        evidence = {"source": "dblp", "raw": json.dumps(hit)}
                        yr = f" {hit['year']}" if hit.get("year") else ""
                        r["venue"] = f"{hit['venue']}{yr}"
                        stats["dblp"] += 1
                    else:
                        r["venue"] = "VERIFY"
                        stats["unresolved"] += 1

        if evidence:
            r["venue_evidence"] = evidence
        status = r["venue"] if r["venue"] != "VERIFY" else "— unresolved —"
        print(f"  [{k:02d}] {aid:>13}  {status[:70]}")

    with open(worklist_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nevidence: kept={stats['kept']}, journal_ref={stats['journal_ref']}, "
          f"accept-comment={stats['comment']}, openalex={stats['openalex']}, "
          f"s2={stats['s2']}, dblp={stats['dblp']} | unresolved (still "
          f"VERIFY): {stats['unresolved']}")
    if any(_ERRS.values()):
        print(f"LOOKUP FAILURES (source unreachable, NOT evidence of "
              f"absence): openalex={_ERRS['openalex']}, s2={_ERRS['s2']}, "
              f"dblp={_ERRS['dblp']} — unresolved counts may shrink on a "
              f"network where these sources are reachable.")
    print("Unresolved entries: replace them, or supply the venue yourself "
          "if you know it. Evidence strings are stored in venue_evidence "
          "for your promote-time check.")
    return stats


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="evaluation.verify_venues",
                                description=__doc__)
    p.add_argument("--worklist",
                   default="data/crossval/published_worklist.jsonl")
    args = p.parse_args(argv)
    try:
        verify(args.worklist)
        return 0
    except EvaluationError as e:
        print(f"EVALUATION ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
