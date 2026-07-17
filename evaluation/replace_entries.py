"""
evaluation/replace_entries.py — replace specific DRAFT entries (by pub-id)
with same-cluster candidates that are BOTH venue-evidence-backed AND
empirically extractable, then redraft + enrich only those slots.

Used when human review finds an entry structurally unfit for the 7-field
tuple protocol (e.g. qualitative/HCI studies with no datasets/metrics/
baselines to extract). Selection criteria for replacements:

  * same topic cluster (query from propose_worklist.QUERY_CLUSTERS);
  * 2024+; not a survey/position paper; not already in the worklist;
  * documented venue evidence (arXiv journal_ref / accept-comment, else
    OpenAlex by DOI) — same standard as the rest of the worklist;
  * EMPIRICAL signal in the abstract (experiments/datasets/metrics
    vocabulary) so the tuple fields exist to be extracted.

All swaps are appended to data/crossval/replaced_unresolved.log. The 45
untouched drafts (including their full-text enrichments) are preserved
byte-for-byte; only the targeted pub-ids are rewritten. The human
promote-gate is unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

from evaluation import EvaluationError
from evaluation.crossval import TUPLE_FIELDS, EXTRACTION_PROMPT_TEMPLATE
from evaluation.propose_worklist import QUERY_CLUSTERS, _DEFAULT_EXCLUDE_TITLE
from evaluation.verify_venues import (
    fetch_arxiv_metadata, openalex_venue, _ACCEPT_RE,
)
from evaluation.draft_published import _EXTRACT_SYSTEM

_DRAFTS = os.path.join("data", "crossval", "published_drafts.jsonl")
_WORKLIST = os.path.join("data", "crossval", "published_worklist.jsonl")
_LOG = os.path.join("data", "crossval", "replaced_unresolved.log")

_EMPIRICAL_RE = re.compile(
    r"experiment|evaluat|benchmark|dataset|accuracy|\bf1\b|outperform|"
    r"baseline|ablation|results show|we (?:train|test|measure|compare)",
    re.IGNORECASE,
)


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _save_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _evidence_for(aid: str, meta: Dict[str, Dict[str, str]]) -> Optional[Dict[str, str]]:
    m = meta.get(aid, {})
    if m.get("journal_ref"):
        return {"source": "arxiv:journal_ref", "raw": m["journal_ref"],
                "venue": m["journal_ref"][:120]}
    if m.get("comment") and _ACCEPT_RE.search(m["comment"]):
        return {"source": "arxiv:comment", "raw": m["comment"][:300],
                "venue": f"per arXiv comment: {m['comment'][:100]}"}
    hit = openalex_venue(aid)
    if hit:
        yr = f" {hit['year']}" if hit.get("year") else ""
        return {"source": "openalex", "raw": json.dumps(hit),
                "venue": f"{hit['venue']}{yr}"}
    return None


def _draft_tuple(client, paper_title: str, abstract: str, venue: str,
                 year: Any) -> Dict[str, str]:
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(
        title=paper_title, venue=venue, year=year, abstract=abstract,
    )
    resp = client.call(system=_EXTRACT_SYSTEM, user=prompt,
                       max_tokens=700, temperature=0.0, json_mode=True)
    if not getattr(resp, "success", False):
        raise EvaluationError(f"redraft: LLM call failed for {paper_title[:50]!r}")
    text = getattr(resp, "text", "") or ""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    parsed = json.loads(m.group(0) if m else text)
    return {f: str(parsed.get(f) or "").strip()[:2000] for f in TUPLE_FIELDS}


def replace_ids(target_ids: List[str], pool_per_cluster: int = 20,
                since: str = "20240101") -> int:
    from tools import arxiv as _arxiv
    from claude_provider import get_claude_client
    client = get_claude_client()
    if client is None:
        raise EvaluationError("no LLM client configured")

    drafts = _load_jsonl(_DRAFTS)
    worklist = _load_jsonl(_WORKLIST)
    by_id = {d["id"]: d for d in drafts}
    wl_by_aid = {w["arxiv_id"]: w for w in worklist}
    current_ids = set(wl_by_aid)
    query_of = {c["cluster"]: c["query"] for c in QUERY_CLUSTERS}
    ex_re = re.compile(_DEFAULT_EXCLUDE_TITLE, re.IGNORECASE)

    # Targets → (draft, worklist row, cluster).
    targets = []
    for tid in target_ids:
        d = by_id.get(tid)
        if d is None:
            raise EvaluationError(f"no draft with id {tid!r}")
        aid = (d.get("provenance") or {}).get("arxiv_id") or ""
        w = wl_by_aid.get(aid)
        if w is None:
            raise EvaluationError(f"{tid}: arxiv_id {aid!r} not in worklist")
        targets.append((d, w, w.get("topic", "")))

    # Candidate pools per cluster + one batched metadata call.
    pools: Dict[str, List[Any]] = {}
    pool_ids: List[str] = []
    for _, _, cluster in targets:
        if cluster in pools:
            continue
        q = query_of.get(cluster)
        if not q:
            raise EvaluationError(f"no query for cluster {cluster!r}")
        papers = _arxiv.search(q, max_results=pool_per_cluster, since=since)
        cands = [
            p for p in papers
            if p.title and p.abstract and len(p.abstract) >= 300
            and p.year and p.year >= 2024
            and not ex_re.search(p.title)
            and p.arxiv_id not in current_ids
            and _EMPIRICAL_RE.search(p.abstract)      # must look extractable
        ]
        pools[cluster] = cands
        pool_ids.extend(p.arxiv_id for p in cands)
    print(f"checking evidence for {len(pool_ids)} empirical candidates…")
    meta = fetch_arxiv_metadata(pool_ids) if pool_ids else {}

    log_lines: List[str] = []
    replaced = 0
    for d, w, cluster in targets:
        cands = pools.get(cluster) or []
        pick = None
        while cands:
            cand = cands.pop(0)
            if cand.arxiv_id in current_ids:
                continue
            ev = _evidence_for(cand.arxiv_id, meta)
            if ev:
                pick = (cand, ev)
                break
        if pick is None:
            print(f"  ! {d['id']} [{cluster}]: no evidence-backed empirical "
                  f"replacement found — entry left unchanged")
            continue
        cand, ev = pick
        old_aid, old_title = w["arxiv_id"], w.get("title", "")

        # 1. worklist row
        w.update({
            "arxiv_id": cand.arxiv_id, "title": cand.title,
            "year": cand.year, "published": cand.published[:10],
            "primary_category": cand.primary_category,
            "venue": ev["venue"],
            "venue_evidence": {"source": ev["source"], "raw": ev["raw"]},
        })
        current_ids.add(cand.arxiv_id)

        # 2. redraft the tuple from the candidate's own abstract
        tup = _draft_tuple(client, cand.title, cand.abstract, ev["venue"],
                           cand.year)
        d.update(tup)
        d["topic"] = cluster
        d["provenance"] = {
            "paper_title": cand.title, "venue": ev["venue"],
            "year": cand.year, "arxiv_id": cand.arxiv_id,
            "extractor": "llm-draft", "human_verified": False,
            "venue_evidence": {"source": ev["source"], "raw": ev["raw"]},
            "replaced_from": f"{old_aid} ({old_title[:60]})",
            "replacement_reason": "structurally unfit for 7-field tuple "
                                  "(human review)",
        }
        replaced += 1
        log_lines.append(
            f"replaced {old_aid} ({old_title[:60]!r}) with {cand.arxiv_id} "
            f"({cand.title[:60]!r}) via {ev['source']} [empirical-required]"
        )
        print(f"  {d['id']} [{cluster}]: {old_aid} -> {cand.arxiv_id} "
              f"({cand.title[:55]})")
        time.sleep(0.5)

    _save_jsonl(worklist, _WORKLIST)
    _save_jsonl(drafts, _DRAFTS)
    if log_lines:
        with open(_LOG, "a", encoding="utf-8") as f:
            for ln in log_lines:
                f.write(ln + "\n")
    print(f"\nreplaced {replaced}/{len(target_ids)} (log -> {_LOG})")
    return replaced


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="evaluation.replace_entries",
                                description=__doc__)
    p.add_argument("ids", help="comma-separated draft ids, e.g. pub-005,pub-015")
    p.add_argument("--pool-per-cluster", type=int, default=20)
    args = p.parse_args(argv)
    try:
        replace_ids([s.strip() for s in args.ids.split(",") if s.strip()],
                    pool_per_cluster=args.pool_per_cluster)
        return 0
    except EvaluationError as e:
        print(f"EVALUATION ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
