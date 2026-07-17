"""
evaluation/replace_unresolved.py — swap venue-unresolved worklist entries for
evidence-backed candidates from the SAME topic cluster.

Rationale: the benchmark protocol requires venue-published papers. Candidates
whose venue no reachable index can attest are most likely recent preprints;
rather than asking the human verifier to research each one, this pass
searches the cluster's arXiv pool for alternatives whose publication venue is
DOCUMENTED (arXiv journal_ref / accept-comment, else OpenAlex by DOI) and
proposes those instead. Entries it cannot replace stay as VERIFY for the
human. Every swap is logged; the removed ids are recorded in
data/crossval/replaced_unresolved.log so nothing disappears silently.

The result is still a PROPOSAL: the human promote-gate (draft_published.py)
is unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

from evaluation import EvaluationError
from evaluation.propose_worklist import QUERY_CLUSTERS, _DEFAULT_EXCLUDE_TITLE
from evaluation.verify_venues import (
    fetch_arxiv_metadata, openalex_venue, _ACCEPT_RE,
)

_LOG = os.path.join("data", "crossval", "replaced_unresolved.log")


def _evidence_for(aid: str, meta: Dict[str, Dict[str, str]]) -> Optional[Dict[str, str]]:
    """Documented venue evidence for one candidate, or None."""
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


def replace(worklist_path: str, pool_per_cluster: int = 15,
            since: str = "20240101") -> Dict[str, int]:
    from tools import arxiv as _arxiv

    rows: List[Dict[str, Any]] = []
    with open(worklist_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    current_ids = {r["arxiv_id"] for r in rows if r.get("arxiv_id")}
    unresolved = [r for r in rows if r.get("venue") == "VERIFY"]
    if not unresolved:
        print("nothing to replace — no VERIFY entries")
        return {"replaced": 0, "unfilled": 0}

    by_cluster: Dict[str, List[Dict[str, Any]]] = {}
    for r in unresolved:
        by_cluster.setdefault(r["topic"], []).append(r)
    query_of = {c["cluster"]: c["query"] for c in QUERY_CLUSTERS}
    ex_re = re.compile(_DEFAULT_EXCLUDE_TITLE, re.IGNORECASE)

    # 1. Gather fresh candidate pools per affected cluster.
    pools: Dict[str, List[Any]] = {}
    pool_ids: List[str] = []
    for cluster in by_cluster:
        q = query_of.get(cluster)
        if not q:
            print(f"  ! no query for cluster {cluster!r} — its entries stay")
            continue
        papers = _arxiv.search(q, max_results=pool_per_cluster, since=since)
        cands = [
            p for p in papers
            if p.title and p.abstract and len(p.abstract) >= 300
            and p.year and p.year >= 2024
            and not ex_re.search(p.title)
            and p.arxiv_id not in current_ids
        ]
        pools[cluster] = cands
        pool_ids.extend(p.arxiv_id for p in cands)

    # 2. One batched arXiv metadata call for every pool candidate.
    print(f"checking venue evidence for {len(pool_ids)} pool candidates…")
    meta = fetch_arxiv_metadata(pool_ids) if pool_ids else {}

    # 3. Per cluster, swap unresolved rows for evidence-backed candidates.
    stats = {"replaced": 0, "unfilled": 0}
    log_lines: List[str] = []
    for cluster, needy in by_cluster.items():
        cands = list(pools.get(cluster) or [])
        for row in needy:
            replacement = None
            while cands:
                cand = cands.pop(0)
                if cand.arxiv_id in current_ids:
                    continue
                ev = _evidence_for(cand.arxiv_id, meta)
                if ev:
                    replacement = (cand, ev)
                    break
            if replacement is None:
                stats["unfilled"] += 1
                print(f"  [{cluster}] no evidence-backed replacement for "
                      f"{row['arxiv_id']} — stays VERIFY")
                continue
            cand, ev = replacement
            log_lines.append(
                f"replaced {row['arxiv_id']} ({row['title'][:60]!r}) with "
                f"{cand.arxiv_id} ({cand.title[:60]!r}) via {ev['source']}"
            )
            row.update({
                "arxiv_id": cand.arxiv_id,
                "title": cand.title,
                "year": cand.year,
                "published": cand.published[:10],
                "primary_category": cand.primary_category,
                "venue": ev["venue"],
                "venue_evidence": {"source": ev["source"], "raw": ev["raw"]},
            })
            current_ids.add(cand.arxiv_id)
            stats["replaced"] += 1
            print(f"  [{cluster}] {log_lines[-1][:100]}")

    with open(worklist_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if log_lines:
        with open(_LOG, "a", encoding="utf-8") as f:
            for ln in log_lines:
                f.write(ln + "\n")
    print(f"\nreplaced={stats['replaced']}, still unresolved={stats['unfilled']} "
          f"(swap log -> {_LOG})")
    return stats


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="evaluation.replace_unresolved",
                                description=__doc__)
    p.add_argument("--worklist",
                   default=os.path.join("data", "crossval",
                                        "published_worklist.jsonl"))
    p.add_argument("--pool-per-cluster", type=int, default=15)
    args = p.parse_args(argv)
    try:
        replace(args.worklist, pool_per_cluster=args.pool_per_cluster)
        return 0
    except EvaluationError as e:
        print(f"EVALUATION ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
