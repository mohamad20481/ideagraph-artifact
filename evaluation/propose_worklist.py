"""
evaluation/propose_worklist.py — propose CANDIDATE papers for the published
half of a CrossVal benchmark, from LIVE arXiv queries.

Guarantees:
  * Every candidate comes from an actual arXiv API result (via the cached
    tools/arxiv adapter) — real IDs, real titles, real abstracts, real dates.
    Nothing is recalled from a model's memory.
  * Queries mirror the topic clusters of the GENERATED half (read from
    data/crossval/), so the two halves cover comparable territory and blind
    judges can't separate them by topic alone.
  * Date window: 2024-01-01 .. run date (submittedDate >= 20240101). The
    benchmark protocol window is 2024-2026 (author decision, 2026-07); the
    paper revision must state this window.
  * venue is written as "VERIFY": arXiv metadata does not state acceptance
    venue. A human must confirm each paper's actual peer-reviewed venue (or
    replace the paper) — draft_published.py refuses placeholder venues, so
    an unverified worklist cannot flow downstream.

Output: data/crossval/published_worklist_PROPOSED.jsonl
Rename to published_worklist.jsonl only after review.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

from evaluation import EvaluationError

_DEF_OUT = os.path.join("data", "crossval", "published_worklist_PROPOSED.jsonl")

# Topic clusters mirroring the generated half (see data/crossval/*.jsonl).
QUERY_CLUSTERS: List[Dict[str, str]] = [
    {"cluster": "multi-agent LLM systems", "query": "multi-agent large language model coordination"},
    {"cluster": "LLM agent memory", "query": "long-term memory LLM agents"},
    {"cluster": "agentic tool use", "query": "LLM agents tool use planning"},
    {"cluster": "LLM reasoning", "query": "large language model reasoning"},
    {"cluster": "retrieval-augmented generation", "query": "retrieval augmented generation"},
    {"cluster": "LLM code generation", "query": "large language model code generation verification"},
    {"cluster": "distillation", "query": "knowledge distillation language models"},
    {"cluster": "reinforcement learning", "query": "reinforcement learning policy optimization convergence"},
    {"cluster": "video diffusion", "query": "video generation diffusion models"},
    {"cluster": "controllable diffusion", "query": "controllable text-to-image diffusion"},
    {"cluster": "image restoration", "query": "face restoration diffusion"},
    {"cluster": "molecular GNNs", "query": "graph neural networks molecular property prediction"},
    {"cluster": "graph transformers", "query": "graph transformers heterogeneous graphs"},
    {"cluster": "physics GNNs", "query": "graph neural network stress field prediction"},
    {"cluster": "LLM social simulation", "query": "large language model agents social simulation norms"},
    {"cluster": "prompt optimization", "query": "evolutionary prompt optimization language models"},
    {"cluster": "multimodal fusion", "query": "multimodal fusion vision language models"},
    {"cluster": "dialogue safety", "query": "conversational agents safety alignment"},
]


# Paper types that extract poorly into a 7-field research-idea tuple:
# surveys/reviews have no single hypothesis; position papers no method/dataset;
# challenge reports describe events, not one idea.
_DEFAULT_EXCLUDE_TITLE = (
    r"\bsurvey\b|\btaxonomy\b|^position\s*:|\bposition paper\b|"
    r"\ba review\b|\breview of\b|in-depth review|challenge on|\broadmap\b"
)


def propose(
    target: int = 50,
    per_query: int = 6,
    since: str = "20240101",
    out_path: str = _DEF_OUT,
    exclude_title: str = _DEFAULT_EXCLUDE_TITLE,
    exclude_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    import re as _re
    from tools import arxiv as _arxiv

    ex_re = _re.compile(exclude_title, _re.IGNORECASE) if exclude_title else None
    # Reviewer-rejected candidates (relevance drift etc.) — excluded by exact
    # ID so the rest of the (cached) candidate pools stay stable.
    seen: set = set(exclude_ids or [])
    per_cluster: Dict[str, List[Dict[str, Any]]] = {}
    for spec in QUERY_CLUSTERS:
        papers = _arxiv.search(spec["query"], max_results=per_query, since=since)
        rows: List[Dict[str, Any]] = []
        for p in papers:
            if not p.title or not p.abstract or len(p.abstract) < 300:
                continue
            if p.year is None or p.year < 2024:
                continue
            if ex_re and ex_re.search(p.title):
                continue
            if p.arxiv_id in seen:
                continue
            seen.add(p.arxiv_id)
            rows.append({
                "arxiv_id": p.arxiv_id,
                "title": p.title,
                "year": p.year,
                "venue": "VERIFY",
                "topic": spec["cluster"],
                "published": p.published[:10],
                "primary_category": p.primary_category,
            })
        per_cluster[spec["cluster"]] = rows
        print(f"  {spec['cluster']:32s} -> {len(rows)} candidates")

    # Round-robin across clusters so no single topic dominates.
    ordered: List[Dict[str, Any]] = []
    idx = 0
    while len(ordered) < target:
        added = False
        for spec in QUERY_CLUSTERS:
            rows = per_cluster.get(spec["cluster"]) or []
            if idx < len(rows):
                ordered.append(rows[idx])
                added = True
                if len(ordered) >= target:
                    break
        if not added:
            break
        idx += 1

    if len(ordered) < target:
        print(f"WARNING: only {len(ordered)} candidates found (target {target}) "
              f"— broaden queries or per_query")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in ordered:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n{len(ordered)} PROPOSED candidates -> {out_path}")
    print("Review each: confirm the real acceptance venue (replace pure "
          "preprints), then rename to published_worklist.jsonl.")
    return ordered


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="evaluation.propose_worklist",
                                description=__doc__)
    p.add_argument("--target", type=int, default=50)
    p.add_argument("--per-query", type=int, default=6)
    p.add_argument("--since", default="20240101")
    p.add_argument("--out", default=_DEF_OUT)
    p.add_argument("--exclude-ids", default="",
                   help="comma-separated arXiv ids rejected during review")
    args = p.parse_args(argv)
    try:
        propose(target=args.target, per_query=args.per_query,
                since=args.since, out_path=args.out,
                exclude_ids=[s.strip() for s in args.exclude_ids.split(",")
                             if s.strip()])
        return 0
    except EvaluationError as e:
        print(f"EVALUATION ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
