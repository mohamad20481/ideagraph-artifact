"""
evaluation/draft_published.py — draft the PUBLISHED half of a CrossVal
benchmark from real papers, with an enforced human-verification gate.

Workflow (three explicit stages, so no draft can silently become data):

  1. worklist  A human writes data/crossval/published_worklist.jsonl:
                   {"arxiv_id": "2403.01234", "venue": "ACL", "year": 2024}
               or {"title": "...", "abstract": "...", "venue": ..., "year": ...}
               Venue and year are REQUIRED — they are the provenance the
               benchmark loader enforces.

  2. draft     `python -m evaluation.draft_published draft`
               Fetches each paper's abstract (tools/arxiv for arXiv ids; the
               provided abstract otherwise) and LLM-drafts the 7-field tuple
               using crossval.EXTRACTION_PROMPT_TEMPLATE (extract-only).
               Output goes to published_drafts.jsonl with
               provenance.extractor = "llm-draft", human_verified = false.
               DRAFTS ARE NOT BENCHMARK ENTRIES.

  3. promote   A HUMAN reads each draft against the actual paper, corrects
               it, then runs:
               `python -m evaluation.draft_published promote --verified-by "Name"`
               which re-validates every draft, stamps
               human_verified = true / verified_by = Name, and writes
               crossval_published.jsonl. Promotion refuses drafts whose
               verified flag was not explicitly set by the --i-have-read-
               every-entry acknowledgement, so bulk-promoting unread drafts
               requires a deliberate lie in the command itself.

The paper must describe this protocol as what it is: LLM-assisted extraction
with human verification, extractor recorded per entry.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

from evaluation import EvaluationError
from evaluation.crossval import (
    CrossValEntry, TUPLE_FIELDS, EXTRACTION_PROMPT_TEMPLATE, save_jsonl,
)

_DEF_DIR = os.path.join("data", "crossval")
WORKLIST = os.path.join(_DEF_DIR, "published_worklist.jsonl")
DRAFTS = os.path.join(_DEF_DIR, "published_drafts.jsonl")
FINAL = os.path.join(_DEF_DIR, "crossval_published.jsonl")

_EXTRACT_SYSTEM = (
    "You extract the core research idea of a published paper into a fixed "
    "7-field tuple. Use ONLY what the provided text states; do not embellish, "
    "modernize, or fill gaps from your own knowledge of the paper. If a field "
    "is not stated in the text, return an empty string for it. Output ONLY "
    "valid JSON."
)


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise EvaluationError(f"not found: {path}")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise EvaluationError(f"{path}:{ln}: bad JSON: {e}") from e
    if not rows:
        raise EvaluationError(f"{path}: empty")
    return rows


def _write_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _fetch_paper(item: Dict[str, Any]) -> Dict[str, str]:
    """Resolve a worklist item to {title, abstract}. arXiv ids are fetched
    live (cached adapter); otherwise both title+abstract must be provided."""
    if item.get("arxiv_id"):
        from tools import arxiv as _arxiv
        p = _arxiv.fetch_by_id(str(item["arxiv_id"]))
        if p is None:
            raise EvaluationError(
                f"arXiv fetch failed for {item['arxiv_id']!r} — check the id"
            )
        return {"title": p.title, "abstract": p.abstract}
    title = str(item.get("title") or "").strip()
    abstract = str(item.get("abstract") or "").strip()
    if not (title and abstract):
        raise EvaluationError(
            "worklist item needs either arxiv_id or BOTH title and abstract: "
            f"{json.dumps(item)[:120]}"
        )
    return {"title": title, "abstract": abstract}


def cmd_draft(args) -> int:
    from claude_provider import get_claude_client
    client = get_claude_client()
    if client is None:
        raise EvaluationError("no LLM client configured")
    worklist = _load_jsonl(args.worklist)
    drafts: List[Dict[str, Any]] = []
    for k, item in enumerate(worklist, 1):
        venue = str(item.get("venue") or "").strip()
        year = item.get("year")
        if not venue or not year:
            raise EvaluationError(
                f"worklist item #{k}: venue and year are REQUIRED provenance"
            )
        # Proposed worklists ship with venue="VERIFY" — a human must confirm
        # the real acceptance venue (or replace the paper) before drafting.
        if re.search(r"verify|tbc|unknown|\?\?", venue, re.IGNORECASE):
            raise EvaluationError(
                f"worklist item #{k} ({item.get('arxiv_id') or item.get('title', '')!r}): "
                f"venue is a placeholder ({venue!r}) — confirm the actual "
                f"publication venue before drafting"
            )
        paper = _fetch_paper(item)
        prompt = EXTRACTION_PROMPT_TEMPLATE.format(
            title=paper["title"], venue=venue, year=year,
            abstract=paper["abstract"],
        )
        resp = client.call(system=_EXTRACT_SYSTEM, user=prompt,
                           max_tokens=700, temperature=0.0, json_mode=True)
        if not getattr(resp, "success", False):
            raise EvaluationError(f"draft #{k}: LLM call failed")
        text = getattr(resp, "text", "") or ""
        m = re.search(r"\{.*\}", text, re.DOTALL)
        try:
            parsed = json.loads(m.group(0) if m else text)
        except Exception as e:
            raise EvaluationError(f"draft #{k}: unparseable: {text[:120]!r}") from e
        draft = {
            "id": f"pub-{k:03d}",
            "source": "published",
            "topic": str(item.get("topic") or paper["title"])[:200],
            "title": "",          # intentionally blank: blinding excludes titles
            "provenance": {
                "paper_title": paper["title"],
                "venue": venue, "year": year,
                "arxiv_id": item.get("arxiv_id", ""),
                "extractor": "llm-draft",
                "human_verified": False,
            },
        }
        for f_name in TUPLE_FIELDS:
            draft[f_name] = str(parsed.get(f_name) or "").strip()[:2000]
        drafts.append(draft)
        print(f"  drafted {draft['id']}: {paper['title'][:60]}")
    _write_jsonl(drafts, args.out)
    print(f"\n{len(drafts)} DRAFTS -> {args.out}")
    print("These are NOT benchmark entries. A human must now read each draft "
          "against the actual paper, correct it in place, and run `promote`.")
    return 0


def cmd_promote(args) -> int:
    if not args.i_have_read_every_entry:
        raise EvaluationError(
            "promote requires --i-have-read-every-entry: a human must have "
            "read and corrected every draft against the actual paper first"
        )
    verifier = (args.verified_by or "").strip()
    if not verifier:
        raise EvaluationError("promote requires --verified-by \"Your Name\"")
    drafts = _load_jsonl(args.drafts)
    entries: List[CrossValEntry] = []
    for d in drafts:
        prov = dict(d.get("provenance") or {})
        prov["human_verified"] = True
        prov["verified_by"] = verifier
        entry = CrossValEntry(
            id=str(d.get("id")), source="published",
            topic=str(d.get("topic") or ""), title="",
            provenance=prov,
            **{f: str(d.get(f) or "") for f in TUPLE_FIELDS},
        )
        entry.validate()   # raises on any empty tuple field → fix draft first
        entries.append(entry)
    save_jsonl(entries, args.out)
    print(f"{len(entries)} verified published entries -> {args.out} "
          f"(verified_by={verifier})")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="evaluation.draft_published",
                                description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draft")
    d.add_argument("--worklist", default=WORKLIST)
    d.add_argument("--out", default=DRAFTS)
    pr = sub.add_parser("promote")
    pr.add_argument("--drafts", default=DRAFTS)
    pr.add_argument("--out", default=FINAL)
    pr.add_argument("--verified-by", default="")
    pr.add_argument("--i-have-read-every-entry", action="store_true")
    args = p.parse_args(argv)
    try:
        return {"draft": cmd_draft, "promote": cmd_promote}[args.cmd](args)
    except EvaluationError as e:
        print(f"EVALUATION ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
