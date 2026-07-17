"""
evaluation/enrich_from_pdf.py — last-resort extract-only pass: fill remaining
empty draft fields from the paper's arXiv PDF.

Corpus escalation for tuple extraction is: abstract (draft step) → arXiv
HTML (enrich_drafts) → PDF full text (this module). Same hard rule as the
other passes: the LLM may return ONLY what the paper's text states, empty
string otherwise. Provenance records enriched_from="arxiv-pdf" per filled
field. Papers whose PDFs can't be fetched/parsed are left for the human.

The human promote-gate is unchanged: filling a field is a DRAFT aid; the
verifier still reads every entry against the paper before promotion.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from typing import Any, Dict, List, Optional

import requests

from evaluation import EvaluationError
from evaluation.crossval import TUPLE_FIELDS
from evaluation.enrich_drafts import _relevant_excerpts, _extract_missing

_PDF_URL = "https://arxiv.org/pdf/{aid}"
_UA = {"User-Agent": "IdeaGraph-eval/1.0 (benchmark tuple enrichment; polite)"}
_GAP_S = 3.0


def _fetch_pdf_text(arxiv_id: str, max_pages: int = 14) -> Optional[str]:
    """Plain text of the paper's PDF (first `max_pages` pages), or None."""
    try:
        r = requests.get(_PDF_URL.format(aid=arxiv_id), headers=_UA, timeout=90)
    except requests.RequestException:
        return None
    if r.status_code != 200 or not r.content[:4] == b"%PDF":
        return None
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(r.content))
        pages = reader.pages[:max_pages]
        text = " ".join((p.extract_text() or "") for p in pages)
    except Exception:
        return None
    text = " ".join(text.split())
    return text if len(text) > 2000 else None


def enrich_pdf(drafts_path: str) -> Dict[str, int]:
    from claude_provider import get_claude_client
    client = get_claude_client()
    if client is None:
        raise EvaluationError("no LLM client configured")

    rows: List[Dict[str, Any]] = []
    with open(drafts_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    stats = {"already_full": 0, "completed": 0, "partial": 0, "no_pdf": 0}
    for k, d in enumerate(rows, 1):
        missing = [f for f in TUPLE_FIELDS if not str(d.get(f, "")).strip()]
        if not missing:
            stats["already_full"] += 1
            continue
        aid = (d.get("provenance") or {}).get("arxiv_id") or ""
        text = _fetch_pdf_text(aid) if aid else None
        time.sleep(_GAP_S)
        if text is None:
            stats["no_pdf"] += 1
            print(f"  [{k:02d}] {d['id']}: PDF unavailable/unparseable — left "
                  f"for human (missing: {','.join(missing)})")
            continue
        got = _extract_missing(client, d, missing, _relevant_excerpts(text, cap=14000))
        filled = [f for f, v in got.items() if v]
        for f_name, v in got.items():
            if v:
                d[f_name] = v
        prov = d.setdefault("provenance", {})
        if filled:
            prov.setdefault("enriched_fields", [])
            prov["enriched_fields"] = sorted(set(prov["enriched_fields"]) | set(filled))
            prov["enriched_from_pdf"] = filled
        still = [f for f in TUPLE_FIELDS if not str(d.get(f, "")).strip()]
        if still:
            stats["partial"] += 1
            print(f"  [{k:02d}] {d['id']}: filled {filled or 'nothing'}; "
                  f"still empty: {','.join(still)}")
        else:
            stats["completed"] += 1
            print(f"  [{k:02d}] {d['id']}: completed via PDF ({','.join(filled)})")

    with open(drafts_path, "w", encoding="utf-8") as f:
        for d in rows:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"\nalready full: {stats['already_full']}, completed via PDF: "
          f"{stats['completed']}, still partial: {stats['partial']}, "
          f"no PDF: {stats['no_pdf']}")
    return stats


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="evaluation.enrich_from_pdf",
                                description=__doc__)
    p.add_argument("--drafts", default="data/crossval/published_drafts.jsonl")
    args = p.parse_args(argv)
    try:
        enrich_pdf(args.drafts)
        return 0
    except EvaluationError as e:
        print(f"EVALUATION ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
