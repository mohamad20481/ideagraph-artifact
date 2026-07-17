"""
evaluation/enrich_drafts.py — fill empty draft fields from the paper's OWN
full text (arXiv HTML), extract-only.

Why: tuple drafting from abstracts leaves dataset/metrics/baselines empty for
most papers (abstracts rarely state them). arXiv has served an HTML full-text
rendering for most 2024+ papers at https://arxiv.org/html/<id>; the
experiments/setup sections do state these fields. This pass:

  * only touches drafts with EMPTY fields;
  * fetches the paper's arXiv HTML (3s polite gap); if unavailable (404 —
    some papers opt out), the draft is left for the human;
  * selects passages around dataset/metric/baseline keywords (caps prompt
    size) and asks the LLM to extract ONLY the missing fields, empty string
    if the text still does not state them;
  * records provenance: enriched_fields + enriched_from="arxiv-html".

The human promote-gate is unchanged: every entry still requires a full
read-and-correct pass against the paper.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from typing import Any, Dict, List, Optional

import requests

from evaluation import EvaluationError
from evaluation.crossval import TUPLE_FIELDS

_HTML_URL = "https://arxiv.org/html/{aid}"
_UA = {"User-Agent": "IdeaGraph-eval/1.0 (benchmark tuple enrichment; polite)"}
_GAP_S = 3.0

_KEYWORD_RE = re.compile(
    r"dataset|corpus|benchmark|metric|f1|accuracy|bleu|rouge|auc|baseline|"
    r"compared? (?:to|with|against)|evaluat|experiment", re.IGNORECASE,
)

_SYSTEM = (
    "You extract structured facts from a research paper's own text. You may "
    "not add, infer, or invent anything the text does not state. If the text "
    "does not state a requested item, return an empty string for it. Output "
    "ONLY valid JSON with exactly the requested keys."
)


def _fetch_html_text(arxiv_id: str) -> Optional[str]:
    """Plain text of the paper's arXiv HTML rendering, or None if absent."""
    try:
        r = requests.get(_HTML_URL.format(aid=arxiv_id), headers=_UA, timeout=60)
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    html = r.text
    # crude but adequate tag stripping
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html,
                  flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    return text if len(text) > 2000 else None


def _relevant_excerpts(text: str, cap: int = 12000) -> str:
    """Sentence-ish chunks around benchmark keywords, up to `cap` chars."""
    chunks: List[str] = []
    total = 0
    for m in _KEYWORD_RE.finditer(text):
        lo = max(0, m.start() - 300)
        hi = min(len(text), m.end() + 300)
        c = text[lo:hi]
        chunks.append(c)
        total += len(c)
        if total >= cap:
            break
    return "\n---\n".join(chunks)[:cap]


def _extract_missing(client, draft: Dict[str, Any], missing: List[str],
                     excerpts: str) -> Dict[str, str]:
    keys = ", ".join(f'"{k}": "<or empty string>"' for k in missing)
    prompt = (
        f"Paper title: {draft['provenance'].get('paper_title','')}\n\n"
        f"Excerpts from the paper's full text:\n{excerpts}\n\n"
        f"From THESE EXCERPTS ONLY, extract the following (empty string if "
        f"not stated):\n{{{keys}}}"
    )
    resp = client.call(system=_SYSTEM, user=prompt, max_tokens=400,
                       temperature=0.0, json_mode=True)
    if not getattr(resp, "success", False):
        raise EvaluationError(f"enrich: LLM call failed on {draft['id']}")
    text = getattr(resp, "text", "") or ""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        parsed = json.loads(m.group(0) if m else text)
    except Exception as e:
        raise EvaluationError(
            f"enrich: unparseable LLM output on {draft['id']}: {text[:100]!r}"
        ) from e
    return {k: str(parsed.get(k) or "").strip()[:600] for k in missing}


def enrich(drafts_path: str) -> Dict[str, int]:
    from claude_provider import get_claude_client
    client = get_claude_client()
    if client is None:
        raise EvaluationError("no LLM client configured")

    rows: List[Dict[str, Any]] = []
    with open(drafts_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    stats = {"already_full": 0, "enriched": 0, "partial": 0, "no_html": 0}
    for k, d in enumerate(rows, 1):
        missing = [f for f in TUPLE_FIELDS if not str(d.get(f, "")).strip()]
        if not missing:
            stats["already_full"] += 1
            continue
        aid = (d.get("provenance") or {}).get("arxiv_id") or ""
        text = _fetch_html_text(aid) if aid else None
        time.sleep(_GAP_S)
        if text is None:
            stats["no_html"] += 1
            print(f"  [{k:02d}] {d['id']}: no arXiv HTML — left for human "
                  f"(missing: {','.join(missing)})")
            continue
        got = _extract_missing(client, d, missing, _relevant_excerpts(text))
        filled = [f for f, v in got.items() if v]
        for f_name, v in got.items():
            if v:
                d[f_name] = v
        prov = d.setdefault("provenance", {})
        prov["enriched_from"] = "arxiv-html"
        prov["enriched_fields"] = filled
        still = [f for f in TUPLE_FIELDS if not str(d.get(f, "")).strip()]
        if still:
            stats["partial"] += 1
            print(f"  [{k:02d}] {d['id']}: filled {filled or 'nothing'}; "
                  f"still empty: {','.join(still)}")
        else:
            stats["enriched"] += 1
            print(f"  [{k:02d}] {d['id']}: completed via full text "
                  f"({','.join(filled)})")

    with open(drafts_path, "w", encoding="utf-8") as f:
        for d in rows:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"\nalready full: {stats['already_full']}, completed: "
          f"{stats['enriched']}, still partial: {stats['partial']}, "
          f"no HTML: {stats['no_html']}")
    print("Partial / no-HTML drafts need the human to fill remaining fields "
          "from the paper (promote validation requires all 7 fields).")
    return stats


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="evaluation.enrich_drafts",
                                description=__doc__)
    p.add_argument("--drafts",
                   default="data/crossval/published_drafts.jsonl")
    args = p.parse_args(argv)
    try:
        enrich(args.drafts)
        return 0
    except EvaluationError as e:
        print(f"EVALUATION ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
