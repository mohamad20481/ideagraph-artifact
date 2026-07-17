"""
evaluation/export_generated.py — export REAL pipeline-run ideas from the
IdeaGraph results database into the CrossVal 7-field schema.

This fills the GENERATED half of a CrossVal benchmark from actual saved runs
(the `results` table written by the app's pipeline runners). Nothing here
invents content:

  * Direct field mapping only:
        motivation        <- idea.motivation
        hypothesis        <- idea.hypothesis
        method_sketch     <- idea.method
        expected_outcome  <- idea.expected_outcome
    The remaining tuple fields (dataset, metrics, baselines) have NO
    dedicated slot in the pipeline's idea schema. They are exported EMPTY
    and the entry is quarantined into a "needs completion" file.
  * Optional `--normalize`: an LLM pass that may ONLY extract dataset /
    metrics / baselines from the idea's own text (title, motivation, method,
    hypothesis, resources, expected_outcome). It is instructed to return ""
    when the idea does not state something. Every normalized entry records
    provenance {"normalizer": "llm-draft", "human_verified": false} — a
    HUMAN must review these before the benchmark is used, and the paper must
    describe this normalization step truthfully.

Selection policy (documented so the paper can state it):
  * runs grouped by lowercased topic; the LATEST run per distinct topic wins
    (repeated experiments on the same topic don't duplicate entries);
  * within a run, ideas sorted by quality_score desc; --top-per-topic taken;
  * --min-quality filters junk; --exclude-topic drops test topics;
  * hard cap --max-entries.

Usage:
    python -m evaluation.export_generated --dry-run
    python -m evaluation.export_generated --out-dir data/crossval
    python -m evaluation.export_generated --out-dir data/crossval --normalize 5
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from typing import Any, Dict, List, Optional, Tuple

from evaluation import EvaluationError
from evaluation.crossval import CrossValEntry, TUPLE_FIELDS, save_jsonl

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "ideagraph.db")

_DIRECT_MAP = {
    "motivation": "motivation",
    "hypothesis": "hypothesis",
    "method_sketch": "method",
    "expected_outcome": "expected_outcome",
}
_LLM_FIELDS = ("dataset", "metrics", "baselines")


def load_runs(db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """All saved runs (read-only), newest last."""
    if not os.path.exists(db_path):
        raise EvaluationError(f"results DB not found: {db_path}")
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT id, user_id, topic, coverage, ideas_count, results_json, "
            "created_at FROM results ORDER BY id"
        ).fetchall()
    finally:
        con.close()
    out = []
    for rid, uid, topic, cov, n, rj, created in rows:
        try:
            results = json.loads(rj)
        except Exception:
            continue
        out.append({
            "run_id": rid, "user_id": uid, "topic": (topic or "").strip(),
            "coverage": cov, "ideas_count": n, "created_at": created,
            "ideas": results.get("ideas") or [],
        })
    return out


def select_ideas(
    runs: List[Dict[str, Any]],
    top_per_topic: int = 1,
    min_quality: float = 0.0,
    max_entries: int = 50,
    exclude_topic: Optional[str] = None,
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """[(run, idea)] under the documented selection policy."""
    ex_re = re.compile(exclude_topic, re.IGNORECASE) if exclude_topic else None
    latest_by_topic: Dict[str, Dict[str, Any]] = {}
    for run in runs:                       # ordered by id → later overwrites
        key = re.sub(r"\s+", " ", run["topic"].lower()).strip()
        if not key or not run["ideas"]:
            continue
        if ex_re and ex_re.search(run["topic"]):
            continue
        latest_by_topic[key] = run

    picked: List[Tuple[Dict[str, Any], Dict[str, Any], int]] = []
    for key in sorted(latest_by_topic):
        run = latest_by_topic[key]
        # carry each idea's position in the run's original ideas array so
        # entry ids stay STABLE across re-runs with different selection knobs
        ideas = sorted(
            ((pos, i) for pos, i in enumerate(run["ideas"])
             if isinstance(i, dict)),
            key=lambda t: float(t[1].get("quality_score") or 0.0),
            reverse=True,
        )
        taken = 0
        for pos, idea in ideas:
            if taken >= top_per_topic:
                break
            if float(idea.get("quality_score") or 0.0) < min_quality:
                continue
            if not all(str(idea.get(src) or "").strip()
                       for src in ("title", "motivation", "method",
                                   "hypothesis", "expected_outcome")):
                continue                    # core content must be real
            picked.append((run, idea, pos))
            taken += 1
    return picked[:max_entries]


def to_entry_dict(run: Dict[str, Any], idea: Dict[str, Any],
                  seq: int) -> Dict[str, Any]:
    """One idea → CrossVal entry dict (dataset/metrics/baselines empty).
    `seq` is the idea's position in the run's original ideas array, so the
    id is stable no matter how the selection knobs change."""
    d: Dict[str, Any] = {
        "id": f"gen-{run['run_id']:04d}-i{seq:02d}",
        "source": "generated",
        "topic": run["topic"],
        "title": str(idea.get("title") or "")[:200],
        "provenance": {
            "run_id": run["run_id"],
            "run_created_at": run["created_at"],
            "source_strategy": idea.get("source_strategy") or "",
            "quality_score": idea.get("quality_score"),
            "methodology_type": idea.get("methodology_type") or "",
            "exporter": "evaluation.export_generated",
        },
    }
    for dst, src in _DIRECT_MAP.items():
        d[dst] = str(idea.get(src) or "").strip()
    for f_name in _LLM_FIELDS:
        d[f_name] = ""
    # keep the raw resources text for the normalizer / human reviewer
    d["provenance"]["resources_text"] = str(idea.get("resources") or "")[:800]
    return d


# ── Optional LLM normalization (extract-only, human-verified later) ──────────

_NORM_SYSTEM = (
    "You extract structured facts from a research idea's OWN text. You may "
    "not add, infer, or invent anything the text does not state. If the "
    "text does not name a dataset / metric / baseline, return an empty "
    "string for that key. Output ONLY valid JSON."
)


def _norm_prompt(d: Dict[str, Any]) -> str:
    prov = d.get("provenance") or {}
    return (
        "Idea text:\n"
        f"  title: {d.get('title','')}\n"
        f"  motivation: {d.get('motivation','')}\n"
        f"  method: {d.get('method_sketch','')}\n"
        f"  hypothesis: {d.get('hypothesis','')}\n"
        f"  resources: {prov.get('resources_text','')}\n"
        f"  expected_outcome: {d.get('expected_outcome','')}\n\n"
        "From THIS TEXT ONLY, extract:\n"
        '{"dataset": "<datasets the text names, comma-separated, or \\"\\">",\n'
        ' "metrics": "<evaluation metrics the text names, or \\"\\">",\n'
        ' "baselines": "<baselines/comparisons the text names, or \\"\\">"}'
    )


def normalize_entries(entries: List[Dict[str, Any]], limit: int) -> int:
    """LLM-draft dataset/metrics/baselines for up to `limit` entries, in
    place. Returns how many entries were actually modified. Raises
    EvaluationError if no client is configured."""
    try:
        from claude_provider import get_claude_client
        client = get_claude_client()
    except Exception as e:
        raise EvaluationError(f"normalize: provider import failed: {e}") from e
    if client is None:
        raise EvaluationError("normalize: no LLM client configured")
    changed = 0
    for d in entries:
        if changed >= limit:
            break
        if all(d.get(f) for f in _LLM_FIELDS):
            continue
        resp = client.call(
            system=_NORM_SYSTEM, user=_norm_prompt(d),
            max_tokens=300, temperature=0.0, json_mode=True,
        )
        if not getattr(resp, "success", False):
            raise EvaluationError(
                f"normalize: LLM call failed on {d['id']}: "
                f"{getattr(resp, 'text', '')[:160]}"
            )
        text = getattr(resp, "text", "") or ""
        m = re.search(r"\{.*\}", text, re.DOTALL)
        try:
            parsed = json.loads(m.group(0) if m else text)
        except Exception as e:
            raise EvaluationError(
                f"normalize: unparseable LLM output on {d['id']}: {text[:120]!r}"
            ) from e
        for f_name in _LLM_FIELDS:
            if not d.get(f_name):
                d[f_name] = str(parsed.get(f_name) or "").strip()[:400]
        d["provenance"]["normalizer"] = "llm-draft"
        d["provenance"]["human_verified"] = False
        changed += 1
    return changed


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="evaluation.export_generated",
                                description=__doc__)
    p.add_argument("--db", default=DB_PATH)
    p.add_argument("--out-dir", default=os.path.join("data", "crossval"))
    p.add_argument("--top-per-topic", type=int, default=1)
    p.add_argument("--min-quality", type=float, default=0.0)
    p.add_argument("--max-entries", type=int, default=50)
    p.add_argument("--exclude-topic", default=r"test topic|^test\b",
                   help="regex of topics to drop (default filters test runs)")
    p.add_argument("--normalize", type=int, default=0, metavar="N",
                   help="LLM-draft dataset/metrics/baselines for up to N "
                        "entries (extract-only; flagged for human review)")
    p.add_argument("--max-complete", type=int, default=0, metavar="N",
                   help="keep only the N highest-quality COMPLETE entries "
                        "(0 = keep all)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    runs = load_runs(args.db)
    picked = select_ideas(
        runs, top_per_topic=args.top_per_topic, min_quality=args.min_quality,
        max_entries=args.max_entries, exclude_topic=args.exclude_topic,
    )
    print(f"runs in DB: {len(runs)} | distinct usable topics picked from: "
          f"{len(picked)} entries selected")
    if args.dry_run:
        for run, idea, _pos in picked[:120]:
            print(f"  run {run['run_id']:>4} q={idea.get('quality_score', 0):.2f} "
                  f"| {run['topic'][:45]:45s} | {str(idea.get('title'))[:50]}")
        return 0

    entries = [to_entry_dict(run, idea, seq=pos)
               for (run, idea, pos) in picked]

    if args.normalize > 0:
        n = normalize_entries(entries, args.normalize)
        print(f"normalized (LLM-draft, needs human verification): {n}")

    complete: List[CrossValEntry] = []
    incomplete: List[Dict[str, Any]] = []
    for d in entries:
        try:
            e = CrossValEntry(**{k: d[k] for k in
                                 ("id", "source", "topic", "title", "provenance")},
                              **{f: d[f] for f in TUPLE_FIELDS})
            e.validate()
            complete.append(e)
        except EvaluationError:
            d["_missing"] = [f for f in TUPLE_FIELDS if not d.get(f)]
            incomplete.append(d)

    # Cap the benchmark at the N highest-quality complete entries (policy:
    # "the --max-complete highest-quality entries among those whose tuples
    # completed"; 0 = keep all).
    if args.max_complete > 0 and len(complete) > args.max_complete:
        complete.sort(
            key=lambda e: float((e.provenance or {}).get("quality_score") or 0.0),
            reverse=True,
        )
        dropped = complete[args.max_complete:]
        complete = complete[:args.max_complete]
        print(f"capped complete entries to top {args.max_complete} by "
              f"quality (dropped {len(dropped)})")

    os.makedirs(args.out_dir, exist_ok=True)
    if complete:
        out = os.path.join(args.out_dir, "crossval_generated.jsonl")
        save_jsonl(complete, out)
        print(f"complete entries -> {out} ({len(complete)})")
    if incomplete:
        q = os.path.join(args.out_dir, "generated_needs_completion.jsonl")
        with open(q, "w", encoding="utf-8") as f:
            for d in incomplete:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"NEEDS COMPLETION (dataset/metrics/baselines) -> {q} "
              f"({len(incomplete)}) — complete via --normalize + human review")
    return 0


if __name__ == "__main__":
    sys.exit(main())
