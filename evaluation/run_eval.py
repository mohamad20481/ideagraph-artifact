"""
evaluation/run_eval.py — CLI orchestrator for the evaluation harness.

Subcommands (each reads/writes plain JSON/JSONL so every step is inspectable
and resumable):

  validate   Check a CrossVal JSONL file (schema, provenance, duplicates).
  blind      Produce a blinded view + unblinding key from a CrossVal file.
  judge      Run the judge panel over a blinded view -> ratings JSON.
  analyze    Unblind ratings, compute Nov-A/Feas-A/NFT per source group,
             run per-dimension non-inferiority TOST (generated vs published),
             and Div-Pair for the generated pool. Emits a markdown report.
  diversity  DI / Div-Pair over a plain ideas JSON file (list of idea dicts).

Example end-to-end:
  python -m evaluation.run_eval validate  data/crossval.jsonl
  python -m evaluation.run_eval blind     data/crossval.jsonl --seed 42 --out-dir out/
  python -m evaluation.run_eval judge     out/blinded.json --out out/ratings.json
  python -m evaluation.run_eval analyze   data/crossval.jsonl out/blinded_key.json \
      out/ratings.json --delta 0.3 --alpha 0.05 --out out/report.md
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List

from evaluation import EvaluationError


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def cmd_validate(args) -> int:
    from evaluation.crossval import load_jsonl
    entries = load_jsonl(args.benchmark)
    n_gen = sum(1 for e in entries if e.source == "generated")
    n_pub = sum(1 for e in entries if e.source == "published")
    print(f"OK: {len(entries)} entries ({n_gen} generated, {n_pub} published), "
          f"{len({e.topic for e in entries})} topics")
    return 0


def cmd_blind(args) -> int:
    import os
    from evaluation.crossval import load_jsonl, blind
    entries = load_jsonl(args.benchmark)
    view, key = blind(entries, seed=args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    view_path = os.path.join(args.out_dir, "blinded.json")
    key_path = os.path.join(args.out_dir, "blinded_key.json")
    _save_json(view, view_path)
    _save_json(key, key_path)
    print(f"blinded view -> {view_path} ({len(view)} items, seed={args.seed})")
    print(f"unblinding key -> {key_path}  (keep away from judges)")
    return 0


def cmd_judge(args) -> int:
    import os
    from evaluation.judges import build_panel, default_panel
    view = _load_json(args.blinded)
    if args.panel:
        panel = build_panel(
            [s.strip() for s in args.panel.split(",") if s.strip()],
            ping=not args.no_ping,
        )
    else:
        panel = default_panel()
    names = [j.name for j in panel]
    if len(set(names)) != len(names):
        raise EvaluationError(f"duplicate judge names: {names}")

    # Checkpointed, resumable loop: every completed rating is persisted
    # immediately, so an interruption at call N costs zero completed work.
    per_item: Dict[str, Any] = {}
    if os.path.exists(args.out):
        try:
            per_item = _load_json(args.out).get("per_item", {})
            already = sum(len(v) for v in per_item.values())
            print(f"resuming: {already} ratings already on disk")
        except Exception:
            per_item = {}
    # Documented skips: "blind_id:judge_name" pairs that persistently fail
    # provider-side. They are recorded in the output for disclosure — the
    # affected item is aggregated from the remaining judges, and analyze
    # must acknowledge each skip via --allow-missing.
    skips = {s.strip() for s in (args.skip or "").split(",") if s.strip()}
    skipped_log: List[str] = []

    total = len(view) * len(panel)
    done = sum(len(v) for v in per_item.values())
    print(f"panel: {names} — {len(view)} items, {total} ratings total")
    if skips:
        print(f"documented skips: {sorted(skips)}")
    for item in view:
        bid = str(item.get("blind_id") or "")
        if not bid:
            raise EvaluationError("blinded item missing blind_id")
        per_item.setdefault(bid, {})
        for judge in panel:
            if judge.name in per_item[bid]:
                continue
            pair = f"{bid}:{judge.name}"
            if pair in skips:
                skipped_log.append(pair)
                done += 1
                print(f"  {done}/{total}  SKIPPED (documented): {pair}")
                continue
            per_item[bid][judge.name] = judge.rate(item, retries=2)
            done += 1
            _save_json({"per_item": per_item,
                        "skipped": sorted(set(skipped_log))}, args.out)
            print(f"  {done}/{total}  {judge.name} on {bid}")
    _save_json({"per_item": per_item,
                "skipped": sorted(set(skipped_log))}, args.out)
    n = sum(len(v) for v in per_item.values())
    print(f"ratings -> {args.out} ({n} ratings"
          f"{', ' + str(len(skipped_log)) + ' documented skips' if skipped_log else ''})")
    return 0


def cmd_analyze(args) -> int:
    from evaluation.crossval import load_jsonl
    from evaluation.judges import PanelRatings, DIMENSIONS
    from evaluation.metrics import nft, div_pair
    from evaluation.tost import noninferiority_test

    entries = load_jsonl(args.benchmark)
    key: Dict[str, str] = _load_json(args.key)
    raw = _load_json(args.ratings)
    ratings = PanelRatings(per_item=raw["per_item"])

    # Panel-completeness gate: every (item, judge) hole must be explicitly
    # acknowledged via --allow-missing (and should match the judge step's
    # documented skips). Silent incompleteness is refused.
    allow = {s.strip() for s in (args.allow_missing or "").split(",") if s.strip()}
    panel_names = ratings.judge_names()
    holes: List[str] = []
    for blind_id, jr in ratings.per_item.items():
        for jn in panel_names:
            if jn not in jr:
                holes.append(f"{blind_id}:{jn}")
    undeclared = [h for h in holes if h not in allow]
    if undeclared:
        raise EvaluationError(
            f"analyze: incomplete panel without acknowledgement: {undeclared[:5]}"
            f"{' …' if len(undeclared) > 5 else ''} — pass --allow-missing to "
            f"acknowledge documented skips"
        )

    by_id = {e.id: e for e in entries}
    groups: Dict[str, List[str]] = {"generated": [], "published": []}
    for blind_id, entry_id in key.items():
        if blind_id not in ratings.per_item:
            raise EvaluationError(f"analyze: no ratings for {blind_id}")
        if entry_id not in by_id:
            raise EvaluationError(f"analyze: key -> missing entry {entry_id}")
        groups[by_id[entry_id].source].append(blind_id)
    if not groups["generated"] or not groups["published"]:
        raise EvaluationError(
            f"analyze: need both groups rated "
            f"(generated={len(groups['generated'])}, "
            f"published={len(groups['published'])})"
        )

    lines: List[str] = ["# Evaluation report", ""]
    lines.append(f"- entries: {len(entries)} "
                 f"(generated={len(groups['generated'])}, "
                 f"published={len(groups['published'])})")
    lines.append(f"- judges: {ratings.judge_names()}")
    lines.append(f"- non-inferiority margin delta={args.delta}, alpha={args.alpha}")
    lines.append("")

    # Nov-A / Feas-A / NFT per group (unit scale).
    lines.append("## Aggregate scores (unit scale 0-1)")
    lines.append("")
    lines.append("| group | Nov-A | Feas-A | NFT |")
    lines.append("|---|---|---|---|")
    for grp, blind_ids in groups.items():
        unit = [ratings.unit_scores(b) for b in blind_ids]
        nov_a = sum(u["novelty"] for u in unit) / len(unit)
        feas_a = sum(u["feasibility"] for u in unit) / len(unit)
        nft_val = nft([{"nov": u["novelty"], "feas": u["feasibility"]} for u in unit])
        lines.append(f"| {grp} | {nov_a:.3f} | {feas_a:.3f} | {nft_val:.3f} |")
    lines.append("")

    # Per-dimension TOST non-inferiority, generated vs published (1-5 scale).
    lines.append(f"## Non-inferiority (generated vs published, delta={args.delta})")
    lines.append("")
    lines.append("| dimension | gen mean | pub mean | diff | p (one-sided) | NI? |")
    lines.append("|---|---|---|---|---|---|")
    for dim in DIMENSIONS:
        gen = ratings.dimension_sample(groups["generated"], dim)
        pub = ratings.dimension_sample(groups["published"], dim)
        r = noninferiority_test(gen, pub, delta=args.delta, alpha=args.alpha)
        lines.append(
            f"| {dim} | {r.mean_test:.2f} | {r.mean_ref:.2f} | "
            f"{r.diff:+.2f} | {r.p_value:.4f} | "
            f"{'YES' if r.established else 'no'} |"
        )
    lines.append("")

    # Div-Pair over the generated pool.
    gen_entries = [by_id[key[b]] for b in groups["generated"]]
    try:
        dp = div_pair([e.tuple_dict() for e in gen_entries])
        lines.append(f"## Div-Pair (generated pool): {dp:.3f}")
    except EvaluationError as e:
        lines.append(f"## Div-Pair: FAILED ({e})")
    lines.append("")

    report = "\n".join(lines)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"report -> {args.out}")
    print(report)
    return 0


def cmd_diversity(args) -> int:
    from evaluation.metrics import diversity_index
    ideas = _load_json(args.ideas)
    if not isinstance(ideas, list):
        raise EvaluationError("diversity: ideas file must be a JSON list")
    di = diversity_index(ideas)
    print(f"DI over {len(ideas)} ideas: {di:.4f}")
    return 0


def main(argv: List[str] = None) -> int:
    p = argparse.ArgumentParser(prog="evaluation.run_eval", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate");   v.add_argument("benchmark")
    b = sub.add_parser("blind");      b.add_argument("benchmark")
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--out-dir", default="out")
    j = sub.add_parser("judge");      j.add_argument("blinded")
    j.add_argument("--out", default="out/ratings.json")
    j.add_argument("--panel", default="",
                   help='comma-separated judge specs, e.g. '
                        '"kimi:moonshot-v1-32k,kimi:kimi-k2.5"')
    j.add_argument("--no-ping", action="store_true")
    j.add_argument("--skip", default="",
                   help='documented "blind_id:judge" pairs to skip')
    a = sub.add_parser("analyze")
    a.add_argument("benchmark"); a.add_argument("key"); a.add_argument("ratings")
    a.add_argument("--delta", type=float, default=0.3)
    a.add_argument("--alpha", type=float, default=0.05)
    a.add_argument("--allow-missing", default="",
                   help='acknowledged "blind_id:judge" holes (documented skips)')
    a.add_argument("--out", default="")
    d = sub.add_parser("diversity");  d.add_argument("ideas")

    args = p.parse_args(argv)
    try:
        return {
            "validate": cmd_validate, "blind": cmd_blind,
            "judge": cmd_judge, "analyze": cmd_analyze,
            "diversity": cmd_diversity,
        }[args.cmd](args)
    except EvaluationError as e:
        print(f"EVALUATION ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
