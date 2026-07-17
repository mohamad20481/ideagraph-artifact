"""
evaluation/crossval.py — CrossVal benchmark schema, IO, and blinding.

A CrossVal benchmark mixes GENERATED ideas with ideas EXTRACTED from
venue-published papers, each normalized to the same 7-field tuple, so that
blind judges cannot use surface format to tell them apart.

The 7 fields (all REQUIRED, all non-empty):
    motivation, hypothesis, method_sketch, dataset, metrics, baselines,
    expected_outcome

IMPORTANT — what this module is and is not:
  * It is the schema, validation, storage, and blinding machinery.
  * It is NOT the benchmark. The benchmark is DATA that must be produced by
    real work: generated entries from actual pipeline runs, and published
    entries extracted from real papers BY PEOPLE (LLM-assisted extraction is
    allowed as a draft step, but every published entry must carry provenance
    — paper title, venue, year — and be human-verified before use).
    This module will refuse to blind a published entry without provenance.

Blinding: `blind(entries, seed)` strips source + provenance, shuffles
deterministically, and returns (blinded_view, key) where `key` maps blind_id
-> original entry id. Judges see ONLY the blinded view; the key stays with
the experimenter for unblinding after ratings are collected.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from evaluation import EvaluationError

TUPLE_FIELDS = (
    "motivation", "hypothesis", "method_sketch", "dataset",
    "metrics", "baselines", "expected_outcome",
)

SOURCES = ("generated", "published")


@dataclass
class CrossValEntry:
    """One benchmark item: a 7-field idea tuple + bookkeeping."""
    id: str
    source: str                          # "generated" | "published"
    topic: str
    motivation: str
    hypothesis: str
    method_sketch: str
    dataset: str
    metrics: str
    baselines: str
    expected_outcome: str
    title: str = ""                      # optional display title
    provenance: Dict[str, Any] = field(default_factory=dict)
    # for published: {"paper_title":..., "venue":..., "year":..., "extractor":...}
    # for generated: {"run_id":..., "strategy":..., "model":...}

    def validate(self) -> None:
        if not self.id or not str(self.id).strip():
            raise EvaluationError("CrossValEntry: empty id")
        if self.source not in SOURCES:
            raise EvaluationError(
                f"CrossValEntry {self.id}: source must be one of {SOURCES}, "
                f"got {self.source!r}"
            )
        if not self.topic or not str(self.topic).strip():
            raise EvaluationError(f"CrossValEntry {self.id}: empty topic")
        for f_name in TUPLE_FIELDS:
            v = getattr(self, f_name)
            if not v or not str(v).strip():
                raise EvaluationError(
                    f"CrossValEntry {self.id}: required field '{f_name}' is empty"
                )
        if self.source == "published":
            prov = self.provenance or {}
            for req in ("paper_title", "venue", "year"):
                if not prov.get(req):
                    raise EvaluationError(
                        f"CrossValEntry {self.id}: published entry missing "
                        f"provenance['{req}'] — published ideas must be "
                        f"traceable to a real paper"
                    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def tuple_dict(self) -> Dict[str, str]:
        """Just the 7 fields (+title if set) — the judge-visible content."""
        d = {f_name: getattr(self, f_name) for f_name in TUPLE_FIELDS}
        if self.title:
            d["title"] = self.title
        return d


def from_dict(raw: Dict[str, Any]) -> CrossValEntry:
    known = {f_name: raw.get(f_name, "") for f_name in TUPLE_FIELDS}
    entry = CrossValEntry(
        id=str(raw.get("id", "")),
        source=str(raw.get("source", "")),
        topic=str(raw.get("topic", "")),
        title=str(raw.get("title", "") or ""),
        provenance=dict(raw.get("provenance") or {}),
        **known,
    )
    entry.validate()
    return entry


# ── JSONL IO ─────────────────────────────────────────────────────────────────

def save_jsonl(entries: Sequence[CrossValEntry], path: str) -> None:
    if not entries:
        raise EvaluationError("save_jsonl: nothing to save")
    seen = set()
    for e in entries:
        e.validate()
        if e.id in seen:
            raise EvaluationError(f"save_jsonl: duplicate id {e.id!r}")
        seen.add(e.id)
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e.to_dict(), ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> List[CrossValEntry]:
    entries: List[CrossValEntry] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                raise EvaluationError(f"{path}:{line_no}: invalid JSON: {e}") from e
            try:
                entries.append(from_dict(raw))
            except EvaluationError as e:
                raise EvaluationError(f"{path}:{line_no}: {e}") from e
    if not entries:
        raise EvaluationError(f"{path}: no entries")
    return entries


# ── Blinding ─────────────────────────────────────────────────────────────────

def blind(
    entries: Sequence[CrossValEntry],
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Strip identity, shuffle deterministically.

    Returns (blinded_view, key):
      blinded_view: [{"blind_id": "b001", <7 fields>, "topic": ...}, ...]
                    — NO source, NO provenance, NO original id.
      key:          {blind_id: original_entry_id} — keep it away from judges.
    """
    if not entries:
        raise EvaluationError("blind: no entries")
    for e in entries:
        e.validate()
    order = list(range(len(entries)))
    random.Random(seed).shuffle(order)
    view: List[Dict[str, Any]] = []
    key: Dict[str, str] = {}
    for pos, idx in enumerate(order):
        e = entries[idx]
        blind_id = f"b{pos + 1:03d}"
        item = {"blind_id": blind_id, "topic": e.topic}
        item.update({f_name: getattr(e, f_name) for f_name in TUPLE_FIELDS})
        # NOTE: title intentionally excluded — generated vs published titles
        # have telltale style differences.
        view.append(item)
        key[blind_id] = e.id
    return view, key


def unblind_scores(
    scores_by_blind_id: Dict[str, Any],
    key: Dict[str, str],
    entries: Sequence[CrossValEntry],
) -> Dict[str, Dict[str, Any]]:
    """Re-attach scores to original entries: {entry_id: {"source":..., "scores":...}}"""
    by_id = {e.id: e for e in entries}
    out: Dict[str, Dict[str, Any]] = {}
    for blind_id, scores in scores_by_blind_id.items():
        if blind_id not in key:
            raise EvaluationError(f"unblind: unknown blind_id {blind_id!r}")
        entry_id = key[blind_id]
        if entry_id not in by_id:
            raise EvaluationError(f"unblind: key points to missing entry {entry_id!r}")
        out[entry_id] = {"source": by_id[entry_id].source, "scores": scores}
    return out


# ── Extraction template (a DRAFTING aid, not a data factory) ─────────────────

EXTRACTION_PROMPT_TEMPLATE = """\
You are extracting the core research idea of a published paper into a fixed
7-field tuple for a blind evaluation benchmark. Use ONLY what the paper
states; do not embellish or modernize. Output JSON with exactly these keys:

{{
  "motivation": "<the problem/why, 1-3 sentences, paper's own framing>",
  "hypothesis": "<the falsifiable core claim>",
  "method_sketch": "<the approach in 2-4 sentences, no results>",
  "dataset": "<datasets used, comma-separated>",
  "metrics": "<evaluation metrics, comma-separated>",
  "baselines": "<baselines compared against, comma-separated>",
  "expected_outcome": "<what success looks like, WITHOUT the actual numbers>"
}}

PAPER TITLE: {title}
VENUE/YEAR: {venue} {year}
ABSTRACT (and intro excerpt if provided):
{abstract}
"""
# Every LLM-drafted extraction MUST be reviewed and corrected by a person
# before entering the benchmark; record the extractor ("llm+<name>") in
# provenance so the paper can report the protocol truthfully.
