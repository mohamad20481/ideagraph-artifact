"""
models/idea.py - Research idea data model for the QD archive.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Behavioural descriptors (define the QD grid) ──────────────────────────────

METHODOLOGY_TYPES: List[str] = [
    "empirical_study",          # row 0 - experiments/benchmarks
    "theoretical_analysis",     # row 1 - proofs/mathematical models
    "system_design",            # row 2 - new architectures/frameworks
    "dataset_creation",         # row 3 - curating/annotating data
    "survey_meta_analysis",     # row 4 - literature synthesis
    "tool_library",             # row 5 - software/infrastructure
    "interdisciplinary_bridge", # row 6 - cross-domain synthesis
]

NOVELTY_LEVELS: List[str] = [
    "incremental",   # col 0 - modest improvement on existing work
    "moderate",      # col 1 - new combination of existing ideas
    "substantial",   # col 2 - genuinely new research direction
]

# Reverse-lookup tables built once at import. Replaces O(n) list.index()
# calls in the analytics chart-build hot path and in Idea.method_idx /
# Idea.novelty_idx (which are called per-idea during ideation).
METHODOLOGY_TYPE_TO_IDX: Dict[str, int] = {v: i for i, v in enumerate(METHODOLOGY_TYPES)}
NOVELTY_LEVEL_TO_IDX: Dict[str, int] = {v: i for i, v in enumerate(NOVELTY_LEVELS)}


# ── Idea dataclass ─────────────────────────────────────────────────────────────
# slots=True: ~40-50% memory reduction per instance (no per-instance __dict__)
# and faster attribute access. Critical for the QD ideation loop, which can
# instantiate thousands of Ideas per pipeline run.
@dataclass(slots=True)
class Idea:
    # Core 7 fields from the IdeaGraph paper (Appendix N)
    title: str
    motivation: str
    method: str
    hypothesis: str
    resources: str
    expected_outcome: str
    risk_assessment: str

    # Provenance
    source_strategy: str = ""          # "A", "B", or "C"

    # QD behavioural coordinates
    methodology_type: Optional[str] = None   # must be in METHODOLOGY_TYPES
    novelty_level: Optional[str] = None      # must be in NOVELTY_LEVELS

    # Probe results
    probe_scores: Optional[Dict[str, float]] = None
    probe_passed: bool = False
    quality_score: float = 0.0

    # Debate results (populated after tournament)
    debate_score: float = 0.0
    debate_history: List[Dict[str, Any]] = field(default_factory=list)
    debate_rank: Optional[int] = None

    # Pipeline-internal: cached (title_tokens, method_tokens, title_simhash, method_simhash)
    # populated by _is_duplicate() to avoid double-compute in _register_archived_title().
    # Declared here because @dataclass(slots=True) blocks dynamic attribute assignment.
    _dedup_cache: Optional[Any] = None

    # Evolution tracking
    generation: int = 0                     # 0 = original, 1+ = refined
    parent_title: Optional[str] = None      # title of the idea this was refined from

    # ── Execution-aware revision (closes the probe → archive feedback loop) ──
    # Populated by agents.execution_revisor when an idea passes the probe stage.
    # The blended posterior overwrites quality_score for archive ranking, but
    # we keep both around so the UI can show the delta and explain the move.
    # Declared explicitly because @dataclass(slots=True) blocks dynamic attrs.
    probe_quality: Optional[float] = None        # original probe-only quality (pre-revision)
    execution_signal: Optional[float] = None     # 0..1, tiny-experiment feasibility
    execution_trust: Optional[float] = None      # 0..1, weight given to exec vs probe
    execution_delta: Optional[float] = None      # blended - probe (signed)
    execution_meta: Optional[Dict[str, Any]] = None  # metric_name, CI, failures, cost

    # ── Provenance (optional, for the Provenance tab + CHI/FAccT study) ──────
    # Pipeline code can opt in via idea_provenance.attach_provenance(idea, ...)
    # to record seed_papers, target_cell, prompt_template_id, etc. The
    # extractor backfills from existing fields when this is missing, so the
    # Provenance tab works even on legacy ideas.
    provenance: Optional[Dict[str, Any]] = None

    # ── Helpers ──────────────────────────────────────────────────────────────
    def method_idx(self) -> int:
        # O(1) dict lookup instead of O(n) list.index() call.
        return METHODOLOGY_TYPE_TO_IDX.get(self.methodology_type or "", 0)

    def novelty_idx(self) -> int:
        return NOVELTY_LEVEL_TO_IDX.get(self.novelty_level or "", 0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "motivation": self.motivation,
            "method": self.method,
            "hypothesis": self.hypothesis,
            "resources": self.resources,
            "expected_outcome": self.expected_outcome,
            "risk_assessment": self.risk_assessment,
            "source_strategy": self.source_strategy,
            "methodology_type": self.methodology_type,
            "novelty_level": self.novelty_level,
            "probe_scores": self.probe_scores,
            "probe_passed": self.probe_passed,
            "quality_score": self.quality_score,
            "debate_score": self.debate_score,
            "debate_history": self.debate_history,
            "debate_rank": self.debate_rank,
            "generation": self.generation,
            "parent_title": self.parent_title,
            "probe_quality": self.probe_quality,
            "execution_signal": self.execution_signal,
            "execution_trust": self.execution_trust,
            "execution_delta": self.execution_delta,
            "execution_meta": self.execution_meta,
            "provenance": self.provenance,
        }

    def to_prompt_str(self) -> str:
        """Compact text representation suitable for passing to execution probes."""
        return (
            f"Title: {self.title}\n"
            f"Motivation: {self.motivation}\n"
            f"Method: {self.method}\n"
            f"Hypothesis: {self.hypothesis}\n"
            f"Resources: {self.resources}\n"
            f"Expected Outcome: {self.expected_outcome}\n"
            f"Risk Assessment: {self.risk_assessment}"
        )
