"""
iqd_controls.py — Interactive Quality-Diversity (iQD) co-steering.

Background (Extension 5 from the paper roadmap):
    Pure-autonomous QD loops produce mathematically diverse ideas that
    may not match real-world lab constraints, funding priorities, or
    human intuition. Interactive iQD lets a researcher sculpt the
    archive: freeze high-value cells (protect them from being
    overwritten by future generations), prune low-feasibility cells
    (exclude them from future generation), and force directed
    crossovers between two parent ideas under a user-supplied constraint.

State model:
    A small dataclass `IQDState` holds frozen/pruned cell sets plus the
    selected-parents queue (up to 2). It serializes cleanly to JSON
    for persistence in session_state or db. The functions below all
    take an IQDState explicitly so they're pure / testable in isolation;
    the Streamlit panel `render_iqd_panel` wires session_state to it.

Archive shape:
    The codebase's QD grid is 7 (methodology) × 3 (novelty) = 21 cells.
    `cell_for_idea(idea)` returns (methodology_idx, novelty_idx) for any
    idea dict; cells where methodology or novelty are unknown map to
    (-1, -1) (parked off-grid).

Directed crossover:
    `directed_crossover(parent_a, parent_b, constraint, llm_client)`
    asks the LLM to synthesize a child idea that incorporates BOTH
    parents AND satisfies the user's constraint string (e.g. "must be
    runnable on a single A100 in under 24h", "must not use proprietary
    data"). Returns a fresh idea dict in the same shape as the rest of
    the pipeline so it slots straight into `st.session_state.results`.

Public API:
    IQDState                                          → dataclass
    cell_for_idea(idea)                               → Tuple[int, int]
    build_archive(ideas)                              → Dict[cell, idea]
    freeze_cell / unfreeze_cell / is_frozen           → state mutations
    prune_cell  / unprune_cell  / is_pruned           → state mutations
    select_parent / clear_parents                     → crossover queue
    directed_crossover(a, b, constraint, llm_client)  → new idea dict
    apply_iqd_generation_filter(state, candidate_cells)
                                                      → filtered cell list
    render_iqd_panel(st, ideas, archive_id, llm_client)
                                                      → Streamlit UI
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from models.idea import (
        METHODOLOGY_TYPES, NOVELTY_LEVELS,
        METHODOLOGY_TYPE_TO_IDX, NOVELTY_LEVEL_TO_IDX,
    )
    _HAS_IDEA_MODEL = True
except Exception:
    METHODOLOGY_TYPES = [
        "empirical_study", "theoretical_analysis", "system_design",
        "dataset_creation", "survey_meta_analysis", "tool_library",
        "interdisciplinary_bridge",
    ]
    NOVELTY_LEVELS = ["incremental", "moderate", "substantial"]
    METHODOLOGY_TYPE_TO_IDX = {v: i for i, v in enumerate(METHODOLOGY_TYPES)}
    NOVELTY_LEVEL_TO_IDX = {v: i for i, v in enumerate(NOVELTY_LEVELS)}
    _HAS_IDEA_MODEL = False


# ── State dataclass ────────────────────────────────────────────────────────

@dataclass
class IQDState:
    """Per-archive sculpting state. Lives in session_state keyed by
    `archive_id` (we use the result_id as the archive id, so each saved
    result has its own freeze/prune sets)."""
    frozen_cells: Set[Tuple[int, int]] = field(default_factory=set)
    pruned_cells: Set[Tuple[int, int]] = field(default_factory=set)
    selected_parents: List[int] = field(default_factory=list)  # idea indices

    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "frozen_cells": [list(c) for c in self.frozen_cells],
            "pruned_cells": [list(c) for c in self.pruned_cells],
            "selected_parents": list(self.selected_parents),
        }

    @classmethod
    def from_jsonable(cls, data: Any) -> "IQDState":
        if not isinstance(data, dict):
            return cls()
        s = cls()
        for c in data.get("frozen_cells") or []:
            if isinstance(c, (list, tuple)) and len(c) == 2:
                try:
                    s.frozen_cells.add((int(c[0]), int(c[1])))
                except (TypeError, ValueError):
                    pass
        for c in data.get("pruned_cells") or []:
            if isinstance(c, (list, tuple)) and len(c) == 2:
                try:
                    s.pruned_cells.add((int(c[0]), int(c[1])))
                except (TypeError, ValueError):
                    pass
        for p in data.get("selected_parents") or []:
            try:
                s.selected_parents.append(int(p))
            except (TypeError, ValueError):
                pass
        return s


# ── Archive utilities ──────────────────────────────────────────────────────

def cell_for_idea(idea: Dict[str, Any]) -> Tuple[int, int]:
    """Return (methodology_idx, novelty_idx) for an idea. Unknown values
    map to -1 so they're trivially detectable as off-grid."""
    if not isinstance(idea, dict):
        return (-1, -1)
    m = METHODOLOGY_TYPE_TO_IDX.get(idea.get("methodology_type") or "", -1)
    n = NOVELTY_LEVEL_TO_IDX.get(idea.get("novelty_level") or "", -1)
    return (m, n)


def build_archive(ideas: List[Dict[str, Any]]) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """Pick the highest-quality idea per cell. Returns a dict mapping
    (mi, ni) → idea. Off-grid ideas (cell == (-1, -1)) are excluded so
    the heatmap only shows the 7×3 valid range."""
    out: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for idea in ideas or []:
        cell = cell_for_idea(idea)
        if cell[0] < 0 or cell[1] < 0:
            continue
        q = float(idea.get("quality_score") or 0.0)
        existing = out.get(cell)
        if existing is None:
            out[cell] = idea
        else:
            existing_q = float(existing.get("quality_score") or 0.0)
            if q > existing_q:
                out[cell] = idea
    return out


# ── State mutations ────────────────────────────────────────────────────────

def freeze_cell(state: IQDState, cell: Tuple[int, int]) -> None:
    """Freeze a cell — its current idea is protected from overwrite
    during future generation. Also unprunes the cell if it was pruned
    (freezing implies the user values that cell)."""
    state.frozen_cells.add(cell)
    state.pruned_cells.discard(cell)


def unfreeze_cell(state: IQDState, cell: Tuple[int, int]) -> None:
    state.frozen_cells.discard(cell)


def is_frozen(state: IQDState, cell: Tuple[int, int]) -> bool:
    return cell in state.frozen_cells


def prune_cell(state: IQDState, cell: Tuple[int, int]) -> None:
    """Prune a cell — exclude it from future generation (user has marked
    it as low-feasibility / not interesting). Also unfreezes."""
    state.pruned_cells.add(cell)
    state.frozen_cells.discard(cell)


def unprune_cell(state: IQDState, cell: Tuple[int, int]) -> None:
    state.pruned_cells.discard(cell)


def is_pruned(state: IQDState, cell: Tuple[int, int]) -> bool:
    return cell in state.pruned_cells


def select_parent(state: IQDState, idea_idx: int) -> None:
    """Add an idea index to the crossover-parents queue (max 2). If
    already at 2, replaces the oldest. Idempotent for already-queued."""
    if idea_idx in state.selected_parents:
        return
    state.selected_parents.append(int(idea_idx))
    if len(state.selected_parents) > 2:
        state.selected_parents = state.selected_parents[-2:]


def clear_parents(state: IQDState) -> None:
    state.selected_parents = []


def apply_iqd_generation_filter(
    state: IQDState,
    candidate_cells: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Remove pruned cells from a list of candidate cells the generator
    is about to pick from. Frozen cells stay in (the generator can
    still TRY them; the freeze logic in the pipeline must refuse to
    overwrite the stored idea).

    Used by the QD loop when iqd_controls is active. If `candidate_cells`
    contains no pruned cells, returns the input list unchanged.
    """
    if not state.pruned_cells:
        return list(candidate_cells)
    return [c for c in candidate_cells if c not in state.pruned_cells]


# ── Directed crossover ─────────────────────────────────────────────────────

_AUTOLOAD = object()


_CROSSOVER_SYSTEM = (
    "You are a research-idea synthesizer. You receive two PARENT "
    "research ideas and a USER CONSTRAINT. Produce one new CHILD "
    "idea that genuinely combines the two parents AND satisfies the "
    "constraint. The child must be a single coherent idea — not just "
    "a paragraph mentioning both parents. It should have a clear "
    "hypothesis the parents alone don't make. "
    "\n\nReply with strict JSON, no prose, no markdown fences:\n"
    "{\n"
    '  "title": "<10-words>",\n'
    '  "motivation": "<why this matters; cite the gap both parents leave>",\n'
    '  "method": "<concrete approach combining parents>",\n'
    '  "hypothesis": "<falsifiable claim>",\n'
    '  "resources": "<rough compute/time/data requirements>",\n'
    '  "expected_outcome": "<what success looks like>",\n'
    '  "risk_assessment": "<the main failure mode>",\n'
    '  "methodology_type": "<one of: empirical_study | theoretical_analysis | system_design | dataset_creation | survey_meta_analysis | tool_library | interdisciplinary_bridge>",\n'
    '  "novelty_level": "<one of: incremental | moderate | substantial>"\n'
    "}"
)


def _idea_block(idea: Dict[str, Any], label: str) -> str:
    return (
        f"=== {label} ===\n"
        f"Title:       {idea.get('title', '?')}\n"
        f"Motivation:  {idea.get('motivation', '?')}\n"
        f"Method:      {idea.get('method', '?')}\n"
        f"Hypothesis:  {idea.get('hypothesis', '?')}\n"
        f"Risks:       {idea.get('risk_assessment', '?')}\n"
        f"Methodology: {idea.get('methodology_type', '?')}\n"
        f"Novelty:     {idea.get('novelty_level', '?')}\n"
    )


def _parse_idea_json(raw: str) -> Optional[Dict[str, Any]]:
    """Tolerant JSON parser: strips markdown fences, trims to outer braces."""
    if not raw:
        return None
    s = raw.strip()
    # Strip ```json ... ``` fences.
    if s.startswith("```"):
        nl = s.find("\n")
        if nl > 0:
            s = s[nl + 1:]
        if s.endswith("```"):
            s = s[:-3].strip()
    a, b = s.find("{"), s.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        return json.loads(s[a:b + 1])
    except Exception:
        return None


def directed_crossover(
    parent_a: Dict[str, Any],
    parent_b: Dict[str, Any],
    constraint: str,
    llm_client: Any = _AUTOLOAD,
    max_tokens: int = 700,
    temperature: float = 0.7,
) -> Optional[Dict[str, Any]]:
    """Synthesize a child idea from two parents under a user constraint.

    Returns the new idea dict on success, or None on any failure (no
    client, LLM call failed, JSON parse failed). The caller appends the
    result to `st.session_state.results["ideas"]` so it integrates with
    the rest of the app.

    The result gets `source_strategy="I"` (Interactive crossover), so
    it shows up correctly in lineage views and sort modes.
    """
    if not isinstance(parent_a, dict) or not isinstance(parent_b, dict):
        return None
    if llm_client is _AUTOLOAD:
        try:
            from claude_provider import get_claude_client
            llm_client = get_claude_client()
        except Exception:
            llm_client = None
    if llm_client is None:
        return None

    user_prompt = (
        _idea_block(parent_a, "PARENT A")
        + "\n"
        + _idea_block(parent_b, "PARENT B")
        + f"\nUSER CONSTRAINT:\n{(constraint or '').strip() or 'No specific constraint.'}\n"
        + "\nSynthesize one CHILD idea that combines A and B and "
        "satisfies the constraint. JSON only."
    )

    try:
        resp = llm_client.call(
            system=_CROSSOVER_SYSTEM,
            user=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception:
        return None
    if not getattr(resp, "success", False):
        return None
    parsed = _parse_idea_json(getattr(resp, "text", "") or "")
    if not parsed:
        return None

    # Sanitize & stamp lineage so the rest of the app treats it like a
    # first-class idea.
    parsed.setdefault("title", "Untitled crossover")
    parsed.setdefault("motivation", "")
    parsed.setdefault("method", "")
    parsed.setdefault("hypothesis", "")
    parsed.setdefault("resources", "")
    parsed.setdefault("expected_outcome", "")
    parsed.setdefault("risk_assessment", "")
    # Coerce methodology / novelty to valid values, falling back to
    # parent A's values if the LLM produced something invalid.
    if parsed.get("methodology_type") not in METHODOLOGY_TYPES:
        parsed["methodology_type"] = (
            parent_a.get("methodology_type")
            or parent_b.get("methodology_type")
            or "system_design"
        )
    if parsed.get("novelty_level") not in NOVELTY_LEVELS:
        parsed["novelty_level"] = "moderate"

    parsed["source_strategy"] = "I"     # Interactive crossover marker
    parsed["quality_score"] = 0.0       # not yet scored
    parsed["generation"] = max(
        int(parent_a.get("generation") or 0),
        int(parent_b.get("generation") or 0),
    ) + 1
    # Lineage / provenance.
    meta = parsed.get("execution_meta") or {}
    meta["iqd_crossover"] = {
        "parent_a_title": parent_a.get("title"),
        "parent_b_title": parent_b.get("title"),
        "user_constraint": (constraint or "").strip(),
    }
    parsed["execution_meta"] = meta
    parsed["parent_title"] = (
        f"{parent_a.get('title', '?')} ⨯ {parent_b.get('title', '?')}"
    )

    return parsed


# ── Streamlit UI ───────────────────────────────────────────────────────────

def _get_state(st_module, archive_id: Any) -> IQDState:
    """Load IQDState from session_state, initializing if missing."""
    key = f"_iqd_state:{archive_id}"
    raw = st_module.session_state.get(key)
    if isinstance(raw, IQDState):
        return raw
    s = IQDState.from_jsonable(raw) if raw else IQDState()
    st_module.session_state[key] = s
    return s


def _save_state(st_module, archive_id: Any, state: IQDState) -> None:
    st_module.session_state[f"_iqd_state:{archive_id}"] = state


def render_iqd_panel(
    st_module,
    ideas: List[Dict[str, Any]],
    archive_id: Any = "current",
    llm_client: Any = _AUTOLOAD,
) -> None:
    """Draw the interactive-QD sculpting panel.

    Layout:
      - Archive heatmap (7×3 grid)        — every cell colored by quality,
                                            with ❄️ frozen / 🚫 pruned badges
                                            and freeze/prune/select buttons
      - Selected-parents queue            — shows up to 2 parents picked
      - Directed crossover form           — constraint input + 🧬 Synthesize
    """
    if not ideas:
        st_module.info(
            "🎛️ Interactive iQD is empty until you have ideas. Run a "
            "pipeline first."
        )
        return

    state = _get_state(st_module, archive_id)
    archive = build_archive(ideas)

    st_module.markdown(
        "### 🎛️ Interactive Quality-Diversity (iQD)"
    )
    st_module.caption(
        "Co-steer the MAP-Elites archive: **❄️ Freeze** a cell to "
        "protect its idea from being overwritten by future generation; "
        "**🚫 Prune** to exclude that cell from future search; "
        "**🧬 Select** two parents from any cells to force a directed "
        "crossover under your own constraint."
    )

    # ── Stats row ────────────────────────────────────────────────────────
    n_frozen = len(state.frozen_cells)
    n_pruned = len(state.pruned_cells)
    n_filled = len(archive)
    total_cells = len(METHODOLOGY_TYPES) * len(NOVELTY_LEVELS)
    sc1, sc2, sc3, sc4 = st_module.columns(4)
    sc1.metric("Filled cells", f"{n_filled}/{total_cells}")
    sc2.metric("Frozen", n_frozen)
    sc3.metric("Pruned", n_pruned)
    sc4.metric("Parents queued", len(state.selected_parents))

    # ── Archive heatmap grid ────────────────────────────────────────────
    st_module.markdown("#### 🗺️ Archive map")
    st_module.caption(
        "Rows = methodology type · Columns = novelty level. Cells show "
        "the best idea per (methodology × novelty) pair."
    )
    # Build idea_index lookup so we can record idea indices for parent selection.
    idea_by_id = {id(i): n for n, i in enumerate(ideas)}

    for mi, m_label in enumerate(METHODOLOGY_TYPES):
        row_cols = st_module.columns(
            [1.5] + [1] * len(NOVELTY_LEVELS)
        )
        row_cols[0].markdown(
            f"<div style='padding-top:18px;font-weight:600;"
            f"color:#0c4a6e;font-size:12px'>"
            f"{m_label.replace('_', ' ').title()}</div>",
            unsafe_allow_html=True,
        )
        for ni, n_label in enumerate(NOVELTY_LEVELS):
            cell = (mi, ni)
            idea = archive.get(cell)
            frozen = is_frozen(state, cell)
            pruned = is_pruned(state, cell)
            with row_cols[ni + 1]:
                _render_cell_card(
                    st_module, cell, idea, frozen, pruned,
                    n_label, state, ideas, idea_by_id,
                    archive_id,
                )

    # ── Selected parents queue ──────────────────────────────────────────
    st_module.markdown("---")
    st_module.markdown("#### 🧬 Directed crossover queue")
    if not state.selected_parents:
        st_module.info(
            "Click **🧬 Select** on any filled cell above to add an "
            "idea as a crossover parent. Pick exactly 2 to enable "
            "the synthesizer below."
        )
    else:
        for pos, idx in enumerate(state.selected_parents):
            if 0 <= idx < len(ideas):
                p = ideas[idx]
                st_module.markdown(
                    f"**Parent #{pos + 1}**: {p.get('title', '?')[:60]} "
                    f"<span style='color:#94a3b8'>"
                    f"({p.get('methodology_type', '?')} · "
                    f"{p.get('novelty_level', '?')})</span>",
                    unsafe_allow_html=True,
                )
        if st_module.button(
            "🗑️ Clear parents", key=f"iqd_clear_parents_{archive_id}",
        ):
            clear_parents(state)
            _save_state(st_module, archive_id, state)
            st_module.rerun()

    # ── Crossover synthesis form (only when exactly 2 parents) ──────────
    if len(state.selected_parents) == 2:
        st_module.markdown("---")
        st_module.markdown("#### 🧪 Synthesize child idea")
        constraint = st_module.text_input(
            "Your constraint (optional)",
            key=f"iqd_constraint_{archive_id}",
            placeholder=(
                "e.g. 'must run on a single A100 in <24h', "
                "'must not need proprietary data', "
                "'must produce a falsifiable benchmark'"
            ),
            help="Force the LLM to honor a real-world requirement when "
                  "blending the two parents.",
        )
        if st_module.button(
            "🧬 Synthesize",
            key=f"iqd_synth_{archive_id}",
            type="primary",
        ):
            pa = ideas[state.selected_parents[0]]
            pb = ideas[state.selected_parents[1]]
            with st_module.spinner("Synthesizing…"):
                child = directed_crossover(
                    pa, pb, constraint,
                    llm_client=llm_client,
                )
            if child:
                # Append to the loaded result so it appears in the
                # Ideas tab + everywhere else.
                results = st_module.session_state.get("results") or {}
                ideas_list = results.get("ideas")
                if isinstance(ideas_list, list):
                    ideas_list.append(child)
                    st_module.success(
                        f"✅ Crossover synthesized: **{child['title']}** "
                        f"(now visible in the Ideas tab as a new "
                        f"`source_strategy=I` idea)."
                    )
                    clear_parents(state)
                    _save_state(st_module, archive_id, state)
                else:
                    # No active results — show the dict for inspection.
                    st_module.success(
                        f"✅ Synthesized: **{child['title']}**"
                    )
                    st_module.json(child)
            else:
                st_module.error(
                    "Synthesis failed (LLM unavailable or returned "
                    "malformed JSON). Check the LLM provider config."
                )


def _render_cell_card(
    st_module,
    cell: Tuple[int, int],
    idea: Optional[Dict[str, Any]],
    frozen: bool,
    pruned: bool,
    novelty_label: str,
    state: IQDState,
    ideas: List[Dict[str, Any]],
    idea_by_id: Dict[int, int],
    archive_id: Any,
) -> None:
    """Render one MAP-Elites cell as a card with freeze/prune/select buttons."""
    mi, ni = cell
    if idea:
        q = float(idea.get("quality_score") or 0.0)
        title = idea.get("title") or "?"
        # Background color by quality.
        bg = (
            "#dcfce7" if q >= 0.6 else
            "#fef9c3" if q >= 0.4 else
            "#fee2e2"
        )
        border = "#86efac" if q >= 0.6 else "#fde68a" if q >= 0.4 else "#fecaca"
    else:
        title = "(empty)"
        q = 0.0
        bg = "#f1f5f9"
        border = "#cbd5e1"

    badges = []
    if frozen:
        badges.append("❄️")
    if pruned:
        badges.append("🚫")
    if idea is not None and id(idea) in idea_by_id and \
            idea_by_id[id(idea)] in state.selected_parents:
        badges.append("🧬")
    badge_html = (
        f"<span style='font-size:14px;margin-right:4px'>{''.join(badges)}</span>"
        if badges else ""
    )

    st_module.markdown(
        f"<div style='background:{bg};border:1px solid {border};"
        f"border-radius:8px;padding:8px 10px;min-height:80px;"
        f"margin-bottom:4px'>"
        f"<div style='font-size:10px;color:#64748b;font-weight:600;"
        f"text-transform:uppercase;letter-spacing:0.05em'>"
        f"{novelty_label}</div>"
        f"<div style='font-size:11px;font-weight:600;color:#0c4a6e;"
        f"line-height:1.3;margin-top:2px;overflow:hidden;"
        f"text-overflow:ellipsis;display:-webkit-box;"
        f"-webkit-line-clamp:2;-webkit-box-orient:vertical'>"
        f"{badge_html}{title[:55]}</div>"
        + (
            f"<div style='font-size:10px;color:#64748b;margin-top:3px'>"
            f"q={q:.2f}</div>"
            if idea else ""
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    if idea is None:
        # Empty cell can still be pruned (= don't generate here).
        b1, _ = st_module.columns(2)
        label = "✅ Unprune" if pruned else "🚫 Prune"
        if b1.button(label, key=f"iqd_p_{archive_id}_{mi}_{ni}",
                      use_container_width=True):
            if pruned:
                unprune_cell(state, cell)
            else:
                prune_cell(state, cell)
            _save_state(st_module, archive_id, state)
            st_module.rerun()
        return

    # Filled cell: freeze + prune + select.
    b1, b2, b3 = st_module.columns(3)
    freeze_lbl = "🔥 Unfreeze" if frozen else "❄️ Freeze"
    if b1.button(freeze_lbl, key=f"iqd_f_{archive_id}_{mi}_{ni}",
                  use_container_width=True,
                  help="Protect/release this cell from future overwrite"):
        if frozen:
            unfreeze_cell(state, cell)
        else:
            freeze_cell(state, cell)
        _save_state(st_module, archive_id, state)
        st_module.rerun()
    prune_lbl = "✅ Unprune" if pruned else "🚫 Prune"
    if b2.button(prune_lbl, key=f"iqd_p_{archive_id}_{mi}_{ni}",
                  use_container_width=True,
                  help="Skip/restore this cell in future generation"):
        if pruned:
            unprune_cell(state, cell)
        else:
            prune_cell(state, cell)
        _save_state(st_module, archive_id, state)
        st_module.rerun()
    if b3.button("🧬 Select", key=f"iqd_s_{archive_id}_{mi}_{ni}",
                  use_container_width=True,
                  help="Add as crossover parent"):
        idea_idx = idea_by_id.get(id(idea), -1)
        if idea_idx >= 0:
            select_parent(state, idea_idx)
            _save_state(st_module, archive_id, state)
            st_module.rerun()
