"""
models/archive.py - Quality-Diversity archive (MAP-Elites style 7x3 grid).

Grid dimensions:
  rows (7) : METHODOLOGY_TYPES
  cols (3) : NOVELTY_LEVELS
"""

from __future__ import annotations
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .idea import Idea, METHODOLOGY_TYPES, NOVELTY_LEVELS


# ── Archive cell ──────────────────────────────────────────────────────────────
@dataclass
class ArchiveCell:
    method_idx: int
    novelty_idx: int
    idea: Optional[Idea] = None
    quality: float = 0.0

    @property
    def is_empty(self) -> bool:
        return self.idea is None

    @property
    def key(self) -> Tuple[int, int]:
        return (self.method_idx, self.novelty_idx)


# ── QD Archive ────────────────────────────────────────────────────────────────
class QDArchive:
    """
    7 × 3 MAP-Elites archive.
    Each cell holds at most one idea; an incoming idea replaces the current
    occupant only if its quality score is higher.
    """

    ROWS: int = len(METHODOLOGY_TYPES)   # 7
    COLS: int = len(NOVELTY_LEVELS)      # 3

    def __init__(self) -> None:
        self._grid: Dict[Tuple[int, int], ArchiveCell] = {}
        self._lock = threading.Lock()
        for mi in range(self.ROWS):
            for ni in range(self.COLS):
                self._grid[(mi, ni)] = ArchiveCell(method_idx=mi, novelty_idx=ni)
        # Dirty-flag caches — invalidated on update().
        self._dirty = True
        self._cached_coverage: float = 0.0
        self._cached_filled: int = 0
        self._cached_quality_stats: tuple = (0.0, 0.0, 0.0)
        self._cached_ideas: Optional[List[Idea]] = None

    # ── Cache management ───────────────────────────────────────────────────
    def _recompute(self) -> None:
        """Recompute all cached aggregates in a single pass (21 cells)."""
        total_q = 0.0
        lo = float("inf")
        hi = float("-inf")
        filled = 0
        ideas: List[Idea] = []
        for c in self._grid.values():
            if not c.is_empty:
                filled += 1
                q = c.quality
                total_q += q
                if q < lo:
                    lo = q
                if q > hi:
                    hi = q
                ideas.append(c.idea)  # type: ignore[arg-type]
        total_cells = self.ROWS * self.COLS
        self._cached_coverage = filled / total_cells
        self._cached_filled = filled
        self._cached_quality_stats = (
            (total_q / filled, lo, hi) if filled else (0.0, 0.0, 0.0)
        )
        self._cached_ideas = ideas
        self._dirty = False

    def _ensure_fresh(self) -> None:
        if self._dirty:
            self._recompute()

    # ── Mutation ──────────────────────────────────────────────────────────────
    def update(self, idea: Idea) -> bool:
        """
        Attempt to place *idea* into its behavioural cell (thread-safe).
        Returns True if the cell was updated (new or improved entry).
        """
        mi = idea.method_idx()
        ni = idea.novelty_idx()
        key = (mi, ni)

        with self._lock:
            cell = self._grid[key]
            if cell.is_empty or idea.quality_score > cell.quality:
                cell.idea = idea
                cell.quality = idea.quality_score
                self._dirty = True
                return True
        return False

    # ── Queries ───────────────────────────────────────────────────────────────
    def coverage(self) -> float:
        """Fraction of cells that contain at least one idea (cached)."""
        self._ensure_fresh()
        return self._cached_coverage

    def get_empty_cells(self) -> List[Tuple[Tuple[int, int], ArchiveCell]]:
        """Return (key, cell) pairs for all empty cells."""
        return [(k, c) for k, c in self._grid.items() if c.is_empty]

    def get_low_quality_cells(self) -> List[Tuple[Tuple[int, int], ArchiveCell]]:
        """Return non-empty cells sorted by quality ascending."""
        occupied = [(k, c) for k, c in self._grid.items() if not c.is_empty]
        return sorted(occupied, key=lambda x: x[1].quality)

    def get_all_ideas(self) -> List[Idea]:
        """Return all ideas currently in the archive (cached)."""
        self._ensure_fresh()
        return list(self._cached_ideas) if self._cached_ideas else []

    def get_cell(self, method_idx: int, novelty_idx: int) -> ArchiveCell:
        return self._grid[(method_idx, novelty_idx)]

    # ── Quality statistics (single-pass) ────────────────────────────────────
    # Previously mean/min/max each built an independent list comprehension
    # over _grid — three O(N) passes. Since all three are typically called
    # together (pipeline.py final summary, app.py stats panel), a single
    # combined method amortizes the traversal.

    def quality_stats(self) -> tuple:
        """Return (mean, min, max) quality across filled cells (cached)."""
        self._ensure_fresh()
        return self._cached_quality_stats

    def mean_quality(self) -> float:
        """Mean quality of filled cells. Returns 0.0 if archive is empty."""
        return self.quality_stats()[0]

    def min_quality(self) -> float:
        """Lowest quality in the archive. Returns 0.0 if archive is empty."""
        return self.quality_stats()[1]

    def max_quality(self) -> float:
        """Highest quality in the archive. Returns 0.0 if archive is empty."""
        return self.quality_stats()[2]

    # ── Display ───────────────────────────────────────────────────────────────
    def to_display_dict(self) -> Dict[str, Any]:
        """
        Return a representation suitable for Streamlit display.
        Produces a 2-D list (rows = methods, cols = novelty levels).
        """
        grid_data = []
        for mi, method_name in enumerate(METHODOLOGY_TYPES):
            row = {"methodology": method_name, "cells": []}
            for ni, novelty_name in enumerate(NOVELTY_LEVELS):
                cell = self._grid[(mi, ni)]
                if cell.is_empty:
                    row["cells"].append({"novelty": novelty_name, "quality": None, "title": None})
                else:
                    row["cells"].append({
                        "novelty": novelty_name,
                        "quality": round(cell.quality, 3),
                        "title": cell.idea.title if cell.idea else None,
                    })
            grid_data.append(row)

        self._ensure_fresh()
        return {
            "coverage": self._cached_coverage,
            "total_cells": self.ROWS * self.COLS,
            "filled_cells": self._cached_filled,
            "grid": grid_data,
            "methodology_labels": METHODOLOGY_TYPES,
            "novelty_labels": NOVELTY_LEVELS,
        }
