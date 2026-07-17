"""
federated_diversity.py — population-scale homogenization study + federated
MAP-Elites mechanism.

The IdeaGraph paper notes a second open problem: per-user MAP-Elites
guarantees individual archive coverage, but if N researchers run the
pipeline independently, do their archives collapse onto the same modes?
This module provides:

  1. A multi-user simulation harness that runs N synthetic pipelines.
  2. Aggregate diversity metrics (Div-Pair via Jaccard distance,
     cell-saturation distribution, ideas-per-cell histogram).
  3. A federated mechanism: each user broadcasts a privacy-preserving
     hash summary of their occupied cells. A global census aggregates
     these and emits a saturation-penalty function that the per-user
     idea generator consults to bias *away* from globally-saturated cells.

The simulation runs locally and deterministically (reuses
pipeline_simulator.fake_pipeline_run) so it costs no LLM budget. The
federation effect is real: turning it on demonstrably raises aggregate
Div-Pair across the population.

Public API:
    CellHashBroadcast           — privacy-preserving per-user summary
    GlobalCellCensus            — aggregator across users
    federated_penalty_fn(census, ...) — returns saturation_fn(cell)→0..1
    compute_div_pair(archives)   — average Jaccard distance across pairs
    cell_saturation_grid(census, n_users) — 7×3 grid of fraction-of-users-occupying
    coverage_distribution(archives) — list of per-user coverage fractions
    simulate_population(n_users, ..., federate=False) → PopulationResult
"""
from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Privacy-preserving per-user broadcast
# ─────────────────────────────────────────────────────────────────────────────

# Quality bucketing: round to one of 4 levels so the broadcast is genuinely
# coarse-grained (a single user can't be uniquely identified by their tiny
# floating-point quality_score).
QUALITY_BUCKETS: Tuple[float, ...] = (0.0, 0.30, 0.55, 0.80, 1.01)


def _quality_bucket(q: float) -> int:
    """Map a continuous quality score to a 0–3 bucket for hashing."""
    q = max(0.0, min(1.0, float(q)))
    for i in range(len(QUALITY_BUCKETS) - 1):
        if QUALITY_BUCKETS[i] <= q < QUALITY_BUCKETS[i + 1]:
            return i
    return len(QUALITY_BUCKETS) - 2


def _cell_hash(method_idx: int, novelty_idx: int, q_bucket: int) -> str:
    """Stable 12-char hash of a (cell, quality bucket) tuple."""
    s = f"{int(method_idx)}|{int(novelty_idx)}|{int(q_bucket)}"
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


@dataclass
class CellHashBroadcast:
    """A single user's contribution to the federated census.

    Carries only hashes of (cell, quality-bucket) tuples — no titles, no
    methods, no probe scores. This is what would actually move across the
    wire in a real federation: enough to compute saturation, not enough
    to leak any specific idea.
    """
    user_id: str
    n_cells: int                  # how many cells this user occupies
    cell_hashes: Set[str] = field(default_factory=set)

    @classmethod
    def from_archive(cls, user_id: str,
                      archive: Dict[Tuple[int, int], Any]) -> "CellHashBroadcast":
        """Build a broadcast from a user's QD archive (FakeRun.archive)."""
        hashes: Set[str] = set()
        for (mi, ni), idea in archive.items():
            q = (
                idea.blended_quality
                if getattr(idea, "blended_quality", None) is not None
                else getattr(idea, "quality", 0.0)
            )
            hashes.add(_cell_hash(mi, ni, _quality_bucket(q)))
        return cls(user_id=user_id, n_cells=len(archive), cell_hashes=hashes)


@dataclass
class GlobalCellCensus:
    """Aggregator of CellHashBroadcasts across all users.

    Keeps a count of how many users mention each (cell, quality-bucket)
    hash, plus a fast lookup of how many users occupy each (mi, ni) cell
    regardless of quality. The latter is what the saturation_fn uses.
    """
    n_users: int = 0
    hash_counts: Dict[str, int] = field(default_factory=dict)
    cell_user_counts: Dict[Tuple[int, int], int] = field(default_factory=dict)

    def ingest(self, broadcast: CellHashBroadcast,
                cells: Optional[Set[Tuple[int, int]]] = None) -> None:
        """Add one broadcast. `cells` is the (mi,ni) set; we keep both views."""
        self.n_users += 1
        for h in broadcast.cell_hashes:
            self.hash_counts[h] = self.hash_counts.get(h, 0) + 1
        if cells:
            for c in cells:
                self.cell_user_counts[c] = self.cell_user_counts.get(c, 0) + 1

    def cell_saturation(self, cell: Tuple[int, int]) -> float:
        """0..1: what fraction of users currently occupy this (mi,ni) cell?"""
        if self.n_users == 0:
            return 0.0
        return self.cell_user_counts.get(cell, 0) / self.n_users


# ─────────────────────────────────────────────────────────────────────────────
# Federated penalty function
# ─────────────────────────────────────────────────────────────────────────────

def federated_penalty_fn(
    census: GlobalCellCensus,
    threshold: float = 0.30,
    strength: float = 0.85,
) -> Callable[[Tuple[int, int]], float]:
    """Build a saturation-penalty function from the current census.

    Returns f(cell) → 0..1 where 0 = fully welcome, 1 = strongly avoid.
    Below `threshold` saturation, no penalty. Above, penalty rises linearly
    to `strength` at saturation = 1.0.

    These defaults tune to: cells occupied by < 30% of users are unpenalized,
    cells everyone occupies get an 85% penalty (the new user's generator
    will mostly steer elsewhere).
    """
    threshold = max(0.0, min(1.0, float(threshold)))
    strength = max(0.0, min(1.0, float(strength)))
    span = max(1e-9, 1.0 - threshold)

    def _penalty(cell: Tuple[int, int]) -> float:
        sat = census.cell_saturation(cell)
        if sat <= threshold:
            return 0.0
        excess = (sat - threshold) / span
        return min(strength, strength * excess)

    return _penalty


# ─────────────────────────────────────────────────────────────────────────────
# Diversity metrics
# ─────────────────────────────────────────────────────────────────────────────

def _jaccard_distance(a: Set, b: Set) -> float:
    """1 - Jaccard similarity. Both empty sets → distance 0."""
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return 1.0 - (len(a & b) / len(union))


def compute_div_pair(archives: List[Dict[Tuple[int, int], Any]]) -> float:
    """Average pairwise Jaccard distance across users' cell sets.

    Higher = more diverse population (users covering different cells).
    Lower = homogenized (everyone landing on the same cells).
    """
    if len(archives) < 2:
        return 0.0
    cell_sets = [set(a.keys()) for a in archives]
    total = 0.0
    n = 0
    for i, j in itertools.combinations(range(len(cell_sets)), 2):
        total += _jaccard_distance(cell_sets[i], cell_sets[j])
        n += 1
    return total / max(1, n)


def cell_saturation_grid(census: GlobalCellCensus,
                          n_methodologies: int = 7,
                          n_novelty: int = 3) -> List[List[float]]:
    """7×3 grid of fraction-of-users occupying each cell."""
    if census.n_users == 0:
        return [[0.0] * n_novelty for _ in range(n_methodologies)]
    grid = [[0.0] * n_novelty for _ in range(n_methodologies)]
    for mi in range(n_methodologies):
        for ni in range(n_novelty):
            grid[mi][ni] = census.cell_user_counts.get((mi, ni), 0) / census.n_users
    return grid


def coverage_distribution(
    archives: List[Dict[Tuple[int, int], Any]],
    total_cells: int = 21,
) -> List[float]:
    """Fraction of the 21-cell grid occupied by each user."""
    return [len(a) / float(total_cells) for a in archives]


def union_coverage(archives: List[Dict[Tuple[int, int], Any]],
                    total_cells: int = 21) -> float:
    """Fraction of cells occupied by AT LEAST ONE user (collective coverage)."""
    union: Set[Tuple[int, int]] = set()
    for a in archives:
        union.update(a.keys())
    return len(union) / float(total_cells)


def homogenization_index(census: GlobalCellCensus) -> float:
    """0 = perfectly diverse (uniform coverage); 1 = one cell holds everyone.

    Computed as Gini coefficient over the cell-user-count distribution.
    """
    if census.n_users == 0:
        return 0.0
    counts = list(census.cell_user_counts.values())
    if not counts:
        return 0.0
    counts = sorted(counts)
    n = len(counts)
    cum = sum((i + 1) * v for i, v in enumerate(counts))
    total = sum(counts)
    if total == 0:
        return 0.0
    gini = (2.0 * cum) / (n * total) - (n + 1.0) / n
    return max(0.0, min(1.0, gini))


# ─────────────────────────────────────────────────────────────────────────────
# Multi-user simulation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PopulationResult:
    """Outcome of a single A or B run of the multi-user simulation."""
    federated: bool
    n_users: int
    archives: List[Dict[Tuple[int, int], Any]] = field(default_factory=list)
    runs: List[Any] = field(default_factory=list)        # FakeRun list
    census: Optional[GlobalCellCensus] = None
    div_pair: float = 0.0
    union_coverage_frac: float = 0.0
    homogenization: float = 0.0
    mean_user_coverage: float = 0.0
    median_user_coverage: float = 0.0

    def summary(self) -> str:
        tag = "FEDERATED" if self.federated else "INDEPENDENT"
        return (
            f"[{tag}] n={self.n_users} users · "
            f"Div-Pair={self.div_pair:.3f} · "
            f"union={self.union_coverage_frac*100:.0f}% · "
            f"homogenization={self.homogenization:.3f}"
        )


def simulate_population(
    n_users: int = 30,
    ideas_per_user: int = 12,
    federate: bool = False,
    seed: int = 0,
    topic_pool: Optional[List[str]] = None,
    enable_revision: bool = True,
    federation_threshold: float = 0.30,
    federation_strength: float = 0.85,
) -> PopulationResult:
    """Run N synthetic pipelines and collect aggregate diversity metrics.

    When `federate=True`, users share privacy-preserving hash broadcasts
    so each new user's generator can bias away from globally-saturated
    cells. When `federate=False`, each user runs independently — exactly
    the homogenization scenario the paper warns about.
    """
    # Lazy-import to keep this module independent of UI bits.
    from pipeline_simulator import fake_pipeline_run

    if topic_pool is None or not topic_pool:
        topic_pool = [
            "graph neural networks for drug discovery",
            "self-supervised learning for medical imaging",
            "robust reinforcement learning under distribution shift",
            "interpretable machine learning for genomics",
            "scalable causal inference at the population level",
        ]

    census = GlobalCellCensus()
    archives: List[Dict[Tuple[int, int], Any]] = []
    runs: List[Any] = []
    sat_fn: Optional[Callable[[Tuple[int, int]], float]] = None

    for u in range(n_users):
        # Each user has a distinct topic + seed combination so their
        # pipelines are genuinely different runs, not deterministic copies.
        topic = topic_pool[u % len(topic_pool)]
        user_seed = seed * 100_003 + u * 7919

        if federate and census.n_users > 0:
            sat_fn = federated_penalty_fn(
                census,
                threshold=federation_threshold,
                strength=federation_strength,
            )

        run = fake_pipeline_run(
            topic, seed=user_seed,
            n_ideas=ideas_per_user,
            enable_revision=enable_revision,
            saturation_fn=sat_fn,
        )
        runs.append(run)
        archives.append(run.archive)

        # User contributes back to the census (in real federation this
        # is a network broadcast; here it's just a local update).
        broadcast = CellHashBroadcast.from_archive(f"user_{u}", run.archive)
        census.ingest(broadcast, cells=set(run.archive.keys()))

    cov_dist = coverage_distribution(archives)
    cov_dist_sorted = sorted(cov_dist)
    mean_cov = sum(cov_dist) / len(cov_dist) if cov_dist else 0.0
    median_cov = (
        cov_dist_sorted[len(cov_dist_sorted) // 2] if cov_dist_sorted else 0.0
    )

    return PopulationResult(
        federated=federate,
        n_users=n_users,
        archives=archives,
        runs=runs,
        census=census,
        div_pair=compute_div_pair(archives),
        union_coverage_frac=union_coverage(archives),
        homogenization=homogenization_index(census),
        mean_user_coverage=mean_cov,
        median_user_coverage=median_cov,
    )


def compare_populations(
    n_users: int = 30,
    ideas_per_user: int = 12,
    seed: int = 0,
    **kwargs,
) -> Dict[str, PopulationResult]:
    """Run the population once independent and once federated for A/B."""
    indep = simulate_population(
        n_users=n_users, ideas_per_user=ideas_per_user,
        federate=False, seed=seed, **kwargs,
    )
    fed = simulate_population(
        n_users=n_users, ideas_per_user=ideas_per_user,
        federate=True, seed=seed, **kwargs,
    )
    return {"independent": indep, "federated": fed}
