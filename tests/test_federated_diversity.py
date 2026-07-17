"""Tests for federated_diversity.py — population-scale homogenization study."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import pytest

from federated_diversity import (
    QUALITY_BUCKETS,
    CellHashBroadcast,
    GlobalCellCensus,
    PopulationResult,
    cell_saturation_grid,
    compare_populations,
    compute_div_pair,
    coverage_distribution,
    federated_penalty_fn,
    homogenization_index,
    simulate_population,
    union_coverage,
    _cell_hash,
    _jaccard_distance,
    _quality_bucket,
)


class _Idea:
    """Lightweight stand-in matching FakeIdea/Idea attribute access."""
    def __init__(self, q, blended=None):
        self.quality = q
        self.blended_quality = blended


# ── Helpers ──────────────────────────────────────────────────────────────

def _archive(*cells_q):
    """Build an archive dict from (mi, ni, q) tuples."""
    return {(mi, ni): _Idea(q) for mi, ni, q in cells_q}


# ── Quality bucketing & cell hashing (privacy primitives) ────────────────

class TestQualityBucketing:
    def test_buckets_span_unit_interval(self):
        # Buckets should cover [0, 1] continuously
        assert QUALITY_BUCKETS[0] == 0.0
        assert QUALITY_BUCKETS[-1] > 1.0

    def test_quality_bucket_clamps_out_of_range(self):
        assert _quality_bucket(-1.0) == 0
        assert _quality_bucket(2.0) == len(QUALITY_BUCKETS) - 2

    def test_bucket_assignment_monotonic(self):
        # Same quality → same bucket; higher quality → ≥ bucket
        b1 = _quality_bucket(0.10)
        b2 = _quality_bucket(0.40)
        b3 = _quality_bucket(0.85)
        assert b1 <= b2 <= b3

    def test_cell_hash_is_stable(self):
        # Same input → same hash; tiny quality change inside same bucket → same hash
        h1 = _cell_hash(0, 1, 2)
        h2 = _cell_hash(0, 1, 2)
        assert h1 == h2 and len(h1) == 12

    def test_cell_hash_differs_for_different_cells(self):
        assert _cell_hash(0, 1, 2) != _cell_hash(1, 1, 2)
        assert _cell_hash(0, 1, 2) != _cell_hash(0, 2, 2)
        assert _cell_hash(0, 1, 2) != _cell_hash(0, 1, 3)


class TestBroadcastPrivacy:
    def test_from_archive_collects_cell_count(self):
        arch = _archive((0, 0, 0.5), (1, 2, 0.7))
        b = CellHashBroadcast.from_archive("u_alpha", arch)
        assert b.user_id == "u_alpha"
        assert b.n_cells == 2
        assert len(b.cell_hashes) == 2

    def test_broadcast_does_not_leak_titles_or_scores(self):
        # Build an "idea" with sensitive data and ensure none of it survives
        class Sensitive:
            def __init__(self):
                self.quality = 0.7
                self.blended_quality = None
                self.title = "LEAKY-TITLE-DO-NOT-EMIT"
                self.method = "secret-method"
        arch = {(0, 0): Sensitive(), (3, 1): Sensitive()}
        b = CellHashBroadcast.from_archive("u", arch)
        for h in b.cell_hashes:
            assert "LEAKY-TITLE" not in h
            assert "secret" not in h.lower()
            # 12-char hex only
            assert len(h) == 12 and all(c in "0123456789abcdef" for c in h)

    def test_blended_quality_takes_precedence(self):
        # When execution-revision has filled blended_quality, it should be
        # used for bucketing instead of probe-only quality.
        arch = {(0, 0): _Idea(q=0.4, blended=0.85)}
        b1 = CellHashBroadcast.from_archive("u", arch)
        # Same archive but only probe quality:
        arch2 = {(0, 0): _Idea(q=0.4, blended=None)}
        b2 = CellHashBroadcast.from_archive("u", arch2)
        # Different quality buckets → different hashes
        assert b1.cell_hashes != b2.cell_hashes


# ── Global census aggregation ────────────────────────────────────────────

class TestGlobalCensus:
    def test_ingests_broadcast_and_counts(self):
        census = GlobalCellCensus()
        census.ingest(CellHashBroadcast.from_archive("u1", _archive((0, 0, 0.5))),
                       cells={(0, 0)})
        census.ingest(CellHashBroadcast.from_archive("u2", _archive((0, 0, 0.5))),
                       cells={(0, 0)})
        assert census.n_users == 2
        assert census.cell_user_counts[(0, 0)] == 2

    def test_saturation_zero_when_empty(self):
        census = GlobalCellCensus()
        assert census.cell_saturation((0, 0)) == 0.0

    def test_saturation_fraction_correct(self):
        census = GlobalCellCensus()
        for u in range(4):
            arch = _archive((0, 0, 0.5)) if u < 3 else _archive((1, 1, 0.5))
            census.ingest(
                CellHashBroadcast.from_archive(f"u{u}", arch),
                cells=set(arch.keys()),
            )
        # 3 of 4 users in (0,0) → 0.75; 1 of 4 in (1,1) → 0.25
        assert census.cell_saturation((0, 0)) == pytest.approx(0.75)
        assert census.cell_saturation((1, 1)) == pytest.approx(0.25)
        assert census.cell_saturation((6, 2)) == 0.0  # never seen


# ── Saturation penalty function ──────────────────────────────────────────

class TestPenaltyFn:
    def _saturate(self, frac, cell=(0, 0), n=20):
        census = GlobalCellCensus()
        for u in range(n):
            cells = {cell} if u < int(frac * n) else {(6, 2)}
            census.ingest(
                CellHashBroadcast.from_archive(f"u{u}",
                                                 {c: _Idea(0.5) for c in cells}),
                cells=cells,
            )
        return census

    def test_below_threshold_no_penalty(self):
        # 20% saturation, threshold 0.30 → no penalty
        census = self._saturate(0.20)
        fn = federated_penalty_fn(census, threshold=0.30, strength=0.85)
        assert fn((0, 0)) == 0.0

    def test_above_threshold_penalty_scales(self):
        # 60% saturation, threshold 0.30 → moderate penalty
        census = self._saturate(0.60)
        fn = federated_penalty_fn(census, threshold=0.30, strength=0.85)
        p = fn((0, 0))
        assert 0 < p < 0.85

    def test_full_saturation_caps_at_strength(self):
        census = self._saturate(1.00)
        fn = federated_penalty_fn(census, threshold=0.30, strength=0.85)
        assert fn((0, 0)) == pytest.approx(0.85, abs=0.01)

    def test_unseen_cell_unpenalized(self):
        census = self._saturate(1.00, cell=(0, 0))
        fn = federated_penalty_fn(census, threshold=0.30, strength=0.85)
        assert fn((4, 1)) == 0.0


# ── Diversity metrics ────────────────────────────────────────────────────

class TestDiversityMetrics:
    def test_jaccard_identical_sets_zero(self):
        s = {(0, 0), (1, 1)}
        assert _jaccard_distance(s, s) == 0.0

    def test_jaccard_disjoint_sets_one(self):
        a = {(0, 0)}
        b = {(1, 1)}
        assert _jaccard_distance(a, b) == 1.0

    def test_jaccard_both_empty_is_zero(self):
        # Convention: empty pair is "perfectly similar" (no information)
        assert _jaccard_distance(set(), set()) == 0.0

    def test_div_pair_zero_for_single_user(self):
        assert compute_div_pair([_archive((0, 0, 0.5))]) == 0.0

    def test_div_pair_one_when_users_disjoint(self):
        archives = [
            _archive((0, 0, 0.5)),
            _archive((1, 1, 0.5)),
            _archive((6, 2, 0.5)),
        ]
        # All pairs disjoint → average distance = 1
        assert compute_div_pair(archives) == 1.0

    def test_div_pair_zero_when_all_identical(self):
        archives = [_archive((0, 0, 0.5), (1, 1, 0.5))] * 5
        assert compute_div_pair(archives) == 0.0

    def test_union_coverage_counts_uniquely(self):
        archives = [_archive((0, 0, 0.5)), _archive((0, 0, 0.5), (1, 1, 0.5))]
        # Only 2 distinct cells across the population
        assert union_coverage(archives) == pytest.approx(2 / 21)

    def test_homogenization_low_when_uniform(self):
        # Build a census where each cell has the same user count
        census = GlobalCellCensus()
        for u in range(7):
            for ni in range(3):
                census.cell_user_counts[(u % 7, ni)] = 3
        census.n_users = 9
        h = homogenization_index(census)
        assert h < 0.1  # near zero — perfectly uniform

    def test_homogenization_high_when_concentrated(self):
        # All weight on one cell
        census = GlobalCellCensus()
        census.cell_user_counts[(0, 0)] = 100
        # Add a few empty cells to make Gini meaningful
        for ni in range(3):
            census.cell_user_counts.setdefault((1, ni), 0)
        census.n_users = 100
        h = homogenization_index(census)
        assert h > 0.4


# ── End-to-end population simulation ─────────────────────────────────────

class TestSimulatePopulation:
    def test_returns_correct_n_archives(self):
        r = simulate_population(n_users=8, ideas_per_user=8, seed=42)
        assert isinstance(r, PopulationResult)
        assert len(r.archives) == 8
        assert r.n_users == 8

    def test_indep_run_does_not_share_state(self):
        # Two runs with same params produce same result (deterministic)
        a = simulate_population(n_users=5, ideas_per_user=8, seed=42)
        b = simulate_population(n_users=5, ideas_per_user=8, seed=42)
        assert a.div_pair == b.div_pair
        assert a.union_coverage_frac == b.union_coverage_frac

    def test_federate_changes_outcome(self):
        # Same seed, same N, same ideas — only the federation flag differs
        indep = simulate_population(
            n_users=20, ideas_per_user=10, seed=7, federate=False,
        )
        fed = simulate_population(
            n_users=20, ideas_per_user=10, seed=7, federate=True,
        )
        # Federation should NOT produce identical results — its job is to
        # change cell-selection behavior
        assert indep.div_pair != fed.div_pair

    def test_federation_increases_div_pair(self):
        # The headline empirical claim of the proposed paper.
        # Confirmed across multiple seeds to avoid a flaky single-seed
        # result driving the test.
        wins = 0
        runs = 0
        for s in range(5):
            indep = simulate_population(
                n_users=20, ideas_per_user=10, seed=s, federate=False,
            )
            fed = simulate_population(
                n_users=20, ideas_per_user=10, seed=s, federate=True,
            )
            if fed.div_pair >= indep.div_pair:
                wins += 1
            runs += 1
        # Federation should win the majority of seeds
        assert wins >= runs - 1, f"federation only won {wins}/{runs} seeds"

    def test_federation_reduces_homogenization_on_average(self):
        # Gini on 21 cells is noisier than Div-Pair, so we test the
        # *average* direction across seeds rather than per-seed wins.
        # The claim: across many populations, federated runs have lower
        # mean Gini than independent runs.
        indep_ginis = []
        fed_ginis = []
        for s in range(8):
            indep = simulate_population(
                n_users=25, ideas_per_user=10, seed=s, federate=False,
            )
            fed = simulate_population(
                n_users=25, ideas_per_user=10, seed=s, federate=True,
            )
            indep_ginis.append(indep.homogenization)
            fed_ginis.append(fed.homogenization)
        mean_indep = sum(indep_ginis) / len(indep_ginis)
        mean_fed = sum(fed_ginis) / len(fed_ginis)
        assert mean_fed <= mean_indep, (
            f"federation should lower mean Gini: indep={mean_indep:.3f} "
            f"fed={mean_fed:.3f}"
        )

    def test_each_user_gets_unique_seed(self):
        # Two users in the same population must NOT produce identical archives
        r = simulate_population(n_users=10, ideas_per_user=10, seed=1)
        archive_signatures = [
            frozenset(a.keys()) for a in r.archives
        ]
        assert len(set(archive_signatures)) > 1, \
            "all users produced identical archive cell-sets — seeding broken"


class TestCompareAndGrid:
    def test_compare_returns_both_arms(self):
        results = compare_populations(n_users=10, ideas_per_user=8, seed=42)
        assert "independent" in results
        assert "federated" in results
        assert results["independent"].federated is False
        assert results["federated"].federated is True

    def test_saturation_grid_shape(self):
        r = simulate_population(n_users=10, ideas_per_user=8, seed=1)
        grid = cell_saturation_grid(r.census)
        assert len(grid) == 7 and all(len(row) == 3 for row in grid)
        for row in grid:
            for v in row:
                assert 0.0 <= v <= 1.0

    def test_coverage_distribution_lengths(self):
        r = simulate_population(n_users=10, ideas_per_user=8, seed=1)
        cov = coverage_distribution(r.archives)
        assert len(cov) == 10
        assert all(0.0 <= c <= 1.0 for c in cov)


class TestPipelineSimulatorHook:
    def test_saturation_fn_invoked_during_run(self):
        from pipeline_simulator import fake_pipeline_run
        called = []

        def trace(cell):
            called.append(cell)
            return 0.0

        run = fake_pipeline_run("topic", seed=42, n_ideas=8, saturation_fn=trace)
        # Hook must be called for every (mi, ni) cell × every idea attempt
        assert len(called) >= 21 * 8  # 21 cells × 8 idea attempts
        # All calls should be valid cell tuples
        assert all(isinstance(c, tuple) and len(c) == 2 for c in called)

    def test_extreme_saturation_zero_collapses_to_uniform(self):
        # If every cell is fully penalized, the fallback path should kick
        # in (no all-zero division-by-zero crash) and produce a valid run
        from pipeline_simulator import fake_pipeline_run
        run = fake_pipeline_run(
            "topic", seed=42, n_ideas=10,
            saturation_fn=lambda cell: 1.0,  # everything maxed out
        )
        # Run must still complete successfully
        assert len(run.ideas) == 10

    def test_no_saturation_fn_keeps_default_distribution(self):
        # Backward compat: passing no hook leaves behavior identical
        from pipeline_simulator import fake_pipeline_run
        a = fake_pipeline_run("topic", seed=42, n_ideas=8)
        b = fake_pipeline_run("topic", seed=42, n_ideas=8, saturation_fn=None)
        assert [i.title for i in a.ideas] == [i.title for i in b.ideas]


class TestAdminWiring:
    def test_admin_dashboard_imports_federated_module(self):
        # Confirm the population panel actually wires the new module
        with open("admin_dashboard.py", encoding="utf-8") as f:
            src = f.read()
        assert "from federated_diversity import" in src
        assert "_render_population_panel" in src
        assert "Federation" in src
