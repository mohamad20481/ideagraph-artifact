"""
agents/diversity_manager.py - Quality-Diversity orchestration (no LLM calls).

Responsibilities:
  - Select which archive cells to target next (UCB1-based exploration/exploitation)
  - Choose which ideation strategy to use (A / B / C) with success-rate weighting
  - Identify structural gaps between clusters in the DAG

Optimisations vs original:
  - UCB1 cell selection: cells scored by quality + C·√(ln(total)/attempts)
    → never-tried cells get ∞ priority; tried cells balance quality vs exploration
  - Strategy success tracking: pick_strategy() weights A/B/C by historical hit rate
    per novelty level, automatically learning which strategies work best
  - Source diversity tracking: pick_frontier_paper() and pick_cluster_pair() weight
    candidates by 1/(use_count+1) so the pipeline explores the full knowledge graph
    rather than over-exploiting a small subset of papers/pairs
  - Structural gap cache still keyed by (n_nodes, n_edges)
"""

from __future__ import annotations
import math
import random
import threading
from itertools import combinations
from typing import Dict, List, Optional, Tuple

from models.archive import QDArchive
from models.dag import KnowledgeDAG
from models.idea import METHODOLOGY_TYPES, NOVELTY_LEVELS
from creative_optimization import ThompsonBandit, CuriosityExplorer

_UCB1_C = math.sqrt(2)  # exploration constant (standard UCB1)


class DiversityManager:
    """
    Orchestration helper for Quality-Diversity search.

    Optimisations:
      - UCB1 cell selection with simulated annealing
      - Thompson Sampling (Bayesian bandit) for strategy selection
      - Curiosity-driven exploration for cell prioritization
      - Maturity-aware cluster pairing
      - Source diversity tracking
    """

    def __init__(self) -> None:
        # Cache: (n_nodes, n_edges) → list of gap pairs
        self._gaps_cache: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        self._gaps_lock = threading.Lock()

        # UCB1 tracking: per-cell attempt and success counts
        self._cell_attempts: Dict[Tuple[int, int], int] = {}
        self._cell_successes: Dict[Tuple[int, int], int] = {}
        self._ucb1_lock = threading.Lock()

        # Strategy success tracking: (strategy, novelty_idx) → (attempts, successes)
        self._strat_attempts: Dict[Tuple[str, int], int] = {}
        self._strat_successes: Dict[Tuple[str, int], int] = {}
        self._strat_lock = threading.Lock()

        # Source diversity tracking
        self._paper_use_count: Dict[str, int] = {}
        self._pair_use_count: Dict[Tuple[int, int], int] = {}
        self._source_lock = threading.Lock()

        # Thompson Sampling bandit for strategy selection (Bayesian, replaces
        # the simple success-rate weighting with proper uncertainty modeling)
        self._thompson = ThompsonBandit()
        for strat in ("A", "B", "C"):
            for nlevel in range(3):
                self._thompson.add_arm(f"{strat}:{nlevel}")

        # Curiosity-driven cell exploration (information-gain maximization)
        self._curiosity = CuriosityExplorer(curiosity_weight=0.3)

    def _dag_fingerprint(self, dag: KnowledgeDAG) -> Tuple[int, int]:
        """Cheap proxy for DAG identity (nodes + edges are static during ideation)."""
        return (dag.graph.number_of_nodes(), dag.graph.number_of_edges())

    # ─────────────────────────────────────────────────────────────────────────
    # UCB1 attempt tracking
    # ─────────────────────────────────────────────────────────────────────────
    def record_attempt(self, cell: Tuple[int, int], success: bool, quality: float = 0.0) -> None:
        """
        Record the outcome of a cell attempt — feeds UCB1 + curiosity explorer.
        Call this from the pipeline after each _run_cell() completes.
        """
        with self._ucb1_lock:
            self._cell_attempts[cell] = self._cell_attempts.get(cell, 0) + 1
            if success:
                self._cell_successes[cell] = self._cell_successes.get(cell, 0) + 1
        # Feed curiosity-driven explorer
        cell_key = f"{cell[0]}:{cell[1]}"
        self._curiosity.record_outcome(cell_key, quality if quality > 0 else (1.0 if success else 0.0))

    def _ucb1_score(
        self,
        cell: Tuple[int, int],
        quality: float,
        total_attempts: int,
        iteration: int = 0,
    ) -> float:
        """
        UCB1 score = quality + C(t)·√(ln(total_attempts) / cell_attempts).

        Exploration decay (simulated annealing):
          C(t) = max(0.5, √2 · (1 − 0.04·t))

          Early iterations (t=0): C = √2 ≈ 1.41 → full exploration
          After 10 iterations:    C ≈ 0.85       → balanced
          After ~17 iterations:   C = 0.50        → exploitation-heavy

        Never-tried cells return ∞ so they are always tried first.
        Empty cells still beat occupied cells of equal UCB1 thanks to the
        +1.0 empty-cell bonus in the caller.
        """
        cell_attempts = self._cell_attempts.get(cell, 0)
        if cell_attempts == 0:
            return float("inf")
        C = max(0.5, _UCB1_C * (1.0 - 0.04 * iteration))
        return quality + C * math.sqrt(math.log(max(total_attempts, 1)) / cell_attempts)

    # ─────────────────────────────────────────────────────────────────────────
    # Cell targeting (UCB1-based)
    # ─────────────────────────────────────────────────────────────────────────
    def select_target_cells(
        self,
        archive: QDArchive,
        n: int = 3,
        iteration: int = 0,
    ) -> List[Tuple[int, int]]:
        """
        Return up to *n* (method_idx, novelty_idx) tuples to target using UCB1.

        Score per cell = UCB1(quality, attempts, iteration) + 1.0 for empty cells.
        Cells never attempted get ∞ and are always tried first.
        Exploration constant C decays with iteration (see _ucb1_score).
        """
        with self._ucb1_lock:
            total_attempts = sum(self._cell_attempts.values())

        scored: List[Tuple[float, Tuple[int, int]]] = []
        for key, cell in archive._grid.items():
            quality = 0.0 if cell.is_empty else cell.quality
            if not cell.is_empty and quality >= 0.85:
                continue
            ucb = self._ucb1_score(key, quality, total_attempts, iteration)
            # Empty cell bonus
            bonus = 1.0 if cell.is_empty else 0.0
            # Curiosity bonus: prioritize cells with high information gain
            cell_key = f"{key[0]}:{key[1]}"
            curiosity_bonus = self._curiosity.curiosity_score(cell_key) * 0.3
            scored.append((ucb + bonus + curiosity_bonus, key))

        # Sort descending; shuffle equal-score entries for variety
        scored.sort(key=lambda x: x[0], reverse=True)
        return [key for _, key in scored[:n]]

    # ─────────────────────────────────────────────────────────────────────────
    # Strategy success tracking
    # ─────────────────────────────────────────────────────────────────────────
    def record_strategy_attempt(
        self,
        strategy: str,
        novelty_idx: int,
        success: bool,
    ) -> None:
        """Track strategy outcomes — feeds both simple stats and Thompson Sampling."""
        key = (strategy, novelty_idx)
        with self._strat_lock:
            self._strat_attempts[key] = self._strat_attempts.get(key, 0) + 1
            if success:
                self._strat_successes[key] = self._strat_successes.get(key, 0) + 1
        # Update Thompson Sampling posterior
        self._thompson.update(f"{strategy}:{novelty_idx}", 1.0 if success else 0.0)

    def _strategy_success_rate(self, strategy: str, novelty_idx: int) -> float:
        """
        Success rate for (strategy, novelty_idx) — defaults to 0.5 (neutral prior)
        when there is no history yet.  This avoids cold-start bias.
        """
        key = (strategy, novelty_idx)
        with self._strat_lock:
            attempts = self._strat_attempts.get(key, 0)
            successes = self._strat_successes.get(key, 0)
        if attempts == 0:
            return 0.5  # uniform prior
        return successes / attempts

    # ─────────────────────────────────────────────────────────────────────────
    # Strategy selection (success-rate weighted)
    # ─────────────────────────────────────────────────────────────────────────
    def pick_strategy(
        self,
        dag: KnowledgeDAG,
        target_cell: Tuple[int, int],
        iteration: int,
    ) -> str:
        """
        Return "A", "B", or "C".

        Base rules filter the valid candidate set:
          - A always valid (frontier extension works for any cell)
          - B valid when ≥ 2 clusters
          - C valid when ≥ 2 clusters AND structural gaps exist

        Among valid candidates, pick by weighted random draw using each
        strategy's historical success rate for this novelty level.  This lets
        the pipeline automatically learn which strategies work best per cell.
        """
        _, novelty_idx = target_cell
        n_clusters = len(dag.get_cluster_ids())

        # Build valid strategy set
        valid: List[str] = ["A"]
        if n_clusters >= 2:
            valid.append("B")
            gaps = self.find_structural_gaps(dag)
            if gaps:
                valid.append("C")

        if len(valid) == 1:
            return valid[0]

        # Thompson Sampling: sample from Beta posteriors for each valid strategy
        # This naturally balances exploration (uncertain arms) with exploitation
        # (known-good arms), replacing the simple weighted-random approach.
        arm_keys = [f"{s}:{novelty_idx}" for s in valid]
        best_arm = self._thompson.select(arm_keys)
        if best_arm:
            return best_arm.split(":")[0]

        # Fallback: simple weighted draw
        weights = [
            self._strategy_success_rate(s, novelty_idx) + 0.05
            for s in valid
        ]
        total_w = sum(weights)
        r = random.random() * total_w
        cumulative = 0.0
        for strategy, w in zip(valid, weights):
            cumulative += w
            if r <= cumulative:
                return strategy
        return valid[-1]

    # ─────────────────────────────────────────────────────────────────────────
    # Structural gap detection
    # ─────────────────────────────────────────────────────────────────────────
    def find_structural_gaps(
        self,
        dag: KnowledgeDAG,
    ) -> List[Tuple[int, int]]:
        """
        Return list of (cluster_a_id, cluster_b_id) pairs where
        NO directed edge exists between any papers of those two clusters.

        Result is cached by (n_nodes, n_edges) since the DAG is static
        during the ideation loop — avoids O(C²×N) recomputation per call.
        """
        fp = self._dag_fingerprint(dag)
        with self._gaps_lock:
            if fp in self._gaps_cache:
                return self._gaps_cache[fp]

        cluster_ids = dag.get_cluster_ids()
        gaps = []
        for c1, c2 in combinations(cluster_ids, 2):
            if not dag.has_cross_cluster_edge(c1, c2):
                gaps.append((c1, c2))

        with self._gaps_lock:
            self._gaps_cache[fp] = gaps
        return gaps

    # ─────────────────────────────────────────────────────────────────────────
    # Convenience: pick cluster pair for strategies B / C
    # ─────────────────────────────────────────────────────────────────────────
    def pick_cluster_pair(
        self,
        dag: KnowledgeDAG,
        strategy: str,
    ) -> Optional[Tuple[int, int]]:
        """
        For strategy B: pick a cluster pair weighted by maturity_bonus / (use_count+1).
        For strategy C: prefer structural gaps, also weighted by use count.
        Returns None if not enough clusters.

        Maturity-aware weighting (Strategy B):
          Pairing a mature cluster with an emerging/developing one generates the
          most interesting bridging ideas — established methods meeting new problems.
          Cross-maturity pairs receive a 2× weight bonus; same-maturity pairs score 1×.
          Combined with use-count decay this gives:
            weight = maturity_bonus / (use_count + 1)
        """
        cluster_ids = dag.get_cluster_ids()
        if len(cluster_ids) < 2:
            return None

        def _weighted_choice_simple(pairs: List[Tuple[int, int]]) -> Tuple[int, int]:
            """Gap selection: use-count weighting only (no maturity data needed)."""
            # Pre-canonicalize once per pair instead of 3 separate calls per pair.
            # min/max comparison work shared across the dict-build, weights list,
            # and the final use-count update.
            canon = [(a, b) if a <= b else (b, a) for a, b in pairs]
            with self._source_lock:
                puc = self._pair_use_count
                weights = [1.0 / (puc.get(c, 0) + 1) for c in canon]
            idx = random.choices(range(len(pairs)), weights=weights, k=1)[0]
            chosen = pairs[idx]
            chosen_key = canon[idx]
            with self._source_lock:
                self._pair_use_count[chosen_key] = self._pair_use_count.get(chosen_key, 0) + 1
            return chosen

        if strategy == "C":
            gaps = self.find_structural_gaps(dag)
            if gaps:
                return _weighted_choice_simple(gaps)
            # Fall back to any pair when no gaps found

        # ── Maturity-aware pair weighting (Strategy B) ────────────────────────
        # Hoist the cluster metadata dict + per-cluster maturity lookups so each
        # cluster is touched once instead of per-pair (was O(P) lookups, now O(C)).
        cluster_meta = dag.cluster_metadata
        maturity_by_cluster = {
            cid: (cluster_meta.get(cid) or {}).get("maturity", "developing")
            for cid in cluster_ids
        }

        # Build pairs once, in canonical form, so downstream loops do no min/max work.
        pairs: List[Tuple[int, int]] = []
        canon_pairs: List[Tuple[int, int]] = []
        for i, a in enumerate(cluster_ids):
            mat_a = maturity_by_cluster[a]
            for b in cluster_ids[i + 1:]:
                pairs.append((a, b))
                canon_pairs.append((a, b) if a <= b else (b, a))

        with self._source_lock:
            puc = self._pair_use_count
            counts = [puc.get(c, 0) for c in canon_pairs]
        weights = [
            (2.0 if maturity_by_cluster[a] != maturity_by_cluster[b] else 1.0) / (cnt + 1)
            for (a, b), cnt in zip(pairs, counts)
        ]
        idx = random.choices(range(len(pairs)), weights=weights, k=1)[0]
        chosen = pairs[idx]
        chosen_key = canon_pairs[idx]
        with self._source_lock:
            self._pair_use_count[chosen_key] = self._pair_use_count.get(chosen_key, 0) + 1
        return chosen

    # ─────────────────────────────────────────────────────────────────────────
    # Convenience: pick frontier paper for strategy A (diversity-weighted)
    # ─────────────────────────────────────────────────────────────────────────
    _CURRENT_YEAR = 2026  # recency baseline

    def pick_frontier_paper(self, dag: KnowledgeDAG) -> Optional[str]:
        """
        Return a frontier paper_id weighted by recency_boost / (use_count+1).

        Two signals combined:
          - Recency boost:  year/2026 — a 2025 paper scores ~0.96, a 2015 paper ~0.58.
            Newer papers are better frontier candidates because they already
            incorporate recent advances and have more unexplored extensions.
          - Diversity decay: 1/(use_count+1) — same as before, prevents over-use.

        Combined weight = recency_boost / (use_count + 1).
        """
        frontier = dag.get_frontier_nodes()
        papers = frontier if frontier else dag.get_all_papers()
        if not papers:
            return None

        with self._source_lock:
            counts = {p.paper_id: self._paper_use_count.get(p.paper_id, 0) for p in papers}

        def _recency_boost(paper) -> float:
            year = getattr(paper, "year", 0) or 0
            # Normalise years into 0–1 range relative to 2000–2026 window.
            # 2025 → 0.96, 2020 → 0.77, 2015 → 0.58 — meaningful separation.
            return max(year - 2000, 1) / (self._CURRENT_YEAR - 2000)

        weights = [_recency_boost(p) / (counts[p.paper_id] + 1) for p in papers]
        chosen = random.choices(papers, weights=weights, k=1)[0]

        with self._source_lock:
            self._paper_use_count[chosen.paper_id] = (
                self._paper_use_count.get(chosen.paper_id, 0) + 1
            )
        return chosen.paper_id
