"""
aesthetic_optimization.py - Cross-domain creative optimization for IdeaGraph.

Layer 9: Techniques drawn from finance, pedagogy, philosophy, narrative
theory, ecology, architecture, sports analytics, and music theory.
The most unconventional — and often most powerful — optimization ideas
come from mapping concepts across distant domains.

  1.  PortfolioOptimizer       — Markowitz mean-variance idea portfolio
  2.  SpacedRepetitionEngine   — Leitner-box knowledge retention scheduler
  3.  DialecticalSynthesizer   — Hegelian thesis-antithesis-synthesis refinement
  4.  NarrativeTensionArc      — Dramatic pacing for pipeline stages
  5.  RiskParityAllocator      — Equal-risk-contribution budget allocation
  6.  ZoneOfProximalDev        — Vygotsky's ZPD for difficulty targeting
  7.  DesireLineOptimizer      — Emergent optimal paths from usage patterns
  8.  HarmonicResonance        — Frequency-domain config optimization
  9.  SabermetricRanker        — WAR-like composite stats for idea evaluation
  10. LotkaVolterraDynamics    — Predator-prey ecosystem for idea populations
  11. GoldenRatioComposer      — Aesthetic proportions for prompt structure
  12. HerosJourneyPipeline     — Narrative arc structure for pipeline flow
"""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ============================================================================
# 1. Markowitz Portfolio Optimizer
# ============================================================================

class PortfolioOptimizer:
    """
    Markowitz mean-variance optimization for idea portfolio construction.

    Instead of picking the single "best" idea, construct a diversified
    portfolio that maximizes expected return (quality) for a given level
    of risk (variance), or equivalently minimizes risk for a target return.

    Sharpe ratio: S = (E[R] - R_f) / σ  (higher = better risk-adjusted return)

    An idea portfolio with uncorrelated ideas has lower variance than
    any single idea — diversification reduces risk.
    """

    @dataclass
    class Asset:
        id: str
        expected_return: float  # mean quality
        volatility: float       # quality std dev
        scores: List[float] = field(default_factory=list)

    def __init__(self, risk_free_rate: float = 0.1):
        self.rf = risk_free_rate
        self._assets: Dict[str, "PortfolioOptimizer.Asset"] = {}

    def add_idea(self, idea_id: str, quality_scores: List[float]) -> None:
        """Add an idea with its observed quality scores."""
        if len(quality_scores) < 2:
            quality_scores = quality_scores + [quality_scores[0] if quality_scores else 0.5]
        mean = sum(quality_scores) / len(quality_scores)
        var = sum((s - mean) ** 2 for s in quality_scores) / len(quality_scores)
        self._assets[idea_id] = self.Asset(
            id=idea_id, expected_return=mean,
            volatility=math.sqrt(var + 1e-6), scores=quality_scores,
        )

    def sharpe_ratio(self, idea_id: str) -> float:
        """Compute Sharpe ratio for a single idea."""
        asset = self._assets.get(idea_id)
        if not asset:
            return 0
        return (asset.expected_return - self.rf) / max(asset.volatility, 0.01)

    def optimal_portfolio(self, n_select: int = 3) -> List[Tuple[str, float]]:
        """
        Select n ideas that form the best risk-adjusted portfolio.

        Uses Sharpe ratio ranking + correlation-based diversification.
        """
        if not self._assets:
            return []

        # Rank by Sharpe ratio
        ranked = sorted(
            self._assets.items(),
            key=lambda x: self.sharpe_ratio(x[0]),
            reverse=True,
        )

        # Greedy diversification: add ideas that are least correlated with portfolio
        portfolio = [ranked[0][0]]
        for candidate_id, _ in ranked[1:]:
            if len(portfolio) >= n_select:
                break
            # Check correlation with existing portfolio
            max_corr = max(
                self._correlation(candidate_id, p_id)
                for p_id in portfolio
            )
            if max_corr < 0.8:  # only add if not too correlated
                portfolio.append(candidate_id)

        # Compute weights (inverse volatility weighting)
        total_inv_vol = sum(1 / max(self._assets[p].volatility, 0.01) for p in portfolio)
        weights = [
            (p, (1 / max(self._assets[p].volatility, 0.01)) / total_inv_vol)
            for p in portfolio
        ]
        return weights

    def _correlation(self, id_a: str, id_b: str) -> float:
        """Estimate correlation between two ideas' quality streams."""
        a = self._assets.get(id_a)
        b = self._assets.get(id_b)
        if not a or not b:
            return 0
        n = min(len(a.scores), len(b.scores))
        if n < 2:
            return 0
        ma = sum(a.scores[:n]) / n
        mb = sum(b.scores[:n]) / n
        cov = sum((a.scores[i] - ma) * (b.scores[i] - mb) for i in range(n)) / n
        sa = math.sqrt(sum((s - ma) ** 2 for s in a.scores[:n]) / n + 1e-8)
        sb = math.sqrt(sum((s - mb) ** 2 for s in b.scores[:n]) / n + 1e-8)
        return cov / (sa * sb)

    def stats(self) -> Dict[str, Any]:
        return {
            "assets": len(self._assets),
            "top_sharpe": sorted(
                [(k, round(self.sharpe_ratio(k), 2)) for k in self._assets],
                key=lambda x: x[1], reverse=True,
            )[:5],
        }


# ============================================================================
# 2. Spaced Repetition Engine (Leitner System)
# ============================================================================

class SpacedRepetitionEngine:
    """
    Leitner-box system for knowledge retention across pipeline runs.

    Knowledge items (lessons, facts, patterns) are placed in boxes:
      Box 1: review every run (new/difficult items)
      Box 2: review every 2 runs
      Box 3: review every 4 runs
      Box 4: review every 8 runs (well-learned items)

    Correct recall → promote to next box (longer interval)
    Incorrect recall → demote to Box 1 (re-learn)

    This optimizes which knowledge is injected into prompts:
    well-known facts don't waste tokens; forgotten facts get reinforced.
    """

    @dataclass
    class Card:
        content: str
        box: int = 1  # 1-4
        last_reviewed: int = 0  # run number
        correct_streak: int = 0
        total_reviews: int = 0

    def __init__(self):
        self._cards: Dict[str, "SpacedRepetitionEngine.Card"] = {}
        self._run_number = 0

    def add_card(self, key: str, content: str) -> None:
        if key not in self._cards:
            self._cards[key] = self.Card(content=content)

    def get_due_cards(self, max_cards: int = 5) -> List["SpacedRepetitionEngine.Card"]:
        """Get cards due for review this run."""
        due = []
        for key, card in self._cards.items():
            interval = 2 ** (card.box - 1)  # 1, 2, 4, 8
            runs_since = self._run_number - card.last_reviewed
            if runs_since >= interval:
                due.append((card.box, key, card))  # lower box = higher priority

        due.sort()
        return [card for _, _, card in due[:max_cards]]

    def review(self, key: str, correct: bool) -> None:
        """Record a review outcome."""
        card = self._cards.get(key)
        if not card:
            return
        card.last_reviewed = self._run_number
        card.total_reviews += 1
        if correct:
            card.correct_streak += 1
            card.box = min(4, card.box + 1)
        else:
            card.correct_streak = 0
            card.box = 1

    def advance_run(self) -> None:
        self._run_number += 1

    def get_context_for_run(self, max_tokens: int = 500) -> str:
        """Get knowledge context for current run from due cards."""
        due = self.get_due_cards()
        parts = []
        chars = 0
        for card in due:
            if chars + len(card.content) > max_tokens * 4:
                break
            parts.append(f"- {card.content}")
            chars += len(card.content)
        return "\n".join(parts)

    def stats(self) -> Dict[str, Any]:
        boxes = defaultdict(int)
        for card in self._cards.values():
            boxes[card.box] += 1
        return {
            "total_cards": len(self._cards),
            "run_number": self._run_number,
            "by_box": dict(boxes),
            "due_now": len(self.get_due_cards()),
        }


# ============================================================================
# 3. Dialectical Synthesizer (Hegel)
# ============================================================================

class DialecticalSynthesizer:
    """
    Hegelian thesis-antithesis-synthesis for idea refinement.

    Every idea (thesis) has a counter-argument (antithesis).
    The resolution (synthesis) transcends both by incorporating
    the best elements of each into a higher-order idea.

    This is systematically more creative than simple iteration because
    it REQUIRES confronting contradictions rather than ignoring them.

    Process:
      1. Thesis: the original idea
      2. Antithesis: strongest possible counter-argument
      3. Synthesis: new idea that resolves the contradiction
    """

    @dataclass
    class Dialectic:
        thesis: str
        antithesis: str
        synthesis: str
        iteration: int
        quality_gain: float = 0.0

    def __init__(self):
        self._history: List["DialecticalSynthesizer.Dialectic"] = []
        self._synthesis_count = 0

    def build_antithesis_prompt(self, thesis: Dict[str, Any]) -> Dict[str, str]:
        """Build a prompt to generate the antithesis of an idea."""
        return {
            "system": (
                "You are a dialectical thinker. Given a research thesis, generate "
                "the strongest possible ANTITHESIS — a counter-approach that challenges "
                "the fundamental assumptions. The antithesis should be intellectually "
                "rigorous, not just a negation. It should propose a genuinely different "
                "paradigm for solving the same problem.\n\n"
                "Return JSON: {\"antithesis_title\": \"...\", \"antithesis_method\": \"...\", "
                "\"contradiction\": \"what fundamental assumption does this challenge?\", "
                "\"strength\": \"why this counter-approach might be superior\"}"
            ),
            "user": (
                f"THESIS:\n"
                f"  Title: {thesis.get('title', '')}\n"
                f"  Method: {thesis.get('method', '')}\n"
                f"  Hypothesis: {thesis.get('hypothesis', '')}\n\n"
                f"Generate the strongest possible ANTITHESIS."
            ),
        }

    def build_synthesis_prompt(self, thesis: Dict, antithesis: Dict) -> Dict[str, str]:
        """Build a prompt to synthesize thesis and antithesis."""
        return {
            "system": (
                "You are a dialectical synthesizer. Given a thesis and its antithesis, "
                "create a SYNTHESIS that transcends both by resolving their contradiction. "
                "The synthesis should be stronger than either alone — incorporating the "
                "best elements of both while addressing their weaknesses.\n\n"
                "Return JSON: {\"title\": \"...\", \"method\": \"...\", \"hypothesis\": \"...\", "
                "\"resolution\": \"how this resolves the thesis-antithesis contradiction\", "
                "\"novel_elements\": [\"what's new in the synthesis\"]}"
            ),
            "user": (
                f"THESIS:\n"
                f"  Title: {thesis.get('title', '')}\n"
                f"  Method: {thesis.get('method', '')}\n\n"
                f"ANTITHESIS:\n"
                f"  Title: {antithesis.get('antithesis_title', '')}\n"
                f"  Method: {antithesis.get('antithesis_method', '')}\n"
                f"  Contradiction: {antithesis.get('contradiction', '')}\n\n"
                f"Create a SYNTHESIS that transcends both."
            ),
        }

    def record_synthesis(self, thesis: str, antithesis: str, synthesis: str,
                         quality_gain: float = 0) -> None:
        self._synthesis_count += 1
        self._history.append(self.Dialectic(
            thesis=thesis, antithesis=antithesis, synthesis=synthesis,
            iteration=self._synthesis_count, quality_gain=quality_gain,
        ))

    def stats(self) -> Dict[str, Any]:
        return {
            "syntheses": self._synthesis_count,
            "avg_quality_gain": round(
                sum(d.quality_gain for d in self._history) / max(len(self._history), 1), 3
            ),
        }


# ============================================================================
# 4. Narrative Tension Arc
# ============================================================================

class NarrativeTensionArc:
    """
    Dramatic pacing for pipeline stages — the pipeline as a story.

    Research shows that engagement follows narrative arcs. Apply this
    to pipeline resource allocation:

      Act 1 (Setup, 25%): Ideation — establish the research landscape
      Act 2a (Rising Action, 25%): Design + Code — build toward experiment
      Act 2b (Climax, 25%): Execution + Analysis — the critical test
      Act 3 (Resolution, 25%): Paper + Review — wrap up and evaluate

    Resource tension follows the arc: moderate → rising → peak → falling.
    This naturally frontloads exploration and backloads refinement.
    """

    ACTS = {
        "setup": {"stages": ["ideation", "tree_search"], "tension": 0.4, "resource_pct": 0.25},
        "rising": {"stages": ["experiment_design", "code_generation"], "tension": 0.7, "resource_pct": 0.25},
        "climax": {"stages": ["execution", "analysis"], "tension": 1.0, "resource_pct": 0.30},
        "resolution": {"stages": ["paper_writing", "review"], "tension": 0.5, "resource_pct": 0.20},
    }

    def __init__(self):
        self._current_act = "setup"
        self._tension_history: List[float] = []

    def get_act(self, stage: str) -> str:
        for act_name, act_info in self.ACTS.items():
            if stage in act_info["stages"]:
                return act_name
        return "setup"

    def get_tension(self, stage: str) -> float:
        """Get narrative tension level for a stage (0-1)."""
        act = self.get_act(stage)
        return self.ACTS[act]["tension"]

    def get_resource_allocation(self, stage: str, total: float) -> float:
        """Get resource allocation based on narrative arc."""
        act = self.get_act(stage)
        act_info = self.ACTS[act]
        n_stages = len(act_info["stages"])
        return total * act_info["resource_pct"] / max(n_stages, 1)

    def tension_to_temperature(self, stage: str) -> float:
        """Map narrative tension to LLM temperature."""
        tension = self.get_tension(stage)
        # High tension → higher creativity (more at stake)
        return 0.3 + tension * 0.5

    def record_tension(self, stage: str) -> None:
        tension = self.get_tension(stage)
        self._tension_history.append(tension)
        self._current_act = self.get_act(stage)

    def stats(self) -> Dict[str, Any]:
        return {
            "current_act": self._current_act,
            "tension_history": [round(t, 2) for t in self._tension_history],
            "acts": {k: v["tension"] for k, v in self.ACTS.items()},
        }


# ============================================================================
# 5. Risk Parity Allocator
# ============================================================================

class RiskParityAllocator:
    """
    Equal-risk-contribution budget allocation across pipeline stages.

    From portfolio theory: instead of equal dollar allocation, allocate
    so each stage contributes EQUAL RISK to the total pipeline risk.

    Stages with high variance (unpredictable outcomes) get LESS budget.
    Stages with low variance (reliable outcomes) get MORE budget.

    Risk contribution: RC_i = w_i × σ_i × ρ_i
    Target: RC_i = RC_j for all i, j (equal risk contribution)
    """

    def __init__(self):
        self._stage_volatility: Dict[str, float] = {}
        self._stage_quality: Dict[str, List[float]] = defaultdict(list)

    def record_quality(self, stage: str, quality: float) -> None:
        self._stage_quality[stage].append(quality)
        # Update volatility
        scores = self._stage_quality[stage]
        if len(scores) >= 2:
            mean = sum(scores) / len(scores)
            var = sum((s - mean) ** 2 for s in scores) / len(scores)
            self._stage_volatility[stage] = math.sqrt(var + 1e-6)

    def allocate(self, total_budget: float) -> Dict[str, float]:
        """Allocate budget with equal risk contribution."""
        if not self._stage_volatility:
            # Equal allocation fallback
            stages = list(self._stage_quality.keys()) or ["default"]
            return {s: total_budget / len(stages) for s in stages}

        # Inverse volatility weighting (approximation of risk parity)
        inv_vols = {s: 1.0 / max(v, 0.01) for s, v in self._stage_volatility.items()}
        total_inv = sum(inv_vols.values())
        return {s: total_budget * (iv / total_inv) for s, iv in inv_vols.items()}

    def stats(self) -> Dict[str, Any]:
        return {
            "volatilities": {k: round(v, 3) for k, v in self._stage_volatility.items()},
            "observations": {k: len(v) for k, v in self._stage_quality.items()},
        }


# ============================================================================
# 6. Zone of Proximal Development (Vygotsky)
# ============================================================================

class ZoneOfProximalDev:
    """
    Vygotsky's ZPD applied to LLM task difficulty targeting.

    The Zone of Proximal Development is the sweet spot between:
      - Too easy (boring, no learning, wasted potential)
      - Too hard (impossible, frustrating, produces garbage)
      - Just right (challenging but achievable with scaffolding)

    Applied to LLM prompts:
      - Track task difficulty vs quality achieved
      - Target the difficulty level where quality is 60-80% (ZPD)
      - Too easy (quality > 90%) → increase difficulty
      - Too hard (quality < 40%) → provide more scaffolding
    """

    def __init__(self, zpd_low: float = 0.6, zpd_high: float = 0.8):
        self.zpd_low = zpd_low
        self.zpd_high = zpd_high
        self._difficulty: Dict[str, float] = defaultdict(lambda: 0.5)
        self._history: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

    def record(self, task_type: str, difficulty: float, quality: float) -> None:
        self._history[task_type].append((difficulty, quality))

    def get_target_difficulty(self, task_type: str) -> float:
        """Get the target difficulty for a task type (in the ZPD)."""
        history = self._history.get(task_type, [])
        if len(history) < 3:
            return self._difficulty[task_type]

        # Find difficulty level that produces quality in [zpd_low, zpd_high]
        recent = history[-10:]
        in_zpd = [(d, q) for d, q in recent if self.zpd_low <= q <= self.zpd_high]

        if in_zpd:
            # Average difficulty of in-ZPD attempts
            self._difficulty[task_type] = sum(d for d, _ in in_zpd) / len(in_zpd)
        else:
            # Adjust: if quality too high, increase difficulty; if too low, decrease
            avg_q = sum(q for _, q in recent) / len(recent)
            if avg_q > self.zpd_high:
                self._difficulty[task_type] = min(1.0, self._difficulty[task_type] + 0.1)
            elif avg_q < self.zpd_low:
                self._difficulty[task_type] = max(0.1, self._difficulty[task_type] - 0.1)

        return self._difficulty[task_type]

    def difficulty_to_tokens(self, difficulty: float, base_tokens: int = 4096) -> int:
        """Map difficulty to max_tokens (harder → more tokens for reasoning)."""
        return int(base_tokens * (0.5 + difficulty))

    def difficulty_to_scaffolding(self, difficulty: float) -> str:
        """Get scaffolding level based on difficulty."""
        if difficulty > 0.8:
            return "minimal"  # expert mode
        elif difficulty > 0.5:
            return "moderate"  # some guidance
        return "heavy"  # step-by-step instructions

    def stats(self) -> Dict[str, Any]:
        return {
            "difficulties": {k: round(v, 3) for k, v in self._difficulty.items()},
            "observations": {k: len(v) for k, v in self._history.items()},
        }


# ============================================================================
# 7. Desire Line Optimizer
# ============================================================================

class DesireLineOptimizer:
    """
    Emergent optimal paths from usage patterns.

    In urban planning, "desire lines" are paths worn into grass by
    pedestrians — they reveal the ACTUAL optimal route, not the
    designed one.

    Applied to pipelines: track which stage orderings and skip patterns
    ACTUALLY produce the best results. Let the data reveal the optimal
    pipeline structure instead of imposing it top-down.
    """

    def __init__(self):
        self._paths: List[Tuple[List[str], float]] = []  # (stage_sequence, quality)
        self._edge_quality: Dict[Tuple[str, str], List[float]] = defaultdict(list)

    def record_path(self, stages_executed: List[str], quality: float) -> None:
        """Record a pipeline execution path and its outcome."""
        self._paths.append((stages_executed, quality))
        for i in range(len(stages_executed) - 1):
            edge = (stages_executed[i], stages_executed[i + 1])
            self._edge_quality[edge].append(quality)

    def get_desire_line(self) -> List[str]:
        """Find the emergent optimal path from historical data."""
        if not self._paths:
            return []

        # Weight paths by quality
        weighted = sorted(self._paths, key=lambda x: x[1], reverse=True)
        # Most common path among top 30%
        top_n = max(1, len(weighted) // 3)
        top_paths = [p for p, _ in weighted[:top_n]]

        # Find most common edges
        edge_freq: Dict[Tuple[str, str], int] = defaultdict(int)
        for path in top_paths:
            for i in range(len(path) - 1):
                edge_freq[(path[i], path[i + 1])] += 1

        # Reconstruct best path by following highest-frequency edges
        if not edge_freq:
            return top_paths[0] if top_paths else []

        # Greedy path construction
        starts = set(e[0] for e in edge_freq)
        ends = set(e[1] for e in edge_freq)
        start_nodes = starts - ends
        current = min(start_nodes) if start_nodes else list(starts)[0]

        path = [current]
        visited = {current}
        for _ in range(20):
            next_edges = [(dst, edge_freq[(current, dst)])
                          for dst in set(e[1] for e in edge_freq if e[0] == current)
                          if dst not in visited]
            if not next_edges:
                break
            next_node = max(next_edges, key=lambda x: x[1])[0]
            path.append(next_node)
            visited.add(next_node)
            current = next_node

        return path

    def get_shortcut(self) -> Optional[Tuple[str, str]]:
        """Find the most valuable shortcut (skipping intermediate stages)."""
        if not self._edge_quality:
            return None

        # Find non-adjacent edges with high quality
        best_shortcut = None
        best_quality = 0
        for (src, dst), qualities in self._edge_quality.items():
            avg = sum(qualities) / len(qualities)
            if avg > best_quality:
                best_quality = avg
                best_shortcut = (src, dst)

        return best_shortcut

    def stats(self) -> Dict[str, Any]:
        return {
            "paths_recorded": len(self._paths),
            "desire_line": self.get_desire_line(),
            "edges_tracked": len(self._edge_quality),
        }


# ============================================================================
# 8. Harmonic Resonance Optimizer
# ============================================================================

class HarmonicResonance:
    """
    Frequency-domain analysis for finding resonant pipeline configurations.

    Just as physical systems have resonant frequencies where energy
    transfer is maximized, pipelines have "resonant" configurations
    where quality output is maximized for given input.

    Uses DFT on quality time series to find periodic patterns:
      - If quality oscillates with period P, the system has a resonance
      - Tuning configuration parameters to match resonant frequencies
        amplifies quality

    Practical use: detect quality oscillation (e.g., good→bad→good iterations)
    and dampen or amplify it.
    """

    def __init__(self):
        self._signals: Dict[str, List[float]] = defaultdict(list)

    def record(self, channel: str, value: float) -> None:
        self._signals[channel].append(value)

    def detect_oscillation(self, channel: str) -> Dict[str, Any]:
        """Detect oscillation patterns via simplified DFT."""
        signal = self._signals.get(channel, [])
        if len(signal) < 4:
            return {"oscillating": False, "period": 0}

        n = len(signal)
        mean = sum(signal) / n
        centered = [s - mean for s in signal]

        # Find dominant frequency via autocorrelation
        max_corr = 0
        best_lag = 0
        for lag in range(1, n // 2 + 1):
            corr = sum(centered[i] * centered[i + lag] for i in range(n - lag))
            corr /= max(sum(c * c for c in centered), 1e-8)
            if corr > max_corr:
                max_corr = corr
                best_lag = lag

        oscillating = max_corr > 0.3 and best_lag > 0
        return {
            "oscillating": oscillating,
            "period": best_lag,
            "strength": round(max_corr, 3),
            "recommendation": "dampen" if oscillating and max_corr > 0.5 else "stable",
        }

    def dampen(self, channel: str) -> float:
        """Suggest a dampening factor to reduce oscillation."""
        result = self.detect_oscillation(channel)
        if not result["oscillating"]:
            return 1.0
        return max(0.5, 1.0 - result["strength"] * 0.3)

    def stats(self) -> Dict[str, Any]:
        return {
            channel: self.detect_oscillation(channel)
            for channel in self._signals
        }


# ============================================================================
# 9. Sabermetric Ranker (WAR-like)
# ============================================================================

class SabermetricRanker:
    """
    WAR-like composite statistics for idea evaluation.

    Baseball's WAR (Wins Above Replacement) measures total value by
    comparing a player to a "replacement-level" baseline. Applied to ideas:

    IAR (Ideas Above Replacement):
      IAR = (novelty_above_replacement × 0.3
           + feasibility_above_replacement × 0.25
           + impact_above_replacement × 0.25
           + clarity_above_replacement × 0.2)

    Replacement level: the average quality of a random/baseline idea.
    IAR measures value ABOVE what you'd get with no optimization.
    """

    @dataclass
    class IdeaStats:
        id: str
        novelty: float = 0.0
        feasibility: float = 0.0
        impact: float = 0.0
        clarity: float = 0.0
        iar: float = 0.0  # Ideas Above Replacement

    WEIGHTS = {"novelty": 0.3, "feasibility": 0.25, "impact": 0.25, "clarity": 0.2}

    def __init__(self):
        self._ideas: Dict[str, "SabermetricRanker.IdeaStats"] = {}
        self._replacement_level: Dict[str, float] = {
            "novelty": 0.3, "feasibility": 0.4, "impact": 0.2, "clarity": 0.4,
        }

    def add_idea(self, idea_id: str, novelty: float, feasibility: float,
                 impact: float, clarity: float) -> float:
        """Add an idea and compute its IAR."""
        stats = self.IdeaStats(
            id=idea_id, novelty=novelty, feasibility=feasibility,
            impact=impact, clarity=clarity,
        )

        # Compute IAR
        iar = 0.0
        for dim, weight in self.WEIGHTS.items():
            value = getattr(stats, dim)
            above_replacement = value - self._replacement_level[dim]
            iar += weight * above_replacement

        stats.iar = iar
        self._ideas[idea_id] = stats

        # Update replacement level (running average)
        n = len(self._ideas)
        for dim in self._replacement_level:
            values = [getattr(s, dim) for s in self._ideas.values()]
            self._replacement_level[dim] = sum(values) / n

        return iar

    def rank(self) -> List[Tuple[str, float]]:
        """Rank all ideas by IAR."""
        return sorted(
            [(s.id, round(s.iar, 4)) for s in self._ideas.values()],
            key=lambda x: x[1], reverse=True,
        )

    def get_mvp(self) -> Optional[str]:
        """Most Valuable Paper — highest IAR."""
        rankings = self.rank()
        return rankings[0][0] if rankings else None

    def stats(self) -> Dict[str, Any]:
        return {
            "ideas": len(self._ideas),
            "replacement_level": {k: round(v, 3) for k, v in self._replacement_level.items()},
            "top_iar": self.rank()[:5],
            "mvp": self.get_mvp(),
        }


# ============================================================================
# 10. Lotka-Volterra Dynamics
# ============================================================================

class LotkaVolterraDynamics:
    """
    Predator-prey population dynamics for idea ecosystem management.

    Lotka-Volterra equations model interacting populations:
      dx/dt = αx - βxy    (prey: ideas grow naturally, decrease with critics)
      dy/dt = δxy - γy    (predators: critics grow from ideas, decrease naturally)

    Applied to idea pipeline:
      x = number of active ideas (prey)
      y = strictness of quality filtering (predators)

    When too many ideas → increase filtering strictness.
    When too few ideas → relax filtering to let more through.
    System naturally oscillates toward equilibrium.
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.02,
                 delta: float = 0.01, gamma: float = 0.3, dt: float = 0.1):
        self.alpha = alpha  # idea growth rate
        self.beta = beta    # idea loss from filtering
        self.delta = delta  # filter tightening from ideas
        self.gamma = gamma  # natural filter relaxation
        self.dt = dt
        self._x = 10.0  # idea population
        self._y = 1.0   # filter strictness
        self._history_x: List[float] = [self._x]
        self._history_y: List[float] = [self._y]

    def step(self, actual_ideas: int = None) -> Tuple[float, float]:
        """Advance one time step. Returns (idea_target, filter_level)."""
        if actual_ideas is not None:
            self._x = float(actual_ideas)

        dx = (self.alpha * self._x - self.beta * self._x * self._y) * self.dt
        dy = (self.delta * self._x * self._y - self.gamma * self._y) * self.dt

        self._x = max(1, self._x + dx)
        self._y = max(0.1, self._y + dy)

        self._history_x.append(self._x)
        self._history_y.append(self._y)

        return self._x, self._y

    @property
    def quality_threshold(self) -> float:
        """Current quality threshold based on predator (filter) level."""
        return min(0.9, 0.3 + self._y * 0.1)

    @property
    def target_idea_count(self) -> int:
        return max(3, int(self._x))

    def stats(self) -> Dict[str, Any]:
        return {
            "ideas": round(self._x, 1),
            "filter": round(self._y, 2),
            "quality_threshold": round(self.quality_threshold, 3),
            "target_ideas": self.target_idea_count,
            "equilibrium": abs(self._x - self.gamma / self.delta) < 2 if self.delta > 0 else False,
        }


# ============================================================================
# 11. Golden Ratio Composer
# ============================================================================

class GoldenRatioComposer:
    """
    Aesthetic proportions for prompt and output structure.

    The golden ratio φ ≈ 1.618 appears throughout nature and art as an
    aesthetically pleasing proportion. Apply it to prompt structure:

      - System prompt : User prompt ≈ 1 : φ
      - Context : Instruction ≈ φ : 1
      - Examples : Rules ≈ 1 : φ
      - Total prompt : Expected output ≈ φ : 1

    Research suggests LLMs produce better outputs when prompts have
    good structural proportions (neither too front-heavy nor back-heavy).
    """

    PHI = (1 + math.sqrt(5)) / 2  # ≈ 1.618

    def __init__(self):
        self._quality_by_ratio: List[Tuple[float, float]] = []

    def ideal_proportions(self, total_tokens: int) -> Dict[str, int]:
        """Compute ideal token allocation for a prompt."""
        # System : User = 1 : φ
        system_tokens = int(total_tokens / (1 + self.PHI))
        user_tokens = total_tokens - system_tokens

        # Within user: Context : Instruction = φ : 1
        context_tokens = int(user_tokens * self.PHI / (1 + self.PHI))
        instruction_tokens = user_tokens - context_tokens

        return {
            "system_tokens": system_tokens,
            "user_tokens": user_tokens,
            "context_tokens": context_tokens,
            "instruction_tokens": instruction_tokens,
        }

    def measure_ratio(self, system_len: int, user_len: int) -> float:
        """Measure how close the system/user ratio is to golden."""
        if system_len == 0:
            return 0
        actual_ratio = user_len / max(system_len, 1)
        ideal = self.PHI
        return 1.0 - min(1.0, abs(actual_ratio - ideal) / ideal)

    def suggest_rebalance(self, system: str, user: str) -> Dict[str, str]:
        """Suggest how to rebalance prompt proportions."""
        s_len = len(system.split())
        u_len = len(user.split())
        ratio = u_len / max(s_len, 1)

        if ratio < self.PHI * 0.7:
            return {"action": "expand_user", "reason": "User prompt too short relative to system"}
        elif ratio > self.PHI * 1.5:
            return {"action": "condense_user", "reason": "User prompt too long relative to system"}
        return {"action": "balanced", "reason": "Proportions are near golden ratio"}

    def record_quality(self, ratio: float, quality: float) -> None:
        self._quality_by_ratio.append((ratio, quality))

    def stats(self) -> Dict[str, Any]:
        return {
            "phi": round(self.PHI, 3),
            "observations": len(self._quality_by_ratio),
        }


# ============================================================================
# 12. Hero's Journey Pipeline
# ============================================================================

class HerosJourneyPipeline:
    """
    Joseph Campbell's monomyth applied to pipeline stage structure.

    The Hero's Journey maps perfectly to scientific research:
      1. Ordinary World: existing knowledge (DAG construction)
      2. Call to Adventure: research gap identified (ideation)
      3. Refusal of the Call: adversarial testing (is this worth pursuing?)
      4. Crossing the Threshold: experiment design (commitment to approach)
      5. Tests & Allies: code generation + execution (the hard work)
      6. Ordeal: analysis (confronting results, good or bad)
      7. Reward: paper writing (capturing the discovery)
      8. Return: review (sharing with the world)

    Each stage has a "narrative role" that determines its energy and focus.
    """

    STAGES = {
        "ordinary_world": {
            "pipeline_stage": "dag_construction",
            "energy": 0.3, "focus": "understanding",
            "description": "Survey the existing knowledge landscape",
        },
        "call_to_adventure": {
            "pipeline_stage": "ideation",
            "energy": 0.6, "focus": "discovery",
            "description": "Identify the research gap worth exploring",
        },
        "refusal_of_call": {
            "pipeline_stage": "adversarial_testing",
            "energy": 0.5, "focus": "skepticism",
            "description": "Challenge whether the idea is truly worth pursuing",
        },
        "crossing_threshold": {
            "pipeline_stage": "experiment_design",
            "energy": 0.7, "focus": "commitment",
            "description": "Commit to a specific experimental approach",
        },
        "tests_and_allies": {
            "pipeline_stage": "code_generation",
            "energy": 0.8, "focus": "execution",
            "description": "Build the tools and run the experiments",
        },
        "ordeal": {
            "pipeline_stage": "execution_analysis",
            "energy": 1.0, "focus": "truth",
            "description": "Confront the results — success or failure",
        },
        "reward": {
            "pipeline_stage": "paper_writing",
            "energy": 0.7, "focus": "synthesis",
            "description": "Capture the discovery in a paper",
        },
        "return": {
            "pipeline_stage": "review",
            "energy": 0.4, "focus": "sharing",
            "description": "Share findings with the scientific community",
        },
    }

    def __init__(self):
        self._current_stage = "ordinary_world"
        self._journey_log: List[Dict] = []

    def advance(self, stage_name: str, outcome: str = "success") -> Dict[str, Any]:
        """Advance the hero's journey."""
        for journey_stage, info in self.STAGES.items():
            if info["pipeline_stage"] in stage_name or stage_name in info["pipeline_stage"]:
                self._current_stage = journey_stage
                entry = {
                    "journey_stage": journey_stage,
                    "pipeline_stage": stage_name,
                    "outcome": outcome,
                    "energy": info["energy"],
                }
                self._journey_log.append(entry)
                return entry
        return {"journey_stage": "unknown", "pipeline_stage": stage_name}

    def get_energy(self) -> float:
        """Current narrative energy level."""
        info = self.STAGES.get(self._current_stage, {})
        return info.get("energy", 0.5)

    def get_focus(self) -> str:
        """Current narrative focus."""
        info = self.STAGES.get(self._current_stage, {})
        return info.get("focus", "execution")

    def is_at_ordeal(self) -> bool:
        return self._current_stage == "ordeal"

    def stats(self) -> Dict[str, Any]:
        return {
            "current_stage": self._current_stage,
            "energy": self.get_energy(),
            "focus": self.get_focus(),
            "journey_length": len(self._journey_log),
        }


# ============================================================================
# Master Aesthetic Optimizer
# ============================================================================

class AestheticOptimizer:
    """Aggregates all aesthetic/cross-domain optimization techniques."""

    def __init__(self, enable_all: bool = True):
        self.portfolio = PortfolioOptimizer() if enable_all else None
        self.spaced_rep = SpacedRepetitionEngine() if enable_all else None
        self.dialectic = DialecticalSynthesizer() if enable_all else None
        self.narrative = NarrativeTensionArc() if enable_all else None
        self.risk_parity = RiskParityAllocator() if enable_all else None
        self.zpd = ZoneOfProximalDev() if enable_all else None
        self.desire_line = DesireLineOptimizer() if enable_all else None
        self.harmonic = HarmonicResonance() if enable_all else None
        self.sabermetric = SabermetricRanker() if enable_all else None
        self.lotka_volterra = LotkaVolterraDynamics() if enable_all else None
        self.golden_ratio = GoldenRatioComposer() if enable_all else None
        self.heros_journey = HerosJourneyPipeline() if enable_all else None

    def summary(self) -> Dict[str, Any]:
        result = {}
        if self.portfolio: result["portfolio"] = self.portfolio.stats()
        if self.spaced_rep: result["spaced_repetition"] = self.spaced_rep.stats()
        if self.dialectic: result["dialectical"] = self.dialectic.stats()
        if self.narrative: result["narrative_arc"] = self.narrative.stats()
        if self.risk_parity: result["risk_parity"] = self.risk_parity.stats()
        if self.zpd: result["zone_of_proximal_dev"] = self.zpd.stats()
        if self.desire_line: result["desire_lines"] = self.desire_line.stats()
        if self.harmonic: result["harmonic_resonance"] = self.harmonic.stats()
        if self.sabermetric: result["sabermetrics"] = self.sabermetric.stats()
        if self.lotka_volterra: result["lotka_volterra"] = self.lotka_volterra.stats()
        if self.golden_ratio: result["golden_ratio"] = self.golden_ratio.stats()
        if self.heros_journey: result["heros_journey"] = self.heros_journey.stats()
        return result
