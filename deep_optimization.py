"""
deep_optimization.py - Deep ML-inspired optimization techniques for IdeaGraph.

Advanced techniques that push beyond standard and creative optimization:

  1.  EnsembleDistiller      — Multi-model ensemble with learned weighting
  2.  MultiFidelityOptimizer — Run cheap approximations, promote promising candidates
  3.  BayesianTuner          — Gaussian Process surrogate for pipeline config
  4.  KnapsackBudget         — 0/1 knapsack for optimal stage budget allocation
  5.  ProgressiveElaborator  — Skeleton-first, progressively add detail
  6.  ContrastivePromptPair  — Generate "do/don't" prompt pairs for better output
  7.  AttentionRollback      — Identify which upstream stage to re-run on failure
  8.  ResponseDistiller      — Compress verbose outputs into reusable knowledge nuggets
  9.  EntropyRegularizer     — Maintain diversity pressure across idea populations
  10. RewardShaper           — Hindsight experience replay for failed experiments
  11. TokenBudgetProjector   — Real-time token spend projections with early-stop
  12. DynamicBatchSizer      — Auto-tune batch size based on latency + remaining budget
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ============================================================================
# 1. Ensemble Distiller (Multi-Model Aggregation)
# ============================================================================

class EnsembleDistiller:
    """
    Combine outputs from multiple LLM calls with learned weighting.

    Inspired by mixture-of-experts and ensemble learning in ML.

    Strategy:
      - Run the same prompt through K different temperature/system variations
      - Score each output independently
      - Combine using weighted voting (weights learned from historical accuracy)
      - Distill consensus into a single high-quality output

    Use case: critical decisions (best idea selection, paper review) where
    a single LLM call might be noisy but an ensemble is more robust.
    """

    def __init__(self, n_ensemble: int = 3):
        self.n_ensemble = n_ensemble
        # variation → (successes, total)
        self._weights: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def create_variations(self, system: str, user: str) -> List[Dict[str, Any]]:
        """Create ensemble variations of a prompt."""
        variations = [
            {
                "id": "precise",
                "system": system + "\nBe precise and analytical. Focus on accuracy.",
                "user": user,
                "temperature": 0.2,
            },
            {
                "id": "creative",
                "system": system + "\nBe creative and think broadly. Consider unconventional angles.",
                "user": user,
                "temperature": 0.8,
            },
            {
                "id": "critical",
                "system": system + "\nBe critical and skeptical. Challenge assumptions.",
                "user": user,
                "temperature": 0.5,
            },
        ]
        return variations[:self.n_ensemble]

    def get_weight(self, variation_id: str) -> float:
        """Get learned weight for a variation."""
        with self._lock:
            w = self._weights.get(variation_id, [1.0, 1.0])
            return w[0] / max(w[1], 1)

    def update_weight(self, variation_id: str, quality: float) -> None:
        """Update weight based on observed quality."""
        with self._lock:
            if variation_id not in self._weights:
                self._weights[variation_id] = [0.0, 0.0]
            self._weights[variation_id][0] += quality
            self._weights[variation_id][1] += 1

    def aggregate_scores(self, results: List[Tuple[str, float]]) -> float:
        """Weighted average of scores from ensemble members."""
        total_weight = 0
        weighted_sum = 0
        for var_id, score in results:
            w = self.get_weight(var_id)
            weighted_sum += w * score
            total_weight += w
        return weighted_sum / max(total_weight, 0.001)

    def stats(self) -> Dict[str, Any]:
        return {
            "n_ensemble": self.n_ensemble,
            "weights": {k: round(self.get_weight(k), 3) for k in self._weights},
        }


# ============================================================================
# 2. Multi-Fidelity Optimizer
# ============================================================================

@dataclass
class FidelityLevel:
    """Definition of a fidelity level."""
    name: str
    max_tokens: int
    temperature: float
    cost_multiplier: float  # relative cost vs full fidelity


class MultiFidelityOptimizer:
    """
    Run cheap approximations first, promote only promising candidates to full eval.

    Inspired by Successive Halving (Jamieson & Talwalkar, 2016) and Hyperband.

    Fidelity levels:
      Low:    256 tokens, high temp → quick screening (1x cost)
      Medium: 1024 tokens, balanced → serious evaluation (3x cost)
      High:   4096 tokens, precise → full generation (10x cost)

    Algorithm:
      1. Generate N candidates at low fidelity
      2. Promote top-N/3 to medium fidelity
      3. Promote top-N/9 to high fidelity
      4. Return the best high-fidelity result

    Total cost ≈ N×1 + N/3×3 + N/9×10 ≈ 3.1N vs N×10 = 10N for full eval.
    Savings: ~70% when most candidates are weak.
    """

    FIDELITY_LEVELS = [
        FidelityLevel("low", max_tokens=256, temperature=0.7, cost_multiplier=1.0),
        FidelityLevel("medium", max_tokens=1024, temperature=0.5, cost_multiplier=3.0),
        FidelityLevel("high", max_tokens=4096, temperature=0.3, cost_multiplier=10.0),
    ]

    def __init__(self, halving_rate: float = 3.0):
        self.halving_rate = halving_rate
        self._level_stats: Dict[str, List[float]] = defaultdict(list)

    def get_fidelity_params(self, level: int) -> FidelityLevel:
        """Get parameters for a fidelity level (0=low, 1=medium, 2=high)."""
        return self.FIDELITY_LEVELS[min(level, len(self.FIDELITY_LEVELS) - 1)]

    def compute_promotion_count(self, n_candidates: int, from_level: int) -> int:
        """How many candidates to promote from this level."""
        return max(1, int(n_candidates / (self.halving_rate ** (from_level + 1))))

    def record_level_quality(self, level: str, quality: float) -> None:
        """Track quality at each fidelity level for analysis."""
        self._level_stats[level].append(quality)

    def should_promote(self, quality: float, level: int) -> bool:
        """Should this candidate be promoted based on quality vs level stats?"""
        level_name = self.FIDELITY_LEVELS[min(level, 2)].name
        history = self._level_stats.get(level_name, [])
        if len(history) < 3:
            return quality > 0.3  # default threshold
        # Promote if above median for this level
        sorted_h = sorted(history)
        median = sorted_h[len(sorted_h) // 2]
        return quality >= median

    def estimated_savings(self, n_candidates: int) -> Dict[str, float]:
        """Estimate cost savings vs full evaluation."""
        full_cost = n_candidates * self.FIDELITY_LEVELS[-1].cost_multiplier
        mf_cost = (
            n_candidates * self.FIDELITY_LEVELS[0].cost_multiplier
            + self.compute_promotion_count(n_candidates, 0) * self.FIDELITY_LEVELS[1].cost_multiplier
            + self.compute_promotion_count(n_candidates, 1) * self.FIDELITY_LEVELS[2].cost_multiplier
        )
        return {
            "full_cost_units": full_cost,
            "multi_fidelity_cost_units": round(mf_cost, 1),
            "savings_pct": round((1 - mf_cost / max(full_cost, 1)) * 100, 1),
        }

    def stats(self) -> Dict[str, Any]:
        return {
            "levels": {
                fl.name: {
                    "samples": len(self._level_stats.get(fl.name, [])),
                    "avg_quality": round(
                        sum(self._level_stats.get(fl.name, [0])) / max(len(self._level_stats.get(fl.name, [1])), 1), 3
                    ),
                }
                for fl in self.FIDELITY_LEVELS
            },
        }


# ============================================================================
# 3. Bayesian Tuner (Gaussian Process Surrogate)
# ============================================================================

class BayesianTuner:
    """
    Lightweight Bayesian optimization for pipeline hyperparameter tuning.

    Uses a simple surrogate model (weighted k-NN in parameter space) instead
    of full Gaussian Processes to avoid scipy dependency.

    Strategy:
      - Maintain history of (params, fitness) observations
      - For new params, predict fitness using distance-weighted k-NN
      - Acquisition function: Expected Improvement (EI)
      - Suggest next params by sampling + picking highest EI

    Tunes: temperature, batch_size, tree_depth, reflection_threshold, etc.
    """

    def __init__(self, param_bounds: Dict[str, Tuple[float, float]] = None):
        self.param_bounds = param_bounds or {
            "temperature": (0.1, 0.95),
            "tree_branches": (1, 5),
            "tree_depth": (1, 3),
            "reflection_threshold": (0.3, 0.8),
            "debate_fraction": (0.1, 0.5),
        }
        self._observations: List[Tuple[Dict[str, float], float]] = []
        self._best_fitness = 0.0
        self._lock = threading.Lock()

    def suggest(self, n_suggestions: int = 1) -> List[Dict[str, float]]:
        """Suggest next hyperparameter configurations to try."""
        if len(self._observations) < 3:
            # Random exploration phase
            return [self._random_params() for _ in range(n_suggestions)]

        # Sample candidates and pick those with highest expected improvement
        candidates = [self._random_params() for _ in range(50)]
        scored = [(self._expected_improvement(c), c) for c in candidates]
        scored.sort(reverse=True)
        return [c for _, c in scored[:n_suggestions]]

    def record(self, params: Dict[str, float], fitness: float) -> None:
        """Record observation."""
        with self._lock:
            self._observations.append((params, fitness))
            self._best_fitness = max(self._best_fitness, fitness)

    def _random_params(self) -> Dict[str, float]:
        """Generate random params within bounds."""
        return {
            k: random.uniform(lo, hi)
            for k, (lo, hi) in self.param_bounds.items()
        }

    def _predict(self, params: Dict[str, float]) -> Tuple[float, float]:
        """Predict (mean, std) for params using distance-weighted k-NN."""
        if not self._observations:
            return 0.5, 0.5

        distances = []
        for obs_params, obs_fitness in self._observations:
            dist = sum(
                ((params.get(k, 0) - obs_params.get(k, 0)) / max(hi - lo, 0.001)) ** 2
                for k, (lo, hi) in self.param_bounds.items()
            )
            distances.append((math.sqrt(dist + 1e-8), obs_fitness))

        # K-NN with distance weighting (k = min(5, len))
        distances.sort()
        k = min(5, len(distances))
        neighbors = distances[:k]

        weights = [1.0 / max(d, 0.01) for d, _ in neighbors]
        total_w = sum(weights)

        # Weighted mean
        mean = sum(w * f for (_, f), w in zip(neighbors, weights)) / max(total_w, 0.001)

        # Weighted variance
        var = sum(w * (f - mean) ** 2 for (_, f), w in zip(neighbors, weights)) / max(total_w, 0.001)
        std = math.sqrt(var + 1e-6)

        return mean, std

    def _expected_improvement(self, params: Dict[str, float]) -> float:
        """Expected Improvement acquisition function."""
        mean, std = self._predict(params)
        if std < 1e-6:
            return 0.0
        z = (mean - self._best_fitness) / std
        # Approximate EI using simplified formula
        ei = std * (z * self._norm_cdf(z) + self._norm_pdf(z))
        return ei

    @staticmethod
    def _norm_pdf(x: float) -> float:
        return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

    @staticmethod
    def _norm_cdf(x: float) -> float:
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def best_observed(self) -> Tuple[Dict[str, float], float]:
        """Return best observed configuration."""
        if not self._observations:
            return self._random_params(), 0.0
        return max(self._observations, key=lambda x: x[1])

    def stats(self) -> Dict[str, Any]:
        best_params, best_fit = self.best_observed()
        return {
            "observations": len(self._observations),
            "best_fitness": round(best_fit, 3),
            "best_params": {k: round(v, 3) for k, v in best_params.items()},
        }


# ============================================================================
# 4. Knapsack Budget Solver
# ============================================================================

class KnapsackBudget:
    """
    Solve budget allocation as a 0-1 knapsack problem.

    Each pipeline stage has:
      - cost: estimated token cost
      - value: expected quality contribution
      - mandatory: whether it can be skipped

    Finds the optimal subset of stages to run within budget.
    Uses dynamic programming for exact solution (polynomial for discrete items).
    """

    @dataclass
    class StageItem:
        name: str
        cost: float  # estimated tokens (thousands)
        value: float  # expected quality contribution
        mandatory: bool = False

    DEFAULT_STAGES = [
        StageItem("ideation", 5.0, 0.9, mandatory=True),
        StageItem("tree_search", 3.0, 0.5),
        StageItem("experiment_design", 2.0, 0.7, mandatory=True),
        StageItem("self_reflection_exp", 1.5, 0.3),
        StageItem("code_generation", 3.0, 0.8, mandatory=True),
        StageItem("self_reflection_code", 1.5, 0.4),
        StageItem("execution", 0.5, 0.6, mandatory=True),
        StageItem("analysis", 2.0, 0.6, mandatory=True),
        StageItem("self_reflection_results", 1.0, 0.25),
        StageItem("paper_writing", 4.0, 0.7, mandatory=True),
        StageItem("self_reflection_paper", 1.5, 0.3),
        StageItem("review", 3.0, 0.5, mandatory=True),
    ]

    def __init__(self, stages: List["KnapsackBudget.StageItem"] = None):
        self.stages = stages or list(self.DEFAULT_STAGES)
        self._history: List[Dict] = []

    def solve(self, budget_tokens_k: float) -> Dict[str, str]:
        """
        Solve knapsack: which stages to run within budget.

        Returns: {stage_name: "run" | "skip"} dict.
        Mandatory stages always run.
        """
        # Separate mandatory and optional
        mandatory = [s for s in self.stages if s.mandatory]
        optional = [s for s in self.stages if not s.mandatory]

        mandatory_cost = sum(s.cost for s in mandatory)
        remaining = budget_tokens_k - mandatory_cost

        if remaining <= 0:
            # Only mandatory stages fit
            result = {s.name: "run" for s in mandatory}
            for s in optional:
                result[s.name] = "skip"
            return result

        # DP knapsack for optional stages
        # Discretize budget to integer units (0.5k granularity)
        scale = 2  # 0.5k resolution
        capacity = int(remaining * scale)
        n = len(optional)

        if n == 0 or capacity <= 0:
            result = {s.name: "run" for s in mandatory}
            return result

        # DP table
        dp = [[0.0] * (capacity + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            item = optional[i - 1]
            item_cost = int(item.cost * scale)
            for w in range(capacity + 1):
                dp[i][w] = dp[i - 1][w]
                if item_cost <= w:
                    dp[i][w] = max(dp[i][w], dp[i - 1][w - item_cost] + item.value)

        # Backtrack to find selected items
        selected = set()
        w = capacity
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i - 1][w]:
                selected.add(optional[i - 1].name)
                w -= int(optional[i - 1].cost * scale)

        result = {}
        for s in self.stages:
            if s.mandatory:
                result[s.name] = "run"
            elif s.name in selected:
                result[s.name] = "run"
            else:
                result[s.name] = "skip"

        self._history.append({"budget": budget_tokens_k, "plan": dict(result)})
        return result

    def update_value(self, stage_name: str, observed_quality: float) -> None:
        """Update a stage's expected value based on observed quality."""
        for s in self.stages:
            if s.name == stage_name:
                # EMA update
                s.value = 0.7 * s.value + 0.3 * observed_quality
                break

    def update_cost(self, stage_name: str, observed_cost: float) -> None:
        """Update a stage's expected cost based on observed tokens."""
        for s in self.stages:
            if s.name == stage_name:
                s.cost = 0.7 * s.cost + 0.3 * observed_cost
                break

    def stats(self) -> Dict[str, Any]:
        return {
            "stages": {s.name: {"cost": round(s.cost, 1), "value": round(s.value, 2), "mandatory": s.mandatory} for s in self.stages},
            "history_count": len(self._history),
        }


# ============================================================================
# 5. Progressive Elaborator
# ============================================================================

class ProgressiveElaborator:
    """
    Start with skeleton outputs, progressively add detail.

    Inspired by progressive growing in GANs and iterative refinement.

    Phases:
      Phase 1: Skeleton (128 tokens) — outline/structure only
      Phase 2: Draft (512 tokens) — fill in main content
      Phase 3: Polish (2048 tokens) — add detail, examples, rigor

    Benefits:
      - Early termination: if skeleton is bad, skip draft/polish
      - Budget control: only spend on promising skeletons
      - Quality: structured generation produces more coherent output
    """

    @dataclass
    class Phase:
        name: str
        max_tokens: int
        prompt_suffix: str
        quality_threshold: float  # min quality to proceed to next phase

    PHASES = [
        Phase("skeleton", 192, "\nProvide ONLY a brief structural outline (3-5 bullet points). No detail yet.", 0.3),
        Phase("draft", 768, "\nExpand the outline into a complete draft. Add key details but don't over-elaborate.", 0.5),
        Phase("polish", 2048, "\nPolish and complete with full detail, examples, and rigor.", 0.0),
    ]

    def __init__(self):
        self._phase_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"attempts": 0, "promotions": 0, "terminations": 0})

    def get_phase(self, phase_idx: int) -> "ProgressiveElaborator.Phase":
        return self.PHASES[min(phase_idx, len(self.PHASES) - 1)]

    def build_progressive_prompt(self, base_system: str, base_user: str, phase_idx: int, previous_output: str = "") -> Tuple[str, str, int]:
        """
        Build a prompt for the given elaboration phase.

        Returns (system, user, max_tokens).
        """
        phase = self.get_phase(phase_idx)

        if phase_idx == 0:
            system = base_system + phase.prompt_suffix
            user = base_user
        else:
            system = base_system + phase.prompt_suffix
            user = (
                f"PREVIOUS OUTPUT:\n{previous_output}\n\n"
                f"ORIGINAL REQUEST:\n{base_user}\n\n"
                f"Now {phase.prompt_suffix.strip()}"
            )

        return system, user, phase.max_tokens

    def should_continue(self, phase_idx: int, quality: float) -> bool:
        """Should we proceed to the next phase based on quality?"""
        phase = self.get_phase(phase_idx)
        self._phase_stats[phase.name]["attempts"] += 1

        if quality >= phase.quality_threshold:
            self._phase_stats[phase.name]["promotions"] += 1
            return True
        else:
            self._phase_stats[phase.name]["terminations"] += 1
            return False

    def stats(self) -> Dict[str, Any]:
        return dict(self._phase_stats)


# ============================================================================
# 6. Contrastive Prompt Pair Generator
# ============================================================================

class ContrastivePromptPair:
    """
    Generate "do this / don't do this" prompt pairs for better output quality.

    Research shows LLMs perform better with both positive and negative examples.
    Adding "DO NOT: [common failure mode]" reduces failure rates by 15-30%.

    Auto-generates negative constraints based on:
      - Common LLM failure modes per task type
      - Historical failure patterns from this pipeline
    """

    # Common failure modes by task type
    NEGATIVE_CONSTRAINTS = {
        "ideation": [
            "DO NOT propose ideas that are merely incremental parameter tuning",
            "DO NOT suggest methods that require unavailable datasets",
            "DO NOT output vague hand-wavy descriptions without concrete technical details",
            "DO NOT duplicate existing well-known methods without meaningful novelty",
        ],
        "code_generation": [
            "DO NOT use placeholder comments like 'TODO' or 'implement here'",
            "DO NOT generate code that imports unavailable proprietary libraries",
            "DO NOT skip error handling for file I/O and network operations",
            "DO NOT hardcode paths — use os.path.join and relative paths",
        ],
        "experiment_design": [
            "DO NOT propose experiments requiring >$100 compute cost",
            "DO NOT skip baseline comparisons",
            "DO NOT use only toy datasets — include at least one standard benchmark",
            "DO NOT ignore statistical significance testing in the metrics",
        ],
        "paper_writing": [
            "DO NOT make unsupported claims without citing evidence",
            "DO NOT skip the limitations section",
            "DO NOT use informal language or first person singular",
            "DO NOT leave sections empty or with placeholder text",
        ],
        "review": [
            "DO NOT give uniformly positive reviews — always find constructive criticism",
            "DO NOT focus only on writing quality — evaluate technical contribution",
            "DO NOT ignore reproducibility concerns",
            "DO NOT be vague — cite specific sections when critiquing",
        ],
    }

    def __init__(self):
        self._failure_patterns: Dict[str, List[str]] = defaultdict(list)
        self._lock = threading.Lock()

    def enhance_prompt(self, system: str, task_type: str) -> str:
        """Add contrastive negative constraints to a system prompt."""
        constraints = self.NEGATIVE_CONSTRAINTS.get(task_type, [])

        # Add learned failure patterns
        with self._lock:
            learned = self._failure_patterns.get(task_type, [])[-3:]

        all_constraints = constraints + learned
        if not all_constraints:
            return system

        negative_block = "\n\nCRITICAL CONSTRAINTS:\n" + "\n".join(f"- {c}" for c in all_constraints[:5])
        return system + negative_block

    def record_failure(self, task_type: str, failure_description: str) -> None:
        """Learn new negative constraint from observed failure."""
        with self._lock:
            constraint = f"DO NOT {failure_description}"
            if constraint not in self._failure_patterns[task_type]:
                self._failure_patterns[task_type].append(constraint)
                # Keep last 10
                if len(self._failure_patterns[task_type]) > 10:
                    self._failure_patterns[task_type] = self._failure_patterns[task_type][-10:]

    def stats(self) -> Dict[str, Any]:
        return {
            "task_types": list(self.NEGATIVE_CONSTRAINTS.keys()),
            "learned_patterns": {k: len(v) for k, v in self._failure_patterns.items()},
        }


# ============================================================================
# 7. Attention Rollback
# ============================================================================

class AttentionRollback:
    """
    When a stage fails, identify which upstream stage to re-run.

    Tracks causal dependencies between stages and quality signals.
    When stage N fails, uses a blame attribution model to identify which
    upstream stage (1..N-1) is most likely responsible and should be re-run.

    Blame attribution:
      - Stage with lowest quality in the dependency chain
      - Stage with highest historical failure-to-downstream-failure correlation
    """

    # Causal dependency graph (stage → stages it depends on)
    DEPENDENCIES = {
        "experiment_design": ["ideation"],
        "tree_search": ["ideation"],
        "code_generation": ["experiment_design", "tree_search"],
        "execution": ["code_generation"],
        "analysis": ["execution"],
        "paper_writing": ["analysis", "experiment_design", "ideation"],
        "review": ["paper_writing"],
    }

    def __init__(self):
        self._stage_qualities: Dict[str, List[float]] = defaultdict(list)
        self._failure_correlations: Dict[Tuple[str, str], int] = defaultdict(int)
        self._lock = threading.Lock()

    def record_quality(self, stage: str, quality: float) -> None:
        """Record quality for a stage execution."""
        with self._lock:
            self._stage_qualities[stage].append(quality)

    def record_failure_pair(self, failed_stage: str, blamed_stage: str) -> None:
        """Record that blamed_stage caused failed_stage to fail."""
        with self._lock:
            self._failure_correlations[(failed_stage, blamed_stage)] += 1

    def identify_rollback_target(self, failed_stage: str) -> Optional[str]:
        """
        Identify which upstream stage to re-run when failed_stage fails.

        Returns the stage name to re-run, or None if no clear target.
        """
        deps = self.DEPENDENCIES.get(failed_stage, [])
        if not deps:
            return None

        with self._lock:
            # Strategy 1: find the dependency with lowest recent quality
            worst_stage = None
            worst_quality = float('inf')
            for dep in deps:
                qualities = self._stage_qualities.get(dep, [])
                if qualities:
                    recent_avg = sum(qualities[-3:]) / len(qualities[-3:])
                    if recent_avg < worst_quality:
                        worst_quality = recent_avg
                        worst_stage = dep

            # Strategy 2: check historical failure correlations
            max_corr = 0
            corr_stage = None
            for dep in deps:
                corr = self._failure_correlations.get((failed_stage, dep), 0)
                if corr > max_corr:
                    max_corr = corr
                    corr_stage = dep

            # Prefer correlation-based if strong signal
            if max_corr >= 3 and corr_stage:
                return corr_stage

            return worst_stage

    def stats(self) -> Dict[str, Any]:
        return {
            "tracked_stages": len(self._stage_qualities),
            "failure_correlations": dict(
                ((f"{k[0]}→{k[1]}", v) for k, v in self._failure_correlations.items())
            ),
        }


# ============================================================================
# 8. Response Distiller
# ============================================================================

class ResponseDistiller:
    """
    Compress verbose LLM outputs into reusable knowledge nuggets.

    Instead of caching full responses, extracts and caches key facts/patterns
    that can be injected into future prompts to avoid redundant generation.

    Types of nuggets:
      - Method signatures: "Use XYZ algorithm with parameters A, B, C"
      - Quality patterns: "This domain benefits from ensemble approaches"
      - Failure patterns: "Avoid batch sizes > 64 on this dataset"
      - Domain facts: "State-of-the-art on CIFAR-10 is 99.2%"
    """

    @dataclass
    class Nugget:
        content: str
        domain: str
        nugget_type: str  # method, quality, failure, fact
        confidence: float
        uses: int = 0
        created_at: float = field(default_factory=time.time)

    def __init__(self, max_nuggets: int = 100):
        self._nuggets: Dict[str, "ResponseDistiller.Nugget"] = {}
        self._max_nuggets = max_nuggets
        self._lock = threading.Lock()

    def _nugget_key(self, content: str) -> str:
        return hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()[:12]

    def add(self, content: str, domain: str, nugget_type: str, confidence: float = 0.5) -> None:
        """Add a knowledge nugget."""
        key = self._nugget_key(content)
        with self._lock:
            if len(self._nuggets) >= self._max_nuggets:
                # Evict least-used nugget
                least_key = min(self._nuggets, key=lambda k: self._nuggets[k].uses)
                del self._nuggets[least_key]
            self._nuggets[key] = self.Nugget(
                content=content, domain=domain,
                nugget_type=nugget_type, confidence=confidence,
            )

    def get_relevant(self, domain: str, nugget_type: str = None, n: int = 5) -> List[str]:
        """Get relevant nuggets for a domain, sorted by confidence × uses."""
        domain_lower = domain.lower()
        with self._lock:
            candidates = []
            for nugget in self._nuggets.values():
                if domain_lower in nugget.domain.lower() or nugget.domain == "*":
                    if nugget_type is None or nugget.nugget_type == nugget_type:
                        score = nugget.confidence * (1 + nugget.uses * 0.1)
                        candidates.append((score, nugget))
                        nugget.uses += 1

        candidates.sort(reverse=True)
        return [n.content for _, n in candidates[:n]]

    def inject_into_prompt(self, prompt: str, domain: str) -> str:
        """Inject relevant nuggets into a prompt as context."""
        nuggets = self.get_relevant(domain, n=3)
        if not nuggets:
            return prompt
        context = "\n\nRELEVANT KNOWLEDGE:\n" + "\n".join(f"- {n}" for n in nuggets)
        return prompt + context

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            types = defaultdict(int)
            for n in self._nuggets.values():
                types[n.nugget_type] += 1
            return {
                "total_nuggets": len(self._nuggets),
                "by_type": dict(types),
                "total_uses": sum(n.uses for n in self._nuggets.values()),
            }


# ============================================================================
# 9. Entropy Regularizer
# ============================================================================

class EntropyRegularizer:
    """
    Maintain diversity pressure across idea populations.

    Prevents the pipeline from converging to a single type of idea.
    Tracks distribution across methodology types and novelty levels,
    penalizes over-represented categories.

    Entropy H = -Σ p(x) log p(x)
    Max entropy = log(N) for N categories

    If entropy drops below threshold, increase exploration temperature
    and target under-represented cells.
    """

    def __init__(self, n_categories: int = 21, min_entropy_ratio: float = 0.6):
        self.n_categories = n_categories  # 7 methods × 3 novelty levels
        self.min_entropy_ratio = min_entropy_ratio
        self._counts: Dict[str, int] = defaultdict(int)
        self._total = 0
        self._lock = threading.Lock()

    def record(self, category: str) -> None:
        """Record an idea in a category (e.g., 'method_3:novelty_1')."""
        with self._lock:
            self._counts[category] += 1
            self._total += 1

    def entropy(self) -> float:
        """Compute Shannon entropy of current distribution."""
        with self._lock:
            if self._total == 0:
                return 0.0
            probs = [c / self._total for c in self._counts.values()]
        h = -sum(p * math.log(p + 1e-10) for p in probs)
        return h

    @property
    def max_entropy(self) -> float:
        return math.log(self.n_categories)

    @property
    def entropy_ratio(self) -> float:
        """Current entropy / max possible entropy. 1.0 = perfectly diverse."""
        return self.entropy() / max(self.max_entropy, 0.001)

    @property
    def needs_diversity_boost(self) -> bool:
        """Is diversity dangerously low?"""
        return self._total >= 5 and self.entropy_ratio < self.min_entropy_ratio

    def get_underrepresented(self, all_categories: List[str], n: int = 3) -> List[str]:
        """Get the most under-represented categories."""
        with self._lock:
            scored = [(self._counts.get(c, 0), c) for c in all_categories]
        scored.sort()  # ascending — least represented first
        return [c for _, c in scored[:n]]

    def diversity_temperature_boost(self) -> float:
        """Extra temperature to add when diversity is low."""
        if not self.needs_diversity_boost:
            return 0.0
        # More boost for lower entropy
        deficit = self.min_entropy_ratio - self.entropy_ratio
        return min(0.2, deficit * 0.5)

    def stats(self) -> Dict[str, Any]:
        return {
            "entropy": round(self.entropy(), 3),
            "max_entropy": round(self.max_entropy, 3),
            "entropy_ratio": round(self.entropy_ratio, 3),
            "needs_boost": self.needs_diversity_boost,
            "total_ideas": self._total,
            "categories_seen": len(self._counts),
            "temp_boost": round(self.diversity_temperature_boost(), 3),
        }


# ============================================================================
# 10. Reward Shaper (Hindsight Experience Replay)
# ============================================================================

class RewardShaper:
    """
    Learn from failed experiments by extracting what went right.

    Inspired by Hindsight Experience Replay (Andrychowicz et al., 2017).

    When an experiment fails:
      1. Extract partial successes (code compiled, some metrics computed, etc.)
      2. Re-label: "instead of failing to achieve X, we succeeded at achieving Y"
      3. Store the positive signal for future use

    This prevents the pipeline from learning "this entire direction is bad"
    when only a specific implementation detail failed.
    """

    @dataclass
    class HindsightRecord:
        original_goal: str
        actual_outcome: str
        partial_successes: List[str]
        lessons_learned: List[str]
        reusable_components: List[str]

    def __init__(self):
        self._records: List["RewardShaper.HindsightRecord"] = []
        self._lesson_index: Dict[str, List[str]] = defaultdict(list)
        self._lock = threading.Lock()

    def record_failure(
        self,
        goal: str,
        outcome: str,
        partial_successes: List[str] = None,
        lessons: List[str] = None,
        reusable: List[str] = None,
    ) -> None:
        """Record a failed experiment with hindsight analysis."""
        record = self.HindsightRecord(
            original_goal=goal,
            actual_outcome=outcome,
            partial_successes=partial_successes or [],
            lessons_learned=lessons or [],
            reusable_components=reusable or [],
        )
        with self._lock:
            self._records.append(record)
            for lesson in record.lessons_learned:
                # Index by keywords
                for word in lesson.lower().split():
                    if len(word) > 4:
                        self._lesson_index[word].append(lesson)

    def get_relevant_lessons(self, context: str, n: int = 3) -> List[str]:
        """Get lessons relevant to a context string."""
        with self._lock:
            scored: Dict[str, int] = defaultdict(int)
            for word in context.lower().split():
                if len(word) > 4 and word in self._lesson_index:
                    for lesson in self._lesson_index[word]:
                        scored[lesson] += 1
            # Sort by relevance score
            sorted_lessons = sorted(scored.items(), key=lambda x: x[1], reverse=True)
            return [l for l, _ in sorted_lessons[:n]]

    def get_reusable_components(self) -> List[str]:
        """Get all reusable components from past failures."""
        with self._lock:
            components = []
            for r in self._records[-10:]:  # last 10 failures
                components.extend(r.reusable_components)
            return list(set(components))

    def shaped_reward(self, base_reward: float, partial_successes: List[str]) -> float:
        """
        Shape the reward to account for partial successes.

        Even a failed experiment with partial successes should get some positive signal:
          shaped = base + 0.1 * n_partial_successes (capped at +0.3)
        """
        partial_bonus = min(0.3, 0.1 * len(partial_successes))
        return min(1.0, base_reward + partial_bonus)

    def stats(self) -> Dict[str, Any]:
        return {
            "total_records": len(self._records),
            "lessons_indexed": sum(len(v) for v in self._lesson_index.values()),
            "reusable_components": len(self.get_reusable_components()),
        }


# ============================================================================
# 11. Token Budget Projector
# ============================================================================

class TokenBudgetProjector:
    """
    Real-time token spend projections with early-stop recommendations.

    Tracks token consumption rate and projects when budget will be exhausted.
    Provides early-stop signals to prevent budget overruns.
    """

    def __init__(self, total_budget_usd: float, cost_per_1k_tokens: float = 0.0005):
        self.total_budget = total_budget_usd
        self.cost_per_1k = cost_per_1k_tokens
        self._checkpoints: List[Tuple[float, float]] = []  # (timestamp, cumulative_spend)
        self._lock = threading.Lock()

    def record_spend(self, amount_usd: float) -> None:
        with self._lock:
            cumulative = (self._checkpoints[-1][1] if self._checkpoints else 0) + amount_usd
            self._checkpoints.append((time.time(), cumulative))

    @property
    def total_spent(self) -> float:
        with self._lock:
            return self._checkpoints[-1][1] if self._checkpoints else 0

    @property
    def remaining(self) -> float:
        return max(0, self.total_budget - self.total_spent)

    def burn_rate(self) -> float:
        """USD per second burn rate (rolling 60s window)."""
        with self._lock:
            if len(self._checkpoints) < 2:
                return 0
            now = time.time()
            recent = [(t, s) for t, s in self._checkpoints if now - t < 60]
            if len(recent) < 2:
                recent = self._checkpoints[-2:]
            dt = recent[-1][0] - recent[0][0]
            ds = recent[-1][1] - recent[0][1]
            return ds / max(dt, 0.1)

    def time_to_exhaustion_s(self) -> float:
        """Estimated seconds until budget is exhausted."""
        rate = self.burn_rate()
        if rate <= 0:
            return float('inf')
        return self.remaining / rate

    def should_early_stop(self, min_remaining_pct: float = 10.0) -> bool:
        """Should the pipeline stop to preserve budget?"""
        remaining_pct = (self.remaining / max(self.total_budget, 0.001)) * 100
        return remaining_pct < min_remaining_pct

    def project_stage_cost(self, stage_name: str, stage_history: List[float]) -> float:
        """Estimate cost for the next execution of a stage."""
        if not stage_history:
            return 0.01  # default 1 cent
        # EMA of recent costs
        ema = stage_history[0]
        for c in stage_history[1:]:
            ema = 0.7 * ema + 0.3 * c
        return ema

    def stats(self) -> Dict[str, Any]:
        return {
            "total_budget": self.total_budget,
            "spent": round(self.total_spent, 4),
            "remaining": round(self.remaining, 4),
            "burn_rate_usd_s": round(self.burn_rate(), 6),
            "time_to_exhaustion_s": round(self.time_to_exhaustion_s(), 0),
            "should_stop": self.should_early_stop(),
        }


# ============================================================================
# 12. Dynamic Batch Sizer
# ============================================================================

class DynamicBatchSizer:
    """
    Auto-tune batch sizes based on latency + remaining budget.

    When budget is plentiful and latency is low → large batches for throughput.
    When budget is tight or latency is high → small batches for efficiency.

    Algorithm:
      batch_size = base × budget_factor × latency_factor
      budget_factor = remaining_budget / total_budget (0→1)
      latency_factor = target_latency / actual_latency (caps at 2x)
    """

    def __init__(
        self,
        base_batch_size: int = 3,
        min_batch: int = 1,
        max_batch: int = 8,
        target_latency_s: float = 10.0,
    ):
        self.base = base_batch_size
        self.min_batch = min_batch
        self.max_batch = max_batch
        self.target_latency = target_latency_s
        self._latencies: deque = deque(maxlen=20)
        self._lock = threading.Lock()

    def record_latency(self, latency_s: float) -> None:
        with self._lock:
            self._latencies.append(latency_s)

    def get_batch_size(self, budget_remaining_pct: float = 100) -> int:
        """Compute optimal batch size."""
        with self._lock:
            avg_latency = sum(self._latencies) / len(self._latencies) if self._latencies else self.target_latency

        budget_factor = max(0.3, min(1.5, budget_remaining_pct / 50))
        latency_factor = max(0.5, min(2.0, self.target_latency / max(avg_latency, 0.1)))

        size = int(self.base * budget_factor * latency_factor)
        return max(self.min_batch, min(self.max_batch, size))

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            lats = list(self._latencies)
        return {
            "current_batch_size": self.get_batch_size(),
            "avg_latency": round(sum(lats) / len(lats), 2) if lats else 0,
            "samples": len(lats),
        }


# ============================================================================
# Master Deep Optimizer
# ============================================================================

class DeepOptimizer:
    """Aggregates all deep optimization techniques."""

    def __init__(self, budget_usd: float = 5.0, enable_all: bool = True):
        self.ensemble = EnsembleDistiller() if enable_all else None
        self.multi_fidelity = MultiFidelityOptimizer() if enable_all else None
        self.bayesian = BayesianTuner() if enable_all else None
        self.knapsack = KnapsackBudget() if enable_all else None
        self.progressive = ProgressiveElaborator() if enable_all else None
        self.contrastive = ContrastivePromptPair() if enable_all else None
        self.rollback = AttentionRollback() if enable_all else None
        self.distiller = ResponseDistiller() if enable_all else None
        self.entropy_reg = EntropyRegularizer() if enable_all else None
        self.reward_shaper = RewardShaper() if enable_all else None
        self.budget_projector = TokenBudgetProjector(budget_usd) if enable_all else None
        self.batch_sizer = DynamicBatchSizer() if enable_all else None

    def summary(self) -> Dict[str, Any]:
        result = {}
        if self.ensemble: result["ensemble"] = self.ensemble.stats()
        if self.multi_fidelity: result["multi_fidelity"] = self.multi_fidelity.stats()
        if self.bayesian: result["bayesian_tuner"] = self.bayesian.stats()
        if self.knapsack: result["knapsack_budget"] = self.knapsack.stats()
        if self.progressive: result["progressive_elaboration"] = self.progressive.stats()
        if self.contrastive: result["contrastive_prompts"] = self.contrastive.stats()
        if self.rollback: result["attention_rollback"] = self.rollback.stats()
        if self.distiller: result["response_distiller"] = self.distiller.stats()
        if self.entropy_reg: result["entropy_regularizer"] = self.entropy_reg.stats()
        if self.reward_shaper: result["reward_shaper"] = self.reward_shaper.stats()
        if self.budget_projector: result["budget_projector"] = self.budget_projector.stats()
        if self.batch_sizer: result["batch_sizer"] = self.batch_sizer.stats()
        return result
