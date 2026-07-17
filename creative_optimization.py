"""
creative_optimization.py - Research-inspired optimization techniques for IdeaGraph.

Novel techniques beyond standard engineering optimization:

  1. PromptEvolver       — Genetic algorithm that evolves prompt templates
  2. ThompsonBandit      — Bayesian multi-armed bandit for strategy selection
  3. AnnealingSchedule   — Simulated annealing temperature across iterations
  4. ParetoFront         — Multi-objective Pareto optimization
  5. CascadeRouter       — Cheap-model-first with smart escalation
  6. CuriosityExplorer   — Information-gain maximization for idea selection
  7. MCTSPipelineRouter  — Monte Carlo tree search for stage skip/run decisions
  8. AdversarialTester   — Red-team stress testing for ideas
  9. EloTournament       — Pairwise ELO ranking for ideas
 10. PopulationTrainer   — Evolutionary hyperparameter optimization across runs
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# ============================================================================
# 1. Prompt Evolution (Genetic Algorithm)
# ============================================================================

@dataclass
class PromptIndividual:
    """A single prompt template in the evolving population."""
    template: str
    fitness: float = 0.0
    generation: int = 0
    parent_ids: List[int] = field(default_factory=list)
    id: int = 0


class PromptEvolver:
    """
    Evolve prompt templates using genetic operators to maximize output quality.

    Inspired by EvoPrompt (Guo et al., 2023) and PromptBreeder (Fernando et al., 2023).

    The population of prompts undergoes:
      - Tournament selection (pick best from random subset)
      - Crossover: combine instruction segments from two parents
      - Mutation: randomly rephrase, add constraints, or swap emphasis
      - Elitism: top-K always survive to next generation

    Fitness is measured by the quality score of outputs produced by each prompt.
    """

    MUTATION_OPERATORS = [
        "Add specificity: include a concrete example or constraint",
        "Add format instruction: specify output structure more precisely",
        "Simplify: remove redundant instructions, make more concise",
        "Reorder: put the most important instruction first",
        "Add persona: strengthen the expert role description",
        "Add negative constraint: specify what NOT to do",
        "Quantify: replace vague terms with specific numbers",
        "Add chain-of-thought: request step-by-step reasoning",
    ]

    def __init__(
        self,
        population_size: int = 8,
        elite_count: int = 2,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.5,
    ):
        self.population_size = population_size
        self.elite_count = elite_count
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self._populations: Dict[str, List[PromptIndividual]] = {}
        self._generation: Dict[str, int] = defaultdict(int)
        self._id_counter = 0
        self._lock = threading.Lock()

    def _next_id(self) -> int:
        self._id_counter += 1
        return self._id_counter

    def initialize(self, task_key: str, base_prompt: str, variants: List[str] = None) -> None:
        """Seed the population for a task with a base prompt + variants."""
        pop = [PromptIndividual(template=base_prompt, id=self._next_id())]
        if variants:
            for v in variants[:self.population_size - 1]:
                pop.append(PromptIndividual(template=v, id=self._next_id()))
        # Pad with mutations of base
        while len(pop) < self.population_size:
            mutated = self._mutate_text(base_prompt)
            pop.append(PromptIndividual(template=mutated, id=self._next_id()))
        self._populations[task_key] = pop

    def get_prompt(self, task_key: str) -> str:
        """Get the best prompt for a task (or a random one if no fitness data)."""
        with self._lock:
            pop = self._populations.get(task_key, [])
        if not pop:
            return ""
        # Return best by fitness, breaking ties randomly
        best = max(pop, key=lambda p: p.fitness + random.random() * 0.001)
        return best.template

    def record_fitness(self, task_key: str, prompt: str, fitness: float) -> None:
        """Record fitness for a prompt that was used."""
        with self._lock:
            pop = self._populations.get(task_key, [])
            for p in pop:
                if p.template == prompt:
                    # Exponential moving average for stability
                    p.fitness = 0.7 * p.fitness + 0.3 * fitness if p.fitness > 0 else fitness
                    break

    def evolve(self, task_key: str) -> None:
        """Run one generation of evolution on the task's population."""
        with self._lock:
            pop = self._populations.get(task_key, [])
            if len(pop) < 3:
                return

            self._generation[task_key] += 1
            gen = self._generation[task_key]

            # Sort by fitness
            pop.sort(key=lambda p: p.fitness, reverse=True)

            # Elitism: keep top-K
            new_pop = [copy.copy(p) for p in pop[:self.elite_count]]
            for p in new_pop:
                p.generation = gen

            # Fill rest with offspring
            while len(new_pop) < self.population_size:
                if random.random() < self.crossover_rate and len(pop) >= 2:
                    # Tournament selection (size 3)
                    p1 = self._tournament(pop, k=3)
                    p2 = self._tournament(pop, k=3)
                    child_text = self._crossover(p1.template, p2.template)
                    child = PromptIndividual(
                        template=child_text, generation=gen,
                        parent_ids=[p1.id, p2.id], id=self._next_id(),
                    )
                else:
                    parent = self._tournament(pop, k=3)
                    child_text = self._mutate_text(parent.template)
                    child = PromptIndividual(
                        template=child_text, generation=gen,
                        parent_ids=[parent.id], id=self._next_id(),
                    )

                # Random mutation chance
                if random.random() < self.mutation_rate:
                    child.template = self._mutate_text(child.template)

                new_pop.append(child)

            self._populations[task_key] = new_pop[:self.population_size]

    def _tournament(self, pop: List[PromptIndividual], k: int = 3) -> PromptIndividual:
        """Tournament selection: pick best from k random individuals."""
        contestants = random.sample(pop, min(k, len(pop)))
        return max(contestants, key=lambda p: p.fitness)

    def _crossover(self, parent1: str, parent2: str) -> str:
        """Single-point crossover on sentence boundaries."""
        s1 = [s.strip() for s in parent1.split(".") if s.strip()]
        s2 = [s.strip() for s in parent2.split(".") if s.strip()]
        if len(s1) < 2 or len(s2) < 2:
            return parent1
        cut1 = random.randint(1, len(s1) - 1)
        cut2 = random.randint(1, len(s2) - 1)
        child_sentences = s1[:cut1] + s2[cut2:]
        return ". ".join(child_sentences) + "."

    def _mutate_text(self, text: str) -> str:
        """Apply a random mutation operator (text-level, no LLM needed)."""
        op = random.choice(self.MUTATION_OPERATORS)
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        if not sentences:
            return text

        mutation_type = random.randint(0, 4)
        if mutation_type == 0 and len(sentences) > 2:
            # Swap two random sentences
            i, j = random.sample(range(len(sentences)), 2)
            sentences[i], sentences[j] = sentences[j], sentences[i]
        elif mutation_type == 1:
            # Insert emphasis marker
            idx = random.randint(0, len(sentences) - 1)
            sentences[idx] = "IMPORTANT: " + sentences[idx]
        elif mutation_type == 2 and len(sentences) > 3:
            # Delete a random non-first sentence
            idx = random.randint(1, len(sentences) - 1)
            sentences.pop(idx)
        elif mutation_type == 3:
            # Duplicate an important sentence at the end
            sentences.append(sentences[0])
        else:
            # Append mutation operator as new instruction
            sentences.append(op)

        return ". ".join(sentences) + "."

    def stats(self) -> Dict[str, Any]:
        """Population stats per task."""
        result = {}
        for key, pop in self._populations.items():
            fitnesses = [p.fitness for p in pop]
            result[key] = {
                "generation": self._generation.get(key, 0),
                "population_size": len(pop),
                "best_fitness": max(fitnesses) if fitnesses else 0,
                "avg_fitness": sum(fitnesses) / len(fitnesses) if fitnesses else 0,
                "diversity": len(set(p.template[:50] for p in pop)),
            }
        return result


# ============================================================================
# 2. Thompson Sampling (Bayesian Multi-Armed Bandit)
# ============================================================================

class ThompsonBandit:
    """
    Bayesian multi-armed bandit using Thompson Sampling with Beta priors.

    Better than epsilon-greedy or UCB1 for strategy selection because it
    naturally balances exploration vs exploitation with uncertainty estimates.

    Each arm (strategy) has a Beta(alpha, beta) posterior.
    On each pull, sample from each arm's posterior and pick the highest.
    This automatically explores uncertain arms and exploits known-good ones.
    """

    def __init__(self):
        # arm_key → (alpha, beta) — Beta distribution parameters
        self._arms: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def add_arm(self, key: str, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> None:
        """Register an arm with a Beta prior."""
        with self._lock:
            if key not in self._arms:
                self._arms[key] = [prior_alpha, prior_beta]

    def select(self, arm_keys: List[str] = None) -> str:
        """
        Thompson Sampling: sample from each arm's Beta posterior,
        return the arm with the highest sample.
        """
        with self._lock:
            keys = arm_keys or list(self._arms.keys())
            if not keys:
                return ""

            best_key = keys[0]
            best_sample = -1.0
            for key in keys:
                alpha, beta = self._arms.get(key, [1.0, 1.0])
                sample = random.betavariate(alpha, beta)
                if sample > best_sample:
                    best_sample = sample
                    best_key = key
            return best_key

    def update(self, key: str, reward: float) -> None:
        """
        Update arm posterior with observed reward.

        reward should be in [0, 1]. We treat it as a Bernoulli-like signal:
        alpha += reward, beta += (1 - reward).
        """
        with self._lock:
            if key not in self._arms:
                self._arms[key] = [1.0, 1.0]
            self._arms[key][0] += max(0, min(1, reward))
            self._arms[key][1] += max(0, min(1, 1 - reward))

    def expected_value(self, key: str) -> float:
        """Mean of Beta distribution = alpha / (alpha + beta)."""
        alpha, beta = self._arms.get(key, [1.0, 1.0])
        return alpha / (alpha + beta)

    def uncertainty(self, key: str) -> float:
        """Variance of Beta = alpha*beta / ((a+b)^2 * (a+b+1))."""
        alpha, beta = self._arms.get(key, [1.0, 1.0])
        total = alpha + beta
        return (alpha * beta) / (total * total * (total + 1))

    def stats(self) -> Dict[str, Dict[str, float]]:
        result = {}
        for key, (alpha, beta) in self._arms.items():
            result[key] = {
                "alpha": round(alpha, 2),
                "beta": round(beta, 2),
                "expected": round(alpha / (alpha + beta), 3),
                "uncertainty": round(self.uncertainty(key), 4),
                "pulls": int(alpha + beta - 2),  # subtract priors
            }
        return result


# ============================================================================
# 3. Simulated Annealing Temperature Schedule
# ============================================================================

class AnnealingSchedule:
    """
    Controls LLM temperature across pipeline iterations using simulated annealing.

    Strategy:
      - Early iterations: HIGH temperature (0.8-0.9) → diverse exploration
      - Middle iterations: MEDIUM temperature (0.5-0.7) → balanced
      - Late iterations: LOW temperature (0.2-0.4) → focused refinement

    Also supports per-stage temperature offsets:
      - Ideation gets +0.1 (always more creative)
      - Code generation gets -0.2 (always more precise)
      - Review gets -0.1 (always more analytical)

    Schedule types:
      - Linear: T(t) = T_max - (T_max - T_min) * t/T_total
      - Exponential: T(t) = T_max * decay^t
      - Cosine: T(t) = T_min + 0.5*(T_max-T_min)*(1 + cos(π*t/T_total))
    """

    STAGE_OFFSETS = {
        "ideation": +0.1,
        "experiment_design": 0.0,
        "tree_search": +0.05,
        "code_generation": -0.2,
        "execution": 0.0,
        "analysis": -0.05,
        "paper_writing": -0.05,
        "review": -0.1,
    }

    def __init__(
        self,
        t_max: float = 0.9,
        t_min: float = 0.2,
        total_iterations: int = 3,
        schedule: str = "cosine",
    ):
        self.t_max = t_max
        self.t_min = t_min
        self.total_iterations = max(total_iterations, 1)
        self.schedule = schedule
        self._current_iteration = 0

    def set_iteration(self, iteration: int) -> None:
        self._current_iteration = iteration

    def get_temperature(self, stage: str = "", iteration: int = None) -> float:
        """Get the temperature for a given stage at the current iteration."""
        t = iteration if iteration is not None else self._current_iteration
        progress = t / max(self.total_iterations - 1, 1)  # 0.0 → 1.0

        if self.schedule == "linear":
            base_temp = self.t_max - (self.t_max - self.t_min) * progress
        elif self.schedule == "exponential":
            decay = (self.t_min / self.t_max) ** (1.0 / max(self.total_iterations - 1, 1))
            base_temp = self.t_max * (decay ** t)
        else:  # cosine (default — smooth, gradual, most stable)
            base_temp = self.t_min + 0.5 * (self.t_max - self.t_min) * (
                1 + math.cos(math.pi * progress)
            )

        # Apply stage offset
        offset = self.STAGE_OFFSETS.get(stage, 0.0)
        final = max(0.05, min(1.0, base_temp + offset))
        return round(final, 3)

    def summary(self) -> Dict[str, Any]:
        """Show the full temperature schedule."""
        schedule = {}
        for i in range(self.total_iterations):
            temps = {}
            for stage in self.STAGE_OFFSETS:
                temps[stage] = self.get_temperature(stage, i)
            schedule[f"iteration_{i}"] = temps
        return {
            "schedule_type": self.schedule,
            "t_max": self.t_max,
            "t_min": self.t_min,
            "total_iterations": self.total_iterations,
            "temperatures": schedule,
        }


# ============================================================================
# 4. Pareto Front Optimizer
# ============================================================================

@dataclass
class ParetoCandidate:
    """A candidate solution in the multi-objective space."""
    id: str
    data: Any
    objectives: Dict[str, float]  # e.g., {"quality": 0.8, "novelty": 0.7, "cost": 0.3}
    rank: int = 0  # Pareto rank (0 = front, 1 = next, ...)
    crowding_distance: float = 0.0


class ParetoFront:
    """
    Multi-objective Pareto optimization using NSGA-II-style ranking.

    Instead of a single quality score, evaluates ideas/experiments across
    multiple competing objectives:
      - Quality (higher is better)
      - Novelty (higher is better)
      - Feasibility (higher is better)
      - Cost efficiency (lower cost → higher is better)
      - Risk (lower risk → higher is better)

    Returns the Pareto front: solutions where no other solution is better
    in ALL objectives simultaneously.
    """

    def __init__(self, objectives: List[str] = None, maximize: List[bool] = None):
        self.objectives = objectives or ["quality", "novelty", "feasibility", "cost_efficiency"]
        self.maximize = maximize or [True, True, True, True]
        self._candidates: List[ParetoCandidate] = []
        self._lock = threading.Lock()

    def add(self, id: str, data: Any, objectives: Dict[str, float]) -> None:
        """Add a candidate with its objective values."""
        with self._lock:
            self._candidates.append(ParetoCandidate(
                id=id, data=data,
                objectives={k: objectives.get(k, 0) for k in self.objectives},
            ))

    def _dominates(self, a: ParetoCandidate, b: ParetoCandidate) -> bool:
        """Check if a dominates b (at least as good in all, strictly better in one)."""
        dominated = False
        for obj, maximize in zip(self.objectives, self.maximize):
            va = a.objectives.get(obj, 0)
            vb = b.objectives.get(obj, 0)
            if maximize:
                if va < vb:
                    return False
                if va > vb:
                    dominated = True
            else:
                if va > vb:
                    return False
                if va < vb:
                    dominated = True
        return dominated

    def compute_ranks(self) -> None:
        """Assign NSGA-II Pareto ranks to all candidates."""
        with self._lock:
            remaining = list(range(len(self._candidates)))
            rank = 0
            while remaining:
                front = []
                for i in remaining:
                    dominated = False
                    for j in remaining:
                        if i != j and self._dominates(self._candidates[j], self._candidates[i]):
                            dominated = True
                            break
                    if not dominated:
                        front.append(i)
                for i in front:
                    self._candidates[i].rank = rank
                    remaining.remove(i)
                self._compute_crowding(front)
                rank += 1

    def _compute_crowding(self, front_indices: List[int]) -> None:
        """Compute crowding distance for candidates in a front."""
        if len(front_indices) <= 2:
            for i in front_indices:
                self._candidates[i].crowding_distance = float('inf')
            return

        for i in front_indices:
            self._candidates[i].crowding_distance = 0.0

        for obj in self.objectives:
            sorted_idx = sorted(front_indices, key=lambda i: self._candidates[i].objectives.get(obj, 0))
            self._candidates[sorted_idx[0]].crowding_distance = float('inf')
            self._candidates[sorted_idx[-1]].crowding_distance = float('inf')

            obj_min = self._candidates[sorted_idx[0]].objectives.get(obj, 0)
            obj_max = self._candidates[sorted_idx[-1]].objectives.get(obj, 0)
            obj_range = obj_max - obj_min if obj_max != obj_min else 1.0

            for k in range(1, len(sorted_idx) - 1):
                prev_val = self._candidates[sorted_idx[k - 1]].objectives.get(obj, 0)
                next_val = self._candidates[sorted_idx[k + 1]].objectives.get(obj, 0)
                self._candidates[sorted_idx[k]].crowding_distance += (next_val - prev_val) / obj_range

    def get_front(self, rank: int = 0) -> List[ParetoCandidate]:
        """Get candidates at a specific Pareto rank (0 = optimal front)."""
        self.compute_ranks()
        return sorted(
            [c for c in self._candidates if c.rank == rank],
            key=lambda c: c.crowding_distance, reverse=True,
        )

    def select_best(self, n: int = 1) -> List[ParetoCandidate]:
        """Select n best candidates using Pareto rank + crowding distance."""
        self.compute_ranks()
        sorted_all = sorted(
            self._candidates,
            key=lambda c: (c.rank, -c.crowding_distance),
        )
        return sorted_all[:n]

    def stats(self) -> Dict[str, Any]:
        self.compute_ranks()
        fronts = defaultdict(int)
        for c in self._candidates:
            fronts[c.rank] += 1
        return {
            "total_candidates": len(self._candidates),
            "fronts": dict(fronts),
            "objectives": self.objectives,
            "front_0_size": fronts.get(0, 0),
        }


# ============================================================================
# 5. Cascade Model Router
# ============================================================================

class CascadeRouter:
    """
    Route LLM calls through a cascade of models: cheap first, expensive only if needed.

    Inspired by FrugalGPT (Chen et al., 2023).

    Strategy:
      - First try the cheap/fast model (e.g., Gemini Flash, Groq)
      - Evaluate response confidence/quality
      - If below threshold, escalate to the expensive model (e.g., GPT-4o)
      - Track cascade savings over time

    Confidence estimation (no extra LLM call):
      - JSON completeness: did the response parse correctly?
      - Length ratio: is the response reasonably long?
      - Keyword presence: does it contain expected domain terms?
    """

    def __init__(
        self,
        cheap_model: str = "gemini-2.0-flash",
        expensive_model: str = "deepseek-chat",
        confidence_threshold: float = 0.6,
        cheap_provider: str = "gemini",
        expensive_provider: str = "deepseek",
    ):
        self.cheap_model = cheap_model
        self.expensive_model = expensive_model
        self.confidence_threshold = confidence_threshold
        self.cheap_provider = cheap_provider
        self.expensive_provider = expensive_provider

        self._cheap_calls = 0
        self._escalations = 0
        self._savings_estimate = 0.0
        self._lock = threading.Lock()

    def should_escalate(self, response: str, expected_json: bool = False) -> bool:
        """
        Estimate confidence in a response without an extra LLM call.
        Returns True if the response should be re-tried with the expensive model.
        """
        confidence = self._estimate_confidence(response, expected_json)
        return confidence < self.confidence_threshold

    def _estimate_confidence(self, response: str, expected_json: bool) -> float:
        """Heuristic confidence estimation."""
        if not response or not response.strip():
            return 0.0

        score = 0.5  # base

        # Length check: very short responses are suspicious
        if len(response) > 100:
            score += 0.1
        if len(response) > 500:
            score += 0.1

        # JSON validity check
        if expected_json:
            try:
                parsed = json.loads(response)
                score += 0.15
                # Check if JSON has meaningful content (not empty)
                if isinstance(parsed, dict) and len(parsed) >= 2:
                    score += 0.1
            except (json.JSONDecodeError, ValueError):
                score -= 0.2

        # Code quality signals
        if "def " in response or "class " in response:
            score += 0.05
        if "import " in response:
            score += 0.05

        # Negative signals
        if "I cannot" in response or "I'm not able" in response:
            score -= 0.3
        if "as an AI" in response.lower():
            score -= 0.2
        if response.count("TODO") > 2:
            score -= 0.15

        return max(0.0, min(1.0, score))

    def record_cheap_call(self, escalated: bool, saved_usd: float = 0) -> None:
        with self._lock:
            self._cheap_calls += 1
            if escalated:
                self._escalations += 1
            else:
                self._savings_estimate += saved_usd

    @property
    def escalation_rate(self) -> float:
        return self._escalations / max(self._cheap_calls, 1)

    def stats(self) -> Dict[str, Any]:
        return {
            "cheap_calls": self._cheap_calls,
            "escalations": self._escalations,
            "escalation_rate": f"{self.escalation_rate:.1%}",
            "savings_estimate_usd": round(self._savings_estimate, 4),
        }


# ============================================================================
# 6. Curiosity-Driven Exploration
# ============================================================================

class CuriosityExplorer:
    """
    Selects ideas/cells that maximize information gain rather than raw quality.

    Inspired by intrinsic motivation in RL (Pathak et al., 2017 — ICM).

    Information gain ≈ how surprising/uncertain an area is.
    A cell/domain with high variance in quality scores has more to learn.
    A cell never explored has maximum curiosity.

    Score = quality_potential + curiosity_bonus
    curiosity_bonus = prediction_error / (visit_count + 1)
    prediction_error = |actual_quality - predicted_quality|
    """

    def __init__(self, curiosity_weight: float = 0.5):
        self.curiosity_weight = curiosity_weight
        self._predictions: Dict[str, float] = {}  # key → predicted quality
        self._actuals: Dict[str, List[float]] = defaultdict(list)
        self._visit_count: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def predict_quality(self, key: str) -> float:
        """Predict quality for a key based on history. Returns 0.5 for unknown."""
        with self._lock:
            actuals = self._actuals.get(key, [])
            if not actuals:
                return 0.5
            # Simple exponential moving average
            ema = actuals[0]
            for a in actuals[1:]:
                ema = 0.7 * ema + 0.3 * a
            self._predictions[key] = ema
            return ema

    def record_outcome(self, key: str, actual_quality: float) -> None:
        """Record the actual quality observed for a key."""
        with self._lock:
            self._actuals[key].append(actual_quality)
            self._visit_count[key] += 1

    def curiosity_score(self, key: str) -> float:
        """
        Compute curiosity bonus for a key.
        Higher = more surprising/uncertain = more worth exploring.
        """
        with self._lock:
            actuals = self._actuals.get(key, [])
            visits = self._visit_count.get(key, 0)

        if visits == 0:
            return 1.0  # maximum curiosity for unexplored

        predicted = self.predict_quality(key)
        if not actuals:
            return 1.0

        # Prediction error = variance-like measure
        last_actual = actuals[-1]
        prediction_error = abs(last_actual - predicted)

        # Variance of observations
        if len(actuals) >= 2:
            mean = sum(actuals) / len(actuals)
            variance = sum((a - mean) ** 2 for a in actuals) / len(actuals)
        else:
            variance = 0.5  # high uncertainty for single observation

        # Combine prediction error + variance, decay by visits
        curiosity = (prediction_error + math.sqrt(variance)) / (visits + 1)
        return min(1.0, curiosity * 2)  # scale to [0, 1]

    def select_best(
        self, candidates: Dict[str, float], n: int = 3,
    ) -> List[str]:
        """
        Select n candidates balancing quality potential + curiosity.

        Args:
            candidates: {key: estimated_quality}
        """
        scored = []
        for key, quality in candidates.items():
            curiosity = self.curiosity_score(key)
            combined = (1 - self.curiosity_weight) * quality + self.curiosity_weight * curiosity
            scored.append((combined, key))
        scored.sort(reverse=True)
        return [key for _, key in scored[:n]]

    def stats(self) -> Dict[str, Any]:
        return {
            "tracked_keys": len(self._actuals),
            "total_visits": sum(self._visit_count.values()),
            "most_curious": sorted(
                [(k, self.curiosity_score(k)) for k in self._actuals],
                key=lambda x: x[1], reverse=True,
            )[:5],
        }


# ============================================================================
# 7. MCTS Pipeline Router
# ============================================================================

@dataclass
class MCTSNode:
    """A node in the pipeline MCTS tree."""
    stage: str
    action: str  # "run", "skip", "run_lite"
    visits: int = 0
    total_reward: float = 0.0
    children: Dict[str, "MCTSNode"] = field(default_factory=dict)
    parent: Optional["MCTSNode"] = None

    @property
    def avg_reward(self) -> float:
        return self.total_reward / max(self.visits, 1)

    def ucb1(self, parent_visits: int, c: float = 1.41) -> float:
        if self.visits == 0:
            return float('inf')
        exploitation = self.avg_reward
        exploration = c * math.sqrt(math.log(parent_visits) / self.visits)
        return exploitation + exploration


class MCTSPipelineRouter:
    """
    Monte Carlo Tree Search to decide which pipeline stages to run, skip, or run
    in lightweight mode.

    At each iteration, the router:
      1. SELECT: traverse tree using UCB1
      2. EXPAND: add new stage-action nodes
      3. SIMULATE: estimate outcome (using stage performance history)
      4. BACKPROPAGATE: update rewards up the tree

    Actions per stage:
      - "run": full stage execution
      - "skip": skip entirely (saves time + budget)
      - "run_lite": reduced version (fewer tokens, simpler prompt)

    Learns over iterations which stage combinations produce best outcomes.
    """

    STAGES = [
        "tree_search", "self_reflection_experiment", "self_reflection_code",
        "self_reflection_results", "self_reflection_paper",
    ]
    ACTIONS = ["run", "skip", "run_lite"]

    def __init__(self, exploration_c: float = 1.41, n_simulations: int = 10):
        self.exploration_c = exploration_c
        self.n_simulations = n_simulations
        self.root = MCTSNode(stage="root", action="start")
        self._stage_history: Dict[str, List[float]] = defaultdict(list)

    def get_plan(self) -> Dict[str, str]:
        """Run MCTS and return recommended action per stage."""
        plan = {}
        current = self.root

        for stage in self.STAGES:
            # Ensure children exist
            for action in self.ACTIONS:
                key = f"{stage}:{action}"
                if key not in current.children:
                    current.children[key] = MCTSNode(
                        stage=stage, action=action, parent=current,
                    )

            # Run simulations
            for _ in range(self.n_simulations):
                self._simulate_from(current, 0)

            # Select best child by avg reward
            best_key = max(
                current.children,
                key=lambda k: current.children[k].avg_reward
                if current.children[k].visits > 0 else 0,
            )
            best_child = current.children[best_key]
            plan[stage] = best_child.action
            current = best_child

        return plan

    def _simulate_from(self, node: MCTSNode, depth: int) -> float:
        """One MCTS simulation: select → expand → simulate → backprop."""
        if depth >= len(self.STAGES):
            # Terminal: estimate reward
            return self._estimate_reward(node)

        stage = self.STAGES[depth]
        # UCB1 selection among children
        for action in self.ACTIONS:
            key = f"{stage}:{action}"
            if key not in node.children:
                node.children[key] = MCTSNode(
                    stage=stage, action=action, parent=node,
                )

        # Select child with highest UCB1
        parent_visits = max(node.visits, 1)
        best_key = max(
            node.children,
            key=lambda k: node.children[k].ucb1(parent_visits, self.exploration_c),
        )
        child = node.children[best_key]

        # Recurse
        reward = self._simulate_from(child, depth + 1)

        # Backpropagate
        child.visits += 1
        child.total_reward += reward
        return reward

    def _estimate_reward(self, node: MCTSNode) -> float:
        """Estimate reward for a terminal state using historical data + noise."""
        # Collect actions taken in this path
        actions = []
        current = node
        while current and current.parent:
            actions.append((current.stage, current.action))
            current = current.parent

        reward = 0.5  # base
        for stage, action in actions:
            history = self._stage_history.get(stage, [])
            avg_hist = sum(history) / len(history) if history else 0.5

            if action == "run":
                reward += avg_hist * 0.15
            elif action == "run_lite":
                reward += avg_hist * 0.08  # less benefit
            # "skip" contributes time savings as reward
            elif action == "skip":
                reward += 0.05  # small fixed bonus for time saved

        # Add noise for exploration
        reward += random.gauss(0, 0.05)
        return max(0, min(1, reward))

    def record_outcome(self, stage: str, quality: float) -> None:
        """Record actual stage outcome to improve future simulations."""
        self._stage_history[stage].append(quality)

    def stats(self) -> Dict[str, Any]:
        plan = {}
        current = self.root
        for stage in self.STAGES:
            children = {
                k: {"visits": v.visits, "avg_reward": round(v.avg_reward, 3)}
                for k, v in current.children.items()
            }
            plan[stage] = children
            # Follow best path
            if current.children:
                best_key = max(current.children, key=lambda k: current.children[k].visits)
                current = current.children[best_key]
        return plan


# ============================================================================
# 8. Adversarial Stress Tester
# ============================================================================

class AdversarialTester:
    """
    Red-team ideas before investing in expensive experiments.

    Generates adversarial critiques along multiple attack vectors:
      - Novelty attack: "This is just [existing method] with minor changes"
      - Feasibility attack: "This requires [impossible resource/data]"
      - Soundness attack: "The hypothesis has this logical flaw: [flaw]"
      - Scalability attack: "This won't work beyond toy scale because [reason]"
      - Ethics attack: "This raises concerns about [bias/harm]"

    Ideas that survive adversarial testing are more likely to succeed.
    Returns a resilience score (0-1) based on how many attacks the idea survives.
    """

    ATTACK_VECTORS = [
        {
            "name": "novelty",
            "prompt": (
                "Argue that this idea is NOT novel. Identify the most similar existing "
                "work and explain why this is essentially the same approach with minor changes. "
                "Be specific about which papers/methods it duplicates."
            ),
        },
        {
            "name": "feasibility",
            "prompt": (
                "Argue that this idea is NOT feasible to implement. Identify specific "
                "technical barriers, missing data, compute requirements, or engineering "
                "challenges that make it impractical."
            ),
        },
        {
            "name": "soundness",
            "prompt": (
                "Find logical flaws in the hypothesis or method. Identify confounding "
                "variables, unjustified assumptions, or reasoning errors that would "
                "invalidate the results."
            ),
        },
        {
            "name": "scalability",
            "prompt": (
                "Argue that this won't work beyond toy/demo scale. Identify bottlenecks "
                "that prevent real-world deployment: data volume, latency, cost, or "
                "generalization failures."
            ),
        },
    ]

    def build_attack_prompts(self, idea: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate adversarial attack prompts for an idea."""
        attacks = []
        for vector in self.ATTACK_VECTORS:
            attacks.append({
                "vector": vector["name"],
                "system": (
                    "You are a ruthless but fair scientific critic. Your job is to find "
                    "the strongest possible counter-argument against this research idea. "
                    "Be specific, cite known methods where possible, and don't hold back.\n\n"
                    "Return JSON: {\"attack\": \"your counter-argument\", "
                    "\"severity\": \"low|medium|high|fatal\", "
                    "\"confidence\": 0.0-1.0, "
                    "\"known_similar_work\": [\"...\"]}"
                ),
                "user": (
                    f"{vector['prompt']}\n\n"
                    f"IDEA: {idea.get('title', '')}\n"
                    f"Method: {idea.get('method', '')}\n"
                    f"Hypothesis: {idea.get('hypothesis', '')}\n"
                    f"Domain: {idea.get('domain', '')}"
                ),
            })
        return attacks

    @staticmethod
    def compute_resilience(attack_results: List[Dict]) -> Dict[str, Any]:
        """
        Compute resilience score from attack results.

        Score = 1.0 - weighted_severity
        where fatal=1.0, high=0.7, medium=0.4, low=0.1
        """
        severity_weights = {"fatal": 1.0, "high": 0.7, "medium": 0.4, "low": 0.1}
        total_weight = 0.0
        total_confidence = 0.0

        survived = []
        failed = []

        for result in attack_results:
            severity = result.get("severity", "medium")
            confidence = result.get("confidence", 0.5)
            w = severity_weights.get(severity, 0.4) * confidence
            total_weight += w
            total_confidence += confidence

            if severity in ("fatal", "high") and confidence > 0.7:
                failed.append(result.get("vector", "unknown"))
            else:
                survived.append(result.get("vector", "unknown"))

        n_attacks = len(attack_results) or 1
        resilience = max(0, 1.0 - total_weight / n_attacks)

        return {
            "resilience_score": round(resilience, 3),
            "survived": survived,
            "failed": failed,
            "recommendation": (
                "proceed" if resilience > 0.6
                else "revise" if resilience > 0.3
                else "abandon"
            ),
        }


# ============================================================================
# 9. ELO Tournament Ranking
# ============================================================================

class EloTournament:
    """
    Rank ideas using ELO ratings from pairwise comparisons.

    More robust than absolute scoring because:
      - LLMs are better at relative judgments than absolute scores
      - Reduces positional bias (first idea always scored higher)
      - Produces a stable global ranking from O(n log n) comparisons

    Uses standard ELO: E(A) = 1/(1 + 10^((Rb-Ra)/400))
    K-factor = 32 (standard chess rapid)
    """

    def __init__(self, k_factor: float = 32.0, initial_rating: float = 1500.0):
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self._ratings: Dict[str, float] = {}
        self._match_history: List[Dict] = []
        self._lock = threading.Lock()

    def add_player(self, id: str) -> None:
        if id not in self._ratings:
            self._ratings[id] = self.initial_rating

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Expected score of A vs B."""
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def record_match(self, winner_id: str, loser_id: str, draw: bool = False) -> None:
        """Record a pairwise comparison result."""
        with self._lock:
            self.add_player(winner_id)
            self.add_player(loser_id)

            ra = self._ratings[winner_id]
            rb = self._ratings[loser_id]

            ea = self.expected_score(ra, rb)
            eb = self.expected_score(rb, ra)

            if draw:
                sa, sb = 0.5, 0.5
            else:
                sa, sb = 1.0, 0.0

            self._ratings[winner_id] = ra + self.k_factor * (sa - ea)
            self._ratings[loser_id] = rb + self.k_factor * (sb - eb)

            self._match_history.append({
                "winner": winner_id, "loser": loser_id, "draw": draw,
                "winner_rating": round(self._ratings[winner_id], 1),
                "loser_rating": round(self._ratings[loser_id], 1),
            })

    def get_rankings(self) -> List[Tuple[str, float]]:
        """Get all players sorted by rating (highest first)."""
        with self._lock:
            return sorted(self._ratings.items(), key=lambda x: x[1], reverse=True)

    def get_rating(self, id: str) -> float:
        return self._ratings.get(id, self.initial_rating)

    def generate_matchups(self, ids: List[str], n_matches: int = None) -> List[Tuple[str, str]]:
        """
        Generate efficient matchups using Swiss-system tournament logic.
        Pairs players with similar ratings for maximally informative matches.
        """
        for id in ids:
            self.add_player(id)

        if n_matches is None:
            n_matches = max(len(ids), int(len(ids) * math.log2(max(len(ids), 2))))

        # Sort by rating, pair adjacent
        sorted_ids = sorted(ids, key=lambda x: self._ratings.get(x, self.initial_rating), reverse=True)
        matchups = []
        for i in range(0, len(sorted_ids) - 1, 2):
            matchups.append((sorted_ids[i], sorted_ids[i + 1]))
        # Add some random matchups for diversity
        while len(matchups) < n_matches and len(ids) >= 2:
            a, b = random.sample(ids, 2)
            if (a, b) not in matchups and (b, a) not in matchups:
                matchups.append((a, b))

        return matchups[:n_matches]

    def build_comparison_prompt(
        self, idea_a: Dict, idea_b: Dict,
    ) -> Dict[str, str]:
        """Build a pairwise comparison prompt for two ideas."""
        return {
            "system": (
                "You are a research evaluation expert. Compare two research ideas "
                "and determine which is BETTER overall considering novelty, feasibility, "
                "potential impact, and scientific rigor.\n\n"
                "Return JSON: {\"winner\": \"A\" or \"B\" or \"draw\", "
                "\"reasoning\": \"brief explanation\", \"confidence\": 0.0-1.0}"
            ),
            "user": (
                f"IDEA A: {idea_a.get('title', '')}\n"
                f"  Method: {idea_a.get('method', '')[:200]}\n"
                f"  Hypothesis: {idea_a.get('hypothesis', '')[:200]}\n\n"
                f"IDEA B: {idea_b.get('title', '')}\n"
                f"  Method: {idea_b.get('method', '')[:200]}\n"
                f"  Hypothesis: {idea_b.get('hypothesis', '')[:200]}\n\n"
                f"Which idea is better? Return JSON with winner (A or B or draw)."
            ),
        }

    def stats(self) -> Dict[str, Any]:
        rankings = self.get_rankings()
        return {
            "players": len(rankings),
            "matches_played": len(self._match_history),
            "top_5": [(id, round(r, 1)) for id, r in rankings[:5]],
            "rating_spread": round(rankings[0][1] - rankings[-1][1], 1) if len(rankings) >= 2 else 0,
        }


# ============================================================================
# 10. Population-Based Training (Hyperparameter Evolution)
# ============================================================================

@dataclass
class HyperparamConfig:
    """A set of pipeline hyperparameters."""
    id: int = 0
    generation: int = 0
    params: Dict[str, float] = field(default_factory=dict)
    fitness: float = 0.0
    runs: int = 0

    def copy_mutate(self, mutation_scale: float = 0.2) -> "HyperparamConfig":
        """Create a mutated copy."""
        new_params = {}
        for k, v in self.params.items():
            # Gaussian mutation with bounds
            mutated = v + random.gauss(0, v * mutation_scale)
            # Clamp to reasonable bounds
            if "rate" in k or "threshold" in k or "weight" in k:
                mutated = max(0.01, min(1.0, mutated))
            elif "count" in k or "size" in k:
                mutated = max(1, int(round(mutated)))
            elif "temperature" in k:
                mutated = max(0.05, min(1.0, mutated))
            new_params[k] = mutated
        return HyperparamConfig(params=new_params)


class PopulationTrainer:
    """
    Evolutionary hyperparameter optimization across pipeline runs.

    Inspired by PBT (Jaderberg et al., 2017).

    Maintains a population of hyperparameter configurations. After each
    pipeline run, low-performing configs are replaced by mutations of
    high-performing ones. Over time, this converges to optimal settings.

    Hyperparameters being evolved:
      - ideation_temperature
      - tree_search_branches
      - tree_search_depth
      - code_generation_temperature
      - self_reflection_threshold
      - debate_budget_fraction
      - ucb1_exploration_constant
      - curiosity_weight
    """

    DEFAULT_PARAMS = {
        "ideation_temperature": 0.7,
        "tree_search_branches": 3,
        "tree_search_depth": 2,
        "code_generation_temperature": 0.3,
        "self_reflection_threshold": 0.5,
        "debate_budget_fraction": 0.3,
        "ucb1_exploration_constant": 1.41,
        "curiosity_weight": 0.5,
    }

    def __init__(self, population_size: int = 6, persist_path: str = None):
        self.population_size = population_size
        self.persist_path = persist_path or str(
            Path(__file__).parent / "output" / "pbt_population.json"
        )
        self._population: List[HyperparamConfig] = []
        self._generation = 0
        self._id_counter = 0

        # Try to load persisted population
        self._load()
        if not self._population:
            self._initialize()

    def _initialize(self) -> None:
        """Create initial population with random variations."""
        for _ in range(self.population_size):
            self._id_counter += 1
            config = HyperparamConfig(
                id=self._id_counter, params=dict(self.DEFAULT_PARAMS),
            )
            # Add random variation
            for k in config.params:
                v = config.params[k]
                config.params[k] = v * (0.7 + random.random() * 0.6)  # ±30%
            self._population.append(config)

    def get_config(self) -> HyperparamConfig:
        """Get the best config (or a random one if no fitness data)."""
        evaluated = [c for c in self._population if c.runs > 0]
        if not evaluated:
            return random.choice(self._population)
        return max(evaluated, key=lambda c: c.fitness)

    def get_exploratory_config(self) -> HyperparamConfig:
        """Get a config biased toward exploration (least-tested)."""
        return min(self._population, key=lambda c: c.runs)

    def record_result(self, config_id: int, fitness: float) -> None:
        """Record the fitness of a pipeline run with this config."""
        for c in self._population:
            if c.id == config_id:
                c.fitness = 0.7 * c.fitness + 0.3 * fitness if c.runs > 0 else fitness
                c.runs += 1
                break
        # Evolve after enough data
        evaluated = [c for c in self._population if c.runs > 0]
        if len(evaluated) >= self.population_size // 2:
            self._evolve()

    def _evolve(self) -> None:
        """One generation of PBT: replace bottom 25% with mutations of top 25%."""
        self._generation += 1

        # Sort by fitness
        evaluated = sorted(
            [c for c in self._population if c.runs > 0],
            key=lambda c: c.fitness, reverse=True,
        )
        if len(evaluated) < 4:
            return

        # Bottom 25%
        n_replace = max(1, len(evaluated) // 4)
        bottom = evaluated[-n_replace:]
        top = evaluated[:n_replace]

        for bad in bottom:
            # Replace with mutation of a random top performer
            parent = random.choice(top)
            self._id_counter += 1
            new_config = parent.copy_mutate(mutation_scale=0.2)
            new_config.id = self._id_counter
            new_config.generation = self._generation
            new_config.fitness = 0.0
            new_config.runs = 0

            # Replace in population
            idx = self._population.index(bad)
            self._population[idx] = new_config

        self._save()

    def _save(self) -> None:
        """Persist population to disk."""
        try:
            os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
            data = {
                "generation": self._generation,
                "population": [
                    {
                        "id": c.id, "generation": c.generation,
                        "params": c.params, "fitness": c.fitness, "runs": c.runs,
                    }
                    for c in self._population
                ],
            }
            with open(self.persist_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _load(self) -> None:
        """Load persisted population."""
        try:
            if os.path.exists(self.persist_path):
                with open(self.persist_path) as f:
                    data = json.load(f)
                self._generation = data.get("generation", 0)
                self._population = []
                for item in data.get("population", []):
                    config = HyperparamConfig(
                        id=item["id"], generation=item.get("generation", 0),
                        params=item["params"], fitness=item.get("fitness", 0),
                        runs=item.get("runs", 0),
                    )
                    self._population.append(config)
                    self._id_counter = max(self._id_counter, config.id)
        except Exception:
            pass

    def stats(self) -> Dict[str, Any]:
        evaluated = [c for c in self._population if c.runs > 0]
        best = max(evaluated, key=lambda c: c.fitness) if evaluated else None
        return {
            "generation": self._generation,
            "population_size": len(self._population),
            "evaluated": len(evaluated),
            "best_fitness": round(best.fitness, 3) if best else 0,
            "best_params": {k: round(v, 3) for k, v in best.params.items()} if best else {},
            "total_runs": sum(c.runs for c in self._population),
        }


# ============================================================================
# Master Creative Optimizer (aggregates all 10 techniques)
# ============================================================================

class CreativeOptimizer:
    """
    Master optimizer aggregating all creative optimization techniques.

    Provides a unified interface for the pipeline to use any combination of:
      - Prompt evolution
      - Thompson sampling for strategy selection
      - Simulated annealing temperature schedule
      - Pareto front multi-objective optimization
      - Cascade model routing
      - Curiosity-driven exploration
      - MCTS pipeline routing
      - Adversarial stress testing
      - ELO tournament ranking
      - Population-based training
    """

    def __init__(
        self,
        max_iterations: int = 3,
        enable_all: bool = True,
    ):
        self.prompt_evolver = PromptEvolver() if enable_all else None
        self.thompson = ThompsonBandit() if enable_all else None
        self.annealing = AnnealingSchedule(total_iterations=max_iterations) if enable_all else None
        self.pareto = ParetoFront() if enable_all else None
        self.cascade = CascadeRouter() if enable_all else None
        self.curiosity = CuriosityExplorer() if enable_all else None
        self.mcts_router = MCTSPipelineRouter() if enable_all else None
        self.adversarial = AdversarialTester() if enable_all else None
        self.elo = EloTournament() if enable_all else None
        self.pbt = PopulationTrainer() if enable_all else None

    def summary(self) -> Dict[str, Any]:
        """Aggregate stats from all sub-optimizers."""
        result = {}
        if self.prompt_evolver:
            result["prompt_evolution"] = self.prompt_evolver.stats()
        if self.thompson:
            result["thompson_sampling"] = self.thompson.stats()
        if self.annealing:
            result["annealing"] = self.annealing.summary()
        if self.pareto:
            result["pareto_front"] = self.pareto.stats()
        if self.cascade:
            result["cascade_router"] = self.cascade.stats()
        if self.curiosity:
            result["curiosity_explorer"] = self.curiosity.stats()
        if self.mcts_router:
            result["mcts_router"] = self.mcts_router.stats()
        if self.elo:
            result["elo_tournament"] = self.elo.stats()
        if self.pbt:
            result["population_training"] = self.pbt.stats()
        return result
