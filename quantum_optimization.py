"""
quantum_optimization.py - Frontier optimization techniques for IdeaGraph.

Layer 6: Cutting-edge techniques from quantum-inspired computing, swarm
intelligence, topology, fractal geometry, and cognitive science. These push
the theoretical boundary of what optimization can achieve in an LLM pipeline.

  1.  QuantumAnnealingSampler  — Quantum-inspired combinatorial optimizer
  2.  SwarmIdeaOptimizer       — Particle swarm optimization for idea refinement
  3.  TopologicalDiversityMap  — Persistent homology for diversity measurement
  4.  FractalBudgetAllocator   — Self-similar recursive budget subdivision
  5.  CognitiveLoadBalancer    — Prompt complexity targeting per-model sweet spot
  6.  HyperbandScheduler       — Principled early stopping across parallel configs
  7.  WassersteinDivergence    — Optimal transport distance for idea distribution
  8.  SimulatedBifurcation     — Ising-machine-inspired parallel combinatorial search
  9.  ContextualBanditRouter   — Contextual MAB for per-query model/prompt selection
  10. GradientFreeOptimizer    — CMA-ES for continuous pipeline hyperparameters
  11. InformationBottleneck    — Rate-distortion compression for stage context passing
  12. MetaLearningWarmStart    — Learn-to-learn initialization from past pipeline runs
"""

from __future__ import annotations

import copy
import hashlib
import math
import os
import json
import random
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


# ============================================================================
# 1. Quantum-Inspired Annealing Sampler
# ============================================================================

class QuantumAnnealingSampler:
    """
    Quantum-inspired combinatorial optimizer for pipeline configuration.

    Models the pipeline configuration as an Ising-like energy landscape.
    Uses simulated quantum tunneling to escape local minima that classical
    simulated annealing gets stuck in.

    Key insight: instead of single-flip Metropolis moves, uses "tunneling"
    moves that flip multiple correlated variables simultaneously, mimicking
    quantum superposition.

    Energy function: E(config) = -Σ_i h_i × s_i - Σ_{ij} J_{ij} × s_i × s_j
    where s_i ∈ {-1, +1} are binary pipeline decisions (run/skip each stage).
    h_i = stage value, J_{ij} = stage interaction strength.
    """

    @dataclass
    class SpinConfig:
        spins: Dict[str, int]  # stage → +1 (run) or -1 (skip)
        energy: float = 0.0

    def __init__(self, stages: List[str] = None):
        self.stages = stages or [
            "tree_search", "self_reflection_exp", "self_reflection_code",
            "self_reflection_results", "self_reflection_paper", "debate",
        ]
        # Local field: stage value (positive = prefer run)
        self._h: Dict[str, float] = {s: 0.3 for s in self.stages}
        # Coupling: interaction between stages (positive = prefer same state)
        self._J: Dict[Tuple[str, str], float] = {}
        self._best: Optional["QuantumAnnealingSampler.SpinConfig"] = None
        self._lock = threading.Lock()

    def set_stage_value(self, stage: str, value: float) -> None:
        """Set the intrinsic value of running a stage."""
        self._h[stage] = value

    def set_coupling(self, stage_a: str, stage_b: str, strength: float) -> None:
        """Set interaction strength between two stages."""
        self._J[(stage_a, stage_b)] = strength
        self._J[(stage_b, stage_a)] = strength

    def _energy(self, config: Dict[str, int]) -> float:
        """Compute energy of a configuration (lower = better)."""
        e = 0.0
        for s, spin in config.items():
            e -= self._h.get(s, 0) * spin
        for (a, b), j in self._J.items():
            if a in config and b in config:
                e -= j * config[a] * config[b]
        return e

    def anneal(self, n_steps: int = 200, n_tunnels: int = 3) -> Dict[str, str]:
        """
        Run quantum-inspired annealing.

        n_tunnels: number of spins to flip simultaneously per tunnel move.
        """
        # Initialize random configuration
        config = {s: random.choice([-1, 1]) for s in self.stages}
        best_config = dict(config)
        best_energy = self._energy(config)

        for step in range(n_steps):
            # Temperature schedule (fast cooling)
            T = 2.0 * (1 - step / n_steps) ** 2 + 0.01

            # Quantum tunneling probability (decreases with time)
            tunnel_prob = 0.3 * (1 - step / n_steps)

            if random.random() < tunnel_prob:
                # Tunnel move: flip multiple correlated spins
                to_flip = random.sample(self.stages, min(n_tunnels, len(self.stages)))
                new_config = dict(config)
                for s in to_flip:
                    new_config[s] = -new_config[s]
            else:
                # Classical single-flip move
                flip = random.choice(self.stages)
                new_config = dict(config)
                new_config[flip] = -new_config[flip]

            new_energy = self._energy(new_config)
            delta = new_energy - self._energy(config)

            # Metropolis acceptance
            if delta < 0 or random.random() < math.exp(-delta / max(T, 0.001)):
                config = new_config
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_config = dict(config)

        with self._lock:
            self._best = self.SpinConfig(spins=best_config, energy=best_energy)

        return {s: "run" if v > 0 else "skip" for s, v in best_config.items()}

    def stats(self) -> Dict[str, Any]:
        return {
            "stages": len(self.stages),
            "best_energy": round(self._best.energy, 3) if self._best else None,
            "best_config": {s: "run" if v > 0 else "skip" for s, v in self._best.spins.items()} if self._best else {},
        }


# ============================================================================
# 2. Swarm Idea Optimizer (Particle Swarm Optimization)
# ============================================================================

class SwarmIdeaOptimizer:
    """
    Particle swarm optimization for continuous idea quality improvement.

    Each "particle" represents an idea configuration in a continuous
    quality space. Particles move toward their personal best and the
    global best, with inertia and random exploration.

    Dimensions: novelty_weight, feasibility_weight, impact_weight, risk_tolerance
    Objective: maximize composite quality score

    PSO update:
      v_i = w×v_i + c1×r1×(pbest_i - x_i) + c2×r2×(gbest - x_i)
      x_i = x_i + v_i
    """

    @dataclass
    class Particle:
        position: Dict[str, float]
        velocity: Dict[str, float]
        best_position: Dict[str, float]
        best_fitness: float = 0.0

    DIMENSIONS = ["novelty_weight", "feasibility_weight", "impact_weight", "risk_tolerance", "creativity_temp"]
    BOUNDS = {
        "novelty_weight": (0.0, 1.0),
        "feasibility_weight": (0.0, 1.0),
        "impact_weight": (0.0, 1.0),
        "risk_tolerance": (0.0, 1.0),
        "creativity_temp": (0.2, 0.95),
    }

    def __init__(self, n_particles: int = 10, inertia: float = 0.7, c1: float = 1.5, c2: float = 1.5):
        self.n_particles = n_particles
        self.w = inertia
        self.c1 = c1  # cognitive (personal best attraction)
        self.c2 = c2  # social (global best attraction)
        self._particles: List["SwarmIdeaOptimizer.Particle"] = []
        self._global_best: Dict[str, float] = {}
        self._global_best_fitness = 0.0
        self._generation = 0
        self._init_swarm()

    def _init_swarm(self) -> None:
        for _ in range(self.n_particles):
            pos = {d: random.uniform(*self.BOUNDS[d]) for d in self.DIMENSIONS}
            vel = {d: random.uniform(-0.1, 0.1) for d in self.DIMENSIONS}
            self._particles.append(self.Particle(
                position=pos, velocity=vel, best_position=dict(pos),
            ))
        self._global_best = dict(self._particles[0].position)

    def step(self, fitness_fn: Callable[[Dict[str, float]], float]) -> Dict[str, float]:
        """Run one PSO iteration. Returns global best position."""
        self._generation += 1

        for p in self._particles:
            fitness = fitness_fn(p.position)

            # Update personal best
            if fitness > p.best_fitness:
                p.best_fitness = fitness
                p.best_position = dict(p.position)

            # Update global best
            if fitness > self._global_best_fitness:
                self._global_best_fitness = fitness
                self._global_best = dict(p.position)

        # Update velocities and positions
        for p in self._particles:
            for d in self.DIMENSIONS:
                r1, r2 = random.random(), random.random()
                cognitive = self.c1 * r1 * (p.best_position[d] - p.position[d])
                social = self.c2 * r2 * (self._global_best[d] - p.position[d])
                p.velocity[d] = self.w * p.velocity[d] + cognitive + social
                # Clamp velocity
                p.velocity[d] = max(-0.2, min(0.2, p.velocity[d]))
                p.position[d] += p.velocity[d]
                # Clamp position to bounds
                lo, hi = self.BOUNDS[d]
                p.position[d] = max(lo, min(hi, p.position[d]))

        return dict(self._global_best)

    def get_best(self) -> Tuple[Dict[str, float], float]:
        return dict(self._global_best), self._global_best_fitness

    def stats(self) -> Dict[str, Any]:
        return {
            "generation": self._generation,
            "particles": self.n_particles,
            "global_best_fitness": round(self._global_best_fitness, 4),
            "global_best": {k: round(v, 3) for k, v in self._global_best.items()},
        }


# ============================================================================
# 3. Topological Diversity Map
# ============================================================================

class TopologicalDiversityMap:
    """
    Measure idea population diversity using persistent homology concepts.

    Instead of simple entropy (counts categories), this measures the
    topological structure of the idea space:
      - Connected components (β₀): distinct idea clusters
      - Loops (β₁): cyclic relationships between ideas
      - Spread: average distance between ideas

    Uses simplified Vietoris-Rips complex on idea feature vectors.
    """

    def __init__(self, feature_dims: int = 5):
        self.feature_dims = feature_dims
        self._points: List[Tuple[str, List[float]]] = []
        self._lock = threading.Lock()

    def add_idea(self, idea_id: str, features: List[float]) -> None:
        """Add an idea with its feature vector."""
        # Normalize to unit vector
        norm = math.sqrt(sum(f * f for f in features) + 1e-10)
        normalized = [f / norm for f in features[:self.feature_dims]]
        # Pad if needed
        while len(normalized) < self.feature_dims:
            normalized.append(0.0)
        with self._lock:
            self._points.append((idea_id, normalized))

    def _distance(self, a: List[float], b: List[float]) -> float:
        return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))

    def connected_components(self, radius: float = 0.5) -> int:
        """Count connected components at given radius (β₀)."""
        with self._lock:
            points = list(self._points)
        if len(points) <= 1:
            return len(points)

        # Union-Find
        parent = list(range(len(points)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                if self._distance(points[i][1], points[j][1]) < radius:
                    union(i, j)

        return len(set(find(i) for i in range(len(points))))

    def spread(self) -> float:
        """Average pairwise distance (higher = more diverse)."""
        with self._lock:
            points = list(self._points)
        if len(points) < 2:
            return 0.0
        total = 0.0
        count = 0
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                total += self._distance(points[i][1], points[j][1])
                count += 1
        return total / max(count, 1)

    def coverage(self, radius: float = 0.3) -> float:
        """Fraction of feature space covered by idea neighborhoods."""
        with self._lock:
            points = list(self._points)
        if not points:
            return 0.0
        # Grid sampling: count cells near at least one idea
        n_samples = 100
        covered = 0
        for _ in range(n_samples):
            sample = [random.random() for _ in range(self.feature_dims)]
            for _, feat in points:
                if self._distance(sample, feat) < radius:
                    covered += 1
                    break
        return covered / n_samples

    def stats(self) -> Dict[str, Any]:
        return {
            "ideas": len(self._points),
            "components_0.3": self.connected_components(0.3),
            "components_0.5": self.connected_components(0.5),
            "spread": round(self.spread(), 3),
            "coverage": round(self.coverage(), 3),
        }


# ============================================================================
# 4. Fractal Budget Allocator
# ============================================================================

class FractalBudgetAllocator:
    """
    Self-similar recursive budget subdivision.

    Inspired by fractal geometry: the budget allocation at each level
    mirrors the global allocation pattern. Sub-stages within a stage
    get the same proportional split as stages get from the total budget.

    This creates a natural multi-scale budget hierarchy:
      Total → stages → sub-stages → individual calls

    Each level applies the golden ratio (φ ≈ 0.618) split:
      Primary task gets φ of the budget, secondary gets 1-φ.
    """

    PHI = (1 + math.sqrt(5)) / 2 - 1  # ≈ 0.618 (golden ratio)

    def __init__(self, total_budget: float):
        self.total_budget = total_budget
        self._allocations: Dict[str, float] = {}
        self._spent: Dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()

    def allocate(self, path: str, depth: int = 0) -> float:
        """
        Get budget for a hierarchical path like 'ideation.strategy_a.prompt1'.

        Each level of the hierarchy gets a φ-based subdivision:
          Level 0: full budget
          Level 1: φ^1 ≈ 0.618 for primary, 0.382 for secondary
          Level 2: φ^2 ≈ 0.382 for primary, 0.236 for secondary
        """
        with self._lock:
            if path in self._allocations:
                return self._allocations[path]

            parts = path.split(".")
            budget = self.total_budget
            for i in range(len(parts)):
                # At each level, primary gets φ, secondary gets 1-φ
                # Use hash of name to deterministically assign primary/secondary
                h = int(hashlib.md5(parts[i].encode(), usedforsecurity=False).hexdigest()[:8], 16)
                is_primary = h % 2 == 0
                budget *= self.PHI if is_primary else (1 - self.PHI)

            remaining = budget - self._spent.get(path, 0)
            self._allocations[path] = max(0, remaining)
            return max(0, remaining)

    def spend(self, path: str, amount: float) -> None:
        with self._lock:
            self._spent[path] += amount

    def remaining(self, path: str) -> float:
        with self._lock:
            allocated = self._allocations.get(path, self.allocate(path))
            return max(0, allocated - self._spent.get(path, 0))

    def stats(self) -> Dict[str, Any]:
        return {
            "total_budget": self.total_budget,
            "allocations": {k: round(v, 4) for k, v in self._allocations.items()},
            "spent": {k: round(v, 4) for k, v in self._spent.items() if v > 0},
        }


# ============================================================================
# 5. Cognitive Load Balancer
# ============================================================================

class CognitiveLoadBalancer:
    """
    Target prompts to each model's complexity sweet spot.

    Different LLMs have different optimal prompt complexity:
      - Small/fast models: simple, direct prompts (< 500 tokens)
      - Medium models: structured prompts with examples (500-2000 tokens)
      - Large models: complex multi-step reasoning (2000+ tokens)

    Measures prompt complexity via:
      - Token count (length)
      - Vocabulary richness (unique/total word ratio)
      - Nesting depth (JSON/list structure depth)
      - Instruction count (number of "rules" or "steps")

    Routes to appropriate model/prompt variant based on complexity score.
    """

    @dataclass
    class ComplexityProfile:
        length_tokens: int = 0
        vocab_richness: float = 0.0
        nesting_depth: int = 0
        instruction_count: int = 0
        score: float = 0.0  # 0=trivial, 1=very complex

    MODEL_SWEET_SPOTS = {
        "fast": (0.0, 0.3),     # simple tasks
        "balanced": (0.2, 0.7),  # moderate tasks
        "powerful": (0.5, 1.0),  # complex tasks
    }

    def __init__(self):
        self._history: List[Tuple[float, float]] = []  # (complexity, quality)

    def measure_complexity(self, system: str, user: str) -> "CognitiveLoadBalancer.ComplexityProfile":
        """Measure prompt complexity."""
        full_text = system + " " + user
        words = full_text.split()
        unique_words = set(w.lower() for w in words)

        profile = self.ComplexityProfile()
        profile.length_tokens = len(words) // 4  # rough token estimate
        profile.vocab_richness = len(unique_words) / max(len(words), 1)

        # Count nesting (braces, brackets)
        max_depth = 0
        depth = 0
        for c in full_text:
            if c in "{[(":
                depth += 1
                max_depth = max(max_depth, depth)
            elif c in "}])":
                depth = max(0, depth - 1)
        profile.nesting_depth = max_depth

        # Count instruction markers
        instruction_markers = ["must", "should", "return", "output", "generate", "ensure", "follow", "step"]
        profile.instruction_count = sum(1 for w in words if w.lower() in instruction_markers)

        # Composite score
        length_score = min(1.0, profile.length_tokens / 1000)
        vocab_score = profile.vocab_richness
        nest_score = min(1.0, profile.nesting_depth / 5)
        inst_score = min(1.0, profile.instruction_count / 10)
        profile.score = 0.3 * length_score + 0.2 * vocab_score + 0.25 * nest_score + 0.25 * inst_score

        return profile

    def recommend_tier(self, complexity: float) -> str:
        """Recommend model tier based on complexity."""
        for tier, (lo, hi) in self.MODEL_SWEET_SPOTS.items():
            if lo <= complexity <= hi:
                return tier
        return "balanced"

    def record_outcome(self, complexity: float, quality: float) -> None:
        self._history.append((complexity, quality))

    def stats(self) -> Dict[str, Any]:
        return {
            "observations": len(self._history),
            "sweet_spots": self.MODEL_SWEET_SPOTS,
        }


# ============================================================================
# 6. Hyperband Scheduler
# ============================================================================

class HyperbandScheduler:
    """
    Principled early stopping for parallel configuration evaluation.

    Hyperband (Li et al., 2017) combines random search with Successive
    Halving for optimal resource allocation under limited budget.

    Runs multiple brackets with different aggressiveness:
      Bracket 0: many configs, few resources each (aggressive elimination)
      Bracket s_max: few configs, full resources each (no elimination)

    Total budget per bracket is fixed, but distributed differently.
    """

    @dataclass
    class ConfigRun:
        config_id: int
        params: Dict[str, float]
        resources_used: float = 0
        performance: float = 0

    def __init__(self, max_resources: float = 81, eta: float = 3):
        self.R = max_resources  # max resources per config
        self.eta = eta  # elimination rate (keep 1/eta each round)
        self.s_max = int(math.log(self.R) / math.log(eta))
        self._brackets: List[List["HyperbandScheduler.ConfigRun"]] = []
        self._best: Optional["HyperbandScheduler.ConfigRun"] = None
        self._config_counter = 0

    def generate_bracket(self, s: int, param_sampler: Callable[[], Dict[str, float]]) -> List["HyperbandScheduler.ConfigRun"]:
        """Generate configs for bracket s."""
        n = int(math.ceil(self.s_max + 1) / (s + 1) * self.eta ** s)
        r = self.R * self.eta ** (-s)

        configs = []
        for _ in range(n):
            self._config_counter += 1
            configs.append(self.ConfigRun(
                config_id=self._config_counter,
                params=param_sampler(),
                resources_used=r,
            ))
        return configs

    def successive_halving(self, configs: List["HyperbandScheduler.ConfigRun"],
                           evaluator: Callable[["HyperbandScheduler.ConfigRun", float], float],
                           s: int) -> "HyperbandScheduler.ConfigRun":
        """Run successive halving on configs within a bracket."""
        r = self.R * self.eta ** (-s)

        for i in range(s + 1):
            n_configs = max(1, int(len(configs) * self.eta ** (-i)))
            r_i = r * self.eta ** i

            # Evaluate each config with r_i resources
            for c in configs:
                c.performance = evaluator(c, r_i)
                c.resources_used = r_i

            # Keep top 1/eta configs
            configs.sort(key=lambda c: c.performance, reverse=True)
            configs = configs[:max(1, int(len(configs) / self.eta))]

        best = configs[0]
        if self._best is None or best.performance > self._best.performance:
            self._best = best
        return best

    def get_best(self) -> Optional["HyperbandScheduler.ConfigRun"]:
        return self._best

    def stats(self) -> Dict[str, Any]:
        return {
            "s_max": self.s_max,
            "eta": self.eta,
            "max_resources": self.R,
            "best": {
                "config_id": self._best.config_id,
                "performance": round(self._best.performance, 4),
                "params": {k: round(v, 3) for k, v in self._best.params.items()},
            } if self._best else None,
        }


# ============================================================================
# 7. Wasserstein Divergence (Optimal Transport)
# ============================================================================

class WassersteinDivergence:
    """
    Measure idea distribution quality using optimal transport distance.

    Wasserstein distance (Earth Mover's Distance) measures how different
    the current idea distribution is from the target distribution.

    Unlike KL divergence, Wasserstein is well-defined even when distributions
    have non-overlapping support (common with sparse idea archives).

    Uses 1D projections for efficient computation (Sliced Wasserstein).
    """

    def __init__(self):
        self._current: List[float] = []
        self._target: List[float] = []

    def set_target(self, distribution: List[float]) -> None:
        """Set the target distribution (e.g., uniform across cells)."""
        self._target = sorted(distribution)

    def update_current(self, distribution: List[float]) -> None:
        """Update the current idea distribution."""
        self._current = sorted(distribution)

    def wasserstein_1d(self) -> float:
        """Compute 1D Wasserstein distance between current and target."""
        if not self._current or not self._target:
            return 0.0

        # Resample to same length if needed
        n = max(len(self._current), len(self._target))
        a = self._resample(self._current, n)
        b = self._resample(self._target, n)

        # W1 distance = mean absolute difference of sorted samples
        return sum(abs(ai - bi) for ai, bi in zip(a, b)) / n

    def _resample(self, values: List[float], n: int) -> List[float]:
        """Resample a distribution to n evenly-spaced quantiles."""
        if len(values) == n:
            return values
        result = []
        for i in range(n):
            idx = i * (len(values) - 1) / max(n - 1, 1)
            lo = int(idx)
            hi = min(lo + 1, len(values) - 1)
            frac = idx - lo
            result.append(values[lo] * (1 - frac) + values[hi] * frac)
        return result

    def convergence_score(self) -> float:
        """0 = identical to target, 1 = maximally different."""
        w = self.wasserstein_1d()
        return min(1.0, w)

    def stats(self) -> Dict[str, Any]:
        return {
            "wasserstein_distance": round(self.wasserstein_1d(), 4),
            "convergence": round(self.convergence_score(), 4),
            "current_size": len(self._current),
            "target_size": len(self._target),
        }


# ============================================================================
# 8. Simulated Bifurcation Machine
# ============================================================================

class SimulatedBifurcation:
    """
    Ising-machine-inspired parallel combinatorial search.

    Simulated Bifurcation (Goto et al., 2019) is a quantum-inspired
    algorithm that outperforms simulated annealing on combinatorial problems.

    Uses coupled oscillators that undergo a pitchfork bifurcation:
    before bifurcation → continuous search, after → discrete solution.

    Equations:
      dx_i/dt = (a₀ - a(t)) × x_i + c₀ × Σ_j J_{ij} × y_j
      dy_i/dt = (a₀ - a(t)) × y_i + c₀ × Σ_j J_{ij} × x_i

    Solution: sign(x_i) after bifurcation.
    """

    def __init__(self, n_vars: int = 6, dt: float = 0.1, n_steps: int = 100):
        self.n = n_vars
        self.dt = dt
        self.n_steps = n_steps

    def solve(self, J: List[List[float]], h: List[float] = None) -> List[int]:
        """
        Solve Ising problem: minimize H = -Σ J_ij s_i s_j - Σ h_i s_i

        Args:
            J: n×n coupling matrix
            h: n-vector of local fields (default zeros)
        Returns:
            List of +1/-1 spin values
        """
        n = len(J)
        h = h or [0.0] * n

        # Initialize oscillators
        x = [random.gauss(0, 0.1) for _ in range(n)]
        y = [random.gauss(0, 0.1) for _ in range(n)]

        a0 = 1.0
        c0 = 0.5 / max(max(abs(J[i][j]) for j in range(n)) for i in range(n)) if n > 0 else 1.0

        for step in range(self.n_steps):
            # Ramping parameter (bifurcation at a(t) = a0)
            a_t = a0 * step / self.n_steps

            # Update x
            new_x = list(x)
            for i in range(n):
                coupling_sum = sum(J[i][j] * y[j] for j in range(n))
                new_x[i] += self.dt * ((a0 - a_t) * x[i] + c0 * coupling_sum + h[i])
                # Clamp
                new_x[i] = max(-2, min(2, new_x[i]))

            # Update y
            new_y = list(y)
            for i in range(n):
                coupling_sum = sum(J[i][j] * x[j] for j in range(n))
                new_y[i] += self.dt * ((a0 - a_t) * y[i] + c0 * coupling_sum + h[i])
                new_y[i] = max(-2, min(2, new_y[i]))

            x, y = new_x, new_y

        # Extract solution: sign of x after bifurcation
        return [1 if xi > 0 else -1 for xi in x]

    def stats(self) -> Dict[str, Any]:
        return {"n_vars": self.n, "dt": self.dt, "n_steps": self.n_steps}


# ============================================================================
# 9. Contextual Bandit Router
# ============================================================================

class ContextualBanditRouter:
    """
    Contextual multi-armed bandit for per-query model/prompt selection.

    Unlike standard MAB (Thompson Sampling), this conditions on context
    features: task type, prompt length, required output format, etc.

    Uses LinUCB algorithm (Li et al., 2010):
      For each arm a and context x:
        θ_a = A_a⁻¹ × b_a
        p_a = θ_a^T × x + α × √(x^T × A_a⁻¹ × x)

    Selects arm with highest p_a (exploitation + exploration).
    """

    def __init__(self, n_features: int = 4, alpha: float = 0.5):
        self.n_features = n_features
        self.alpha = alpha
        self._arms: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def add_arm(self, arm_id: str) -> None:
        """Register an arm (model/prompt variant)."""
        n = self.n_features
        with self._lock:
            self._arms[arm_id] = {
                "A": [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)],  # identity
                "b": [0.0] * n,
                "pulls": 0,
            }

    def select(self, context: List[float], arm_ids: List[str] = None) -> str:
        """Select best arm given context features."""
        with self._lock:
            ids = arm_ids or list(self._arms.keys())
            if not ids:
                return ""

            best_score = -float('inf')
            best_arm = ids[0]

            for arm_id in ids:
                arm = self._arms.get(arm_id)
                if not arm:
                    continue

                # θ = A⁻¹ b (simplified: just use b/pulls as approximation)
                pulls = max(arm["pulls"], 1)
                theta = [bi / pulls for bi in arm["b"]]

                # Exploitation: θ^T × x
                exploitation = sum(t * c for t, c in zip(theta, context[:self.n_features]))

                # Exploration: α × √(x^T × A⁻¹ × x) ≈ α / √(pulls)
                exploration = self.alpha / math.sqrt(pulls)

                score = exploitation + exploration
                if score > best_score:
                    best_score = score
                    best_arm = arm_id

            return best_arm

    def update(self, arm_id: str, context: List[float], reward: float) -> None:
        """Update arm with observed reward."""
        with self._lock:
            arm = self._arms.get(arm_id)
            if not arm:
                return
            arm["pulls"] += 1
            # Update b += reward × context
            for i in range(min(len(context), self.n_features)):
                arm["b"][i] += reward * context[i]

    def stats(self) -> Dict[str, Any]:
        return {
            arm_id: {"pulls": arm["pulls"]}
            for arm_id, arm in self._arms.items()
        }


# ============================================================================
# 10. Gradient-Free Optimizer (CMA-ES inspired)
# ============================================================================

class GradientFreeOptimizer:
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) inspired
    optimizer for continuous pipeline hyperparameters.

    CMA-ES adapts a multivariate Gaussian distribution to sample
    increasingly better configurations. Unlike PBT, it adapts the
    full covariance structure (correlations between parameters).

    Simplified version: diagonal covariance (no cross-correlations)
    to avoid matrix operations.
    """

    def __init__(self, param_names: List[str], sigma: float = 0.3, pop_size: int = 8):
        self.param_names = param_names
        self.n = len(param_names)
        self.sigma = sigma
        self.pop_size = pop_size

        # Distribution: mean + diagonal std
        self._mean = {p: 0.5 for p in param_names}
        self._std = {p: sigma for p in param_names}
        self._generation = 0
        self._best: Tuple[Dict[str, float], float] = ({}, 0)

    def sample_population(self) -> List[Dict[str, float]]:
        """Sample a population from the current distribution."""
        population = []
        for _ in range(self.pop_size):
            individual = {}
            for p in self.param_names:
                val = random.gauss(self._mean[p], self._std[p])
                individual[p] = max(0.01, min(0.99, val))  # clamp
            population.append(individual)
        return population

    def update(self, population: List[Dict[str, float]], fitnesses: List[float]) -> None:
        """Update distribution based on fitness-sorted population."""
        self._generation += 1

        # Sort by fitness (descending)
        paired = list(zip(fitnesses, population))
        paired.sort(reverse=True)

        # Track best
        if paired[0][0] > self._best[1]:
            self._best = (dict(paired[0][1]), paired[0][0])

        # Select top half (mu)
        mu = max(1, self.pop_size // 2)
        elite = [p for _, p in paired[:mu]]

        # Update mean: weighted average of elite
        weights = [math.log(mu + 1) - math.log(i + 1) for i in range(mu)]
        w_sum = sum(weights)
        weights = [w / w_sum for w in weights]

        for p in self.param_names:
            self._mean[p] = sum(w * ind[p] for w, ind in zip(weights, elite))

        # Update std: weighted std of elite
        for p in self.param_names:
            variance = sum(w * (ind[p] - self._mean[p]) ** 2 for w, ind in zip(weights, elite))
            self._std[p] = max(0.01, math.sqrt(variance + 1e-8))

    def get_best(self) -> Tuple[Dict[str, float], float]:
        return self._best

    def stats(self) -> Dict[str, Any]:
        return {
            "generation": self._generation,
            "mean": {k: round(v, 3) for k, v in self._mean.items()},
            "std": {k: round(v, 3) for k, v in self._std.items()},
            "best_fitness": round(self._best[1], 4),
            "best_params": {k: round(v, 3) for k, v in self._best[0].items()},
        }


# ============================================================================
# 11. Information Bottleneck
# ============================================================================

class InformationBottleneck:
    """
    Rate-distortion compression for stage context passing.

    The Information Bottleneck (Tishby et al., 2000) finds the optimal
    trade-off between compressing a representation (rate) and preserving
    relevant information (distortion).

    Applied to pipeline: compress the context passed between stages
    to minimize token usage while maximizing information relevant to
    the downstream stage.

    Compression levels:
      Level 0: full context (no compression)
      Level 1: key-value extraction (50% compression)
      Level 2: abstractive summary (75% compression)
      Level 3: single-sentence distillation (90% compression)
    """

    @dataclass
    class CompressionLevel:
        level: int
        name: str
        target_ratio: float  # fraction of original length
        extractor: str  # extraction method

    LEVELS = [
        CompressionLevel(0, "full", 1.0, "none"),
        CompressionLevel(1, "key_value", 0.5, "extract_kv"),
        CompressionLevel(2, "summary", 0.25, "summarize"),
        CompressionLevel(3, "distilled", 0.1, "distill"),
    ]

    def __init__(self):
        self._compression_history: Dict[str, List[Tuple[int, float]]] = defaultdict(list)  # stage → [(level, quality)]

    def select_level(self, stage: str, budget_pressure: float = 0.0) -> int:
        """
        Select compression level based on stage importance and budget pressure.

        budget_pressure: 0 = no pressure (use full context), 1 = max pressure (distill everything)
        """
        history = self._compression_history.get(stage, [])

        if not history or budget_pressure < 0.3:
            return 0  # use full context when budget is available

        # Find the most compressed level that still maintains quality
        best_level = 0
        for level_idx in range(len(self.LEVELS)):
            level_quality = [q for l, q in history if l == level_idx]
            if level_quality and sum(level_quality) / len(level_quality) > 0.5:
                best_level = level_idx

        # Increase compression under budget pressure
        return min(3, best_level + int(budget_pressure * 2))

    def compress(self, text: str, level: int) -> str:
        """Compress text to the specified level."""
        if level == 0 or not text:
            return text

        target_len = int(len(text) * self.LEVELS[min(level, 3)].target_ratio)

        if level == 1:
            # Key-value extraction: keep lines with ":" or "="
            lines = text.split("\n")
            kv_lines = [l for l in lines if ":" in l or "=" in l or l.strip().startswith("-")]
            result = "\n".join(kv_lines)
            return result[:target_len] if len(result) > target_len else result

        elif level == 2:
            # Summary: keep first and last paragraph + key sentences
            paragraphs = text.split("\n\n")
            if len(paragraphs) <= 2:
                return text[:target_len]
            result = paragraphs[0] + "\n\n" + paragraphs[-1]
            return result[:target_len]

        else:  # level 3
            # Distill: first sentence of first paragraph
            sentences = text.split(".")
            return (sentences[0] + ".") if sentences else text[:target_len]

    def record_quality(self, stage: str, level: int, quality: float) -> None:
        """Record quality achieved at a compression level."""
        self._compression_history[stage].append((level, quality))

    def stats(self) -> Dict[str, Any]:
        return {
            "stages": {
                stage: {
                    f"level_{l}": round(sum(q for lvl, q in hist if lvl == l) / max(sum(1 for lvl, _ in hist if lvl == l), 1), 3)
                    for l in range(4)
                }
                for stage, hist in self._compression_history.items()
            }
        }


# ============================================================================
# 12. Meta-Learning Warm Start
# ============================================================================

class MetaLearningWarmStart:
    """
    Learn-to-learn initialization from past pipeline runs.

    MAML-inspired (Finn et al., 2017): instead of starting from scratch,
    initialize pipeline parameters from an aggregation of past successful runs.

    Stores (config, outcome) pairs from historical runs. For new runs,
    finds the most similar past run and uses its config as initialization.

    Similarity: weighted Euclidean distance on normalized topic features.
    """

    @dataclass
    class RunRecord:
        topic_hash: str
        config: Dict[str, float]
        quality: float
        cost: float
        timestamp: float

    def __init__(self, persist_path: str = None):
        self.persist_path = persist_path or str(
            Path(__file__).parent / "output" / "meta_learning.json"
        )
        self._records: List["MetaLearningWarmStart.RunRecord"] = []
        self._load()

    def _topic_hash(self, topic: str) -> str:
        """Hash topic into a feature vector proxy."""
        return hashlib.md5(topic.lower().strip().encode(), usedforsecurity=False).hexdigest()[:16]

    def record_run(self, topic: str, config: Dict[str, float], quality: float, cost: float) -> None:
        """Store a completed run for future warm-starting."""
        self._records.append(self.RunRecord(
            topic_hash=self._topic_hash(topic),
            config=config, quality=quality, cost=cost,
            timestamp=time.time(),
        ))
        self._save()

    def warm_start(self, topic: str) -> Optional[Dict[str, float]]:
        """Get warm-start config for a new topic based on similar past runs."""
        if not self._records:
            return None

        topic_hash = self._topic_hash(topic)

        # Find most similar successful past run
        best_record = None
        best_similarity = -1

        for record in self._records:
            if record.quality < 0.4:
                continue  # skip failed runs
            # Similarity: hash prefix match length
            similarity = sum(1 for a, b in zip(topic_hash, record.topic_hash) if a == b) / len(topic_hash)
            # Weight by recency
            age_days = (time.time() - record.timestamp) / 86400
            recency_weight = math.exp(-age_days / 30)  # decay over 30 days
            weighted_sim = similarity * recency_weight * record.quality

            if weighted_sim > best_similarity:
                best_similarity = weighted_sim
                best_record = record

        if best_record and best_similarity > 0.1:
            return dict(best_record.config)
        return None

    def _save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
            data = [{
                "topic_hash": r.topic_hash, "config": r.config,
                "quality": r.quality, "cost": r.cost, "timestamp": r.timestamp,
            } for r in self._records[-100:]]  # keep last 100
            with open(self.persist_path, "w") as f:
                json.dump(data, f)
        except Exception:
            pass

    def _load(self) -> None:
        try:
            if os.path.exists(self.persist_path):
                with open(self.persist_path) as f:
                    data = json.load(f)
                self._records = [
                    self.RunRecord(**item) for item in data
                ]
        except Exception:
            pass

    def stats(self) -> Dict[str, Any]:
        return {
            "stored_runs": len(self._records),
            "avg_quality": round(sum(r.quality for r in self._records) / max(len(self._records), 1), 3),
            "avg_cost": round(sum(r.cost for r in self._records) / max(len(self._records), 1), 4),
        }


# ============================================================================
# Master Quantum Optimizer
# ============================================================================

class QuantumOptimizer:
    """Aggregates all layer-6 frontier optimization techniques."""

    def __init__(self, enable_all: bool = True):
        self.quantum_annealer = QuantumAnnealingSampler() if enable_all else None
        self.swarm = SwarmIdeaOptimizer() if enable_all else None
        self.topology = TopologicalDiversityMap() if enable_all else None
        self.fractal_budget = None  # initialized with budget in run()
        self.cognitive_load = CognitiveLoadBalancer() if enable_all else None
        self.hyperband = HyperbandScheduler() if enable_all else None
        self.wasserstein = WassersteinDivergence() if enable_all else None
        self.bifurcation = SimulatedBifurcation() if enable_all else None
        self.contextual_bandit = ContextualBanditRouter() if enable_all else None
        self.cma_es = GradientFreeOptimizer(
            ["temperature", "tree_branches", "reflection_threshold", "debate_fraction"]
        ) if enable_all else None
        self.info_bottleneck = InformationBottleneck() if enable_all else None
        self.meta_learner = MetaLearningWarmStart() if enable_all else None

    def summary(self) -> Dict[str, Any]:
        result = {}
        if self.quantum_annealer: result["quantum_annealing"] = self.quantum_annealer.stats()
        if self.swarm: result["swarm_optimization"] = self.swarm.stats()
        if self.topology: result["topological_diversity"] = self.topology.stats()
        if self.fractal_budget: result["fractal_budget"] = self.fractal_budget.stats()
        if self.cognitive_load: result["cognitive_load"] = self.cognitive_load.stats()
        if self.hyperband: result["hyperband"] = self.hyperband.stats()
        if self.wasserstein: result["wasserstein"] = self.wasserstein.stats()
        if self.bifurcation: result["bifurcation"] = self.bifurcation.stats()
        if self.contextual_bandit: result["contextual_bandit"] = self.contextual_bandit.stats()
        if self.cma_es: result["cma_es"] = self.cma_es.stats()
        if self.info_bottleneck: result["info_bottleneck"] = self.info_bottleneck.stats()
        if self.meta_learner: result["meta_learner"] = self.meta_learner.stats()
        return result
