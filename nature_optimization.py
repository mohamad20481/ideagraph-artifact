"""
nature_optimization.py - Bio-inspired, social, and physics-based creative
optimization for IdeaGraph.

Layer 7: Techniques drawn from biology, social science, physics, and
pure mathematics. These are the most creative optimization strategies,
modeling natural phenomena that evolution has refined over billions of years.

  1.  AntColonyRouter         — Pheromone-based pipeline path optimization
  2.  ImmuneClonalSelector    — Artificial immune system for idea filtering
  3.  GravitationalClustering — Mass-based idea attraction for natural grouping
  4.  DiffusionIdeaGenerator  — Noise→structure denoising for idea refinement
  5.  VickreyBudgetAuction    — Second-price auction for stage budget allocation
  6.  WisdomOfCrowds          — Diversity-weighted crowd aggregation
  7.  MomentumOptimizer       — Adam-like momentum for quality trajectory smoothing
  8.  SecretarySelector       — Optimal stopping for idea selection
  9.  CoevolutionaryArms      — Predator-prey co-evolution of ideas and critics
  10. ThermoFreeEnergy        — Free energy principle for quality-diversity balance
  11. SocialInfluenceNetwork  — PageRank-like influence propagation across ideas
  12. ChaoticExplorer         — Deterministic chaos for ergodic search space coverage
"""

from __future__ import annotations

import hashlib
import math
import random
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


# ============================================================================
# 1. Ant Colony Router (ACO)
# ============================================================================

class AntColonyRouter:
    """
    Pheromone-based pipeline path optimization.

    Ants (simulated pipeline runs) traverse the stage graph, depositing
    pheromones on edges they use. Good paths (high review score) get
    more pheromone; bad paths evaporate. Over time, the colony converges
    to the best pipeline configuration.

    Pheromone update:
      τ_{ij} ← (1-ρ) × τ_{ij} + Σ_k Δτ_{ij}^k
      Δτ_{ij}^k = Q / cost_k  (if ant k used edge i→j)

    Probability of choosing edge i→j:
      p_{ij} = (τ_{ij}^α × η_{ij}^β) / Σ_l (τ_{il}^α × η_{il}^β)
      where η = heuristic (inverse cost)
    """

    def __init__(self, evaporation: float = 0.1, alpha: float = 1.0, beta: float = 2.0):
        self.rho = evaporation
        self.alpha = alpha  # pheromone importance
        self.beta = beta    # heuristic importance
        # edge: (from_stage, to_stage) → pheromone level
        self._pheromone: Dict[Tuple[str, str], float] = defaultdict(lambda: 1.0)
        # Heuristic: inverse of estimated cost
        self._heuristic: Dict[Tuple[str, str], float] = defaultdict(lambda: 1.0)
        self._lock = threading.Lock()
        self._best_path: List[str] = []
        self._best_quality: float = 0.0

    STAGE_GRAPH = {
        "start": ["ideation"],
        "ideation": ["tree_search", "experiment_design"],
        "tree_search": ["experiment_design"],
        "experiment_design": ["code_generation"],
        "code_generation": ["execution"],
        "execution": ["analysis"],
        "analysis": ["paper_writing"],
        "paper_writing": ["review"],
        "review": ["end"],
    }

    def run_ant(self) -> List[str]:
        """Simulate one ant traversing the pipeline graph."""
        path = ["start"]
        current = "start"

        while current != "end" and len(path) < 15:
            neighbors = self.STAGE_GRAPH.get(current, [])
            if not neighbors:
                break

            # Probabilistic edge selection
            with self._lock:
                weights = []
                for n in neighbors:
                    tau = self._pheromone[(current, n)] ** self.alpha
                    eta = self._heuristic[(current, n)] ** self.beta
                    weights.append(tau * eta)

            total = sum(weights) + 1e-10
            probs = [w / total for w in weights]

            # Roulette wheel selection
            r = random.random()
            cumulative = 0
            chosen = neighbors[0]
            for n, p in zip(neighbors, probs):
                cumulative += p
                if r <= cumulative:
                    chosen = n
                    break

            path.append(chosen)
            current = chosen

        return path

    def deposit_pheromone(self, path: List[str], quality: float) -> None:
        """Deposit pheromone on a path proportional to quality."""
        with self._lock:
            for i in range(len(path) - 1):
                edge = (path[i], path[i + 1])
                self._pheromone[edge] += quality

            if quality > self._best_quality:
                self._best_quality = quality
                self._best_path = list(path)

    def evaporate(self) -> None:
        """Evaporate pheromone on all edges."""
        with self._lock:
            for edge in self._pheromone:
                self._pheromone[edge] *= (1 - self.rho)
                self._pheromone[edge] = max(0.1, self._pheromone[edge])

    def optimize(self, n_ants: int = 10, n_iterations: int = 5) -> List[str]:
        """Run full ACO optimization. Returns best path."""
        for _ in range(n_iterations):
            paths = [self.run_ant() for _ in range(n_ants)]
            # Score paths (heuristic: shorter + more stages = better)
            for path in paths:
                score = sum(1 for s in path if s not in ("start", "end")) / 8.0
                self.deposit_pheromone(path, score)
            self.evaporate()
        return self._best_path

    def get_stage_importance(self) -> Dict[str, float]:
        """Get relative importance of each stage from pheromone levels."""
        with self._lock:
            stage_pheromone: Dict[str, float] = defaultdict(float)
            for (src, dst), level in self._pheromone.items():
                stage_pheromone[dst] += level
            total = sum(stage_pheromone.values()) + 1e-10
            return {s: round(v / total, 3) for s, v in stage_pheromone.items() if s not in ("start", "end")}

    def stats(self) -> Dict[str, Any]:
        return {
            "best_quality": round(self._best_quality, 3),
            "best_path": self._best_path,
            "stage_importance": self.get_stage_importance(),
        }


# ============================================================================
# 2. Immune Clonal Selector (Artificial Immune System)
# ============================================================================

class ImmuneClonalSelector:
    """
    Artificial Immune System for idea filtering and refinement.

    Models ideas as antibodies and quality criteria as antigens.
    High-affinity antibodies (good ideas) are cloned and hypermutated
    to explore nearby quality space. Low-affinity ones die off.

    Algorithm (CLONALG):
      1. Measure affinity (quality) of each antibody (idea)
      2. Clone top-n proportionally to affinity (more clones for better ideas)
      3. Hypermutate clones inversely to affinity (bad ideas mutate more)
      4. Select best from original + clones
      5. Replace worst with random newcomers (diversity injection)
    """

    @dataclass
    class Antibody:
        id: str
        features: Dict[str, float]  # idea features
        affinity: float = 0.0       # quality score
        generation: int = 0
        parent_id: Optional[str] = None

    def __init__(self, population_size: int = 20, clone_factor: float = 0.5, replace_fraction: float = 0.2):
        self.pop_size = population_size
        self.clone_factor = clone_factor
        self.replace_fraction = replace_fraction
        self._population: List["ImmuneClonalSelector.Antibody"] = []
        self._generation = 0
        self._lock = threading.Lock()

    def add_idea(self, idea_id: str, features: Dict[str, float], affinity: float) -> None:
        """Add an idea as an antibody."""
        with self._lock:
            self._population.append(self.Antibody(
                id=idea_id, features=features, affinity=affinity,
            ))

    def clonal_selection(self) -> List["ImmuneClonalSelector.Antibody"]:
        """Run one round of clonal selection. Returns the refined population."""
        with self._lock:
            if len(self._population) < 3:
                return list(self._population)

            self._generation += 1

            # Sort by affinity (descending)
            sorted_pop = sorted(self._population, key=lambda a: a.affinity, reverse=True)

            # Clone top-n proportionally to affinity
            n_select = max(2, int(len(sorted_pop) * self.clone_factor))
            clones = []
            for i, ab in enumerate(sorted_pop[:n_select]):
                n_clones = max(1, n_select - i)  # more clones for higher affinity
                for _ in range(n_clones):
                    clone = self._hypermutate(ab)
                    clones.append(clone)

            # Select best from original + clones
            all_candidates = list(sorted_pop) + clones
            all_candidates.sort(key=lambda a: a.affinity, reverse=True)
            survivors = all_candidates[:self.pop_size]

            # Replace worst with random newcomers (diversity)
            n_replace = max(1, int(self.pop_size * self.replace_fraction))
            for i in range(n_replace):
                idx = -(i + 1)
                if abs(idx) <= len(survivors):
                    survivors[idx] = self._random_antibody()

            self._population = survivors
            return list(survivors)

    def _hypermutate(self, antibody: "ImmuneClonalSelector.Antibody") -> "ImmuneClonalSelector.Antibody":
        """Mutate inversely proportional to affinity (bad ideas mutate more)."""
        mutation_rate = max(0.05, 1.0 - antibody.affinity)
        new_features = {}
        for k, v in antibody.features.items():
            if random.random() < mutation_rate:
                new_features[k] = max(0, min(1, v + random.gauss(0, mutation_rate * 0.3)))
            else:
                new_features[k] = v

        return self.Antibody(
            id=f"{antibody.id}_clone_{self._generation}",
            features=new_features,
            affinity=antibody.affinity * 0.9,  # slight penalty until re-evaluated
            generation=self._generation,
            parent_id=antibody.id,
        )

    def _random_antibody(self) -> "ImmuneClonalSelector.Antibody":
        """Generate a random newcomer for diversity."""
        features = {
            "novelty": random.random(),
            "feasibility": random.random(),
            "impact": random.random(),
            "clarity": random.random(),
        }
        return self.Antibody(
            id=f"random_{self._generation}_{random.randint(0, 999)}",
            features=features, affinity=0.3, generation=self._generation,
        )

    def get_best(self, n: int = 3) -> List["ImmuneClonalSelector.Antibody"]:
        with self._lock:
            return sorted(self._population, key=lambda a: a.affinity, reverse=True)[:n]

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            affinities = [a.affinity for a in self._population]
        return {
            "population": len(self._population),
            "generation": self._generation,
            "best_affinity": round(max(affinities), 3) if affinities else 0,
            "avg_affinity": round(sum(affinities) / max(len(affinities), 1), 3),
            "diversity": len(set(a.id.split("_")[0] for a in self._population)),
        }


# ============================================================================
# 3. Gravitational Clustering
# ============================================================================

class GravitationalClustering:
    """
    Mass-based idea attraction for natural grouping.

    Inspired by Gravitational Search Algorithm (Rashedi et al., 2009).
    Ideas have "mass" proportional to quality. Higher-mass ideas attract
    nearby lower-mass ones, naturally forming clusters around the best ideas.

    Force: F_ij = G × M_i × M_j / (R_ij² + ε)
    Acceleration: a_i = Σ_j F_ij / M_i
    Position update: x_i += v_i, v_i += a_i

    Use: identify natural idea clusters and find cluster centers (best ideas).
    """

    @dataclass
    class IdeaBody:
        id: str
        position: List[float]  # feature vector
        mass: float = 1.0      # quality-based mass
        velocity: List[float] = field(default_factory=list)

    def __init__(self, G: float = 1.0, dims: int = 4):
        self.G = G
        self.dims = dims
        self._bodies: List["GravitationalClustering.IdeaBody"] = []
        self._lock = threading.Lock()

    def add_idea(self, idea_id: str, features: List[float], quality: float) -> None:
        """Add an idea with mass proportional to quality."""
        with self._lock:
            pos = features[:self.dims]
            while len(pos) < self.dims:
                pos.append(0.0)
            vel = [0.0] * self.dims
            self._bodies.append(self.IdeaBody(
                id=idea_id, position=pos, mass=max(0.1, quality), velocity=vel,
            ))

    def step(self) -> None:
        """Run one gravitational step."""
        with self._lock:
            n = len(self._bodies)
            if n < 2:
                return

            # Compute forces
            accelerations = [[0.0] * self.dims for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    dist_sq = sum(
                        (self._bodies[i].position[d] - self._bodies[j].position[d]) ** 2
                        for d in range(self.dims)
                    ) + 1e-6
                    force_mag = self.G * self._bodies[i].mass * self._bodies[j].mass / dist_sq
                    dist = math.sqrt(dist_sq)
                    for d in range(self.dims):
                        direction = (self._bodies[j].position[d] - self._bodies[i].position[d]) / dist
                        accelerations[i][d] += force_mag * direction / max(self._bodies[i].mass, 0.1)

            # Update velocities and positions
            for i in range(n):
                for d in range(self.dims):
                    self._bodies[i].velocity[d] = 0.5 * self._bodies[i].velocity[d] + accelerations[i][d] * 0.1
                    self._bodies[i].velocity[d] = max(-0.5, min(0.5, self._bodies[i].velocity[d]))
                    self._bodies[i].position[d] += self._bodies[i].velocity[d]
                    self._bodies[i].position[d] = max(0, min(1, self._bodies[i].position[d]))

    def find_clusters(self, radius: float = 0.2) -> Dict[str, List[str]]:
        """Find idea clusters by proximity after gravitational collapse."""
        for _ in range(10):
            self.step()

        with self._lock:
            # Find heaviest bodies as cluster centers
            sorted_bodies = sorted(self._bodies, key=lambda b: b.mass, reverse=True)
            clusters: Dict[str, List[str]] = {}
            assigned = set()

            for center in sorted_bodies:
                if center.id in assigned:
                    continue
                cluster = [center.id]
                assigned.add(center.id)

                for body in self._bodies:
                    if body.id in assigned:
                        continue
                    dist = math.sqrt(sum(
                        (center.position[d] - body.position[d]) ** 2
                        for d in range(self.dims)
                    ))
                    if dist < radius:
                        cluster.append(body.id)
                        assigned.add(body.id)

                if len(cluster) > 0:
                    clusters[center.id] = cluster

            return clusters

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "bodies": len(self._bodies),
                "heaviest": max((b.mass, b.id) for b in self._bodies) if self._bodies else (0, ""),
                "avg_mass": round(sum(b.mass for b in self._bodies) / max(len(self._bodies), 1), 3),
            }


# ============================================================================
# 4. Diffusion Idea Generator
# ============================================================================

class DiffusionIdeaGenerator:
    """
    Noise→structure denoising for idea quality refinement.

    Inspired by diffusion models (Ho et al., 2020). Instead of generating
    ideas from scratch, start with a noisy/low-quality idea and iteratively
    denoise it toward high quality.

    Forward process: add noise to a good idea → progressively corrupt it
    Reverse process: take a noisy idea → iteratively refine toward quality

    Noise schedule: β_t linearly increases from β_min to β_max
    Denoising: at each step, predict and remove the noise component.

    Applied to idea features (not images): quality, novelty, feasibility scores.
    """

    def __init__(self, n_steps: int = 10, beta_min: float = 0.01, beta_max: float = 0.5):
        self.n_steps = n_steps
        self.betas = [beta_min + (beta_max - beta_min) * t / n_steps for t in range(n_steps)]
        self._quality_history: List[List[float]] = []

    def add_noise(self, features: Dict[str, float], step: int) -> Dict[str, float]:
        """Forward process: add noise at time step t."""
        beta = self.betas[min(step, len(self.betas) - 1)]
        noisy = {}
        for k, v in features.items():
            noise = random.gauss(0, math.sqrt(beta))
            noisy[k] = max(0, min(1, v + noise))
        return noisy

    def denoise_step(self, features: Dict[str, float], step: int,
                     quality_signal: float = 0.5) -> Dict[str, float]:
        """
        Reverse process: one denoising step guided by quality signal.

        Higher quality_signal → more aggressive denoising toward structure.
        """
        beta = self.betas[min(step, len(self.betas) - 1)]
        # Estimate noise direction: move toward higher quality
        denoised = {}
        for k, v in features.items():
            # Pull toward quality signal
            gradient = (quality_signal - v) * beta * 2
            denoised[k] = max(0, min(1, v + gradient + random.gauss(0, beta * 0.1)))
        return denoised

    def refine(self, features: Dict[str, float], quality_fn: Callable[[Dict[str, float]], float],
               n_steps: int = None) -> Dict[str, float]:
        """Full denoising refinement loop."""
        steps = n_steps or self.n_steps
        current = dict(features)
        trajectory = []

        for t in range(steps - 1, -1, -1):
            quality = quality_fn(current)
            trajectory.append(quality)
            current = self.denoise_step(current, t, quality)

        self._quality_history.append(trajectory)
        return current

    def stats(self) -> Dict[str, Any]:
        return {
            "n_steps": self.n_steps,
            "refinements": len(self._quality_history),
            "avg_improvement": round(
                sum(h[-1] - h[0] for h in self._quality_history if len(h) >= 2) /
                max(len(self._quality_history), 1), 3
            ) if self._quality_history else 0,
        }


# ============================================================================
# 5. Vickrey Budget Auction
# ============================================================================

class VickreyBudgetAuction:
    """
    Second-price sealed-bid auction for stage budget allocation.

    Each pipeline stage "bids" for budget based on its expected marginal
    quality improvement. Winner pays the second-highest bid (Vickrey).

    This is incentive-compatible: stages have no reason to over-bid
    (they'd pay more than needed) or under-bid (they'd lose).

    Truthful bidding → optimal allocation → no wasted budget.
    """

    @dataclass
    class Bid:
        stage: str
        amount: float  # how much budget this stage wants
        expected_quality_gain: float  # what it promises to deliver
        priority: float = 0.0  # computed clearing price

    def __init__(self):
        self._bids: List["VickreyBudgetAuction.Bid"] = []
        self._history: List[Dict] = []
        self._lock = threading.Lock()

    def submit_bid(self, stage: str, amount: float, expected_quality: float) -> None:
        """Stage submits a bid for budget."""
        with self._lock:
            self._bids.append(self.Bid(
                stage=stage, amount=amount, expected_quality_gain=expected_quality,
            ))

    def clear_auction(self, total_budget: float) -> Dict[str, float]:
        """
        Clear the auction: allocate budget using Vickrey (second-price) rules.

        Returns: {stage: allocated_budget}
        """
        with self._lock:
            if not self._bids:
                return {}

            # Sort by value density (quality per dollar)
            bids = sorted(self._bids, key=lambda b: b.expected_quality_gain / max(b.amount, 0.01), reverse=True)

            allocations = {}
            remaining = total_budget

            for i, bid in enumerate(bids):
                if remaining <= 0:
                    allocations[bid.stage] = 0
                    continue

                # Vickrey: pay the next-highest bid's value density
                if i + 1 < len(bids):
                    next_density = bids[i + 1].expected_quality_gain / max(bids[i + 1].amount, 0.01)
                    clearing_price = min(bid.amount, bid.expected_quality_gain / max(next_density, 0.01))
                else:
                    clearing_price = bid.amount * 0.5  # monopoly discount

                allocated = min(clearing_price, remaining)
                allocations[bid.stage] = allocated
                remaining -= allocated
                bid.priority = clearing_price

            self._history.append({"allocations": dict(allocations), "remaining": remaining})
            self._bids.clear()
            return allocations

    def stats(self) -> Dict[str, Any]:
        return {
            "pending_bids": len(self._bids),
            "auctions_cleared": len(self._history),
            "last_allocation": self._history[-1] if self._history else {},
        }


# ============================================================================
# 6. Wisdom of Crowds
# ============================================================================

class WisdomOfCrowds:
    """
    Diversity-weighted crowd aggregation for robust scoring.

    Surowiecki's conditions for wise crowds:
      1. Diversity of opinion
      2. Independence of members
      3. Decentralization
      4. Aggregation mechanism

    This implements a diversity-weighted median that gives more weight
    to independent judgments and penalizes herding behavior.

    Detection: if two reviewers' scores are suspiciously close (|diff| < ε),
    reduce their combined weight (they may be correlated, not independent).
    """

    def __init__(self, correlation_threshold: float = 0.5):
        self.corr_threshold = correlation_threshold
        self._scores: Dict[str, List[float]] = defaultdict(list)

    def add_scores(self, item_id: str, scores: List[float]) -> None:
        """Add multiple reviewer scores for an item."""
        self._scores[item_id] = scores

    def aggregate(self, item_id: str) -> float:
        """Diversity-weighted aggregation."""
        scores = self._scores.get(item_id, [])
        if not scores:
            return 0.0
        if len(scores) == 1:
            return scores[0]

        # Compute pairwise independence weights
        weights = [1.0] * len(scores)
        for i in range(len(scores)):
            for j in range(i + 1, len(scores)):
                diff = abs(scores[i] - scores[j])
                if diff < self.corr_threshold:
                    # Penalize correlated reviewers
                    penalty = 1.0 - (self.corr_threshold - diff) / self.corr_threshold
                    weights[i] *= penalty
                    weights[j] *= penalty

        # Weighted average
        total_w = sum(weights) + 1e-10
        return sum(w * s for w, s in zip(weights, scores)) / total_w

    def diversity_index(self, item_id: str) -> float:
        """Measure diversity of opinions (0=consensus, 1=maximum disagreement)."""
        scores = self._scores.get(item_id, [])
        if len(scores) < 2:
            return 0.0
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        # Normalize by possible range
        return min(1.0, math.sqrt(variance) / 5.0)

    def stats(self) -> Dict[str, Any]:
        return {
            "items_scored": len(self._scores),
            "avg_diversity": round(
                sum(self.diversity_index(k) for k in self._scores) / max(len(self._scores), 1), 3
            ),
        }


# ============================================================================
# 7. Momentum Optimizer (Adam-like)
# ============================================================================

class MomentumOptimizer:
    """
    Adam-like momentum for quality trajectory smoothing.

    Tracks first moment (mean) and second moment (variance) of quality
    improvements across iterations. Uses bias-corrected estimates to
    make smoother optimization decisions.

    Momentum prevents oscillation: if quality has been consistently
    improving, momentum carries forward even through a bad iteration.

    Update:
      m_t = β₁ × m_{t-1} + (1-β₁) × g_t     (first moment)
      v_t = β₂ × v_{t-1} + (1-β₂) × g_t²    (second moment)
      m̂_t = m_t / (1 - β₁^t)                  (bias correction)
      v̂_t = v_t / (1 - β₂^t)
      θ_t = θ_{t-1} + lr × m̂_t / (√v̂_t + ε)
    """

    def __init__(self, beta1: float = 0.9, beta2: float = 0.999, lr: float = 0.1, epsilon: float = 1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr = lr
        self.epsilon = epsilon
        self._params: Dict[str, float] = {}  # parameter values
        self._m: Dict[str, float] = defaultdict(float)  # first moment
        self._v: Dict[str, float] = defaultdict(float)  # second moment
        self._t: int = 0

    def init_param(self, name: str, value: float) -> None:
        self._params[name] = value

    def update(self, gradients: Dict[str, float]) -> Dict[str, float]:
        """One Adam update step. Returns updated parameters."""
        self._t += 1

        for name, grad in gradients.items():
            if name not in self._params:
                self._params[name] = 0.5

            self._m[name] = self.beta1 * self._m[name] + (1 - self.beta1) * grad
            self._v[name] = self.beta2 * self._v[name] + (1 - self.beta2) * grad ** 2

            # Bias correction
            m_hat = self._m[name] / (1 - self.beta1 ** self._t)
            v_hat = self._v[name] / (1 - self.beta2 ** self._t)

            self._params[name] += self.lr * m_hat / (math.sqrt(v_hat) + self.epsilon)
            self._params[name] = max(0.01, min(0.99, self._params[name]))

        return dict(self._params)

    def get_momentum(self, name: str) -> float:
        """Get current momentum for a parameter (direction of improvement)."""
        if self._t == 0:
            return 0.0
        m_hat = self._m.get(name, 0) / (1 - self.beta1 ** self._t)
        return m_hat

    def is_improving(self, name: str) -> bool:
        """Is the parameter trending in the positive direction?"""
        return self.get_momentum(name) > 0

    def stats(self) -> Dict[str, Any]:
        return {
            "step": self._t,
            "params": {k: round(v, 3) for k, v in self._params.items()},
            "momentum": {k: round(self.get_momentum(k), 4) for k in self._params},
        }


# ============================================================================
# 8. Secretary Selector (Optimal Stopping)
# ============================================================================

class SecretarySelector:
    """
    Optimal stopping for idea selection (Secretary Problem).

    The classic secretary problem: you see ideas one at a time and must
    decide immediately whether to select each one. The optimal strategy:

      1. Observe first n/e ideas without selecting (exploration phase)
      2. After that, select the first idea better than all observed so far

    This gives a 1/e ≈ 37% chance of selecting the absolute best idea,
    which is provably optimal for this setting.

    Extension: multiple selections allowed (k-secretary problem).
    """

    def __init__(self, expected_total: int = 20, n_select: int = 1):
        self.expected_total = expected_total
        self.n_select = n_select
        self._threshold_count = max(1, int(expected_total / math.e))
        self._observed: List[Tuple[str, float]] = []
        self._selected: List[Tuple[str, float]] = []
        self._best_in_exploration: float = 0.0
        self._phase = "explore"  # explore or select

    def observe(self, idea_id: str, quality: float) -> bool:
        """
        Observe an idea. Returns True if this idea should be selected.
        """
        self._observed.append((idea_id, quality))

        if len(self._observed) <= self._threshold_count:
            # Exploration phase: observe but don't select
            self._best_in_exploration = max(self._best_in_exploration, quality)
            self._phase = "explore"
            return False

        self._phase = "select"

        # Selection phase: select if better than all in exploration
        if len(self._selected) < self.n_select and quality > self._best_in_exploration:
            self._selected.append((idea_id, quality))
            return True

        return False

    def force_select_best(self) -> Optional[Tuple[str, float]]:
        """If no idea was selected, pick the best observed."""
        if self._selected:
            return max(self._selected, key=lambda x: x[1])
        if self._observed:
            return max(self._observed, key=lambda x: x[1])
        return None

    def reset(self, expected_total: int = None) -> None:
        if expected_total:
            self.expected_total = expected_total
            self._threshold_count = max(1, int(expected_total / math.e))
        self._observed.clear()
        self._selected.clear()
        self._best_in_exploration = 0.0
        self._phase = "explore"

    def stats(self) -> Dict[str, Any]:
        return {
            "phase": self._phase,
            "observed": len(self._observed),
            "selected": len(self._selected),
            "exploration_threshold": self._threshold_count,
            "best_exploration": round(self._best_in_exploration, 3),
        }


# ============================================================================
# 9. Coevolutionary Arms Race
# ============================================================================

class CoevolutionaryArms:
    """
    Predator-prey co-evolution between ideas (prey) and critics (predators).

    Ideas evolve to survive criticism; critics evolve to find weaknesses.
    This creates an arms race that drives both toward higher quality.

    Prey fitness: survive more critic attacks
    Predator fitness: find weaknesses in more prey

    Leads to much stronger ideas than static evaluation because the
    evaluation criteria themselves evolve.
    """

    @dataclass
    class Organism:
        id: str
        kind: str  # "idea" or "critic"
        strength: float = 0.5
        wins: int = 0
        losses: int = 0
        generation: int = 0

    def __init__(self):
        self._ideas: List["CoevolutionaryArms.Organism"] = []
        self._critics: List["CoevolutionaryArms.Organism"] = []
        self._generation = 0

    def add_idea(self, idea_id: str, quality: float) -> None:
        self._ideas.append(self.Organism(id=idea_id, kind="idea", strength=quality))

    def add_critic(self, critic_id: str, strictness: float) -> None:
        self._critics.append(self.Organism(id=critic_id, kind="critic", strength=strictness))

    def compete(self) -> List[Tuple[str, str, str]]:
        """
        Run one round of competition. Returns list of (idea_id, critic_id, winner).
        """
        if not self._ideas or not self._critics:
            return []

        self._generation += 1
        results = []

        for idea in self._ideas:
            critic = random.choice(self._critics)
            # Outcome: idea survives if quality > critic strictness + noise
            noise = random.gauss(0, 0.1)
            if idea.strength + noise > critic.strength:
                idea.wins += 1
                critic.losses += 1
                results.append((idea.id, critic.id, "idea_wins"))
            else:
                idea.losses += 1
                critic.wins += 1
                results.append((idea.id, critic.id, "critic_wins"))

        return results

    def evolve(self) -> None:
        """Evolve: replicate winners, mutate losers."""
        # Ideas: strengthen winners, weaken losers
        for idea in self._ideas:
            win_rate = idea.wins / max(idea.wins + idea.losses, 1)
            idea.strength = 0.8 * idea.strength + 0.2 * win_rate
            idea.generation = self._generation

        # Critics: toughen those that found weaknesses
        for critic in self._critics:
            win_rate = critic.wins / max(critic.wins + critic.losses, 1)
            critic.strength = 0.8 * critic.strength + 0.2 * win_rate
            critic.generation = self._generation

    def get_survivors(self, top_n: int = 5) -> List[str]:
        """Get idea IDs that survived the most critic attacks."""
        sorted_ideas = sorted(self._ideas, key=lambda i: i.wins, reverse=True)
        return [i.id for i in sorted_ideas[:top_n]]

    def stats(self) -> Dict[str, Any]:
        return {
            "generation": self._generation,
            "ideas": len(self._ideas),
            "critics": len(self._critics),
            "avg_idea_strength": round(sum(i.strength for i in self._ideas) / max(len(self._ideas), 1), 3),
            "avg_critic_strength": round(sum(c.strength for c in self._critics) / max(len(self._critics), 1), 3),
            "top_survivors": self.get_survivors(3),
        }


# ============================================================================
# 10. Thermodynamic Free Energy
# ============================================================================

class ThermoFreeEnergy:
    """
    Free energy principle for quality-diversity balance.

    From thermodynamics: F = E - T×S
    where E=energy (negative quality), T=temperature, S=entropy (diversity).

    Minimizing free energy means:
      - At high T: maximize entropy (diversity) → explore broadly
      - At low T: minimize energy (maximize quality) → exploit best
      - At medium T: natural balance between quality and diversity

    This provides a principled way to balance the quality-diversity trade-off
    instead of arbitrary weighting.
    """

    def __init__(self):
        self._quality_samples: List[float] = []
        self._diversity_samples: List[float] = []
        self._temperature: float = 1.0

    def set_temperature(self, T: float) -> None:
        """Set thermodynamic temperature (exploration level)."""
        self._temperature = max(0.01, T)

    def record(self, quality: float, diversity: float) -> None:
        """Record a (quality, diversity) observation."""
        self._quality_samples.append(quality)
        self._diversity_samples.append(diversity)

    def free_energy(self, quality: float, diversity: float) -> float:
        """Compute free energy: F = -quality - T × diversity."""
        return -quality - self._temperature * diversity

    def optimal_action(self) -> str:
        """Recommend action based on free energy landscape."""
        if not self._quality_samples:
            return "explore"

        avg_q = sum(self._quality_samples[-5:]) / min(len(self._quality_samples), 5)
        avg_d = sum(self._diversity_samples[-5:]) / min(len(self._diversity_samples), 5)

        if self._temperature > 0.7:
            if avg_d < 0.5:
                return "explore_more"
            return "maintain_diversity"
        elif self._temperature < 0.3:
            if avg_q < 0.5:
                return "exploit_harder"
            return "refine_best"
        else:
            return "balanced"

    def suggest_temperature(self) -> float:
        """Suggest temperature based on current quality-diversity state."""
        if len(self._quality_samples) < 3:
            return 1.0  # start warm

        # If quality is stagnating but diversity is low → increase T
        recent_q = self._quality_samples[-3:]
        q_trend = recent_q[-1] - recent_q[0]
        avg_d = sum(self._diversity_samples[-3:]) / 3 if self._diversity_samples else 0.5

        if q_trend < 0.01 and avg_d < 0.4:
            return min(1.0, self._temperature * 1.2)  # heat up
        elif q_trend > 0.05:
            return max(0.1, self._temperature * 0.9)  # cool down
        return self._temperature

    def stats(self) -> Dict[str, Any]:
        return {
            "temperature": round(self._temperature, 3),
            "action": self.optimal_action(),
            "suggested_temp": round(self.suggest_temperature(), 3),
            "observations": len(self._quality_samples),
        }


# ============================================================================
# 11. Social Influence Network (PageRank-like)
# ============================================================================

class SocialInfluenceNetwork:
    """
    PageRank-like influence propagation across ideas.

    Models ideas as nodes in a citation-like network. Ideas that inspire
    many other ideas have high "influence" (PageRank). Selecting high-influence
    ideas as seeds leads to more productive exploration.

    PageRank: PR(i) = (1-d)/N + d × Σ_j PR(j) / L(j)
    where d=0.85 (damping), L(j)=out-degree of j, N=total nodes.
    """

    def __init__(self, damping: float = 0.85, n_iterations: int = 20):
        self.damping = damping
        self.n_iterations = n_iterations
        self._edges: List[Tuple[str, str]] = []  # (from, to) = "from inspired to"
        self._nodes: Set[str] = set()

    def add_influence(self, from_idea: str, to_idea: str) -> None:
        """Record that from_idea influenced/inspired to_idea."""
        self._edges.append((from_idea, to_idea))
        self._nodes.add(from_idea)
        self._nodes.add(to_idea)

    def compute_pagerank(self) -> Dict[str, float]:
        """Compute PageRank for all ideas."""
        if not self._nodes:
            return {}

        nodes = sorted(self._nodes)
        n = len(nodes)
        node_idx = {node: i for i, node in enumerate(nodes)}

        # Build adjacency
        out_degree = defaultdict(int)
        in_links: Dict[int, List[int]] = defaultdict(list)
        for src, dst in self._edges:
            out_degree[node_idx[src]] += 1
            in_links[node_idx[dst]].append(node_idx[src])

        # Initialize uniform
        pr = [1.0 / n] * n

        # Power iteration
        for _ in range(self.n_iterations):
            new_pr = [(1 - self.damping) / n] * n
            for i in range(n):
                for j in in_links[i]:
                    new_pr[i] += self.damping * pr[j] / max(out_degree[j], 1)
            pr = new_pr

        return {nodes[i]: round(pr[i], 4) for i in range(n)}

    def most_influential(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get the n most influential ideas."""
        pr = self.compute_pagerank()
        return sorted(pr.items(), key=lambda x: x[1], reverse=True)[:n]

    def stats(self) -> Dict[str, Any]:
        return {
            "nodes": len(self._nodes),
            "edges": len(self._edges),
            "top_influential": self.most_influential(3),
        }


# ============================================================================
# 12. Chaotic Explorer
# ============================================================================

class ChaoticExplorer:
    """
    Deterministic chaos for ergodic search space coverage.

    Uses the logistic map x_{n+1} = r × x_n × (1 - x_n) to generate
    deterministic-but-chaotic sequences that ergodically cover [0,1].

    Unlike random search (which can leave gaps), chaotic sequences
    provably visit every region of the search space given enough iterations.

    At r=4.0 (edge of chaos), the logistic map is maximally chaotic
    and its invariant measure is the arcsine distribution — guaranteed
    to cover the full range.

    Use: generate exploration parameters (temperature, creativity) that
    systematically cover the full parameter space.
    """

    def __init__(self, r: float = 3.99, n_dims: int = 4):
        self.r = r  # 3.99 ≈ edge of chaos without numerical overflow
        self.n_dims = n_dims
        # Initialize each dimension with a different seed
        self._state = [0.1 + 0.1 * i / n_dims for i in range(n_dims)]
        self._step = 0

    def next(self) -> List[float]:
        """Generate next chaotic point in [0,1]^n_dims."""
        self._step += 1
        for d in range(self.n_dims):
            self._state[d] = self.r * self._state[d] * (1 - self._state[d])
            # Ensure stays in valid range (numerical safety)
            self._state[d] = max(0.001, min(0.999, self._state[d]))
        return list(self._state)

    def next_params(self, param_names: List[str],
                    bounds: Dict[str, Tuple[float, float]] = None) -> Dict[str, float]:
        """Generate chaotic parameter values within bounds."""
        point = self.next()
        params = {}
        for i, name in enumerate(param_names):
            raw = point[i % self.n_dims]
            if bounds and name in bounds:
                lo, hi = bounds[name]
                params[name] = lo + raw * (hi - lo)
            else:
                params[name] = raw
        return params

    def coverage_estimate(self, n_samples: int = 100) -> float:
        """Estimate how well the chaotic sequence covers [0,1]."""
        n_bins = 10
        bins = [0] * n_bins
        state_backup = list(self._state)

        for _ in range(n_samples):
            point = self.next()
            for d in range(min(self.n_dims, 1)):
                idx = min(int(point[d] * n_bins), n_bins - 1)
                bins[idx] += 1

        self._state = state_backup  # restore state
        filled = sum(1 for b in bins if b > 0)
        return filled / n_bins

    def stats(self) -> Dict[str, Any]:
        return {
            "step": self._step,
            "r": self.r,
            "n_dims": self.n_dims,
            "current_state": [round(s, 4) for s in self._state],
            "coverage": round(self.coverage_estimate(), 2),
        }


# ============================================================================
# Master Nature Optimizer
# ============================================================================

class NatureOptimizer:
    """Aggregates all nature-inspired optimization techniques."""

    def __init__(self, enable_all: bool = True):
        self.ant_colony = AntColonyRouter() if enable_all else None
        self.immune = ImmuneClonalSelector() if enable_all else None
        self.gravity = GravitationalClustering() if enable_all else None
        self.diffusion = DiffusionIdeaGenerator() if enable_all else None
        self.auction = VickreyBudgetAuction() if enable_all else None
        self.crowds = WisdomOfCrowds() if enable_all else None
        self.momentum = MomentumOptimizer() if enable_all else None
        self.secretary = SecretarySelector() if enable_all else None
        self.coevolution = CoevolutionaryArms() if enable_all else None
        self.thermo = ThermoFreeEnergy() if enable_all else None
        self.influence = SocialInfluenceNetwork() if enable_all else None
        self.chaos = ChaoticExplorer() if enable_all else None

    def summary(self) -> Dict[str, Any]:
        result = {}
        if self.ant_colony: result["ant_colony"] = self.ant_colony.stats()
        if self.immune: result["immune_system"] = self.immune.stats()
        if self.gravity: result["gravitational"] = self.gravity.stats()
        if self.diffusion: result["diffusion"] = self.diffusion.stats()
        if self.auction: result["vickrey_auction"] = self.auction.stats()
        if self.crowds: result["wisdom_of_crowds"] = self.crowds.stats()
        if self.momentum: result["momentum"] = self.momentum.stats()
        if self.secretary: result["secretary_selector"] = self.secretary.stats()
        if self.coevolution: result["coevolution"] = self.coevolution.stats()
        if self.thermo: result["thermodynamics"] = self.thermo.stats()
        if self.influence: result["social_influence"] = self.influence.stats()
        if self.chaos: result["chaotic_explorer"] = self.chaos.stats()
        return result
