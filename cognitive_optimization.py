"""
cognitive_optimization.py - Cognitive science & advanced mathematical
optimization for IdeaGraph.

Layer 8: Techniques from cognitive architecture, linguistic analysis,
game theory (Shapley values), geometric optimization, robustness
engineering, and temporal scheduling. These model how intelligent
systems think, communicate, and allocate attention.

  1.  EpisodicMemoryBank      — Recall specific past runs for context
  2.  SemanticMemoryGraph      — Generalized knowledge across all runs
  3.  ShapleyContributor       — Fair attribution of each stage's contribution
  4.  ChainOfThoughtOptimizer  — Optimize reasoning chain structure
  5.  ManifoldExplorer         — Low-dimensional idea space navigation
  6.  ChaosEngineer            — Random fault injection for robustness testing
  7.  CanaryDeployer           — Shadow-test new optimizations safely
  8.  PriorityAgingQueue       — Priority queue with age-based promotion
  9.  TimeBoxedExecutor        — Hard time budgets per stage with preemption
  10. LinguisticComplexity     — Measure and target prompt readability
  11. PromptAlgebra            — Compositional prompt construction operators
  12. GeodeticIdeaDistance     — Manifold-aware idea similarity metric
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import random
import re
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


# ============================================================================
# 1. Episodic Memory Bank
# ============================================================================

class EpisodicMemoryBank:
    """
    Recall specific past pipeline runs for context.

    Episodic memory stores specific experiences (not generalizations):
      - What topic was explored
      - What strategies worked/failed
      - What reviewer feedback was given
      - What the final quality was

    When starting a new run, retrieves the most relevant past episodes
    to seed context and avoid repeating mistakes.
    """

    @dataclass
    class Episode:
        topic: str
        timestamp: float
        strategies_used: List[str]
        quality: float
        reviewer_feedback: List[str]
        key_decisions: List[str]
        outcome: str  # "accept", "reject", "error"
        lessons: List[str]

    def __init__(self, persist_path: str = None, max_episodes: int = 100):
        self.persist_path = persist_path or str(
            Path(__file__).parent / "output" / "episodic_memory.json"
        )
        self.max_episodes = max_episodes
        self._episodes: List["EpisodicMemoryBank.Episode"] = []
        self._load()

    def record(self, topic: str, strategies: List[str], quality: float,
               feedback: List[str], decisions: List[str], outcome: str,
               lessons: List[str] = None) -> None:
        """Store a complete episode from a pipeline run."""
        self._episodes.append(self.Episode(
            topic=topic, timestamp=time.time(),
            strategies_used=strategies, quality=quality,
            reviewer_feedback=feedback[:5], key_decisions=decisions[:5],
            outcome=outcome, lessons=lessons or [],
        ))
        if len(self._episodes) > self.max_episodes:
            self._episodes = self._episodes[-self.max_episodes:]
        self._save()

    def recall(self, topic: str, n: int = 3) -> List["EpisodicMemoryBank.Episode"]:
        """Retrieve most relevant past episodes for a topic."""
        if not self._episodes:
            return []

        # Score by topic word overlap + recency
        topic_words = set(topic.lower().split())
        scored = []
        now = time.time()
        for ep in self._episodes:
            ep_words = set(ep.topic.lower().split())
            overlap = len(topic_words & ep_words) / max(len(topic_words | ep_words), 1)
            recency = math.exp(-(now - ep.timestamp) / (86400 * 30))  # 30-day decay
            score = 0.7 * overlap + 0.3 * recency
            scored.append((score, ep))

        scored.sort(reverse=True)
        return [ep for _, ep in scored[:n]]

    def recall_lessons(self, topic: str) -> List[str]:
        """Get accumulated lessons from similar past runs."""
        episodes = self.recall(topic, n=5)
        lessons = []
        for ep in episodes:
            lessons.extend(ep.lessons)
        return list(set(lessons))[:10]

    def _save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
            data = [{
                "topic": e.topic, "timestamp": e.timestamp,
                "strategies": e.strategies_used, "quality": e.quality,
                "feedback": e.reviewer_feedback, "decisions": e.key_decisions,
                "outcome": e.outcome, "lessons": e.lessons,
            } for e in self._episodes]
            with open(self.persist_path, "w") as f:
                json.dump(data, f)
        except Exception:
            pass

    def _load(self) -> None:
        try:
            if os.path.exists(self.persist_path):
                with open(self.persist_path) as f:
                    data = json.load(f)
                self._episodes = [self.Episode(**d) for d in data]
        except Exception:
            pass

    def stats(self) -> Dict[str, Any]:
        outcomes = defaultdict(int)
        for e in self._episodes:
            outcomes[e.outcome] += 1
        return {
            "total_episodes": len(self._episodes),
            "outcomes": dict(outcomes),
            "avg_quality": round(sum(e.quality for e in self._episodes) / max(len(self._episodes), 1), 3),
            "total_lessons": sum(len(e.lessons) for e in self._episodes),
        }


# ============================================================================
# 2. Semantic Memory Graph
# ============================================================================

class SemanticMemoryGraph:
    """
    Generalized knowledge graph across all pipeline runs.

    Unlike episodic memory (specific events), semantic memory stores
    generalized facts and relationships:
      - "Transformers work well for NLP tasks"
      - "Batch size > 64 causes OOM on small GPUs"
      - "Cross-domain bridging produces more novel ideas"

    Organized as a weighted graph: concept → concept with relationship type.
    Strengthened by repetition (Hebbian learning: neurons that fire together
    wire together).
    """

    @dataclass
    class Relation:
        source: str
        target: str
        relation_type: str  # "causes", "improves", "conflicts_with", "requires"
        strength: float = 1.0
        evidence_count: int = 1

    def __init__(self):
        self._relations: Dict[Tuple[str, str], "SemanticMemoryGraph.Relation"] = {}
        self._concept_frequency: Dict[str, int] = defaultdict(int)
        # Inverted index: concept (lowercased) → set of (src, tgt) keys it
        # appears in (as either source or target). Maintained incrementally
        # on add/remove so queries become O(C) substring scan over UNIQUE
        # concepts + O(matches) collection — the prior code did 2 substring
        # checks per relation, scaling badly with thousands of relations.
        self._concept_to_keys: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
        self._lock = threading.Lock()

    def add_relation(self, source: str, target: str, relation_type: str, strength: float = 1.0) -> None:
        """Add or strengthen a semantic relation (Hebbian learning)."""
        with self._lock:
            src_l = source.lower()
            tgt_l = target.lower()
            key = (src_l, tgt_l)
            if key in self._relations:
                rel = self._relations[key]
                # Hebbian: strengthen with repetition
                rel.strength = min(10.0, rel.strength + strength * 0.3)
                rel.evidence_count += 1
            else:
                self._relations[key] = self.Relation(
                    source=src_l, target=tgt_l,
                    relation_type=relation_type, strength=strength,
                )
                # Maintain inverted index only on first insert (key reused on update)
                self._concept_to_keys[src_l].add(key)
                self._concept_to_keys[tgt_l].add(key)
            self._concept_frequency[src_l] += 1
            self._concept_frequency[tgt_l] += 1

    def query(self, concept: str, relation_type: str = None, n: int = 5) -> List["SemanticMemoryGraph.Relation"]:
        """Find relations involving a concept (substring match, top-n by strength)."""
        with self._lock:
            concept_lower = concept.lower()
            # Exact-match: O(1) inverted-index lookup. Most queries use full
            # concept names (see get_context), so this is the common case.
            keys = set(self._concept_to_keys.get(concept_lower, ()))
            # Substring match: scan UNIQUE concept names (typically a tiny
            # fraction of total relations) and union their keys. Self-key
            # already added above.
            for ck in self._concept_to_keys:
                if ck != concept_lower and concept_lower in ck:
                    keys.update(self._concept_to_keys[ck])

            # Filter by relation_type and rank by strength.
            relations = self._relations
            if relation_type is None:
                rels = [relations[k] for k in keys if k in relations]
            else:
                rels = [
                    r for r in (relations.get(k) for k in keys)
                    if r is not None and r.relation_type == relation_type
                ]
            # heapq.nlargest is O(M log n) — beats O(M log M) full sort + slice
            # when n ≪ M, which it almost always is (n=5 default).
            return heapq.nlargest(n, rels, key=lambda r: r.strength)

    def get_context(self, concepts: List[str], max_facts: int = 5) -> str:
        """Generate context string from semantic memory for given concepts."""
        facts = []
        for concept in concepts:
            rels = self.query(concept, n=3)
            for rel in rels:
                if rel.strength > 1.5:  # only strong facts
                    facts.append(f"{rel.source} {rel.relation_type} {rel.target} (strength={rel.strength:.1f})")
        return "\n".join(facts[:max_facts])

    def decay(self, factor: float = 0.99) -> None:
        """Apply memory decay (forgetting curve)."""
        with self._lock:
            to_remove = []
            for key, rel in self._relations.items():
                rel.strength *= factor
                if rel.strength < 0.1:
                    to_remove.append(key)
            for key in to_remove:
                del self._relations[key]
                # Keep inverted index consistent so query() doesn't return
                # ghost references.
                src_l, tgt_l = key
                src_keys = self._concept_to_keys.get(src_l)
                if src_keys is not None:
                    src_keys.discard(key)
                    if not src_keys:
                        del self._concept_to_keys[src_l]
                tgt_keys = self._concept_to_keys.get(tgt_l)
                if tgt_keys is not None and src_l != tgt_l:
                    tgt_keys.discard(key)
                    if not tgt_keys:
                        del self._concept_to_keys[tgt_l]

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "relations": len(self._relations),
                "concepts": len(self._concept_frequency),
                "strongest": sorted(
                    [(f"{r.source}→{r.target}", round(r.strength, 2))
                     for r in self._relations.values()],
                    key=lambda x: x[1], reverse=True,
                )[:5],
            }


# ============================================================================
# 3. Shapley Value Contributor
# ============================================================================

class ShapleyContributor:
    """
    Fair attribution of each pipeline stage's marginal contribution.

    Shapley values (from cooperative game theory) compute the average
    marginal contribution of each stage across all possible orderings.

    phi_i = Sum over S: |S|!(|N|-|S|-1)!/|N|! * [v(S+i) - v(S)]

    This is the ONLY attribution method that satisfies:
      - Efficiency: contributions sum to total value
      - Symmetry: equal contributors get equal credit
      - Null player: non-contributors get zero
      - Additivity: additive across games

    Uses Monte Carlo sampling (O(n×m) instead of O(2^n)) for tractability.
    """

    def __init__(self, n_samples: int = 100):
        self.n_samples = n_samples
        self._stages: List[str] = []
        self._value_fn_cache: Dict[frozenset, float] = {}
        self._shapley_values: Dict[str, float] = {}
        self._lock = threading.Lock()

    def set_stages(self, stages: List[str]) -> None:
        self._stages = stages

    def record_coalition_value(self, stages_active: Set[str], value: float) -> None:
        """Record the value achieved by a coalition of stages."""
        key = frozenset(stages_active)
        with self._lock:
            self._value_fn_cache[key] = value

    def _value(self, coalition: Set[str]) -> float:
        """Get value of a coalition (from cache or interpolation)."""
        key = frozenset(coalition)
        if key in self._value_fn_cache:
            return self._value_fn_cache[key]
        # Interpolate: assume additive model as fallback
        total = 0
        for stage in coalition:
            single_key = frozenset({stage})
            if single_key in self._value_fn_cache:
                total += self._value_fn_cache[single_key]
        return total * 0.8  # slight discount for assumed interactions

    def compute_shapley(self) -> Dict[str, float]:
        """Compute Shapley values via Monte Carlo sampling."""
        if not self._stages:
            return {}

        n = len(self._stages)
        shapley = {s: 0.0 for s in self._stages}

        for _ in range(self.n_samples):
            perm = list(self._stages)
            random.shuffle(perm)

            coalition = set()
            prev_value = 0.0

            for stage in perm:
                coalition.add(stage)
                new_value = self._value(coalition)
                marginal = new_value - prev_value
                shapley[stage] += marginal / self.n_samples
                prev_value = new_value

        with self._lock:
            self._shapley_values = shapley
        return shapley

    def get_attribution(self) -> List[Tuple[str, float]]:
        """Get stages ranked by Shapley value."""
        if not self._shapley_values:
            self.compute_shapley()
        return sorted(self._shapley_values.items(), key=lambda x: x[1], reverse=True)

    def most_valuable_stage(self) -> str:
        attr = self.get_attribution()
        return attr[0][0] if attr else ""

    def least_valuable_stage(self) -> str:
        attr = self.get_attribution()
        return attr[-1][0] if attr else ""

    def stats(self) -> Dict[str, Any]:
        return {
            "stages": len(self._stages),
            "cached_coalitions": len(self._value_fn_cache),
            "shapley_values": {k: round(v, 4) for k, v in self.get_attribution()},
        }


# ============================================================================
# 4. Chain-of-Thought Optimizer
# ============================================================================

class ChainOfThoughtOptimizer:
    """
    Optimize reasoning chain structure in LLM prompts.

    Different tasks benefit from different CoT structures:
      - Linear: A → B → C → answer (good for sequential reasoning)
      - Tree: branch at uncertainty points (good for exploration)
      - Graph: allow revisiting prior steps (good for complex problems)
      - None: direct answer (good for simple/factual tasks)

    Learns which CoT structure works best per task type by tracking
    quality outcomes.
    """

    COT_TEMPLATES = {
        "none": "",
        "linear": "\n\nThink step by step:\n1. First, analyze the key requirements\n2. Then, identify the approach\n3. Finally, produce the output",
        "tree": "\n\nConsider multiple approaches:\n- Approach A: [describe]\n- Approach B: [describe]\nSelect the best approach and explain why, then produce the output",
        "graph": "\n\nReason carefully:\n1. State your initial understanding\n2. Identify potential issues\n3. Revise your understanding if needed\n4. Verify consistency\n5. Produce the output",
        "socratic": "\n\nBefore answering, ask yourself:\n- What is the core question?\n- What assumptions am I making?\n- What could go wrong?\nThen provide your answer",
    }

    def __init__(self):
        self._outcomes: Dict[Tuple[str, str], List[float]] = defaultdict(list)  # (task_type, cot_style) → qualities
        self._lock = threading.Lock()

    def select_cot(self, task_type: str) -> str:
        """Select the best CoT style for a task type based on history."""
        with self._lock:
            best_style = "linear"  # default
            best_avg = 0.0

            for style in self.COT_TEMPLATES:
                outcomes = self._outcomes.get((task_type, style), [])
                if outcomes:
                    avg = sum(outcomes) / len(outcomes)
                    if avg > best_avg:
                        best_avg = avg
                        best_style = style

            # Exploration: 20% chance of trying a random style
            if random.random() < 0.2:
                best_style = random.choice(list(self.COT_TEMPLATES.keys()))

            return best_style

    def get_cot_suffix(self, task_type: str) -> str:
        """Get the CoT prompt suffix for a task."""
        style = self.select_cot(task_type)
        return self.COT_TEMPLATES.get(style, "")

    def record_outcome(self, task_type: str, cot_style: str, quality: float) -> None:
        with self._lock:
            self._outcomes[(task_type, cot_style)].append(quality)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            results = {}
            for (task, style), outcomes in self._outcomes.items():
                if task not in results:
                    results[task] = {}
                results[task][style] = {
                    "avg": round(sum(outcomes) / len(outcomes), 3),
                    "n": len(outcomes),
                }
            return {"task_cot_performance": results}


# ============================================================================
# 5. Manifold Explorer
# ============================================================================

class ManifoldExplorer:
    """
    Low-dimensional idea space navigation via PCA-like projection.

    The idea space is high-dimensional but ideas likely lie on a
    low-dimensional manifold. This projects ideas to 2D/3D for:
      - Identifying unexplored regions
      - Measuring true diversity (manifold distance, not Euclidean)
      - Guiding exploration toward sparse regions

    Uses incremental PCA (no numpy dependency): tracks running mean
    and covariance, projects onto top-k principal components.
    """

    def __init__(self, input_dims: int = 5, output_dims: int = 2):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self._points: List[List[float]] = []
        self._mean: List[float] = [0.0] * input_dims
        self._n = 0

    def add_point(self, features: List[float]) -> None:
        """Add a point to the manifold."""
        padded = (features + [0.0] * self.input_dims)[:self.input_dims]
        self._points.append(padded)
        # Update running mean
        self._n += 1
        for d in range(self.input_dims):
            self._mean[d] += (padded[d] - self._mean[d]) / self._n

    def project(self, features: List[float]) -> List[float]:
        """Project a point to the low-dimensional manifold."""
        if len(self._points) < 3:
            return features[:self.output_dims]

        padded = (features + [0.0] * self.input_dims)[:self.input_dims]
        centered = [padded[d] - self._mean[d] for d in range(self.input_dims)]

        # Simple projection: use first output_dims dimensions of centered point
        # (Approximation of PCA without full SVD)
        return centered[:self.output_dims]

    def find_sparse_region(self) -> List[float]:
        """Find the sparsest region of the manifold for targeted exploration."""
        if len(self._points) < 3:
            return [random.random() for _ in range(self.input_dims)]

        # Grid-based sparsity: find grid cell with fewest points
        n_bins = 5
        projected = [self.project(p) for p in self._points]
        grid: Dict[Tuple, int] = defaultdict(int)

        for proj in projected:
            cell = tuple(min(int(p * n_bins), n_bins - 1) for p in proj if 0 <= p <= 1)
            if cell:
                grid[cell] += 1

        # Find empty or sparse cell
        all_cells = [(i, j) for i in range(n_bins) for j in range(n_bins)]
        sparse_cells = sorted(all_cells, key=lambda c: grid.get(c, 0))
        target_cell = sparse_cells[0]

        # Convert back to feature space (center of sparse cell)
        target = [(c + 0.5) / n_bins for c in target_cell]
        # Pad to input dims
        while len(target) < self.input_dims:
            target.append(0.5)
        return target

    def stats(self) -> Dict[str, Any]:
        return {
            "points": len(self._points),
            "input_dims": self.input_dims,
            "output_dims": self.output_dims,
        }


# ============================================================================
# 6. Chaos Engineer
# ============================================================================

class ChaosEngineer:
    """
    Random fault injection for pipeline robustness testing.

    Inspired by Netflix's Chaos Monkey. Randomly injects failures to
    verify the pipeline gracefully handles:
      - API timeouts
      - Malformed LLM responses
      - Missing data
      - Budget exhaustion mid-run
      - Stage failures

    Builds confidence that optimizations don't make the pipeline fragile.
    """

    FAULT_TYPES = [
        "timeout", "empty_response", "malformed_json", "budget_exceeded",
        "stage_crash", "slow_response", "truncated_output",
    ]

    def __init__(self, fault_probability: float = 0.05, enabled: bool = False):
        self.fault_probability = fault_probability
        self.enabled = enabled
        self._injected: List[Dict] = []
        self._recovered: int = 0
        self._failed: int = 0

    def should_inject(self) -> bool:
        """Should we inject a fault right now?"""
        return self.enabled and random.random() < self.fault_probability

    def inject(self, stage: str) -> Optional[Dict[str, Any]]:
        """Inject a random fault. Returns fault details or None."""
        if not self.should_inject():
            return None

        fault_type = random.choice(self.FAULT_TYPES)
        fault = {"stage": stage, "type": fault_type, "timestamp": time.time()}
        self._injected.append(fault)
        return fault

    def simulate_fault(self, fault_type: str) -> Any:
        """Simulate the effect of a fault type."""
        if fault_type == "timeout":
            time.sleep(0.1)  # brief simulated delay
            return ""
        elif fault_type == "empty_response":
            return ""
        elif fault_type == "malformed_json":
            return "{invalid json here"
        elif fault_type == "truncated_output":
            return "This response was trunca"
        elif fault_type == "budget_exceeded":
            return ""
        return None

    def record_recovery(self, success: bool) -> None:
        if success:
            self._recovered += 1
        else:
            self._failed += 1

    @property
    def recovery_rate(self) -> float:
        total = self._recovered + self._failed
        return self._recovered / max(total, 1)

    def stats(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "fault_probability": self.fault_probability,
            "injected": len(self._injected),
            "recovered": self._recovered,
            "failed": self._failed,
            "recovery_rate": f"{self.recovery_rate:.1%}",
        }


# ============================================================================
# 7. Canary Deployer
# ============================================================================

class CanaryDeployer:
    """
    Shadow-test new optimizations safely before full deployment.

    Runs the new optimization alongside the old one on a fraction of
    calls. Compares results without affecting the main pipeline.
    Promotes to full deployment only when the canary outperforms.

    Canary stages:
      1. Shadow (5% traffic): run both, use old result, log new
      2. Canary (20% traffic): use new result for some calls
      3. Ramp (50% traffic): gradually increase new usage
      4. Full (100%): fully deployed
    """

    @dataclass
    class Canary:
        name: str
        stage: str  # "shadow", "canary", "ramp", "full", "rolled_back"
        traffic_pct: float
        old_results: List[float] = field(default_factory=list)
        new_results: List[float] = field(default_factory=list)
        created_at: float = field(default_factory=time.time)

    def __init__(self):
        self._canaries: Dict[str, "CanaryDeployer.Canary"] = {}
        self._lock = threading.Lock()

    def create_canary(self, name: str, initial_traffic: float = 0.05) -> None:
        with self._lock:
            self._canaries[name] = self.Canary(
                name=name, stage="shadow", traffic_pct=initial_traffic,
            )

    def should_use_new(self, name: str) -> bool:
        """Should this call use the new optimization?"""
        with self._lock:
            canary = self._canaries.get(name)
            if not canary or canary.stage == "rolled_back":
                return False
            return random.random() < canary.traffic_pct

    def record_result(self, name: str, is_new: bool, quality: float) -> None:
        with self._lock:
            canary = self._canaries.get(name)
            if not canary:
                return
            if is_new:
                canary.new_results.append(quality)
            else:
                canary.old_results.append(quality)

            # Auto-promote or rollback based on results
            self._evaluate_canary(canary)

    def _evaluate_canary(self, canary: "CanaryDeployer.Canary") -> None:
        """Evaluate canary and potentially promote or rollback."""
        min_samples = 5
        if len(canary.new_results) < min_samples or len(canary.old_results) < min_samples:
            return

        new_avg = sum(canary.new_results[-10:]) / min(len(canary.new_results), 10)
        old_avg = sum(canary.old_results[-10:]) / min(len(canary.old_results), 10)

        if new_avg < old_avg * 0.9:
            # New is significantly worse → rollback
            canary.stage = "rolled_back"
            canary.traffic_pct = 0.0
        elif new_avg >= old_avg:
            # New is at least as good → promote
            if canary.stage == "shadow":
                canary.stage = "canary"
                canary.traffic_pct = 0.2
            elif canary.stage == "canary":
                canary.stage = "ramp"
                canary.traffic_pct = 0.5
            elif canary.stage == "ramp":
                canary.stage = "full"
                canary.traffic_pct = 1.0

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                name: {
                    "stage": c.stage, "traffic": c.traffic_pct,
                    "new_avg": round(sum(c.new_results[-10:]) / max(len(c.new_results[-10:]), 1), 3),
                    "old_avg": round(sum(c.old_results[-10:]) / max(len(c.old_results[-10:]), 1), 3),
                }
                for name, c in self._canaries.items()
            }


# ============================================================================
# 8. Priority Aging Queue
# ============================================================================

class PriorityAgingQueue:
    """
    Priority queue with age-based promotion to prevent starvation.

    Standard priority queues can starve low-priority items forever.
    Aging: every tick, increase priority of waiting items by a small amount.
    Eventually, even the lowest-priority item gets promoted.

    Aging rate: priority += age_factor × time_waiting
    This guarantees bounded wait time for all items.
    """

    @dataclass
    class QueueItem:
        id: str
        data: Any
        base_priority: float
        enqueued_at: float = field(default_factory=time.time)
        age_bonus: float = 0.0

        @property
        def effective_priority(self) -> float:
            return self.base_priority + self.age_bonus

    def __init__(self, age_factor: float = 0.01):
        self.age_factor = age_factor
        self._queue: List["PriorityAgingQueue.QueueItem"] = []
        self._lock = threading.Lock()

    def enqueue(self, id: str, data: Any, priority: float) -> None:
        with self._lock:
            self._queue.append(self.QueueItem(id=id, data=data, base_priority=priority))

    def dequeue(self) -> Optional["PriorityAgingQueue.QueueItem"]:
        """Remove and return highest-priority item (with aging applied)."""
        with self._lock:
            if not self._queue:
                return None

            now = time.time()
            for item in self._queue:
                wait_time = now - item.enqueued_at
                item.age_bonus = self.age_factor * wait_time

            self._queue.sort(key=lambda x: x.effective_priority, reverse=True)
            return self._queue.pop(0)

    def peek(self) -> Optional["PriorityAgingQueue.QueueItem"]:
        with self._lock:
            if not self._queue:
                return None
            now = time.time()
            for item in self._queue:
                item.age_bonus = self.age_factor * (now - item.enqueued_at)
            return max(self._queue, key=lambda x: x.effective_priority)

    @property
    def size(self) -> int:
        return len(self._queue)

    def stats(self) -> Dict[str, Any]:
        return {
            "size": len(self._queue),
            "age_factor": self.age_factor,
        }


# ============================================================================
# 9. Time-Boxed Executor
# ============================================================================

class TimeBoxedExecutor:
    """
    Hard time budgets per pipeline stage with preemption.

    Each stage gets a time budget. If exceeded, the stage is interrupted
    and a fallback result is used. This prevents any single stage from
    consuming the entire pipeline runtime.

    Budget allocation: proportional to stage importance (from Shapley values
    or historical duration).
    """

    def __init__(self, total_time_s: float = 600):
        self.total_time = total_time_s
        self._budgets: Dict[str, float] = {}
        self._actual: Dict[str, float] = {}
        self._overruns: Dict[str, int] = defaultdict(int)

    DEFAULT_FRACTIONS = {
        "ideation": 0.30,
        "experiment_design": 0.10,
        "code_generation": 0.15,
        "execution": 0.20,
        "analysis": 0.08,
        "paper_writing": 0.10,
        "review": 0.07,
    }

    def allocate(self, stage_fractions: Dict[str, float] = None) -> Dict[str, float]:
        """Allocate time budgets. Returns {stage: seconds}."""
        fractions = stage_fractions or self.DEFAULT_FRACTIONS
        self._budgets = {s: self.total_time * f for s, f in fractions.items()}
        return dict(self._budgets)

    def get_budget(self, stage: str) -> float:
        """Get time budget for a stage in seconds."""
        return self._budgets.get(stage, self.total_time * 0.1)

    def record_actual(self, stage: str, duration_s: float) -> None:
        """Record actual stage duration."""
        self._actual[stage] = duration_s
        budget = self._budgets.get(stage, float('inf'))
        if duration_s > budget:
            self._overruns[stage] += 1

    def remaining_time(self) -> float:
        """Total remaining time across all stages."""
        used = sum(self._actual.values())
        return max(0, self.total_time - used)

    def should_abort(self, stage: str, elapsed_s: float) -> bool:
        """Should this stage be aborted due to time overrun?"""
        budget = self._budgets.get(stage, float('inf'))
        return elapsed_s > budget * 1.5  # 50% grace period

    def stats(self) -> Dict[str, Any]:
        return {
            "total_time": self.total_time,
            "budgets": {k: round(v, 1) for k, v in self._budgets.items()},
            "actual": {k: round(v, 1) for k, v in self._actual.items()},
            "overruns": dict(self._overruns),
            "remaining": round(self.remaining_time(), 1),
        }


# ============================================================================
# 10. Linguistic Complexity Analyzer
# ============================================================================

class LinguisticComplexity:
    """
    Measure and target prompt readability for optimal LLM comprehension.

    Research shows LLMs have a "sweet spot" of prompt complexity:
      - Too simple: underspecified, vague outputs
      - Too complex: confused, inconsistent outputs
      - Just right: clear, high-quality outputs

    Metrics:
      - Flesch-Kincaid grade level
      - Average sentence length
      - Vocabulary diversity (type-token ratio)
      - Instruction density (imperatives per sentence)
    """

    TARGET_RANGES = {
        "grade_level": (8, 14),     # high school to college
        "avg_sentence_len": (10, 25),  # words per sentence
        "vocab_diversity": (0.4, 0.8),  # type-token ratio
        "instruction_density": (0.2, 0.6),  # fraction of imperative sentences
    }

    def __init__(self):
        self._analyses: Dict[str, List[Dict]] = defaultdict(list)

    def analyze(self, text: str) -> Dict[str, float]:
        """Compute linguistic complexity metrics."""
        words = text.split()
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

        n_words = len(words)
        n_sentences = max(len(sentences), 1)
        n_syllables = sum(self._count_syllables(w) for w in words)

        avg_sentence_len = n_words / n_sentences
        avg_word_len = n_syllables / max(n_words, 1)

        # Flesch-Kincaid grade level
        grade = 0.39 * avg_sentence_len + 11.8 * avg_word_len - 15.59

        # Type-token ratio (vocabulary diversity)
        unique_words = set(w.lower() for w in words)
        ttr = len(unique_words) / max(n_words, 1)

        # Instruction density
        imperatives = ["must", "should", "return", "generate", "output", "ensure",
                       "provide", "create", "use", "include", "avoid", "do not"]
        imp_count = sum(1 for w in words if w.lower() in imperatives)
        inst_density = imp_count / max(n_words, 1) * 10  # scale up

        return {
            "grade_level": round(grade, 1),
            "avg_sentence_len": round(avg_sentence_len, 1),
            "vocab_diversity": round(ttr, 3),
            "instruction_density": round(inst_density, 3),
            "word_count": n_words,
        }

    def is_in_sweet_spot(self, metrics: Dict[str, float]) -> bool:
        """Check if metrics fall within the optimal range."""
        for metric, (lo, hi) in self.TARGET_RANGES.items():
            val = metrics.get(metric, 0)
            if val < lo or val > hi:
                return False
        return True

    def suggest_adjustment(self, metrics: Dict[str, float]) -> List[str]:
        """Suggest prompt adjustments to reach the sweet spot."""
        suggestions = []
        for metric, (lo, hi) in self.TARGET_RANGES.items():
            val = metrics.get(metric, 0)
            if val < lo:
                if metric == "grade_level":
                    suggestions.append("Increase complexity: add more technical detail")
                elif metric == "instruction_density":
                    suggestions.append("Add more explicit instructions")
            elif val > hi:
                if metric == "grade_level":
                    suggestions.append("Simplify language: use shorter sentences")
                elif metric == "avg_sentence_len":
                    suggestions.append("Break long sentences into shorter ones")
        return suggestions

    @staticmethod
    def _count_syllables(word: str) -> int:
        """Approximate syllable count."""
        word = word.lower().strip(".,!?;:")
        if len(word) <= 3:
            return 1
        vowels = "aeiouy"
        count = 0
        prev_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        if word.endswith("e"):
            count = max(1, count - 1)
        return max(1, count)

    def stats(self) -> Dict[str, Any]:
        return {"target_ranges": self.TARGET_RANGES}


# ============================================================================
# 11. Prompt Algebra
# ============================================================================

class PromptAlgebra:
    """
    Compositional prompt construction using algebraic operators.

    Prompts as first-class objects with operations:
      - compose(A, B): sequential composition (A then B)
      - parallel(A, B): combine independent instructions
      - condition(A, pred): apply A only if predicate is true
      - negate(A): reverse the instruction ("do X" → "do NOT X")
      - amplify(A, k): repeat/strengthen instruction k times
      - abstract(A): extract the core instruction pattern
    """

    @dataclass
    class PromptFragment:
        content: str
        weight: float = 1.0
        conditional: Optional[str] = None
        tags: List[str] = field(default_factory=list)

    def __init__(self):
        self._library: Dict[str, "PromptAlgebra.PromptFragment"] = {}

    def register(self, name: str, content: str, tags: List[str] = None) -> None:
        """Register a reusable prompt fragment."""
        self._library[name] = self.PromptFragment(content=content, tags=tags or [])

    def compose(self, *fragment_names: str) -> str:
        """Sequential composition: apply fragments in order."""
        parts = []
        for name in fragment_names:
            frag = self._library.get(name)
            if frag:
                if frag.conditional:
                    parts.append(f"If {frag.conditional}: {frag.content}")
                else:
                    parts.append(frag.content)
        return "\n\n".join(parts)

    def parallel(self, *fragment_names: str) -> str:
        """Parallel combination: merge as concurrent instructions."""
        parts = []
        for name in fragment_names:
            frag = self._library.get(name)
            if frag:
                parts.append(f"- {frag.content}")
        return "Simultaneously:\n" + "\n".join(parts)

    def amplify(self, fragment_name: str, strength: int = 2) -> str:
        """Strengthen an instruction by emphasis."""
        frag = self._library.get(fragment_name)
        if not frag:
            return ""
        emphasis = ["IMPORTANT: ", "CRITICAL: ", "ESSENTIAL: "][min(strength - 1, 2)]
        return emphasis + frag.content

    def negate(self, fragment_name: str) -> str:
        """Negate an instruction."""
        frag = self._library.get(fragment_name)
        if not frag:
            return ""
        content = frag.content
        # Simple negation rules (compute .lower() once)
        content_lower = content.lower()
        if content_lower.startswith("do "):
            return "Do NOT " + content[3:]
        if content_lower.startswith("use "):
            return "Do NOT use " + content[4:]
        if content_lower.startswith("generate "):
            return "Do NOT generate " + content[9:]
        return "Avoid: " + content

    def condition(self, fragment_name: str, predicate: str) -> str:
        """Apply fragment conditionally."""
        frag = self._library.get(fragment_name)
        if not frag:
            return ""
        return f"If {predicate}, then: {frag.content}"

    def build_prompt(self, recipe: List[Tuple[str, str]]) -> str:
        """
        Build a prompt from a recipe of (operation, fragment_name) pairs.

        Example: [("compose", "intro"), ("amplify", "key_rule"), ("negate", "bad_habit")]
        """
        parts = []
        for op, name in recipe:
            if op == "compose":
                parts.append(self.compose(name))
            elif op == "amplify":
                parts.append(self.amplify(name))
            elif op == "negate":
                parts.append(self.negate(name))
            elif op == "parallel" and "," in name:
                parts.append(self.parallel(*name.split(",")))
        return "\n\n".join(p for p in parts if p)

    def stats(self) -> Dict[str, Any]:
        return {
            "library_size": len(self._library),
            "fragments": list(self._library.keys()),
        }


# ============================================================================
# 12. Geodetic Idea Distance
# ============================================================================

class GeodeticIdeaDistance:
    """
    Manifold-aware idea similarity metric.

    Euclidean distance in feature space doesn't account for the curved
    manifold structure of the idea space. Geodesic distance follows the
    manifold surface, giving more meaningful similarity.

    Approximation via graph distance:
      1. Build k-NN graph of ideas
      2. Geodesic distance ≈ shortest path in k-NN graph (Dijkstra)
      3. Far more meaningful than Euclidean for curved spaces

    Use: better idea deduplication and diversity measurement.
    """

    def __init__(self, k_neighbors: int = 5):
        self.k = k_neighbors
        self._points: Dict[str, List[float]] = {}
        self._graph: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

    def add_idea(self, idea_id: str, features: List[float]) -> None:
        self._points[idea_id] = features

    def _euclidean(self, a: List[float], b: List[float]) -> float:
        return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)) + 1e-10)

    def build_graph(self) -> None:
        """Build k-NN graph for geodesic computation."""
        self._graph.clear()
        ids = list(self._points.keys())
        # Hoist into a parallel list so the inner loop avoids repeated dict
        # lookups (was self._points[id_j] per j × N times per i).
        points = [self._points[i] for i in ids]
        n = len(ids)
        k = self.k
        eucl = self._euclidean
        graph = self._graph

        for i in range(n):
            pi = points[i]
            # Build per-i distance list once, then take the k smallest with
            # heapq.nsmallest — O(N log k), beats the prior full sort which
            # is O(N log N). For typical k=5, N=100 that's ~4× faster.
            distances = [
                (eucl(pi, points[j]), ids[j]) for j in range(n) if j != i
            ]
            id_i = ids[i]
            for d, id_j in heapq.nsmallest(k, distances):
                graph[id_i].append((id_j, d))
                graph[id_j].append((id_i, d))

    def geodesic_distance(self, id_a: str, id_b: str) -> float:
        """Compute geodesic distance via Dijkstra on k-NN graph."""
        if id_a not in self._points or id_b not in self._points:
            return float('inf')

        if not self._graph:
            self.build_graph()

        # Dijkstra with heap-based priority queue. Prior version called
        # queue.sort() in the loop, making each pop O(Q log Q) instead of
        # the heap's O(log Q) — a quadratic blowup on large graphs.
        dist: Dict[str, float] = {id_a: 0.0}
        visited: Set[str] = set()
        heap: List[Tuple[float, str]] = [(0.0, id_a)]
        graph = self._graph

        while heap:
            d, current = heapq.heappop(heap)
            if current in visited:
                continue
            if current == id_b:
                return d
            visited.add(current)

            for neighbor, edge_dist in graph.get(current, ()):
                if neighbor in visited:
                    continue
                new_dist = d + edge_dist
                if new_dist < dist.get(neighbor, float('inf')):
                    dist[neighbor] = new_dist
                    heapq.heappush(heap, (new_dist, neighbor))

        # Disconnected: fall back to Euclidean
        return self._euclidean(self._points[id_a], self._points[id_b])

    def most_distant_pair(self) -> Tuple[str, str, float]:
        """Find the most diverse pair of ideas."""
        if not self._graph:
            self.build_graph()

        max_dist = 0
        best_pair = ("", "")
        ids = list(self._points.keys())[:20]  # cap for performance
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                d = self.geodesic_distance(a, b)
                if d > max_dist:
                    max_dist = d
                    best_pair = (a, b)
        return (best_pair[0], best_pair[1], max_dist)

    def stats(self) -> Dict[str, Any]:
        return {
            "ideas": len(self._points),
            "graph_edges": sum(len(v) for v in self._graph.values()) // 2,
            "k_neighbors": self.k,
        }


# ============================================================================
# Master Cognitive Optimizer
# ============================================================================

class CognitiveOptimizer:
    """Aggregates all cognitive/mathematical optimization techniques."""

    def __init__(self, enable_all: bool = True):
        self.episodic = EpisodicMemoryBank() if enable_all else None
        self.semantic = SemanticMemoryGraph() if enable_all else None
        self.shapley = ShapleyContributor() if enable_all else None
        self.cot = ChainOfThoughtOptimizer() if enable_all else None
        self.manifold = ManifoldExplorer() if enable_all else None
        self.chaos_eng = ChaosEngineer() if enable_all else None
        self.canary = CanaryDeployer() if enable_all else None
        self.priority_queue = PriorityAgingQueue() if enable_all else None
        self.time_box = TimeBoxedExecutor() if enable_all else None
        self.linguistic = LinguisticComplexity() if enable_all else None
        self.algebra = PromptAlgebra() if enable_all else None
        self.geodetic = GeodeticIdeaDistance() if enable_all else None

    def summary(self) -> Dict[str, Any]:
        result = {}
        if self.episodic: result["episodic_memory"] = self.episodic.stats()
        if self.semantic: result["semantic_memory"] = self.semantic.stats()
        if self.shapley: result["shapley_values"] = self.shapley.stats()
        if self.cot: result["chain_of_thought"] = self.cot.stats()
        if self.manifold: result["manifold_explorer"] = self.manifold.stats()
        if self.chaos_eng: result["chaos_engineering"] = self.chaos_eng.stats()
        if self.canary: result["canary_deployer"] = self.canary.stats()
        if self.priority_queue: result["priority_queue"] = self.priority_queue.stats()
        if self.time_box: result["time_boxed"] = self.time_box.stats()
        if self.linguistic: result["linguistic"] = self.linguistic.stats()
        if self.algebra: result["prompt_algebra"] = self.algebra.stats()
        if self.geodetic: result["geodetic_distance"] = self.geodetic.stats()
        return result
