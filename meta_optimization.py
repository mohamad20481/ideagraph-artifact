"""
meta_optimization.py - Cross-disciplinary meta-optimization for IdeaGraph.

Layer 5: Techniques drawn from signal processing, game theory, information
theory, compiler optimization, and control theory. These operate at a higher
abstraction level — optimizing the optimizers themselves.

  1.  KalmanQualityEstimator   — Kalman filter for robust quality prediction
  2.  MutualInfoStageRanker    — Information-theoretic stage importance ranking
  3.  NashReviewerConsensus    — Game-theoretic multi-reviewer aggregation
  4.  WorkingMemoryGate        — Attention-gated context passing between stages
  5.  DeadStageEliminator      — Compiler-style dead code elimination for pipeline
  6.  PredictiveQualityGate    — ML-based early abort before expensive stages
  7.  BloomDedup               — Probabilistic O(1) idea deduplication
  8.  ExponentialForecaster    — Holt-Winters forecasting for quality/cost trends
  9.  ABPromptTester           — Automated split testing for prompt variants
  10. CausalImpactAnalyzer     — Difference-in-differences for optimization impact
  11. ReservoirSampler         — Memory-efficient representative sampling
  12. FeedbackLoopDetector     — Detect and break negative feedback loops
"""

from __future__ import annotations

import hashlib
import math
import random
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ============================================================================
# 1. Kalman Filter Quality Estimator
# ============================================================================

class KalmanQualityEstimator:
    """
    Kalman filter for robust quality score estimation.

    LLM quality scores are noisy — the same idea evaluated twice may get
    different scores. A Kalman filter optimally fuses noisy observations
    into a smooth, accurate estimate.

    State: estimated true quality (x)
    Measurement: observed quality score (z)
    Process noise (Q): how much quality changes between evaluations
    Measurement noise (R): how noisy the LLM scorer is

    Update equations:
      Predict:  x̂ = x, P̂ = P + Q
      Update:   K = P̂/(P̂ + R), x = x̂ + K(z - x̂), P = (1-K)P̂
    """

    @dataclass
    class State:
        x: float = 0.5     # estimated quality
        P: float = 0.25    # estimation uncertainty
        history: List[float] = field(default_factory=list)

    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 0.1):
        self.Q = process_noise    # how much true quality drifts
        self.R = measurement_noise  # how noisy LLM scoring is
        self._states: Dict[str, "KalmanQualityEstimator.State"] = {}
        self._lock = threading.Lock()

    def update(self, key: str, observed_quality: float) -> float:
        """Update quality estimate with a new observation. Returns filtered estimate."""
        with self._lock:
            if key not in self._states:
                self._states[key] = self.State(x=observed_quality, P=self.R)
                self._states[key].history.append(observed_quality)
                return observed_quality

            s = self._states[key]
            # Predict
            x_pred = s.x
            P_pred = s.P + self.Q

            # Update (Kalman gain)
            K = P_pred / (P_pred + self.R)
            s.x = x_pred + K * (observed_quality - x_pred)
            s.P = (1 - K) * P_pred
            s.history.append(observed_quality)

            return s.x

    def get_estimate(self, key: str) -> Tuple[float, float]:
        """Get (estimated_quality, uncertainty) for a key."""
        with self._lock:
            s = self._states.get(key)
            if not s:
                return 0.5, 0.5
            return s.x, math.sqrt(s.P)

    def confidence_interval(self, key: str, z: float = 1.96) -> Tuple[float, float]:
        """95% confidence interval for quality estimate."""
        est, unc = self.get_estimate(key)
        return (max(0, est - z * unc), min(1, est + z * unc))

    def is_reliable(self, key: str, min_observations: int = 3) -> bool:
        """Is the estimate reliable (enough observations)?"""
        with self._lock:
            s = self._states.get(key)
            return s is not None and len(s.history) >= min_observations

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "tracked_keys": len(self._states),
                "estimates": {
                    k: {"quality": round(s.x, 3), "uncertainty": round(math.sqrt(s.P), 3), "observations": len(s.history)}
                    for k, s in list(self._states.items())[:10]
                },
            }


# ============================================================================
# 2. Mutual Information Stage Ranker
# ============================================================================

class MutualInfoStageRanker:
    """
    Information-theoretic ranking of pipeline stage importance.

    Measures how much information each stage contributes to the final
    review score using empirical mutual information:

      I(stage; review) ≈ H(review) - H(review | stage)

    Stages with high MI are essential; stages with low MI can be skipped
    or run in lite mode without quality loss.

    Uses binned histograms for entropy estimation (no parametric assumptions).
    """

    def __init__(self, n_bins: int = 5):
        self.n_bins = n_bins
        self._stage_scores: Dict[str, List[float]] = defaultdict(list)
        self._review_scores: List[float] = []
        self._joint: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

    def record(self, stage: str, stage_quality: float, review_score: float) -> None:
        """Record a (stage_quality, review_score) pair."""
        self._stage_scores[stage].append(stage_quality)
        self._review_scores.append(review_score)
        self._joint[stage].append((stage_quality, review_score))

    def _entropy(self, values: List[float]) -> float:
        """Compute entropy using binned histogram."""
        if len(values) < 3:
            return 0.0
        min_v, max_v = min(values), max(values)
        if max_v == min_v:
            return 0.0
        bin_width = (max_v - min_v) / self.n_bins
        counts = [0] * self.n_bins
        for v in values:
            idx = min(int((v - min_v) / bin_width), self.n_bins - 1)
            counts[idx] += 1
        n = len(values)
        h = 0.0
        for c in counts:
            if c > 0:
                p = c / n
                h -= p * math.log(p + 1e-10)
        return h

    def mutual_information(self, stage: str) -> float:
        """Compute I(stage; review) for a stage."""
        joint = self._joint.get(stage, [])
        if len(joint) < 5:
            return 0.0  # not enough data

        stage_vals = [s for s, _ in joint]
        review_vals = [r for _, r in joint]

        h_review = self._entropy(review_vals)

        # Conditional entropy: H(review | stage) via binned conditioning
        min_s, max_s = min(stage_vals), max(stage_vals)
        if max_s == min_s:
            return 0.0
        bin_width = (max_s - min_s) / self.n_bins
        bins: Dict[int, List[float]] = defaultdict(list)
        for s, r in joint:
            idx = min(int((s - min_s) / bin_width), self.n_bins - 1)
            bins[idx].append(r)

        h_conditional = 0.0
        n = len(joint)
        for idx, reviews in bins.items():
            weight = len(reviews) / n
            h_conditional += weight * self._entropy(reviews)

        return max(0, h_review - h_conditional)

    def rank_stages(self) -> List[Tuple[str, float]]:
        """Rank stages by mutual information with review score."""
        mi = [(stage, self.mutual_information(stage)) for stage in self._stage_scores]
        mi.sort(key=lambda x: x[1], reverse=True)
        return mi

    def recommend_skip(self, threshold: float = 0.05) -> List[str]:
        """Stages with MI below threshold — safe to skip."""
        return [stage for stage, mi in self.rank_stages() if mi < threshold]

    def stats(self) -> Dict[str, Any]:
        return {
            "rankings": [(s, round(mi, 4)) for s, mi in self.rank_stages()],
            "skip_candidates": self.recommend_skip(),
            "observations": len(self._review_scores),
        }


# ============================================================================
# 3. Nash Reviewer Consensus
# ============================================================================

class NashReviewerConsensus:
    """
    Game-theoretic aggregation of multiple reviewer scores.

    Instead of simple averaging, models reviewers as strategic agents.
    Each reviewer has a bias (optimistic/pessimistic) and precision
    (how noisy their scores are). Nash equilibrium finds the consensus
    score that no reviewer would unilaterally deviate from.

    Weighted aggregation: w_i = precision_i / Σ precision_j
    Bias correction: score_i' = score_i - bias_i
    Consensus = Σ w_i × score_i' (bias-corrected weighted average)
    """

    @dataclass
    class ReviewerProfile:
        id: str
        scores: List[float] = field(default_factory=list)
        bias: float = 0.0  # positive = optimistic
        precision: float = 1.0  # inverse variance

    def __init__(self):
        self._reviewers: Dict[str, "NashReviewerConsensus.ReviewerProfile"] = {}
        self._ground_truth: List[float] = []  # consensus from past rounds
        self._lock = threading.Lock()

    def record_review(self, reviewer_id: str, score: float) -> None:
        """Record a review score."""
        with self._lock:
            if reviewer_id not in self._reviewers:
                self._reviewers[reviewer_id] = self.ReviewerProfile(id=reviewer_id)
            self._reviewers[reviewer_id].scores.append(score)

    def update_profiles(self, consensus: float) -> None:
        """Update reviewer bias and precision based on consensus."""
        with self._lock:
            self._ground_truth.append(consensus)
            for rid, profile in self._reviewers.items():
                if profile.scores:
                    latest = profile.scores[-1]
                    error = latest - consensus
                    # Update bias (EMA)
                    profile.bias = 0.8 * profile.bias + 0.2 * error
                    # Update precision (inverse of rolling MSE)
                    errors = [s - gt for s, gt in zip(profile.scores[-5:], self._ground_truth[-5:])]
                    if errors:
                        mse = sum(e ** 2 for e in errors) / len(errors)
                        profile.precision = 1.0 / max(mse, 0.01)

    def compute_consensus(self, scores: Dict[str, float]) -> float:
        """
        Compute Nash consensus from multiple reviewer scores.

        Args:
            scores: {reviewer_id: score}
        Returns:
            Bias-corrected, precision-weighted consensus score.
        """
        with self._lock:
            weighted_sum = 0.0
            total_weight = 0.0

            for rid, score in scores.items():
                profile = self._reviewers.get(rid)
                if profile:
                    corrected = score - profile.bias
                    weight = profile.precision
                else:
                    corrected = score
                    weight = 1.0

                weighted_sum += weight * corrected
                total_weight += weight

            consensus = weighted_sum / max(total_weight, 0.001)
            consensus = max(0, min(10, consensus))

            # Update profiles with this consensus
            self.update_profiles(consensus)
            return consensus

    def reviewer_reliability(self) -> Dict[str, float]:
        """Get reliability score per reviewer."""
        with self._lock:
            return {
                rid: round(p.precision / max(sum(pp.precision for pp in self._reviewers.values()), 0.001), 3)
                for rid, p in self._reviewers.items()
            }

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "reviewers": len(self._reviewers),
                "profiles": {
                    rid: {"bias": round(p.bias, 3), "precision": round(p.precision, 2), "reviews": len(p.scores)}
                    for rid, p in self._reviewers.items()
                },
                "consensus_rounds": len(self._ground_truth),
            }


# ============================================================================
# 4. Working Memory with Attention Gate
# ============================================================================

class WorkingMemoryGate:
    """
    Selective context passing between pipeline stages.

    Instead of dumping ALL prior context into each stage's prompt,
    this filters context by relevance — like attention in transformers.

    Memory slots:
      - hypothesis: the current hypothesis being tested
      - findings: key experimental findings
      - errors: recent errors and fixes
      - insights: cross-stage insights

    Gate function: relevance(slot, stage) → [0, 1]
    Only slots with relevance > threshold are included in the prompt.
    Saves tokens by excluding irrelevant context.
    """

    @dataclass
    class MemorySlot:
        key: str
        content: str
        importance: float = 0.5
        created_at: float = field(default_factory=time.time)
        access_count: int = 0
        source_stage: str = ""

    # Relevance matrix: which slots are relevant to which stages
    RELEVANCE = {
        "ideation": {"hypothesis": 0.3, "findings": 0.8, "errors": 0.2, "insights": 0.9, "method": 0.7},
        "experiment_design": {"hypothesis": 0.9, "findings": 0.5, "errors": 0.3, "insights": 0.6, "method": 0.9},
        "code_generation": {"hypothesis": 0.4, "findings": 0.3, "errors": 0.9, "insights": 0.2, "method": 0.8},
        "execution": {"hypothesis": 0.2, "findings": 0.1, "errors": 0.8, "insights": 0.1, "method": 0.3},
        "analysis": {"hypothesis": 0.9, "findings": 0.9, "errors": 0.3, "insights": 0.7, "method": 0.5},
        "paper_writing": {"hypothesis": 0.9, "findings": 0.9, "errors": 0.1, "insights": 0.8, "method": 0.8},
        "review": {"hypothesis": 0.7, "findings": 0.8, "errors": 0.2, "insights": 0.6, "method": 0.5},
    }

    def __init__(self, max_slots: int = 20, relevance_threshold: float = 0.4):
        self.max_slots = max_slots
        self.threshold = relevance_threshold
        self._slots: Dict[str, "WorkingMemoryGate.MemorySlot"] = {}
        self._lock = threading.Lock()

    def store(self, key: str, content: str, importance: float = 0.5, source_stage: str = "") -> None:
        """Store or update a memory slot."""
        with self._lock:
            if len(self._slots) >= self.max_slots and key not in self._slots:
                # Evict least important + oldest
                evict_key = min(self._slots, key=lambda k: self._slots[k].importance * 0.7 + (1 / (time.time() - self._slots[k].created_at + 1)) * 0.3)
                del self._slots[evict_key]
            self._slots[key] = self.MemorySlot(
                key=key, content=content[:500],  # cap content
                importance=importance, source_stage=source_stage,
            )

    def retrieve(self, stage: str, max_tokens: int = 2000) -> str:
        """Retrieve relevant context for a stage via attention gating."""
        relevance_map = self.RELEVANCE.get(stage, {})
        with self._lock:
            candidates = []
            for key, slot in self._slots.items():
                # Compute relevance from slot type matching
                slot_type = key.split("_")[0] if "_" in key else key
                rel = relevance_map.get(slot_type, 0.3)
                gated_score = rel * slot.importance
                if gated_score >= self.threshold:
                    candidates.append((gated_score, slot))
                    slot.access_count += 1

        candidates.sort(reverse=True)

        # Build context string within token budget
        context_parts = []
        total_chars = 0
        for _, slot in candidates:
            if total_chars + len(slot.content) > max_tokens * 4:  # ~4 chars per token
                break
            context_parts.append(f"[{slot.key}] {slot.content}")
            total_chars += len(slot.content)

        return "\n".join(context_parts) if context_parts else ""

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "slots": len(self._slots),
                "total_accesses": sum(s.access_count for s in self._slots.values()),
                "slot_keys": list(self._slots.keys()),
            }


# ============================================================================
# 5. Dead Stage Eliminator
# ============================================================================

class DeadStageEliminator:
    """
    Compiler-style dead code elimination applied to pipeline stages.

    A stage is "dead" if its output is never consumed by any downstream
    stage that ultimately affects the final review score.

    Analysis:
      1. Build data-flow graph: which stages consume which outputs
      2. Mark stages reachable from the final output (review)
      3. All unreachable stages are dead — skip them

    Also detects:
      - Redundant stages: two stages producing equivalent output
      - No-op stages: stages that pass input through unchanged
    """

    # Data flow: stage → stages that consume its output
    DATA_FLOW = {
        "ideation": ["experiment_design", "tree_search", "paper_writing"],
        "tree_search": ["experiment_design"],
        "experiment_design": ["code_generation", "paper_writing"],
        "self_reflection_exp": ["experiment_design"],
        "code_generation": ["execution"],
        "self_reflection_code": ["code_generation"],
        "execution": ["analysis"],
        "analysis": ["paper_writing"],
        "self_reflection_results": ["analysis"],
        "paper_writing": ["review"],
        "self_reflection_paper": ["paper_writing"],
        "review": [],  # terminal
    }

    def __init__(self):
        self._stage_impact: Dict[str, float] = {}
        self._no_op_stages: Set[str] = set()

    def find_live_stages(self, terminal: str = "review") -> set:
        """BFS backwards from terminal to find all live stages."""
        # Build reverse graph
        reverse: Dict[str, List[str]] = defaultdict(list)
        for src, dsts in self.DATA_FLOW.items():
            for dst in dsts:
                reverse[dst].append(src)

        live = {terminal}
        queue = [terminal]
        while queue:
            current = queue.pop(0)
            for parent in reverse.get(current, []):
                if parent not in live:
                    live.add(parent)
                    queue.append(parent)
        return live

    def find_dead_stages(self) -> List[str]:
        """Stages that don't contribute to the final output."""
        live = self.find_live_stages()
        all_stages = set(self.DATA_FLOW.keys())
        return sorted(all_stages - live)

    def record_stage_impact(self, stage: str, impact: float) -> None:
        """Record measured impact of a stage on final quality."""
        self._stage_impact[stage] = 0.7 * self._stage_impact.get(stage, impact) + 0.3 * impact

    def record_no_op(self, stage: str) -> None:
        """Mark a stage as no-op (output ≈ input)."""
        self._no_op_stages.add(stage)

    def get_elimination_plan(self, min_impact: float = 0.05) -> Dict[str, str]:
        """Recommend which stages to eliminate or demote."""
        dead = set(self.find_dead_stages())
        plan = {}
        for stage in self.DATA_FLOW:
            if stage in dead:
                plan[stage] = "eliminate"
            elif stage in self._no_op_stages:
                plan[stage] = "eliminate_no_op"
            elif self._stage_impact.get(stage, 1.0) < min_impact:
                plan[stage] = "demote_to_lite"
            else:
                plan[stage] = "keep"
        return plan

    def stats(self) -> Dict[str, Any]:
        return {
            "live_stages": sorted(self.find_live_stages()),
            "dead_stages": self.find_dead_stages(),
            "no_op_stages": sorted(self._no_op_stages),
            "impact_scores": {k: round(v, 3) for k, v in self._stage_impact.items()},
        }


# ============================================================================
# 6. Predictive Quality Gate
# ============================================================================

class PredictiveQualityGate:
    """
    ML-based early abort before expensive stages.

    Uses lightweight features to predict whether an idea/experiment will
    succeed BEFORE running expensive code generation or execution.

    Features:
      - Idea quality score
      - Experiment plan confidence
      - Code quality score
      - Historical success rate for similar ideas
      - Adversarial resilience score

    Prediction: logistic regression trained online.
    If P(success) < threshold, skip expensive stages and move to next idea.
    """

    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold
        self._weights: Dict[str, float] = {
            "idea_quality": 2.0,
            "plan_confidence": 1.5,
            "code_quality": 1.0,
            "adversarial_resilience": 1.0,
            "historical_success": 1.5,
            "bias": -2.0,
        }
        self._training_data: List[Tuple[Dict[str, float], bool]] = []
        self._lock = threading.Lock()

    def predict(self, features: Dict[str, float]) -> float:
        """Predict P(success) using logistic regression."""
        z = self._weights.get("bias", 0)
        for feature, value in features.items():
            z += self._weights.get(feature, 0) * value
        return 1.0 / (1.0 + math.exp(-z))

    def should_proceed(self, features: Dict[str, float]) -> bool:
        """Should the pipeline proceed to expensive stages?"""
        prob = self.predict(features)
        return prob >= self.threshold

    def record_outcome(self, features: Dict[str, float], success: bool) -> None:
        """Update weights with a new observation (online logistic regression)."""
        with self._lock:
            self._training_data.append((features, success))
            # SGD update
            prob = self.predict(features)
            y = 1.0 if success else 0.0
            error = y - prob
            lr = 0.1

            for feature, value in features.items():
                if feature in self._weights:
                    self._weights[feature] += lr * error * value
            self._weights["bias"] += lr * error

    def stats(self) -> Dict[str, Any]:
        return {
            "weights": {k: round(v, 3) for k, v in self._weights.items()},
            "training_samples": len(self._training_data),
            "threshold": self.threshold,
        }


# ============================================================================
# 7. Bloom Filter Dedup
# ============================================================================

class BloomDedup:
    """
    Probabilistic O(1) idea deduplication using Bloom filters.

    Faster than SimHash + Jaccard for large idea pools.
    False positive rate ~1% with 10 hash functions and 10x overprovisioning.
    NO false negatives — if Bloom says "not seen", it's definitely new.

    Use: quick pre-filter before expensive similarity checks.
    """

    def __init__(self, expected_items: int = 1000, fp_rate: float = 0.01):
        # Optimal Bloom filter size
        self.size = max(64, int(-expected_items * math.log(fp_rate) / (math.log(2) ** 2)))
        self.n_hashes = max(1, int(self.size / expected_items * math.log(2)))
        self._bits = [False] * self.size
        self._count = 0
        self._lock = threading.Lock()

    def _hashes(self, item: str) -> List[int]:
        """Generate k hash positions for an item."""
        positions = []
        for i in range(self.n_hashes):
            h = hashlib.md5(f"{i}:{item}".encode(), usedforsecurity=False).hexdigest()
            positions.append(int(h, 16) % self.size)
        return positions

    def add(self, item: str) -> None:
        """Add an item to the filter."""
        with self._lock:
            for pos in self._hashes(item):
                self._bits[pos] = True
            self._count += 1

    def might_contain(self, item: str) -> bool:
        """Check if item might be in the filter. False = definitely not seen."""
        with self._lock:
            return all(self._bits[pos] for pos in self._hashes(item))

    def is_new(self, item: str) -> bool:
        """Check if item is definitely new (not in filter)."""
        return not self.might_contain(item)

    def add_if_new(self, item: str) -> bool:
        """Add item if new, return True if it was new."""
        if self.is_new(item):
            self.add(item)
            return True
        return False

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            fill_ratio = sum(self._bits) / self.size
        return {
            "items": self._count,
            "size_bits": self.size,
            "n_hashes": self.n_hashes,
            "fill_ratio": round(fill_ratio, 3),
            "estimated_fp_rate": round(fill_ratio ** self.n_hashes, 6),
        }


# ============================================================================
# 8. Exponential Forecaster (Holt-Winters)
# ============================================================================

class ExponentialForecaster:
    """
    Holt-Winters double exponential smoothing for quality/cost trend prediction.

    Forecasts future values using level + trend components:
      Level: L_t = α × y_t + (1-α) × (L_{t-1} + T_{t-1})
      Trend: T_t = β × (L_t - L_{t-1}) + (1-β) × T_{t-1}
      Forecast: F_{t+h} = L_t + h × T_t

    Use cases:
      - Predict future iteration quality → early stop if converging
      - Predict budget burn rate → trigger alerts
      - Predict stage duration → optimize scheduling
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.1):
        self.alpha = alpha
        self.beta = beta
        self._series: Dict[str, List[float]] = defaultdict(list)
        self._levels: Dict[str, float] = {}
        self._trends: Dict[str, float] = {}

    def observe(self, key: str, value: float) -> None:
        """Add a new observation to a time series."""
        self._series[key].append(value)
        series = self._series[key]

        if len(series) == 1:
            self._levels[key] = value
            self._trends[key] = 0
        elif len(series) == 2:
            self._levels[key] = value
            self._trends[key] = value - series[0]
        else:
            prev_level = self._levels[key]
            prev_trend = self._trends[key]
            self._levels[key] = self.alpha * value + (1 - self.alpha) * (prev_level + prev_trend)
            self._trends[key] = self.beta * (self._levels[key] - prev_level) + (1 - self.beta) * prev_trend

    def forecast(self, key: str, horizon: int = 1) -> float:
        """Forecast h steps ahead."""
        level = self._levels.get(key, 0.5)
        trend = self._trends.get(key, 0)
        return level + horizon * trend

    def is_converging(self, key: str, threshold: float = 0.01) -> bool:
        """Is the series converging (trend near zero)?"""
        return abs(self._trends.get(key, 1.0)) < threshold

    def is_improving(self, key: str) -> bool:
        """Is the trend positive?"""
        return self._trends.get(key, 0) > 0

    def stats(self) -> Dict[str, Any]:
        return {
            key: {
                "level": round(self._levels.get(key, 0), 3),
                "trend": round(self._trends.get(key, 0), 4),
                "forecast_1": round(self.forecast(key, 1), 3),
                "forecast_3": round(self.forecast(key, 3), 3),
                "converging": self.is_converging(key),
                "improving": self.is_improving(key),
                "observations": len(self._series.get(key, [])),
            }
            for key in self._series
        }


# ============================================================================
# 9. A/B Prompt Tester
# ============================================================================

class ABPromptTester:
    """
    Automated split testing for prompt variants.

    Randomly assigns each call to variant A or B, tracks quality outcomes,
    and uses a statistical test to determine which variant wins.

    Uses Welch's t-test for significance testing (no equal-variance assumption).
    Declares winner when p < 0.05 and minimum sample size reached.
    """

    @dataclass
    class Variant:
        name: str
        prompt_modifier: Callable[[str], str]
        scores: List[float] = field(default_factory=list)

    def __init__(self, min_samples: int = 10, significance: float = 0.05):
        self.min_samples = min_samples
        self.significance = significance
        self._tests: Dict[str, Dict[str, "ABPromptTester.Variant"]] = {}
        self._lock = threading.Lock()

    def create_test(self, test_id: str, variant_a_mod: Callable, variant_b_mod: Callable) -> None:
        """Create an A/B test with two prompt modifiers."""
        self._tests[test_id] = {
            "A": self.Variant(name="A", prompt_modifier=variant_a_mod),
            "B": self.Variant(name="B", prompt_modifier=variant_b_mod),
        }

    def get_variant(self, test_id: str) -> Tuple[str, Callable]:
        """Randomly assign a variant. Returns (variant_name, modifier_fn)."""
        with self._lock:
            test = self._tests.get(test_id, {})
            if not test:
                return "A", lambda x: x
            # Assign randomly, but prefer under-sampled variant
            a_count = len(test["A"].scores)
            b_count = len(test["B"].scores)
            if a_count <= b_count:
                chosen = "A"
            elif b_count < a_count:
                chosen = "B"
            else:
                chosen = random.choice(["A", "B"])
            return chosen, test[chosen].prompt_modifier

    def record_result(self, test_id: str, variant: str, score: float) -> None:
        with self._lock:
            test = self._tests.get(test_id, {})
            if variant in test:
                test[variant].scores.append(score)

    def get_winner(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Check if test has a statistically significant winner."""
        with self._lock:
            test = self._tests.get(test_id, {})
            if not test:
                return None
            a_scores = test["A"].scores
            b_scores = test["B"].scores

        if len(a_scores) < self.min_samples or len(b_scores) < self.min_samples:
            return {"status": "collecting", "a_samples": len(a_scores), "b_samples": len(b_scores)}

        # Welch's t-test
        mean_a = sum(a_scores) / len(a_scores)
        mean_b = sum(b_scores) / len(b_scores)
        var_a = sum((x - mean_a) ** 2 for x in a_scores) / (len(a_scores) - 1) if len(a_scores) > 1 else 0
        var_b = sum((x - mean_b) ** 2 for x in b_scores) / (len(b_scores) - 1) if len(b_scores) > 1 else 0

        se = math.sqrt(var_a / len(a_scores) + var_b / len(b_scores) + 1e-10)
        t_stat = (mean_a - mean_b) / max(se, 1e-10)

        # Approximate p-value using normal distribution (good for n > 10)
        p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2))))

        if p_value < self.significance:
            winner = "A" if mean_a > mean_b else "B"
            return {
                "status": "significant",
                "winner": winner,
                "mean_a": round(mean_a, 3), "mean_b": round(mean_b, 3),
                "t_stat": round(t_stat, 3), "p_value": round(p_value, 4),
                "effect_size": round(abs(mean_a - mean_b), 3),
            }
        return {
            "status": "inconclusive",
            "mean_a": round(mean_a, 3), "mean_b": round(mean_b, 3),
            "p_value": round(p_value, 4),
        }

    def stats(self) -> Dict[str, Any]:
        return {
            test_id: self.get_winner(test_id) or {}
            for test_id in self._tests
        }


# ============================================================================
# 10. Causal Impact Analyzer
# ============================================================================

class CausalImpactAnalyzer:
    """
    Difference-in-differences estimator for optimization impact.

    Measures the TRUE causal effect of an optimization by comparing:
      - Before vs after enabling
      - Treatment group vs control group (with vs without)

    DiD estimator: δ = (Y_after_treatment - Y_before_treatment) - (Y_after_control - Y_before_control)

    This controls for time trends that would confound simple before/after comparison.
    """

    @dataclass
    class Observation:
        timestamp: float
        value: float
        treatment: bool  # True if optimization was active
        period: str  # "before" or "after"

    def __init__(self):
        self._observations: Dict[str, List["CausalImpactAnalyzer.Observation"]] = defaultdict(list)

    def record(self, optimization: str, value: float, treatment: bool, period: str) -> None:
        self._observations[optimization].append(
            self.Observation(time.time(), value, treatment, period)
        )

    def estimate_impact(self, optimization: str) -> Dict[str, Any]:
        """Compute difference-in-differences estimate."""
        obs = self._observations.get(optimization, [])
        if len(obs) < 4:
            return {"status": "insufficient_data", "observations": len(obs)}

        # Split into 4 groups
        before_treatment = [o.value for o in obs if o.period == "before" and o.treatment]
        after_treatment = [o.value for o in obs if o.period == "after" and o.treatment]
        before_control = [o.value for o in obs if o.period == "before" and not o.treatment]
        after_control = [o.value for o in obs if o.period == "after" and not o.treatment]

        if not all((before_treatment, after_treatment, before_control, after_control)):
            return {"status": "missing_groups"}

        mean = lambda xs: sum(xs) / len(xs) if xs else 0
        did = (mean(after_treatment) - mean(before_treatment)) - (mean(after_control) - mean(before_control))

        return {
            "status": "estimated",
            "causal_impact": round(did, 4),
            "direction": "positive" if did > 0 else "negative",
            "mean_before_treatment": round(mean(before_treatment), 3),
            "mean_after_treatment": round(mean(after_treatment), 3),
            "mean_before_control": round(mean(before_control), 3),
            "mean_after_control": round(mean(after_control), 3),
        }

    def stats(self) -> Dict[str, Any]:
        return {
            opt: self.estimate_impact(opt)
            for opt in self._observations
        }


# ============================================================================
# 11. Reservoir Sampler
# ============================================================================

class ReservoirSampler:
    """
    Memory-efficient representative sampling from large idea streams.

    Vitter's Algorithm R: maintains a uniform random sample of size k
    from a stream of unknown length n. Each item has equal probability
    k/n of being in the sample, using O(k) memory regardless of n.

    Use: when generating many ideas, keep a representative sample
    for diversity analysis without storing all of them.
    """

    def __init__(self, k: int = 50):
        self.k = k
        self._reservoir: List[Any] = []
        self._n = 0
        self._lock = threading.Lock()

    def add(self, item: Any) -> bool:
        """Add item to stream. Returns True if item entered the reservoir."""
        with self._lock:
            self._n += 1
            if len(self._reservoir) < self.k:
                self._reservoir.append(item)
                return True
            # Replace with probability k/n
            j = random.randint(0, self._n - 1)
            if j < self.k:
                self._reservoir[j] = item
                return True
            return False

    def sample(self) -> List[Any]:
        """Get the current reservoir sample."""
        with self._lock:
            return list(self._reservoir)

    def clear(self) -> None:
        with self._lock:
            self._reservoir.clear()
            self._n = 0

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "reservoir_size": len(self._reservoir),
                "items_seen": self._n,
                "capacity": self.k,
                "sampling_rate": f"{self.k / max(self._n, 1):.1%}",
            }


# ============================================================================
# 12. Feedback Loop Detector
# ============================================================================

class FeedbackLoopDetector:
    """
    Detect and break negative feedback loops in the pipeline.

    Negative loops: quality drops → more aggressive generation → worse quality → ...
    Positive loops: good idea → good experiment → good paper → better next iteration

    Detection: track quality trajectory per stage. If a stage's quality is
    monotonically decreasing for N consecutive observations, flag it.

    Breaking strategies:
      - Reset temperature to default
      - Clear caches (stale data causing loops)
      - Inject random perturbation
      - Skip the problematic stage for one iteration
    """

    def __init__(self, window_size: int = 4, min_decline_streak: int = 3):
        self.window_size = window_size
        self.min_streak = min_decline_streak
        self._trajectories: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self._loops_detected: Dict[str, int] = defaultdict(int)
        self._interventions: List[Dict] = []
        self._lock = threading.Lock()

    def record(self, stage: str, quality: float) -> None:
        with self._lock:
            self._trajectories[stage].append(quality)

    def detect_negative_loop(self, stage: str) -> bool:
        """Is the stage in a negative feedback loop?"""
        with self._lock:
            traj = list(self._trajectories.get(stage, []))

        if len(traj) < self.min_streak:
            return False

        # Check for monotonic decline
        recent = traj[-self.min_streak:]
        declining = all(recent[i] > recent[i + 1] for i in range(len(recent) - 1))

        if declining:
            with self._lock:
                self._loops_detected[stage] += 1
            return True

        return False

    def detect_stagnation(self, stage: str, threshold: float = 0.02) -> bool:
        """Is the stage stagnating (quality not changing)?"""
        with self._lock:
            traj = list(self._trajectories.get(stage, []))
        if len(traj) < 3:
            return False
        variance = sum((x - sum(traj) / len(traj)) ** 2 for x in traj) / len(traj)
        return variance < threshold ** 2

    def get_intervention(self, stage: str) -> Optional[Dict[str, Any]]:
        """Get recommended intervention for a problematic stage."""
        if self.detect_negative_loop(stage):
            intervention = {
                "stage": stage,
                "type": "negative_loop",
                "actions": [
                    "reset_temperature",
                    "clear_stage_cache",
                    "inject_randomness",
                ],
                "severity": "high",
            }
            self._interventions.append(intervention)
            return intervention

        if self.detect_stagnation(stage):
            intervention = {
                "stage": stage,
                "type": "stagnation",
                "actions": [
                    "increase_temperature",
                    "try_different_strategy",
                ],
                "severity": "medium",
            }
            self._interventions.append(intervention)
            return intervention

        return None

    def scan_all(self) -> List[Dict[str, Any]]:
        """Scan all tracked stages for issues."""
        issues = []
        for stage in list(self._trajectories.keys()):
            intervention = self.get_intervention(stage)
            if intervention:
                issues.append(intervention)
        return issues

    def stats(self) -> Dict[str, Any]:
        return {
            "tracked_stages": len(self._trajectories),
            "loops_detected": dict(self._loops_detected),
            "total_interventions": len(self._interventions),
            "current_issues": self.scan_all(),
        }


# ============================================================================
# Master Meta Optimizer
# ============================================================================

class MetaOptimizer:
    """Aggregates all meta-optimization techniques."""

    def __init__(self, enable_all: bool = True):
        self.kalman = KalmanQualityEstimator() if enable_all else None
        self.mutual_info = MutualInfoStageRanker() if enable_all else None
        self.nash = NashReviewerConsensus() if enable_all else None
        self.memory = WorkingMemoryGate() if enable_all else None
        self.dead_stage = DeadStageEliminator() if enable_all else None
        self.quality_gate = PredictiveQualityGate() if enable_all else None
        self.bloom = BloomDedup() if enable_all else None
        self.forecaster = ExponentialForecaster() if enable_all else None
        self.ab_tester = ABPromptTester() if enable_all else None
        self.causal = CausalImpactAnalyzer() if enable_all else None
        self.reservoir = ReservoirSampler() if enable_all else None
        self.feedback = FeedbackLoopDetector() if enable_all else None

    def summary(self) -> Dict[str, Any]:
        result = {}
        if self.kalman: result["kalman_estimator"] = self.kalman.stats()
        if self.mutual_info: result["mutual_info_ranker"] = self.mutual_info.stats()
        if self.nash: result["nash_consensus"] = self.nash.stats()
        if self.memory: result["working_memory"] = self.memory.stats()
        if self.dead_stage: result["dead_stage_eliminator"] = self.dead_stage.stats()
        if self.quality_gate: result["predictive_gate"] = self.quality_gate.stats()
        if self.bloom: result["bloom_dedup"] = self.bloom.stats()
        if self.forecaster: result["forecaster"] = self.forecaster.stats()
        if self.ab_tester: result["ab_tester"] = self.ab_tester.stats()
        if self.causal: result["causal_impact"] = self.causal.stats()
        if self.reservoir: result["reservoir_sampler"] = self.reservoir.stats()
        if self.feedback: result["feedback_detector"] = self.feedback.stats()
        return result
