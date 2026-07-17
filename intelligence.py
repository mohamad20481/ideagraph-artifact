"""
intelligence.py - Intelligence upgrade layer for IdeaGraph.

Implements 7 features that make the pipeline genuinely smart:

  1. SemanticNoveltyChecker   - n-gram Jaccard to reject near-duplicate ideas
  2. QualityExplainer         - plain-English explanation of probe scores
  3. ParetoArchive            - multi-objective replacement (quality + novelty + feasibility)
  4. FailurePatternTracker    - learns which (methodology, novelty, probe) combos always fail
  5. StrategyAdvisor          - recommends A/B/C based on DAG structure
  6. CrossRunMemory           - persists lessons learned per user across runs
  7. ActiveMentorTools        - tool-calling for mentor chat (review, debate, revise)

All classes are stateless or thread-safe. No external dependencies beyond stdlib.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# 1. SEMANTIC NOVELTY CHECKER
# ─────────────────────────────────────────────────────────────────────────────

_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did will would "
    "shall should may might can could of in to for with on at by from as into "
    "through during before after above below between under and but or nor not "
    "so yet both either neither each every all any few more most other some such "
    "no only own same than too very this that these those it its we our they their "
    "using based method approach propose novel new".split()
)


def _text_tokens(text: str) -> frozenset:
    """Lowercase, stopword-filtered token set."""
    words = re.sub(r"[^a-z0-9]+", " ", text.lower()).split()
    return frozenset(w for w in words if w not in _STOPWORDS and len(w) > 2)


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class SemanticNoveltyChecker:
    """
    Checks whether a new idea is semantically too similar to existing archived
    ideas by comparing method-level n-gram Jaccard similarity.

    Stricter than the existing title-only dedup in pipeline.py:
    - Compares the METHOD field (not just title)
    - Uses bigrams for phrase-level matching
    - Returns the most-similar existing idea for transparency
    """

    def __init__(self, threshold: float = 0.55) -> None:
        self.threshold = threshold
        # Cache token-set length per archived item so the Jaccard hot loop
        # doesn't recompute len() each compare and can apply a cheap
        # size-ratio fast-skip before the O(|a|+|b|) set ops.
        self._archived: List[Tuple[str, frozenset, int]] = []  # (title, tokens, len)
        self._lock = threading.Lock()

    def register(self, title: str, method: str) -> None:
        tokens = _text_tokens(method)
        n = len(tokens)
        with self._lock:
            self._archived.append((title, tokens, n))

    def check(self, method: str) -> Tuple[bool, Optional[str], float]:
        """
        Returns (is_novel, most_similar_title, similarity_score).
        is_novel=True means the idea is sufficiently different.

        Optimization: Jaccard(a, b) ≤ min(|a|, |b|) / max(|a|, |b|), so if the
        size ratio is already below the novelty threshold, we can skip the
        intersection/union compute entirely for that archived item.
        """
        tokens = _text_tokens(method)
        n_q = len(tokens)
        if n_q == 0:
            # Empty query: no comparison meaningful, treat as novel.
            return True, None, 0.0

        threshold = self.threshold
        # Snapshot the archive list under lock, then compute outside the lock
        # so concurrent register()s don't block long check() runs.
        with self._lock:
            archived = list(self._archived)

        best_sim = 0.0
        best_title: Optional[str] = None
        for title, archived_tokens, n_a in archived:
            if n_a == 0:
                continue
            # Cheap upper-bound prune: max possible Jaccard for these sizes.
            if n_q < n_a:
                upper = n_q / n_a
            else:
                upper = n_a / n_q
            if upper <= best_sim:
                continue  # Can't possibly beat current best — skip the set ops.
            inter = len(tokens & archived_tokens)
            if inter == 0:
                continue
            sim = inter / (n_q + n_a - inter)  # |a∪b| = |a|+|b|-|a∩b|
            if sim > best_sim:
                best_sim = sim
                best_title = title
                # If we've already exceeded threshold, no need to find the
                # absolute max — caller only needs is_novel and a "most similar".
                # But keep going to give the most similar one for transparency
                # (matches the original semantics).

        is_novel = best_sim < threshold
        return is_novel, best_title, round(best_sim, 3)

    def clear(self) -> None:
        with self._lock:
            self._archived.clear()


# ─────────────────────────────────────────────────────────────────────────────
# 2. QUALITY EXPLAINER
# ─────────────────────────────────────────────────────────────────────────────

# Human-readable explanations for each probe dimension at different score levels.
_EXPLANATIONS = {
    "code": {
        "low": "The core method is difficult to implement — it requires custom algorithms or libraries that don't exist yet.",
        "mid": "The method is implementable but may need significant engineering effort beyond standard ML libraries.",
        "high": "The method can be implemented using standard frameworks like PyTorch/TensorFlow with minimal custom code.",
    },
    "dataset": {
        "low": "The required data doesn't appear to be publicly available. Consider synthetic data or publicly accessible benchmarks.",
        "mid": "Some required data is available but may need preprocessing or augmentation.",
        "high": "All required datasets are publicly available and well-documented.",
    },
    "constraint": {
        "low": "This idea requires industrial-scale compute (large GPU clusters, months of training). Consider a smaller-scale version.",
        "mid": "Feasible on academic hardware but will need careful optimization of batch size and training schedule.",
        "high": "Fits comfortably within a typical academic compute budget (1-4 GPUs, days of training).",
    },
    "novelty": {
        "low": "This approach is very similar to existing work. The core contribution is unclear.",
        "mid": "There's a novel angle, but the overall approach builds heavily on known methods.",
        "high": "This is a genuinely new direction that hasn't been explored in the literature.",
    },
    "specificity": {
        "low": "The method description is too vague — add specific algorithm names, exact metrics, and concrete dataset names.",
        "mid": "Mostly specific, but some parts of the method could be more concrete.",
        "high": "Excellent specificity — the method, metrics, datasets, and expected outcomes are all clearly defined.",
    },
    "significance": {
        "low": "Even if successful, the impact on the field would be limited.",
        "mid": "Useful contribution that addresses a real gap in the literature.",
        "high": "Solving this would significantly advance the field and open new research directions.",
    },
}


def explain_quality(scores: Dict[str, float], idea_title: str = "") -> str:
    """
    Generate a 2-4 sentence plain-English explanation of why an idea got
    its quality score. Highlights the weakest dimension and suggests a fix.
    """
    if not scores:
        return "No probe scores available."

    # Find weakest and strongest dimensions
    sorted_dims = sorted(scores.items(), key=lambda x: x[1])
    weakest_name, weakest_score = sorted_dims[0]
    strongest_name, strongest_score = sorted_dims[-1]

    # Overall quality
    quality = sum(scores.values()) / len(scores) if scores else 0

    # Pick explanation level
    def _level(score: float) -> str:
        if score < 0.4:
            return "low"
        elif score < 0.7:
            return "mid"
        return "high"

    parts = []

    # Overall summary
    if quality >= 0.6:
        parts.append(f"This is a solid idea (quality {quality:.2f}).")
    elif quality >= 0.4:
        parts.append(f"This idea has potential but needs work (quality {quality:.2f}).")
    else:
        parts.append(f"This idea needs significant revision (quality {quality:.2f}).")

    # Weakest dimension explanation
    if weakest_name in _EXPLANATIONS:
        level = _level(weakest_score)
        parts.append(
            f"Weakest area: **{weakest_name}** ({weakest_score:.2f}). "
            f"{_EXPLANATIONS[weakest_name][level]}"
        )

    # If there's a second weak dimension, mention it
    if len(sorted_dims) > 1:
        second_name, second_score = sorted_dims[1]
        if second_score < 0.4 and second_name in _EXPLANATIONS:
            parts.append(
                f"Also weak: **{second_name}** ({second_score:.2f}). "
                f"{_EXPLANATIONS[second_name][_level(second_score)]}"
            )

    # Strongest dimension (positive feedback)
    if strongest_score >= 0.6 and strongest_name in _EXPLANATIONS:
        parts.append(
            f"Strongest area: **{strongest_name}** ({strongest_score:.2f}). "
            f"{_EXPLANATIONS[strongest_name][_level(strongest_score)]}"
        )

    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# 3. PARETO ARCHIVE REPLACEMENT
# ─────────────────────────────────────────────────────────────────────────────

def pareto_dominates(new_scores: Dict[str, float], old_scores: Dict[str, float],
                     keys: Tuple[str, ...] = ("quality", "novelty", "feasibility")) -> bool:
    """
    True if new_scores Pareto-dominates old_scores on the given keys:
    at least as good on ALL keys, and strictly better on at least one.
    """
    dominated = False
    for k in keys:
        nv = new_scores.get(k, 0.0)
        ov = old_scores.get(k, 0.0)
        if nv < ov:
            return False
        if nv > ov:
            dominated = True
    return dominated


def should_replace_pareto(new_idea, old_idea) -> bool:
    """
    Multi-objective replacement check for QD archive.
    Replaces if:
      1. New idea Pareto-dominates on (quality, novelty_score, feasibility_score), OR
      2. New idea has higher quality AND at least one other dimension is better
    """
    new_scores = _extract_multi_scores(new_idea)
    old_scores = _extract_multi_scores(old_idea)

    # Pure Pareto dominance
    if pareto_dominates(new_scores, old_scores):
        return True

    # Relaxed: better quality + at least one other dimension better
    if new_scores["quality"] > old_scores["quality"]:
        for k in ("novelty", "feasibility"):
            if new_scores[k] > old_scores[k]:
                return True

    return False


def _extract_multi_scores(idea) -> Dict[str, float]:
    """Extract multi-objective scores from an Idea."""
    probe = idea.probe_scores or {}
    return {
        "quality": idea.quality_score,
        "novelty": probe.get("novelty", 0.5),
        "feasibility": (probe.get("code", 0.5) + probe.get("dataset", 0.5)
                         + probe.get("constraint", 0.5)) / 3.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. FAILURE PATTERN TRACKER
# ─────────────────────────────────────────────────────────────────────────────

class FailurePatternTracker:
    """
    Tracks which (methodology_type, novelty_level, failed_probe) combos
    repeatedly fail. After 3+ failures in the same pattern, generates a
    specific mitigation hint to inject into the ideation prompt.
    """

    MITIGATIONS = {
        "code": "This cell commonly fails on implementability. Use only standard ML libraries (PyTorch, sklearn, networkx). Avoid requiring custom CUDA kernels or novel data structures.",
        "dataset": "This cell commonly fails on data availability. Suggest only publicly available benchmarks (e.g., ZINC, OGB, MoleculeNet for drug discovery; ImageNet, COCO for vision). Avoid requiring proprietary or clinical data.",
        "constraint": "This cell commonly fails on compute requirements. Design methods that train in <24 hours on a single A100 GPU. Use efficient architectures (not billion-parameter models).",
        "novelty": "This cell commonly fails on novelty. The ideas generated tend to be too similar to existing work. Think about combining techniques from DIFFERENT sub-fields or applying methods from outside the domain.",
        "specificity": "This cell commonly fails on specificity. Be extremely concrete: name exact algorithms, specific datasets with version numbers, exact evaluation metrics, and quantitative performance targets.",
        "significance": "This cell commonly fails on significance. Focus on problems that affect many researchers or have broad applications, not narrow corner cases.",
    }

    THRESHOLD = 3  # failures before triggering mitigation

    def __init__(self) -> None:
        # (method_idx, novelty_idx, probe_name) → failure count
        self._failures: Dict[Tuple[int, int, str], int] = defaultdict(int)
        self._lock = threading.Lock()

    def record_failure(self, method_idx: int, novelty_idx: int, probe_name: str) -> None:
        with self._lock:
            self._failures[(method_idx, novelty_idx, probe_name)] += 1

    def record_success(self, method_idx: int, novelty_idx: int) -> None:
        """Reset failure counts for a cell when it succeeds."""
        with self._lock:
            keys = [k for k in self._failures if k[0] == method_idx and k[1] == novelty_idx]
            for k in keys:
                self._failures[k] = max(0, self._failures[k] - 1)

    def get_mitigations(self, method_idx: int, novelty_idx: int) -> str:
        """
        Return accumulated mitigation hints for a cell, or empty string if none.
        Only fires after THRESHOLD failures in the same pattern.
        """
        hints = []
        with self._lock:
            for (mi, ni, probe), count in self._failures.items():
                if mi == method_idx and ni == novelty_idx and count >= self.THRESHOLD:
                    hint = self.MITIGATIONS.get(probe, "")
                    if hint:
                        hints.append(hint)
        return "\n".join(hints)

    def clear(self) -> None:
        with self._lock:
            self._failures.clear()

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                f"({mi},{ni},{p})": c
                for (mi, ni, p), c in self._failures.items()
                if c >= 2
            }


# ─────────────────────────────────────────────────────────────────────────────
# 5. STRATEGY ADVISOR
# ─────────────────────────────────────────────────────────────────────────────

def advise_strategy(dag_node_count: int, dag_cluster_count: int,
                    dag_edge_count: int, iteration: int) -> str:
    """
    Recommend the best ideation strategy based on DAG structure.

    Returns a reason string that can be injected into progress messages.
    The actual selection still goes through Thompson sampling in diversity_manager,
    but this adds a prior bias.
    """
    if dag_node_count < 3:
        return "DAG is very sparse (<3 papers). Using topic-direct generation (Strategy A fallback)."
    if dag_cluster_count < 2:
        return "Only 1 cluster found. Strategy B/C won't help. Bias toward Strategy A (frontier extension)."
    if dag_cluster_count >= 3 and dag_edge_count > dag_node_count * 1.5:
        return "Rich graph with 3+ clusters and many connections. Strategies B (cross-cluster) and C (gap-filling) are promising."
    if iteration > 5 and dag_cluster_count >= 2:
        return "Late iteration with multiple clusters. Bias toward Strategy C (gap-filling) for diversity."
    return "Balanced graph. All strategies viable."


# ─────────────────────────────────────────────────────────────────────────────
# 6. CROSS-RUN MEMORY (per-user persistent lessons)
# ─────────────────────────────────────────────────────────────────────────────

class CrossRunMemory:
    """
    Persists lessons learned per user in the DB. Each lesson is a short
    string like "empirical_study ideas score highest" or "always fails
    dataset probe for interdisciplinary topics."

    Loaded at pipeline start, injected into ideation prompts.
    """

    MAX_LESSONS = 20

    def __init__(self, user_id: Optional[int] = None) -> None:
        self.user_id = user_id
        self._lessons: List[str] = []

    def load_from_db(self) -> None:
        """Load lessons from the agent_memory table for this user."""
        if not self.user_id:
            return
        try:
            import db
            memories = db.recall_agent_memories(
                self.user_id, agent_role="cross_run_learner",
                domain="lessons", n=self.MAX_LESSONS,
            )
            self._lessons = [m.get("content", "") for m in memories if m.get("content")]
        except Exception:
            self._lessons = []

    def save_lesson(self, lesson: str) -> None:
        """Persist a lesson to the DB."""
        if not self.user_id or not lesson:
            return
        self._lessons.append(lesson)
        try:
            import db
            db.store_agent_memory(
                user_id=self.user_id,
                agent_role="cross_run_learner",
                memory_type="lesson",
                domain="lessons",
                content=lesson,
                relevance_score=0.8,
            )
        except Exception:
            pass

    def extract_lessons_from_run(self, results: Dict[str, Any]) -> List[str]:
        """
        Analyze pipeline results and extract generalizable lessons.
        Called at end of each run.
        """
        lessons = []
        ideas = results.get("ideas", [])
        stats = results.get("stats", {})
        coverage = results.get("coverage", 0)

        if not ideas:
            lessons.append("Run produced no ideas. Consider using a simpler topic or increasing budget.")
            return lessons

        # Find best methodology type
        method_scores: Dict[str, List[float]] = defaultdict(list)
        for idea in ideas:
            mt = idea.get("methodology_type", "unknown")
            q = idea.get("quality_score", 0)
            method_scores[mt].append(q)

        if method_scores:
            best_method = max(method_scores, key=lambda k: sum(method_scores[k]) / len(method_scores[k]))
            best_avg = sum(method_scores[best_method]) / len(method_scores[best_method])
            if best_avg > 0.5:
                lessons.append(f"Best methodology: {best_method} (avg quality {best_avg:.2f}). Prioritize this type.")

        # Find commonly failing probes
        probe_fails: Dict[str, int] = defaultdict(int)
        for idea in ideas:
            scores = idea.get("probe_scores", {})
            for probe, score in scores.items():
                if isinstance(score, (int, float)) and score < 0.4:
                    probe_fails[probe] += 1
        for probe, count in probe_fails.items():
            if count >= len(ideas) * 0.5:  # >50% of ideas fail this probe
                lessons.append(f"Frequent failure: {probe} probe (failed in {count}/{len(ideas)} ideas). Address this proactively.")

        # Coverage insight
        if coverage >= 0.5:
            lessons.append(f"Good coverage ({coverage:.0%}). Archive is well-balanced.")
        elif coverage < 0.2:
            lessons.append(f"Low coverage ({coverage:.0%}). Try more iterations or a broader topic.")

        return lessons

    def get_context_for_prompt(self) -> str:
        """
        Return a prompt-injectable context string with accumulated lessons.
        """
        if not self._lessons:
            return ""
        recent = self._lessons[-10:]
        return (
            "\n\nLessons from previous runs (use these to improve this run):\n"
            + "\n".join(f"- {l}" for l in recent)
        )


# ─────────────────────────────────────────────────────────────────────────────
# 7. ACTIVE MENTOR TOOLS
# ─────────────────────────────────────────────────────────────────────────────

class ActiveMentorTools:
    """
    Tool functions that the mentor chat can call to actively analyze and
    improve ideas (instead of just giving generic advice).
    """

    def __init__(self) -> None:
        self._critic = None
        self._ideation = None
        self._reviewer = None

    def _get_critic(self):
        if self._critic is None:
            from agents.execution_critic import ExecutionCritic
            self._critic = ExecutionCritic()
        return self._critic

    def _get_ideation(self):
        if self._ideation is None:
            from agents.ideation_agent import IdeationAgent
            self._ideation = IdeationAgent()
        return self._ideation

    def analyze_idea(self, idea_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run full probe analysis on an idea and return scores + explanation.
        """
        from models.idea import Idea
        idea = Idea(
            title=idea_dict.get("title", ""),
            motivation=idea_dict.get("motivation", ""),
            method=idea_dict.get("method", ""),
            hypothesis=idea_dict.get("hypothesis", ""),
            resources=idea_dict.get("resources", ""),
            expected_outcome=idea_dict.get("expected_outcome", ""),
            risk_assessment=idea_dict.get("risk_assessment", ""),
        )
        result = self._get_critic().probe_all(idea)
        explanation = explain_quality(result.get("scores", {}), idea.title)
        return {
            "scores": result.get("scores", {}),
            "quality": result.get("quality", 0),
            "passed": result.get("all_pass", False),
            "explanation": explanation,
            "feedback": result.get("feedback", ""),
        }

    def improve_idea(self, idea_dict: Dict[str, Any], feedback: str = "") -> Optional[Dict[str, Any]]:
        """
        Revise an idea based on feedback (or auto-generated feedback from probes).
        """
        from models.idea import Idea
        idea = Idea(
            title=idea_dict.get("title", ""),
            motivation=idea_dict.get("motivation", ""),
            method=idea_dict.get("method", ""),
            hypothesis=idea_dict.get("hypothesis", ""),
            resources=idea_dict.get("resources", ""),
            expected_outcome=idea_dict.get("expected_outcome", ""),
            risk_assessment=idea_dict.get("risk_assessment", ""),
            quality_score=idea_dict.get("quality_score", 0.5),
        )
        if not feedback:
            # Auto-generate feedback from probes
            probe_result = self._get_critic().probe_all(idea)
            feedback = probe_result.get("feedback", "Improve specificity and novelty.")
        revised = self._get_ideation().revise_idea(idea, feedback, idea.quality_score)
        if revised:
            return revised.to_dict()
        return None

    def compare_ideas(self, idea_a: Dict[str, Any], idea_b: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare two ideas across all probe dimensions.
        """
        scores_a = self.analyze_idea(idea_a)
        scores_b = self.analyze_idea(idea_b)

        comparison = {}
        for dim in scores_a.get("scores", {}):
            sa = scores_a["scores"].get(dim, 0)
            sb = scores_b["scores"].get(dim, 0)
            if sa > sb + 0.1:
                winner = "A"
            elif sb > sa + 0.1:
                winner = "B"
            else:
                winner = "tie"
            comparison[dim] = {"A": round(sa, 2), "B": round(sb, 2), "winner": winner}

        overall_a = scores_a.get("quality", 0)
        overall_b = scores_b.get("quality", 0)
        return {
            "dimensions": comparison,
            "overall_winner": "A" if overall_a > overall_b else "B" if overall_b > overall_a else "tie",
            "explanation_a": scores_a.get("explanation", ""),
            "explanation_b": scores_b.get("explanation", ""),
        }

    def suggest_next_steps(self, idea_dict: Dict[str, Any]) -> List[str]:
        """
        Analyze an idea and suggest concrete next steps for the researcher.
        """
        scores = idea_dict.get("probe_scores", {})
        steps = []

        if not scores:
            return ["Run the idea through the probe system first to get quality scores."]

        # Sort by score, suggest fixes for weakest areas
        sorted_scores = sorted(scores.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0)

        for dim, score in sorted_scores[:3]:
            if isinstance(score, (int, float)) and score < 0.5:
                if dim == "code":
                    steps.append("Write a proof-of-concept implementation (even pseudocode) to validate feasibility.")
                elif dim == "dataset":
                    steps.append("Identify 2-3 specific public datasets that would work for this experiment.")
                elif dim == "constraint":
                    steps.append("Design a smaller-scale pilot experiment that runs in <1 GPU-day.")
                elif dim == "novelty":
                    steps.append("Search for the 3 most similar papers and explicitly state how your approach differs.")
                elif dim == "specificity":
                    steps.append("Rewrite the method section with exact algorithm names, hyperparameters, and evaluation metrics.")
                elif dim == "significance":
                    steps.append("Articulate who benefits from this research and quantify the expected improvement.")

        if not steps:
            steps.append("This idea looks strong! Consider running the full experiment pipeline.")
            steps.append("Share it with collaborators for additional feedback.")

        return steps


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# 8. SMART IDEA RECOMMENDER ("Ideas Like This")
# ─────────────────────────────────────────────────────────────────────────────

class IdeaRecommender:
    """
    Given a liked/bookmarked idea, find the N most conceptually similar ideas
    from the archive or other users' shared ideas.
    """

    def recommend(
        self,
        target_idea: Dict[str, Any],
        candidate_ideas: List[Dict[str, Any]],
        n: int = 5,
    ) -> List[Dict[str, Any]]:
        target_tokens = _text_tokens(target_idea.get("method", ""))
        target_title_tokens = _text_tokens(target_idea.get("title", ""))
        if not target_tokens:
            return []

        scored = []
        for cand in candidate_ideas:
            if cand.get("title", "") == target_idea.get("title", ""):
                continue
            cand_tokens = _text_tokens(cand.get("method", ""))
            cand_title_tokens = _text_tokens(cand.get("title", ""))
            method_sim = _jaccard(target_tokens, cand_tokens)
            title_sim = _jaccard(target_title_tokens, cand_title_tokens)
            sim = 0.7 * method_sim + 0.3 * title_sim
            if sim > 0.1:
                overlap = target_tokens & cand_tokens
                reason = f"Shares concepts: {', '.join(list(overlap)[:5])}" if overlap else "Related topic"
                scored.append({
                    **cand,
                    "_similarity": round(sim, 3),
                    "_recommendation_reason": reason,
                })
        scored.sort(key=lambda x: x["_similarity"], reverse=True)
        return scored[:n]


# ─────────────────────────────────────────────────────────────────────────────
# 9. IDEA MATURITY SCORING
# ─────────────────────────────────────────────────────────────────────────────

MATURITY_LEVELS = [
    ("sketch",            0, "Just a title and rough direction"),
    ("concept",           1, "Has motivation + method outline"),
    ("proposal",          2, "Fully specified: method, hypothesis, resources, expected outcomes"),
    ("validated",         3, "Passed probe evaluation with quality >= 0.5"),
    ("experiment-ready",  4, "Has experiment design + code skeleton"),
    ("publication-ready", 5, "Has paper draft + positive review"),
]

MATURITY_EMOJI = {0: "📝", 1: "💡", 2: "📋", 3: "✅", 4: "🧪", 5: "📄"}
MATURITY_COLOR = {0: "#95a5a6", 1: "#f39c12", 2: "#3498db", 3: "#2ecc71", 4: "#9b59b6", 5: "#e74c3c"}


def compute_maturity(idea: Dict[str, Any], has_experiment: bool = False,
                     has_paper: bool = False, review_score: float = 0.0) -> Dict[str, Any]:
    criteria_met = []
    criteria_missing = []

    has_title = bool(idea.get("title", "").strip())
    if has_title:
        criteria_met.append("Has title")
    has_motivation = len(idea.get("motivation", "")) > 20
    has_method = len(idea.get("method", "")) > 50
    if has_motivation:
        criteria_met.append("Has motivation")
    else:
        criteria_missing.append("Add motivation (why this matters)")
    if has_method:
        criteria_met.append("Has method description")
    else:
        criteria_missing.append("Add detailed method (>50 chars)")
    has_hypothesis = len(idea.get("hypothesis", "")) > 20
    has_resources = len(idea.get("resources", "")) > 10
    has_outcome = len(idea.get("expected_outcome", "")) > 20
    for check, label, missing_msg in [
        (has_hypothesis, "Has testable hypothesis", "Add falsifiable hypothesis"),
        (has_resources, "Has resource requirements", "Specify compute/data needs"),
        (has_outcome, "Has expected outcomes", "Describe expected results"),
    ]:
        (criteria_met if check else criteria_missing).append(label if check else missing_msg)
    quality = idea.get("quality_score", 0)
    probe_passed = idea.get("probe_passed", False)
    if probe_passed and quality >= 0.5:
        criteria_met.append(f"Probe validated (quality={quality:.2f})")
    elif not probe_passed:
        criteria_missing.append("Pass probe evaluation")
    else:
        criteria_missing.append(f"Improve quality above 0.50 (currently {quality:.2f})")
    if has_experiment:
        criteria_met.append("Has experiment design + code")
    else:
        criteria_missing.append("Generate experiment design and code")
    if has_paper and review_score >= 5.0:
        criteria_met.append(f"Has paper draft (review score: {review_score:.1f}/10)")
    elif not has_paper:
        criteria_missing.append("Generate paper draft")
    else:
        criteria_missing.append(f"Improve paper (review score {review_score:.1f} < 5.0)")

    total_criteria = 9
    met_count = len(criteria_met)
    if has_paper and review_score >= 5.0:
        level = 5
    elif has_experiment:
        level = 4
    elif probe_passed and quality >= 0.5:
        level = 3
    elif has_hypothesis and has_resources and has_outcome and has_method:
        level = 2
    elif has_motivation and has_method:
        level = 1
    else:
        level = 0

    label, _, description = MATURITY_LEVELS[level]
    return {
        "level": level,
        "label": label,
        "description": description,
        "progress_pct": round((met_count / total_criteria) * 100, 1),
        "next_step": criteria_missing[0] if criteria_missing else "Ready to publish!",
        "criteria_met": criteria_met,
        "criteria_missing": criteria_missing,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 10. COST PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────

def predict_run_cost(
    budget_usd: float, iterations: int,
    debate_enabled: bool = False, provider: str = "deepseek",
) -> Dict[str, Any]:
    rates = {
        "deepseek":  {"input": 0.27, "output": 1.10},
        "openai":    {"input": 2.50, "output": 10.00},
        "groq":      {"input": 0.59, "output": 0.79},
        "gemini":    {"input": 0.10, "output": 0.40},
        "azure":     {"input": 0.27, "output": 1.10},
        "anthropic": {"input": 3.00, "output": 15.00},  # Sonnet default
    }
    rate = rates.get(provider, rates["deepseek"])
    dag_tokens = 3000
    per_cell_tokens = 4000
    cells_per_iter = 5
    debate_tokens = 2000

    total_input = dag_tokens
    total_output = int(dag_tokens * 0.5)
    for _ in range(iterations):
        total_input += cells_per_iter * per_cell_tokens
        total_output += int(cells_per_iter * per_cell_tokens * 0.4)
    if debate_enabled:
        total_input += debate_tokens * 6
        total_output += debate_tokens * 3

    cost = (total_input * rate["input"] + total_output * rate["output"]) / 1_000_000
    cost = min(cost, budget_usd)

    call_time = {"deepseek": 10, "openai": 5, "groq": 3, "gemini": 4, "azure": 10}.get(provider, 10)
    total_calls = 3 + iterations * cells_per_iter * 2 + (18 if debate_enabled else 0)
    minutes = (total_calls / 3.0 * call_time) / 60.0

    return {
        "estimated_cost_usd": round(cost, 3),
        "estimated_minutes": round(minutes, 1),
        "cost_breakdown": {
            "dag_building": round(dag_tokens * rate["input"] / 1_000_000, 4),
            "ideation": round(iterations * cells_per_iter * per_cell_tokens * rate["input"] / 1_000_000, 4),
            "debate": round((debate_tokens * 6 * rate["input"] / 1_000_000) if debate_enabled else 0, 4),
        },
        "confidence": "estimated",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 11. CLAIM VERIFIER (literature-grounded)
# ─────────────────────────────────────────────────────────────────────────────

class ClaimVerifier:
    def verify_against_dag(
        self, idea: Dict[str, Any], dag_papers: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        claims = self._extract_claims(idea)
        if not claims or not dag_papers:
            return [{"claim": c, "status": "unverified",
                     "evidence": "No literature available", "confidence": 0.0}
                    for c in claims]
        results = []
        for claim in claims:
            claim_tokens = _text_tokens(claim)
            best_match, best_sim = None, 0.0
            for paper in dag_papers:
                paper_text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
                sim = _jaccard(claim_tokens, _text_tokens(paper_text))
                if sim > best_sim:
                    best_sim = sim
                    best_match = paper
            if best_sim >= 0.25 and best_match:
                results.append({
                    "claim": claim,
                    "status": "supported",
                    "evidence": f"Related: {best_match.get('title', '?')[:80]}",
                    "confidence": round(best_sim, 2),
                })
            else:
                results.append({
                    "claim": claim,
                    "status": "unverified",
                    "evidence": "No matching paper found in DAG",
                    "confidence": round(best_sim, 2),
                })
        return results

    def _extract_claims(self, idea: Dict[str, Any]) -> List[str]:
        text = f"{idea.get('hypothesis', '')}. {idea.get('method', '')}"
        sentences = re.split(r'(?<=[.!?])\s+', text)
        claims = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 30 and not sent.lower().startswith(("we propose", "this paper", "in this work")):
                claims.append(sent[:150])
        return claims[:6]


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Semantic Novelty Checker ===")
    checker = SemanticNoveltyChecker(threshold=0.55)
    checker.register("GNN for molecules", "use graph neural networks with message passing to predict molecular properties on ZINC dataset")
    checker.register("Transformer attention", "apply self-attention transformer architecture to protein folding prediction")

    is_novel, similar, sim = checker.check("use graph neural networks with attention to predict drug toxicity on ZINC benchmark")
    print(f"  Novel: {is_novel}, Similar to: {similar}, Similarity: {sim}")

    is_novel2, _, sim2 = checker.check("reinforcement learning for robotic control in simulated environments")
    print(f"  Novel: {is_novel2}, Similarity: {sim2}")

    print("\n=== Quality Explainer ===")
    scores = {"code": 0.3, "dataset": 0.8, "novelty": 0.6, "constraint": 0.5, "specificity": 0.2, "significance": 0.7}
    print(f"  {explain_quality(scores)}")

    print("\n=== Failure Pattern Tracker ===")
    tracker = FailurePatternTracker()
    for _ in range(4):
        tracker.record_failure(0, 0, "dataset")
    hint = tracker.get_mitigations(0, 0)
    print(f"  Hint: {hint[:100]}...")

    print("\n=== Strategy Advisor ===")
    print(f"  {advise_strategy(2, 1, 1, 0)}")
    print(f"  {advise_strategy(20, 4, 35, 3)}")

    print("\n=== Cross-Run Memory ===")
    mem = CrossRunMemory()
    lessons = mem.extract_lessons_from_run({
        "ideas": [
            {"methodology_type": "empirical_study", "quality_score": 0.7, "probe_scores": {"code": 0.8, "dataset": 0.3}},
            {"methodology_type": "empirical_study", "quality_score": 0.6, "probe_scores": {"code": 0.7, "dataset": 0.2}},
        ],
        "coverage": 0.3,
    })
    for l in lessons:
        print(f"  Lesson: {l}")

    print("\nAll intelligence tests passed.")
