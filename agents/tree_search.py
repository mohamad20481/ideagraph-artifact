"""
agents/tree_search.py - Agentic tree search for experiment exploration.

Tries multiple experimental approaches in parallel, evaluates each,
and expands the most promising branches. Inspired by AI Scientist-v2.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from agents.base_agent import BaseAgent


@dataclass(slots=True)
class ExperimentNode:
    """A node in the experiment search tree."""
    id: int
    approach: str                           # description of this approach
    experiment_plan: Dict[str, Any]         # full experiment plan
    code_files: Optional[Dict[str, str]] = None
    execution_result: Optional[Dict] = None
    analysis: Optional[Dict] = None
    score: float = 0.0                      # quality score (0-1)
    parent_id: Optional[int] = None
    children_ids: List[int] = field(default_factory=list)
    depth: int = 0
    status: str = "pending"                 # pending, running, completed, failed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "approach": self.approach,
            "score": self.score, "depth": self.depth, "status": self.status,
            "parent_id": self.parent_id, "children_ids": self.children_ids,
        }


class ExperimentTreeSearch(BaseAgent):
    """
    Agentic tree search over experimental approaches.

    Given an idea, generates multiple experimental approaches (branching),
    evaluates each, and expands the most promising ones.
    """

    def __init__(self):
        super().__init__(temperature=0.7)

    def search(
        self,
        idea: Dict[str, Any],
        domain: str,
        max_branches: int = 3,
        max_depth: int = 2,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> List[ExperimentNode]:
        """
        Explore multiple experimental approaches via tree search.

        Returns list of all nodes (best-first sorted by score).
        """
        if on_progress:
            on_progress(f"Tree search: exploring {max_branches} approaches, depth {max_depth}")

        nodes: List[ExperimentNode] = []
        node_counter = 0

        # Generate root branches — different experimental approaches
        approaches = self._generate_approaches(idea, domain, max_branches)

        # Defensive early-exit: if the LLM returned no approaches (call
        # failed, parse failed, rate-limited, no API key), don't try to
        # expand or sort an empty tree. Returning [] lets the caller
        # gracefully fall back via `tree_nodes[0] if tree_nodes else None`.
        if not approaches:
            if on_progress:
                on_progress(
                    "Tree search: no approaches generated — LLM may be "
                    "unavailable or rate-limited. Skipping tree expansion."
                )
            return []

        for approach in approaches:
            node = ExperimentNode(
                id=node_counter,
                approach=approach.get("approach", "default approach"),
                experiment_plan=approach, depth=0,
            )
            nodes.append(node)
            node_counter += 1

        if on_progress:
            on_progress(f"Generated {len(approaches)} root approaches")

        # Evaluate root nodes
        for node in nodes:
            if on_progress:
                on_progress(f"  Evaluating approach: {node.approach[:50]}...")
            node.score = self._evaluate_approach(node.experiment_plan, idea, domain)
            node.status = "completed"

        # Expand best nodes up to max_depth
        for depth in range(1, max_depth):
            # Sort by score, take top branch
            sorted_nodes = sorted(
                [n for n in nodes if n.depth == depth - 1 and n.status == "completed"],
                key=lambda n: n.score, reverse=True,
            )
            if not sorted_nodes:
                break

            best = sorted_nodes[0]
            if on_progress:
                on_progress(f"Expanding best approach (score={best.score:.2f}): {best.approach[:40]}")

            # Generate refined variants
            variants = self._refine_approach(best.experiment_plan, idea, domain, 2)
            for variant in variants:
                child = ExperimentNode(
                    id=node_counter, approach=variant.get("approach", ""),
                    experiment_plan=variant, depth=depth,
                    parent_id=best.id,
                )
                best.children_ids.append(node_counter)
                child.score = self._evaluate_approach(variant, idea, domain)
                child.status = "completed"
                nodes.append(child)
                node_counter += 1

        # Sort all by score
        nodes.sort(key=lambda n: n.score, reverse=True)

        if on_progress:
            # Guard the [0] access — nodes could have been emptied by
            # filtering above or remain empty if every evaluate call also
            # failed (rare but possible during proxy outages).
            if nodes:
                on_progress(
                    f"Tree search complete: {len(nodes)} nodes explored, "
                    f"best score={nodes[0].score:.2f}"
                )
            else:
                on_progress(
                    "Tree search complete: 0 nodes explored "
                    "(all approach evaluations failed)."
                )

        return nodes

    def _generate_approaches(
        self, idea: Dict, domain: str, n: int,
    ) -> List[Dict[str, Any]]:
        """Generate n different experimental approaches for an idea."""
        system = (
            "You are an expert experimental scientist. Given a research idea, "
            "propose multiple DIFFERENT experimental approaches to test it. "
            "Each approach should use different methods, datasets, or evaluation strategies.\n\n"
            f"Return JSON: {{\"approaches\": [{{\"approach\": \"brief description\", "
            f"\"method_variant\": \"...\", \"dataset\": \"...\", \"key_difference\": \"...\"}}]}} "
            f"with exactly {n} approaches."
        )
        user = (
            f"Domain: {domain}\n"
            f"Idea: {idea.get('title', '')}\n"
            f"Method: {idea.get('method', '')}\n"
            f"Hypothesis: {idea.get('hypothesis', '')}\n\n"
            f"Propose {n} different experimental approaches."
        )
        result = self._call_json(system, user, max_tokens=1500)
        approaches = result.get("approaches", [])
        # Ensure each has required fields
        for a in approaches:
            a.setdefault("approach", "default approach")
            a["idea_title"] = idea.get("title", "")
            a["hypothesis"] = idea.get("hypothesis", "")
        return approaches[:n]

    def _evaluate_approach(
        self, plan: Dict, idea: Dict, domain: str,
    ) -> float:
        """Score an experimental approach (0-1) without executing it."""
        system = (
            "You are a research evaluation expert. Score this experimental approach "
            "on a scale of 0.0 to 1.0 based on: feasibility (0.3), novelty (0.3), "
            "expected impact (0.2), and clarity (0.2).\n\n"
            "Return JSON: {\"score\": float, \"reasoning\": \"...\"}"
        )
        user = (
            f"Idea: {idea.get('title', '')}\n"
            f"Approach: {plan.get('approach', '')}\n"
            f"Method variant: {plan.get('method_variant', '')}\n"
            f"Dataset: {plan.get('dataset', '')}\n"
            f"Key difference: {plan.get('key_difference', '')}"
        )
        result = self._call_json(system, user, max_tokens=256, temperature=0.2)
        score = result.get("score", 0.5)
        return max(0.0, min(1.0, float(score)))

    def _refine_approach(
        self, plan: Dict, idea: Dict, domain: str, n: int,
    ) -> List[Dict[str, Any]]:
        """Generate refined variants of a promising approach."""
        system = (
            f"You are improving an experimental approach. Generate {n} refined "
            "variants that address potential weaknesses while keeping the core strengths.\n\n"
            f"Return JSON: {{\"variants\": [{{\"approach\": \"...\", \"improvement\": \"...\", "
            f"\"method_variant\": \"...\", \"dataset\": \"...\"}}]}}"
        )
        user = (
            f"Original approach: {plan.get('approach', '')}\n"
            f"Method: {plan.get('method_variant', '')}\n"
            f"For idea: {idea.get('title', '')}"
        )
        result = self._call_json(system, user, max_tokens=1000)
        variants = result.get("variants", [])
        for v in variants:
            v["idea_title"] = idea.get("title", "")
            v["hypothesis"] = idea.get("hypothesis", "")
        return variants[:n]
