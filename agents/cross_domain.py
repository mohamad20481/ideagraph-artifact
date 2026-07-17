"""
agents/cross_domain.py - Cross-domain idea synthesis.

Combines ideas from different research domains to generate novel hybrid ideas.
Domain A agents challenge Domain B ideas and vice versa.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import config
from agents.base_agent import BaseAgent
from agents.agent_memory import AgentMemoryManager
from agents.execution_critic import ExecutionCritic
from models.idea import Idea


@dataclass
class CrossDomainResult:
    source_domains: List[str] = field(default_factory=list)
    source_ideas: List[Dict[str, Any]] = field(default_factory=list)
    hybrid_ideas: List[Dict[str, Any]] = field(default_factory=list)
    challenge_exchanges: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_domains": self.source_domains,
            "source_ideas": self.source_ideas,
            "hybrid_ideas": self.hybrid_ideas,
            "challenge_exchanges": self.challenge_exchanges,
        }


class CrossDomainSynthesizer(BaseAgent):
    """Synthesizes ideas across different domain runs."""

    def __init__(self, memory_manager: Optional[AgentMemoryManager] = None):
        super().__init__(temperature=0.8)
        self.memory = memory_manager

    def synthesize(
        self, runs: List[Dict[str, Any]],
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> CrossDomainResult:
        """Cross-pollinate ideas from multiple saved pipeline results.

        Args:
            runs: List of pipeline result dicts (each has 'topic', 'ideas', etc.)
        """
        def progress(msg: str) -> None:
            if on_progress:
                on_progress(msg)

        result = CrossDomainResult()
        result.source_domains = [r.get("topic", "Unknown")[:60] for r in runs]

        # Extract top ideas from each run
        all_domain_ideas: List[List[Dict]] = []
        for run in runs:
            ideas = run.get("ideas", [])
            # Top 3 by quality (O(n log 3) via heapq instead of O(n log n) sort)
            top = heapq.nlargest(3, ideas, key=lambda x: x.get("quality_score", 0))
            all_domain_ideas.append(top)
            result.source_ideas.extend(top)

        # Cross-domain challenges for each pair of domains
        pairs = self._select_domain_pairs(runs)
        progress(f"  Cross-domain: {len(pairs)} domain pairs to synthesize")

        for i, (run_a, run_b) in enumerate(pairs):
            domain_a = run_a.get("topic", "Domain A")[:60]
            domain_b = run_b.get("topic", "Domain B")[:60]
            progress(f"    Pair {i+1}: \"{domain_a[:30]}\" × \"{domain_b[:30]}\"")

            ideas_a = heapq.nlargest(2, run_a.get("ideas", []), key=lambda x: x.get("quality_score", 0))
            ideas_b = heapq.nlargest(2, run_b.get("ideas", []), key=lambda x: x.get("quality_score", 0))

            # Challenge: A critiques B's top idea
            if ideas_b:
                exchange = self._cross_challenge(ideas_b[0], domain_a, domain_b)
                result.challenge_exchanges.append({
                    "challenger_domain": domain_a,
                    "defender_domain": domain_b,
                    "idea_title": ideas_b[0].get("title", ""),
                    **exchange,
                })

            # Generate hybrid idea
            if ideas_a and ideas_b:
                idea_a_dict = ideas_a[0]
                idea_b_dict = ideas_b[0]
                challenge_ctx = exchange.get("defense", "") if ideas_b else ""

                hybrid = self._generate_hybrid(
                    idea_a_dict, idea_b_dict, domain_a, domain_b, challenge_ctx,
                )
                if hybrid:
                    result.hybrid_ideas.append(hybrid)
                    progress(f"      Hybrid idea: \"{hybrid.get('title', 'Untitled')[:50]}\"")

        progress(f"  Cross-domain complete: {len(result.hybrid_ideas)} hybrid ideas generated")
        return result

    def _select_domain_pairs(
        self, runs: List[Dict[str, Any]],
    ) -> List[tuple]:
        """Generate all unique pairs of runs."""
        pairs = []
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                pairs.append((runs[i], runs[j]))
        return pairs[:6]  # cap at 6 pairs

    def _cross_challenge(
        self, idea_dict: Dict[str, Any], challenger_domain: str, defender_domain: str,
    ) -> Dict[str, str]:
        """One round of cross-domain challenge."""
        system = (
            f"You are a researcher from the field of '{challenger_domain}'. "
            f"You are critiquing an idea from '{defender_domain}'."
        )

        # Challenger argues
        challenge_prompt = (
            f"From the perspective of {challenger_domain}, critique this idea:\n"
            f"Title: {idea_dict.get('title', '')}\n"
            f"Method: {idea_dict.get('method', '')}\n"
            f"Hypothesis: {idea_dict.get('hypothesis', '')}\n\n"
            f"Identify weaknesses and suggest how methods from {challenger_domain} "
            f"could improve this idea. Be specific and constructive."
        )
        challenge = self._call(system, challenge_prompt, max_tokens=512, temperature=0.7)

        # Defender responds
        defense_system = (
            f"You are a researcher from '{defender_domain}'. "
            f"Defend your idea and adapt it incorporating cross-domain feedback."
        )
        defense_prompt = (
            f"Your idea:\nTitle: {idea_dict.get('title', '')}\n"
            f"Method: {idea_dict.get('method', '')}\n\n"
            f"Critique from {challenger_domain}:\n{challenge}\n\n"
            f"Defend your idea and propose specific adaptations that incorporate "
            f"the best insights from {challenger_domain}."
        )
        defense = self._call(defense_system, defense_prompt, max_tokens=512, temperature=0.6)

        return {"challenge": challenge, "defense": defense}

    def _generate_hybrid(
        self, idea_a: Dict, idea_b: Dict,
        domain_a: str, domain_b: str, challenge_ctx: str,
    ) -> Optional[Dict[str, Any]]:
        """Generate a hybrid idea combining insights from two domain ideas."""
        system = (
            "You are an interdisciplinary research scientist who excels at "
            "combining methods and insights from different fields to create "
            "groundbreaking new research directions."
        )

        user = (
            f"Combine these two ideas from different domains into a novel hybrid:\n\n"
            f"DOMAIN A ({domain_a[:40]}):\n"
            f"  Title: {idea_a.get('title', '')}\n"
            f"  Method: {idea_a.get('method', '')}\n\n"
            f"DOMAIN B ({domain_b[:40]}):\n"
            f"  Title: {idea_b.get('title', '')}\n"
            f"  Method: {idea_b.get('method', '')}\n\n"
            f"Cross-domain insights:\n{challenge_ctx[:500]}\n\n"
            f"Generate a hybrid research idea as JSON:\n"
            '{"title": "...", "motivation": "...", "method": "...", '
            '"hypothesis": "...", "resources": "...", "expected_outcome": "...", '
            '"risk_assessment": "...", "methodology_type": "interdisciplinary_bridge", '
            '"novelty_level": "substantial", "source_strategy": "cross_domain"}'
        )

        result = self._call_json(system, user, max_tokens=1024, temperature=0.8)
        if result and result.get("title"):
            result["quality_score"] = 0.0  # not yet probed
            result["probe_scores"] = {}
            return result
        return None
