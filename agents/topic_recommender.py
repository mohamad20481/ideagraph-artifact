"""
agents/topic_recommender.py - Smart topic recommendation based on portfolio gap analysis.
"""

from __future__ import annotations
from typing import Any, Dict, List

from agents.base_agent import BaseAgent
from models.idea import METHODOLOGY_TYPES, NOVELTY_LEVELS


class TopicRecommender(BaseAgent):
    """Analyzes gaps in the user's idea portfolio and suggests new research directions."""

    def __init__(self):
        super().__init__(temperature=0.8)

    def recommend(
        self, ideas: List[Dict[str, Any]], domains: List[str],
    ) -> List[Dict[str, str]]:
        """Analyze portfolio gaps and suggest 3-5 unexplored directions.

        Returns list of {"topic": str, "rationale": str, "gap_type": str}
        """
        portfolio = self._summarize_portfolio(ideas, domains)

        system = (
            "You are a research strategy advisor who identifies gaps and opportunities "
            "in a researcher's idea portfolio. You suggest new research directions that "
            "would maximize coverage and impact."
        )
        user = (
            f"Here is a researcher's current idea portfolio:\n\n{portfolio}\n\n"
            "Based on this portfolio, suggest 3-5 unexplored research directions that would:\n"
            "1. Fill methodology or novelty gaps\n"
            "2. Bridge unconnected domains\n"
            "3. Explore high-impact emerging areas\n\n"
            "Return JSON: {\"recommendations\": [{\"topic\": \"...\", \"rationale\": \"...\", "
            "\"gap_type\": \"methodology_gap|novelty_gap|domain_gap|emerging_area\"}]}"
        )

        result = self._call_json(system, user)
        return result.get("recommendations", [])

    def _summarize_portfolio(
        self, ideas: List[Dict], domains: List[str],
    ) -> str:
        """Build a concise text summary of the portfolio for the LLM."""
        # Single pass: tally counts, fill the methodology×novelty grid, and
        # accumulate quality stats. The previous version walked `ideas` once
        # per methodology, again per novelty, again to build filled_cells, and
        # one more time for scores — 11 full passes for a 7×3 grid + scores.
        method_counts = {mt: 0 for mt in METHODOLOGY_TYPES}
        novelty_counts = {nl: 0 for nl in NOVELTY_LEVELS}
        filled_cells: set = set()
        q_total = 0.0
        q_count = 0
        for i in ideas:
            mt = i.get("methodology_type", "")
            nl = i.get("novelty_level", "")
            if mt in method_counts:
                method_counts[mt] += 1
            if nl in novelty_counts:
                novelty_counts[nl] += 1
            if mt and nl:
                filled_cells.add((mt, nl))
            q = i.get("quality_score", 0)
            if q:
                q_total += q
                q_count += 1

        avg_q = q_total / q_count if q_count else 0

        # Empty cells — still O(7*3) = 21 ops, dominated by string formatting.
        empty_cells: List[str] = []
        for mt in METHODOLOGY_TYPES:
            mt_display = mt.replace("_", " ")
            for nl in NOVELTY_LEVELS:
                if (mt, nl) not in filled_cells:
                    empty_cells.append(f"{mt_display} × {nl}")

        lines = [
            f"Total ideas: {len(ideas)}",
            f"Domains covered: {', '.join(domains[:5])}",
            f"Average quality: {avg_q:.3f}",
            "",
            "Methodology distribution:",
        ]
        for mt, count in method_counts.items():
            lines.append(f"  {mt.replace('_', ' ')}: {count} ideas")

        lines.append("\nNovelty distribution:")
        for nl, count in novelty_counts.items():
            lines.append(f"  {nl}: {count} ideas")

        if empty_cells:
            lines.append(f"\nEmpty cells ({len(empty_cells)}/{len(METHODOLOGY_TYPES) * len(NOVELTY_LEVELS)}):")
            for cell in empty_cells[:10]:
                lines.append(f"  - {cell}")

        return "\n".join(lines)
