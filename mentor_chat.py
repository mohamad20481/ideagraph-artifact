"""
mentor_chat.py - AI Research Mentor that knows your ideas and helps improve them.

The mentor has access to:
  - All ideas from the current/past runs
  - DAG summary (what papers exist)
  - Quality scores and probe feedback
  - User's bookmarks and notes

Provides:
  - Research methodology guidance
  - Idea improvement suggestions
  - Literature gap analysis
  - Experiment design help
  - Writing advice
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from agents.base_agent import BaseAgent
from intelligence import ActiveMentorTools


class ResearchMentor(BaseAgent):
    """AI research mentor with context about user's ideas and pipeline results.

    Enhanced with ActiveMentorTools: can actually analyze, improve, and compare
    ideas (not just chat about them).
    """

    def __init__(self):
        super().__init__(temperature=0.6)
        self._context: str = ""
        self._history: List[Dict[str, str]] = []
        self.tools = ActiveMentorTools()

    def set_context(self, results: Dict[str, Any] = None,
                    bookmarks: List[Dict] = None, topic: str = "") -> None:
        """Load research context from pipeline results + bookmarks."""
        parts = []

        if topic:
            parts.append(f"Current research topic: {topic}")

        if results:
            ideas = results.get("ideas", [])
            if ideas:
                top_ideas = sorted(ideas, key=lambda x: x.get("quality_score", 0), reverse=True)[:5]
                parts.append(f"\nUser has {len(ideas)} generated research ideas. Top 5:")
                for i, idea in enumerate(top_ideas, 1):
                    parts.append(
                        f"  {i}. \"{idea.get('title', '?')[:60]}\" "
                        f"(quality={idea.get('quality_score', 0):.2f}, "
                        f"type={idea.get('methodology_type', '?')}, "
                        f"novelty={idea.get('novelty_level', '?')})"
                    )
                    if idea.get("method"):
                        parts.append(f"     Method: {idea['method'][:100]}")

            coverage = results.get("coverage", 0)
            parts.append(f"\nArchive coverage: {coverage:.1%}")

            dag = results.get("dag_summary", {})
            if dag:
                parts.append(f"Knowledge DAG: {dag.get('node_count', 0)} papers, {dag.get('edge_count', 0)} edges, {dag.get('cluster_count', 0)} clusters")

            stats = results.get("stats", {})
            if stats:
                parts.append(f"Quality: mean={stats.get('quality_mean', 0):.3f}, max={stats.get('quality_max', 0):.3f}")

            review = results.get("final_review")
            if review:
                parts.append(f"Review: score={review.get('overall_score', 0):.1f}/10, decision={review.get('decision', '?')}")

        if bookmarks:
            parts.append(f"\nUser has {len(bookmarks)} bookmarked ideas:")
            for bk in bookmarks[:3]:
                parts.append(f"  - {bk.get('idea_title', '?')[:50]} (note: {bk.get('note', '')[:50]})")

        self._context = "\n".join(parts)

    def chat(self, user_message: str) -> str:
        """Send a message to the mentor and get a response."""
        system = (
            "You are an expert research mentor for the IdeaGraph platform. "
            "You help researchers improve their ideas, design experiments, "
            "identify literature gaps, and write better papers.\n\n"
            "You have full context about the user's research:\n"
            f"{self._context}\n\n"
            "Guidelines:\n"
            "- Be constructive and specific, not vague\n"
            "- Reference the user's actual ideas by title when relevant\n"
            "- Suggest concrete next steps, not just general advice\n"
            "- If asked about methodology, be technically precise\n"
            "- If asked to improve an idea, give specific changes\n"
            "- Keep answers concise (2-4 paragraphs max)\n"
        )

        # Build conversation history
        messages_text = ""
        for msg in self._history[-6:]:  # last 3 exchanges
            messages_text += f"\nUser: {msg['user']}\nMentor: {msg['assistant']}\n"

        user_prompt = f"{messages_text}\nUser: {user_message}\nMentor:"

        response = self._call(system, user_prompt, max_tokens=1024, temperature=0.6, use_cache=False)

        self._history.append({"user": user_message, "assistant": response})
        return response

    def suggest_improvements(self, idea: Dict[str, Any]) -> str:
        """Get specific improvement suggestions for an idea."""
        return self.chat(
            f"Please analyze this research idea and suggest 3 specific improvements:\n"
            f"Title: {idea.get('title', '?')}\n"
            f"Method: {idea.get('method', '?')}\n"
            f"Hypothesis: {idea.get('hypothesis', '?')}\n"
            f"Quality score: {idea.get('quality_score', 0):.2f}\n"
            f"Probe scores: {idea.get('probe_scores', {})}"
        )

    def explain_gap(self, topic: str) -> str:
        """Explain what research gaps exist in a topic."""
        return self.chat(
            f"Based on the knowledge DAG and generated ideas, what are the most "
            f"important research gaps in '{topic}' that haven't been addressed yet?"
        )

    # ── Active tools: analyze, improve, compare ideas in real-time ─────────

    def deep_analyze(self, idea: Dict[str, Any]) -> str:
        """Run full probe analysis and return detailed explanation."""
        analysis = self.tools.analyze_idea(idea)
        parts = [
            f"**Deep Analysis: {idea.get('title', '?')[:60]}**\n",
            f"Quality: {analysis['quality']:.2f} | Passed: {analysis['passed']}",
            f"\n{analysis['explanation']}",
        ]
        if analysis.get("feedback"):
            parts.append(f"\n**Detailed feedback:**\n{analysis['feedback'][:500]}")
        # Add next steps
        steps = self.tools.suggest_next_steps(idea)
        if steps:
            parts.append("\n**Recommended next steps:**")
            for i, step in enumerate(steps, 1):
                parts.append(f"{i}. {step}")
        return "\n".join(parts)

    def active_improve(self, idea: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Actually revise an idea using the ideation agent (not just suggest)."""
        return self.tools.improve_idea(idea)

    def active_compare(self, idea_a: Dict[str, Any], idea_b: Dict[str, Any]) -> str:
        """Compare two ideas across all dimensions with a verdict."""
        result = self.tools.compare_ideas(idea_a, idea_b)
        parts = [
            f"**Idea A:** {idea_a.get('title', '?')[:50]}",
            f"**Idea B:** {idea_b.get('title', '?')[:50]}",
            f"**Winner:** {result['overall_winner']}\n",
        ]
        for dim, data in result.get("dimensions", {}).items():
            marker = "**" if data["winner"] != "tie" else ""
            parts.append(
                f"  {dim}: A={data['A']:.2f} vs B={data['B']:.2f} "
                f"→ {marker}{data['winner']}{marker}"
            )
        return "\n".join(parts)

    def get_quick_suggestions(self) -> List[str]:
        """Return quick-access suggestion prompts."""
        return [
            "Which of my ideas has the most potential?",
            "How can I improve my highest-quality idea?",
            "What research gaps haven't been covered?",
            "Suggest an experiment design for my best idea",
            "What methodology should I try next?",
            "How does my work compare to the state of the art?",
            "Deep-analyze my best idea (run probes)",
            "Help me write an abstract for my best idea",
        ]
