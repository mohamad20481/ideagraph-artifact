"""
agents/agent_memory.py - Persistent memory system for debate agents.

Stores reusable insights, argumentation patterns, and domain knowledge
across sessions so agents become smarter over time.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from agents.base_agent import BaseAgent
import db


class AgentMemoryManager(BaseAgent):
    """Manages persistent agent memory in SQLite."""

    def __init__(self, user_id: int):
        super().__init__(temperature=0.2)
        self.user_id = user_id

    # ── Store ─────────────────────────────────────────────────────────────────

    def store_memory(
        self, agent_role: str, memory_type: str, domain: str,
        pattern: str, context: str, outcome: str, confidence: float = 0.5,
    ) -> int:
        """Store a new memory entry. Returns the memory row id."""
        content = {
            "pattern": pattern,
            "context": context,
            "outcome": outcome,
            "confidence": confidence,
        }
        return db.save_agent_memory(
            self.user_id, agent_role, memory_type, domain, content, confidence,
        )

    # ── Recall ────────────────────────────────────────────────────────────────

    def recall(self, agent_role: str, domain: str, limit: int = 5) -> List[Dict]:
        """Retrieve relevant memories for an agent in a domain."""
        return db.query_agent_memory(self.user_id, agent_role, domain, limit)

    def build_memory_context(self, agent_role: str, domain: str) -> str:
        """Format recalled memories as a prompt string for system prompt injection.
        Returns empty string if no memories exist."""
        memories = self.recall(agent_role, domain, limit=5)
        if not memories:
            return ""

        lines = ["From your past experience in this domain:"]
        for m in memories:
            c = m["content"]
            lines.append(f"- {c.get('pattern', '')} (confidence: {c.get('confidence', 0.5):.1f})")
        return "\n".join(lines)

    # ── Extract insights after debate ─────────────────────────────────────────

    def extract_and_store_insights(
        self, agent_role: str, domain: str,
        debate_exchange: str, outcome: str,
    ) -> None:
        """Use LLM to extract reusable patterns from a debate exchange and store them."""
        system = (
            "You are an AI research meta-analyst. Analyze the following debate exchange "
            "and extract reusable argumentation patterns and domain insights."
        )
        user = (
            f"Domain: {domain}\nAgent role: {agent_role}\nOutcome: {outcome}\n\n"
            f"Debate exchange:\n{debate_exchange}\n\n"
            "Extract 1-3 key insights as JSON:\n"
            '{"insights": [{"pattern": "...", "context": "...", "confidence": 0.0-1.0}]}'
        )

        try:
            result = self._call_json(system, user, max_tokens=512)
            insights = result.get("insights", [])
            for ins in insights[:3]:
                self.store_memory(
                    agent_role=agent_role,
                    memory_type="debate_insight",
                    domain=domain,
                    pattern=ins.get("pattern", ""),
                    context=ins.get("context", ""),
                    outcome=outcome,
                    confidence=float(ins.get("confidence", 0.5)),
                )
        except Exception:
            pass  # memory extraction is best-effort
