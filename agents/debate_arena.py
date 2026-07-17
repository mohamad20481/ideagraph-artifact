"""
agents/debate_arena.py - Tournament-style debate engine.

Ideas compete head-to-head in a single-elimination bracket.
Two specialized agents argue for/against, a judge picks the winner.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import config
from agents.base_agent import BaseAgent
from agents.agent_memory import AgentMemoryManager
from models.idea import Idea


# ── Agent role personas ──────────────────────────────────────────────────────

DEBATE_ROLES: Dict[str, str] = {
    "innovator": (
        "You are the Innovator — a bold, creative researcher who champions "
        "unconventional ideas. You emphasize novelty, potential breakthroughs, "
        "and transformative impact. You see possibilities where others see obstacles."
    ),
    "critic": (
        "You are the Critic — a rigorous, skeptical scientist who demands evidence "
        "and identifies weaknesses. You challenge assumptions, find logical gaps, "
        "and ensure only robust ideas survive."
    ),
    "pragmatist": (
        "You are the Pragmatist — an engineering-minded researcher focused on "
        "implementation feasibility. You evaluate resource requirements, timeline, "
        "and practical steps needed to make an idea work."
    ),
    "connector": (
        "You are the Connector — a cross-domain thinker who links ideas to other "
        "fields. You find unexpected applications, draw analogies from distant "
        "disciplines, and identify synergies."
    ),
    "synthesizer": (
        "You are the Synthesizer — an integrative thinker who merges the best "
        "points from competing perspectives. You find common ground, resolve "
        "contradictions, and build consensus."
    ),
}

_ROLE_LIST = list(DEBATE_ROLES.keys())


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class DebateMatch:
    idea_a: Dict[str, Any]       # idea.to_dict()
    idea_b: Dict[str, Any]
    advocate_a_role: str
    advocate_b_role: str
    exchanges: List[Dict[str, Any]] = field(default_factory=list)
    judge_verdict: Dict[str, Any] = field(default_factory=dict)
    winner_side: str = ""        # "a" or "b"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "idea_a": self.idea_a,
            "idea_b": self.idea_b,
            "advocate_a_role": self.advocate_a_role,
            "advocate_b_role": self.advocate_b_role,
            "exchanges": self.exchanges,
            "judge_verdict": self.judge_verdict,
            "winner_side": self.winner_side,
        }


@dataclass
class Tournament:
    ideas: List[Idea]
    bracket: List[List[DebateMatch]] = field(default_factory=list)
    champion: Optional[Idea] = None
    champion_title: str = ""
    all_matches: List[DebateMatch] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entrant_count": len(self.ideas),
            "rounds": [
                [m.to_dict() for m in round_matches]
                for round_matches in self.bracket
            ],
            "champion_title": self.champion_title,
            "total_matches": len(self.all_matches),
        }


# ── Debate Arena ─────────────────────────────────────────────────────────────

class DebateArena(BaseAgent):
    """Runs a single-elimination tournament where ideas compete head-to-head."""

    def __init__(self, memory_manager: Optional[AgentMemoryManager] = None):
        super().__init__(temperature=0.7)
        self.memory = memory_manager

    def run_tournament(
        self, ideas: List[Idea], domain: str,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> Tournament:
        """Run a full single-elimination bracket tournament."""
        if len(ideas) < 2:
            t = Tournament(ideas=ideas)
            if ideas:
                t.champion = ideas[0]
                t.champion_title = ideas[0].title
            return t

        def progress(msg: str) -> None:
            if on_progress:
                on_progress(msg)

        # Seed: sort by quality descending
        seeded = sorted(ideas, key=lambda i: i.quality_score, reverse=True)

        # Pad to next power of 2 with None (byes)
        n = len(seeded)
        bracket_size = 1
        while bracket_size < n:
            bracket_size *= 2
        padded: List[Optional[Idea]] = list(seeded) + [None] * (bracket_size - n)

        # Standard seeding: #1 vs #N, #2 vs #N-1, etc.
        pairs: List[Tuple[Optional[Idea], Optional[Idea]]] = []
        for i in range(bracket_size // 2):
            pairs.append((padded[i], padded[bracket_size - 1 - i]))

        tournament = Tournament(ideas=ideas)
        current_round = pairs
        round_num = 0

        while len(current_round) > 0:
            round_num += 1
            progress(f"  Debate Round {round_num}: {len(current_round)} matches")
            round_matches: List[DebateMatch] = []
            winners: List[Idea] = []

            for idx, (a, b) in enumerate(current_round):
                # Handle byes
                if a is None and b is None:
                    continue
                if a is None:
                    winners.append(b)
                    continue
                if b is None:
                    winners.append(a)
                    continue

                progress(f"    Match {idx+1}: \"{a.title[:40]}\" vs \"{b.title[:40]}\"")
                match = self._run_match(a, b, domain)
                round_matches.append(match)
                tournament.all_matches.append(match)

                if match.winner_side == "a":
                    winners.append(a)
                else:
                    winners.append(b)

            tournament.bracket.append(round_matches)

            if len(winners) <= 1:
                break

            # Build next round pairs
            next_pairs = []
            for i in range(0, len(winners), 2):
                if i + 1 < len(winners):
                    next_pairs.append((winners[i], winners[i + 1]))
                else:
                    next_pairs.append((winners[i], None))
            current_round = next_pairs

        if winners:
            tournament.champion = winners[0]
            tournament.champion_title = winners[0].title

        # Assign debate ranks
        for idea in ideas:
            idea.debate_rank = None
        if tournament.champion:
            tournament.champion.debate_rank = 1
            tournament.champion.debate_score = 1.0

        # Build a title→Idea index ONCE so the rank-assignment loop is
        # O(rounds × matches) instead of O(rounds × matches × ideas). For an
        # N=64 tournament this is 6×32×64 ≈ 12 k iterations down to ~200.
        n_ideas = len(ideas)
        n_rounds = max(len(tournament.bracket), 1)
        title_to_idea = {idea.title: idea for idea in ideas}

        for round_idx, round_matches in enumerate(tournament.bracket):
            for match in round_matches:
                loser_title = (
                    match.idea_a["title"] if match.winner_side == "b"
                    else match.idea_b["title"]
                )
                idea = title_to_idea.get(loser_title)
                if idea is not None and idea.debate_rank is None:
                    idea.debate_rank = n_ideas - round_idx
                    idea.debate_score = round_idx / n_rounds

        progress(f"  Tournament complete! Champion: \"{tournament.champion_title[:50]}\"")
        return tournament

    def _run_match(self, idea_a: Idea, idea_b: Idea, domain: str) -> DebateMatch:
        """Run a head-to-head debate between two ideas."""
        role_a, role_b = self._assign_roles()

        match = DebateMatch(
            idea_a=idea_a.to_dict(),
            idea_b=idea_b.to_dict(),
            advocate_a_role=role_a,
            advocate_b_role=role_b,
        )

        for round_num in range(1, config.DEBATE_ROUNDS_PER_MATCH + 1):
            # Advocate A argues for idea_a
            arg_a = self._generate_argument(
                role=role_a, idea=idea_a, opponent_idea=idea_b,
                previous_exchanges=match.exchanges, domain=domain,
                side="for",
            )
            match.exchanges.append({
                "round": round_num, "side": "a", "role": role_a,
                "argument": arg_a,
            })

            # Advocate B argues for idea_b, responding to A
            arg_b = self._generate_argument(
                role=role_b, idea=idea_b, opponent_idea=idea_a,
                previous_exchanges=match.exchanges, domain=domain,
                side="for",
            )
            match.exchanges.append({
                "round": round_num, "side": "b", "role": role_b,
                "argument": arg_b,
            })

        # Judge evaluates
        verdict = self._judge_match(idea_a, idea_b, match.exchanges)
        match.judge_verdict = verdict
        match.winner_side = verdict.get("winner", "a")

        # Store memory if available
        if self.memory:
            exchange_text = "\n".join(
                f"[{e['role']}] {e['argument'][:200]}" for e in match.exchanges
            )
            winner_title = idea_a.title if match.winner_side == "a" else idea_b.title
            try:
                self.memory.extract_and_store_insights(
                    agent_role=role_a, domain=domain,
                    debate_exchange=exchange_text,
                    outcome=f"Winner: {winner_title}",
                )
            except Exception:
                pass

        return match

    def _generate_argument(
        self, role: str, idea: Idea, opponent_idea: Idea,
        previous_exchanges: List[Dict], domain: str, side: str,
    ) -> str:
        """Generate one argument from a role's perspective."""
        role_prompt = DEBATE_ROLES.get(role, DEBATE_ROLES["innovator"])

        # Build memory context if available
        memory_ctx = ""
        if self.memory:
            memory_ctx = self.memory.build_memory_context(role, domain)
            if memory_ctx:
                memory_ctx = f"\n\n{memory_ctx}"

        system = f"{role_prompt}{memory_ctx}"

        # Build exchange history
        history = ""
        if previous_exchanges:
            history = "\nPrevious arguments:\n"
            for ex in previous_exchanges[-4:]:
                history += f"  [{ex['role']}] {ex['argument'][:300]}\n"

        user = (
            f"You are arguing FOR this research idea in a debate:\n"
            f"Title: {idea.title}\n"
            f"Method: {idea.method}\n"
            f"Hypothesis: {idea.hypothesis}\n\n"
            f"The opposing idea is:\n"
            f"Title: {opponent_idea.title}\n"
            f"Method: {opponent_idea.method}\n\n"
            f"{history}\n"
            f"Make a compelling argument (2-3 paragraphs) for why YOUR idea is superior. "
            f"Address the opponent's strengths while highlighting your idea's advantages."
        )

        return self._call(system, user, max_tokens=512, temperature=0.7)

    def _judge_match(
        self, idea_a: Idea, idea_b: Idea, exchanges: List[Dict],
    ) -> Dict[str, Any]:
        """Impartial judge evaluates the debate and picks a winner."""
        system = (
            "You are an impartial research evaluation judge at a top AI conference. "
            "Evaluate both ideas based on the debate arguments. Be fair and analytical."
        )

        exchange_text = "\n\n".join(
            f"[Round {e['round']}, {e['role']} for Idea {'A' if e['side']=='a' else 'B'}]\n{e['argument']}"
            for e in exchanges
        )

        user = (
            f"IDEA A: {idea_a.title}\nMethod: {idea_a.method}\n\n"
            f"IDEA B: {idea_b.title}\nMethod: {idea_b.method}\n\n"
            f"DEBATE:\n{exchange_text}\n\n"
            "Judge the debate. Return JSON:\n"
            '{"winner": "a" or "b", "score_a": 0.0-1.0, "score_b": 0.0-1.0, '
            '"reasoning": "2-3 sentence justification", '
            '"criteria": {"novelty": "a or b", "feasibility": "a or b", '
            '"impact": "a or b", "clarity": "a or b"}}'
        )

        result = self._call_json(system, user, max_tokens=512, temperature=0.2)

        # Ensure valid winner
        if result.get("winner") not in ("a", "b"):
            result["winner"] = "a"  # default to higher-seeded idea

        return result

    def _assign_roles(self) -> Tuple[str, str]:
        """Pick two different roles for a match."""
        roles = random.sample(_ROLE_LIST, 2)
        return roles[0], roles[1]

    # ── Refinement & Consensus ──────────────────────────────────────────

    def refine_losers(
        self, tournament_data: Dict[str, Any], domain: str,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Refine losing ideas using debate feedback. Returns list of improved idea dicts."""
        refined = []
        rounds = tournament_data.get("rounds", [])

        for rd in rounds:
            for match in rd:
                winner_side = match.get("winner_side", "a")
                loser = match["idea_b"] if winner_side == "a" else match["idea_a"]
                verdict = match.get("judge_verdict", {})
                feedback = verdict.get("reasoning", "No feedback available.")

                if on_progress:
                    on_progress(f"Refining: {loser.get('title', '')[:40]}...")

                try:
                    revised = self._refine_idea(loser, feedback, domain)
                    if revised:
                        refined.append(revised)
                except Exception:
                    continue

        return refined

    def _refine_idea(
        self, idea_dict: Dict[str, Any], feedback: str, domain: str,
    ) -> Optional[Dict[str, Any]]:
        """Revise a losing idea based on debate feedback."""
        system = (
            "You are a research idea improver. Given an idea that lost a debate "
            "and the judge's feedback, revise the idea to address the weaknesses "
            "while preserving its strengths. Return the improved idea as JSON with "
            "all 7 fields: title, motivation, method, hypothesis, resources, "
            "expected_outcome, risk_assessment, plus methodology_type and novelty_level."
        )
        user = (
            f"Domain: {domain}\n\n"
            f"ORIGINAL IDEA:\n"
            f"Title: {idea_dict.get('title', '')}\n"
            f"Motivation: {idea_dict.get('motivation', '')}\n"
            f"Method: {idea_dict.get('method', '')}\n"
            f"Hypothesis: {idea_dict.get('hypothesis', '')}\n\n"
            f"JUDGE FEEDBACK (why it lost):\n{feedback}\n\n"
            f"Revise this idea to address the weaknesses. Keep the core insight "
            f"but strengthen the method, hypothesis, and feasibility."
        )

        result = self._call_json(system, user, max_tokens=1024, temperature=0.6)
        if result and result.get("title"):
            result["generation"] = idea_dict.get("generation", 0) + 1
            result["parent_title"] = idea_dict.get("title", "")
            result["source_strategy"] = "refined"
            result["quality_score"] = idea_dict.get("quality_score", 0)  # will be re-scored
            return result
        return None

    def synthesize_consensus(
        self, match: Dict[str, Any], domain: str,
    ) -> Optional[Dict[str, Any]]:
        """Merge the best parts of both debated ideas into a hybrid."""
        idea_a = match.get("idea_a", {})
        idea_b = match.get("idea_b", {})
        exchanges = match.get("exchanges", [])
        verdict = match.get("judge_verdict", {})

        # Build exchange summary
        exchange_text = ""
        for ex in exchanges[:4]:  # limit to avoid token overflow
            side = "A" if ex.get("side") == "a" else "B"
            exchange_text += f"[{ex.get('role', '?')} for {side}]: {ex.get('argument', '')[:200]}\n\n"

        system = (
            "You are a research synthesizer. Given two research ideas that competed "
            "in a debate and the arguments made, create a NEW hybrid idea that "
            "combines the strongest elements of both. The hybrid should be stronger "
            "than either original. Return JSON with all 7 fields: title, motivation, "
            "method, hypothesis, resources, expected_outcome, risk_assessment, "
            "plus methodology_type and novelty_level."
        )
        user = (
            f"Domain: {domain}\n\n"
            f"IDEA A: {idea_a.get('title', '')}\n"
            f"Method A: {idea_a.get('method', '')}\n"
            f"Hypothesis A: {idea_a.get('hypothesis', '')}\n\n"
            f"IDEA B: {idea_b.get('title', '')}\n"
            f"Method B: {idea_b.get('method', '')}\n"
            f"Hypothesis B: {idea_b.get('hypothesis', '')}\n\n"
            f"DEBATE HIGHLIGHTS:\n{exchange_text}\n"
            f"JUDGE REASONING: {verdict.get('reasoning', '')[:300]}\n\n"
            f"Create a hybrid idea combining the best of both."
        )

        result = self._call_json(system, user, max_tokens=1024, temperature=0.7)
        if result and result.get("title"):
            result["source_strategy"] = "consensus"
            result["generation"] = 0
            return result
        return None
