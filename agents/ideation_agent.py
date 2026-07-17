"""
agents/ideation_agent.py - Generates research ideas using three strategies:
  A. Frontier Extension
  B. Cross-Cluster Bridging
  C. Gap-Filling

Prompts follow Appendix N of the IdeaGraph paper.

Optimisations vs original:
  - Adaptive temperature: higher creativity (0.85) for radical novelty cells,
    lower (0.65) for incremental cells.  Radical ideas need more exploration;
    incremental extensions benefit from focused, coherent output.
  - Shared paper_summary() helper eliminates duplicate code across B and C.
"""

from __future__ import annotations
import random
import threading
from typing import Any, Dict, List, Optional, Tuple

from models.dag import KnowledgeDAG
from models.idea import Idea, METHODOLOGY_TYPES, NOVELTY_LEVELS
from .base_agent import BaseAgent


def _novelty_temperature(novelty_idx: int) -> float:
    """
    Map novelty index → LLM temperature.

    novelty_idx 0 (Incremental) → 0.65: focused, coherent extensions
    novelty_idx 1 (Moderate)    → 0.70: balanced exploration
    novelty_idx 2 (Radical)     → 0.85: high creativity for disruptive ideas
    """
    _TEMPS = [0.65, 0.70, 0.85]
    return _TEMPS[novelty_idx] if 0 <= novelty_idx < len(_TEMPS) else 0.70


def _paper_summary(papers) -> str:
    """Compact list of paper titles+years for use in prompts."""
    return "; ".join(f'"{p.title}" ({p.year})' for p in papers)


# ── Shared prompt fragments ────────────────────────────────────────────────────
_SYSTEM = (
    "You are an expert research scientist specialising in ideation. "
    "Generate a concrete, novel, and executable research idea. "
    "Output ONLY valid JSON with exactly these keys: "
    "title, motivation, method, hypothesis, resources, expected_outcome, "
    "risk_assessment, source_strategy, methodology_type, novelty_level. "
    f"methodology_type must be one of: {', '.join(METHODOLOGY_TYPES)}. "
    f"novelty_level must be one of: {', '.join(NOVELTY_LEVELS)}."
)

_SYSTEM_BATCH = (
    "You are an expert research scientist specialising in ideation. "
    "Generate TWO distinct, concrete, novel, and executable research ideas. "
    'Output ONLY valid JSON: {"ideas": [idea1, idea2]} where each idea has keys: '
    "title, motivation, method, hypothesis, resources, expected_outcome, "
    "risk_assessment, source_strategy, methodology_type, novelty_level. "
    f"methodology_type must be one of: {', '.join(METHODOLOGY_TYPES)}. "
    f"novelty_level must be one of: {', '.join(NOVELTY_LEVELS)}. "
    "The two ideas should take DIFFERENT technical approaches to fill the same research niche."
)

_IDEA_SCHEMA = (
    "Return JSON:\n"
    "{\n"
    '  "title": "<concise title>",\n'
    '  "motivation": "<why this matters>",\n'
    '  "method": "<concrete technical approach>",\n'
    '  "hypothesis": "<testable prediction>",\n'
    '  "resources": "<datasets, compute, software needed>",\n'
    '  "expected_outcome": "<measurable results>",\n'
    '  "risk_assessment": "<main risks and mitigations>",\n'
    '  "source_strategy": "<A|B|C>",\n'
    f'  "methodology_type": "<one of: {", ".join(METHODOLOGY_TYPES)}>",\n'
    f'  "novelty_level": "<one of: {", ".join(NOVELTY_LEVELS)}>"\n'
    "}"
)


# ── Domain-specific methodology hints ─────────────────────────────────────────
_METHOD_HINTS = {
    "empirical_study": (
        "Design a rigorous experiment with clear baselines, controlled variables, "
        "statistical significance tests, and reproducible results on standard benchmarks."
    ),
    "theoretical_analysis": (
        "Develop a formal mathematical framework with provable guarantees, "
        "bounds, or convergence properties. Include theorems and proofs sketch."
    ),
    "system_design": (
        "Propose a novel architecture, framework, or system with clear components, "
        "interfaces, and scalability analysis. Include a system diagram description."
    ),
    "dataset_creation": (
        "Design a new dataset with clear annotation guidelines, quality metrics, "
        "diversity analysis, and comparison to existing datasets."
    ),
    "survey_meta_analysis": (
        "Propose a systematic review with clear inclusion criteria, taxonomy, "
        "quantitative meta-analysis, and identification of open problems."
    ),
    "tool_library": (
        "Design a reusable software library or tool with clear API, documentation plan, "
        "benchmarks against existing tools, and adoption strategy."
    ),
    "interdisciplinary_bridge": (
        "Connect two distinct research fields by identifying shared structures, "
        "transferable methods, or analogous problems. Explain why the bridge is non-obvious."
    ),
}


# ── Multi-perspective expert personas ─────────────────────────────────────────
_EXPERT_PERSONAS = [
    {"role": "methodologist", "instruction": "Focus on methodological rigor. Propose ideas with clear experimental design, proper controls, and statistical validity."},
    {"role": "theorist", "instruction": "Focus on theoretical contribution. Propose ideas that advance fundamental understanding with provable properties or novel frameworks."},
    {"role": "practitioner", "instruction": "Focus on practical impact. Propose ideas that solve real-world problems and can be deployed in production within 6 months."},
    {"role": "contrarian", "instruction": "Challenge conventional wisdom. Propose ideas that go against the mainstream approach — what if the standard assumption is wrong?"},
    {"role": "synthesizer", "instruction": "Find unexpected connections. Propose ideas that bridge two fields that haven't been connected before."},
    {"role": "futurist", "instruction": "Think 5 years ahead. Propose ideas that anticipate where the field is heading, not where it is now."},
]

# ── Topic decomposition templates ─────────────────────────────────────────────
_DECOMPOSITION_PROMPT = (
    "You are a research strategist. Given a broad research topic, decompose it into "
    "3-5 specific, actionable sub-problems that each represent a distinct research "
    "opportunity. Output ONLY valid JSON: {\"sub_problems\": [\"...\", \"...\"]}"
)


class IdeationAgent(BaseAgent):

    def __init__(self) -> None:
        super().__init__(temperature=0.7)
        self._lock = threading.Lock()
        self._probe_failure_counts: Dict[str, int] = {
            "code": 0, "dataset": 0, "constraint": 0, "novelty": 0,
        }
        self._exemplar_ideas: List[Dict[str, str]] = []
        self._topic_sub_problems: List[str] = []
        self._current_persona_idx: int = 0
        self._literature_context: str = ""  # injected paper abstracts from DAG

    def record_probe_failure(self, probe_name: str) -> None:
        """Track which probes fail so prompts can address weak dimensions (thread-safe)."""
        with self._lock:
            if probe_name in self._probe_failure_counts:
                self._probe_failure_counts[probe_name] += 1

    def set_exemplars(self, exemplars: List[Dict[str, str]]) -> None:
        """Set top-quality archived ideas as few-shot examples (thread-safe)."""
        with self._lock:
            self._exemplar_ideas = exemplars[:2]

    def set_literature_context(self, papers: list) -> None:
        """Inject real paper titles+abstracts from the DAG for grounded generation (thread-safe)."""
        if not papers:
            with self._lock:
                self._literature_context = ""
            return
        parts = ["Relevant papers in this field (use these as grounding — your idea must go BEYOND these):"]
        for p in papers[:5]:
            title = p.get("title", "") if isinstance(p, dict) else str(p)
            abstract = p.get("abstract", "")[:150] if isinstance(p, dict) else ""
            year = p.get("year", "") if isinstance(p, dict) else ""
            parts.append(f"  - \"{title}\" ({year}): {abstract}")
        with self._lock:
            self._literature_context = "\n".join(parts)

    def _build_smart_context(self, target_cell: Tuple[int, int]) -> str:
        """Build adaptive context: domain hints + failure awareness + few-shot (thread-safe)."""
        # Snapshot mutable state under the lock once, then build strings freely.
        with self._lock:
            failure_counts = dict(self._probe_failure_counts)
            exemplars = list(self._exemplar_ideas)
            lit_ctx = self._literature_context

        parts = []
        method_idx = target_cell[0]
        method_type = METHODOLOGY_TYPES[method_idx] if method_idx < len(METHODOLOGY_TYPES) else ""

        # 1. Domain-specific methodology hint
        hint = _METHOD_HINTS.get(method_type, "")
        if hint:
            parts.append(f"Methodology guidance: {hint}")

        # 2. Failure-aware prompting: address the most-failed probe
        if sum(failure_counts.values()) >= 3:
            worst_probe = max(failure_counts, key=failure_counts.get)
            failure_hints = {
                "code": "IMPORTANT: Ensure the method can be implemented in Python with standard ML libraries (PyTorch, sklearn, numpy).",
                "dataset": "IMPORTANT: Use ONLY publicly available datasets (HuggingFace, UCI, Kaggle). Specify exact dataset names.",
                "constraint": "IMPORTANT: The experiment must run on a single GPU (A100) within 1 day. Keep scale realistic.",
                "novelty": "IMPORTANT: This MUST be clearly different from existing work. Explain what specific gap this fills that no prior work addresses.",
            }
            if worst_probe in failure_hints:
                parts.append(failure_hints[worst_probe])

        # 3. Few-shot exemplars from archive
        if exemplars:
            parts.append("Here are examples of HIGH-QUALITY ideas (use as reference for quality, NOT to copy):")
            for i, ex in enumerate(exemplars[:2]):
                parts.append(
                    f"  Example {i+1}: \"{ex.get('title', '')[:60]}\" — "
                    f"Method: {ex.get('method', '')[:100]}"
                )

        # 4. Literature context (real papers from DAG)
        if lit_ctx:
            parts.append(lit_ctx)

        # 5. Specificity reminder (always present to reduce vague ideas)
        parts.append(
            "CRITICAL: Be SPECIFIC. Name exact algorithms, specific datasets (by name), "
            "concrete metrics (e.g., 'BLEU score', 'F1 on SQuAD'), and quantitative targets "
            "(e.g., '5% improvement over baseline'). Vague ideas will be rejected."
        )

        return "\n".join(parts) if parts else ""

    # ─────────────────────────────────────────────────────────────────────────
    # Topic Decomposition (split broad topic → actionable sub-problems)
    # ─────────────────────────────────────────────────────────────────────────
    def decompose_topic(self, topic: str) -> List[str]:
        """
        Decompose a broad topic into 3-5 specific sub-problems.
        Cached per topic to avoid repeated LLM calls.
        """
        if self._topic_sub_problems:
            return self._topic_sub_problems

        result = self._call_json(
            _DECOMPOSITION_PROMPT,
            f"Topic: {topic}\n\nDecompose into 3-5 specific sub-problems.",
            max_tokens=512, temperature=0.5,
        )
        subs = result.get("sub_problems", [])
        if isinstance(subs, list) and subs:
            self._topic_sub_problems = [str(s)[:200] for s in subs[:5]]
        return self._topic_sub_problems

    # ─────────────────────────────────────────────────────────────────────────
    # Multi-Perspective Generation (rotate expert personas)
    # ─────────────────────────────────────────────────────────────────────────
    def _get_next_persona(self) -> Dict[str, str]:
        """Rotate through expert personas for diverse perspectives."""
        persona = _EXPERT_PERSONAS[self._current_persona_idx % len(_EXPERT_PERSONAS)]
        self._current_persona_idx += 1
        return persona

    def _build_persona_system_prompt(self, target_cell: Tuple[int, int]) -> str:
        """Build a system prompt enhanced with the current expert persona."""
        persona = self._get_next_persona()
        base = (
            f"You are an expert research scientist specialising in ideation, "
            f"taking the role of a {persona['role']}. {persona['instruction']}\n\n"
            "Generate a concrete, novel, and executable research idea. "
            "Output ONLY valid JSON with exactly these keys: "
            "title, motivation, method, hypothesis, resources, expected_outcome, "
            "risk_assessment, source_strategy, methodology_type, novelty_level. "
            f"methodology_type must be one of: {', '.join(METHODOLOGY_TYPES)}. "
            f"novelty_level must be one of: {', '.join(NOVELTY_LEVELS)}."
        )
        return base

    # ─────────────────────────────────────────────────────────────────────────
    # Chain-of-Thought Structured Ideation
    # ─────────────────────────────────────────────────────────────────────────
    def generate_with_chain_of_thought(
        self,
        topic: str,
        target_cell: Tuple[int, int],
        context: str = "",
    ) -> Optional[Idea]:
        """
        Generate an idea using structured chain-of-thought reasoning.

        Steps the LLM through:
          1. Identify the key challenge in this topic
          2. List 3 possible approaches
          3. Pick the most novel one
          4. Flesh it out into a full idea
        """
        method_hint = METHODOLOGY_TYPES[target_cell[0]] if target_cell[0] < len(METHODOLOGY_TYPES) else ""
        novelty_hint = NOVELTY_LEVELS[target_cell[1]] if target_cell[1] < len(NOVELTY_LEVELS) else ""
        smart_ctx = self._build_smart_context(target_cell)

        system = self._build_persona_system_prompt(target_cell)
        user = (
            f"Research topic: {topic}\n"
            f"Target methodology: {method_hint}\n"
            f"Target novelty: {novelty_hint}\n\n"
            f"{smart_ctx}\n\n"
            f"{context}\n\n"
            "Think step by step:\n"
            "1. What is the most important unsolved challenge in this topic?\n"
            "2. List 3 possible technical approaches to address it.\n"
            "3. Which approach is most novel and feasible?\n"
            "4. Flesh it out into a complete research idea.\n\n"
            "Now generate the idea as JSON with all required fields.\n\n"
            + _IDEA_SCHEMA
        )

        temp = _novelty_temperature(target_cell[1])
        result = self._call_json(system, user, max_tokens=1024, temperature=temp)
        return _dict_to_idea(result, "A")

    # ─────────────────────────────────────────────────────────────────────────
    # Self-Improvement Loop (generate → critique → refine)
    # ─────────────────────────────────────────────────────────────────────────
    def generate_with_self_improvement(
        self,
        topic: str,
        target_cell: Tuple[int, int],
        context: str = "",
        rounds: int = 2,
    ) -> Optional[Idea]:
        """
        Generate an idea, then self-critique and refine it for N rounds.
        Each round: critique the current idea → produce an improved version.
        """
        # Round 0: initial generation
        idea = self.generate_with_chain_of_thought(topic, target_cell, context)
        if idea is None:
            return None

        for _round in range(rounds):
            # Critique
            critique_system = (
                "You are a harsh but constructive research reviewer. "
                "Given a research idea, identify its 3 biggest weaknesses. "
                "Output ONLY JSON: {\"weaknesses\": [\"...\"], \"suggestions\": [\"...\"]}"
            )
            critique_user = (
                f"Idea: {idea.title}\n"
                f"Method: {idea.method}\n"
                f"Hypothesis: {idea.hypothesis}\n"
                f"Rate the novelty, feasibility, and clarity. What's wrong?"
            )
            critique = self._call_json(critique_system, critique_user, max_tokens=512, temperature=0.3)
            weaknesses = critique.get("weaknesses", [])
            suggestions = critique.get("suggestions", [])

            if not weaknesses:
                break  # idea is good enough

            # Refine based on critique
            refine_feedback = (
                "Weaknesses found:\n"
                + "\n".join(f"- {w}" for w in weaknesses[:3])
                + "\nSuggested fixes:\n"
                + "\n".join(f"- {s}" for s in suggestions[:3])
            )
            revised = self.revise_idea(idea, refine_feedback, idea.quality_score or 0.5)
            if revised and len(revised.method or "") > len(idea.method or ""):
                idea = revised  # only keep if revised version is more detailed

        return idea

    # ─────────────────────────────────────────────────────────────────────────
    # Strategy A – Frontier Extension
    # ─────────────────────────────────────────────────────────────────────────
    def generate_strategy_a(
        self,
        dag: KnowledgeDAG,
        frontier_paper_id: str,
        target_cell: Tuple[int, int],
        avoid_context: str = "",
    ) -> Optional[Idea]:
        """
        Propose the next step(s) from a frontier paper.
        Target cell hints guide the methodology_type and novelty_level.
        """
        paper = dag.get_paper(frontier_paper_id)
        if paper is None:
            return None

        method_hint = METHODOLOGY_TYPES[target_cell[0]]
        novelty_hint = NOVELTY_LEVELS[target_cell[1]]

        # Build context from the paper's cluster
        cluster_context = ""
        if paper.cluster_id is not None:
            meta = dag.cluster_metadata.get(paper.cluster_id, {})
            cluster_context = (
                f"Cluster theme: {meta.get('theme', '')}\n"
                f"Open questions: {'; '.join(meta.get('open_questions', []))}\n"
            )

        smart_context = self._build_smart_context(target_cell)

        user = (
            "## Strategy A: Frontier Extension\n\n"
            f"Frontier paper:\n"
            f"  Title: {paper.title} ({paper.year})\n"
            f"  Abstract: {paper.abstract[:250]}\n\n"
            f"{cluster_context}"
            f"Target methodology type (hint): {method_hint}\n"
            f"Target novelty level (hint): {novelty_hint}\n\n"
            f"{smart_context}\n\n"
            "Propose ONE next research idea that directly extends this frontier paper. "
            "Be specific about the technical approach and what makes it novel.\n\n"
            + _IDEA_SCHEMA
            + (f"\n\n{avoid_context}" if avoid_context else "")
        )

        temp = _novelty_temperature(target_cell[1])
        result = self._call_json(_SYSTEM, user, max_tokens=1024, temperature=temp)
        return _dict_to_idea(result, "A")

    # ─────────────────────────────────────────────────────────────────────────
    # Strategy B – Cross-Cluster Bridging
    # ─────────────────────────────────────────────────────────────────────────
    def generate_strategy_b(
        self,
        dag: KnowledgeDAG,
        cluster_a_id: int,
        cluster_b_id: int,
        target_cell: Tuple[int, int],
        avoid_context: str = "",
    ) -> Optional[Idea]:
        """
        Combine insights from two different research clusters.
        """
        method_hint = METHODOLOGY_TYPES[target_cell[0]]
        novelty_hint = NOVELTY_LEVELS[target_cell[1]]

        meta_a = dag.cluster_metadata.get(cluster_a_id, {})
        meta_b = dag.cluster_metadata.get(cluster_b_id, {})

        papers_a = dag.get_papers_in_cluster(cluster_a_id)[:3]
        papers_b = dag.get_papers_in_cluster(cluster_b_id)[:3]

        smart_context = self._build_smart_context(target_cell)

        user = (
            "## Strategy B: Cross-Cluster Bridging\n\n"
            f"Cluster A — Theme: {meta_a.get('theme', f'Cluster {cluster_a_id}')}\n"
            f"  Key papers: {_paper_summary(papers_a)}\n"
            f"  Open questions: {'; '.join(meta_a.get('open_questions', []))}\n\n"
            f"Cluster B — Theme: {meta_b.get('theme', f'Cluster {cluster_b_id}')}\n"
            f"  Key papers: {_paper_summary(papers_b)}\n"
            f"  Open questions: {'; '.join(meta_b.get('open_questions', []))}\n\n"
            f"Target methodology type (hint): {method_hint}\n"
            f"Target novelty level (hint): {novelty_hint}\n\n"
            f"{smart_context}\n\n"
            "Design ONE research idea that creatively bridges insights from BOTH clusters. "
            "Explain the synergy and why the combination is more powerful than either alone.\n\n"
            + _IDEA_SCHEMA
            + (f"\n\n{avoid_context}" if avoid_context else "")
        )

        temp = _novelty_temperature(target_cell[1])
        result = self._call_json(_SYSTEM, user, max_tokens=1024, temperature=temp)
        return _dict_to_idea(result, "B")

    # ─────────────────────────────────────────────────────────────────────────
    # Strategy C – Gap-Filling
    # ─────────────────────────────────────────────────────────────────────────
    def generate_strategy_c(
        self,
        dag: KnowledgeDAG,
        cluster_a_id: int,
        cluster_b_id: int,
        target_cell: Tuple[int, int],
        avoid_context: str = "",
    ) -> Optional[Idea]:
        """
        Fill a structural gap between two clusters with no existing edge.
        """
        method_hint = METHODOLOGY_TYPES[target_cell[0]]
        novelty_hint = NOVELTY_LEVELS[target_cell[1]]

        meta_a = dag.cluster_metadata.get(cluster_a_id, {})
        meta_b = dag.cluster_metadata.get(cluster_b_id, {})

        papers_a = dag.get_papers_in_cluster(cluster_a_id)[:3]
        papers_b = dag.get_papers_in_cluster(cluster_b_id)[:3]

        smart_context = self._build_smart_context(target_cell)

        user = (
            "## Strategy C: Gap-Filling\n\n"
            f"Cluster A — Theme: {meta_a.get('theme', f'Cluster {cluster_a_id}')}\n"
            f"  Key papers: {_paper_summary(papers_a)}\n\n"
            f"Cluster B — Theme: {meta_b.get('theme', f'Cluster {cluster_b_id}')}\n"
            f"  Key papers: {_paper_summary(papers_b)}\n\n"
            "These two clusters have NO existing research connections. "
            "There is a structural gap in the literature.\n\n"
            f"Target methodology type (hint): {method_hint}\n"
            f"Target novelty level (hint): {novelty_hint}\n\n"
            f"{smart_context}\n\n"
            "Propose ONE research idea that fills this gap — a study that could plausibly "
            "create a link between these two previously disconnected research streams. "
            "This should be a genuinely novel direction.\n\n"
            + _IDEA_SCHEMA
            + (f"\n\n{avoid_context}" if avoid_context else "")
        )

        temp = _novelty_temperature(target_cell[1])
        result = self._call_json(_SYSTEM, user, max_tokens=1024, temperature=temp)
        return _dict_to_idea(result, "C")

    # ─────────────────────────────────────────────────────────────────────────
    # Batch generation (2-for-1): single LLM call returns 2 ideas
    # ─────────────────────────────────────────────────────────────────────────
    def generate_batch_strategy_a(
        self,
        dag: KnowledgeDAG,
        frontier_paper_id: str,
        target_cell: Tuple[int, int],
        avoid_context: str = "",
    ) -> List[Idea]:
        """Generate 2 ideas from one frontier paper in a single LLM call."""
        paper = dag.get_paper(frontier_paper_id)
        if paper is None:
            return []

        method_hint = METHODOLOGY_TYPES[target_cell[0]]
        novelty_hint = NOVELTY_LEVELS[target_cell[1]]

        cluster_context = ""
        if paper.cluster_id is not None:
            meta = dag.cluster_metadata.get(paper.cluster_id, {})
            cluster_context = (
                f"Cluster theme: {meta.get('theme', '')}\n"
                f"Open questions: {'; '.join(meta.get('open_questions', []))}\n"
            )

        user = (
            "## Strategy A: Frontier Extension (generate 2 ideas)\n\n"
            f"Frontier paper:\n"
            f"  Title: {paper.title} ({paper.year})\n"
            f"  Abstract: {paper.abstract[:400]}\n\n"
            f"{cluster_context}"
            f"Target methodology type (hint): {method_hint}\n"
            f"Target novelty level (hint): {novelty_hint}\n\n"
            "Propose TWO distinct research ideas that extend this frontier paper. "
            "Each should take a DIFFERENT technical approach.\n\n"
            'Return JSON: {"ideas": [{idea1}, {idea2}]}'
            + (f"\n\n{avoid_context}" if avoid_context else "")
        )

        temp = _novelty_temperature(target_cell[1])
        return self._parse_batch_response(
            self._call(
                _SYSTEM_BATCH, user, max_tokens=1536,
                temperature=temp, json_mode=True,
            ),
            "A",
        )

    def _parse_batch_response(self, raw: str, strategy: str) -> List[Idea]:
        """Parse a batch LLM response containing 2 idea dicts.

        Handles multiple formats:
          - {"ideas": [{...}, {...}]}   (json_mode wraps in object)
          - [{...}, {...}]              (raw array)
          - {...}                       (single idea)
        """
        if not raw:
            return []
        import json as _json
        import re as _re
        try:
            parsed = _json.loads(raw)
        except _json.JSONDecodeError:
            # Try extracting a JSON array or object from the response
            match = _re.search(r"[\[{][\s\S]*[\]}]", raw)
            if match:
                try:
                    parsed = _json.loads(match.group(0))
                except _json.JSONDecodeError:
                    return []
            else:
                return []
        # Unwrap {"ideas": [...]} wrapper from json_mode
        if isinstance(parsed, dict) and "ideas" in parsed and isinstance(parsed["ideas"], list):
            parsed = parsed["ideas"]
        elif isinstance(parsed, dict):
            # Single idea object — check if it has a "title" key
            if "title" in parsed:
                parsed = [parsed]
            else:
                # Try any list value in the dict
                for v in parsed.values():
                    if isinstance(v, list) and len(v) > 0:
                        parsed = v
                        break
                else:
                    return []
        if not isinstance(parsed, list):
            return []
        ideas = []
        for d in parsed[:2]:
            if isinstance(d, dict):
                idea = _dict_to_idea(d, strategy)
                if idea is not None:
                    ideas.append(idea)
        return ideas

    # ─────────────────────────────────────────────────────────────────────────
    # Revision after probe failure
    # ─────────────────────────────────────────────────────────────────────────
    def revise_idea(
        self,
        idea: Idea,
        feedback: str,
        quality_score: float = 0.5,
    ) -> Optional[Idea]:
        """
        Revise a failing idea based on execution-probe feedback.
        Preserves the original strategy and behavioural coordinates.

        Adaptive temperature based on quality_score:
          < 0.30  → 0.75  badly failing: need a substantially different approach
          < 0.45  → 0.60  moderately failing: moderate creativity
          ≥ 0.45  → 0.45  close to passing: fine-tune, don't stray far

        Lower quality → higher temperature → more exploratory revision.
        """
        if quality_score < 0.30:
            temp = 0.75
        elif quality_score < 0.45:
            temp = 0.60
        else:
            temp = 0.45

        user = (
            "## Idea Revision\n\n"
            "The following research idea FAILED execution probes:\n\n"
            f"{idea.to_prompt_str()}\n\n"
            f"Probe feedback (most critical issues first):\n{feedback}\n\n"
            "Please revise the idea to address the specific issues raised. "
            "Keep the core insight but make it more feasible, better resourced, "
            "and clearer in its novelty claims.\n\n"
            + _IDEA_SCHEMA
        )
        result = self._call_json(_SYSTEM, user, max_tokens=1024, temperature=temp)
        if not result:
            return None
        revised = _dict_to_idea(result, idea.source_strategy)
        if revised is None:
            return None
        # Preserve original behavioural classification if revision doesn't provide one
        if revised.methodology_type is None:
            revised.methodology_type = idea.methodology_type
        if revised.novelty_level is None:
            revised.novelty_level = idea.novelty_level
        return revised


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────
def _dict_to_idea(d: Dict[str, Any], strategy: str) -> Optional[Idea]:
    if not d:
        return None

    methodology_type = d.get("methodology_type")
    if methodology_type not in METHODOLOGY_TYPES:
        methodology_type = METHODOLOGY_TYPES[0]

    novelty_level = d.get("novelty_level")
    if novelty_level not in NOVELTY_LEVELS:
        novelty_level = NOVELTY_LEVELS[0]

    return Idea(
        title=str(d.get("title", "Untitled Idea")),
        motivation=str(d.get("motivation", "")),
        method=str(d.get("method", "")),
        hypothesis=str(d.get("hypothesis", "")),
        resources=str(d.get("resources", "")),
        expected_outcome=str(d.get("expected_outcome", "")),
        risk_assessment=str(d.get("risk_assessment", "")),
        source_strategy=strategy,
        methodology_type=methodology_type,
        novelty_level=novelty_level,
    )
