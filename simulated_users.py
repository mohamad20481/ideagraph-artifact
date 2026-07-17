"""
simulated_users.py - Smart synthetic agents that simulate real users using IdeaGraph.

Features:
  - 8 user personas (PhD student, postdoc, professor, industry researcher, etc.)
  - Topic pools per field (ML, biology, physics, CS, medicine, etc.)
  - Runs the pipeline autonomously with realistic topics
  - Collects synthetic feedback (quality ratings, satisfaction)
  - Auto-optimizes config based on aggregated feedback
  - Saves all interaction data for offline analysis
  - Exports to CSV, JSON, and Markdown reports

Usage:
    from simulated_users import AgentSimulator

    sim = AgentSimulator()
    sim.run_batch(n_agents=10, topics_per_agent=2)
    report = sim.export_report()
"""

from __future__ import annotations

import csv
import json
import os
import random
import statistics
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# USER PERSONAS
# ═══════════════════════════════════════════════════════════════════════════

PERSONAS = {
    "phd_student_cs": {
        "name": "PhD Student (Computer Science)",
        "field": "computer_science",
        "experience": "junior",
        "budget": 1.0,
        "iterations": 8,
        "novelty_preference": "substantial",  # wants breakthrough ideas
        "quality_threshold": 0.5,
        "feedback_style": "eager",  # rates high when novel
        "topics": [
            "transformer attention mechanisms for long documents",
            "self-supervised learning for low-resource NLP",
            "graph neural networks for protein structure prediction",
            "reinforcement learning for combinatorial optimization",
            "diffusion models for 3D scene generation",
            "federated learning with privacy guarantees",
            "neural architecture search for edge devices",
            "multimodal foundation models for robotics",
        ],
    },
    "postdoc_bio": {
        "name": "Postdoc (Computational Biology)",
        "field": "biology",
        "experience": "mid",
        "budget": 2.0,
        "iterations": 12,
        "novelty_preference": "moderate",  # wants feasible ideas
        "quality_threshold": 0.6,
        "feedback_style": "critical",
        "topics": [
            "single-cell RNA sequencing batch effect correction",
            "protein language models for drug discovery",
            "CRISPR gene editing off-target prediction",
            "spatial transcriptomics cell-cell interaction inference",
            "microbiome dysbiosis biomarker discovery",
            "cryo-EM protein structure refinement",
            "evolutionary genomics of antibiotic resistance",
            "cancer neoantigen prediction from tumor sequencing",
        ],
    },
    "professor_physics": {
        "name": "Professor (Physics)",
        "field": "physics",
        "experience": "senior",
        "budget": 3.0,
        "iterations": 15,
        "novelty_preference": "substantial",
        "quality_threshold": 0.7,
        "feedback_style": "demanding",
        "topics": [
            "quantum error correction for fault-tolerant computing",
            "dark matter detection via axion haloscopes",
            "topological superconductivity in twisted bilayer graphene",
            "neutrino mass hierarchy measurement techniques",
            "gravitational wave astronomy with pulsar timing arrays",
            "Majorana fermions in semiconductor nanowires",
            "quantum supremacy benchmarks for NISQ devices",
            "high-temperature superconductivity in hydride compounds",
        ],
    },
    "industry_researcher_ml": {
        "name": "Industry Researcher (ML/AI)",
        "field": "industry_ml",
        "experience": "mid",
        "budget": 1.5,
        "iterations": 10,
        "novelty_preference": "moderate",
        "quality_threshold": 0.55,
        "feedback_style": "practical",  # cares about feasibility
        "topics": [
            "LLM inference acceleration with speculative decoding",
            "efficient fine-tuning with LoRA variants",
            "vector database retrieval optimization",
            "multi-agent LLM orchestration frameworks",
            "on-device AI model compression",
            "long-context attention with state space models",
            "reinforcement learning from human feedback scaling",
            "synthetic data generation for fine-tuning",
        ],
    },
    "grad_student_medicine": {
        "name": "Grad Student (Medicine)",
        "field": "medicine",
        "experience": "junior",
        "budget": 1.0,
        "iterations": 8,
        "novelty_preference": "moderate",
        "quality_threshold": 0.5,
        "feedback_style": "curious",
        "topics": [
            "AI-assisted early detection of pancreatic cancer",
            "wearable sensors for continuous glucose monitoring",
            "personalized cancer immunotherapy selection",
            "antibiotic resistance prediction from genomic data",
            "Alzheimer's biomarkers in cerebrospinal fluid",
            "telemedicine for rural mental health care",
            "gut-brain axis in neurological disorders",
            "machine learning for radiology triage",
        ],
    },
    "researcher_materials": {
        "name": "Researcher (Materials Science)",
        "field": "materials",
        "experience": "mid",
        "budget": 1.5,
        "iterations": 10,
        "novelty_preference": "substantial",
        "quality_threshold": 0.6,
        "feedback_style": "technical",
        "topics": [
            "machine learning for perovskite solar cell design",
            "high-entropy alloys for extreme environments",
            "2D materials beyond graphene for electronics",
            "solid-state battery electrolytes discovery",
            "metal-organic frameworks for CO2 capture",
            "biodegradable polymers for medical implants",
            "nanostructured catalysts for green hydrogen",
            "topological insulators for spintronic devices",
        ],
    },
    "social_scientist": {
        "name": "Social Scientist",
        "field": "social_science",
        "experience": "mid",
        "budget": 1.0,
        "iterations": 8,
        "novelty_preference": "moderate",
        "quality_threshold": 0.5,
        "feedback_style": "interpretive",
        "topics": [
            "social media misinformation cascade dynamics",
            "algorithmic bias in hiring decisions",
            "urban gentrification and displacement patterns",
            "climate change policy public opinion shifts",
            "remote work impact on organizational culture",
            "digital divide in education outcomes",
            "political polarization on social platforms",
            "cryptocurrency adoption socioeconomic factors",
        ],
    },
    "interdisciplinary_researcher": {
        "name": "Interdisciplinary Researcher",
        "field": "interdisciplinary",
        "experience": "senior",
        "budget": 2.0,
        "iterations": 12,
        "novelty_preference": "substantial",
        "quality_threshold": 0.65,
        "feedback_style": "exploratory",
        "topics": [
            "AI for climate change mitigation strategies",
            "neuroscience-inspired machine learning architectures",
            "bio-inspired robotics and swarm intelligence",
            "quantum computing for drug discovery",
            "synthetic biology for sustainable manufacturing",
            "computational sociology with agent-based models",
            "cognitive science and human-AI collaboration",
            "network science in epidemic spreading",
        ],
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AgentRun:
    """A single pipeline run by a synthetic agent."""
    agent_id: str
    persona: str
    topic: str
    timestamp: str
    duration_s: float
    budget_usd: float
    iterations: int

    # Results
    ideas_count: int = 0
    coverage: float = 0.0
    quality_mean: float = 0.0
    quality_max: float = 0.0
    cost_usd: float = 0.0
    api_calls: int = 0
    cache_hit_rate: float = 0.0
    error: Optional[str] = None

    # Agent feedback
    satisfaction: float = 0.0  # 0-1
    would_recommend: bool = False
    feedback_text: str = ""
    pain_points: List[str] = field(default_factory=list)
    liked_features: List[str] = field(default_factory=list)


@dataclass
class OptimizationSuggestion:
    """Config tweak derived from agent feedback."""
    parameter: str
    old_value: Any
    new_value: Any
    reason: str
    expected_improvement: float  # 0-1


# ═══════════════════════════════════════════════════════════════════════════
# SYNTHETIC AGENT
# ═══════════════════════════════════════════════════════════════════════════

class SyntheticAgent:
    """A synthetic user agent that runs the pipeline and provides feedback."""

    def __init__(self, persona_key: str, agent_id: str = None):
        if persona_key not in PERSONAS:
            raise ValueError(f"Unknown persona: {persona_key}")
        self.persona_key = persona_key
        self.persona = PERSONAS[persona_key]
        self.agent_id = agent_id or f"{persona_key}_{random.randint(1000, 9999)}"
        self.runs: List[AgentRun] = []

    def pick_topic(self) -> str:
        """Choose a topic from the persona's pool."""
        return random.choice(self.persona["topics"])

    def run_pipeline(self, topic: str = None, use_real_pipeline: bool = False) -> AgentRun:
        """
        Run the pipeline on a topic. If use_real_pipeline=False, uses mock data
        for fast testing. Set to True to call the actual AutomatedScientist.
        """
        topic = topic or self.pick_topic()
        start = time.time()
        timestamp = datetime.now().isoformat()

        run = AgentRun(
            agent_id=self.agent_id,
            persona=self.persona_key,
            topic=topic,
            timestamp=timestamp,
            duration_s=0,
            budget_usd=self.persona["budget"],
            iterations=self.persona["iterations"],
        )

        try:
            if use_real_pipeline:
                results = self._run_real_pipeline(topic)
            else:
                results = self._run_mock_pipeline(topic)

            run.duration_s = time.time() - start
            run.ideas_count = len(results.get("ideas", []))
            run.coverage = results.get("coverage", 0.0)
            stats = results.get("stats", {})
            run.quality_mean = stats.get("quality_mean", 0.0)
            run.quality_max = stats.get("quality_max", 0.0)
            run.cost_usd = stats.get("estimated_cost_usd", 0.0)
            call_metrics = results.get("call_metrics", {})
            run.api_calls = call_metrics.get("calls", 0)
            run.cache_hit_rate = (call_metrics.get("cache_hits", 0) / max(call_metrics.get("calls", 1), 1))

            # Generate feedback
            self._generate_feedback(run, results)

        except Exception as e:
            run.error = str(e)[:200]
            run.duration_s = time.time() - start

        self.runs.append(run)
        return run

    def _run_mock_pipeline(self, topic: str) -> Dict[str, Any]:
        """Fast mock pipeline for testing without API calls."""
        time.sleep(random.uniform(0.1, 0.3))  # simulate processing

        # Quality distribution depends on persona preference
        base_q = {"substantial": 0.55, "moderate": 0.65, "incremental": 0.70}
        mean_q = base_q.get(self.persona["novelty_preference"], 0.6)
        noise = random.uniform(-0.15, 0.15)

        n_ideas = random.randint(8, self.persona["iterations"] * 3)
        ideas = []
        for i in range(n_ideas):
            q = max(0.1, min(0.95, random.gauss(mean_q + noise, 0.15)))
            ideas.append({
                "title": f"Mock idea {i+1} on {topic[:40]}",
                "quality_score": q,
                "methodology_type": random.choice([
                    "empirical_study", "theoretical_analysis", "system_design",
                    "dataset_creation", "survey_meta_analysis",
                ]),
                "novelty_level": self.persona["novelty_preference"],
                "source_strategy": random.choice(["A", "B", "C"]),
                "motivation": f"Investigate {topic}",
                "method": f"Apply novel approach to {topic}",
                "hypothesis": f"Our method improves over baseline on {topic}",
            })

        qualities = [i["quality_score"] for i in ideas]
        return {
            "ideas": ideas,
            "coverage": len(ideas) / 21.0,  # 7 methods × 3 novelty
            "stats": {
                "quality_mean": statistics.mean(qualities) if qualities else 0,
                "quality_max": max(qualities) if qualities else 0,
                "estimated_cost_usd": random.uniform(0.001, 0.05),
                "iterations": random.randint(3, self.persona["iterations"]),
            },
            "call_metrics": {
                "calls": random.randint(20, 100),
                "cache_hits": random.randint(0, 30),
                "errors": random.randint(0, 3),
            },
            "topic": topic,
        }

    def _run_real_pipeline(self, topic: str) -> Dict[str, Any]:
        """Run the actual automated scientist pipeline."""
        from pipeline_v2 import AutomatedScientist

        scientist = AutomatedScientist()
        return scientist.run(
            topic=topic,
            budget_usd=self.persona["budget"],
            max_ideation_iterations=self.persona["iterations"],
            max_scientist_iterations=1,
            execution_timeout=300,
            on_progress=None,
            debate_enabled=False,
        )

    def _generate_feedback(self, run: AgentRun, results: Dict[str, Any]) -> None:
        """Simulate realistic user feedback based on persona."""
        style = self.persona["feedback_style"]
        threshold = self.persona["quality_threshold"]

        # Satisfaction based on quality vs threshold
        quality_ratio = run.quality_mean / max(threshold, 0.01)
        base_satisfaction = min(1.0, quality_ratio * 0.8)

        # Style modifiers
        style_modifier = {
            "eager": 0.15,       # PhD student: enthusiastic
            "critical": -0.10,   # Postdoc: critical eye
            "demanding": -0.15,  # Professor: very demanding
            "practical": 0.0,    # Industry: balanced
            "curious": 0.10,
            "technical": -0.05,
            "interpretive": 0.05,
            "exploratory": 0.10,
        }.get(style, 0.0)

        run.satisfaction = max(0.0, min(1.0, base_satisfaction + style_modifier + random.uniform(-0.1, 0.1)))
        run.would_recommend = run.satisfaction >= 0.6

        # Pain points (realistic complaints)
        if run.ideas_count < 5:
            run.pain_points.append("Too few ideas generated")
        if run.quality_mean < 0.4:
            run.pain_points.append("Ideas lack depth and specificity")
        if run.duration_s > 120:
            run.pain_points.append("Pipeline too slow")
        if run.cost_usd > self.persona["budget"] * 0.9:
            run.pain_points.append("Expensive for the value")
        if run.coverage < 0.3:
            run.pain_points.append("Low diversity across methodology types")
        if run.error:
            run.pain_points.append(f"Pipeline failed: {run.error[:50]}")

        # Liked features
        if run.quality_max >= 0.7:
            run.liked_features.append("At least one high-quality idea found")
        if run.cache_hit_rate > 0.2:
            run.liked_features.append("Fast re-runs via caching")
        if run.ideas_count >= 15:
            run.liked_features.append("Good variety of ideas")
        if run.cost_usd < self.persona["budget"] * 0.3:
            run.liked_features.append("Very cost-effective")

        # Feedback text (templated based on style)
        templates = {
            "eager": [
                "Wow, these are some interesting directions!",
                "I love the variety here — exciting to explore!",
                "This could actually work for my thesis chapter.",
            ],
            "critical": [
                "Some ideas are promising but others lack rigor.",
                "The feasibility analysis needs improvement.",
                "I appreciate the breadth but want more depth.",
            ],
            "demanding": [
                "Most of these wouldn't pass peer review.",
                "The novelty claims are often overstated.",
                "I need citation-grounded novelty assessments.",
            ],
            "practical": [
                "Good starting points for proof-of-concept.",
                "I can see deploying 2-3 of these in production.",
                "Need more attention to scalability concerns.",
            ],
            "curious": [
                "Learning a lot from exploring these directions!",
                "The cross-domain connections are intriguing.",
                "Would love to see more examples.",
            ],
            "technical": [
                "Good technical depth on some ideas.",
                "Need more specific experimental protocols.",
                "The resource estimates seem realistic.",
            ],
            "interpretive": [
                "Interesting framings — some novel angles.",
                "The qualitative analysis is missing.",
                "Would benefit from more theoretical grounding.",
            ],
            "exploratory": [
                "Great for broadening my research horizons!",
                "The interdisciplinary connections are valuable.",
                "Would like more risk assessment detail.",
            ],
        }
        run.feedback_text = random.choice(templates.get(style, ["Neutral feedback."]))


# ═══════════════════════════════════════════════════════════════════════════
# AGENT SIMULATOR (batch runner)
# ═══════════════════════════════════════════════════════════════════════════

class AgentSimulator:
    """Runs batches of synthetic agents and collects aggregate data."""

    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir or (Path(__file__).parent / "output" / "agent_simulations"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.all_runs: List[AgentRun] = []
        self.agents: List[SyntheticAgent] = []

    def run_batch(
        self,
        n_agents: int = 10,
        topics_per_agent: int = 2,
        personas: List[str] = None,
        use_real_pipeline: bool = False,
        on_progress: Callable[[str], None] = None,
    ) -> List[AgentRun]:
        """Run a batch of agents simulating user sessions."""
        personas = personas or list(PERSONAS.keys())
        runs = []

        total = n_agents * topics_per_agent
        done = 0

        for i in range(n_agents):
            persona = random.choice(personas)
            agent = SyntheticAgent(persona, agent_id=f"agent_{i+1:03d}")
            self.agents.append(agent)

            for j in range(topics_per_agent):
                topic = agent.pick_topic()
                run = agent.run_pipeline(topic, use_real_pipeline=use_real_pipeline)
                runs.append(run)
                self.all_runs.append(run)
                done += 1

                if on_progress:
                    on_progress(
                        f"[{done}/{total}] {agent.agent_id} ({persona}): "
                        f"'{topic[:40]}' → {run.ideas_count} ideas, "
                        f"q={run.quality_mean:.2f}, sat={run.satisfaction:.2f}"
                    )

        return runs

    def aggregate_stats(self) -> Dict[str, Any]:
        """Compute aggregate statistics across all runs."""
        if not self.all_runs:
            return {}

        successful = [r for r in self.all_runs if not r.error]

        return {
            "total_runs": len(self.all_runs),
            "successful_runs": len(successful),
            "failed_runs": len(self.all_runs) - len(successful),
            "total_ideas_generated": sum(r.ideas_count for r in successful),
            "avg_quality_mean": statistics.mean([r.quality_mean for r in successful]) if successful else 0,
            "avg_quality_max": statistics.mean([r.quality_max for r in successful]) if successful else 0,
            "avg_satisfaction": statistics.mean([r.satisfaction for r in successful]) if successful else 0,
            "recommend_rate": sum(1 for r in successful if r.would_recommend) / max(len(successful), 1),
            "total_cost_usd": sum(r.cost_usd for r in successful),
            "avg_duration_s": statistics.mean([r.duration_s for r in successful]) if successful else 0,
            "personas_used": list(set(r.persona for r in self.all_runs)),
            "unique_topics": len(set(r.topic for r in self.all_runs)),
        }

    def pain_point_analysis(self) -> Dict[str, int]:
        """Count frequency of pain points across all runs."""
        pains = {}
        for run in self.all_runs:
            for pain in run.pain_points:
                pains[pain] = pains.get(pain, 0) + 1
        return dict(sorted(pains.items(), key=lambda x: x[1], reverse=True))

    def liked_features_analysis(self) -> Dict[str, int]:
        """Count frequency of liked features."""
        likes = {}
        for run in self.all_runs:
            for like in run.liked_features:
                likes[like] = likes.get(like, 0) + 1
        return dict(sorted(likes.items(), key=lambda x: x[1], reverse=True))

    def generate_optimization_suggestions(self) -> List[OptimizationSuggestion]:
        """Analyze feedback and suggest config tweaks."""
        suggestions = []
        stats = self.aggregate_stats()
        pains = self.pain_point_analysis()

        if not stats:
            return suggestions

        # Low quality → increase iterations
        if stats.get("avg_quality_mean", 0) < 0.5:
            suggestions.append(OptimizationSuggestion(
                parameter="MAX_ITERATIONS",
                old_value=10, new_value=15,
                reason=f"Avg quality {stats['avg_quality_mean']:.2f} below 0.5 threshold",
                expected_improvement=0.15,
            ))

        # Too few ideas → lower quality floor
        if pains.get("Too few ideas generated", 0) >= 2:
            suggestions.append(OptimizationSuggestion(
                parameter="quality_floor_base",
                old_value=0.3, new_value=0.25,
                reason=f"{pains['Too few ideas generated']} agents reported too few ideas",
                expected_improvement=0.20,
            ))

        # Too slow → reduce iterations or enable parallelism
        if pains.get("Pipeline too slow", 0) >= 2:
            suggestions.append(OptimizationSuggestion(
                parameter="MAX_PARALLEL_CELLS",
                old_value=6, new_value=8,
                reason=f"{pains['Pipeline too slow']} agents complained about speed",
                expected_improvement=0.25,
            ))

        # Too expensive → enable compression + cascade routing
        if pains.get("Expensive for the value", 0) >= 2:
            suggestions.append(OptimizationSuggestion(
                parameter="ENABLE_CASCADE_ROUTING",
                old_value=False, new_value=True,
                reason="Multiple agents found the cost high — use cheap models for simple tasks",
                expected_improvement=0.40,
            ))

        # Low coverage → boost diversity
        if pains.get("Low diversity across methodology types", 0) >= 2:
            suggestions.append(OptimizationSuggestion(
                parameter="CURIOSITY_WEIGHT",
                old_value=0.3, new_value=0.5,
                reason="Low diversity reported — increase curiosity-driven exploration",
                expected_improvement=0.20,
            ))

        # Satisfaction < 0.5 → multiple tweaks
        if stats.get("avg_satisfaction", 0) < 0.5:
            suggestions.append(OptimizationSuggestion(
                parameter="ENABLE_DIALECTICAL",
                old_value=False, new_value=True,
                reason="Low satisfaction — enable dialectical synthesis for deeper ideas",
                expected_improvement=0.15,
            ))

        return suggestions

    def export_csv(self, filename: str = None) -> str:
        """Export all runs as CSV for Excel/analysis."""
        filename = filename or f"agent_runs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        path = self.output_dir / filename

        if not self.all_runs:
            return str(path)

        fieldnames = list(asdict(self.all_runs[0]).keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for run in self.all_runs:
                row = asdict(run)
                row["pain_points"] = " | ".join(row["pain_points"])
                row["liked_features"] = " | ".join(row["liked_features"])
                writer.writerow(row)
        return str(path)

    def export_json(self, filename: str = None) -> str:
        """Export all runs + aggregate stats as JSON."""
        filename = filename or f"agent_runs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path = self.output_dir / filename

        data = {
            "timestamp": datetime.now().isoformat(),
            "aggregate_stats": self.aggregate_stats(),
            "pain_points": self.pain_point_analysis(),
            "liked_features": self.liked_features_analysis(),
            "optimization_suggestions": [asdict(s) for s in self.generate_optimization_suggestions()],
            "runs": [asdict(r) for r in self.all_runs],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        return str(path)

    def export_markdown_report(self, filename: str = None) -> str:
        """Export a human-readable Markdown report."""
        filename = filename or f"agent_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        path = self.output_dir / filename

        stats = self.aggregate_stats()
        pains = self.pain_point_analysis()
        likes = self.liked_features_analysis()
        suggestions = self.generate_optimization_suggestions()

        lines = [
            "# Agent Simulation Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "## Summary",
            f"- **Total runs:** {stats.get('total_runs', 0)}",
            f"- **Successful:** {stats.get('successful_runs', 0)}",
            f"- **Failed:** {stats.get('failed_runs', 0)}",
            f"- **Total ideas generated:** {stats.get('total_ideas_generated', 0)}",
            f"- **Average quality:** {stats.get('avg_quality_mean', 0):.3f}",
            f"- **Average satisfaction:** {stats.get('avg_satisfaction', 0):.3f}",
            f"- **Recommend rate:** {stats.get('recommend_rate', 0):.1%}",
            f"- **Total cost:** ${stats.get('total_cost_usd', 0):.4f}",
            f"- **Average duration:** {stats.get('avg_duration_s', 0):.1f}s",
            f"- **Personas tested:** {len(stats.get('personas_used', []))}",
            f"- **Unique topics:** {stats.get('unique_topics', 0)}",
            "",
            "## Top Pain Points",
        ]
        for pain, count in list(pains.items())[:10]:
            lines.append(f"- **{count}x** — {pain}")

        lines.extend(["", "## Top Liked Features"])
        for like, count in list(likes.items())[:10]:
            lines.append(f"- **{count}x** — {like}")

        lines.extend(["", "## Optimization Suggestions"])
        for s in suggestions:
            lines.append(
                f"- **{s.parameter}**: `{s.old_value}` → `{s.new_value}` "
                f"(+{s.expected_improvement:.0%} expected improvement)\n"
                f"  - {s.reason}"
            )

        # Per-persona breakdown
        lines.extend(["", "## Per-Persona Breakdown"])
        by_persona = {}
        for run in self.all_runs:
            by_persona.setdefault(run.persona, []).append(run)

        for persona_key, runs in by_persona.items():
            persona = PERSONAS.get(persona_key, {})
            successful = [r for r in runs if not r.error]
            if not successful:
                continue
            avg_q = statistics.mean([r.quality_mean for r in successful])
            avg_s = statistics.mean([r.satisfaction for r in successful])
            lines.append(
                f"\n### {persona.get('name', persona_key)}"
                f"\n- Runs: {len(runs)}"
                f"\n- Avg quality: {avg_q:.3f}"
                f"\n- Avg satisfaction: {avg_s:.3f}"
                f"\n- Field: {persona.get('field', '?')}"
            )

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return str(path)

    def export_all(self) -> Dict[str, str]:
        """Export all formats. Returns dict of format → path."""
        return {
            "csv": self.export_csv(),
            "json": self.export_json(),
            "markdown": self.export_markdown_report(),
        }
